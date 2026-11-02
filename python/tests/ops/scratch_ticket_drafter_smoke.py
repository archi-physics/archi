"""Smoke run of the ticket-triage reply drafter against the real model (a paid call; not part of the suite).

    OPENAI_API_KEY=... python python/tests/ops/scratch_ticket_drafter_smoke.py --cap 0.03 --max-calls 30 \
        --price-in 0.10 --price-out 0.50 --price-cached 0.01 [--rounds 1]

It reads the fixture queue (the three arriving tickets of the unit tests: Redmine, JIRA and ServiceNow) with
the real context reader, and asks the drafter for each ticket's reply through the real TriageProposer. It
counts each call's cost from the reply's token usage, and before every call it stops if the cost so far
plus the call's bound could pass `--cap`. It prints one JSON line per call (tokens, dollars, whether the
reply was used and why not) and a total. The key is read from the environment and never printed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from archi.evaluation.agent import make_model  # noqa: E402
from archi.ops.proposers import ProposeRequest  # noqa: E402
from archi.triage.context import read_ticket_context  # noqa: E402
from archi.triage.drafter import INSTRUCTIONS, Drafter, records_text  # noqa: E402
from archi.triage.proposer import TriageProposer  # noqa: E402
from test_ticket_triage import NAME, graph_with_arrivals  # noqa: E402

ENTITIES = ["redmine:4820", "jira:OPS-15", "servicenow:INC0010003"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, required=True, help="stop before a call that could pass this many dollars")
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--price-in", type=float, required=True)
    ap.add_argument("--price-out", type=float, required=True)
    ap.add_argument("--price-cached", type=float, required=True)
    args = ap.parse_args(argv)

    usage = []

    def counting(url, model, **kw):
        client = make_model(url, model, **kw)
        inner = client.complete

        def complete(messages, tools):
            reply = inner(messages, tools)
            usage.append({"prompt_tokens": reply.prompt_tokens, "cached_tokens": reply.cached_tokens,
                          "completion_tokens": reply.completion_tokens, "reasoning_tokens": reply.reasoning_tokens,
                          "usage_reported": reply.usage_reported, "finish": reply.finish_reason})
            return reply

        client.complete = complete
        return client

    config = {"model": args.model, "max_output_tokens": 1000,
              "price": {"input_per_million": args.price_in, "output_per_million": args.price_out,
                        "cached_input_per_million": args.price_cached}}
    drafter = Drafter(config, make_model=counting)
    proposer = TriageProposer({"doc_subtypes": ["servicenow_kb_article"]}, drafter=drafter,
                              read_text=lambda t, g: "Jobs fail with ENOSPC on /scratch.")
    g = graph_with_arrivals()
    spent, calls = 0.0, 0
    for _ in range(args.rounds):
        for entity in ENTITIES:
            d = read_ticket_context(g, entity, g.generation, {"doc_subtypes": ["servicenow_kb_article"]})
            ctx = d["context"]
            user = records_text(ctx["ticket"], "Jobs fail with ENOSPC on /scratch.", ctx["related"], ctx["docs"],
                                ctx["precedents"])
            bound = drafter.bound(len(INSTRUCTIONS) + len(user))
            if calls >= args.max_calls or spent + bound > args.cap:
                print(json.dumps({"stopped": True, "calls": calls, "spent": round(spent, 6), "next_bound": bound}))
                return 0
            before = len(usage)
            out = proposer.propose(ProposeRequest(NAME, {"id": "s", "entity": entity}, ctx, g.generation, 1))
            calls += 1
            spent += out.cost
            p = out.proposal
            used = "the model reply was not used" not in p["summary"]
            print(json.dumps({"entity": entity, "cost": round(out.cost, 6), "used": used,
                              "why_not": None if used else p["summary"].split("the model reply was not used")[1],
                              **(usage[before] if len(usage) > before else {}),
                              "reply": p["payload"]["body"].split("Suggested reply")[1].split("\nSources:")[0][:600]}))
    print(json.dumps({"total_calls": calls, "spent": round(spent, 6), "per_ticket": round(spent / max(calls, 1), 6)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
