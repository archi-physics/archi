"""Do a question set's gold node ids still name the same records in a later generation?

The agent arms resolve each gold id through the MCP `inspect` tool (`raw_arm.inspect_node`).
That is only sound if a published generation keeps the ids of the records it carries over.
This command samples gold ids from a question file and inspects each at two published
generations of one instance:

    python -m archi.evaluation.gold_check --questions FILE --generation-a G1 --generation-b G2 \\
        [--sample 20] [--seed 0] [--corpus DIR] [--instance DIR | --mcp-url URL] [--json OUT]

A sampled id is stable when it is visible at both generations with the same subtype and
source, and (with --corpus) resolves to the same corpus record at both. Exit 0 when every
sampled id is stable, 1 when one is not, 2 when the check could not run. It calls only the
MCP endpoint: no model, nothing paid.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

TOKEN_VAR = "OKG_CHAT_MCP_TOKEN"
DEFAULT_SAMPLE = 20


async def check(tools: Any, gold_ids: list[str], generation_a: str, generation_b: str,
                keys: Optional[dict] = None) -> list[dict[str, Any]]:
    from archi.evaluation.raw_arm import inspect_node, record_for

    rows = []
    for gid in gold_ids:
        seen = {}
        for label, generation in (("a", generation_a), ("b", generation_b)):
            info, why = await inspect_node(tools, gid, generation=generation)
            entry: dict[str, Any] = {"visible": info is not None}
            if info is None:
                entry["why"] = why
            else:
                entry.update(subtype=info.subtype, source=info.source)
                if keys is not None:
                    record = record_for(info, keys)
                    entry["record"] = record if isinstance(record, str) else (record.path or record.status)
            seen[label] = entry
        a, b = seen["a"], seen["b"]
        problems = []
        if not a["visible"] or not b["visible"]:
            problems.append("not visible at " + " and ".join(k for k in ("a", "b") if not seen[k]["visible"]))
        else:
            for name in ("subtype", "source", "record"):
                if a.get(name) != b.get(name):
                    problems.append(f"{name} {a.get(name)!r} -> {b.get(name)!r}")
        rows.append({"id": gid, "stable": not problems, "problems": problems, "a": a, "b": b})
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m archi.evaluation.gold_check", description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True, help="the context question file")
    parser.add_argument("--generation-a", required=True, help="the earlier published generation id")
    parser.add_argument("--generation-b", required=True, help="the later published generation id")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="how many gold ids to check (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="the sampling seed (default 0)")
    parser.add_argument("--corpus", type=Path, help="a raw corpus: also check each id resolves to the same record")
    parser.add_argument("--instance", type=Path, default=Path("."), help="instance directory (default: .)")
    parser.add_argument("--mcp-url", help="an MCP endpoint instead of the instance's; the token comes from "
                                          f"${TOKEN_VAR}")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args(argv)

    from archi.evaluation import agent as agent_mod
    from archi.evaluation.context import ContextSetError, load_context

    try:
        cset = load_context(args.questions)
    except ContextSetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ids = sorted({g for q in cset.questions for g in q.gold})
    sample = sorted(random.Random(args.seed).sample(ids, min(max(args.sample, 1), len(ids))))
    token, url = os.environ.get(TOKEN_VAR, ""), args.mcp_url
    if not url:
        from archi.stack.instance import Instance, InstanceError, read_env

        try:
            instance = Instance.load(args.instance)
            token = read_env(instance.env_path).get(TOKEN_VAR) or token
        except InstanceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        url = f"http://127.0.0.1:{instance.ports.mcp}/mcp"
    if not token:
        print(f"error: no MCP bearer token ({TOKEN_VAR})", file=sys.stderr)
        return 2
    keys = None
    if args.corpus:
        from archi.evaluation.raw_corpus import CorpusError, load_corpus

        try:
            keys = load_corpus(args.corpus).keys()
        except (CorpusError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    async def go():
        async with agent_mod.McpToolBox(url, token) as tools:
            return await check(tools, sample, args.generation_a, args.generation_b, keys)

    try:
        rows = asyncio.run(go())
    except Exception as exc:  # noqa: BLE001 - an endpoint that cannot be used is a run that could not execute
        print(f"error: the MCP endpoint could not be used: {agent_mod._describe(exc)}", file=sys.stderr)
        return 2
    for row in rows:
        print(f"{'stable  ' if row['stable'] else 'CHANGED '} {row['id']}" +
              ("" if row["stable"] else f": {'; '.join(row['problems'])}"))
    stable = sum(r["stable"] for r in rows)
    print(f"{stable} of {len(rows)} sampled gold ids are stable between {args.generation_a} and {args.generation_b} "
          f"({len(ids)} gold ids in {args.questions.name})")
    if args.json:
        args.json.write_text(json.dumps({"generation_a": args.generation_a, "generation_b": args.generation_b,
                                         "seed": args.seed, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0 if stable == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
