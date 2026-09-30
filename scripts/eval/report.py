#!/usr/bin/env python3
"""Compare `archi eval qa` runs of several setups on the same dataset; prints a Markdown report.

Usage:
  python scripts/eval/report.py --labels datasets/v1/labels.json \
      --pair prod:new-model --no-tools no-tools \
      prod=runs/prod new-model=runs/new-model no-tools=runs/no-tools > report.md

Each positional argument is NAME=RUN_DIR (a scored run folder). Metrics:
  correct      per question: the share of its required atoms judged `entailed`, averaged over its
               attempts (a failed attempt counts 0); then averaged over questions. The 95% range is a
               bootstrap over questions (4,000 resamples, fixed seed).
  wrong fact   the share of answered, scored attempts with at least one atom judged `contradicted`.
  --pair A:B   B minus A per question, on the questions both scored: mean difference with its 95%
               range, and how many questions got better, worse or stayed equal.
  by source    the share of required atoms found, per atom `answer_source` (from labels.json).
  by label     `correct` per value of every item label in labels.json (task_type, domain, ...).
  --no-tools   a setup run without tools marks each question `parametric` (full / partial / no) by how
               much of it that setup answered; the report then shows scores without those questions.
Failed attempts are labelled: context_overflow, tool_output_too_large, answer_timeout, step_limit, api_error,
other.
Token columns appear when the run recorded usage per attempt.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

LABEL_SKIP = {"atoms", "source_file", "paired_with", "provenance_ref"}


def failure_label(row: dict) -> str:
    err = row.get("error") or {}
    text = f"{err.get('type', '')} {err.get('message', '')}".lower()
    if "context" in text and any(k in text for k in ("window", "length", "overflow", "too long")):
        return "context_overflow"
    if "string_above_max_length" in text or "maximum length" in text:
        return "tool_output_too_large"  # one tool result over the model API's input size limit
    if "answertimelimitexceeded" in text or "timeout" in text:
        return "answer_timeout"
    if "recursion" in text or "step limit" in text or "max_turns" in text:
        return "step_limit"
    if any(k in text for k in ("apierror", "internalservererror", "ratelimit", "apiconnection",
                               "serviceunavailable", "rate limit")):
        return "api_error"
    return "other"


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(run: Path, labels: dict) -> list:
    """One row per attempt, with its recall over required atoms and failure label."""
    evals = {e["attempt_id"]: e for e in read_jsonl(run / "evaluation_results.jsonl")}
    rows = []
    for a in read_jsonl(run / "answers.jsonl"):
        lab = labels.get(a["item_id"], {})
        req = [x for x in lab.get("atoms", []) if x.get("required")]
        e = evals.get(a["attempt_id"], {})
        outcomes = {j.get("atom_id"): j.get("outcome") for j in e.get("judgments") or []}
        answered = a.get("status") == "answer_ready"
        if not answered:
            recall = 0.0
        elif outcomes and req:
            recall = sum(outcomes.get(x["id"]) == "entailed" for x in req) / len(req)
        else:
            recall = None  # answered but not scored (e.g. the judge failed): left out
        usage = a.get("usage") or {}
        rows.append({
            "attempt_id": a["attempt_id"], "item_id": a["item_id"], "status": a.get("status"),
            "failure": None if answered else failure_label(a),
            "recall": recall, "scored": bool(outcomes), "outcomes": outcomes,
            "contradicted": answered and any(v == "contradicted" for v in outcomes.values()),
            "duration_s": int(a.get("duration_ms") or 0) / 1000,
            "tool_calls": len(a.get("tool_calls") or []),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        })
    return rows


def per_question(rows: list) -> dict:
    by = defaultdict(list)
    for r in rows:
        if r["recall"] is not None:
            by[r["item_id"]].append(r["recall"])
    return {q: sum(v) / len(v) for q, v in by.items()}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def bootstrap(values: list, seed: int = 1, n: int = 4000):
    if not values:
        return None, None
    rnd, k = random.Random(seed), len(values)
    means = sorted(sum(rnd.choice(values) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def pct(x, signed=False) -> str:
    if x is None:
        return "–"
    return f"{100 * x:+.0f}" if signed else f"{100 * x:.0f}%"


def source_recall(rows: list, labels: dict) -> dict:
    """{answer_source: (found, total, questions)} over required atoms; failed attempts count 0."""
    found, total, items = defaultdict(float), defaultdict(int), defaultdict(set)
    by_q = defaultdict(list)
    for r in rows:
        if r["recall"] is not None:
            by_q[r["item_id"]].append(r)
    for q, attempts in by_q.items():
        for atom in labels.get(q, {}).get("atoms", []):
            if not atom.get("required") or not atom.get("answer_source"):
                continue
            src = str(atom["answer_source"])
            hit = mean([1.0 if r["outcomes"].get(atom["id"]) == "entailed" else 0.0 for r in attempts])
            found[src] += hit
            total[src] += 1
            items[src].add(q)
    return {s: (found[s], total[s], len(items[s])) for s in total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path, help="labels.json from export_dataset.py")
    ap.add_argument("--pair", action="append", default=[], help="A:B, compare setup B with setup A per question")
    ap.add_argument("--no-tools", help="name of the setup run without tools (marks parametric questions)")
    ap.add_argument("--json", type=Path, help="also write the per-attempt rows as JSON")
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR")
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text())
    setups = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=RUN_DIR, got {spec!r}")
        setups[name] = load_run(Path(path), labels)
    scores = {name: per_question(rows) for name, rows in setups.items()}

    print("# Evaluation report\n")
    print("| setup | questions | attempts | failed | correct | 95% range | wrong fact | median time | median tool calls | tokens in / out per attempt |")
    print("|---|---:|---:|---|---:|---|---:|---:|---:|---|")
    for name, rows in setups.items():
        fails = defaultdict(int)
        for r in rows:
            if r["failure"]:
                fails[r["failure"]] += 1
        q = scores[name]
        lo, hi = bootstrap(list(q.values()))
        scored = [r for r in rows if r["status"] == "answer_ready" and r["scored"]]
        wrong = mean(1.0 if r["contradicted"] else 0.0 for r in scored)
        tin, tout = mean(r["input_tokens"] for r in rows), mean(r["output_tokens"] for r in rows)
        tokens = f"{tin:,.0f} / {tout:,.0f}" if tin is not None else "–"
        failed = f"{sum(fails.values())}" + (f" ({', '.join(f'{k} {v}' for k, v in sorted(fails.items()))})" if fails else "")
        print(f"| {name} | {len(q)} | {len(rows)} | {failed} | {pct(mean(q.values()))} | {pct(lo)}–{pct(hi)} "
              f"| {pct(wrong)} | {statistics.median([r['duration_s'] for r in rows]):.0f} s "
              f"| {statistics.median([r['tool_calls'] for r in rows]):.0f} | {tokens} |")

    for pair in args.pair:
        a, _, b = pair.partition(":")
        if a not in scores or b not in scores:
            print(f"\n(pair {pair}: unknown setup)")
            continue
        common = sorted(set(scores[a]) & set(scores[b]))
        diff = [scores[b][q] - scores[a][q] for q in common]
        lo, hi = bootstrap(diff)
        better, worse = sum(d > 1e-9 for d in diff), sum(d < -1e-9 for d in diff)
        print(f"\n## {b} vs {a} ({len(common)} questions scored in both)\n")
        print(f"{b} − {a}: **{pct(mean(diff), True)} points** (95% range {pct(lo, True)} to {pct(hi, True)}); "
              f"better on {better}, worse on {worse}, equal on {len(common) - better - worse}.\n")
        moved = [(q, d) for q, d in zip(common, diff) if abs(d) >= 0.25]
        if moved:
            print(f"| question | {a} | {b} |\n|---|---:|---:|")
            for q, d in sorted(moved, key=lambda t: t[1]):
                print(f"| {q} | {pct(scores[a][q])} | {pct(scores[b][q])} |")

    if args.no_tools and args.no_tools in scores:
        par = {q: ("full" if v >= 0.999 else "partial" if v >= 0.5 else "no") for q, v in scores[args.no_tools].items()}
        full = {q for q, v in par.items() if v == "full"}
        part = {q for q, v in par.items() if v == "partial"}
        for q in labels:
            labels[q]["parametric"] = par.get(q, "no")
        print(f"\n## Without questions the model answers unaided (from `{args.no_tools}`: full {len(full)}, partial {len(part)})\n")
        print("| setup | all questions | without full | without full and partial |\n|---|---:|---:|---:|")
        for name, q in scores.items():
            print(f"| {name} | {pct(mean(q.values()))} | {pct(mean(v for k, v in q.items() if k not in full))} "
                  f"| {pct(mean(v for k, v in q.items() if k not in full | part))} |")

    per_source = {name: source_recall(rows, labels) for name, rows in setups.items()}
    sources = sorted(set().union(*(set(s) for s in per_source.values())))
    if sources:
        print("\n## Required atoms found, by answer source\n")
        print("| answer source | questions | atoms | " + " | ".join(setups) + " |")
        print("|---|---:|---:|" + "---:|" * len(setups))
        for s in sorted(sources, key=lambda s: -max(ps.get(s, (0, 0, 0))[1] for ps in per_source.values())):
            ref = next(ps[s] for ps in per_source.values() if s in ps)
            cells = [pct(ps[s][0] / ps[s][1]) if s in ps and ps[s][1] else "–" for ps in per_source.values()]
            print(f"| {s} | {ref[2]} | {ref[1]} | " + " | ".join(cells) + " |")

    keys = sorted({k for lab in labels.values() for k in lab if k not in LABEL_SKIP})
    for key in keys:
        values = sorted({json.dumps(lab.get(key)) for lab in labels.values()})
        if len(values) < 2:
            continue
        print(f"\n## Correct, by {key}\n")
        print(f"| {key} | questions | " + " | ".join(setups) + " |")
        print("|---|---:|" + "---:|" * len(setups))
        for v in values:
            qs = {q for q, lab in labels.items() if json.dumps(lab.get(key)) == v}
            cells = [pct(mean(scores[n][q] for q in qs if q in scores[n])) for n in setups]
            print(f"| {json.loads(v)} | {len(qs)} | " + " | ".join(cells) + " |")

    failed = [(n, r) for n, rows in setups.items() for r in rows if r["failure"]]
    if failed:
        print("\n## Failed attempts\n\n| setup | attempt | label |\n|---|---|---|")
        for n, r in failed:
            print(f"| {n} | {r['attempt_id']} | {r['failure']} |")
    if args.json:
        args.json.write_text(json.dumps(setups, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
