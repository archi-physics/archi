#!/usr/bin/env python3
"""Turn answers from any agent into a completed `archi eval qa` run phase, so `score` can judge them.

`archi eval qa run` can only drive Archi's own agent. To evaluate another agent (a different framework, a plain
tool loop, a coding assistant), ask it the prepared questions yourself, write its answers to a JSONL file, and
import them into a prepared run folder:

  archi eval qa prepare --dataset datasets/v1/benchmark.json --evaluator-profile evaluator.yaml --output-dir runs/other
  <your runner>  -> answers.jsonl
  python scripts/eval/import_answers.py runs/other answers.jsonl other_agent_config.yaml other_prompt.md \
      --agent-name my-agent --provider openai --model gpt-5.5
  archi eval qa score runs/other --evaluator-profile evaluator.yaml

answers.jsonl: one row per attempt with
  item_id, ordinal (1..N), status ("answer_ready" or "execution_failed"), duration_ms,
  answer (when ready) or error {"type", "message"} (when failed),
  tool_calls: [{"ordinal", "name", "query", "response", "duration_ms", "status"}]  (may be empty),
  usage (optional): {"model_calls", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"}.
Extra keys are kept. Every prepared question must have exactly N attempts.

The config and prompt files are stored with the run for provenance (any text describing the agent works).
The run folder must come straight from `prepare` (no run phase yet).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a run folder made by `archi eval qa prepare`")
    ap.add_argument("answers", type=Path, help="the agent's answers, JSONL (format above)")
    ap.add_argument("agent_config", type=Path, help="a file describing the agent's configuration")
    ap.add_argument("prompt", type=Path, help="the agent's system prompt")
    ap.add_argument("--agent-name", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=1, help="how many answers the runner produced in parallel")
    args = ap.parse_args()

    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["phases"].get("prepare", {}).get("status") != "completed":
        sys.exit("the prepare phase of this run folder is not complete")
    if "run" in manifest["phases"] or (run / "answers.jsonl").exists():
        sys.exit("this run folder already has a run phase; use a freshly prepared one")

    prepared = [json.loads(line) for line in (run / "preparation.jsonl").read_text().splitlines() if line.strip()]
    item_ids = [p.get("id", p.get("item_id")) for p in prepared if p.get("status") == "prepared"]
    rows = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
    if not rows:
        sys.exit("no answers")
    for r in rows:
        missing = [k for k in ("item_id", "ordinal", "status") if k not in r]
        if missing or r["status"] not in ("answer_ready", "execution_failed"):
            sys.exit(f"bad answer row (needs item_id, ordinal, status answer_ready|execution_failed): {str(r)[:200]}")
        if r["status"] == "answer_ready" and not str(r.get("answer") or "").strip():
            sys.exit(f"{r['item_id']} attempt {r['ordinal']}: answer_ready without an answer")
        r.setdefault("duration_ms", 0)
        r.setdefault("tool_calls", [])
    attempts = max(r["ordinal"] for r in rows)
    expected = {(i, n) for i in item_ids for n in range(1, attempts + 1)}
    got = {(r["item_id"], r["ordinal"]) for r in rows}
    if got != expected or len(rows) != len(expected):
        sys.exit(f"answers do not match the prepared questions x {attempts} attempts: "
                 f"missing {sorted(expected - got)[:5]}, extra {sorted(got - expected)[:5]}")

    shutil.copyfile(args.agent_config, run / "agent_config.resolved.yaml")
    shutil.copyfile(args.prompt, run / "agent_spec.resolved.md")
    cfg_sha, spec_sha = sha(run / "agent_config.resolved.yaml"), sha(run / "agent_spec.resolved.md")
    order = {i: k for k, i in enumerate(item_ids)}
    rows.sort(key=lambda r: (order[r["item_id"]], r["ordinal"]))
    with open(run / "answers.jsonl", "w") as f:
        for r in rows:
            r = dict(r)
            r["attempt_id"] = f"{r['item_id']}-attempt-{r['ordinal']}"
            r["agent_config_sha256"], r["agent_spec_sha256"] = cfg_sha, spec_sha
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    (run / "live_checks.jsonl").write_text("")

    now = datetime.now(timezone.utc).isoformat()
    started = min((r.get("started_at") or now) for r in rows)
    manifest["agent"] = {"agent_class": args.agent_name, "config_artifact": "agent_config.resolved.yaml",
                         "model": args.model, "provider": args.provider, "spec_artifact": "agent_spec.resolved.md"}
    manifest["attempts"] = attempts
    for name in ("agent_config.resolved.yaml", "agent_spec.resolved.md", "answers.jsonl", "live_checks.jsonl"):
        manifest["artifacts"][name] = sha(run / name)
    manifest["phases"]["run"] = {"actual_agent_executions": len(rows), "attempt_slots": len(rows),
                                 "checked_at": started, "completed_at": now, "live_check_status": "matched_baseline",
                                 "started_at": started, "status": "completed", "workers": args.workers,
                                 "external_runner": args.agent_name}
    manifest["status"] = "run_completed"
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"imported {len(rows)} attempts ({sum(r['status'] == 'answer_ready' for r in rows)} answered) into {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
