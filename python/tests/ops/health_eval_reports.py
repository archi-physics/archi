"""Fixture `report.json` files for the instance-health eval signal.

Each file is the dict `archi.evaluation.runner.Report.to_dict()` makes, written where `archi eval` writes it,
so a change to the report format shows up here. No eval runs: the questions are given their status directly.
"""
import json
from pathlib import Path

from archi.evaluation.runner import QuestionResult, Report


def write_report(directory: Path, *, bundle: str = "cern-team", started_at: str, statuses, threshold: int = 7,
                 model: str = "qwen3:8b", error=None, sub: str = "", prompt_hash: str = "sha256:p1") -> Path:
    """`<directory>/<bundle>/[<sub>/]report.json` with one question per entry of `statuses`."""
    report = Report(bundle=bundle, model=model, model_url="http://127.0.0.1:11434/v1",
                    mcp_url="http://127.0.0.1:18080/mcp", pass_threshold=threshold,
                    smoke_file=f"bundles/{bundle}/eval/smoke.yaml", prompt_hash=prompt_hash,
                    questions=[QuestionResult(f"q{i}", f"question {i}?", s) for i, s in enumerate(statuses)],
                    started_at=started_at, error=error)
    out = directory / bundle / sub if sub else directory / bundle
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def scored(passed: int, total: int = 10) -> list[str]:
    return ["pass"] * passed + ["fail"] * (total - passed)
