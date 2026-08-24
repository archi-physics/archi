# isort: skip_file
from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, TextIO

from .artifacts import AtomicTextWriter
from .constants import ATTEMPT_LIFECYCLE_STATUSES, ITEM_LIFECYCLE_STATUSES
from .preparation import PreparationRecord
from .validation import Atom, Judgment

OUTCOME_VALUES = {"entailed": 1, "not_mentioned": 0, "contradicted": -1}


def score_attempt(
    gold_atoms: Sequence[Atom], judgments: Sequence[Judgment]
) -> Dict[str, Any]:
    by_id = {judgment.atom_id: judgment for judgment in judgments}
    required_count = sum(1 for atom in gold_atoms if atom.required)
    entailed_required = sum(
        1
        for atom in gold_atoms
        if atom.required and by_id[atom.id].outcome == "entailed"
    )
    value_sum = sum(OUTCOME_VALUES[by_id[atom.id].outcome] for atom in gold_atoms)
    return {
        "atom_score": max(0.0, value_sum / len(gold_atoms)),
        "required_atom_recall": entailed_required / required_count,
        "passed": entailed_required == required_count,
    }


def build_summary(
    preparation: Iterable[PreparationRecord],
    evaluation_results: Iterable[Dict[str, Any]],
    live_checks: Iterable[Dict[str, Any]] = (),
    *,
    item_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    item_lifecycle = Counter()
    attempt_lifecycle = Counter()
    aggregate_outcomes = Counter()
    quality_attempt_count = 0
    passed_attempt_count = 0
    atom_score_sum = 0.0
    required_recall_sum = 0.0
    scored_attempt_count = 0
    oracle_calls = Counter()
    with tempfile.TemporaryDirectory(prefix=".qa-summary-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "summary.sqlite3"))
        try:
            connection.executescript(
                """
                CREATE TABLE items (
                    ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT UNIQUE NOT NULL,
                    atoms_json TEXT NOT NULL,
                    requested INTEGER NOT NULL DEFAULT 0,
                    quality_k INTEGER NOT NULL DEFAULT 0,
                    scored INTEGER NOT NULL DEFAULT 0,
                    execution_failed INTEGER NOT NULL DEFAULT 0,
                    evaluation_failed INTEGER NOT NULL DEFAULT 0,
                    live_validation_failed INTEGER NOT NULL DEFAULT 0,
                    passed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE entailed (
                    item_id TEXT NOT NULL,
                    atom_id TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (item_id, atom_id)
                );
                """
            )
            for record in preparation:
                item_lifecycle[record.status] += 1
                for call in record.oracle_calls or ():
                    oracle_calls["succeeded" if call.success else "failed"] += 1
                if record.status == "prepared":
                    connection.execute(
                        "INSERT INTO items (item_id, atoms_json) VALUES (?, ?)",
                        (
                            record.item_id,
                            json.dumps(
                                [atom.to_dict() for atom in record.prepared_gold_atoms],
                                ensure_ascii=False,
                            ),
                        ),
                    )
            connection.commit()

            for result in evaluation_results:
                status = result["status"]
                attempt_lifecycle[status] += 1
                item_id = result["item_id"]
                cursor = connection.execute(
                    "UPDATE items SET requested = requested + 1 WHERE item_id = ?",
                    (item_id,),
                )
                if cursor.rowcount != 1:
                    raise ValueError(
                        "evaluation result references an unknown prepared item"
                    )
                if status == "execution_failed":
                    connection.execute(
                        "UPDATE items SET execution_failed = execution_failed + 1, quality_k = quality_k + 1 WHERE item_id = ?",
                        (item_id,),
                    )
                    quality_attempt_count += 1
                elif status == "evaluation_failed":
                    connection.execute(
                        "UPDATE items SET evaluation_failed = evaluation_failed + 1 WHERE item_id = ?",
                        (item_id,),
                    )
                elif status == "live_validation_failed":
                    connection.execute(
                        "UPDATE items SET live_validation_failed = live_validation_failed + 1 WHERE item_id = ?",
                        (item_id,),
                    )
                elif status == "scored":
                    connection.execute(
                        "UPDATE items SET scored = scored + 1, quality_k = quality_k + 1, passed = passed + ? WHERE item_id = ?",
                        (1 if result["passed"] else 0, item_id),
                    )
                    quality_attempt_count += 1
                    scored_attempt_count += 1
                    atom_score_sum += result["atom_score"]
                    required_recall_sum += result["required_atom_recall"]
                    if result["passed"]:
                        passed_attempt_count += 1
                    for judgment in result["judgments"]:
                        aggregate_outcomes[judgment["outcome"]] += 1
                        if judgment["outcome"] == "entailed":
                            connection.execute(
                                "INSERT INTO entailed (item_id, atom_id, count) VALUES (?, ?, 1) "
                                "ON CONFLICT(item_id, atom_id) DO UPDATE SET count = count + 1",
                                (item_id, judgment["atom_id"]),
                            )
                else:
                    raise ValueError(f"unsupported evaluation status: {status}")
            connection.commit()

            item_summaries = [] if item_sink is None else None
            item_rate_sum = 0.0
            item_rate_count = 0
            item_macro_exclusion_count = 0
            rows = connection.execute(
                "SELECT item_id, atoms_json, requested, quality_k, scored, execution_failed, "
                "evaluation_failed, live_validation_failed, passed FROM items ORDER BY ordinal"
            )
            for row in rows:
                (
                    item_id,
                    atoms_json,
                    requested,
                    k,
                    scored,
                    execution_failed,
                    evaluation_failed,
                    live_validation_failed,
                    passed,
                ) = row
                entailed = dict(
                    connection.execute(
                        "SELECT atom_id, count FROM entailed WHERE item_id = ?",
                        (item_id,),
                    )
                )
                item_summary = {
                    "item_id": item_id,
                    "k": k,
                    "requested_attempts": requested,
                    "scored_attempts": scored,
                    "execution_failed_attempts": execution_failed,
                    "evaluation_failed_attempts": evaluation_failed,
                    "live_validation_failed_attempts": live_validation_failed,
                    "item_pass_count": passed,
                    "item_pass_rate": passed / k if k else None,
                    "gold_atom_pass_rates": [
                        {
                            "atom_id": atom["id"],
                            "atom_pass_count": entailed.get(atom["id"], 0),
                            "k": k,
                            "atom_pass_rate": (
                                entailed.get(atom["id"], 0) / k if k else None
                            ),
                        }
                        for atom in json.loads(atoms_json)
                    ],
                }
                if item_summary["item_pass_rate"] is None:
                    item_macro_exclusion_count += 1
                else:
                    item_rate_sum += item_summary["item_pass_rate"]
                    item_rate_count += 1
                if item_sink is None:
                    assert item_summaries is not None
                    item_summaries.append(item_summary)
                else:
                    item_sink(item_summary)
        finally:
            connection.close()
    for check in live_checks:
        for call in check["calls"]:
            oracle_calls["succeeded" if call["success"] else "failed"] += 1
    return {
        "item_lifecycle_counts": {
            status: item_lifecycle.get(status, 0) for status in ITEM_LIFECYCLE_STATUSES
        },
        "attempt_lifecycle_counts": {
            status: attempt_lifecycle.get(status, 0)
            for status in ATTEMPT_LIFECYCLE_STATUSES
        },
        "quality_accounted_attempts": quality_attempt_count,
        "passed_attempts": passed_attempt_count,
        "overall_attempt_pass_rate": (
            passed_attempt_count / quality_attempt_count
            if quality_attempt_count
            else None
        ),
        "macro_mean_item_pass_rate": (
            item_rate_sum / item_rate_count if item_rate_count else None
        ),
        "item_macro_exclusion_count": item_macro_exclusion_count,
        "macro_mean_scored_attempt_atom_score": (
            atom_score_sum / scored_attempt_count if scored_attempt_count else None
        ),
        "macro_mean_scored_attempt_required_atom_recall": (
            required_recall_sum / scored_attempt_count if scored_attempt_count else None
        ),
        "aggregate_atom_outcomes": {
            outcome: aggregate_outcomes.get(outcome, 0)
            for outcome in ("entailed", "not_mentioned", "contradicted")
        },
        "oracle_calls_succeeded": oracle_calls["succeeded"],
        "oracle_calls_failed": oracle_calls["failed"],
        **({"items": item_summaries} if item_summaries is not None else {}),
    }


def write_summary(
    path: Path, summary: Dict[str, Any], items: Iterable[Dict[str, Any]]
) -> None:
    """Stream the dataset-sized item projection into the machine summary."""
    with AtomicTextWriter(path) as handle:
        handle.write("{")
        first = True
        for key in sorted(summary):
            if key == "items":
                continue
            if not first:
                handle.write(",")
            handle.write(json.dumps(key))
            handle.write(":")
            handle.write(json.dumps(summary[key], ensure_ascii=False, sort_keys=True))
            first = False
        if not first:
            handle.write(",")
        handle.write('"items":[')
        first_item = True
        for item in items:
            if not first_item:
                handle.write(",")
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            first_item = False
        handle.write("]}\n")


def _report_header(summary: Dict[str, Any], manifest: Dict[str, Any]) -> list[str]:
    agent = manifest.get("agent") or {}
    provenance = summary.get("provenance") or {}
    return [
        "# Archi QA Evaluation Report",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Attempts per prepared item: `{manifest.get('attempts', 'unavailable')}`",
        f"- Agent class: `{agent.get('agent_class', 'unavailable')}`",
        f"- Agent config SHA-256: `{provenance.get('agent_config_sha256', 'unavailable')}`",
        f"- Agent spec SHA-256: `{provenance.get('agent_spec_sha256', 'unavailable')}`",
        f"- Evaluator profile SHA-256: `{provenance.get('evaluator_profile_sha256', 'unavailable')}`",
        f"- Overall attempt pass rate: `{_rate(summary['overall_attempt_pass_rate'])}` "
        f"({summary['passed_attempts']}/{summary['quality_accounted_attempts']})",
        f"- Macro mean item pass rate: `{_rate(summary['macro_mean_item_pass_rate'])}`",
        f"- Macro mean atom score: `{_rate(summary['macro_mean_scored_attempt_atom_score'])}`",
        f"- Macro mean required-atom recall: `{_rate(summary['macro_mean_scored_attempt_required_atom_recall'])}`",
        "",
        "## Lifecycle counts",
        "",
        f"- Items: `{summary['item_lifecycle_counts']}`",
        f"- Attempts: `{summary['attempt_lifecycle_counts']}`",
        f"- Items excluded from item macros (`k = 0`): `{summary['item_macro_exclusion_count']}`",
        "",
        "## Per-item results",
        "",
    ]


def _write_item_report(handle: TextIO, item: Dict[str, Any]) -> None:
    lines = [
        f"### {item['item_id']}",
        "",
        f"- Item pass rate: `{_rate(item['item_pass_rate'])}` "
        f"({item['item_pass_count']}/{item['k']})",
        f"- Scored / execution failed / evaluation failed: "
        f"`{item['scored_attempts']} / {item['execution_failed_attempts']} / "
        f"{item['evaluation_failed_attempts']}`",
        f"- Live validation failed: `{item.get('live_validation_failed_attempts', 0)}`",
        "- Gold atom rates:",
        "",
    ]
    handle.write("\n".join(lines) + "\n")
    for atom in item["gold_atom_pass_rates"]:
        rendered = (
            "unavailable"
            if atom["atom_pass_rate"] is None
            else f"{atom['atom_pass_rate']:.3f}@{atom['k']}"
        )
        handle.write(
            f"  - `{atom['atom_id']}`: `{rendered}` "
            f"({atom['atom_pass_count']}/{atom['k']})\n"
        )
    handle.write("\n")


def write_report(
    path: Path,
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
    items: Iterable[Dict[str, Any]],
) -> None:
    with AtomicTextWriter(path) as handle:
        handle.write("\n".join(_report_header(summary, manifest)) + "\n")
        for item in items:
            _write_item_report(handle, item)


def _rate(value: Optional[float]) -> str:
    return "unavailable" if value is None else f"{value:.3f}"
