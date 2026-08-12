from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence

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
    preparation: Sequence[PreparationRecord],
    evaluation_results: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    item_lifecycle = Counter(record.status for record in preparation)
    attempt_lifecycle = Counter()
    prepared_by_id = {
        record.item_id: record
        for record in preparation
        if record.status == "prepared"
    }
    item_totals = {
        item_id: {
            "requested": 0,
            "k": 0,
            "scored": 0,
            "execution_failed": 0,
            "evaluation_failed": 0,
            "passed": 0,
            "entailed_atoms": Counter(),
        }
        for item_id in prepared_by_id
    }
    aggregate_outcomes = Counter()
    quality_attempt_count = 0
    passed_attempt_count = 0
    atom_score_sum = 0.0
    required_recall_sum = 0.0
    scored_attempt_count = 0
    for result in evaluation_results:
        status = result["status"]
        attempt_lifecycle[status] += 1
        totals = item_totals[result["item_id"]]
        totals["requested"] += 1
        if status == "execution_failed":
            totals["execution_failed"] += 1
            totals["k"] += 1
            quality_attempt_count += 1
        elif status == "evaluation_failed":
            totals["evaluation_failed"] += 1
        elif status == "scored":
            totals["scored"] += 1
            totals["k"] += 1
            quality_attempt_count += 1
            scored_attempt_count += 1
            atom_score_sum += result["atom_score"]
            required_recall_sum += result["required_atom_recall"]
            if result["passed"]:
                totals["passed"] += 1
                passed_attempt_count += 1
            for judgment in result["judgments"]:
                aggregate_outcomes[judgment["outcome"]] += 1
                if judgment["outcome"] == "entailed":
                    totals["entailed_atoms"][judgment["atom_id"]] += 1
        else:
            raise ValueError(f"unsupported evaluation status: {status}")

    item_summaries = []
    for item_id, prepared in prepared_by_id.items():
        totals = item_totals[item_id]
        k = totals["k"]
        item_summaries.append(
            {
                "item_id": item_id,
                "k": k,
                "requested_attempts": totals["requested"],
                "scored_attempts": totals["scored"],
                "execution_failed_attempts": totals["execution_failed"],
                "evaluation_failed_attempts": totals["evaluation_failed"],
                "item_pass_count": totals["passed"],
                "item_pass_rate": totals["passed"] / k if k else None,
                "gold_atom_pass_rates": [
                    {
                        "atom_id": atom.id,
                        "atom_pass_count": totals["entailed_atoms"][atom.id],
                        "k": k,
                        "atom_pass_rate": (
                            totals["entailed_atoms"][atom.id] / k if k else None
                        ),
                    }
                    for atom in prepared.prepared_gold_atoms
                ],
            }
        )
    item_rate_sum = sum(
        row["item_pass_rate"]
        for row in item_summaries
        if row["item_pass_rate"] is not None
    )
    item_rate_count = sum(
        1 for row in item_summaries if row["item_pass_rate"] is not None
    )
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
        "item_macro_exclusion_count": sum(
            1 for row in item_summaries if row["item_pass_rate"] is None
        ),
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
        "items": item_summaries,
    }


def _rate(value: Optional[float]) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def render_report(summary: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    agent = manifest.get("agent") or {}
    provenance = summary.get("provenance") or {}
    lines = [
        "# Archi QA Evaluation Report",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Attempts per prepared item: `{manifest.get('attempts', 'unavailable')}`",
        f"- Agent class: `{agent.get('agent_class', 'unavailable')}`",
        f"- Agent config SHA-256: `{provenance.get('agent_config_sha256', 'unavailable')}`",
        f"- Agent spec SHA-256: `{provenance.get('agent_spec_sha256', 'unavailable')}`",
        f"- Evaluator profile SHA-256: "
        f"`{provenance.get('evaluator_profile_sha256', 'unavailable')}`",
        f"- Overall attempt pass rate: `{_rate(summary['overall_attempt_pass_rate'])}` "
        f"({summary['passed_attempts']}/{summary['quality_accounted_attempts']})",
        f"- Macro mean item pass rate: `{_rate(summary['macro_mean_item_pass_rate'])}`",
        f"- Macro mean atom score: `{_rate(summary['macro_mean_scored_attempt_atom_score'])}`",
        f"- Macro mean required-atom recall: "
        f"`{_rate(summary['macro_mean_scored_attempt_required_atom_recall'])}`",
        "",
        "## Lifecycle counts",
        "",
        f"- Items: `{summary['item_lifecycle_counts']}`",
        f"- Attempts: `{summary['attempt_lifecycle_counts']}`",
        f"- Items excluded from item macros (`k = 0`): "
        f"`{summary['item_macro_exclusion_count']}`",
        "",
        "## Per-item results",
        "",
    ]
    for item in summary["items"]:
        lines.extend(
            [
                f"### {item['item_id']}",
                "",
                f"- Item pass rate: `{_rate(item['item_pass_rate'])}` "
                f"({item['item_pass_count']}/{item['k']})",
                f"- Scored / execution failed / evaluation failed: "
                f"`{item['scored_attempts']} / {item['execution_failed_attempts']} / "
                f"{item['evaluation_failed_attempts']}`",
                "- Gold atom rates:",
                "",
            ]
        )
        for atom in item["gold_atom_pass_rates"]:
            rendered = (
                "unavailable"
                if atom["atom_pass_rate"] is None
                else f"{atom['atom_pass_rate']:.3f}@{atom['k']}"
            )
            lines.append(
                f"  - `{atom['atom_id']}`: `{rendered}` "
                f"({atom['atom_pass_count']}/{atom['k']})"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
