# isort: skip_file
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from .dataset import (
    ANSWER_MODE_VALUES,
    Atom,
    DatasetGateway,
    DatasetItem,
    DatasetItemState,
    DatasetSchemaVersion,
    dataset_source_format,
    derive_item_id,
    iter_dataset_items,
    load_dataset,
    load_dataset_bytes,
    validate_atoms,
    validate_dataset_rows,
    validate_nonempty_string,
    validate_optional_enum,
    validate_optional_nonempty_string,
)

OUTCOME_VALUES = {"entailed", "not_mentioned", "contradicted", "unjudgeable"}


@dataclass(frozen=True)
class Judgment:
    atom_id: str
    outcome: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _strict_keys(value: Dict[str, Any], allowed: set, context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")


def validate_gold_output(raw: Any, *, context: str) -> List[Atom]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    _strict_keys(raw, {"atoms"}, context)
    return validate_atoms(raw.get("atoms"), context=f"{context}.atoms")


def validate_judgments(
    raw: Any, *, gold_atoms: Sequence[Atom], context: str
) -> List[Judgment]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    _strict_keys(raw, {"judgments"}, context)
    raw_judgments = raw.get("judgments")
    if not isinstance(raw_judgments, list):
        raise ValueError(f"{context}.judgments must be a list")
    gold_ids = {atom.id for atom in gold_atoms}
    judgments: List[Judgment] = []
    seen = set()
    allowed = {"atom_id", "outcome", "rationale"}
    for index, raw_judgment in enumerate(raw_judgments):
        item_context = f"{context}.judgments[{index}]"
        if not isinstance(raw_judgment, dict):
            raise ValueError(f"{item_context} must be an object")
        _strict_keys(raw_judgment, allowed, item_context)
        atom_id = validate_nonempty_string(
            raw_judgment.get("atom_id"), f"{item_context}.atom_id"
        )
        if atom_id not in gold_ids:
            raise ValueError(f"{item_context} references unknown gold atom '{atom_id}'")
        if atom_id in seen:
            raise ValueError(f"{context} contains duplicate judgment for '{atom_id}'")
        seen.add(atom_id)
        outcome = raw_judgment.get("outcome")
        if outcome not in OUTCOME_VALUES:
            raise ValueError(f"{item_context}.outcome must be a supported outcome")
        rationale = validate_nonempty_string(
            raw_judgment.get("rationale"), f"{item_context}.rationale"
        )
        judgments.append(
            Judgment(atom_id=atom_id, outcome=outcome, rationale=rationale)
        )
    missing = sorted(gold_ids - seen)
    if missing:
        raise ValueError(f"{context} is missing judgment(s) for: {', '.join(missing)}")
    return judgments


__all__ = [
    "ANSWER_MODE_VALUES",
    "Atom",
    "DatasetGateway",
    "DatasetItem",
    "DatasetItemState",
    "DatasetSchemaVersion",
    "Judgment",
    "dataset_source_format",
    "derive_item_id",
    "iter_dataset_items",
    "load_dataset",
    "load_dataset_bytes",
    "validate_atoms",
    "validate_dataset_rows",
    "validate_gold_output",
    "validate_judgments",
    "validate_nonempty_string",
    "validate_optional_enum",
    "validate_optional_nonempty_string",
]
