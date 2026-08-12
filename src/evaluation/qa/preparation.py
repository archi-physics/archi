from __future__ import annotations

# isort: off
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from .artifacts import iter_jsonl
from .validation import (
    ANSWER_MODE_VALUES,
    Atom,
    DatasetItem,
    validate_atoms,
    validate_gold_output,
    validate_nonempty_string,
    validate_optional_enum,
    validate_optional_nonempty_string,
)

# isort: on

PreparationStatus = Literal[
    "prepared",
    "skipped_time_sensitive",
    "preparation_failed",
]
AtomSource = Literal["supplied", "inferred"]


class GoldExtractor(Protocol):
    def extract_gold(self, question: str, answer: str) -> object: ...


class AnswerComparator(Protocol):
    def compare(
        self,
        question: str,
        gold_atoms: Sequence[Atom],
        answer: str,
    ) -> object: ...


@dataclass(frozen=True)
class PreparationRecord:
    item_id: str
    status: PreparationStatus
    category: Optional[str] = None
    answer_mode: Optional[str] = None
    answer_source: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    time_sensitive: Optional[bool] = None
    gold_atoms: Optional[Tuple[Atom, ...]] = None
    atom_source: Optional[AtomSource] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        validate_nonempty_string(self.item_id, "preparation item_id")
        validate_optional_nonempty_string(
            self.category,
            "preparation category",
        )
        validate_optional_enum(
            self.answer_mode,
            ANSWER_MODE_VALUES,
            "preparation answer_mode",
        )
        validate_optional_nonempty_string(
            self.answer_source,
            "preparation answer_source",
        )
        if self.status not in {
            "prepared",
            "skipped_time_sensitive",
            "preparation_failed",
        }:
            raise ValueError("unsupported preparation status")
        if self.status == "prepared":
            normalized_question = validate_nonempty_string(
                self.question,
                "prepared record question",
                normalize_newlines=True,
            )
            normalized_answer = validate_nonempty_string(
                self.answer,
                "prepared record answer",
                normalize_newlines=True,
            )
            if normalized_question != self.question or normalized_answer != self.answer:
                raise ValueError("prepared record text must use normalized newlines")
            if self.time_sensitive is not False:
                raise ValueError("time-sensitive item cannot be prepared")
            if not self.gold_atoms:
                raise ValueError("prepared record requires gold atoms")
            if self.atom_source not in {"supplied", "inferred"}:
                raise ValueError("prepared record requires an atom source")
            if self.error is not None:
                raise ValueError("prepared record cannot contain an error")
            return
        if self.status == "preparation_failed":
            if self.error is None:
                raise ValueError("failed preparation requires an error")
            if not isinstance(self.error, str):
                raise ValueError("failed preparation error must be a string")
            if (
                self.question is not None
                or self.answer is not None
                or self.time_sensitive is not None
                or self.gold_atoms is not None
                or self.atom_source is not None
            ):
                raise ValueError("failed preparation cannot contain prepared output")
            return
        if (
            self.question is not None
            or self.answer is not None
            or self.time_sensitive is not None
            or self.gold_atoms is not None
            or self.atom_source is not None
            or self.error is not None
        ):
            raise ValueError("skipped preparation cannot contain output")

    @property
    def prepared_gold_atoms(self) -> Tuple[Atom, ...]:
        if self.status != "prepared" or self.gold_atoms is None:
            raise ValueError("preparation record is not prepared")
        return self.gold_atoms

    @property
    def prepared_question(self) -> str:
        if self.status != "prepared" or self.question is None:
            raise ValueError("preparation record is not prepared")
        return self.question

    @property
    def prepared_answer(self) -> str:
        if self.status != "prepared" or self.answer is None:
            raise ValueError("preparation record is not prepared")
        return self.answer

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "item_id": self.item_id,
            "status": self.status,
            "category": self.category,
            "answer_mode": self.answer_mode,
            "answer_source": self.answer_source,
        }
        if self.status == "prepared":
            row.update(
                {
                    "question": self.prepared_question,
                    "answer": self.prepared_answer,
                    "time_sensitive": self.time_sensitive,
                    "atom_source": self.atom_source,
                    "gold_atoms": [atom.to_dict() for atom in self.prepared_gold_atoms],
                }
            )
        elif self.status == "preparation_failed":
            row["error"] = self.error
        return row


def prepare_dataset_items(
    items: Sequence[DatasetItem], extractor: GoldExtractor
) -> List[PreparationRecord]:
    """Prepare one immutable terminal record for every validated dataset item."""
    return [prepare_dataset_item(item, extractor) for item in items]


def prepare_dataset_item(
    item: DatasetItem, extractor: GoldExtractor
) -> PreparationRecord:
    """Prepare one validated dataset item into one terminal record."""
    if item.time_sensitive:
        return PreparationRecord(
            item_id=item.id,
            status="skipped_time_sensitive",
            category=item.category,
            answer_mode=item.answer_mode,
            answer_source=item.answer_source,
        )
    try:
        if item.expected_atoms is not None:
            gold_atoms = item.expected_atoms
            atom_source: AtomSource = "supplied"
        else:
            gold_atoms = validate_gold_output(
                extractor.extract_gold(item.question, item.answer),
                context=f"gold extraction for item {item.id}",
            )
            atom_source = "inferred"
    except Exception as exc:
        return PreparationRecord(
            item_id=item.id,
            status="preparation_failed",
            category=item.category,
            answer_mode=item.answer_mode,
            answer_source=item.answer_source,
            error=str(exc),
        )
    return PreparationRecord(
        item_id=item.id,
        status="prepared",
        category=item.category,
        answer_mode=item.answer_mode,
        answer_source=item.answer_source,
        question=item.question,
        answer=item.answer,
        time_sensitive=item.time_sensitive,
        gold_atoms=tuple(gold_atoms),
        atom_source=atom_source,
    )


def _require_exact_keys(
    row: Dict[str, Any], expected: Set[str], *, context: str
) -> None:
    missing = sorted(expected - set(row))
    unknown = sorted(set(row) - expected)
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unknown:
        details.append("unknown: " + ", ".join(unknown))
    if details:
        raise ValueError(f"{context} has invalid fields ({'; '.join(details)})")


def _record_from_row(row: Dict[str, Any], *, index: int) -> PreparationRecord:
    context = f"preparation row {index}"
    status = row.get("status")
    if status not in {
        "prepared",
        "skipped_time_sensitive",
        "preparation_failed",
    }:
        raise ValueError(f"{context} has an unsupported status")
    base_fields = {
        "item_id",
        "status",
        "category",
        "answer_mode",
        "answer_source",
    }
    status_fields = {
        "prepared": {
            "question",
            "answer",
            "time_sensitive",
            "atom_source",
            "gold_atoms",
        },
        "preparation_failed": {"error"},
        "skipped_time_sensitive": set(),
    }[status]
    _require_exact_keys(row, base_fields | status_fields, context=context)
    common = {
        "item_id": row["item_id"],
        "category": row["category"],
        "answer_mode": row["answer_mode"],
        "answer_source": row["answer_source"],
    }

    if status == "prepared":
        atom_source = row["atom_source"]
        gold_atoms = tuple(
            validate_atoms(row["gold_atoms"], context=f"{context}.gold_atoms")
        )
        return PreparationRecord(
            **common,
            status="prepared",
            question=row["question"],
            answer=row["answer"],
            time_sensitive=row["time_sensitive"],
            gold_atoms=gold_atoms,
            atom_source=atom_source,
        )

    if status == "preparation_failed":
        return PreparationRecord(
            **common,
            status="preparation_failed",
            error=row["error"],
        )
    return PreparationRecord(**common, status="skipped_time_sensitive")


def _validate_expected_count(expected_count: Optional[int]) -> None:
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("preparation input item count must be a positive integer")


def _iter_preparation_rows(
    rows: Iterable[Dict[str, Any]], *, expected_count: Optional[int] = None
) -> Iterator[PreparationRecord]:
    _validate_expected_count(expected_count)
    seen_ids: Set[str] = set()
    record_count = 0
    for record_count, row in enumerate(rows, 1):
        record = _record_from_row(row, index=record_count)
        if record.item_id in seen_ids:
            raise ValueError("preparation artifact contains duplicate item IDs")
        seen_ids.add(record.item_id)
        yield record
    if record_count == 0:
        raise ValueError("preparation artifact must contain at least one row")
    if expected_count is not None and record_count != expected_count:
        raise ValueError(
            "preparation artifact must contain exactly one row per input item"
        )


def iter_preparation_records(
    path: Path, *, expected_count: Optional[int] = None
) -> Iterator[PreparationRecord]:
    """Validate and yield the authoritative preparation artifact incrementally."""
    yield from _iter_preparation_rows(
        iter_jsonl(path), expected_count=expected_count
    )


def load_preparation_records(
    path: Path, *, expected_count: Optional[int] = None
) -> List[PreparationRecord]:
    return list(iter_preparation_records(path, expected_count=expected_count))


def load_preparation_rows(
    rows: Iterable[Dict[str, Any]], *, expected_count: Optional[int] = None
) -> List[PreparationRecord]:
    """Validate a projected row stream using the canonical record contract."""
    return list(_iter_preparation_rows(rows, expected_count=expected_count))
