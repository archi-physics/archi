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
    Union,
)

from .artifacts import iter_jsonl
from .oracle import (
    DIAGNOSTIC_LIMIT,
    OracleCallEvidence,
    OracleRecipe,
    OracleResolutionError,
    OracleResolver,
    answer_sha256,
    bounded_diagnostic,
    canonical_json,
    oracle_call_evidence_from_dict,
    parse_oracle_recipe,
    validate_json_value,
)
from .validation import (
    ANSWER_MODE_VALUES,
    Atom,
    DatasetItem,
    DatasetItemState,
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
    "skipped_live",
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
    answer: Optional[Union[str, Dict[str, Any]]] = None
    time_sensitive: Optional[bool] = None
    gold_atoms: Optional[Tuple[Atom, ...]] = None
    atom_source: Optional[AtomSource] = None
    error: Optional[str] = None
    oracle: Optional[OracleRecipe] = None
    answer_sha256: Optional[str] = None
    oracle_metadata: Optional[Dict[str, Any]] = None
    oracle_calls: Optional[Tuple[OracleCallEvidence, ...]] = None

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
            "skipped_live",
            "preparation_failed",
        }:
            raise ValueError("unsupported preparation status")
        if self.status == "prepared":
            normalized_question = validate_nonempty_string(
                self.question,
                "prepared record question",
                normalize_newlines=True,
            )
            if normalized_question != self.question:
                raise ValueError("prepared record text must use normalized newlines")
            if not self.gold_atoms:
                raise ValueError("prepared record requires gold atoms")
            if self.atom_source not in {"supplied", "inferred"}:
                raise ValueError("prepared record requires an atom source")
            if self.error is not None:
                raise ValueError("prepared record cannot contain an error")
            if self.time_sensitive is False:
                normalized_answer = validate_nonempty_string(
                    self.answer,
                    "prepared record answer",
                    normalize_newlines=True,
                )
                if normalized_answer != self.answer:
                    raise ValueError(
                        "prepared record text must use normalized newlines"
                    )
                if any(
                    value is not None
                    for value in (
                        self.oracle,
                        self.answer_sha256,
                        self.oracle_metadata,
                        self.oracle_calls,
                    )
                ):
                    raise ValueError("static prepared record has live-only fields")
                return
            if self.time_sensitive is not True:
                raise ValueError("prepared record requires time_sensitive boolean")
            if (
                not isinstance(self.answer, dict)
                or not self.answer
                or self.oracle is None
                or self.answer_sha256 != answer_sha256(self.answer)
                or not isinstance(self.oracle_metadata, dict)
                or not isinstance(self.oracle_calls, tuple)
            ):
                raise ValueError("live prepared record has invalid live truth fields")
            validate_json_value(self.oracle_metadata, "live prepared oracle_metadata")
            return
        if self.status == "preparation_failed":
            if self.error is None:
                raise ValueError("failed preparation requires an error")
            if (
                not isinstance(self.error, str)
                or not self.error
                or len(self.error) > DIAGNOSTIC_LIMIT
            ):
                raise ValueError(
                    "failed preparation error must be bounded non-empty text"
                )
            validate_nonempty_string(self.error, "failed preparation error")
            if (
                self.question is not None
                or self.answer is not None
                or self.time_sensitive is not None
                or self.gold_atoms is not None
                or self.atom_source is not None
                or self.oracle is not None
                or self.answer_sha256 is not None
                or self.oracle_metadata is not None
                or (
                    self.oracle_calls is not None
                    and not isinstance(self.oracle_calls, tuple)
                )
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
            or self.oracle is not None
            or self.answer_sha256 is not None
            or self.oracle_metadata is not None
            or self.oracle_calls is not None
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
    def prepared_answer(self) -> Union[str, Dict[str, Any]]:
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
            if self.time_sensitive:
                assert self.oracle is not None
                assert self.answer_sha256 is not None
                assert self.oracle_metadata is not None
                assert self.oracle_calls is not None
                row.update(
                    {
                        "oracle": self.oracle.to_dict(),
                        "answer_sha256": self.answer_sha256,
                        "oracle_metadata": self.oracle_metadata,
                        "oracle_calls": [
                            evidence.to_dict() for evidence in self.oracle_calls
                        ],
                    }
                )
        elif self.status == "preparation_failed":
            row["error"] = self.error
            if self.oracle_calls is not None:
                row["oracle_calls"] = [
                    evidence.to_dict() for evidence in self.oracle_calls
                ]
        return row


def prepare_dataset_items(
    items: Sequence[DatasetItem], extractor: GoldExtractor
) -> List[PreparationRecord]:
    """Prepare one immutable terminal record for every validated dataset item."""
    return [prepare_dataset_item(item, extractor) for item in items]


def prepare_dataset_item(
    item: DatasetItem,
    extractor: GoldExtractor,
    oracle_resolver: Optional[OracleResolver] = None,
    *,
    skip_live: bool = False,
) -> PreparationRecord:
    """Prepare one validated dataset item into one terminal record."""
    if item.state is DatasetItemState.LEGACY_TIME_SENSITIVE:
        return PreparationRecord(
            item_id=item.id,
            status="skipped_time_sensitive",
            category=item.category,
            answer_mode=item.answer_mode,
            answer_source=item.answer_source,
        )
    if item.is_live and skip_live:
        return PreparationRecord(
            item_id=item.id,
            status="skipped_live",
            category=item.category,
            answer_mode=item.answer_mode,
            answer_source=item.answer_source,
        )
    try:
        resolved = None
        if item.state is DatasetItemState.UNRESOLVED_LIVE:
            if oracle_resolver is None or item.oracle is None:
                raise ValueError("live preparation requires an evaluator MCP registry")
            resolved = oracle_resolver.resolve(item.oracle)
            answer: Union[str, Dict[str, Any]] = resolved.answer
        elif item.state is DatasetItemState.MATERIALIZED_LIVE:
            if not isinstance(item.answer, dict) or item.oracle is None:
                raise ValueError("materialized live item is incomplete")
            answer = item.answer
        else:
            answer = item.answer_for_extraction()
        if item.expected_atoms is not None:
            gold_atoms = item.expected_atoms
            atom_source: AtomSource = "supplied"
        else:
            gold_atoms = validate_gold_output(
                extractor.extract_gold(
                    item.question,
                    (
                        item.answer_for_extraction()
                        if isinstance(answer, str)
                        else canonical_json(answer)
                    ),
                ),
                context=f"gold extraction for item {item.id}",
            )
            atom_source = "inferred"
    except Exception as exc:
        if isinstance(exc, OracleResolutionError):
            detail: Any = exc.detail
        elif item.is_live:
            detail = f"Gold extraction failed ({type(exc).__name__})."
        else:
            detail = exc
        return PreparationRecord(
            item_id=item.id,
            status="preparation_failed",
            category=item.category,
            answer_mode=item.answer_mode,
            answer_source=item.answer_source,
            error=bounded_diagnostic(detail),
            oracle_calls=(
                exc.calls
                if isinstance(exc, OracleResolutionError)
                else (resolved.calls if resolved is not None else None)
            ),
        )
    live_fields: Dict[str, Any] = {}
    if item.is_live:
        assert isinstance(answer, dict)
        assert item.oracle is not None
        live_fields = {
            "oracle": item.oracle,
            "answer_sha256": (
                resolved.answer_sha256
                if resolved is not None
                else answer_sha256(answer)
            ),
            "oracle_metadata": resolved.metadata if resolved is not None else {},
            "oracle_calls": resolved.calls if resolved is not None else (),
        }
    return PreparationRecord(
        item_id=item.id,
        status="prepared",
        category=item.category,
        answer_mode=item.answer_mode,
        answer_source=item.answer_source,
        question=item.question,
        answer=answer,
        time_sensitive=item.time_sensitive,
        gold_atoms=tuple(gold_atoms),
        atom_source=atom_source,
        **live_fields,
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
        "skipped_live",
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
        "skipped_live": set(),
    }[status]
    if status == "prepared" and row.get("time_sensitive") is True:
        status_fields |= {
            "oracle",
            "answer_sha256",
            "oracle_metadata",
            "oracle_calls",
        }
    if status == "preparation_failed" and "oracle_calls" in row:
        status_fields.add("oracle_calls")
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
            oracle=(
                parse_oracle_recipe(row["oracle"], f"{context}.oracle")
                if row["time_sensitive"] is True
                else None
            ),
            answer_sha256=row.get("answer_sha256"),
            oracle_metadata=row.get("oracle_metadata"),
            oracle_calls=(
                tuple(
                    oracle_call_evidence_from_dict(
                        evidence,
                        f"{context}.oracle_calls[{call_index}]",
                    )
                    for call_index, evidence in enumerate(row.get("oracle_calls", []))
                )
                if row["time_sensitive"] is True
                else None
            ),
        )

    if status == "preparation_failed":
        return PreparationRecord(
            **common,
            status="preparation_failed",
            error=row["error"],
            oracle_calls=(
                tuple(
                    oracle_call_evidence_from_dict(
                        evidence,
                        f"{context}.oracle_calls[{call_index}]",
                    )
                    for call_index, evidence in enumerate(row.get("oracle_calls", []))
                )
                if "oracle_calls" in row
                else None
            ),
        )
    return PreparationRecord(**common, status=status)


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
    yield from _iter_preparation_rows(iter_jsonl(path), expected_count=expected_count)


def preparation_record_from_dict(
    row: Dict[str, Any], *, context_index: int = 1
) -> PreparationRecord:
    """Validate one already-decoded preparation row."""
    return _record_from_row(row, index=context_index)


def load_preparation_records(
    path: Path, *, expected_count: Optional[int] = None
) -> List[PreparationRecord]:
    return list(iter_preparation_records(path, expected_count=expected_count))


def load_preparation_rows(
    rows: Iterable[Dict[str, Any]], *, expected_count: Optional[int] = None
) -> List[PreparationRecord]:
    """Validate a projected row stream using the canonical record contract."""
    return list(_iter_preparation_rows(rows, expected_count=expected_count))
