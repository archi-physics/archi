from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ijson import JSONError
from ijson.backends import python as ijson_backend

DATASET_FIELDS = {
    "id",
    "question",
    "answer",
    "time_sensitive",
    "category",
    "answer_mode",
    "answer_source",
    "expected_atoms",
}
ANSWER_MODE_VALUES = {"direct_answer", "needs_information", "escalate", "refuse"}
OUTCOME_VALUES = {"entailed", "not_mentioned", "contradicted", "unjudgeable"}


@dataclass(frozen=True)
class Atom:
    id: str
    text: str
    required: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetItem:
    id: str
    question: str
    answer: str
    time_sensitive: bool
    category: Optional[str]
    answer_mode: Optional[str]
    answer_source: Optional[str]
    expected_atoms: Optional[List[Atom]]


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


def validate_nonempty_string(
    value: Any, context: str, *, normalize_newlines: bool = False
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{context} must not contain NUL characters")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise ValueError(f"{context} must contain valid Unicode scalar values")
    if normalize_newlines:
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return value


def validate_optional_enum(value: Any, allowed: set, context: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{context} must be one of: {', '.join(sorted(allowed))}")
    return value


def validate_optional_nonempty_string(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    return validate_nonempty_string(value, context)


def validate_atoms(
    raw_atoms: Any,
    *,
    context: str,
) -> List[Atom]:
    if not isinstance(raw_atoms, list):
        raise ValueError(f"{context} must be a list")
    if not raw_atoms:
        raise ValueError(f"{context} must contain at least one atom")
    atoms: List[Atom] = []
    seen = set()
    for index, raw in enumerate(raw_atoms):
        atom_context = f"{context}[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{atom_context} must be an object")
        _strict_keys(raw, {"id", "text", "required"}, atom_context)
        atom_id = validate_nonempty_string(raw.get("id"), f"{atom_context}.id")
        text = validate_nonempty_string(raw.get("text"), f"{atom_context}.text")
        required = raw.get("required")
        if not isinstance(required, bool):
            raise ValueError(f"{atom_context}.required must be a boolean")
        if atom_id in seen:
            raise ValueError(f"{context} contains duplicate atom id '{atom_id}'")
        seen.add(atom_id)
        atoms.append(Atom(id=atom_id, text=text, required=required))
    if not any(atom.required for atom in atoms):
        raise ValueError(f"{context} must contain at least one required atom")
    return atoms


def derive_item_id(question: str, answer: str) -> str:
    canonical = json.dumps(
        {"answer": answer, "question": question},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"qa-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def _validate_dataset_row(raw: Any, *, index: int, seen_ids: Set[str]) -> DatasetItem:
    context = f"dataset row {index}"
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    _strict_keys(raw, DATASET_FIELDS, context)
    question = validate_nonempty_string(
        raw.get("question"), f"{context}.question", normalize_newlines=True
    )
    answer = validate_nonempty_string(
        raw.get("answer"),
        f"{context}.answer",
        normalize_newlines=True,
    )
    time_sensitive = raw.get("time_sensitive")
    if not isinstance(time_sensitive, bool):
        raise ValueError(f"{context}.time_sensitive must be a boolean")
    explicit_id = raw.get("id")
    item_id = (
        validate_nonempty_string(explicit_id, f"{context}.id")
        if explicit_id is not None
        else derive_item_id(question, answer)
    )
    if item_id in seen_ids:
        raise ValueError(f"dataset contains duplicate or colliding id '{item_id}'")
    seen_ids.add(item_id)
    expected_atoms = None
    if "expected_atoms" in raw:
        expected_atoms = validate_atoms(
            raw["expected_atoms"],
            context=f"{context}.expected_atoms",
        )
    return DatasetItem(
        id=item_id,
        question=question,
        answer=answer,
        time_sensitive=time_sensitive,
        category=validate_optional_nonempty_string(
            raw.get("category"), f"{context}.category"
        ),
        answer_mode=validate_optional_enum(
            raw.get("answer_mode"), ANSWER_MODE_VALUES, f"{context}.answer_mode"
        ),
        answer_source=validate_optional_nonempty_string(
            raw.get("answer_source"), f"{context}.answer_source"
        ),
        expected_atoms=expected_atoms,
    )


def validate_dataset_rows(raw_rows: Any) -> List[DatasetItem]:
    if not isinstance(raw_rows, list):
        raise ValueError("dataset JSON must contain an array")
    if not raw_rows:
        raise ValueError("dataset must contain at least one row")
    items: List[DatasetItem] = []
    seen_ids: Set[str] = set()
    for index, raw in enumerate(raw_rows, 1):
        items.append(_validate_dataset_row(raw, index=index, seen_ids=seen_ids))
    return items


def _iter_json_rows(path: Path) -> Iterator[Any]:
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                first_token = next(
                    (byte for byte in chunk if byte not in b" \t\r\n"),
                    None,
                )
                if first_token is not None:
                    if first_token != ord("["):
                        raise ValueError("dataset JSON must contain an array")
                    break
            else:
                raise ValueError("invalid JSON dataset: empty input")
            handle.seek(0)
            # YAJL replaces unmatched surrogate escapes; the Python backend
            # preserves them so the shared string validator can reject them.
            yield from ijson_backend.items(handle, "item")
    except JSONError as exc:
        raise ValueError(f"invalid JSON dataset: {exc}") from exc


def _iter_jsonl_rows(path: Path) -> Iterator[Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at line {line_number}: {exc.msg}"
                    ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError("dataset must be UTF-8 encoded") from exc


def dataset_source_format(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise ValueError(f"dataset must be an existing file: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("dataset must use .json or .jsonl")
    return suffix[1:]


def iter_dataset_items(path: Path) -> Iterator[DatasetItem]:
    """Validate and yield JSON or JSONL rows without loading the file at once."""
    source_format = dataset_source_format(path)
    if source_format == "json":
        raw_rows = _iter_json_rows(path)
    else:
        raw_rows = _iter_jsonl_rows(path)

    seen_ids: Set[str] = set()
    item_index = 0
    for raw in raw_rows:
        item_index += 1
        yield _validate_dataset_row(
            raw,
            index=item_index,
            seen_ids=seen_ids,
        )
    if item_index == 0:
        raise ValueError("dataset must contain at least one row")


def load_dataset_bytes(
    filename: str, raw_bytes: bytes
) -> Tuple[str, List[DatasetItem], bytes]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("dataset must use .json or .jsonl")
    try:
        text = raw_bytes.decode("utf-8")
        if suffix == ".json":
            rows = json.loads(text)
        else:
            rows = []
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at line {line_number}: {exc.msg}"
                    ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError("dataset must be UTF-8 encoded") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON dataset: {exc.msg}") from exc
    return suffix[1:], validate_dataset_rows(rows), raw_bytes


def load_dataset(path: Path) -> Tuple[str, List[DatasetItem], bytes]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"dataset must be an existing file: {path}")
    return load_dataset_bytes(path.name, path.read_bytes())


def validate_gold_output(raw: Any, *, context: str) -> List[Atom]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    _strict_keys(raw, {"atoms"}, context)
    return validate_atoms(
        raw.get("atoms"),
        context=f"{context}.atoms",
    )


def validate_judgments(
    raw: Any,
    *,
    gold_atoms: Sequence[Atom],
    context: str,
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
            Judgment(
                atom_id=atom_id,
                outcome=outcome,
                rationale=rationale,
            )
        )
    missing = sorted(gold_ids - seen)
    if missing:
        raise ValueError(f"{context} is missing judgment(s) for: {', '.join(missing)}")
    return judgments
