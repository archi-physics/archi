# isort: skip_file
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Optional, Protocol, Set, Tuple, Union

from ijson import JSONError
from ijson.backends import python as ijson_backend
from ijson.common import ObjectBuilder

from .oracle import (
    OracleRecipe,
    canonical_json,
    parse_oracle_recipe,
    validate_json_value,
)

DATASET_V2_SCHEMA_VERSION = "qa-dataset-v2"
ANSWER_MODE_VALUES = {"direct_answer", "needs_information", "escalate", "refuse"}
COMMON_ITEM_FIELDS = {
    "id",
    "question",
    "answer",
    "time_sensitive",
    "category",
    "answer_mode",
    "answer_source",
    "expected_atoms",
}
V1_ITEM_FIELDS = COMMON_ITEM_FIELDS
V2_ITEM_FIELDS = COMMON_ITEM_FIELDS | {"oracle"}


class DatasetSchemaVersion(str, Enum):
    V1 = "qa-dataset-v1"
    V2 = DATASET_V2_SCHEMA_VERSION


class DatasetItemState(str, Enum):
    STATIC = "static"
    LEGACY_TIME_SENSITIVE = "legacy_time_sensitive"
    UNRESOLVED_LIVE = "unresolved_live"
    MATERIALIZED_LIVE = "materialized_live"


class PhysicalFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"


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
    answer: Optional[Union[str, Dict[str, Any]]]
    time_sensitive: bool
    category: Optional[str]
    answer_mode: Optional[str]
    answer_source: Optional[str]
    expected_atoms: Optional[List[Atom]]
    oracle: Optional[OracleRecipe] = None
    schema_version: DatasetSchemaVersion = DatasetSchemaVersion.V1
    state: Optional[DatasetItemState] = None

    def __post_init__(self) -> None:
        if self.state is None:
            if self.schema_version is DatasetSchemaVersion.V1:
                state = (
                    DatasetItemState.LEGACY_TIME_SENSITIVE
                    if self.time_sensitive
                    else DatasetItemState.STATIC
                )
            elif self.time_sensitive and self.answer is None:
                state = DatasetItemState.UNRESOLVED_LIVE
            elif self.time_sensitive:
                state = DatasetItemState.MATERIALIZED_LIVE
            else:
                state = DatasetItemState.STATIC
            object.__setattr__(self, "state", state)

    @property
    def is_live(self) -> bool:
        return self.state in {
            DatasetItemState.UNRESOLVED_LIVE,
            DatasetItemState.MATERIALIZED_LIVE,
        }

    @property
    def is_materialized_live(self) -> bool:
        return self.state is DatasetItemState.MATERIALIZED_LIVE

    def answer_for_extraction(self) -> str:
        if isinstance(self.answer, str):
            return self.answer
        if isinstance(self.answer, dict):
            return canonical_json(self.answer)
        raise ValueError(f"dataset item '{self.id}' has no resolved answer")


@dataclass(frozen=True)
class ContainerDescriptor:
    physical_format: PhysicalFormat
    declared_schema_version: Optional[str]
    historical_headerless: bool
    first_jsonl_value: Optional[Any] = None
    first_jsonl_line_number: Optional[int] = None


@dataclass(frozen=True)
class DatasetRead:
    descriptor: ContainerDescriptor
    schema_version: DatasetSchemaVersion
    items: Iterator[DatasetItem]

    def __iter__(self) -> Iterator[DatasetItem]:
        return self.items


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


def validate_optional_nonempty_string(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    return validate_nonempty_string(value, context)


def validate_optional_enum(value: Any, allowed: set, context: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{context} must be one of: {', '.join(sorted(allowed))}")
    return value


def validate_atoms(raw_atoms: Any, *, context: str) -> List[Atom]:
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
        unknown = sorted(set(raw) - {"id", "text", "required"})
        missing = sorted({"id", "text", "required"} - set(raw))
        if unknown or missing:
            raise ValueError(f"{atom_context} has invalid fields")
        atom_id = validate_nonempty_string(raw["id"], f"{atom_context}.id")
        text = validate_nonempty_string(raw["text"], f"{atom_context}.text")
        required = raw["required"]
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


def dataset_source_format(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise ValueError(f"dataset must be an existing file: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("dataset must use .json or .jsonl")
    return suffix[1:]


def _first_non_whitespace(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                raise ValueError("invalid JSON dataset: empty input")
            for byte in chunk:
                if byte not in b" \t\r\n":
                    return byte


def _validate_json_structure(path: Path) -> None:
    """Reject malformed JSON and duplicate keys without materializing the file."""
    object_keys: List[Set[str]] = []
    try:
        with path.open("rb") as handle:
            for _prefix, event, value in ijson_backend.parse(handle, use_float=True):
                if event == "start_map":
                    object_keys.append(set())
                elif event == "map_key":
                    if not object_keys:
                        raise ValueError("invalid JSON dataset: map key outside object")
                    if value in object_keys[-1]:
                        raise ValueError(
                            f"dataset JSON contains duplicate key '{value}'"
                        )
                    object_keys[-1].add(value)
                elif event == "end_map":
                    object_keys.pop()
    except JSONError as exc:
        raise ValueError(f"invalid JSON dataset: {exc}") from exc


def _inspect_v2_json(path: Path) -> ContainerDescriptor:
    top_keys: Set[str] = set()
    declared: Any = None
    items_is_array = False
    try:
        with path.open("rb") as handle:
            for prefix, event, value in ijson_backend.parse(handle, use_float=True):
                if prefix == "" and event == "map_key":
                    if value in top_keys:
                        raise ValueError(
                            f"dataset JSON object contains duplicate key '{value}'"
                        )
                    top_keys.add(value)
                elif prefix == "schema_version" and event in {
                    "string",
                    "null",
                    "boolean",
                    "number",
                }:
                    declared = value
                elif prefix == "items" and event == "start_array":
                    items_is_array = True
                if (
                    top_keys == {"schema_version", "items"}
                    and items_is_array
                    and isinstance(declared, str)
                ):
                    return ContainerDescriptor(
                        physical_format=PhysicalFormat.JSON,
                        declared_schema_version=declared,
                        historical_headerless=False,
                    )
    except JSONError as exc:
        raise ValueError(f"invalid JSON dataset: {exc}") from exc
    if "schema_version" not in top_keys:
        raise ValueError(
            "dataset JSON must contain an array or an explicit Dataset V2 object"
        )
    if top_keys != {"schema_version", "items"} or not items_is_array:
        raise ValueError(
            "Dataset V2 JSON must contain exactly schema_version and items"
        )
    if not isinstance(declared, str):
        raise ValueError("dataset schema_version must be a string")
    return ContainerDescriptor(
        physical_format=PhysicalFormat.JSON,
        declared_schema_version=declared,
        historical_headerless=False,
    )


def _strict_json_line(line: str, line_number: int) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite number {value}")

    def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key '{key}'")
            value[key] = item
        return value

    try:
        return json.loads(
            line,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise ValueError(f"invalid JSONL at line {line_number}: {message}") from exc


def _first_jsonl_value(path: Path) -> Tuple[int, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    return line_number, _strict_json_line(line, line_number)
    except UnicodeDecodeError as exc:
        raise ValueError("dataset must be UTF-8 encoded") from exc
    raise ValueError("dataset must contain at least one row")


class PhysicalDatasetCodec(Protocol):
    def describe(self, path: Path) -> ContainerDescriptor: ...

    def iter_rows(
        self, path: Path, descriptor: ContainerDescriptor
    ) -> Iterator[Any]: ...


class JsonDatasetCodec:
    def describe(self, path: Path) -> ContainerDescriptor:
        first = _first_non_whitespace(path)
        if first == ord("["):
            return ContainerDescriptor(
                physical_format=PhysicalFormat.JSON,
                declared_schema_version=None,
                historical_headerless=True,
            )
        if first == ord("{"):
            return _inspect_v2_json(path)
        raise ValueError("dataset JSON must contain an array or Dataset V2 object")

    def iter_rows(self, path: Path, descriptor: ContainerDescriptor) -> Iterator[Any]:
        prefix = "item" if descriptor.historical_headerless else "items.item"
        try:
            with path.open("rb") as handle:
                builder: Optional[ObjectBuilder] = None
                depth = 0
                object_keys: List[Set[str]] = []
                top_keys: Set[str] = set()
                for event_prefix, event, value in ijson_backend.parse(
                    handle, use_float=True
                ):
                    if not descriptor.historical_headerless:
                        if event_prefix == "" and event == "map_key":
                            if value in top_keys:
                                raise ValueError(
                                    f"dataset JSON object contains duplicate key '{value}'"
                                )
                            if value not in {"schema_version", "items"}:
                                raise ValueError(
                                    "Dataset V2 JSON must contain exactly "
                                    "schema_version and items"
                                )
                            top_keys.add(value)
                    if builder is None:
                        if event_prefix != prefix:
                            continue
                        if event not in {"start_map", "start_array"}:
                            yield value
                            continue
                        builder = ObjectBuilder()
                        depth = 1
                    elif event in {"start_map", "start_array"}:
                        depth += 1

                    if event == "start_map":
                        object_keys.append(set())
                    elif event == "map_key":
                        if not object_keys:
                            raise ValueError(
                                "invalid JSON dataset: map key outside object"
                            )
                        if value in object_keys[-1]:
                            raise ValueError(
                                f"dataset JSON contains duplicate key '{value}'"
                            )
                        object_keys[-1].add(value)

                    builder.event(event, value)

                    if event == "end_map":
                        object_keys.pop()
                    if event in {"end_map", "end_array"}:
                        depth -= 1
                        if depth == 0:
                            yield builder.value
                            builder = None
                if not descriptor.historical_headerless and top_keys != {
                    "schema_version",
                    "items",
                }:
                    raise ValueError(
                        "Dataset V2 JSON must contain exactly schema_version and items"
                    )
        except JSONError as exc:
            raise ValueError(f"invalid JSON dataset: {exc}") from exc


class JsonlDatasetCodec:
    @staticmethod
    def descriptor_from_first(line_number: int, first: Any) -> ContainerDescriptor:
        if isinstance(first, dict) and "schema_version" in first:
            if set(first) != {"schema_version"}:
                raise ValueError(
                    "Dataset V2 JSONL header must contain exactly schema_version"
                )
            declared = first["schema_version"]
            if not isinstance(declared, str):
                raise ValueError("dataset schema_version must be a string")
            return ContainerDescriptor(
                physical_format=PhysicalFormat.JSONL,
                declared_schema_version=declared,
                historical_headerless=False,
                first_jsonl_line_number=line_number,
            )
        return ContainerDescriptor(
            physical_format=PhysicalFormat.JSONL,
            declared_schema_version=None,
            historical_headerless=True,
            first_jsonl_value=first,
            first_jsonl_line_number=line_number,
        )

    @staticmethod
    def iter_numbered_rows(path: Path) -> Iterator[Tuple[int, Any]]:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        yield line_number, _strict_json_line(line, line_number)
        except UnicodeDecodeError as exc:
            raise ValueError("dataset must be UTF-8 encoded") from exc

    def describe(self, path: Path) -> ContainerDescriptor:
        line_number, first = _first_jsonl_value(path)
        return self.descriptor_from_first(line_number, first)

    def iter_rows(self, path: Path, descriptor: ContainerDescriptor) -> Iterator[Any]:
        try:
            assert descriptor.first_jsonl_line_number is not None
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    if line_number < descriptor.first_jsonl_line_number:
                        continue
                    if line_number == descriptor.first_jsonl_line_number:
                        if descriptor.historical_headerless:
                            yield descriptor.first_jsonl_value
                        continue
                    yield _strict_json_line(line, line_number)
        except UnicodeDecodeError as exc:
            raise ValueError("dataset must be UTF-8 encoded") from exc


class DatasetVersionResolver:
    def resolve(self, descriptor: ContainerDescriptor) -> DatasetSchemaVersion:
        if descriptor.declared_schema_version is not None:
            if descriptor.declared_schema_version == DATASET_V2_SCHEMA_VERSION:
                return DatasetSchemaVersion.V2
            raise ValueError(
                "unsupported dataset schema_version "
                f"'{descriptor.declared_schema_version}'"
            )
        if descriptor.historical_headerless:
            return DatasetSchemaVersion.V1
        raise ValueError("dataset container has no supported schema version")


def _strict_row_keys(raw: Dict[str, Any], allowed: set, context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _common_fields(raw: Dict[str, Any], context: str) -> Dict[str, Any]:
    question = validate_nonempty_string(
        raw.get("question"), f"{context}.question", normalize_newlines=True
    )
    time_sensitive = raw.get("time_sensitive")
    if not isinstance(time_sensitive, bool):
        raise ValueError(f"{context}.time_sensitive must be a boolean")
    return {
        "question": question,
        "time_sensitive": time_sensitive,
        "category": validate_optional_nonempty_string(
            raw.get("category"), f"{context}.category"
        ),
        "answer_mode": validate_optional_enum(
            raw.get("answer_mode"), ANSWER_MODE_VALUES, f"{context}.answer_mode"
        ),
        "answer_source": validate_optional_nonempty_string(
            raw.get("answer_source"), f"{context}.answer_source"
        ),
    }


class V1DatasetReader:
    def read_row(self, raw: Any, *, index: int) -> DatasetItem:
        context = f"dataset row {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        _strict_row_keys(raw, V1_ITEM_FIELDS, context)
        common = _common_fields(raw, context)
        answer = validate_nonempty_string(
            raw.get("answer"), f"{context}.answer", normalize_newlines=True
        )
        explicit_id = raw.get("id")
        item_id = (
            validate_nonempty_string(explicit_id, f"{context}.id")
            if explicit_id is not None
            else derive_item_id(common["question"], answer)
        )
        atoms = (
            validate_atoms(raw["expected_atoms"], context=f"{context}.expected_atoms")
            if "expected_atoms" in raw
            else None
        )
        return DatasetItem(
            id=item_id,
            answer=answer,
            expected_atoms=atoms,
            schema_version=DatasetSchemaVersion.V1,
            **common,
        )


class V2DatasetReader:
    def __init__(self, *, allow_materialized_live: bool):
        self._allow_materialized_live = allow_materialized_live

    def read_row(self, raw: Any, *, index: int) -> DatasetItem:
        context = f"dataset row {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        _strict_row_keys(raw, V2_ITEM_FIELDS, context)
        common = _common_fields(raw, context)
        item_id = validate_nonempty_string(raw.get("id"), f"{context}.id")
        time_sensitive = common["time_sensitive"]
        if not time_sensitive:
            if "oracle" in raw:
                raise ValueError(f"{context} static item must not contain oracle")
            answer = validate_nonempty_string(
                raw.get("answer"), f"{context}.answer", normalize_newlines=True
            )
            atoms = (
                validate_atoms(
                    raw["expected_atoms"], context=f"{context}.expected_atoms"
                )
                if "expected_atoms" in raw
                else None
            )
            return DatasetItem(
                id=item_id,
                answer=answer,
                expected_atoms=atoms,
                schema_version=DatasetSchemaVersion.V2,
                **common,
            )
        if "oracle" not in raw:
            raise ValueError(f"{context} live item requires oracle")
        recipe = parse_oracle_recipe(raw["oracle"], f"{context}.oracle")
        has_answer = "answer" in raw
        has_atoms = "expected_atoms" in raw
        if not has_answer and not has_atoms:
            return DatasetItem(
                id=item_id,
                answer=None,
                expected_atoms=None,
                oracle=recipe,
                schema_version=DatasetSchemaVersion.V2,
                **common,
            )
        if has_atoms and not has_answer:
            raise ValueError(f"{context} live item cannot contain atoms without answer")
        if not has_atoms:
            raise ValueError(
                f"{context} materialized live item requires expected_atoms"
            )
        answer = raw["answer"]
        if not isinstance(answer, dict) or not answer:
            raise ValueError(
                f"{context} materialized live answer must be a non-empty object"
            )
        validate_json_value(answer, f"{context}.answer")
        if not self._allow_materialized_live:
            raise ValueError(
                "external materialized live item is not portable; import the "
                "unresolved parent and materialize it locally"
            )
        return DatasetItem(
            id=item_id,
            answer=answer,
            expected_atoms=validate_atoms(
                raw["expected_atoms"], context=f"{context}.expected_atoms"
            ),
            oracle=recipe,
            schema_version=DatasetSchemaVersion.V2,
            **common,
        )


class DatasetGateway:
    def __init__(self) -> None:
        self._codecs: Dict[PhysicalFormat, PhysicalDatasetCodec] = {
            PhysicalFormat.JSON: JsonDatasetCodec(),
            PhysicalFormat.JSONL: JsonlDatasetCodec(),
        }
        self._resolver = DatasetVersionResolver()
        self._reader_factories = {
            DatasetSchemaVersion.V1: lambda _allow: V1DatasetReader(),
            DatasetSchemaVersion.V2: lambda allow: V2DatasetReader(
                allow_materialized_live=allow
            ),
        }

    def read(self, path: Path, *, allow_materialized_live: bool = False) -> DatasetRead:
        source_format = PhysicalFormat(dataset_source_format(path))
        codec = self._codecs[source_format]
        if isinstance(codec, JsonlDatasetCodec):
            numbered_rows = codec.iter_numbered_rows(path)
            try:
                first_line_number, first_value = next(numbered_rows)
            except StopIteration:
                raise ValueError("dataset must contain at least one row") from None
            descriptor = codec.descriptor_from_first(first_line_number, first_value)

            def raw_rows() -> Iterator[Any]:
                if descriptor.historical_headerless:
                    yield first_value
                for _line_number, value in numbered_rows:
                    yield value

        else:
            descriptor = codec.describe(path)

            def raw_rows() -> Iterator[Any]:
                yield from codec.iter_rows(path, descriptor)

        schema_version = self._resolver.resolve(descriptor)
        reader = self._reader_factories[schema_version](allow_materialized_live)

        def items() -> Iterator[DatasetItem]:
            seen_ids: Set[str] = set()
            count = 0
            for count, raw in enumerate(raw_rows(), 1):
                item = reader.read_row(raw, index=count)
                if item.id in seen_ids:
                    raise ValueError(
                        f"dataset contains duplicate or colliding id '{item.id}'"
                    )
                seen_ids.add(item.id)
                yield item
            if count == 0:
                raise ValueError("dataset must contain at least one row")

        return DatasetRead(
            descriptor=descriptor,
            schema_version=schema_version,
            items=items(),
        )


DEFAULT_DATASET_GATEWAY = DatasetGateway()


def iter_dataset_items(
    path: Path, *, allow_materialized_live: bool = False
) -> Iterator[DatasetItem]:
    yield from DEFAULT_DATASET_GATEWAY.read(
        path, allow_materialized_live=allow_materialized_live
    )


def validate_dataset_rows(raw_rows: Any) -> List[DatasetItem]:
    if not isinstance(raw_rows, list):
        raise ValueError("dataset JSON must contain an array")
    if not raw_rows:
        raise ValueError("dataset must contain at least one row")
    reader = V1DatasetReader()
    seen: Set[str] = set()
    items = []
    for index, raw in enumerate(raw_rows, 1):
        item = reader.read_row(raw, index=index)
        if item.id in seen:
            raise ValueError(f"dataset contains duplicate or colliding id '{item.id}'")
        seen.add(item.id)
        items.append(item)
    return items


def load_dataset_bytes(
    filename: str, raw_bytes: bytes, *, allow_materialized_live: bool = False
) -> Tuple[str, List[DatasetItem], bytes]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("dataset must use .json or .jsonl")
    with TemporaryDirectory(prefix="archi-qa-dataset-") as directory:
        path = Path(directory) / f"source{suffix}"
        path.write_bytes(raw_bytes)
        items = list(
            iter_dataset_items(path, allow_materialized_live=allow_materialized_live)
        )
    return suffix[1:], items, raw_bytes


def load_dataset(
    path: Path, *, allow_materialized_live: bool = False
) -> Tuple[str, List[DatasetItem], bytes]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"dataset must be an existing file: {path}")
    return load_dataset_bytes(
        path.name,
        path.read_bytes(),
        allow_materialized_live=allow_materialized_live,
    )


def dataset_item_to_dict(item: DatasetItem) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "answer": deepcopy(item.answer),
        "time_sensitive": item.time_sensitive,
    }
    if item.answer is None:
        value.pop("answer")
    for name, field_value in (
        ("category", item.category),
        ("answer_mode", item.answer_mode),
        ("answer_source", item.answer_source),
    ):
        if field_value is not None:
            value[name] = field_value
    if item.expected_atoms is not None:
        value["expected_atoms"] = [atom.to_dict() for atom in item.expected_atoms]
    if item.oracle is not None:
        value["oracle"] = item.oracle.to_dict()
    return value


def v2_json_document(items: Iterator[DatasetItem]) -> Iterator[str]:
    yield '{"schema_version":"qa-dataset-v2","items":['
    first = True
    for item in items:
        if not first:
            yield ","
        first = False
        yield canonical_json(dataset_item_to_dict(item))
    yield "]}\n"
