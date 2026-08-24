# isort: skip_file
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import uuid
from enum import Enum
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .artifacts import (
    AtomicJsonlWriter,
    copy_file_atomic,
    iter_jsonl,
    read_json,
    sha256_file,
    utc_now,
    write_bytes,
    write_json,
)
from .dataset import (
    DATASET_V2_SCHEMA_VERSION,
    DatasetItemState,
    dataset_item_to_dict,
    iter_dataset_items,
)
from .oracle import OracleResolver
from .preparation import (
    GoldExtractor,
    PreparationRecord,
    prepare_dataset_item,
)
from .profile import DEFAULT_PROFILE, _parse_profile
from .validation import (  # isort: skip
    DatasetItem,
    validate_atoms,
)

MAX_IMPORT_BYTES = 25 * 1024 * 1024
CATALOG_INTEGRITY_SCHEMA_VERSION = "qa-catalog-integrity-v1"
LEGACY_CATALOG_METADATA_FIELDS = {
    "id",
    "name",
    "source_filename",
    "format",
    "sha256",
    "item_count",
    "eligible_item_count",
    "time_sensitive_item_count",
    "supplied_atom_item_count",
    "atom_count",
    "categories",
    "answer_sources",
    "parent_dataset_id",
    "created_at",
}


class DatasetRole(str, Enum):
    LEGACY = "legacy"
    DEFINITION_PARENT = "definition_parent"
    APPROVED_CHILD = "approved_child"


def dataset_role(metadata: Dict[str, Any]) -> DatasetRole:
    try:
        return DatasetRole(metadata.get("dataset_role"))
    except ValueError as exc:
        raise ValueError("dataset catalog metadata has an invalid role") from exc


def _validate_catalog_role(metadata: Dict[str, Any]) -> None:
    role = dataset_role(metadata)
    parent_id = metadata.get("parent_dataset_id")
    schema_version = metadata.get("schema_version")
    if role is DatasetRole.LEGACY:
        valid = schema_version != DATASET_V2_SCHEMA_VERSION and parent_id is None
    elif role is DatasetRole.DEFINITION_PARENT:
        valid = schema_version == DATASET_V2_SCHEMA_VERSION and parent_id is None
    else:
        valid = (
            isinstance(parent_id, str)
            and schema_version in {"qa-dataset-v1", DATASET_V2_SCHEMA_VERSION}
            and metadata.get("generation_scope")
            in {"review", "complete", "static_only", "refresh_live", "add_live"}
        )
    if not valid:
        raise ValueError("dataset catalog metadata role conflicts with lineage")
    if not isinstance(metadata.get("contains_live_answers"), bool):
        raise ValueError("dataset catalog metadata has invalid live content state")


def _catalog_id(value: str, *, allow_builtin: bool = False) -> str:
    if allow_builtin and value == "builtin":
        return value
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("invalid catalog identifier") from exc
    if str(parsed) != value:
        raise ValueError("invalid catalog identifier")
    return value


def _display_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    cleaned = value.strip()
    if len(cleaned) > 160 or "\x00" in cleaned:
        raise ValueError(f"{context} is invalid")
    return cleaned


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _dataset_row(
    item: DatasetItem, atoms: Optional[Sequence[Dict[str, Any]]]
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "answer": item.answer,
        "time_sensitive": item.time_sensitive,
    }
    for field in ("category", "answer_mode", "answer_source"):
        value = getattr(item, field)
        if value is not None:
            row[field] = value
    if atoms is not None:
        row["expected_atoms"] = list(atoms)
    if item.oracle is not None:
        row["oracle"] = item.oracle.to_dict()
    return row


def _generated_draft_row(
    item: DatasetItem,
    record: PreparationRecord,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "item_id": item.id,
        "question": item.question,
        "answer": item.answer,
        "time_sensitive": item.time_sensitive,
        "status": record.status,
    }
    if item.answer_source is not None:
        row["answer_source"] = item.answer_source
    if record.status == "prepared":
        row["atom_source"] = record.atom_source
        row["atoms"] = [atom.to_dict() for atom in record.prepared_gold_atoms]
        if record.time_sensitive:
            row.update(
                {
                    "answer": record.prepared_answer,
                    "answer_sha256": record.answer_sha256,
                    "oracle": record.oracle.to_dict(),
                    "oracle_metadata": record.oracle_metadata,
                    "oracle_calls": [
                        call.to_dict() for call in (record.oracle_calls or ())
                    ],
                    "live_state": "resolved",
                }
            )
    elif record.status == "skipped_live":
        row["live_state"] = "omitted"
    elif record.error is not None:
        row["error"] = record.error
    return row


class _ReviewedChildPublisher:
    """Serialize an approved child in its selected persistence format."""

    def __init__(self, publication_schema: str):
        self._is_v2 = publication_schema == DATASET_V2_SCHEMA_VERSION

    @property
    def header(self) -> str:
        return '{"schema_version":"qa-dataset-v2","items":[' if self._is_v2 else "["

    @property
    def footer(self) -> str:
        return "]}" if self._is_v2 else "]"

    def row(
        self,
        item: DatasetItem,
        draft_row: Dict[str, Any],
        supplied_atoms: Optional[str],
    ) -> Dict[str, Any]:
        if self._is_v2:
            if draft_row.get("status") != "prepared" or supplied_atoms is None:
                if draft_row.get("status") == "skipped_time_sensitive":
                    # A review-atoms draft skips live rows at creation, so no
                    # hand-supplied atom can make one resolve here. Name the way
                    # out, or the operator dead-ends on the generic message.
                    raise ValueError(
                        "live item "
                        f"'{draft_row.get('item_id')}' cannot resolve in this "
                        "draft: review-atoms drafts skip live rows. Live rows "
                        "are materialized only in a generate-atoms draft — save "
                        "that draft instead (its id is the generating job "
                        "result's draft_id), or omit the live item from "
                        "reviewed_items."
                    )
                raise ValueError(
                    "included draft items must resolve and contain valid atoms"
                )
            row = dataset_item_to_dict(item)
            row["expected_atoms"] = json.loads(supplied_atoms)
            if item.is_live:
                answer = draft_row.get("answer")
                if not isinstance(answer, dict) or not answer:
                    raise ValueError(
                        "included live draft item requires a resolved answer"
                    )
                row["answer"] = answer
            return row
        return _dataset_row(
            item,
            (
                json.loads(supplied_atoms)
                if supplied_atoms is not None
                else (
                    [atom.to_dict() for atom in item.expected_atoms]
                    if item.expected_atoms is not None
                    else None
                )
            ),
        )


def _draft_includes(draft: Dict[str, Any], draft_row: Dict[str, Any]) -> bool:
    return not (
        dataset_role(draft) is DatasetRole.DEFINITION_PARENT
        and draft_row.get("status") == "skipped_live"
    )


def _draft_requires_review(
    draft: Dict[str, Any], item: DatasetItem, draft_row: Dict[str, Any]
) -> bool:
    if dataset_role(draft) is DatasetRole.DEFINITION_PARENT:
        return draft_row.get("status") == "prepared"
    return not item.time_sensitive


def _draft_generation_scope(draft: Dict[str, Any]) -> str:
    if dataset_role(draft) is DatasetRole.DEFINITION_PARENT:
        return draft.get("generation_scope") or "complete"
    return "review"


class EvaluationCatalog:
    """Immutable dataset/profile catalogs plus mutable atom-review drafts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.datasets_dir = self.root / "datasets"
        self.profiles_dir = self.root / "profiles"
        self.drafts_dir = self.root / "drafts"
        self.runs_dir = self.root / "runs"
        self.jobs_dir = self.root / "jobs"
        for directory in (
            self.datasets_dir,
            self.profiles_dir,
            self.drafts_dir,
            self.runs_dir,
            self.jobs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _streaming_dataset_metadata(
        dataset_id: str,
        name: str,
        source_filename: str,
        source_format: str,
        digest: str,
        items: Iterable[DatasetItem],
        *,
        parent_dataset_id: Optional[str] = None,
        based_on_child_id: Optional[str] = None,
        generation_scope: Optional[str] = None,
        approval_actor: Optional[str] = None,
        approval_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        categories = set()
        sources = set()
        item_count = 0
        eligible_item_count = 0
        time_sensitive_item_count = 0
        supplied_atom_item_count = 0
        atom_count = 0
        contains_live_answers = False
        schema_version = None
        for item in items:
            item_count += 1
            schema_version = item.schema_version.value
            if item.category:
                categories.add(item.category)
            if item.answer_source:
                sources.add(item.answer_source)
            eligible_item_count += item.state not in {
                DatasetItemState.LEGACY_TIME_SENSITIVE,
                DatasetItemState.UNRESOLVED_LIVE,
            }
            time_sensitive_item_count += item.time_sensitive
            supplied_atom_item_count += item.expected_atoms is not None
            atom_count += len(item.expected_atoms or [])
            contains_live_answers = contains_live_answers or (
                item.state is DatasetItemState.MATERIALIZED_LIVE
            )
        if item_count == 0 or schema_version is None:
            raise ValueError("dataset must contain at least one row")
        dataset_role = (
            DatasetRole.APPROVED_CHILD.value
            if parent_dataset_id is not None
            else (
                DatasetRole.DEFINITION_PARENT.value
                if schema_version == DATASET_V2_SCHEMA_VERSION
                else DatasetRole.LEGACY.value
            )
        )
        return {
            "id": dataset_id,
            "name": name,
            "source_filename": source_filename,
            "format": source_format,
            "sha256": digest,
            "item_count": item_count,
            "eligible_item_count": eligible_item_count,
            "time_sensitive_item_count": time_sensitive_item_count,
            "supplied_atom_item_count": supplied_atom_item_count,
            "atom_count": atom_count,
            "categories": sorted(categories),
            "answer_sources": sorted(sources),
            "parent_dataset_id": parent_dataset_id,
            "schema_version": schema_version,
            "dataset_role": dataset_role,
            "publication_schema": schema_version,
            "based_on_child_id": based_on_child_id,
            "generation_scope": generation_scope,
            "approval_actor": approval_actor,
            "approval_time": approval_time,
            "contains_live_answers": contains_live_answers,
            "created_at": utc_now(),
        }

    def _find_by_hash(self, directory: Path, digest: str) -> Optional[Dict[str, Any]]:
        for metadata_path in directory.glob("*/metadata.json"):
            try:
                metadata = read_json(metadata_path)
            except ValueError:
                continue
            if metadata.get("sha256") == digest:
                return metadata
        return None

    @staticmethod
    def _write_dataset_integrity(directory: Path, source_name: str) -> None:
        metadata_path = directory / "metadata.json"
        source_path = directory / source_name
        write_json(
            directory / "integrity.json",
            {
                "schema_version": CATALOG_INTEGRITY_SCHEMA_VERSION,
                "metadata_sha256": sha256_file(metadata_path),
                "source_sha256": sha256_file(source_path),
            },
        )

    def _read_verified_dataset_metadata(self, directory: Path) -> Dict[str, Any]:
        metadata_path = directory / "metadata.json"
        integrity_path = directory / "integrity.json"
        if not metadata_path.is_file():
            raise ValueError("dataset catalog integrity manifest is missing")
        if not integrity_path.is_file():
            metadata = read_json(metadata_path)
            source_format = metadata.get("format")
            source_path = directory / f"source.{source_format}"
            if (
                source_format not in {"json", "jsonl"}
                or not source_path.is_file()
                or set(metadata) != LEGACY_CATALOG_METADATA_FIELDS
                or sha256_file(source_path) != metadata.get("sha256")
            ):
                raise ValueError("dataset catalog integrity manifest is missing")
            for item in iter_dataset_items(source_path):
                if item.schema_version.value == DATASET_V2_SCHEMA_VERSION:
                    raise ValueError("dataset catalog integrity manifest is missing")
            with self._lock:
                if not integrity_path.is_file():
                    self._write_dataset_integrity(directory, source_path.name)
        integrity = read_json(integrity_path)
        if (
            set(integrity)
            != {
                "schema_version",
                "metadata_sha256",
                "source_sha256",
            }
            or integrity.get("schema_version") != CATALOG_INTEGRITY_SCHEMA_VERSION
        ):
            raise ValueError("dataset catalog integrity manifest is invalid")
        metadata = read_json(metadata_path)
        if "dataset_role" in metadata:
            _validate_catalog_role(metadata)
        elif set(metadata) != LEGACY_CATALOG_METADATA_FIELDS:
            raise ValueError("dataset catalog metadata has an invalid legacy shape")
        source_format = metadata.get("format")
        if source_format not in {"json", "jsonl"}:
            raise ValueError("dataset catalog metadata is invalid")
        source_path = directory / f"source.{source_format}"
        if (
            not source_path.is_file()
            or sha256_file(metadata_path) != integrity.get("metadata_sha256")
            or sha256_file(source_path) != integrity.get("source_sha256")
            or metadata.get("sha256") != integrity.get("source_sha256")
        ):
            raise ValueError("dataset catalog integrity verification failed")
        if "dataset_role" not in metadata:
            metadata = dict(metadata)
            metadata.update(
                {
                    "schema_version": "qa-dataset-v1",
                    "dataset_role": DatasetRole.LEGACY.value,
                    "publication_schema": "qa-dataset-v1",
                    "based_on_child_id": None,
                    "generation_scope": None,
                    "approval_actor": None,
                    "approval_time": None,
                    "contains_live_answers": False,
                }
            )
        return metadata

    def _find_dataset_by_hash(self, digest: str) -> Optional[Dict[str, Any]]:
        for directory in self.datasets_dir.iterdir():
            if not directory.is_dir():
                continue
            try:
                metadata = self._read_verified_dataset_metadata(directory)
            except ValueError:
                continue
            if metadata.get("sha256") == digest:
                return metadata
        return None

    def import_dataset(
        self,
        name: str,
        filename: str,
        blob: bytes,
        *,
        parent_dataset_id: Optional[str] = None,
        allow_materialized_live: bool = False,
        based_on_child_id: Optional[str] = None,
        generation_scope: Optional[str] = None,
        approval_actor: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        name = _display_name(name, "dataset name")
        if not isinstance(blob, bytes) or len(blob) > MAX_IMPORT_BYTES:
            raise ValueError("dataset upload exceeds the 25 MB limit")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError("dataset must use .json or .jsonl")
        source_format = suffix[1:]
        digest = _sha256(blob)
        with tempfile.TemporaryDirectory(
            prefix=".dataset-", dir=str(self.datasets_dir)
        ) as temporary:
            temporary_path = Path(temporary)
            source_path = temporary_path / f"source.{source_format}"
            write_bytes(source_path, blob)

            # Complete preflight validation is a separate bounded pass. Metadata
            # is then accumulated from a fresh read without retaining the rows.
            for _item in iter_dataset_items(
                source_path,
                allow_materialized_live=allow_materialized_live,
            ):
                pass
            metadata_items = iter_dataset_items(
                source_path,
                allow_materialized_live=allow_materialized_live,
            )
            dataset_id = str(uuid.uuid4())
            metadata = self._streaming_dataset_metadata(
                dataset_id,
                name,
                Path(filename).name,
                source_format,
                digest,
                metadata_items,
                parent_dataset_id=parent_dataset_id,
                based_on_child_id=based_on_child_id,
                generation_scope=generation_scope,
                approval_actor=approval_actor,
                approval_time=utc_now() if approval_actor is not None else None,
            )
            with self._lock:
                if parent_dataset_id is None:
                    existing = self._find_dataset_by_hash(digest)
                    if existing is not None:
                        return existing, False
                elif parent_dataset_id is not None:
                    _catalog_id(parent_dataset_id)
                write_json(temporary_path / "metadata.json", metadata)
                self._write_dataset_integrity(temporary_path, f"source.{source_format}")
                target = self.datasets_dir / dataset_id
                os.replace(str(temporary_path), str(target))
            return metadata, True

    def _import_dataset_file(
        self,
        name: str,
        filename: str,
        source_path: Path,
        *,
        parent_dataset_id: str,
        based_on_child_id: Optional[str],
        generation_scope: str,
        approval_actor: str,
    ) -> Dict[str, Any]:
        """Publish a generated child without loading its source into memory."""
        name = _display_name(name, "dataset name")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError("dataset must use .json or .jsonl")
        source_format = suffix[1:]
        digest = sha256_file(source_path)
        dataset_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory(
            prefix=".dataset-", dir=str(self.datasets_dir)
        ) as temporary:
            temporary_path = Path(temporary)
            staged_source = temporary_path / f"source.{source_format}"
            copy_file_atomic(source_path, staged_source)
            for _item in iter_dataset_items(
                staged_source,
                allow_materialized_live=True,
            ):
                pass
            metadata = self._streaming_dataset_metadata(
                dataset_id,
                name,
                Path(filename).name,
                source_format,
                digest,
                iter_dataset_items(staged_source, allow_materialized_live=True),
                parent_dataset_id=parent_dataset_id,
                based_on_child_id=based_on_child_id,
                generation_scope=generation_scope,
                approval_actor=approval_actor,
                approval_time=utc_now(),
            )
            write_json(temporary_path / "metadata.json", metadata)
            self._write_dataset_integrity(temporary_path, f"source.{source_format}")
            with self._lock:
                _catalog_id(parent_dataset_id)
                os.replace(str(temporary_path), str(self.datasets_dir / dataset_id))
        return metadata

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets = []
        for path in self.datasets_dir.glob("*/metadata.json"):
            try:
                datasets.append(self._read_verified_dataset_metadata(path.parent))
            except ValueError:
                continue
        return sorted(
            datasets, key=lambda item: item.get("created_at", ""), reverse=True
        )

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        dataset_id = _catalog_id(dataset_id)
        path = self.datasets_dir / dataset_id / "metadata.json"
        if not path.is_file():
            raise LookupError("dataset not found")
        return self._read_verified_dataset_metadata(path.parent)

    def dataset_path(self, dataset_id: str) -> Path:
        metadata = self.get_dataset(dataset_id)
        path = self.datasets_dir / dataset_id / f"source.{metadata['format']}"
        if not path.is_file():
            raise LookupError("dataset source is missing")
        return path

    def dataset_items(self, dataset_id: str) -> List[DatasetItem]:
        metadata = self.get_dataset(dataset_id)
        return list(
            iter_dataset_items(
                self.dataset_path(dataset_id),
                allow_materialized_live=bool(metadata.get("contains_live_answers")),
            )
        )

    def import_profile(
        self, name: str, filename: str, blob: bytes
    ) -> Tuple[Dict[str, Any], bool]:
        name = _display_name(name, "profile name")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".yaml", ".yml"}:
            raise ValueError("evaluator profile must use .yaml or .yml")
        if not isinstance(blob, bytes) or len(blob) > MAX_IMPORT_BYTES:
            raise ValueError("profile upload exceeds the 25 MB limit")
        try:
            raw = yaml.safe_load(blob.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValueError(f"invalid evaluator profile YAML: {exc}") from exc
        profile = _parse_profile(raw)
        digest = _sha256(blob)
        with self._lock:
            existing = self._find_by_hash(self.profiles_dir, digest)
            if existing is not None:
                return existing, False
            profile_id = str(uuid.uuid4())
            metadata = {
                "id": profile_id,
                "name": name,
                "source_filename": Path(filename).name,
                "sha256": digest,
                "created_at": utc_now(),
                "components": {
                    component: descriptor.to_dict()
                    for component, descriptor in profile.components()
                },
            }
            target = self.profiles_dir / profile_id
            with tempfile.TemporaryDirectory(
                prefix=".profile-", dir=str(self.profiles_dir)
            ) as temporary:
                temporary_path = Path(temporary)
                write_bytes(temporary_path / "profile.yaml", blob)
                write_json(temporary_path / "metadata.json", metadata)
                os.replace(str(temporary_path), str(target))
            return metadata, True

    def list_profiles(self) -> List[Dict[str, Any]]:
        built_in = {
            "id": "builtin",
            "name": "Built-in QA profile",
            "source_filename": None,
            "sha256": None,
            "created_at": None,
            "built_in": True,
            "components": {
                component: descriptor.to_dict()
                for component, descriptor in DEFAULT_PROFILE.components()
            },
        }
        profiles = [built_in]
        for path in self.profiles_dir.glob("*/metadata.json"):
            try:
                metadata = read_json(path)
                metadata["built_in"] = False
                profiles.append(metadata)
            except ValueError:
                continue
        return profiles

    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        profile_id = _catalog_id(profile_id, allow_builtin=True)
        for profile in self.list_profiles():
            if profile["id"] == profile_id:
                return profile
        raise LookupError("profile not found")

    def profile_path(self, profile_id: str) -> Optional[Path]:
        self.get_profile(profile_id)
        if profile_id == "builtin":
            return None
        path = self.profiles_dir / profile_id / "profile.yaml"
        if not path.is_file():
            raise LookupError("profile source is missing")
        return path

    def create_atom_draft(
        self,
        dataset_id: str,
        profile_id: str,
        evaluator: GoldExtractor,
        oracle_resolver: Optional[OracleResolver] = None,
        *,
        static_only: bool = False,
        include_items: bool = True,
    ) -> Dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        is_definition = dataset_role(dataset) is DatasetRole.DEFINITION_PARENT
        if not is_definition and dataset["atom_count"] != 0:
            raise ValueError(
                "atom generation requires a dataset with zero atoms; review its "
                "existing atoms instead"
            )
        self.get_profile(profile_id)

        def draft_items() -> Iterable[Dict[str, Any]]:
            for item in iter_dataset_items(self.dataset_path(dataset_id)):
                record = (
                    prepare_dataset_item(
                        item,
                        evaluator,
                        oracle_resolver,
                        skip_live=static_only,
                    )
                    if item.is_live
                    else prepare_dataset_item(item, evaluator)
                )
                yield _generated_draft_row(item, record)

        return self._persist_atom_draft(
            dataset,
            profile_id,
            draft_items(),
            generation_scope="static_only" if static_only else "complete",
            include_items=include_items,
        )

    def create_atom_review_draft(self, dataset_id: str) -> Dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if dataset["atom_count"] == 0:
            raise ValueError(
                "atom review requires a dataset with at least one atom; generate "
                "atoms first"
            )

        def draft_items() -> Iterable[Dict[str, Any]]:
            for item in iter_dataset_items(self.dataset_path(dataset_id)):
                if item.time_sensitive:
                    status = "skipped_time_sensitive"
                elif item.expected_atoms is None:
                    status = "missing_atoms"
                else:
                    status = "prepared"
                row: Dict[str, Any] = {
                    "item_id": item.id,
                    "question": item.question,
                    "answer": item.answer,
                    "time_sensitive": item.time_sensitive,
                    "status": status,
                    "atoms": [atom.to_dict() for atom in (item.expected_atoms or [])],
                }
                if item.answer_source is not None:
                    row["answer_source"] = item.answer_source
                if item.expected_atoms is not None:
                    row["atom_source"] = "supplied"
                yield row

        return self._persist_atom_draft(dataset, None, draft_items())

    def _persist_atom_draft(
        self,
        dataset: Dict[str, Any],
        profile_id: Optional[str],
        draft_items: Iterable[Dict[str, Any]],
        *,
        generation_scope: Optional[str] = None,
        based_on_child_id: Optional[str] = None,
        include_items: bool = True,
    ) -> Dict[str, Any]:
        draft_id = str(uuid.uuid4())
        draft = {
            "id": draft_id,
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "schema_version": dataset.get("schema_version"),
            "dataset_role": dataset.get("dataset_role"),
            "publication_schema": dataset.get("publication_schema"),
            "profile_id": profile_id,
            "status": "open",
            "created_at": utc_now(),
            "generation_scope": generation_scope,
            "based_on_child_id": based_on_child_id,
        }
        target = self.drafts_dir / draft_id
        with tempfile.TemporaryDirectory(
            prefix=".draft-", dir=str(self.drafts_dir)
        ) as temporary:
            temporary_path = Path(temporary)
            write_json(temporary_path / "draft.json", draft)
            with AtomicJsonlWriter(temporary_path / "items.jsonl") as writer:
                for row in draft_items:
                    writer.write(row)
            os.replace(str(temporary_path), str(target))
        return self.get_atom_draft(draft_id) if include_items else draft

    def get_atom_draft(self, draft_id: str) -> Dict[str, Any]:
        draft = self.get_atom_draft_header(draft_id)
        path = self.drafts_dir / draft["id"] / "draft.json"
        items_path = path.parent / "items.jsonl"
        if items_path.is_file():
            if items_path.stat().st_size > MAX_IMPORT_BYTES:
                raise ValueError("atom draft detail exceeds the 25 MB limit")
            draft["items"] = list(iter_jsonl(items_path))
        elif "items" not in draft:
            draft["items"] = []
        return draft

    def get_atom_draft_header(self, draft_id: str) -> Dict[str, Any]:
        draft_id = _catalog_id(draft_id)
        path = self.drafts_dir / draft_id / "draft.json"
        if not path.is_file():
            raise LookupError("atom draft not found")
        return read_json(path)

    def _write_atom_draft_header(self, draft: Dict[str, Any]) -> None:
        header = {key: value for key, value in draft.items() if key != "items"}
        write_json(self.drafts_dir / draft["id"] / "draft.json", header)

    def _iter_atom_draft_items(self, draft_id: str) -> Iterable[Dict[str, Any]]:
        path = self.drafts_dir / draft_id / "items.jsonl"
        if path.is_file():
            return iter_jsonl(path)
        return iter(self.get_atom_draft(draft_id)["items"])

    def make_atom_draft_static_only(self, draft_id: str) -> Dict[str, Any]:
        """Omit every live row from an open V2 draft without altering its parent."""
        with self._lock:
            draft_id = _catalog_id(draft_id)
            draft = read_json(self.drafts_dir / draft_id / "draft.json")
            if draft.get("status") != "open":
                raise ValueError("atom draft has already been saved")
            if dataset_role(draft) is not DatasetRole.DEFINITION_PARENT:
                raise ValueError("static-only scope is available only for Dataset V2")
            if draft.get("generation_scope") == "static_only":
                return draft
            if draft.get("generation_scope") not in {"complete", None}:
                raise ValueError(
                    "a live refresh draft cannot switch to static-only scope"
                )
            live_count = sum(
                1
                for row in self._iter_atom_draft_items(draft_id)
                if row.get("time_sensitive")
            )
            if not live_count:
                raise ValueError("atom draft has no live items to omit")
            items_path = self.drafts_dir / draft_id / "items.jsonl"
            with AtomicJsonlWriter(items_path) as writer:
                for row in self._iter_atom_draft_items(draft_id):
                    if row.get("time_sensitive"):
                        row = {
                            "item_id": row.get("item_id"),
                            "question": row.get("question"),
                            "answer": None,
                            "time_sensitive": True,
                            "status": "skipped_live",
                            "live_state": "omitted",
                        }
                    writer.write(row)
            draft["generation_scope"] = "static_only"
            self._write_atom_draft_header(draft)
            return self.get_atom_draft(draft_id)

    def atom_retry_details(self, draft_id: str) -> Dict[str, Any]:
        with self._lock:
            draft_id = _catalog_id(draft_id)
            draft = read_json(self.drafts_dir / draft_id / "draft.json")
            if draft.get("status") != "open":
                raise ValueError("atom draft has already been saved")
            profile_id = draft.get("profile_id")
            if not isinstance(profile_id, str):
                raise ValueError("only generated atom drafts can retry failed items")
            self.get_profile(profile_id)
            item_ids = [
                row["item_id"]
                for row in self._iter_atom_draft_items(draft_id)
                if row.get("status") == "preparation_failed"
            ]
            if not item_ids:
                raise ValueError("atom draft has no failed items to retry")
            return {
                "draft_id": draft["id"],
                "dataset_id": draft["dataset_id"],
                "profile_id": profile_id,
                "item_ids": item_ids,
            }

    def retry_failed_atom_items(
        self,
        draft_id: str,
        evaluator: GoldExtractor,
        oracle_resolver: Optional[OracleResolver] = None,
        *,
        include_items: bool = True,
    ) -> Dict[str, Any]:
        details = self.atom_retry_details(draft_id)
        retry_ids = set(details["item_ids"])
        with tempfile.TemporaryDirectory(prefix=".retry-index-") as temporary:
            connection = sqlite3.connect(str(Path(temporary) / "retry.sqlite3"))
            try:
                connection.execute(
                    "CREATE TABLE replacements (item_id TEXT PRIMARY KEY, row_json TEXT NOT NULL)"
                )
                for item in iter_dataset_items(
                    self.dataset_path(details["dataset_id"])
                ):
                    if item.id not in retry_ids:
                        continue
                    record = (
                        prepare_dataset_item(item, evaluator, oracle_resolver)
                        if item.is_live
                        else prepare_dataset_item(item, evaluator)
                    )
                    connection.execute(
                        "INSERT INTO replacements VALUES (?, ?)",
                        (
                            item.id,
                            json.dumps(
                                _generated_draft_row(item, record),
                                ensure_ascii=False,
                            ),
                        ),
                    )
                found_ids = {
                    row[0]
                    for row in connection.execute("SELECT item_id FROM replacements")
                }
                missing = sorted(retry_ids - found_ids)
                if missing:
                    raise ValueError(
                        "atom draft references missing dataset item(s): "
                        + ", ".join(missing)
                    )

                with self._lock:
                    draft = read_json(
                        self.drafts_dir / details["draft_id"] / "draft.json"
                    )
                    if draft.get("status") != "open":
                        raise ValueError("atom draft has already been saved")
                    current_failed_ids = {
                        row["item_id"]
                        for row in self._iter_atom_draft_items(draft_id)
                        if row.get("status") == "preparation_failed"
                    }
                    if not retry_ids.issubset(current_failed_ids):
                        raise ValueError(
                            "atom draft changed while failed items were retried"
                        )
                    items_path = self.drafts_dir / details["draft_id"] / "items.jsonl"
                    with AtomicJsonlWriter(items_path) as writer:
                        for row in self._iter_atom_draft_items(draft_id):
                            replacement = connection.execute(
                                "SELECT row_json FROM replacements WHERE item_id = ?",
                                (row["item_id"],),
                            ).fetchone()
                            writer.write(
                                json.loads(replacement[0])
                                if replacement is not None
                                else row
                            )
                    return self.get_atom_draft(draft_id) if include_items else draft
            finally:
                connection.close()

    def create_refresh_draft(
        self,
        child_dataset_id: str,
        profile_id: str,
        evaluator: GoldExtractor,
        oracle_resolver: OracleResolver,
        *,
        include_items: bool = True,
    ) -> Dict[str, Any]:
        child = self.get_dataset(child_dataset_id)
        parent_id = child.get("parent_dataset_id")
        if not isinstance(parent_id, str):
            raise ValueError("live refresh requires an approved child dataset")
        parent = self.get_dataset(parent_id)
        if dataset_role(parent) is not DatasetRole.DEFINITION_PARENT:
            raise ValueError("live refresh requires a Dataset V2 definition parent")
        if sha256_file(self.dataset_path(child_dataset_id)) != child["sha256"]:
            raise ValueError("selected child source hash does not match its manifest")
        if sha256_file(self.dataset_path(parent_id)) != parent["sha256"]:
            raise ValueError(
                "definition parent source hash does not match its manifest"
            )
        self.get_profile(profile_id)
        with tempfile.TemporaryDirectory(prefix=".refresh-index-") as temporary:
            connection = sqlite3.connect(str(Path(temporary) / "child.sqlite3"))
            try:
                connection.execute(
                    "CREATE TABLE child (id TEXT PRIMARY KEY, question TEXT NOT NULL, "
                    "answer_json TEXT, atoms_json TEXT)"
                )
                for item in iter_dataset_items(
                    self.dataset_path(child_dataset_id),
                    allow_materialized_live=bool(child.get("contains_live_answers")),
                ):
                    connection.execute(
                        "INSERT INTO child VALUES (?, ?, ?, ?)",
                        (
                            item.id,
                            item.question,
                            json.dumps(item.answer, ensure_ascii=False),
                            (
                                json.dumps(
                                    [atom.to_dict() for atom in item.expected_atoms],
                                    ensure_ascii=False,
                                )
                                if item.expected_atoms is not None
                                else None
                            ),
                        ),
                    )
                connection.commit()

                for item in iter_dataset_items(self.dataset_path(parent_id)):
                    if item.is_live:
                        continue
                    carried = connection.execute(
                        "SELECT question, answer_json, atoms_json FROM child WHERE id = ?",
                        (item.id,),
                    ).fetchone()
                    if (
                        carried is None
                        or carried[0] != item.question
                        or json.loads(carried[1]) != item.answer
                        or carried[2] is None
                    ):
                        raise ValueError(
                            f"carried static item '{item.id}' does not match its definition parent"
                        )

                def draft_items() -> Iterable[Dict[str, Any]]:
                    for item in iter_dataset_items(self.dataset_path(parent_id)):
                        carried = connection.execute(
                            "SELECT answer_json, atoms_json FROM child WHERE id = ?",
                            (item.id,),
                        ).fetchone()
                        if not item.is_live:
                            assert carried is not None and carried[1] is not None
                            yield {
                                "item_id": item.id,
                                "question": item.question,
                                "answer": item.answer,
                                "time_sensitive": False,
                                "status": "prepared",
                                "atom_source": "supplied",
                                "atoms": json.loads(carried[1]),
                            }
                            continue
                        record = prepare_dataset_item(item, evaluator, oracle_resolver)
                        row = _generated_draft_row(item, record)
                        previous_answer = (
                            json.loads(carried[0]) if carried is not None else None
                        )
                        if not isinstance(previous_answer, dict):
                            row["live_state"] = (
                                "unavailable" if record.status != "prepared" else "new"
                            )
                        elif record.status != "prepared":
                            row["live_state"] = "unavailable"
                            row["previous_answer"] = previous_answer
                        else:
                            row["previous_answer"] = previous_answer
                            row["live_state"] = (
                                "unchanged"
                                if previous_answer == record.prepared_answer
                                else "changed"
                            )
                        yield row

                scope = (
                    "refresh_live" if child.get("contains_live_answers") else "add_live"
                )
                return self._persist_atom_draft(
                    parent,
                    profile_id,
                    draft_items(),
                    generation_scope=scope,
                    based_on_child_id=child_dataset_id,
                    include_items=include_items,
                )
            finally:
                connection.close()

    def save_reviewed_dataset(
        self,
        draft_id: str,
        name: str,
        reviewed_items: Any,
        *,
        approval_actor: str = "operator",
    ) -> Dict[str, Any]:
        with self._lock:
            draft_id = _catalog_id(draft_id)
            draft = read_json(self.drafts_dir / draft_id / "draft.json")
            if draft.get("status") != "open":
                raise ValueError("atom draft has already been saved")
            if not isinstance(reviewed_items, list):
                raise ValueError("reviewed_items must be a list")
            parent_id = draft["dataset_id"]
            parent = self.get_dataset(parent_id)
            if "dataset_role" not in draft:
                draft = {
                    **draft,
                    "schema_version": parent.get("schema_version"),
                    "dataset_role": parent["dataset_role"],
                    "publication_schema": parent.get("publication_schema"),
                    "generation_scope": draft.get("generation_scope"),
                }
            publisher = _ReviewedChildPublisher(
                draft.get("publication_schema")
                or parent.get("publication_schema")
                or parent.get("schema_version")
            )
            with tempfile.TemporaryDirectory(prefix=".approval-index-") as temporary:
                connection = sqlite3.connect(str(Path(temporary) / "review.sqlite3"))
                child_path = Path(temporary) / "child.json"
                try:
                    connection.execute(
                        "CREATE TABLE reviewed (id TEXT PRIMARY KEY, atoms_json TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0)"
                    )
                    for index, reviewed in enumerate(reviewed_items):
                        if not isinstance(reviewed, dict) or set(reviewed) != {
                            "item_id",
                            "atoms",
                        }:
                            raise ValueError(
                                f"reviewed_items[{index}] must contain item_id and atoms"
                            )
                        item_id = reviewed["item_id"]
                        if not isinstance(item_id, str) or not item_id:
                            raise ValueError(
                                "reviewed item IDs must be non-empty and unique"
                            )
                        atoms = [
                            atom.to_dict()
                            for atom in validate_atoms(
                                reviewed["atoms"],
                                context=f"reviewed_items[{index}].atoms",
                            )
                        ]
                        try:
                            connection.execute(
                                "INSERT INTO reviewed (id, atoms_json) VALUES (?, ?)",
                                (item_id, json.dumps(atoms, ensure_ascii=False)),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(
                                "reviewed item IDs must be non-empty and unique"
                            ) from exc
                    connection.commit()

                    with child_path.open("w", encoding="utf-8") as child_file:
                        child_file.write(publisher.header)
                        first = True
                        item_rows = iter_dataset_items(self.dataset_path(parent_id))
                        draft_rows = self._iter_atom_draft_items(draft_id)
                        for item, draft_row in zip_longest(item_rows, draft_rows):
                            if item is None or draft_row is None:
                                raise ValueError(
                                    "atom draft membership does not match its definition parent"
                                )
                            if draft_row.get("item_id") != item.id:
                                raise ValueError(
                                    "atom draft order does not match its definition parent"
                                )
                            if not _draft_includes(draft, draft_row):
                                continue
                            eligible = _draft_requires_review(draft, item, draft_row)
                            supplied = connection.execute(
                                "SELECT atoms_json FROM reviewed WHERE id = ?",
                                (item.id,),
                            ).fetchone()
                            if eligible and supplied is None:
                                raise ValueError(
                                    "reviewed items do not match eligible dataset items "
                                    f"(missing: {item.id})"
                                )
                            if supplied is not None:
                                connection.execute(
                                    "UPDATE reviewed SET used = 1 WHERE id = ?",
                                    (item.id,),
                                )
                            row = publisher.row(
                                item,
                                draft_row,
                                supplied[0] if supplied is not None else None,
                            )
                            if not first:
                                child_file.write(",")
                            child_file.write(
                                json.dumps(
                                    row,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            )
                            first = False
                        child_file.write(publisher.footer)
                        child_file.write("\n")
                    unknown = connection.execute(
                        "SELECT id FROM reviewed WHERE used = 0 LIMIT 1"
                    ).fetchone()
                    if unknown is not None:
                        raise ValueError(
                            "reviewed items do not match eligible dataset items "
                            f"(unknown: {unknown[0]})"
                        )
                    metadata = self._import_dataset_file(
                        name,
                        f"{name}.json",
                        child_path,
                        parent_dataset_id=parent_id,
                        based_on_child_id=draft.get("based_on_child_id"),
                        generation_scope=_draft_generation_scope(draft),
                        approval_actor=_display_name(approval_actor, "approval actor"),
                    )
                finally:
                    connection.close()
            draft["status"] = "saved"
            draft["saved_at"] = utc_now()
            draft["child_dataset_id"] = metadata["id"]
            self._write_atom_draft_header(draft)
            return metadata
