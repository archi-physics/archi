# isort: skip_file
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .artifacts import read_json, utc_now, write_bytes, write_json
from .preparation import GoldExtractor, PreparationRecord, prepare_dataset_items
from .profile import DEFAULT_PROFILE, _parse_profile
from .validation import (  # isort: skip
    DatasetItem,
    load_dataset,
    load_dataset_bytes,
    validate_atoms,
)

MAX_IMPORT_BYTES = 25 * 1024 * 1024


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
    elif record.error is not None:
        row["error"] = record.error
    return row


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
    def _dataset_metadata(
        dataset_id: str,
        name: str,
        source_filename: str,
        source_format: str,
        blob: bytes,
        items: Sequence[DatasetItem],
        *,
        parent_dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        categories = sorted({item.category for item in items if item.category})
        sources = sorted({item.answer_source for item in items if item.answer_source})
        return {
            "id": dataset_id,
            "name": name,
            "source_filename": source_filename,
            "format": source_format,
            "sha256": _sha256(blob),
            "item_count": len(items),
            "eligible_item_count": sum(not item.time_sensitive for item in items),
            "time_sensitive_item_count": sum(item.time_sensitive for item in items),
            "supplied_atom_item_count": sum(
                item.expected_atoms is not None for item in items
            ),
            "atom_count": sum(len(item.expected_atoms or []) for item in items),
            "categories": categories,
            "answer_sources": sources,
            "parent_dataset_id": parent_dataset_id,
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

    def import_dataset(
        self,
        name: str,
        filename: str,
        blob: bytes,
        *,
        parent_dataset_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        name = _display_name(name, "dataset name")
        if not isinstance(blob, bytes) or len(blob) > MAX_IMPORT_BYTES:
            raise ValueError("dataset upload exceeds the 25 MB limit")
        source_format, items, _ = load_dataset_bytes(filename, blob)
        digest = _sha256(blob)
        with self._lock:
            if parent_dataset_id is None:
                existing = self._find_by_hash(self.datasets_dir, digest)
                if existing is not None:
                    return existing, False
            dataset_id = str(uuid.uuid4())
            if parent_dataset_id is not None:
                _catalog_id(parent_dataset_id)
            metadata = self._dataset_metadata(
                dataset_id,
                name,
                Path(filename).name,
                source_format,
                blob,
                items,
                parent_dataset_id=parent_dataset_id,
            )
            target = self.datasets_dir / dataset_id
            with tempfile.TemporaryDirectory(
                prefix=".dataset-", dir=str(self.datasets_dir)
            ) as temporary:
                temporary_path = Path(temporary)
                write_bytes(temporary_path / f"source.{source_format}", blob)
                write_json(temporary_path / "metadata.json", metadata)
                os.replace(str(temporary_path), str(target))
            return metadata, True

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets = []
        for path in self.datasets_dir.glob("*/metadata.json"):
            try:
                datasets.append(read_json(path))
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
        return read_json(path)

    def dataset_path(self, dataset_id: str) -> Path:
        metadata = self.get_dataset(dataset_id)
        path = self.datasets_dir / dataset_id / f"source.{metadata['format']}"
        if not path.is_file():
            raise LookupError("dataset source is missing")
        return path

    def dataset_items(self, dataset_id: str) -> List[DatasetItem]:
        return load_dataset(self.dataset_path(dataset_id))[1]

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
        self, dataset_id: str, profile_id: str, evaluator: GoldExtractor
    ) -> Dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if dataset["atom_count"] != 0:
            raise ValueError(
                "atom generation requires a dataset with zero atoms; review its "
                "existing atoms instead"
            )
        self.get_profile(profile_id)
        items = self.dataset_items(dataset_id)
        records = prepare_dataset_items(items, evaluator)
        records_by_id = {record.item_id: record for record in records}
        draft_items = [
            _generated_draft_row(item, records_by_id[item.id])
            for item in items
        ]
        return self._persist_atom_draft(dataset, profile_id, draft_items)

    def create_atom_review_draft(self, dataset_id: str) -> Dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if dataset["atom_count"] == 0:
            raise ValueError(
                "atom review requires a dataset with at least one atom; generate "
                "atoms first"
            )
        draft_items = []
        for item in self.dataset_items(dataset_id):
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
            draft_items.append(row)
        return self._persist_atom_draft(dataset, None, draft_items)

    def _persist_atom_draft(
        self,
        dataset: Dict[str, Any],
        profile_id: Optional[str],
        draft_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        draft_id = str(uuid.uuid4())
        draft = {
            "id": draft_id,
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "profile_id": profile_id,
            "status": "open",
            "created_at": utc_now(),
            "items": draft_items,
        }
        target = self.drafts_dir / draft_id
        with tempfile.TemporaryDirectory(
            prefix=".draft-", dir=str(self.drafts_dir)
        ) as temporary:
            temporary_path = Path(temporary)
            write_json(temporary_path / "draft.json", draft)
            os.replace(str(temporary_path), str(target))
        return draft

    def get_atom_draft(self, draft_id: str) -> Dict[str, Any]:
        draft_id = _catalog_id(draft_id)
        path = self.drafts_dir / draft_id / "draft.json"
        if not path.is_file():
            raise LookupError("atom draft not found")
        return read_json(path)

    def atom_retry_details(self, draft_id: str) -> Dict[str, Any]:
        with self._lock:
            draft = self.get_atom_draft(draft_id)
            if draft.get("status") != "open":
                raise ValueError("atom draft has already been saved")
            profile_id = draft.get("profile_id")
            if not isinstance(profile_id, str):
                raise ValueError("only generated atom drafts can retry failed items")
            self.get_profile(profile_id)
            item_ids = [
                row["item_id"]
                for row in draft["items"]
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
        self, draft_id: str, evaluator: GoldExtractor
    ) -> Dict[str, Any]:
        details = self.atom_retry_details(draft_id)
        items_by_id = {
            item.id: item for item in self.dataset_items(details["dataset_id"])
        }
        missing = sorted(set(details["item_ids"]) - set(items_by_id))
        if missing:
            raise ValueError(
                "atom draft references missing dataset item(s): " + ", ".join(missing)
            )
        selected_items = [items_by_id[item_id] for item_id in details["item_ids"]]
        records = prepare_dataset_items(selected_items, evaluator)
        records_by_id = {record.item_id: record for record in records}
        replacements = {
            item.id: _generated_draft_row(item, records_by_id[item.id])
            for item in selected_items
        }

        with self._lock:
            draft = self.get_atom_draft(draft_id)
            if draft.get("status") != "open":
                raise ValueError("atom draft has already been saved")
            current_failed_ids = {
                row["item_id"]
                for row in draft["items"]
                if row.get("status") == "preparation_failed"
            }
            if not set(details["item_ids"]).issubset(current_failed_ids):
                raise ValueError("atom draft changed while failed items were retried")
            draft["items"] = [
                replacements.get(row["item_id"], row) for row in draft["items"]
            ]
            write_json(
                self.drafts_dir / details["draft_id"] / "draft.json",
                draft,
            )
            return draft

    def save_reviewed_dataset(
        self, draft_id: str, name: str, reviewed_items: Any
    ) -> Dict[str, Any]:
        with self._lock:
            draft = self.get_atom_draft(draft_id)
            if draft.get("status") != "open":
                raise ValueError("atom draft has already been saved")
            if not isinstance(reviewed_items, list):
                raise ValueError("reviewed_items must be a list")
            supplied: Dict[str, List[Dict[str, Any]]] = {}
            for index, row in enumerate(reviewed_items):
                if not isinstance(row, dict) or set(row) != {"item_id", "atoms"}:
                    raise ValueError(
                        f"reviewed_items[{index}] must contain item_id and atoms"
                    )
                item_id = row["item_id"]
                if not isinstance(item_id, str) or not item_id or item_id in supplied:
                    raise ValueError("reviewed item IDs must be non-empty and unique")
                atoms = validate_atoms(
                    row["atoms"],
                    context=f"reviewed_items[{index}].atoms",
                )
                supplied[item_id] = [atom.to_dict() for atom in atoms]

            parent_id = draft["dataset_id"]
            items = self.dataset_items(parent_id)
            eligible_ids = {item.id for item in items if not item.time_sensitive}
            if set(supplied) != eligible_ids:
                missing = sorted(eligible_ids - set(supplied))
                unknown = sorted(set(supplied) - eligible_ids)
                details = []
                if missing:
                    details.append("missing: " + ", ".join(missing))
                if unknown:
                    details.append("unknown: " + ", ".join(unknown))
                raise ValueError(
                    "reviewed items do not match eligible dataset items ("
                    + "; ".join(details)
                    + ")"
                )

            rows = [
                _dataset_row(
                    item,
                    (
                        supplied[item.id]
                        if not item.time_sensitive
                        else (
                            [atom.to_dict() for atom in item.expected_atoms]
                            if item.expected_atoms is not None
                            else None
                        )
                    ),
                )
                for item in items
            ]
            blob = (
                json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            metadata, _created = self.import_dataset(
                name,
                f"{name}.json",
                blob,
                parent_dataset_id=parent_id,
            )
            draft["status"] = "saved"
            draft["saved_at"] = utc_now()
            draft["child_dataset_id"] = metadata["id"]
            write_json(self.drafts_dir / draft_id / "draft.json", draft)
            return metadata
