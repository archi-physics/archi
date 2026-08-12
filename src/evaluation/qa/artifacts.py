from __future__ import annotations

import hashlib
import json
import os
import tempfile
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, TextIO

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def write_yaml(path: Path, value: Any) -> None:
    _atomic_write(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def write_text(path: Path, value: str) -> None:
    _atomic_write(path, value)


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def copy_file_atomic(source: Path, target: Path) -> None:
    """Copy a file in bounded chunks and atomically replace the target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    try:
        with (
            os.fdopen(descriptor, "wb") as target_handle,
            source.open("rb") as source_handle,
        ):
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                target_handle.write(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class AtomicJsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._handle: Optional[TextIO] = None
        self._temp_name: Optional[str] = None

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, self._temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=str(self.path.parent)
        )
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")
        return self

    def write(self, row: Dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JSONL writer is not open")
        self._handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def __exit__(self, exc_type, exc, traceback) -> None:
        commit = False
        try:
            if self._handle is not None and exc_type is None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                commit = True
        finally:
            if self._handle is not None:
                self._handle.close()
            if self._temp_name is not None:
                try:
                    if commit:
                        os.replace(self._temp_name, self.path)
                finally:
                    if os.path.exists(self._temp_name):
                        os.unlink(self._temp_name)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with AtomicJsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON artifact {path.name}: {exc}") from exc


def read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read valid YAML artifact {path.name}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not an object")
                yield row
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"cannot read valid JSONL artifact {path.name}: {exc}"
        ) from exc


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def artifact_hashes(run_dir: Path, names: Iterable[str]) -> Dict[str, str]:
    return {name: sha256_file(run_dir / name) for name in sorted(names)}


def verify_hashes(
    run_dir: Path, expected: typing.Mapping[str, str], names: Iterable[str]
) -> None:
    for name in names:
        path = run_dir / name
        if not path.exists() or not path.is_file():
            raise ValueError(f"required workspace artifact is missing: {name}")
        if expected.get(name) != sha256_file(path):
            raise ValueError(f"workspace artifact hash mismatch: {name}")
