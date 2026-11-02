"""Shared parts of the cluster-manager readers (ServiceNow, tickets, Slurm, HTCondor, node health).

Each reader in this family reads one JSON cache that a downloader (or an
operator's export job) wrote under ``${archi_data_root}/<name>/``. The run
shape is the same as ``archi.sources.dqm`` and ``archi.sources.jira``:

- A cache that is missing, empty, truncated or of the wrong JSON shape is a
  failed fetch, never an empty catalog: no facts and no complete scope.
- A cache item that cannot be parsed is skipped, and the run then claims no
  complete scope (a complete scope over the survivors would retract the
  records the drifted items used to produce).
- When ``meta_path`` names a ``meta.json`` with a ``record_count``, a parsed
  count that differs means a truncated or stale cache: facts are emitted, the
  health is ``endpoint_failed`` and no complete scope is claimed.

A subclass sets ``name``, ``profile``, ``change_probe_kind`` and
``description`` and implements :meth:`parse` and :meth:`facts`. It never talks
to the network: credentials are named by environment variable in the registry
entry (``credential_refs``) and used only by the downloader.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    NodeFact,
    PreflightResult,
)

from archi.auth.cache import (
    cache_or_forced_live_change_probe,
    content_hash,
    content_hash_change_probe,
    resolve_repo_path,
)
from archi.enrichment.anonymizer import redact_email_addresses
from archi.sources._cache_report import (
    CacheUnusable,
    read_cache_json,
    skipped_items_status,
    unusable_cache_preflight,
    unusable_cache_run,
)


def checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(value: Any) -> str:
    """One cache value as plain text.

    A ServiceNow reference field can arrive as ``{"display_value": ...}``
    (or ``{"value": ...}``); other systems send ``{"name": ...}``. NUL bytes
    become spaces and email addresses are removed, so no address reaches a
    node attribute.
    """
    if value is None or isinstance(value, bool):
        return "" if value is None else str(value).lower()
    if isinstance(value, Mapping):
        for key in ("display_value", "name", "displayName", "login", "value"):
            if value.get(key) not in (None, ""):
                return clean(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(part for part in (clean(v) for v in value) if part)
    return redact_email_addresses(str(value).replace("\x00", " ")).strip()


def clean_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = [part for part in re.split(r"[,\n]", value)]
    if isinstance(value, Mapping):
        value = [value]
    return [text for text in (clean(v) for v in value) if text]


def number(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def real(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def person_id(system: str, display_name: str) -> str:
    """``person:<sha>`` for a name in one system.

    The system is hashed in, so the same name in ServiceNow and in Redmine
    does not become one node two sources both write.
    """
    return "person:" + sha256(f"{system}|{display_name.casefold().strip()}")[:16]


def person_fact(system: str, display_name: str, revision: Mapping[str, Any]) -> NodeFact:
    node_id = person_id(system, display_name)
    return NodeFact(
        node_id=node_id,
        subtype="person",
        attrs={"person_id": node_id, "display_name": display_name, "text": display_name},
        source_record_id={"person": f"{system}:{display_name}"},
        source_revision=dict(revision),
    )


def expected_record_count(meta_path: str | None, *, base: str | None = None) -> int | None:
    """``record_count`` from ``meta.json``, or None when there is no usable one."""
    if not meta_path:
        return None
    path = resolve_repo_path(meta_path, base=base)
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("record_count")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class CacheSource:
    """Base of the cache-backed cluster-manager readers."""

    name = ""
    profile = ""
    change_probe_kind = ""
    #: how the cache is named in health reasons, e.g. "ServiceNow records"
    description = ""
    #: a mutable upstream (tickets) or a snapshot (scheduler state)
    mutable = False

    def __init__(
        self,
        *,
        records_path: str,
        meta_path: str | None = None,
        credential_refs: Sequence[str] = (),
        base: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.records_path = records_path
        self.meta_path = meta_path
        self.credential_refs = tuple(credential_refs)
        self.base = base
        probe = cache_or_forced_live_change_probe if self.mutable else content_hash_change_probe
        self.change_probe = probe(
            cache_paths=self.cache_paths,
            config={"records_path": records_path, "meta_path": meta_path, **dict(config or {})},
            emit_targets=type(self),
            base=base,
        )

    # -- subclass hooks ----------------------------------------------------

    #: JSON type the cache holds (``list`` or ``dict``)
    expect: type | tuple[type, ...] = list

    def parse(self, payload: Any) -> tuple[list[Any], int]:
        """``(records, skipped item count)`` for the cache payload."""
        raise NotImplementedError

    def facts(self, records: list[Any], revision: dict[str, Any]) -> Iterator[Any]:
        raise NotImplementedError

    def record_total(self, records: list[Any]) -> int:
        """How many records the cache holds, as the downloader counts them in meta.json."""
        return len(records)

    # -- the reader contract -------------------------------------------------

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def _read(self) -> tuple[list[Any], int]:
        payload = read_cache_json(self.records_path, expect=self.expect, base=self.base)
        return self.parse(payload)

    def _verdict(self, records: list[Any], skipped: int, ok_reason: str) -> tuple[str, str, bool]:
        """``(status, reason, may claim a complete scope)``, shared by preflight and run."""
        count = self.record_total(records)
        expected = expected_record_count(self.meta_path, base=self.base)
        if expected is not None and expected != count:
            return (
                "endpoint_failed",
                f"{self.description} cache holds {count} records but meta.json reports {expected}; "
                "the cache is truncated or stale, so no complete scope is claimed",
                False,
            )
        status, reason = skipped_items_status(
            status="ok", reason=ok_reason, record_count=count, skipped_count=skipped
        )
        return status, reason, not skipped

    def preflight(self, mode: str = "live") -> PreflightResult:
        try:
            records, skipped = self._read()
        except CacheUnusable as exc:
            return unusable_cache_preflight(
                exc, source_name=self.name, required=True, credential_refs=self.credential_refs
            )
        status, reason, _ = self._verdict(records, skipped, f"local {self.description} cache present")
        return PreflightResult(
            source_name=self.name,
            status=status,
            mode="cache",
            required=True,
            credential_refs=self.credential_refs,
            record_count=self.record_total(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason=reason,
            checked_at=checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        try:
            records, skipped = self._read()
        except CacheUnusable as exc:
            return unusable_cache_run(exc, mode=mode, credential_refs=self.credential_refs)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": self.record_total(records),
        }
        status, reason, complete = self._verdict(records, skipped, f"local {self.description} cache used")
        return ConnectorRun(
            facts=self.facts(records, revision),
            completed_scope=mode in {"scope_complete", "reconcile"} and complete,
            run_mode=mode,
            health=ConnectorHealth(
                status=status,
                mode="cache",
                credential_refs=self.credential_refs,
                record_count=self.record_total(records),
                content_hash=revision["content_hash"],
                reason=reason,
            ),
        )


def write_cache(output_dir: Path, payload: Any, record_count: int, **meta: Any) -> dict[str, Any]:
    """Write ``records.json`` and ``meta.json`` atomically (the downloaders' shared tail).

    ``meta.json`` carries ``record_count`` and the content hash, which the
    readers cross-check; the rest of ``meta`` is the downloader's own notes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    info = {
        "fetched_at": datetime.now(timezone.utc).timestamp(),
        "record_count": record_count,
        "content_hash": hashlib.sha256(data).hexdigest(),
        **meta,
    }
    for name, body in (
        ("records.json", data),
        ("meta.json", json.dumps(info, indent=2, sort_keys=True).encode() + b"\n"),
    ):
        target = output_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(target)
    return info
