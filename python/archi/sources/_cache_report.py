"""Cache preflight/health report helpers for optional cache-backed sources.

Ported from okg-deployments ``cms/cms_sources/_cache.py`` at
``main@f33a9c4`` (req.w2.sources-catalogs) — the ``cache_preflight_result``
/ ``cache_source_health`` pair that ``archi/auth/cache.py`` deliberately
left behind ("they move with the source ports that consume them").
Consumed by :mod:`archi.sources.dbs`, :mod:`archi.sources.conddb`, and
:mod:`archi.sources.wmstats`. Only change from the original: path
resolution takes the explicit ``base`` argument of
:func:`archi.auth.cache.resolve_repo_path` instead of assuming a
deployments-repo checkout.

Archi deviation (circleback-fixes): :func:`skipped_items_status` is new
— it is the shared policy for item-level parse tolerance. Sources that
skip unparseable cached items must not claim a completed scope over the
survivors (``missing_from_completed_scope`` would retract every record
the drifted items used to produce), and zero parsed records from a
non-empty payload is an endpoint failure rather than an empty success.
Also consumed by the inline-health catalog sources (cmssw, dqm, gocdb,
indico, hypernews).

Archi deviation (empty-cache-fails-loudly): :func:`read_cache_json` is
the shared reader for a JSON cache file. A cache that is missing,
unreadable, zero bytes, not valid JSON (a truncated write), or an empty
list/object is never an empty catalog: the reader raises
:class:`CacheUnusable`, and :func:`unusable_cache_run` /
:func:`unusable_cache_preflight` turn it into ``cache_missing`` health
with the path and ``completed_scope=False``. A payload of the wrong
JSON shape (drift, an error body) is ``endpoint_failed``. Under
``missing_from_completed_scope`` a complete scope over zero records is
an order to retract every record the source ever ingested, and okg
honours that order for ``ok`` and ``skipped_optional`` health alike —
so ``skipped_optional`` is no longer reported for an empty cache.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    PreflightResult,
)

from archi.auth.cache import content_hash, resolve_repo_path


class CacheUnusable(Exception):
    """A cache file that cannot back a run: no facts, no scope claim.

    ``status`` is ``cache_missing`` when there is no usable data (the
    file is missing, unreadable, zero bytes, truncated/invalid JSON,
    or an empty list/object) and ``endpoint_failed`` when the file
    holds valid JSON of the wrong shape (upstream drift or an error
    body written in place of the data).
    """

    def __init__(self, path: Path, problem: str, *, status: str) -> None:
        self.path = path
        self.problem = problem
        self.status = status
        super().__init__(self.reason)

    @property
    def reason(self) -> str:
        return f"cache file {self.path} {self.problem}"


def read_cache_json(
    path: str | Path,
    *,
    expect: type | tuple[type, ...],
    base: str | Path | None = None,
    allow_empty: bool = False,
) -> Any:
    """Load a JSON cache file or raise :class:`CacheUnusable`.

    ``expect`` is ``list`` or ``dict``, or a tuple of both. An empty
    container raises
    unless ``allow_empty`` — set only by a source whose upstream can
    genuinely be empty and which then refuses to claim a complete
    scope itself.
    """
    resolved = resolve_repo_path(path, base=base)
    if not resolved.is_file():
        raise CacheUnusable(resolved, "is missing", status="cache_missing")
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise CacheUnusable(
            resolved,
            f"could not be read ({exc.strerror or exc})",
            status="cache_missing",
        ) from exc
    if not raw.strip():
        raise CacheUnusable(
            resolved,
            f"is empty ({len(raw)} bytes); a failed or unfinished sync, "
            "not an empty catalog",
            status="cache_missing",
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CacheUnusable(
            resolved, "is not UTF-8 text", status="cache_missing"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CacheUnusable(
            resolved,
            f"is not valid JSON ({exc.msg} at line {exc.lineno} column "
            f"{exc.colno}); likely a truncated or interrupted write",
            status="cache_missing",
        ) from exc
    if not isinstance(payload, expect):
        raise CacheUnusable(
            resolved,
            f"holds a JSON {_json_kind(payload)} where a JSON "
            f"{_expected_kind(expect)} was expected (drifted or "
            "error-shaped payload)",
            status="endpoint_failed",
        )
    if not payload and not allow_empty:
        raise CacheUnusable(
            resolved,
            f"holds an empty JSON {_json_kind(payload)}; a failed or "
            "unsynced fetch, not an empty catalog",
            status="cache_missing",
        )
    return payload


def unusable_cache_run(
    error: CacheUnusable,
    *,
    mode: str,
    health_mode: str = "cache",
    credential_refs: tuple[str, ...] = (),
    alias_refs: Mapping[str, tuple[str, ...]] | None = None,
) -> ConnectorRun:
    """The run for an unusable cache: no facts, no complete scope."""
    extra: dict[str, Any] = {}
    if alias_refs:
        extra["alias_refs"] = dict(alias_refs)
    return ConnectorRun(
        facts=(),
        completed_scope=False,
        run_mode=mode,
        health=ConnectorHealth(
            status=error.status,
            mode=health_mode,
            credential_refs=credential_refs,
            cache_path=str(error.path),
            record_count=0,
            reason=(
                f"{error.reason}; no facts emitted and no complete "
                "scope claimed"
            ),
            checked_at=_checked_at(),
            **extra,
        ),
    )


def unusable_cache_preflight(
    error: CacheUnusable,
    *,
    source_name: str,
    required: bool,
    mode: str = "cache",
    credential_refs: tuple[str, ...] = (),
    alias_refs: Mapping[str, tuple[str, ...]] | None = None,
) -> PreflightResult:
    """The preflight verdict :func:`unusable_cache_run` would reach."""
    extra: dict[str, Any] = {}
    if alias_refs:
        extra["alias_refs"] = dict(alias_refs)
    return PreflightResult(
        source_name=source_name,
        status=error.status,
        mode=mode,
        required=required,
        credential_refs=credential_refs,
        cache_path=str(error.path),
        record_count=0,
        reason=error.reason,
        checked_at=_checked_at(),
        **extra,
    )


def _expected_kind(expect: type | tuple[type, ...]) -> str:
    kinds = expect if isinstance(expect, tuple) else (expect,)
    return " or ".join(_json_kind(kind()) for kind in kinds)


def _json_kind(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def cache_preflight_result(
    *,
    source_name: str,
    description: str,
    cache_paths: Iterable[str | Path],
    records: Iterable[Any] | None,
    required: bool = False,
    mode: str = "cache",
    base: str | None = None,
    skipped_count: int = 0,
) -> PreflightResult:
    paths = tuple(cache_paths)
    resolved_paths = tuple(resolve_repo_path(p, base=base) for p in paths)
    missing = [p for p in resolved_paths if not p.is_file()]
    if missing:
        return PreflightResult(
            source_name=source_name,
            status="cache_missing",
            mode=mode,
            required=required,
            cache_path=", ".join(str(p) for p in missing),
            reason=f"{description} cache file is missing",
            checked_at=_checked_at(),
        )
    count = len(tuple(records or ()))
    # Same verdict cache_source_health gives run(): all items
    # unparseable is endpoint_failed, not an empty success.
    status, reason = skipped_items_status(
        status=_cache_status(count),
        reason=_cache_reason(description, count, observed=True),
        record_count=count,
        skipped_count=skipped_count,
    )
    return PreflightResult(
        source_name=source_name,
        status=status,
        mode=mode,
        required=required,
        record_count=count,
        content_hash=content_hash(paths, base=base),
        reason=reason,
        checked_at=_checked_at(),
    )


def cache_source_health(
    *,
    description: str,
    cache_paths: Iterable[str | Path],
    record_count: int,
    skipped_count: int = 0,
    base: str | None = None,
) -> ConnectorHealth:
    status, reason = skipped_items_status(
        status=_cache_status(record_count),
        reason=_cache_reason(description, record_count, observed=False),
        record_count=record_count,
        skipped_count=skipped_count,
    )
    return ConnectorHealth(
        status=status,
        mode="cache",
        record_count=record_count,
        content_hash=content_hash(cache_paths, base=base),
        reason=reason,
        checked_at=_checked_at(),
    )


def skipped_items_status(
    *,
    status: str,
    reason: str,
    record_count: int,
    skipped_count: int,
) -> tuple[str, str]:
    """Degrade a health claim when item-level parsing skipped records.

    Per-item tolerance stays (one bad record must not fail the whole
    run), but the run must stop claiming completeness: callers gate
    ``completed_scope`` on ``skipped_count == 0`` so schema drift never
    turns into a healthy completed-scope run that retracts the skipped
    records. Zero parsed records from a non-empty payload becomes
    ``endpoint_failed`` instead of an empty success.
    """
    if not skipped_count:
        return status, reason
    if not record_count:
        return (
            "endpoint_failed",
            f"{reason}; all {skipped_count} cached item(s) failed to "
            "parse (possible schema drift); no records emitted and no "
            "complete scope claimed",
        )
    return (
        status,
        f"{reason}; skipped {skipped_count} unparseable cached item(s); "
        "no complete scope claimed",
    )


def _cache_status(record_count: int) -> str:
    # Zero records with nothing skipped means the payload itself was
    # empty. read_cache_json refuses that before a caller gets here;
    # this is the backstop, and it must not be skipped_optional, which
    # okg accepts as healthy enough to retract under a complete scope.
    return "ok" if record_count else "cache_missing"


def _cache_reason(
    description: str,
    record_count: int,
    *,
    observed: bool,
) -> str:
    if record_count:
        action = "observed" if observed else "used"
        return f"local {description} cache {action}"
    return (
        f"local {description} cache yielded no records"
    )


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()
