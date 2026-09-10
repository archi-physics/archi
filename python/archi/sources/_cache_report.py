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
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from okg.deployment import (
    ConnectorHealth,
    PreflightResult,
)

from archi.auth.cache import content_hash, resolve_repo_path


def cache_preflight_result(
    *,
    source_name: str,
    description: str,
    cache_paths: Iterable[str | Path],
    records: Iterable[Any] | None,
    required: bool = False,
    mode: str = "cache",
    base: str | None = None,
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
    return PreflightResult(
        source_name=source_name,
        status=_cache_status(count),
        mode=mode,
        required=required,
        record_count=count,
        content_hash=content_hash(paths, base=base),
        reason=_cache_reason(description, count, observed=True),
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
    return "ok" if record_count else "skipped_optional"


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
        f"local {description} cache is present but empty; "
        "no live data fetch attempted"
    )


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()
