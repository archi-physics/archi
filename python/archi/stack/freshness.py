"""Source freshness: each source's age against its pack's `max_staleness`.

The SLOs come from `archi.lock` (`composed.freshness`, written by `archi
render` from the packs' `freshness:` keys). The ages come from okg's
watermarks through `okg doctor --check sources --deployment <name> --json`:
its `sources` list holds, per source, `coverage.watermark.last_reconcile_at`,
the time of the source's last completed reconcile (okg.source_watermarks).

`evaluate` is pure so doctor, context gates and tests share it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from archi.bundles.duration import format_age, parse_duration


def parse_timestamp(value: Any) -> Optional[datetime]:
    """An ISO-8601 timestamp as an aware UTC datetime; None when absent or unreadable."""
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _find(rows: Sequence[Mapping[str, Any]], source_id: str) -> Optional[Mapping[str, Any]]:
    """The okg source row for a rendered source id (`<pack>.<source>`).

    okg names a registry entry by the template's top-level key, which is the
    rendered id. No looser match: a bare `<source>` name could belong to
    another pack's source and pass for this one.
    """
    for row in rows:
        if row.get("name") == source_id:
            return row
    return None


def evaluate(
    slos: Mapping[str, Mapping[str, str]],
    okg_sources: Sequence[Mapping[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    """One `{source, age, max_staleness, ok, ...}` row per source with an SLO.

    A source okg does not list, or with no completed run, is not ok: its
    age is unknown and an unknown age cannot meet an SLO.
    """
    out: list[dict[str, Any]] = []
    for source_id in sorted(slos):
        limit_text = slos[source_id]["max_staleness"]
        limit = parse_duration(limit_text)
        row = _find(okg_sources, source_id)
        stamp = None
        reason = ""
        if row is None:
            reason = "okg's source registry does not list this source"
        else:
            stamp = parse_timestamp(((row.get("coverage") or {}).get("watermark") or {}).get("last_reconcile_at"))
            if stamp is None:
                reason = "no completed run recorded in okg's watermarks"
        age = max((now - stamp).total_seconds(), 0.0) if stamp else None
        ok = age is not None and age <= limit
        if age is not None and not ok:
            reason = f"last run {format_age(age)} ago, over the {limit_text} limit"
        out.append({
            "source": source_id,
            "age": format_age(age) if age is not None else None,
            "max_staleness": limit_text,
            "ok": ok,
            "age_seconds": round(age) if age is not None else None,
            "max_staleness_seconds": limit,
            "last_reconcile_at": stamp.isoformat() if stamp else None,
            "reason": reason,
        })
    return out
