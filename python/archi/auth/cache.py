"""Cache-file helpers for cache-backed Archi sources.

Rewritten from okg-deployments ``cms/cms_sources/_cache.py`` (207 LOC)
unified with its trimmed fork ``wisdqm/wisdqm_sources/_cache.py``
(55 LOC), both at ``main@f33a9c4``. Changes from the originals:

- ``resolve_repo_path`` no longer assumes it runs inside a deployments
  repo checkout (the originals walked up to a hardcoded repo root). The
  base directory is now explicit: a ``base`` argument, else the
  ``ARCHI_DATA_ROOT`` environment variable, else the current working
  directory.
- The per-deployment ``data/<name>`` prefix stripping
  (``OKG_CMS_DATA_ROOT`` / ``OKG_WISDQM_DATA_ROOT``) is dropped;
  instances point ``ARCHI_DATA_ROOT`` at whichever directory their
  relative cache paths are written against.
- The cache preflight/health helpers (``cache_preflight_result``,
  ``cache_source_health``, ``json_record_count``) are not ported here;
  they move with the source ports that consume them.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from okg.deployment import ContentHashProbe
from okg.deployment import MutableApiProbe

DATA_ROOT_ENV = "ARCHI_DATA_ROOT"


def data_root(base: str | Path | None = None) -> Path:
    """Return the directory relative cache paths resolve against."""
    if base is not None:
        return Path(base).expanduser()
    raw = os.environ.get(DATA_ROOT_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path.cwd()


def resolve_repo_path(
    path: str | Path,
    *,
    base: str | Path | None = None,
) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return data_root(base) / p


def load_json(path: str | Path, *, base: str | Path | None = None) -> Any:
    p = resolve_repo_path(path, base=base)
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def content_hash(
    paths: Iterable[str | Path],
    *,
    base: str | Path | None = None,
) -> str:
    digest = hashlib.sha256()
    resolved = sorted((str(p), resolve_repo_path(p, base=base)) for p in paths)
    for label, path in resolved:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def content_hash_change_probe(
    *,
    cache_paths: Iterable[str | Path],
    config: dict[str, Any],
    emit_targets: Any,
    base: str | Path | None = None,
) -> ContentHashProbe:
    """Change probe over cache files that exist at probe time."""
    paths = tuple(cache_paths)

    def _items() -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        for raw in paths:
            path = resolve_repo_path(raw, base=base)
            if path.is_file():
                out.append((str(raw), path))
        return out

    return ContentHashProbe(
        content_items=_items,
        config=config,
        emit_targets=emit_targets,
    )


def cache_or_forced_live_change_probe(
    *,
    cache_paths: Iterable[str | Path],
    config: dict[str, Any],
    emit_targets: Any,
    base: str | Path | None = None,
) -> MutableApiProbe:
    """Probe mutable sources that have cache artifacts but no cheap live token.

    When a cache exists, use its content hash as the version token. Without a
    cache, return a fresh token so the runner performs the full live read
    rather than silently skipping a mutable upstream.
    """
    paths = tuple(cache_paths)

    def _version() -> str:
        existing = [
            (raw, resolve_repo_path(raw, base=base))
            for raw in paths
            if resolve_repo_path(raw, base=base).is_file()
        ]
        if not existing:
            return datetime.now(timezone.utc).isoformat()
        digest = hashlib.sha256()
        for raw, path in sorted(existing, key=lambda item: str(item[0])):
            digest.update(str(raw).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    return MutableApiProbe(
        version_fn=_version,
        config=config,
        emit_targets=emit_targets,
    )
