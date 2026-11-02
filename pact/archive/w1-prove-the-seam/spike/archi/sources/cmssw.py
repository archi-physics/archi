"""CMSSW release catalog source — W1 seam proof.

Fetches the public cms-bot ``releases.map`` (no auth) into a local
cache, then emits one ``cmssw_release`` node per release label.
Mirrors the proven adapter idiom from okg-deployments/cms (cache file
+ content-hash probe), with the fetch folded into the adapter so W1
proves a real network read. W2 note: a periodically-refreshed remote
catalog wants a mutable_api probe — the content-hash probe here only
sees the cache, so it cannot detect upstream change on its own.
"""
from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from okg.substrate.library.sources.base import (
    NodeFact,
    SourceHealth,
    SourceRun,
)
from okg.substrate.library.sources.content_hash_probe import ContentHashProbe

RELEASES_MAP_URL = (
    "https://raw.githubusercontent.com/cms-sw/cms-bot/master/releases.map"
)
_VERSION_RE = re.compile(r"^CMSSW_(\d+)_(\d+)_(\d+)(?:_(.+))?$")


def _parse_releases(raw: str, limit: int) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = dict(
            part.split("=", 1) for part in line.split(";") if "=" in part
        )
        label = fields.get("label", "")
        m = _VERSION_RE.match(label)
        if not m:
            continue
        rec = by_label.setdefault(
            label,
            {
                "release": label,
                "release_type": fields.get("type", ""),
                "state": fields.get("state", ""),
                "architecture": [],
                "major": int(m.group(1)),
                "minor": int(m.group(2)),
                "patch": int(m.group(3)),
                "suffix": m.group(4) or "",
                "text": f"{label} {fields.get('type', '')} {fields.get('state', '')}".strip(),
            },
        )
        arch = fields.get("architecture", "")
        if arch and arch not in rec["architecture"]:
            rec["architecture"].append(arch)
    records = sorted(
        by_label.values(),
        key=lambda r: (r["major"], r["minor"], r["patch"], r["release"]),
    )
    for rec in records:
        rec["architecture"] = sorted(rec["architecture"])
    if limit > 0:
        records = records[-limit:]
    return records


class CMSSWReleaseSource:
    """Public CMSSW release catalog (cms-bot releases.map)."""

    name = "cmssw_releases"
    profile = "reference_catalog"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        cache_path: str,
        url: str = RELEASES_MAP_URL,
        limit: int = 300,
        fetch: bool = True,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.url = url
        self.limit = int(limit)
        self.fetch = bool(fetch)
        self.change_probe = ContentHashProbe(
            content_items=self._probe_items,
            config={
                "cache_path": str(cache_path),
                "url": url,
                "limit": self.limit,
            },
            emit_targets=CMSSWReleaseSource,
        )

    def _probe_items(self) -> list[tuple[str, Path]]:
        if self.cache_path.is_file():
            return [(str(self.cache_path), self.cache_path)]
        return []

    def _download(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.url, timeout=60) as resp:
            self.cache_path.write_bytes(resp.read())

    def run(
        self,
        run_id: str,
        *,
        mode: str = "cursor",
        sync_scope: Optional[Mapping[str, Any]] = None,
    ) -> SourceRun:
        if self.fetch or not self.cache_path.is_file():
            self._download()
        raw = self.cache_path.read_text(encoding="utf-8", errors="replace")
        records = _parse_releases(raw, self.limit)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        def _facts() -> Iterator[NodeFact]:
            for rec in records:
                yield NodeFact(
                    node_id=f"cmssw_release:{rec['release']}",
                    subtype="cmssw_release",
                    attrs=rec,
                    source_record_id={"release": rec["release"]},
                    source_revision={
                        "content_hash": digest,
                        "run_id": run_id,
                    },
                )

        return SourceRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=SourceHealth(
                status="ok",
                mode="live",
                record_count=len(records),
                content_hash=digest,
                reason="releases.map fetched from cms-bot",
            ),
        )
