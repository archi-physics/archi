"""CMSSW release catalog source — cache-backed, with releases.map option.

Merge of two lineages (req.w2.sources-catalogs):

- okg-deployments ``cms/cms_sources/cmssw.py`` (324 LOC,
  ``CMSSWReleaseSource``) at ``main@f33a9c4`` — canonical. Reads a JSON
  records cache and emits ``cmssw_release`` nodes (individual releases
  + ``CMSSW_X_Y_X`` family nodes) plus deterministic ``supersedes``
  edges between known predecessor releases.
- The archi W1 seam-proof module this file previously held — the
  public cms-bot ``releases.map`` fetch (no auth). Kept as an *option*:
  pass ``releases_map_url``/``map_cache_path`` (and ``fetch=True``) to
  build the same record set from the fetched map instead of the JSON
  cache. Emission is the cms path's in both modes (families +
  supersedes edges), so the W1 mode now also emits supersedes edges.

Changes from the cms original: cache/probe helpers come from
:mod:`archi.auth.cache` with an explicit ``base`` parameter; the
hardcoded ``data/cms/cmssw-releases/records.json`` path is a parameter
(default keeps the cms layout minus the ``cms/`` segment).

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template (compose
``archi/schemas/operations.yaml`` + ``archi/schemas/bridges/operations.yaml``
into the deployment; ``output_scope_summary`` must accompany
``output_signature``; standard ``sync:`` block). ::

    cmssw_releases:
      module: archi.sources.cmssw
      class: CMSSWReleaseSource
      ownership_id: <instance>.cmssw-releases
      admission_policy:
        producer_id: <instance>.cmssw-releases
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: cmssw_releases
        output_signature:
          nodes:
            - {subtype: cmssw_release}
          edges:
            - {src_subtype: cmssw_release, edge_type: supersedes, dst_subtype: cmssw_release}
        output_scope_summary:
          summary: CMSSW releases, release families, and supersedes chains
          nodes: [cmssw_release]
          edges:
            - cmssw_release supersedes cmssw_release
      source_class: reference_catalog
      record_identity_kind: domain_key
      record_identity_fields: [release]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: true
      params:
        # cms default; the cms deployment used data/cms/cmssw-releases/
        records_path: data/cmssw-releases/records.json
        # W1 live option (instead of a maintained records cache):
        # map_cache_path: data/cmssw-releases/releases.map
        # releases_map_url: https://raw.githubusercontent.com/cms-sw/cms-bot/master/releases.map
        # fetch: true
        # limit: 300
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
    ConnectorRun,
)

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
    load_json,
    resolve_repo_path,
)
from archi.sources._cache_report import skipped_items_status

RELEASES_MAP_URL = (
    "https://raw.githubusercontent.com/cms-sw/cms-bot/master/releases.map"
)

_VERSION_RE = re.compile(
    r"^CMSSW_(\d+)_(\d+)_(\d+)"
    r"(?:_(patch\d+|pre\d+|hltpatch\d+|ROOT\d+.*|.*))?$"
)


@dataclass(frozen=True)
class CMSSWReleaseRecord:
    label: str
    xml_type: str
    state: str
    architecture: tuple[str, ...]
    release_notes: str = ""
    release_date: str = ""

    @property
    def node_id(self) -> str:
        return f"cmssw_release:{self.label}"


class CMSSWReleaseSource:
    """Cache-backed CMSSW release catalog source.

    Default mode reads a local JSON records cache. When
    ``map_cache_path`` is set the source instead builds the record set
    from the public cms-bot ``releases.map`` (fetched into that cache
    when ``fetch`` is true or the cache is absent — the W1 seam-proof
    path). Both modes emit one ``cmssw_release`` node per release plus
    family nodes and deterministic ``supersedes`` edges.
    """

    name = "cmssw_releases"
    profile = "reference_catalog"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/cmssw-releases/records.json",
        map_cache_path: str | None = None,
        releases_map_url: str = RELEASES_MAP_URL,
        fetch: bool = False,
        limit: int = 0,
        base: str | None = None,
        cache_path: str | None = None,
    ) -> None:
        # W1 compatibility: `cache_path` was the W1 name for the
        # releases.map cache location.
        if cache_path is not None and map_cache_path is None:
            map_cache_path = cache_path
        self.records_path = records_path
        self.map_cache_path = map_cache_path
        self.releases_map_url = releases_map_url
        self.fetch = bool(fetch)
        self.limit = int(limit)
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={
                "records_path": self.records_path,
                "map_cache_path": self.map_cache_path,
                "releases_map_url": self.releases_map_url,
                "limit": self.limit,
            },
            emit_targets=CMSSWReleaseSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        if self.map_cache_path is not None:
            return (self.map_cache_path,)
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        path = resolve_repo_path(self.cache_paths[0], base=self.base)
        if not path.is_file() and not (
            self.map_cache_path is not None and self.fetch
        ):
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="cache",
                required=True,
                cache_path=str(path),
                reason="CMSSW release cache file is missing",
                checked_at=_checked_at(),
            )
        if not path.is_file():
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="live",
                required=True,
                endpoint=self.releases_map_url,
                reason="releases.map will be fetched from cms-bot",
                checked_at=_checked_at(),
            )
        records = self._records()
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=True,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason="local CMSSW release cache present",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        records, skipped, truncated = self._records_with_details()
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
            "n_release_families": len(_release_families(records)),
        }

        def _facts() -> Iterator[Any]:
            for family in _release_families(records):
                yield _family_node_fact(family, revision)
            for record in records:
                yield _node_fact(record, revision)
            yield from _supersedes_edges(records, revision)

        status, reason = skipped_items_status(
            status="ok",
            reason=(
                "local CMSSW release cache used"
                if self.map_cache_path is None
                else "cms-bot releases.map used"
            ),
            record_count=len(records),
            skipped_count=skipped,
        )
        if truncated:
            # A truncated newest-N window must not claim a complete
            # scope: every upstream append would retract the oldest
            # in-window release (sliding retention window).
            reason += (
                f"; limit={self.limit} truncated the release list; "
                "no complete scope claimed"
            )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"}
                and not skipped
                and not truncated
            ),
            run_mode=mode,
            health=ConnectorHealth(
                status=status,
                mode="cache",
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason=reason,
            ),
        )

    def _records(self) -> list[CMSSWReleaseRecord]:
        return self._records_with_details()[0]

    def _records_with_details(
        self,
    ) -> tuple[list[CMSSWReleaseRecord], int, bool]:
        """Records plus (skipped_item_count, limit_truncated)."""
        if self.map_cache_path is not None:
            return self._records_from_map()
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of releases"
            )
        records: list[CMSSWReleaseRecord] = []
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                skipped += 1
                continue
            architecture_raw = item.get("architecture") or ()
            if isinstance(architecture_raw, str):
                architecture = (architecture_raw,)
            else:
                architecture = tuple(str(v) for v in architecture_raw)
            records.append(CMSSWReleaseRecord(
                label=label,
                xml_type=str(item.get("type") or ""),
                state=str(item.get("state") or ""),
                architecture=architecture,
                release_notes=str(item.get("release_notes") or ""),
                release_date=str(item.get("release_date") or ""),
            ))
        return records, skipped, False

    def _records_from_map(
        self,
    ) -> tuple[list[CMSSWReleaseRecord], int, bool]:
        """W1 path: build records from the cms-bot ``releases.map`` cache."""
        path = resolve_repo_path(self.map_cache_path, base=self.base)
        if self.fetch or not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(
                self.releases_map_url, timeout=60
            ) as resp:
                path.write_bytes(resp.read())
        raw = path.read_text(encoding="utf-8", errors="replace")
        full, skipped = _parse_map_with_skips(raw)
        records = full[-self.limit:] if self.limit > 0 else full
        return records, skipped, len(records) < len(full)


def parse_releases_map(raw: str, limit: int = 0) -> list[CMSSWReleaseRecord]:
    """Parse cms-bot ``releases.map`` lines into release records (W1 path)."""
    records, _skipped = _parse_map_with_skips(raw)
    if limit > 0:
        return records[-limit:]
    return records


def _parse_map_with_skips(
    raw: str,
) -> tuple[list[CMSSWReleaseRecord], int]:
    """Parse map lines, counting the ones that yield no release.

    A non-empty line without a ``label`` matching the CMSSW version
    pattern is a skip: format drift (e.g. a renamed key) must degrade
    the run's scope claim instead of silently emptying the record set
    under a healthy completed scope.
    """
    by_label: dict[str, dict[str, Any]] = {}
    skipped = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = dict(
            part.split("=", 1) for part in line.split(";") if "=" in part
        )
        label = fields.get("label", "")
        if not _VERSION_RE.match(label):
            skipped += 1
            continue
        rec = by_label.setdefault(
            label,
            {
                "label": label,
                "type": fields.get("type", ""),
                "state": fields.get("state", ""),
                "architecture": [],
            },
        )
        arch = fields.get("architecture", "")
        if arch and arch not in rec["architecture"]:
            rec["architecture"].append(arch)
    ordered = sorted(
        by_label.values(),
        key=lambda r: _sort_key(r["label"]),
    )
    return [
        CMSSWReleaseRecord(
            label=rec["label"],
            xml_type=rec["type"],
            state=rec["state"],
            architecture=tuple(sorted(rec["architecture"])),
        )
        for rec in ordered
    ], skipped


def _sort_key(label: str) -> tuple[int, int, int, str]:
    match = _VERSION_RE.match(label)
    if not match:
        return (0, 0, 0, label)
    major, minor, patch, _suffix = match.groups()
    return (int(major), int(minor), int(patch), label)


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_version(label: str) -> dict[str, Any]:
    match = _VERSION_RE.match(label)
    if not match:
        return {}
    major, minor, patch, suffix = match.groups()
    parsed: dict[str, Any] = {
        "major": int(major),
        "minor": int(minor),
        "patch": int(patch),
    }
    if suffix:
        parsed["suffix"] = suffix
    return parsed


def _release_type(label: str, xml_type: str, suffix: str | None) -> str:
    if not suffix:
        return xml_type or "release"
    lowered = suffix.lower()
    if lowered.startswith("pre"):
        return "pre-release"
    if lowered.startswith("hltpatch"):
        return "hlt-patch"
    if lowered.startswith("patch"):
        return "patch"
    return xml_type or "variant"


def _node_fact(
    record: CMSSWReleaseRecord,
    revision: dict[str, Any],
) -> NodeFact:
    version = _parse_version(record.label)
    suffix = version.get("suffix")
    release_type = _release_type(record.label, record.xml_type, suffix)
    architecture = ",".join(record.architecture)
    text_parts = [
        record.label,
        release_type,
        record.state,
        architecture,
        record.release_notes,
    ]
    attrs = {
        "label": record.label,
        "name": record.label,
        "release": record.label,
        "release_type": release_type,
        "state": record.state,
        "architecture": architecture,
        "release_date": record.release_date,
        "text": " ".join(p for p in text_parts if p).strip(),
        **version,
    }
    return NodeFact(
        node_id=record.node_id,
        subtype="cmssw_release",
        attrs=attrs,
        source_record_id={"release": record.label},
        source_revision=revision,
    )


def _release_families(records: list[CMSSWReleaseRecord]) -> tuple[str, ...]:
    families: set[str] = set()
    for record in records:
        version = _parse_version(record.label)
        major = version.get("major")
        minor = version.get("minor")
        if major is None or minor is None:
            continue
        families.add(f"CMSSW_{major}_{minor}_X")
    return tuple(sorted(families))


def _family_node_fact(
    family: str,
    revision: dict[str, Any],
) -> NodeFact:
    version = _parse_family(family)
    return NodeFact(
        node_id=f"cmssw_release:{family}",
        subtype="cmssw_release",
        attrs={
            "label": family,
            "name": family,
            "release": family,
            "release_type": "release_family",
            "state": "family",
            "architecture": "",
            "release_date": "",
            "text": family,
            **version,
        },
        source_record_id={"release_family": family},
        source_revision=revision,
    )


def _parse_family(label: str) -> dict[str, Any]:
    match = re.match(r"^CMSSW_(\d+)_(\d+)_X$", label)
    if not match:
        return {}
    major, minor = match.groups()
    return {
        "major": int(major),
        "minor": int(minor),
        "family": True,
    }


def _supersedes_edges(
    records: list[CMSSWReleaseRecord],
    revision: dict[str, Any],
) -> Iterator[EdgeFact]:
    known = {record.label for record in records}
    for record in records:
        predecessor = _find_predecessor(record.label)
        if predecessor not in known:
            continue
        yield EdgeFact(
            src=record.node_id,
            dst=f"cmssw_release:{predecessor}",
            edge_type="supersedes",
            provenance="derived_deterministic",
            confidence=1.0,
            source_record_id={"release": record.label},
            source_revision=revision,
        )


def _find_predecessor(label: str) -> str | None:
    match = _VERSION_RE.match(label)
    if not match:
        return None
    major, minor, patch, suffix = match.groups()
    base = f"CMSSW_{major}_{minor}_{patch}"

    if not suffix:
        patch_n = int(patch)
        if patch_n == 0:
            return None
        return f"CMSSW_{major}_{minor}_{patch_n - 1}"

    patch_match = re.match(r"^patch(\d+)$", suffix)
    if patch_match:
        patch_n = int(patch_match.group(1))
        if patch_n > 1:
            return f"{base}_patch{patch_n - 1}"
        return base

    pre_match = re.match(r"^pre(\d+)$", suffix)
    if pre_match:
        pre_n = int(pre_match.group(1))
        if pre_n > 1:
            return f"{base}_pre{pre_n - 1}"
        return None

    hlt_match = re.match(r"^hltpatch(\d+)$", suffix)
    if hlt_match:
        hlt_n = int(hlt_match.group(1))
        if hlt_n > 1:
            return f"{base}_hltpatch{hlt_n - 1}"
        return base

    return base
