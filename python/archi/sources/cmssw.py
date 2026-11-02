"""CMSSW release catalog source — cache-backed, with releases.map option.

Merge of two lineages (req.w2.sources-catalogs):

- okg-deployments ``cms/cms_sources/cmssw.py`` (324 LOC,
  ``CMSSWReleaseSource``) at ``main@f33a9c4`` — canonical. Reads a JSON
  records cache and emits ``cmssw_release`` nodes (individual releases
  + one release-family node per ``major_minor`` series) plus
  deterministic ``supersedes`` edges between known predecessor releases.
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

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from okg.deployment import (
    ConnectorHealth,
    ConnectorRun,
    EdgeFact,
    NodeFact,
    PreflightResult,
)

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
    load_json,
    resolve_repo_path,
)
from archi.sources._cache_report import skipped_items_status
from archi.sources._run_modes import COMPLETED_SCOPE_RUN_MODES
from archi.sources._sdk_adapter import ReaderAdapter

RELEASES_MAP_URL = (
    "https://raw.githubusercontent.com/cms-sw/cms-bot/master/releases.map"
)

# A release is three numbers, optionally followed by a build suffix. The
# suffix arrives in one of two shapes, and they are deliberately NOT given
# the same freedom:
#
#   `_<suffix>`  -- the usual form (`_patch2`, `_pre3`, `_ROOT6...`). The
#                   trailing `.*` alternative accepts any separated suffix.
#                   That is pre-existing latitude, left as it is here;
#   `<suffix>`   -- no separator, which cms-bot writes for a few real old
#                   builds: `CMSSW_1_4_3g483`, a Geant4 8.3 variant carried
#                   on two architectures beside `CMSSW_1_4_3`. Restricted to
#                   lowercase letters followed by digits, because without a
#                   separator there is nothing to tell a build suffix from a
#                   typo glued to a valid triple -- `CMSSW_14_0_1TYPO` would
#                   otherwise become a release with a `supersedes` edge to
#                   the real one.
#
# The two shapes capture into groups 4 and 5; `_version_parts` collapses
# them so no caller has to know which branch matched.
_VERSION_RE = re.compile(
    r"^CMSSW_(\d+)_(\d+)_(\d+)"
    r"(?:_(patch\d+|pre\d+|hltpatch\d+|ROOT\d+.*|.*)|([a-z]+\d+))?$"
)
# Products cms-bot ships in the same map that are not CMSSW releases. Closed
# by design; see `_parse_map_with_skips`.
_OTHER_PRODUCTS = ("ECALTBH4_",)
# Integration-build FAMILY rows: two version numbers, then `X`, with optional
# word segments before it (`CMSSW_9_4_AN_X`) or after (`CMSSW_6_2_X_SLHC`).
# A family names a release series, not a release. cms-bot lists these rows
# with `type=Development` and `state=IB`; each one is kept as a release
# record of its own (`release_kind: integration-series`), under its own label.
_FAMILY_RE = re.compile(
    r"^CMSSW_\d+_\d+_(?:[A-Za-z0-9]+_)*X(?:_[A-Za-z0-9]+)*$"
)
# Major and minor of an integration-series label (`CMSSW_9_4_AN_X` -> 9, 4).
_SERIES_RE = re.compile(r"^CMSSW_(\d+)_(\d+)_")

# Release-family nodes are DERIVED: one per major/minor series that has at
# least one concrete release. They live in their own id namespace,
# `cmssw_release_family:CMSSW_<major>_<minor>`, so they can never take the id
# `cmssw_release:<label>` that belongs to a source row. They once used
# `cmssw_release:CMSSW_<major>_<minor>_X`, which is also the label of the
# cms-bot integration-series row for that series, so one of the two records
# was lost. Every `cmssw_release:<label>` id now names exactly the source row
# with that label.
FAMILY_ID_PREFIX = "cmssw_release_family"


def family_key(major: int | str, minor: int | str) -> str:
    """The family's own name and alias value: `CMSSW_15_0`."""
    return f"CMSSW_{int(major)}_{int(minor)}"


def family_node_id(major: int | str, minor: int | str) -> str:
    """`cmssw_release_family:CMSSW_15_0` for the 15.0 series."""
    return f"{FAMILY_ID_PREFIX}:{family_key(major, minor)}"


@dataclass(frozen=True)
class CMSSWReleaseRecord:
    label: str
    xml_type: str
    state: str
    architecture: tuple[str, ...]
    release_notes: str = ""
    release_date: str = ""
    # Architectures the map marks `prodarch=1`; empty when none is marked.
    production_architecture: tuple[str, ...] = ()

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
        map_cache_digest: str | None = None,
    ) -> None:
        # W1 compatibility: `cache_path` was the W1 name for the
        # releases.map cache location.
        if cache_path is not None and map_cache_path is None:
            map_cache_path = cache_path
        if map_cache_digest is not None:
            if not isinstance(map_cache_digest, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", map_cache_digest
            ):
                raise ValueError("map_cache_digest must be sha256:<64 lowercase hex>")
            if not map_cache_path or fetch:
                raise ValueError(
                    "a frozen CMSSW map requires map_cache_path and fetch=False"
                )
        self.records_path = records_path
        self.map_cache_path = map_cache_path
        self.map_cache_digest = map_cache_digest
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
                "map_cache_digest": self.map_cache_digest,
                "input_mode": "frozen" if self.map_cache_digest else "unpinned",
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
        if not path.is_file() and not (self.map_cache_path is not None and self.fetch):
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
        if self.map_cache_digest is not None:
            records, _, _, revision_hash = self._read_frozen_map()
        else:
            records = self._records()
            revision_hash = content_hash(self.cache_paths, base=self.base)
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=True,
            record_count=len(records),
            content_hash=revision_hash,
            reason="local CMSSW release cache present",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        if self.map_cache_digest is not None:
            records, skipped, truncated, revision_hash = self._read_frozen_map()
        else:
            records, skipped, truncated = self._records_with_details()
            revision_hash = content_hash(self.cache_paths, base=self.base)
        revision = {
            "run_id": run_id,
            "content_hash": revision_hash,
            "n_records": len(records),
            "n_release_families": len(_release_families(records)),
        }
        if self.map_cache_digest is not None:
            revision["map_cache_digest"] = self.map_cache_digest

        def _facts() -> Iterator[Any]:
            for major, minor in _release_families(records):
                yield family_node_fact(major, minor, revision)
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
        if not records and not skipped and not truncated:
            # Last line of defence. A complete scope over zero releases is a
            # retraction order for the whole catalog -- `deletion_semantics:
            # missing_from_completed_scope` -- so it may never be claimed by
            # accident. A real catalog is never empty; an empty record set
            # means the input was missing, empty, or shaped in a way this
            # parser did not recognise, whether or not anything was counted
            # as a skip. Covers the frozen mode too: both modes end here.
            # Only worded when nothing else already withheld the claim, so the
            # health reason never says it twice.
            reason += "; no releases parsed; no complete scope claimed"
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(
                mode in COMPLETED_SCOPE_RUN_MODES
                and not skipped
                and not truncated
                and bool(records)
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
        if self.map_cache_digest is not None:
            return self._read_frozen_map()[:3]
        if self.map_cache_path is not None:
            return self._records_from_map()
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(f"{self.records_path}: expected a JSON list of releases")
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
            records.append(
                CMSSWReleaseRecord(
                    label=label,
                    xml_type=str(item.get("type") or ""),
                    state=str(item.get("state") or ""),
                    architecture=architecture,
                    release_notes=str(item.get("release_notes") or ""),
                    release_date=str(item.get("release_date") or ""),
                    production_architecture=_as_tuple(
                        item.get("production_architecture")
                    ),
                )
            )
        return records, skipped, False

    def _read_frozen_map(
        self,
    ) -> tuple[list[CMSSWReleaseRecord], int, bool, str]:
        """Verify, parse and describe one buffer without fetching or rereading it."""
        path = resolve_repo_path(self.map_cache_path, base=self.base)
        body = path.read_bytes()
        if "sha256:" + hashlib.sha256(body).hexdigest() != self.map_cache_digest:
            raise ValueError("frozen CMSSW map digest mismatch")
        full, skipped = _parse_map_with_skips(body.decode("utf-8"))
        records = full[-self.limit :] if self.limit > 0 else full
        # Keep the cache helper's path-aware revision format, using the bytes
        # already verified rather than reading a potentially changed file.
        revision_hash = hashlib.sha256(
            str(self.map_cache_path).encode("utf-8") + b"\0" + body + b"\0"
        ).hexdigest()
        return records, skipped, len(records) < len(full), revision_hash

    def _records_from_map(
        self,
    ) -> tuple[list[CMSSWReleaseRecord], int, bool]:
        """W1 path: build records from the cms-bot ``releases.map`` cache."""
        path = resolve_repo_path(self.map_cache_path, base=self.base)
        if self.fetch or not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(self.releases_map_url, timeout=60) as resp:
                path.write_bytes(resp.read())
        raw = path.read_text(encoding="utf-8", errors="replace")
        full, skipped = _parse_map_with_skips(raw)
        records = full[-self.limit :] if self.limit > 0 else full
        return records, skipped, len(records) < len(full)


def parse_releases_map(raw: str, limit: int = 0) -> list[CMSSWReleaseRecord]:
    """Parse cms-bot ``releases.map`` lines into release records (W1 path)."""
    records, _skipped = _parse_map_with_skips(raw)
    if limit > 0:
        return records[-limit:]
    return records


def _version_parts(label: str) -> tuple[str, str, str, str | None] | None:
    """(major, minor, patch, suffix) for a release label, else None.

    Collapses the two suffix shapes -- separated (`_patch2`, group 4) and
    separator-less (`g483`, group 5) -- into one value, so callers never have
    to know which branch of `_VERSION_RE` matched.
    """
    match = _VERSION_RE.match(label)
    if not match:
        return None
    major, minor, patch, sep_suffix, bare_suffix = match.groups()
    return major, minor, patch, sep_suffix or bare_suffix or None

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
        fields = dict(part.split("=", 1) for part in line.split(";") if "=" in part)
        label = fields.get("label", "")
        if not (_VERSION_RE.match(label) or _FAMILY_RE.match(label)):
            # Neither a release label nor an integration-series label
            # (`CMSSW_14_0_X`, kept as a release record of its own, with
            # `release_kind: integration-series`). One kind of row is out of
            # scope rather than unreadable, and must not cost the run its
            # completed-scope claim: rows for a NAMED other product
            # (`ECALTBH4_0_2_2`) are outside a CMSSW release catalog's
            # declared scope entirely.
            #
            # The product list is closed on purpose. "Anything that does not
            # start with CMSSW_" would also absorb a wholesale upstream
            # rename: a map where every label became `CMSSW14_0_1` would parse
            # to an EMPTY record set under a healthy completed scope, and with
            # `deletion_semantics: missing_from_completed_scope` the next
            # publish would retract every release in the graph and report
            # success. A product cms-bot adds later is a fact about the
            # upstream we want to be told about, so it degrades the scope
            # until someone looks and adds it here deliberately.
            #
            # Anything else -- an empty or absent label, a malformed `CMSSW_*`
            # label, an unknown product -- counts as a skip and degrades the
            # scope claim.
            if label.startswith(_OTHER_PRODUCTS):
                continue
            skipped += 1
            continue
        rec = by_label.setdefault(
            label,
            {
                "label": label,
                "type": fields.get("type", ""),
                "state": fields.get("state", ""),
                "architecture": [],
                "production_architecture": [],
            },
        )
        arch = fields.get("architecture", "")
        if arch and arch not in rec["architecture"]:
            rec["architecture"].append(arch)
        # `prodarch=1` marks the production architecture of the release.
        # `prodarch=0`, or no `prodarch` key, marks nothing. Older releases
        # can carry the marker on several architectures; all are kept.
        if (
            arch
            and fields.get("prodarch", "").strip() == "1"
            and arch not in rec["production_architecture"]
        ):
            rec["production_architecture"].append(arch)
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
            production_architecture=tuple(sorted(rec["production_architecture"])),
        )
        for rec in ordered
    ], skipped


def _sort_key(label: str) -> tuple[int, int, int, str]:
    parts = _version_parts(label)
    if parts is None:
        series = _SERIES_RE.match(label) if _FAMILY_RE.match(label) else None
        if series:
            # An integration series sorts just before its first release.
            return (int(series.group(1)), int(series.group(2)), -1, label)
        return (0, 0, 0, label)
    major, minor, patch, _suffix = parts
    return (int(major), int(minor), int(patch), label)


def _as_tuple(value: Any) -> tuple[str, ...]:
    """A cache value that is a string or a list, as a tuple of strings."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(sorted(part.strip() for part in value.split(",") if part.strip()))
    return tuple(sorted(str(v) for v in value))


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_version(label: str) -> dict[str, Any]:
    parts = _version_parts(label)
    if parts is None:
        series = _SERIES_RE.match(label) if _FAMILY_RE.match(label) else None
        if series:
            return {"major": int(series.group(1)), "minor": int(series.group(2))}
        return {}
    major, minor, patch, suffix = parts
    parsed: dict[str, Any] = {
        "major": int(major),
        "minor": int(minor),
        "patch": int(patch),
    }
    if suffix:
        parsed["suffix"] = suffix
    return parsed


def release_kind(label: str) -> str:
    """okg's own classification of a release label.

    Derived from the label alone, never from the source `type`: that value is
    kept unchanged in `release_type` (cms-bot writes only `Production` or
    `Development`). `CMSSW_15_0_19_patch2` is a `Production` release of kind
    `patch`; before this split it was stored as `release_type: patch` and the
    source type was lost.
    """
    parts = _version_parts(label)
    if parts is None:
        if _FAMILY_RE.match(label):
            return "integration-series"
        return "unrecognized"
    suffix = parts[3]
    if not suffix:
        return "release"
    lowered = suffix.lower()
    if lowered.startswith("pre"):
        return "pre-release"
    if lowered.startswith("hltpatch"):
        return "hlt-patch"
    if lowered.startswith("patch"):
        return "patch"
    return "variant"


def _node_fact(
    record: CMSSWReleaseRecord,
    revision: dict[str, Any],
) -> NodeFact:
    version = _parse_version(record.label)
    kind = release_kind(record.label)
    architecture = ",".join(record.architecture)
    production_architecture = ",".join(record.production_architecture)
    text_parts = [
        record.label,
        record.xml_type,
        kind,
        record.state,
        architecture,
        (
            f"production architecture: {production_architecture}"
            if production_architecture
            else ""
        ),
        record.release_notes,
    ]
    attrs = {
        "label": record.label,
        "name": record.label,
        "release": record.label,
        # The source's own type, unchanged (Production or Development).
        "release_type": record.xml_type,
        "release_kind": kind,
        "state": record.state,
        "architecture": architecture,
        "release_date": record.release_date,
        "text": " ".join(p for p in text_parts if p).strip(),
        **version,
    }
    if production_architecture:
        attrs["production_architecture"] = production_architecture
    return NodeFact(
        node_id=record.node_id,
        subtype="cmssw_release",
        attrs=attrs,
        source_record_id={"release": record.label},
        source_revision=revision,
    )


def _release_families(
    records: list[CMSSWReleaseRecord],
) -> tuple[tuple[int, int], ...]:
    """(major, minor) of every series with at least one concrete release.

    An integration-series row alone (`CMSSW_20_1_X` before any 20.1 release)
    mints no family; it is a release record of its own.
    """
    families: set[tuple[int, int]] = set()
    for record in records:
        parts = _version_parts(record.label)
        if parts is None:
            continue
        families.add((int(parts[0]), int(parts[1])))
    return tuple(sorted(families))


def family_node_fact(
    major: int,
    minor: int,
    revision: dict[str, Any],
    *,
    state: str = "family",
) -> NodeFact:
    """The derived release-family node for one major/minor series.

    It carries no `release_type`: that attribute holds only a source value,
    and a family is not a source row.
    """
    key = family_key(major, minor)
    series = f"{key}_X"
    return NodeFact(
        node_id=family_node_id(major, minor),
        subtype="cmssw_release",
        attrs={
            "label": key,
            "name": key,
            "release": key,
            "release_kind": "release_family",
            "series": series,
            "state": state,
            "architecture": "",
            "release_date": "",
            "text": f"{key} release family {series}",
            "major": int(major),
            "minor": int(minor),
            "family": True,
        },
        source_record_id={"release_family": key},
        source_revision=revision,
    )


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
    parts = _version_parts(label)
    if parts is None:
        return None
    major, minor, patch, suffix = parts
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


class CMSSWReleaseAdapter(ReaderAdapter):
    """Registry adapter for :class:`CMSSWReleaseSource`.

    The bundle's `cmssw_releases` source names this class. The reader's
    behavior is unchanged; this class only drives it through the
    substrate's adapter contract.

    ``profile`` and ``change_probe_kind`` must be string literals; see
    ``ReaderAdapter``. ``test_bundle_source_adapters.py`` parses this
    file and holds them equal to the reader's own values.
    """

    reader_class = CMSSWReleaseSource
    profile = "reference_catalog"
    change_probe_kind = "content_hash"
