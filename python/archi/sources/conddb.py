"""Cache-backed CondDB global tag source.

Ported from okg-deployments ``cms/cms_sources/conddb.py`` (269 LOC,
``CondDBGlobalTagSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes: cache
helpers come from :mod:`archi.auth.cache` /
:mod:`archi.sources._cache_report` with an explicit ``base`` parameter,
and the hardcoded ``data/cms/...`` paths are parameters (defaults keep
the cms layout minus the ``cms/`` segment). As in the original the
CMSSW records cache is optional (missing -> no ``cmssw_targets``, so
``depends_on`` edges only reach releases named by CondDB itself), and
the source is optional and offline-only; a missing, unreadable, truncated or
empty records cache is ``cache_missing`` with ``completed_scope=False`` from
both ``preflight`` and ``run()`` (empty-cache-fails-loudly; the original
raised from ``run()`` on a missing cache and reported an empty one as
``skipped_optional``, which okg accepts as a complete scope to retract).

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``global_tag`` and
``cmssw_release`` ship in ``archi/schemas/operations.yaml``. ::

    conddb_global_tags:
      module: archi.sources.conddb
      class: CondDBGlobalTagAdapter
      ownership_id: <instance>.conddb-global-tags
      admission_policy:
        producer_id: <instance>.conddb-global-tags
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: conddb_global_tags
        output_signature:
          nodes:
            - {subtype: global_tag}
            - {subtype: cmssw_release}
          edges:
            - {src_subtype: global_tag, edge_type: supersedes, dst_subtype: global_tag}
            - {src_subtype: global_tag, edge_type: depends_on, dst_subtype: cmssw_release}
        output_scope_summary:
          summary: CondDB global tags, supersedes chains, and release dependencies
          nodes: [global_tag, cmssw_release]
          edges:
            - global_tag supersedes global_tag
            - global_tag depends_on cmssw_release
      source_class: reference_catalog
      record_identity_kind: domain_key
      record_identity_fields: [global_tag]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: false
      params:
        # cms defaults; the cms deployment used data/cms/...
        records_path: data/conddb-global-tags/records.json
        cmssw_records_path: data/cmssw-releases/records.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    PreflightResult,
    ConnectorRun,
)

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
    load_json,
)
from archi.sources._cache_report import (
    CacheUnusable,
    cache_preflight_result,
    cache_source_health,
    read_cache_json,
    unusable_cache_preflight,
    unusable_cache_run,
)
from archi.sources._run_modes import (
    COMPLETED_SCOPE_RUN_MODES,
    RELEASE_RUN_MODES,
    input_content_checkpoint,
)
from archi.sources._sdk_adapter import ReaderAdapter
from archi.sources.cmssw import release_kind

_GT_VERSION_RE = re.compile(r"_v(\d+)$")


@dataclass(frozen=True)
class GlobalTagRecord:
    name: str
    release: str = ""
    scenario: str = ""
    description: str = ""
    created_at: str = ""

    @property
    def node_id(self) -> str:
        return f"global_tag:{self.name}"


class CondDBGlobalTagSource:
    """Cache-backed CondDB global tag source.

    This adapter is intentionally offline-only: it reads the local JSON
    cache and emits no placeholder facts when the cache is empty.
    """

    name = "conddb_global_tags"
    profile = "reference_catalog"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/conddb-global-tags/records.json",
        cmssw_records_path: str = "data/cmssw-releases/records.json",
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.cmssw_records_path = cmssw_records_path
        self.base = base
        # The probe must cover every cache run() reads: run() also
        # consults cmssw_records_path (release-target gating), so a
        # cmssw-cache update changes emitted facts and must fire the
        # probe. `cache_paths` stays records-only — it defines this
        # source's own record authority (preflight/content_hash), and
        # the cmssw cache is optional.
        self.change_probe = content_hash_change_probe(
            cache_paths=(self.records_path, self.cmssw_records_path),
            config={
                "records_path": self.records_path,
                "cmssw_records_path": self.cmssw_records_path,
            },
            emit_targets=CondDBGlobalTagSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        # Same verdict run() reaches: a missing, unreadable, truncated
        # or empty cache is cache_missing, never an empty success.
        try:
            records, skipped = self._records_with_skips()
        except CacheUnusable as exc:
            return unusable_cache_preflight(
                exc, source_name=self.name, required=False
            )
        return cache_preflight_result(
            source_name=self.name,
            description="CondDB global tag",
            cache_paths=self.cache_paths,
            records=records,
            required=False,
            base=self.base,
            skipped_count=skipped,
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        try:
            records, skipped = self._records_with_skips()
        except CacheUnusable as exc:
            # An empty cache is a failed fetch, not an empty catalog: a
            # complete scope over zero records would retract every
            # record ingested before (okg honours that order for
            # skipped_optional health too), so report cache_missing.
            return unusable_cache_run(exc, mode=mode)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }
        cmssw_targets = _known_cmssw_targets(
            self.cmssw_records_path, base=self.base
        )
        conddb_release_targets = {
            f"cmssw_release:{record.release}"
            for record in records
            if record.release
        }
        conddb_only_release_targets = conddb_release_targets - cmssw_targets

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _node_fact(record, revision)
            for node_id in sorted(conddb_only_release_targets):
                yield _referenced_release_node_fact(node_id, revision)
            yield from _edge_facts(
                records,
                revision,
                cmssw_targets | conddb_only_release_targets,
            )

        # In okg's reference-catalog modes (release_new, release_unchanged)
        # this source deliberately claims NO complete scope, so okg never
        # deletes records missing from the cache. A cache is a selection
        # exported by hand or by a downloader; a narrower export would
        # otherwise retract an unbounded share of the catalog, and okg has
        # no guard on retraction size yet (mitdbg/akmon#19). okg never skips
        # a read for this reader (it does not declare
        # probe_short_circuit_safe), so every run re-reads the cache in
        # full. The checkpoint only gives the source stored state, which is
        # what lets okg accept an unchanged rerun as a no-op.
        release_mode = mode in RELEASE_RUN_MODES
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(
                mode in COMPLETED_SCOPE_RUN_MODES and not release_mode and not skipped
            ),
            next_checkpoint=(
                input_content_checkpoint(
                    (self.records_path, self.cmssw_records_path),
                    base=self.base,
                )
                if release_mode
                else None
            ),
            run_mode=mode,
            health=cache_source_health(
                description="CondDB global tag",
                cache_paths=self.cache_paths,
                record_count=len(records),
                skipped_count=skipped,
                base=self.base,
            ),
        )

    def _records(self) -> list[GlobalTagRecord]:
        return self._records_with_skips()[0]

    def _records_with_skips(self) -> tuple[list[GlobalTagRecord], int]:
        payload = read_cache_json(
            self.records_path, expect=list, base=self.base
        )
        records: list[GlobalTagRecord] = []
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            name = str(item.get("name") or item.get("tag_name") or "").strip()
            if not name:
                skipped += 1
                continue
            records.append(GlobalTagRecord(
                name=name,
                release=str(item.get("release") or ""),
                scenario=str(item.get("scenario") or ""),
                description=str(item.get("description") or ""),
                created_at=str(
                    item.get("created_at")
                    or item.get("snapshot_time")
                    or ""
                ),
            ))
        return records, skipped


def _node_fact(
    record: GlobalTagRecord,
    revision: dict[str, Any],
) -> NodeFact:
    text = f"{record.name} {record.scenario} {record.release}".strip()
    return NodeFact(
        node_id=record.node_id,
        subtype="global_tag",
        attrs={
            "label": record.name,
            "name": record.name,
            "tag_name": record.name,
            "release": record.release,
            "scenario": record.scenario,
            "description": record.description,
            "created_at": record.created_at,
            "text": text,
        },
        source_record_id={"global_tag": record.name},
        source_revision=revision,
    )


def _edge_facts(
    records: list[GlobalTagRecord],
    revision: dict[str, Any],
    cmssw_targets: set[str],
) -> Iterator[EdgeFact]:
    known = {record.name for record in records}
    for record in records:
        predecessor = _find_predecessor(record.name)
        if predecessor in known:
            yield EdgeFact(
                src=record.node_id,
                dst=f"global_tag:{predecessor}",
                edge_type="supersedes",
                provenance="derived_deterministic",
                source_record_id={"global_tag": record.name},
                source_revision=revision,
            )
        release_target = f"cmssw_release:{record.release}"
        if record.release and release_target in cmssw_targets:
            yield EdgeFact(
                src=record.node_id,
                dst=release_target,
                edge_type="depends_on",
                provenance="derived_deterministic",
                source_record_id={"global_tag": record.name},
                source_revision=revision,
            )


def _find_predecessor(name: str) -> str | None:
    match = _GT_VERSION_RE.search(name)
    if not match:
        return None
    version = int(match.group(1))
    if version <= 1:
        return None
    return f"{name[:match.start()]}_v{version - 1}"


def _referenced_release_node_fact(
    node_id: str,
    revision: dict[str, Any],
) -> NodeFact:
    """A release CondDB names that the CMSSW cache does not list.

    Keyed by the label CondDB wrote, so a series value (`CMSSW_14_0_X`)
    names the integration-series record of that label, the record the
    GlobalTagReleaseLinker rule (`release` = `release`, exact) also reaches.
    It carries no `release_type`: that attribute holds only the CMSSW
    source's own value, and CondDB does not know it.
    """
    _prefix, _sep, key = node_id.partition(":")
    return NodeFact(
        node_id=node_id,
        subtype="cmssw_release",
        attrs={
            "label": key,
            "name": key,
            "release": key,
            "release_kind": release_kind(key),
            "state": "referenced_by_conddb",
            "text": key,
        },
        source_record_id={"release": key},
        source_revision=revision,
    )


def _known_cmssw_targets(
    records_path: str, *, base: str | None = None
) -> set[str]:
    try:
        payload = load_json(records_path, base=base)
    except FileNotFoundError:
        return set()
    if not isinstance(payload, list):
        return set()
    targets: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        # Only the label itself. The CMSSW source no longer mints a node at
        # `cmssw_release:CMSSW_<m>_<n>_X` from concrete releases (families
        # are `cmssw_release_family:*`), so that id is known only when the
        # cache lists the integration-series row.
        targets.add(f"cmssw_release:{label}")
    return targets


class CondDBGlobalTagAdapter(ReaderAdapter):
    """Registry adapter for :class:`CondDBGlobalTagSource`.

    The registry-entry template in this module's docstring names this
    class. The reader's behavior is unchanged; this class only drives it
    through the substrate's adapter contract, which a bare reader cannot
    satisfy (its ``ConnectorRun`` has no ``next_cursor``).

    ``profile`` and ``change_probe_kind`` must be string literals; see
    ``ReaderAdapter``. ``test_bundle_source_adapters.py`` parses this
    file and holds them equal to the reader's own values.
    """

    reader_class = CondDBGlobalTagSource
    profile = "reference_catalog"
    change_probe_kind = "content_hash"
