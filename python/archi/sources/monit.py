"""MONIT (CERN Grafana/OpenSearch) monitoring-overlay sources.

Rewritten from okg-deployments ``cms/cms_sources/monit.py`` (1,931 LOC)
at ``main@f33a9c4`` for the archi v3 package (req.w2.sources,
task.w2.sources-monit). The cms module is the canonical base; the four
classes keep its record shapes, query bodies, parsing, node/edge
emission, and cache-or-live run shape. Changes from the original:

- **De-CMS-ified**: the Grafana base URL, datasource ids, index
  patterns, cache paths, credential env-var name, and time windows are
  constructor parameters. The CMS values remain the documented defaults
  (and are spelled out in the registry templates below) so the cms
  parity corpus does not churn at cutover. Two CMS-specific query
  fragments are parameters too: the SAM/SiteMon Lucene filter
  (``query_filter``, default ``data.vo:cms AND
  data.profile:CMS_CRITICAL``) and the dataset-overlay exclusion
  patterns (``exclude_dataset_patterns``, default the CMS functional
  test datasets). The remaining ``data.*`` field names in the query
  bodies are part of the MONIT document schema of the default indices;
  pointing ``index``/``datasource_id`` at a differently-shaped index
  requires a subclass overriding the query/parse pair, not just params.
- Cache handling goes through :mod:`archi.auth.cache` (explicit
  ``base`` parameter instead of repo-layout assumptions); the
  ``mutable_api`` change probes come from
  :func:`archi.auth.cache.cache_or_forced_live_change_probe` (cache
  present -> content-hash token, cache absent -> fresh token so a
  mutable upstream is never silently skipped).
- Credentials are env-var *references* only (``token_env`` names the
  variable; its value never appears in health payloads or probe
  config).
- ``source_name`` is a constructor parameter (multiple instances of one
  class can coexist in a registry), defaulting to the cms names.

Failure semantics (the docs.py/twiki.py completeness discipline; the
2026-08 adversarial review hardened the cms original's version of it —
see ``pact/changes/circleback-fixes/notes-live.md``):

- A missing configured credential (no cache file and ``token_env``
  unset) is ``missing_credential`` health with ``completed_scope=False``
  — never an empty-but-complete run that would retract every previously
  ingested record under ``missing_from_completed_scope``.
- Any cache-read or live-query failure is ``endpoint_failed`` health
  with ``completed_scope=False``.
- ``ConnectorHealth.status`` values come only from okg's closed
  ``PREFLIGHT_STATUSES`` vocabulary (``ok`` / ``skipped_optional`` /
  ``missing_credential`` / ``auth_failed`` / ``tls_failed`` /
  ``endpoint_failed`` / ``cache_missing`` / ``not_applicable`` / ...);
  ``ConnectorHealth.__post_init__`` rejects anything else.
- A *partial* OpenSearch response — ``timed_out: true``, failed shards,
  or a bucket-limited terms aggregation that dropped buckets
  (``sum_other_doc_count`` / ``doc_count_error_upper_bound`` > 0) —
  still emits the records it carries but reports ``endpoint_failed``
  (the closed vocabulary has no ``degraded``) with
  ``completed_scope=False``: a truncated snapshot must never become the
  retraction baseline. The dataset overlay's streaming path instead
  raises on any partial page so the runner fails the run rather than
  committing a partial scope.
- A *successful* read that returns zero buckets reports
  ``skipped_optional`` with ``completed_scope=False``: under the
  ``ignore_unavailable: true`` wildcard-index queries these sources
  issue, a renamed index or stalled ingestion pipeline is
  indistinguishable from a genuinely empty window, so an empty result
  never claims a complete (empty) scope that would retract every
  previously ingested record.
- The dataset overlay's live page cache is keyed by a fingerprint that
  includes the resolved snapshot date, and an entry only satisfies a
  run when its ``cached_at`` is on the same UTC day and younger than
  ``page_cache_ttl_hours`` — a cached page from a previous day can
  never satisfy a new day's run. ``cache_live_pages`` stays ``true`` by
  default on purpose: with date-scoped keys the cache can only replay
  the current snapshot day's pages, which is its role (crash/resume
  within a day without re-querying MONIT).

Registry-entry templates — same three prerequisites as the
archi/sources/jira.py and archi/sources/docs.py templates, plus one
specific to this family:

1. Ontology: ``monitoring_snapshot`` and ``transfer_job`` ship in
   ``archi/schemas/sources.yaml`` (added with this port); ``dataset``
   comes from the substrate ``dataset`` module — compose it in
   ``deployment.yaml`` (``modules: [dataset, ...]``). Copy the packaged
   schema slice into the deployment: ``archi/schemas/sources.yaml`` ->
   ``<deployment>/schemas/`` and ``archi/schemas/bridges/sources.yaml``
   -> ``<deployment>/schemas/bridges/`` (narrowings outside
   ``schemas/bridges/`` are silently ignored and fail only at ingest).
2. **Edge endpoints**: every edge these sources emit starts at a
   ``site`` (or, for dataset replicas, a ``storage_endpoint``) node
   owned by the catalogs port (``archi/schemas/operations.yaml`` +
   ``archi/schemas/bridges/operations.yaml``). The bridge
   narrowings for these edges are declared with
   ``optional_when_subtypes_missing: 'true'`` so composing a catalog
   without ``site``/``storage_endpoint`` stays green — but then the
   emitted edges are rejected at ingest as narrowing violations.
   Deploy this family together with a schema slice that provides
   ``site`` (and ``storage_endpoint`` for monit_rucio_datasets).
3. ``output_scope_summary`` must accompany ``output_signature``, and
   every entry needs the standard ``sync:`` block. ::

    monit_sam:
      module: archi.sources.monit
      class: MONITSAMSource
      ownership_id: <instance>.monit-sam
      admission_policy:
        producer_id: <instance>.monit-sam
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: monit_sam
        output_signature:
          nodes:
            - {subtype: monitoring_snapshot}
          edges:
            - {src_subtype: site, edge_type: hosts, dst_subtype: monitoring_snapshot}
        output_scope_summary:
          summary: daily SAM/SiteMon site-availability snapshots from MONIT
          nodes: [monitoring_snapshot]
          edges:
            - site hosts monitoring_snapshot
      source_class: mutable_api
      record_identity_kind: remote_id
      record_identity_fields: [snapshot_id]
      source_revision_kind: updated_at
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [MONIT_GRAFANA_TOKEN]
      required_for_baseline: false
      params:
        records_path: data/monit-sam/records.json  # cms: data/cms/monit-sam/records.json
        token_env: MONIT_GRAFANA_TOKEN
        from_time: now-24h
        to_time: now
        max_sites: 500
        # CMS defaults, shown for overriding on another instance:
        # grafana_base_url: https://monit-grafana.cern.ch
        # datasource_id: 9841
        # index: monit_prod_sitemon_agg_site-*
        # query_filter: "data.vo:cms AND data.profile:CMS_CRITICAL"
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete

    monit_condor:
      module: archi.sources.monit
      class: MONITCondorSource
      ownership_id: <instance>.monit-condor
      admission_policy:
        producer_id: <instance>.monit-condor
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: monit_condor
        output_signature:
          nodes:
            - {subtype: monitoring_snapshot}
          edges:
            - {src_subtype: site, edge_type: hosts, dst_subtype: monitoring_snapshot}
        output_scope_summary:
          summary: site-level HTCondor compute summaries from MONIT
          nodes: [monitoring_snapshot]
          edges:
            - site hosts monitoring_snapshot
      source_class: mutable_api
      record_identity_kind: remote_id
      record_identity_fields: [snapshot_id]
      source_revision_kind: updated_at
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [MONIT_GRAFANA_TOKEN]
      required_for_baseline: false
      params:
        records_path: data/monit-condor/records.json  # cms: data/cms/monit-condor/records.json
        token_env: MONIT_GRAFANA_TOKEN
        from_time: now-7d
        to_time: now
        max_sites: 500
        # CMS defaults, shown for overriding on another instance:
        # grafana_base_url: https://monit-grafana.cern.ch
        # datasource_id: 9582
        # index: monit_prod_condor_agg_metric-*
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete

    monit_rucio_transfer:
      module: archi.sources.monit
      class: MONITRucioTransferSource
      ownership_id: <instance>.monit-rucio-transfer
      admission_policy:
        producer_id: <instance>.monit-rucio-transfer
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: monit_rucio_transfer
        output_signature:
          nodes:
            - {subtype: transfer_job}
          edges:
            - {src_subtype: site, edge_type: hosts, dst_subtype: transfer_job}
            - {src_subtype: site, edge_type: transfers_to, dst_subtype: site}
        output_scope_summary:
          summary: per-site-pair Rucio transfer aggregates from MONIT enriched events
          nodes: [transfer_job]
          edges:
            - site hosts transfer_job
            - site transfers_to site
      source_class: mutable_api
      record_identity_kind: remote_id
      record_identity_fields: [transfer_id]
      source_revision_kind: updated_at
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [MONIT_GRAFANA_TOKEN]
      required_for_baseline: false
      params:
        records_path: data/monit-rucio-transfer/records.json  # cms: data/cms/monit-rucio-transfer/records.json
        token_env: MONIT_GRAFANA_TOKEN
        from_time: now-7d
        to_time: now
        max_src_sites: 500
        max_dst_sites: 500
        # CMS defaults, shown for overriding on another instance:
        # grafana_base_url: https://monit-grafana.cern.ch
        # datasource_id: 9732
        # index: monit_prod_cms_rucio_enr_events-*
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete

    monit_rucio_datasets:
      module: archi.sources.monit
      class: MONITRucioDatasetSource
      ownership_id: <instance>.monit-rucio-datasets
      admission_policy:
        producer_id: <instance>.monit-rucio-datasets
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: monit_rucio_datasets
        output_signature:
          nodes:
            - {subtype: dataset}
          edges:
            - {src_subtype: site, edge_type: hosts, dst_subtype: dataset}
            - {src_subtype: storage_endpoint, edge_type: hosts, dst_subtype: dataset}
        output_scope_summary:
          summary: dataset replica overlays from MONIT Rucio daily stats
          nodes: [dataset]
          edges:
            - site hosts dataset
            - storage_endpoint hosts dataset
      source_class: mutable_api
      record_identity_kind: remote_id
      record_identity_fields: [dataset]
      source_revision_kind: updated_at
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [MONIT_GRAFANA_TOKEN]
      required_for_baseline: false
      params:
        records_path: data/monit-rucio-datasets/records.json  # cms: data/cms/monit-rucio-datasets/records.json
        token_env: MONIT_GRAFANA_TOKEN
        from_time: now-24h
        to_time: now
        page_size: 250
        max_replica_rses: 250
        page_cache_dir: data/monit-rucio-datasets/pages  # cms: data/cms/monit-rucio-datasets/pages
        # cache_live_pages stays true on purpose (2026-08 review): the
        # page-cache fingerprint includes the resolved snapshot date and
        # entries expire after page_cache_ttl_hours (same UTC day only),
        # so the cache can only replay the current day's pages — its
        # crash/resume role — never yesterday's datasets stamped with
        # today's date.
        cache_live_pages: true
        page_cache_ttl_hours: 24.0
        # CMS defaults, shown for overriding on another instance:
        # grafana_base_url: https://monit-grafana.cern.ch
        # datasource_id: 10151
        # index: monit_prod_cms_rucio_raw_daily_stats-*
        # exclude_dataset_patterns: [GenericTTbar, SAM-CMSSW, HC-CMSSW, /store/test/]
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import requests

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ProgressMarker,
    ConnectorHealth,
    PreflightResult,
    ConnectorRun,
)

from archi.auth.cache import (
    cache_or_forced_live_change_probe,
    load_json,
    resolve_repo_path,
)


DEFAULT_GRAFANA_BASE_URL = "https://monit-grafana.cern.ch"
DEFAULT_TOKEN_ENV = "MONIT_GRAFANA_TOKEN"
# CMS MONIT datasource ids / index patterns (documented defaults).
DS_SITEMON_SITE = 9841
DS_CONDOR_AGG = 9582
DS_RUCIO_ENRICHED = 9732
DS_RUCIO_DAILY_STATS = 10151
SITEMON_INDEX = "monit_prod_sitemon_agg_site-*"
CONDOR_AGG_INDEX = "monit_prod_condor_agg_metric-*"
RUCIO_ENRICHED_INDEX = "monit_prod_cms_rucio_enr_events-*"
RUCIO_DAILY_INDEX = "monit_prod_cms_rucio_raw_daily_stats-*"
DEFAULT_SAM_QUERY_FILTER = "data.vo:cms AND data.profile:CMS_CRITICAL"
DEFAULT_EXCLUDED_DATASET_PATTERNS = (
    "GenericTTbar",
    "SAM-CMSSW",
    "HC-CMSSW",
    "/store/test/",
)
TIME_FORMAT = "strict_date_optional_time||epoch_millis"
RSE_SUFFIXES = ("_Tape", "_Disk", "_Buffer", "_Export", "_Test", "_MSS")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiteAvailabilityRecord:
    site: str
    snapshot_date: str
    availability_pct: float
    status: str
    total_tests: int
    ok_count: int = 0
    warning_count: int = 0
    critical_count: int = 0
    unknown_count: int = 0

    @property
    def node_id(self) -> str:
        return f"monitoring_snapshot:sam:{self.site}:{self.snapshot_date}"


@dataclass(frozen=True)
class CondorComputeRecord:
    site: str
    snapshot_date: str
    total_jobs: int
    core_hours: float = 0.0
    cpu_time_hours: float = 0.0
    avg_queue_hours: float | None = None
    jobs_running: int = 0
    jobs_idle: int = 0
    jobs_held: int = 0
    status_breakdown: dict[str, int] | None = None
    job_type_breakdown: dict[str, int] | None = None
    error_breakdown: dict[str, int] | None = None

    @property
    def node_id(self) -> str:
        return f"monitoring_snapshot:condor:{self.site}:{self.snapshot_date}"

    @property
    def cpu_efficiency(self) -> float | None:
        if self.core_hours <= 0:
            return None
        return self.cpu_time_hours / self.core_hours


@dataclass(frozen=True)
class RucioTransferRecord:
    src_site: str
    dst_site: str
    snapshot_date: str
    done_count: int = 0
    failed_count: int = 0
    bytes_total: int = 0
    avg_duration_s: float = 0.0
    top_activity: str = ""
    failure_reasons: dict[str, int] | None = None

    @property
    def node_id(self) -> str:
        src = _safe_node_part(self.src_site)
        dst = _safe_node_part(self.dst_site)
        return f"transfer_job:rucio:{src}:{dst}:{self.snapshot_date}"

    @property
    def transfer_id(self) -> str:
        return self.node_id

    @property
    def total_count(self) -> int:
        return self.done_count + self.failed_count

    @property
    def success_rate(self) -> float:
        return self.done_count / self.total_count if self.total_count else 0.0


@dataclass(frozen=True)
class RucioDatasetRecord:
    dataset: str
    snapshot_date: str
    data_tier: str = ""
    acquisition_era: str = ""
    event_count: int = 0
    file_count: int = 0
    size_bytes: int = 0
    physics_group: str = ""
    replica_sites: tuple[str, ...] = ()
    replica_rses: tuple[str, ...] = ()

    @property
    def node_id(self) -> str:
        return f"dataset:{self.dataset}"


@dataclass(frozen=True)
class _ResponseQuality:
    """Completeness markers of one MONIT OpenSearch response.

    An HTTP-200 ``_msearch`` response can still be partial:
    ``timed_out: true``, failed shards, or bucket-limited terms
    aggregations that dropped buckets (``sum_other_doc_count`` /
    ``doc_count_error_upper_bound`` > 0). Any of those must keep the
    run from claiming a completed scope — under
    ``missing_from_completed_scope`` a partial snapshot claimed
    complete would retract every record the response happened to drop.
    """

    timed_out: bool = False
    shards_failed: int = 0
    shards_total: int = 0
    truncated_aggs: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not (
            self.timed_out or self.shards_failed or self.truncated_aggs
        )

    def describe(self) -> str:
        parts = []
        if self.timed_out:
            parts.append("timed_out=true")
        if self.shards_failed:
            total = self.shards_total or "?"
            parts.append(f"shards failed {self.shards_failed}/{total}")
        if self.truncated_aggs:
            parts.append(
                "truncated terms aggregations: "
                + ", ".join(self.truncated_aggs)
            )
        return "; ".join(parts) or "complete"


def _response_quality(payload: dict[str, Any]) -> _ResponseQuality:
    response = _first_response(payload)
    shards = response.get("_shards") or {}
    truncated: list[str] = []
    _collect_agg_truncations(
        response.get("aggregations") or {}, "", truncated
    )
    return _ResponseQuality(
        timed_out=bool(response.get("timed_out")),
        shards_failed=int(shards.get("failed") or 0),
        shards_total=int(shards.get("total") or 0),
        truncated_aggs=tuple(truncated),
    )


def _collect_agg_truncations(node: Any, path: str, out: list[str]) -> None:
    """Record bucket-limited aggregations that silently dropped buckets.

    Walks the aggregation tree (nested aggregations live inside each
    bucket) and flags any ``buckets`` aggregation whose
    ``sum_other_doc_count`` or ``doc_count_error_upper_bound`` is
    positive — the fixed ``size`` cap truncated it.
    """
    if not isinstance(node, dict):
        return
    for name, value in node.items():
        if not isinstance(value, dict):
            continue
        child = f"{path}.{name}" if path else name
        buckets = value.get("buckets")
        if not isinstance(buckets, list):
            continue
        other = int(value.get("sum_other_doc_count") or 0)
        bound = int(value.get("doc_count_error_upper_bound") or 0)
        if other > 0 or bound > 0:
            out.append(
                f"{child} (sum_other_doc_count={other}, "
                f"doc_count_error_upper_bound={bound})"
            )
        for bucket in buckets:
            _collect_agg_truncations(bucket, child, out)


def _merge_quality(
    left: _ResponseQuality,
    right: _ResponseQuality,
) -> _ResponseQuality:
    return _ResponseQuality(
        timed_out=left.timed_out or right.timed_out,
        shards_failed=left.shards_failed + right.shards_failed,
        shards_total=max(left.shards_total, right.shards_total),
        truncated_aggs=tuple(dict.fromkeys(
            (*left.truncated_aggs, *right.truncated_aggs)
        )),
    )


class MONITSAMSource:
    """Emit site-level SAM/SiteMon monitoring snapshots."""

    name = "monit_sam"
    profile = "mutable_api"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        source_name: str = "monit_sam",
        records_path: str = "data/monit-sam/records.json",
        token_env: str = DEFAULT_TOKEN_ENV,
        grafana_base_url: str = DEFAULT_GRAFANA_BASE_URL,
        datasource_id: int = DS_SITEMON_SITE,
        index: str = SITEMON_INDEX,
        query_filter: str = DEFAULT_SAM_QUERY_FILTER,
        from_time: str = "now-24h",
        to_time: str = "now",
        max_sites: int = 500,
        timeout: float = 60.0,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.records_path = records_path
        self.token_env = token_env
        self.grafana_base_url = grafana_base_url.rstrip("/")
        self.datasource_id = datasource_id
        self.index = index
        self.query_filter = query_filter
        self.from_time = from_time
        self.to_time = to_time
        self.max_sites = max_sites
        self.timeout = timeout
        self.base = base
        self.change_probe = _monit_change_probe(self)

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        return _monit_preflight(
            source_name=self.name,
            records_path=self.records_path,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            cache_loader=self._records_from_cache,
            cache_reason="local MONIT SAM cache present",
            base=self.base,
        )

    def run(
        self,
        run_id: str,
        *,
        mode: str = "cursor",
        cursor: Any = None,
        **_: Any,
    ) -> ConnectorRun:
        try:
            records, run_mode, revision, quality = self._load_records(run_id)
        except MissingMONITCredential as exc:
            return _missing_credential_run(mode, self.token_env, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _endpoint_failed_run(
                mode, self.token_env, exc, endpoint=self.grafana_base_url
            )

        def _facts() -> Iterator[NodeFact | EdgeFact]:
            yield from _sam_facts_for_records(records, revision)

        return _source_run(
            facts=_facts(),
            mode=mode,
            run_mode=run_mode,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            revision=revision,
            records=records,
            ok_reason="MONIT SAM/SiteMon records loaded",
            empty_reason="MONIT SAM/SiteMon query returned no site buckets",
            quality=quality,
        )

    def _load_records(
        self,
        run_id: str,
    ) -> tuple[
        list[SiteAvailabilityRecord], str, dict[str, Any], _ResponseQuality
    ]:
        records, run_mode, hash_value, quality = _load_cache_or_live(
            records_path=self.records_path,
            token_env=self.token_env,
            cache_loader=self._records_from_cache,
            live_loader=self._records_from_live,
            base=self.base,
        )
        return records, run_mode, _revision(
            run_id=run_id,
            source=self.name,
            run_mode=run_mode,
            from_time=self.from_time,
            to_time=self.to_time,
            hash_value=hash_value,
            n_records=len(records),
        ), quality

    def _records_from_cache(
        self,
    ) -> tuple[list[SiteAvailabilityRecord], _ResponseQuality]:
        payload = load_json(self.records_path, base=self.base)
        if isinstance(payload, dict) and "responses" in payload:
            return (
                _parse_sitemon_response(payload, snapshot_date=_today()),
                _response_quality(payload),
            )
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a list of records or "
                "MONIT _msearch response"
            )
        return (
            [_sam_record_from_mapping(item) for item in payload if item],
            _ResponseQuality(),
        )

    def _records_from_live(
        self, token: str
    ) -> tuple[list[SiteAvailabilityRecord], _ResponseQuality]:
        response = _monit_msearch(
            grafana_base_url=self.grafana_base_url,
            datasource_id=self.datasource_id,
            index=self.index,
            token=token,
            query=_sitemon_query(
                query_filter=self.query_filter,
                from_time=self.from_time,
                to_time=self.to_time,
                max_sites=self.max_sites,
            ),
            timeout=self.timeout,
        )
        return (
            _parse_sitemon_response(response, snapshot_date=_today()),
            _response_quality(response),
        )


class MONITCondorSource:
    """Emit site-level HTCondor compute monitoring snapshots."""

    name = "monit_condor"
    profile = "mutable_api"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        source_name: str = "monit_condor",
        records_path: str = "data/monit-condor/records.json",
        token_env: str = DEFAULT_TOKEN_ENV,
        grafana_base_url: str = DEFAULT_GRAFANA_BASE_URL,
        datasource_id: int = DS_CONDOR_AGG,
        index: str = CONDOR_AGG_INDEX,
        from_time: str = "now-7d",
        to_time: str = "now",
        max_sites: int = 500,
        timeout: float = 60.0,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.records_path = records_path
        self.token_env = token_env
        self.grafana_base_url = grafana_base_url.rstrip("/")
        self.datasource_id = datasource_id
        self.index = index
        self.from_time = from_time
        self.to_time = to_time
        self.max_sites = max_sites
        self.timeout = timeout
        self.base = base
        self.change_probe = _monit_change_probe(self)

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        return _monit_preflight(
            source_name=self.name,
            records_path=self.records_path,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            cache_loader=self._records_from_cache,
            cache_reason="local MONIT Condor cache present",
            base=self.base,
        )

    def run(
        self,
        run_id: str,
        *,
        mode: str = "cursor",
        cursor: Any = None,
        **_: Any,
    ) -> ConnectorRun:
        try:
            records, run_mode, revision, quality = self._load_records(run_id)
        except MissingMONITCredential as exc:
            return _missing_credential_run(mode, self.token_env, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _endpoint_failed_run(
                mode, self.token_env, exc, endpoint=self.grafana_base_url
            )

        def _facts() -> Iterator[NodeFact | EdgeFact]:
            yield from _condor_facts_for_records(records, revision)

        return _source_run(
            facts=_facts(),
            mode=mode,
            run_mode=run_mode,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            revision=revision,
            records=records,
            ok_reason="MONIT Condor records loaded",
            empty_reason="MONIT Condor query returned no site buckets",
            quality=quality,
        )

    def _load_records(
        self,
        run_id: str,
    ) -> tuple[
        list[CondorComputeRecord], str, dict[str, Any], _ResponseQuality
    ]:
        records, run_mode, hash_value, quality = _load_cache_or_live(
            records_path=self.records_path,
            token_env=self.token_env,
            cache_loader=self._records_from_cache,
            live_loader=self._records_from_live,
            base=self.base,
        )
        return records, run_mode, _revision(
            run_id=run_id,
            source=self.name,
            run_mode=run_mode,
            from_time=self.from_time,
            to_time=self.to_time,
            hash_value=hash_value,
            n_records=len(records),
        ), quality

    def _records_from_cache(
        self,
    ) -> tuple[list[CondorComputeRecord], _ResponseQuality]:
        payload = load_json(self.records_path, base=self.base)
        if isinstance(payload, dict) and "responses" in payload:
            return (
                _parse_condor_response(payload, snapshot_date=_today()),
                _response_quality(payload),
            )
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a list of records or "
                "MONIT _msearch response"
            )
        return (
            [_condor_record_from_mapping(item) for item in payload if item],
            _ResponseQuality(),
        )

    def _records_from_live(
        self, token: str
    ) -> tuple[list[CondorComputeRecord], _ResponseQuality]:
        response = _monit_msearch(
            grafana_base_url=self.grafana_base_url,
            datasource_id=self.datasource_id,
            index=self.index,
            token=token,
            query=_condor_query(
                from_time=self.from_time,
                to_time=self.to_time,
                max_sites=self.max_sites,
            ),
            timeout=self.timeout,
        )
        return (
            _parse_condor_response(response, snapshot_date=_today()),
            _response_quality(response),
        )


class MONITRucioTransferSource:
    """Emit Rucio transfer aggregate nodes from MONIT enriched events."""

    name = "monit_rucio_transfer"
    profile = "mutable_api"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        source_name: str = "monit_rucio_transfer",
        records_path: str = "data/monit-rucio-transfer/records.json",
        token_env: str = DEFAULT_TOKEN_ENV,
        grafana_base_url: str = DEFAULT_GRAFANA_BASE_URL,
        datasource_id: int = DS_RUCIO_ENRICHED,
        index: str = RUCIO_ENRICHED_INDEX,
        from_time: str = "now-7d",
        to_time: str = "now",
        max_src_sites: int = 500,
        max_dst_sites: int = 500,
        timeout: float = 60.0,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.records_path = records_path
        self.token_env = token_env
        self.grafana_base_url = grafana_base_url.rstrip("/")
        self.datasource_id = datasource_id
        self.index = index
        self.from_time = from_time
        self.to_time = to_time
        self.max_src_sites = max_src_sites
        self.max_dst_sites = max_dst_sites
        self.timeout = timeout
        self.base = base
        self.change_probe = _monit_change_probe(self)

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        return _monit_preflight(
            source_name=self.name,
            records_path=self.records_path,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            cache_loader=self._records_from_cache,
            cache_reason="local MONIT Rucio transfer cache present",
            base=self.base,
        )

    def run(
        self,
        run_id: str,
        *,
        mode: str = "cursor",
        cursor: Any = None,
        **_: Any,
    ) -> ConnectorRun:
        try:
            records, run_mode, revision, quality = self._load_records(run_id)
        except MissingMONITCredential as exc:
            return _missing_credential_run(mode, self.token_env, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _endpoint_failed_run(
                mode, self.token_env, exc, endpoint=self.grafana_base_url
            )

        def _facts() -> Iterator[NodeFact | EdgeFact]:
            yield from _rucio_transfer_facts_for_records(records, revision)

        return _source_run(
            facts=_facts(),
            mode=mode,
            run_mode=run_mode,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            revision=revision,
            records=records,
            ok_reason="MONIT Rucio transfer records loaded",
            empty_reason="MONIT Rucio transfer query returned no transfer buckets",
            quality=quality,
        )

    def _load_records(
        self,
        run_id: str,
    ) -> tuple[
        list[RucioTransferRecord], str, dict[str, Any], _ResponseQuality
    ]:
        records, run_mode, hash_value, quality = _load_cache_or_live(
            records_path=self.records_path,
            token_env=self.token_env,
            cache_loader=self._records_from_cache,
            live_loader=self._records_from_live,
            base=self.base,
        )
        return records, run_mode, _revision(
            run_id=run_id,
            source=self.name,
            run_mode=run_mode,
            from_time=self.from_time,
            to_time=self.to_time,
            hash_value=hash_value,
            n_records=len(records),
        ), quality

    def _records_from_cache(
        self,
    ) -> tuple[list[RucioTransferRecord], _ResponseQuality]:
        payload = load_json(self.records_path, base=self.base)
        if isinstance(payload, dict) and "responses" in payload:
            return (
                _parse_rucio_transfer_response(
                    payload, snapshot_date=_today()
                ),
                _response_quality(payload),
            )
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a list of records or "
                "MONIT _msearch response"
            )
        return (
            _dedupe_rucio_transfer_records([
                _rucio_transfer_record_from_mapping(item)
                for item in payload
                if item
            ]),
            _ResponseQuality(),
        )

    def _records_from_live(
        self, token: str
    ) -> tuple[list[RucioTransferRecord], _ResponseQuality]:
        response = _monit_msearch(
            grafana_base_url=self.grafana_base_url,
            datasource_id=self.datasource_id,
            index=self.index,
            token=token,
            query=_rucio_transfer_query(
                from_time=self.from_time,
                to_time=self.to_time,
                max_src_sites=self.max_src_sites,
                max_dst_sites=self.max_dst_sites,
            ),
            timeout=self.timeout,
        )
        return (
            _parse_rucio_transfer_response(response, snapshot_date=_today()),
            _response_quality(response),
        )


class MONITRucioDatasetSource:
    """Emit dataset replica overlay records from MONIT Rucio daily stats.

    Live runs use OpenSearch composite aggregation pagination rather than a
    fixed top-N terms aggregation cap.
    """

    name = "monit_rucio_datasets"
    profile = "mutable_api"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        source_name: str = "monit_rucio_datasets",
        records_path: str = "data/monit-rucio-datasets/records.json",
        token_env: str = DEFAULT_TOKEN_ENV,
        grafana_base_url: str = DEFAULT_GRAFANA_BASE_URL,
        datasource_id: int = DS_RUCIO_DAILY_STATS,
        index: str = RUCIO_DAILY_INDEX,
        from_time: str = "now-24h",
        to_time: str = "now",
        page_size: int = 250,
        max_replica_rses: int = 250,
        page_cache_dir: str = "data/monit-rucio-datasets/pages",
        cache_live_pages: bool = True,
        page_cache_ttl_hours: float = 24.0,
        exclude_dataset_patterns: tuple[str, ...] = (
            DEFAULT_EXCLUDED_DATASET_PATTERNS
        ),
        timeout: float = 90.0,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.records_path = records_path
        self.token_env = token_env
        self.grafana_base_url = grafana_base_url.rstrip("/")
        self.datasource_id = datasource_id
        self.index = index
        self.from_time = from_time
        self.to_time = to_time
        self.page_size = page_size
        self.max_replica_rses = max_replica_rses
        self.page_cache_dir = page_cache_dir
        self.cache_live_pages = cache_live_pages
        self.page_cache_ttl_hours = page_cache_ttl_hours
        self.exclude_dataset_patterns = tuple(exclude_dataset_patterns)
        self.timeout = timeout
        self.base = base
        self.change_probe = _monit_change_probe(self)

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        return _monit_preflight(
            source_name=self.name,
            records_path=self.records_path,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            cache_loader=self._records_from_cache,
            cache_reason="local MONIT Rucio dataset cache present",
            base=self.base,
        )

    def run(
        self,
        run_id: str,
        *,
        mode: str = "cursor",
        cursor: Any = None,
        since_progress: dict[str, str] | None = None,
        **_: Any,
    ) -> ConnectorRun:
        path = resolve_repo_path(self.records_path, base=self.base)
        if not path.is_file():
            token = os.environ.get(self.token_env)
            if not token:
                return _missing_credential_run(
                    mode,
                    self.token_env,
                    f"{self.token_env} is not set and {path} is missing",
                )
            revision = _revision(
                run_id=run_id,
                source=self.name,
                run_mode="live",
                from_time=self.from_time,
                to_time=self.to_time,
                hash_value=self._query_fingerprint(),
                n_records=-1,
            )

            # Fetch the first page eagerly: a degraded first page must
            # fail the run before any scope claim exists, and a
            # zero-bucket wildcard result (ignore_unavailable masks a
            # renamed index or a stalled pipeline) must never claim a
            # complete empty scope — that would retract every
            # previously ingested dataset.
            try:
                first_page = self._load_live_page(
                    token=token, page_number=0, after_key=None
                )
            except Exception as exc:  # noqa: BLE001
                return _endpoint_failed_run(
                    mode, self.token_env, exc, endpoint=self.grafana_base_url
                )
            first_datasets = (
                _first_response(first_page)
                .get("aggregations", {})
                .get("datasets", {})
            )
            if not first_datasets.get("buckets"):
                return ConnectorRun(
                    facts=[],
                    completed_scope=False,
                    run_mode=mode,
                    health=ConnectorHealth(
                        status="skipped_optional",
                        mode="live",
                        credential_refs=(self.token_env,),
                        record_count=0,
                        endpoint=self.grafana_base_url,
                        reason=(
                            "MONIT Rucio dataset query returned no dataset "
                            "buckets; with ignore_unavailable a zero-bucket "
                            "wildcard result is indistinguishable from a "
                            "renamed index or stalled pipeline, so no "
                            "complete scope is claimed"
                        ),
                        checked_at=_checked_at(),
                    ),
                )

            def _live() -> Iterator[NodeFact | EdgeFact | ProgressMarker]:
                yield from self._live_facts(
                    token=token,
                    revision=revision,
                    since_progress=since_progress or {},
                    first_page=first_page,
                )

            return ConnectorRun(
                facts=_live(),
                completed_scope=(mode in {"scope_complete", "reconcile"}),
                run_mode=mode,
                health=ConnectorHealth(
                    status="ok",
                    mode="live",
                    credential_refs=(self.token_env,),
                    content_hash=revision["content_hash"],
                    endpoint=self.grafana_base_url,
                    reason=(
                        "MONIT Rucio dataset records stream from "
                        "composite pagination"
                    ),
                    checked_at=_checked_at(),
                ),
            )

        try:
            records, run_mode, revision, quality = self._load_records(run_id)
        except MissingMONITCredential as exc:
            return _missing_credential_run(mode, self.token_env, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _endpoint_failed_run(
                mode, self.token_env, exc, endpoint=self.grafana_base_url
            )

        def _facts() -> Iterator[NodeFact | EdgeFact]:
            yield from _rucio_dataset_facts_for_records(records, revision)

        return _source_run(
            facts=_facts(),
            mode=mode,
            run_mode=run_mode,
            token_env=self.token_env,
            endpoint=self.grafana_base_url,
            revision=revision,
            records=records,
            ok_reason="MONIT Rucio dataset records loaded",
            empty_reason="MONIT Rucio dataset query returned no dataset buckets",
            quality=quality,
        )

    def _load_records(
        self,
        run_id: str,
    ) -> tuple[
        list[RucioDatasetRecord], str, dict[str, Any], _ResponseQuality
    ]:
        records, run_mode, hash_value, quality = _load_cache_or_live(
            records_path=self.records_path,
            token_env=self.token_env,
            cache_loader=self._records_from_cache,
            live_loader=self._records_from_live,
            base=self.base,
        )
        return records, run_mode, _revision(
            run_id=run_id,
            source=self.name,
            run_mode=run_mode,
            from_time=self.from_time,
            to_time=self.to_time,
            hash_value=hash_value,
            n_records=len(records),
        ), quality

    def _records_from_cache(
        self,
    ) -> tuple[list[RucioDatasetRecord], _ResponseQuality]:
        payload = load_json(self.records_path, base=self.base)
        if isinstance(payload, dict) and "responses" in payload:
            return self._parse_response(payload), _response_quality(payload)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a list of records or "
                "MONIT _msearch response"
            )
        return (
            [
                _rucio_dataset_record_from_mapping(item)
                for item in payload
                if item
            ],
            _ResponseQuality(),
        )

    def _records_from_live(
        self, token: str
    ) -> tuple[list[RucioDatasetRecord], _ResponseQuality]:
        records: list[RucioDatasetRecord] = []
        quality = _ResponseQuality()
        after_key: dict[str, Any] | None = None
        while True:
            response = _monit_msearch(
                grafana_base_url=self.grafana_base_url,
                datasource_id=self.datasource_id,
                index=self.index,
                token=token,
                query=_rucio_dataset_query(
                    from_time=self.from_time,
                    to_time=self.to_time,
                    page_size=self.page_size,
                    max_replica_rses=self.max_replica_rses,
                    after_key=after_key,
                ),
                timeout=self.timeout,
            )
            records.extend(self._parse_response(response))
            quality = _merge_quality(quality, _response_quality(response))
            response0 = _first_response(response)
            after_key = (
                response0.get("aggregations", {})
                .get("datasets", {})
                .get("after_key")
            )
            if not after_key:
                break
        return records, quality

    def _parse_response(
        self, payload: dict[str, Any]
    ) -> list[RucioDatasetRecord]:
        return _parse_rucio_dataset_response(
            payload,
            snapshot_date=_today(),
            exclude_patterns=self.exclude_dataset_patterns,
        )

    def _query_fingerprint(self) -> str:
        # The snapshot date is part of the key: the query body's literal
        # "now-24h"/"now" resolve to a different window every day, so a
        # date-free fingerprint would let one day's cached pages satisfy
        # every later day's run forever (2026-08 review blocker).
        return _query_fingerprint(
            source=self.name,
            from_time=self.from_time,
            to_time=self.to_time,
            page_size=self.page_size,
            max_replica_rses=self.max_replica_rses,
            snapshot_date=_today(),
        )

    def _live_facts(
        self,
        *,
        token: str,
        revision: dict[str, Any],
        since_progress: dict[str, str],
        first_page: dict[str, Any] | None = None,
    ) -> Iterator[NodeFact | EdgeFact | ProgressMarker]:
        after_key: dict[str, Any] | None = None
        page_number = 0
        n_records = 0
        while True:
            if page_number == 0 and first_page is not None:
                response = first_page
            else:
                response = self._load_live_page(
                    token=token,
                    page_number=page_number,
                    after_key=after_key,
                )
            records = self._parse_response(response)
            for record in records:
                fingerprint = _record_fingerprint(record)
                progress_key = f"record:{record.dataset}"
                if since_progress.get(progress_key) == fingerprint:
                    yield ProgressMarker(
                        record_id=record.dataset,
                        fingerprint=fingerprint,
                    )
                    continue
                yield from _rucio_dataset_facts_for_records(
                    [record],
                    revision,
                )
                yield ProgressMarker(
                    record_id=record.dataset,
                    fingerprint=fingerprint,
                )
                n_records += 1
            if page_number % 20 == 0:
                log.info(
                    "MONIT Rucio dataset overlay: page=%d records_seen=%d",
                    page_number,
                    n_records,
                )
            response0 = _first_response(response)
            after_key = (
                response0.get("aggregations", {})
                .get("datasets", {})
                .get("after_key")
            )
            if not after_key:
                break
            page_number += 1

    def _load_live_page(
        self,
        *,
        token: str,
        page_number: int,
        after_key: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cache_path = _page_cache_path(
            self.page_cache_dir,
            query_fingerprint=self._query_fingerprint(),
            page_number=page_number,
            base=self.base,
        )
        if self.cache_live_pages and cache_path.is_file():
            cached = json.loads(cache_path.read_text())
            if cached.get("after_key_in") == after_key and (
                _page_cache_entry_fresh(
                    cached, ttl_hours=self.page_cache_ttl_hours
                )
            ):
                return cached["response"]

        response = _monit_msearch(
            grafana_base_url=self.grafana_base_url,
            datasource_id=self.datasource_id,
            index=self.index,
            token=token,
            query=_rucio_dataset_query(
                from_time=self.from_time,
                to_time=self.to_time,
                page_size=self.page_size,
                max_replica_rses=self.max_replica_rses,
                after_key=after_key,
            ),
            timeout=self.timeout,
        )
        quality = _response_quality(response)
        if not quality.complete:
            # The streaming path cannot retroactively downgrade the
            # already-returned health/scope claim, so a partial page
            # fails the run (and is never cached) instead of letting a
            # partial scope be committed.
            raise RuntimeError(
                f"partial MONIT OpenSearch response on page {page_number}: "
                f"{quality.describe()}"
            )
        if self.cache_live_pages:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({
                "after_key_in": after_key,
                "response": response,
                "cached_at": _checked_at(),
            }, sort_keys=True))
            tmp_path.replace(cache_path)
        return response


class MissingMONITCredential(RuntimeError):
    pass


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _monit_change_probe(source: Any) -> Any:
    config = {
        key: value
        for key, value in vars(source).items()
        if key != "change_probe"
    }
    # Several cache parsers stamp records with `_today()` when the cached MONIT
    # payload is a raw response. Include that day bucket so the probe cannot
    # skip across midnight while emitted record IDs/revisions would change.
    config["snapshot_date"] = _today()
    return cache_or_forced_live_change_probe(
        cache_paths=source.cache_paths,
        config=config,
        emit_targets=source.__class__,
        base=source.base,
    )


def _monit_preflight(
    *,
    source_name: str,
    records_path: str,
    token_env: str,
    endpoint: str,
    cache_loader: Callable[[], tuple[list[Any], _ResponseQuality]],
    cache_reason: str,
    base: str | None = None,
) -> PreflightResult:
    path = resolve_repo_path(records_path, base=base)
    if path.is_file():
        records, _ = cache_loader()
        return PreflightResult(
            source_name=source_name,
            status="ok",
            mode="cache",
            required=False,
            record_count=len(records),
            content_hash=_file_hash(path),
            cache_path=str(path),
            reason=cache_reason,
            checked_at=_checked_at(),
        )
    if os.environ.get(token_env):
        return PreflightResult(
            source_name=source_name,
            status="ok",
            mode="live",
            required=False,
            credential_refs=(token_env,),
            endpoint=endpoint,
            reason="MONIT Grafana token present",
            checked_at=_checked_at(),
        )
    return PreflightResult(
        source_name=source_name,
        status="missing_credential",
        mode="live",
        required=False,
        credential_refs=(token_env,),
        cache_path=str(path),
        reason=f"{token_env} is not set and {path} is missing",
        checked_at=_checked_at(),
    )


def _load_cache_or_live(
    *,
    records_path: str,
    token_env: str,
    cache_loader: Callable[[], tuple[list[Any], _ResponseQuality]],
    live_loader: Callable[[str], tuple[list[Any], _ResponseQuality]],
    base: str | None = None,
) -> tuple[list[Any], str, str, _ResponseQuality]:
    path = resolve_repo_path(records_path, base=base)
    if path.is_file():
        records, quality = cache_loader()
        return records, "cache", _file_hash(path), quality
    token = os.environ.get(token_env)
    if not token:
        raise MissingMONITCredential(f"{token_env} is not set and {path} is missing")
    records, quality = live_loader(token)
    return records, "live", _records_hash(records), quality


def _missing_credential_run(
    mode: str,
    token_env: str,
    reason: str,
) -> ConnectorRun:
    # A missing configured credential must never claim completed_scope:
    # under missing_from_completed_scope semantics an empty-but-complete
    # run would retract every previously ingested record.
    return ConnectorRun(
        facts=[],
        completed_scope=False,
        run_mode=mode,
        health=ConnectorHealth(
            status="missing_credential",
            mode="live",
            credential_refs=(token_env,),
            reason=reason,
            checked_at=_checked_at(),
        ),
    )


def _endpoint_failed_run(
    mode: str,
    token_env: str,
    exc: Exception,
    *,
    endpoint: str,
) -> ConnectorRun:
    # Same discipline for failed reads: emit nothing and claim nothing.
    return ConnectorRun(
        facts=[],
        completed_scope=False,
        run_mode=mode,
        health=ConnectorHealth(
            status="endpoint_failed",
            mode="live",
            credential_refs=(token_env,),
            endpoint=endpoint,
            reason=f"{type(exc).__name__}: {exc}",
            checked_at=_checked_at(),
        ),
    )


def _source_run(
    *,
    facts: Iterator[NodeFact | EdgeFact],
    mode: str,
    run_mode: str,
    token_env: str,
    endpoint: str,
    revision: dict[str, Any],
    records: list[Any],
    ok_reason: str,
    empty_reason: str,
    quality: _ResponseQuality | None = None,
) -> ConnectorRun:
    partial = quality is not None and not quality.complete
    if partial:
        # HTTP 200 but the response itself is partial (timed out, shard
        # failures, or capped terms aggregations): emit the records it
        # carries, but the closed status vocabulary has no 'degraded',
        # so report endpoint_failed — and never claim the scope, or
        # missing_from_completed_scope would retract every record the
        # partial response happened to drop.
        status = "endpoint_failed"
        reason = (
            f"partial MONIT OpenSearch response ({quality.describe()}); "
            f"{len(records)} records emitted, no complete scope claimed"
        )
        completed = False
    elif not records:
        # Zero buckets under an ignore_unavailable wildcard-index query
        # is indistinguishable from a renamed index or a stalled
        # ingestion pipeline, so never claim a complete empty scope (it
        # would retract every previously ingested record).
        status = "skipped_optional"
        reason = f"{empty_reason}; no complete scope claimed"
        completed = False
    else:
        status = "ok"
        reason = ok_reason
        completed = mode in {"scope_complete", "reconcile"}
    return ConnectorRun(
        facts=facts,
        completed_scope=completed,
        run_mode=mode,
        health=ConnectorHealth(
            status=status,
            mode=run_mode,
            credential_refs=((token_env,) if run_mode == "live" else ()),
            record_count=len(records),
            content_hash=revision["content_hash"],
            endpoint=endpoint if run_mode == "live" else None,
            reason=reason,
            checked_at=_checked_at(),
        ),
    )


def _sitemon_query(
    *,
    query_filter: str,
    from_time: str,
    to_time: str,
    max_sites: int,
) -> dict[str, Any]:
    query = _time_range_query(from_time=from_time, to_time=to_time)
    query["bool"]["must"] = [{
        "query_string": {
            "query": query_filter,
            "analyze_wildcard": True,
        }
    }]
    return {
        "size": 0,
        "query": query,
        "aggs": {
            "groups": {
                "terms": {
                    "field": "data.dst_experiment_site",
                    "size": max_sites,
                },
                "aggs": {
                    "status_breakdown": {
                        "terms": {"field": "data.status", "size": 10}
                    }
                },
            }
        },
    }


def _condor_query(
    *,
    from_time: str,
    to_time: str,
    max_sites: int,
) -> dict[str, Any]:
    return {
        "size": 0,
        "query": _time_range_query(from_time=from_time, to_time=to_time),
        "aggs": {
            "sites": {
                "terms": {"field": "data.Site", "size": max_sites},
                "aggs": {
                    "by_status": {
                        "terms": {"field": "data.Status", "size": 20}
                    },
                    "by_job_type": {
                        "terms": {"field": "data.CMS_JobType", "size": 20}
                    },
                    "by_error": {
                        "terms": {"field": "data.ErrorClass", "size": 20}
                    },
                    "total_core_hrs": {"sum": {"field": "data.sum_CoreHr"}},
                    "total_cpu_time_hrs": {
                        "sum": {"field": "data.sum_CpuTimeHr"}
                    },
                    "avg_queue_hrs": {"avg": {"field": "data.avg_QueueHrs"}},
                    "total_jobs": {"sum": {"field": "data.sum_count"}},
                },
            }
        },
    }


def _rucio_transfer_query(
    *,
    from_time: str,
    to_time: str,
    max_src_sites: int,
    max_dst_sites: int,
) -> dict[str, Any]:
    query = _time_range_query(from_time=from_time, to_time=to_time)
    query["bool"]["must"] = [{
        "query_string": {
            "query": "data.event_type:(transfer-done OR transfer-failed)",
            "analyze_wildcard": True,
        }
    }]
    return {
        "size": 0,
        "query": query,
        "aggs": {
            "by_src_rse": {
                "terms": {"field": "data.src_rse", "size": max_src_sites},
                "aggs": {
                    "by_dst_rse": {
                        "terms": {
                            "field": "data.dst_rse",
                            "size": max_dst_sites,
                        },
                        "aggs": {
                            "bytes_total": {"sum": {"field": "data.bytes"}},
                            "avg_duration": {
                                "avg": {"field": "data.duration"}
                            },
                            "by_event_type": {
                                "terms": {
                                    "field": "data.event_type",
                                    "size": 10,
                                }
                            },
                            "top_activity": {
                                "terms": {"field": "data.activity", "size": 5}
                            },
                            "failure_reasons": {
                                "terms": {"field": "data.reason", "size": 5}
                            },
                        },
                    }
                },
            }
        },
    }


def _rucio_dataset_query(
    *,
    from_time: str,
    to_time: str,
    page_size: int,
    max_replica_rses: int,
    after_key: dict[str, Any] | None,
) -> dict[str, Any]:
    composite: dict[str, Any] = {
        "size": page_size,
        "sources": [{"dataset": {"terms": {"field": "data.dataset"}}}],
    }
    if after_key:
        composite["after"] = after_key
    query = _time_range_query(from_time=from_time, to_time=to_time)
    query["bool"]["must"] = [{
        "query_string": {
            "query": "data.dataset_access_type:VALID",
            "analyze_wildcard": True,
        }
    }]
    return {
        "size": 0,
        "query": query,
        "aggs": {
            "datasets": {
                "composite": composite,
                "aggs": {
                    "replicas": {
                        "terms": {
                            "field": "data.rse",
                            "size": max_replica_rses,
                        }
                    },
                    "sample": {
                        "top_hits": {
                            "size": 1,
                            "_source": {
                                "includes": [
                                    "data.data_tier_name",
                                    "data.acquisition_era_name",
                                    "data.dbs_event_count",
                                    "data.dbs_n_files",
                                    "data.dbs_size",
                                    "data.physics_group_name",
                                ]
                            },
                        }
                    },
                },
            }
        },
    }


def _time_range_query(*, from_time: str, to_time: str) -> dict[str, Any]:
    return {
        "bool": {
            "filter": [{
                "range": {
                    "metadata.timestamp": {
                        "gte": from_time,
                        "lte": to_time,
                        "format": TIME_FORMAT,
                    }
                }
            }],
        }
    }


def _monit_msearch(
    *,
    grafana_base_url: str,
    datasource_id: int,
    index: str,
    token: str,
    query: dict[str, Any],
    timeout: float,
    retries: int = 8,
) -> dict[str, Any]:
    url = (
        f"{grafana_base_url}/api/datasources/proxy/"
        f"{datasource_id}/_msearch"
    )
    meta = {
        "search_type": "query_then_fetch",
        "ignore_unavailable": True,
        "index": [index],
    }
    payload = json.dumps(meta) + "\n" + json.dumps(query) + "\n"
    response = None
    retry_statuses = {429, 502, 503, 504}
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                data=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == retries:
                raise
            sleep_s = min(30.0, 2.0 ** attempt)
            log.warning(
                "MONIT _msearch request failed: %s; retrying in %.1fs",
                type(exc).__name__,
                sleep_s,
            )
            time.sleep(sleep_s)
            continue
        if response.status_code not in retry_statuses:
            break
        if attempt == retries:
            break
        sleep_s = min(30.0, 2.0 ** attempt)
        log.warning(
            "MONIT _msearch transient status=%s; retrying in %.1fs",
            response.status_code,
            sleep_s,
        )
        time.sleep(sleep_s)
    if response is None:
        assert last_exc is not None
        raise last_exc
    response.raise_for_status()
    data = response.json()
    responses = data.get("responses", [data])
    if responses and responses[0].get("error"):
        error = responses[0]["error"]
        raise RuntimeError(
            f"OpenSearch error: {error.get('type', '?')}: "
            f"{error.get('reason', error)}"
        )
    return data


def _parse_sitemon_response(
    payload: dict[str, Any],
    *,
    snapshot_date: str,
) -> list[SiteAvailabilityRecord]:
    response = _first_response(payload)
    buckets = (
        response.get("aggregations", {})
        .get("groups", {})
        .get("buckets", [])
    )
    records = []
    for bucket in buckets:
        site = str(bucket.get("key") or "").strip()
        if not site:
            continue
        counts = _bucket_dict(
            bucket.get("status_breakdown", {}).get("buckets", [])
        )
        total = sum(counts.values())
        ok = counts.get("OK", 0)
        availability = (ok / total * 100.0) if total else 0.0
        records.append(SiteAvailabilityRecord(
            site=site,
            snapshot_date=snapshot_date,
            availability_pct=round(availability, 2),
            status=_classify_availability(availability),
            total_tests=total,
            ok_count=ok,
            warning_count=counts.get("WARNING", 0),
            critical_count=counts.get("CRITICAL", 0),
            unknown_count=counts.get("UNKNOWN", 0),
        ))
    return records


def _parse_condor_response(
    payload: dict[str, Any],
    *,
    snapshot_date: str,
) -> list[CondorComputeRecord]:
    response = _first_response(payload)
    buckets = (
        response.get("aggregations", {})
        .get("sites", {})
        .get("buckets", [])
    )
    records = []
    for bucket in buckets:
        site = str(bucket.get("key") or "").strip()
        if not site:
            continue
        status_breakdown = _bucket_dict(
            bucket.get("by_status", {}).get("buckets", [])
        )
        records.append(CondorComputeRecord(
            site=site,
            snapshot_date=snapshot_date,
            total_jobs=int(
                bucket.get("total_jobs", {}).get("value")
                or bucket.get("doc_count")
                or 0
            ),
            core_hours=_float_agg(bucket, "total_core_hrs"),
            cpu_time_hours=_float_agg(bucket, "total_cpu_time_hrs"),
            avg_queue_hours=_optional_float_agg(bucket, "avg_queue_hrs"),
            jobs_running=status_breakdown.get("Running", 0),
            jobs_idle=status_breakdown.get("Idle", 0),
            jobs_held=status_breakdown.get("Held", 0),
            status_breakdown=status_breakdown,
            job_type_breakdown=_bucket_dict(
                bucket.get("by_job_type", {}).get("buckets", [])
            ),
            error_breakdown=_bucket_dict(
                bucket.get("by_error", {}).get("buckets", [])
            ),
        ))
    return records


def _parse_rucio_transfer_response(
    payload: dict[str, Any],
    *,
    snapshot_date: str,
) -> list[RucioTransferRecord]:
    response = _first_response(payload)
    src_buckets = (
        response.get("aggregations", {})
        .get("by_src_rse", {})
        .get("buckets", [])
    )
    records: dict[tuple[str, str, str], RucioTransferRecord] = {}
    for src_bucket in src_buckets:
        src_site = _rse_to_site(str(src_bucket.get("key") or ""))
        if not src_site:
            continue
        for dst_bucket in src_bucket.get("by_dst_rse", {}).get("buckets", []):
            dst_site = _rse_to_site(str(dst_bucket.get("key") or ""))
            if not dst_site:
                continue
            event_counts = _bucket_dict(
                dst_bucket.get("by_event_type", {}).get("buckets", [])
            )
            activity_buckets = (
                dst_bucket.get("top_activity", {}).get("buckets", [])
            )
            record = RucioTransferRecord(
                src_site=src_site,
                dst_site=dst_site,
                snapshot_date=snapshot_date,
                done_count=event_counts.get("transfer-done", 0),
                failed_count=event_counts.get("transfer-failed", 0),
                bytes_total=int(
                    dst_bucket.get("bytes_total", {}).get("value") or 0
                ),
                avg_duration_s=float(
                    dst_bucket.get("avg_duration", {}).get("value") or 0.0
                ),
                top_activity=(
                    str(activity_buckets[0].get("key"))
                    if activity_buckets else ""
                ),
                failure_reasons=_bucket_dict(
                    dst_bucket.get("failure_reasons", {}).get("buckets", [])
                ),
            )
            key = (record.src_site, record.dst_site, record.snapshot_date)
            records[key] = _merge_rucio_transfer_records(
                records.get(key),
                record,
            )
    return list(records.values())


def _parse_rucio_dataset_response(
    payload: dict[str, Any],
    *,
    snapshot_date: str,
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDED_DATASET_PATTERNS,
) -> list[RucioDatasetRecord]:
    response = _first_response(payload)
    buckets = (
        response.get("aggregations", {})
        .get("datasets", {})
        .get("buckets", [])
    )
    records: list[RucioDatasetRecord] = []
    for bucket in buckets:
        key = bucket.get("key")
        dataset = key.get("dataset") if isinstance(key, dict) else key
        dataset = str(dataset or "").strip()
        if not dataset or _is_excluded_dataset(dataset, exclude_patterns):
            continue
        replica_rses = sorted({
            _normalize_rse(str(item.get("key") or ""))
            for item in bucket.get("replicas", {}).get("buckets", [])
            if item.get("key")
        })
        replica_sites = sorted({_rse_to_site(rse) for rse in replica_rses})
        sample_hits = (
            bucket.get("sample", {})
            .get("hits", {})
            .get("hits", [])
        )
        meta: dict[str, Any] = {}
        if sample_hits:
            meta = sample_hits[0].get("_source", {}).get("data", {})
        records.append(RucioDatasetRecord(
            dataset=dataset,
            snapshot_date=snapshot_date,
            data_tier=str(meta.get("data_tier_name") or ""),
            acquisition_era=str(meta.get("acquisition_era_name") or ""),
            event_count=int(meta.get("dbs_event_count") or 0),
            file_count=int(meta.get("dbs_n_files") or 0),
            size_bytes=int(meta.get("dbs_size") or 0),
            physics_group=str(meta.get("physics_group_name") or ""),
            replica_sites=tuple(replica_sites),
            replica_rses=tuple(replica_rses),
        ))
    return records


def _first_response(payload: dict[str, Any]) -> dict[str, Any]:
    responses = payload.get("responses")
    if isinstance(responses, list):
        return responses[0] if responses else {}
    return payload


def _bucket_dict(buckets: Iterable[dict[str, Any]]) -> dict[str, int]:
    return {
        str(bucket.get("key") or "UNKNOWN"): int(bucket.get("doc_count") or 0)
        for bucket in buckets
    }


def _float_agg(bucket: dict[str, Any], name: str) -> float:
    return float(bucket.get(name, {}).get("value") or 0.0)


def _optional_float_agg(bucket: dict[str, Any], name: str) -> float | None:
    value = bucket.get(name, {}).get("value")
    return float(value) if value is not None else None


def _classify_availability(availability: float) -> str:
    if availability >= 90.0:
        return "OK"
    if availability >= 70.0:
        return "WARNING"
    return "CRITICAL"


def _sam_record_from_mapping(item: dict[str, Any]) -> SiteAvailabilityRecord:
    return SiteAvailabilityRecord(
        site=str(item["site"]),
        snapshot_date=str(item.get("snapshot_date") or item.get("date") or _today()),
        availability_pct=float(item.get("availability_pct") or 0.0),
        status=str(item.get("status") or "UNKNOWN"),
        total_tests=int(item.get("total_tests") or 0),
        ok_count=int(item.get("ok_count") or 0),
        warning_count=int(item.get("warning_count") or 0),
        critical_count=int(item.get("critical_count") or 0),
        unknown_count=int(item.get("unknown_count") or 0),
    )


def _condor_record_from_mapping(item: dict[str, Any]) -> CondorComputeRecord:
    return CondorComputeRecord(
        site=str(item["site"]),
        snapshot_date=str(item.get("snapshot_date") or item.get("date") or _today()),
        total_jobs=int(item.get("total_jobs") or 0),
        core_hours=float(item.get("core_hours") or 0.0),
        cpu_time_hours=float(item.get("cpu_time_hours") or 0.0),
        avg_queue_hours=(
            float(item["avg_queue_hours"])
            if item.get("avg_queue_hours") is not None else None
        ),
        jobs_running=int(item.get("jobs_running") or 0),
        jobs_idle=int(item.get("jobs_idle") or 0),
        jobs_held=int(item.get("jobs_held") or 0),
        status_breakdown=dict(item.get("status_breakdown") or {}),
        job_type_breakdown=dict(item.get("job_type_breakdown") or {}),
        error_breakdown=dict(item.get("error_breakdown") or {}),
    )


def _rucio_transfer_record_from_mapping(
    item: dict[str, Any],
) -> RucioTransferRecord:
    return RucioTransferRecord(
        src_site=_rse_to_site(str(item.get("src_site") or item.get("src_rse"))),
        dst_site=_rse_to_site(str(item.get("dst_site") or item.get("dst_rse"))),
        snapshot_date=str(item.get("snapshot_date") or item.get("date") or _today()),
        done_count=int(item.get("done_count") or 0),
        failed_count=int(item.get("failed_count") or 0),
        bytes_total=int(item.get("bytes_total") or 0),
        avg_duration_s=float(item.get("avg_duration_s") or 0.0),
        top_activity=str(item.get("top_activity") or ""),
        failure_reasons=dict(item.get("failure_reasons") or {}),
    )


def _merge_rucio_transfer_records(
    left: RucioTransferRecord | None,
    right: RucioTransferRecord,
) -> RucioTransferRecord:
    if left is None:
        return right
    left_weight = left.total_count
    right_weight = right.total_count
    total_weight = left_weight + right_weight
    if total_weight:
        avg_duration = (
            (left.avg_duration_s * left_weight)
            + (right.avg_duration_s * right_weight)
        ) / total_weight
    else:
        avg_duration = 0.0
    failure_reasons = dict(left.failure_reasons or {})
    for reason, count in (right.failure_reasons or {}).items():
        failure_reasons[reason] = failure_reasons.get(reason, 0) + count
    return RucioTransferRecord(
        src_site=left.src_site,
        dst_site=left.dst_site,
        snapshot_date=left.snapshot_date,
        done_count=left.done_count + right.done_count,
        failed_count=left.failed_count + right.failed_count,
        bytes_total=left.bytes_total + right.bytes_total,
        avg_duration_s=avg_duration,
        top_activity=left.top_activity or right.top_activity,
        failure_reasons=failure_reasons,
    )


def _dedupe_rucio_transfer_records(
    records: list[RucioTransferRecord],
) -> list[RucioTransferRecord]:
    merged: dict[tuple[str, str, str], RucioTransferRecord] = {}
    for record in records:
        key = (record.src_site, record.dst_site, record.snapshot_date)
        merged[key] = _merge_rucio_transfer_records(merged.get(key), record)
    return list(merged.values())


def _rucio_dataset_record_from_mapping(
    item: dict[str, Any],
) -> RucioDatasetRecord:
    raw_replica_rses = item.get("replica_rses") or ()
    raw_replica_sites = item.get("replica_sites") or ()
    replica_rses = tuple(sorted({
        _normalize_rse(str(rse)) for rse in raw_replica_rses if str(rse)
    }))
    if not replica_rses:
        replica_rses = tuple(sorted({
            _normalize_rse(str(value))
            for value in raw_replica_sites
            if str(value)
        }))
    replica_sites = tuple(sorted({
        _rse_to_site(str(value)) for value in (*raw_replica_sites, *replica_rses)
        if str(value)
    }))
    return RucioDatasetRecord(
        dataset=str(item.get("dataset") or item.get("dataset_path")),
        snapshot_date=str(item.get("snapshot_date") or item.get("date") or _today()),
        data_tier=str(item.get("data_tier") or ""),
        acquisition_era=str(item.get("acquisition_era") or ""),
        event_count=int(item.get("event_count") or 0),
        file_count=int(item.get("file_count") or 0),
        size_bytes=int(item.get("size_bytes") or 0),
        physics_group=str(item.get("physics_group") or ""),
        replica_sites=replica_sites,
        replica_rses=replica_rses,
    )


def _records_hash(records: list[Any]) -> str:
    import hashlib

    payload = json.dumps(
        [asdict(record) for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record_fingerprint(record: Any) -> str:
    import hashlib

    payload = json.dumps(
        asdict(record),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _query_fingerprint(
    *,
    source: str,
    from_time: str,
    to_time: str,
    page_size: int,
    max_replica_rses: int,
    snapshot_date: str,
) -> str:
    import hashlib

    payload = json.dumps({
        "source": source,
        "from_time": from_time,
        "to_time": to_time,
        "page_size": page_size,
        "max_replica_rses": max_replica_rses,
        # Relative time expressions ("now-24h") resolve differently
        # every day; the snapshot date keys the cache to the day the
        # window actually covered. v3 also invalidates every date-free
        # v2 cache entry.
        "snapshot_date": snapshot_date,
        "query_version": "rucio_dataset_composite_v3",
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _page_cache_entry_fresh(
    cached: dict[str, Any],
    *,
    ttl_hours: float,
) -> bool:
    """True when a live page-cache entry may satisfy this run.

    The entry must carry a parseable ``cached_at`` stamped on the same
    UTC day as this run's snapshot date and be younger than
    *ttl_hours*. A page from a previous day (or of unknown age) must
    never satisfy a new day's run — replaying it would emit
    yesterday's datasets stamped with today's date under a completed
    scope claim, and the substrate would retract everything else.
    """
    raw = cached.get("cached_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        cached_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)
    cached_at = cached_at.astimezone(timezone.utc)
    if cached_at.strftime("%Y-%m-%d") != _today():
        return False
    age = datetime.now(timezone.utc) - cached_at
    return timedelta(0) <= age <= timedelta(hours=ttl_hours)


def _page_cache_path(
    page_cache_dir: str,
    *,
    query_fingerprint: str,
    page_number: int,
    base: str | None = None,
) -> Path:
    root = resolve_repo_path(page_cache_dir, base=base)
    return root / query_fingerprint[:16] / f"page-{page_number:06d}.json"


def _file_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _revision(
    *,
    run_id: str,
    source: str,
    run_mode: str,
    from_time: str,
    to_time: str,
    hash_value: str,
    n_records: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "source": source,
        "mode": run_mode,
        "from_time": from_time,
        "to_time": to_time,
        "content_hash": hash_value,
        "n_records": n_records,
    }


def _sam_facts_for_records(
    records: list[SiteAvailabilityRecord],
    revision: dict[str, Any],
) -> Iterator[NodeFact | EdgeFact]:
    for record in records:
        yield _sam_snapshot_node(record, revision)
        yield EdgeFact(
            src=f"site:{record.site}",
            dst=record.node_id,
            edge_type="hosts",
            attrs={
                "relationship": "monitoring_snapshot",
                "metric": "sam_site_availability",
            },
            source_record_id={"snapshot_id": record.node_id},
            source_revision=revision,
        )


def _condor_facts_for_records(
    records: list[CondorComputeRecord],
    revision: dict[str, Any],
) -> Iterator[NodeFact | EdgeFact]:
    for record in records:
        yield _condor_snapshot_node(record, revision)
        yield EdgeFact(
            src=f"site:{record.site}",
            dst=record.node_id,
            edge_type="hosts",
            attrs={
                "relationship": "monitoring_snapshot",
                "metric": "condor_compute_summary",
            },
            source_record_id={"snapshot_id": record.node_id},
            source_revision=revision,
        )


def _rucio_transfer_facts_for_records(
    records: list[RucioTransferRecord],
    revision: dict[str, Any],
) -> Iterator[NodeFact | EdgeFact]:
    seen_site_edges: set[tuple[str, str]] = set()
    for record in records:
        yield _rucio_transfer_node(record, revision)
        for site, role in (
            (record.src_site, "transfer_source"),
            (record.dst_site, "transfer_destination"),
        ):
            yield EdgeFact(
                src=f"site:{site}",
                dst=record.node_id,
                edge_type="hosts",
                attrs={"role": role, "metric": "rucio_transfer_summary"},
                source_record_id={
                    "transfer_id": record.transfer_id,
                    "role": role,
                },
                source_revision=revision,
            )
        site_edge = (record.src_site, record.dst_site)
        if site_edge not in seen_site_edges:
            seen_site_edges.add(site_edge)
            yield EdgeFact(
                src=f"site:{record.src_site}",
                dst=f"site:{record.dst_site}",
                edge_type="transfers_to",
                attrs={
                    "metric": "rucio_transfer_summary",
                    "snapshot_date": record.snapshot_date,
                },
                source_record_id={
                    "src_site": record.src_site,
                    "dst_site": record.dst_site,
                    "snapshot_date": record.snapshot_date,
                },
                source_revision=revision,
            )


def _rucio_dataset_facts_for_records(
    records: list[RucioDatasetRecord],
    revision: dict[str, Any],
) -> Iterator[NodeFact | EdgeFact]:
    for record in records:
        yield _rucio_dataset_node(record, revision)
        for site in sorted(set(record.replica_sites)):
            yield EdgeFact(
                src=f"site:{site}",
                dst=record.node_id,
                edge_type="hosts",
                attrs={
                    "relationship": "dataset_replica",
                    "snapshot_date": record.snapshot_date,
                },
                source_record_id={
                    "dataset": record.dataset,
                    "site": site,
                    "snapshot_date": record.snapshot_date,
                },
                source_revision=revision,
            )
        for rse in sorted(set(record.replica_rses)):
            site = _rse_to_site(rse)
            yield EdgeFact(
                src=f"se:{rse}",
                dst=record.node_id,
                edge_type="hosts",
                attrs={
                    "relationship": "dataset_replica_rse",
                    "rse": rse,
                    "site": site,
                    "snapshot_date": record.snapshot_date,
                },
                source_record_id={
                    "dataset": record.dataset,
                    "rse": rse,
                    "snapshot_date": record.snapshot_date,
                },
                source_revision=revision,
            )


def _sam_snapshot_node(
    record: SiteAvailabilityRecord,
    revision: dict[str, Any],
) -> NodeFact:
    text = (
        f"SAM/SiteMon availability for {record.site} on "
        f"{record.snapshot_date}: {record.availability_pct:.2f}% "
        f"({record.status}); {record.ok_count}/{record.total_tests} OK."
    )
    return NodeFact(
        node_id=record.node_id,
        subtype="monitoring_snapshot",
        attrs={
            "label": f"SAM {record.site} {record.snapshot_date}",
            "snapshot_id": record.node_id,
            "subject": record.site,
            "metric": "sam_site_availability",
            "value": record.availability_pct,
            "unit": "percent",
            "observed_at": record.snapshot_date,
            "status": record.status,
            "total_tests": record.total_tests,
            "ok_count": record.ok_count,
            "warning_count": record.warning_count,
            "critical_count": record.critical_count,
            "unknown_count": record.unknown_count,
            "text": text,
        },
        source_record_id={"snapshot_id": record.node_id},
        source_revision=revision,
    )


def _condor_snapshot_node(
    record: CondorComputeRecord,
    revision: dict[str, Any],
) -> NodeFact:
    cpu_eff = record.cpu_efficiency
    cpu_eff_text = f"{cpu_eff * 100:.1f}%" if cpu_eff is not None else "unknown"
    text = (
        f"Condor compute summary for {record.site} on "
        f"{record.snapshot_date}: {record.total_jobs} jobs, "
        f"{record.core_hours:.1f} core-hours, CPU efficiency "
        f"{cpu_eff_text}, {record.jobs_running} running, "
        f"{record.jobs_idle} idle, {record.jobs_held} held."
    )
    return NodeFact(
        node_id=record.node_id,
        subtype="monitoring_snapshot",
        attrs={
            "label": f"Condor {record.site} {record.snapshot_date}",
            "snapshot_id": record.node_id,
            "subject": record.site,
            "metric": "condor_compute_summary",
            "value": float(record.total_jobs),
            "unit": "jobs",
            "observed_at": record.snapshot_date,
            "total_jobs": record.total_jobs,
            "core_hours": round(record.core_hours, 2),
            "cpu_time_hours": round(record.cpu_time_hours, 2),
            "cpu_efficiency": round(cpu_eff, 4) if cpu_eff is not None else None,
            "avg_queue_hours": (
                round(record.avg_queue_hours, 2)
                if record.avg_queue_hours is not None else None
            ),
            "jobs_running": record.jobs_running,
            "jobs_idle": record.jobs_idle,
            "jobs_held": record.jobs_held,
            "status_breakdown": record.status_breakdown or {},
            "job_type_breakdown": record.job_type_breakdown or {},
            "error_breakdown": record.error_breakdown or {},
            "text": text,
        },
        source_record_id={"snapshot_id": record.node_id},
        source_revision=revision,
    )


def _rucio_transfer_node(
    record: RucioTransferRecord,
    revision: dict[str, Any],
) -> NodeFact:
    state = "ok" if record.failed_count == 0 else "mixed"
    text = (
        f"Rucio transfer summary {record.src_site} to {record.dst_site} "
        f"on {record.snapshot_date}: {record.done_count} done, "
        f"{record.failed_count} failed, {record.bytes_total} bytes, "
        f"success rate {record.success_rate:.1%}."
    )
    return NodeFact(
        node_id=record.node_id,
        subtype="transfer_job",
        attrs={
            "label": (
                f"Rucio {record.src_site} -> {record.dst_site} "
                f"{record.snapshot_date}"
            ),
            "transfer_id": record.transfer_id,
            "src_site": record.src_site,
            "dst_site": record.dst_site,
            "state": state,
            "num_files": record.total_count,
            "total_bytes": record.bytes_total,
            "completed_at": record.snapshot_date,
            "done_count": record.done_count,
            "failed_count": record.failed_count,
            "success_rate": round(record.success_rate, 4),
            "avg_duration_s": round(record.avg_duration_s, 2),
            "top_activity": record.top_activity,
            "failure_reasons": record.failure_reasons or {},
            "text": text,
        },
        source_record_id={"transfer_id": record.transfer_id},
        source_revision=revision,
    )


def _rucio_dataset_node(
    record: RucioDatasetRecord,
    revision: dict[str, Any],
) -> NodeFact:
    text = (
        f"Dataset {record.dataset}; tier={record.data_tier}; "
        f"era={record.acquisition_era}; replicas={len(record.replica_sites)}."
    )
    return NodeFact(
        node_id=record.node_id,
        subtype="dataset",
        attrs={
            "label": _dataset_label(record.dataset),
            "dataset_id": record.node_id,
            "name": record.dataset,
            "dataset_name": record.dataset,
            "tier": record.data_tier,
            "era": record.acquisition_era,
            "event_count": record.event_count,
            "file_count": record.file_count,
            "total_size_bytes": record.size_bytes,
            "physics_group": record.physics_group,
            "replica_sites": list(record.replica_sites),
            "replica_rses": list(record.replica_rses),
            "replica_count": len(record.replica_sites),
            "rse_replica_count": len(record.replica_rses),
            "observed_at": record.snapshot_date,
            "overlay_source": "monit_rucio_daily_stats",
            "text": text,
        },
        source_record_id={"dataset": record.dataset},
        source_revision=revision,
    )


def _safe_node_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _normalize_rse(rse: str) -> str:
    return _safe_node_part(rse.strip())


def _rse_to_site(rse: str) -> str:
    value = _normalize_rse(rse)
    for suffix in RSE_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _is_excluded_dataset(
    dataset: str,
    patterns: tuple[str, ...] = DEFAULT_EXCLUDED_DATASET_PATTERNS,
) -> bool:
    return any(pattern in dataset for pattern in patterns)


def _dataset_label(dataset: str, *, budget: int = 120) -> str:
    if len(dataset) <= budget:
        return dataset
    parts = dataset.split("/")
    if len(parts) >= 4 and parts[0] == "":
        primary = parts[1]
        tier = parts[-1]
        reserved = len(f"/{primary}/.../{tier}")
        middle_room = max(0, budget - reserved - 3)
        middle = parts[2]
        if len(middle) > middle_room:
            middle = middle[:middle_room] + "..."
        return f"/{primary}/{middle}/{tier}"
    return dataset[: budget - 3] + "..."
