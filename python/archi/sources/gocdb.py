"""Cache-backed GOCDB downtime source.

Ported from okg-deployments ``cms/cms_sources/gocdb.py`` (280 LOC,
``GoCDBDowntimeSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes: cache
helpers come from :mod:`archi.auth.cache` with an explicit ``base``
parameter, and the hardcoded ``data/cms/...`` paths are parameters
(defaults keep the cms layout minus the ``cms/`` segment).

Kept-behavior note: as in the original, the CRIC ``sites_path`` and
CRIC-core ``services_path`` topology caches are *required* — they are
part of ``cache_paths`` (probe/preflight hash them). They are not
optional reference targets here because ``affects`` edges are only
emitted to nodes those caches prove exist.

Empty-cache-fails-loudly: a missing, unreadable, zero-byte, truncated
or empty downtime list is ``cache_missing``, and so is a missing or
empty CRIC topology cache (an empty one would drop every ``affects``
edge under a complete scope); an error-shaped payload is
``endpoint_failed``. Each gives no facts and ``completed_scope=False``
from both ``preflight()`` and ``run()`` (the original raised
``FileNotFoundError`` from ``run()`` only and claimed a complete scope
over ``[]``). Topology entries that are not objects are skipped and
stop the scope claim, as unparseable downtimes do.

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``downtime``, ``site``, and
``infrastructure_service`` ship in ``archi/schemas/operations.yaml``. ::

    gocdb_downtimes:
      module: archi.sources.gocdb
      class: GoCDBDowntimeAdapter
      ownership_id: <instance>.gocdb-downtimes
      admission_policy:
        producer_id: <instance>.gocdb-downtimes
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: gocdb_downtimes
        output_signature:
          nodes:
            - {subtype: downtime}
          edges:
            - {src_subtype: downtime, edge_type: affects, dst_subtype: site}
            - {src_subtype: downtime, edge_type: affects, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: GOCDB downtimes with affects edges to known sites/services
          nodes: [downtime]
          edges:
            - downtime affects site
            - downtime affects infrastructure_service
      source_class: discovery_crawl
      record_identity_kind: scoped_locator
      record_identity_fields: [downtime_id]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: true
      params:
        # cms defaults; the cms deployment used data/cms/...
        records_path: data/gocdb-downtimes/records.json
        sites_path: data/cric/sites.json
        services_path: data/cric-core/services.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

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
)
from archi.sources._cache_report import (
    CacheUnusable,
    read_cache_json,
    skipped_items_status,
    unusable_cache_preflight,
    unusable_cache_run,
)
from archi.sources._sdk_adapter import ReaderAdapter
from archi.sources.cric import _CMS_SITE_NAME, _Skipped, _read_objects


@dataclass(frozen=True)
class DowntimeRecord:
    downtime_id: int
    primary_key: str
    classification: str
    severity: str
    description: str
    start_date: str
    end_date: str
    hosted_by: str
    affected_services: tuple[str, ...] = ()
    affected_hostnames: tuple[str, ...] = ()

    @property
    def node_id(self) -> str:
        return f"downtime:{self.downtime_id}"


class GoCDBDowntimeSource:
    """Cache-backed GOCDB downtime source.

    Downtime records are deduplicated by `downtime_id`. Edges are emitted
    only to site / CRIC-core service nodes that exist in the local
    topology caches, so the source does not create dangling affects edges.
    """

    name = "gocdb_downtimes"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/gocdb-downtimes/records.json",
        sites_path: str = "data/cric/sites.json",
        services_path: str = "data/cric-core/services.json",
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.sites_path = sites_path
        self.services_path = services_path
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"cache_paths": self.cache_paths},
            emit_targets=GoCDBDowntimeSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path, self.sites_path, self.services_path)

    def preflight(self, mode: str = "live") -> PreflightResult:
        # Same verdict run() reaches for the same caches.
        try:
            records, skipped = self._records_with_skips()
            _sites, _services, topo_skipped = self._topology()
        except CacheUnusable as exc:
            return unusable_cache_preflight(
                exc, source_name=self.name, required=True
            )
        status, reason, _complete = _verdict(
            "local GOCDB downtime cache present",
            record_count=len(records),
            skipped_count=skipped + topo_skipped,
            records_path=self.records_path,
        )
        return PreflightResult(
            source_name=self.name,
            status=status,
            mode="cache",
            required=True,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason=reason,
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        try:
            records, skipped = self._records_with_skips()
            known_sites, service_lookup, topo_skipped = self._topology()
        except CacheUnusable as exc:
            return unusable_cache_run(exc, mode=mode)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _downtime_node(record, revision)
                yield from _affects_edges(
                    record,
                    revision,
                    known_sites=known_sites,
                    service_lookup=service_lookup,
                )

        status, reason, complete = _verdict(
            "local GOCDB downtime cache used",
            record_count=len(records),
            skipped_count=skipped + topo_skipped,
            records_path=self.records_path,
        )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and complete
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

    def _topology(self) -> tuple[set[str], dict[str, str], int]:
        """The required CRIC topology caches (see the module note).

        Read like the CRIC source reads them: a missing, unreadable,
        truncated or empty cache is ``cache_missing``, an API error body
        or a sites cache with no CMS site name is ``endpoint_failed``,
        in preflight() and run() alike. An empty topology is refused
        rather than accepted: it would drop every ``affects`` edge while
        the run claimed a complete scope. Returns the lookups plus the
        number of non-object entries skipped (which stops the claim).
        """
        skipped = _Skipped()
        sites = _read_objects(
            self.sites_path, base=self.base, skipped=skipped,
            key_pattern=_CMS_SITE_NAME,
        )
        services = _read_objects(
            self.services_path, base=self.base, skipped=skipped
        )
        return _known_sites(sites), _service_lookup(services), skipped.total

    def _records(self) -> list[DowntimeRecord]:
        return self._records_with_skips()[0]

    def _records_with_skips(self) -> tuple[list[DowntimeRecord], int]:
        # An empty downtime list is refused as cache_missing (operator
        # decision 2026-09-28: an empty cache is a loud failure). The
        # snapshot builder already refuses an empty gocdb group.
        payload = read_cache_json(
            self.records_path, expect=list, base=self.base
        )
        grouped: dict[int, dict[str, Any]] = {}
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            try:
                downtime_id = int(item.get("downtime_id") or 0)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if downtime_id <= 0:
                skipped += 1
                continue
            entry = grouped.setdefault(downtime_id, {
                "downtime_id": downtime_id,
                "primary_key": str(item.get("primary_key") or ""),
                "classification": str(item.get("classification") or ""),
                "severity": str(item.get("severity") or ""),
                "description": str(item.get("description") or ""),
                "start_date": str(item.get("start_date") or ""),
                "end_date": str(item.get("end_date") or ""),
                "hosted_by": str(item.get("hosted_by") or ""),
                "services": set(),
                "hostnames": set(),
            })
            service_type = str(item.get("service_type") or "")
            hostname = str(item.get("hostname") or "")
            if service_type:
                entry["services"].add(service_type)
            if hostname:
                entry["hostnames"].add(hostname)
        return [
            DowntimeRecord(
                downtime_id=entry["downtime_id"],
                primary_key=entry["primary_key"],
                classification=entry["classification"],
                severity=entry["severity"],
                description=entry["description"],
                start_date=entry["start_date"],
                end_date=entry["end_date"],
                hosted_by=entry["hosted_by"],
                affected_services=tuple(sorted(entry["services"])),
                affected_hostnames=tuple(sorted(entry["hostnames"])),
            )
            for entry in grouped.values()
        ], skipped


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verdict(
    ok_reason: str,
    *,
    record_count: int,
    skipped_count: int,
    records_path: str,
) -> tuple[str, str, bool]:
    """(status, reason, may claim a complete scope) for one cache read,
    shared by preflight() and run(). An empty downtime list never gets
    here (read_cache_json refuses it)."""
    status, reason = skipped_items_status(
        status="ok",
        reason=ok_reason,
        record_count=record_count,
        skipped_count=skipped_count,
    )
    return status, reason, not skipped_count


def _known_sites(payload: dict[str, Any]) -> set[str]:
    return {str(k) for k in payload}


def _service_lookup(payload: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for service_name, service in payload.items():
        if not isinstance(service, dict):
            continue
        node_id = f"svc:{service_name}"
        endpoint = str(service.get("endpoint") or "")
        if endpoint:
            host = _endpoint_host(endpoint)
            if host:
                lookup.setdefault(host, node_id)
        name_host = str(service_name).rsplit("-", 1)[-1]
        if "." in name_host:
            lookup.setdefault(name_host, node_id)
    return lookup


def _endpoint_host(endpoint: str) -> str:
    """Hostname of a service endpoint, scheme-prefixed or bare.

    The naive ``split("/", 1)[0]`` parse turned
    ``https://host.cern.ch:8443/path`` into ``https:``, silently
    breaking hostname -> service ``affects`` matching; use
    :func:`urllib.parse.urlparse` when a scheme is present.
    """
    if "://" in endpoint:
        return urlparse(endpoint).hostname or ""
    return endpoint.split("/", 1)[0].split(":", 1)[0]


def _downtime_node(
    record: DowntimeRecord,
    revision: dict[str, Any],
) -> NodeFact:
    services = ", ".join(record.affected_services)
    hosts = ", ".join(record.affected_hostnames)
    text = (
        f"{record.classification.lower()} downtime "
        f"{record.severity.lower()} at {record.hosted_by}: "
        f"{record.description} [{record.start_date} - {record.end_date}] "
        f"{services} {hosts}"
    ).strip()
    return NodeFact(
        node_id=record.node_id,
        subtype="downtime",
        attrs={
            "label": f"Downtime {record.downtime_id}",
            "downtime_id": str(record.downtime_id),
            "primary_key": record.primary_key,
            "site": record.hosted_by,
            "service": services,
            "severity": record.severity.lower(),
            "classification": record.classification.lower(),
            "start_time": record.start_date,
            "end_time": record.end_date,
            "description": record.description,
            "affected_services": list(record.affected_services),
            "affected_hostnames": list(record.affected_hostnames),
            "text": text,
        },
        source_record_id={"downtime_id": record.downtime_id},
        source_revision=revision,
    )


def _affects_edges(
    record: DowntimeRecord,
    revision: dict[str, Any],
    *,
    known_sites: set[str],
    service_lookup: dict[str, str],
) -> Iterator[EdgeFact]:
    if record.hosted_by in known_sites:
        yield EdgeFact(
            src=record.node_id,
            dst=f"site:{record.hosted_by}",
            edge_type="affects",
            attrs={"relationship": "hosted_by"},
            source_record_id={"downtime_id": record.downtime_id},
            source_revision=revision,
        )
    emitted_services: set[str] = set()
    for hostname in record.affected_hostnames:
        target = service_lookup.get(hostname)
        if not target or target in emitted_services:
            continue
        emitted_services.add(target)
        yield EdgeFact(
            src=record.node_id,
            dst=target,
            edge_type="affects",
            attrs={
                "relationship": "affected_service_host",
                "hostname": hostname,
            },
            source_record_id={
                "downtime_id": record.downtime_id,
                "hostname": hostname,
            },
            source_revision=revision,
        )


class GoCDBDowntimeAdapter(ReaderAdapter):
    """Registry adapter for :class:`GoCDBDowntimeSource`.

    The registry-entry template in this module's docstring names this
    class. The reader's behavior is unchanged; this class only drives it
    through the substrate's adapter contract, which a bare reader cannot
    satisfy (its ``ConnectorRun`` has no ``next_cursor``).

    ``profile`` and ``change_probe_kind`` must be string literals; see
    ``ReaderAdapter``. ``test_bundle_source_adapters.py`` parses this
    file and holds them equal to the reader's own values.
    """

    reader_class = GoCDBDowntimeSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
