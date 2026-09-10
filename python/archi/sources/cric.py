"""Cache-backed CRIC topology + CRIC core service/federation sources.

Ported from okg-deployments ``cms/cms_sources/cric.py`` (349 LOC,
``CRICSource``) and ``cms/cms_sources/cric_core.py`` (279 LOC,
``CRICCoreSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs), merged into one module. Behavior is kept
verbatim; the only changes:

- Cache/probe helpers come from :mod:`archi.auth.cache` with an
  explicit ``base`` parameter (the originals assumed a deployments-repo
  checkout and the ``data/cms/`` prefix).
- The hardcoded CMS cache paths are constructor parameters; defaults
  keep the cms layout minus the ``cms/`` segment (``data/cric/...``,
  ``data/cric-core/...``) so the registry template below matches the
  reference-target paths already documented in jira.py/docs.py.

Registry-entry templates — same three prerequisites as
``archi/sources/jira.py``'s template (compose the deployment schema
slices ``archi/schemas/operations.yaml`` +
``archi/schemas/bridges/operations.yaml`` into ``<deployment>/schemas/``
and ``schemas/bridges/``; ``output_scope_summary`` must accompany
``output_signature``; add the standard ``sync:`` block). ::

    cric:
      module: archi.sources.cric
      class: CRICSource
      ownership_id: <instance>.cric
      admission_policy:
        producer_id: <instance>.cric
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: cric
        output_signature:
          nodes:
            - {subtype: facility}
            - {subtype: site}
            - {subtype: storage_endpoint}
            - {subtype: compute_endpoint}
            - {subtype: operator}
          edges:
            - {src_subtype: facility, edge_type: contains, dst_subtype: site}
            - {src_subtype: site, edge_type: contains, dst_subtype: compute_endpoint}
            - {src_subtype: site, edge_type: contains, dst_subtype: storage_endpoint}
            - {src_subtype: operator, edge_type: responsible_for, dst_subtype: site}
        output_scope_summary:
          summary: CRIC topology - facilities, sites, storage/compute endpoints, operators
          nodes: [facility, site, storage_endpoint, compute_endpoint, operator]
          edges:
            - facility contains site
            - site contains compute_endpoint
            - site contains storage_endpoint
            - operator responsible_for site
      source_class: discovery_crawl
      record_identity_kind: remote_id
      record_identity_fields: [name, kind]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: true
      params:
        # cms defaults; the cms deployment used data/cms/cric/*.json
        sites_path: data/cric/sites.json
        storage_units_path: data/cric/storage_units.json
        compute_units_path: data/cric/compute_units.json
        facilities_path: data/cric/facilities.json
        responsibilities_path: data/cric/responsibilities.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete

    cric_core:
      module: archi.sources.cric
      class: CRICCoreSource
      ownership_id: <instance>.cric-core
      admission_policy:
        producer_id: <instance>.cric-core
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: cric_core
        output_signature:
          nodes:
            - {subtype: infrastructure_service}
            - {subtype: federation}
          edges:
            - {src_subtype: site, edge_type: contains, dst_subtype: infrastructure_service}
            - {src_subtype: site, edge_type: member_of, dst_subtype: federation}
        output_scope_summary:
          summary: CRIC core infrastructure services and WLCG federations
          nodes: [infrastructure_service, federation]
          edges:
            - site contains infrastructure_service
            - site member_of federation
      source_class: discovery_crawl
      record_identity_kind: remote_id
      record_identity_fields: [name, kind]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: true
      params:
        # cms defaults; the cms deployment used data/cms/cric-core/*.json
        services_path: data/cric-core/services.json
        rcsites_path: data/cric-core/rcsites.json
        federations_path: data/cric-core/federations.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Literal

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

_TIER_MAP = {0: "T0", 1: "T1", 2: "T2", 3: "T3"}
_NODE_ID_PREFIX = {
    "facility": "facility:",
    "site": "site:",
    "storage_endpoint": "se:",
    "compute_endpoint": "ce:",
    "operator": "op:",
}


@dataclass(frozen=True)
class CRICRecord:
    kind: str
    name: str
    attrs: dict[str, Any]
    contains_ids: tuple[str, ...] = ()
    contained_by: str = ""
    responsibilities: tuple[tuple[str, str], ...] = ()

    @property
    def node_id(self) -> str:
        return f"{_NODE_ID_PREFIX[self.kind]}{self.name}"


class CRICSource:
    """Cache-backed CRIC topology source.

    Emits site/facility/storage/compute/operator facts from a local
    CRIC JSON cache. Live CRIC download will be a later extension; this
    adapter deliberately performs no network I/O.
    """

    name = "cric"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        sites_path: str = "data/cric/sites.json",
        storage_units_path: str = "data/cric/storage_units.json",
        compute_units_path: str = "data/cric/compute_units.json",
        facilities_path: str = "data/cric/facilities.json",
        responsibilities_path: str = "data/cric/responsibilities.json",
        base: str | None = None,
    ) -> None:
        self.sites_path = sites_path
        self.storage_units_path = storage_units_path
        self.compute_units_path = compute_units_path
        self.facilities_path = facilities_path
        self.responsibilities_path = responsibilities_path
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"cache_paths": self.cache_paths},
            emit_targets=CRICSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (
            self.sites_path,
            self.storage_units_path,
            self.compute_units_path,
            self.facilities_path,
            self.responsibilities_path,
        )

    def preflight(self, mode: str = "live") -> PreflightResult:
        missing = [
            str(resolve_repo_path(p, base=self.base))
            for p in self.cache_paths
            if not resolve_repo_path(p, base=self.base).is_file()
        ]
        if missing:
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="cache",
                required=True,
                cache_path=", ".join(missing),
                reason="one or more CRIC cache files are missing",
                checked_at=_checked_at(),
            )
        try:
            records = self._records()
        except ValueError as exc:
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode="cache",
                required=True,
                reason=str(exc),
                checked_at=_checked_at(),
            )
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=True,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason="local CRIC cache present",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        records = self._records()
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _node_fact(record, revision)
            for record in records:
                yield from _edge_facts(record, revision)

        return ConnectorRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="cache",
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason="local CRIC cache used",
            ),
        )

    def _records(self) -> list[CRICRecord]:
        return _build_records(
            sites=load_json(self.sites_path, base=self.base),
            storage_units=load_json(self.storage_units_path, base=self.base),
            compute_units=load_json(self.compute_units_path, base=self.base),
            facilities=load_json(self.facilities_path, base=self.base),
            responsibilities=_responsibilities_result(
                load_json(self.responsibilities_path, base=self.base),
                self.responsibilities_path,
            ),
        )


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _responsibilities_result(payload: Any, path: str) -> list[list[Any]]:
    """Extract the ``result`` rows, failing loudly on schema drift.

    The former ``.get("result", [])`` default silently read an
    error-shaped or drifted payload as "no responsibilities", which a
    completed-scope run would then commit by retracting every operator
    record. A payload without a ``result`` list must abort the run
    instead of emptying it.
    """
    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError(
            f"{path}: expected a dict with a 'result' list of "
            "responsibility rows; refusing to treat a drifted or "
            "error-shaped payload as zero responsibilities"
        )
    result = payload["result"]
    if not isinstance(result, list):
        raise ValueError(
            f"{path}: 'result' must be a list of responsibility rows"
        )
    return result


def _build_records(
    *,
    sites: dict[str, Any],
    storage_units: dict[str, Any],
    compute_units: dict[str, Any],
    facilities: dict[str, Any],
    responsibilities: list[list[Any]],
) -> list[CRICRecord]:
    records: list[CRICRecord] = []
    site_names = set(sites)
    title_map: dict[str, str] = {}
    for name, site in sites.items():
        sitedb_title = site.get("sitedb_title")
        if sitedb_title:
            title_map[sitedb_title] = name
        title_map[name] = name

    for name, facility in facilities.items():
        contained_sites = []
        for cms_site in facility.get("cmssites", []):
            site_name = (
                cms_site.get("name")
                if isinstance(cms_site, dict) else cms_site
            )
            if site_name in site_names:
                contained_sites.append(f"site:{site_name}")
        records.append(CRICRecord(
            kind="facility",
            name=name,
            attrs={
                "country": facility.get("country", ""),
                "timezone": facility.get("timezone", ""),
                "fullname": facility.get("fullname", ""),
                "state": facility.get("state", ""),
            },
            contains_ids=tuple(contained_sites),
        ))

    for name, site in sites.items():
        tier_int = site.get("tier_level", 0)
        tier_str = _TIER_MAP.get(tier_int, f"T{tier_int}")
        compute_units_for_site = site.get("computeunits", {})
        cu_ids = []
        if isinstance(compute_units_for_site, dict):
            cu_ids = [f"ce:{cu_name}" for cu_name in compute_units_for_site]
        records.append(CRICRecord(
            kind="site",
            name=name,
            attrs={
                "tier_level": tier_str,
                "geographic_location": site.get("country", ""),
                "country_code": site.get("country_code", ""),
                "facility": site.get("facility", ""),
                "status": site.get("status", ""),
                "state": site.get("state", ""),
                "sitedb_title": site.get("sitedb_title") or name,
            },
            contains_ids=tuple(cu_ids),
        ))

    for name, storage_unit in storage_units.items():
        site_info = storage_unit.get("site", {})
        parent_site = (
            site_info.get("name") if isinstance(site_info, dict) else None
        )
        records.append(CRICRecord(
            kind="storage_endpoint",
            name=name,
            attrs={
                "type": storage_unit.get("type", ""),
                "pledged_cms": storage_unit.get("pledged-CMS", 0.0),
                "state": storage_unit.get("state", ""),
            },
            contained_by=(
                f"site:{parent_site}"
                if parent_site and parent_site in site_names else ""
            ),
        ))

    for name, compute_unit in compute_units.items():
        records.append(CRICRecord(
            kind="compute_endpoint",
            name=name,
            attrs={
                "corepower": compute_unit.get("corepower", 0.0),
                "pledged_cms": compute_unit.get("pledged_cms", 0.0),
                "potential_max": compute_unit.get("potential_max", 0.0),
                "promised": compute_unit.get("promised", 0.0),
                "state": compute_unit.get("state", ""),
            },
        ))

    records.extend(_operator_records(responsibilities, title_map))
    return records


def _operator_records(
    responsibilities: list[list[Any]],
    title_map: dict[str, str],
) -> Iterable[CRICRecord]:
    user_resps: dict[str, list[tuple[str, str]]] = {}
    seen_edges: set[tuple[str, str, str]] = set()
    for row in responsibilities:
        if len(row) < 3:
            continue
        username, site_title, role = row[0], row[1], row[2]
        if not username or not site_title:
            continue
        site_name = title_map.get(site_title)
        if site_name is None:
            continue
        edge_key = (username, site_name, role)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        user_resps.setdefault(username, []).append((f"site:{site_name}", role))
    for username, resps in user_resps.items():
        yield CRICRecord(
            kind="operator",
            name=username,
            attrs={},
            responsibilities=tuple(resps),
        )


def _node_fact(record: CRICRecord, revision: dict[str, Any]) -> NodeFact:
    attrs = dict(record.attrs)
    if record.kind == "site":
        label = attrs.get("sitedb_title", record.name)
        attrs.pop("sitedb_title", None)
        text = (
            f"{record.name} {label} {attrs.get('tier_level', '')} "
            f"{attrs.get('geographic_location', '')}"
        ).strip()
    elif record.kind == "facility":
        label = attrs.get("fullname") or record.name
        text = f"{record.name} {label} {attrs.get('country', '')}".strip()
    elif record.kind == "storage_endpoint":
        label = record.name
        text = f"{record.name} {attrs.get('type', '')} storage endpoint"
    elif record.kind == "compute_endpoint":
        label = record.name
        text = f"{record.name} compute endpoint"
    elif record.kind == "operator":
        label = record.name
        roles = " ".join(role for _, role in record.responsibilities)
        text = f"{record.name} operator {roles}".strip()
    else:
        label = record.name
        text = record.name
    attrs.update({"label": label, "name": record.name, "text": text})
    return NodeFact(
        node_id=record.node_id,
        subtype=record.kind,
        attrs=attrs,
        source_record_id={"name": record.name, "kind": record.kind},
        source_revision=revision,
    )


def _edge_facts(
    record: CRICRecord,
    revision: dict[str, Any],
) -> Iterator[EdgeFact]:
    for child_id in record.contains_ids:
        yield EdgeFact(
            src=record.node_id,
            dst=child_id,
            edge_type="contains",
            source_record_id={"name": record.name, "kind": record.kind},
            source_revision=revision,
        )
    if record.contained_by:
        yield EdgeFact(
            src=record.contained_by,
            dst=record.node_id,
            edge_type="contains",
            source_record_id={"name": record.name, "kind": record.kind},
            source_revision=revision,
        )
    for site_id, role in record.responsibilities:
        yield EdgeFact(
            src=record.node_id,
            dst=site_id,
            edge_type="responsible_for",
            attrs={"role": role},
            source_record_id={"name": record.name, "kind": record.kind},
            source_revision=revision,
        )


@dataclass(frozen=True)
class CRICCoreRecord:
    kind: Literal["service", "federation"]
    name: str
    attrs: dict[str, Any]
    cms_sites: tuple[str, ...]

    @property
    def node_id(self) -> str:
        if self.kind == "service":
            return f"svc:{self.name}"
        return f"fed:{self.name}"

    @property
    def subtype(self) -> str:
        if self.kind == "service":
            return "infrastructure_service"
        return "federation"


class CRICCoreSource:
    """Cache-backed CRIC core source with no network I/O."""

    name = "cric_core"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        services_path: str = "data/cric-core/services.json",
        rcsites_path: str = "data/cric-core/rcsites.json",
        federations_path: str = "data/cric-core/federations.json",
        base: str | None = None,
    ) -> None:
        self.services_path = services_path
        self.rcsites_path = rcsites_path
        self.federations_path = federations_path
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"cache_paths": self.cache_paths},
            emit_targets=CRICCoreSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (
            self.services_path,
            self.rcsites_path,
            self.federations_path,
        )

    def preflight(self, mode: str = "live") -> PreflightResult:
        missing = [
            str(resolve_repo_path(p, base=self.base))
            for p in self.cache_paths
            if not resolve_repo_path(p, base=self.base).is_file()
        ]
        if missing:
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="cache",
                required=True,
                cache_path=", ".join(missing),
                reason="one or more CRIC core cache files are missing",
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
            reason="local CRIC core cache present",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        records = self._records()
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _core_node_fact(record, revision)
            for record in records:
                yield from _core_edge_facts(record, revision)

        return ConnectorRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="cache",
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason="local CRIC core cache used",
            ),
        )

    def _records(self) -> list[CRICCoreRecord]:
        return _build_core_records(
            services=load_json(self.services_path, base=self.base),
            rcsites=load_json(self.rcsites_path, base=self.base),
            federations=load_json(self.federations_path, base=self.base),
        )


def _build_rcsite_to_cms_sites(
    rcsites: dict[str, Any],
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for rcsite_name, rcsite in rcsites.items():
        cms_sites = [
            site["name"]
            for site in rcsite.get("sites", [])
            if isinstance(site, dict) and site.get("vo_name") == "cms"
        ]
        if cms_sites:
            mapping[rcsite_name] = cms_sites
    return mapping


def _extract_latest_cms_pledge(pledges: dict[str, Any]) -> dict[str, Any]:
    if not pledges:
        return {}
    latest_year = max(pledges)
    quarters = pledges[latest_year]
    latest_quarter = max(quarters)
    cms = quarters[latest_quarter].get("cms", {})
    if not cms:
        return {}
    return {
        "pledge_cpu": cms.get("CPU", 0),
        "pledge_disk": cms.get("Disk", 0),
        "pledge_year": latest_year,
        "pledge_quarter": latest_quarter,
    }


def _build_core_records(
    *,
    services: dict[str, Any],
    rcsites: dict[str, Any],
    federations: dict[str, Any],
) -> list[CRICCoreRecord]:
    rcsite_to_cms = _build_rcsite_to_cms_sites(rcsites)
    records: list[CRICCoreRecord] = []
    for name, service in services.items():
        rcsite_name = service.get("rcsite", "")
        records.append(CRICCoreRecord(
            kind="service",
            name=name,
            attrs={
                "service_type": service.get("type", ""),
                "flavour": service.get("flavour") or "",
                "endpoint": service.get("endpoint", ""),
                "is_monitored": service.get("is_monitored", False),
                "rcsite": rcsite_name,
            },
            cms_sites=tuple(rcsite_to_cms.get(rcsite_name, [])),
        ))
    for name, federation in federations.items():
        pledges = federation.get("pledges", {})
        vos = federation.get("vos", [])
        has_cms = "cms" in vos or any(
            "cms" in vo_data
            for quarters in pledges.values()
            for vo_data in quarters.values()
        )
        if not has_cms:
            continue
        cms_sites: list[str] = []
        for rcsite_name in federation.get("rcsites", []):
            cms_sites.extend(rcsite_to_cms.get(rcsite_name, []))
        records.append(CRICCoreRecord(
            kind="federation",
            name=name,
            attrs={
                "accounting_name": federation.get("accounting_name", ""),
                "tier_level": federation.get("tier_level"),
                "country": federation.get("country", ""),
                "infrastructure": federation.get("infrastructure", ""),
                **_extract_latest_cms_pledge(pledges),
            },
            cms_sites=tuple(cms_sites),
        ))
    return records


def _core_node_fact(
    record: CRICCoreRecord,
    revision: dict[str, Any],
) -> NodeFact:
    attrs = dict(record.attrs)
    if record.kind == "service":
        label = record.name
        text = (
            f"{record.name} {attrs.get('service_type', '')} "
            f"{attrs.get('flavour', '')} infrastructure service"
        ).strip()
    else:
        label = attrs.get("accounting_name") or record.name
        text = (
            f"{record.name} {attrs.get('accounting_name', '')} "
            f"{attrs.get('country', '')} tier {attrs.get('tier_level', '')} "
            "federation"
        ).strip()
    attrs.update({"label": label, "name": record.name, "text": text})
    return NodeFact(
        node_id=record.node_id,
        subtype=record.subtype,
        attrs=attrs,
        source_record_id={"name": record.name, "kind": record.kind},
        source_revision=revision,
    )


def _core_edge_facts(
    record: CRICCoreRecord,
    revision: dict[str, Any],
) -> Iterator[EdgeFact]:
    if record.kind == "service":
        for cms_site in record.cms_sites:
            yield EdgeFact(
                src=f"site:{cms_site}",
                dst=record.node_id,
                edge_type="contains",
                source_record_id={
                    "name": record.name,
                    "kind": record.kind,
                },
                source_revision=revision,
            )
    else:
        for cms_site in record.cms_sites:
            yield EdgeFact(
                src=f"site:{cms_site}",
                dst=record.node_id,
                edge_type="member_of",
                source_record_id={
                    "name": record.name,
                    "kind": record.kind,
                },
                source_revision=revision,
            )
