"""Type-aware projection alias backend.

Provenance: ported from ``cms/cms_sources/alias.py`` (192 LOC,
okg-deployments ``main@f33a9c4``). Changes: class de-CMS-ified
(``CMSProjectionAliasBackend`` -> :class:`ProjectionAliasBackend`);
the entity-type/prefix tables (which mirror the packaged extraction
rules' ``cms_*`` id types), the backend ``name`` (``cms_projection``,
kept for cutover parity with the comp-ops instance's alias-resolver
config), and SQL are unchanged. Matching differs from the port source
in two ways (circleback review, see
``pact/changes/circleback-fixes/notes-enrichment.md``):

- Dataset alias matching is case-sensitive: DBS dataset paths are
  case-sensitive identifiers, and case-folding collapsed case-distinct
  canonical ids (``/A/B/RAW`` vs ``/a/b/raw``) into whichever loaded
  last. Sites, releases, JIRA keys, hostnames, and endpoints stay
  case-insensitive — for those types case is transcription noise.
- Dataset misses are no longer cached: the old negative cache grew
  unboundedly with every unresolved mention, and memoized nothing (the
  fallback lookup hit the same index).

Deployment wiring (alias_resolver block)::

    alias_resolver:
      on_empty: drop
      backends:
        - class: archi.enrichment.alias.ProjectionAliasBackend
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Optional

import psycopg

from okg.substrate.alias.protocol import AliasMatch
from okg.substrate.library.linkers import _chronos


class ProjectionAliasBackend:
    """Resolve extracted identifiers only to compatible live nodes.

    The default SQL alias backend is intentionally generic and ignores
    entity type. HEP deployments have ambiguous aliases such as
    site-like values that can point at storage endpoints, so
    projection-time linking needs a stricter backend to avoid narrowing
    violations.
    """

    name = "cms_projection"
    accept_threshold = 1.0

    _ENTITY_PREFIXES = {
        "cms_site": ("site",),
        "cmssw_release": ("cmssw_release",),
        "cms_jira_key": ("jira",),
        "global_tag": ("global_tag",),
        "cms_dataset": ("dataset",),
        "cms_run_number": ("run",),
        "cms_workflow": ("workflow",),
    }
    _PREFIX_ENTITY_TYPES = {
        prefix: entity_type
        for entity_type, prefixes in _ENTITY_PREFIXES.items()
        for prefix in prefixes
    }
    _SUBTYPE_ENTITY_TYPES = {
        "site": "cms_site",
        "cmssw_release": "cmssw_release",
        "jira_issue": "cms_jira_key",
        "global_tag": "global_tag",
        "dataset": "cms_dataset",
        "run": "cms_run_number",
        "workflow": "cms_workflow",
    }

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn
        self._direct_by_type: dict[str, dict[str, str]] = {
            entity_type: {}
            for entity_type in self._ENTITY_PREFIXES
        }
        self._service_by_endpoint: dict[str, str] = {}
        # Case-sensitive dataset-path -> node id index (DBS paths are
        # case-sensitive). Holds only live nodes; misses are not cached.
        self._dataset_by_value: dict[str, str] = {}
        self._dataset_index_loaded = False
        self._load()

    def match(
        self,
        needle: str,
        *,
        entity_type: Optional[str] = None,
    ) -> list[AliasMatch]:
        value = needle.strip()
        if not value or entity_type is None:
            return []
        if entity_type == "cms_hostname":
            canonical = self._service_by_endpoint.get(_norm(value))
            if canonical is None:
                canonical = self._service_by_endpoint.get(
                    _norm(_host_without_port(value))
                )
            return _match(canonical, "cms_hostname")
        if entity_type == "cms_dataset":
            return _match(self._resolve_dataset(value), "cms_dataset")
        if entity_type not in self._ENTITY_PREFIXES:
            return []
        key = _norm(value)
        canonical = self._direct_by_type[entity_type].get(key)
        return _match(canonical, entity_type)

    def _load(self) -> None:
        endpoint_candidates: defaultdict[str, set[str]] = defaultdict(set)
        with self._conn.cursor() as cur:
            rows = _chronos.query(
                cur,
                """
                SELECT node_id, subtype, attrs
                  FROM okg.graph_nodes
                 WHERE subtype = ANY(:subtypes)
                 ORDER BY node_id
                """,
                {
                    "subtypes": [
                        "site",
                        "cmssw_release",
                        "jira_issue",
                        "infrastructure_service",
                        "global_tag",
                        "run",
                        "workflow",
                    ],
                },
            )
            for row in rows:
                node_id = str(row["node_id"])
                subtype = str(row["subtype"])
                attrs = row["attrs"]
                if subtype == "infrastructure_service":
                    self._index_service_endpoint(
                        endpoint_candidates,
                        node_id,
                        attrs,
                    )
                    continue
                entity_type = self._SUBTYPE_ENTITY_TYPES.get(subtype)
                prefix, _, value = node_id.partition(":")
                if entity_type is None or not value:
                    continue
                if prefix not in self._ENTITY_PREFIXES[entity_type]:
                    continue
                self._direct_by_type[entity_type][_norm(value)] = node_id
            self._service_by_endpoint = {
                endpoint: next(iter(nodes))
                for endpoint, nodes in endpoint_candidates.items()
                if len(nodes) == 1
            }

    def _resolve_dataset(self, value: str) -> str | None:
        self._load_dataset_index()
        # Exact-case lookup: DBS dataset paths are case-sensitive, so
        # /A/B/RAW and /a/b/raw are distinct canonical ids.
        canonical = self._dataset_by_value.get(value)
        if canonical is not None:
            return canonical
        return self._dataset_by_value.get(value.removeprefix("dataset:"))

    def _load_dataset_index(self) -> None:
        if self._dataset_index_loaded:
            return
        with self._conn.cursor() as cur:
            rows = _chronos.query(
                cur,
                """
                SELECT node_id
                  FROM okg.graph_nodes
                 WHERE subtype = 'dataset'
                """,
            )
        for row in rows:
            node_id = str(row["node_id"])
            prefix, _, value = node_id.partition(":")
            if prefix == "dataset" and value:
                self._dataset_by_value[value.strip()] = node_id
        self._dataset_index_loaded = True

    def _index_service_endpoint(
        self,
        endpoint_candidates: defaultdict[str, set[str]],
        node_id: str,
        attrs: object,
    ) -> None:
        if not isinstance(attrs, Mapping):
            return
        endpoint = attrs.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            return
        endpoint_candidates[_norm(endpoint)].add(node_id)
        endpoint_candidates[_norm(_host_without_port(endpoint))].add(node_id)


def _host_without_port(value: str) -> str:
    if value.count(":") == 1:
        host, port = value.rsplit(":", 1)
        if port.isdigit():
            return host
    return value


def _norm(value: str) -> str:
    return value.strip().lower()


def _match(canonical: str | None, method: str) -> list[AliasMatch]:
    if canonical is None:
        return []
    return [AliasMatch(canonical=canonical, similarity=1.0, method=method)]
