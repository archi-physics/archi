"""SITECONF site-configuration source (GitLab group crawl).

Ported from okg-deployments ``cms/cms_sources/siteconf.py`` (442 LOC,
``SITECONFSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes:

- The CMS constants are constructor parameters with the cms values as
  defaults: ``base_url`` (``https://gitlab.cern.ch``), ``group_id``
  (4099 — the CMS SITECONF GitLab group), ``token_env``
  (``CERN_GITLAB_TOKEN`` with alias ``GITLAB_CERN_TOKEN``), and
  ``sites_path`` (CRIC sites cache used to gate ``site contains
  site_config`` edges).
- Cache reads go through :mod:`archi.auth.cache` with an explicit
  ``base`` parameter.

Kept-behavior note: as in the original, ``run()`` calls
``_known_sites(sites_path)`` unconditionally — a missing CRIC sites
cache raises ``FileNotFoundError`` even in fixture mode. Credentials
are env-var references only; the fetch itself needs a live token.

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``site_config`` and ``site``
ship in ``archi/schemas/operations.yaml``. ::

    siteconf:
      module: archi.sources.siteconf
      class: SITECONFSource
      ownership_id: <instance>.siteconf
      admission_policy:
        producer_id: <instance>.siteconf
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: siteconf
        output_signature:
          nodes:
            - {subtype: site_config}
          edges:
            - {src_subtype: site, edge_type: contains, dst_subtype: site_config}
        output_scope_summary:
          summary: per-site SITECONF configuration records from CERN GitLab
          nodes: [site_config]
          edges:
            - site contains site_config
      source_class: discovery_crawl
      record_identity_kind: remote_id
      record_identity_fields: [site_name]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [CERN_GITLAB_TOKEN]
      credential_aliases:
        CERN_GITLAB_TOKEN: [GITLAB_CERN_TOKEN]
      required_for_baseline: true
      params:
        # cms defaults
        base_url: https://gitlab.cern.ch
        group_id: 4099
        token_env: CERN_GITLAB_TOKEN
        sites_path: data/cric/sites.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import quote_plus

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
)
from okg.substrate.library.sources.base import SourceRun
from okg.deployment import MutableApiProbe
from okg.deployment import credential_preflight

from archi.auth.cache import load_json

_DEFAULT_BASE_URL = "https://gitlab.cern.ch"
_DEFAULT_GROUP_ID = 4099  # CMS SITECONF group on CERN GitLab


@dataclass(frozen=True)
class SiteConfRecord:
    site_name: str
    storage_protocols: tuple[str, ...] = ()
    se_endpoints: tuple[str, ...] = ()
    ce_endpoints: tuple[str, ...] = ()
    local_stage_out: str = ""
    fallback_stage_out: str = ""
    frontier_config: str = ""
    raw_storage_json: str = ""

    @property
    def node_id(self) -> str:
        return f"siteconf:{self.site_name}"


class SITECONFSource:
    """Fetch SITECONF project data from a GitLab group."""

    name = "siteconf"
    profile = "discovery_crawl"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        token_env: str = "CERN_GITLAB_TOKEN",
        aliases: list[str] | None = None,
        group_id: int = _DEFAULT_GROUP_ID,
        required: bool = True,
        max_projects: int | None = None,
        sites_path: str = "data/cric/sites.json",
        records: list[SiteConfRecord] | None = None,
        base: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.aliases = tuple(aliases or ("GITLAB_CERN_TOKEN",))
        self.group_id = group_id
        self.required = required
        self.max_projects = max_projects
        self.sites_path = sites_path
        self._records = records
        self.base = base
        self.change_probe = MutableApiProbe(
            version_fn=self._probe_version,
            config={
                "base_url": self.base_url,
                "token_env": self.token_env,
                "aliases": self.aliases,
                "group_id": self.group_id,
                "max_projects": self.max_projects,
                "sites_path": self.sites_path,
            },
            emit_targets=SITECONFSource,
        )

    def _probe_version(self) -> str:
        if self._records is None:
            return _checked_at()
        payload = json.dumps(
            [record.__dict__ for record in self._records],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def preflight(self, mode: str = "live") -> PreflightResult:
        if self._records is not None:
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="fixture",
                required=self.required,
                record_count=len(self._records),
                reason="in-memory SITECONF records supplied",
                checked_at=_checked_at(),
            )
        env_result = credential_preflight(
            self.name,
            self.token_env,
            aliases=self.aliases,
            required=self.required,
            mode=mode,
        )
        if env_result.status != "ok" or mode != "live":
            return env_result
        return self._token_probe(env_result)

    def _token_probe(
        self,
        env_result: PreflightResult,
    ) -> PreflightResult:
        import requests

        token = _env_value(self.token_env, self.aliases)
        try:
            resp = requests.get(
                f"{self.base_url}/api/v4/user",
                headers={"PRIVATE-TOKEN": token},
                timeout=20,
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode="live",
                required=self.required,
                credential_refs=env_result.credential_refs,
                alias_refs=env_result.alias_refs,
                endpoint=f"{self.base_url}/api/v4/user",
                reason=f"GitLab token probe failed: {type(exc).__name__}",
                checked_at=_checked_at(),
            )
        if resp.status_code in {401, 403}:
            return PreflightResult(
                source_name=self.name,
                status="auth_failed",
                mode="live",
                required=self.required,
                credential_refs=env_result.credential_refs,
                alias_refs=env_result.alias_refs,
                endpoint=f"{self.base_url}/api/v4/user",
                reason="GitLab token was rejected",
                checked_at=_checked_at(),
            )
        if resp.status_code >= 400:
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode="live",
                required=self.required,
                credential_refs=env_result.credential_refs,
                alias_refs=env_result.alias_refs,
                endpoint=f"{self.base_url}/api/v4/user",
                reason=(
                    f"GitLab token probe returned HTTP {resp.status_code}"
                ),
                checked_at=_checked_at(),
            )
        return self._group_probe(env_result)

    def _group_probe(
        self,
        env_result: PreflightResult,
    ) -> PreflightResult:
        """Probe group visibility, not just token validity.

        ``/api/v4/user`` accepts any live token; a token without group
        visibility then gets an HTTP-200 *empty* project list from the
        group crawl, which used to become a healthy completed scope
        over zero records (retract-all).
        """
        import requests

        token = _env_value(self.token_env, self.aliases)
        endpoint = f"{self.base_url}/api/v4/groups/{self.group_id}/projects"
        try:
            # Mirror the crawl's listing exactly (include_subgroups):
            # a group whose projects live only in subgroups must not
            # get a false-negative preflight that blocks run().
            resp = requests.get(
                endpoint,
                params={"include_subgroups": "true", "per_page": 1},
                headers={"PRIVATE-TOKEN": token},
                timeout=20,
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode="live",
                required=self.required,
                credential_refs=env_result.credential_refs,
                alias_refs=env_result.alias_refs,
                endpoint=endpoint,
                reason=f"GitLab group probe failed: {type(exc).__name__}",
                checked_at=_checked_at(),
            )
        if resp.status_code in {401, 403}:
            return PreflightResult(
                source_name=self.name,
                status="auth_failed",
                mode="live",
                required=self.required,
                credential_refs=env_result.credential_refs,
                alias_refs=env_result.alias_refs,
                endpoint=endpoint,
                reason=(
                    f"GitLab token cannot access group {self.group_id}"
                ),
                checked_at=_checked_at(),
            )
        payload: Any = None
        if resp.status_code < 400:
            try:
                payload = resp.json()
            except ValueError:
                payload = None
        if (
            resp.status_code >= 400
            or not isinstance(payload, list)
            or not payload
        ):
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode="live",
                required=self.required,
                credential_refs=env_result.credential_refs,
                alias_refs=env_result.alias_refs,
                endpoint=endpoint,
                reason=(
                    f"GitLab group {self.group_id} project list is not "
                    "visible to this token (HTTP "
                    f"{resp.status_code}, empty or non-list payload)"
                ),
                checked_at=_checked_at(),
            )
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="live",
            required=self.required,
            credential_refs=env_result.credential_refs,
            alias_refs=env_result.alias_refs,
            endpoint=endpoint,
            reason="GitLab token accepted and group projects visible",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> SourceRun:
        preflight = self.preflight(mode="live")
        if preflight.status != "ok":
            return SourceRun(
                facts=[],
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status=preflight.status,
                    mode=preflight.mode,
                    reason=preflight.reason,
                    credential_refs=preflight.credential_refs,
                ),
            )

        if self._records is not None:
            records = self._records
            truncated = False
        else:
            records, n_projects, truncated = self._fetch()
            if not records:
                # An HTTP-200 empty project list (token without group
                # visibility) or a crawl that parsed zero site configs
                # must never become a healthy completed scope over zero
                # records — that would retract every site_config.
                return SourceRun(
                    facts=[],
                    completed_scope=False,
                    run_mode=mode,
                    health=ConnectorHealth(
                        status="endpoint_failed",
                        mode="live",
                        credential_refs=(self.token_env,),
                        record_count=0,
                        endpoint=self.base_url,
                        reason=(
                            (
                                f"GitLab group {self.group_id} returned "
                                "an empty project list; the token may "
                                "lack group visibility"
                            )
                            if n_projects == 0
                            else (
                                f"{n_projects} GitLab projects listed "
                                "but zero SITECONF records parsed"
                            )
                        )
                        + "; no scope claimed",
                    ),
                )
        record_hash = _records_hash(records)
        revision = {
            "run_id": run_id,
            "content_hash": record_hash,
            "n_records": len(records),
            "group_id": self.group_id,
        }
        known_sites = _known_sites(self.sites_path, base=self.base)

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _node_fact(record, revision)
                if record.site_name in known_sites:
                    yield EdgeFact(
                        src=f"site:{record.site_name}",
                        dst=record.node_id,
                        edge_type="contains",
                        provenance="authoritative",
                        source_record_id={"site_name": record.site_name},
                        source_revision=revision,
                    )

        reason = "SITECONF records fetched from GitLab"
        if truncated:
            # A truncated project list is a sliding window, not the
            # full group scope.
            reason += (
                f"; max_projects={self.max_projects} truncated the "
                "project list; no complete scope claimed"
            )
        return SourceRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and not truncated
            ),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="live" if self._records is None else "fixture",
                credential_refs=(
                    (self.token_env,) if self._records is None else ()
                ),
                record_count=len(records),
                content_hash=record_hash,
                reason=reason,
            ),
        )

    def _fetch(self) -> tuple[list[SiteConfRecord], int, bool]:
        """(records, listed_project_count, cap_truncated)."""
        import requests

        token = _env_value(self.token_env, self.aliases)
        headers = {"PRIVATE-TOKEN": token} if token else {}
        session = requests.Session()
        projects, truncated = self._list_projects(session, headers)
        records: list[SiteConfRecord] = []
        for project in projects:
            site_name = _pg_text(str(project.get("path") or ""))
            project_id = project.get("id")
            if not project_id or not re.match(r"T\d_", site_name):
                continue
            record = self._fetch_site_config(
                session,
                site_name=site_name,
                project_id=str(project_id),
                headers=headers,
            )
            if record is not None:
                records.append(record)
        return records, len(projects), truncated

    def _list_projects(
        self,
        session: Any,
        headers: dict[str, str],
    ) -> tuple[list[dict[str, Any]], bool]:
        projects: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = session.get(
                f"{self.base_url}/api/v4/groups/{self.group_id}/projects",
                params={
                    "include_subgroups": "true",
                    "per_page": 100,
                    "page": page,
                },
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            projects.extend(item for item in batch if isinstance(item, dict))
            if (
                self.max_projects is not None
                and len(projects) >= self.max_projects
            ):
                # The cap fired: pages/items beyond it were not listed.
                return projects[:self.max_projects], True
            page += 1
        return projects, False

    def _fetch_site_config(
        self,
        session: Any,
        *,
        site_name: str,
        project_id: str,
        headers: dict[str, str],
    ) -> SiteConfRecord | None:
        storage_protocols: set[str] = set()
        se_endpoints: set[str] = set()
        ce_endpoints: set[str] = set()
        local_stage_out = ""
        fallback_stage_out = ""
        frontier_config = ""
        raw_storage_json = ""

        storage_resp = _fetch_raw(
            session,
            self.base_url,
            project_id,
            "storage.json",
            headers,
        )
        if storage_resp:
            raw_storage_json = _pg_text(storage_resp)
            _parse_storage_json(
                raw_storage_json,
                storage_protocols=storage_protocols,
                se_endpoints=se_endpoints,
            )

        slc_resp = _fetch_raw(
            session,
            self.base_url,
            project_id,
            "JobConfig/site-local-config.xml",
            headers,
        )
        if slc_resp:
            for match in re.finditer(
                r'local-stage-out[^"]*"([^"]+)"', slc_resp
            ):
                local_stage_out = _pg_text(match.group(1))
            for match in re.finditer(
                r'fallback-stage-out[^"]*"([^"]+)"', slc_resp
            ):
                fallback_stage_out = _pg_text(match.group(1))
            for match in re.finditer(
                r'frontier-connect[^"]*"([^"]+)"', slc_resp
            ):
                frontier_config = _pg_text(match.group(1))
            for match in re.finditer(r'se-name="([^"]+)"', slc_resp):
                se_endpoints.add(_pg_text(match.group(1)))
            for match in re.finditer(r'ce-name="([^"]+)"', slc_resp):
                ce_endpoints.add(_pg_text(match.group(1)))

        if not storage_protocols and not se_endpoints and not raw_storage_json:
            return None
        return SiteConfRecord(
            site_name=site_name,
            storage_protocols=tuple(sorted(storage_protocols)),
            se_endpoints=tuple(sorted(se_endpoints)),
            ce_endpoints=tuple(sorted(ce_endpoints)),
            local_stage_out=local_stage_out,
            fallback_stage_out=fallback_stage_out,
            frontier_config=frontier_config,
            raw_storage_json=raw_storage_json,
        )


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_raw(
    session: Any,
    base_url: str,
    project_id: str,
    path: str,
    headers: dict[str, str],
) -> str:
    resp = session.get(
        f"{base_url}/api/v4/projects/{project_id}"
        f"/repository/files/{quote_plus(path)}/raw",
        params={"ref": "master"},
        headers=headers,
        timeout=20,
    )
    if resp.status_code == 404:
        return ""
    resp.raise_for_status()
    return resp.text


def _parse_storage_json(
    raw: str,
    *,
    storage_protocols: set[str],
    se_endpoints: set[str],
) -> None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return
    entries = parsed if isinstance(parsed, list) else [parsed]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        protocol = entry.get("protocol")
        if protocol:
            storage_protocols.add(_pg_text(str(protocol)))
        endpoint = entry.get("se") or entry.get("endpoint")
        if endpoint:
            se_endpoints.add(_pg_text(str(endpoint)))


def _node_fact(
    record: SiteConfRecord,
    revision: dict[str, Any],
) -> NodeFact:
    text_parts = [f"SITECONF for {record.site_name}"]
    if record.storage_protocols:
        text_parts.append(
            f"Protocols: {', '.join(record.storage_protocols)}"
        )
    if record.se_endpoints:
        text_parts.append(f"SE: {', '.join(record.se_endpoints[:5])}")
    return NodeFact(
        node_id=record.node_id,
        subtype="site_config",
        attrs={
            "label": f"SITECONF {record.site_name}",
            "site_name": record.site_name,
            "storage_protocols": list(record.storage_protocols),
            "se_endpoints": list(record.se_endpoints),
            "ce_endpoints": list(record.ce_endpoints),
            "local_stage_out": record.local_stage_out,
            "fallback_stage_out": record.fallback_stage_out,
            "frontier_config": record.frontier_config,
            "raw_storage_json": record.raw_storage_json,
            "text": "\n".join(text_parts),
        },
        source_record_id={"site_name": record.site_name},
        source_revision=revision,
    )


def _records_hash(records: list[SiteConfRecord]) -> str:
    payload = json.dumps(
        [record.__dict__ for record in records],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _known_sites(path: str, *, base: str | None = None) -> set[str]:
    payload = load_json(path, base=base)
    if isinstance(payload, dict):
        return {str(key) for key in payload}
    return set()


def _env_value(name: str, aliases: tuple[str, ...]) -> str:
    for ref in (name,) + aliases:
        value = os.environ.get(ref)
        if value:
            return value
    return ""


def _pg_text(text: str) -> str:
    return text.replace("\x00", " ")
