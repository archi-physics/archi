"""Cache-backed documentation sources with chunk/reference emission.

Rewritten from okg-deployments ``cms/cms_sources/docs.py`` (1,396 LOC)
at ``main@f33a9c4`` for the archi v3 package (req.w2.sources-parity).
Changes from the original:

- **Dropped** ``GitHubFileContentSource`` (removed upstream 2026-06-29
  as redundant with the code_repos family) together with the manifest
  parsing, repo-file classification, service-alias extraction, and
  repo-link machinery only it exercised (~700 LOC). Only that class
  produced records with ``record_kind == "github_file_content"`` or
  service aliases, so docsite/gitlab_docs/SSO emissions are unchanged;
  ``DocumentationRecord`` is trimmed to the fields the surviving
  producers set. Dropping it also removes the PyYAML dependency.
- **Renamed** ``CMSWebDocsSource`` -> :class:`SSOCookieDocsSource`: the
  class is generic-CERN (sitemap crawl through a CERN SSO cookie jar),
  not CMS-specific, once ``sitemap_url`` and ``cookie_file_env`` are
  required parameters instead of CMS defaults.
- Cache handling goes through :mod:`archi.auth.cache` (explicit
  ``base`` parameter instead of repo-layout assumptions); cookie
  handling goes through :mod:`archi.auth.cookies` (jar loading,
  offline freshness in preflight, login-bounce detection during the
  crawl — the original ingested SSO login pages it was bounced to).
- ``SSOCookieDocsSource.run`` reports a missing/unreadable cookie as
  ``auth_failed`` and any per-page fetch failure or login bounce as a
  partial crawl; neither may claim ``completed_scope`` — the original
  reported ``ok``/complete in both cases, which under
  ``missing_from_completed_scope`` deletion semantics would retract
  previously ingested pages. A bounce means landing on an SSO host (or
  a redirect to a login-looking body), never just ``/login`` in a
  page's own path or ``auth.cern.ch`` in its text.
- 2026-08 adversarial review (see
  ``pact/changes/circleback-fixes/notes-live.md``): a ``max_pages`` cap
  that actually truncates the sitemap frontier is surfaced in health
  and never claims ``completed_scope`` (the un-crawled pages would be
  retracted under ``missing_from_completed_scope``); non-HTML sitemap
  entries (PDFs/binaries, decided by response ``Content-Type``) are
  skipped and counted instead of being regex-stripped into mojibake
  pages — they are excluded from the source's scope *by design*, so a
  clean crawl that skipped them may still claim the scope (a
  previously mis-ingested binary is then correctly retracted).
- ``DocumentationSource.run`` reports ``cache_missing`` health when the
  records cache is absent instead of raising from ``load_json``.
- The reference-target caches (sites / releases / jira / services) are
  optional (default ``None`` -> no reference edges of that kind). As in
  the original, a *configured* sites/releases/services path whose file
  is missing raises; only unconfigured caches are skipped silently.
- The CMS project-key regex for issue references is generalized to any
  JIRA-style key; matches are still bounded by intersection with the
  keys actually present in the configured jira records cache.
- ``repo_base_url`` (default ``https://gitlab.cern.ch``) and
  ``chunker_name`` (default kept at ``cms_document_window_v1`` so the
  parity corpus does not churn at cutover) are parameters.

Registry-entry templates — INGEST-PROVEN 2026-08-11 on a scratch
instance (okg dev@21c5b8c3e); both docsite-style families use the
same class with different params. Same three prerequisites as
archi/sources/jira.py's template: (1) compose the ``person`` +
``extraction`` modules and copy ``archi/schemas/sources.yaml`` +
``archi/schemas/bridges/sources.yaml`` into the deployment (narrowings
outside ``schemas/bridges/`` are silently ignored and fail only at
ingest); (2) ``output_scope_summary`` must accompany
``output_signature``; (3) add the standard ``sync:`` block. ::

    docsite:
      module: archi.sources.docs
      class: DocumentationSource
      ownership_id: <instance>.docsite
      admission_policy:
        producer_id: <instance>.docsite
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: docsite
        output_signature:
          nodes:
            - {subtype: documentation_page}
            - {subtype: software_repository}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: software_repository, edge_type: contains, dst_subtype: documentation_page}
            - {src_subtype: documentation_page, edge_type: contains, dst_subtype: document_chunk}
            - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
            # Uncomment together with the matching params below:
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: cmssw_release}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: site}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: documentation pages, their repos, and text chunks from the records cache
          nodes: [documentation_page, software_repository, document_chunk]
          edges:
            - software_repository contains documentation_page
            - documentation_page contains document_chunk
            - document_chunk references jira_issue
            # Uncomment together with the matching params below:
            # - document_chunk references cmssw_release
            # - document_chunk references site
            # - document_chunk references infrastructure_service
      source_class: discovery_crawl
      record_identity_kind: scoped_locator
      record_identity_fields: [url]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: true
      params:
        source_name: docsite
        required: true
        records_path: data/docsite/records.json
        # Enables the document_chunk references jira_issue edges
        # declared above (as in bundles/cern-team/source-defaults/
        # docsite.yaml); drop it together with those edge lines.
        jira_records_path: data/jira/records.json
        # Optional reference-target caches — commented out on purpose.
        # WARNING: enabling any of them requires (a) uncommenting the
        # matching document_chunk references edges in output_signature
        # AND output_scope_summary above, and (b) the target subtype +
        # narrowing in the deployment schema: cmssw_release ships in
        # archi/schemas/operations.yaml with its narrowing in
        # archi/schemas/bridges/sources.yaml; site and
        # infrastructure_service ship in the catalogs port
        # (archi/schemas/operations.yaml + bridges/operations.yaml).
        # sites_path: data/cric/sites.json
        # releases_path: data/cmssw-releases/records.json
        # services_path: data/cric-core/services.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete

    gitlab_docs:
      module: archi.sources.docs
      class: DocumentationSource
      ownership_id: <instance>.gitlab-docs
      admission_policy:
        producer_id: <instance>.gitlab-docs
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: gitlab_docs
        output_signature:
          nodes:
            - {subtype: documentation_page}
            - {subtype: software_repository}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: software_repository, edge_type: contains, dst_subtype: documentation_page}
            - {src_subtype: documentation_page, edge_type: contains, dst_subtype: document_chunk}
            # Uncomment together with the matching reference-cache
            # params (same warning as the docsite entry above):
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: cmssw_release}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: site}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: GitLab-exported documentation pages, their repos, and text chunks from the records cache
          nodes: [documentation_page, software_repository, document_chunk]
          edges:
            - software_repository contains documentation_page
            - documentation_page contains document_chunk
            # Uncomment together with the matching reference-cache params:
            # - document_chunk references jira_issue
            # - document_chunk references cmssw_release
            # - document_chunk references site
            # - document_chunk references infrastructure_service
      source_class: discovery_crawl
      record_identity_kind: scoped_locator
      record_identity_fields: [project, path]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [CERN_GITLAB_TOKEN]
      credential_aliases:
        CERN_GITLAB_TOKEN: [GITLAB_CERN_TOKEN]
      required_for_baseline: true
      params:
        source_name: gitlab_docs
        required: true
        records_path: data/gitlab-docs/records.json
        # Optional reference-target caches: see the docsite entry above
        # for the warning; uncommenting one requires the matching edge
        # lines in output_signature AND output_scope_summary.
        # jira_records_path: data/jira/records.json
        # sites_path: data/cric/sites.json
        # releases_path: data/cmssw-releases/records.json
        # services_path: data/cric-core/services.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete

    cmsweb_docs:                        # cms instance of the renamed class
      module: archi.sources.docs
      class: SSOCookieDocsSource
      ownership_id: <instance>.cmsweb-docs
      admission_policy:
        producer_id: <instance>.cmsweb-docs
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: cmsweb_docs
        output_signature:
          nodes:
            - {subtype: documentation_page}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: documentation_page, edge_type: contains, dst_subtype: document_chunk}
            # Uncomment together with the matching reference-cache
            # params (same warning as the docsite entry above):
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: cmssw_release}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: site}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: SSO-crawled documentation pages and text chunks from the live sitemap
          nodes: [documentation_page, document_chunk]
          edges:
            - documentation_page contains document_chunk
            # Uncomment together with the matching reference-cache params:
            # - document_chunk references jira_issue
            # - document_chunk references cmssw_release
            # - document_chunk references site
            # - document_chunk references infrastructure_service
      source_class: discovery_crawl
      record_identity_kind: scoped_locator
      record_identity_fields: [url]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [CMS_HTTP_GROUP_DOCS_COOKIE_FILE]
      required_for_baseline: false
      params:
        source_name: cmsweb_docs
        sitemap_url: https://cms-http-group.docs.cern.ch/sitemap.xml
        cookie_file_env: CMS_HTTP_GROUP_DOCS_COOKIE_FILE
        # Optional reference-target caches: see the docsite entry above.
        # jira_records_path: data/jira/records.json
        # sites_path: data/cric/sites.json
        # releases_path: data/cmssw-releases/records.json
        # services_path: data/cric-core/services.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
)
from okg.substrate.library.sources.base import SourceRun

from archi.auth.cache import (
    cache_or_forced_live_change_probe,
    content_hash,
    content_hash_change_probe,
    load_json,
    resolve_repo_path,
)
from archi.auth.cookies import (
    check_cookie_file,
    load_cookie_jar_from_env,
    looks_like_login_page,
)

_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200
_SITE_RE = re.compile(r"\bT[0-3]_[A-Z]{2}_[A-Za-z0-9_]+(?:_[A-Za-z0-9]+)?\b")
_CMSSW_RE = re.compile(r"\bCMSSW_\d+_\d+_\d+(?:_[A-Za-z0-9_]+)?\b")
# Generalized from the original's hardcoded CMS project-key alternation;
# reference edges stay bounded by the known-key intersection below.
_ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9_]*-\d+\b")
_HOST_RE = re.compile(
    r"(?<![\w.@-])"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,}(?::\d{2,5})?"
    r"(?![\w.-])"
)

DEFAULT_REPO_BASE_URL = "https://gitlab.cern.ch"
DEFAULT_CHUNKER_NAME = "cms_document_window_v1"


@dataclass(frozen=True)
class DocumentationRecord:
    title: str
    url: str
    body: str
    site_name: str = ""
    source_repo: str = ""
    path: str = ""

    @property
    def node_id(self) -> str:
        if self.source_repo and self.path:
            return (
                f"documentation_page:{_slug(self.source_repo)}:{_slug(self.path)}"
            )
        return f"documentation_page:{_sha256(self.url)[:16]}"

    @property
    def source_record_id(self) -> dict[str, Any]:
        return {"url": self.url}


class DocumentationSource:
    """Cache-backed documentation source with chunk/reference emission."""

    name = "docsite"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        source_name: str = "docsite",
        records_path: str = "data/docsite/records.json",
        required: bool = True,
        sites_path: str | None = None,
        releases_path: str | None = None,
        jira_records_path: str | None = None,
        services_path: str | None = None,
        repo_base_url: str = DEFAULT_REPO_BASE_URL,
        chunker_name: str = DEFAULT_CHUNKER_NAME,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.records_path = records_path
        self.required = required
        self.sites_path = sites_path
        self.releases_path = releases_path
        self.jira_records_path = jira_records_path
        self.services_path = services_path
        self.repo_base_url = repo_base_url.rstrip("/")
        self.chunker_name = chunker_name
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"records_path": self.records_path},
            emit_targets=DocumentationSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        path = resolve_repo_path(self.records_path, base=self.base)
        if not path.is_file():
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="cache",
                required=self.required,
                cache_path=str(path),
                reason=f"{self.name} records cache file is missing",
                checked_at=_checked_at(),
            )
        records = self._records()
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=self.required,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason=f"local {self.name} documentation cache present",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> SourceRun:
        path = resolve_repo_path(self.records_path, base=self.base)
        if not path.is_file():
            return SourceRun(
                facts=(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="cache_missing",
                    mode="cache",
                    cache_path=str(path),
                    reason=(
                        f"{self.name} records cache is absent; "
                        "no facts emitted"
                    ),
                    checked_at=_checked_at(),
                ),
            )
        records = self._records()
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }
        targets = self._reference_targets()

        def _facts() -> Iterator[Any]:
            yield from _facts_for_records(
                records,
                revision,
                targets,
                repo_base_url=self.repo_base_url,
                chunker_name=self.chunker_name,
            )

        return SourceRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="cache",
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason=f"local {self.name} documentation cache used",
            ),
        )

    def _reference_targets(self) -> dict[str, Any]:
        return _reference_targets(
            sites_path=self.sites_path,
            releases_path=self.releases_path,
            jira_records_path=self.jira_records_path,
            services_path=self.services_path,
            base=self.base,
        )

    def _records(self) -> list[DocumentationRecord]:
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of docs"
            )
        return _records_from_payload(payload)


class SSOCookieDocsSource(DocumentationSource):
    """Live sitemap-driven docs source read through a CERN SSO cookie jar.

    Renamed from the cms original ``CMSWebDocsSource``; the sitemap URL
    and the cookie-file env-var reference are required parameters. The
    cookie file is Netscape/Mozilla format, produced out-of-band (see
    :mod:`archi.auth.cookies` for the acquisition contract); only its
    *path* travels through the environment variable named by
    ``cookie_file_env`` — never a credential value.

    Scope semantics: when ``max_pages`` actually truncates the sitemap
    frontier the run emits what it crawled but never claims
    ``completed_scope`` (the un-crawled pages would otherwise be
    retracted under ``missing_from_completed_scope``). Non-HTML sitemap
    entries (by response Content-Type; a missing header is treated as
    HTML) are skipped and counted in health, and are excluded from the
    source's scope by design — a clean crawl that skipped them may
    still claim the scope.
    """

    name = "sso_docs"
    profile = "discovery_crawl"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        sitemap_url: str,
        cookie_file_env: str,
        source_name: str = "sso_docs",
        max_pages: int | None = None,
        cookie_max_age_hours: float | None = None,
        sites_path: str | None = None,
        releases_path: str | None = None,
        jira_records_path: str | None = None,
        services_path: str | None = None,
        repo_base_url: str = DEFAULT_REPO_BASE_URL,
        chunker_name: str = DEFAULT_CHUNKER_NAME,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.sitemap_url = sitemap_url
        self.cookie_file_env = cookie_file_env
        self.max_pages = max_pages
        self.cookie_max_age_hours = cookie_max_age_hours
        self.records_path = ""
        self.required = False
        self.sites_path = sites_path
        self.releases_path = releases_path
        self.jira_records_path = jira_records_path
        self.services_path = services_path
        self.repo_base_url = repo_base_url.rstrip("/")
        self.chunker_name = chunker_name
        self.base = base
        self.change_probe = cache_or_forced_live_change_probe(
            cache_paths=self.cache_paths,
            config={
                "source_name": self.name,
                "sitemap_url": self.sitemap_url,
                "cookie_file_env": self.cookie_file_env,
                "max_pages": self.max_pages,
            },
            emit_targets=SSOCookieDocsSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return ()

    def preflight(self, mode: str = "live") -> PreflightResult:
        cookie_file = os.environ.get(self.cookie_file_env, "")
        if not cookie_file or not Path(cookie_file).is_file():
            return PreflightResult(
                source_name=self.name,
                status="missing_credential",
                mode="live",
                required=False,
                credential_refs=(self.cookie_file_env,),
                reason=(
                    f"{self.cookie_file_env} file is not set "
                    "or does not exist"
                ),
                checked_at=_checked_at(),
            )
        max_age = (
            timedelta(hours=self.cookie_max_age_hours)
            if self.cookie_max_age_hours is not None
            else None
        )
        status = check_cookie_file(cookie_file, max_age=max_age)
        return PreflightResult(
            source_name=self.name,
            status="ok" if status.fresh else "auth_failed",
            mode="live",
            required=False,
            credential_refs=(self.cookie_file_env,),
            cache_path=cookie_file,
            reason=f"SSO cookie file: {status.reason}",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> SourceRun:
        session = self._cookie_session()
        if session is None:
            # A missing/unreadable cookie is an auth failure, not an
            # empty-but-complete crawl: with completed_scope=True the
            # registry's missing_from_completed_scope semantics would
            # retract every previously ingested page.
            return SourceRun(
                facts=(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="auth_failed",
                    mode="live",
                    credential_refs=(self.cookie_file_env,),
                    record_count=0,
                    content_hash=None,
                    reason=(
                        f"SSO cookie file referenced by "
                        f"{self.cookie_file_env} is missing or unreadable; "
                        "no pages fetched and no complete scope claimed"
                    ),
                    checked_at=_checked_at(),
                ),
            )
        crawl = self._crawl(session)
        records = list(crawl.records)
        record_hash = _records_hash(records)
        revision = {
            "run_id": run_id,
            "content_hash": record_hash,
            "n_records": len(records),
            "sitemap_url": self.sitemap_url,
        }
        targets = self._reference_targets()

        def _facts() -> Iterator[Any]:
            yield from _facts_for_records(
                records,
                revision,
                targets,
                repo_base_url=self.repo_base_url,
                chunker_name=self.chunker_name,
            )

        truncation_note = ""
        if crawl.truncated:
            truncation_note = (
                f"; sitemap frontier truncated by max_pages="
                f"{self.max_pages} ({crawl.total_urls}/"
                f"{crawl.sitemap_total} sitemap URLs crawled)"
            )
        skip_note = ""
        if crawl.skipped_non_html:
            skip_note = (
                f"; skipped {len(crawl.skipped_non_html)} non-HTML "
                "sitemap entries (excluded from scope by design)"
            )
        if crawl.failed_urls:
            # A partially failed crawl must never claim a complete scope
            # (missing_from_completed_scope would retract the failed
            # pages' records); emit what succeeded and report the rest.
            samples = ", ".join(crawl.failed_urls[:3])
            return SourceRun(
                facts=_facts(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="endpoint_failed",
                    mode="live",
                    credential_refs=(self.cookie_file_env,),
                    record_count=len(records),
                    content_hash=record_hash,
                    reason=(
                        f"partial crawl: {len(crawl.failed_urls)}/"
                        f"{crawl.total_urls} sitemap pages failed (fetch "
                        f"error or SSO login bounce), e.g. {samples}"
                        f"{truncation_note}{skip_note}; "
                        "no complete scope claimed"
                    ),
                ),
            )
        if crawl.truncated:
            # max_pages actually cut the sitemap frontier: the
            # un-crawled pages would be retracted under
            # missing_from_completed_scope if this run claimed a
            # complete scope. Emit what was crawled and claim nothing.
            return SourceRun(
                facts=_facts(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="ok",
                    mode="live",
                    credential_refs=(self.cookie_file_env,),
                    record_count=len(records),
                    content_hash=record_hash,
                    reason=(
                        "SSO docs fetched through CERN SSO cookie"
                        f"{truncation_note}{skip_note}; "
                        "no complete scope claimed"
                    ),
                ),
            )
        return SourceRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="live",
                credential_refs=(self.cookie_file_env,),
                record_count=len(records),
                content_hash=record_hash,
                reason=(
                    "SSO docs fetched through CERN SSO cookie"
                    f"{skip_note}"
                ),
            ),
        )

    def _cookie_session(self) -> requests.Session | None:
        """Session carrying the SSO cookie jar; ``None`` when the cookie
        env var is unset or the file is missing/unparseable."""
        jar = load_cookie_jar_from_env(self.cookie_file_env)
        if jar is None:
            return None
        session = requests.Session()
        session.cookies = jar
        return session

    def _crawl(self, session: requests.Session) -> _CrawlOutcome:
        sitemap = session.get(self.sitemap_url, timeout=30)
        sitemap.raise_for_status()
        if _login_bounce(
            self.sitemap_url,
            str(getattr(sitemap, "url", "") or ""),
            getattr(sitemap, "text", "") or "",
        ):
            raise RuntimeError(
                f"{self.name}: sitemap fetch was bounced to a CERN SSO "
                f"login page; refresh the cookie file referenced by "
                f"{self.cookie_file_env}"
            )
        all_urls = _parse_sitemap(sitemap.content)
        urls = all_urls
        truncated = False
        if self.max_pages is not None and len(all_urls) > self.max_pages:
            urls = all_urls[: self.max_pages]
            truncated = True
        records: list[DocumentationRecord] = []
        failed_urls: list[str] = []
        skipped_non_html: list[str] = []
        for url in urls:
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
            except Exception:  # noqa: BLE001
                failed_urls.append(url)
                continue
            content_type = _content_type(resp)
            if content_type and not _is_html_content_type(content_type):
                # PDFs/binaries listed in the sitemap are not
                # documentation pages; regex tag-stripping would turn
                # them into mojibake. Counted, excluded from scope by
                # design (not a crawl failure).
                skipped_non_html.append(url)
                continue
            final_url = str(getattr(resp, "url", url) or url)
            text = getattr(resp, "text", "") or ""
            if _login_bounce(url, final_url, text):
                failed_urls.append(url)
                continue
            title = _extract_title(text) or urlparse(url).path.rstrip(
                "/"
            ).split("/")[-1]
            body = _extract_body(text)
            if not body:
                continue
            records.append(
                DocumentationRecord(
                    title=title,
                    url=url,
                    body=body,
                    site_name=urlparse(url).netloc,
                )
            )
        return _CrawlOutcome(
            records=tuple(records),
            failed_urls=tuple(failed_urls),
            total_urls=len(urls),
            sitemap_total=len(all_urls),
            truncated=truncated,
            skipped_non_html=tuple(skipped_non_html),
        )


@dataclass(frozen=True)
class _CrawlOutcome:
    """One sitemap crawl: emitted records plus the pages that failed.

    ``total_urls`` is the crawled frontier (after any ``max_pages``
    cap); ``sitemap_total`` is the full sitemap size and ``truncated``
    is True when the cap actually cut the frontier. Non-HTML sitemap
    entries (by Content-Type) are counted in ``skipped_non_html`` —
    excluded from the source's scope by design, not crawl failures.
    """

    records: tuple[DocumentationRecord, ...]
    failed_urls: tuple[str, ...]
    total_urls: int
    sitemap_total: int = 0
    truncated: bool = False
    skipped_non_html: tuple[str, ...] = ()


def _content_type(response: Any) -> str:
    """Normalized media type of a response ('' when unavailable)."""
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("Content-Type") or headers.get("content-type") or ""
    return str(raw).split(";")[0].strip().lower()


def _is_html_content_type(content_type: str) -> bool:
    return "html" in content_type


_SSO_LOGIN_HOSTS = frozenset({"auth.cern.ch", "login.cern.ch"})


def _is_sso_login_host(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host.lower() in _SSO_LOGIN_HOSTS


def _login_bounce(requested_url: str, final_url: str, text: str) -> bool:
    """True only when the fetch actually landed on CERN SSO.

    A bounce means the final response URL's host *is* an SSO login host,
    or the request was redirected away from *requested_url* and the body
    reads as a login page. A non-redirected 200 whose own path merely
    contains ``/login`` or ``/sso/``, or whose body just mentions
    ``auth.cern.ch``, is a legitimate docs page — the pre-fix substring
    checks silently dropped such pages.
    """
    if _is_sso_login_host(final_url):
        return True
    if final_url and final_url != requested_url:
        return looks_like_login_page(text[:2000])
    return False


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _records_from_payload(payload: list[Any]) -> list[DocumentationRecord]:
    records: list[DocumentationRecord] = []
    seen_urls: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = _pg_text(str(item.get("url") or ""))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        records.append(
            DocumentationRecord(
                title=_pg_text(str(item.get("title") or "")),
                url=url,
                body=_pg_text(str(item.get("body") or "")),
                site_name=_pg_text(str(item.get("site_name") or "")),
                source_repo=_pg_text(
                    str(item.get("source_repo") or item.get("repo") or "")
                ),
                path=_pg_text(
                    str(item.get("path") or item.get("file_path") or "")
                ),
            )
        )
    return records


def _facts_for_records(
    records: list[DocumentationRecord],
    revision: dict[str, Any],
    targets: dict[str, Any],
    *,
    repo_base_url: str = DEFAULT_REPO_BASE_URL,
    chunker_name: str = DEFAULT_CHUNKER_NAME,
) -> Iterator[NodeFact | EdgeFact]:
    emitted_repos: set[str] = set()
    for record in records:
        if record.source_repo and record.source_repo not in emitted_repos:
            emitted_repos.add(record.source_repo)
            yield _repo_node(
                record.source_repo, revision, repo_base_url=repo_base_url
            )
        yield _page_node(record, revision)
        if record.source_repo:
            yield EdgeFact(
                src=f"software_repository:{record.source_repo}",
                dst=record.node_id,
                edge_type="contains",
                attrs={"relationship": "repo_documentation"},
                source_record_id=record.source_record_id,
                source_revision=revision,
            )
        for chunk_index, offset, chunk_text in _chunks(record.body):
            chunk_hash = _sha256(f"{record.node_id}\0{chunk_index}\0{chunk_text}")
            chunk_id = f"chunk:{chunk_hash[:16]}"
            chunk_record_id = {
                **record.source_record_id,
                "chunk_index": chunk_index,
            }
            yield NodeFact(
                node_id=chunk_id,
                subtype="document_chunk",
                attrs={
                    "chunk_id": chunk_id,
                    "content_sha256": _sha256(chunk_text),
                    "text": chunk_text,
                    "char_offset": offset,
                    "char_length": len(chunk_text),
                    "chunker_name": chunker_name,
                    "heading_path": record.title,
                },
                source_record_id=chunk_record_id,
                source_revision=revision,
            )
            yield EdgeFact(
                src=record.node_id,
                dst=chunk_id,
                edge_type="contains",
                attrs={"chunk_index": chunk_index},
                source_record_id=chunk_record_id,
                source_revision=revision,
            )
            yield from _reference_edges(
                chunk_id,
                chunk_text,
                chunk_record_id,
                revision,
                targets,
            )


def _repo_node(
    repo: str,
    revision: dict[str, Any],
    *,
    repo_base_url: str = DEFAULT_REPO_BASE_URL,
) -> NodeFact:
    url = f"{repo_base_url.rstrip('/')}/{repo}"
    return NodeFact(
        node_id=f"software_repository:{repo}",
        subtype="software_repository",
        attrs={
            "label": repo,
            "name": repo,
            "full_name": repo,
            "url": url,
            "text": repo,
        },
        source_record_id={"repo": repo},
        source_revision=revision,
    )


def _page_node(
    record: DocumentationRecord,
    revision: dict[str, Any],
) -> NodeFact:
    return NodeFact(
        node_id=record.node_id,
        subtype="documentation_page",
        attrs={
            "label": record.title or record.url,
            "title": record.title,
            "url": record.url,
            "body": record.body,
            "source_repo": record.source_repo,
            "site_name": record.site_name,
            "path": record.path,
            "text": " ".join(filter(None, [record.title, record.body])),
            # Parity with the original's optional-attr filter: of the
            # github_file_content extras only these two survived its
            # ``value not in ("", 0)`` check on docsite/gitlab/SSO pages
            # (record_kind is always set; ``[]`` passes the filter).
            "record_kind": "documentation_page",
            "service_aliases": [],
        },
        source_record_id=record.source_record_id,
        source_revision=revision,
    )


def _reference_edges(
    chunk_id: str,
    text: str,
    source_record_id: dict[str, Any],
    revision: dict[str, Any],
    targets: dict[str, Any],
) -> Iterator[EdgeFact]:
    for site in sorted(set(_SITE_RE.findall(text)) & targets["site"]):
        yield _reference_edge(
            chunk_id,
            f"site:{site}",
            source_record_id,
            revision,
            match_type="cms_site",
        )
    for release in sorted(set(_CMSSW_RE.findall(text)) & targets["release"]):
        yield _reference_edge(
            chunk_id,
            f"cmssw_release:{release}",
            source_record_id,
            revision,
            match_type="cmssw_release",
        )
    for issue in sorted(set(_ISSUE_KEY_RE.findall(text)) & targets["jira"]):
        yield _reference_edge(
            chunk_id,
            f"jira:{issue}",
            source_record_id,
            revision,
            match_type="jira_issue",
        )
    service_lookup = targets["service"]
    matched_services: set[str] = set()
    for host in _HOST_RE.findall(text):
        for service_id in service_lookup.get(_normalize_endpoint_alias(host), ()):
            matched_services.add(service_id)
    for service_id in sorted(matched_services):
        yield _reference_edge(
            chunk_id,
            service_id,
            source_record_id,
            revision,
            match_type="infrastructure_service",
        )


def _reference_edge(
    src: str,
    dst: str,
    source_record_id: dict[str, Any],
    revision: dict[str, Any],
    *,
    match_type: str,
    attrs: dict[str, Any] | None = None,
) -> EdgeFact:
    edge_attrs = {"match_type": match_type}
    if attrs:
        edge_attrs.update(attrs)
    return EdgeFact(
        src=src,
        dst=dst,
        edge_type="references",
        provenance="derived_deterministic",
        confidence=0.95,
        attrs=edge_attrs,
        source_record_id=source_record_id,
        source_revision=revision,
    )


def _reference_targets(
    *,
    sites_path: str | None = None,
    releases_path: str | None = None,
    jira_records_path: str | None = None,
    services_path: str | None = None,
    base: str | None = None,
) -> dict[str, Any]:
    """Known reference targets.

    Unconfigured caches (``None``) are skipped silently (no reference
    edges of that kind). A *configured* sites/releases/services path
    whose file is missing raises, as in the original — a silently empty
    target set would drop every reference edge of that kind without a
    trace. The jira cache keeps the original's skip-when-absent
    behavior.
    """
    return {
        "site": _known_sites(sites_path, base=base),
        "release": _known_releases(releases_path, base=base),
        "jira": _known_jira(jira_records_path, base=base),
        "service": _known_services(services_path, base=base),
    }


def _configured_json(path: str | None, *, base: str | None = None) -> Any:
    """Load a configured reference cache; ``None`` means not configured.

    A configured-but-missing file raises so misconfiguration surfaces
    instead of yielding a silently empty reference-target set.
    """
    if not path:
        return None
    resolved = resolve_repo_path(path, base=base)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{path}: configured reference cache is missing ({resolved})"
        )
    return load_json(path, base=base)


def _optional_json(path: str | None, *, base: str | None = None) -> Any:
    if not path or not resolve_repo_path(path, base=base).is_file():
        return None
    return load_json(path, base=base)


def _known_sites(path: str | None, *, base: str | None = None) -> set[str]:
    payload = _configured_json(path, base=base)
    if isinstance(payload, dict):
        return {str(k) for k in payload}
    return set()


def _known_releases(path: str | None, *, base: str | None = None) -> set[str]:
    payload = _configured_json(path, base=base)
    if not isinstance(payload, list):
        return set()
    return {
        str(item.get("label") or "")
        for item in payload
        if isinstance(item, dict) and item.get("label")
    }


def _known_jira(path: str | None, *, base: str | None = None) -> set[str]:
    payload = _optional_json(path, base=base)
    if not isinstance(payload, list):
        return set()
    return {
        str(item.get("key") or item.get("issue_key") or "")
        for item in payload
        if isinstance(item, dict) and (item.get("key") or item.get("issue_key"))
    }


def _known_services(
    path: str | None, *, base: str | None = None
) -> dict[str, set[str]]:
    payload = _configured_json(path, base=base)
    if not isinstance(payload, dict):
        return {}
    lookup: dict[str, set[str]] = {}
    for name, service in payload.items():
        if not isinstance(service, dict):
            continue
        service_id = f"svc:{name}"
        endpoint = str(service.get("endpoint") or "")
        for alias in _service_aliases(str(name), endpoint):
            lookup.setdefault(alias, set()).add(service_id)
    return lookup


def _service_aliases(name: str, endpoint: str) -> set[str]:
    aliases = {_normalize_endpoint_alias(name)}
    endpoint_alias = _normalize_endpoint_alias(endpoint)
    if endpoint_alias:
        aliases.add(endpoint_alias)
        aliases.add(_endpoint_host(endpoint_alias))
    return {alias for alias in aliases if alias}


def _normalize_endpoint_alias(value: str) -> str:
    value = value.strip().lower()
    if "://" in value:
        parsed = urlparse(value)
        value = parsed.netloc or parsed.path
    return value.rstrip("/.,;:)")


def _endpoint_host(value: str) -> str:
    if not value:
        return ""
    if value.count(":") == 1:
        host, port = value.rsplit(":", 1)
        if port.isdigit():
            return host
    return value


def _chunks(text: str) -> Iterator[tuple[int, int, str]]:
    if not text:
        return
    step = max(1, _CHUNK_SIZE - _CHUNK_OVERLAP)
    idx = 0
    offset = 0
    while offset < len(text):
        chunk = _pg_text(text[offset:offset + _CHUNK_SIZE].strip())
        if chunk:
            yield idx, offset, chunk
            idx += 1
        if offset + _CHUNK_SIZE >= len(text):
            break
        offset += step


def _parse_sitemap(xml_content: bytes) -> list[str]:
    root = ElementTree.fromstring(xml_content)  # noqa: S314
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]


def _extract_title(markup: str) -> str:
    match = re.search(r"<title[^>]*>([^<]+)</title>", markup, re.IGNORECASE)
    return html.unescape(match.group(1).strip()) if match else ""


def _extract_body(markup: str) -> str:
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        markup,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return _pg_text(re.sub(r"\s+", " ", text).strip())


def _records_hash(records: list[DocumentationRecord]) -> str:
    data = json.dumps(
        [asdict(record) for record in records],
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(data)


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return slug or _sha256(text)[:16]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pg_text(text: str) -> str:
    return text.replace("\x00", " ")
