"""HyperNews forum-thread source (SSO cookie crawl + records cache).

Ported from okg-deployments ``cms/cms_sources/hypernews.py`` (586 LOC,
``HyperNewsSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; changes:

- Cache/probe helpers come from :mod:`archi.auth.cache` with an
  explicit ``base`` parameter; the CERN login-page check comes from
  :func:`archi.auth.cookies.looks_like_login_page` (same markers as the
  cms ``is_cern_login_page``).
- Chunking and reference-edge emission are shared with
  :mod:`archi.sources.docs` (same import direction as the original).
- Hardcoded CMS values are parameters with cms defaults documented in
  the template: ``base_url`` (``https://hypernews.cern.ch``), ``forums``
  (``comp-ops``, ``mcOps``), ``records_path``. The reference-target
  caches (sites/releases/jira/services) default to ``None`` (no
  reference edges of that kind), matching the landed jira/docs idiom;
  the cms deployment's paths are in the template below.
- Parity wart kept on purpose (as in jira.py): chunk ids hash with a
  literal backslash-zero separator, not a NUL byte.

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``forum_thread`` ships in
``archi/schemas/operations.yaml``; ``document_chunk`` comes from the
``extraction`` module. ::

    hypernews:
      module: archi.sources.hypernews
      class: HyperNewsSource
      ownership_id: <instance>.hypernews
      admission_policy:
        producer_id: <instance>.hypernews
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: hypernews
        output_signature:
          nodes:
            - {subtype: forum_thread}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: forum_thread, edge_type: contains, dst_subtype: document_chunk}
            # Uncomment together with the matching params below:
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: site}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: cmssw_release}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: HyperNews forum threads and their text chunks
          nodes: [forum_thread, document_chunk]
          edges:
            - forum_thread contains document_chunk
            # Uncomment together with the matching params below:
            # - document_chunk references site
            # - document_chunk references cmssw_release
            # - document_chunk references jira_issue
            # - document_chunk references infrastructure_service
      source_class: discovery_crawl
      record_identity_kind: remote_id
      record_identity_fields: [thread_id]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [HYPERNEWS_COOKIE_FILE]
      required_for_baseline: false
      params:
        # cms defaults; the cms deployment used data/cms/hypernews/
        base_url: https://hypernews.cern.ch
        forums: [comp-ops, mcOps]
        records_path: data/hypernews/records.json
        cookie_file_env: HYPERNEWS_COOKIE_FILE
        # Optional reference-target caches (cms deployment values) —
        # enabling any requires the matching references edges above
        # and the target subtype in the deployment schema:
        # sites_path: data/cric/sites.json
        # releases_path: data/cmssw-releases/records.json
        # jira_records_path: data/jira/records.json
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
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
)
from okg.substrate.library.sources.base import SourceRun
from okg.deployment import MutableApiProbe
from okg.deployment import file_preflight

from archi.auth.cache import load_json, resolve_repo_path
from archi.auth.cookies import looks_like_login_page
from archi.sources._cache_report import skipped_items_status
from archi.sources.docs import (
    _chunks,
    _pg_text,
    _reference_edges,
    _reference_targets,
)

_DEFAULT_BASE_URL = "https://hypernews.cern.ch"
_DEFAULT_FORUMS = ("comp-ops", "mcOps")
DEFAULT_CHUNKER_NAME = "cms_hypernews_window_v1"


@dataclass(frozen=True)
class HyperNewsRecord:
    thread_id: str
    title: str
    url: str
    forum_name: str
    body: str = ""
    author: str = ""
    date: str = ""
    reply_count: int = 0

    @property
    def node_id(self) -> str:
        return f"hn:{self.thread_id}"


@dataclass
class _FetchOutcome:
    """Result of a live crawl, with every failure made visible.

    A crawl that lost forums, threads, or the cookie must never be
    mistaken for a complete one: ``missing_from_completed_scope`` would
    retract every record the failed fetches used to produce.
    """

    records: list[HyperNewsRecord]
    total_forums: int
    failed_forums: list[str]
    failed_hydrations: int = 0
    truncated: bool = False
    listed_threads: int = 0
    error: str = ""

    @property
    def partial(self) -> bool:
        return bool(self.failed_forums or self.failed_hydrations)


class HyperNewsSource:
    """Fetch and chunk HyperNews threads through SSO cookies."""

    name = "hypernews"
    profile = "discovery_crawl"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        cookie_file_env: str = "HYPERNEWS_COOKIE_FILE",
        forums: list[str] | None = None,
        max_threads: int | None = None,
        max_workers: int = 8,
        fetch_timeout: int = 12,
        records_path: str = "data/hypernews/records.json",
        required: bool = False,
        sites_path: str | None = None,
        releases_path: str | None = None,
        jira_records_path: str | None = None,
        services_path: str | None = None,
        chunker_name: str = DEFAULT_CHUNKER_NAME,
        records: list[HyperNewsRecord] | None = None,
        base: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cookie_file_env = cookie_file_env
        self.forums = tuple(forums or _DEFAULT_FORUMS)
        self.max_threads = max_threads
        self.max_workers = max_workers
        self.fetch_timeout = fetch_timeout
        self.records_path = records_path
        self.required = required
        self.sites_path = sites_path
        self.releases_path = releases_path
        self.jira_records_path = jira_records_path
        self.services_path = services_path
        self.chunker_name = chunker_name
        self._records = records
        self.base = base
        self.change_probe = MutableApiProbe(
            version_fn=self._probe_version,
            config={
                "base_url": self.base_url,
                "cookie_file_env": self.cookie_file_env,
                "forums": self.forums,
                "max_threads": self.max_threads,
                "records_path": self.records_path,
                "sites_path": self.sites_path,
                "releases_path": self.releases_path,
                "jira_records_path": self.jira_records_path,
                "services_path": self.services_path,
            },
            emit_targets=HyperNewsSource,
        )

    def _probe_version(self) -> str:
        if self._records is not None:
            payload = json.dumps(
                [record.__dict__ for record in self._records],
                sort_keys=True,
                default=str,
            )
            return _probe_hash(
                ("records", payload.encode("utf-8")),
                *self._auxiliary_probe_items(),
            )
        items = [
            item
            for item in [
                self._path_probe_item("records", self.records_path),
                *self._auxiliary_probe_items(),
            ]
            if item is not None
        ]
        if items:
            return _probe_hash(*items)
        return _checked_at()

    def _path_probe_item(
        self, label: str, raw_path: str
    ) -> tuple[str, bytes] | None:
        path = resolve_repo_path(raw_path, base=self.base)
        if not path.is_file():
            return None
        return label, path.read_bytes()

    def _auxiliary_probe_items(self) -> list[tuple[str, bytes]]:
        paths = (
            self.sites_path,
            self.releases_path,
            self.jira_records_path,
            self.services_path,
        )
        return [
            item
            for index, raw_path in enumerate(paths)
            if raw_path is not None
            for item in [
                self._path_probe_item(f"aux:{index}:{raw_path}", raw_path)
            ]
            if item is not None
        ]

    def preflight(self, mode: str = "live") -> PreflightResult:
        if self._records is not None:
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="fixture",
                required=self.required,
                record_count=len(self._records),
                reason="in-memory HyperNews records supplied",
                checked_at=_checked_at(),
            )
        cache_path = resolve_repo_path(self.records_path, base=self.base)
        if cache_path.is_file():
            try:
                records, _skipped = self._records_from_cache()
            except ValueError as exc:
                return PreflightResult(
                    source_name=self.name,
                    status="endpoint_failed",
                    mode="cache",
                    required=self.required,
                    cache_path=str(cache_path),
                    reason=str(exc),
                    checked_at=_checked_at(),
                )
            if not records:
                # Match run()'s refusal: an empty cache must not look
                # healthy in preflight while run() rejects it.
                return PreflightResult(
                    source_name=self.name,
                    status="endpoint_failed",
                    mode="cache",
                    required=self.required,
                    cache_path=str(cache_path),
                    record_count=0,
                    reason=self._empty_cache_reason(),
                    checked_at=_checked_at(),
                )
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="cache",
                required=self.required,
                cache_path=str(cache_path),
                record_count=len(records),
                content_hash=_records_hash(records),
                reason="local HyperNews records cache present",
                checked_at=_checked_at(),
            )
        file_result = file_preflight(
            self.name,
            self.cookie_file_env,
            required=self.required,
            mode=mode,
        )
        if file_result.status != "ok" or mode != "live":
            return file_result
        return self._endpoint_probe(file_result)

    def _endpoint_probe(
        self,
        file_result: PreflightResult,
    ) -> PreflightResult:
        import http.cookiejar
        import requests

        cookie_file = os.environ.get(self.cookie_file_env, "")
        try:
            jar = http.cookiejar.MozillaCookieJar(cookie_file)
            jar.load(ignore_discard=True, ignore_expires=True)
            session = requests.Session()
            session.cookies = jar
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode="live",
                required=self.required,
                credential_refs=file_result.credential_refs,
                endpoint=self.base_url,
                reason=f"HyperNews probe failed: {type(exc).__name__}",
                checked_at=_checked_at(),
            )
        failures: list[str] = []
        for forum in self.forums:
            endpoint = f"{self.base_url}/HyperNews/CMS/get/{forum}.html"
            try:
                resp = session.get(endpoint, timeout=45)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{forum}: {type(exc).__name__}")
                continue
            if looks_like_login_page(resp.text) or "auth.cern.ch" in resp.url:
                return PreflightResult(
                    source_name=self.name,
                    status="auth_failed",
                    mode="live",
                    required=self.required,
                    credential_refs=file_result.credential_refs,
                    endpoint=endpoint,
                    reason="HyperNews cookie reached CERN login page",
                    checked_at=_checked_at(),
                )
            if resp.status_code >= 400:
                failures.append(f"{forum}: HTTP {resp.status_code}")
                continue
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="live",
                required=self.required,
                credential_refs=file_result.credential_refs,
                endpoint=endpoint,
                reason="HyperNews forum listing reachable",
                checked_at=_checked_at(),
            )
        return PreflightResult(
            source_name=self.name,
            status="endpoint_failed",
            mode="live",
            required=self.required,
            credential_refs=file_result.credential_refs,
            endpoint=self.base_url,
            reason=(
                "No configured HyperNews forum listing reachable"
                + (f": {'; '.join(failures)}" if failures else "")
            ),
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

        records = self._records
        run_health_mode = "fixture"
        run_status = "ok"
        run_reason = "HyperNews fixture records supplied"
        credential_refs: tuple[str, ...] = ()
        allow_scope = True
        if records is None:
            cache_path = resolve_repo_path(self.records_path, base=self.base)
            if cache_path.is_file():
                records, skipped = self._records_from_cache()
                run_health_mode = "cache"
                if not records:
                    # An empty cache would otherwise claim a healthy
                    # complete scope over zero threads (and replay
                    # forever); refuse instead of retracting everything.
                    # (Preflight reports the same refusal, so this is
                    # normally caught before reaching here.)
                    return SourceRun(
                        facts=[],
                        completed_scope=False,
                        run_mode=mode,
                        health=ConnectorHealth(
                            status="endpoint_failed",
                            mode="cache",
                            record_count=0,
                            reason=self._empty_cache_reason(),
                        ),
                    )
                run_status, run_reason = skipped_items_status(
                    status="ok",
                    reason="HyperNews records read from local cache",
                    record_count=len(records),
                    skipped_count=skipped,
                )
                allow_scope = not skipped
            else:
                run_health_mode = "live"
                credential_refs = (self.cookie_file_env,)
                outcome = self._fetch()
                if not outcome.records:
                    return SourceRun(
                        facts=[],
                        completed_scope=False,
                        run_mode=mode,
                        health=ConnectorHealth(
                            status="endpoint_failed",
                            mode="live",
                            credential_refs=credential_refs,
                            record_count=0,
                            endpoint=self.base_url,
                            reason=_total_failure_reason(outcome),
                        ),
                    )
                records = outcome.records
                run_reason = (
                    "HyperNews records fetched through SSO cookie"
                )
                problems: list[str] = []
                if outcome.failed_forums:
                    samples = "; ".join(outcome.failed_forums[:3])
                    problems.append(
                        f"{len(outcome.failed_forums)}/"
                        f"{outcome.total_forums} forum listings failed "
                        f"(e.g. {samples})"
                    )
                if outcome.failed_hydrations:
                    problems.append(
                        f"{outcome.failed_hydrations}/"
                        f"{outcome.listed_threads} thread fetches "
                        "failed and were dropped"
                    )
                if problems:
                    # Emit what succeeded, but never claim a complete
                    # scope over a partially failed crawl (docs.py
                    # partial-crawl idiom).
                    run_status = "endpoint_failed"
                    allow_scope = False
                    run_reason = (
                        "partial HyperNews crawl: "
                        + "; ".join(problems)
                        + "; no complete scope claimed"
                    )
                if outcome.truncated:
                    allow_scope = False
                    run_reason += (
                        f"; max_threads={self.max_threads} truncated "
                        "forum listings; no complete scope claimed"
                    )
                # Only a full, untruncated crawl may be persisted: a
                # partial cache would replay on the next run as a
                # healthy complete scope and retract the missing
                # threads.
                if not problems and not outcome.truncated:
                    self._write_cache(records)
        record_hash = _records_hash(records)
        revision = {
            "run_id": run_id,
            "content_hash": record_hash,
            "n_records": len(records),
            "forums": list(self.forums),
        }
        targets = _reference_targets(
            sites_path=self.sites_path,
            releases_path=self.releases_path,
            jira_records_path=self.jira_records_path,
            services_path=self.services_path,
            base=self.base,
        )

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _thread_node(record, revision)
                body = "\n\n".join(
                    part for part in [record.title, record.body] if part
                )
                for chunk_index, offset, chunk_text in _chunks(body):
                    chunk_id = f"chunk:{_sha256(f'{record.node_id}\\0{chunk_index}\\0{chunk_text}')[:16]}"
                    chunk_record_id = {
                        "thread_id": record.thread_id,
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
                            "chunker_name": self.chunker_name,
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

        return SourceRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and allow_scope
            ),
            run_mode=mode,
            health=ConnectorHealth(
                status=run_status,
                mode=run_health_mode,
                credential_refs=credential_refs,
                record_count=len(records),
                content_hash=record_hash,
                reason=run_reason,
            ),
        )

    def _empty_cache_reason(self) -> str:
        return (
            f"HyperNews records cache {self.records_path} contains no "
            "usable threads; refusing an empty complete scope (delete "
            "the cache to force a live re-fetch)"
        )

    def _records_from_cache(self) -> tuple[list[HyperNewsRecord], int]:
        raw = load_json(self.records_path, base=self.base)
        if isinstance(raw, dict):
            if "records" not in raw:
                raise ValueError(
                    f"{self.records_path}: expected a 'records' list in "
                    "the cache dict; refusing to treat a drifted payload "
                    "as zero threads"
                )
            raw = raw["records"]
        if not isinstance(raw, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of threads"
            )
        records: list[HyperNewsRecord] = []
        skipped = 0
        for item in raw:
            if not isinstance(item, dict):
                skipped += 1
                continue
            record = _record_from_dict(item)
            if not record.thread_id:
                skipped += 1
                continue
            records.append(record)
        return records, skipped

    def _write_cache(self, records: list[HyperNewsRecord]) -> None:
        path = resolve_repo_path(self.records_path, base=self.base)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps([asdict(record) for record in records], indent=2)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def _fetch(self) -> _FetchOutcome:
        import http.cookiejar
        import requests

        cookie_file = os.environ.get(self.cookie_file_env, "")
        if not cookie_file or not Path(cookie_file).is_file():
            return _FetchOutcome(
                records=[],
                total_forums=len(self.forums),
                failed_forums=list(self.forums),
                error="cookie file missing or unreadable",
            )
        jar = http.cookiejar.MozillaCookieJar(cookie_file)
        jar.load(ignore_discard=True, ignore_expires=True)
        session = requests.Session()
        session.cookies = jar
        records: list[HyperNewsRecord] = []
        failed_forums: list[str] = []
        truncated = False
        for forum in self.forums:
            forum_url = f"{self.base_url}/HyperNews/CMS/get/{forum}.html"
            try:
                resp = session.get(forum_url, timeout=self.fetch_timeout)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                failed_forums.append(f"{forum}: {type(exc).__name__}")
                continue
            parsed = self._parse_threads(resp.text, forum)
            if (
                self.max_threads is not None
                and len(parsed) >= self.max_threads
            ):
                truncated = True
            records.extend(parsed)

        listed_threads = len(records)
        if not records:
            return _FetchOutcome(
                records=[],
                total_forums=len(self.forums),
                failed_forums=failed_forums,
                truncated=truncated,
            )

        thread_local = threading.local()

        def _session() -> Any:
            existing = getattr(thread_local, "session", None)
            if existing is not None:
                return existing
            worker_jar = http.cookiejar.MozillaCookieJar(cookie_file)
            worker_jar.load(ignore_discard=True, ignore_expires=True)
            worker_session = requests.Session()
            worker_session.cookies = worker_jar
            thread_local.session = worker_session
            return worker_session

        def _hydrate(thread: HyperNewsRecord) -> HyperNewsRecord | None:
            try:
                thread_resp = _session().get(
                    thread.url, timeout=self.fetch_timeout,
                )
                thread_resp.raise_for_status()
            except Exception:  # noqa: BLE001
                # Emitting the un-hydrated listing stub would blank the
                # thread's body/author and drop its chunks under a
                # complete scope; drop it and report instead.
                return None
            return HyperNewsRecord(
                thread_id=thread.thread_id,
                title=thread.title,
                url=thread.url,
                forum_name=thread.forum_name,
                body=_extract_body(thread_resp.text),
                author=_extract_author(thread_resp.text),
                date=_extract_date(thread_resp.text),
                reply_count=_count_replies(thread_resp.text),
            )

        workers = max(1, int(self.max_workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_hydrate, records))
        hydrated = [record for record in results if record is not None]
        return _FetchOutcome(
            records=hydrated,
            total_forums=len(self.forums),
            failed_forums=failed_forums,
            failed_hydrations=len(results) - len(hydrated),
            truncated=truncated,
            listed_threads=listed_threads,
        )

    def _parse_threads(self, markup: str, forum: str) -> list[HyperNewsRecord]:
        records: list[HyperNewsRecord] = []
        for match in re.finditer(
            r'<li\s+value="(\d+)"[^>]*>.*?<a\s+href="([^"]+)"[^>]*>'
            r"\s*([^<]+?)\s*</a>",
            markup,
            re.DOTALL | re.IGNORECASE,
        ):
            thread_id = match.group(1)
            href = re.sub(r"get//+", "get/", match.group(2))
            url = href if href.startswith("http") else f"{self.base_url}{href}"
            records.append(HyperNewsRecord(
                thread_id=f"{forum}/{thread_id}",
                title=_pg_text(html.unescape(match.group(3).strip())),
                url=url,
                forum_name=forum,
            ))
            if (
                self.max_threads is not None
                and len(records) >= self.max_threads
            ):
                break
        return records


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _total_failure_reason(outcome: _FetchOutcome) -> str:
    """Reason string for a live crawl that produced zero records."""
    if outcome.error:
        detail = f"HyperNews fetch failed: {outcome.error}"
    elif outcome.failed_forums:
        samples = "; ".join(outcome.failed_forums[:3])
        detail = (
            f"{len(outcome.failed_forums)}/{outcome.total_forums} "
            f"forum listings failed (e.g. {samples})"
        )
        if len(outcome.failed_forums) < outcome.total_forums:
            detail += (
                "; the reachable forum listings parsed to zero threads"
            )
    else:
        detail = (
            f"all {outcome.total_forums} forum listings were reachable "
            "but parsed to zero threads (markup drift?)"
        )
    return (
        f"{detail}; treated as fetch failure — no records cache "
        "written and no scope claimed"
    )


def _thread_node(
    record: HyperNewsRecord,
    revision: dict[str, Any],
) -> NodeFact:
    body = _clean_body(record.body)
    return NodeFact(
        node_id=record.node_id,
        subtype="forum_thread",
        attrs={
            "label": record.title,
            "title": record.title,
            "url": record.url,
            "forum_name": record.forum_name,
            "body": body,
            "author": record.author,
            "date": record.date,
            "reply_count": record.reply_count,
            "text": "\n\n".join(
                part for part in [record.title, body] if part
            ),
        },
        source_record_id={"thread_id": record.thread_id},
        source_revision=revision,
    )


def _record_from_dict(raw: dict[str, Any]) -> HyperNewsRecord:
    return HyperNewsRecord(
        thread_id=str(raw.get("thread_id") or ""),
        title=_pg_text(str(raw.get("title") or "")),
        url=str(raw.get("url") or ""),
        forum_name=str(raw.get("forum_name") or ""),
        body=_clean_body(str(raw.get("body") or "")),
        author=_pg_text(str(raw.get("author") or "")),
        date=str(raw.get("date") or ""),
        reply_count=int(raw.get("reply_count") or 0),
    )


def _extract_body(markup: str) -> str:
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        markup,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return _clean_body(_pg_text(re.sub(r"\s+", " ", text).strip()))


def _clean_body(text: str) -> str:
    text = re.sub(
        r",?\s*(?:annotation_type|content_type|name|from_|num_messages|"
        r"last_message_date|last_mod|message_id|num|previous_num|"
        r"next_num|up_url|up_rel|node_type)"
        r"=(?:<[^>]+>|'[^']*'|datetime\.datetime\([^)]+\)|\d+)",
        "",
        text,
    )
    text = re.sub(r"URCMessage\([^)]*\)", " ", text)
    text = re.sub(r"(?:&nbsp;)?\s*Next-in-(?:Thread|Forum)\b", " ", text)
    text = re.sub(r"Click to see raw info\s*,?", " ", text)
    text = re.sub(
        r"This site runs HyperNewsViewer.*", " ", text, flags=re.DOTALL
    )
    return re.sub(r"\s+", " ", text).strip()


def _extract_author(markup: str) -> str:
    match = re.search(r'<i>From:</i>\s*<a[^>]*><i></i>([^<&]+)', markup)
    if match:
        return _pg_text(html.unescape(match.group(1).strip()))
    match = re.search(
        r"(?:Posted by|From|Author)[:\s]*([^<\n]{3,80})",
        markup,
        re.IGNORECASE,
    )
    return _pg_text(html.unescape(match.group(1).strip())) if match else ""


def _extract_date(markup: str) -> str:
    match = re.search(r"<i>Date:</i>\s*(\d{1,2}\s+\w{3},?\s+\d{4})", markup)
    if match:
        raw = match.group(1).strip()
        for fmt in ("%d %b, %Y", "%d %b %Y"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return raw
    match = re.search(
        r"(?:Date|Posted)[:\s]*(\d{4}[-/]\d{2}[-/]\d{2}(?:\s+\d{2}:\d{2})?)",
        markup,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _count_replies(markup: str) -> int:
    replies = re.findall(r'<a\s+name="(\d+)"', markup)
    return max(0, len(replies) - 1)


def _records_hash(records: list[HyperNewsRecord]) -> str:
    payload = json.dumps(
        [record.__dict__ for record in records],
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(payload)


def _probe_hash(*items: tuple[str, bytes]) -> str:
    digest = hashlib.sha256()
    for label, data in sorted(items, key=lambda item: item[0]):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).hexdigest().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
