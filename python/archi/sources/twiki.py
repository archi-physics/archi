"""TWiki sources: EOS-snapshot reader and seeded live crawler.

Two ingestors over one parser core (:mod:`archi.sources._twiki_parse`),
consolidated for the archi v3 package (req.w2.sources,
task.w2.sources-twiki) from:

- okg-deployments ``cms/cms_sources/twiki_eos.py`` (515 LOC) at
  ``main@f33a9c4`` — the canonical base. :class:`TwikiEOSSource` is its
  lift: same preflight/probe/run shape, same emission (page node attrs,
  page-to-page reference kinds and their dedup order, chunking over
  ``title + body``, chunk entity references), same default chunker name
  ``cms_twiki_window_v1`` so the cms parity corpus does not churn.
- okg-deployments ``wisdqm/wisdqm_sources/docs.py`` +
  ``wisdqm/scripts/download_sources.py`` at ``main@f33a9c4`` — the
  fidelity flags on the parser core, the viewauth->view URL
  canonicalization, and the seeded/depth-bounded snapshot-crawl shape
  (``seed_topics`` + ``max_depth``, snapshot-existence-checked link
  following) that :class:`TwikiEOSSource` gains as optional behavior.
- okg-deployments ``cern-twiki`` — nothing (maintainer decision).
- archi v2 ``dev@28b977d1``
  ``src/data_manager/collectors/scrapers/spiders/twiki.py`` (+
  ``parsers/twiki.py``, ``auth/cern_sso.py``): taken case-by-case into
  :class:`TwikiCrawlSource` — seeded start URLs with a bounded depth
  limit and page cap, structural-topic deny-listing (WebChanges /
  WebIndex / WebLeftBar / ..., covered here by the parser core's
  ``DEFAULT_SKIP_PATTERNS``), and URL normalization that drops query +
  fragment. Left behind: the Scrapy engine, the Playwright interactive
  SSO login (v3 never runs an in-process login; cookies come from
  :mod:`archi.auth.cookies` files produced out-of-band), and the
  rendered-HTML DOM extraction — the dev spider deny-listed
  ``/bin/raw`` and scraped ``#patternMainContents`` HTML, whereas this
  source fetches the raw topic markup (``?raw=all``, which is
  ``?raw=text`` plus the ``%META`` lines that ``parse_meta`` and
  topic_parent references need) so both ingestors share one parser.

Decisions: two-ingestor design + fidelity flags approved by the
maintainer 2026-08-12; cern-twiki hardening excluded.

Failure semantics follow the parity-review lessons encoded in
:mod:`archi.sources.docs`: per-topic fetch failures are tracked; any
failure (or a missing/unparseable cookie when ``cookie_file_env`` is
configured) means ``completed_scope=False`` plus a health reason with
failed/total counts and samples, using only the closed preflight status
vocabulary (``ok`` / ``missing_credential`` / ``auth_failed`` /
``tls_failed`` / ``endpoint_failed`` / ``cache_missing``). Login
bounces are detected with docs.py's ``_login_bounce`` (real SSO final
host, or redirect + login-looking body — never a mere ``/login``
substring in a page's own path).

Parity wart kept on purpose (as in archi/sources/jira.py): chunk ids
hash their input with a literal backslash-zero separator (the cms
original's ``f'..\\0..'`` inside an f-string), not the NUL byte docs.py
uses. Changing it would re-key every twiki chunk at cutover.

Deliberate parity deviations from the cms parser (its ``=code=``
unwrap regex paired ``=`` across lines/assignments and its heading
``\\s*`` absorbed the next line after a bare marker) are fixed in
:mod:`archi.sources._twiki_parse` — see its docstring; TWiki is not in
archi's v2 byte-parity corpus, and the resulting chunk re-keying on
affected topics was accepted.

Registry-entry templates — same three prerequisites as the
archi/sources/jira.py and archi/sources/docs.py templates: (1) compose
the ``extraction`` module in ``deployment.yaml`` (provides
``document_chunk``; the ``person`` module is not needed by these
sources but is by jira) and copy the packaged schema slice into the
deployment — ``archi/schemas/sources.yaml`` -> ``<deployment>/schemas/``
(provides ``documentation_page``) and
``archi/schemas/bridges/sources.yaml`` ->
``<deployment>/schemas/bridges/`` (provides the
``documentation_page contains document_chunk`` and
``documentation_page references documentation_page`` narrowings these
sources need; narrowings outside ``schemas/bridges/`` are silently
ignored and fail only at ingest); (2) ``output_scope_summary`` must
accompany ``output_signature``; (3) add the standard ``sync:`` block. ::

    twiki_eos:
      module: archi.sources.twiki
      class: TwikiEOSSource
      ownership_id: <instance>.twiki-eos
      admission_policy:
        producer_id: <instance>.twiki-eos
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: twiki_eos
        output_signature:
          nodes:
            - {subtype: documentation_page}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: documentation_page, edge_type: references, dst_subtype: documentation_page}
            - {src_subtype: documentation_page, edge_type: contains, dst_subtype: document_chunk}
            # Uncomment together with the matching params below:
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: cmssw_release}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: site}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: TWiki topics from a local EOS snapshot, their text chunks, and topic-to-topic references
          nodes: [documentation_page, document_chunk]
          edges:
            - documentation_page references documentation_page
            - documentation_page contains document_chunk
            # Uncomment together with the matching params below:
            # - document_chunk references jira_issue
            # - document_chunk references cmssw_release
            # - document_chunk references site
            # - document_chunk references infrastructure_service
      source_class: discovery_crawl
      record_identity_kind: scoped_locator
      record_identity_fields: [path]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [TWIKI_EOS_ROOT]
      required_for_baseline: false
      params:
        eos_root_env: TWIKI_EOS_ROOT
        web_root: CMS
        required: false
        # Optional seeded subtree instead of the whole snapshot:
        # seed_topics: [CMS/CompOpsHome]
        # max_depth: 2
        # Optional reference-target caches — commented out on purpose.
        # WARNING: enabling any of them requires (a) uncommenting the
        # matching document_chunk references edges in output_signature
        # AND output_scope_summary above, and (b) the target subtype +
        # narrowing in the deployment schema: jira_issue ships in
        # archi/schemas/sources.yaml and cmssw_release in
        # archi/schemas/operations.yaml with narrowings in
        # archi/schemas/bridges/sources.yaml; site and
        # infrastructure_service ship in the catalogs port
        # (archi/schemas/operations.yaml + bridges/operations.yaml).
        # sites_path: data/cric/sites.json
        # releases_path: data/cmssw-releases/records.json
        # jira_records_path: data/jira/records.json
        # services_path: data/cric-core/services.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete

    twiki_crawl:
      module: archi.sources.twiki
      class: TwikiCrawlSource
      ownership_id: <instance>.twiki-crawl
      admission_policy:
        producer_id: <instance>.twiki-crawl
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: twiki_crawl
        output_signature:
          nodes:
            - {subtype: documentation_page}
            - {subtype: document_chunk}
          edges:
            - {src_subtype: documentation_page, edge_type: references, dst_subtype: documentation_page}
            - {src_subtype: documentation_page, edge_type: contains, dst_subtype: document_chunk}
            # Uncomment together with the matching params below:
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: cmssw_release}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: site}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: seeded live TWiki topics, their text chunks, and topic-to-topic references
          nodes: [documentation_page, document_chunk]
          edges:
            - documentation_page references documentation_page
            - documentation_page contains document_chunk
            # Uncomment together with the matching params below:
            # - document_chunk references jira_issue
            # - document_chunk references cmssw_release
            # - document_chunk references site
            # - document_chunk references infrastructure_service
      source_class: discovery_crawl
      record_identity_kind: scoped_locator
      record_identity_fields: [path]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      # Only for SSO-protected webs; public webs need no credential.
      credential_refs: [CERN_TWIKI_COOKIE_FILE]
      required_for_baseline: false
      params:
        source_name: twiki_crawl
        base_url: https://twiki.cern.ch/twiki
        seed_topics: [CMSPublic/SWGuideCrab]
        max_depth: 1
        # cookie_file_env: CERN_TWIKI_COOKIE_FILE  # SSO-protected webs only
        # max_pages: 250
        # Optional reference-target caches: same warning as twiki_eos.
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
import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
    ConnectorRun,
)
from okg.deployment import ContentHashProbe
from okg.deployment import file_preflight

from archi.auth.cache import (
    cache_or_forced_live_change_probe,
    resolve_repo_path,
)
from archi.auth.cookies import check_cookie_file, load_cookie_jar_from_env
from archi.sources._twiki_parse import (
    DEFAULT_SKIP_PATTERNS,  # noqa: F401  (re-exported: part of the core API)
    canonical_twiki_url,
    extract_bare_wikiwords,
    extract_wiki_links,
    is_real_page,
    parse_meta,
    strip_twiki,
    topic_page_id,
    twiki_node_id,
    twiki_page_id_from_url,
)
from archi.sources.docs import (
    _chunks,
    _login_bounce,
    _reference_edges,
    _reference_targets,
    _sha256,
)

DEFAULT_EOS_ROOT_ENV = "TWIKI_EOS_ROOT"
DEFAULT_EOS_WEB_ROOT = "CMS"
DEFAULT_EOS_VIEW_BASE_URL = "https://twiki.cern.ch/twiki/bin/view"
DEFAULT_EOS_CHUNKER_NAME = "cms_twiki_window_v1"
DEFAULT_CRAWL_CHUNKER_NAME = "archi_twiki_crawl_v1"


@dataclass(frozen=True)
class TwikiRecord:
    """One parsed TWiki topic (shared by both ingestors)."""

    page_id: str
    web_name: str
    title: str
    url: str
    source_path: str
    body: str
    last_modified: str = ""
    author: str = ""
    parent_topic: str = ""
    version: str = ""
    web_root: str = ""
    wiki_links: tuple[str, ...] = ()
    bare_wikiwords: tuple[str, ...] = ()

    @property
    def node_id(self) -> str:
        return twiki_node_id(self.page_id)


@dataclass(frozen=True)
class _EOSSeedWalk:
    """One seeded snapshot walk: found records + absent seed topics."""

    records: tuple[TwikiRecord, ...]
    missing_seeds: tuple[str, ...]


class TwikiEOSSource:
    """Read TWiki topics from a local EOS snapshot mirror.

    Lift of the cms ``TwikiEOSSource``: whole-tree ``rglob("*.txt")``
    by default, or a seeded subtree when ``seed_topics`` is given
    (links are followed through the snapshot up to ``max_depth``,
    silently skipping followed-link targets whose files are absent —
    the wisdqm snapshot-crawl behavior — while a configured *seed*
    whose file is absent fails the run's scope loudly).
    ``web_root`` un-hardcodes the cms "CMS" prefix; the
    dropped cms operator-path defaults mean the root must arrive via
    the ``eos_root`` param or the env var named by ``eos_root_env``.
    Reference-target caches are optional; a configured-but-missing
    cache raises (docs.py's ``_configured_json`` contract).
    """

    name = "twiki_eos"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        eos_root: str | None = None,
        eos_root_env: str = DEFAULT_EOS_ROOT_ENV,
        web_root: str = DEFAULT_EOS_WEB_ROOT,
        required: bool = False,
        max_files: int | None = None,
        seed_topics: list[str] | tuple[str, ...] | None = None,
        max_depth: int = 1,
        sites_path: str | None = None,
        releases_path: str | None = None,
        jira_records_path: str | None = None,
        services_path: str | None = None,
        view_base_url: str = DEFAULT_EOS_VIEW_BASE_URL,
        chunker_name: str = DEFAULT_EOS_CHUNKER_NAME,
        heading_style: str = "drop",
        whitespace: str = "collapse",
        preserve_tables: bool = False,
        preserve_images: bool = False,
        skip_patterns: tuple[str, ...] | list[str] = DEFAULT_SKIP_PATTERNS,
        records: list[TwikiRecord] | None = None,
        base: str | None = None,
    ) -> None:
        self.eos_root = _resolve_root(eos_root, eos_root_env)
        self.eos_root_env = eos_root_env
        self.web_root = web_root.strip("/")
        self.required = required
        if max_files is not None and max_files < 1:
            raise ValueError(
                f"max_files must be >= 1 (or None for unlimited), "
                f"got {max_files}"
            )
        self.max_files = max_files
        if seed_topics is not None and not seed_topics:
            raise ValueError(
                "seed_topics must be a non-empty list (or None for the "
                "whole snapshot)"
            )
        self.seed_topics = (
            tuple(self._seed_page_id(seed) for seed in seed_topics)
            if seed_topics is not None else None
        )
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        self.max_depth = max_depth
        self.sites_path = sites_path
        self.releases_path = releases_path
        self.jira_records_path = jira_records_path
        self.services_path = services_path
        self.view_base_url = view_base_url.rstrip("/")
        self.chunker_name = chunker_name
        self.heading_style = heading_style
        self.whitespace = whitespace
        self.preserve_tables = preserve_tables
        self.preserve_images = preserve_images
        self.skip_patterns = tuple(skip_patterns)
        self._records = records
        self.base = base
        self.change_probe = ContentHashProbe(
            content_items=self._probe_content_items,
            config={
                "eos_root": self.eos_root,
                "eos_root_env": self.eos_root_env,
                "web_root": self.web_root,
                "max_files": self.max_files,
                "seed_topics": list(self.seed_topics or ()),
                "max_depth": self.max_depth,
                "skip_patterns": list(self.skip_patterns),
                "sites_path": self.sites_path,
                "releases_path": self.releases_path,
                "jira_records_path": self.jira_records_path,
                "services_path": self.services_path,
            },
            emit_targets=TwikiEOSSource,
        )

    def _probe_content_items(self) -> list[tuple[str, Any]]:
        auxiliary_items = self._probe_aux_items()
        if self._records is not None:
            return [
                (
                    record.page_id,
                    json.dumps(asdict(record), sort_keys=True).encode("utf-8"),
                )
                for record in self._records
            ] + auxiliary_items
        if not self.eos_root:
            return auxiliary_items
        root = Path(self.eos_root).expanduser()
        if not root.is_dir():
            return auxiliary_items
        return [
            (path.relative_to(root).as_posix(), path)
            for path in self._paths()
        ] + auxiliary_items

    def _probe_aux_items(self) -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        for raw in (
            self.sites_path,
            self.releases_path,
            self.jira_records_path,
            self.services_path,
        ):
            if not raw:
                continue
            path = resolve_repo_path(raw, base=self.base)
            if path.is_file():
                out.append((f"aux:{raw}", path))
        return out

    def preflight(self, mode: str = "live") -> PreflightResult:
        if self._records is not None:
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="fixture",
                required=self.required,
                record_count=len(self._records),
                reason="in-memory TWiki records supplied",
                checked_at=_checked_at(),
            )
        root = Path(self.eos_root).expanduser() if self.eos_root else None
        if root is not None and root.is_dir():
            credential_refs = (
                (self.eos_root_env,)
                if os.environ.get(self.eos_root_env) else
                ()
            )
            if self.seed_topics is None:
                files = self._paths()
                reason = "local TWiki EOS snapshot directory present"
            else:
                # Seeded scope: count/hash the seeded closure, not the
                # whole tree, so preflight describes what run() emits.
                walk = self._seed_walk(root)
                files = [root / record.source_path for record in walk.records]
                reason = (
                    f"local TWiki EOS snapshot directory present; seeded "
                    f"closure: {len(files)} topics from "
                    f"{len(self.seed_topics)} seeds at max depth "
                    f"{self.max_depth}"
                )
                if walk.missing_seeds:
                    reason += (
                        f" ({len(walk.missing_seeds)} seeds missing from "
                        "the snapshot)"
                    )
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="filesystem",
                required=self.required,
                credential_refs=credential_refs,
                cache_path=str(root),
                record_count=len(files),
                content_hash=_listing_hash(root, files),
                reason=reason,
                checked_at=_checked_at(),
            )
        return file_preflight(
            self.name,
            self.eos_root_env,
            required=self.required,
            mode=mode,
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        missing_seeds: tuple[str, ...] = ()
        if self._records is not None:
            records = self._records
        else:
            # No preflight() call here: its only run-relevant work for a
            # present snapshot is the tree walk, which run() performs
            # itself — one listing per run, not two.
            root = Path(self.eos_root).expanduser() if self.eos_root else None
            if root is None or not root.is_dir():
                preflight = file_preflight(
                    self.name,
                    self.eos_root_env,
                    required=self.required,
                    mode="live",
                )
                return ConnectorRun(
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
            records, missing_seeds = self._records_from_root(root)
        content_hash = _records_hash(records)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash,
            "n_records": len(records),
            "eos_root": self.eos_root,
        }
        targets = _reference_targets(
            sites_path=self.sites_path,
            releases_path=self.releases_path,
            jira_records_path=self.jira_records_path,
            services_path=self.services_path,
            base=self.base,
        )

        def _facts() -> Iterator[Any]:
            yield from _facts_for_twiki_records(
                records,
                revision,
                targets,
                chunker_name=self.chunker_name,
            )

        if missing_seeds:
            # A configured seed without a snapshot file is a scope
            # failure, never a silent skip: with completed_scope=True
            # the registry's missing_from_completed_scope semantics
            # would retract every topic under the absent seed. (Only
            # operator-configured seeds fail loudly; followed-link
            # targets missing from the snapshot stay silently skipped —
            # wisdqm-parity behavior.)
            total = len(self.seed_topics or ())
            samples = ", ".join(missing_seeds[:3])
            all_missing = len(missing_seeds) == total
            return ConnectorRun(
                facts=_facts(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="cache_missing" if all_missing else "endpoint_failed",
                    mode="filesystem",
                    credential_refs=(self.eos_root_env,),
                    record_count=len(records),
                    content_hash=content_hash,
                    reason=(
                        f"{len(missing_seeds)}/{total} configured seed "
                        f"topics missing from the EOS snapshot, e.g. "
                        f"{samples}; no complete scope claimed"
                    ),
                ),
            )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="filesystem" if self._records is None else "fixture",
                credential_refs=(
                    (self.eos_root_env,) if self._records is None else ()
                ),
                record_count=len(records),
                content_hash=content_hash,
                reason="TWiki EOS snapshot read from local filesystem",
            ),
        )

    def _paths(self) -> list[Path]:
        root = Path(self.eos_root).expanduser()
        paths = [
            path for path in root.rglob("*.txt")
            if path.is_file() and is_real_page(path.name, self.skip_patterns)
        ]
        paths.sort(key=lambda p: p.relative_to(root).as_posix())
        if self.max_files is not None:
            return paths[:self.max_files]
        return paths

    def _records_from_root(
        self, root: Path
    ) -> tuple[list[TwikiRecord], tuple[str, ...]]:
        """(records, missing seed page ids) for the configured scope."""
        if self.seed_topics is None:
            records = [
                self._record_for_path(root, path) for path in self._paths()
            ]
            return records, ()
        walk = self._seed_walk(root)
        return list(walk.records), walk.missing_seeds

    def _seed_walk(self, root: Path) -> _EOSSeedWalk:
        """Seeded subtree walk through the snapshot (wisdqm behavior):
        follow parsed topic references up to ``max_depth``, silently
        skipping followed-link targets whose snapshot files are absent.
        Operator-configured seeds are held to stricter semantics: a
        seed without a snapshot file is returned in ``missing_seeds``
        so run() can fail loudly instead of silently shrinking scope."""
        seed_set = set(self.seed_topics or ())
        queue: deque[tuple[str, int]] = deque(
            (page_id, 0) for page_id in self.seed_topics or ()
        )
        seen: set[str] = set()
        records: list[TwikiRecord] = []
        missing_seeds: list[str] = []
        while queue:
            page_id, depth = queue.popleft()
            if not page_id or page_id in seen:
                continue
            seen.add(page_id)
            path = self._page_path(root, page_id)
            if path is None:
                if page_id in seed_set:
                    missing_seeds.append(page_id)
                continue
            if not is_real_page(path.name, self.skip_patterns):
                continue
            record = self._record_for_path(root, path)
            records.append(record)
            if depth >= self.max_depth:
                continue
            for _kind, target in _topic_reference_page_ids(record):
                if target and target not in seen:
                    queue.append((target, depth + 1))
        records.sort(key=lambda record: record.source_path)
        return _EOSSeedWalk(
            records=tuple(records),
            missing_seeds=tuple(missing_seeds),
        )

    def _seed_page_id(self, seed: str) -> str:
        """Seed as ``Topic`` / ``Sub/Topic`` / ``Web.Topic`` / view URL
        -> full page id under ``web_root``."""
        seed = seed.strip()
        if "://" in seed:
            page_id = twiki_page_id_from_url(canonical_twiki_url(seed))
        else:
            page_id = seed.replace(".", "/").strip("/")
        if not page_id:
            raise ValueError(f"unusable TWiki seed topic: {seed!r}")
        if not self.web_root:
            return page_id
        if page_id == self.web_root:
            raise ValueError(
                f"seed {seed!r} names the web root, not a topic"
            )
        if page_id.startswith(f"{self.web_root}/"):
            return page_id
        return f"{self.web_root}/{page_id}"

    def _page_path(self, root: Path, page_id: str) -> Path | None:
        rel = page_id
        if self.web_root:
            if not page_id.startswith(f"{self.web_root}/"):
                return None
            rel = page_id[len(self.web_root) + 1:]
        path = root / f"{rel}.txt"
        return path if path.is_file() else None

    def _record_for_path(self, root: Path, path: Path) -> TwikiRecord:
        raw = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root).as_posix()
        rel_no_ext = rel.rsplit(".", 1)[0] if "." in rel else rel
        page_id = (
            f"{self.web_root}/{rel_no_ext}" if self.web_root else rel_no_ext
        )
        web_name = (
            page_id.rsplit("/", 1)[0] if "/" in page_id else page_id
        )
        return _record_from_raw(
            raw,
            page_id=page_id,
            web_name=web_name,
            web_root=self.web_root,
            title=path.stem,
            url=f"{self.view_base_url}/{page_id}",
            source_path=rel,
            heading_style=self.heading_style,
            whitespace=self.whitespace,
            preserve_tables=self.preserve_tables,
            preserve_images=self.preserve_images,
        )


class TwikiCrawlSource:
    """Seeded live TWiki crawler over the raw-text endpoint.

    Fetches ``<base_url>/bin/view/<Web>/<Topic>?raw=all`` (raw topic
    markup including the ``%META`` lines, so the shared parser core
    sees exactly what the EOS snapshot files contain) for the seed
    topics and the explicit references they link to, out to
    ``max_depth``. ``seed_topics`` and ``max_depth`` are required — no
    whole-site sweeps. Bare CamelCase WikiWords are *not* followed live
    (each would be a speculative fetch whose 404 would poison the
    failure count); they still yield reference edges between crawled
    topics. Public webs need no credential; ``cookie_file_env`` names
    an env var holding an SSO cookie-file *path* (never a value) for
    protected webs.
    """

    name = "twiki_crawl"
    profile = "discovery_crawl"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        base_url: str,
        seed_topics: list[str] | tuple[str, ...],
        max_depth: int,
        source_name: str = "twiki_crawl",
        cookie_file_env: str | None = None,
        cookie_max_age_hours: float | None = None,
        max_pages: int | None = None,
        timeout: int = 30,
        sites_path: str | None = None,
        releases_path: str | None = None,
        jira_records_path: str | None = None,
        services_path: str | None = None,
        chunker_name: str = DEFAULT_CRAWL_CHUNKER_NAME,
        heading_style: str = "markdown",
        whitespace: str = "preserve",
        preserve_tables: bool = True,
        preserve_images: bool = True,
        skip_patterns: tuple[str, ...] | list[str] = DEFAULT_SKIP_PATTERNS,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.base_url = base_url.rstrip("/")
        self.seed_topics = tuple(
            _crawl_seed_page_id(seed) for seed in seed_topics
        )
        if not self.seed_topics:
            raise ValueError(
                "TwikiCrawlSource requires at least one seed topic"
            )
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        self.max_depth = max_depth
        self.cookie_file_env = cookie_file_env
        self.cookie_max_age_hours = cookie_max_age_hours
        if max_pages is not None and max_pages < 1:
            raise ValueError(
                f"max_pages must be >= 1 (or None for unlimited), "
                f"got {max_pages}"
            )
        self.max_pages = max_pages
        self.timeout = timeout
        self.sites_path = sites_path
        self.releases_path = releases_path
        self.jira_records_path = jira_records_path
        self.services_path = services_path
        self.chunker_name = chunker_name
        self.heading_style = heading_style
        self.whitespace = whitespace
        self.preserve_tables = preserve_tables
        self.preserve_images = preserve_images
        self.skip_patterns = tuple(skip_patterns)
        self.base = base
        self.change_probe = cache_or_forced_live_change_probe(
            cache_paths=(),
            config={
                "source_name": self.name,
                "base_url": self.base_url,
                "seed_topics": list(self.seed_topics),
                "max_depth": self.max_depth,
                "cookie_file_env": self.cookie_file_env,
                "max_pages": self.max_pages,
                "skip_patterns": list(self.skip_patterns),
            },
            emit_targets=TwikiCrawlSource,
            base=base,
        )

    def _credential_refs(self) -> tuple[str, ...]:
        return (self.cookie_file_env,) if self.cookie_file_env else ()

    def preflight(self, mode: str = "live") -> PreflightResult:
        if not self.cookie_file_env:
            return PreflightResult(
                source_name=self.name,
                status="ok",
                mode="live",
                required=False,
                reason=(
                    f"public TWiki crawl: {len(self.seed_topics)} seed "
                    f"topics, max depth {self.max_depth}; no SSO cookie "
                    "configured"
                ),
                checked_at=_checked_at(),
            )
        cookie_file = os.environ.get(self.cookie_file_env, "")
        if not cookie_file or not Path(cookie_file).is_file():
            return PreflightResult(
                source_name=self.name,
                status="missing_credential",
                mode="live",
                required=False,
                credential_refs=self._credential_refs(),
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
            credential_refs=self._credential_refs(),
            cache_path=cookie_file,
            reason=f"SSO cookie file: {status.reason}",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        session = self._session()
        if session is None:
            # A missing/unparseable cookie is an auth failure, not an
            # empty-but-complete crawl: with completed_scope=True the
            # registry's missing_from_completed_scope semantics would
            # retract every previously ingested topic.
            return ConnectorRun(
                facts=(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="auth_failed",
                    mode="live",
                    credential_refs=self._credential_refs(),
                    record_count=0,
                    content_hash=None,
                    reason=(
                        f"SSO cookie file referenced by "
                        f"{self.cookie_file_env} is missing or unreadable; "
                        "no topics fetched and no complete scope claimed"
                    ),
                    checked_at=_checked_at(),
                ),
            )
        crawl = self._crawl(session)
        records = list(crawl.records)
        content_hash = _records_hash(records)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash,
            "n_records": len(records),
            "base_url": self.base_url,
        }
        targets = _reference_targets(
            sites_path=self.sites_path,
            releases_path=self.releases_path,
            jira_records_path=self.jira_records_path,
            services_path=self.services_path,
            base=self.base,
        )

        def _facts() -> Iterator[Any]:
            yield from _facts_for_twiki_records(
                records,
                revision,
                targets,
                chunker_name=self.chunker_name,
            )

        missing_note = (
            f"; {len(crawl.missing_link_targets)} dangling wiki-link "
            "targets skipped (missing_link_targets)"
            if crawl.missing_link_targets else ""
        )
        truncation_note = (
            f"crawl truncated at max_pages={self.max_pages} with "
            f"{crawl.truncated_queued} queued"
            if crawl.truncated_queued else ""
        )
        if crawl.failed_urls:
            # A partially failed crawl must never claim a complete scope
            # (missing_from_completed_scope would retract the failed
            # topics' records); emit what succeeded and report the rest.
            samples = ", ".join(crawl.failed_urls[:3])
            trailer = f"; {truncation_note}" if truncation_note else ""
            return ConnectorRun(
                facts=_facts(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="endpoint_failed",
                    mode="live",
                    credential_refs=self._credential_refs(),
                    record_count=len(records),
                    content_hash=content_hash,
                    reason=(
                        f"partial crawl: {len(crawl.failed_urls)}/"
                        f"{crawl.total_topics} seeded TWiki topics failed "
                        f"(fetch error or SSO login bounce), e.g. {samples}"
                        f"{missing_note}{trailer}; "
                        "no complete scope claimed"
                    ),
                ),
            )
        if crawl.truncated_queued:
            # max_pages stopped the crawl with work still queued: every
            # fetch succeeded, but the scope was not covered, so it must
            # not be claimed complete in any mode (the queued topics'
            # previously ingested records would be retracted under
            # missing_from_completed_scope).
            return ConnectorRun(
                facts=_facts(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="ok",
                    mode="live",
                    credential_refs=self._credential_refs(),
                    record_count=len(records),
                    content_hash=content_hash,
                    reason=(
                        f"{truncation_note}{missing_note}; "
                        "no complete scope claimed"
                    ),
                ),
            )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="live",
                credential_refs=self._credential_refs(),
                record_count=len(records),
                content_hash=content_hash,
                reason=(
                    "seeded TWiki topics fetched from the raw-text endpoint"
                    f"{missing_note}"
                ),
            ),
        )

    def _session(self) -> requests.Session | None:
        """Crawl session; ``None`` only when a configured cookie file is
        missing or unparseable (public crawls need no cookie)."""
        if not self.cookie_file_env:
            return requests.Session()
        jar = load_cookie_jar_from_env(self.cookie_file_env)
        if jar is None:
            return None
        session = requests.Session()
        session.cookies = jar
        return session

    def _crawl(self, session: requests.Session) -> _TwikiCrawlOutcome:
        seed_set = set(self.seed_topics)
        queue: deque[tuple[str, int]] = deque(
            (page_id, 0) for page_id in self.seed_topics
        )
        seen: set[str] = set()
        records: list[TwikiRecord] = []
        failed_urls: list[str] = []
        missing_link_targets: list[str] = []
        truncated_queued = 0
        while queue:
            page_id, depth = queue.popleft()
            if not page_id or page_id in seen:
                continue
            if self.max_pages is not None and len(seen) >= self.max_pages:
                # Record how many distinct topics were still queued so
                # run() can refuse to claim a complete scope.
                remaining = {
                    queued_id for queued_id, _ in queue
                    if queued_id and queued_id not in seen
                }
                remaining.add(page_id)
                truncated_queued = len(remaining)
                break
            seen.add(page_id)
            view_url = f"{self.base_url}/bin/view/{page_id}"
            fetch_url = f"{view_url}?raw=all"
            try:
                resp = session.get(fetch_url, timeout=self.timeout)
            except Exception:  # noqa: BLE001
                failed_urls.append(view_url)
                continue
            # ?raw=all bodies frequently ship without a charset in the
            # Content-Type header; requests would then guess ISO-8859-1
            # (RFC default) and mangle UTF-8. Force UTF-8 unless the
            # server declared a charset (requests decodes .text with
            # errors="replace" semantics).
            headers = getattr(resp, "headers", None) or {}
            if "charset" not in str(headers.get("content-type", "")).lower():
                resp.encoding = "utf-8"
            final_url = str(getattr(resp, "url", fetch_url) or fetch_url)
            text = getattr(resp, "text", "") or ""
            if (
                getattr(resp, "status_code", 0) == 404
                or _twiki_oops_page(final_url, text)
            ):
                # An absent topic (404 or a TWiki oops page) reached by
                # following a link is a dangling wiki link, not a crawl
                # failure: record it, emit nothing (never ingest the
                # oops body), and keep it out of failed_urls. Seeds are
                # operator configuration and keep strict semantics.
                if page_id in seed_set:
                    failed_urls.append(view_url)
                else:
                    missing_link_targets.append(view_url)
                continue
            try:
                resp.raise_for_status()
            except Exception:  # noqa: BLE001
                # Real transport-level errors (5xx and other non-404
                # statuses) are failures even on followed links.
                failed_urls.append(view_url)
                continue
            if _login_bounce(fetch_url, final_url, text):
                failed_urls.append(view_url)
                continue
            record = _record_from_raw(
                text,
                page_id=page_id,
                web_name=page_id.rsplit("/", 1)[0],
                web_root="",
                title=page_id.rsplit("/", 1)[-1],
                url=view_url,
                source_path=page_id,
                heading_style=self.heading_style,
                whitespace=self.whitespace,
                preserve_tables=self.preserve_tables,
                preserve_images=self.preserve_images,
            )
            records.append(record)
            if depth >= self.max_depth:
                continue
            for kind, target in _topic_reference_page_ids(record):
                if kind == "bare_wikiword":
                    # Never fetch speculative CamelCase words live; a
                    # 404 per guess would poison the failure count.
                    continue
                if not target or target in seen:
                    continue
                if not is_real_page(
                    f"{target.rsplit('/', 1)[-1]}.txt", self.skip_patterns
                ):
                    continue
                queue.append((target, depth + 1))
        return _TwikiCrawlOutcome(
            records=tuple(records),
            failed_urls=tuple(failed_urls),
            missing_link_targets=tuple(missing_link_targets),
            total_topics=len(seen),
            truncated_queued=truncated_queued,
        )


@dataclass(frozen=True)
class _TwikiCrawlOutcome:
    """One seeded crawl: parsed records plus what could not be covered.

    ``failed_urls`` are real failures (transport errors, non-404 HTTP
    errors, SSO login bounces, and absent *seed* topics);
    ``missing_link_targets`` are dangling wiki links (404/oops on a
    followed link) that are excluded from failure semantics;
    ``truncated_queued`` counts distinct topics still queued when the
    crawl stopped at ``max_pages`` (0 = not truncated).
    """

    records: tuple[TwikiRecord, ...]
    failed_urls: tuple[str, ...]
    missing_link_targets: tuple[str, ...]
    total_topics: int
    truncated_queued: int = 0


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_root(explicit: str | None, env_ref: str) -> str:
    """EOS root from the explicit param, else the env var, else empty.

    The cms original also fell back to hardcoded operator paths
    (``/eos/cms/...``, ``~/data/twiki-cms``); those defaults are
    dropped — instances configure the root explicitly.
    """
    if explicit:
        return os.path.expanduser(explicit)
    value = os.environ.get(env_ref)
    if value:
        return os.path.expanduser(value)
    return ""


_OOPS_URL_MARKER = "/bin/oops/"
_OOPS_BODY_MARKERS = ("topicdoesnotexist", "oopsmissing")


def _twiki_oops_page(final_url: str, body: str) -> bool:
    """True when a ``?raw=all`` response is a TWiki 'oops' error page
    (topic does not exist), not raw topic markup.

    twiki.cern.ch could not be probed unauthenticated (every request
    bounces to CERN SSO first, which ``_login_bounce`` already
    handles); stock TWiki serves a missing topic either as an HTTP 404
    (checked by the caller before this function) or as a redirect to
    ``/bin/oops/<Web>/<Topic>?template=oopsmissing`` / a small oops
    body naming the ``TopicDoesNotExist``/``oopsmissing`` template, so
    both shapes are detected here. Raw topic markup always carries a
    ``%META:TOPICINFO`` line, which oops pages never do — that guards
    real topics that merely mention the marker strings in their text.
    """
    if _OOPS_URL_MARKER in urlparse(final_url).path.lower():
        return True
    lowered = body.lower()
    if "%meta:topicinfo" in lowered:
        return False
    return any(marker in lowered for marker in _OOPS_BODY_MARKERS)


def _crawl_seed_page_id(seed: str) -> str:
    """Crawl seed as ``Web/Topic`` / ``Web.Topic`` / view URL -> page id."""
    seed = seed.strip()
    if "://" in seed:
        page_id = twiki_page_id_from_url(canonical_twiki_url(seed))
    else:
        page_id = seed.replace(".", "/").strip("/")
    if not page_id or "/" not in page_id:
        raise ValueError(
            f"crawl seed topics must be Web/Topic (or a TWiki view URL), "
            f"got: {seed!r}"
        )
    return page_id


def _record_from_raw(
    raw: str,
    *,
    page_id: str,
    web_name: str,
    web_root: str,
    title: str,
    url: str,
    source_path: str,
    heading_style: str,
    whitespace: str,
    preserve_tables: bool,
    preserve_images: bool,
) -> TwikiRecord:
    meta = parse_meta(raw)
    return TwikiRecord(
        page_id=page_id,
        web_name=web_name,
        web_root=web_root,
        title=title,
        url=url,
        source_path=source_path,
        body=strip_twiki(
            raw,
            heading_style=heading_style,
            whitespace=whitespace,
            preserve_tables=preserve_tables,
            preserve_images=preserve_images,
        ),
        last_modified=meta["last_modified"],
        author=meta["author"],
        parent_topic=meta["parent_topic"],
        version=meta["version"],
        wiki_links=tuple(sorted(extract_wiki_links(raw))),
        bare_wikiwords=tuple(sorted(extract_bare_wikiwords(raw))),
    )


def _topic_reference_page_ids(
    record: TwikiRecord,
) -> Iterator[tuple[str, str]]:
    """(kind, target page id) pairs, in the cms dedup-priority order."""
    if record.parent_topic:
        yield "topic_parent", topic_page_id(
            record.parent_topic,
            web_name=record.web_name,
            web_root=record.web_root,
        )
    for target in record.wiki_links:
        yield "wiki_link", topic_page_id(
            target, web_name=record.web_name, web_root=record.web_root
        )
    for target in record.bare_wikiwords:
        yield "bare_wikiword", topic_page_id(
            target, web_name=record.web_name, web_root=record.web_root
        )


def _facts_for_twiki_records(
    records: list[TwikiRecord],
    revision: dict[str, Any],
    targets: dict[str, Any],
    *,
    chunker_name: str,
) -> Iterator[NodeFact | EdgeFact]:
    known_node_ids = {record.node_id for record in records}
    for record in records:
        yield _page_node(record, revision)
        yielded_targets: set[str] = set()
        for kind, target_page_id in _topic_reference_page_ids(record):
            target = twiki_node_id(target_page_id) if target_page_id else ""
            if (
                not target
                or target == record.node_id
                or target not in known_node_ids
                or target in yielded_targets
            ):
                continue
            yielded_targets.add(target)
            yield EdgeFact(
                src=record.node_id,
                dst=target,
                edge_type="references",
                provenance="derived_deterministic",
                attrs={"kind": kind},
                source_record_id={"path": record.source_path},
                source_revision=revision,
            )
        full_text = " ".join(
            part for part in [record.title, record.body] if part
        )
        for chunk_index, offset, chunk_text in _chunks(full_text):
            # Parity wart kept on purpose: a literal backslash-zero
            # separator (not NUL), as in the cms original and jira.py.
            chunk_seed = f"{record.node_id}\\0{chunk_index}\\0{chunk_text}"
            chunk_id = f"chunk:{_sha256(chunk_seed)[:16]}"
            chunk_record_id = {
                "path": record.source_path,
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


def _page_node(record: TwikiRecord, revision: dict[str, Any]) -> NodeFact:
    text = " ".join(
        filter(None, [record.title, record.parent_topic, record.web_name])
    )
    return NodeFact(
        node_id=record.node_id,
        subtype="documentation_page",
        attrs={
            "label": record.title,
            "title": record.title,
            "url": record.url,
            "source_path": record.source_path,
            "source_repo": "",
            "site_name": urlparse(record.url).netloc,
            "web_name": record.web_name,
            "last_updated": record.last_modified,
            "author": record.author,
            "parent_topic": record.parent_topic,
            "version": record.version,
            "text": text,
        },
        source_record_id={"path": record.source_path},
        source_revision=revision,
    )


def _listing_hash(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}\0{stat.st_size}\0{int(stat.st_mtime)}\n".encode())
    return digest.hexdigest()


def _records_hash(records: list[TwikiRecord]) -> str:
    payload = json.dumps(
        [
            {
                "page_id": record.page_id,
                "source_path": record.source_path,
                "last_modified": record.last_modified,
                "body_hash": _sha256(record.body),
            }
            for record in records
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(payload)
