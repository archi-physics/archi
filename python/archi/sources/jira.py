"""Cache-backed JIRA issue source (records + meta cache files).

Rewritten for the archi v3 package (req.w2.sources-parity) as a merge
of two lineages:

- okg-deployments ``cms/cms_sources/jira.py`` (``JiraIssueSource``,
  505 LOC) at ``main@f33a9c4`` — the record parsing, node/edge/chunk
  emission, and cache-backed run/preflight shape.
- archi v2 at ``main@9c9e1cb0``:
  ``src/utils/jira.py`` (26 LOC) — :func:`parse_jira_project_keys` and
  :func:`quote_jql_string`, folded in verbatim as module functions —
  and ``src/data_manager/collectors/tickets/integrations/jira.py``
  (``JiraClient`` collector, 236 LOC), from which two behaviors the
  cms source lacked are taken: the ``<base_url>/browse/<key>`` issue
  URL (now a ``url`` attr on every ``jira_issue`` node) and the JQL
  query construction for live fetches (:func:`issues_jql` /
  :func:`format_jira_datetime`, rewritten from the collector's
  ``project=X and created >= ... and updated >= ...`` builder with
  quoting via :func:`quote_jql_string`).

Deliberately **not** ported:

- ``src/interfaces/jira.py`` — that is the v2 *service*, not a source.
- The v2 collector's anonymizer. The collector optionally ran
  ``Anonymizer().anonymize(issue_text)`` over the built issue text
  (config flag ``anonymize_data``) before storage. In v3 that concern
  moves to the enrichment port (``archi/enrichment``, dev-branch
  anonymizer per the porting matrix); the hook point to preserve there
  is the text surface this module emits — ``jira_issue.attrs`` text
  fields and ``document_chunk.attrs["text"]`` — before embedding.
  The cms source's narrower, always-on email removal *is* kept: every
  string read from the cache goes through :func:`_pg_text`, which
  calls :func:`archi.enrichment.anonymizer.redact_email_addresses`.
- The live JIRA fetch itself: like the cms original this source reads
  an externally maintained records cache; credentials stay env-var
  references only.

Changes from the cms original:

- De-CMS-ified: cache paths, JIRA ``base_url`` (default CERN JIRA,
  ``https://its.cern.ch/jira``), ``project_keys``, and the credential
  ref/aliases (default ``CERN_JIRA_TOKEN`` / alias ``JIRA_CERN_TOKEN``)
  are constructor parameters. The hardcoded CMS project-key regex is
  replaced by :func:`issue_key_pattern` built from ``project_keys``
  (validated with :func:`parse_jira_project_keys`); without
  ``project_keys`` extraction is bounded to the project prefixes of
  the issue keys present in the records cache (a bare generic
  ``KEY-123`` pattern would match strings like ``COVID-19``).
- Change probe: the original declared ``change_probe_kind =
  "content_hash"`` over its own cache, which silently skips when the
  cache is absent. Per the W2 porting matrix this mutable upstream now
  uses ``cache_or_forced_live_change_probe`` (``mutable_api``): cache
  present -> content-hash token, cache absent -> fresh token.
- Cache helpers come from :mod:`archi.auth.cache` (explicit ``base``);
  chunking/reference emission is shared with :mod:`archi.sources.docs`
  (same direction as the original import).
- The reference-target caches (sites/releases/services) are optional
  parameters (no CMS data-path defaults); issue self-references use
  this source's own ``records_path``.
- ``chunker_name`` is a parameter, default kept at
  ``cms_jira_window_v1`` so the parity corpus does not churn.
- Parity wart kept on purpose: chunk ids hash their input with a
  literal backslash-zero separator (the original's ``f'..\\0..'``
  inside an f-string), *not* the NUL byte docs.py uses. Changing it
  would re-key every jira chunk at cutover.
- 2026-08 adversarial review (see
  ``pact/changes/circleback-fixes/notes-live.md``): ``run()`` now
  cross-checks the parsed record count against ``meta.json``'s
  ``record_count`` (the original consulted the meta file only in the
  cache-missing preflight branch). On a mismatch the cache is treated
  as truncated/stale: facts are still emitted, health is
  ``endpoint_failed`` (the closed status vocabulary has no
  ``degraded``) with both counts in the reason, and the run never
  claims ``completed_scope`` — a truncated cache claimed complete
  would retract every record it happened to drop.
- 2026-09 (empty-cache-fails-loudly): a records cache that is
  unreadable, zero bytes, truncated JSON or an empty list is
  ``cache_missing`` with no facts and no complete scope, whatever
  ``meta.json`` says — a complete scope over zero issues retracts every
  issue. Items without an issue key are skipped per item but stop the
  run claiming a complete scope. ``preflight()`` and ``run()`` share
  one verdict, so preflight now also reports the meta mismatch.
- 2026-10 (jira-comment-records): with ``comment_records=True`` each
  comment is also a ``jira_comment`` record (``jira_issue has_comment
  jira_comment``, ``jira_comment authored_by person``) named by a hash
  of its content (its time taken as a UTC instant), and the issue
  carries ``comment_count``. Comments are not in the full-text index;
  the chunks stay the search unit for comment text. The issue
  node always carries the fields that were chunk-only before
  (environment, fix_versions, parent_key, issue_links, subtasks). The
  chunks are byte-identical either way.

Registry-entry template — INGEST-PROVEN 2026-08-11 on a scratch
instance (okg dev@21c5b8c3e). Three things beyond the registry entry
are required or ingest fails after a green lint:

1. Ontology: compose the ``person`` and ``extraction`` modules in
   ``deployment.yaml`` (they provide ``person``/``document_chunk``),
   and copy the packaged schema slice into the deployment —
   ``archi/schemas/sources.yaml`` -> ``<deployment>/schemas/`` and
   ``archi/schemas/bridges/sources.yaml`` ->
   ``<deployment>/schemas/bridges/`` (narrowings placed in plain
   ``schemas/*.yaml`` are silently ignored by the catalog composer;
   the failure only surfaces at ingest as ProducerPolicyViolation).
2. ``output_scope_summary`` must accompany ``output_signature``
   (lint blocker ``admission_policy_block_drift`` otherwise).
3. A ``sync:`` block (triggers: [manual, reconcile],
   default_event_mode/reconcile_mode: scope_complete) like every
   other entry.

::

    jira:
      module: archi.sources.jira
      class: JiraIssueSource
      ownership_id: <instance>.jira
      admission_policy:
        producer_id: <instance>.jira
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: jira
        output_signature:
          nodes:
            - {subtype: jira_issue}
            - {subtype: person}
            - {subtype: document_chunk}
            # Uncomment together with `comment_records: true`:
            # - {subtype: jira_comment}
          edges:
            - {src_subtype: jira_issue, edge_type: assigned_to, dst_subtype: person}
            - {src_subtype: jira_issue, edge_type: reported_by, dst_subtype: person}
            - {src_subtype: jira_issue, edge_type: contains, dst_subtype: document_chunk}
            - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
            # Uncomment together with `issue_link_edges: true` below:
            # - {src_subtype: jira_issue, edge_type: linked_to, dst_subtype: jira_issue}
            # Uncomment together with `comment_records: true`:
            # - {src_subtype: jira_issue, edge_type: has_comment, dst_subtype: jira_comment}
            # - {src_subtype: jira_comment, edge_type: authored_by, dst_subtype: person}
            # Uncomment together with the matching params below:
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: cmssw_release}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: site}
            # - {src_subtype: document_chunk, edge_type: references, dst_subtype: infrastructure_service}
        output_scope_summary:
          summary: JIRA issues, participants, and issue-text chunks from the records cache
          nodes: [jira_issue, person, document_chunk]
          edges:
            - jira_issue assigned_to person
            - jira_issue reported_by person
            - jira_issue contains document_chunk
            - document_chunk references jira_issue
            # Uncomment together with `issue_link_edges: true` below:
            # - jira_issue linked_to jira_issue
            # Uncomment together with `comment_records: true`:
            # - jira_issue has_comment jira_comment
            # - jira_comment authored_by person
            # Uncomment together with the matching params below:
            # - document_chunk references cmssw_release
            # - document_chunk references site
            # - document_chunk references infrastructure_service
      source_class: mutable_api
      record_identity_kind: remote_id
      record_identity_fields: [issue_key]
      source_revision_kind: updated_at
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      credential_refs: [CERN_JIRA_TOKEN]
      credential_aliases:
        CERN_JIRA_TOKEN: [JIRA_CERN_TOKEN]
      required_for_baseline: true
      params:
        records_path: data/jira/records.json
        meta_path: data/jira/meta.json
        base_url: https://its.cern.ch/jira
        project_keys: [CMSCOMPPR, CMSPROD, CMSRUCIO, CMSALCA, CMSDM,
                       CMSTRANSF, CMSMONIT, CMSVOC, CMSTZ, PRCAMPAIGNS]
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
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Sequence

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
    ConnectorRun,
)

from archi.enrichment.anonymizer import redact_email_addresses
from archi.sources._cache_report import (
    CacheUnusable,
    read_cache_json,
    skipped_items_status,
    unusable_cache_preflight,
    unusable_cache_run,
)
from archi.sources._sdk_adapter import ReaderAdapter
from archi.auth.cache import (
    cache_or_forced_live_change_probe,
    content_hash,
    load_json,
    resolve_repo_path,
)
from archi.sources.docs import _chunks, _reference_edges, _reference_targets

DEFAULT_JIRA_BASE_URL = "https://its.cern.ch/jira"
DEFAULT_CHUNKER_NAME = "cms_jira_window_v1"

JIRA_PROJECT_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_GENERIC_ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9_]*-\d+\b")
_ISSUE_KEY_PREFIX_RE = re.compile(r"([A-Z][A-Z0-9_]*)-\d+")
_MATCH_NO_KEY_RE = re.compile(r"(?!)")  # never matches


def parse_jira_project_keys(value: object, error_message: str) -> list[str]:
    """Validate a list of JIRA project keys (from archi v2 src/utils/jira.py)."""
    if not isinstance(value, list):
        raise ValueError(error_message)

    projects = []
    for project in value:
        if not isinstance(project, str):
            raise ValueError(error_message)
        project = project.strip()
        if not project or not JIRA_PROJECT_KEY_PATTERN.fullmatch(project):
            raise ValueError(error_message)
        projects.append(project)

    if not projects:
        raise ValueError(error_message)
    return projects


def quote_jql_string(value: str) -> str:
    """Quote a string literal for JQL (from archi v2 src/utils/jira.py)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_jira_datetime(value: str) -> str:
    """ISO-8601 -> the ``YYYY/MM/DD HH:MM`` form JQL date clauses expect.

    Rewritten from the v2 collector's ``_format_jira_datetime``; invalid
    input raises ``ValueError`` instead of being silently dropped.
    """
    return datetime.fromisoformat(value).strftime("%Y/%m/%d %H:%M")


def issues_jql(
    project_key: str,
    *,
    created_after: str | None = None,
    updated_after: str | None = None,
) -> str:
    """JQL for one project's issues, optionally windowed by ISO datetimes.

    Rewritten from the v2 collector's query builder (cutoff_date /
    since_iso clauses), with the project key validated and quoted.
    """
    key = parse_jira_project_keys(
        [project_key], f"invalid JIRA project key: {project_key!r}"
    )[0]
    parts = [f"project = {quote_jql_string(key)}"]
    if created_after:
        parts.append(f'created >= "{format_jira_datetime(created_after)}"')
    if updated_after:
        parts.append(f'updated >= "{format_jira_datetime(updated_after)}"')
    return " and ".join(parts)


def issue_key_pattern(project_keys: Sequence[str] | None = None) -> re.Pattern[str]:
    """Regex matching issue keys, restricted to *project_keys* when given.

    Replaces the cms original's hardcoded CMS project alternation. With
    no keys the generic pattern matches any ``KEY-123`` token (beware:
    that includes strings like ``COVID-19``; pass project keys to
    restrict extraction).
    """
    if not project_keys:
        return _GENERIC_ISSUE_KEY_RE
    keys = parse_jira_project_keys(
        list(project_keys),
        "project_keys must be a non-empty list of JIRA project keys "
        "([A-Z][A-Z0-9_]*)",
    )
    alternation = "|".join(re.escape(key) for key in keys)
    return re.compile(rf"\b(?:{alternation})-\d+\b")


@dataclass(frozen=True)
class JiraIssueRecord:
    key: str
    summary: str = ""
    description: str = ""
    project: str = ""
    status: str = ""
    priority: str = ""
    issue_type: str = ""
    assignee: str = ""
    reporter: str = ""
    created: str = ""
    updated: str = ""
    environment: str = ""
    resolution: str = ""
    parent_key: str = ""
    comment_count: int = 0
    components: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    fix_versions: tuple[str, ...] = ()
    issue_links: tuple[str, ...] = ()
    subtasks: tuple[str, ...] = ()
    recent_comments: tuple[str, ...] = ()
    # (author, created, body) per comment, the same comments in the same
    # order as recent_comments; feeds the jira_comment records.
    comments: tuple[tuple[str, str, str], ...] = ()
    # (linked key, link type name, "outward" | "inward") for each link
    # whose type and direction the cache records; feeds linked_to edges.
    typed_links: tuple[tuple[str, str, str], ...] = ()

    @property
    def node_id(self) -> str:
        return f"jira:{self.key}"


class JiraIssueSource:
    """Load JIRA issue records from an externally supplied cache file."""

    name = "jira"
    profile = "mutable_api"
    change_probe_kind = "mutable_api"

    def __init__(
        self,
        *,
        records_path: str = "data/jira/records.json",
        meta_path: str = "data/jira/meta.json",
        base_url: str = DEFAULT_JIRA_BASE_URL,
        project_keys: list[str] | None = None,
        credential_ref: str = "CERN_JIRA_TOKEN",
        credential_aliases: list[str] | None = None,
        sites_path: str | None = None,
        releases_path: str | None = None,
        services_path: str | None = None,
        chunker_name: str = DEFAULT_CHUNKER_NAME,
        issue_link_edges: bool = False,
        comment_records: bool = False,
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.meta_path = meta_path
        self.base_url = base_url.rstrip("/")
        self.project_keys = (
            parse_jira_project_keys(
                project_keys,
                "project_keys must be a non-empty list of JIRA project "
                "keys ([A-Z][A-Z0-9_]*)",
            )
            if project_keys is not None
            else None
        )
        self.issue_key_re = issue_key_pattern(self.project_keys)
        self.credential_ref = credential_ref
        self.credential_aliases = tuple(
            credential_aliases
            if credential_aliases is not None
            else ("JIRA_CERN_TOKEN",)
        )
        self.sites_path = sites_path
        self.releases_path = releases_path
        self.services_path = services_path
        self.chunker_name = chunker_name
        # Issue links and sub-task parents as jira_issue linked_to
        # jira_issue edges. Opt-in: the edge is a new output the
        # deployment's output_signature must list, so a deployment that
        # says nothing emits exactly what it emitted before.
        self.issue_link_edges = bool(issue_link_edges)
        # Each comment as a jira_comment record (jira_issue has_comment
        # jira_comment, jira_comment authored_by person). Opt-in for the
        # same reason: new outputs the output_signature must list. The
        # chunks are unchanged either way.
        self.comment_records = bool(comment_records)
        self.base = base
        self.change_probe = cache_or_forced_live_change_probe(
            cache_paths=self.cache_paths,
            config={
                "records_path": self.records_path,
                "meta_path": self.meta_path,
                # Present only when on, so a deployment that never set
                # the option keeps its old config hash.
                **({"issue_link_edges": True}
                   if self.issue_link_edges else {}),
                **({"comment_records": True}
                   if self.comment_records else {}),
            },
            # The comment id code lives in module functions; naming them
            # makes a change to the id scheme change the probe token, so
            # a cursor run over an unchanged cache still re-emits.
            emit_targets=(
                JiraIssueSource,
                _comment_parts,
                _comment_line,
                _created_utc,
                comment_ids,
                _comment_facts,
            ),
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    @property
    def _credential_refs(self) -> tuple[str, ...]:
        return (self.credential_ref,)

    @property
    def _alias_refs(self) -> dict[str, tuple[str, ...]]:
        if not self.credential_aliases:
            return {}
        return {self.credential_ref: self.credential_aliases}

    def preflight(self, mode: str = "live") -> PreflightResult:
        path = resolve_repo_path(self.records_path, base=self.base)
        if not path.is_file():
            expected = _expected_count(self.meta_path, base=self.base)
            reason = f"JIRA records cache file {path} is missing"
            if expected is not None:
                reason += f"; metadata reports {expected} records"
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="cache",
                required=True,
                credential_refs=self._credential_refs,
                alias_refs=self._alias_refs,
                cache_path=str(path),
                reason=reason,
                checked_at=_checked_at(),
            )
        # Same verdict run() reaches for the same cache.
        try:
            records, skipped = self._records_with_skips()
        except CacheUnusable as exc:
            return unusable_cache_preflight(
                exc,
                source_name=self.name,
                required=True,
                credential_refs=self._credential_refs,
                alias_refs=self._alias_refs,
            )
        status, reason, _complete = self._verdict(
            len(records), skipped, "local JIRA records cache present"
        )
        return PreflightResult(
            source_name=self.name,
            status=status,
            mode="cache",
            required=True,
            credential_refs=self._credential_refs,
            alias_refs=self._alias_refs,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason=reason,
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        path = resolve_repo_path(self.records_path, base=self.base)
        if not path.is_file():
            return ConnectorRun(
                facts=(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="cache_missing",
                    mode="cache",
                    credential_refs=self._credential_refs,
                    alias_refs=self._alias_refs,
                    cache_path=str(path),
                    reason=(
                        f"JIRA records cache {path} is absent; no facts "
                        "emitted and no complete scope claimed"
                    ),
                    checked_at=_checked_at(),
                ),
            )
        try:
            records, skipped = self._records_with_skips()
        except CacheUnusable as exc:
            # Unreadable, truncated or empty: a failed fetch, never an
            # empty project. A complete scope over zero issues would
            # retract every issue ingested before, and meta.json cannot
            # vouch for an empty cache.
            return unusable_cache_run(
                exc,
                mode=mode,
                credential_refs=self._credential_refs,
                alias_refs=self._alias_refs,
            )
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }
        targets = _reference_targets(
            sites_path=self.sites_path,
            releases_path=self.releases_path,
            jira_records_path=self.records_path,
            services_path=self.services_path,
            base=self.base,
        )

        links_by_src = (
            _links_by_source(records) if self.issue_link_edges else {}
        )

        def _facts() -> Iterator[Any]:
            emitted_people: set[str] = set()
            for record in records:
                yield self._node_fact(record, revision)
                for dst, link_types in sorted(
                    links_by_src.get(record.key, {}).items()
                ):
                    yield EdgeFact(
                        src=f"jira:{record.key}",
                        dst=f"jira:{dst}",
                        edge_type="linked_to",
                        attrs={
                            "link_kind": _strongest_kind(link_types),
                            "link_type": "; ".join(sorted(link_types)),
                        },
                        source_record_id={
                            "issue_key": record.key,
                            "linked_issue_key": dst,
                        },
                        source_revision=revision,
                    )
                for role, display_name in (
                    ("assignee", record.assignee),
                    ("reporter", record.reporter),
                ):
                    if not display_name:
                        continue
                    person_id = _person_id(display_name)
                    if person_id not in emitted_people:
                        emitted_people.add(person_id)
                        yield _person_node(display_name, revision)
                    yield EdgeFact(
                        src=record.node_id,
                        dst=person_id,
                        edge_type=(
                            "assigned_to" if role == "assignee" else "reported_by"
                        ),
                        source_record_id={"issue_key": record.key, "role": role},
                        source_revision=revision,
                    )
                if self.comment_records:
                    for fact in _comment_facts(record, revision):
                        if (isinstance(fact, NodeFact)
                                and fact.subtype == "person"):
                            if fact.node_id in emitted_people:
                                continue
                            emitted_people.add(fact.node_id)
                        yield fact
                text = _chunk_text(record)
                for chunk_index, offset, chunk_text in _chunks(text):
                    # Parity wart, kept deliberately: the separator is a
                    # literal backslash-zero (two characters), matching
                    # the cms original's escaped f-string — not the NUL
                    # byte docs.py uses. Changing it re-keys every jira
                    # chunk at cutover.
                    chunk_id = f"chunk:{_sha256(f'{record.node_id}\\0{chunk_index}\\0{chunk_text}')[:16]}"
                    chunk_record_id = {
                        "issue_key": record.key,
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
                            "char_end": offset + len(chunk_text),
                            "chunker_name": self.chunker_name,
                            "heading_path": record.key,
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

        status, reason, complete = self._verdict(
            len(records), skipped, "local JIRA records cache used"
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
                credential_refs=self._credential_refs,
                alias_refs=self._alias_refs,
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason=reason,
            ),
        )

    def _verdict(
        self, record_count: int, skipped: int, ok_reason: str
    ) -> tuple[str, str, bool]:
        """(status, reason, may claim a complete scope), shared by
        preflight() and run() so the two cannot disagree."""
        expected = _expected_count(self.meta_path, base=self.base)
        if expected is not None and expected != record_count:
            # meta.json says the fetch produced `expected` records but
            # the cache parsed a different number: the records file is
            # truncated or stale relative to its own metadata. Emit
            # what parsed, but never claim a completed scope — under
            # missing_from_completed_scope a truncated cache claimed
            # complete would retract every record it dropped. The
            # closed status vocabulary has no 'degraded', so report
            # endpoint_failed.
            return (
                "endpoint_failed",
                f"JIRA records cache parsed {record_count} records "
                f"but {self.meta_path} reports record_count="
                f"{expected}; cache looks truncated or stale, "
                "facts emitted, no complete scope claimed",
                False,
            )
        # Items without an issue key (or not objects at all) are
        # skipped per item, but the run then stops claiming a complete
        # scope: the skipped items' issues would otherwise be retracted.
        status, reason = skipped_items_status(
            status="ok",
            reason=ok_reason,
            record_count=record_count,
            skipped_count=skipped,
        )
        return status, reason, not skipped

    def issue_url(self, key: str) -> str:
        """Browse URL for an issue (v2 collector's ticket metadata URL)."""
        return f"{self.base_url}/browse/{key}" if self.base_url else ""

    def _node_fact(
        self,
        record: JiraIssueRecord,
        revision: dict[str, Any],
    ) -> NodeFact:
        text = " ".join(filter(None, [
            record.key,
            record.summary,
            record.description,
            record.status,
            record.priority,
        ]))
        return NodeFact(
            node_id=record.node_id,
            subtype="jira_issue",
            attrs={
                "label": record.key,
                "issue_key": record.key,
                "summary": record.summary,
                "description": record.description,
                "project": record.project,
                "status": record.status,
                "priority": record.priority,
                "issue_type": record.issue_type,
                "assignee": record.assignee,
                "reporter": record.reporter,
                "created": record.created,
                "updated": record.updated,
                "resolution": record.resolution,
                "labels": list(record.labels),
                "components": list(record.components),
                # Before 2026-10 these were in the chunk text only.
                "environment": record.environment,
                "fix_versions": list(record.fix_versions),
                "parent_key": record.parent_key,
                "issue_links": list(record.issue_links),
                "subtasks": list(record.subtasks),
                # Only with comment records on: okg reads a ticket from
                # its jira_comment records when it carries this count
                # (okg_record_parts count_attr), from its chunks if not.
                **({"comment_count": len(record.comments)}
                   if self.comment_records else {}),
                "url": self.issue_url(record.key),
                "text": text,
            },
            source_record_id={"issue_key": record.key},
            source_revision=revision,
        )

    def _records_with_skips(self) -> tuple[list[JiraIssueRecord], int]:
        payload = read_cache_json(
            self.records_path, expect=list, base=self.base
        )
        items = [item for item in payload if isinstance(item, dict)]
        skipped = len(payload) - len(items)
        # Without configured project_keys the generic KEY-123 pattern
        # would extract strings like COVID-19 from free text; mirror
        # docs.py's intersect-with-known-keys discipline by bounding
        # extraction to the project prefixes present in this cache.
        key_re = (
            self.issue_key_re
            if self.project_keys
            else _cache_bounded_key_pattern(items)
        )
        records: list[JiraIssueRecord] = []
        for item in items:
            key = _pg_text(str(item.get("key") or item.get("issue_key") or ""))
            if not key:
                skipped += 1
                continue
            fields = (
                item.get("fields") if isinstance(item.get("fields"), dict) else {}
            )
            project = _name_or_key(fields.get("project") or item.get("project"))
            comments = _comment_parts(
                item.get("recent_comments")
                or item.get("comments")
                or (fields.get("comment") or {}).get("comments")
            )
            records.append(JiraIssueRecord(
                key=key,
                summary=_pg_text(str(
                    item.get("summary") or fields.get("summary") or ""
                )),
                description=_pg_text(str(
                    item.get("description")
                    or fields.get("description")
                    or ""
                )),
                project=_pg_text(project),
                status=_name_or_key(item.get("status") or fields.get("status")),
                priority=_name_or_key(
                    item.get("priority") or fields.get("priority")
                ),
                issue_type=_name_or_key(
                    item.get("issue_type") or fields.get("issuetype")
                ),
                assignee=_display_name(
                    item.get("assignee") or fields.get("assignee")
                ),
                reporter=_display_name(
                    item.get("reporter") or fields.get("reporter")
                ),
                created=_pg_text(str(
                    item.get("created") or fields.get("created") or ""
                )),
                updated=_pg_text(str(
                    item.get("updated") or fields.get("updated") or ""
                )),
                environment=_pg_text(str(
                    item.get("environment") or fields.get("environment") or ""
                )),
                resolution=_name_or_key(
                    item.get("resolution") or fields.get("resolution")
                ),
                parent_key=_name_or_key(
                    item.get("parent_key")
                    or item.get("parent")
                    or fields.get("parent")
                ),
                comment_count=int(item.get("comment_count") or 0),
                components=tuple(_strings_from_value(
                    item.get("components") or fields.get("components")
                )),
                labels=tuple(_strings_from_value(
                    item.get("labels") or fields.get("labels")
                )),
                fix_versions=tuple(_strings_from_value(
                    item.get("fix_versions")
                    or fields.get("fixVersions")
                    or fields.get("fix_versions")
                )),
                issue_links=tuple(self._issue_keys_from_value(
                    item.get("issue_links") or fields.get("issuelinks"),
                    key_re,
                )),
                subtasks=tuple(self._issue_keys_from_value(
                    item.get("subtasks") or fields.get("subtasks"),
                    key_re,
                )),
                recent_comments=tuple(
                    _comment_line(*comment) for comment in comments
                ),
                comments=tuple(comments),
                typed_links=tuple(_typed_links(
                    item.get("issue_links") or fields.get("issuelinks"),
                    key_re,
                )),
            ))
        return records, skipped

    def _issue_keys_from_value(
        self, value: Any, key_re: re.Pattern[str] | None = None
    ) -> list[str]:
        pattern = self.issue_key_re if key_re is None else key_re
        text = " ".join(_strings_from_value(value))
        return sorted(set(pattern.findall(text)))


def _cache_bounded_key_pattern(
    items: Sequence[dict[str, Any]],
) -> re.Pattern[str]:
    """Issue-key pattern bounded to the project prefixes in *items*.

    Used when ``project_keys`` is omitted: extraction only accepts keys
    whose project prefix belongs to an issue actually present in the
    records cache, so free text like ``COVID-19`` cannot masquerade as
    an issue key.
    """
    prefixes: set[str] = set()
    for item in items:
        key = _pg_text(str(item.get("key") or item.get("issue_key") or ""))
        match = _ISSUE_KEY_PREFIX_RE.fullmatch(key)
        if match:
            prefixes.add(match.group(1))
    if not prefixes:
        return _MATCH_NO_KEY_RE
    alternation = "|".join(re.escape(prefix) for prefix in sorted(prefixes))
    return re.compile(rf"\b(?:{alternation})-\d+\b")


SUBTASK_LINK_TYPE = "Sub-task"

#: The small set a JIRA link type name collapses to. Instances define
#: their own link types (and misspell them: "Duplibcate"), so the raw
#: name stays on the edge as ``link_type`` and ``link_kind`` is the
#: value to filter or group on. Ordered most specific first: a pair of
#: issues linked by several types takes the first kind present.
LINK_KINDS = ("child", "blocks", "duplicates", "clones", "gantt", "relates")
_LINK_KIND_MARKERS = (
    # First match wins. Gantt first: the WBS-Gantt plugin's types
    # ("Hierarchy link (WBSGantt)") are scheduling links, not parents.
    ("gantt", "gantt"),
    ("sub-task", "child"),
    ("child", "child"),
    ("parent", "child"),
    ("block", "blocks"),
    ("clon", "clones"),
    ("dupl", "duplicates"),
)


def link_kind(link_type: str) -> str:
    """Collapse a JIRA link type name to one of :data:`LINK_KINDS`.

    Anything unrecognised (Relates, Relation, Causal, Dependency,
    Follows, ...) is ``relates``: the edge still says the two issues
    are linked and ``link_type`` keeps the exact name.
    """
    name = link_type.lower()
    for marker, kind in _LINK_KIND_MARKERS:
        if marker in name:
            return kind
    return "relates"


def _strongest_kind(link_types: Iterable[str]) -> str:
    """The one kind for a pair of issues linked by several link types.

    The graph holds one ``linked_to`` edge per ordered pair, so a pair
    that is both a Sub-task and a Relates link gets the more specific
    kind; :data:`LINK_KINDS` is in that order, ``relates`` last.
    """
    kinds = {link_kind(link_type) for link_type in link_types}
    return next(kind for kind in LINK_KINDS if kind in kinds)


def _links_by_source(
    records: list[JiraIssueRecord],
) -> dict[str, dict[str, set[str]]]:
    """``{src key: {dst key: link type names}}`` over the whole cache.

    Both ends must be issues in the cache: a link to a ticket outside
    the fetched projects would dangle. Each link is cached on both of
    its issues, once as outward and once as inward, and lands here
    once. Several link types between the same ordered pair are kept
    together, because the graph holds one edge per pair.
    """
    cached_keys = {record.key for record in records}
    out: dict[str, dict[str, set[str]]] = {}
    for record in records:
        for src, dst, link_type in _issue_link_triples(record):
            if src == dst or src not in cached_keys or dst not in cached_keys:
                continue
            out.setdefault(src, {}).setdefault(dst, set()).add(link_type)
    return out


def _typed_links(
    value: Any, key_re: re.Pattern[str]
) -> list[tuple[str, str, str]]:
    """``(key, link type, direction)`` for each typed issue link.

    Reads both cache shapes: the downloader's flat
    ``{"key", "type", "direction"}`` and the JIRA REST
    ``{"type": {"name"}, "outwardIssue"|"inwardIssue": {"key"}}``.
    A link with no direction, or whose key is not an issue key, is
    left out; its key still reaches the chunk text via ``issue_links``.
    """
    if not isinstance(value, list):
        return []
    out: list[tuple[str, str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("outwardIssue") or item.get("inwardIssue"):
            direction = "outward" if item.get("outwardIssue") else "inward"
            key = _name_or_key(item.get(f"{direction}Issue"))
        else:
            direction = str(item.get("direction") or "")
            key = _pg_text(str(item.get("key") or ""))
        if direction not in ("outward", "inward") or not key_re.fullmatch(key):
            continue
        out.append((key, _name_or_key(item.get("type")), direction))
    return out


def _issue_link_triples(
    record: JiraIssueRecord,
) -> Iterator[tuple[str, str, str]]:
    """``(src key, dst key, link type)`` for every link this issue states.

    The source is the issue on the link's outward side, so a link reads
    the same from either end; a sub-task's parent is the source of a
    ``Sub-task`` link. The cache keeps a link type's name, not its
    outward wording, so for an instance-defined parent/child type the
    outward side is taken to be the parent (true of CERN JIRA's
    Child-Issue, "is parent of"); check that before relying on the
    direction of ``child`` edges on another instance.
    """
    for key, link_type, direction in record.typed_links:
        if direction == "outward":
            yield record.key, key, link_type
        else:
            yield key, record.key, link_type
    if record.parent_key:
        yield record.parent_key, record.key, SUBTASK_LINK_TYPE
    for subtask in record.subtasks:
        yield record.key, subtask, SUBTASK_LINK_TYPE


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expected_count(path: str, *, base: str | None = None) -> int | None:
    p = resolve_repo_path(path, base=base)
    if not p.is_file():
        return None
    try:
        payload = load_json(path, base=base)
    except (ValueError, OSError):
        # The count is informational; corrupt/unreadable meta must
        # degrade the preflight report, not crash it.
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("record_count")
    return int(value) if isinstance(value, int) else None


def _name_or_key(value: Any) -> str:
    if isinstance(value, dict):
        return _pg_text(str(value.get("name") or value.get("key") or ""))
    return _pg_text(str(value or ""))


def _display_name(value: Any) -> str:
    if isinstance(value, dict):
        return _pg_text(str(
            value.get("displayName")
            or value.get("name")
            or value.get("key")
            or ""
        ))
    return _pg_text(str(value or ""))


def _chunk_text(record: JiraIssueRecord) -> str:
    """Full text surface for chunk-level extraction.

    The parent ``jira_issue`` node stays compact, but chunks need the
    operational link surface: linked issues, parent/subtask keys,
    components, labels, environment, and recent comments.
    """
    parts = [
        ("summary", record.summary),
        ("description", record.description),
        ("environment", record.environment),
        ("status", record.status),
        ("priority", record.priority),
        ("resolution", record.resolution),
        ("parent", record.parent_key),
        ("components", " ".join(record.components)),
        ("labels", " ".join(record.labels)),
        ("fix_versions", " ".join(record.fix_versions)),
        ("issue_links", " ".join(record.issue_links)),
        ("subtasks", " ".join(record.subtasks)),
        ("recent_comments", "\n".join(record.recent_comments)),
    ]
    return "\n\n".join(
        f"{label}: {text}" for label, text in parts if text
    )


def _pg_text(text: str) -> str:
    """Normalize one source string: NUL to space, email addresses removed.

    Every string this source reads from the records cache passes through
    here (key, summary, description, environment, dates, status-like
    names, people, labels, components, comment authors and bodies), so
    no address reaches a ``jira_issue`` or ``person`` attr or
    ``document_chunk`` text. Parity with okg-deployments
    ``cms/cms_sources/jira.py`` (commit b25e36f06c).
    """
    return redact_email_addresses(text.replace("\x00", " "))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _person_id(display_name: str) -> str:
    slug = _sha256(display_name.casefold().strip())[:16]
    return f"person:{slug}"


def _strings_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _pg_text(value).strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        preferred = (
            value.get("name")
            or value.get("key")
            or value.get("value")
            or value.get("displayName")
        )
        if preferred:
            return _strings_from_value(preferred)
        out: list[str] = []
        for nested in value.values():
            out.extend(_strings_from_value(nested))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for nested in value:
            out.extend(_strings_from_value(nested))
        return out
    return _strings_from_value(str(value))


def _comment_parts(value: Any) -> list[tuple[str, str, str]]:
    """``(author, created, body)`` for each comment, in source order.

    One parse feeds both the chunk text (:func:`_comment_line`) and the
    ``jira_comment`` records, so the two always hold the same comments
    with the same cleaning: every string through :func:`_pg_text`, the
    body stripped, a comment with an empty body dropped. A comment the
    cache holds as a bare string has no author or time.
    """
    if not value:
        return []
    comments = value if isinstance(value, list) else [value]
    out: list[tuple[str, str, str]] = []
    for comment in comments:
        if isinstance(comment, dict):
            author = _display_name(comment.get("author"))
            created = _pg_text(str(comment.get("created") or ""))
            body = _pg_text(str(
                comment.get("body")
                or comment.get("text")
                or comment.get("renderedBody")
                or ""
            )).strip()
            if body:
                out.append((author, created, body))
        else:
            out.extend(("", "", text) for text in _strings_from_value(comment))
    return out


def _comment_line(author: str, created: str, body: str) -> str:
    """The ``"author created: body"`` line the chunk text carries.

    Chunk ids hash the chunk text, so this format must not change.
    """
    prefix = " ".join(p for p in (author, created) if p)
    return f"{prefix}: {body}" if prefix else body


#: A JIRA time with a zone: ``2026-01-06T08:00:00.000+0200``, ``...Z``
#: or ``...+02:00``; the seconds and fraction are optional.
_JIRA_TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})"
    r"(?::(\d{2})(?:\.(\d{1,9}))?)?"
    r"(Z|[+-]\d{2}:?\d{2})$"
)

#: ``id_basis`` of a comment whose id hashes its time as a UTC instant.
ID_BASIS_CONTENT = "content"
#: ``id_basis`` of a comment whose time did not parse (or was absent), so
#: its id hashes the raw time string as the cache holds it.
ID_BASIS_CONTENT_RAW_TIME = "content_raw_created"


def _created_utc(created: str) -> str | None:
    """*created* as a UTC instant in one fixed format, or None.

    JIRA renders a comment's time in the zone of the account or server
    that fetched it (``+0200`` and ``+0100`` both occur in one cache), so
    the same comment fetched under another zone has another string. The
    id hashes this form so a refetch in another zone keeps every id.
    None for an empty time, a time without a zone, or one that does not
    parse.
    """
    match = _JIRA_TIME_RE.match(created.strip())
    if not match:
        return None
    year, month, day, hour, minute, second, fraction, zone = match.groups()
    if zone == "Z":
        offset = timedelta(0)
    else:
        sign = -1 if zone[0] == "-" else 1
        digits = zone[1:].replace(":", "")
        offset = sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
    try:
        moment = datetime(
            int(year), int(month), int(day), int(hour), int(minute),
            int(second or 0), int((fraction or "0").ljust(6, "0")[:6]),
            tzinfo=timezone(offset),
        )
    except ValueError:
        return None
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def comment_ids(comments: Sequence[tuple[str, str, str]]) -> list[str]:
    """The ``#c-...`` id part of each comment of one issue, in order.

    The cache keeps no JIRA comment id, so a comment is named by its
    content: ``c-`` and the first 12 hex characters of
    ``sha256(created|author|body)`` over the cleaned strings, with
    ``created`` as its UTC instant (:func:`_created_utc`) or, when it does
    not parse, as the raw string. An exact repeat inside the same issue
    gets ``-1``, ``-2``, ... in source order. Editing a body, renaming an
    author or deleting an earlier repeat therefore re-keys that comment
    (one removal, one addition); a refetch in another time zone does not.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for author, created, body in comments:
        when = _created_utc(created) or created
        base = "c-" + _sha256(f"{when}|{author}|{body}")[:12]
        repeat = seen.get(base, 0)
        seen[base] = repeat + 1
        out.append(f"{base}-{repeat}" if repeat else base)
    return out


def _comment_facts(
    record: JiraIssueRecord, revision: dict[str, Any]
) -> Iterator[NodeFact | EdgeFact]:
    """One ``jira_comment`` record per comment, with its two edges.

    Node id ``jira:<KEY>#c-<hash>`` (see :func:`comment_ids`). Each
    comment carries the issue key and the issue's ``project``, and comes
    from the same source, so an access rule on the project or on the
    source matches the comments. A rule on the issue's own record does
    not: the comment's ``source_record_id`` (issue key plus comment id)
    is a different record ref from the issue's. okg's
    ``test_comment_lineage_filter.py`` is the gate before a deployment
    turns ``comment_records`` on.

    The comment has no ``text`` attribute, so okg's full-text index (on
    ``text``) does not hold it: the issue's chunks stay the one search
    unit for comment text, and ``body`` holds the comment for reads.
    An author's ``person`` node is yielded before its first
    ``authored_by`` edge; the caller drops repeats.
    """
    for position, ((author, created, body), cid) in enumerate(
        zip(record.comments, comment_ids(record.comments))
    ):
        node_id = f"{record.node_id}#{cid}"
        record_id = {"issue_key": record.key, "comment": cid}
        yield NodeFact(
            node_id=node_id,
            subtype="jira_comment",
            attrs={
                "issue_key": record.key,
                "project": record.project,
                "position": position,
                "author": author,
                "created": created,
                "body": body,
                "id_basis": (
                    ID_BASIS_CONTENT if _created_utc(created)
                    else ID_BASIS_CONTENT_RAW_TIME
                ),
            },
            source_record_id=record_id,
            source_revision=revision,
        )
        yield EdgeFact(
            src=record.node_id,
            dst=node_id,
            edge_type="has_comment",
            attrs={"position": position},
            source_record_id=record_id,
            source_revision=revision,
        )
        if author:
            yield _person_node(author, revision)
            yield EdgeFact(
                src=node_id,
                dst=_person_id(author),
                edge_type="authored_by",
                source_record_id=record_id,
                source_revision=revision,
            )


def _person_node(
    display_name: str,
    revision: dict[str, Any],
) -> NodeFact:
    node_id = _person_id(display_name)
    return NodeFact(
        node_id=node_id,
        subtype="person",
        attrs={
            "person_id": node_id,
            "display_name": display_name,
            "text": display_name,
        },
        source_record_id={"person": display_name},
        source_revision=revision,
    )


class JiraIssueAdapter(ReaderAdapter):
    """Registry adapter for :class:`JiraIssueSource`.

    The bundle's `jira` source names this class. The reader's
    behavior is unchanged; this class only drives it through the
    substrate's adapter contract.

    ``profile`` and ``change_probe_kind`` must be string literals; see
    ``ReaderAdapter``. ``test_bundle_source_adapters.py`` parses this
    file and holds them equal to the reader's own values.
    """

    reader_class = JiraIssueSource
    profile = "mutable_api"
    change_probe_kind = "mutable_api"
