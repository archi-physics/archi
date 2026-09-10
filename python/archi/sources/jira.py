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
          edges:
            - {src_subtype: jira_issue, edge_type: assigned_to, dst_subtype: person}
            - {src_subtype: jira_issue, edge_type: reported_by, dst_subtype: person}
            - {src_subtype: jira_issue, edge_type: contains, dst_subtype: document_chunk}
            - {src_subtype: document_chunk, edge_type: references, dst_subtype: jira_issue}
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
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
    ConnectorRun,
)

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
        self.base = base
        self.change_probe = cache_or_forced_live_change_probe(
            cache_paths=self.cache_paths,
            config={
                "records_path": self.records_path,
                "meta_path": self.meta_path,
            },
            emit_targets=JiraIssueSource,
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
            reason = "JIRA records cache file is missing"
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
        records = self._records()
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=True,
            credential_refs=self._credential_refs,
            alias_refs=self._alias_refs,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason="local JIRA records cache present",
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
                    reason="JIRA records cache is absent; no facts emitted",
                    checked_at=_checked_at(),
                ),
            )
        records = self._records()
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

        def _facts() -> Iterator[Any]:
            emitted_people: set[str] = set()
            for record in records:
                yield self._node_fact(record, revision)
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

        expected = _expected_count(self.meta_path, base=self.base)
        if expected is not None and expected != len(records):
            # meta.json says the fetch produced `expected` records but
            # the cache parsed a different number: the records file is
            # truncated or stale relative to its own metadata. Emit
            # what parsed, but never claim a completed scope — under
            # missing_from_completed_scope a truncated cache claimed
            # complete would retract every record it dropped. The
            # closed status vocabulary has no 'degraded', so report
            # endpoint_failed.
            return ConnectorRun(
                facts=_facts(),
                completed_scope=False,
                run_mode=mode,
                health=ConnectorHealth(
                    status="endpoint_failed",
                    mode="cache",
                    credential_refs=self._credential_refs,
                    alias_refs=self._alias_refs,
                    record_count=len(records),
                    content_hash=revision["content_hash"],
                    reason=(
                        f"JIRA records cache parsed {len(records)} records "
                        f"but {self.meta_path} reports record_count="
                        f"{expected}; cache looks truncated or stale, "
                        "facts emitted, no complete scope claimed"
                    ),
                    checked_at=_checked_at(),
                ),
            )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="cache",
                credential_refs=self._credential_refs,
                alias_refs=self._alias_refs,
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason="local JIRA records cache used",
            ),
        )

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
                "url": self.issue_url(record.key),
                "text": text,
            },
            source_record_id={"issue_key": record.key},
            source_revision=revision,
        )

    def _records(self) -> list[JiraIssueRecord]:
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of JIRA records"
            )
        items = [item for item in payload if isinstance(item, dict)]
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
                continue
            fields = (
                item.get("fields") if isinstance(item.get("fields"), dict) else {}
            )
            project = _name_or_key(fields.get("project") or item.get("project"))
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
                recent_comments=tuple(_comment_texts(
                    item.get("recent_comments")
                    or item.get("comments")
                    or (fields.get("comment") or {}).get("comments")
                )),
            ))
        return records

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
    return text.replace("\x00", " ")


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


def _comment_texts(value: Any) -> list[str]:
    if not value:
        return []
    comments = value if isinstance(value, list) else [value]
    out: list[str] = []
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
                prefix = " ".join(p for p in (author, created) if p)
                out.append(f"{prefix}: {body}" if prefix else body)
        else:
            out.extend(_strings_from_value(comment))
    return out


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
