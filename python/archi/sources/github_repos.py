"""Configured GitHub repository identity source.

Ported from okg-deployments ``cms/cms_sources/github_repos.py``
(193 LOC, ``GitHubRepoSource``) at ``main@f33a9c4`` for the archi v3
package (req.w2.sources-catalogs). Behavior kept verbatim; only change:
the hardcoded CMS repository list is the *default* for the ``repos``
constructor parameter (it already was a parameter upstream — nothing
else was CMS-bound; the source reads no caches and does no network I/O,
it emits registry-configured repository identities only).

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``software_repository`` ships in
``archi/schemas/sources.yaml`` (extended with the repo-metadata attrs
this source emits); the generic ``repo`` subtype comes from a substrate
module (code_repos family), as in the cms deployment. ::

    github_repos:
      module: archi.sources.github_repos
      class: GitHubRepoSource
      ownership_id: <instance>.github-repos
      admission_policy:
        producer_id: <instance>.github-repos
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: github_repos
        output_signature:
          nodes:
            - {subtype: software_repository}
            - {subtype: repo}
          edges:
            - {src_subtype: software_repository, edge_type: references, dst_subtype: repo}
        output_scope_summary:
          summary: configured GitHub repository identities (no live metadata)
          nodes: [software_repository, repo]
          edges:
            - software_repository references repo
      source_class: reference_catalog
      record_identity_kind: remote_id
      record_identity_fields: [repo]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: false
      params:
        # cms defaults (DEFAULT_REPOS); override per instance
        # repos: [dmwm/WMCore, cms-sw/cmssw, ...]
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ConnectorHealth,
    PreflightResult,
    ConnectorRun,
)
from okg.deployment import ContentHashProbe

DEFAULT_REPOS = (
    "dmwm/WMCore",
    "dmwm/CMSMonitoring",
    "dmwm/DBS",
    "dmwm/CMSRucio",
    "dmwm/CRABServer",
    "dmwm/CRABClient",
    "dmwm/WMArchive",
    "cms-sw/cmssw",
    "cms-sw/cmsdist",
    "cms-sw/cms-bot",
    "rucio/rucio",
    "xrootd/xrootd",
    "glideinWMS/glideinwms",
    "dmwm/reqmgr2",
    "dmwm/T0",
)


@dataclass(frozen=True)
class GitHubRepoRecord:
    slug: str

    @property
    def name(self) -> str:
        return self.slug.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"https://github.com/{self.slug}"

    @property
    def software_node_id(self) -> str:
        return f"software_repository:{self.slug}"

    @property
    def repo_node_id(self) -> str:
        return f"repo:{self.slug}"


class GitHubRepoSource:
    """Configured repository identity source.

    This source deliberately emits only registry-configured repository
    identities. Live GitHub metadata/docs can extend this adapter once
    a cache or token-backed download path exists.
    """

    name = "github_repos"
    profile = "reference_catalog"
    change_probe_kind = "content_hash"

    def __init__(self, *, repos: list[str] | None = None) -> None:
        self.repos = tuple(repos or DEFAULT_REPOS)
        self.change_probe = ContentHashProbe(
            content_items=lambda: [
                ("repos", "\n".join(sorted(self.repos)).encode("utf-8")),
            ],
            config={"repos": self.repos},
            emit_targets=GitHubRepoSource,
        )

    def preflight(self, mode: str = "live") -> PreflightResult:
        records = self._records()
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="registry_seed",
            required=False,
            record_count=len(records),
            content_hash=_repo_hash(self.repos),
            reason=(
                "configured repository identities present; no live "
                "GitHub metadata cache was read"
            ),
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        records = self._records()
        revision = {
            "run_id": run_id,
            "content_hash": _repo_hash(self.repos),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            for record in records:
                yield _software_repository_node(record, revision)
                yield _repo_node(record, revision)
                yield EdgeFact(
                    src=record.software_node_id,
                    dst=record.repo_node_id,
                    edge_type="references",
                    source_record_id={"repo": record.slug},
                    source_revision=revision,
                )

        return ConnectorRun(
            facts=_facts(),
            completed_scope=(mode in {"scope_complete", "reconcile"}),
            run_mode=mode,
            health=ConnectorHealth(
                status="ok",
                mode="registry_seed",
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason="configured GitHub repository identities used",
            ),
        )

    def _records(self) -> list[GitHubRepoRecord]:
        seen: set[str] = set()
        records: list[GitHubRepoRecord] = []
        for repo in self.repos:
            slug = repo.strip()
            if not slug or "/" not in slug or slug in seen:
                continue
            seen.add(slug)
            records.append(GitHubRepoRecord(slug=slug))
        return records


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_hash(repos: tuple[str, ...]) -> str:
    payload = "\n".join(sorted(repos)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _software_repository_node(
    record: GitHubRepoRecord,
    revision: dict[str, Any],
) -> NodeFact:
    return NodeFact(
        node_id=record.software_node_id,
        subtype="software_repository",
        attrs={
            "label": record.name,
            "name": record.name,
            "full_name": record.slug,
            "url": record.url,
            "description": "",
            "language": "",
            "stars": 0,
            "topics": [],
            "default_branch": "",
            "last_commit_date": "",
            "text": f"{record.slug} {record.name}".strip(),
        },
        source_record_id={"repo": record.slug},
        source_revision=revision,
    )


def _repo_node(
    record: GitHubRepoRecord,
    revision: dict[str, Any],
) -> NodeFact:
    return NodeFact(
        node_id=record.repo_node_id,
        subtype="repo",
        attrs={
            "label": record.slug,
            "repo_id": record.repo_node_id,
            "slug": record.slug,
            "name": record.name,
            "url": record.url,
        },
        source_record_id={"repo": record.slug},
        source_revision=revision,
    )
