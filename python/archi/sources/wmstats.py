"""Cache-backed WMStats workflow source.

Ported from okg-deployments ``cms/cms_sources/wmstats.py`` (257 LOC,
``WMStatsWorkflowSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes: cache
helpers come from :mod:`archi.auth.cache` /
:mod:`archi.sources._cache_report` with an explicit ``base`` parameter,
and the hardcoded ``data/cms/wmstats-workflows/records.json`` path is a
parameter (default keeps the cms layout minus the ``cms/`` segment).
As in the original the source is optional and offline-only; a missing, unreadable, truncated or
empty records cache is ``cache_missing`` with ``completed_scope=False`` from
both ``preflight`` and ``run()`` (empty-cache-fails-loudly; the original
raised from ``run()`` on a missing cache and reported an empty one as
``skipped_optional``, which okg accepts as a complete scope to retract). The
``depends_on`` -> ``cmssw_release`` edge is emitted whether or not that
release node exists (the original did not check).

The cache is either a plain list of workflow records or the WMStats
collection shape (wmstats-collection-shape)::

    {"cutoff_utc": "...",
     "status_payloads": {"<status>": {"result": [{"<name>": {record}}]}}}

one WMStats API answer per workflow status. Its records are flattened in
file order. A workflow name seen twice is skipped and counted, so the run
emits it once and claims no complete scope: the collector already merges
duplicates (``authority.json`` ``deliberate_merge_count``), so a repeat
means the export contradicts itself. No workflow at all under
``status_payloads`` is ``cache_missing``; an object without a
``status_payloads`` object, or a status without a ``result`` list, is
``endpoint_failed``. When ``authority.json`` sits next to the records file,
the number of records in the file must equal its ``record_count``, else
``endpoint_failed``.

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``workflow`` and
``cmssw_release`` ship in ``archi/schemas/operations.yaml``;
``dataset`` comes from a substrate module. ::

    wmstats_workflows:
      module: archi.sources.wmstats
      class: WMStatsWorkflowAdapter
      ownership_id: <instance>.wmstats-workflows
      admission_policy:
        producer_id: <instance>.wmstats-workflows
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: wmstats_workflows
        output_signature:
          nodes:
            - {subtype: workflow}
            - {subtype: dataset}
          edges:
            - {src_subtype: workflow, edge_type: consumes, dst_subtype: dataset}
            - {src_subtype: workflow, edge_type: produces, dst_subtype: dataset}
            - {src_subtype: workflow, edge_type: depends_on, dst_subtype: cmssw_release}
        output_scope_summary:
          summary: WMStats workflows with dataset input/output and release dependencies
          nodes: [workflow, dataset]
          edges:
            - workflow consumes dataset
            - workflow produces dataset
            - workflow depends_on cmssw_release
      source_class: mutable_api
      record_identity_kind: remote_id
      record_identity_fields: [workflow]
      source_revision_kind: updated_at
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: false
      params:
        # cms default; the cms deployment used data/cms/wmstats-workflows/
        records_path: data/wmstats-workflows/records.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    PreflightResult,
    ConnectorRun,
)

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
    resolve_repo_path,
)
from archi.sources._cache_report import (
    CacheUnusable,
    cache_preflight_result,
    cache_source_health,
    read_cache_json,
    unusable_cache_preflight,
    unusable_cache_run,
)
from archi.sources._sdk_adapter import ReaderAdapter


@dataclass(frozen=True)
class WorkflowRecord:
    workflow_name: str
    request_type: str = ""
    status: str = ""
    campaign: str = ""
    prep_id: str = ""
    priority: int = 0
    cmssw_version: str = ""
    input_dataset: str = ""
    output_datasets: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""

    @property
    def node_id(self) -> str:
        return f"workflow:{self.workflow_name}"


class WMStatsWorkflowSource:
    """Cache-backed WMStats workflow source.

    This adapter is intentionally offline-only: it reads the local JSON
    cache and emits no placeholder facts when the cache is empty.
    """

    name = "wmstats_workflows"
    profile = "mutable_api"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/wmstats-workflows/records.json",
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"records_path": self.records_path},
            emit_targets=WMStatsWorkflowSource,
            base=base,
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return (self.records_path,)

    def preflight(self, mode: str = "live") -> PreflightResult:
        # Same verdict run() reaches: a missing, unreadable, truncated
        # or empty cache is cache_missing, never an empty success.
        try:
            records, skipped = self._records_with_skips()
        except CacheUnusable as exc:
            return unusable_cache_preflight(
                exc, source_name=self.name, required=False
            )
        return cache_preflight_result(
            source_name=self.name,
            description="WMStats workflow",
            cache_paths=self.cache_paths,
            records=records,
            required=False,
            base=self.base,
            skipped_count=skipped,
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        try:
            records, skipped = self._records_with_skips()
        except CacheUnusable as exc:
            # An empty cache is a failed fetch, not an empty catalog: a
            # complete scope over zero records would retract every
            # record ingested before (okg honours that order for
            # skipped_optional health too), so report cache_missing.
            return unusable_cache_run(exc, mode=mode)
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            yield from _facts_for_records(records, revision)

        return ConnectorRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and not skipped
            ),
            run_mode=mode,
            health=cache_source_health(
                description="WMStats workflow",
                cache_paths=self.cache_paths,
                record_count=len(records),
                skipped_count=skipped,
                base=self.base,
            ),
        )

    def _records(self) -> list[WorkflowRecord]:
        return self._records_with_skips()[0]

    def _records_with_skips(self) -> tuple[list[WorkflowRecord], int]:
        payload = read_cache_json(
            self.records_path, expect=(list, dict), base=self.base
        )
        path = resolve_repo_path(self.records_path, base=self.base)
        items = _workflow_items(payload, path)
        _check_authority_count(path, len(items))
        records: list[WorkflowRecord] = []
        seen: set[str] = set()
        skipped = 0
        for item in items:
            if not isinstance(item, dict):
                skipped += 1
                continue
            name = str(
                item.get("workflow_name")
                or item.get("request_name")
                or item.get("RequestName")
                or ""
            ).strip()
            if not name or name in seen:
                skipped += 1
                continue
            seen.add(name)
            output = (
                item.get("output_datasets")
                or item.get("OutputDatasets")
                or ()
            )
            if isinstance(output, str):
                output_datasets = (output,)
            else:
                output_datasets = tuple(str(v) for v in output if v)
            records.append(WorkflowRecord(
                workflow_name=name,
                request_type=str(
                    item.get("request_type")
                    or item.get("RequestType")
                    or ""
                ),
                status=str(
                    item.get("status")
                    or item.get("RequestStatus")
                    or ""
                ),
                campaign=str(item.get("campaign") or item.get("Campaign") or ""),
                prep_id=str(item.get("prep_id") or item.get("PrepID") or ""),
                priority=int(
                    item.get("priority") or item.get("RequestPriority") or 0
                ),
                cmssw_version=str(
                    item.get("cmssw_version")
                    or item.get("CMSSWVersion")
                    or ""
                ),
                input_dataset=str(
                    item.get("input_dataset")
                    or item.get("InputDataset")
                    or ""
                ),
                output_datasets=output_datasets,
                created_at=str(
                    item.get("created_at")
                    or item.get("RequestDate")
                    or ""
                ),
                updated_at=str(item.get("updated_at") or ""),
            ))
        return records, skipped


def _workflow_items(payload: Any, path: Path) -> list[Any]:
    """The workflow records of either cache shape, in file order."""
    if isinstance(payload, list):
        return payload
    by_status = payload.get("status_payloads")
    if not isinstance(by_status, dict):
        raise CacheUnusable(
            path,
            "holds a JSON object without a status_payloads object "
            "(drifted or error-shaped payload)",
            status="endpoint_failed",
        )
    items: list[Any] = []
    for status, body in by_status.items():
        result = body.get("result") if isinstance(body, dict) else None
        if not isinstance(result, list):
            raise CacheUnusable(
                path,
                f"status_payloads[{status!r}] has no result list "
                "(drifted or error-shaped payload)",
                status="endpoint_failed",
            )
        for entry in result:
            # Each result entry maps workflow name -> record.
            items.extend(entry.values() if isinstance(entry, dict) else [entry])
    if not items:
        raise CacheUnusable(
            path,
            "holds no workflow under status_payloads; a failed or "
            "unsynced fetch, not an empty catalog",
            status="cache_missing",
        )
    return items


def _check_authority_count(path: Path, count: int) -> None:
    """The collector's ``authority.json`` record count, when it is there."""
    authority = path.parent / "authority.json"
    if not authority.is_file():
        return
    try:
        expected = json.loads(authority.read_text(encoding="utf-8"))["record_count"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CacheUnusable(
            authority,
            f"has no readable record_count ({exc.__class__.__name__})",
            status="endpoint_failed",
        ) from exc
    if expected != count:
        raise CacheUnusable(
            path,
            f"holds {count} workflow records but {authority} counts "
            f"{expected}; a partial or mixed export",
            status="endpoint_failed",
        )


def _facts_for_records(
    records: list[WorkflowRecord],
    revision: dict[str, Any],
) -> Iterator[NodeFact | EdgeFact]:
    seen_datasets: set[str] = set()
    for record in records:
        yield _workflow_node(record, revision)
        if record.input_dataset:
            if record.input_dataset not in seen_datasets:
                seen_datasets.add(record.input_dataset)
                yield _dataset_node(record.input_dataset, revision)
            yield EdgeFact(
                src=record.node_id,
                dst=f"dataset:{record.input_dataset}",
                edge_type="consumes",
                source_record_id={"workflow": record.workflow_name},
                source_revision=revision,
            )
        for dataset in record.output_datasets:
            if dataset not in seen_datasets:
                seen_datasets.add(dataset)
                yield _dataset_node(dataset, revision)
            yield EdgeFact(
                src=record.node_id,
                dst=f"dataset:{dataset}",
                edge_type="produces",
                source_record_id={"workflow": record.workflow_name},
                source_revision=revision,
            )
        if record.cmssw_version:
            yield EdgeFact(
                src=record.node_id,
                dst=f"cmssw_release:{record.cmssw_version}",
                edge_type="depends_on",
                provenance="derived_deterministic",
                source_record_id={"workflow": record.workflow_name},
                source_revision=revision,
            )


def _workflow_node(
    record: WorkflowRecord,
    revision: dict[str, Any],
) -> NodeFact:
    text = " ".join(filter(None, [
        record.workflow_name,
        record.request_type,
        record.status,
        record.campaign,
        record.prep_id,
        record.cmssw_version,
        record.input_dataset,
        " ".join(record.output_datasets),
    ]))
    return NodeFact(
        node_id=record.node_id,
        subtype="workflow",
        attrs={
            "label": record.workflow_name,
            "workflow_name": record.workflow_name,
            "request_type": record.request_type,
            "status": record.status,
            "campaign": record.campaign,
            "prep_id": record.prep_id,
            "priority": record.priority,
            "input_dataset": record.input_dataset,
            "output_dataset": ",".join(record.output_datasets),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "text": text,
        },
        source_record_id={"workflow": record.workflow_name},
        source_revision=revision,
    )


def _dataset_node(dataset: str, revision: dict[str, Any]) -> NodeFact:
    return NodeFact(
        node_id=f"dataset:{dataset}",
        subtype="dataset",
        attrs={
            "label": dataset,
            "dataset_id": f"dataset:{dataset}",
            "name": dataset,
            "dataset_name": dataset,
            "text": dataset,
        },
        source_record_id={"dataset": dataset},
        source_revision=revision,
    )


class WMStatsWorkflowAdapter(ReaderAdapter):
    """Registry adapter for :class:`WMStatsWorkflowSource`.

    The registry-entry template in this module's docstring names this
    class. The reader's behavior is unchanged; this class only drives it
    through the substrate's adapter contract, which a bare reader cannot
    satisfy (its ``ConnectorRun`` has no ``next_cursor``).

    ``profile`` and ``change_probe_kind`` must be string literals; see
    ``ReaderAdapter``. ``test_bundle_source_adapters.py`` parses this
    file and holds them equal to the reader's own values.
    """

    reader_class = WMStatsWorkflowSource
    profile = "mutable_api"
    change_probe_kind = "content_hash"
