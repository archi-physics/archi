"""Cache-backed WMStats workflow source.

Ported from okg-deployments ``cms/cms_sources/wmstats.py`` (257 LOC,
``WMStatsWorkflowSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes: cache
helpers come from :mod:`archi.auth.cache` /
:mod:`archi.sources._cache_report` with an explicit ``base`` parameter,
and the hardcoded ``data/cms/wmstats-workflows/records.json`` path is a
parameter (default keeps the cms layout minus the ``cms/`` segment).
As in the original the source is optional and offline-only; ``run()``
raises on a missing cache (only ``preflight`` reports it), and the
``depends_on`` -> ``cmssw_release`` edge is emitted whether or not that
release node exists (the original did not check).

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``workflow`` and
``cmssw_release`` ship in ``archi/schemas/operations.yaml``;
``dataset`` comes from a substrate module. ::

    wmstats_workflows:
      module: archi.sources.wmstats
      class: WMStatsWorkflowSource
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
      source_revision_kind: content_hash
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

from dataclasses import dataclass
from typing import Any, Iterator

from okg.deployment import (
    EdgeFact,
    NodeFact,
    PreflightResult,
)
from okg.substrate.library.sources.base import SourceRun

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
    load_json,
)
from archi.sources._cache_report import (
    cache_preflight_result,
    cache_source_health,
)


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
        try:
            records = self._records()
        except FileNotFoundError:
            records = None
        return cache_preflight_result(
            source_name=self.name,
            description="WMStats workflow",
            cache_paths=self.cache_paths,
            records=records,
            required=False,
            base=self.base,
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> SourceRun:
        records, skipped = self._records_with_skips()
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            yield from _facts_for_records(records, revision)

        return SourceRun(
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
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of workflows"
            )
        records: list[WorkflowRecord] = []
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            name = str(
                item.get("workflow_name")
                or item.get("request_name")
                or item.get("RequestName")
                or ""
            ).strip()
            if not name:
                skipped += 1
                continue
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
