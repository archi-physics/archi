"""Cache-backed DBS dataset source.

Ported from okg-deployments ``cms/cms_sources/dbs.py`` (258 LOC,
``DBSDatasetSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes: cache
helpers come from :mod:`archi.auth.cache` /
:mod:`archi.sources._cache_report` with an explicit ``base`` parameter,
and the hardcoded ``data/cms/dbs-datasets/records.json`` path is a
parameter (default keeps the cms layout minus the ``cms/`` segment).
As in the original the source is optional (``required=False``) and
offline-only; a missing cache raises ``FileNotFoundError`` from
``run()`` (only ``preflight`` reports it), matching the original.

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; the ``dataset`` subtype comes
from a substrate module (the cms deployment composed it), not from
``archi/schemas/operations.yaml``. ::

    dbs_datasets:
      module: archi.sources.dbs
      class: DBSDatasetSource
      ownership_id: <instance>.dbs-datasets
      admission_policy:
        producer_id: <instance>.dbs-datasets
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: dbs_datasets
        output_signature:
          nodes:
            - {subtype: dataset}
          edges:
            - {src_subtype: dataset, edge_type: derives_from, dst_subtype: dataset}
        output_scope_summary:
          summary: DBS datasets and deterministic tier-chain derives_from edges
          nodes: [dataset]
          edges:
            - dataset derives_from dataset
      source_class: reference_catalog
      record_identity_kind: remote_id
      record_identity_fields: [dataset]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: false
      params:
        # cms default; the cms deployment used data/cms/dbs-datasets/
        records_path: data/dbs-datasets/records.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import re
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
class DBSDatasetRecord:
    dataset_name: str
    data_tier: str = ""
    primary_dataset: str = ""
    processed_dataset: str = ""
    physics_group: str = ""
    creation_date: str = ""
    dataset_access_type: str = ""
    total_size_bytes: int = 0
    total_files: int = 0
    total_events: int = 0

    @property
    def node_id(self) -> str:
        return f"dataset:{self.dataset_name}"


class DBSDatasetSource:
    """Cache-backed DBS dataset source.

    This adapter is intentionally offline-only: it reads the local JSON
    cache and emits no placeholder facts when the cache is empty.
    """

    name = "dbs_datasets"
    profile = "reference_catalog"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/dbs-datasets/records.json",
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"records_path": self.records_path},
            emit_targets=DBSDatasetSource,
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
            description="DBS dataset",
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
            for record in records:
                yield _node_fact(record, revision)
            yield from _edge_facts(records, revision)

        return SourceRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and not skipped
            ),
            run_mode=mode,
            health=cache_source_health(
                description="DBS dataset",
                cache_paths=self.cache_paths,
                record_count=len(records),
                skipped_count=skipped,
                base=self.base,
            ),
        )

    def _records(self) -> list[DBSDatasetRecord]:
        return self._records_with_skips()[0]

    def _records_with_skips(self) -> tuple[list[DBSDatasetRecord], int]:
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of datasets"
            )
        records: list[DBSDatasetRecord] = []
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            dataset_name = str(
                item.get("dataset_name") or item.get("dataset") or ""
            ).strip()
            if not dataset_name:
                skipped += 1
                continue
            records.append(DBSDatasetRecord(
                dataset_name=dataset_name,
                data_tier=str(
                    item.get("data_tier")
                    or item.get("data_tier_name")
                    or ""
                ),
                primary_dataset=str(
                    item.get("primary_dataset")
                    or item.get("primary_ds_name")
                    or ""
                ),
                processed_dataset=str(
                    item.get("processed_dataset")
                    or item.get("processed_ds_name")
                    or ""
                ),
                physics_group=str(
                    item.get("physics_group")
                    or item.get("physics_group_name")
                    or ""
                ),
                creation_date=str(item.get("creation_date") or ""),
                dataset_access_type=str(
                    item.get("dataset_access_type") or ""
                ),
                total_size_bytes=int(
                    item.get("total_size_bytes")
                    or item.get("dataset_size")
                    or 0
                ),
                total_files=int(
                    item.get("total_files") or item.get("nfiles") or 0
                ),
                total_events=int(
                    item.get("total_events") or item.get("nevents") or 0
                ),
            ))
        return records, skipped


def _node_fact(
    record: DBSDatasetRecord,
    revision: dict[str, Any],
) -> NodeFact:
    era = _era(record.processed_dataset)
    text = " ".join(filter(None, [
        record.dataset_name,
        record.data_tier,
        record.primary_dataset,
        record.processed_dataset,
        era,
    ]))
    return NodeFact(
        node_id=record.node_id,
        subtype="dataset",
        attrs={
            "label": record.dataset_name,
            "name": record.dataset_name,
            "dataset_id": record.node_id,
            "dataset_name": record.dataset_name,
            "tier": record.data_tier,
            "era": era,
            "primary_dataset": record.primary_dataset,
            "processed_dataset": record.processed_dataset,
            "creation_date": record.creation_date,
            "dataset_access_type": record.dataset_access_type,
            "physics_group": record.physics_group,
            "total_size_bytes": record.total_size_bytes,
            "total_files": record.total_files,
            "total_events": record.total_events,
            "text": text,
        },
        source_record_id={"dataset": record.dataset_name},
        source_revision=revision,
    )


def _edge_facts(
    records: list[DBSDatasetRecord],
    revision: dict[str, Any],
) -> Iterator[EdgeFact]:
    records_by_group: dict[tuple[str, str], list[DBSDatasetRecord]] = {}
    for record in records:
        key = (record.primary_dataset, _campaign(record.processed_dataset))
        records_by_group.setdefault(key, []).append(record)

    for grouped in records_by_group.values():
        if len(grouped) < 2:
            continue
        ordered = sorted(grouped, key=lambda r: _tier_order(r.data_tier))
        for src, dst in zip(ordered[1:], ordered):
            if _tier_order(src.data_tier) == _tier_order(dst.data_tier):
                continue
            yield EdgeFact(
                src=src.node_id,
                dst=dst.node_id,
                edge_type="derives_from",
                provenance="derived_deterministic",
                attrs={
                    "relationship": "tier_chain",
                    "src_tier": src.data_tier,
                    "dst_tier": dst.data_tier,
                },
                source_record_id={"dataset": src.dataset_name},
                source_revision=revision,
            )


_TIER_ORDER = {
    "GEN": 0,
    "LHE": 0,
    "GEN-SIM": 1,
    "SIM": 1,
    "RAW": 2,
    "DIGI": 2,
    "DIGI-RECO": 3,
    "RECO": 3,
    "AOD": 4,
    "AODSIM": 4,
    "MINIAOD": 5,
    "MINIAODSIM": 5,
    "NANOAOD": 6,
    "NANOAODSIM": 6,
}


def _tier_order(tier: str) -> int:
    return _TIER_ORDER.get(tier, 99)


def _era(processed_dataset: str) -> str:
    parts = processed_dataset.split("-", 1)
    return parts[0] if parts else ""


def _campaign(processed_dataset: str) -> str:
    match = re.match(r"([A-Za-z]+\d{4}[A-Za-z]*)", processed_dataset)
    return match.group(1) if match else _era(processed_dataset)
