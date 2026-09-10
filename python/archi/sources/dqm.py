"""Cache-backed DQM data certification source.

Ported from okg-deployments ``cms/cms_sources/dqm.py`` (281 LOC,
``DQMSource``) at ``main@f33a9c4`` for the archi v3 package
(req.w2.sources-catalogs). Behavior kept verbatim; only changes: cache
helpers come from :mod:`archi.auth.cache` with an explicit ``base``
parameter, and the hardcoded ``data/cms/dqm/records.json`` path is a
parameter (default keeps the cms layout minus the ``cms/`` segment).

Registry-entry template — same three prerequisites as
``archi/sources/jira.py``'s template; ``data_certification`` and
``run`` ship in ``archi/schemas/operations.yaml``; ``dataset`` comes
from a substrate module. ::

    dqm:
      module: archi.sources.dqm
      class: DQMSource
      ownership_id: <instance>.dqm
      admission_policy:
        producer_id: <instance>.dqm
        producer_kind: source
        trust_label: implicit_legacy_trusted
        admission_mode: fast_track
        authority_scope:
          source_family: <family>
          source_name: dqm
        output_signature:
          nodes:
            - {subtype: data_certification}
            - {subtype: run}
            - {subtype: dataset}
          edges:
            - {src_subtype: data_certification, edge_type: recorded_during, dst_subtype: run}
            - {src_subtype: data_certification, edge_type: references, dst_subtype: dataset}
        output_scope_summary:
          summary: DQM certifications with run-range boundary runs and referenced datasets
          nodes: [data_certification, run, dataset]
          edges:
            - data_certification recorded_during run
            - data_certification references dataset
      source_class: discovery_crawl
      record_identity_kind: remote_id
      record_identity_fields: [certification_id]
      source_revision_kind: content_hash
      deletion_semantics: missing_from_completed_scope
      publication_mode: published_generation
      required_for_baseline: true
      params:
        # cms default; the cms deployment used data/cms/dqm/records.json
        records_path: data/dqm/records.json
      sync:
        triggers: [manual, reconcile]
        default_event_mode: scope_complete
        reconcile_mode: scope_complete
"""
from __future__ import annotations

import re
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

from archi.auth.cache import (
    content_hash,
    content_hash_change_probe,
    load_json,
    resolve_repo_path,
)
from archi.sources._cache_report import skipped_items_status

_GROUP_RE = re.compile(r"Cert_((?:Collisions|Cosmics|Commissioning)\d+)")
_CERT_TYPE_MAP = {
    "golden": "golden",
    "muon": "muon",
    "silver": "silver",
    "dconly": "dcs_only",
    "dcsonly": "dcs_only",
    "dcs_only": "dcs_only",
}


@dataclass(frozen=True)
class DQMRecord:
    filename: str
    cert_name: str
    run_range: tuple[int, int] | None = None
    num_lumi_sections: int = 0
    datasets: tuple[str, ...] = ()

    @property
    def node_id(self) -> str:
        return f"data_certification:{self.cert_name}"


class DQMSource:
    """Cache-backed DQM certification source."""

    name = "dqm"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"

    def __init__(
        self,
        *,
        records_path: str = "data/dqm/records.json",
        base: str | None = None,
    ) -> None:
        self.records_path = records_path
        self.base = base
        self.change_probe = content_hash_change_probe(
            cache_paths=self.cache_paths,
            config={"records_path": self.records_path},
            emit_targets=DQMSource,
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
                required=True,
                cache_path=str(path),
                reason="DQM certification cache file is missing",
                checked_at=_checked_at(),
            )
        records = self._records()
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=True,
            record_count=len(records),
            content_hash=content_hash(self.cache_paths, base=self.base),
            reason="local DQM certification cache present",
            checked_at=_checked_at(),
        )

    def run(self, run_id: str, *, mode: str = "cursor") -> ConnectorRun:
        records, skipped = self._records_with_skips()
        revision = {
            "run_id": run_id,
            "content_hash": content_hash(self.cache_paths, base=self.base),
            "n_records": len(records),
        }

        def _facts() -> Iterator[Any]:
            yield from _facts_for_records(records, revision)

        status, reason = skipped_items_status(
            status="ok",
            reason="local DQM certification cache used",
            record_count=len(records),
            skipped_count=skipped,
        )
        return ConnectorRun(
            facts=_facts(),
            completed_scope=(
                mode in {"scope_complete", "reconcile"} and not skipped
            ),
            run_mode=mode,
            health=ConnectorHealth(
                status=status,
                mode="cache",
                record_count=len(records),
                content_hash=revision["content_hash"],
                reason=reason,
            ),
        )

    def _records(self) -> list[DQMRecord]:
        return self._records_with_skips()[0]

    def _records_with_skips(self) -> tuple[list[DQMRecord], int]:
        payload = load_json(self.records_path, base=self.base)
        if not isinstance(payload, list):
            raise ValueError(
                f"{self.records_path}: expected a JSON list of DQM records"
            )
        records: list[DQMRecord] = []
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue
            cert_name = str(item.get("cert_name") or "").strip()
            if not cert_name:
                skipped += 1
                continue
            run_range = item.get("run_range")
            parsed_range = None
            if isinstance(run_range, list) and len(run_range) == 2:
                parsed_range = (int(run_range[0]), int(run_range[1]))
            records.append(DQMRecord(
                filename=str(item.get("filename") or ""),
                cert_name=cert_name,
                run_range=parsed_range,
                num_lumi_sections=int(item.get("num_lumi_sections") or 0),
                datasets=tuple(
                    str(v) for v in item.get("datasets") or () if v
                ),
            ))
        return records, skipped


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _facts_for_records(
    records: list[DQMRecord],
    revision: dict[str, Any],
) -> Iterator[NodeFact | EdgeFact]:
    seen_runs: set[int] = set()
    seen_datasets: set[str] = set()
    for record in records:
        yield _cert_node(record, revision)
        if record.run_range:
            for run_number in record.run_range:
                if run_number not in seen_runs:
                    seen_runs.add(run_number)
                    yield _run_node(run_number, revision)
                yield EdgeFact(
                    src=record.node_id,
                    dst=f"run:{run_number}",
                    edge_type="recorded_during",
                    attrs={"relationship": "run_range_boundary"},
                    source_record_id={"certification_id": record.cert_name},
                    source_revision=revision,
                )
        for dataset in record.datasets:
            if dataset not in seen_datasets:
                seen_datasets.add(dataset)
                yield _dataset_node(dataset, revision)
            yield EdgeFact(
                src=record.node_id,
                dst=f"dataset:{dataset}",
                edge_type="references",
                attrs={"field": "datasets"},
                source_record_id={"certification_id": record.cert_name},
                source_revision=revision,
            )


def _cert_node(record: DQMRecord, revision: dict[str, Any]) -> NodeFact:
    cert_type = _cert_type(record.cert_name)
    group, year = _group_and_year(record.cert_name)
    run_text = ""
    if record.run_range:
        run_text = f"runs {record.run_range[0]}-{record.run_range[1]}"
    text = " ".join(filter(None, [
        record.cert_name,
        group,
        cert_type,
        run_text,
        " ".join(record.datasets),
    ]))
    attrs: dict[str, Any] = {
        "label": record.cert_name,
        "certification_id": record.cert_name,
        "certification_name": record.cert_name,
        "certification_type": cert_type,
        "dataset_group": group,
        "status": cert_type,
        "filename": record.filename,
        "num_lumi_sections": record.num_lumi_sections,
        "datasets": list(record.datasets),
        "text": text,
    }
    if record.run_range:
        attrs.update({
            "run": f"{record.run_range[0]}-{record.run_range[1]}",
            "run_min": record.run_range[0],
            "run_max": record.run_range[1],
        })
    if record.datasets:
        attrs["dataset"] = ",".join(record.datasets)
    if year is not None:
        attrs["year"] = year
    return NodeFact(
        node_id=record.node_id,
        subtype="data_certification",
        attrs=attrs,
        source_record_id={"certification_id": record.cert_name},
        source_revision=revision,
    )


def _run_node(run_number: int, revision: dict[str, Any]) -> NodeFact:
    return NodeFact(
        node_id=f"run:{run_number}",
        subtype="run",
        attrs={
            "label": f"Run {run_number}",
            "run_number": run_number,
            "text": f"CMS run {run_number}",
        },
        source_record_id={"run_number": run_number},
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


def _cert_type(name: str) -> str:
    lowered = name.lower()
    for key, value in _CERT_TYPE_MAP.items():
        if key in lowered:
            return value
    return "certification"


def _group_and_year(name: str) -> tuple[str, int | None]:
    match = _GROUP_RE.search(name)
    if not match:
        return name, None
    group = match.group(1)
    year_match = re.search(r"(\d{4})$", group)
    if year_match:
        return group, int(year_match.group(1))
    year_match = re.search(r"(\d{2})$", group)
    if year_match:
        return group, 2000 + int(year_match.group(1))
    return group, None
