"""Cache-backed node-health source: per-node check results from a health export.

The PACT names the node-health pack without describing its input, so this is
the smallest useful shape: one report per compute node, each with the named
checks a health tool ran on it (LBNL NHC, a Nagios or Prometheus export, a
site's own script). It reads ``${archi_data_root}/node-health/records.json``,
which the site's exporter writes; nothing in archi runs a check.

Cache shape: a JSON list

    {"node": "node042", "cluster": "submit", "observed_at": "2026-10-20T08:00:00Z",
     "checks": [{"name": "disk_scratch", "status": "ok" | "warn" | "fail",
                 "detail": "...", "observed_at": "..."}]}

The node's overall ``health`` is the worst check (``fail`` -> ``unhealthy``,
``warn`` -> ``degraded``, else ``healthy``) unless the report gives a
``status`` of its own. A report with no node name is skipped. The checks live
on ``node_health_check`` records so a search finds the failing check by name.

Registry entry: ``bundles/packs/node-health/sources/node_health.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from okg.deployment import EdgeFact, NodeFact

from archi.sources._cluster_cache import CacheSource, clean
from archi.sources._sdk_adapter import ReaderAdapter

_CHECK_STATUS = {"ok": "ok", "pass": "ok", "passed": "ok", "good": "ok", "warn": "warn", "warning": "warn",
                 "fail": "fail", "failed": "fail", "critical": "fail", "error": "fail"}
_HEALTH = {"fail": "unhealthy", "warn": "degraded", "ok": "healthy"}
_REPORTED = {"healthy": "healthy", "degraded": "degraded", "unhealthy": "unhealthy", "ok": "healthy",
             "warn": "degraded", "warning": "degraded", "fail": "unhealthy", "critical": "unhealthy"}


@dataclass(frozen=True)
class NodeReport:
    cluster: str
    node: str
    observed_at: str
    health: str
    checks: tuple[tuple[str, str, str, str], ...]  # name, status, detail, observed_at

    @property
    def node_id(self) -> str:
        return f"compute_node:{self.cluster}:{self.node}"


class NodeHealthSource(CacheSource):
    """Load node-health reports from the exporter's cache."""

    name = "node_health"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
    description = "node-health reports"

    def __init__(
        self,
        *,
        records_path: str = "node-health/records.json",
        cluster_name: str = "cluster",
        base: str | None = None,
    ) -> None:
        self.cluster_name = cluster_name
        super().__init__(records_path=records_path, base=base, config={"cluster_name": cluster_name})

    def parse(self, payload: Any) -> tuple[list[NodeReport], int]:
        reports: list[NodeReport] = []
        seen: set[str] = set()
        skipped = 0
        for item in payload:
            report = _report(item, self.cluster_name) if isinstance(item, dict) else None
            if report is None or report.node_id in seen:
                skipped += 1
                continue
            seen.add(report.node_id)
            reports.append(report)
        return reports, skipped

    def facts(self, records: list[NodeReport], revision: dict[str, Any]) -> Iterator[Any]:
        for report in records:
            failing = [name for name, status, _, _ in report.checks if status == "fail"]
            warning = [name for name, status, _, _ in report.checks if status == "warn"]
            yield NodeFact(
                node_id=report.node_id,
                subtype="compute_node",
                attrs={
                    "label": report.node,
                    "cluster": report.cluster,
                    "node_name": report.node,
                    "health": report.health,
                    "failing_checks": failing,
                    "warning_checks": warning,
                    "check_count": len(report.checks),
                    "observed_at": report.observed_at,
                    "text": " ".join(filter(None, [
                        f"node {report.node} on {report.cluster} is {report.health}",
                        "failing " + " ".join(failing) if failing else "",
                        "warning " + " ".join(warning) if warning else "",
                    ])),
                },
                source_record_id={"cluster": report.cluster, "kind": "node", "name": report.node},
                source_revision=revision,
            )
            for name, status, detail, observed in report.checks:
                check_id = f"node_health_check:{report.cluster}:{report.node}:{name}"
                yield NodeFact(
                    node_id=check_id,
                    subtype="node_health_check",
                    attrs={
                        "label": f"{report.node} {name}",
                        "cluster": report.cluster,
                        "node_name": report.node,
                        "check_name": name,
                        "status": status,
                        "detail": detail,
                        "observed_at": observed or report.observed_at,
                        "text": f"node {report.node} check {name} {status} {detail}".strip(),
                    },
                    source_record_id={"cluster": report.cluster, "kind": "check", "name": f"{report.node}/{name}"},
                    source_revision=revision,
                )
                yield EdgeFact(
                    src=report.node_id,
                    dst=check_id,
                    edge_type="contains",
                    source_record_id={"cluster": report.cluster, "node": report.node, "check": name},
                    source_revision=revision,
                )


def _report(item: dict[str, Any], default_cluster: str) -> NodeReport | None:
    node = clean(item.get("node") or item.get("hostname") or item.get("name"))
    if not node:
        return None
    observed_at = clean(item.get("observed_at"))
    checks: dict[str, tuple[str, str, str, str]] = {}
    for check in item.get("checks") or []:
        if not isinstance(check, dict) or not clean(check.get("name")):
            continue
        name = clean(check["name"])
        status = _CHECK_STATUS.get(clean(check.get("status")).lower(), "warn")
        checks[name] = (name, status, clean(check.get("detail")), clean(check.get("observed_at")))
    worst = "fail" if any(c[1] == "fail" for c in checks.values()) else (
        "warn" if any(c[1] == "warn" for c in checks.values()) else "ok")
    health = _REPORTED.get(clean(item.get("status")).lower()) or _HEALTH[worst]
    return NodeReport(
        cluster=clean(item.get("cluster")) or default_cluster,
        node=node,
        observed_at=observed_at,
        health=health,
        checks=tuple(checks.values()),
    )


class NodeHealthAdapter(ReaderAdapter):
    """Registry adapter for :class:`NodeHealthSource` (see ``ReaderAdapter``)."""

    reader_class = NodeHealthSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
