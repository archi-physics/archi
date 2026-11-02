"""Cache-backed HTCondor source: a pool snapshot (machines, schedds and totals).

It reads the snapshot that ``python -m archi.downloaders.htcondor`` writes under
``${archi_data_root}/htcondor/`` from ``condor_status`` output. The reader never
runs an HTCondor command; ``archi.tools.htcondor`` holds the live read tools,
which reach a pool only through an injected runner.

Cache shape: one JSON object

    {"pool": "submit-pool", "collected_at": "2026-10-20T08:00:00Z",
     "totals": {"claimed_cpus": 410, "unclaimed_cpus": 90, "owner_cpus": 0},
     "machines": [{"name", "state", "activity", "cpus", "memory_mb", "load_avg",
                   "opsys", "arch", "slots"}],
     "schedds":  [{"name", "running_jobs", "idle_jobs", "held_jobs"}]}

A snapshot with no machines is a failed export, not an empty pool.

Registry entry: ``bundles/packs/htcondor/sources/htcondor.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from okg.deployment import EdgeFact, NodeFact

from archi.auth.cache import resolve_repo_path
from archi.sources._cache_report import CacheUnusable
from archi.sources._cluster_cache import CacheSource, clean, number, real
from archi.sources._sdk_adapter import ReaderAdapter


@dataclass(frozen=True)
class PoolSnapshot:
    pool: str
    collected_at: str
    totals: dict[str, int]
    machines: tuple[dict[str, Any], ...]
    schedds: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.machines) + len(self.schedds)


class HTCondorSource(CacheSource):
    """Load an HTCondor pool snapshot from the exporter's cache."""

    name = "htcondor"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
    description = "HTCondor pool snapshot"
    expect = dict

    def __init__(
        self,
        *,
        records_path: str = "htcondor/records.json",
        meta_path: str = "htcondor/meta.json",
        pool_name: str = "pool",
        base: str | None = None,
    ) -> None:
        self.pool_name = pool_name
        super().__init__(records_path=records_path, meta_path=meta_path, base=base, config={"pool_name": pool_name})

    def record_total(self, records: list[PoolSnapshot]) -> int:
        return sum(snap.count for snap in records)

    def parse(self, payload: Any) -> tuple[list[PoolSnapshot], int]:
        skipped = 0
        kept: dict[str, list[dict[str, Any]]] = {}
        for key in ("machines", "schedds"):
            rows = payload.get(key) or []
            if not isinstance(rows, list):
                skipped += 1
                rows = []
            seen: set[str] = set()
            kept[key] = []
            for item in rows:
                name = clean(item.get("name")) if isinstance(item, dict) else ""
                if not name or name in seen:
                    skipped += 1
                    continue
                seen.add(name)
                kept[key].append(item)
        if not kept["machines"]:
            raise CacheUnusable(
                resolve_repo_path(self.records_path, base=self.base),
                "holds no machines; a failed export, not an empty pool",
                status="endpoint_failed",
            )
        totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
        snapshot = PoolSnapshot(
            pool=clean(payload.get("pool")) or self.pool_name,
            collected_at=clean(payload.get("collected_at")),
            totals={clean(k): number(v) for k, v in totals.items() if clean(k)},
            machines=tuple(kept["machines"]),
            schedds=tuple(kept["schedds"]),
        )
        return [snapshot], skipped

    def facts(self, records: list[PoolSnapshot], revision: dict[str, Any]) -> Iterator[Any]:
        for snap in records:
            pool_id = f"condor_pool:{snap.pool}"
            totals = " ".join(f"{k}={v}" for k, v in sorted(snap.totals.items()))
            yield NodeFact(
                node_id=pool_id,
                subtype="condor_pool",
                attrs={
                    "label": snap.pool,
                    "pool_name": snap.pool,
                    "machine_count": len(snap.machines),
                    "schedd_count": len(snap.schedds),
                    "claimed_cpus": snap.totals.get("claimed_cpus", 0),
                    "unclaimed_cpus": snap.totals.get("unclaimed_cpus", 0),
                    "owner_cpus": snap.totals.get("owner_cpus", 0),
                    "observed_at": snap.collected_at,
                    "text": f"HTCondor pool {snap.pool} {len(snap.machines)} machines {totals}".strip(),
                },
                source_record_id={"pool": snap.pool, "kind": "pool", "name": snap.pool},
                source_revision=revision,
            )
            for machine in snap.machines:
                name = clean(machine["name"])
                machine_id = f"condor_machine:{snap.pool}:{name}"
                state = clean(machine.get("state"))
                activity = clean(machine.get("activity"))
                yield NodeFact(
                    node_id=machine_id,
                    subtype="condor_machine",
                    attrs={
                        "label": name,
                        "pool_name": snap.pool,
                        "machine_name": name,
                        "state": state,
                        "activity": activity,
                        "cpus": number(machine.get("cpus")),
                        "memory_mb": number(machine.get("memory_mb")),
                        "load_avg": real(machine.get("load_avg")),
                        "opsys": clean(machine.get("opsys")),
                        "arch": clean(machine.get("arch")),
                        "slots": number(machine.get("slots")),
                        "observed_at": snap.collected_at,
                        "text": f"HTCondor machine {name} in {snap.pool} {state} {activity}".strip(),
                    },
                    source_record_id={"pool": snap.pool, "kind": "machine", "name": name},
                    source_revision=revision,
                )
                yield EdgeFact(
                    src=pool_id,
                    dst=machine_id,
                    edge_type="contains",
                    source_record_id={"pool": snap.pool, "machine": name},
                    source_revision=revision,
                )
            for schedd in snap.schedds:
                name = clean(schedd["name"])
                schedd_id = f"condor_schedd:{snap.pool}:{name}"
                yield NodeFact(
                    node_id=schedd_id,
                    subtype="condor_schedd",
                    attrs={
                        "label": name,
                        "pool_name": snap.pool,
                        "schedd_name": name,
                        "running_jobs": number(schedd.get("running_jobs")),
                        "idle_jobs": number(schedd.get("idle_jobs")),
                        "held_jobs": number(schedd.get("held_jobs")),
                        "observed_at": snap.collected_at,
                        "text": (
                            f"HTCondor schedd {name} in {snap.pool} running {number(schedd.get('running_jobs'))} "
                            f"idle {number(schedd.get('idle_jobs'))} held {number(schedd.get('held_jobs'))}"
                        ),
                    },
                    source_record_id={"pool": snap.pool, "kind": "schedd", "name": name},
                    source_revision=revision,
                )
                yield EdgeFact(
                    src=pool_id,
                    dst=schedd_id,
                    edge_type="contains",
                    source_record_id={"pool": snap.pool, "schedd": name},
                    source_revision=revision,
                )


class HTCondorAdapter(ReaderAdapter):
    """Registry adapter for :class:`HTCondorSource` (see ``ReaderAdapter``)."""

    reader_class = HTCondorSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
