"""Cache-backed Slurm source: partitions, nodes, reservations and job summaries.

It reads the snapshot that ``python -m archi.downloaders.slurm`` writes under
``${archi_data_root}/slurm/`` from ``sinfo``, ``scontrol show reservation`` and
``sacct`` output (or from slurmrestd). The reader never runs a Slurm command;
the live tools in ``archi.tools.slurm`` are the only code here that can reach a
scheduler, and only through an injected runner.

Cache shape: one JSON object

    {"cluster": "submit", "collected_at": "2026-10-20T08:00:00Z",
     "partitions":   [{"name", "state", "default", "max_time", "total_nodes", "total_cpus"}],
     "nodes":        [{"name", "state", "reason", "partitions": [..], "cpus", "memory_mb",
                       "gres", "features": [..], "cpu_load", "free_mem_mb", "boot_time"}],
     "reservations": [{"name", "start", "end", "nodes": [..], "users", "accounts", "flags"}],
     "jobs":         [{"job_id", "name", "user", "account", "partition", "state", "submit",
                       "start", "end", "elapsed", "nodelist": [..], "req_cpus", "req_mem",
                       "exit_code"}]}

A snapshot with no partitions and no nodes is not a snapshot of anything and
reads as a failed export. Job users are kept as an attribute, not as person
nodes: a scheduler's accounts are not people the knowledge base should hold.

Registry entry: ``bundles/packs/slurm/sources/slurm.yaml``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

from okg.deployment import EdgeFact, NodeFact

from archi.auth.cache import resolve_repo_path
from archi.sources._cache_report import CacheUnusable
from archi.sources._cluster_cache import CacheSource, clean, clean_list, number, real
from archi.sources._sdk_adapter import ReaderAdapter

#: Slurm marks a node state with flags: ``*`` not responding, ``~`` powered off,
#: ``#`` powering up, ``%`` powering down, ``$`` maintenance, ``@`` reboot pending.
_STATE_FLAGS = re.compile(r"[*~#%$@^!]+$")
_NODE_RANGE = re.compile(r"^(?P<prefix>[^\[\]]*)\[(?P<body>[^\[\]]+)\](?P<suffix>.*)$")


def base_state(state: str) -> str:
    """``DRAIN*`` -> ``drain``; ``mixed+drain`` -> ``mixed``."""
    first = re.split(r"[+\s]", state.strip(), maxsplit=1)[0] if state.strip() else ""
    return _STATE_FLAGS.sub("", first).lower()


def expand_nodelist(value: Any) -> list[str]:
    """``n[01-03,07],gpu1`` -> ``[n01, n02, n03, n07, gpu1]``. A list passes through."""
    if isinstance(value, (list, tuple)):
        return [name for name in (clean(v) for v in value) if name]
    text = clean(value)
    names: list[str] = []
    depth, start = 0, 0
    parts: list[str] = []
    for i, ch in enumerate(text):
        depth += ch == "["
        depth -= ch == "]"
        if ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    for part in (p.strip() for p in parts):
        if not part:
            continue
        match = _NODE_RANGE.match(part)
        if not match:
            names.append(part)
            continue
        for piece in match["body"].split(","):
            lo, _, hi = piece.partition("-")
            if hi and lo.isdigit() and hi.isdigit():
                width = len(lo)
                names.extend(f"{match['prefix']}{n:0{width}d}{match['suffix']}" for n in range(int(lo), int(hi) + 1))
            else:
                names.append(f"{match['prefix']}{piece}{match['suffix']}")
    return names


@dataclass(frozen=True)
class Snapshot:
    cluster: str
    collected_at: str
    partitions: tuple[dict[str, Any], ...]
    nodes: tuple[dict[str, Any], ...]
    reservations: tuple[dict[str, Any], ...]
    jobs: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.partitions) + len(self.nodes) + len(self.reservations) + len(self.jobs)


class SlurmSource(CacheSource):
    """Load a Slurm snapshot from the exporter's cache."""

    name = "slurm"
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
    description = "Slurm snapshot"
    expect = dict

    def __init__(
        self,
        *,
        records_path: str = "slurm/records.json",
        meta_path: str = "slurm/meta.json",
        cluster_name: str = "cluster",
        base: str | None = None,
    ) -> None:
        self.cluster_name = cluster_name
        super().__init__(
            records_path=records_path, meta_path=meta_path, base=base, config={"cluster_name": cluster_name}
        )

    def parse(self, payload: Any) -> tuple[list[Snapshot], int]:
        skipped = 0
        out: dict[str, list[dict[str, Any]]] = {}
        for key, ident in (("partitions", "name"), ("nodes", "name"), ("reservations", "name"), ("jobs", "job_id")):
            rows = payload.get(key) or []
            kept = []
            seen: set[str] = set()
            for item in rows if isinstance(rows, list) else []:
                name = clean(item.get(ident)) if isinstance(item, dict) else ""
                if not name or name in seen:
                    skipped += 1
                    continue
                seen.add(name)
                kept.append(item)
            if not isinstance(rows, list):
                skipped += 1
            out[key] = kept
        snapshot = Snapshot(
            cluster=clean(payload.get("cluster")) or self.cluster_name,
            collected_at=clean(payload.get("collected_at")),
            partitions=tuple(out["partitions"]),
            nodes=tuple(out["nodes"]),
            reservations=tuple(out["reservations"]),
            jobs=tuple(out["jobs"]),
        )
        if not snapshot.partitions and not snapshot.nodes:
            # nothing to describe the cluster by: a failed export, not a quiet cluster
            raise CacheUnusable(
                resolve_repo_path(self.records_path, base=self.base),
                "holds no partitions and no nodes; a failed export, not an empty cluster",
                status="endpoint_failed",
            )
        return [snapshot], skipped

    def record_total(self, records: list[Snapshot]) -> int:
        return sum(snap.count for snap in records)

    def facts(self, records: list[Snapshot], revision: dict[str, Any]) -> Iterator[Any]:
        for snap in records:
            yield from self._snapshot_facts(snap, revision)

    def _snapshot_facts(self, snap: Snapshot, revision: dict[str, Any]) -> Iterator[Any]:
        def nid(kind: str, name: str) -> str:
            return f"{kind}:{snap.cluster}:{name}"

        partitions = {clean(p["name"]) for p in snap.partitions}
        nodes = {clean(n["name"]) for n in snap.nodes}
        for part in snap.partitions:
            name = clean(part["name"])
            yield NodeFact(
                node_id=nid("slurm_partition", name),
                subtype="slurm_partition",
                attrs={
                    "label": f"{snap.cluster}/{name}",
                    "cluster": snap.cluster,
                    "partition_name": name,
                    "state": clean(part.get("state")),
                    "is_default": bool(part.get("default")),
                    "max_time": clean(part.get("max_time")),
                    "total_nodes": number(part.get("total_nodes")),
                    "total_cpus": number(part.get("total_cpus")),
                    "observed_at": snap.collected_at,
                    "text": f"Slurm partition {name} on {snap.cluster} {clean(part.get('state'))}",
                },
                source_record_id={"cluster": snap.cluster, "kind": "partition", "name": name},
                source_revision=revision,
            )
        for node in snap.nodes:
            name = clean(node["name"])
            state = clean(node.get("state"))
            node_parts = clean_list(node.get("partitions"))
            yield NodeFact(
                node_id=nid("slurm_node", name),
                subtype="slurm_node",
                attrs={
                    "label": name,
                    "cluster": snap.cluster,
                    "node_name": name,
                    "state": state,
                    "state_base": base_state(state),
                    "reason": clean(node.get("reason")),
                    "partitions": node_parts,
                    "cpus": number(node.get("cpus")),
                    "memory_mb": number(node.get("memory_mb")),
                    "gres": clean(node.get("gres")),
                    "features": clean_list(node.get("features")),
                    "cpu_load": real(node.get("cpu_load")),
                    "free_mem_mb": number(node.get("free_mem_mb")),
                    "boot_time": clean(node.get("boot_time")),
                    "observed_at": snap.collected_at,
                    "text": " ".join(filter(None, [
                        f"Slurm node {name} on {snap.cluster}", state, clean(node.get("reason")),
                        " ".join(node_parts), clean(node.get("gres")),
                    ])),
                },
                source_record_id={"cluster": snap.cluster, "kind": "node", "name": name},
                source_revision=revision,
            )
            for part in node_parts:
                if part in partitions:
                    yield EdgeFact(
                        src=nid("slurm_partition", part),
                        dst=nid("slurm_node", name),
                        edge_type="contains",
                        source_record_id={"cluster": snap.cluster, "partition": part, "node": name},
                        source_revision=revision,
                    )
        for res in snap.reservations:
            name = clean(res["name"])
            res_nodes = expand_nodelist(res.get("nodes"))
            yield NodeFact(
                node_id=nid("slurm_reservation", name),
                subtype="slurm_reservation",
                attrs={
                    "label": name,
                    "cluster": snap.cluster,
                    "reservation_name": name,
                    "start": clean(res.get("start")),
                    "end": clean(res.get("end")),
                    "users": clean(res.get("users")),
                    "accounts": clean(res.get("accounts")),
                    "flags": clean(res.get("flags")),
                    "nodes": res_nodes,
                    "observed_at": snap.collected_at,
                    "text": f"Slurm reservation {name} on {snap.cluster} {' '.join(res_nodes)} {clean(res.get('flags'))}",
                },
                source_record_id={"cluster": snap.cluster, "kind": "reservation", "name": name},
                source_revision=revision,
            )
            for node in res_nodes:
                if node in nodes:
                    yield EdgeFact(
                        src=nid("slurm_reservation", name),
                        dst=nid("slurm_node", node),
                        edge_type="references",
                        source_record_id={"cluster": snap.cluster, "reservation": name, "node": node},
                        source_revision=revision,
                    )
        for job in snap.jobs:
            job_id = clean(job["job_id"])
            job_nodes = expand_nodelist(job.get("nodelist"))
            partition = clean(job.get("partition"))
            state = clean(job.get("state"))
            yield NodeFact(
                node_id=nid("slurm_job", job_id),
                subtype="slurm_job",
                attrs={
                    "label": f"{snap.cluster} job {job_id}",
                    "cluster": snap.cluster,
                    "job_id": job_id,
                    "job_name": clean(job.get("name")),
                    "user": clean(job.get("user")),
                    "account": clean(job.get("account")),
                    "partition": partition,
                    "state": state,
                    "submit_time": clean(job.get("submit")),
                    "start_time": clean(job.get("start")),
                    "end_time": clean(job.get("end")),
                    "elapsed": clean(job.get("elapsed")),
                    "nodelist": job_nodes,
                    "req_cpus": number(job.get("req_cpus")),
                    "req_mem": clean(job.get("req_mem")),
                    "exit_code": clean(job.get("exit_code")),
                    "observed_at": snap.collected_at,
                    "text": " ".join(filter(None, [
                        f"Slurm job {job_id} on {snap.cluster}", clean(job.get("name")), state, partition,
                        " ".join(job_nodes), clean(job.get("exit_code")),
                    ])),
                },
                source_record_id={"cluster": snap.cluster, "kind": "job", "name": job_id},
                source_revision=revision,
            )
            if partition in partitions:
                yield EdgeFact(
                    src=nid("slurm_job", job_id),
                    dst=nid("slurm_partition", partition),
                    edge_type="member_of",
                    source_record_id={"cluster": snap.cluster, "job_id": job_id, "partition": partition},
                    source_revision=revision,
                )
            for node in job_nodes:
                if node in nodes:
                    yield EdgeFact(
                        src=nid("slurm_node", node),
                        dst=nid("slurm_job", job_id),
                        edge_type="executes",
                        source_record_id={"cluster": snap.cluster, "job_id": job_id, "node": node},
                        source_revision=revision,
                    )


class SlurmAdapter(ReaderAdapter):
    """Registry adapter for :class:`SlurmSource` (see ``ReaderAdapter``)."""

    reader_class = SlurmSource
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
