"""Export a Slurm snapshot into the cache ``archi.sources.slurm`` reads.

Run it where the Slurm client commands work (a login node or the controller),
as the account the site gives the exporter::

    python -m archi.downloaders.slurm --cluster submit --since-days 2

It runs four read-only commands, ``sinfo`` (partitions, then nodes),
``scontrol show reservation --oneliner`` and ``sacct`` (job summaries), through
a runner and parses their parseable output. No credential is read: Slurm's own
account on the host is the access. The commands never change the scheduler.
Tests inject a fake runner and never start a process; see ``tests/sources/test_slurm.py``.

``records.json`` and ``meta.json`` are written atomically under ``--output-dir``
(default ``slurm`` under ``ARCHI_DATA_ROOT``, where the pack reads it). An export whose ``sinfo``
fails writes nothing: a half snapshot would retract the missing half.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from archi.sources._cluster_cache import write_cache
from archi.sources.slurm import expand_nodelist
from archi.tools._live import Runner, RunResult, subprocess_runner

DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
TIMEOUT = 120

PARTITION_FORMAT = "%P|%a|%l|%D|%C"
NODE_FORMAT = "%N|%P|%T|%c|%m|%G|%f|%O|%e|%E"
SACCT_FIELDS = ("JobID", "JobName", "User", "Account", "Partition", "State", "Submit", "Start", "End",
                "Elapsed", "NodeList", "ReqCPUS", "ReqMem", "ExitCode")


class ExportError(RuntimeError):
    pass


def _run(runner: Runner, argv: Sequence[str]) -> str:
    result = runner(argv, timeout=TIMEOUT)
    if result.returncode != 0:
        raise ExportError(f"{argv[0]} exited {result.returncode}: {result.stderr.strip()[:200]}")
    return result.stdout


def parse_partitions(text: str) -> list[dict[str, Any]]:
    """``sinfo -h -o '%P|%a|%l|%D|%C'``: name (``*`` marks the default), availability, time limit, node count, CPUs A/I/O/T."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5 or not parts[0]:
            continue
        name = parts[0].rstrip("*")
        cpus = parts[4].split("/")
        rows.append({
            "name": name,
            "state": "up" if parts[1] == "up" else parts[1],
            "default": parts[0].endswith("*"),
            "max_time": parts[2],
            "total_nodes": int(parts[3]) if parts[3].isdigit() else 0,
            "total_cpus": int(cpus[3]) if len(cpus) == 4 and cpus[3].isdigit() else 0,
        })
    return rows


def parse_nodes(text: str) -> list[dict[str, Any]]:
    """``sinfo -h -N -o`` rows: a node in several partitions has one row each, merged here."""
    nodes: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("|", 9)
        if len(parts) < 10 or not parts[0].strip():
            continue
        name, partition, state, cpus, memory, gres, features, load, free_mem, reason = (p.strip() for p in parts)
        node = nodes.setdefault(name, {
            "name": name, "state": state, "reason": "" if reason == "none" else reason, "partitions": [],
            "cpus": cpus, "memory_mb": memory, "gres": "" if gres == "(null)" else gres,
            "features": [] if features == "(null)" else features.split(","), "cpu_load": load,
            "free_mem_mb": free_mem,
        })
        part = partition.rstrip("*")
        if part and part not in node["partitions"]:
            node["partitions"].append(part)
    return list(nodes.values())


def _null(value: str) -> str:
    return "" if value == "(null)" else value


_KV = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|\s*$)")


def parse_reservations(text: str) -> list[dict[str, Any]]:
    """``scontrol show reservation --oneliner``: one ``Key=Value`` line per reservation."""
    rows = []
    for line in text.splitlines():
        fields = dict(_KV.findall(line.strip()))
        if not fields.get("ReservationName"):
            continue
        rows.append({
            "name": fields["ReservationName"],
            "start": fields.get("StartTime", ""),
            "end": fields.get("EndTime", ""),
            "nodes": expand_nodelist(fields.get("Nodes", "")),
            "users": _null(fields.get("Users", "")),
            "accounts": _null(fields.get("Accounts", "")),
            "flags": fields.get("Flags", ""),
        })
    return rows


def parse_jobs(text: str) -> list[dict[str, Any]]:
    """``sacct -a -X -P --noheader -o ...``: one summary row per job allocation."""
    rows = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("|")
        if len(parts) != len(SACCT_FIELDS) or not parts[0].strip():
            continue
        row = dict(zip(SACCT_FIELDS, parts))
        rows.append({
            "job_id": row["JobID"], "name": row["JobName"], "user": row["User"], "account": row["Account"],
            "partition": row["Partition"], "state": row["State"].split()[0] if row["State"] else "",
            "submit": row["Submit"], "start": row["Start"], "end": row["End"], "elapsed": row["Elapsed"],
            "nodelist": expand_nodelist(row["NodeList"]) if row["NodeList"] not in ("None assigned", "") else [],
            "req_cpus": row["ReqCPUS"], "req_mem": row["ReqMem"], "exit_code": row["ExitCode"],
        })
    return rows


def export(runner: Runner, cluster: str, *, since_days: int = 1, max_jobs: int = 5000) -> dict[str, Any]:
    """The snapshot object ``archi.sources.slurm`` reads. Raises ``ExportError`` on any failed command."""
    partitions = parse_partitions(_run(runner, ["sinfo", "--noheader", "-o", PARTITION_FORMAT]))
    nodes = parse_nodes(_run(runner, ["sinfo", "--noheader", "-N", "-o", NODE_FORMAT]))
    if not partitions and not nodes:
        raise ExportError("sinfo listed no partitions and no nodes; not writing an empty snapshot")
    reservations = parse_reservations(_run(runner, ["scontrol", "show", "reservation", "--oneliner"]))
    jobs = parse_jobs(_run(runner, [
        "sacct", "--allusers", "--allocations", "--parsable2", "--noheader", f"--starttime=now-{since_days}days",
        f"--format={','.join(SACCT_FIELDS)}",
    ]))[:max_jobs]
    return {
        "cluster": cluster,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "partitions": partitions, "nodes": nodes, "reservations": reservations, "jobs": jobs,
    }


def default_output_dir() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV)
    return (Path(raw).expanduser() if raw else Path.cwd()) / "slurm"


def main(argv: Optional[Sequence[str]] = None, *, runner: Optional[Runner] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cluster", required=True, help="cluster name used in the record ids")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--since-days", type=int, default=1, help="how far back sacct looks")
    parser.add_argument("--max-jobs", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        snapshot = export(runner or subprocess_runner, args.cluster, since_days=args.since_days, max_jobs=args.max_jobs)
    except (ExportError, OSError) as exc:
        print(f"slurm export stopped: {exc}", file=sys.stderr)
        return 1
    total = sum(len(snapshot[k]) for k in ("partitions", "nodes", "reservations", "jobs"))
    info = write_cache(Path(args.output_dir) if args.output_dir else default_output_dir(), snapshot, total, cluster=args.cluster)
    print(json.dumps({"record_count": info["record_count"], "content_hash": info["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
