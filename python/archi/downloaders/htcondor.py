"""Export an HTCondor pool snapshot into the cache ``archi.sources.htcondor`` reads.

Run it where the HTCondor client commands work, as the account the site gives
the exporter::

    python -m archi.downloaders.htcondor --pool submit-pool

It runs two read-only ``condor_status`` queries (slots, then schedds) through a
runner and sums the CPUs by slot state. No credential is read: the host's
HTCondor configuration is the access. The commands never change the pool.
Tests inject a fake runner and never start a process; see ``tests/sources/test_htcondor.py``.

``records.json`` and ``meta.json`` are written atomically under ``--output-dir``
(default ``htcondor`` under ``ARCHI_DATA_ROOT``, where the pack reads it). A query that fails or an
empty pool writes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from archi.sources._cluster_cache import number, write_cache
from archi.tools._live import Runner, subprocess_runner

DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
TIMEOUT = 120
MACHINE_ATTRS = ("Machine", "State", "Activity", "Cpus", "Memory", "LoadAvg", "OpSys", "Arch")
SCHEDD_ATTRS = ("Name", "TotalRunningJobs", "TotalIdleJobs", "TotalHeldJobs")


class ExportError(RuntimeError):
    pass


def _rows(runner: Runner, argv: Sequence[str], names: Sequence[str]) -> list[dict[str, str]]:
    result = runner(argv, timeout=TIMEOUT)
    if result.returncode != 0:
        raise ExportError(f"{argv[0]} exited {result.returncode}: {result.stderr.strip()[:200]}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        parts += [""] * (len(names) - len(parts))
        rows.append(dict(zip(names, (p.strip() for p in parts))))
    return rows


def export(runner: Runner, pool: str) -> dict[str, Any]:
    """The snapshot object ``archi.sources.htcondor`` reads; one machine per name (slots summed)."""
    slots = _rows(runner, ["condor_status", "-af:t", *MACHINE_ATTRS], MACHINE_ATTRS)
    if not slots:
        raise ExportError("condor_status listed no slots; not writing an empty snapshot")
    machines: dict[str, dict[str, Any]] = {}
    totals = {"claimed_cpus": 0, "unclaimed_cpus": 0, "owner_cpus": 0}
    for slot in slots:
        # a slot name is slot1@host: group the slots of one host
        host = slot["Machine"] or "unknown"
        machine = machines.setdefault(host, {
            "name": host, "state": slot["State"], "activity": slot["Activity"], "cpus": 0, "memory_mb": 0,
            "load_avg": slot["LoadAvg"], "opsys": slot["OpSys"], "arch": slot["Arch"], "slots": 0,
        })
        machine["cpus"] += number(slot["Cpus"])
        machine["memory_mb"] += number(slot["Memory"])
        machine["slots"] += 1
        key = f"{slot['State'].lower()}_cpus"
        if key in totals:
            totals[key] += number(slot["Cpus"])
    schedds = [
        {"name": r["Name"], "running_jobs": number(r["TotalRunningJobs"]), "idle_jobs": number(r["TotalIdleJobs"]),
         "held_jobs": number(r["TotalHeldJobs"])}
        for r in _rows(runner, ["condor_status", "-schedd", "-af:t", *SCHEDD_ATTRS], SCHEDD_ATTRS) if r["Name"]
    ]
    return {
        "pool": pool, "collected_at": datetime.now(timezone.utc).isoformat(), "totals": totals,
        "machines": list(machines.values()), "schedds": schedds,
    }


def default_output_dir() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV)
    return (Path(raw).expanduser() if raw else Path.cwd()) / "htcondor"


def main(argv: Optional[Sequence[str]] = None, *, runner: Optional[Runner] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", required=True, help="pool name used in the record ids")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    try:
        snapshot = export(runner or subprocess_runner, args.pool)
    except (ExportError, OSError) as exc:
        print(f"htcondor export stopped: {exc}", file=sys.stderr)
        return 1
    total = len(snapshot["machines"]) + len(snapshot["schedds"])
    info = write_cache(Path(args.output_dir) if args.output_dir else default_output_dir(), snapshot, total, pool=args.pool)
    print(json.dumps({"record_count": info["record_count"], "content_hash": info["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
