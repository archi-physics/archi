"""HTCondor live read tools: the pool's machines and the job queue.

Read-only, boundary ``external_live``. There are no act tools for HTCondor in
3.0: nothing in the cluster-manager bundle changes a pool. Wiring for an
instance's ``agent_tools``::

    agent_tools:
      archi_condor_status:
        module: archi.tools.htcondor
        function: archi_condor_status
        boundary: external_live
        optional: true
      archi_condor_q:
        module: archi.tools.htcondor
        function: archi_condor_q
        boundary: external_live
        optional: true

Commands (``condor_status``, ``condor_q``) run through an injected ``runner``
(see :mod:`archi.tools._live`) and are built as argument lists from validated
names: an argument can never become an option or a ClassAd expression.
Credentials: none in archi; the runner runs as the account the instance gives it.
"""
from __future__ import annotations

from typing import Any, Optional

from archi.tools import _live
from archi.tools._live import Runner, ToolError, bounded, name_arg, run_read

__all__ = ["TOOLS", "archi_condor_status", "archi_condor_q"]

TOOLS: dict[str, dict[str, Any]] = {
    "archi_condor_status": {"kind": "read", "idempotent": True, "boundary": _live.BOUNDARY},
    "archi_condor_q": {"kind": "read", "idempotent": True, "boundary": _live.BOUNDARY},
}

STATUS_ATTRS = ("Machine", "State", "Activity", "Cpus", "Memory", "LoadAvg", "OpSys", "Arch")
Q_ATTRS = ("ClusterId", "ProcId", "Owner", "JobStatus", "RemoteHost", "HoldReason")
#: JobStatus codes in the HTCondor manual
JOB_STATUS = {"1": "Idle", "2": "Running", "3": "Removed", "4": "Completed", "5": "Held", "6": "Transferring", "7": "Suspended"}
_STATES = {"owner", "unclaimed", "matched", "claimed", "preempting", "drained", "backfill"}


def _tab_rows(names: tuple[str, ...]):
    def parse(text: str) -> list[dict[str, Any]]:
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", len(names) - 1)
            parts += [""] * (len(names) - len(parts))
            rows.append({name: part.strip() for name, part in zip(names, parts)})
        return rows

    return parse


def archi_condor_status(
    machine: Optional[str] = None,
    state: Optional[str] = None,
    max_results: int = 50,
    runner: Optional[Runner] = None,
) -> dict[str, Any]:
    """Live ``condor_status``: slots with state, activity, cores, memory and load."""
    limit = bounded(max_results)
    argv = ["condor_status", "-limit", str(limit + 1), "-af:t", *STATUS_ATTRS]
    try:
        constraints = []
        if machine:
            constraints.append(f'Machine == "{name_arg(machine, "machine")}"')
        if state:
            if state.lower() not in _STATES:
                raise ToolError(f"state {state!r} is not one of {sorted(_STATES)}")
            constraints.append(f'State == "{state.capitalize()}"')
        if constraints:
            argv += ["-constraint", " && ".join(constraints)]
    except ToolError as exc:
        return _live.error_payload(argv, str(exc))
    return run_read(argv, runner, _tab_rows(STATUS_ATTRS), limit)


def _q_rows(text: str) -> list[dict[str, Any]]:
    rows = _tab_rows(Q_ATTRS)(text)
    for row in rows:
        row["JobStatus"] = JOB_STATUS.get(row["JobStatus"], row["JobStatus"])
    return rows


def archi_condor_q(
    owner: Optional[str] = None,
    cluster_id: Optional[str] = None,
    held_only: bool = False,
    max_results: int = 50,
    runner: Optional[Runner] = None,
) -> dict[str, Any]:
    """Live ``condor_q``: jobs, optionally for one owner, one cluster id or the held ones."""
    limit = bounded(max_results)
    argv = ["condor_q", "-allusers", "-limit", str(limit + 1), "-af:t", *Q_ATTRS]
    try:
        constraints = []
        if owner:
            constraints.append(f'Owner == "{name_arg(owner, "owner")}"')
        if cluster_id:
            text = str(cluster_id)
            if not (text.isascii() and text.isdigit()):
                raise ToolError(f"cluster id {cluster_id!r} is not a number")
            constraints.append(f"ClusterId == {int(text)}")
        if held_only:
            constraints.append("JobStatus == 5")
        if constraints:
            argv += ["-constraint", " && ".join(constraints)]
    except ToolError as exc:
        return _live.error_payload(argv, str(exc))
    return run_read(argv, runner, _q_rows, limit)
