"""Slurm live tools: read the queue and the nodes, drain or resume a node, requeue a job.

Every tool here has the boundary ``external_live``: what it returns is a live
reading from the scheduler, not graph evidence. Wiring for an instance's
``agent_tools`` (the two read tools; the act tools are not agent tools)::

    agent_tools:
      archi_slurm_squeue:
        module: archi.tools.slurm
        function: archi_slurm_squeue
        boundary: external_live
        optional: true
      archi_slurm_sinfo:
        module: archi.tools.slurm
        function: archi_slurm_sinfo
        boundary: external_live
        optional: true

**Read tools** (``archi_slurm_squeue``, ``archi_slurm_sinfo``) build one
``squeue`` or ``sinfo`` command from validated arguments and return bounded rows.

**Act tools** (``drain_node``, ``resume_node``, ``requeue_job``) are the mediators
of the action classes in this pack's ``action_classes.yaml``, each at most L2:

    drain_node(payload={"node", "reason"}, context=...)
    resume_node(payload={"node"}, context=...)
    requeue_job(payload={"job_id"}, context=...)

They are not agent tools. The ops runtime calls them only after a person
approved the escalation, and each checks that again before it builds a command
(:func:`archi.tools._live.require_approval`): a call without an approved
decision raises ``ActRefused`` and runs nothing. Each is idempotent. The
runtime passes the run's ``idempotency_key``; the tool reads the node or job
first and does nothing when it is already in the requested state, and a drain
writes the key into the node's Slurm reason so a repeat finds its own mark.

The commands (``squeue``, ``sinfo``, ``scontrol``) run through an injected
``runner``; see :mod:`archi.tools._live`. Credentials: none in archi. The
runner runs as the account the instance gives it (a Slurm operator account on
the controller or a login node).
"""
from __future__ import annotations

import subprocess
from typing import Any, Mapping, Optional

from archi.tools import _live
from archi.tools._live import (
    ActRefused,
    Runner,
    ToolError,
    Verifier,
    bounded,
    job_id_arg,
    name_arg,
    nodelist_arg,
    outcome,
    require_approval,
    run_read,
    split_fields,
    subprocess_runner,
    validate_payload,
)

__all__ = [
    "ActRefused", "ToolError", "TOOLS", "PAYLOAD_SCHEMAS",
    "archi_slurm_squeue", "archi_slurm_sinfo", "drain_node", "resume_node", "requeue_job",
]

SQUEUE_FIELDS = ("job_id", "partition", "name", "user", "state", "elapsed", "nodes", "reason_or_nodelist")
SQUEUE_FORMAT = "%i|%P|%j|%u|%T|%M|%D|%R"
SINFO_FIELDS = ("node", "partition", "state", "cpus", "memory_mb", "reason")
SINFO_FORMAT = "%N|%P|%T|%c|%m|%E"
JOB_FORMAT = "%i|%T|%k"  # id, state, comment
#: what archi writes into a node's reason or a job's comment, followed by [<idempotency key>]
MARK = "archi"

#: states `scontrol` reports for a node that is out of service by an operator or by Slurm
_OUT_OF_SERVICE = ("drain", "drng", "down", "fail", "failg", "maint")

PAYLOAD_SCHEMAS: dict[str, dict[str, Any]] = {
    "drain_node": {
        "type": "object",
        "required": ["node", "reason"],
        "additionalProperties": False,
        "properties": {
            "node": {"type": "string", "pattern": _live.NAME_PATTERN},
            "reason": {"type": "string", "minLength": 3, "maxLength": 200, "pattern": r"^[^\x00-\x1f\x7f]+$"},
        },
    },
    "resume_node": {
        "type": "object",
        "required": ["node"],
        "additionalProperties": False,
        "properties": {
            "node": {"type": "string", "pattern": _live.NAME_PATTERN},
            # without it only a node archi drained is resumed; set, any drained or down node is
            "force": {"type": "boolean"},
        },
    },
    "requeue_job": {
        "type": "object",
        "required": ["job_id"],
        "additionalProperties": False,
        "properties": {"job_id": {"type": "string", "pattern": _live.JOB_ID_PATTERN}},
    },
}

#: What the pack ships, in one place: the tests hold this equal to pack.yaml and action_classes.yaml.
TOOLS: dict[str, dict[str, Any]] = {
    "archi_slurm_squeue": {"kind": "read", "idempotent": True, "boundary": _live.BOUNDARY},
    "archi_slurm_sinfo": {"kind": "read", "idempotent": True, "boundary": _live.BOUNDARY},
    "drain_node": {"kind": "act", "idempotent": True, "boundary": _live.BOUNDARY, "max_level": "L2"},
    "resume_node": {"kind": "act", "idempotent": True, "boundary": _live.BOUNDARY, "max_level": "L2"},
    "requeue_job": {"kind": "act", "idempotent": True, "boundary": _live.BOUNDARY, "max_level": "L2"},
}


# --- read tools -----------------------------------------------------------------


def _squeue_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) > len(SQUEUE_FIELDS):
            # a job name may hold the separator: the name is the field after the third
            extra = len(parts) - len(SQUEUE_FIELDS)
            parts = parts[:2] + ["|".join(parts[2:3 + extra])] + parts[3 + extra:]
        row = dict(zip(SQUEUE_FIELDS, (p.strip() for p in parts)))
        rows.append(row)
    return rows


def archi_slurm_squeue(
    user: Optional[str] = None,
    partition: Optional[str] = None,
    state: Optional[str] = None,
    job_id: Optional[str] = None,
    max_results: int = 50,
    runner: Optional[Runner] = None,
) -> dict[str, Any]:
    """Live ``squeue``: jobs in the queue, optionally for one user, partition, state or job id."""
    limit = bounded(max_results)
    argv = ["squeue", "--noheader", f"--format={SQUEUE_FORMAT}"]
    try:
        if user:
            argv += ["--user", name_arg(user, "user")]
        if partition:
            argv += ["--partition", name_arg(partition, "partition")]
        if state:
            argv += ["--states", name_arg(state.upper(), "state")]
        if job_id:
            argv += ["--jobs", job_id_arg(job_id)]
    except ToolError as exc:
        return _live.error_payload(argv, str(exc))
    return run_read(argv, runner, _squeue_rows, limit)


def _sinfo_rows(text: str) -> list[dict[str, Any]]:
    return [split_fields(line, SINFO_FIELDS) for line in text.splitlines() if line.strip()]


def archi_slurm_sinfo(
    partition: Optional[str] = None,
    node: Optional[str] = None,
    state: Optional[str] = None,
    max_results: int = 50,
    runner: Optional[Runner] = None,
) -> dict[str, Any]:
    """Live ``sinfo``: nodes with their state and, for a drained node, the reason."""
    limit = bounded(max_results)
    argv = ["sinfo", "--noheader", "--Node", f"--format={SINFO_FORMAT}"]
    try:
        if partition:
            argv += ["--partition", name_arg(partition, "partition")]
        if node:
            argv += ["--nodes", nodelist_arg(node)]
        if state:
            argv += ["--states", name_arg(state.lower(), "state")]
    except ToolError as exc:
        return _live.error_payload(argv, str(exc))
    return run_read(argv, runner, _sinfo_rows, limit)


# --- act tools --------------------------------------------------------------------
#
# An act returns {"status", "result", ...}. `status` is one of
#   succeeded  the tool changed the scheduler as asked
#   no_change  the scheduler was already in the requested state, or the tool left it alone on purpose;
#              nothing was changed and `result` says why
#   failed     nothing changed (a refused read, a command that exited non-zero)
#   unknown    a write command timed out or could not be started; it may have applied, check the node or job


def _node_state(node: str, runner: Runner) -> Optional[dict[str, str]]:
    """The node as ``sinfo`` shows it now, or None when Slurm does not know it."""
    result = runner(["sinfo", "--noheader", "--Node", f"--format={SINFO_FORMAT}", "--nodes", node],
                    timeout=_live.DEFAULT_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"sinfo exited {result.returncode}: {result.stderr.strip()[:200]}")
    rows = _sinfo_rows(result.stdout)
    return rows[0] if rows else None


def _out_of_service(state: str) -> bool:
    return any(word in state.lower() for word in _OUT_OF_SERVICE)


def _is_drained(state: str) -> bool:
    return "drain" in state.lower() or "drng" in state.lower()


def _prepare(
    payload: Any, context: Optional[Mapping[str, Any]], verify: Optional[Verifier], tool: str
) -> dict[str, Any]:
    # first: a refused call validates and runs nothing; the approval is for this payload exactly
    require_approval(context, verify, payload=payload if isinstance(payload, Mapping) else None, action_class=tool)
    data = validate_payload(payload, PAYLOAD_SCHEMAS[tool])
    # A schema `pattern` is a search and `$` also matches before a final newline, so the names are
    # checked again with fullmatch, and a reason is one line.
    if "node" in data:
        name_arg(data["node"], "node")
    if "job_id" in data:
        job_id_arg(data["job_id"])
    if any(ord(ch) < 32 or ch == "\x7f" for ch in data.get("reason", "")):
        raise ToolError("reason may not hold control characters")
    return data


def _read_failed(what: str, exc: Exception) -> dict[str, Any]:
    return outcome("failed", f"could not read {what} ({type(exc).__name__}: {exc}); nothing changed")


def _write(argv: list[str], runner: Runner, what: str) -> Optional[dict[str, Any]]:
    """Run one `scontrol` write. None on success; else the outcome to return.

    A command that timed out or could not be started may still have applied, so that is
    `unknown`, never `failed`; one that exited non-zero changed nothing.
    """
    try:
        result = runner(argv, timeout=_live.DEFAULT_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return outcome("unknown", f"{type(exc).__name__} running scontrol: may have applied; check {what}")
    if result.returncode != 0:
        return outcome("failed", f"scontrol exited {result.returncode}: {result.stderr.strip()[:200]}")
    return None


def drain_node(
    *,
    payload: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
    runner: Optional[Runner] = None,
    verify: Optional[Verifier] = None,
) -> dict[str, Any]:
    """Drain one node: running jobs finish, no new job starts. Reversible with ``resume_node``."""
    data = _prepare(payload, context, verify, "drain_node")
    run = runner or subprocess_runner
    key = context["idempotency_key"]
    node = data["node"]
    try:
        state = _node_state(node, run)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        return _read_failed(f"node {node}", exc)
    if state is None:
        return outcome("failed", f"node {node} is not known to Slurm; nothing changed")
    mark = f"{MARK}[{key}]"
    if _is_drained(state["state"]):
        if mark in state["reason"]:
            return outcome("no_change", f"node {node} is already drained by this call ({state['state']})")
        return outcome(
            "no_change",
            f"node {node} is already {state['state']} with another reason ({state['reason'] or 'none'}); left as is",
        )
    failed = _write(
        ["scontrol", "update", f"NodeName={node}", "State=DRAIN", f"Reason={mark} {data['reason']}"], run, f"node {node}"
    )
    return failed or outcome("succeeded", f"node {node} set to DRAIN ({mark})")


def resume_node(
    *,
    payload: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
    runner: Optional[Runner] = None,
    verify: Optional[Verifier] = None,
) -> dict[str, Any]:
    """Return a node that archi drained to service.

    A node drained or downed by anyone else is left alone unless the payload says ``force: true``;
    the flag is part of the payload schema, so it is in front of the person who approves.
    """
    data = _prepare(payload, context, verify, "resume_node")
    run = runner or subprocess_runner
    node = data["node"]
    try:
        state = _node_state(node, run)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        return _read_failed(f"node {node}", exc)
    if state is None:
        return outcome("failed", f"node {node} is not known to Slurm; nothing changed")
    if not _out_of_service(state["state"]):
        return outcome("no_change", f"node {node} is already in service ({state['state']}); nothing to resume")
    if not data.get("force") and f"{MARK}[" not in state["reason"]:
        return outcome(
            "no_change",
            f"node {node} is {state['state']} ({state['reason'] or 'no reason'}) and archi did not drain it; "
            "left as is (set force: true to resume it anyway)",
        )
    failed = _write(["scontrol", "update", f"NodeName={node}", "State=RESUME"], run, f"node {node}")
    return failed or outcome("succeeded", f"node {node} set to RESUME (was {state['state']})")


def _job(job_id: str, runner: Runner) -> Optional[dict[str, str]]:
    """The job as ``squeue`` shows it now (state and comment), or None when Slurm does not list it."""
    result = runner(["squeue", "--noheader", f"--format={JOB_FORMAT}", "--jobs", job_id], timeout=_live.DEFAULT_TIMEOUT)
    if result.returncode != 0:
        return None  # squeue exits non-zero for a job id it no longer knows
    rows = [split_fields(line, ("job_id", "state", "comment")) for line in result.stdout.splitlines() if line.strip()]
    return rows[0] if rows else None


def requeue_job(
    *,
    payload: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
    runner: Optional[Runner] = None,
    verify: Optional[Verifier] = None,
) -> dict[str, Any]:
    """Requeue one running job so it starts again from the queue, once per idempotency key.

    The job's own state decides, and the key is recorded only after the requeue is confirmed:

    1. read the job; a comment already holding this call's mark means it was requeued, so nothing is done
       (even when the job is running again by then);
    2. `scontrol requeue`; a failure or a timeout leaves no mark, so a retry reads the job again and acts on
       what it finds (waiting: nothing to do; still running: requeue);
    3. read the job again; only when it is waiting is the mark appended to the job's comment, after what the
       user wrote there, within Slurm's comment length. A job still running after a requeue that exited 0
       is `unknown`.

    A timeout leaves one case a mark cannot settle: if the requeue applied and the job started again before
    the retry, the retry sees it running and requeues once more. The result of a timeout says to check the job.
    """
    data = _prepare(payload, context, verify, "requeue_job")
    run = runner or subprocess_runner
    key = context["idempotency_key"]
    job_id = data["job_id"]
    mark = f"{MARK}[{key}]"
    try:
        job = _job(job_id, run)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _read_failed(f"job {job_id}", exc)
    if job is None:
        return outcome("failed", f"job {job_id} is not in the queue; nothing changed")
    if mark in job["comment"]:
        return outcome("no_change", f"job {job_id} was already requeued by this call ({job['state']})")
    state = job["state"].upper()
    if state in _WAITING:
        return outcome("no_change", f"job {job_id} is already waiting in the queue ({state}); nothing to requeue")
    if state != "RUNNING":
        return outcome("failed", f"job {job_id} is {state}, not running; nothing changed")
    failed = _write(["scontrol", "requeue", job_id], run, f"job {job_id}")
    if failed:
        return failed  # no mark was written: a retry reads the job again
    try:
        after = _job(job_id, run)
    except (OSError, subprocess.TimeoutExpired):
        after = None
    if after is None or after["state"].upper() not in _WAITING:
        seen = "not listed" if after is None else after["state"]
        return outcome("unknown", f"scontrol requeue exited 0 but job {job_id} is {seen}; check the job")
    comment = _with_mark(after["comment"], mark)
    marked = _write(["scontrol", "update", f"JobId={job_id}", f"Comment={comment}"], run, f"job {job_id}")
    note = "" if marked is None else "; the comment mark could not be written, so a repeat checks the job's state instead"
    return outcome("succeeded", f"job {job_id} requeued ({after['state']}){note}")


#: Slurm job states that mean a job is waiting for a node again
_WAITING = ("PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED")
#: a job comment is kept within this many characters
COMMENT_LIMIT = 255


def _with_mark(comment: str, mark: str) -> str:
    """The user's comment, then the mark; the user's text is cut (never the mark) to stay within the limit."""
    existing = "" if comment in ("", "(null)") else comment
    room = COMMENT_LIMIT - len(mark) - 1
    return f"{existing[:max(room, 0)]} {mark}".strip()
