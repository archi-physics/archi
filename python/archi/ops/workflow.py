"""The ops loop as an OKG deployment workflow (ADR 0002 section 7.1, PACT r3-ops-runtime).

This is the only module that imports `okg.deployment`. The worker loads it
through the deployment's `runtime.workflows: [archi.ops.workflow]` (the
renderer writes that line before `okg install` whenever an automation pack is
selected; editing it afterwards makes publish refuse until `okg catalog load
--apply`).

    archi_ops_scan   a short @deployment_scheduled scanner: adds the signals of
                     schedule triggers (a cron time) and of ticket_queue
                     triggers (a new or updated ticket, archi.triage.signals),
                     then starts one run per signal with DBOS.start_workflow.
                     Never on the okg-<deployment> queue (an admission submit
                     from a run queued there deadlocks behind it). The workflow
                     id is make_run_id(automation, signal id, attempt).
    archi_ops_run    one run: `archi.ops.loop.run_once` with real dependencies.
                     Every external effect is its own @deployment_step.
                     Anything that touches the clock, the graph, the budget,
                     the records or the mediator is a step, so a recovered run
                     replays its cached answers and repeats nothing. The
                     escalation wait (request_escalation) runs in the workflow
                     body, as DBOS requires.

Proven on a real okg 0.2.0 worker (spike-results.md, 2026-10-25):
- `bind_sdk_context` with our own autocommit connection makes `cost_budget` and
  `request_escalation` work in a plain workflow.
- A late approval is accepted after the worker restarts, so the mediator step
  re-checks halt and the escalation itself (`may_execute`) before it acts.
- Cancel stops a run only at a step boundary and does not wake a recv; halt
  closes each open escalation first (archi.cli.ops).
- Record steps carry the idempotency key "<workflow id>|record-<n>".

A mediator runs at most once per idempotency key. The mediator step first looks for an earlier
attempt (`archi_ops.mediations`), then re-checks halt and the escalation, then records the attempt,
then calls the tool with the key. A replay that finds an attempt with no recorded end does not call
again: the action is recorded with status `unknown` ("possible duplicate prevented; check the
external system"). A live tool used at L2 must also declare `idempotent_tools:` in its pack and honour
the key itself, because the step cannot tell whether a tool that crashed mid-call finished.

Everything here that reaches into okg beyond the public `okg.deployment` surface, with the upstream
ask (docs/ops-runtime-akmon-asks.md) that removes it. `archi.ops.okg_io` lists its own.

| What | Where | Ask |
|---|---|---|
| `okg.deployment._internal.context.bind_sdk_context` | `_private_okg` | O9: a public way to bind a workflow's deployment context |
| `okg.deployment._internal.escalation.EscalationTimeoutError` | `_private_okg` | O9 |
| `okg.deployment._internal.cost_budget.BudgetExhaustedError` | `_private_okg` | O9 |
| `okg.substrate.runtime.escalation.EscalationStateError` (a replay of an applied escalation raises it) | `_private_okg` | O14: `request_escalation` is not replay-safe after the row is applied |
| the escalation id rule `"esc-" + sha1("<workflow id>|_create_escalation")[:24]` | `expected_escalation_id` | O3: `request_escalation` should return the id before it waits |
| raw SQL on `okg.escalation_live` (through `OpsIO`) | `_decision_step`, `run_mediation` | O3 |
| `okg.deployment_cost_caps` (cap lookup before a charge) | `_cap_of` | O4 |
| `dbos.DBOS.retrieve_workflow` and `OpsIO.schedules` (the halt check) | `halt_reason` | O5: a first-class kill switch |
| `DBOS.recv` on `archi_ops_test_gate` (test only) | `_Mediator.execute` | none: a test hook, off unless ops.yaml sets `test_gate_seconds` |
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.parse
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from dbos import DBOS, SetWorkflowID

from okg.deployment import (
    cost_budget,
    deployment_scheduled,
    deployment_step,
    deployment_workflow,
    request_escalation,
)

from archi.ops.context_gate import (
    GateCheck,
    GateFacts,
    GateResult,
    GateSpec,
    context_eval_reading,
    evaluate_gate,
)
from archi.ops.escalation import (
    DECISION_SCHEMA,
    Decision,
    Escalation,
    decision_from_row,
    may_execute,
    subject_for,
)
from archi.ops.loop import (
    DEMOTED_PREFIX,
    BudgetCheck,
    BudgetExceeded,
    Demotion,
    Deps,
    MediationRefused,
    MediatorResult,
    OutcomeResult,
    PromotionState,
    ReadResult,
    Signal,
    make_run_id,
    run_once,
)
from archi.ops.okg_io import OpsIO
from archi.ops.proposers import Proposed, ProposerError, ProposeRequest, build_proposer
from archi.ops.record import OpsRecords, RecordError, submit
from archi.ops.runtime_config import RUN_WORKFLOW, SCANNER, RuntimeConfig, load_runtime
from archi.ops.schedule import latest_fire, tick_id
from archi.ops.steps import local_step
from archi.ops.stubs import marker_outcome, resolve_mediator
from archi.triage.context import read_ticket_context
from archi.triage.signals import add_ticket_signals, packs_of

log = logging.getLogger(__name__)
DECISION_SCHEMA_REF = "schema:literal:" + urllib.parse.quote(json.dumps(DECISION_SCHEMA))
#: The DBOS topic a test sends to, to release a run held by `test_gate_seconds`.
TEST_GATE_TOPIC = "archi_ops_test_gate"


# --- the private okg imports, in one place -------------------------------------------


def _private_okg():
    """okg 0.2.0 has no public way to bind the SDK context outside the SDK's own
    publish cycle, nor to name its escalation timeout or budget errors.

    UPSTREAM ASK O9 (docs/ops-runtime-akmon-asks.md): a public workflow-context binding and public
    error types. Until then archi pins okg 0.2.0 and imports these names from
    `okg.deployment._internal`, here only.
    """
    from okg.deployment._internal.context import bind_sdk_context
    from okg.deployment._internal.cost_budget import BudgetExhaustedError
    from okg.deployment._internal.escalation import EscalationTimeoutError
    from okg.substrate.runtime.escalation import EscalationStateError

    return bind_sdk_context, EscalationTimeoutError, BudgetExhaustedError, EscalationStateError


def _bind(conn, deployment: str):
    return _private_okg()[0](deployment=deployment, conn=conn)


# --- helpers shared by the steps -------------------------------------------------------


def _runtime() -> RuntimeConfig:
    """Re-read on every call: the instance's ops.yaml and archi.yaml are the source of truth."""
    return load_runtime()


def _io(rt: RuntimeConfig) -> OpsIO:
    return OpsIO(rt.dsn, rt.deployment)


def is_applied_replay(error: Exception, row: Mapping[str, Any]) -> bool:
    """Is this EscalationStateError the applied -> applied move of a replay? Only then it is harmless:
    the row is `applied` now and the error says it was asked to become applied. Any other state error
    (a decided row that cannot be consumed, a timed-out one, ...) takes the loop's error path."""
    status = (row.get("attrs") or {}).get("status")
    return status == "applied" and "applied" in str(error)


def expected_escalation_id(workflow_id: str) -> str:
    """okg names the escalation of a workflow body "esc-" + sha1("<workflow id>|_create_escalation")[:24]
    (okg 0.2.0, substrate/runtime/escalation.py). If okg changes that, the row is not found
    and the run fails closed (the decision is an error, nothing executes)."""
    return "esc-" + hashlib.sha1(f"{workflow_id}|_create_escalation".encode()).hexdigest()[:24]


def halt_reason(io: OpsIO) -> Optional[str]:
    """Why this run must not act, or None. Halted means the scanner schedule is paused
    (`archi ops halt`) or this run's workflow was cancelled. Raises if it cannot tell."""
    schedules = io.schedules()
    if not schedules:
        raise RuntimeError(f"schedule {SCANNER!r} not found; cannot tell whether the instance is halted")
    if any(schedules.values()):
        return f"the {SCANNER} schedule is paused (archi ops halt)"
    status = DBOS.retrieve_workflow(DBOS.workflow_id).get_status().status
    if str(status).upper() == "CANCELLED":
        return "this run was cancelled"
    return None


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text)


# --- steps ---------------------------------------------------------------------------


@deployment_step(name="archi_ops_now")
def _now_step() -> str:
    return datetime.now(timezone.utc).isoformat()


@deployment_step(name="archi_ops_halted")
def _halted_step() -> Optional[str]:
    return halt_reason(_io(_runtime()))


@deployment_step(name="archi_ops_gate")
def _gate_step(automation: str) -> dict:
    """Source ages from okg's watermarks (`okg doctor --check sources`) against the gate."""
    from archi.bundles.duration import parse_duration
    from archi.bundles.lock import read_lock
    from archi.stack.freshness import parse_timestamp

    rt = _runtime()
    spec: GateSpec = rt.automation(automation).gate
    rows = {r.get("name"): r for r in _io(rt).doctor_sources()} if spec.sources else {}
    published = None
    if spec.published:
        try:
            _io(rt).pin_generation()
            published = True
        except RecordError:
            published = False
    try:
        slos = read_lock(rt.instance_dir).get("composed", {}).get("freshness") or {}
    except Exception:  # no lock: sources are held to the gate's own max_staleness, or fail
        slos = {}
    now = datetime.now(timezone.utc)
    freshness = []
    for source in spec.sources:
        row = rows.get(source)
        if row is None:
            continue  # evaluate_gate fails a source that has no row
        stamp = parse_timestamp(((row.get("coverage") or {}).get("watermark") or {}).get("last_reconcile_at"))
        slo = (slos.get(source) or {}).get("max_staleness")
        freshness.append({
            "source": source,
            "age_seconds": max((now - stamp).total_seconds(), 0.0) if stamp else None,
            "max_staleness_seconds": parse_duration(slo) if slo else None,
            "reason": "" if stamp else "no completed run recorded in okg's watermarks",
        })
    # The newest `archi eval --context` run of the gate's bundle. Never raises: an unreadable, foreign,
    # failed or too old run is a failing `context_eval` check that names the reason.
    score, problem = None, None
    if spec.min_context_eval_score is not None:
        reading = context_eval_reading(spec, rt.instance_dir)
        score, problem = reading.score, reading.problem
    result = evaluate_gate(spec, GateFacts(None, tuple(freshness), {}, score, published, problem))
    return result.as_dict()


@deployment_step(name="archi_ops_pin")
def _pin_step() -> str:
    return _io(_runtime()).pin_generation()


@deployment_step(name="archi_ops_read")
def _read_step(entity: str, generation: str) -> dict:
    """Precedent incidents on the same entity, read at the pinned generation only."""
    rows = _io(_runtime()).read_nodes("ops_incident", generation, {"entity": entity})
    return {
        "rows": rows,
        "context": {"precedents": [{"node_id": r["node_id"], "title": (r.get("attrs") or {}).get("title"),
                                    "status": (r.get("attrs") or {}).get("status")} for r in rows]},
        "node_ids": sorted(r["node_id"] for r in rows),
    }


@deployment_step(name="archi_ops_ticket_read")
def _ticket_read_step(automation: str, entity: str, generation: str) -> dict:
    """A ticket_queue automation's read (archi.triage.context): the ticket, related tickets, documentation and
    precedents, at the pinned generation only. The proposer block of ops.yaml sets how much it reads."""
    rt = _runtime()
    return read_ticket_context(_io(rt), entity, generation, rt.settings[automation].proposer,
                               packs=packs_of(rt.automation(automation)))


@deployment_step(name="archi_ops_propose")
def _propose_step(automation: str, request: dict) -> dict:
    rt = _runtime()
    cfg = rt.settings[automation].proposer
    try:
        proposer = build_proposer(cfg, rt)
    except ProposerError as exc:
        raise RuntimeError(f"automation {automation!r}: {exc}") from exc
    proposed = proposer.propose(ProposeRequest(**request))
    return {"proposal": dict(proposed.proposal), "cost": proposed.cost}


@deployment_step(name="archi_ops_budget_check")
def _budget_check_step(scope: str, estimate: float, cap: float) -> dict:
    rt = _runtime()
    with _io(rt).connect() as conn, _bind(conn, rt.deployment):
        with cost_budget(usd=cap, scope=scope) as budget:
            consumed = budget.consumed
    return {"ok": consumed + estimate <= cap + 1e-9, "detail": f"{consumed:g} of {cap:g} spent"}


@deployment_step(name="archi_ops_budget_record")
def _budget_record_step(scope: str, cost: float, consumption_id: str) -> bool:
    """Charge `cost` with the cap enforced. Returns False when the charge would pass the cap
    (okg then writes nothing to the ledger)."""
    rt = _runtime()
    exhausted_error = _private_okg()[2]
    with _io(rt).connect() as conn, _bind(conn, rt.deployment):
        with cost_budget(usd=_cap_of(conn, scope), scope=scope) as budget:
            try:
                budget.consume(cost, consumption_id=consumption_id)
            except exhausted_error:
                return False
    return True


def _cap_of(conn, scope: str) -> float:
    row = conn.execute("SELECT cap_usd FROM okg.deployment_cost_caps WHERE scope = %s", (scope,)).fetchone()
    if row is None:
        raise RuntimeError(f"cost_budget scope {scope!r} has no cap; check() runs before record()")
    return float(row[0])


@deployment_step(name="archi_ops_record")
def _record_step(payload: dict, key: str) -> str:
    rt = _runtime()
    records = OpsRecords(nodes={n["node_id"]: n for n in payload["nodes"]}, edges=list(payload["edges"]))
    reply = submit(records, deployment=rt.deployment, dsn=rt.dsn, idempotency_key=key)
    return str((reply.get("output") or reply).get("status"))


DUPLICATE_PREVENTED = "possible duplicate prevented; check the external system"


def log_refusal(verdict: str, reason: str) -> None:
    """Say, in the worker log, that the mediator step itself refused to act."""
    message = f"archi_ops_mediate refused ({verdict}): {reason}"
    log.warning(message)
    print(message, file=sys.stderr, flush=True)


def run_mediation(
    io: Any,
    *,
    mediator_entry: Callable[[], str],
    action_class: str,
    payload: Mapping[str, Any],
    proposal_id: str,
    idempotency_key: str,
    esc: Optional[Mapping[str, Any]],
    workflow_id: str,
    dsn: str,
    halt: Callable[[Any], Optional[str]] = halt_reason,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    resolve: Callable[[str], Callable[..., Mapping[str, Any]]] = resolve_mediator,
    refused: Callable[[str, str], None] = log_refusal,
    instance_dir: Optional[str] = None,
    at_l1: bool = False,
) -> dict:
    """The body of the mediator step: the only place an action happens.

    Order: (1) an earlier attempt for this idempotency key decides everything (a replay never
    asks again); (2) halt; (3) the escalation row, read again, must pass `may_execute` (a workflow step
    with an implementation has no escalation only at L1, where it has no external effect; at L2 the
    escalation is re-read for a step too); (4) the
    attempt is recorded durably; (5) the mediator is called with the key. Any exception in
    1 to 3 is a refusal with verdict `error`, never a failed action. Returns `{"refused":
    [verdict, reason]}` or `{"status", "result"}` (`unknown` when an earlier attempt has no recorded end).
    """
    try:
        earlier = io.mediation(idempotency_key)
        if earlier is not None:
            return _earlier_attempt(earlier)
        reason = halt(io)
        if reason:
            refused("halted", reason)
            return {"refused": ["halted", reason]}
        entry = mediator_entry()
        if not (at_l1 and local_step(entry) is not None):
            if esc is None:
                refused("error", "no escalation was opened for this proposal")
                return {"refused": ["error", "no escalation was opened for this proposal"]}
            escalation = Escalation(esc["escalation_id"], esc["proposal_id"], _moment(esc["opened_at"]),
                                    _moment(esc["timeout_at"]))
            allowed = may_execute(io.decision_for(escalation.escalation_id), escalation, now())
            if not allowed:
                verdict = "timed_out" if allowed.late else "error"
                refused(verdict, allowed.reason)
                return {"refused": [verdict, allowed.reason]}
    except Exception as exc:
        reason = f"the mediator step could not verify it may act ({type(exc).__name__}: {exc})"
        refused("error", reason)
        return {"refused": ["error", reason]}
    earlier = io.begin_mediation(idempotency_key, proposal_id)
    if earlier is not None:  # another attempt got in between
        return _earlier_attempt(earlier)
    try:
        context = {"dsn": dsn, "proposal_id": proposal_id, "idempotency_key": idempotency_key,
                   "workflow_id": workflow_id}
        if instance_dir is not None:
            context["instance_dir"] = instance_dir
        out = resolve(entry)(payload=dict(payload), context=context)
        status, result = out["status"], str(out.get("result", ""))
    except Exception as exc:
        status, result = "failed", f"{type(exc).__name__}: {exc}"
    io.finish_mediation(idempotency_key, status, result)
    return {"status": status, "result": result}


def _earlier_attempt(earlier: Mapping[str, Any]) -> dict:
    if earlier["status"] == "attempted":  # a call began; whether it ended is not recorded
        return {"status": "unknown", "result": DUPLICATE_PREVENTED}
    return {"status": earlier["status"], "result": earlier["result"] or ""}


@deployment_step(name="archi_ops_mediate")
def _mediate_step(automation: str, action_class: str, payload: dict, proposal_id: str,
                  idempotency_key: str, esc: Optional[dict], level: str = "L2") -> dict:
    """The mediator step. The loop has already decided; this re-checks everything (see run_mediation)."""
    rt = _runtime()
    return run_mediation(
        _io(rt), mediator_entry=lambda: rt.automation(automation).action_classes[action_class].mediator,
        action_class=action_class, payload=payload, proposal_id=proposal_id, idempotency_key=idempotency_key,
        esc=esc, workflow_id=DBOS.workflow_id, dsn=rt.dsn, instance_dir=str(rt.instance_dir),
        at_l1=level == "L1")


@deployment_step(name="archi_ops_decision")
def _decision_step(escalation_id: str) -> dict:
    """The escalation row as okg holds it now (a step, so a replay sees the same answer)."""
    found = _io(_runtime()).escalation(escalation_id)
    return {"found": found is not None, "subject": found[0] if found else None, "attrs": found[1] if found else None}


@deployment_step(name="archi_ops_demotions")
def _demotion_step(automation: str, generation: str) -> Optional[dict]:
    """The latest demotion of the automation to shadow: the synchronous log (`archi_ops.demotions`,
    written when a run is demoted) and the published ops records at the pinned generation (history).
    The later of the two."""
    io = _io(_runtime())
    candidates = []
    logged = io.latest_demotion(automation)
    if logged:
        candidates.append(logged)
    for row in io.read_nodes("ops_proposal", generation, {"automation": automation, "mode": "shadow"}):
        attrs = row.get("attrs") or {}
        reason, at = attrs.get("mode_reason") or "", attrs.get("proposed_at")
        if reason.startswith(DEMOTED_PREFIX) and at:
            candidates.append({"at": at, "reason": reason})
    return max(candidates, key=lambda c: _moment(c["at"])) if candidates else None


@deployment_step(name="archi_ops_demotion_record")
def _demotion_record_step(automation: str, run_id: str, at: str, reason: str) -> None:
    _io(_runtime()).record_demotion(automation, run_id, _moment(at), reason)


@deployment_step(name="archi_ops_promotion")
def _promotion_step(automation: str) -> dict:
    """Is the automation's promotion approved? Read from okg's decision ledger, in a step."""
    rt = _runtime()
    change = rt.settings[automation].promotion
    return _io(rt).promotion_status(rt.instance_dir, change, automation)


@deployment_step(name="archi_ops_outcome")
def _outcome_step(automation: str, entity: str) -> dict:
    rt = _runtime()
    cfg = rt.settings[automation].outcome
    if cfg.get("kind") != "marker":
        return {"result": "unknown", "resolution": "no outcome check is configured"}
    result, resolution = marker_outcome(dsn=rt.dsn, entity=entity)
    return {"result": result, "resolution": resolution}


# --- the real dependencies -------------------------------------------------------------


class _Clock:
    def now(self) -> datetime:
        return _moment(_now_step())


class _Halt:
    def halted(self) -> bool:
        return _halted_step() is not None


class _Gate:
    def __init__(self, automation: str):
        self.automation = automation

    def check(self, spec: GateSpec) -> GateResult:
        d = _gate_step(self.automation)
        return GateResult(d["generation"], tuple(GateCheck(**c) for c in d["checks"]))


class _Reader:
    def __init__(self, automation: str = "", trigger: Optional[Mapping[str, Any]] = None):
        #: the full rows the pinned read returned, by node id, so the recorder can carry
        #: a cited precedent with the proposal (okg admits an edge only with both ends in the payload)
        self.rows: dict[str, dict] = {}
        self.automation = automation
        #: a ticket_queue automation reads the ticket and what is around it; any other reads precedents
        self.tickets = (trigger or {}).get("type") == "ticket_queue"

    def pin(self) -> str:
        return _pin_step()

    def read(self, signal: Signal, generation: str) -> ReadResult:
        if self.tickets:
            d = _ticket_read_step(self.automation, signal.entity, generation)
        else:
            d = _read_step(signal.entity, generation)
        self.rows = {r["node_id"]: r for r in d["rows"]}
        return ReadResult(d["context"], frozenset(d["node_ids"]))


class _Proposer:
    def __init__(self, automation: str):
        self.automation = automation

    def propose(self, request: ProposeRequest) -> Proposed:
        d = _propose_step(self.automation, {**asdict(request), "context": dict(request.context)})
        return Proposed(d["proposal"], d["cost"])


class _Budget:
    def check(self, scope: str, estimate: float, cap: float) -> BudgetCheck:
        d = _budget_check_step(scope, estimate, cap)
        return BudgetCheck(d["ok"], d["detail"])

    def record(self, scope: str, cost: float) -> None:
        if not _budget_record_step(scope, cost, f"{DBOS.workflow_id}|{scope}"):
            raise BudgetExceeded(scope)


class _Demotions:
    def latest(self, automation: str, generation: str) -> Optional[Demotion]:
        d = _demotion_step(automation, generation)
        return None if d is None else Demotion(_moment(d["at"]), d["reason"])

    def record(self, automation: str, run_id: str, at: datetime, reason: str) -> None:
        _demotion_record_step(automation, run_id, at.isoformat(), reason)


class _Promotions:
    def check(self, automation: str, change: Optional[str]) -> PromotionState:
        d = _promotion_step(automation)
        return PromotionState(bool(d["approved"]), _moment(d["at"]) if d.get("at") else None, d.get("reason") or "")


class _Recorder:
    """Collects records in memory; `flush` submits them all in one admission step.

    Each flush carries every record so far (a record linked to an earlier one must
    travel with it), with the key "<workflow id>|record-<n>"."""

    def __init__(self, reader: _Reader):
        self.records = OpsRecords()
        self.reader = reader
        self.flushes = 0
        self.last = ""

    def add(self, kind: str, *, evidence=(), **attrs: Any) -> str:
        evidence = list(evidence)
        for cited in evidence:  # a cited precedent travels with the proposal that cites it
            if str(cited).startswith("ops_incident:") and cited in self.reader.rows:
                self.records.carry(self.reader.rows[cited])
        return self.records.add(kind, evidence=evidence, **attrs)

    def flush(self) -> None:
        payload = self.records.payload()
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if digest == self.last or not payload["nodes"]:
            return
        self.flushes += 1
        status = _record_step(payload, f"{DBOS.workflow_id}|record-{self.flushes}")
        if status != "accepted":
            raise RuntimeError(f"okg admission did not accept the ops records: {status}")
        self.last = digest


#: starts the one line of an escalation reason that records the action (line 2; see `escalation_reason`)
ACTION_PREFIX = "Action: "


def _one_line(text: Any) -> str:
    """Free text from a proposal on one line: no newline can start a line of the escalation reason."""
    return " ".join(str(text).split())


def escalation_reason(automation: str, proposal: Any) -> str:
    """The reason an escalation is opened with, in a fixed layout the approver and the live tools both read.

    Line 1 is the proposer's free-text summary, line 2 is the one `Action: <class> <payload JSON>` line, then
    `Rollback:` and `Evidence:`. Every proposer-written piece is put on one line first, so a summary holding
    a newline and "Action: ..." cannot make a second Action line or move the real one. A live tool binds an
    approval to the payload by reading line 2, and refuses a reason with any other line that starts with
    `Action: ` (archi.tools._live.ops_verifier).
    """
    return (f"{automation}: {_one_line(proposal.summary)}\n"
            f"{ACTION_PREFIX}{_one_line(proposal.action_class)} {json.dumps(dict(proposal.payload), sort_keys=True)}\n"
            f"Rollback: {_one_line(proposal.rollback)}\n"
            f"Evidence: {_one_line(', '.join(proposal.evidence_ids))}")


class _Decider:
    """Opens the escalation through request_escalation and reads the decision back from okg."""

    def __init__(self, rt: RuntimeConfig):
        self.rt, self.reason = rt, ""

    def open(self, *, automation, proposal_id, proposal, timeout_at) -> Escalation:
        self.reason = escalation_reason(automation, proposal)
        return Escalation(expected_escalation_id(DBOS.workflow_id), proposal_id, _Clock().now(), timeout_at)

    def wait(self, escalation: Escalation) -> Decision:
        timeout_error, state_error = _private_okg()[1], _private_okg()[3]
        replay_error: Optional[Exception] = None
        remaining = max((escalation.timeout_at - _Clock().now()).total_seconds(), 1.0)
        try:
            request_escalation(subject=subject_for(escalation.proposal_id), reason=self.reason,
                               decision_schema=DECISION_SCHEMA_REF, timeout_seconds=remaining)
        except timeout_error:
            pass  # okg marked the row timed_out; the row below says so
        except state_error as exc:
            replay_error = exc
        row = _decision_step(escalation.escalation_id)  # a step: a replay sees the row it saw before
        if replay_error is not None and not is_applied_replay(replay_error, row):
            raise replay_error
        if replay_error is not None:
            # A replay of a run whose escalation okg already marked `applied`: request_escalation asks to move
            # applied -> applied and is refused (upstream ask O14). The recv already returned the recorded
            # decision, so the row is read exactly as in the first execution.
            log.info("request_escalation: escalation %s was already applied; reading its row", escalation.escalation_id)
        return decision_from_row(escalation.escalation_id, row["subject"], row["attrs"])


class _Mediator:
    def __init__(self, automation: str):
        self.automation = automation

    def execute(self, *, action_class, payload, proposal_id, idempotency_key, generation, escalation=None,
                level="L2"):
        esc = None if escalation is None else {
            "escalation_id": escalation.escalation_id, "proposal_id": escalation.proposal_id,
            "opened_at": escalation.opened_at.isoformat(), "timeout_at": escalation.timeout_at.isoformat()}
        gate_seconds = _runtime().settings[self.automation].test_gate_seconds
        if gate_seconds > 0:
            # TEST ONLY (ops.yaml test_gate_seconds): hold here, after the loop's decision and before the
            # mediator step, until a message arrives on the topic, so a test can halt the instance or
            # change the escalation in this window.
            print(f"archi_ops test gate: waiting up to {gate_seconds:g}s on {TEST_GATE_TOPIC!r}",
                  file=sys.stderr, flush=True)
            DBOS.recv(topic=TEST_GATE_TOPIC, timeout_seconds=gate_seconds)
        d = _mediate_step(self.automation, action_class, dict(payload), proposal_id, idempotency_key, esc, level)
        if "refused" in d:
            raise MediationRefused(*d["refused"])
        return MediatorResult(d["status"], d["result"])


class _Outcome:
    def __init__(self, automation: str):
        self.automation = automation

    def check(self, *, signal: Signal, action: MediatorResult, generation: str) -> OutcomeResult:
        d = _outcome_step(self.automation, signal.entity)
        return OutcomeResult(d["result"], d["resolution"])


def real_deps(rt: RuntimeConfig, automation: str) -> Deps:
    reader = _Reader(automation, rt.automation(automation).trigger)
    return Deps(gate=_Gate(automation), reader=reader, proposer=_Proposer(automation),
                decider=_Decider(rt), mediator=_Mediator(automation), outcome=_Outcome(automation),
                recorder=_Recorder(reader), halt=_Halt(), clock=_Clock(), budget=_Budget(),
                demotions=_Demotions(), promotions=_Promotions())


# --- the workflows ---------------------------------------------------------------------


@deployment_workflow(name=RUN_WORKFLOW)
def archi_ops_run(automation: str, signal: dict) -> dict:
    """One run of one automation for one signal. The workflow id is the run id."""
    rt = _runtime()
    config = rt.automation(automation)
    source = signal.get("source") or "signal"
    # a ticket signal's entity is <system>:<key>; the incident names the ticket, so a later ticket resolution
    # can become its outcome (archi.ops.record, `ticket_keys`)
    keys = (signal["entity"].partition(":")[2],) if source == "ticket" and ":" in signal["entity"] else ()
    sig = Signal(id=signal["id"], entity=signal["entity"], title=signal["title"], source=source,
                 summary=signal.get("summary"), ticket_keys=keys)
    with _io(rt).connect() as conn, _bind(conn, rt.deployment):
        result = run_once(config, sig, real_deps(rt, automation), run_id=DBOS.workflow_id)
    return {**asdict(result), "records": list(result.records)}


def start_run(run_id: str, automation: str, signal: dict) -> None:
    """Start one run as its own workflow. Never on the okg-<deployment> queue: an admission
    submit made by a run queued there waits behind that run and deadlocks (spike-results.md g)."""
    with SetWorkflowID(run_id):
        DBOS.start_workflow(archi_ops_run, automation, signal)


def add_schedule_ticks(rt: Any, io: Any, now: datetime, say: Callable[[str], None] = log.info) -> list[str]:
    """For each automation with a `schedule` trigger, add the signal of its latest cron time.

    The signal id is `tick-<time>`, so a scan that runs again within the same cron time adds nothing
    new, and a worker that was down for a day adds one tick, not a backlog. Returns the ids added or
    already present, as `<automation>/<id>`.
    """
    load = getattr(rt, "automation", None)
    if load is None:
        return []
    ticks = []
    for name in sorted(rt.settings):
        try:
            trigger = load(name).trigger
            if trigger.get("type") != "schedule":
                continue
            fire = latest_fire(trigger["cron"], now)
        except Exception as exc:  # a bad automation file or cron stops that automation's ticks, not the scan
            say(f"archi_ops_scan: {name}: cannot work out its schedule ({type(exc).__name__}: {exc})")
            continue
        if fire is None:
            continue
        io.add_signal(name, tick_id(fire), rt.deployment, f"scheduled run {fire:%Y-%m-%d %H:%MZ}",
                      source="schedule")
        ticks.append(f"{name}/{tick_id(fire)}")
    return ticks


def scan_once(rt: RuntimeConfig, io: Any, start: Callable[[str, str, dict], None] = start_run,
              say: Callable[[str], None] = log.info, now: Optional[datetime] = None) -> list[str]:
    """Start at most one run per automation, and only when that automation has none in flight.

    One at a time keeps an L2 automation's budget and escalations simple: a run waiting hours for a
    person holds the next signal in the table (status `new`) until it finishes. Nothing starts while
    the scanner schedule is paused (a manual trigger of a paused schedule still runs this function).
    """
    started: list[str] = []
    io.ensure_tables()  # idempotent: the worker does not depend on init or halt having made them
    if any(io.schedules().values()):
        say("archi_ops_scan: halted (schedule paused); starting nothing")
        return started
    add_schedule_ticks(rt, io, now or datetime.now(timezone.utc), say)
    add_ticket_signals(rt, io, say)
    in_flight = {run.automation for run in io.runs()}
    for name in sorted(rt.settings):
        if name in in_flight:
            say(f"archi_ops_scan: {name} has a run in progress; its next signal waits")
            continue
        for s in io.new_signals(name)[:1]:
            try:
                run_id = make_run_id(name, s["id"], s["attempt"])
            except ValueError as exc:
                io.mark_signal(name, s["id"], "invalid")
                say(f"archi_ops_scan: signal {s['id']!r} refused: {exc}")
                continue
            start(run_id, name, {k: s[k] for k in ("id", "entity", "title", "source", "summary")})
            io.mark_signal_started(name, s["id"])
            started.append(run_id)
            say(f"archi_ops_scan: started {run_id}")
    return started


@deployment_scheduled(SCANNER, default_cron="*/1 * * * *", automatic_backfill=False, guard_body=False)
def archi_ops_scan(scheduled_time, context):
    """Start the next run of each idle automation. Short: the runs are separate workflows (queue None)."""
    rt = _runtime()
    scan_once(rt, _io(rt))
