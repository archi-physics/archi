"""The operations loop (ADR 0002 section 7.1, PACT r3-ops-runtime req.loop).

One run takes a signal through: halt check, effective level, context gate,
pinned generation, budget, proposal, decision by level, mediator, outcome
check, and the ops records of each step. This module is plain Python with no
okg imports; everything it touches outside itself comes in through the small
Protocols in `Deps`, so tests can fake each one. The workflow module wraps
each Protocol call (and `_mediate`) in a durable step.

Levels come from `archi.ops.levels` (ADR 0002 D13: L3 and unknown levels are
refused). The effective level is the lowest of the instance's setting, the
bundle bound and the pack maximum. Each action class also caps the level it
may run at: a proposal whose class allows less than the effective level is
recorded as `action_class_capped` and never executes.

    L0  record the incident, the proposal and an `observed` decision.
    L1  as L0, with the draft on the proposal and a `drafted` decision.
        Nothing is sent. When the action class's mediator is a workflow step with an
        implementation (`archi.ops.steps`, e.g. `draft_issue`), the step runs: a step has no
        external effect, so it needs no escalation. A live tool never runs at L1.
    L2  the Decider opens an escalation and later reports a decision. Only an
        approval bound to that escalation and proposal, by a named person,
        made between the escalation opening and its deadline, reaches the
        mediator. The first statement of `_mediate` is the HaltCheck.

Shadow mode (a live automation whose context gate fails drops to shadow for
the run; a shadow automation stays there) never opens an escalation and never
mediates. It records what it would have done, with an `observed` decision.

Failing closed. An exception from the halt check, gate, pin, budget, proposer,
decider or a record write never executes anything. The run records what it
can: after the incident exists, an `error` decision; before that, a run-level
record (below).

Run-level records. A decision needs a proposal, and a proposal needs a pinned
generation, so a run that stops before reading anything has neither. When such
a run is refused (L3 or unknown level) or fails (the gate, the pin or the
halt check raised), it writes one `ops_incident` for the signal whose summary
says "run not started: <reason>" and carries no `generation_read`: that record
exists so the operator sees the signal and why nothing ran. A run that stops
because the system is halted writes nothing and returns status `halted`:
halt means stop, and the caller logs the structured result.

A run that stops after the incident (budget, a refused proposal, a proposer
error) records a placeholder proposal with `mode: placeholder` and a decision.
Shadow scoring reads only `mode: shadow`, so placeholders are skipped.

`AutomationConfig.from_block` builds the configuration from the rendered
automation block, `<instance>/automations/<name>.yaml`; the instance's own
level (archi.yaml) and the mode are passed alongside.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol

import yaml

from archi.ops.context_gate import GateResult, GateSpec
from archi.ops.escalation import (
    APPROVED,
    ERROR,
    REJECTED,
    TIMED_OUT,
    Decision,
    Escalation,
    may_execute,
)
from archi.ops.levels import LevelError, level_problem, rank, resolve_level
from archi.ops.proposers import Proposed, ProposeRequest, Proposer
from archi.ops.steps import get_step

MODES = ("shadow", "live")
#: Proposal `mode_reason` prefix of a live run whose promotion had no shadow trial (`promote --no-trial`).
NO_SHADOW_EVIDENCE_PREFIX = "no shadow evidence: "
#: Proposal `mode_reason` prefix when `mode: live` was not honoured: no approved promotion.
NOT_PROMOTED_PREFIX = "not promoted: "
#: Proposal `mode_reason` prefix when a live automation was dropped to shadow.
DEMOTED_PREFIX = "demoted to shadow: "
#: Proposal `mode` of a placeholder written when no real proposal exists.
PLACEHOLDER_MODE = "placeholder"
ACTOR_KIND = "automation"


# --- configuration -----------------------------------------------------------------


@dataclass(frozen=True)
class ActionClassConfig:
    #: the highest level the class may run at
    max_level: str
    mediator: str = ""
    #: JSON Schema the proposal's payload must satisfy
    payload_schema: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class AutomationConfig:
    name: str
    default_level: str
    pack_max_level: str
    instance_level: Optional[str] = None
    bundle_max_level: Optional[str] = None
    #: every automation starts in shadow
    mode: str = "shadow"
    gate: Optional[GateSpec] = None
    action_classes: Mapping[str, ActionClassConfig] = field(default_factory=dict)
    #: JSON Schema the raw proposal must satisfy
    proposal_schema: Mapping[str, Any] = field(default_factory=dict)
    cost_per_run: Optional[float] = None
    cost_per_day: Optional[float] = None
    #: what one proposal is expected to cost, checked against the caps first
    estimate_cost: float = 0.0
    #: how long an L2 escalation waits for a person
    timeout_seconds: int = 4 * 3600
    #: when an operator last approved the automation's promotion. The runtime fills it from the PACT
    #: ledger (`Deps.promotions`); a demotion recorded after it keeps the automation in shadow.
    promoted_at: Optional[datetime] = None
    #: the PACT change `archi ops promote` opened (ops.yaml `promotion`). `mode: live` counts only while
    #: this change is approved.
    promotion_change: Optional[str] = None
    #: the rendered trigger (`{type: schedule, cron: ...}`); the scanner makes a signal per cron time
    trigger: Mapping[str, Any] = field(default_factory=dict)
    #: the pack's opt-in `evidence: own_signals` (a monitor): a proposal may cite its own run's incident,
    #: and a step with an implementation runs at L1. False for every other automation.
    own_signals: bool = False

    @classmethod
    def from_block(
        cls,
        block: Mapping[str, Any],
        *,
        instance_level: Optional[str] = None,
        mode: str = "shadow",
        timeout_seconds: int = 4 * 3600,
        promoted_at: Optional[datetime] = None,
        promotion_change: Optional[str] = None,
        estimate_cost: Optional[float] = None,
    ) -> "AutomationConfig":
        """From a rendered automation block (`<instance>/automations/<name>.yaml`)."""
        defs = block.get("action_class_definitions") or {}
        classes = {
            name: ActionClassConfig(d.get("max_level"), d.get("mediator", ""), d.get("payload_schema"))
            for name, d in defs.items()
        }
        budget = block.get("budget") or {}
        gate = block.get("context_gate")
        return cls(
            name=block["pack"],
            default_level=block["default_level"],
            pack_max_level=block["pack_max_level"],
            instance_level=instance_level,
            bundle_max_level=block.get("bundle_max_level"),
            mode=mode,
            gate=GateSpec.from_mapping(gate) if gate else None,
            action_classes=classes,
            proposal_schema=dict(block.get("proposal_schema") or {}),
            cost_per_run=budget.get("cost_per_run"),
            cost_per_day=budget.get("cost_per_day"),
            # what one proposal is expected to cost: the pack's per-run budget unless the instance says otherwise
            estimate_cost=float(estimate_cost if estimate_cost is not None else (budget.get("cost_per_run") or 0.0)),
            timeout_seconds=timeout_seconds,
            promoted_at=promoted_at,
            promotion_change=promotion_change,
            trigger=dict(block.get("trigger") or {}),
            own_signals=block.get("evidence") == "own_signals",
        )


def load_automation(instance_dir: Path | str, name: str, **kwargs: Any) -> AutomationConfig:
    """Read `<instance>/automations/<name>.yaml` as the runtime configuration."""
    path = Path(instance_dir) / "automations" / f"{name}.yaml"
    return AutomationConfig.from_block(yaml.safe_load(path.read_text(encoding="utf-8")), **kwargs)


def resolved_level(automation: AutomationConfig) -> str:
    """The effective level ("L0".."L2"). Raises LevelError on L3 or an unknown level,
    including an action class that allows one."""
    effective = resolve_level(
        instance_level=automation.instance_level,
        default_level=automation.default_level,
        bundle_max=automation.bundle_max_level,
        pack_max=automation.pack_max_level,
    ).level
    for name, cls in automation.action_classes.items():
        message = level_problem(cls.max_level, f"action class {name!r} max_level")
        if message:
            raise LevelError(message)
    return effective


#: Joins the parts of a run id. Neither an automation name nor a signal id may contain it, so
#: two different (automation, signal, attempt) triples never make the same id.
RUN_ID_SEPARATOR = "~"


def make_run_id(automation: str, signal_id: str, attempt: int | str) -> str:
    """The run id for (automation, signal, attempt). Deterministic: a replay of the
    same workflow gets the same id, so its records overwrite rather than duplicate.

    Raises ValueError for an empty part, a part holding the separator `~`, or an attempt
    that is not a positive integer."""
    for label, part in (("automation", automation), ("signal id", signal_id)):
        if not isinstance(part, str) or not part or RUN_ID_SEPARATOR in part:
            raise ValueError(f"{label} {part!r} must be a non-empty string without {RUN_ID_SEPARATOR!r}")
    try:
        n = int(attempt)
    except (TypeError, ValueError):
        n = 0
    if n < 1 or str(n) != str(attempt):
        raise ValueError(f"attempt {attempt!r} must be a positive integer")
    return RUN_ID_SEPARATOR.join((automation, signal_id, str(n)))


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    """(automation, signal id, attempt) of a run id made by make_run_id."""
    parts = run_id.split(RUN_ID_SEPARATOR)
    if len(parts) != 3:
        raise ValueError(f"{run_id!r} is not an ops run id")
    return parts[0], parts[1], int(parts[2])


# --- the signal and the proposal ---------------------------------------------------


@dataclass(frozen=True)
class Signal:
    id: str
    entity: str
    title: str
    #: what raised it: "schedule", "ticket", "alert"
    source: str = "signal"
    summary: Optional[str] = None
    entity_kind: Optional[str] = None
    error_signature: Optional[str] = None
    severity: Optional[str] = None
    ticket_keys: tuple[str, ...] = ()
    detected_at: Optional[datetime] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "entity": self.entity, "title": self.title, "source": self.source,
            "summary": self.summary, "entity_kind": self.entity_kind,
            "error_signature": self.error_signature, "severity": self.severity,
            "ticket_keys": list(self.ticket_keys),
        }


class ProposalError(ValueError):
    """The proposal does not match the automation's proposal schema."""


@dataclass(frozen=True)
class Proposal:
    summary: str
    evidence_ids: tuple[str, ...]
    action_class: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    rollback: Optional[str] = None
    draft: Optional[str] = None


def _schema_errors(schema: Mapping[str, Any], instance: Any, where: str) -> list[str]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(dict(schema))
    return [
        f"{where}{'/' + '/'.join(map(str, e.absolute_path)) if e.absolute_path else ''}: {e.message}"
        for e in sorted(validator.iter_errors(instance), key=lambda e: list(map(str, e.absolute_path)))
    ]


def validate_proposal(
    raw: Any,
    automation: AutomationConfig,
    level: str,
    known_evidence: Optional[Iterable[str]] = None,
) -> Proposal:
    """Check a raw proposal against the automation's proposal schema.

    The rendered `proposal_schema` (JSON Schema) is the check; on top of it the
    loop requires at least one cited evidence id, an action class at L2 that the
    automation declares and whose payload schema the payload meets, and a draft
    or payload at L1. When `known_evidence` is given, every cited id must be in it
    (the ids the pinned read returned).
    """
    if not isinstance(raw, Mapping):
        raise ProposalError("proposal must be an object")
    errors = _schema_errors(automation.proposal_schema, raw, "proposal") if automation.proposal_schema else []
    evidence = raw.get("evidence_ids")
    if (not isinstance(evidence, (list, tuple)) or not evidence
            or not all(isinstance(e, str) and e.strip() for e in evidence)):
        if not any("evidence_ids" in e for e in errors):
            errors.append("evidence_ids must list at least one cited okg node id")
    elif known_evidence is not None:
        known = set(known_evidence)
        outside = [e for e in evidence if e not in known]
        if outside:
            errors.append(f"evidence_ids not in the pinned read: {outside}")
    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary is required")

    action_class = raw.get("action_class")
    payload = raw.get("payload", {})
    if not isinstance(payload, Mapping):
        errors.append("payload must be an object")
        payload = {}
    if action_class is not None and not isinstance(action_class, str):
        errors.append("action_class must be a string")
    elif action_class is None:
        if level == "L2":
            errors.append("action_class is required at L2")
    elif action_class not in automation.action_classes:
        errors.append(f"action_class {action_class!r} is not declared; known: {sorted(automation.action_classes)}")
    else:
        schema = automation.action_classes[action_class].payload_schema
        if schema:
            errors += _schema_errors(schema, payload, f"payload for {action_class!r}")
    draft = raw.get("draft")
    if draft is not None and not isinstance(draft, str):
        errors.append("draft must be a string")
    if level == "L1" and not (isinstance(draft, str) and draft.strip()) and not payload:
        errors.append("a draft (draft, or a payload) is required at L1")
    if errors:
        raise ProposalError("; ".join(errors))
    confidence = raw.get("confidence")
    if draft is None and level == "L1" and isinstance(payload.get("body"), str):
        draft = payload["body"]  # a draft_reply's body is the draft (proposal schemas carry no `draft` key)
    return Proposal(
        summary=summary,
        evidence_ids=tuple(dict.fromkeys(evidence)),
        action_class=action_class,
        payload=dict(payload),
        confidence=None if confidence is None else float(confidence),
        rollback=raw.get("rollback"),
        draft=draft,
    )


# --- dependencies ------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetCheck:
    ok: bool
    detail: str = ""


class BudgetExceeded(Exception):
    """`BudgetGuard.record` refused the charge: it would take the scope past its cap."""


@dataclass(frozen=True)
class Demotion:
    """The latest recorded demotion of an automation to shadow."""

    at: datetime
    #: the `mode_reason` the demoted run's proposal carried
    reason: str


class DemotionLog(Protocol):
    def latest(self, automation: str, generation: str) -> Optional[Demotion]:
        """The most recent demotion of `automation`: the synchronous log first, then the ops records
        published at `generation` (history). A demotion never waits for a publish to take effect."""

    def record(self, automation: str, run_id: str, at: datetime, reason: str) -> None:
        """Write a demotion to the synchronous log, once per run id."""


@dataclass(frozen=True)
class PromotionState:
    """Whether an operator approved the automation's promotion (an approved PACT change)."""

    approved: bool
    #: when it was approved
    at: Optional[datetime] = None
    #: why not, when it is not approved
    reason: str = ""
    #: the decision that waived the shadow trial (`archi ops promote --no-trial`), when there was one
    no_trial: str = ""


class PromotionCheck(Protocol):
    def check(self, automation: str, change: Optional[str]) -> PromotionState: ...


class BudgetGuard(Protocol):
    def check(self, scope: str, estimate: float, cap: float) -> BudgetCheck:
        """Would spending `estimate` more in `scope` stay within `cap`?"""

    def record(self, scope: str, cost: float) -> None:
        """Charge `cost` to `scope`. Raises BudgetExceeded if the cap would be passed (nothing is charged)."""


class GateChecker(Protocol):
    def check(self, spec: GateSpec) -> GateResult: ...


@dataclass(frozen=True)
class ReadResult:
    context: Mapping[str, Any]
    #: the okg node ids the pinned read returned; a proposal may cite only these
    node_ids: frozenset[str]


class Reader(Protocol):
    def pin(self) -> str:
        """The generation this run reads. Called once."""

    def read(self, signal: "Signal", generation: str) -> ReadResult:
        """The context for the signal, read at `generation` only."""


class Decider(Protocol):
    def open(self, *, automation: str, proposal_id: str, proposal: Proposal, timeout_at: datetime) -> Escalation:
        """Open an escalation for the proposal."""

    def wait(self, escalation: Escalation) -> Decision:
        """Wait for it. Return approved, rejected, timed_out or error."""


@dataclass(frozen=True)
class MediatorResult:
    status: str  # succeeded | failed
    result: str = ""


class MediationRefused(Exception):
    """The mediator step refused to run (it re-checked halt or the escalation).

    `verdict` is the decision recorded in place of the approval: `halted`,
    `timed_out` or `error`.
    """

    def __init__(self, verdict: str, reason: str):
        super().__init__(reason)
        self.verdict, self.reason = verdict, reason


class Mediator(Protocol):
    def execute(
        self, *, action_class: str, payload: Mapping[str, Any], proposal_id: str, idempotency_key: str,
        generation: str, escalation: Optional[Escalation] = None, level: str = "L2",
    ) -> MediatorResult:
        """Run the action. May raise MediationRefused after re-checking halt and the
        escalation (the durable step does both before it acts)."""


@dataclass(frozen=True)
class OutcomeResult:
    result: str  # resolved | not_resolved | partial | unknown
    resolution: str = ""


class OutcomeChecker(Protocol):
    def check(self, *, signal: Signal, action: MediatorResult, generation: str) -> OutcomeResult: ...


class Recorder(Protocol):
    def add(self, kind: str, *, evidence: Iterable[Any] = (), **attrs: Any) -> str:
        """Add one ops record in `archi.ops.record.OpsRecords.add` format; return its node id."""

    def flush(self) -> None:
        """Make every record added so far durable (the loop calls it at each point where
        a person or a later run must be able to see the state)."""


class HaltCheck(Protocol):
    def halted(self) -> bool: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


@dataclass(frozen=True)
class Deps:
    gate: GateChecker
    reader: Reader
    proposer: Proposer
    decider: Decider
    mediator: Mediator
    outcome: OutcomeChecker
    recorder: Recorder
    halt: HaltCheck
    clock: Clock
    budget: BudgetGuard
    #: where demotions are read from; None means demotions are not persisted
    demotions: Optional[DemotionLog] = None
    #: where promotions are checked; when given, `mode: live` counts only for an approved promotion
    promotions: Optional[PromotionCheck] = None


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


# --- the run -----------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    run_id: str
    #: executed | observed | drafted | rejected | timed_out | halted | refused | budget_exhausted |
    #: invalid_proposal | action_class_capped | error
    status: str
    reason: str = ""
    level: Optional[str] = None
    mode: Optional[str] = None
    generation: Optional[str] = None
    #: node ids of the ops records written, in order
    records: tuple[str, ...] = ()
    mediated: bool = False
    outcome: Optional[str] = None


PERSISTS = "demotion persists, last recorded "


def persisted_demotion_reason(demotion: Demotion) -> str:
    """The mode_reason of a run kept in shadow by an earlier demotion. It starts with
    DEMOTED_PREFIX, so the next run finds this one as the latest demotion and the chain holds."""
    core = demotion.reason[len(DEMOTED_PREFIX):] if demotion.reason.startswith(DEMOTED_PREFIX) else demotion.reason
    if core.startswith(PERSISTS):  # already a persisted reason: keep the original cause once
        core = core.split("): ", 1)[1] if "): " in core else core
    return f"{DEMOTED_PREFIX}{PERSISTS}{_iso(demotion.at)} (until `archi ops promote`): {core}"


def _iso(moment: datetime) -> str:
    """UTC with a Z. A naive time is written as given, without a Z, so it shows."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        return moment.isoformat()
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_once(automation: AutomationConfig, signal: Signal, deps: Deps, *, run_id: str) -> RunResult:
    """Run the loop once for one signal. `run_id` comes from the caller (see make_run_id)."""
    return _Run(automation, signal, deps, run_id).run()


class _Run:
    def __init__(self, automation: AutomationConfig, signal: Signal, deps: Deps, run_id: str):
        self.a, self.signal, self.deps, self.run_id = automation, signal, deps, run_id
        self.nodes: list[str] = []
        self.level: Optional[str] = None
        self.mode: Optional[str] = None
        self.mode_reason: Optional[str] = None
        self.generation: Optional[str] = None
        self.incident_written = False
        self.proposal_written = False
        self.incident_id = f"inc-{run_id}"
        self.proposal_id = f"prop-{run_id}"
        self.decision_id = f"dec-{run_id}"
        self.incident_node = ""

    # records

    def write(self, kind: str, record_id: str, *, evidence: Iterable[Any] = (), **attrs: Any) -> str:
        common = {"automation": self.a.name, "actor_kind": ACTOR_KIND, "actor": self.a.name,
                  "generation_read": self.generation}
        nid = self.deps.recorder.add(kind, evidence=evidence, record_id=record_id, **{**common, **attrs})
        self.nodes.append(nid)
        return nid

    def flush(self) -> None:
        flush = getattr(self.deps.recorder, "flush", None)
        if flush is not None:
            flush()

    def result(self, status: str, reason: str = "", **extra: Any) -> RunResult:
        return RunResult(self.run_id, status, reason, level=self.level, mode=self.mode,
                         generation=self.generation, records=tuple(self.nodes), **extra)

    def now(self) -> str:
        return _iso(self.deps.clock.now())

    def write_incident(self, summary: Optional[str] = None) -> None:
        s = self.signal
        self.incident_node = self.write(
            "incident", self.incident_id, title=s.title, summary=summary or s.summary, entity=s.entity,
            entity_kind=s.entity_kind, error_signature=s.error_signature, severity=s.severity,
            status="open", detected_at=_iso(s.detected_at or self.deps.clock.now()),
            ticket_keys=list(s.ticket_keys) or None, signal=f"{s.source}:{s.id}",
        )
        self.incident_written = True

    def run_level(self, status: str, reason: str) -> RunResult:
        """Stopped before reading anything: one incident, no generation (see module docstring)."""
        self.write_incident(f"run not started: {reason}")
        self.flush()
        return self.result(status, reason)

    def write_proposal(self, evidence: Iterable[Any], *, flush: bool = True, **attrs: Any) -> None:
        self.write("proposal", self.proposal_id, evidence=evidence, incident_id=self.incident_id,
                   autonomy_level=self.level, proposed_at=self.now(), **attrs)
        self.proposal_written = True
        if flush:  # a person deciding on it, or the next run, must be able to see it
            self.flush()

    def decide(self, verdict: str, reason: str, record_id: Optional[str] = None, **attrs: Any) -> None:
        attrs.setdefault("decided_at", self.now())
        self.write("decision", record_id or self.decision_id, proposal_id=self.proposal_id,
                   verdict=verdict, reason=reason, **attrs)
        self.flush()

    def stop(self, verdict: str, reason: str) -> RunResult:
        """No real proposal: a placeholder (skipped by shadow scoring) and a decision."""
        note = f"placeholder: no proposal was made ({reason}); run mode {self.mode}"
        if self.mode_reason:
            note += f"; {self.mode_reason}"
        self.write_proposal([self.incident_node], flush=False, summary=f"no proposal: {reason}",
                            mode=PLACEHOLDER_MODE, mode_reason=note)
        self.decide(verdict, reason)
        return self.result(verdict, reason)

    # the run

    def run(self) -> RunResult:
        try:
            return self._run()
        except Exception as exc:  # fail closed, and say so
            reason = f"{type(exc).__name__}: {exc}"
            if not self.incident_written:
                return self.run_level(ERROR, reason)
            if not self.proposal_written:
                return self.stop(ERROR, reason)
            self.decide(ERROR, reason, f"{self.decision_id}-error")
            return self.result(ERROR, reason)

    def _run(self) -> RunResult:
        a, deps = self.a, self.deps
        started = deps.clock.now()
        if started.tzinfo is None or started.utcoffset() is None:
            raise ValueError("clock returned a naive datetime")
        if deps.halt.halted():
            return self.result("halted", "halted before the run started")

        try:
            self.level = resolved_level(a)
        except LevelError as exc:
            return self.run_level("refused", str(exc))
        if a.mode not in MODES:
            return self.run_level("refused", f"mode {a.mode!r} is not one of {list(MODES)}")
        self.mode = a.mode
        level = rank(self.level)

        gate = deps.gate.check(a.gate) if a.gate is not None else None
        self.generation = deps.reader.pin()
        failure: Optional[str] = None
        if gate is not None and not gate.ok:
            failure = gate.message
        elif gate is not None and gate.generation not in (None, self.generation):
            failure = f"context gate read generation {gate.generation} but the run pinned {self.generation}"
        promoted_at = a.promoted_at
        if self.mode == "live" and deps.promotions is not None:
            # `mode: live` in ops.yaml is a request. It counts only while the PACT change that
            # `archi ops promote` opened is approved by a person; hand-editing ops.yaml is not enough.
            promotion = deps.promotions.check(a.name, a.promotion_change)
            if not promotion.approved:
                self.mode_reason = f"{NOT_PROMOTED_PREFIX}{promotion.reason}"
                self.mode = "shadow"
            else:
                promoted_at = promotion.at or promoted_at
                if promotion.no_trial:
                    self.mode_reason = (f"{NO_SHADOW_EVIDENCE_PREFIX}promotion {a.promotion_change} went live with no "
                                        f"shadow trial, on the decision: {promotion.no_trial}")
        if failure is not None:
            if self.mode == "live":
                self.mode_reason = DEMOTED_PREFIX + failure
                if deps.demotions is not None:
                    # written now, not at the next publish: the next run reads it
                    deps.demotions.record(a.name, self.run_id, deps.clock.now(), self.mode_reason)
            else:
                earlier = f"{self.mode_reason}; " if self.mode_reason else ""
                self.mode_reason = f"{earlier}context gate failed: {failure}"
            self.mode = "shadow"

        if self.mode == "live" and failure is None and deps.demotions is not None:
            demoted = deps.demotions.latest(a.name, self.generation)
            if demoted is not None and (promoted_at is None or demoted.at > promoted_at):
                # A failing gate demoted this automation earlier; a passing gate does not undo that.
                # Only `archi ops promote` does (it stamps promoted_at after the demotion).
                self.mode_reason = persisted_demotion_reason(demoted)
                self.mode = "shadow"

        self.write_incident()

        # budget, before the model is asked: per run and per day
        scopes = self.budget_scopes(started)
        for scope, cap in scopes:
            check = deps.budget.check(scope, a.estimate_cost, cap)
            if not check.ok:
                return self.stop("budget_exhausted", f"{scope.split(':')[0]} budget exhausted: {check.detail}")

        # propose, from the pinned generation only
        try:
            read = deps.reader.read(self.signal, self.generation)
            # the incident this run just wrote is a record the proposer may cite: it holds the signal
            # that started the run (a monitor's evidence is the instance's state, not a graph node)
            proposed = deps.proposer.propose(
                ProposeRequest(a.name, self.signal.as_dict(), {**read.context, "run_incident": self.incident_node},
                               self.generation, level))
        except Exception as exc:
            return self.stop(ERROR, f"proposer failed: {type(exc).__name__}: {exc}")
        exceeded: Optional[BudgetExceeded] = None
        for scope, _ in scopes:
            try:
                deps.budget.record(scope, proposed.cost)
            except BudgetExceeded as exc:
                exceeded = exc
                exceeded_scope = scope
        try:
            proposal = validate_proposal(proposed.proposal, a, self.level, ({*read.node_ids, self.incident_node} if a.own_signals else read.node_ids))
        except ProposalError as exc:
            return self.stop("invalid_proposal", f"proposal refused: {exc}")

        self.write_proposal(
            proposal.evidence_ids, summary=proposal.summary, action_class=proposal.action_class,
            payload=json.dumps(dict(proposal.payload), sort_keys=True) if proposal.payload else None,
            confidence=proposal.confidence, rollback=proposal.rollback, draft=proposal.draft,
            mode=self.mode, mode_reason=self.mode_reason,
        )

        # the model call may have used up a cap: stop before anything is escalated or run
        if exceeded is not None:
            reason = (f"{exceeded_scope.split(':')[0]} budget exhausted: charging {proposed.cost:g} "
                      "would pass the cap (the charge was refused)")
            self.decide("budget_exhausted", reason)
            return self.result("budget_exhausted", reason)
        for scope, cap in scopes:
            check = deps.budget.check(scope, 0.0, cap)
            if not check.ok:
                reason = f"{scope.split(':')[0]} budget exhausted: {check.detail}"
                self.decide("budget_exhausted", reason)
                return self.result("budget_exhausted", reason)

        # the class caps the level it may run at
        cls = a.action_classes.get(proposal.action_class) if proposal.action_class else None
        if cls is not None and rank(cls.max_level) < level:
            reason = (f"action class {proposal.action_class!r} allows at most {cls.max_level}, "
                      f"below the run's {self.level}; it is never executed")
            self.decide("action_class_capped", reason)
            return self.result("action_class_capped", reason)

        if self.mode == "shadow":
            would = {0: "observe", 1: "draft for a person to send", 2: "escalate for approval and then act"}[level]
            self.decide("observed", f"shadow: recorded only; a live run at {self.level} would {would}")
            return self.result("observed", "shadow")
        if level == 0:
            self.decide("observed", "L0: observed only")
            return self.result("observed", "L0")
        if level == 1:
            self.decide("drafted", "L1: draft recorded for a person to send; nothing was sent")
            return self.run_draft_step(proposal)
        return self.escalate(proposal)

    def run_draft_step(self, proposal: Proposal) -> RunResult:
        """At L1, run the action class's mediator when it is a workflow step with an implementation.

        A step has no external effect (`archi.ops.steps`), so it needs no escalation: it stores the
        draft (`draft_issue` writes a file in the instance repository). The halt check, the mediator
        record and the at-most-once guard are the same as for a live tool. A class whose mediator is a
        live tool, or a step with no implementation, is only recorded: nothing runs at L1.
        """
        cls = self.a.action_classes.get(proposal.action_class or "")
        step = get_step(cls.mediator) if cls and cls.mediator else None
        if step is None or step.impl is None or not self.a.own_signals:
            return self.result("drafted", "L1")
        mediated, why, refused = self._mediate(proposal, None)
        if mediated is None:
            self.decide(refused, f"{why}; the draft step did not run")  # supersedes "drafted" (same record id)
            return self.result(refused, why)
        if mediated.status != "succeeded":
            # a broken draft step is an error, not a draft: the decision says so (and the error rate counts it)
            reason = f"the draft step {cls.mediator} {mediated.status}: {mediated.result}"[:500]
            self.write(
                "action", f"act-{self.run_id}", proposal_id=self.proposal_id, decision_id=self.decision_id,
                action_class=proposal.action_class, mediator=cls.mediator,
                payload=json.dumps(dict(proposal.payload), sort_keys=True) if proposal.payload else None,
                status=mediated.status, result=mediated.result, executed_at=self.now(),
            )
            self.decide(ERROR, reason)
            return self.result(ERROR, reason, mediated=True)
        self.write(
            "action", f"act-{self.run_id}", proposal_id=self.proposal_id, decision_id=self.decision_id,
            action_class=proposal.action_class, mediator=cls.mediator,
            payload=json.dumps(dict(proposal.payload), sort_keys=True) if proposal.payload else None,
            status=mediated.status, result=mediated.result, executed_at=self.now(),
        )
        self.flush()
        return self.result("drafted", f"step {cls.mediator}: {mediated.status}", mediated=True)

    def budget_scopes(self, started: datetime) -> list[tuple[str, float]]:
        a = self.a
        day = started.astimezone(timezone.utc).date().isoformat()
        scopes = []
        if a.cost_per_run is not None:
            scopes.append((f"run:{self.run_id}", float(a.cost_per_run)))
        if a.cost_per_day is not None:
            scopes.append((f"day:{a.name}:{day}", float(a.cost_per_day)))
        return scopes

    def escalate(self, proposal: Proposal) -> RunResult:
        a, deps = self.a, self.deps
        timeout_at = deps.clock.now() + timedelta(seconds=a.timeout_seconds)
        escalation: Optional[Escalation] = None
        try:
            opened = deps.decider.open(automation=a.name, proposal_id=self.proposal_id, proposal=proposal,
                                       timeout_at=timeout_at)
            if opened.proposal_id != self.proposal_id:
                raise ValueError(f"escalation {opened.escalation_id!r} was opened for {opened.proposal_id!r}")
            escalation = replace(opened, timeout_at=timeout_at)
            decision = deps.decider.wait(escalation)
        except Exception as exc:
            decision = Decision(ERROR, error=f"{type(exc).__name__}: {exc}")
        allowed = may_execute(decision, escalation, deps.clock.now())

        verdict, attrs = decision.status, {}
        reason = {
            APPROVED: "approved by escalation",
            REJECTED: "rejected by escalation",
            TIMED_OUT: "no decision before the deadline",
            ERROR: f"escalation failed: {decision.error or 'unknown error'}",
        }.get(decision.status)
        if reason is None:
            verdict, reason = ERROR, f"unknown decision status {decision.status!r}"
        if decision.status == APPROVED and not allowed:
            verdict, reason = (TIMED_OUT if allowed.late else ERROR), f"not executed: {allowed.reason}"
        if decision.note:
            reason = f"{reason}; note: {decision.note}"
        if decision.status in (APPROVED, REJECTED) and decision.decided_by:
            attrs.update(actor_kind="person", actor=decision.decided_by)
        escalation_id = escalation.escalation_id if escalation else decision.escalation_id
        self.decide(verdict, reason, escalation_id=escalation_id,
                    decided_at=_iso(decision.decided_at) if decision.decided_at else self.now(), **attrs)
        if not allowed:
            return self.result(verdict, reason)

        mediated, why, refused = self._mediate(proposal, escalation)
        if mediated is None:
            # one decision supersedes the approved one (same record id)
            reason = f"{why}; supersedes the approval by {decision.decided_by} ({escalation_id})"
            self.decide(refused, reason, escalation_id=escalation_id, **attrs)
            return self.result(refused, reason)
        action = mediated
        action_id = f"act-{self.run_id}"
        cls = a.action_classes.get(proposal.action_class or "")
        self.write(
            "action", action_id, proposal_id=self.proposal_id, decision_id=self.decision_id,
            action_class=proposal.action_class, mediator=(cls.mediator or None) if cls else None,
            payload=json.dumps(dict(proposal.payload), sort_keys=True) if proposal.payload else None,
            status=action.status, result=action.result, executed_at=self.now(),
        )
        try:
            if action.status == "unknown":  # the mediator may have run; do not look to the checker for a verdict
                outcome = OutcomeResult("unknown", action.result)
            else:
                outcome = deps.outcome.check(signal=self.signal, action=action, generation=self.generation)
        except Exception as exc:
            outcome = OutcomeResult("unknown", f"outcome check failed: {type(exc).__name__}: {exc}")
        self.write(
            "outcome", f"out-{self.run_id}", incident_id=self.incident_id, action_id=action_id,
            result=outcome.result, resolution=outcome.resolution or None, check_method="outcome_check",
            checked_at=self.now(),
        )
        self.flush()
        return self.result("executed", f"mediator {action.status}", mediated=True, outcome=outcome.result)

    def _mediate(self, proposal: Proposal, escalation: Optional[Escalation]
                 ) -> tuple[Optional[MediatorResult], str, str]:
        """Run the mediator unless halted or refused. The HaltCheck is the first statement,
        so nothing can run between it and the external call (the durable step wraps this
        method; the real mediator step checks halt and the escalation again inside).
        Returns (result, "", "") or (None, why it did not run, the verdict to record)."""
        try:
            if self.deps.halt.halted():
                return None, "halted after approval, before the mediator ran", "halted"
        except Exception as exc:
            return None, f"halt check failed ({type(exc).__name__}: {exc}); failing closed", "halted"
        try:
            return self.deps.mediator.execute(
                action_class=proposal.action_class or "", payload=dict(proposal.payload),
                proposal_id=self.proposal_id, idempotency_key=self.run_id, generation=self.generation or "",
                escalation=escalation, level=self.level or "L2",
            ), "", ""
        except MediationRefused as exc:
            return None, exc.reason, exc.verdict
        except Exception as exc:
            return MediatorResult("failed", f"{type(exc).__name__}: {exc}"), "", ""
