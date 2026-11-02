"""Escalation decisions for L2 proposals (ADR 0002 section 7.3, D13).

An L2 automation proposes; it acts only after a person approves through an
okg escalation. This module holds the parts that need no database:

- the schema of the decision payload a person answers with
  (`{"verdict": "approve" | "reject", "note": "..."}`), and
- the deadline rule: a proposal executes only if it was approved, at a known
  decision time, no later than the proposal's timeout.

The okg calls (open an escalation, read its state, respond) live in the
workflow module, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

VERDICTS = ("approve", "reject")

#: The schema the escalation is opened with (okg `schema:literal:` decision schema).
DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "note": {"type": "string"},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}

#: Where a decision can end up. Only `approved` can lead to execution.
APPROVED, REJECTED, TIMED_OUT, ERROR = "approved", "rejected", "timed_out", "error"
STATUSES = (APPROVED, REJECTED, TIMED_OUT, ERROR)


class DecisionPayloadError(ValueError):
    """A decision payload that does not match DECISION_SCHEMA."""


def parse_decision_payload(payload: Any) -> tuple[str, Optional[str]]:
    """Return `(verdict, note)` or raise DecisionPayloadError."""
    if not isinstance(payload, Mapping):
        raise DecisionPayloadError("decision payload must be an object")
    extra = sorted(set(payload) - {"verdict", "note"})
    if extra:
        raise DecisionPayloadError(f"unknown decision keys {extra}")
    verdict = payload.get("verdict")
    if verdict not in VERDICTS:
        raise DecisionPayloadError(f"verdict must be one of {list(VERDICTS)}, got {verdict!r}")
    note = payload.get("note")
    if note is not None and not isinstance(note, str):
        raise DecisionPayloadError("note must be a string")
    return verdict, note


def status_from_payload(payload: Any) -> str:
    """`approved` or `rejected` for a valid payload."""
    verdict, _ = parse_decision_payload(payload)
    return APPROVED if verdict == "approve" else REJECTED


@dataclass(frozen=True)
class Escalation:
    """The escalation the loop opened for one proposal."""

    escalation_id: str
    proposal_id: str
    opened_at: datetime
    #: The loop sets this from its own clock; it does not trust the Decider's copy.
    timeout_at: datetime


@dataclass(frozen=True)
class Decision:
    """What the Decider reports about one escalation once it is decided."""

    status: str
    proposal_id: Optional[str] = None
    escalation_id: Optional[str] = None
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    note: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class Execution:
    """Whether a decision lets the mediator run, and if not, why."""

    ok: bool
    reason: str
    #: True when the only fault is that the approval came after the deadline
    late: bool = False

    def __bool__(self) -> bool:
        return self.ok


def _aware(*moments: Optional[datetime]) -> bool:
    return all(m is not None and m.tzinfo is not None and m.utcoffset() is not None for m in moments)


def may_execute(decision: Decision, escalation: Optional[Escalation], now: datetime) -> Execution:
    """True only for an approval that is bound to this escalation and in time.

    All of these must hold: the decision is `approved`; it names the proposal
    and the escalation the loop opened (not just any); a person (`decided_by`)
    made it; it has a decision time at or after the escalation opened, at or
    before the deadline, and not after `now`. Naive datetimes are refused,
    not guessed at. A decision we cannot show was valid is not valid.
    """
    if decision.status != APPROVED:
        return Execution(False, f"decision is {decision.status}, not approved")
    if escalation is None:
        return Execution(False, "no escalation was opened for this proposal")
    if decision.proposal_id != escalation.proposal_id:
        return Execution(False, f"decision is for proposal {decision.proposal_id!r}, not {escalation.proposal_id!r}")
    if not decision.escalation_id or decision.escalation_id != escalation.escalation_id:
        return Execution(False, f"decision is for escalation {decision.escalation_id!r}, not {escalation.escalation_id!r}")
    if not decision.decided_by:
        return Execution(False, "approval names no person (decided_by)")
    if decision.decided_at is None:
        return Execution(False, "approval has no decision time")
    if not _aware(decision.decided_at, escalation.opened_at, escalation.timeout_at, now):
        return Execution(False, "a time is naive (no timezone); refusing to compare")
    if decision.decided_at < escalation.opened_at:
        return Execution(False, f"approved at {decision.decided_at.isoformat()}, before the escalation opened")
    if decision.decided_at > now:
        return Execution(False, f"decision time {decision.decided_at.isoformat()} is after now")
    if decision.decided_at > escalation.timeout_at:
        return Execution(
            False,
            f"approved at {decision.decided_at.isoformat()}, after the deadline "
            f"{escalation.timeout_at.isoformat()}",
            late=True,
        )
    return Execution(True, "approved within the deadline")


# --- reading an escalation row back (okg.escalation_live) ---------------------------

SUBJECT_PREFIX = "ops_proposal:"
OPEN_STATES = ("pending", "routed")


def subject_for(proposal_id: str) -> str:
    """The node id an escalation names as its subject."""
    return SUBJECT_PREFIX + proposal_id


def _moment(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def decision_from_row(escalation_id: str, subject: Optional[str], attrs: Optional[Mapping[str, Any]]) -> Decision:
    """The Decision an `okg.escalation_live` row (current version) describes.

    `subject` is the row's `subject_node_id`; the proposal the person decided on
    is read from it, never assumed. A row still open, missing, or with a
    verdict outside the schema is an error decision, so it can never execute.
    """
    proposal_id = subject[len(SUBJECT_PREFIX):] if subject and subject.startswith(SUBJECT_PREFIX) else None
    if attrs is None:
        return Decision(ERROR, proposal_id, escalation_id, error="escalation row not found")
    status = attrs.get("status")
    common = dict(proposal_id=proposal_id, escalation_id=escalation_id, decided_by=attrs.get("decided_by"),
                  decided_at=_moment(attrs.get("decided_at")))
    if status == "timed_out":
        return Decision(TIMED_OUT, **common)
    if status not in ("decided", "applied"):
        return Decision(ERROR, **{**common, "decided_at": None}, error=f"escalation is {status}, not decided")
    try:
        verdict, note = parse_decision_payload(attrs.get("decision"))
    except DecisionPayloadError as exc:
        return Decision(ERROR, **common, error=f"decision payload refused: {exc}")
    return Decision(APPROVED if verdict == "approve" else REJECTED, **common, note=note)
