"""Workflow steps an action class may name as its mediator (ADR 0002 section 7.2).

This registry is the one place steps are registered; the runtime adds its
implementations against these names. A step is a mediator for work with **no
external effect**, so its highest level is L0 or L1. Anything that changes
an external system (a ticket update, a restart) must be a live tool from a
tool pack: an automation never calls an external system any other way.

    record_finding   L0  writes a finding to the ops record
    draft_reply      L1  stores a draft reply for a person to send; never sends
    draft_issue      L1  writes a draft issue file in the instance repository; never reaches a forge

A step with an implementation (`impl`, a `module:function` called as `fn(payload=..., context=...)`)
runs at L1 with no escalation, because it has no external effect: `draft_issue` writes files under
`.archi/issues/` in the instance's own repository and commits them there, nothing else. A step with no
implementation (`record_finding`, `draft_reply`) is recorded on the proposal and never executed. The
GitLab and GitHub issue backends are not steps: they reach external systems, so they are a live tool
(`archi.tools.instance_health`) that runs at L2 after approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from archi.ops.levels import level_problem, rank

#: The highest level a workflow step may carry. L2 means an approved
#: escalation runs it against an external system, which only a live tool may do.
STEP_CEILING = "L1"


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    max_level: str
    description: str
    #: `module:function` the runtime calls at L1 (no escalation); None: recorded, never executed
    impl: Optional[str] = None


_STEPS: dict[str, WorkflowStep] = {}


def register_step(name: str, max_level: str, description: str, impl: Optional[str] = None) -> WorkflowStep:
    """Register a step. Refuses a repeated name, a level above L1 and an unknown level."""
    message = level_problem(max_level, f"step {name!r} max_level")
    if message:
        raise ValueError(message)
    if rank(max_level) > rank(STEP_CEILING):
        raise ValueError(
            f"step {name!r} max_level {max_level}: a workflow step has no external effect, "
            f"so it stops at {STEP_CEILING}; an action that changes an external system "
            "must be a live tool from a tool pack (ADR 0002 section 7.2)"
        )
    if name in _STEPS:
        raise ValueError(f"workflow step {name!r} is already registered")
    if impl is not None and (impl.count(":") != 1 or not all(impl.split(":"))):
        raise ValueError(f"step {name!r} impl {impl!r} is not module:function")
    step = WorkflowStep(name, max_level, description, impl)
    _STEPS[name] = step
    return step


def get_step(name: str) -> WorkflowStep | None:
    return _STEPS.get(name)


def local_step(entry: str) -> Optional[WorkflowStep]:
    """The registered step with an implementation that a mediator entry names, else None.

    Such a step runs at L1 with no escalation (see the module docstring). A `module:function` entry
    (a live tool) is never a step.
    """
    if ":" in entry:
        return None
    step = _STEPS.get(entry)
    return step if step is not None and step.impl is not None else None


def step_names() -> tuple[str, ...]:
    return tuple(_STEPS)


register_step("record_finding", "L0", "Write a finding to the ops record. Nothing else changes.")
register_step("draft_reply", "L1", "Store a draft reply for a person to send. Never sends it.")
register_step(
    "draft_issue", "L1",
    "Write a draft issue file in the instance repository (.archi/issues/) and commit it there. "
    "Never reaches a forge.",
    impl="archi.health.steps:draft_issue",
)
