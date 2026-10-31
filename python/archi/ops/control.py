"""Operator control: `archi ops status | halt | resume` (PACT r3-ops-runtime req.control).

Only the logic is here, as pure functions over lists the caller has read:
what status shows, and what halt and resume should pause and cancel. The okg
and DBOS calls (list schedules, pause, cancel, write the `halted` decisions)
come later and just carry out these plans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence

from archi.ops.levels import LevelError
from archi.ops.loop import DEMOTED_PREFIX, AutomationConfig, resolved_level

#: Run states that can still reach a mediator. Everything else is final.
LIVE_RUN_STATES = frozenset({"PENDING", "ENQUEUED", "DELAYED"})


@dataclass(frozen=True)
class BudgetUsage:
    """Caps and spend for one automation. None means no cap or not measured."""

    per_run_cap: Optional[float] = None
    per_day_cap: Optional[float] = None
    day_spent: Optional[float] = None

    @property
    def day_remaining(self) -> Optional[float]:
        if self.per_day_cap is None or self.day_spent is None:
            return None
        return max(self.per_day_cap - self.day_spent, 0.0)

    @property
    def exhausted(self) -> bool:
        return self.day_remaining == 0.0


@dataclass(frozen=True)
class ShadowSummary:
    shadow_proposals: int = 0
    live_proposals: int = 0
    #: live automations dropped to shadow by a failing context gate
    demotions: int = 0


@dataclass(frozen=True)
class AutomationStatus:
    name: str
    #: effective level as "L0".."L2", or None when the configuration is refused
    level: Optional[str]
    level_problem: Optional[str]
    mode: str
    schedule_paused: bool
    budget: BudgetUsage
    shadow: ShadowSummary

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "level": self.level, "level_problem": self.level_problem,
            "mode": self.mode, "schedule_paused": self.schedule_paused,
            "budget": {"per_run_cap": self.budget.per_run_cap, "per_day_cap": self.budget.per_day_cap,
                       "day_spent": self.budget.day_spent, "day_remaining": self.budget.day_remaining,
                       "exhausted": self.budget.exhausted},
            "shadow": {"shadow_proposals": self.shadow.shadow_proposals,
                       "live_proposals": self.shadow.live_proposals, "demotions": self.shadow.demotions},
        }


def shadow_summary(proposals: Iterable[Mapping[str, Any]]) -> dict[str, ShadowSummary]:
    """Counts per automation from ops_proposal attrs (`automation`, `mode`, `mode_reason`)."""
    counts: dict[str, list[int]] = {}
    for attrs in proposals:
        row = counts.setdefault(str(attrs.get("automation")), [0, 0, 0])
        if attrs.get("mode") == "shadow":
            row[0] += 1
            if str(attrs.get("mode_reason") or "").startswith(DEMOTED_PREFIX):
                row[2] += 1
        elif attrs.get("mode") == "live":
            row[1] += 1
    return {name: ShadowSummary(*row) for name, row in counts.items()}


def build_status(
    automations: Sequence[AutomationConfig],
    *,
    paused_schedules: Collection[str] = (),
    budgets: Mapping[str, BudgetUsage] = {},
    proposals: Iterable[Mapping[str, Any]] = (),
) -> list[AutomationStatus]:
    """One row per automation: level, mode, schedule state, budget, shadow counts."""
    shadow = shadow_summary(proposals)
    rows = []
    for automation in automations:
        try:
            level, problem = resolved_level(automation), None
        except LevelError as exc:
            level, problem = None, str(exc)
        rows.append(AutomationStatus(
            name=automation.name, level=level, level_problem=problem, mode=automation.mode,
            schedule_paused=automation.name in paused_schedules,
            budget=budgets.get(automation.name, BudgetUsage()),
            shadow=shadow.get(automation.name, ShadowSummary()),
        ))
    return rows


@dataclass(frozen=True)
class RunInfo:
    workflow_id: str
    automation: str
    status: str
    #: the entity of the signal the run was started for (empty when not known)
    entity: str = ""


@dataclass(frozen=True)
class HaltPlan:
    #: schedules to pause now (not already paused)
    pause: tuple[str, ...]
    #: schedules that were paused before halt; resume must leave them paused
    already_paused: tuple[str, ...]
    #: runs to cancel, each followed by a `halted` decision record
    cancel: tuple[RunInfo, ...]
    #: runs already final, left alone
    final: tuple[RunInfo, ...] = field(default=())


def halt_plan(schedules: Mapping[str, bool], runs: Sequence[RunInfo]) -> HaltPlan:
    """What halt does. `schedules` maps schedule name to whether it is paused."""
    pause = tuple(sorted(name for name, paused in schedules.items() if not paused))
    already = tuple(sorted(name for name, paused in schedules.items() if paused))
    cancel = tuple(r for r in runs if r.status in LIVE_RUN_STATES)
    final = tuple(r for r in runs if r.status not in LIVE_RUN_STATES)
    return HaltPlan(pause, already, cancel, final)


def resume_plan(schedules: Mapping[str, bool], paused_by_halt: Collection[str]) -> tuple[str, ...]:
    """Schedules resume should un-pause: those halt paused and that are still paused.

    Cancelled runs are not resumed; a new signal starts a new run.
    """
    return tuple(sorted(name for name, paused in schedules.items() if paused and name in paused_by_halt))
