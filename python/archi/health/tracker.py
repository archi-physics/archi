"""The issue tracker interface and the reconcile step (PACT r3-ops-self-monitoring).

`reconcile(tracker, problems, scope)` makes the tracker match what was found, one issue per key:

    key not in the tracker            -> create
    open issue, text differs          -> update
    open issue, text the same         -> unchanged (nothing is written)
    resolved issue, problem is back   -> reopen it (the same issue; never a second one)
    open issue, key not found now     -> resolve, if its family was read in this run

An issue a person closed is not reopened while the problem lasts: the run notes once, on the issue, that the
problem is still found, and leaves it closed (it was the person's decision). Only an issue this automation
resolved itself is reopened when its problem returns.

A family that could not be read in this run is not in `scope`: its issues stay as they are. Reading
nothing is not evidence that the problem is gone.

Backends: `archi.health.local` (files in the instance repository; the default, no credential) and
`archi.health.forge` (GitLab, GitHub). The forge backends reach an external system, so they run only
from the live tool `archi.tools.instance_health` (ADR 0002 section 7.2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

from archi.health.problems import Problem

OPEN, RESOLVED = "open", "resolved"
LABEL = "archi-health"
#: `<!-- archi-health key=K family=F [resolved-by=archi] [noted=1] -->`: how a forge issue is found by its body
MARKER = re.compile(r"<!--\s*archi-health\s+key=(\S+)\s+family=(\S+)((?:\s+[a-z-]+=[^\s>]+)*)\s*-->")
STILL_PRESENT = ("The problem is still found, but this issue was closed by a person, so it stays closed. "
                 "Reopen it, or fix the problem.")


class TrackerError(RuntimeError):
    """The tracker could not be read or written. The run fails; it never pretends it worked."""


@dataclass(frozen=True)
class Issue:
    key: str
    family: str
    title: str
    #: the problem's rendered text (`render_body`), as the tracker holds it
    body: str
    state: str
    #: where a person finds it: a repository path, or the forge's issue URL
    ref: str = ""
    #: for a resolved issue: did this automation resolve it (False: a person closed it)
    resolved_by_us: bool = True
    #: a person-closed issue already carries the note that the problem is still found
    noted: bool = False


@dataclass(frozen=True)
class Change:
    key: str
    action: str  # created | updated | reopened | resolved | noted | unchanged
    ref: str = ""


class IssueTracker(Protocol):
    def open_issues(self) -> list[Issue]:
        """Every open issue this automation made (the ones that carry a key)."""

    def find(self, key: str) -> Optional[Issue]:
        """The issue for `key`, open or resolved; None if there never was one."""

    def create(self, problem: Problem) -> Issue: ...

    def update(self, issue: Issue, problem: Problem) -> Issue:
        """Rewrite the issue's text, and reopen it if it was resolved."""

    def resolve(self, issue: Issue, note: str) -> Issue: ...

    def note_still_present(self, issue: Issue, text: str) -> Issue:
        """Record, once, on an issue a person closed, that the problem is still found. Does not reopen it."""

    def finish(self) -> None:
        """Make the run's changes durable (the local backend commits them). A no-op for a forge."""


def render_body(problem: Problem) -> str:
    """The issue text for a problem: the detail, then what it is and how it closes."""
    return (
        f"{problem.detail.strip()}\n\n"
        f"Severity: {problem.severity}. Found by the instance-health automation (signal: {problem.family}). "
        "It updates this issue while the problem lasts and marks it resolved when a later check finds "
        "the problem gone.\n"
    )


def same_text(a: str, b: str) -> bool:
    """Equal after line-ending and trailing-space differences a forge may introduce."""
    def norm(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n"))

    return norm(a) == norm(b)


def needs_update(issue: Issue, problem: Problem) -> bool:
    return issue.state != OPEN or issue.title != problem.title or not same_text(issue.body, render_body(problem))


def reconcile(
    tracker: IssueTracker,
    problems: Sequence[Problem],
    scope: Iterable[str],
    *,
    resolve_note: str = "The problem is no longer found.",
) -> list[Change]:
    """Make the tracker match `problems`. Returns what was done, in key order."""
    by_key: dict[str, Problem] = {}
    for problem in problems:
        if problem.key in by_key:
            raise ValueError(f"two problems share the key {problem.key!r}; a key names one problem")
        by_key[problem.key] = problem
    scope = set(scope)
    changes: list[Change] = []
    try:
        for key in sorted(by_key):
            problem = by_key[key]
            issue = tracker.find(key)
            if issue is None:
                made = tracker.create(problem)
                changes.append(Change(key, "created", made.ref))
            elif issue.state != OPEN and not issue.resolved_by_us:
                if issue.noted:
                    changes.append(Change(key, "unchanged", issue.ref))
                else:
                    made = tracker.note_still_present(issue, STILL_PRESENT)
                    changes.append(Change(key, "noted", made.ref))
            elif needs_update(issue, problem):
                reopened = issue.state != OPEN
                made = tracker.update(issue, problem)
                changes.append(Change(key, "reopened" if reopened else "updated", made.ref))
            else:
                changes.append(Change(key, "unchanged", issue.ref))
        for issue in sorted(tracker.open_issues(), key=lambda i: i.key):
            if issue.key not in by_key and issue.family in scope:
                done = tracker.resolve(issue, resolve_note)
                changes.append(Change(issue.key, "resolved", done.ref))
    finally:
        tracker.finish()  # what was changed before a failure is still made durable
    return sorted(changes, key=lambda c: c.key)


def summarize(changes: Sequence[Change]) -> str:
    """One line for the mediator's result."""
    counts: dict[str, int] = {}
    for change in changes:
        counts[change.action] = counts.get(change.action, 0) + 1
    if not counts:
        return "no problems found and no open issues"
    return ", ".join(f"{n} {action}" for action, n in sorted(counts.items()))
