"""A problem the instance-health automation found, and its stable key.

One problem is one issue. The key is what makes that true: it names the check and the thing the check
is about (`source-freshness/jira.jira`), never a time or a count, so the same problem found on the next
run has the same key and finds the same issue.

    key      <check>/<subject>   check: a doctor check id, `ingest`, `automation-errors`, `eval`, `monitor`
    family   which signal found it: doctor | ingest | automation-errors | eval | monitor

A run closes an issue only when its family was read successfully in that run and the key is no longer
among the problems (see `archi.health.tracker.reconcile`).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

FAMILIES = ("doctor", "ingest", "automation-errors", "eval", "monitor")
SEVERITIES = ("error", "warning")

_CHECK = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SUBJECT = re.compile(r"[^A-Za-z0-9._-]+")
KEY = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9._-]+$")


class ProblemError(ValueError):
    """A problem or its key is malformed."""


def make_key(check: str, subject: str) -> str:
    """`<check>/<subject>`, all lowercase (a key is a file name, and file systems and forges may ignore case).

    A subject that is not already lowercase letters, digits, `.`, `_` and `-` is lowercased and cleaned (any
    other character becomes `_`), and a short hash of the original is added, so `Foo` and `foo`, or `a b` and
    `a_b`, never make the same key.
    """
    if not _CHECK.match(check):
        raise ProblemError(f"check id {check!r} must be lowercase letters, digits, '.', '_' or '-'")
    raw = str(subject)
    cleaned = _SUBJECT.sub("_", raw.lower()).strip("_") or "instance"
    if cleaned != raw:
        cleaned = f"{cleaned}-{hashlib.sha1(raw.encode()).hexdigest()[:6]}" if raw else "instance"
    return f"{check}/{cleaned}"


def slug(key: str) -> str:
    """A file name for a key: `source-freshness/jira.jira` -> `source-freshness--jira.jira`."""
    if not KEY.match(key):
        raise ProblemError(f"problem key {key!r} is not <check>/<subject>")
    return key.replace("/", "--")


@dataclass(frozen=True)
class Problem:
    key: str
    family: str
    title: str
    #: Markdown. Stable between runs while the problem is unchanged (no ages, no run times), because
    #: an issue is rewritten whenever this text changes.
    detail: str
    severity: str = "error"

    def __post_init__(self) -> None:
        if not KEY.match(self.key):
            raise ProblemError(f"problem key {self.key!r} is not <check>/<subject>")
        if self.family not in FAMILIES:
            raise ProblemError(f"problem {self.key}: family {self.family!r} is not one of {list(FAMILIES)}")
        if self.severity not in SEVERITIES:
            raise ProblemError(f"problem {self.key}: severity {self.severity!r} is not one of {list(SEVERITIES)}")
        if not self.title.strip():
            raise ProblemError(f"problem {self.key}: a title is required")

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "family": self.family, "title": self.title, "detail": self.detail,
                "severity": self.severity}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Problem":
        return cls(key=data["key"], family=data["family"], title=data["title"], detail=data["detail"],
                   severity=data.get("severity", "error"))
