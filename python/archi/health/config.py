"""What the instance configures for the health automation (`ops.yaml`).

    automations:
      instance-health:
        mode: shadow                    # or live, after `archi ops promote`
        proposer:
          kind: instance-health
          tracker:                      # optional; the default is the local tracker
            kind: local                 # local | gitlab | github
          # kind: gitlab
          # url: https://gitlab.cern.ch
          # project: cms-comp-ops/instance-issues      # path or numeric id
          # token_env: ARCHI_HEALTH_GITLAB_TOKEN       # the NAME of the variable; never the token
          # kind: github
          # repo: myorg/my-instance
          # token_env: ARCHI_HEALTH_GITHUB_TOKEN
          # api_url: https://api.github.com           # optional (GitHub Enterprise); https only
          # Issue bodies carry doctor messages and errors: point a forge tracker at a PRIVATE project.
          ignore_checks: []             # doctor check ids to leave out (e.g. services)
          include_warnings: false       # doctor `warn` results are problems too
          error_rate: {window_hours: 24, min_decisions: 3, max_rate: 0.5}
          eval:
            min_drop: 0.1               # a drop in score (passed / total) larger than this is a problem
            reports_dir: null           # default <data root>/eval, where `archi eval` writes; relative = from the instance
            max_age_days: 14            # a newest run older than this is noted in the proposal (not an issue)

Only the name of the token variable is written here; a value in this file is refused.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from archi.health.tracker import IssueTracker, TrackerError

KINDS = ("local", "gitlab", "github")
ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ALLOWED = {"kind", "url", "project", "repo", "token_env", "api_url"}


@dataclass(frozen=True)
class TrackerSpec:
    kind: str = "local"
    url: Optional[str] = None
    project: Optional[str] = None
    repo: Optional[str] = None
    token_env: Optional[str] = None
    api_url: Optional[str] = None

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in (("kind", self.kind), ("url", self.url), ("project", self.project),
                                  ("repo", self.repo), ("token_env", self.token_env), ("api_url", self.api_url)) if v}

    @property
    def action_class(self) -> str:
        """`draft_issue` (a step, runs at L1) for local files; a forge needs the live tool (L2)."""
        return "draft_issue" if self.kind == "local" else "sync_forge_issues"

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "TrackerSpec":
        data = dict(data or {})
        unknown = sorted(set(data) - _ALLOWED)
        if unknown:
            raise TrackerError(f"tracker: unknown keys {unknown}; allowed: {sorted(_ALLOWED)}")
        kind = data.get("kind", "local")
        if kind not in KINDS:
            raise TrackerError(f"tracker.kind {kind!r} is not one of {list(KINDS)}")
        spec = cls(kind, *(data.get(k) for k in ("url", "project", "repo", "token_env", "api_url")))
        if kind == "local":
            extra = sorted(k for k in data if k != "kind")
            if extra:
                raise TrackerError(f"tracker: the local tracker takes no {extra}")
            return spec
        if not spec.token_env:
            raise TrackerError(f"tracker: {kind} needs token_env, the NAME of the environment variable that holds the token")
        if not ENV_NAME.match(spec.token_env):
            raise TrackerError("tracker.token_env is not an environment variable name; "
                               "write the name of the variable, never the token itself")
        if kind == "gitlab" and not (spec.url and spec.project):
            raise TrackerError("tracker: gitlab needs url (https://gitlab.example) and project (path or id)")
        for label, url in (("url", spec.url), ("api_url", spec.api_url)):
            if url and not url.lower().startswith("https://"):
                raise TrackerError(f"tracker.{label} must be an https:// URL (the token is sent with every request)")
        if kind == "github" and not spec.repo:
            raise TrackerError("tracker: github needs repo (owner/name)")
        return spec


@dataclass(frozen=True)
class HealthConfig:
    tracker: TrackerSpec = field(default_factory=TrackerSpec)
    ignore_checks: tuple[str, ...] = ()
    include_warnings: bool = False
    window_hours: float = 24.0
    min_decisions: int = 3
    max_error_rate: float = 0.5
    #: a fall in eval score (passed / total, 0..1) larger than this, against the mean of the earlier runs, is a
    #: problem. 0.1 is one question of a ten-question set: one flip is noise, two are a finding.
    eval_min_drop: float = 0.1
    #: where `archi eval` reports are read from; None: `<data root>/eval`, the `archi eval` default
    eval_reports_dir: Optional[str] = None
    #: a bundle whose newest eval run is older than this many days gets a note in the proposal, not an issue
    eval_max_age_days: float = 14.0

    @classmethod
    def from_proposer(cls, config: Mapping[str, Any]) -> "HealthConfig":
        known = {"kind", "tracker", "ignore_checks", "include_warnings", "error_rate", "eval"}
        unknown = sorted(set(config) - known)
        if unknown:
            raise TrackerError(f"proposer: unknown keys {unknown}; allowed: {sorted(known)}")
        rate = dict(config.get("error_rate") or {})
        ev = dict(config.get("eval") or {})
        for name, allowed in (("error_rate", {"window_hours", "min_decisions", "max_rate"}),
                              ("eval", {"min_drop", "reports_dir", "max_age_days"})):
            bad = sorted(set(config.get(name) or {}) - allowed)
            if bad:
                raise TrackerError(f"proposer.{name}: unknown keys {bad}; allowed: {sorted(allowed)}")
        ignore = config.get("ignore_checks") or []
        if not isinstance(ignore, list) or not all(isinstance(i, str) for i in ignore):
            raise TrackerError("proposer.ignore_checks must be a list of check ids")
        return cls(
            tracker=TrackerSpec.from_mapping(config.get("tracker")),
            ignore_checks=tuple(ignore),
            include_warnings=bool(config.get("include_warnings", False)),
            window_hours=float(rate.get("window_hours", 24.0)),
            min_decisions=int(rate.get("min_decisions", 3)),
            max_error_rate=float(rate.get("max_rate", 0.5)),
            eval_min_drop=_fraction(ev.get("min_drop", 0.1), "proposer.eval.min_drop"),
            eval_reports_dir=_optional_text(ev.get("reports_dir"), "proposer.eval.reports_dir"),
            eval_max_age_days=_positive(ev.get("max_age_days", 14.0), "proposer.eval.max_age_days"),
        )


def _fraction(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TrackerError(f"{label} must be a number between 0 and 1") from None
    if not 0 <= number <= 1:
        raise TrackerError(f"{label} must be between 0 and 1 (a share of the questions), not {number:g}")
    return number


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TrackerError(f"{label} must be a number of days") from None
    if not number > 0:
        raise TrackerError(f"{label} must be more than 0, not {number:g}")
    return number


def _optional_text(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TrackerError(f"{label} must be a path")
    return value


def build_tracker(spec: TrackerSpec, *, instance_dir: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
                  session: Any = None) -> IssueTracker:
    """The tracker a spec names. A forge reads its token from the variable `token_env` names."""
    env = os.environ if env is None else env
    if spec.kind == "local":
        from archi.health.local import LocalTracker

        if not instance_dir:
            raise TrackerError("the local tracker needs the instance directory")
        return LocalTracker(instance_dir)
    token = env.get(spec.token_env or "")
    if not token:
        raise TrackerError(f"environment variable {spec.token_env} is not set; the {spec.kind} tracker needs its token")
    from archi.health import forge

    if spec.kind == "gitlab":
        return forge.GitLabTracker(spec.url, spec.project, token, session)
    return forge.GitHubTracker(spec.repo, token, session, api_url=spec.api_url or "https://api.github.com")
