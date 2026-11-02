"""The `draft_issue` workflow step: write the run's problems as issue files in the instance repository.

A step has no external effect (`archi.ops.steps`), and this one touches only `<instance>/.archi/issues/`
and the instance's own commit history. It runs at L1 with no escalation. It is called as every mediator is,
`draft_issue(payload=..., context=...)`, with `context["instance_dir"]`, and returns
`{"status", "result"}`. Running it twice with the same payload changes nothing the second time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from archi.health.config import TrackerSpec
from archi.health.local import LocalTracker
from archi.health.problems import Problem
from archi.health.tracker import TrackerError, reconcile, summarize


def parse_payload(payload: Mapping[str, Any]) -> tuple[TrackerSpec, list[Problem], list[str], datetime]:
    spec = TrackerSpec.from_mapping(payload.get("tracker"))
    problems = [Problem.from_dict(p) for p in payload.get("problems") or []]
    observed = datetime.fromisoformat(str(payload["observed_at"]).replace("Z", "+00:00"))
    return spec, problems, list(payload.get("scope") or []), observed


def describe(changes) -> str:
    parts = [summarize(changes)]
    parts += [f"{c.action} {c.key} -> {c.ref}" for c in changes if c.action != "unchanged"]
    return "; ".join(parts)


def draft_issue(*, payload: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, str]:
    spec, problems, scope, observed = parse_payload(payload)
    if spec.kind != "local":
        raise TrackerError(f"draft_issue writes local files only; the {spec.kind} tracker is the live tool "
                           "archi.tools.instance_health:sync_forge_issues (L2)")
    if not context.get("instance_dir"):
        raise TrackerError("draft_issue needs context['instance_dir']")
    tracker = LocalTracker(context["instance_dir"], now=lambda: observed.astimezone(timezone.utc))
    return {"status": "succeeded", "result": describe(reconcile(tracker, problems, scope))}
