"""Live tool: sync the instance-health problems to a GitLab or GitHub project.

This is the only place the health automation reaches a forge (ADR 0002 section 7.2: an external system is
reached only through a live tool). It is the mediator of the action class `sync_forge_issues`, which runs at
L2 only, after a person approves the escalation. Opening an issue on a forge is an external post, and at L1
the ADR has the automation draft for a person to send, so each run that would post waits for an approval. The
pack allows L2 (default L1): an instance that wants forge issues sets its level to L2 in `archi.yaml`, and at
L1 the run only drafts (and its proposal says that the forge tracker was not used).

    sync_forge_issues(payload=..., context=...) -> {"status", "result"}

`payload` is the proposal payload (`archi.health.proposer`): the tracker block (kind, url or repo,
`token_env`), the problems, and the scope. The token is read from the environment variable `token_env`
names. Idempotent: issues are found by the key in their body, so a second call with the same payload (or a
repeat after a crash between the forge call and its record) creates nothing new.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from archi.health.config import build_tracker
from archi.health.steps import describe, parse_payload
from archi.health.tracker import TrackerError, reconcile


def sync_forge_issues(*, payload: Mapping[str, Any], context: Mapping[str, Any], session: Optional[Any] = None,
                      env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    spec, problems, scope, _ = parse_payload(payload)
    if spec.kind == "local":
        raise TrackerError("sync_forge_issues is for gitlab or github; the local tracker is the draft_issue step")
    tracker = build_tracker(spec, env=os.environ if env is None else env, session=session)
    return {"status": "succeeded", "result": f"{spec.kind}: {describe(reconcile(tracker, problems, scope))}"}
