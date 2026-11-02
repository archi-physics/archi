"""The proposer of the instance-health automation: algorithmic, no model call (ADR 0002: algorithmic first).

It reads the signals (`archi.health.signals`) and proposes one thing per run: reconcile the tracker with
what was found. The payload carries the whole picture, so the mediator is a single idempotent call:

    action_class  draft_issue         the local tracker (a step, runs at L1)
                  sync_forge_issues   GitLab or GitHub (a live tool, L2 only)
    payload       {tracker, observed_at, problems: [...], scope: [...], skipped: [...]}

`scope` lists the signal families that were read in this run; only those may close an issue. The proposal
cites the run's own incident (`context["run_incident"]`) as its evidence: the monitor's evidence is the
instance's state at that moment, which is in the payload, not a node of the graph.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from archi.health import signals
from archi.health.config import HealthConfig
from archi.ops.proposers import Proposed, ProposeRequest

ROLLBACK = {
    "draft_issue": "Revert the commit under .archi/issues/ in the instance repository, or delete the issue file.",
    "sync_forge_issues": "Close the issue on the forge; reopened issues can be closed again. Nothing else changed.",
}


class HealthProposer:
    def __init__(self, config: HealthConfig, *, instance_dir, deployment: str, dsn: str,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 read_doctor: Optional[Callable[[], Mapping[str, Any]]] = None,
                 read_frontier: Optional[Callable[[], Mapping[str, Any]]] = None,
                 read_ops_records: Optional[Callable[[str], list]] = None):
        self.config, self.instance_dir, self.now = config, instance_dir, now
        self.read_doctor = read_doctor or (lambda: signals.run_doctor(instance_dir))
        self.read_frontier = read_frontier or (lambda: signals.run_frontier(deployment, dsn))
        if read_ops_records is None:
            from archi.ops.shadow import okg_source

            read_ops_records = okg_source(deployment, dsn=dsn)
        self.read_ops_records = read_ops_records

    def propose(self, request: ProposeRequest) -> Proposed:
        observed = self.now()
        obs = signals.collect(self.config, instance_dir=self.instance_dir, now=observed,
                              read_doctor=self.read_doctor, read_frontier=self.read_frontier,
                              read_ops_records=self.read_ops_records)
        spec = self.config.tracker
        keys = [p.key for p in obs.problems]
        if keys:
            summary = f"Instance health: {len(keys)} problem(s): {', '.join(keys)}"
        else:
            summary = f"Instance health: no problems found in {', '.join(s for s in obs.scope if s != 'monitor')}"
        if spec.kind != "local" and request.level < 2:
            summary += (f"; WARNING: the {spec.kind} tracker is configured but this run is at L{request.level}: "
                        "an issue on a forge is an external post, so it is only drafted here and nothing was sent. "
                        "Set the instance level to L2 (an approval per run) or use the local tracker")
        if obs.skipped:
            summary += "; skipped: " + "; ".join(f"{s['signal']} ({s['reason']})" for s in obs.skipped)
        return Proposed({
            "summary": summary,
            "evidence_ids": [request.context["run_incident"]],
            "action_class": spec.action_class,
            "payload": {
                "tracker": spec.as_dict(),
                "observed_at": observed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "problems": [p.as_dict() for p in obs.problems],
                "scope": obs.scope,
                "skipped": obs.skipped,
            },
            "confidence": 1.0,
            "rollback": ROLLBACK[spec.action_class],
        }, cost=0.0)


def build(config: Mapping[str, Any], runtime: Any) -> HealthProposer:
    """`archi.ops.proposers.PROPOSER_KINDS["instance-health"]`: from the `proposer:` block of `ops.yaml`."""
    return HealthProposer(HealthConfig.from_proposer(config), instance_dir=runtime.instance_dir,
                          deployment=runtime.deployment, dsn=runtime.dsn)
