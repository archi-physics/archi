"""What the ops runtime reads from an instance (PACT r3-ops-runtime).

The worker and the `archi ops` commands share this module. It reads no
database and imports no okg:

- `<instance>/archi.yaml`: each automation's level (`automations.<name>.level`).
  It is read as plain YAML here, not through `Instance.load`, because the
  runtime must refuse L3 at run start (ADR 0002 D13) rather than fail to start.
- `<instance>/automations/<name>.yaml`: the rendered automation block
  (`archi.ops.loop.load_automation`).
- `<instance>/ops.yaml`: what the instance decides about running them:

      automations:
        scratch-triage:
          mode: live              # shadow (default) | live
          timeout_seconds: 60     # how long an L2 escalation waits
          estimate_cost: 0.1      # checked against the budget before each proposal (default: the pack's cost_per_run)
          promotion: promote-scratch-triage-20261026t090000z  # set by `archi ops promote`; live counts only while that PACT change is approved
          proposer: {kind: stub, fixture: {...}}   # see archi.ops.proposers.StubProposer
          outcome: {kind: marker}                  # the outcome check (stub for now)
          test_gate_seconds: 0    # TEST ONLY: wait this long for a release message just before the mediator step
          queues: [Cluster Ops]   # ticket_queue triggers: only tickets in these queues (project, or ServiceNow
                                  # assignment group); unset means every ticket the instance ingests

Environment: ARCHI_INSTANCE (the instance directory), OKG_DSN (the graph
database), ARCHI_OPS_DEPLOYMENT (the okg deployment name; default: the
instance's name).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from archi.ops.loop import AutomationConfig, load_automation

OPS_FILE = "ops.yaml"
INSTANCE_FILE = "archi.yaml"
SCANNER = "archi_ops_scan"
RUN_WORKFLOW = "archi_ops_run"


class RuntimeConfigError(ValueError):
    """The instance's ops configuration cannot be used."""


@dataclass(frozen=True)
class AutomationSettings:
    name: str
    mode: str = "shadow"
    timeout_seconds: int = 4 * 3600
    estimate_cost: Optional[float] = None
    #: the PACT change `archi ops promote` opened; `mode: live` counts only while it is approved
    promotion: Optional[str] = None
    #: test-only: the run waits on DBOS topic `archi_ops_test_gate` for this long between the
    #: loop's decision and the mediator step, so a test can change the world in that window
    test_gate_seconds: float = 0.0
    proposer: Mapping[str, Any] = field(default_factory=dict)
    outcome: Mapping[str, Any] = field(default_factory=dict)
    #: a ticket_queue trigger's queues (project, or ServiceNow assignment group); empty means all
    queues: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    instance_dir: Path
    deployment: str
    dsn: str
    settings: Mapping[str, AutomationSettings]

    def automation(self, name: str) -> AutomationConfig:
        """The loop's configuration for `name`. A level that archi.yaml sets wrongly
        (L3, unknown) is passed on as it is, so the run refuses it and records why."""
        if name not in self.settings:
            raise RuntimeConfigError(f"{self.instance_dir / OPS_FILE}: no automations.{name}")
        s = self.settings[name]
        return load_automation(
            self.instance_dir, name, instance_level=raw_levels(self.instance_dir).get(name),
            mode=s.mode, timeout_seconds=s.timeout_seconds, estimate_cost=s.estimate_cost,
            promotion_change=s.promotion,
        )


def raw_levels(instance_dir: Path) -> dict[str, Optional[str]]:
    """`automations.<name>.level` from archi.yaml, unvalidated (None when unset)."""
    path = Path(instance_dir) / INSTANCE_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: (body or {}).get("level") for name, body in (data.get("automations") or {}).items()}


def parse_moment(value: Any) -> Optional[datetime]:
    """An ISO time (or a YAML timestamp) as an aware UTC datetime; naive means UTC."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def read_settings(instance_dir: Path) -> dict[str, AutomationSettings]:
    path = Path(instance_dir) / OPS_FILE
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for name, body in (data.get("automations") or {}).items():
        body = body or {}
        unknown = sorted(set(body) - {"mode", "timeout_seconds", "estimate_cost", "proposer", "outcome",
                                  "promotion", "test_gate_seconds", "queues"})
        if unknown:
            raise RuntimeConfigError(f"{path}: automations.{name}: unknown keys {unknown}")
        queues = body.get("queues") or []
        if not isinstance(queues, list) or not all(isinstance(q, str) and q for q in queues):
            raise RuntimeConfigError(f"{path}: automations.{name}.queues must be a list of queue names")
        out[name] = AutomationSettings(
            name=name,
            mode=body.get("mode", "shadow"),
            timeout_seconds=int(body.get("timeout_seconds", 4 * 3600)),
            estimate_cost=None if body.get("estimate_cost") is None else float(body["estimate_cost"]),
            promotion=body.get("promotion") or None,
            test_gate_seconds=float(body.get("test_gate_seconds", 0.0)),
            proposer=dict(body.get("proposer") or {}),
            outcome=dict(body.get("outcome") or {}),
            queues=tuple(queues),
        )
    return out


def load_runtime(env: Optional[Mapping[str, str]] = None) -> RuntimeConfig:
    """The runtime configuration from the environment and the instance directory."""
    env = os.environ if env is None else env
    if not env.get("ARCHI_INSTANCE"):
        raise RuntimeConfigError("ARCHI_INSTANCE is not set (the instance directory)")
    if not env.get("OKG_DSN"):
        raise RuntimeConfigError("OKG_DSN is not set (the graph database)")
    instance_dir = Path(env["ARCHI_INSTANCE"]).resolve()
    deployment = env.get("ARCHI_OPS_DEPLOYMENT")
    if not deployment:
        data = yaml.safe_load((instance_dir / INSTANCE_FILE).read_text(encoding="utf-8")) or {}
        deployment = data.get("name")
    if not deployment:
        raise RuntimeConfigError("no deployment name: set ARCHI_OPS_DEPLOYMENT or name the instance")
    return RuntimeConfig(instance_dir, deployment, env["OKG_DSN"], read_settings(instance_dir))
