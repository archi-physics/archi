"""Proposers: the agent step of the loop (ADR 0002 D11, agents propose).

A proposer reads the pinned context and returns a raw proposal mapping:
`summary`, `evidence_ids` (okg node ids it cites, taken from the pinned read),
and as the automation's proposal schema allows `action_class`, `payload`,
`confidence`, `rollback` and `draft`. The loop
validates the mapping; a proposer holds no execution credential.

`StubProposer` returns fixed answers from a fixture, for tests and scratch
runs. The model-backed proposer comes later.

An instance names its proposer in `ops.yaml` (`automations.<name>.proposer.kind`). `PROPOSER_KINDS`
maps each kind to the `module:function` that builds it (`build(config, runtime) -> Proposer`):

    stub             fixed answers from `proposer.fixture`
    instance-health  the algorithmic health monitor (archi.health.proposer); no model call
    ticket-triage    the algorithmic ticket triage (archi.triage.proposer); no model call
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProposeRequest:
    automation: str
    signal: Mapping[str, Any]
    context: Mapping[str, Any]
    generation: str
    #: 0, 1 or 2
    level: int


@dataclass(frozen=True)
class Proposed:
    """A raw proposal and what the model call cost (charged to the budget)."""

    proposal: Mapping[str, Any]
    cost: float = 0.0


@runtime_checkable
class Proposer(Protocol):
    def propose(self, request: ProposeRequest) -> Proposed: ...


class ProposerError(RuntimeError):
    """The proposer could not produce a proposal."""


#: kind -> "module:function" of a builder `(config, runtime_config) -> Proposer`
PROPOSER_KINDS = {
    "stub": "archi.ops.proposers:build_stub",
    "instance-health": "archi.health.proposer:build",
    "ticket-triage": "archi.triage.proposer:build",
}


def build_proposer(config: Mapping[str, Any], runtime: Any) -> "Proposer":
    """The proposer an automation's `ops.yaml` block names."""
    import importlib

    kind = config.get("kind")
    entry = PROPOSER_KINDS.get(kind)
    if entry is None:
        raise ProposerError(f"proposer kind {kind!r} is not available (known: {sorted(PROPOSER_KINDS)})")
    module, _, name = entry.partition(":")
    return getattr(importlib.import_module(module), name)(config, runtime)


def build_stub(config: Mapping[str, Any], runtime: Any) -> "Proposer":
    return StubProposer(config.get("fixture") or {})


@dataclass
class StubProposer:
    """Deterministic proposer.

    fixture = {
        "by_signal": {"sig-1": {...proposal...}},   # looked up by signal id
        "default": {...proposal...},                # used when no signal matches
        "fail": {"sig-2": "model timeout"},         # raise ProposerError for these
        "cost": 0.1,                                # reported cost per call
    }
    """

    fixture: Mapping[str, Any]
    calls: list[ProposeRequest] = field(default_factory=list)

    def propose(self, request: ProposeRequest) -> Proposed:
        self.calls.append(request)
        signal_id = request.signal.get("id")
        cost = float(self.fixture.get("cost", 0.0))
        failures = self.fixture.get("fail") or {}
        if signal_id in failures:
            raise ProposerError(failures[signal_id])
        by_signal = self.fixture.get("by_signal") or {}
        if signal_id in by_signal:
            return Proposed(copy.deepcopy(by_signal[signal_id]), cost)
        if "default" in self.fixture:
            return Proposed(copy.deepcopy(self.fixture["default"]), cost)
        raise ProposerError(f"fixture has no proposal for signal {signal_id!r}")
