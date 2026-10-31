"""req.r3-ops-runtime.loop: the operations loop at L0, L1 and L2 with fakes for
every dependency (acc.r3-ops-runtime.timeout, acc.r3-ops-runtime.no-l3 at run
start). Each test asserts the exact ops records written and whether the
mediator was called."""
from datetime import datetime, timedelta

import pytest

from archi.ops.escalation import Decision, Escalation
from archi.ops.loop import (
    BudgetExceeded,
    Demotion,
    DEMOTED_PREFIX as _PREFIX,
    parse_run_id,
    persisted_demotion_reason,
    MediationRefused,
    DEMOTED_PREFIX,
    AutomationConfig,
    Deps,
    MediatorResult,
    Signal,
    make_run_id,
    run_once,
    validate_proposal,
)
from archi.ops.proposers import StubProposer
from loop_fakes import (
    Demotions,
    Promotions,
    FAIL,
    GEN,
    PASS,
    T0,
    Budget,
    Clock,
    Decider,
    Gate,
    Halt,
    Mediator,
    Outcome,
    Reader,
    Recorder,
    at,
    iso,
)

TIMEOUT = 600
RUN = make_run_id("triage", "sig-1", 1)
SIGNAL = Signal(id="sig-1", entity="T2_US_MIT", title="Site down", source="ticket")
PROPOSAL = {
    "summary": "restart the agent",
    "evidence_ids": ["jira:A-1"],
    "action_class": "restart_service",
    "payload": {"service": "xrootd"},
    "confidence": 0.9,
    "rollback": "start it again",
}
DRAFT_PROPOSAL = {
    "summary": "reply to the site", "evidence_ids": ["jira:A-1"], "action_class": "draft_reply",
    "payload": {"ticket_key": "A-1", "body": "We restarted it."}, "draft": "We restarted it.",
}
# The shape of a rendered automation block (compare the fixture-triage render).
BLOCK = {
    "pack": "triage",
    "default_level": "L1",
    "pack_max_level": "L2",
    "bundle_max_level": None,
    "context_gate": {"sources": ["s"]},
    "proposal_schema": {
        "type": "object",
        "required": ["summary", "evidence_ids"],
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string", "minLength": 1},
            "evidence_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "action_class": {"type": "string"},
            "payload": {"type": "object"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rollback": {"type": "string"},
            "draft": {"type": "string"},
        },
    },
    "budget": {"cost_per_run": 0.5, "cost_per_day": 10},
    "action_class_definitions": {
        "restart_service": {
            "max_level": "L2", "mediator": "slurm.restart",
            "payload_schema": {"type": "object", "required": ["service"], "additionalProperties": False,
                               "properties": {"service": {"type": "string"}}},
        },
        "draft_reply": {
            "max_level": "L1", "mediator": "draft_reply",
            "payload_schema": {"type": "object", "required": ["ticket_key", "body"],
                               "properties": {"ticket_key": {"type": "string"}, "body": {"type": "string"}}},
        },
    },
}


def config(level="L2", mode="live", **over):
    return AutomationConfig.from_block(BLOCK, instance_level=level, mode=mode, timeout_seconds=TIMEOUT, **over)


class World:
    def __init__(self, proposal=PROPOSAL, gate=PASS, decision=None, after=60, halt=(False,), budget=None,
                 cost=0.1, clock=None, gate_raises=None, pin_raises=None, nodes=None, demotions=None, promotions=None, **decider):
        self.clock = clock or Clock()
        self.gate, self.reader = Gate(gate, gate_raises), Reader(pin_raises, **({"nodes": nodes} if nodes else {}))
        self.proposer = StubProposer({"default": proposal, "cost": cost})
        self.decider = Decider(self.clock, decision, after, **decider)
        self.mediator, self.outcome = Mediator(), Outcome()
        self.recorder, self.halt = Recorder(), Halt(*halt)
        self.budget = budget or Budget()
        self.demotions = demotions
        self.promotions = promotions

    def run(self, automation, run_id=RUN):
        deps = Deps(self.gate, self.reader, self.proposer, self.decider, self.mediator, self.outcome,
                    self.recorder, self.halt, self.clock, self.budget, self.demotions, self.promotions)
        return run_once(automation, SIGNAL, deps, run_id=run_id)

    @property
    def kinds(self):
        return [c[0] for c in self.recorder.calls]


# --- expected records ----------------------------------------------------------------

COMMON = {"automation": "triage", "actor_kind": "automation", "actor": "triage", "generation_read": GEN}


def rec(kind, *, evidence=(), **attrs):
    return (kind, list(evidence), {**COMMON, **attrs})


def incident(**over):
    return rec("incident", **{**dict(record_id=f"inc-{RUN}", title="Site down", entity="T2_US_MIT",
                                     status="open", detected_at=iso(), signal="ticket:sig-1"), **over})


def run_level_incident(reason):
    kind, ev, attrs = incident(summary=f"run not started: {reason}")
    del attrs["generation_read"]
    return kind, ev, attrs


def proposal(level="L2", mode="live", at_=0, **over):
    attrs = dict(record_id=f"prop-{RUN}", incident_id=f"inc-{RUN}", summary="restart the agent",
                 action_class="restart_service", payload='{"service": "xrootd"}', confidence=0.9,
                 rollback="start it again", autonomy_level=level, mode=mode, proposed_at=iso(at_))
    attrs.update(over)
    return rec("proposal", evidence=["jira:A-1"], **attrs)


def placeholder(reason, level="L2", mode="live", note_extra=""):
    return rec("proposal", evidence=[f"ops_incident:inc-{RUN}"], record_id=f"prop-{RUN}",
               incident_id=f"inc-{RUN}", summary=f"no proposal: {reason}", autonomy_level=level,
               mode="placeholder", proposed_at=iso(),
               mode_reason=f"placeholder: no proposal was made ({reason}); run mode {mode}{note_extra}")


def decision(verdict, reason, at_=0, **over):
    attrs = dict(record_id=f"dec-{RUN}", proposal_id=f"prop-{RUN}", verdict=verdict, reason=reason,
                 decided_at=iso(at_))
    attrs.update(over)
    return rec("decision", **attrs)


def approved(**over):
    base = dict(status="approved", proposal_id=f"prop-{RUN}", escalation_id="esc-1", decided_by="alice",
                decided_at=at(60))
    return Decision(**{**base, **over})


ALICE = {"actor_kind": "person", "actor": "alice", "escalation_id": "esc-1"}


def action(at_):
    return rec("action", record_id=f"act-{RUN}", proposal_id=f"prop-{RUN}", decision_id=f"dec-{RUN}",
               action_class="restart_service", mediator="slurm.restart", payload='{"service": "xrootd"}',
               status="succeeded", result="restarted", executed_at=iso(at_))


def outcome(at_):
    return rec("outcome", record_id=f"out-{RUN}", incident_id=f"inc-{RUN}", action_id=f"act-{RUN}",
               result="resolved", resolution="service up", check_method="outcome_check", checked_at=iso(at_))


# --- L0, L1 --------------------------------------------------------------------------


def test_l0_records_observed_and_never_acts():
    w = World()
    result = w.run(config("L0"))
    assert result.status == "observed" and not result.mediated
    assert w.recorder.calls == [
        incident(),
        proposal("L0"),
        decision("observed", "L0: observed only"),
    ]
    assert w.decider.calls == [] and w.mediator.calls == []
    assert w.reader.pins == 1 and w.reader.reads == [("sig-1", GEN)]
    assert w.recorder.records.dangling() == []


def test_l1_records_the_draft_and_never_sends():
    w = World(proposal=DRAFT_PROPOSAL)
    result = w.run(config("L1"))
    assert result.status == "drafted" and not result.mediated
    assert w.recorder.calls == [
        incident(),
        rec("proposal", evidence=["jira:A-1"], record_id=f"prop-{RUN}", incident_id=f"inc-{RUN}",
            summary="reply to the site", action_class="draft_reply",
            payload='{"body": "We restarted it.", "ticket_key": "A-1"}', draft="We restarted it.",
            autonomy_level="L1", mode="live", proposed_at=iso()),
        decision("drafted", "L1: draft recorded for a person to send; nothing was sent"),
    ]
    assert w.decider.calls == [] and w.mediator.calls == []


def test_l1_proposal_with_neither_draft_nor_payload_is_refused():
    w = World(proposal={"summary": "s", "evidence_ids": ["jira:A-1"]})
    result = w.run(config("L1"))
    assert result.status == "invalid_proposal" and "required at L1" in result.reason
    assert w.mediator.calls == []


# --- L2 ------------------------------------------------------------------------------


def test_l2_approved_runs_the_mediator_then_checks_the_outcome():
    w = World(decision=approved())
    result = w.run(config())
    assert (result.status, result.mediated, result.outcome) == ("executed", True, "resolved")
    assert w.recorder.calls == [
        incident(),
        proposal(),
        decision("approved", "approved by escalation", at_=60, **ALICE),
        action(60),
        outcome(60),
    ]
    assert w.mediator.calls == [{
        "action_class": "restart_service", "payload": {"service": "xrootd"}, "proposal_id": f"prop-{RUN}",
        "idempotency_key": RUN, "generation": GEN,
        "escalation": Escalation("esc-1", f"prop-{RUN}", at(0), at(TIMEOUT)), "level": "L2",
    }]
    assert w.decider.opened[0]["timeout_at"] == at(TIMEOUT)
    assert w.halt.calls == 2  # at the start, and as the first statement of the mediator step
    assert len(result.records) == 5 and w.recorder.records.dangling() == []


def test_l2_rejected_never_reaches_the_mediator():
    w = World(decision=approved(status="rejected", note="wrong site"))
    result = w.run(config())
    assert result.status == "rejected" and not result.mediated
    assert w.recorder.calls == [
        incident(), proposal(),
        decision("rejected", "rejected by escalation; note: wrong site", at_=60, **ALICE),
    ]
    assert w.mediator.calls == [] and w.outcome.calls == []


def test_l2_timeout_provably_does_not_execute():
    w = World(decision=Decision("timed_out", proposal_id=f"prop-{RUN}", escalation_id="esc-1"), after=TIMEOUT)
    result = w.run(config())
    assert result.status == "timed_out" and not result.mediated
    assert w.recorder.calls == [
        incident(), proposal(),
        decision("timed_out", "no decision before the deadline", at_=TIMEOUT, escalation_id="esc-1"),
    ]
    assert w.mediator.calls == [] and w.outcome.calls == [] and "action" not in w.kinds
    assert w.halt.calls == 1


def test_l2_late_approval_does_not_execute():
    w = World(decision=approved(decided_at=at(TIMEOUT + 1)), after=TIMEOUT + 100)
    result = w.run(config())
    assert result.status == "timed_out" and not result.mediated
    reason = (f"not executed: approved at {iso(TIMEOUT + 1).replace('Z', '+00:00')}, "
              f"after the deadline {iso(TIMEOUT).replace('Z', '+00:00')}")
    assert w.recorder.calls == [
        incident(), proposal(),
        decision("timed_out", reason, at_=TIMEOUT + 1, **ALICE),
    ]
    assert w.mediator.calls == [] and w.halt.calls == 1


@pytest.mark.parametrize("decision_over, fragment, extra_args", [
    ({"proposal_id": "prop-other"}, "decision is for proposal 'prop-other'", {}),
    ({"proposal_id": None}, "decision is for proposal None", {}),
    ({"escalation_id": "esc-9"}, "decision is for escalation 'esc-9', not 'esc-1'", {}),
    ({"escalation_id": None}, "decision is for escalation None", {}),
    ({"decided_by": None}, "names no person", {}),
    ({"decided_by": ""}, "names no person", {}),
    ({"decided_at": None}, "no decision time", {}),
    ({"decided_at": at(-5)}, "before the escalation opened", {}),
    ({"decided_at": datetime(2026, 10, 25, 12, 1)}, "naive", {}),
])
def test_an_approval_not_bound_to_this_escalation_does_not_execute(decision_over, fragment, extra_args):
    w = World(decision=approved(**decision_over))
    result = w.run(config())
    assert result.status == "error" and not result.mediated
    kind, _, attrs = w.recorder.calls[-1]
    assert (kind, attrs["verdict"]) == ("decision", "error")
    assert attrs["reason"].startswith("not executed: ") and fragment in attrs["reason"]
    assert w.mediator.calls == [] and "action" not in w.kinds and w.halt.calls == 1


def test_a_decision_for_an_escalation_opened_for_another_proposal_is_an_error():
    w = World(decision=approved(), open_proposal_id="prop-other")
    result = w.run(config())
    assert result.status == "error" and w.mediator.calls == []
    assert "opened for 'prop-other'" in w.recorder.calls[-1][2]["reason"]


def test_escalation_that_cannot_be_opened_does_not_execute():
    w = World(decision=approved(), open_raises=RuntimeError("db down"))
    result = w.run(config())
    assert result.status == "error"
    assert w.recorder.calls[-1] == decision("error", "escalation failed: RuntimeError: db down")
    assert w.mediator.calls == []


def test_escalation_error_does_not_execute():
    w = World(decision=Decision("error", error="boom"))
    result = w.run(config())
    assert result.status == "error" and w.mediator.calls == []
    assert w.recorder.calls[-1] == decision("error", "escalation failed: boom", at_=60, escalation_id="esc-1")


def test_halt_before_the_mediator_writes_one_halted_decision_that_supersedes_the_approval():
    w = World(decision=approved(), halt=(False, True))
    result = w.run(config())
    reason = ("halted after approval, before the mediator ran; "
              "supersedes the approval by alice (esc-1)")
    assert result.status == "halted" and not result.mediated
    assert w.recorder.calls == [
        incident(), proposal(),
        decision("approved", "approved by escalation", at_=60, **ALICE),
        decision("halted", reason, at_=60, **ALICE),  # same record id: replaces the approval
    ]
    assert w.mediator.calls == [] and "action" not in w.kinds
    # in the written batch only the halted decision is left
    nodes = w.recorder.records.nodes
    assert nodes[f"ops_decision:dec-{RUN}"]["attrs"]["verdict"] == "halted"


def test_halt_check_raising_before_the_mediator_fails_closed():
    w = World(decision=approved(), halt=(False, RuntimeError("no db")))
    result = w.run(config())
    assert result.status == "halted" and "failing closed" in result.reason and "no db" in result.reason
    assert w.mediator.calls == []
    assert w.recorder.calls[-1][2]["verdict"] == "halted"


def test_halt_at_the_start_writes_nothing_and_returns_a_structured_halt():
    w = World(decision=approved(), halt=(True,))
    result = w.run(config())
    assert (result.status, result.records, result.reason) == ("halted", (), "halted before the run started")
    assert w.recorder.calls == [] and w.gate.calls == 0 and w.reader.pins == 0


def test_mediator_failure_is_recorded_and_the_outcome_still_checked():
    w = World(decision=approved())
    w.mediator = Mediator(MediatorResult("failed", "exit 1"))
    result = w.run(config())
    assert result.status == "executed"
    kind, _, attrs = w.recorder.calls[3]
    assert (kind, attrs["status"], attrs["result"]) == ("action", "failed", "exit 1")


# --- levels (acc.r3-ops-runtime.no-l3, run-start half) -------------------------------


@pytest.mark.parametrize("over", [
    {"instance_level": "L3"},
    {"bundle_max_level": "L3"},
    {"pack_max_level": "L3"},
    {"default_level": "L3", "instance_level": None},
    {"instance_level": "L9"},
    {"instance_level": "act"},
    {"instance_level": "l2"},        # lowercase is not accepted
    {"instance_level": "L²"},   # superscript two
    {"instance_level": "L٢"},   # Arabic-indic two
    {"instance_level": "L２"},   # fullwidth two
    {"instance_level": 2},
])
def test_level_above_l2_or_unknown_is_refused_at_run_start(over):
    from dataclasses import replace

    w = World(decision=approved())
    result = w.run(replace(config(), **over))
    assert result.status == "refused" and "ADR 0002 D13" in result.reason
    # one run-level incident, no generation, and nothing was read, proposed or run
    assert w.recorder.calls == [run_level_incident(result.reason)]
    assert w.gate.calls == 0 and w.reader.pins == 0 and w.proposer.calls == [] and w.mediator.calls == []


def test_an_action_class_above_l2_is_refused_at_run_start():
    from dataclasses import replace
    from archi.ops.loop import ActionClassConfig

    bad = replace(config(), action_classes={"restart_service": ActionClassConfig("L3")})
    w = World()
    result = w.run(bad)
    assert result.status == "refused" and "restart_service" in result.reason
    assert w.mediator.calls == []


def test_effective_level_is_the_minimum_of_instance_bundle_and_pack():
    w = World(proposal=DRAFT_PROPOSAL)
    bounded = AutomationConfig.from_block({**BLOCK, "bundle_max_level": "L1"}, instance_level="L2",
                                          mode="live", timeout_seconds=TIMEOUT)
    result = w.run(bounded)
    assert (result.level, result.status) == ("L1", "drafted")


def test_the_pack_default_stands_in_when_the_instance_sets_no_level():
    w = World(proposal=DRAFT_PROPOSAL)
    assert w.run(config(None)).level == "L1"


def test_an_action_class_capped_below_the_run_level_never_executes():
    # draft_reply allows L1; the run is L2 and the approval would otherwise run it.
    w = World(proposal=DRAFT_PROPOSAL, decision=approved())
    result = w.run(config("L2"))
    reason = "action class 'draft_reply' allows at most L1, below the run's L2; it is never executed"
    assert result.status == "action_class_capped" and not result.mediated
    assert w.recorder.calls[-1] == decision("action_class_capped", reason)
    assert "action" not in w.kinds
    assert w.decider.calls == [] and w.mediator.calls == []


# --- gate, shadow, fail closed -------------------------------------------------------


def test_failing_gate_drops_a_live_run_to_shadow_with_the_reason_recorded():
    w = World(decision=approved(), gate=FAIL)
    result = w.run(config())
    reason = DEMOTED_PREFIX + FAIL.message
    assert result.status == "observed" and result.mode == "shadow" and not result.mediated
    assert w.recorder.calls == [
        incident(),
        proposal(mode="shadow", mode_reason=reason),
        decision("observed", "shadow: recorded only; a live run at L2 would escalate for approval and then act"),
    ]
    assert "freshness:s" in reason
    assert w.decider.calls == [] and w.mediator.calls == []


def test_shadow_never_mediates_even_with_a_passing_gate():
    w = World(decision=approved())
    result = w.run(config(mode="shadow"))
    assert result.status == "observed" and result.mode == "shadow"
    assert w.recorder.calls == [
        incident(),
        proposal(mode="shadow"),
        decision("observed", "shadow: recorded only; a live run at L2 would escalate for approval and then act"),
    ]
    assert w.decider.calls == [] and w.mediator.calls == [] and w.outcome.calls == []


def test_shadow_with_a_failing_gate_keeps_the_reason_without_claiming_a_demotion():
    w = World(gate=FAIL)
    w.run(config("L0", mode="shadow"))
    assert w.recorder.calls[1][2]["mode_reason"] == "context gate failed: " + FAIL.message


def test_gate_and_pin_on_different_generations_demote_to_shadow():
    from archi.ops.context_gate import GateResult

    w = World(gate=GateResult("gen-6", PASS.checks), decision=approved())
    result = w.run(config())
    assert result.mode == "shadow" and w.mediator.calls == []
    reason = w.recorder.calls[1][2]["mode_reason"]
    assert "gen-6" in reason and GEN in reason


@pytest.mark.parametrize("kwargs", [{"gate_raises": RuntimeError("gate down")},
                                    {"pin_raises": RuntimeError("no generation")},
                                    {"halt": (RuntimeError("halt table gone"),)}])
def test_exceptions_before_anything_is_read_fail_closed_with_a_run_level_record(kwargs):
    w = World(decision=approved(), **kwargs)
    result = w.run(config())
    assert result.status == "error"
    assert w.recorder.calls == [run_level_incident(result.reason)]
    assert w.proposer.calls == [] and w.mediator.calls == []


def test_a_budget_exception_fails_closed_with_an_error_decision():
    w = World(decision=approved(), budget=Budget(raises=RuntimeError("ledger down")))
    result = w.run(config())
    reason = "RuntimeError: ledger down"
    assert result.status == "error"
    assert w.recorder.calls == [incident(), placeholder(reason), decision("error", reason)]
    assert w.proposer.calls == [] and w.mediator.calls == []


def test_a_naive_clock_is_refused_not_compared():
    w = World(decision=approved(), clock=Clock(datetime(2026, 10, 25, 12, 0)))
    result = w.run(config())
    assert result.status == "error" and "naive" in result.reason
    assert w.mediator.calls == [] and w.kinds == ["incident"]
    assert w.recorder.calls[0][2]["detected_at"] == "2026-10-25T12:00:00"  # visibly without a Z


def test_proposer_failure_is_recorded_as_an_error():
    w = World()
    w.proposer = StubProposer({"fail": {"sig-1": "model timeout"}})
    result = w.run(config())
    reason = "proposer failed: ProposerError: model timeout"
    assert result.status == "error"
    assert w.recorder.calls[-2:] == [placeholder(reason), decision("error", reason)]
    assert w.mediator.calls == []


# --- the run id ----------------------------------------------------------------------


def test_run_id_is_deterministic_and_comes_from_the_caller():
    assert make_run_id("triage", "sig-1", 3) == "triage~sig-1~3"
    first, second = World(), World(clock=Clock(at(5000)))
    first.run(config("L0"), run_id="triage~sig-1~1")
    second.run(config("L0"), run_id="triage~sig-1~1")
    ids = lambda w: [c[2]["record_id"] for c in w.recorder.calls]  # noqa: E731
    assert ids(first) == ids(second) == ["inc-triage~sig-1~1", "prop-triage~sig-1~1", "dec-triage~sig-1~1"]
    other = World()
    other.run(config("L0"), run_id="triage~sig-1~2")
    assert ids(other)[0] == "inc-triage~sig-1~2"


# --- budget --------------------------------------------------------------------------

RUN_SCOPE, DAY_SCOPE = f"run:{RUN}", "day:triage:2026-10-25"


def test_budget_is_checked_per_run_and_per_day_before_proposing_and_charged_after():
    w = World(decision=approved(), cost=0.1)
    w.run(config())
    # the estimate defaults to the pack's per-run budget
    assert w.budget.checks[:2] == [(RUN_SCOPE, 0.5, 0.5), (DAY_SCOPE, 0.5, 10.0)]
    assert w.budget.recorded == [(RUN_SCOPE, 0.1), (DAY_SCOPE, 0.1)]


def test_exhausted_day_budget_records_a_decision_and_stops_before_proposing():
    reason = "day budget exhausted: 9.8 of 10 spent"
    w = World(decision=approved(), budget=Budget({DAY_SCOPE: 9.8}))
    result = w.run(config(estimate_cost=0.4))
    assert result.status == "budget_exhausted"
    assert w.recorder.calls == [incident(), placeholder(reason), decision("budget_exhausted", reason)]
    assert w.proposer.calls == [] and w.decider.calls == [] and w.mediator.calls == []


def test_per_run_cap_used_up_by_the_proposal_stops_before_escalation_and_mediator():
    w = World(decision=approved(), cost=0.8)  # the run cap is 0.5
    result = w.run(config())
    reason = "run budget exhausted: 0.8 of 0.5 spent"
    assert result.status == "budget_exhausted"
    assert w.recorder.calls == [incident(), proposal(), decision("budget_exhausted", reason)]
    assert w.decider.calls == [] and w.mediator.calls == []


# --- the proposal --------------------------------------------------------------------


def test_proposal_without_evidence_ids_is_refused():
    bare = {k: v for k, v in PROPOSAL.items() if k != "evidence_ids"}
    w = World(proposal=bare, decision=approved())
    result = w.run(config())
    assert result.status == "invalid_proposal"
    assert "evidence_ids" in result.reason
    assert w.recorder.calls[-1][2]["verdict"] == "invalid_proposal"
    assert w.decider.calls == [] and w.mediator.calls == []


def test_a_cited_id_the_pinned_read_did_not_return_is_refused():
    w = World(proposal={**PROPOSAL, "evidence_ids": ["jira:A-1", "jira:Z-9"]}, decision=approved())
    result = w.run(config())
    reason = "proposal refused: evidence_ids not in the pinned read: ['jira:Z-9']"
    assert result.status == "invalid_proposal" and result.reason == reason
    assert w.recorder.calls[-2:] == [placeholder(reason), decision("invalid_proposal", reason)]
    assert w.mediator.calls == []


@pytest.mark.parametrize("bad, fragment", [
    ({"evidence_ids": []}, "evidence_ids"),
    ({"evidence_ids": [""]}, "evidence_ids"),
    ({"action_class": "drop_database"}, "not declared"),
    ({"action_class": None}, "required at L2"),
    ({"payload": {}}, "payload for 'restart_service': "),
    ({"payload": {"service": "x", "rm": "-rf"}}, "Additional properties"),
    ({"confidence": 3}, "proposal/confidence"),
    ({"surprise": 1}, "Additional properties"),
    ({"summary": ""}, "summary"),
])
def test_proposal_schema_violations_are_refused(bad, fragment):
    w = World(proposal={**PROPOSAL, **bad}, decision=approved())
    result = w.run(config())
    assert result.status == "invalid_proposal" and fragment in result.reason
    assert w.mediator.calls == []


def test_validate_proposal_uses_the_automations_json_schema():
    automation = config()
    ok = validate_proposal(PROPOSAL, automation, "L2", {"jira:A-1"})
    assert ok.action_class == "restart_service" and ok.evidence_ids == ("jira:A-1",)
    with pytest.raises(Exception, match="proposal/confidence"):
        validate_proposal({**PROPOSAL, "confidence": "high"}, automation, "L2")


def test_placeholder_proposals_are_marked_so_shadow_scoring_skips_them():
    w = World(budget=Budget({DAY_SCOPE: 9.8}))
    w.run(config("L0", mode="shadow", estimate_cost=0.4))
    _, _, attrs = w.recorder.calls[1]
    assert attrs["mode"] == "placeholder"  # shadow scoring reads only mode == "shadow"
    assert attrs["mode_reason"].startswith("placeholder: no proposal was made")


# --- durable points and a refusing mediator step -------------------------------------


def test_records_are_flushed_where_a_person_or_the_next_run_must_see_them():
    w = World(decision=approved())
    w.run(config())
    # after the proposal (before the wait), after the decision, after the outcome
    assert w.recorder.flushed_at == [2, 3, 5]


def test_a_stopped_run_flushes_its_placeholder_and_decision_once():
    w = World(budget=Budget({DAY_SCOPE: 9.8}))
    w.run(config(estimate_cost=0.4))
    assert w.recorder.flushed_at == [3]


@pytest.mark.parametrize("verdict", ["halted", "timed_out", "error"])
def test_a_mediator_step_that_refuses_supersedes_the_approval(verdict):
    w = World(decision=approved())
    w.mediator = Mediator(refuse=MediationRefused(verdict, "guard says no"))
    result = w.run(config())
    assert result.status == verdict and not result.mediated
    assert w.recorder.calls[-1] == decision(
        verdict, "guard says no; supersedes the approval by alice (esc-1)", at_=60, **ALICE)
    assert w.recorder.calls[-2][2]["verdict"] == "approved" and "action" not in w.kinds


def test_an_l1_draft_reply_body_is_recorded_as_the_draft():
    reply = {"summary": "reply", "evidence_ids": ["jira:A-1"], "action_class": "draft_reply",
             "payload": {"ticket_key": "A-1", "body": "We restarted it."}}
    w = World(proposal=reply)
    w.run(config("L1"))
    assert w.recorder.calls[1][2]["draft"] == "We restarted it."
    w2 = World(proposal={**reply, "action_class": "restart_service", "payload": {"service": "x"}})
    w2.run(config("L1"))
    assert "draft" not in w2.recorder.calls[1][2]  # no body, no draft


# --- run ids, estimates, enforced caps, unknown actions, persisted demotion ---------------


def test_run_ids_cannot_collide():
    assert make_run_id("a-b", "c", 1) != make_run_id("a", "b-c", 1)  # the old "-" join made these equal
    assert parse_run_id(make_run_id("triage", "sig-1", 3)) == ("triage", "sig-1", 3)
    for bad in (("a~b", "s", 1), ("a", "s~t", 1), ("", "s", 1), ("a", "", 1), ("a", "s", 0), ("a", "s", "x"),
                ("a", "s", "01"), ("a", "s", -1)):
        with pytest.raises(ValueError):
            make_run_id(*bad)
    with pytest.raises(ValueError):
        parse_run_id("no-separators")


def test_estimate_comes_from_the_pack_unless_the_instance_says_otherwise():
    assert config().estimate_cost == 0.5
    assert config(estimate_cost=0.2).estimate_cost == 0.2


def test_a_charge_the_cap_refuses_stops_with_budget_exhausted_after_the_real_proposal():
    w = World(decision=approved(), cost=0.8, budget=Budget(enforce=True))
    result = w.run(config())
    reason = "run budget exhausted: charging 0.8 would pass the cap (the charge was refused)"
    assert result.status == "budget_exhausted"
    assert w.recorder.calls == [incident(), proposal(), decision("budget_exhausted", reason)]
    assert w.decider.calls == [] and w.mediator.calls == []
    assert w.budget.recorded == [(DAY_SCOPE, 0.8)]  # only the scope the cap refused went uncharged


def test_an_unknown_mediator_result_records_outcome_unknown_without_asking_the_checker():
    w = World(decision=approved())
    w.mediator = Mediator(MediatorResult("unknown", "possible duplicate prevented; check the external system"))
    result = w.run(config())
    assert result.status == "executed" and result.outcome == "unknown"
    assert w.outcome.calls == []
    assert w.recorder.calls[-2][2]["status"] == "unknown"
    assert w.recorder.calls[-1] == rec(
        "outcome", record_id=f"out-{RUN}", incident_id=f"inc-{RUN}", action_id=f"act-{RUN}", result="unknown",
        resolution="possible duplicate prevented; check the external system", check_method="outcome_check",
        checked_at=iso(60))


DEMOTED_AT = at(-3600)
EARLIER = Demotion(DEMOTED_AT, DEMOTED_PREFIX + "context gate failed: freshness:s: last run 3d ago")


def test_a_demotion_persists_when_the_gate_passes_again():
    log = Demotions(EARLIER)
    w = World(decision=approved(), demotions=log)
    result = w.run(config())
    reason = persisted_demotion_reason(EARLIER)
    assert result.mode == "shadow" and result.status == "observed" and not result.mediated
    assert reason.startswith(DEMOTED_PREFIX) and "freshness:s: last run 3d ago" in reason
    assert w.recorder.calls[1] == proposal(mode="shadow", mode_reason=reason)
    assert log.calls == [("triage", GEN)]
    assert w.decider.calls == [] and w.mediator.calls == []


def test_a_persisted_reason_does_not_nest_and_keeps_the_original_cause():
    once = persisted_demotion_reason(EARLIER)
    twice = persisted_demotion_reason(Demotion(at(0), once))
    assert twice.count("demotion persists") == 1 and "freshness:s: last run 3d ago" in twice


def test_promotion_after_the_demotion_lifts_it():
    w = World(decision=approved(), demotions=Demotions(EARLIER))
    result = w.run(config(promoted_at=at(-60)))
    assert result.mode == "live" and result.status == "executed"


def test_a_demotion_after_the_promotion_still_counts():
    w = World(decision=approved(), demotions=Demotions(EARLIER))
    assert w.run(config(promoted_at=at(-7200))).mode == "shadow"


def test_without_a_recorded_demotion_a_live_run_stays_live():
    w = World(decision=approved(), demotions=Demotions(None))
    assert w.run(config()).mode == "live"


def test_a_failing_gate_demotes_even_when_a_demotion_is_already_recorded():
    log = Demotions(EARLIER)
    w = World(decision=approved(), gate=FAIL, demotions=log)
    w.run(config())
    assert w.recorder.calls[1][2]["mode_reason"] == DEMOTED_PREFIX + FAIL.message
    assert log.calls == []


def test_a_demotion_takes_effect_before_it_is_published():
    """The first run's demotion is in the synchronous log; a second run starts at once, with nothing published."""
    log = Demotions(None)
    first = World(decision=approved(), gate=FAIL, demotions=log)
    first.run(config())
    assert [(name, run_id) for name, run_id, _, _ in log.recorded] == [("triage", RUN)]
    assert log.recorded[0][3] == DEMOTED_PREFIX + FAIL.message
    second = World(decision=approved(), demotions=log)  # the gate passes now; no published record exists
    result = second.run(config(), run_id=make_run_id("triage", "sig-2", 1))
    assert result.mode == "shadow" and result.status == "observed" and second.decider.calls == []
    assert second.recorder.calls[1][2]["mode_reason"].startswith(DEMOTED_PREFIX + PERSISTS_MARK)
    assert log.recorded and len(log.recorded) == 1  # a persisted shadow run does not write a new demotion


PERSISTS_MARK = "demotion persists"


def test_live_without_an_approved_promotion_runs_in_shadow_and_says_why():
    promotions = Promotions(approved=False, reason="the PACT change promote-x is awaiting_review")
    w = World(decision=approved(), promotions=promotions)
    result = w.run(config(promotion_change="promote-x"))
    reason = "not promoted: the PACT change promote-x is awaiting_review"
    assert result.mode == "shadow" and result.status == "observed" and not result.mediated
    assert w.recorder.calls[1] == proposal(mode="shadow", mode_reason=reason)
    assert promotions.calls == [("triage", "promote-x")]
    assert w.decider.calls == [] and w.mediator.calls == []


def test_an_approved_promotion_makes_live_count():
    w = World(decision=approved(), promotions=Promotions(approved=True, at=at(-60)))
    assert w.run(config(promotion_change="promote-x")).mode == "live"


def test_a_live_run_of_a_promotion_with_no_trial_records_that_there_is_no_shadow_evidence():
    """Board d717: `archi ops promote --no-trial --note <decision>`. The proposal of every live run says so."""
    w = World(decision=approved(), promotions=Promotions(approved=True, at=at(-60), no_trial="board d717"))
    assert w.run(config(promotion_change="promote-x")).mode == "live"
    reason = w.recorder.calls[1][2]["mode_reason"]
    assert reason.startswith("no shadow evidence: ") and "promote-x" in reason and "board d717" in reason


def test_a_shadow_automation_never_asks_about_promotion():
    promotions = Promotions(approved=False)
    World(promotions=promotions).run(config(mode="shadow", level="L0"))
    assert promotions.calls == []


def test_the_approval_time_lifts_an_earlier_demotion_and_a_later_demotion_still_counts():
    earlier = Demotions(Demotion(at(-3600), DEMOTED_PREFIX + "x"))
    assert World(decision=approved(), demotions=earlier, promotions=Promotions(True, at(-60))).run(config()).mode == "live"
    later = Demotions(Demotion(at(-3600), DEMOTED_PREFIX + "x"))
    assert World(decision=approved(), demotions=later, promotions=Promotions(True, at(-7200))).run(config()).mode == "shadow"


def test_an_old_promotion_is_superseded_by_a_later_demotion():
    """Approved last week, demoted yesterday: the approval predates the demotion, so the run stays in shadow
    until a promotion is approved after it."""
    demotions = Demotions(Demotion(at(-3600), DEMOTED_PREFIX + "context gate failed: stale"))
    old_approval = Promotions(approved=True, at=at(-86400))
    w = World(decision=approved(), demotions=demotions, promotions=old_approval)
    result = w.run(config(promotion_change="promote-old"))
    assert result.mode == "shadow" and not result.mediated
    assert w.recorder.calls[1][2]["mode_reason"].startswith(DEMOTED_PREFIX + "demotion persists")
    fresh_approval = Promotions(approved=True, at=at(-60))
    assert World(decision=approved(), demotions=demotions, promotions=fresh_approval).run(
        config(promotion_change="promote-new")).mode == "live"


def test_a_failing_gate_on_an_unpromoted_automation_adds_to_the_reason_and_records_no_demotion():
    log = Demotions(None)
    w = World(gate=FAIL, demotions=log, promotions=Promotions(False, reason="no promotion"))
    w.run(config())
    assert w.recorder.calls[1][2]["mode_reason"] == "not promoted: no promotion; context gate failed: " + FAIL.message
    assert log.recorded == []
