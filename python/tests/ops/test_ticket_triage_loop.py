"""acc.r3-auto-ticket-triage.fixture, in process: a fixture queue through the real loop.

The pieces are the shipped ones: the automation block rendered from the pack (cluster-manager with
servicenow and jira), the scanner (`archi.triage.signals`), the read (`archi.triage.context`), the
proposer, the loop (`archi.ops.loop.run_once`), the mediator step's guard (`archi.ops.workflow.run_mediation`)
and the live tools with their own approval check (`archi.tools._live.ops_verifier`, reading the escalation
row and its `Action:` line). Only the graph, the ticket system, the clock and the person deciding are
fakes. The same queue runs on a real okg worker in scratch_ticket_triage_run.sh.
"""
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

pytest.importorskip("okg.deployment")
pytest.importorskip("dbos")

from archi.bundles import discovery  # noqa: E402
from archi.bundles.lock import render_and_lock  # noqa: E402
from archi.ops import workflow as wf  # noqa: E402
from archi.ops.context_gate import GateResult  # noqa: E402
from archi.ops.escalation import Escalation, decision_from_row, subject_for  # noqa: E402
from archi.ops.loop import Deps, MediationRefused, MediatorResult, ReadResult, Signal, load_automation, make_run_id, run_once  # noqa: E402
from archi.ops.runtime_config import read_settings  # noqa: E402
from archi.tools import tickets as tools  # noqa: E402
from archi.tools._live import ops_verifier  # noqa: E402
from archi.triage.context import read_ticket_context  # noqa: E402
from archi.triage.proposer import TriageProposer  # noqa: E402
from archi.triage.signals import scan_queue  # noqa: E402
from loop_fakes import PASS, Budget, Clock, Gate, Halt, Outcome, Recorder, at  # noqa: E402
from ticket_fakes import TOKEN, FakeTicketSystem, Graph, arrival, queue_rows, to_row  # noqa: E402

NAME = "ticket-triage"
REPO = Path(__file__).resolve().parents[3]
NOW = "2026-10-25T12:00:00Z"
URL = "https://redmine.example.org"
CONFIG = {"doc_subtypes": ["servicenow_kb_article"],
          "systems": {"redmine": {"url": URL, "token_env": "RM_TOKEN", "triaged_status": "In Progress",
                                  "triage_from": ["New"]}}}


@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    root = tmp_path_factory.mktemp("triage")
    catalog = discovery.from_roots([REPO / "bundles" / "packs"], [REPO / "bundles"])
    assert catalog.problems == []
    render_and_lock(catalog, root / "inst", ["cluster-manager"], with_packs=["servicenow", "jira", NAME])
    # the live tools send only to an endpoint the instance configures (ops.yaml proposer.systems)
    ops = root / "inst" / "ops.yaml"
    data = (yaml.safe_load(ops.read_text()) if ops.is_file() else None) or {}
    block = data.setdefault("automations", {}).setdefault(NAME, {})
    block["proposer"] = {"kind": "ticket-triage", **CONFIG}
    block["mode"] = "live"  # what the mediator step reads again (the rollback test sets it back to shadow)
    ops.write_text(yaml.safe_dump(data))
    return root / "inst"


class GraphReader:
    def __init__(self, graph):
        self.graph = graph

    def pin(self):
        return self.graph.generation

    def read(self, signal, generation):
        d = read_ticket_context(self.graph, signal.entity, generation, CONFIG)
        return ReadResult(d["context"], frozenset(d["node_ids"]))


class EscalationIO:
    """The escalation row as okg would hold it, and the mediation attempts (what run_mediation and the
    tool's own check read)."""

    def __init__(self):
        self.row, self.attempts = None, {}

    def escalation(self, escalation_id):
        return self.row if self.row and self.row[1]["id"] == escalation_id else None

    def schedules(self):
        return {"archi_ops_scan": False}

    def decision_for(self, escalation_id):
        found = self.escalation(escalation_id)
        return decision_from_row(escalation_id, *(found or (None, None)))

    def mediation(self, key):
        return self.attempts.get(key)

    def begin_mediation(self, key, proposal_id):
        if key in self.attempts:
            return self.attempts[key]
        self.attempts[key] = {"status": "attempted", "result": None}

    def finish_mediation(self, key, status, result):
        self.attempts[key] = {"status": status, "result": result}


class Person:
    """Opens the escalation the way the workflow does (its id, its reason with the Action line) and answers
    `verdict` (approve, reject or None: nobody answers before the deadline)."""

    def __init__(self, clock, esc_io, run_id, verdict, on_wait=None):
        self.clock, self.io, self.run_id, self.verdict, self.opened = clock, esc_io, run_id, verdict, []
        self.on_wait = on_wait

    def open(self, *, automation, proposal_id, proposal, timeout_at):
        eid = wf.expected_escalation_id(self.run_id)
        self.io.row = (subject_for(proposal_id), {
            "id": eid, "status": "pending", "requested_by_workflow": self.run_id, "timeout_at": timeout_at.isoformat(),
            "reason": wf.escalation_reason(automation, proposal)})
        self.opened.append(proposal)
        return Escalation(eid, proposal_id, self.clock.now(), timeout_at)

    def wait(self, escalation):
        if self.on_wait:
            self.on_wait()
        attrs = self.io.row[1]
        if self.verdict is None:
            self.clock.t = escalation.timeout_at + timedelta(seconds=1)
            attrs["status"] = "timed_out"
        else:
            self.clock.t = escalation.opened_at + timedelta(seconds=60)
            attrs.update(status="decided", decision={"verdict": self.verdict}, decided_by="operator-x",
                         decided_at=self.clock.t.isoformat())
        return decision_from_row(escalation.escalation_id, *self.io.row)


class ToolMediator:
    """The mediator step as the worker runs it: run_mediation's guard, then the real live tool, whose own
    approval check reads the same escalation row."""

    def __init__(self, auto, clock, esc_io, system, run_id, instance_dir):
        self.auto, self.clock, self.io, self.system, self.run_id = auto, clock, esc_io, system, run_id
        self.instance_dir = str(instance_dir)
        self.calls = []

    def execute(self, *, action_class, payload, proposal_id, idempotency_key, generation, escalation=None, level="L2"):
        def tool(entry):
            fn = {"archi.tools.tickets:assign_ticket": tools.assign_ticket,
                  "archi.tools.tickets:set_ticket_status": tools.set_ticket_status}[entry]
            verify = lambda c, p, a: ops_verifier(c, p, a, io=self.io, now=self.clock.now, halt=lambda io: None)  # noqa: E731
            return lambda *, payload, context: fn(payload=payload, context=context, session=self.system.session(),
                                                  env={"RM_TOKEN": TOKEN}, verify=verify)

        esc = None if escalation is None else {
            "escalation_id": escalation.escalation_id, "proposal_id": escalation.proposal_id,
            "opened_at": escalation.opened_at.isoformat(), "timeout_at": escalation.timeout_at.isoformat()}
        self.calls.append(action_class)
        out = wf.run_mediation(
            self.io, mediator_entry=lambda: self.auto.action_classes[action_class].mediator, action_class=action_class,
            payload=payload, proposal_id=proposal_id, idempotency_key=idempotency_key, esc=esc,
            workflow_id=self.run_id, dsn="postgresql://fixture", halt=lambda io: None, now=self.clock.now,
            resolve=tool, refused=lambda verdict, reason: None, instance_dir=self.instance_dir, at_l1=level == "L1",
            mode_now=lambda: read_settings(Path(self.instance_dir))[NAME].mode)
        if "refused" in out:
            raise MediationRefused(*out["refused"])
        return MediatorResult(out["status"], out["result"])


class World:
    """The fixture queue after one round of arrivals, scanned into signals."""

    def __init__(self):
        self.graph = Graph(queue_rows())
        assert scan_queue(self.graph, NAME, ["redmine", "jira", "servicenow"]) == []  # the first read sets the cursor
        self.graph.publish([
            to_row("redmine", arrival("redmine", "4820", "Scratch filesystem full on node051",
                                      "Jobs on node051 fail with ENOSPC on /scratch.", NOW)),
            to_row("jira", arrival("jira", "OPS-15", "Transfers failing with SRM_TIMEOUT after the upgrade",
                                   "FTS reports SRM_TIMEOUT on srm-put.", "2026-10-25T12:01:00.000+0000")),
        ])
        self.signals = {s.split("/", 1)[1]: None for s in scan_queue(self.graph, NAME, ["redmine", "jira", "servicenow"])}
        self.system = FakeTicketSystem()
        self.system.add("redmine", "4820", status="New", assignee="")

    def run(self, instance, signal_id, *, level, mode="live", verdict="approve", halt=(False,), on_wait=None):
        auto = load_automation(instance, NAME, instance_level=level, mode=mode, timeout_seconds=600)
        s = self.graph.signals[(NAME, signal_id)]
        run_id = make_run_id(NAME, signal_id, 1)
        clock, esc_io, recorder = Clock(at(0)), EscalationIO(), Recorder()
        person = Person(clock, esc_io, run_id, verdict, on_wait)
        mediator = ToolMediator(auto, clock, esc_io, self.system, run_id, instance)
        deps = Deps(gate=Gate(GateResult(None, PASS.checks)), reader=GraphReader(self.graph), proposer=TriageProposer(CONFIG), decider=person,
                    mediator=mediator, outcome=Outcome(), recorder=recorder, halt=Halt(*halt), clock=clock,
                    budget=Budget())
        signal = Signal(id=s["id"], entity=s["entity"], title=s["title"], source=s["source"],
                        ticket_keys=(s["entity"].split(":", 1)[1],))
        result = run_once(auto, signal, deps, run_id=run_id)
        return result, recorder, person, mediator


def records(recorder, kind):
    """The attributes of each record of `kind`, with a proposal's cited evidence as `evidence_ids`."""
    return [{**attrs, **({"evidence_ids": list(ev)} if ev else {})} for k, ev, attrs in recorder.calls if k == kind]


def test_the_queue_yields_one_signal_per_arrival():
    w = World()
    assert sorted(w.signals) == ["jira-OPS-15-20261025T120100Z", "redmine-4820-20261025T120000Z"]


def test_shadow_is_the_default_and_records_the_cited_draft_without_acting(instance):
    w = World()
    auto = load_automation(instance, NAME)
    assert auto.mode == "shadow"
    result, rec, person, mediator = w.run(instance, "redmine-4820-20261025T120000Z", level=None, mode="shadow")
    assert (result.status, result.mode, result.level) == ("observed", "shadow", "L1")
    (proposal,) = records(rec, "proposal")
    assert proposal["action_class"] == "draft_reply" and "Sources:" in proposal["draft"]
    assert "tracker_ticket:redmine:4820" in proposal["evidence_ids"]
    (incident,) = records(rec, "incident")
    assert incident["entity"] == "redmine:4820" and incident["ticket_keys"] == ["4820"]
    assert person.opened == [] and mediator.calls == [] and w.system.writes() == []


def test_l1_drafts_cites_and_sends_nothing(instance):
    w = World()
    result, rec, person, mediator = w.run(instance, "jira-OPS-15-20261025T120100Z", level="L1")
    assert (result.status, result.mode) == ("drafted", "live") and not result.mediated
    (proposal,) = records(rec, "proposal")
    assert "SRM_TIMEOUT" in proposal["draft"] and "jira:OPS-11" in proposal["evidence_ids"]
    (decision,) = records(rec, "decision")
    assert decision["verdict"] == "drafted" and "nothing was sent" in decision["reason"]
    assert person.opened == [] and mediator.calls == [] and w.system.requests == []


def test_l2_approved_assigns_the_ticket_and_notes_the_escalation_on_it(instance):
    w = World()
    result, rec, person, mediator = w.run(instance, "redmine-4820-20261025T120000Z", level="L2", verdict="approve")
    assert (result.status, result.mediated) == ("executed", True), result
    (proposal,) = person.opened
    assert proposal.action_class == "assign_ticket" and proposal.payload["value"] == "Grace Hopper"
    ticket = w.system.ticket("redmine", "4820")
    assert ticket["assignee"] == "Grace Hopper"
    eid = wf.expected_escalation_id(make_run_id(NAME, "redmine-4820-20261025T120000Z", 1))
    assert f"escalation {eid}" in ticket["notes"][0] and "operator-x" in ticket["notes"][0]
    (action,) = records(rec, "action")
    assert action["status"] == "succeeded" and action["mediator"] == "archi.tools.tickets:assign_ticket"
    assert records(rec, "decision")[0]["actor"] == "operator-x"


def test_l2_rejected_never_reaches_the_ticket_system(instance):
    w = World()
    result, rec, person, mediator = w.run(instance, "redmine-4820-20261025T120000Z", level="L2", verdict="reject")
    assert result.status == "rejected" and person.opened and mediator.calls == []
    assert w.system.requests == [] and w.system.ticket("redmine", "4820")["assignee"] == ""


def test_l2_timed_out_never_reaches_the_ticket_system(instance):
    w = World()
    result, rec, person, mediator = w.run(instance, "redmine-4820-20261025T120000Z", level="L2", verdict=None)
    assert result.status == "timed_out" and mediator.calls == [] and w.system.requests == []
    assert records(rec, "decision")[0]["verdict"] == "timed_out"


def test_halt_after_the_approval_stops_the_run_before_the_tool(instance):
    w = World()
    result, rec, person, mediator = w.run(instance, "redmine-4820-20261025T120000Z", level="L2", halt=(False, True))
    assert result.status == "halted" and person.opened and mediator.calls == [] and w.system.requests == []
    assert "supersedes the approval" in records(rec, "decision")[-1]["reason"]


def test_an_approval_for_another_payload_is_refused_by_the_tool_itself(instance):
    """The person approved the Action line they saw; if the payload the tool gets differs, it refuses."""
    w = World()
    original = ToolMediator.execute

    def tampered(self, **kw):
        return original(self, **{**kw, "payload": {**kw["payload"], "value": "Alan Turing"}})

    ToolMediator.execute = tampered
    try:
        result, rec, *_ = w.run(instance, "redmine-4820-20261025T120000Z", level="L2")
    finally:
        ToolMediator.execute = original
    (action,) = records(rec, "action")
    assert action["status"] == "failed" and "is not the one being run" in action["result"]
    assert w.system.requests == [] and w.system.ticket("redmine", "4820")["assignee"] == ""


def test_l3_is_refused_at_run_start(instance):
    w = World()
    result, rec, person, mediator = w.run(instance, "redmine-4820-20261025T120000Z", level="L3")
    assert result.status == "refused" and "D13" in result.reason
    assert person.opened == [] and mediator.calls == [] and w.system.requests == []
    (incident,) = records(rec, "incident")
    assert incident["summary"].startswith("run not started")


def test_the_rollback_is_one_line_of_ops_yaml_and_holds_for_a_run_already_waiting(instance):
    """Board d717 rollback: `mode: shadow` for ticket-triage in ops.yaml. A run waiting for its approval when
    the line changes is approved after it, and still sends nothing: the mediator step reads ops.yaml again."""
    ops = instance / "ops.yaml"
    before = ops.read_text()

    def roll_back():
        data = yaml.safe_load(ops.read_text())
        data["automations"][NAME]["mode"] = "shadow"
        ops.write_text(yaml.safe_dump(data))

    try:
        w = World()
        result, rec, person, mediator = w.run(instance, "redmine-4820-20261025T120000Z", level="L2",
                                              verdict="approve", on_wait=roll_back)
        assert person.opened and not result.mediated, result
        assert "mode 'shadow'" in result.reason and "nothing was sent" in result.reason
        assert w.system.writes() == [] and w.system.ticket("redmine", "4820")["assignee"] == ""
    finally:
        ops.write_text(before)
