"""req.r3-ops-self-monitoring.health on the ops runtime: the shipped pack, the L1 draft step, the schedule tick,
the proposer, and the mediator step's rules for a step versus a live tool."""
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.lock import render_and_lock
from archi.health.config import HealthConfig
from archi.health.proposer import HealthProposer, build
from archi.ops import workflow as wf
from archi.ops.context_gate import GateFacts, GateSpec, GateSpecError, evaluate_gate
from archi.ops.loop import AutomationConfig, Deps, MediatorResult, Signal, load_automation, make_run_id, run_once
from archi.ops.proposers import PROPOSER_KINDS, ProposeRequest, ProposerError, StubProposer, build_proposer
from archi.ops.schedule import CronError, latest_fire, tick_id

from bundles.bundle_support import build_cern_team_catalog
from loop_fakes import GEN, PASS, Budget, Clock, Decider, Gate, Halt, Mediator, Outcome, Reader, Recorder
from test_workflow import FakeIO, Tool

NAME = "instance-health"
T = datetime(2026, 10, 25, 12, 7, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    root = tmp_path_factory.mktemp("health")
    build_cern_team_catalog(root / "cat")
    catalog = discovery.from_roots([root / "cat" / "packs"], [root / "cat"])
    assert catalog.problems == []
    render_and_lock(catalog, root / "inst", ["cern-team"], with_packs=["ops-record", NAME])
    return root / "inst"


@pytest.fixture
def auto(instance):
    return load_automation(instance, NAME, instance_level="L1", mode="live")


# --- the pack as rendered --------------------------------------------------------------------


def test_the_pack_is_valid_and_pulls_in_the_ops_record_it_writes_to(instance):
    from archi.bundles.lock import read_lock

    lock = read_lock(instance)
    assert NAME in lock["selection"]["with"] or NAME in str(lock["composed"]["automations"])
    block = lock["composed"]["automations"][NAME]
    assert block["pack_max_level"] == "L2" and block["default_level"] == "L1" and block["evidence"] == "own_signals"
    assert (instance / "profiles" / "cern-team" / "skills" / "health.md").is_file()


def test_it_runs_on_a_schedule_within_its_budget_and_asks_only_for_a_published_graph(auto):
    assert auto.trigger == {"type": "schedule", "cron": "7 * * * *"}
    assert auto.gate.published and auto.gate.sources == ()
    assert auto.pack_max_level == "L2" and auto.cost_per_run == 0.01 and auto.own_signals
    assert {n: c.max_level for n, c in auto.action_classes.items()} == {"draft_issue": "L1", "sync_forge_issues": "L2"}
    assert auto.action_classes["draft_issue"].mediator == "draft_issue"
    assert auto.action_classes["sync_forge_issues"].mediator == "archi.tools.instance_health:sync_forge_issues"


def test_the_gate_asks_whether_the_graph_is_published_and_nothing_about_freshness():
    spec = GateSpec.from_mapping({"published": True})
    assert evaluate_gate(spec, GateFacts(published=True)).ok
    failed = evaluate_gate(spec, GateFacts(published=False))
    assert not failed.ok and failed.failing == ("published",)
    assert not evaluate_gate(spec, GateFacts()).ok  # unknown is not a pass
    with pytest.raises(GateSpecError):
        GateSpec.from_mapping({"published": "yes"})


# --- the schedule ----------------------------------------------------------------------------------


@pytest.mark.parametrize("cron, now, expect", [
    ("7 * * * *", datetime(2026, 10, 25, 12, 30, tzinfo=timezone.utc), "2026-10-25T12:07"),
    ("7 * * * *", datetime(2026, 10, 25, 12, 7, tzinfo=timezone.utc), "2026-10-25T12:07"),
    ("7 * * * *", datetime(2026, 10, 25, 12, 6, tzinfo=timezone.utc), "2026-10-25T11:07"),
    ("*/15 9-17 * * *", datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc), "2026-10-25T17:45"),
    ("0 6 * * 1", datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc), "2026-10-19T06:00"),  # a Monday
    ("0 0 1 * *", datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc), "2026-10-01T00:00"),
])
def test_the_latest_cron_time_at_or_before_now(cron, now, expect):
    assert latest_fire(cron, now).strftime("%Y-%m-%dT%H:%M") == expect


def test_a_tick_id_is_a_legal_run_id_part_and_is_the_same_within_one_cron_time():
    a, b = latest_fire("7 * * * *", T + timedelta(minutes=3)), latest_fire("7 * * * *", T + timedelta(minutes=50))
    assert tick_id(a) == tick_id(b) == "tick-20261025T1207Z"
    make_run_id(NAME, tick_id(a), 1)


def test_a_bad_cron_is_refused():
    for bad in ("* * *", "61 * * * *", "a * * * *", "*/0 * * * *"):
        with pytest.raises(CronError):
            latest_fire(bad, T)


class TickIO:
    def __init__(self):
        self.signals = {}

    def add_signal(self, automation, signal_id, entity, title, *, summary=None, source="table"):
        self.signals.setdefault((automation, signal_id), (entity, title, source))


def test_each_scan_adds_the_latest_tick_once_and_only_for_scheduled_automations():
    scheduled = SimpleNamespace(trigger={"type": "schedule", "cron": "7 * * * *"})
    queued = SimpleNamespace(trigger={"type": "signal", "source": "x", "condition": "y"})
    rt = SimpleNamespace(settings={NAME: object(), "other": object()}, deployment="inst",
                         automation=lambda n: scheduled if n == NAME else queued)
    io = TickIO()
    wf.add_schedule_ticks(rt, io, T + timedelta(minutes=1))
    wf.add_schedule_ticks(rt, io, T + timedelta(minutes=2))  # the next scan, same cron time
    assert io.signals == {(NAME, "tick-20261025T1207Z"): ("inst", "scheduled run 2026-10-25 12:07Z", "schedule")}
    wf.add_schedule_ticks(rt, io, T + timedelta(hours=1))
    assert (NAME, "tick-20261025T1307Z") in io.signals and len(io.signals) == 2  # a new hour, a new tick


def test_a_broken_automation_file_does_not_stop_the_scan():
    def boom(name):
        raise FileNotFoundError("no automations/x.yaml")

    said = []
    wf.add_schedule_ticks(SimpleNamespace(settings={"x": object()}, deployment="i", automation=boom), TickIO(), T, said.append)
    assert said and "cannot work out its schedule" in said[0]


# --- the L1 draft step in the loop ------------------------------------------------------------------


class World:
    def __init__(self, proposal, **mediator):
        self.clock = Clock()
        self.recorder, self.mediator = Recorder(), Mediator(**mediator)
        self.proposer = StubProposer({"default": proposal})
        self.deps = Deps(Gate(PASS), Reader(), self.proposer, Decider(self.clock, None), self.mediator, Outcome(),
                         self.recorder, Halt(False), self.clock, Budget())

    def run(self, auto, signal_id="tick-1"):
        return run_once(auto, Signal(signal_id, "inst", "scheduled run", source="schedule"), self.deps,
                        run_id=make_run_id(NAME, signal_id, 1))


def health_proposal(action_class="draft_issue", tracker=None):
    return {"summary": "Instance health: 1 problem(s)", "evidence_ids": [f"ops_incident:inc-{make_run_id(NAME, 'tick-1', 1)}"],
            "action_class": action_class, "confidence": 1.0, "rollback": "revert",
            "payload": {"tracker": tracker or {"kind": "local"}, "observed_at": "2026-10-25T12:07:00Z", "problems": [],
                        "scope": ["doctor"], "skipped": []}}


def test_at_l1_the_draft_step_runs_with_no_escalation_and_its_action_is_recorded(auto):
    w = World(health_proposal())
    result = w.run(auto)
    assert result.status == "drafted" and result.mediated
    (call,) = w.mediator.calls
    assert call["action_class"] == "draft_issue" and call["escalation"] is None
    assert call["idempotency_key"] == make_run_id(NAME, "tick-1", 1)
    kinds = [c[0] for c in w.recorder.calls]
    assert kinds == ["incident", "proposal", "decision", "action"]
    assert w.recorder.calls[2][2]["verdict"] == "drafted"
    assert w.recorder.calls[3][2]["status"] == "succeeded" and w.recorder.calls[3][2]["mediator"] == "draft_issue"
    assert w.deps.decider.calls == []  # no escalation


def test_a_proposal_may_cite_the_incident_of_its_own_run(auto):
    w = World(health_proposal())
    assert w.run(auto).status == "drafted"
    assert w.recorder.calls[1][1] == [f"ops_incident:inc-{make_run_id(NAME, 'tick-1', 1)}"]
    assert w.proposer.calls[0].context["run_incident"] == f"ops_incident:inc-{make_run_id(NAME, 'tick-1', 1)}"


def test_evidence_that_is_not_a_node_of_the_pinned_read_or_this_run_is_still_refused(auto):
    bad = {**health_proposal(), "evidence_ids": ["jira:NOT-READ"]}
    assert World(bad).run(auto).status == "invalid_proposal"


def test_the_forge_live_tool_never_runs_at_l1(auto):
    tracker = {"kind": "github", "repo": "o/r", "token_env": "MY_TOKEN"}
    w = World(health_proposal("sync_forge_issues", tracker))
    result = w.run(auto)
    # the class allows L2 but the pack's maximum is L1: the run is L1, and a live tool is recorded, never run
    assert result.status == "drafted" and not result.mediated and w.mediator.calls == []


def test_a_refused_step_supersedes_the_drafted_decision_and_writes_no_action(auto):
    from archi.ops.loop import MediationRefused

    w = World(health_proposal(), refuse=MediationRefused("halted", "the schedule is paused"))
    assert w.run(auto).status == "halted"
    assert [c[2].get("verdict") for c in w.recorder.calls if c[0] == "decision"] == ["drafted", "halted"]
    assert "action" not in [c[0] for c in w.recorder.calls]


def test_in_shadow_the_step_does_not_run(instance):
    shadow = load_automation(instance, NAME, instance_level="L1", mode="shadow")
    w = World(health_proposal())
    assert w.run(shadow).status == "observed" and w.mediator.calls == []


# --- the mediator step: a step needs no escalation, a live tool still does ---------------------------------


def mediate(io, entry, tool, esc=None, at_l1=True):
    return wf.run_mediation(
        io, mediator_entry=lambda: entry, action_class="x", payload={"p": 1}, proposal_id="prop-1",
        idempotency_key="k", esc=esc, workflow_id="k", dsn="postgresql://x", halt=lambda io: None,
        resolve=lambda e: tool, refused=lambda v, r: None, instance_dir="/inst", at_l1=at_l1)


def test_a_step_runs_without_an_escalation_and_gets_the_instance_directory():
    io, tool = FakeIO(), Tool()
    out = mediate(io, "draft_issue", tool)
    assert out == {"status": "succeeded", "result": "done"}
    assert tool.calls[0][1]["instance_dir"] == "/inst" and io.begun == [("k", "prop-1")]


def test_a_live_tool_with_no_escalation_is_still_refused():
    io, tool = FakeIO(), Tool()
    out = mediate(io, "archi.tools.instance_health:sync_forge_issues", tool)
    assert out["refused"][0] == "error" and "no escalation" in out["refused"][1] and tool.calls == [] and io.begun == []


def test_a_step_is_still_stopped_by_halt_and_by_an_earlier_attempt():
    io, tool = FakeIO(attempts={"k": {"status": "attempted", "result": None}}), Tool()
    assert mediate(io, "draft_issue", tool)["status"] == "unknown" and tool.calls == []
    halted = wf.run_mediation(FakeIO(), mediator_entry=lambda: "draft_issue", action_class="x", payload={}, proposal_id="p",
                              idempotency_key="k2", esc=None, workflow_id="k2", dsn="d", halt=lambda io: "paused",
                              resolve=lambda e: tool, refused=lambda v, r: None, at_l1=True)
    assert halted["refused"][0] == "halted" and tool.calls == []


def test_draft_issue_resolves_to_its_implementation_and_a_step_without_one_does_not():
    from archi.ops.stubs import resolve_mediator

    assert resolve_mediator("draft_issue").__module__ == "archi.health.steps"
    with pytest.raises(ValueError):
        resolve_mediator("draft_reply")


# --- the proposer -----------------------------------------------------------------------------------------------


def doctor_report():
    return {"ok": False, "checks": [{"id": "source-freshness", "status": "fail", "message": "m", "remediation": "r", "footgun": "",
                                      "data": {"sources": [{"source": "jira.jira", "ok": False, "last_reconcile_at": "2026-09-20T10:00:00+00:00",
                                                            "max_staleness": "24h"}]}}]}


def proposer(tmp_path, **kw):
    return HealthProposer(kw.get("config") or HealthConfig(eval_reports_dir="eval"), instance_dir=tmp_path, deployment="inst", dsn="postgresql://x", now=lambda: T,
                          read_doctor=kw.get("doctor", doctor_report), read_frontier=lambda: {"rows": []},
                          read_ops_records=lambda subtype: [])


def request(**ctx):
    return ProposeRequest(NAME, {"id": "tick-1"}, {"run_incident": "ops_incident:inc-1", **ctx}, GEN, 1)


def test_the_proposal_names_the_problems_the_skip_and_validates_against_the_pack(tmp_path, auto):
    from archi.ops.loop import validate_proposal

    proposed = proposer(tmp_path).propose(request())
    raw = proposed.proposal
    assert proposed.cost == 0.0 and raw["action_class"] == "draft_issue" and raw["evidence_ids"] == ["ops_incident:inc-1"]
    assert "source-freshness/jira.jira" in raw["summary"] and "skipped: eval (no eval report exists" in raw["summary"]
    assert raw["payload"]["scope"] == ["doctor", "ingest", "automation-errors", "monitor"]
    assert raw["payload"]["observed_at"] == "2026-10-25T12:07:00Z"
    validate_proposal(raw, auto, "L1", {"ops_incident:inc-1"})  # the pack's own proposal and payload schemas


def test_a_healthy_instance_still_gets_a_proposal_so_the_issues_can_close(tmp_path, auto):
    from archi.ops.loop import validate_proposal

    raw = proposer(tmp_path, doctor=lambda: {"ok": True, "checks": []}).propose(request()).proposal
    assert raw["payload"]["problems"] == [] and raw["summary"].startswith("Instance health: no problems found in doctor")
    validate_proposal(raw, auto, "L1", {"ops_incident:inc-1"})


def test_eval_reports_under_the_instance_data_root_reach_the_proposal(tmp_path, auto):
    """The default place: `archi eval` writes <data root>/eval/<bundle>/<UTC stamp>/report.json (an older archi wrote
    <data root>/eval/<bundle>/report.json; both are read); the proposer reads it there."""
    from archi.ops.loop import validate_proposal
    from archi.stack.instance import Instance, Ports
    from health_eval_reports import scored, write_report

    inst = Instance(dir=tmp_path / "inst", name="demo", bundles=["cern-team"], data_root=tmp_path / "data",
                    runtime="docker", ports=Ports(graph_db=15432, mcp=18080, webui=13000))
    inst.dir.mkdir()
    inst.save()
    write_report(tmp_path / "data" / "eval", started_at="2026-10-25T03:00:00Z", statuses=["execution_failed"] * 10,
                 error="the MCP endpoint offers no tools, so the agent has nothing to read with")
    raw = proposer(inst.dir, config=HealthConfig(), doctor=lambda: {"ok": True, "checks": []}).propose(request()).proposal
    (problem,) = raw["payload"]["problems"]
    assert problem["key"] == "eval/cern-team" and problem["family"] == "eval" and problem["severity"] == "error"
    assert "not a low score" in problem["detail"] and "eval" in raw["payload"]["scope"]
    assert "eval/cern-team" in raw["summary"] and "skipped" not in raw["summary"]  # a failed run is judged, not skipped
    validate_proposal(raw, auto, "L1", {"ops_incident:inc-1"})


def test_the_forge_tracker_picks_the_live_tool_class(tmp_path, auto):
    from archi.ops.loop import validate_proposal

    cfg = HealthConfig.from_proposer({"kind": "instance-health", "tracker": {"kind": "gitlab", "url": "https://gl.example",
                                                                             "project": "g/p", "token_env": "T_ENV"}})
    p = HealthProposer(cfg, instance_dir=tmp_path, deployment="i", dsn="d", now=lambda: T, read_doctor=lambda: {"checks": []},
                       read_frontier=lambda: {"rows": []}, read_ops_records=lambda s: [])
    raw = p.propose(request()).proposal
    assert raw["action_class"] == "sync_forge_issues" and raw["payload"]["tracker"]["token_env"] == "T_ENV"
    validate_proposal(raw, auto, "L2", {"ops_incident:inc-1"})


def test_the_ops_yaml_block_builds_the_proposer_and_an_unknown_kind_or_key_is_refused(tmp_path):
    rt = SimpleNamespace(instance_dir=tmp_path, deployment="inst", dsn="postgresql://x")
    assert isinstance(build_proposer({"kind": "instance-health", "tracker": {"kind": "local"}}, rt), HealthProposer)
    assert isinstance(build_proposer({"kind": "stub", "fixture": {}}, rt), StubProposer)
    with pytest.raises(ProposerError, match="not available"):
        build_proposer({"kind": "gpt"}, rt)
    with pytest.raises(Exception, match="unknown keys"):
        build({"kind": "instance-health", "token": "x"}, rt)
    assert set(PROPOSER_KINDS) == {"stub", "instance-health"}


# --- end to end without okg: the loop, the real step, a real repository --------------------------------------------


def test_a_stale_source_over_two_runs_leaves_exactly_one_committed_issue(tmp_path, auto):
    """The loop runs the proposer and then the real draft_issue step through run_mediation, twice."""
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], capture_output=True, text=True, check=True).stdout
    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    (tmp_path / "archi.yaml").write_text("name: t\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    prop = proposer(tmp_path)

    class RealMediator:
        def __init__(self):
            self.io = FakeIO()

        def execute(self, *, action_class, payload, proposal_id, idempotency_key, generation, escalation=None, level="L2"):
            from archi.ops.stubs import resolve_mediator

            out = wf.run_mediation(self.io, mediator_entry=lambda: auto.action_classes[action_class].mediator,
                                   action_class=action_class, payload=payload, proposal_id=proposal_id,
                                   idempotency_key=idempotency_key, esc=None, workflow_id=idempotency_key, dsn="d",
                                   halt=lambda io: None, instance_dir=str(tmp_path), at_l1=level == "L1")
            return MediatorResult(out["status"], out["result"])

    results = []
    for signal_id in ("tick-1", "tick-2"):
        run_id = make_run_id(NAME, signal_id, 1)
        world = World({})
        world.proposer = SimpleNamespace(propose=lambda req: prop.propose(req), calls=[])
        world.deps = Deps(Gate(PASS), Reader(), world.proposer, Decider(world.clock, None), RealMediator(), Outcome(),
                          world.recorder, Halt(False), world.clock, Budget())
        sig = Signal(signal_id, "inst", "scheduled run", source="schedule")
        results.append(run_once(auto, sig, world.deps, run_id=run_id))
    assert [r.status for r in results] == ["drafted", "drafted"]
    files = sorted(p.name for p in (tmp_path / ".archi" / "issues").glob("*.md"))
    assert files == ["source-freshness--jira.jira.md"]
    assert git("log", "--format=%s").splitlines() == ["health: open source-freshness/jira.jira", "init"]
    assert git("status", "--porcelain") == ""


# --- review fixes -----------------------------------------------------------------------------------------------------

import dataclasses
import shutil

from archi.bundles.manifest import validate_pack
from archi.health.signals import ERROR_VERDICTS, automation_error_problems

SHIPPED = Path(__file__).resolve().parents[3] / "bundles" / "packs" / NAME


def pack_problems(tmp_path, change):
    dest = tmp_path / NAME
    shutil.copytree(SHIPPED, dest)
    data = yaml.safe_load((dest / "pack.yaml").read_text())
    change(data["automation"])
    (dest / "pack.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return [p.message for p in validate_pack(dest, defer_requires=True)]


def test_the_shipped_pack_is_valid_with_its_opt_in(tmp_path):
    assert pack_problems(tmp_path, lambda a: None) == []


def test_a_published_only_gate_is_refused_without_the_opt_in(tmp_path):
    def drop(a):
        del a["evidence"]

    messages = pack_problems(tmp_path, drop)
    assert any("`published: true` alone is allowed only with `automation.evidence: own_signals`" in m for m in messages)
    assert any("draft_issue" in m and "own_signals" in m for m in messages)  # the L1 step needs the opt-in too


def test_a_gate_with_a_freshness_check_needs_no_opt_in_for_its_gate(tmp_path):
    def swap(a):
        del a["evidence"]
        a["context_gate"] = {"sources": ["ops-record.x"], "max_staleness": "24h"}

    messages = pack_problems(tmp_path, swap)
    assert not any("declares no check" in m for m in messages)


def test_the_opt_in_is_for_a_monitor_that_starts_at_l1_or_below(tmp_path):
    def high(a):
        a["default_level"] = "L2"

    assert any("starts at L0 or L1" in m for m in pack_problems(tmp_path, high))


def test_without_the_opt_in_a_run_cannot_cite_its_own_incident(auto):
    plain = dataclasses.replace(auto, own_signals=False)
    result = World(health_proposal()).run(plain)
    assert result.status == "invalid_proposal"
    assert World(health_proposal()).run(auto).status == "drafted"  # the opt-in


def test_without_the_opt_in_the_draft_step_does_not_run_at_l1(auto):
    plain = dataclasses.replace(auto, own_signals=False)
    cited = {**health_proposal(), "evidence_ids": ["jira:A-1"]}  # cites the pinned read, as any automation must
    w = World(cited)
    assert w.run(plain).status == "drafted" and w.mediator.calls == []


def test_at_l2_a_step_still_needs_the_escalation_to_be_re_read():
    io, tool = FakeIO(), Tool()
    out = mediate(io, "draft_issue", tool, at_l1=False)
    assert out["refused"][0] == "error" and "no escalation" in out["refused"][1] and tool.calls == []


def test_a_failing_draft_step_is_an_error_decision_not_a_draft(auto):
    w = World(health_proposal(), result=MediatorResult("failed", "TrackerError: git commit failed"))
    result = w.run(auto)
    assert result.status == "error" and "git commit failed" in result.reason
    verdicts = [c[2]["verdict"] for c in w.recorder.calls if c[0] == "decision"]
    assert verdicts == ["drafted", "error"]
    assert [c[2]["status"] for c in w.recorder.calls if c[0] == "action"] == ["failed"]
    assert "error" in ERROR_VERDICTS


def test_an_automation_whose_draft_step_keeps_failing_shows_up_in_the_error_rate():
    rows = [{"node_id": f"d{i}", "attrs": {"automation": NAME, "verdict": "error", "decided_at": "2026-10-25T11:00:00Z"}}
            for i in range(3)]
    (p,) = automation_error_problems(lambda s: rows, T)
    assert p.key == f"automation-errors/{NAME}"


def test_a_forge_tracker_at_l1_says_in_the_proposal_that_nothing_was_posted(tmp_path):
    cfg = HealthConfig.from_proposer({"kind": "instance-health", "tracker": {
        "kind": "github", "repo": "o/r", "token_env": "T_ENV"}})
    p = HealthProposer(cfg, instance_dir=tmp_path, deployment="i", dsn="d", now=lambda: T, read_doctor=lambda: {"checks": []},
                       read_frontier=lambda: {"rows": []}, read_ops_records=lambda s: [])
    l1 = p.propose(request()).proposal["summary"]
    l2 = p.propose(ProposeRequest(NAME, {"id": "t"}, {"run_incident": "ops_incident:inc-1"}, GEN, 2)).proposal["summary"]
    assert "WARNING" in l1 and "only drafted" in l1 and "WARNING" not in l2
