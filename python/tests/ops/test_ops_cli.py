"""The workflow-free pieces of the runtime and `archi ops`: the renderer's runtime line,
the instance configuration, reading an escalation row back, and the halt, resume,
status and answer logic over a fake OpsIO."""
import json
import shutil
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.lock import render_and_lock
from archi.bundles.cli import build_parser
from archi.cli import ops
from archi.ops.control import RunInfo
from archi.ops.escalation import Decision, decision_from_row, may_execute, Escalation
from archi.ops.record import RecordError
from archi.ops.runtime_config import RuntimeConfigError, load_runtime, raw_levels, read_settings

LIBRARY = Path(__file__).resolve().parent / "fixtures" / "library"
ALPHA = Path(__file__).resolve().parents[1] / "bundles" / "fixtures" / "library" / "packs" / "alpha"
T0 = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)


@pytest.fixture
def instance(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(LIBRARY, root)
    shutil.copytree(ALPHA, root / "packs" / "alpha")
    bdir = root / "bundles" / "triage-team"
    bdir.mkdir(parents=True)
    (bdir / "bundle.yaml").write_text(yaml.safe_dump({
        "name": "triage-team", "version": "0.1.0", "description": "t",
        "packs": {"default": ["alpha", "fixture-triage"]}, "okg": ">=0.2.0,<0.3",
    }, sort_keys=False))
    catalog = discovery.from_roots([root / "packs"], [root / "bundles"])
    inst = tmp_path / "inst"
    render_and_lock(catalog, inst, ["triage-team"])
    return inst, catalog, root


# --- the renderer declares the runtime workflow -------------------------------------


def test_render_declares_the_ops_workflow_when_an_automation_is_selected(instance):
    inst, _, _ = instance
    defaults = yaml.safe_load(next(inst.glob("profiles/*/deployment-defaults.yaml")).read_text())
    assert defaults["runtime"] == {"enabled": True, "workflows": ["archi.ops.workflow"]}


def test_render_leaves_the_runtime_off_without_an_automation(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(LIBRARY, root)
    shutil.copytree(ALPHA, root / "packs" / "alpha")
    bdir = root / "bundles" / "plain"
    bdir.mkdir(parents=True)
    (bdir / "bundle.yaml").write_text(yaml.safe_dump({
        "name": "plain", "version": "0.1.0", "description": "t", "packs": {"default": ["alpha"]},
        "okg": ">=0.2.0,<0.3"}, sort_keys=False))
    catalog = discovery.from_roots([root / "packs"], [root / "bundles"])
    render_and_lock(catalog, tmp_path / "inst", ["plain"])
    defaults = yaml.safe_load(next((tmp_path / "inst").glob("profiles/*/deployment-defaults.yaml")).read_text())
    assert defaults["runtime"] == {"enabled": False}


# --- configuration ----------------------------------------------------------------------


def write_instance(inst: Path, levels=None, ops_yaml=None):
    (inst / "archi.yaml").write_text(yaml.safe_dump({"name": "dep", "automations": {
        n: {"level": lv} for n, lv in (levels or {}).items()}}))
    if ops_yaml is not None:
        (inst / "ops.yaml").write_text(yaml.safe_dump(ops_yaml))


def test_runtime_config_reads_levels_leniently_so_l3_reaches_the_run(instance):
    inst, _, _ = instance
    write_instance(inst, {"fixture-triage": "L3"}, {"automations": {"fixture-triage": {"mode": "live", "timeout_seconds": 30}}})
    assert raw_levels(inst) == {"fixture-triage": "L3"}
    rt = load_runtime({"ARCHI_INSTANCE": str(inst), "OKG_DSN": "postgresql://x"})
    assert rt.deployment == "dep"
    config = rt.automation("fixture-triage")
    assert (config.instance_level, config.mode, config.timeout_seconds) == ("L3", "live", 30)


def test_runtime_config_defaults_to_shadow_and_refuses_unknown_keys(instance, tmp_path):
    inst, _, _ = instance
    write_instance(inst, ops_yaml={"automations": {"a": {}}})
    assert read_settings(inst)["a"].mode == "shadow"
    write_instance(inst, ops_yaml={"automations": {"a": {"modee": "live"}}})
    with pytest.raises(RuntimeConfigError, match="unknown keys"):
        read_settings(inst)
    with pytest.raises(RuntimeConfigError, match="ARCHI_INSTANCE"):
        load_runtime({})
    with pytest.raises(RuntimeConfigError, match="OKG_DSN"):
        load_runtime({"ARCHI_INSTANCE": str(inst)})


# --- reading an escalation row back -----------------------------------------------------


def row(status="applied", decision=None, by="alice", at="2026-10-25T12:01:00+00:00"):
    return {"status": status, "decided_by": by, "decided_at": at,
            "decision": decision if decision is not None else {"verdict": "approve"}}


@pytest.mark.parametrize("attrs, status, error", [
    (row(), "approved", None),
    (row("decided", {"verdict": "reject", "note": "no"}), "rejected", None),
    (row("timed_out", {}), "timed_out", None),
    (row("pending", {}), "error", "pending, not decided"),
    (row("decided", {"verdict": "maybe"}), "error", "verdict must be one of"),
    (None, "error", "not found"),
])
def test_decision_from_row(attrs, status, error):
    d = decision_from_row("esc-1", "ops_proposal:prop-9", attrs)
    assert (d.status, d.proposal_id, d.escalation_id) == (status, "prop-9", "esc-1")
    assert (error is None) or error in d.error


def test_a_row_for_another_subject_cannot_execute():
    d = decision_from_row("esc-1", "jira:X-1", row())
    esc = Escalation("esc-1", "prop-9", T0, T0.replace(hour=13))
    assert d.proposal_id is None and not may_execute(d, esc, T0.replace(minute=30))
    good = decision_from_row("esc-1", "ops_proposal:prop-9", row())
    assert may_execute(good, esc, T0.replace(minute=30))


# --- archi ops --------------------------------------------------------------------------


class FakeIO:
    deployment, dsn = "dep", "postgresql://x"

    def __init__(self, schedules=None, runs=(), escalations=(), halts=()):
        self._schedules = dict(schedules if schedules is not None else {"archi_ops_scan": False})
        self._runs, self._escalations, self._halts = list(runs), list(escalations), list(halts)
        self.calls = []

    def ensure_tables(self): self.calls.append(("ensure",))
    def schedules(self): return dict(self._schedules)
    def runs(self, live_only=True): return list(self._runs)
    def run_input(self, wid): return {"id": "sig-1", "entity": "T2_US_MIT", "title": "Site down"}
    def pin_generation(self): return "gen-1"
    def read_nodes(self, subtype, generation, where=None): return []

    def pause(self, name):
        self.calls.append(("pause", name))
        self._schedules[name] = True

    def resume(self, name):
        self.calls.append(("resume", name))
        self._schedules[name] = False

    def record_halt(self, names, by): self.calls.append(("record_halt", tuple(names), by)); self._halts += list(names)
    def paused_by_halt(self): return list(self._halts)
    def clear_halt(self, name): self.calls.append(("clear_halt", name)); self._halts.remove(name)
    def cancel(self, wid): self.calls.append(("cancel", wid))
    def cancel_unconfirmed_admission(self, key): self.calls.append(("cancel_admission", key)); return ["adm-1"]

    def escalations(self, open_only=True, workflow_ids=None):
        return [e for e in self._escalations if workflow_ids is None or e["requested_by_workflow"] in workflow_ids]

    def respond(self, escalation_id, verdict, *, note, decided_by):
        self.calls.append(("respond", escalation_id, verdict, note, decided_by))

    def day_spend(self, automation): return (3.0, 10.0)
    def escalation(self, esc_id): return None
    def requeue_cancelled_signals(self): return []


def test_halt_pauses_then_closes_escalations_then_cancels_and_records():
    io = FakeIO(
        runs=[RunInfo("r1", "fixture-triage", "PENDING"), RunInfo("r2", "fixture-triage", "SUCCESS")],
        escalations=[{"escalation_id": "esc-1", "proposal_id": "prop-r1", "requested_by_workflow": "r1",
                      "timeout_at": "x", "reason": "r"}])
    submitted = []
    out = ops.do_halt(io, who="jason", log=lambda _: None,
                      submit_records=lambda records, **kw: submitted.append((records, kw)))
    assert [c[0] for c in io.calls] == ["ensure", "pause", "record_halt", "respond", "cancel"]
    assert io.calls[3] == ("respond", "esc-1", "reject", "closed by archi ops halt", "archi ops halt")
    assert out == {"paused": ["archi_ops_scan"], "already_paused": [], "closed": ["esc-1"],
                   "stayed_open": [], "cancelled": ["r1"], "recorded": ["r1"], "unrecorded": [],
                   "left_running": [], "left_final": ["r2"]}
    records, kw = submitted[0]
    verdicts = {n["attrs"]["record_id"]: n["attrs"].get("verdict") for n in records.nodes.values()}
    assert verdicts["dec-r1-halt"] == "halted"
    assert records.nodes["ops_proposal:prop-r1-halt"]["attrs"]["mode"] == "placeholder"
    assert records.dangling() == [] and kw["idempotency_key"] == "r1|halt"
    assert kw["max_wait_seconds"] == ops.SUBMIT_WAIT_SECONDS


def test_halt_goes_on_when_the_records_cannot_be_confirmed():
    io = FakeIO(runs=[RunInfo("r1", "a", "PENDING")])
    messages = []

    def unconfirmed(*a, **k):
        raise RecordError("okg admission submit failed: pending")

    out = ops.do_halt(io, log=messages.append, submit_records=unconfirmed)
    assert out["cancelled"] == ["r1"] and out["recorded"] == [] and out["unrecorded"] == ["r1"]
    assert any("not confirmed" in m for m in messages)
    # the queued admission is cancelled, or okg would refuse to start a worker again
    assert ("cancel_admission", "r1|halt") in io.calls


def test_halt_survives_an_escalation_that_is_already_closed():
    class Closed(FakeIO):
        def respond(self, *a, **k):
            raise RuntimeError("is in status 'applied'")

    io = Closed(runs=[RunInfo("r1", "a", "PENDING")], escalations=[
        {"escalation_id": "esc-1", "proposal_id": "p", "requested_by_workflow": "r1", "timeout_at": "x", "reason": ""}])
    out = ops.do_halt(io, log=lambda _: None, submit_records=lambda *a, **k: None)
    assert out["closed"] == [] and out["cancelled"] == ["r1"]


def test_resume_only_resumes_what_halt_paused():
    io = FakeIO(schedules={"archi_ops_scan": True}, halts=["archi_ops_scan"])
    assert ops.do_resume(io, log=lambda _: None)["resumed"] == ["archi_ops_scan"]
    assert ("clear_halt", "archi_ops_scan") in io.calls
    operator_paused = FakeIO(schedules={"archi_ops_scan": True}, halts=[])
    assert ops.do_resume(operator_paused, log=lambda _: None)["resumed"] == []


def test_halt_reports_an_escalation_it_could_not_close_and_does_not_hide_it():
    class Stuck(FakeIO):
        def respond(self, *a, **k):
            raise RuntimeError("db write refused")

        def escalation(self, esc_id):  # still open after the failed answer
            return ("ops_proposal:p", {"status": "pending"})

    io = Stuck(runs=[RunInfo("r1", "a", "PENDING")], escalations=[
        {"escalation_id": "esc-1", "proposal_id": "p", "requested_by_workflow": "r1", "timeout_at": "x", "reason": ""}])
    messages = []
    out = ops.do_halt(io, log=messages.append, submit_records=lambda *a, **k: None)
    assert out["closed"] == []
    assert out["stayed_open"] == [{"escalation_id": "esc-1", "error": "RuntimeError: db write refused"}]
    assert out["cancelled"] == ["r1"] and any("STAYED OPEN" in m for m in messages)


def test_halt_counts_an_escalation_decided_meanwhile_as_closed():
    class Raced(FakeIO):
        def respond(self, *a, **k):
            raise RuntimeError("is in status 'decided'")

        def escalation(self, esc_id):
            return ("ops_proposal:p", {"status": "decided"})

    io = Raced(runs=[RunInfo("r1", "a", "PENDING")], escalations=[
        {"escalation_id": "esc-1", "proposal_id": "p", "requested_by_workflow": "r1", "timeout_at": "x", "reason": ""}])
    out = ops.do_halt(io, log=lambda _: None, submit_records=lambda *a, **k: None)
    assert out["closed"] == ["esc-1"] and out["stayed_open"] == []


def test_halt_with_no_cancel_pauses_and_leaves_runs_and_escalations_alone():
    io = FakeIO(runs=[RunInfo("r1", "a", "PENDING")], escalations=[
        {"escalation_id": "esc-1", "proposal_id": "p", "requested_by_workflow": "r1", "timeout_at": "x", "reason": ""}])
    out = ops.do_halt(io, cancel=False, log=lambda _: None, submit_records=lambda *a, **k: pytest.fail("no records"))
    assert out["paused"] == ["archi_ops_scan"] and out["cancelled"] == [] and out["left_running"] == ["r1"]
    assert [c[0] for c in io.calls] == ["ensure", "pause", "record_halt"]


def test_halt_exits_nonzero_when_an_escalation_stayed_open(monkeypatch, capsys):
    monkeypatch.setattr(ops, "make_io", lambda args: object())
    monkeypatch.setattr(ops, "do_halt", lambda io, **kw: {"stayed_open": [{"escalation_id": "esc-1", "error": "x"}],
                                                          "paused": [], "closed": [], "cancelled": [], "left_running": []})
    args = build_parser().parse_args(["ops", "halt"])
    assert args.func(args) == 1
    assert "stayed open: esc-1" in capsys.readouterr().err


def test_resume_gives_a_halted_signal_a_new_attempt():
    class Requeue(FakeIO):
        def requeue_cancelled_signals(self):
            return [("fixture-triage", "sig-1", 2)]

    out = ops.do_resume(Requeue(schedules={"archi_ops_scan": True}, halts=["archi_ops_scan"]), log=lambda _: None)
    assert out["requeued"] == [["fixture-triage", "sig-1", 2]]


def escalation_io(subject="ops_proposal:prop-1", status="pending", workflow="fixture-triage~sig-1~1", by=None):
    class Io(FakeIO):
        def escalation(self, esc_id):
            if esc_id != "esc-1":
                return None
            return (subject, {"status": status, "requested_by_workflow": workflow, "decided_by": by})

    return Io()


def test_approve_records_the_os_user_and_a_claimed_name_is_marked_as_claimed(monkeypatch):
    monkeypatch.setattr(ops, "os_user", lambda: "jdoe")
    io = escalation_io()
    out = ops.do_respond(io, "esc-1", "approve", "ok", None, automations=["fixture-triage"])
    assert out["decided_by"] == "jdoe"
    out = ops.do_respond(io, "esc-1", "reject", None, "alice", automations=["fixture-triage"])
    assert out["decided_by"] == "jdoe for alice"
    assert [c[4] for c in io.calls if c[0] == "respond"] == ["jdoe", "jdoe for alice"]
    assert ops.decider_name("jdoe") == "jdoe"


@pytest.mark.parametrize("io, fragment", [
    (escalation_io(subject="jira:X-1"), "not an ops proposal"),
    (escalation_io(workflow="somebody-elses-workflow"), "not opened by an archi ops run"),
    (escalation_io(workflow="other-automation~sig-1~1"), "which this instance does not have"),
    (escalation_io(status="timed_out"), "is timed_out; it can no longer be answered"),
    (escalation_io(status="decided", by="alice"), "is decided, by alice; it can no longer be answered"),
    (escalation_io(status="applied"), "can no longer be answered"),
])
def test_approve_refuses_what_is_not_an_open_ops_proposal_of_this_instance(io, fragment):
    with pytest.raises(ops.OpsCliError, match=fragment):
        ops.do_respond(io, "esc-1", "approve", None, None, automations=["fixture-triage"])
    assert not any(c[0] == "respond" for c in io.calls)


def test_approve_of_an_unknown_escalation_is_refused():
    with pytest.raises(ops.OpsCliError, match="no escalation 'esc-9'"):
        ops.do_respond(escalation_io(), "esc-9", "approve", None, None)


def test_a_refused_approval_exits_nonzero_with_the_message(monkeypatch, capsys):
    monkeypatch.setattr(ops, "make_io", lambda args: escalation_io(status="timed_out"))
    monkeypatch.setattr(ops, "automations", lambda d: [SimpleNamespace(name="fixture-triage")])
    args = build_parser().parse_args(["ops", "approve", "esc-1"])
    assert args.func(args) == 2
    assert "is timed_out; it can no longer be answered" in capsys.readouterr().err


def test_status_reports_levels_schedules_budgets_and_pending(instance):
    inst, _, _ = instance
    write_instance(inst, {"fixture-triage": "L3"}, {"automations": {"fixture-triage": {"mode": "live"}}})
    io = FakeIO(schedules={"archi_ops_scan": True}, escalations=[
        {"escalation_id": "esc-1", "proposal_id": "p", "requested_by_workflow": "r", "timeout_at": "t", "reason": ""}])
    report = ops.status_report(inst, io, proposals=[{"attrs": {"automation": "fixture-triage", "mode": "shadow"}}],
                               shadow=lambda name: {"incidents": 0})
    assert report["scanner"] == {"archi_ops_scan": "paused"}
    (row,) = report["automations"]
    assert row["name"] == "fixture-triage" and row["level"] is None and "L3" in row["level_problem"]
    assert row["mode"] == "live" and row["schedule_paused"] is False  # keyed by automation name, not the scanner
    assert row["budget"]["day_spent"] == 3.0 and row["budget"]["per_day_cap"] == 10.0
    assert row["shadow"]["shadow_proposals"] == 1 and row["shadow_report"] == {"incidents": 0}
    assert len(report["pending"]) == 1
    json.dumps(report, default=str)


def test_status_says_so_when_the_shadow_report_cannot_be_built(instance):
    inst, _, _ = instance
    write_instance(inst, {"fixture-triage": "L1"})

    def broken(name):
        raise RuntimeError("no records yet")

    report = ops.status_report(inst, FakeIO(), shadow=broken)
    assert report["automations"][0]["level"] == "L1"
    assert report["automations"][0]["shadow_report"] == {"error": "RuntimeError: no records yet"}


def test_the_ops_commands_are_registered():
    parser = build_parser()
    for argv, command in (
        (["ops", "status", "--json"], "status"), (["ops", "halt"], "halt"), (["ops", "resume"], "resume"),
        (["ops", "pending"], "pending"), (["ops", "approve", "esc-1", "--note", "n", "--as", "bob"], "approve"),
        (["ops", "reject", "esc-1"], "reject"),
    ):
        args = parser.parse_args(argv)
        assert args.command == "ops" and args.command_name == command
    approve = parser.parse_args(["ops", "approve", "esc-1", "--as", "bob", "--note", "n"])
    assert (approve.escalation_id, approve.as_user, approve.note) == ("esc-1", "bob", "n")


def test_dsn_comes_from_the_flag_then_the_environment(monkeypatch, tmp_path):
    parser = build_parser()
    monkeypatch.setenv("OKG_DSN", "postgresql://env")
    assert ops.resolve_dsn(parser.parse_args(["ops", "--dsn", "postgresql://flag", "pending"])) == "postgresql://flag"
    assert ops.resolve_dsn(parser.parse_args(["ops", "pending"])) == "postgresql://env"
    monkeypatch.delenv("OKG_DSN")
    with pytest.raises(ops.OpsCliError, match="no database"):
        ops.resolve_dsn(parser.parse_args(["ops", "--instance", str(tmp_path), "pending"]))


def test_submit_passes_the_idempotency_key(monkeypatch):
    from archi.ops import record

    seen = {}

    class Done:
        returncode, stdout, stderr = 0, json.dumps({"output": {"status": "accepted"}}), ""

    monkeypatch.setattr(record.subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd) and Done())
    records = record.OpsRecords()
    records.add("incident", record_id="i1", title="t", entity="e")
    record.submit(records, deployment="dep", idempotency_key="wf|record-1", max_wait_seconds=20)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--idempotency-key") + 1] == "wf|record-1"
    assert cmd[cmd.index("--max-wait-seconds") + 1] == "20"


def test_ops_yaml_names_the_promotion_change_and_a_hand_written_promoted_at_is_refused(instance):
    inst, _, _ = instance
    write_instance(inst, {"fixture-triage": "L2"},
                   {"automations": {"fixture-triage": {"mode": "live", "promotion": "promote-x"}}})
    rt = load_runtime({"ARCHI_INSTANCE": str(inst), "OKG_DSN": "postgresql://x"})
    config = rt.automation("fixture-triage")
    assert (config.mode, config.promotion_change, config.promoted_at) == ("live", "promote-x", None)
    write_instance(inst, ops_yaml={"automations": {"fixture-triage": {"mode": "live",
                                                                       "promoted_at": "2026-10-25T00:00:00Z"}}})
    with pytest.raises(RuntimeConfigError, match="unknown keys"):
        read_settings(inst)
