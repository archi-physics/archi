"""The workflow module's own logic, driven without a worker: the guard inside the mediator step,
the at-most-once attempt marker, the scanner, and the thin wrappers around the steps.

These import `archi.ops.workflow`, which imports okg and dbos (CI installs the okg wheel)."""
import inspect
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("okg.deployment")
pytest.importorskip("dbos")

from archi.ops import workflow as wf  # noqa: E402
from archi.ops.control import RunInfo  # noqa: E402
from archi.ops.escalation import Escalation  # noqa: E402
from archi.ops.loop import BudgetExceeded, MediationRefused, Signal  # noqa: E402
from archi.ops.proposers import ProposeRequest  # noqa: E402
from archi.ops.runtime_config import AutomationSettings  # noqa: E402

T0 = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
OPENED, DEADLINE = T0, T0 + timedelta(seconds=600)
ESC = {"escalation_id": "esc-1", "proposal_id": "prop-1", "opened_at": OPENED.isoformat(),
       "timeout_at": DEADLINE.isoformat()}
KEY = "triage~sig-1~1"


def row(*, status="applied", verdict="approve", by="alice", at=T0 + timedelta(seconds=60), subject="ops_proposal:prop-1"):
    attrs = {"status": status, "decision": {"verdict": verdict}, "decided_by": by, "decided_at": at.isoformat()}
    return subject, attrs


class FakeIO:
    """Just the OpsIO methods run_mediation uses."""

    def __init__(self, escalation_row=None, attempts=None, read_raises=None):
        self.row = escalation_row if escalation_row is not None else row()
        self.attempts = dict(attempts or {})
        self.read_raises = read_raises
        self.begun, self.finished = [], []

    def mediation(self, key):
        return self.attempts.get(key)

    def decision_for(self, escalation_id):
        from archi.ops.escalation import decision_from_row

        if self.read_raises:
            raise self.read_raises
        return decision_from_row(escalation_id, *self.row)

    def begin_mediation(self, key, proposal_id):
        self.begun.append((key, proposal_id))
        if key in self.attempts:
            return self.attempts[key]
        self.attempts[key] = {"status": "attempted", "result": None}
        return None

    def finish_mediation(self, key, status, result):
        self.finished.append((key, status, result))
        self.attempts[key] = {"status": status, "result": result}


class Tool:
    def __init__(self, result=None, raises=None):
        self.calls, self.result, self.raises = [], result or {"status": "succeeded", "result": "done"}, raises

    def __call__(self, *, payload, context):
        self.calls.append((payload, context))
        if self.raises:
            raise self.raises
        return self.result


def mediate(io, tool=None, *, halt=lambda io: None, now=T0 + timedelta(seconds=120), esc=ESC, entry=lambda: "m:f"):
    tool = tool or Tool()
    refusals = []
    out = wf.run_mediation(
        io, mediator_entry=entry, action_class="write_marker", payload={"entity": "T2"}, proposal_id="prop-1",
        idempotency_key=KEY, esc=esc, workflow_id=KEY, dsn="postgresql://x", halt=halt, now=lambda: now,
        resolve=lambda entry: tool, refused=lambda verdict, reason: refusals.append((verdict, reason)))
    return out, tool, refusals


# --- the guard inside the mediator step ---------------------------------------------------


def test_an_approval_in_time_runs_the_tool_once_with_the_key_and_records_the_attempt():
    io = FakeIO()
    out, tool, refusals = mediate(io)
    assert out == {"status": "succeeded", "result": "done"} and refusals == []
    (payload, context), = tool.calls
    assert payload == {"entity": "T2"}
    assert context == {"dsn": "postgresql://x", "proposal_id": "prop-1", "idempotency_key": KEY, "workflow_id": KEY}
    assert io.begun == [(KEY, "prop-1")] and io.finished == [(KEY, "succeeded", "done")]


def test_halt_refuses_before_anything_is_recorded_or_called():
    io = FakeIO()
    out, tool, refusals = mediate(io, halt=lambda io: "the archi_ops_scan schedule is paused (archi ops halt)")
    assert out == {"refused": ["halted", "the archi_ops_scan schedule is paused (archi ops halt)"]}
    assert refusals == [("halted", "the archi_ops_scan schedule is paused (archi ops halt)")]
    assert tool.calls == [] and io.begun == []


def test_an_approval_made_late_since_the_loop_looked_is_refused_as_timed_out():
    io = FakeIO(row(at=DEADLINE + timedelta(seconds=1)))
    out, tool, refusals = mediate(io, now=DEADLINE + timedelta(seconds=30))
    assert out["refused"][0] == "timed_out" and "after the deadline" in out["refused"][1]
    assert refusals and tool.calls == [] and io.begun == []


@pytest.mark.parametrize("escalation_row, fragment", [
    (row(verdict="reject"), "decision is rejected"),            # superseded by a rejection
    (row(status="timed_out"), "decision is timed_out"),
    (row(status="pending"), "decision is error"),               # not decided at all
    (row(by=None), "names no person"),
    (row(subject="ops_proposal:prop-9"), "decision is for proposal 'prop-9'"),
    (row(at=OPENED - timedelta(seconds=5)), "before the escalation opened"),
])
def test_a_changed_or_superseded_escalation_row_is_refused(escalation_row, fragment):
    io = FakeIO(escalation_row)
    out, tool, refusals = mediate(io)
    assert out["refused"][0] == "error" and fragment in out["refused"][1]
    assert refusals and tool.calls == [] and io.begun == []


def test_no_escalation_is_refused():
    out, tool, _ = mediate(FakeIO(), esc=None)
    assert out == {"refused": ["error", "no escalation was opened for this proposal"]} and tool.calls == []


@pytest.mark.parametrize("make", [
    lambda: dict(halt=lambda io: (_ for _ in ()).throw(RuntimeError("halt table gone"))),
    lambda: dict(io=FakeIO(read_raises=RuntimeError("db down"))),
    lambda: dict(entry=lambda: (_ for _ in ()).throw(KeyError("no such action class"))),
])
def test_an_exception_while_checking_is_an_error_refusal_not_a_failed_action(make):
    kwargs = make()
    io = kwargs.pop("io", FakeIO())
    out, tool, refusals = mediate(io, **kwargs)
    assert out["refused"][0] == "error" and "could not verify it may act" in out["refused"][1]
    assert refusals and tool.calls == [] and io.begun == [] and io.finished == []


# --- at most once across replay ----------------------------------------------------------------


def test_a_replay_that_finds_an_attempt_with_no_end_does_not_call_the_tool_again():
    io = FakeIO(attempts={KEY: {"status": "attempted", "result": None}})
    out, tool, refusals = mediate(io, halt=lambda io: pytest.fail("a replay must not even re-check halt"))
    assert out == {"status": "unknown", "result": "possible duplicate prevented; check the external system"}
    assert tool.calls == [] and io.begun == [] and refusals == []


def test_a_replay_of_a_finished_attempt_returns_the_recorded_result():
    io = FakeIO(attempts={KEY: {"status": "failed", "result": "exit 1"}})
    out, tool, _ = mediate(io)
    assert out == {"status": "failed", "result": "exit 1"} and tool.calls == []


def test_two_attempts_racing_for_one_key_call_the_tool_once():
    class Racy(FakeIO):
        def begin_mediation(self, key, proposal_id):
            return {"status": "attempted", "result": None}  # someone else got there first

    out, tool, _ = mediate(Racy())
    assert out["status"] == "unknown" and tool.calls == []


def test_a_tool_that_raises_is_a_failed_action_with_the_attempt_closed():
    io = FakeIO()
    out, tool, _ = mediate(io, Tool(raises=RuntimeError("exit 3")))
    assert out == {"status": "failed", "result": "RuntimeError: exit 3"}
    assert io.finished == [(KEY, "failed", "RuntimeError: exit 3")]


def test_a_second_run_with_the_same_key_after_a_clean_finish_does_not_call_again():
    io = FakeIO()
    first, tool, _ = mediate(io)
    second, tool2, _ = mediate(io)
    assert first == second == {"status": "succeeded", "result": "done"} and tool2.calls == []


def test_the_log_line_names_the_step_as_the_source_of_the_refusal(capsys, caplog):
    wf.log_refusal("halted", "the schedule is paused")
    assert "archi_ops_mediate refused (halted): the schedule is paused" in capsys.readouterr().err


# --- the wrappers around the steps ----------------------------------------------------------


def settings(**kw):
    return SimpleNamespace(settings={"triage": AutomationSettings("triage", **kw)})


def test_the_mediator_wrapper_raises_a_refusal_and_maps_an_unknown_result(monkeypatch):
    monkeypatch.setattr(wf, "_runtime", lambda: settings())
    escalation = Escalation("esc-1", "prop-1", OPENED, DEADLINE)
    seen = {}

    def step(automation, action_class, payload, proposal_id, key, esc, level):
        seen["esc"] = esc
        return {"refused": ["halted", "paused"]}

    monkeypatch.setattr(wf, "_mediate_step", step)
    mediator = wf._Mediator("triage")
    with pytest.raises(MediationRefused) as excinfo:
        mediator.execute(action_class="a", payload={}, proposal_id="prop-1", idempotency_key=KEY, generation="g",
                         escalation=escalation)
    assert (excinfo.value.verdict, excinfo.value.reason) == ("halted", "paused")
    assert seen["esc"] == ESC
    monkeypatch.setattr(wf, "_mediate_step", lambda *a: {"status": "unknown", "result": wf.DUPLICATE_PREVENTED})
    result = mediator.execute(action_class="a", payload={}, proposal_id="prop-1", idempotency_key=KEY, generation="g")
    assert (result.status, result.result) == ("unknown", wf.DUPLICATE_PREVENTED)


def test_the_test_gate_waits_between_the_decision_and_the_step_only_when_asked(monkeypatch):
    order = []
    monkeypatch.setattr(wf.DBOS, "recv", lambda **kw: order.append(("recv", kw)), raising=False)
    monkeypatch.setattr(wf, "_mediate_step", lambda *a: order.append("step") or {"status": "succeeded", "result": ""})
    mediator = wf._Mediator("triage")
    monkeypatch.setattr(wf, "_runtime", lambda: settings())
    mediator.execute(action_class="a", payload={}, proposal_id="p", idempotency_key=KEY, generation="g")
    assert order == ["step"]
    order.clear()
    monkeypatch.setattr(wf, "_runtime", lambda: settings(test_gate_seconds=90))
    mediator.execute(action_class="a", payload={}, proposal_id="p", idempotency_key=KEY, generation="g")
    assert order == [("recv", {"topic": "archi_ops_test_gate", "timeout_seconds": 90.0}), "step"]


def test_a_charge_the_cap_refuses_raises_budget_exceeded(monkeypatch):
    monkeypatch.setattr(wf.DBOS, "workflow_id", "wid", raising=False)
    monkeypatch.setattr(wf, "_budget_record_step", lambda scope, cost, cid: cost < 1)
    wf._Budget().record("run:x", 0.5)
    with pytest.raises(BudgetExceeded):
        wf._Budget().record("run:x", 2.0)


def test_demotions_come_back_as_datetimes(monkeypatch):
    monkeypatch.setattr(wf, "_demotion_step", lambda a, g: {"at": "2026-10-25T11:00:00Z", "reason": "demoted to shadow: x"})
    d = wf._Demotions().latest("triage", "gen-1")
    assert d.at == datetime(2026, 10, 25, 11, tzinfo=timezone.utc) and d.reason == "demoted to shadow: x"
    monkeypatch.setattr(wf, "_demotion_step", lambda a, g: None)
    assert wf._Demotions().latest("triage", "gen-1") is None


def test_the_decider_reads_the_row_in_a_step_and_treats_a_timeout_as_the_rows_word(monkeypatch):
    class Timeout(Exception):
        pass

    class StateError(Exception):
        pass

    monkeypatch.setattr(wf, "_private_okg", lambda: (None, Timeout, None, StateError))
    monkeypatch.setattr(wf, "request_escalation", lambda **kw: (_ for _ in ()).throw(Timeout()))
    monkeypatch.setattr(wf, "_Clock", lambda: SimpleNamespace(now=lambda: T0))
    subject, attrs = row(status="timed_out", verdict="approve")
    monkeypatch.setattr(wf, "_decision_step", lambda esc_id: {"found": True, "subject": subject, "attrs": attrs})
    decision = wf._Decider(None).wait(Escalation("esc-1", "prop-1", OPENED, DEADLINE))
    assert (decision.status, decision.proposal_id, decision.escalation_id) == ("timed_out", "prop-1", "esc-1")


def test_the_escalation_id_follows_okgs_rule():
    assert wf.expected_escalation_id("triage~sig-1~1").startswith("esc-") and len(wf.expected_escalation_id("x")) == 28


# --- the scanner -------------------------------------------------------------------------------


class ScanIO:
    def __init__(self, signals, runs=(), paused=False):
        self.signals, self._runs, self.paused = dict(signals), list(runs), paused
        self.marked, self.ensured = [], 0

    def ensure_tables(self):
        self.ensured += 1

    def schedules(self):
        return {"archi_ops_scan": self.paused}

    def runs(self):
        return list(self._runs)

    def new_signals(self, automation):
        return list(self.signals.get(automation, []))

    def mark_signal_started(self, automation, signal_id):
        self.marked.append((automation, signal_id, "started"))

    def mark_signal(self, automation, signal_id, status):
        self.marked.append((automation, signal_id, status))


def sig(id, attempt=1):
    return {"id": id, "entity": "T2", "title": "t", "source": "table", "summary": None, "attempt": attempt}


def rt(*names):
    return SimpleNamespace(settings={n: object() for n in names})


def test_the_scanner_starts_one_run_per_idle_automation_and_leaves_the_rest_waiting():
    io = ScanIO({"a": [sig("s1"), sig("s2")], "b": [sig("s3")]})
    started = []
    out = wf.scan_once(rt("a", "b"), io, lambda run_id, name, signal: started.append((run_id, name, signal["id"])),
                       say=lambda m: None)
    assert out == ["a~s1~1", "b~s3~1"] and started == [("a~s1~1", "a", "s1"), ("b~s3~1", "b", "s3")]
    assert io.marked == [("a", "s1", "started"), ("b", "s3", "started")]


def test_a_run_in_flight_holds_back_its_automations_next_signal():
    io = ScanIO({"a": [sig("s1")], "b": [sig("s3")]}, runs=[RunInfo("a~s0~1", "a", "PENDING")])
    out = wf.scan_once(rt("a", "b"), io, lambda *a: None, say=lambda m: None)
    assert out == ["b~s3~1"]


def ticket_rt(*names):
    """Automations with a ticket_queue trigger (one run at a time per ticket)."""
    auto = SimpleNamespace(trigger={"type": "ticket_queue", "system": ["redmine"], "queue": "*"})
    return SimpleNamespace(settings={n: object() for n in names}, automation=lambda n: auto)


def tsig(id, entity):
    return {**sig(id), "entity": entity, "source": "ticket"}


def test_a_ticket_run_waiting_for_approval_holds_back_only_its_own_ticket():
    """#746 finding 7: one L2 run waiting for a person blocked the whole ticket queue."""
    io = ScanIO({"t": [tsig("s1", "redmine:1"), tsig("s2", "redmine:2")]},
                runs=[RunInfo("t~s0~1", "t", "PENDING", "redmine:1")])
    said = []
    assert wf.scan_once(ticket_rt("t"), io, lambda *a: None, say=said.append) == ["t~s2~1"]
    io = ScanIO({"t": [tsig("s1", "redmine:1")]}, runs=[RunInfo("t~s0~1", "t", "PENDING", "redmine:1")])
    assert wf.scan_once(ticket_rt("t"), io, lambda *a: None, say=said.append) == []
    assert any("wait for a run in progress on the same ticket" in s for s in said)


def test_a_ticket_automation_has_at_most_a_few_runs_in_flight():
    runs = [RunInfo(f"t~s{i}~1", "t", "PENDING", f"redmine:{i}") for i in range(wf.TICKET_RUNS_IN_FLIGHT)]
    io = ScanIO({"t": [tsig("s9", "redmine:9")]}, runs=runs)
    assert wf.scan_once(ticket_rt("t"), io, lambda *a: None, say=lambda m: None) == []
    io = ScanIO({"t": [tsig("s9", "redmine:9")]}, runs=runs[1:])
    assert wf.scan_once(ticket_rt("t"), io, lambda *a: None, say=lambda m: None) == ["t~s9~1"]


def test_an_unreadable_automation_file_keeps_one_run_at_a_time():
    def broken(name):
        raise OSError("gone")

    rt = SimpleNamespace(settings={"t": object()}, automation=broken)
    io = ScanIO({"t": [tsig("s2", "redmine:2")]}, runs=[RunInfo("t~s0~1", "t", "PENDING", "redmine:1")])
    assert wf.scan_once(rt, io, lambda *a: None, say=lambda m: None) == []




def test_the_scanner_makes_the_archi_ops_tables_before_anything_else_even_when_paused():
    io = ScanIO({"a": [sig("s1")]}, paused=True)
    wf.scan_once(rt("a"), io, lambda *a: None, say=lambda m: None)
    assert io.ensured == 1


def test_a_paused_scanner_starts_nothing():
    io = ScanIO({"a": [sig("s1")]}, paused=True)
    assert wf.scan_once(rt("a"), io, lambda *a: pytest.fail("started while halted"), say=lambda m: None) == []


def test_a_signal_id_that_would_collide_is_refused_not_started():
    io = ScanIO({"a": [sig("s~1")]})
    assert wf.scan_once(rt("a"), io, lambda *a: pytest.fail("started"), say=lambda m: None) == []
    assert io.marked == [("a", "s~1", "invalid")]


def test_a_requeued_attempt_makes_a_new_run_id():
    io = ScanIO({"a": [sig("s1", attempt=2)]})
    assert wf.scan_once(rt("a"), io, lambda *a: None, say=lambda m: None) == ["a~s1~2"]


def test_start_workflow_is_never_given_a_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(wf, "DBOS", SimpleNamespace(start_workflow=lambda *a, **kw: calls.append((a, kw))))
    class Ctx:
        def __init__(self, run_id):
            calls.append(("id", run_id))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(wf, "SetWorkflowID", Ctx)
    wf.start_run("a~s1~1", "a", {"id": "s1"})
    assert calls == [("id", "a~s1~1"), ((wf.archi_ops_run, "a", {"id": "s1"}), {})]  # no kwargs: no queue
    source = inspect.getsource(wf)
    for forbidden in (".enqueue(", "Queue(", "queue_name", "queue=", "okg-{"):
        assert forbidden not in source.replace("okg-<deployment>", "")


def test_a_replay_of_an_applied_escalation_still_reads_the_row(monkeypatch):
    """okg refuses applied -> applied when request_escalation runs again after a restart (O14)."""
    class StateError(Exception):
        pass

    monkeypatch.setattr(wf, "_private_okg", lambda: (None, RuntimeError, None, StateError))
    monkeypatch.setattr(wf, "request_escalation", lambda **kw: (_ for _ in ()).throw(StateError("applied -> applied")))
    monkeypatch.setattr(wf, "_Clock", lambda: SimpleNamespace(now=lambda: T0))
    subject, attrs = row(status="applied")
    monkeypatch.setattr(wf, "_decision_step", lambda esc_id: {"found": True, "subject": subject, "attrs": attrs})
    decision = wf._Decider(None).wait(Escalation("esc-1", "prop-1", OPENED, DEADLINE))
    assert (decision.status, decision.decided_by) == ("approved", "alice")


def test_any_other_error_from_request_escalation_still_propagates_to_the_loop(monkeypatch):
    monkeypatch.setattr(wf, "_private_okg", lambda: (None, ValueError, None, KeyError))
    monkeypatch.setattr(wf, "request_escalation", lambda **kw: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(wf, "_Clock", lambda: SimpleNamespace(now=lambda: T0))
    with pytest.raises(RuntimeError, match="db down"):
        wf._Decider(None).wait(Escalation("esc-1", "prop-1", OPENED, DEADLINE))


def test_only_the_applied_to_applied_replay_is_tolerated(monkeypatch):
    class StateError(Exception):
        pass

    monkeypatch.setattr(wf, "_private_okg", lambda: (None, RuntimeError, None, StateError))
    monkeypatch.setattr(wf, "_Clock", lambda: SimpleNamespace(now=lambda: T0))

    def wait_with(status, message):
        monkeypatch.setattr(wf, "request_escalation", lambda **kw: (_ for _ in ()).throw(StateError(message)))
        subject, attrs = row(status=status)
        monkeypatch.setattr(wf, "_decision_step", lambda esc_id: {"found": True, "subject": subject, "attrs": attrs})
        return wf._Decider(None).wait(Escalation("esc-1", "prop-1", OPENED, DEADLINE))

    assert wait_with("applied", "escalation esc-1: applied -> applied is illegal").status == "approved"
    for status in ("decided", "pending", "timed_out"):
        with pytest.raises(StateError):
            wait_with(status, "escalation esc-1 is in status 'decided'; expected applied")
    with pytest.raises(StateError):  # an applied row but an error that is about something else
        wait_with("applied", "escalation esc-1 is in status 'pending'; expected decided")


def test_is_applied_replay():
    assert wf.is_applied_replay(Exception("applied -> applied"), {"attrs": {"status": "applied"}})
    assert not wf.is_applied_replay(Exception("applied -> applied"), {"attrs": {"status": "decided"}})
    assert not wf.is_applied_replay(Exception("anything"), {"attrs": {"status": "applied"}})
    assert not wf.is_applied_replay(Exception("applied"), {"attrs": None})


def test_promotion_is_read_from_the_ledger_through_a_step(monkeypatch):
    monkeypatch.setattr(wf, "_promotion_step", lambda a: {"approved": True, "at": "2026-10-25T11:00:00+00:00", "reason": ""})
    state = wf._Promotions().check("triage", "promote-x")
    assert state.approved and state.at == datetime(2026, 10, 25, 11, tzinfo=timezone.utc)
    monkeypatch.setattr(wf, "_promotion_step", lambda a: {"approved": False, "at": None, "reason": "awaiting_review"})
    state = wf._Promotions().check("triage", "promote-x")
    assert (state.approved, state.at, state.reason) == (False, None, "awaiting_review")


def test_a_demotion_is_logged_through_a_step(monkeypatch):
    seen = []
    monkeypatch.setattr(wf, "_demotion_record_step", lambda *a: seen.append(a))
    wf._Demotions().record("triage", "triage~s~1", T0, "demoted to shadow: x")
    assert seen == [("triage", "triage~s~1", T0.isoformat(), "demoted to shadow: x")]


def test_the_demotion_step_takes_the_later_of_the_log_and_the_published_history(monkeypatch):
    class Io:
        def __init__(self, logged, published):
            self.logged, self.published = logged, published

        def latest_demotion(self, automation):
            return self.logged

        def read_nodes(self, subtype, generation, where):
            return [{"attrs": {"mode_reason": r, "proposed_at": at}} for r, at in self.published]

    def step(io):
        monkeypatch.setattr(wf, "_io", lambda rt: io)
        monkeypatch.setattr(wf, "_runtime", lambda: None)
        return wf._demotion_step("triage", "gen-1")

    old = ("demoted to shadow: old", "2026-10-25T10:00:00Z")
    new_logged = {"at": "2026-10-25T11:00:00+00:00", "reason": "demoted to shadow: logged"}
    assert step(Io(new_logged, [old]))["reason"] == "demoted to shadow: logged"
    assert step(Io(None, [old, ("not a demotion", "2026-10-25T12:00:00Z")]))["reason"] == "demoted to shadow: old"
    assert step(Io({"at": "2026-10-25T09:00:00+00:00", "reason": "demoted to shadow: stale log"}, [old]))["reason"] == "demoted to shadow: old"
    assert step(Io(None, [])) is None


# --- the gate step's context-eval check (PACT r3-context-quality) -------------------------------


def gate_step_on(monkeypatch, tmp_path, report=None):
    """`_gate_step` for a gate that asks only for a context-eval score, on an instance whose only bundle
    is `demo`, with `report` (a dict, or raw text) as its newest context-eval run."""
    from archi.ops.context_gate import GateSpec
    from archi.stack.instance import Instance, Ports

    inst = Instance(dir=tmp_path / "inst", name="demo", bundles=["demo"], data_root=tmp_path / "data",
                    runtime="podman", ports=Ports(graph_db=1, mcp=2, webui=3))
    inst.dir.mkdir()
    inst.save()
    if report is not None:
        run = tmp_path / "data" / "eval" / "context" / "demo" / "20261020T120000Z"
        run.mkdir(parents=True)
        (run / "report.json").write_text(report if isinstance(report, str) else json.dumps(report))
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.5})
    rt = SimpleNamespace(instance_dir=inst.dir, automation=lambda name: SimpleNamespace(gate=spec))
    monkeypatch.setattr(wf, "_runtime", lambda: rt)
    return wf._gate_step("triage")


def good_report(finished_at):
    return {"schema": "archi-context-eval-report/1", "bundle": "demo", "status": "pass", "errors": 0,
            "score": 0.75, "arms": {"graph": {"errors": 0, "tool": "search", "arguments": {}}},
            "k": 10, "questions_sha256": "sha256:" + "0" * 64,
            "question_count": 1, "questions": [{"id": "q"}], "finished_at": finished_at, "generation": "41"}


def test_the_gate_step_passes_on_a_recent_good_context_eval(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = gate_step_on(monkeypatch, tmp_path, good_report(now))
    assert result["ok"] and result["checks"] == [
        {"name": "context_eval", "ok": True, "detail": "score 0.75 meets the minimum 0.5"}]


def test_the_gate_step_fails_an_old_context_eval_by_name(monkeypatch, tmp_path):
    result = gate_step_on(monkeypatch, tmp_path, good_report("2026-01-01T00:00:00Z"))
    assert result["failing"] == ["context_eval"] and "more than the 14 days" in result["message"]


def test_the_gate_step_fails_an_unreadable_context_eval_without_raising(monkeypatch, tmp_path):
    result = gate_step_on(monkeypatch, tmp_path, "{not json")
    assert result["failing"] == ["context_eval"] and "cannot be read" in result["message"]


def test_the_gate_step_fails_when_no_context_eval_ran(monkeypatch, tmp_path):
    result = gate_step_on(monkeypatch, tmp_path)
    assert result["failing"] == ["context_eval"] and "run `archi eval --context`" in result["message"]
