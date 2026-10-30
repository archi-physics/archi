"""req.r3-ops-self-monitoring.health: the signals the instance-health automation reads, as plain data."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from archi.health import signals
from archi.health.config import HealthConfig
from archi.health.problems import Problem, ProblemError, make_key, slug
from health_eval_reports import scored, write_report

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def doctor(*checks):
    return {"ok": not any(c["status"] == "fail" for c in checks), "instance": "/i", "checks": list(checks)}


def check(id_, status, message="m", remediation="", footgun="", data=None):
    return {"id": id_, "status": status, "message": message, "remediation": remediation, "footgun": footgun,
            "data": data or {}}


def freshness(*rows):
    return check("source-freshness", "fail", "2 of 3 sources over their max_staleness", "okg ingest",
                 data={"sources": list(rows)})


def row(source, ok, last="2026-09-20T10:00:00+00:00", limit="24h", age="35d", reason="last run 35d ago, over the 24h limit"):
    return {"source": source, "ok": ok, "last_reconcile_at": last, "max_staleness": limit, "age": age, "reason": reason}


# --- keys ------------------------------------------------------------------------


def test_a_key_names_the_check_and_the_subject_and_a_file():
    assert make_key("source-freshness", "jira.jira") == "source-freshness/jira.jira"
    weird = make_key("services", "weird name!")
    assert weird.startswith("services/weird_name-") and weird == make_key("services", "weird name!")
    assert slug("source-freshness/jira.jira") == "source-freshness--jira.jira"
    with pytest.raises(ProblemError):
        slug("no-subject")
    with pytest.raises(ProblemError):
        Problem(key="a/b", family="nonsense", title="t", detail="d")


# --- doctor ------------------------------------------------------------------------


def test_each_late_source_is_its_own_problem_and_the_fresh_ones_are_not():
    report = doctor(freshness(row("jira.jira", False), row("docsite.docs", True, age="1h"),
                              row("twiki.twiki", False, last=None, reason="no completed run")))
    problems = signals.doctor_problems(report)
    assert [p.key for p in problems] == ["source-freshness/jira.jira", "source-freshness/twiki.twiki"]
    assert all(p.family == "doctor" and p.severity == "error" for p in problems)
    assert "2026-09-20T10:00:00+00:00" in problems[0].detail and "24h" in problems[0].detail
    assert "none recorded" in problems[1].detail


def test_a_late_source_issue_text_has_no_age_so_it_does_not_change_between_runs():
    a = signals.doctor_problems(doctor(freshness(row("jira.jira", False, age="35d"))))
    b = signals.doctor_problems(doctor(freshness(row("jira.jira", False, age="36d", reason="last run 36d ago"))))
    assert a == b


def test_other_failing_checks_are_one_problem_each_with_the_remediation():
    report = doctor(check("services", "fail", "mcp: absent", "archi up", "FG-16"), check("published", "pass"),
                    check("okg-lint", "skip"), check("data-root", "warn", "slow"))
    problems = signals.doctor_problems(report)
    assert [p.key for p in problems] == ["services/instance"]
    assert "archi up" in problems[0].detail and "FG-16" in problems[0].detail


def test_warnings_count_only_when_asked_and_ignored_checks_never():
    report = doctor(check("data-root", "warn", "slow"), check("services", "fail", "down"))
    assert [p.key for p in signals.doctor_problems(report, include_warnings=True)] == ["data-root/instance", "services/instance"]
    assert signals.doctor_problems(report, include_warnings=True)[0].severity == "warning"
    assert [p.key for p in signals.doctor_problems(report, ignore=["services"])] == []


# --- ingest ------------------------------------------------------------------------


def test_a_blocked_frontier_is_an_ingest_problem_and_a_fresh_one_is_not():
    report = {"rows": [
        {"registry_source_name": "jira.jira", "frontier_state": "blocked", "last_attempt_status": "FAILED",
         "last_attempt_run_id": "run-9", "last_error": "401 from jira", "read_visibility_state": "last_run_failed"},
        {"registry_source_name": "docsite.docs", "frontier_state": "fresh"},
        {"registry_source_name": "twiki.twiki", "frontier_state": "publish_pending"},
    ]}
    (problem,) = signals.frontier_problems(report)
    assert problem.key == "ingest/jira.jira" and problem.family == "ingest"
    assert "401 from jira" in problem.detail and "run-9" in problem.detail


# --- automation errors ---------------------------------------------------------------


def decisions(automation, verdicts, hours_ago=1):
    when = (NOW - timedelta(hours=hours_ago)).isoformat()
    return [{"node_id": f"ops_decision:{automation}-{i}", "attrs": {"automation": automation, "verdict": v, "decided_at": when}}
            for i, v in enumerate(verdicts)]


def reader(rows):
    return lambda subtype: rows if subtype == "ops_decision" else []


def test_an_automation_whose_decisions_are_mostly_errors_is_a_problem():
    rows = decisions("triage", ["error", "error", "observed", "invalid_proposal"]) + decisions("health", ["observed"] * 5)
    (problem,) = signals.automation_error_problems(reader(rows), NOW)
    assert problem.key == "automation-errors/triage"
    assert "3 of the last 4" in problem.detail


def test_too_few_decisions_or_old_ones_do_not_make_a_rate():
    assert signals.automation_error_problems(reader(decisions("t", ["error", "error"])), NOW) == []
    old = decisions("t", ["error"] * 5, hours_ago=48)
    assert signals.automation_error_problems(reader(old), NOW) == []
    capped = decisions("t", ["budget_exhausted"] * 5 + ["halted"] * 5)
    assert signals.automation_error_problems(reader(capped), NOW) == []


# --- eval -------------------------------------------------------------------------
# Fixture report.json files only (health_eval_reports.write_report): no eval runs here.


def runs(directory, *passed_counts, bundle="cern-team", model="qwen3:8b", threshold=7):
    """One run per count (None: execution_failed), a day apart, kept as `archi eval --out <dir>/<bundle>/<date>`."""
    for day, passed in enumerate(passed_counts, start=1):
        statuses = ["execution_failed"] * 10 if passed is None else scored(passed)
        write_report(directory, bundle=bundle, model=model, threshold=threshold, started_at=f"2026-10-{day:02d}T03:00:00Z",
                     statuses=statuses, error="MCP endpoint not usable: refused" if passed is None else None,
                     sub=f"2026-10-{day:02d}")


def test_with_no_eval_report_the_signal_is_skipped_and_says_so(tmp_path):
    problems, notes = signals.eval_problems(tmp_path / "eval")
    assert problems == [] and [n["signal"] for n in notes] == ["eval"] and "no eval report exists" in notes[0]["reason"]


def test_a_latest_run_below_the_bundle_threshold_is_an_error_even_with_no_history(tmp_path):
    write_report(tmp_path, started_at="2026-10-26T03:00:00Z", statuses=scored(5))  # the archi eval default place
    (problem,), notes = signals.eval_problems(tmp_path)
    assert problem.key == "eval/cern-team" and problem.severity == "error"
    assert problem.title == "Eval of cern-team is below its threshold"
    assert "passed 5 of 10" in problem.detail and "threshold of 7" in problem.detail
    assert "cern-team/report.json" in problem.detail
    assert [n["signal"] for n in notes] == ["eval-trend"] and "no earlier run" in notes[0]["reason"]


def test_a_passing_run_with_no_history_is_healthy_and_says_the_trend_was_not_judged(tmp_path):
    write_report(tmp_path, started_at="2026-10-26T03:00:00Z", statuses=scored(8))
    problems, notes = signals.eval_problems(tmp_path)
    assert problems == [] and [n["signal"] for n in notes] == ["eval-trend"]


def test_a_drop_beyond_the_margin_is_a_warning_while_the_threshold_still_holds(tmp_path):
    runs(tmp_path, 9, 9, 8, 7)  # mean of the three before: 0.87; latest 0.70, still at the threshold of 7
    (problem,), notes = signals.eval_problems(tmp_path)
    assert notes == [] and problem.key == "eval/cern-team" and problem.severity == "warning"
    assert problem.title == "Eval score of cern-team dropped"
    assert "0.17 below the mean score 0.87 of the 3 earlier run(s)" in problem.detail


def test_below_threshold_and_dropped_is_one_error_that_says_both(tmp_path):
    runs(tmp_path, 9, 6)
    (problem,), _ = signals.eval_problems(tmp_path)
    assert problem.severity == "error" and "threshold of 7" in problem.detail and "0.30 below" in problem.detail


def test_a_one_question_dip_is_within_the_default_margin_and_the_margin_is_a_setting(tmp_path):
    runs(tmp_path, 8, 7)  # 0.8 -> 0.7: exactly the default 0.1, which is not more than it
    assert signals.eval_problems(tmp_path) == ([], [])
    (problem,), _ = signals.eval_problems(tmp_path, min_drop=0.05)
    assert problem.severity == "warning" and "allowed drop is 0.05" in problem.detail


def test_only_the_last_three_earlier_runs_make_the_baseline(tmp_path):
    runs(tmp_path, 10, 10, 7, 7, 7, 7)  # the two 10s are older than the window
    assert signals.eval_problems(tmp_path) == ([], [])


def test_an_execution_failed_run_is_a_failed_signal_not_a_low_score(tmp_path):
    runs(tmp_path, 9, None)
    (problem,), _ = signals.eval_problems(tmp_path)
    assert problem.key == "eval/cern-team" and problem.severity == "error"
    assert problem.title == "Eval of cern-team could not run"
    assert "execution_failed" in problem.detail and "MCP endpoint not usable: refused" in problem.detail
    assert "failed health signal, not a low score" in problem.detail and "passed" not in problem.detail


def test_an_execution_failed_run_is_left_out_of_the_baseline(tmp_path):
    runs(tmp_path, 9, None, 9)  # the failed run in between is not a 0 that would make the 9 look like a rise
    assert signals.eval_problems(tmp_path) == ([], [])
    runs(tmp_path / "b", 9, None, 7)  # nor does it pull the baseline down and hide a real drop
    (problem,), _ = signals.eval_problems(tmp_path / "b")
    assert "0.20 below the mean score 0.90 of the 1 earlier run(s)" in problem.detail


def test_a_grown_question_set_is_not_compared_with_the_old_one(tmp_path):
    runs(tmp_path, 9, 9)  # 0.90 on ten questions
    write_report(tmp_path, started_at="2026-10-26T03:00:00Z", statuses=scored(9, total=12))  # 0.75, same smoke file
    problems, notes = signals.eval_problems(tmp_path)
    assert problems == [] and [n["signal"] for n in notes] == ["eval-trend"] and "same question set" in notes[0]["reason"]


def test_a_changed_prompt_hash_is_another_measurement(tmp_path):
    runs(tmp_path, 9, 9)
    write_report(tmp_path, started_at="2026-10-26T03:00:00Z", statuses=scored(7), prompt_hash="sha256:p2")
    problems, notes = signals.eval_problems(tmp_path)
    assert problems == [] and [n["signal"] for n in notes] == ["eval-trend"]


def test_questions_that_could_not_be_asked_are_named_and_not_read_as_wrong_answers(tmp_path):
    runs(tmp_path, 9, 9)
    write_report(tmp_path, started_at="2026-10-26T03:00:00Z",
                 statuses=["pass"] * 6 + ["execution_failed"] * 3 + ["fail"])  # a fail run, not execution_failed
    (problem,), _ = signals.eval_problems(tmp_path)
    assert problem.title == "Eval of cern-team is below its threshold" and "passed 6 of 10" in problem.detail
    assert "3 of the 10 questions could not be asked (`execution_failed`: q6, q7, q8)" in problem.detail
    assert "not wrong answers" in problem.detail


def test_a_run_with_questions_that_could_not_be_asked_is_never_a_baseline(tmp_path):
    runs(tmp_path, 9)
    write_report(tmp_path, started_at="2026-10-02T03:00:00Z", statuses=["pass"] * 7 + ["execution_failed"] * 3,
                 sub="2026-10-02")
    write_report(tmp_path, started_at="2026-10-03T03:00:00Z", statuses=scored(8), sub="2026-10-03")
    # with the partial run the mean would be 0.80 and 0.80 no drop; without it, 0.90 -> 0.80
    (problem,), _ = signals.eval_problems(tmp_path, min_drop=0.05)
    assert "0.10 below the mean score 0.90 of the 1 earlier run(s)" in problem.detail


def test_only_the_bundles_the_instance_lists_are_judged(tmp_path):
    runs(tmp_path, 4, bundle="cern-team")
    runs(tmp_path, 4, bundle="retired-bundle")
    (problem,), notes = signals.eval_problems(tmp_path, bundles=lambda: ["cern-team"])
    assert problem.key == "eval/cern-team"
    assert "eval-unlisted" in [n["signal"] for n in notes]
    assert "retired-bundle" in next(n["reason"] for n in notes if n["signal"] == "eval-unlisted")


def test_an_old_newest_run_is_a_note_not_an_issue(tmp_path):
    runs(tmp_path, 9, 9)  # newest started 2026-10-02, 23 days before NOW
    problems, notes = signals.eval_problems(tmp_path, now=NOW)
    assert problems == [] and [n["signal"] for n in notes] == ["eval-stale"] and "14 days" in notes[0]["reason"]
    assert signals.eval_problems(tmp_path, now=NOW, max_age_days=30) == ([], [])


def test_runs_of_another_model_are_not_compared(tmp_path):
    runs(tmp_path, 10, model="big-model")
    write_report(tmp_path, model="qwen3:8b", started_at="2026-10-26T03:00:00Z", statuses=scored(7))
    problems, notes = signals.eval_problems(tmp_path)
    assert problems == [] and [n["signal"] for n in notes] == ["eval-trend"] and "qwen3:8b" in notes[0]["reason"]


def test_each_bundle_is_its_own_problem(tmp_path):
    runs(tmp_path, 9, 9, bundle="cern-team")
    runs(tmp_path, 9, 4, bundle="cms-physics")
    (problem,), notes = signals.eval_problems(tmp_path)
    assert problem.key == "eval/cms-physics" and notes == []


def test_the_same_report_twice_is_one_run_and_the_text_is_stable(tmp_path):
    runs(tmp_path, 9, 6)
    first = signals.eval_problems(tmp_path)
    write_report(tmp_path, started_at="2026-10-02T03:00:00Z", statuses=scored(6), sub="copy")
    assert signals.eval_problems(tmp_path) == first  # no new run, no new text, so the issue is not rewritten


def test_a_report_json_of_another_kind_is_left_out(tmp_path):
    (tmp_path / "context").mkdir()
    (tmp_path / "context" / "report.json").write_text(json.dumps({"schema": "archi-context-report/1", "score": 0.1}))
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "report.json").write_text(json.dumps({"score": 0.1}))
    assert [n["signal"] for n in signals.eval_problems(tmp_path)[1]] == ["eval"]


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(schema="archi-eval-report/2"), "not one this reader knows"),
    (lambda d: d.pop("pass_threshold"), "not a usable eval report"),
    (lambda d: d.update(started_at=""), "started_at must be a non-empty string"),
    (lambda d: d.update(status="maybe"), "status 'maybe'"),
])
def test_an_eval_report_that_cannot_be_read_raises(tmp_path, mutate, message):
    path = write_report(tmp_path, started_at="2026-10-26T03:00:00Z", statuses=scored(8))
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        signals.eval_problems(tmp_path)


def test_the_reports_are_found_under_the_data_root_or_the_configured_directory(tmp_path):
    from archi.stack.instance import Instance, Ports

    inst = Instance(dir=tmp_path / "inst", name="demo", bundles=["cern-team"], data_root=tmp_path / "data",
                    runtime="docker", ports=Ports(graph_db=15432, mcp=18080, webui=13000))
    inst.dir.mkdir()
    inst.save()
    assert signals.eval_reports_dir(inst.dir) == tmp_path / "data" / "eval"
    assert signals.eval_reports_dir(inst.dir, "evals") == inst.dir / "evals"
    assert signals.eval_reports_dir(inst.dir, str(tmp_path / "x")) == tmp_path / "x"


def test_the_eval_settings_are_read_and_checked():
    from archi.health.tracker import TrackerError

    assert HealthConfig.from_proposer({}).eval_min_drop == 0.1 and HealthConfig().eval_reports_dir is None
    cfg = HealthConfig.from_proposer({"eval": {"min_drop": 0.2, "reports_dir": "/srv/evals"}})
    assert (cfg.eval_min_drop, cfg.eval_reports_dir) == (0.2, "/srv/evals")
    for bad in ({"min_drop": 2}, {"min_drop": "lots"}, {"reports_dir": ""}, {"history": 3}):
        with pytest.raises(TrackerError):
            HealthConfig.from_proposer({"eval": bad})


# --- collect --------------------------------------------------------------------------


def collect(tmp_path, *, doctor_report=None, frontier=None, rows=(), **kw):
    def fail(msg):
        def go():
            raise RuntimeError(msg)
        return go

    return signals.collect(
        HealthConfig(eval_reports_dir=kw.get("eval_dir", "eval")), instance_dir=tmp_path, now=NOW,
        read_doctor=kw.get("read_doctor") or (lambda: doctor_report or doctor()),
        read_frontier=kw.get("read_frontier") or (lambda: frontier or {"rows": []}),
        read_ops_records=reader(list(rows)),
        read_bundles=kw.get("read_bundles", lambda: ["cern-team", "cms-physics"]),
    )


def test_collect_gathers_every_signal_and_reports_the_eval_skip(tmp_path):
    obs = collect(tmp_path, doctor_report=doctor(freshness(row("jira.jira", False))))
    assert [p.key for p in obs.problems] == ["source-freshness/jira.jira"]
    assert obs.scope == ["doctor", "ingest", "automation-errors", "monitor"]  # eval was not judged
    assert [s["signal"] for s in obs.skipped] == ["eval"] and "no eval report" in obs.skipped[0]["reason"]


def test_a_signal_that_cannot_be_read_becomes_a_monitor_problem_and_leaves_its_scope(tmp_path):
    def boom():
        raise RuntimeError("okg is down")

    obs = collect(tmp_path, read_frontier=boom)
    assert [p.key for p in obs.problems] == ["monitor/ingest"]
    assert "okg is down" in obs.problems[0].detail
    assert "ingest" not in obs.scope and "doctor" in obs.scope and "monitor" in obs.scope


def test_a_malformed_eval_report_is_a_monitor_problem_not_a_skip(tmp_path):
    (tmp_path / "eval" / "cern-team").mkdir(parents=True)
    (tmp_path / "eval" / "cern-team" / "report.json").write_text("{not json")
    obs = collect(tmp_path)
    assert [p.key for p in obs.problems] == ["monitor/eval"] and obs.skipped == []
    assert "eval" not in obs.scope


def test_with_no_archi_yaml_the_default_reports_place_is_unknown_and_that_is_a_monitor_problem(tmp_path):
    obs = collect(tmp_path, eval_dir=None)
    assert [p.key for p in obs.problems] == ["monitor/eval"] and "archi.yaml" in obs.problems[0].detail


def test_an_eval_finding_is_judged_and_a_missing_trend_does_not_take_eval_out_of_scope(tmp_path):
    write_report(tmp_path / "eval", started_at="2026-10-26T03:00:00Z", statuses=scored(5))
    write_report(tmp_path / "eval", bundle="cms-physics", started_at="2026-10-26T04:00:00Z", statuses=scored(9))
    obs = collect(tmp_path)
    assert [p.key for p in obs.problems] == ["eval/cern-team"]
    assert obs.scope == ["doctor", "ingest", "automation-errors", "eval", "monitor"]
    assert [s["signal"] for s in obs.skipped] == ["eval-trend", "eval-trend"]  # each bundle has one run so far


def test_a_healthy_eval_run_lets_its_old_issue_close(tmp_path):
    runs(tmp_path / "eval", 5, 9)
    obs = collect(tmp_path)
    assert obs.problems == [] and "eval" in obs.scope
    assert [s["signal"] for s in obs.skipped] == ["eval-stale"]  # 2026-10-02 is 23 days before NOW: a note only


def test_a_removed_bundle_cannot_hold_its_issue_open(tmp_path):
    runs(tmp_path / "eval", 4, bundle="retired-bundle")
    obs = collect(tmp_path, read_bundles=lambda: ["cern-team"])
    assert obs.problems == [] and "eval" in obs.scope  # judged, nothing found: eval/retired-bundle closes
    assert [s["signal"] for s in obs.skipped] == ["eval-unlisted"]


def test_the_bundle_list_comes_from_archi_yaml_by_default(tmp_path):
    from archi.stack.instance import Instance, Ports

    Instance(dir=tmp_path, name="demo", bundles=["cms-physics"], data_root=tmp_path / "data", runtime="docker",
             ports=Ports(graph_db=15432, mcp=18080, webui=13000)).save()
    runs(tmp_path / "eval", 4, bundle="cern-team")
    runs(tmp_path / "eval", 4, bundle="cms-physics")
    obs = collect(tmp_path, read_bundles=None)
    assert [p.key for p in obs.problems] == ["eval/cms-physics"]


def test_context_eval_runs_and_hidden_directories_are_never_read(tmp_path):
    """`archi eval --context` runs live under `eval/context/`, and a run being written sits in a hidden
    directory: neither is read, so a broken file there cannot break the eval signal."""
    good = write_report(tmp_path, started_at="2026-10-26T10:00:00Z", statuses=scored(8))
    claims = '{"schema": "archi-eval-report/1", "bundle": '  # claims the schema, then breaks off
    for rel in ("context/demo/20261027T030144Z/report.json", "context/report.json",
                "cern-team/.20261027T030144Z.tmp-7/report.json", ".partial/report.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(claims)
    assert [r.path for r in signals.read_eval_runs(tmp_path)] == [good.relative_to(tmp_path).as_posix()]
    broken = tmp_path / "cms-physics" / "report.json"
    broken.parent.mkdir()
    broken.write_text(claims)
    with pytest.raises(ValueError, match="not readable JSON"):  # a visible smoke report is still checked
        signals.read_eval_runs(tmp_path)
