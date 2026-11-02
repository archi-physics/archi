"""req.r3-ops-runtime.shadow: a shadow report is produced from a fixture
history, scoring incidents against later human actions
(acc.r3-ops-runtime.score)."""
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from archi.ops import shadow
from archi.ops.record import RecordError, load_records
from archi.ops.shadow import okg_source, parse_time, promotion_readiness, render_markdown, score_automation

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "shadow" / "history.jsonl"
UNTIL = "2026-10-31T00:00:00Z"
A = "ticket-triage"


def _rows():
    return [{"node_id": n["node_id"], "subtype": n["subtype"], "attrs": dict(n["attrs"], text="okg-stamped")}
            for n in load_records(FIXTURE).payload()["nodes"]]


def _source(rows):
    return lambda subtype: [r for r in rows if r["subtype"] == subtype]


def _row(subtype, **attrs):
    return {"node_id": f"{subtype}:{attrs['record_id']}", "subtype": subtype, "attrs": attrs}


def _set(rows, record_id, **changes):
    for r in rows:
        if r["attrs"].get("record_id") == record_id:
            r["attrs"].update(changes)
    return rows


def _report(rows=None, **kw):
    kw.setdefault("until", UNTIL)
    return score_automation(_source(rows if rows is not None else _rows()), A, **kw)


def _classes(report):
    return {d["incident_id"]: d["class"] for d in report["details"]}


def test_report_from_the_fixture_history_has_exact_numbers():
    report = _report()
    assert _classes(report) == {
        "inc-agree": "agree",            # two shadow proposals, scored once
        "inc-disagree": "disagree",      # other service; the action time carries a +02:00 offset
        "inc-false": "false",            # a person outcome says no action was needed
        "inc-undecidable": "undecidable",
        "inc-late": "late",              # matching action five days later
        "inc-approved": "approved",      # a person approved; not agreement
        "inc-classonly": "agree",        # empty payload: action class only
        "inc-preempted": "preempted",    # a person acted before the proposal
        "inc-pending": "pending",
    }
    assert report["incidents"] == 9 and report["proposals"] == 10 and report["scored"] == 7
    assert report["ignored_non_shadow"] == 1
    assert report["counts"] == {"agree": 2, "disagree": 1, "false": 1, "approved": 1, "preempted": 1,
                                "late": 1, "undecidable": 1, "pending": 1}
    assert report["agree_class_only"] == 1
    assert report["answered"] == 4
    assert report["agreement_rate"] == 0.6667            # 2 / (2 + 1); approved left out
    assert report["response_coverage"] == 0.5714         # 4 / 7
    assert report["false_proposal_rate"] == 0.1667       # 1 / (7 - 1)
    assert report["false_proposal_rate_upper"] == 0.2857  # (1 + 1) / 7
    assert not any(report["dropped"].values())

    ttfr = report["time_to_first_response"]
    # Pending is out of the automation figure; the late response is out of the human one.
    assert ttfr["automation"] == {"n": 8, "median": 1050, "p90": None, "p90_note": "p90 needs n >= 10"}
    assert ttfr["human"] == {"n": 5, "median": 3600, "p90": None, "p90_note": "p90 needs n >= 10"}
    assert ttfr["median_lead_seconds"] == 2700
    assert ttfr["excluded"] == 0
    assert report["time_to_resolution_human"]["n"] == 1
    assert report["time_to_resolution_human"]["median"] == 27600
    json.dumps(report)


def test_several_proposals_on_one_incident_count_once_through_the_first_in_the_window():
    report = _report()
    assert report["proposals"] == 10 and report["incidents"] == 9
    first = {d["incident_id"]: d["proposal_id"] for d in report["details"]}
    assert first["inc-agree"] == "prop-agree"
    # Window opens after the first proposal: the second one is first in the window,
    # and the person already acted at 09:00, so it is preempted.
    later = _report(since="2026-09-14T09:30:00Z", until="2026-09-14T23:00:00Z")
    assert [(d["proposal_id"], d["class"]) for d in later["details"]] == [("prop-agree-2", "preempted")]


def test_empty_or_unparseable_payload_agrees_on_action_class_only():
    assert _report()["agree_class_only"] == 1
    rows = _set(_rows(), "prop-agree", payload="not json")
    report = _report(rows)
    assert report["agree_class_only"] == 2 and report["counts"]["agree"] == 2
    rows = _set(_rows(), "prop-classonly", action_class="drain_site")  # class differs: disagree
    assert _classes(_report(rows))["inc-classonly"] == "disagree"


def test_approval_is_not_agreement_and_a_rejection_is_disagreement():
    report = _report()
    assert report["counts"]["approved"] == 1 and report["counts"]["agree"] == 2
    rows = _set(_rows(), "dec-approved", verdict="rejected")
    assert _classes(_report(rows))["inc-approved"] == "disagree"
    rows = _set(_rows(), "dec-approved", verdict="timed_out", actor_kind="automation")
    assert _classes(_report(rows))["inc-approved"] == "undecidable"


def test_key_fields_limit_what_must_match():
    report = _report(key_fields={"restart_storage_door": ["site"]})
    assert _classes(report)["inc-disagree"] == "agree"
    assert report["agree_class_only"] == 1  # key field `site` is in the payload, so not class-only


def test_a_person_action_before_the_proposal_is_preempted_and_out_of_the_rates():
    report = _report()
    assert _classes(report)["inc-preempted"] == "preempted"
    assert report["scored"] == 7  # 9 incidents - pending - preempted
    assert report["counts"]["agree"] + report["counts"]["disagree"] == 3
    # The person was first: lead is negative for that incident, and it is in the human figure.
    assert report["time_to_first_response"]["human"]["n"] == 5


def test_late_response_is_reported_but_left_out_of_first_response_figures():
    rows = _rows()
    assert _classes(_report(rows))["inc-late"] == "late"
    assert _report(rows)["counts"]["late"] == 1
    # Without it, the human figure would hold 6 incidents.
    assert _report(rows)["time_to_first_response"]["human"]["n"] == 5


def test_horizon_edge_is_inclusive():
    on_edge = _set(_rows(), "act-agree", executed_at="2026-09-17T08:10:00Z")   # proposal 09-14T08:10 + 72 h
    assert _classes(_report(on_edge))["inc-agree"] == "agree"
    past = _set(_rows(), "act-agree", executed_at="2026-09-17T08:10:01Z")
    assert _classes(_report(past))["inc-agree"] == "late"
    at_proposal = _set(_rows(), "act-agree", executed_at="2026-09-14T08:10:00Z")
    assert _classes(_report(at_proposal))["inc-agree"] == "agree"
    just_before = _set(_rows(), "act-agree", executed_at="2026-09-14T08:09:59Z")
    assert _classes(_report(just_before))["inc-agree"] == "preempted"


def test_window_edges_are_inclusive():
    t = "2026-09-14T08:10:00Z"  # prop-agree
    assert "inc-agree" in _classes(_report(since=t, until="2026-09-14T08:10:00Z"))
    assert _report(since="2026-09-14T08:10:01Z")["details"][0]["proposal_id"] == "prop-agree-2"
    assert "inc-agree" not in _classes(_report(since="2026-09-14T10:00:01Z"))
    assert "inc-agree" not in _classes(_report(until="2026-09-14T08:09:59Z"))


def test_records_after_until_are_not_read():
    # until is 20 minutes after the proposal: the 09:00 person action is not read, the horizon is open.
    report = _report(until="2026-09-14T08:30:00Z")
    assert _classes(report) == {"inc-agree": "pending"}
    assert report["window"]["as_of"] == "2026-09-14T08:30:00Z"
    assert report["proposals"] == 1  # prop-agree-2 (10:00) is after until


def test_without_until_as_of_is_now_not_the_last_event_seen():
    rows = _rows()
    soon = score_automation(_source(rows), A, now=datetime(2026, 10, 30, 21, 0, tzinfo=timezone.utc))
    assert soon["window"]["as_of"] == "2026-10-30T21:00:00Z"
    assert _classes(soon)["inc-pending"] == "pending"
    later = score_automation(_source(rows), A, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert _classes(later)["inc-pending"] == "undecidable"


def test_non_utc_offsets_are_normalised():
    assert parse_time("2026-09-15T13:30:00+02:00") == parse_time("2026-09-15T11:30:00Z")
    assert parse_time("2026-09-15T11:30:00") == parse_time("2026-09-15T11:30:00Z")
    assert parse_time("garbage") is None and parse_time(None) is None
    # The fixture's disagree action (13:30+02:00 = 11:30Z) is after the 10:30Z proposal: in horizon.
    assert _classes(_report())["inc-disagree"] == "disagree"
    rows = _set(_rows(), "act-disagree-human", executed_at="2026-09-15T12:20:00+02:00")  # 10:20Z, before
    assert _classes(_report(rows))["inc-disagree"] == "preempted"


def test_no_action_needed_counts_only_from_a_person_outcome_at_or_after_the_proposal():
    assert _classes(_report())["inc-false"] == "false"
    before = _set(_rows(), "out-false", checked_at="2026-09-16T12:19:59Z")      # proposal at 12:20
    assert _classes(_report(before))["inc-false"] == "undecidable"
    on_time = _set(_rows(), "out-false", checked_at="2026-09-16T12:20:00Z")
    assert _classes(_report(on_time))["inc-false"] == "false"
    automated = _set(_rows(), "out-false", actor_kind="automation")
    assert _classes(_report(automated))["inc-false"] == "undecidable"


def test_a_person_false_alarm_status_counts_but_a_resolved_incident_does_not():
    rows = _set(_rows(), "inc-undecidable", status="false_alarm", actor_kind="person",
                resolved_at="2026-09-17T09:00:00Z")
    assert _classes(_report(rows))["inc-undecidable"] == "false"
    by_automation = _set(_rows(), "inc-undecidable", status="false_alarm")
    assert _classes(_report(by_automation))["inc-undecidable"] == "undecidable"
    rows = _rows() + [_row("ops_outcome", record_id="o1", incident_id="inc-undecidable", result="resolved",
                           checked_at="2026-09-18T00:00:00Z", actor_kind="person", actor="p")]
    assert _classes(_report(rows))["inc-undecidable"] == "undecidable"


def test_a_ticket_resolution_is_not_a_first_response():
    # inc-undecidable has no action; a person closing the ticket does not make it answered.
    rows = _rows() + [_row("ops_outcome", record_id="o2", incident_id="inc-undecidable", result="resolved",
                           check_method="ticket_resolution", checked_at="2026-09-17T09:00:00Z",
                           actor_kind="person", actor="p")]
    report = _report(rows)
    assert _classes(report)["inc-undecidable"] == "undecidable"
    assert report["time_to_first_response"]["human"]["n"] == 5
    assert report["time_to_resolution_human"]["n"] == 2


def test_dropped_records_are_counted():
    rows = _rows() + [
        _row("ops_proposal", record_id="p-nodate", incident_id="inc-agree", mode="shadow", automation=A),
        _row("ops_proposal", record_id="p-baddate", incident_id="inc-agree", mode="shadow", automation=A,
             proposed_at="yesterday"),
        _row("ops_proposal", record_id="p-noinc", mode="shadow", automation=A, proposed_at="2026-09-20T00:00:00Z"),
        _row("ops_proposal", record_id="p-gone", incident_id="inc-nowhere", mode="shadow", automation=A,
             proposed_at="2026-09-20T00:00:00Z"),
        {"node_id": "ops_proposal:x", "subtype": "ops_proposal",
         "attrs": {"incident_id": "inc-agree", "mode": "shadow", "automation": A,
                   "proposed_at": "2026-09-20T00:00:00Z"}},
        _row("ops_action", record_id="a-undated", proposal_id="prop-undecidable", actor_kind="person",
             action_class="restart_storage_door"),
    ]
    report = _report(rows)
    assert report["dropped"] == {"proposal_no_record_id": 1, "proposal_bad_proposed_at": 2,
                                 "proposal_no_incident_id": 1, "proposal_incident_missing": 1,
                                 "undated_human_records": 1}
    assert report["incidents"] == 9  # the good history is unchanged
    assert "Dropped records: {" in render_markdown(report)


def test_ignored_non_shadow_counts_only_proposals_in_the_window():
    assert _report()["ignored_non_shadow"] == 1  # prop-live, 09-25
    assert _report(until="2026-09-20T00:00:00Z")["ignored_non_shadow"] == 0


def test_p90_appears_only_from_ten_incidents():
    rows = []
    for i in range(10):
        rows.append(_row("ops_incident", record_id=f"i{i}", title="t", entity="E",
                         detected_at="2026-09-01T00:00:00Z"))
        rows.append(_row("ops_proposal", record_id=f"p{i}", incident_id=f"i{i}", mode="shadow", automation=A,
                         action_class="x", proposed_at=f"2026-09-01T00:{i + 1:02d}:00Z"))
    report = _report(rows)
    summary = report["time_to_first_response"]["automation"]
    assert summary["n"] == 10 and summary["p90"] == 9 * 60 and "p90_note" not in summary
    assert "n/a (n < 10)" in render_markdown(_report())


def test_no_proposals_gives_an_empty_report():
    report = score_automation(_source(_rows()), "no-such-automation", until=UNTIL)
    assert report["incidents"] == 0 and report["scored"] == 0
    assert report["agreement_rate"] is None and report["false_proposal_rate"] is None
    assert report["response_coverage"] is None and report["false_proposal_rate_upper"] is None
    assert report["time_to_first_response"]["automation"]["n"] == 0
    assert "n/a" in render_markdown(report)


def test_no_human_records_leaves_agreement_undefined():
    rows = [r for r in _rows() if r["attrs"].get("actor_kind") != "person"]
    report = _report(rows)
    assert report["agreement_rate"] is None
    assert report["counts"]["agree"] == report["counts"]["disagree"] == report["counts"]["late"] == 0
    assert report["time_to_first_response"]["human"]["n"] == 0
    assert report["counts"]["undecidable"] == 8 and report["counts"]["pending"] == 1
    assert not promotion_readiness(report, min_proposals=1, min_answered=0)["ready"]


def test_an_incident_without_detected_at_is_excluded_from_response_time():
    rows = _rows()
    for r in rows:
        if r["node_id"] == "ops_incident:inc-false":
            del r["attrs"]["detected_at"]
    report = _report(rows)
    assert report["time_to_first_response"]["excluded"] == 1
    assert report["time_to_first_response"]["automation"]["n"] == 7


def test_bad_window_timestamp_is_an_error():
    with pytest.raises(RecordError):
        score_automation(_source(_rows()), A, since="yesterday")


def test_markdown_rendering_carries_the_headline_numbers():
    text = render_markdown(_report())
    assert text.startswith("# Shadow report: ticket-triage")
    assert "Agreement: 0.6667" in text
    assert "of which 1 on action class only" in text and "Approved without agreement: 1" in text
    assert "upper bound 0.2857" in text and "1 preempted" in text
    assert "| first response, human | 5 | 1.0 h | n/a (n < 10) |" in text
    assert "| resolution, human | 1 | 7.7 h | n/a (n < 10) |" in text


def test_readiness_helper_names_each_failing_threshold():
    report = _report()
    verdict = promotion_readiness(report)
    assert verdict["ready"] is False and len(verdict["reasons"]) == 4
    text = " ".join(verdict["reasons"])
    for needle in ("7 scored", "4 answered", "agreement 0.67", "0.29"):
        assert needle in text

    loose = dict(min_proposals=7, min_answered=4, min_response_coverage=0.5, min_agreement=0.6,
                 max_false_rate=0.3)
    assert promotion_readiness(report, **loose) == {"ready": True, "reasons": []}
    assert promotion_readiness(report, **dict(loose, min_answered=5))["ready"] is False
    assert promotion_readiness(report, **dict(loose, min_response_coverage=0.6))["ready"] is False
    edge = promotion_readiness(report, **dict(loose, min_agreement=0.6667, max_false_rate=0.2857))
    assert edge["ready"] is True  # thresholds are inclusive


def test_readiness_fails_when_rates_are_missing():
    verdict = promotion_readiness({"scored": 0, "answered": 0}, min_proposals=0, min_answered=0)
    assert verdict["ready"] is False and len(verdict["reasons"]) == 3


def test_okg_source_reads_each_subtype_once_in_the_okg_row_shape(monkeypatch):
    rows = _rows()
    calls = []

    def fake_run(cmd, capture_output, text):
        subtype = cmd[cmd.index("--type") + 1]
        calls.append(subtype)
        assert cmd[:2] == ["okg", "filter"] and "--all" in cmd and "--jsonl" in cmd
        out = "\n".join(json.dumps(r) for r in copy.deepcopy(rows) if r["subtype"] == subtype)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr("archi.ops.record.subprocess.run", fake_run)
    report = score_automation(okg_source("myteam", okg="okg"), A, until=UNTIL)
    assert report["counts"]["agree"] == 2
    assert sorted(calls) == sorted(shadow.SUBTYPES)  # five calls, one per subtype


def test_okg_source_reports_a_failed_read(monkeypatch):
    monkeypatch.setattr("archi.ops.record.subprocess.run",
                        lambda cmd, capture_output, text: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(RecordError):
        score_automation(okg_source("myteam", okg="okg"), A, until=UNTIL)


def test_cli_prints_json_and_markdown(monkeypatch, capsys):
    monkeypatch.setattr(shadow, "okg_source", lambda deployment, dsn=None: _source(_rows()))
    assert shadow.main(["--deployment", "d", "--automation", A, "--until", UNTIL]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["agree"] == 2
    assert shadow.main(["--deployment", "d", "--automation", A, "--until", UNTIL, "--format", "markdown"]) == 0
    assert capsys.readouterr().out.startswith("# Shadow report")


def test_the_vocabulary_the_scorer_reads_is_documented_and_accepted():
    from archi.ops.record import OpsRecords, vocabulary
    root = Path(__file__).resolve().parents[3] / "bundles/packs/ops-record"
    schema = (root / "schemas/ops_record.yaml").read_text()
    assert "open | mitigated | resolved | false_alarm" in schema
    assert "unknown | no_action_needed" in schema
    skill = (root / "skills/ops_precedents.md").read_text()
    assert shadow.NO_ACTION_STATUS in skill and shadow.NO_ACTION_RESULT in skill
    records = OpsRecords(vocab=vocabulary())
    records.add("incident", record_id="i1", title="t", entity="E", status=shadow.NO_ACTION_STATUS)
    records.add("outcome", record_id="o1", incident_id="i1", result=shadow.NO_ACTION_RESULT)
