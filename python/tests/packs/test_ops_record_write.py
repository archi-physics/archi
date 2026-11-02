"""PACT r3-ops-record, task.r3-ops-record.write: the ops-record writer.

What the writer hands to `okg admission submit`, checked without a database.
The admission itself (acc.r3-ops-record.admit) is proven on a scratch
instance by `scratch_ops_record_run.sh`.
"""
import json
import subprocess
from pathlib import Path

import pytest

from archi.ops import record
from archi.ops.record import OpsRecords, RecordError, load_records, resolution_outcomes

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ops_record"


def _run1():
    return load_records(FIXTURES / "run1.jsonl")


def test_a_workflow_run_becomes_one_payload_with_the_loop_edges():
    records = _run1()
    payload = records.payload()
    subtypes = sorted(n["subtype"] for n in payload["nodes"])
    assert subtypes == [
        "ops_action", "ops_decision", "ops_incident", "ops_incident", "ops_outcome", "ops_proposal",
    ]
    edges = {(e["src"].split(":")[0], e["edge_type"], e["dst"]) for e in payload["edges"]}
    assert edges == {
        ("ops_proposal", "addresses", "ops_incident:inc-20260914-t2usmit-srm"),
        ("ops_decision", "decides", "ops_proposal:prop-20260914-t2usmit-srm-1"),
        ("ops_action", "executes", "ops_proposal:prop-20260914-t2usmit-srm-1"),
        ("ops_action", "authorized_by", "ops_decision:dec-20260914-t2usmit-srm-1"),
        ("ops_outcome", "evaluates", "ops_action:act-20260914-t2usmit-srm-1"),
        ("ops_outcome", "outcome_of", "ops_incident:inc-20260914-t2usmit-srm"),
    }
    assert records.dangling() == []
    proposal = records.nodes["ops_proposal:prop-20260914-t2usmit-srm-1"]["attrs"]
    assert proposal["evidence_ids"] == ["jira:OPS-11", "monit:fixture:transfers:T2_US_MIT"]
    decision = records.nodes["ops_decision:dec-20260914-t2usmit-srm-1"]["attrs"]
    assert decision["escalation_id"].startswith("esc-") and decision["actor_kind"] == "person"


def test_the_producer_policy_declares_exactly_what_the_payload_writes():
    records = _run1()
    policy = records.authority("workflow:archi-ops")["producer_policy"]
    assert policy["producer_kind"] == "workflow"
    assert policy["admission_mode"] == "fast_track"
    signature = policy["output_signature"]
    assert {n["subtype"] for n in signature["nodes"]} == {
        n["subtype"] for n in records.payload()["nodes"]
    }
    assert len(signature["edges"]) == 6
    # Every triple is one the pack's bridge declares, so the catalog check passes.
    vocab = record.vocabulary()
    for e in signature["edges"]:
        assert (e["src_subtype"], e["edge_type"], e["dst_subtype"]) in vocab.edges


@pytest.mark.parametrize(
    "kind, attrs, message",
    [
        ("incident", {"record_id": "i", "title": "t"}, "missing required ['entity']"),
        ("proposal", {"record_id": "p", "incident_id": "i", "summary": "s",
                      "generation_read": "g"}, "missing required ['evidence_ids']"),
        ("decision", {"record_id": "d", "proposal_id": "p", "verdict": "approved",
                      "actor_kind": "person", "actor": "a", "colour": "red"}, "unknown attributes ['colour']"),
        ("postmortem", {"record_id": "x"}, "unknown record kind"),
    ],
)
def test_a_record_the_schema_would_refuse_is_refused_here(kind, attrs, message):
    with pytest.raises(RecordError, match=message.replace("[", r"\[").replace("]", r"\]")):
        OpsRecords().add(kind, **attrs)


def test_a_link_to_an_earlier_record_needs_that_record_carried():
    records = OpsRecords()
    records.add("outcome", record_id="o", incident_id="inc-old", result="resolved",
                actor_kind="automation", actor="w")
    assert records.dangling() == ["ops_incident:inc-old"]
    with pytest.raises(RecordError, match="carry"):
        record.submit(records, deployment="d")
    records.carry({"node_id": "ops_incident:inc-old", "subtype": "ops_incident",
                   "attrs": {"record_id": "inc-old", "title": "t", "entity": "e", "text": "x",
                             "nomos_source_lineage": {}}})
    assert records.dangling() == []
    # okg's own attributes are not re-asserted.
    assert set(records.nodes["ops_incident:inc-old"]["attrs"]) == {"record_id", "title", "entity"}


def test_a_cited_precedent_gets_an_edge_other_evidence_stays_an_attribute():
    records = OpsRecords()
    records.carry({"node_id": "ops_incident:inc-old", "subtype": "ops_incident",
                   "attrs": {"record_id": "inc-old", "title": "t", "entity": "e"}})
    records.add("incident", record_id="inc-new", title="t", entity="e")
    records.add("proposal", record_id="p", incident_id="inc-new", summary="s", generation_read="g",
                evidence=["ops_incident:inc-old", {"node_id": "jira:OPS-1"}])
    cites = [e for e in records.edges if e["edge_type"] == "cites"]
    assert [(e["dst"], e["dst_subtype"]) for e in cites] == [("ops_incident:inc-old", "ops_incident")]
    assert records.dangling() == []


def test_submit_runs_okg_admission_submit_as_a_workflow_producer(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output, text):
        seen["cmd"] = cmd
        seen["payload"] = json.loads(Path(cmd[cmd.index("--payload-file") + 1]).read_text())
        return subprocess.CompletedProcess(cmd, 0, stdout='{"output": {"status": "accepted"}}', stderr="")

    monkeypatch.setattr(record.subprocess, "run", fake_run)
    reply = record.submit(_run1(), deployment="myteam", okg="okg")
    assert reply == {"output": {"status": "accepted"}}
    cmd = seen["cmd"]
    assert cmd[:3] == ["okg", "admission", "submit"]
    assert cmd[cmd.index("--producer-kind") + 1] == "workflow"
    assert cmd[cmd.index("--producer-id") + 1] == record.PRODUCER_ID
    assert cmd[cmd.index("--intent") + 1] == "ops_record"
    authority = json.loads(cmd[cmd.index("--authority-json") + 1])
    assert authority["producer_policy"]["output_signature"]["edges"]
    assert len(seen["payload"]["nodes"]) == 6


def test_submit_reports_okg_refusal(monkeypatch):
    monkeypatch.setattr(
        record.subprocess, "run",
        lambda cmd, capture_output, text: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="producer_catalog_narrowing_missing"),
    )
    with pytest.raises(RecordError, match="producer_catalog_narrowing_missing"):
        record.submit(_run1(), deployment="myteam", okg="okg")


# ---------------------------------------------------------------------------
# Ticket resolutions
# ---------------------------------------------------------------------------


def _rows(records):
    return [{"node_id": n["node_id"], "subtype": n["subtype"], "attrs": n["attrs"]}
            for n in records.payload()["nodes"]]


def _tickets(name):
    issues = json.loads((FIXTURES / name).read_text())
    return [{"node_id": f"jira:{i['key']}", "subtype": "jira_issue", "attrs": i} for i in issues]


def test_a_resolved_ticket_becomes_the_incidents_outcome():
    incidents = [r for r in _rows(load_records(FIXTURES / "run2.jsonl"))]
    assert resolution_outcomes(incidents, _tickets("jira_records_open.json")).nodes == {}

    records = resolution_outcomes(incidents, _tickets("jira_records_resolved.json"))
    outcome = records.nodes["ops_outcome:inc-20261003-t2usmit-srm.ticket.OPS-12"]["attrs"]
    assert outcome["result"] == "resolved"
    assert outcome["check_method"] == "ticket_resolution"
    assert outcome["ticket_key"] == "OPS-12"
    assert outcome["actor_kind"] == "person" and outcome["actor"] == "Site Admin MIT"
    incident = records.nodes["ops_incident:inc-20261003-t2usmit-srm"]["attrs"]
    assert incident["status"] == "resolved"
    assert incident["resolved_at"] == "2026-10-03T11:10:00.000+0000"
    assert [(e["edge_type"], e["dst"]) for e in records.edges] == [
        ("outcome_of", "ops_incident:inc-20261003-t2usmit-srm")
    ]
    assert records.dangling() == []


def test_ticket_outcomes_are_written_once_and_a_non_fix_is_unknown():
    incidents = _rows(load_records(FIXTURES / "run2.jsonl"))
    tickets = _tickets("jira_records_resolved.json")
    first = resolution_outcomes(incidents, tickets)
    again = resolution_outcomes(incidents, tickets, outcomes=_rows(first))
    assert again.nodes == {}

    tickets[1]["attrs"]["resolution"] = "Won't Fix"
    records = resolution_outcomes(incidents, tickets)
    outcome = records.nodes["ops_outcome:inc-20261003-t2usmit-srm.ticket.OPS-12"]["attrs"]
    assert outcome["result"] == "unknown"
    # The incident stays open but travels with the outcome for the edge.
    assert records.nodes["ops_incident:inc-20261003-t2usmit-srm"]["attrs"]["status"] == "open"
    assert records.dangling() == []
