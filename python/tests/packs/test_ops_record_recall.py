"""PACT r3-ops-record, acc.r3-ops-record.precedent: a fixture incident
retrieves its precedent with outcome.

The fixture is two workflow runs (fixtures/ops_record): a resolved SRM-timeout
incident at T2_US_MIT with its full loop, an unrelated incident at T1_DE_KIT,
and a new SRM-timeout incident at T2_US_MIT. The reader answers
`archi.ops.precedents` from those records exactly as `okg filter` would from
the published graph (`scratch_ops_record_run.sh` runs the same lookup against
a real instance).
"""
import json
from pathlib import Path

import pytest

from archi.ops.precedents import find_precedents, okg_reader
from archi.ops.record import RecordError, load_records

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ops_record"


def _graph():
    rows = []
    for run in ("run1.jsonl", "run2.jsonl"):
        for node in load_records(FIXTURES / run).payload()["nodes"]:
            rows.append({"node_id": node["node_id"], "subtype": node["subtype"],
                         "attrs": dict(node["attrs"], text="okg-stamped")})
    return rows


def _reader(rows):
    def read(subtype, attr, value):
        return [r for r in rows if r["subtype"] == subtype and r["attrs"].get(attr) == value]
    return read


def test_a_fixture_incident_retrieves_its_precedent_with_outcome():
    found = find_precedents(_reader(_graph()), incident="inc-20261003-t2usmit-srm")
    assert found["entity"] == "T2_US_MIT"
    assert [p["record_id"] for p in found["precedents"]] == ["inc-20260914-t2usmit-srm"]
    precedent = found["precedents"][0]
    assert precedent["same_error_signature"] is True
    assert "text" not in precedent

    (proposal,) = precedent["proposals"]
    assert proposal["action_class"] == "restart_storage_door"
    assert [d["verdict"] for d in proposal["decisions"]] == ["approved"]
    assert proposal["decisions"][0]["escalation_id"].startswith("esc-")
    assert [a["status"] for a in proposal["actions"]] == ["succeeded"]

    (outcome,) = precedent["outcomes"]
    assert outcome["result"] == "resolved"
    assert outcome["node_id"] == "ops_outcome:out-20260914-t2usmit-srm-1"
    assert "Restarting the SRM door" in outcome["resolution"]


def test_by_entity_most_recent_first_and_matching_signature_first():
    rows = _graph()
    found = find_precedents(_reader(rows), entity="T2_US_MIT")
    assert [p["record_id"] for p in found["precedents"]] == [
        "inc-20261003-t2usmit-srm", "inc-20260914-t2usmit-srm",
    ]
    # An older incident sharing the signature outranks a newer one that does not.
    rows.append({"node_id": "ops_incident:inc-x", "subtype": "ops_incident",
                 "attrs": {"record_id": "inc-x", "title": "other", "entity": "T2_US_MIT",
                           "error_signature": "DISK_FULL", "detected_at": "2026-10-01T00:00:00Z"}})
    found = find_precedents(_reader(rows), incident="inc-20261003-t2usmit-srm")
    assert [p["record_id"] for p in found["precedents"]] == ["inc-20260914-t2usmit-srm", "inc-x"]
    assert found["precedents"][1]["outcomes"] == []


def test_outcomes_are_listed_oldest_first():
    rows = _graph()
    # A later ticket resolution, inserted ahead of the outcome check it follows.
    rows.insert(0, {"node_id": "ops_outcome:out-late", "subtype": "ops_outcome",
                    "attrs": {"record_id": "out-late", "incident_id": "inc-20260914-t2usmit-srm",
                              "result": "unknown", "check_method": "ticket_resolution",
                              "checked_at": "2026-09-20T09:00:00Z"}})
    found = find_precedents(_reader(rows), incident="inc-20261003-t2usmit-srm")
    outcomes = found["precedents"][0]["outcomes"]
    assert [o["record_id"] for o in outcomes] == ["out-20260914-t2usmit-srm-1", "out-late"]


def test_an_entity_with_no_history_has_no_precedents():
    assert find_precedents(_reader(_graph()), entity="T3_XX_NOWHERE")["precedents"] == []


def test_an_unknown_incident_is_an_error():
    with pytest.raises(RecordError, match="no ops_incident"):
        find_precedents(_reader(_graph()), incident="inc-missing")


def test_the_okg_reader_filters_by_attribute(monkeypatch):
    import subprocess

    seen = []

    def fake_run(cmd, capture_output, text):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(
            {"node_id": "ops_incident:i", "subtype": "ops_incident", "attrs": {"record_id": "i"}}) + "\n",
            stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rows = okg_reader("myteam", okg="okg")("ops_incident", "entity", "T2_US_MIT")
    assert rows[0]["node_id"] == "ops_incident:i"
    cmd = seen[0]
    assert cmd[:2] == ["okg", "filter"] and "--all" in cmd and "--jsonl" in cmd
    assert json.loads(cmd[cmd.index("--where") + 1]) == {
        "type": "ops_incident", "attrs.entity": "T2_US_MIT"}
