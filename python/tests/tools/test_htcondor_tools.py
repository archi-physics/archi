"""The HTCondor live read tools, on a fake runner. Read-only: the pack ships no act tool, and no process is started."""
import subprocess
from pathlib import Path

import pytest
import yaml

from archi.tools import htcondor
from packs.cluster_fixtures import FakeRunner, text

PACK = Path(__file__).resolve().parents[3] / "bundles" / "packs" / "htcondor"


@pytest.fixture(autouse=True)
def no_process(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"a real process was started: {args!r}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


def test_condor_status_lists_slots_with_a_constraint_built_from_validated_names():
    runner = FakeRunner().on(lambda a: a[0] == "condor_status", text("condor_status.out"))
    out = htcondor.archi_condor_status(machine="node001.example.org", state="claimed", runner=runner)
    assert runner.calls == [[
        "condor_status", "-limit", "51", "-af:t", *htcondor.STATUS_ATTRS,
        "-constraint", 'Machine == "node001.example.org" && State == "Claimed"',
    ]]
    assert out["boundary"] == "external_live" and out["observed_at"] and out["count"] == 4
    assert out["rows"][0] == {"Machine": "node001.example.org", "State": "Claimed", "Activity": "Busy", "Cpus": "8",
                              "Memory": "16384", "LoadAvg": "7.50", "OpSys": "LINUX", "Arch": "X86_64"}


def test_condor_q_translates_job_status_codes_and_keeps_an_empty_remote_host():
    runner = FakeRunner().on(lambda a: a[0] == "condor_q", text("condor_q.out"))
    out = htcondor.archi_condor_q(owner="alan", cluster_id="4002", held_only=True, runner=runner)
    assert runner.calls[0][-2:] == ["-constraint", 'Owner == "alan" && ClusterId == 4002 && JobStatus == 5']
    assert [r["JobStatus"] for r in out["rows"]] == ["Running", "Held", "Idle"]
    held = [r for r in out["rows"] if r["JobStatus"] == "Held"][0]
    assert held["HoldReason"] == "Job held by user" and held["RemoteHost"] == ""
    assert out["boundary"] == "external_live"


def test_no_constraint_when_no_filter_is_given():
    runner = FakeRunner().on(lambda a: True, text("condor_q.out"))
    htcondor.archi_condor_q(runner=runner)
    assert "-constraint" not in runner.calls[0]


@pytest.mark.parametrize("tool, kwargs", [
    (htcondor.archi_condor_status, {"machine": 'x" || true || "'}),
    (htcondor.archi_condor_status, {"machine": "-pool evil"}),
    (htcondor.archi_condor_status, {"state": "Claimed\" || 1"}),
    (htcondor.archi_condor_q, {"owner": "a b"}),
    (htcondor.archi_condor_q, {"cluster_id": "1 || 2"}),
    (htcondor.archi_condor_q, {"cluster_id": "\u0663\u0664"}),    # Arabic-Indic digits: isdigit() is true, int() accepts them
    (htcondor.archi_condor_q, {"cluster_id": "\u00b2"}),          # a superscript two: isdigit() true, int() raises
])
def test_an_argument_that_could_become_a_classad_or_an_option_never_reaches_a_command(tool, kwargs):
    runner = FakeRunner()
    out = tool(runner=runner, **kwargs)
    assert "error" in out and runner.calls == []


def test_a_failing_command_is_an_error_payload():
    runner = FakeRunner().on(lambda a: True, "", returncode=1, stderr="Error: communication error")
    assert "communication error" in htcondor.archi_condor_status(runner=runner)["error"]


def test_the_pack_ships_only_idempotent_read_tools_and_no_action_class():
    pack = yaml.safe_load((PACK / "pack.yaml").read_text())
    declared = {f"archi.tools.htcondor:{name}" for name in htcondor.TOOLS}
    assert set(pack["live_tools"]) == declared == set(pack["idempotent_tools"])
    assert all(meta["kind"] == "read" and meta["boundary"] == "external_live" for meta in htcondor.TOOLS.values())
    assert "action_classes" not in pack


def test_the_query_and_the_output_are_both_capped():
    runner = FakeRunner().on(lambda a: True, text("condor_q.out") * 40)
    out = htcondor.archi_condor_q(max_results=2, runner=runner)
    assert runner.calls[0][:5] == ["condor_q", "-allusers", "-limit", "3", "-af:t"], "the tool asks for limit + 1 rows"
    assert len(out["rows"]) == 2 and out["truncated"] is True, "and a reply that ignores the flag is cut before parsing"
