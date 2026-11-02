"""task.r3-bundle-cluster-manager.htcondor: the HTCondor pool-snapshot reader and its exporter, offline.

The exporter runs through a fake runner; no HTCondor command is ever started.
"""
import json
from collections import Counter

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.downloaders import htcondor as exporter
from archi.sources.htcondor import HTCondorAdapter, HTCondorSource
from packs.cluster_fixtures import FakeRunner, condor_export_runner, write_condor_cache


def _run(tmp_path, **params):
    run = HTCondorSource(base=str(tmp_path), **params).run("run-1", mode="scope_complete")
    return run, list(run.facts)


def test_the_exporter_sums_slots_into_machines_and_cpus_by_state():
    snapshot = exporter.export(condor_export_runner(), "submit-pool")
    machines = {m["name"]: m for m in snapshot["machines"]}
    assert set(machines) == {"node001.example.org", "node002.example.org", "node003.example.org"}
    assert machines["node001.example.org"]["slots"] == 2 and machines["node001.example.org"]["cpus"] == 16
    assert machines["node001.example.org"]["memory_mb"] == 32768
    assert snapshot["totals"] == {"claimed_cpus": 24, "unclaimed_cpus": 8, "owner_cpus": 16}
    assert snapshot["schedds"] == [
        {"name": "submit01.example.org", "running_jobs": 120, "idle_jobs": 30, "held_jobs": 2},
        {"name": "submit02.example.org", "running_jobs": 15, "idle_jobs": 0, "held_jobs": 0},
    ]


def test_the_exporter_runs_two_read_only_condor_status_queries():
    runner = condor_export_runner()
    exporter.export(runner, "p")
    assert [c[0] for c in runner.calls] == ["condor_status", "condor_status"]
    assert "-schedd" in runner.calls[1]


@pytest.mark.parametrize("runner", [
    FakeRunner().on(lambda a: True, "", returncode=1, stderr="Failed to connect to collector"),
    FakeRunner().on(lambda a: True, ""),
])
def test_a_failed_or_empty_query_writes_nothing(tmp_path, runner):
    assert exporter.main(["--pool", "p", "--output-dir", str(tmp_path / "out")], runner=runner) == 1
    assert not (tmp_path / "out").exists()


def test_the_recorded_snapshot_becomes_a_pool_machines_and_schedds(tmp_path):
    snapshot = write_condor_cache(tmp_path)
    run, facts = _run(tmp_path)
    assert run.completed_scope is True and run.health.status == "ok" and run.health.record_count == 5
    assert (tmp_path / "htcondor" / "meta.json").is_file() and snapshot["pool"] == "submit-pool"
    assert Counter(f.subtype for f in facts if isinstance(f, NodeFact)) == {
        "condor_pool": 1, "condor_machine": 3, "condor_schedd": 2,
    }
    nodes = {f.node_id: f.attrs for f in facts if isinstance(f, NodeFact)}
    pool = nodes["condor_pool:submit-pool"]
    assert (pool["claimed_cpus"], pool["unclaimed_cpus"], pool["owner_cpus"], pool["machine_count"]) == (24, 8, 16, 3)
    machine = nodes["condor_machine:submit-pool:node002.example.org"]
    assert (machine["cpus"], machine["state"], machine["activity"], machine["load_avg"]) == (16, "Claimed", "Busy", 15.9)
    schedd = nodes["condor_schedd:submit-pool:submit01.example.org"]
    assert (schedd["running_jobs"], schedd["idle_jobs"], schedd["held_jobs"]) == (120, 30, 2)
    edges = [(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)]
    assert len(edges) == 5 and all(s == "condor_pool:submit-pool" and t == "contains" and d in nodes for s, t, d in edges)


def test_a_snapshot_without_machines_is_a_failed_export(tmp_path):
    (tmp_path / "htcondor").mkdir()
    (tmp_path / "htcondor" / "records.json").write_text(json.dumps({"pool": "p", "schedds": [{"name": "s"}]}))
    source = HTCondorSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == [] and run.completed_scope is False and run.health.status == "endpoint_failed"


@pytest.mark.parametrize("payload", [None, "", "{", "{}", "[]"])
def test_a_missing_or_empty_cache_is_a_failed_fetch(tmp_path, payload):
    if payload is not None:
        (tmp_path / "htcondor").mkdir()
        (tmp_path / "htcondor" / "records.json").write_text(payload)
    run = HTCondorSource(base=str(tmp_path)).run("run-1", mode="scope_complete")
    assert list(run.facts) == [] and run.completed_scope is False
    assert run.health.status in {"cache_missing", "endpoint_failed"}


def test_a_meta_count_that_disagrees_withholds_the_complete_scope(tmp_path):
    write_condor_cache(tmp_path)
    (tmp_path / "htcondor" / "meta.json").write_text(json.dumps({"record_count": 1}))
    run, _ = _run(tmp_path)
    assert run.completed_scope is False and run.health.status == "endpoint_failed"


def test_a_nameless_machine_is_skipped_and_the_scope_is_not_complete(tmp_path):
    (tmp_path / "htcondor").mkdir()
    (tmp_path / "htcondor" / "records.json").write_text(json.dumps({"machines": [{"name": "m"}, {"cpus": 4}]}))
    run, facts = _run(tmp_path, pool_name="p")
    assert run.completed_scope is False
    assert [f.node_id for f in facts if isinstance(f, NodeFact) and f.subtype == "condor_machine"] == ["condor_machine:p:m"]
    assert HTCondorAdapter.profile == "discovery_crawl"
