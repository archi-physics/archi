"""task.r3-bundle-cluster-manager.slurm: the Slurm snapshot reader and its exporter, over recorded command output, offline.

The exporter runs through a fake runner here; no Slurm command is ever started.
The gated live tools are tested in tests/tools/test_slurm_tools.py.
"""
import json
from collections import Counter

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.downloaders import slurm as exporter
from archi.sources.slurm import SlurmAdapter, SlurmSource, base_state, expand_nodelist
from packs.cluster_fixtures import FakeRunner, slurm_export_runner, write_slurm_cache


def _run(tmp_path, mode="scope_complete", **params):
    source = SlurmSource(base=str(tmp_path), **params)
    run = source.run("run-1", mode=mode)
    return run, list(run.facts)


def _nodes(facts, subtype):
    return {f.node_id: f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype}


# --- the exporter ---------------------------------------------------------------


def test_the_exporter_parses_the_recorded_output():
    snapshot = exporter.export(slurm_export_runner(), "submit")
    assert [p["name"] for p in snapshot["partitions"]] == ["cpu", "gpu", "debug"]
    assert snapshot["partitions"][0] == {
        "name": "cpu", "state": "up", "default": True, "max_time": "7-00:00:00", "total_nodes": 4, "total_cpus": 256,
    }
    nodes = {n["name"]: n for n in snapshot["nodes"]}
    assert set(nodes) == {"node001", "node002", "node003", "node042", "gpu01", "gpu02"}
    assert nodes["node003"]["partitions"] == ["cpu", "debug"], "one row per partition is merged"
    assert nodes["node042"]["reason"].startswith("archi[run-77]: scratch full")
    assert nodes["node001"]["reason"] == "" and nodes["node001"]["gres"] == ""
    assert nodes["gpu01"]["features"] == ["avx512", "a100"] and nodes["gpu01"]["gres"] == "gpu:a100:4"
    reservation = snapshot["reservations"][0]
    assert reservation["name"] == "maint_oct" and reservation["nodes"] == ["node001", "node002"]
    assert reservation["users"] == "root" and reservation["accounts"] == "" and reservation["flags"] == "MAINT,IGNORE_JOBS"
    jobs = {j["job_id"]: j for j in snapshot["jobs"]}
    assert set(jobs) == {"1001", "1002", "1003", "1004_7", "1005"}
    assert jobs["1002"]["nodelist"] == ["node001", "node002"] and jobs["1004_7"]["nodelist"] == []
    assert jobs["1005"]["state"] == "CANCELLED"


def test_the_exporter_asks_only_read_only_commands():
    runner = slurm_export_runner()
    exporter.export(runner, "submit", since_days=3)
    assert [c[0] for c in runner.calls] == ["sinfo", "sinfo", "scontrol", "sacct"]
    scontrol = [c for c in runner.calls if c[0] == "scontrol"][0]
    assert scontrol[1:3] == ["show", "reservation"], "never an update"
    assert any("--starttime=now-3days" in part for part in runner.calls[3])


def test_a_failed_command_writes_nothing(tmp_path):
    runner = FakeRunner().on(lambda a: True, "", returncode=1, stderr="slurm_load_partitions: Unable to contact slurm controller")
    assert exporter.main(["--cluster", "submit", "--output-dir", str(tmp_path / "out")], runner=runner) == 1
    assert not (tmp_path / "out").exists()


def test_an_empty_sinfo_writes_nothing(tmp_path):
    runner = FakeRunner().on(lambda a: True, "")
    assert exporter.main(["--cluster", "submit", "--output-dir", str(tmp_path / "out")], runner=runner) == 1
    assert not (tmp_path / "out").exists()


def test_the_exporter_writes_a_cache_whose_meta_counts_every_item(tmp_path):
    snapshot = write_slurm_cache(tmp_path)
    meta = json.loads((tmp_path / "slurm" / "meta.json").read_text())
    assert meta["record_count"] == sum(len(snapshot[k]) for k in ("partitions", "nodes", "reservations", "jobs")) == 15
    assert meta["cluster"] == "submit"


# --- the reader -----------------------------------------------------------------


def test_the_recorded_snapshot_becomes_partitions_nodes_reservations_and_jobs(tmp_path):
    write_slurm_cache(tmp_path)
    run, facts = _run(tmp_path)
    assert run.completed_scope is True and run.health.status == "ok" and run.health.record_count == 15
    assert Counter(f.subtype for f in facts if isinstance(f, NodeFact)) == {
        "slurm_partition": 3, "slurm_node": 6, "slurm_reservation": 1, "slurm_job": 5,
    }
    node = _nodes(facts, "slurm_node")["slurm_node:submit:node042"].attrs
    assert node["state"] == "drained" and node["state_base"] == "drained"
    assert "scratch full" in node["reason"] and "scratch full" in node["text"]
    down = _nodes(facts, "slurm_node")["slurm_node:submit:gpu02"].attrs
    assert (down["state"], down["state_base"]) == ("down*", "down")
    assert down["cpus"] == 32 and down["memory_mb"] == 512000 and down["gres"] == "gpu:a100:4"
    job = _nodes(facts, "slurm_job")["slurm_job:submit:1001"].attrs
    assert (job["state"], job["partition"], job["nodelist"], job["req_cpus"]) == ("RUNNING", "gpu", ["gpu01"], 8)
    assert job["user"] == "ada"


def test_edges_join_partitions_nodes_jobs_and_reservations_and_end_at_emitted_nodes(tmp_path):
    write_slurm_cache(tmp_path)
    _, facts = _run(tmp_path)
    ids = {f.node_id for f in facts if isinstance(f, NodeFact)}
    edges = {(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)}
    assert all(e[0] in ids and e[2] in ids for e in edges)
    assert ("slurm_partition:submit:cpu", "contains", "slurm_node:submit:node042") in edges
    assert ("slurm_partition:submit:debug", "contains", "slurm_node:submit:node003") in edges
    assert ("slurm_node:submit:node001", "executes", "slurm_job:submit:1002") in edges
    assert ("slurm_node:submit:node002", "executes", "slurm_job:submit:1002") in edges
    assert ("slurm_job:submit:1001", "member_of", "slurm_partition:submit:gpu") in edges
    assert ("slurm_reservation:submit:maint_oct", "references", "slurm_node:submit:node001") in edges
    assert not [e for e in edges if e[0] == "slurm_node:submit:node001" and e[2] == "slurm_job:submit:1001"]
    assert Counter(e[1] for e in edges) == {"contains": 7, "executes": 5, "member_of": 5, "references": 2}


def test_an_edge_to_a_node_the_snapshot_does_not_list_is_not_emitted(tmp_path):
    snapshot = {
        "cluster": "c", "partitions": [{"name": "p"}],
        "nodes": [{"name": "n1", "partitions": ["p", "gone"]}],
        "reservations": [{"name": "r", "nodes": "n[1-2]"}],
        "jobs": [{"job_id": "5", "partition": "gone", "nodelist": ["n1", "n9"]}],
    }
    (tmp_path / "slurm").mkdir()
    (tmp_path / "slurm" / "records.json").write_text(json.dumps(snapshot))
    _, facts = _run(tmp_path)
    edges = {(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)}
    assert edges == {
        ("slurm_partition:c:p", "contains", "slurm_node:c:n1"),
        ("slurm_reservation:c:r", "references", "slurm_node:c:n1"),
        ("slurm_node:c:n1", "executes", "slurm_job:c:5"),
    }


def test_a_meta_count_that_disagrees_withholds_the_complete_scope(tmp_path):
    write_slurm_cache(tmp_path)
    (tmp_path / "slurm" / "meta.json").write_text(json.dumps({"record_count": 99}))
    run, facts = _run(tmp_path)
    assert run.completed_scope is False and run.health.status == "endpoint_failed" and facts


@pytest.mark.parametrize("payload", [
    None, "", "{", "{}", "[]", json.dumps({"cluster": "c", "jobs": [{"job_id": "1"}]}),
])
def test_a_missing_empty_or_nodeless_snapshot_is_a_failed_export(tmp_path, payload):
    if payload is not None:
        (tmp_path / "slurm").mkdir()
        (tmp_path / "slurm" / "records.json").write_text(payload)
    source = SlurmSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == [] and run.completed_scope is False
    assert run.health.status in {"cache_missing", "endpoint_failed"}
    assert source.preflight().status == run.health.status


def test_a_duplicate_or_nameless_item_is_skipped_and_the_scope_is_not_complete(tmp_path):
    snapshot = {"nodes": [{"name": "n1"}, {"name": "n1"}, {"state": "idle"}], "partitions": []}
    (tmp_path / "slurm").mkdir()
    (tmp_path / "slurm" / "records.json").write_text(json.dumps(snapshot))
    run, facts = _run(tmp_path, cluster_name="c")
    assert run.completed_scope is False
    assert [f.node_id for f in facts if isinstance(f, NodeFact)] == ["slurm_node:c:n1"]


def test_state_and_node_list_helpers():
    assert [base_state(s) for s in ("DRAIN*", "mixed+drain", "idle~", "", "down*")] == ["drain", "mixed", "idle", "", "down"]
    assert expand_nodelist("n[01-03,07],gpu1,x[8-9]y") == ["n01", "n02", "n03", "n07", "gpu1", "x8y", "x9y"]
    assert expand_nodelist(["a", "", "b"]) == ["a", "b"] and expand_nodelist("") == []


def test_the_reader_names_no_credential(tmp_path):
    write_slurm_cache(tmp_path)
    assert SlurmSource(base=str(tmp_path)).preflight().credential_refs == ()
    assert SlurmAdapter.profile == "discovery_crawl"
