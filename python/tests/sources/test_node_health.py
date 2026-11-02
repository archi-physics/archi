"""task.r3-bundle-cluster-manager.htcondor (node-health half): the node-health reader over a recorded health export, offline."""
import json
from collections import Counter

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.sources.node_health import NodeHealthAdapter, NodeHealthSource
from packs.cluster_fixtures import write_node_health_cache


def _run(tmp_path, **params):
    run = NodeHealthSource(base=str(tmp_path), **params).run("run-1", mode="scope_complete")
    return run, list(run.facts)


def _by_id(facts):
    return {f.node_id: f.attrs for f in facts if isinstance(f, NodeFact)}


def test_the_recorded_reports_become_nodes_and_checks(tmp_path):
    write_node_health_cache(tmp_path)
    run, facts = _run(tmp_path)
    assert run.completed_scope is True and run.health.status == "ok" and run.health.record_count == 3
    assert Counter(f.subtype for f in facts if isinstance(f, NodeFact)) == {"compute_node": 3, "node_health_check": 7}
    nodes = _by_id(facts)
    healthy = nodes["compute_node:submit:node001"]
    assert healthy["health"] == "healthy" and healthy["failing_checks"] == [] and healthy["check_count"] == 3
    bad = nodes["compute_node:submit:node042"]
    assert bad["health"] == "unhealthy", "the worst check decides"
    assert bad["failing_checks"] == ["disk_scratch"] and bad["warning_checks"] == ["ib_link"]
    assert "failing disk_scratch" in bad["text"]
    check = nodes["node_health_check:submit:node042:disk_scratch"]
    assert (check["status"], check["detail"], check["node_name"]) == ("fail", "100% used, 0 bytes free", "node042")


def test_a_reported_status_wins_over_the_checks(tmp_path):
    write_node_health_cache(tmp_path)
    _, facts = _run(tmp_path)
    assert _by_id(facts)["compute_node:submit:gpu02"]["health"] == "unhealthy"
    rows = [{"node": "n1", "status": "degraded", "checks": [{"name": "a", "status": "ok"}]}]
    (tmp_path / "node-health" / "records.json").write_text(json.dumps(rows))
    _, facts = _run(tmp_path)
    assert _by_id(facts)["compute_node:cluster:n1"]["health"] == "degraded"


def test_every_check_hangs_from_its_node(tmp_path):
    write_node_health_cache(tmp_path)
    _, facts = _run(tmp_path)
    ids = set(_by_id(facts))
    edges = [(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)]
    assert len(edges) == 7 and all(t == "contains" and s in ids and d in ids for s, t, d in edges)
    assert ("compute_node:submit:node001", "contains", "node_health_check:submit:node001:munge") in edges


def test_an_unknown_check_status_is_a_warning_not_a_pass(tmp_path):
    (tmp_path / "node-health").mkdir()
    rows = [{"node": "n1", "checks": [{"name": "x", "status": "mystery"}, {"name": "y", "status": "PASS"}]}]
    (tmp_path / "node-health" / "records.json").write_text(json.dumps(rows))
    _, facts = _run(tmp_path, cluster_name="c")
    nodes = _by_id(facts)
    assert nodes["node_health_check:c:n1:x"]["status"] == "warn" and nodes["node_health_check:c:n1:y"]["status"] == "ok"
    assert nodes["compute_node:c:n1"]["health"] == "degraded"


def test_a_nameless_or_repeated_report_is_skipped_and_the_scope_is_not_complete(tmp_path):
    (tmp_path / "node-health").mkdir()
    rows = [{"node": "n1"}, {"node": "n1"}, {"checks": []}, "x"]
    (tmp_path / "node-health" / "records.json").write_text(json.dumps(rows))
    run, facts = _run(tmp_path, cluster_name="c")
    assert run.completed_scope is False
    assert [f.node_id for f in facts if isinstance(f, NodeFact)] == ["compute_node:c:n1"]


@pytest.mark.parametrize("payload", [None, "", "[", "[]", "{}"])
def test_a_missing_or_empty_cache_is_a_failed_fetch(tmp_path, payload):
    if payload is not None:
        (tmp_path / "node-health").mkdir()
        (tmp_path / "node-health" / "records.json").write_text(payload)
    source = NodeHealthSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == [] and run.completed_scope is False
    assert run.health.status in {"cache_missing", "endpoint_failed"}
    assert source.preflight().status == run.health.status


def test_the_reader_names_no_credential(tmp_path):
    write_node_health_cache(tmp_path)
    assert NodeHealthSource(base=str(tmp_path)).preflight().credential_refs == ()
    assert NodeHealthAdapter.profile == "discovery_crawl"
