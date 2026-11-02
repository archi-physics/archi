"""req.r3-bundle-cluster-manager.connectors: the ServiceNow reader over recorded Table API rows, offline."""
import json
from collections import Counter

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.sources.servicenow import CREDENTIALS, ServiceNowAdapter, ServiceNowSource
from packs.cluster_fixtures import load, write_servicenow_cache


def _run(tmp_path, mode="scope_complete", **params):
    source = ServiceNowSource(base=str(tmp_path), **params)
    run = source.run("run-1", mode=mode)
    return run, list(run.facts)


def _nodes(facts, subtype):
    return {f.node_id: f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype}


def test_the_recorded_rows_become_incidents_requests_articles_and_people(tmp_path):
    write_servicenow_cache(tmp_path)
    run, facts = _run(tmp_path, instance_url="https://example.service-now.com/")
    assert run.completed_scope is True
    assert run.health.status == "ok" and run.health.record_count == 7

    counts = Counter(f.subtype for f in facts if isinstance(f, NodeFact))
    assert counts == {
        "servicenow_incident": 3, "servicenow_request": 2, "servicenow_kb_article": 2, "person": 5,
    }
    incidents = _nodes(facts, "servicenow_incident")
    first = incidents["servicenow_incident:INC0010001"].attrs
    assert first["title"] == "node042 drained, scratch filesystem full"
    assert (first["state"], first["priority"], first["assignment_group"]) == (
        "In Progress", "2 - High", "Cluster Operations")
    assert first["config_item"] == "node042"
    assert first["url"] == "https://example.service-now.com/incident.do?sys_id=a1b2c3d4e5f60718293a4b5c6d7e8f90"
    assert "node042" in first["text"] and "INC0010001" in first["text"]


def test_reference_fields_given_as_objects_read_as_their_display_value(tmp_path):
    write_servicenow_cache(tmp_path)
    _, facts = _run(tmp_path)
    second = _nodes(facts, "servicenow_incident")["servicenow_incident:INC0010002"].attrs
    assert second["state"] == "Resolved"
    assert second["assignee"] == "Grace Hopper" and second["reporter"] == "Alan Turing"
    assert second["resolution"] == "Rolled back the kernel and pinned the version."


def test_no_email_address_reaches_a_node(tmp_path):
    write_servicenow_cache(tmp_path)
    _, facts = _run(tmp_path)
    blob = json.dumps([f.attrs for f in facts if isinstance(f, NodeFact)])
    assert "@" not in blob


def test_people_edges_follow_the_roles(tmp_path):
    write_servicenow_cache(tmp_path)
    _, facts = _run(tmp_path)
    edges = {(e.src, e.edge_type): e.dst for e in facts if isinstance(e, EdgeFact)}
    people = {f.node_id: f.attrs["display_name"] for f in facts if isinstance(f, NodeFact) and f.subtype == "person"}
    assert people[edges[("servicenow_incident:INC0010001", "reported_by")]] == "Ada Lovelace"
    assert people[edges[("servicenow_incident:INC0010001", "assigned_to")]] == "Grace Hopper"
    assert people[edges[("servicenow_request:REQ0020001", "reported_by")]] == "Katherine Johnson"
    assert people[edges[("servicenow_kb_article:KB0000010", "authored_by")]] == "Grace Hopper"
    # an unassigned incident has no assigned_to edge
    assert ("servicenow_incident:INC0010003", "assigned_to") not in edges
    # every edge ends at an emitted node
    ids = {f.node_id for f in facts if isinstance(f, NodeFact)}
    assert all(e.dst in ids and e.src in ids for e in facts if isinstance(e, EdgeFact))


def test_a_knowledge_article_carries_its_body_and_base(tmp_path):
    write_servicenow_cache(tmp_path)
    _, facts = _run(tmp_path)
    article = _nodes(facts, "servicenow_kb_article")["servicenow_kb_article:KB0000010"].attrs
    assert article["knowledge_base"] == "Cluster Operations" and article["author"] == "Grace Hopper"
    assert "Drain the node" in article["description"]
    assert "priority" not in article


def test_the_same_run_twice_emits_the_same_ids(tmp_path):
    write_servicenow_cache(tmp_path)
    ids = [sorted(f.node_id for f in _run(tmp_path)[1] if isinstance(f, NodeFact)) for _ in range(2)]
    assert ids[0] == ids[1]


def test_a_meta_count_that_disagrees_withholds_the_complete_scope(tmp_path):
    write_servicenow_cache(tmp_path)
    (tmp_path / "servicenow" / "meta.json").write_text(json.dumps({"record_count": 9}))
    run, facts = _run(tmp_path)
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed" and "9" in run.health.reason
    assert facts, "the records are still emitted"


def test_an_item_that_cannot_be_parsed_is_skipped_and_the_scope_is_not_complete(tmp_path):
    records = load("servicenow_records.json") + [{"table": "incident"}, {"table": "problem", "number": "PRB1"}, "x"]
    (tmp_path / "servicenow").mkdir()
    (tmp_path / "servicenow" / "records.json").write_text(json.dumps(records))
    run, facts = _run(tmp_path)
    assert run.completed_scope is False
    assert len(_nodes(facts, "servicenow_incident")) == 3
    assert "skipped 3" in run.health.reason


@pytest.mark.parametrize("content", [None, "", "[", "[]", "{\"result\": []}"])
def test_a_missing_empty_or_wrong_cache_is_a_failed_fetch_never_an_empty_catalog(tmp_path, content):
    if content is not None:
        (tmp_path / "servicenow").mkdir()
        (tmp_path / "servicenow" / "records.json").write_text(content)
    source = ServiceNowSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status in {"cache_missing", "endpoint_failed"}
    assert source.preflight().status == run.health.status


def test_credentials_are_named_by_auth_method_and_never_read(tmp_path, monkeypatch):
    for variable in sum(CREDENTIALS.values(), ()):
        monkeypatch.delenv(variable, raising=False)
    write_servicenow_cache(tmp_path)
    for method, names in CREDENTIALS.items():
        source = ServiceNowSource(base=str(tmp_path), auth_method=method)
        assert source.preflight().credential_refs == names
        assert source.preflight().status == "ok", "the cache needs no credential to be read"
    with pytest.raises(ValueError, match="auth_method"):
        ServiceNowSource(base=str(tmp_path), auth_method="token")


def test_the_adapter_declares_the_reader_profile(tmp_path):
    assert ServiceNowAdapter.profile == ServiceNowSource.profile == "mutable_api"
    write_servicenow_cache(tmp_path)
    assert ServiceNowAdapter(base=str(tmp_path)).preflight().record_count == 7
