"""task.r3-bundle-cluster-manager.tickets: the generic ticket reader over recorded Redmine and JIRA issues, offline."""
import json
from collections import Counter

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.sources.tickets import TicketAdapter, TicketSource
from packs.cluster_fixtures import write_jira_tickets_cache, write_tickets_cache


def _run(tmp_path, mode="scope_complete", **params):
    source = TicketSource(base=str(tmp_path), **params)
    run = source.run("run-1", mode=mode)
    return run, list(run.facts)


def _tickets(facts):
    return {f.node_id: f for f in facts if isinstance(f, NodeFact) and f.subtype == "tracker_ticket"}


def test_the_recorded_issues_become_tickets_and_people(tmp_path):
    write_tickets_cache(tmp_path)
    run, facts = _run(tmp_path, base_url="https://redmine.example.org/")
    assert run.completed_scope is True and run.health.record_count == 3
    assert Counter(f.subtype for f in facts if isinstance(f, NodeFact)) == {"tracker_ticket": 3, "person": 3}

    tickets = _tickets(facts)
    one = tickets["tracker_ticket:redmine:4812"].attrs
    assert one["ticket_id"] == "4812" and one["system"] == "redmine"
    assert one["subject"] == "Scratch filesystem full on node042"
    assert (one["project"], one["tracker"], one["status"], one["priority"]) == (
        "Cluster Ops", "Support", "In Progress", "High")
    assert one["url"] == "https://redmine.example.org/issues/4812"
    assert one["created"] == "2026-10-12T08:14:03Z" and one["closed"] == ""
    closed = tickets["tracker_ticket:redmine:4799"].attrs
    assert closed["closed"] == "2026-10-08T15:30:00Z" and closed["assignee"] == ""


def test_journal_notes_are_searchable_text_and_empty_journals_are_not_notes(tmp_path):
    write_tickets_cache(tmp_path)
    _, facts = _run(tmp_path)
    one = _tickets(facts)["tracker_ticket:redmine:4812"].attrs
    assert one["note_count"] == 1
    assert "Grace Hopper: Drained the node" in one["text"]


def test_no_email_address_reaches_a_node(tmp_path):
    write_tickets_cache(tmp_path)
    _, facts = _run(tmp_path)
    assert "@" not in json.dumps([f.attrs for f in facts if isinstance(f, NodeFact)])


def test_author_and_assignee_edges(tmp_path):
    write_tickets_cache(tmp_path)
    _, facts = _run(tmp_path)
    people = {f.node_id: f.attrs["display_name"] for f in facts if isinstance(f, NodeFact) and f.subtype == "person"}
    edges = {(e.src, e.edge_type): people[e.dst] for e in facts if isinstance(e, EdgeFact)}
    assert edges[("tracker_ticket:redmine:4812", "reported_by")] == "Ada Lovelace"
    assert edges[("tracker_ticket:redmine:4812", "assigned_to")] == "Grace Hopper"
    assert ("tracker_ticket:redmine:4799", "assigned_to") not in edges


def test_the_flat_shape_reads_like_the_rest_shape(tmp_path):
    flat = [{
        "system": "jira-like", "key": "OPS-7", "subject": "Disk alarm", "status": "Open", "author": "Ada Lovelace",
        "notes": [{"author": "Grace Hopper", "created": "2026-10-13T00:00:00Z", "body": "looking"}],
    }]
    (tmp_path / "tickets").mkdir()
    (tmp_path / "tickets" / "records.json").write_text(json.dumps(flat))
    run, facts = _run(tmp_path, system="redmine")
    ticket = _tickets(facts)["tracker_ticket:jira-like:OPS-7"].attrs
    assert ticket["note_count"] == 1 and ticket["label"] == "jira-like#OPS-7"


def test_the_same_number_in_two_systems_stays_two_tickets_and_two_people(tmp_path):
    rows = [
        {"system": "redmine", "key": "1", "subject": "a", "author": "Ada Lovelace"},
        {"system": "other", "key": "1", "subject": "b", "author": "Ada Lovelace"},
    ]
    (tmp_path / "tickets").mkdir()
    (tmp_path / "tickets" / "records.json").write_text(json.dumps(rows))
    _, facts = _run(tmp_path)
    assert len(_tickets(facts)) == 2
    assert len([f for f in facts if isinstance(f, NodeFact) and f.subtype == "person"]) == 2


def test_a_duplicate_or_keyless_item_is_skipped_and_the_scope_is_not_complete(tmp_path):
    rows = [{"key": "1", "subject": "a"}, {"key": "1", "subject": "again"}, {"subject": "no key"}]
    (tmp_path / "tickets").mkdir()
    (tmp_path / "tickets" / "records.json").write_text(json.dumps(rows))
    run, facts = _run(tmp_path)
    assert run.completed_scope is False and len(_tickets(facts)) == 1
    assert "skipped 2" in run.health.reason


def test_a_meta_count_that_disagrees_withholds_the_complete_scope(tmp_path):
    write_tickets_cache(tmp_path)
    (tmp_path / "tickets" / "meta.json").write_text(json.dumps({"record_count": 50}))
    run, _ = _run(tmp_path)
    assert run.completed_scope is False and run.health.status == "endpoint_failed"


@pytest.mark.parametrize("content", [None, "", "[", "[]", "{}"])
def test_a_missing_or_empty_cache_is_a_failed_fetch(tmp_path, content):
    if content is not None:
        (tmp_path / "tickets").mkdir()
        (tmp_path / "tickets" / "records.json").write_text(content)
    run = TicketSource(base=str(tmp_path)).run("run-1", mode="scope_complete")
    assert list(run.facts) == [] and run.completed_scope is False
    assert run.health.status in {"cache_missing", "endpoint_failed"}


def test_the_key_is_named_not_read_and_the_system_is_validated(tmp_path, monkeypatch):
    monkeypatch.delenv("REDMINE_API_KEY", raising=False)
    write_tickets_cache(tmp_path)
    source = TicketSource(base=str(tmp_path))
    assert source.preflight().credential_refs == ("REDMINE_API_KEY",)
    assert source.preflight().status == "ok"
    with pytest.raises(ValueError, match="system"):
        TicketSource(base=str(tmp_path), system="Red Mine")
    assert TicketAdapter.profile == "mutable_api"


# --- JIRA (the tickets downloader's JIRA reader, over recorded search pages) -------

def test_jira_issues_become_tickets_keyed_apart_from_redmine(tmp_path):
    write_jira_tickets_cache(tmp_path)
    run, facts = _run(tmp_path, system="jira", base_url="https://jira.example.org")
    assert run.completed_scope is True and run.health.record_count == 3
    tickets = _tickets(facts)
    assert set(tickets) == {"tracker_ticket:jira:OPS-12", "tracker_ticket:jira:OPS-9", "tracker_ticket:jira:OPS-3"}
    one = tickets["tracker_ticket:jira:OPS-12"].attrs
    assert (one["ticket_id"], one["system"], one["label"]) == ("OPS-12", "jira", "jira#OPS-12")
    assert (one["project"], one["tracker"], one["status"], one["priority"]) == (
        "Cluster Operations", "Incident", "In Progress", "High")
    assert one["url"] == "https://jira.example.org/browse/OPS-12"
    assert one["note_count"] == 1 and "Dorothy Vaughan: Reseated the GPU" in one["text"]
    assert tickets["tracker_ticket:jira:OPS-9"].attrs["closed"] == "2026-10-06T16:45:00.000+0000"


def test_jira_reporter_and_assignee_edges_and_no_address_reaches_a_node(tmp_path):
    write_jira_tickets_cache(tmp_path)
    _, facts = _run(tmp_path)
    people = {f.node_id: f.attrs["display_name"] for f in facts if isinstance(f, NodeFact) and f.subtype == "person"}
    assert sorted(people.values()) == ["Dorothy Vaughan", "Katherine Johnson", "Margaret Hamilton"]
    edges = {(e.src, e.edge_type): people[e.dst] for e in facts if isinstance(e, EdgeFact)}
    assert edges[("tracker_ticket:jira:OPS-12", "reported_by")] == "Katherine Johnson"
    assert edges[("tracker_ticket:jira:OPS-12", "assigned_to")] == "Dorothy Vaughan"
    assert ("tracker_ticket:jira:OPS-9", "assigned_to") not in edges
    # The recorded description carries an address; the reader removes it as it does for Redmine.
    assert "@" not in json.dumps([f.attrs for f in facts if isinstance(f, NodeFact)])


def test_a_missing_jira_cache_names_the_jira_token_not_the_redmine_key(tmp_path):
    source = TicketSource(base=str(tmp_path), system="jira")
    assert source.preflight().credential_refs == ("JIRA_API_TOKEN",)
    run = source.run("run-1", mode="scope_complete")
    assert run.health.status == "cache_missing" and run.health.credential_refs == ("JIRA_API_TOKEN",)
    assert TicketSource(base=str(tmp_path)).run("r", mode="scope_complete").health.credential_refs == ("REDMINE_API_KEY",)
    assert TicketSource(base=str(tmp_path), system="other", credential_ref="").credential_refs == ()


def test_restricted_comments_and_mentions_never_reach_a_node(tmp_path):
    write_jira_tickets_cache(tmp_path)
    _, facts = _run(tmp_path, system="jira")
    attrs = json.dumps([f.attrs for f in facts if isinstance(f, NodeFact)])
    assert "Admins only" not in attrs and "[~mhamilton]" not in attrs and "[~user]" in attrs
    assert _tickets(facts)["tracker_ticket:jira:OPS-3"].attrs["note_count"] == 0


def test_a_jira_ticket_without_a_usable_link_of_its_own_links_under_browse(tmp_path):
    (tmp_path / "tickets").mkdir()
    rows = [{"system": "jira", "key": "OPS-1", "subject": "a"},
            {"system": "jira", "key": "OPS-2", "subject": "b", "url": "javascript:alert(1)"}]
    (tmp_path / "tickets" / "records.json").write_text(json.dumps(rows))
    _, facts = _run(tmp_path, base_url="https://jira.example.org/")
    tickets = _tickets(facts)
    assert tickets["tracker_ticket:jira:OPS-1"].attrs["url"] == "https://jira.example.org/browse/OPS-1"
    assert tickets["tracker_ticket:jira:OPS-2"].attrs["url"] == "https://jira.example.org/browse/OPS-2"
