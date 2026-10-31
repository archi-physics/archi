"""req.r3-auto-ticket-triage.automation, the parts that need no worker: the shipped pack and how it renders
in both bundles, the ticket shape, the scanner's new-or-updated rule, the run's read, and the proposer."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.automation import automation_problems
from archi.bundles.render import render
from archi.bundles.resolve import CompositionError, resolve
from archi.ops.proposers import PROPOSER_KINDS, ProposeRequest, ProposerError, build_proposer
from archi.triage.context import ContextError, read_ticket_context
from archi.triage.proposer import TriageProposer
from archi.triage.signals import add_ticket_signals, new_or_updated, scan_queue
from archi.triage.tickets import Ticket, TicketError, parse_entity, parse_time, signal_id
from ticket_fakes import Graph, arrival, load_cache, queue_rows, to_row

REPO = Path(__file__).resolve().parents[3]
PACK = REPO / "bundles" / "packs" / "ticket-triage"
NAME = "ticket-triage"
NOW = "2026-10-26T09:00:00Z"


@pytest.fixture(scope="module")
def catalog():
    found = discovery.from_roots([REPO / "bundles" / "packs"], [REPO / "bundles"])
    assert found.problems == [], [str(p) for p in found.problems]
    return found


def block(catalog, tmp_path, bundle, with_packs):
    rendered = render(resolve(catalog, [bundle], with_packs=with_packs), tmp_path / "instance")
    return yaml.safe_load((tmp_path / "instance" / "automations" / f"{NAME}.yaml").read_text()), rendered


# --- the pack ----------------------------------------------------------------------------


def test_the_pack_is_an_automation_that_starts_at_l1_and_allows_at_most_l2(catalog):
    pack = catalog.packs[NAME]
    auto = pack.manifest["automation"]
    assert pack.kind == "automation" and auto["default_level"] == "L1" and auto["max_level"] == "L2"
    assert auto["trigger"] == {"type": "ticket_queue", "system": ["redmine", "jira", "servicenow"], "queue": "*"}
    assert auto["context_gate"]["sources"] == ["tickets.tickets", "jira.jira", "servicenow.servicenow"]
    assert (PACK / "skills" / auto["playbook"]).is_file()


def test_the_action_classes_are_one_draft_at_l1_and_two_live_tools_at_l2(catalog):
    classes = yaml.safe_load((PACK / "action_classes.yaml").read_text())["action_classes"]
    assert {n: (c["mediator"], c["max_level"]) for n, c in classes.items()} == {
        "draft_reply": ("draft_reply", "L1"),
        "set_ticket_status": ("archi.tools.tickets:set_ticket_status", "L2"),
        "assign_ticket": ("archi.tools.tickets:assign_ticket", "L2"),
    }


def test_the_tool_payload_schemas_are_the_action_class_schemas():
    from archi.tools import tickets

    classes = yaml.safe_load((PACK / "action_classes.yaml").read_text())["action_classes"]
    for name, schema in tickets.PAYLOAD_SCHEMAS.items():
        assert classes[name]["payload_schema"] == schema, name
    manifest = yaml.safe_load((PACK / "pack.yaml").read_text())
    tools = {f"archi.tools.tickets:{n}" for n in tickets.TOOLS}
    assert set(manifest["live_tools"]) == tools and set(manifest["idempotent_tools"]) == tools


def test_the_pack_is_generic():
    """cluster-manager is for any cluster operator: nothing in the pack names a lab or an experiment."""
    for path in PACK.rglob("*"):
        if path.is_file():
            assert not re.search(r"\b(cern|cms|submit|mit)\b", path.read_text(), re.I), path


def test_the_proposer_kind_is_registered():
    assert PROPOSER_KINDS[NAME] == "archi.triage.proposer:build"
    assert isinstance(build_proposer({"kind": NAME}, None), TriageProposer)


@pytest.mark.parametrize("trigger, fragment", [
    ({"type": "ticket_queue", "system": ["redmine", "gitlab"], "queue": "*"}, "'gitlab' is not one of"),
    ({"type": "ticket_queue", "system": ["jira", "jira"], "queue": "*"}, "listed more than once"),
    ({"type": "ticket_queue", "system": "trac", "queue": "q"}, "'trac' is not one of"),
])
def test_validate_refuses_a_bad_ticket_queue_trigger(trigger, fragment):
    data = yaml.safe_load((PACK / "pack.yaml").read_text())
    data["automation"]["trigger"] = trigger
    messages = [p.message for p in automation_problems(PACK, data, check_requires=False)]
    assert any(fragment in m for m in messages), messages


def test_validate_accepts_the_shipped_pack_and_refuses_narrowing_a_gate_with_no_source():
    data = yaml.safe_load((PACK / "pack.yaml").read_text())
    assert automation_problems(PACK, data, check_requires=False) == []
    data["automation"]["context_gate"] = {"invariants": ["tickets_coverage"]}
    messages = [p.message for p in automation_problems(PACK, data, check_requires=False)]
    assert any("narrow_to_selected: the context gate names no source" in m for m in messages)


# --- rendered in the two bundles ----------------------------------------------------------


def test_cluster_manager_triages_the_tickets_pack_redmine_and_jira_and_gates_on_it_alone(catalog, tmp_path):
    """The tickets pack reads Redmine and any JIRA, so both systems stay; ServiceNow needs its own pack."""
    auto, _ = block(catalog, tmp_path, "cluster-manager", [NAME])
    assert auto["trigger"]["system"] == ["redmine", "jira"] and auto["context_gate"]["sources"] == ["tickets.tickets"]
    assert auto["not_selected"] == {"sources": ["jira.jira", "servicenow.servicenow"], "systems": ["servicenow"]}
    assert "narrow_to_selected" not in auto and auto["pack_max_level"] == "L2"


def test_cluster_manager_with_servicenow_triages_both(catalog, tmp_path):
    auto, rendered = block(catalog, tmp_path, "cluster-manager", ["servicenow", NAME])
    assert auto["trigger"]["system"] == ["redmine", "jira", "servicenow"]
    assert auto["context_gate"]["sources"] == ["tickets.tickets", "servicenow.servicenow"]
    assert (rendered.profile_dir / "skills" / "triage.md").is_file()


def test_cms_comp_ops_triages_the_jira_queue(catalog, tmp_path):
    auto, _ = block(catalog, tmp_path, "cms-comp-ops", ["jira", NAME])
    assert auto["trigger"]["system"] == ["jira"] and auto["context_gate"]["sources"] == ["jira.jira"]


def test_an_instance_with_no_ticket_pack_is_refused_at_render(catalog, tmp_path):
    with pytest.raises(CompositionError) as err:
        render(resolve(catalog, ["cms-comp-ops"], with_packs=[NAME]), tmp_path / "instance")
    assert err.value.code == "automation_inputs_not_selected" and "jira" in str(err.value)


# --- tickets -----------------------------------------------------------------------------


def test_a_generic_jira_ticket_from_the_tickets_pack_is_a_jira_ticket():
    rows = {r["node_id"]: r for r in queue_rows()}
    t = Ticket.from_row(rows["tracker_ticket:jira:SUP-1"])
    assert (t.system, t.entity, t.key, t.closed, t.assigned_to, t.queue) == (
        "jira", "jira:SUP-1", "SUP-1", True, "Grace Hopper", "SUP")
    other = {"node_id": "tracker_ticket:trac:9", "subtype": "tracker_ticket", "attrs": {"ticket_id": "9", "system": "trac"}}
    with pytest.raises(TicketError, match="'trac' ticket"):
        Ticket.from_row(other)
    unnamed = {"node_id": "tracker_ticket:x:9", "subtype": "tracker_ticket", "attrs": {"ticket_id": "9"}}
    assert Ticket.from_row(unnamed).system == "redmine"


def test_each_system_reads_into_one_shape():
    rows = {r["node_id"]: r for r in queue_rows()}
    rm = Ticket.from_row(rows["tracker_ticket:redmine:4799"])
    assert (rm.entity, rm.closed, rm.assigned_to, rm.category, rm.components) == (
        "redmine:4799", True, "Grace Hopper", "Support", ("Cluster Ops",))
    jr = Ticket.from_row(rows["jira:OPS-14"])
    assert (jr.entity, jr.closed, jr.queue, jr.components, jr.category) == ("jira:OPS-14", False, "OPS", ("monitoring",), "Task")
    sn = Ticket.from_row(rows["servicenow_incident:INC0010001"])
    assert (sn.entity, sn.closed, sn.assigned_to, sn.table, sn.queue) == (
        "servicenow:INC0010001", True, "Cluster Operations", "incident", "Cluster Operations")
    with pytest.raises(TicketError):
        Ticket.from_row({"node_id": "x", "subtype": "documentation_page", "attrs": {}})


@pytest.mark.parametrize("text", ["2026-09-14T08:15:00.000+0000", "2026-09-14T08:15:00Z", "2026-09-14 08:15:00",
                                  "2026-09-14T10:15:00+02:00"])
def test_ticket_times_read_in_every_system_format(text):
    assert parse_time(text) == datetime(2026, 9, 14, 8, 15, tzinfo=timezone.utc)


def test_a_signal_id_is_one_per_ticket_version_and_a_legal_run_id_part():
    t = Ticket.from_row(to_row("jira", arrival("jira", "OPS-15", "x", "y", "2026-10-26T09:00:00.000+0000")))
    assert signal_id(t) == "jira-OPS-15-20261026T090000Z" and "~" not in signal_id(t)
    assert parse_entity("servicenow:INC0010003") == ("servicenow", "INC0010003")
    with pytest.raises(TicketError):
        parse_entity("gitlab:12")


# --- signals -----------------------------------------------------------------------------


def cache_item(system, key):
    return next(i for i in load_cache(system) if i.get("key", i.get("number")) == key)


def _new(system, key, subject, updated, **kw):
    return to_row(system, arrival(system, key, subject, "jobs fail with ENOSPC on /scratch", updated, **kw))


def test_the_first_read_sets_the_cursor_and_triages_nothing():
    g = Graph(queue_rows())
    assert scan_queue(g, NAME, ["redmine", "jira", "servicenow"]) == []
    assert g.signals == {}
    rm = g.cursors[(NAME, "redmine:tracker_ticket@https://redmine.example.org/issues/")]
    assert rm["updated"] == parse_time("2026-10-10T14:00:00Z") and rm["seen"] == ["redmine:4811"]
    assert g.cursors[(NAME, "jira:jira_issue@https://jira.example.org/browse/")]["generation"] == "gen-1"
    assert g.cursors[(NAME, "jira/*")]["generation"] == "gen-1"


def test_a_new_or_updated_open_ticket_in_a_newer_generation_becomes_one_signal():
    g = Graph(queue_rows())
    scan_queue(g, NAME, ["redmine", "jira", "servicenow"])
    reopened = to_row("redmine", {**cache_item("redmine", "4811"), "updated": NOW})
    closed = to_row("redmine", {**arrival("redmine", "4830", "Closed already", "x", NOW), "status": "Closed"})
    g.publish([_new("redmine", "4820", "Scratch filesystem full on node051", NOW), reopened,
               _new("jira", "OPS-15", "SRM_TIMEOUT after the upgrade", "2026-10-26T09:01:00.000+0000"),
               _new("servicenow", "INC0010003", "scratch filesystem full on node051", NOW), closed])
    added = scan_queue(g, NAME, ["redmine", "jira", "servicenow"])
    assert sorted(added) == sorted(f"{NAME}/{s}" for s in (
        "redmine-4811-20261026T090000Z", "redmine-4820-20261026T090000Z", "jira-OPS-15-20261026T090100Z",
        "servicenow-INC0010003-20261026T090000Z"))
    sig = g.signals[(NAME, "redmine-4820-20261026T090000Z")]
    assert sig["entity"] == "redmine:4820" and sig["source"] == "ticket" and "node051" in sig["title"]
    assert scan_queue(g, NAME, ["redmine", "jira", "servicenow"]) == []  # same generation: not read again
    g.publish([])
    assert scan_queue(g, NAME, ["redmine", "jira", "servicenow"]) == []  # nothing newer than the cursor


def test_jira_tickets_come_from_either_pack_and_one_issue_makes_one_signal():
    g = Graph(queue_rows())
    scan_queue(g, NAME, ["redmine", "jira"])
    generic = to_row("redmine", arrival("redmine", "SUP-3", "Jobs held: home quota exceeded", "quota", NOW, tracker="jira"))
    both = [_new("jira", "OPS-20", "Same issue in both packs", NOW),
            to_row("redmine", arrival("redmine", "OPS-20", "Same issue in both packs", "x", NOW, tracker="jira"))]
    g.publish([generic, *both])
    added = scan_queue(g, NAME, ["redmine", "jira"])
    assert sorted(added) == [f"{NAME}/jira-OPS-20-20261026T090000Z", f"{NAME}/jira-SUP-3-20261026T090000Z"]
    assert g.signals[(NAME, "jira-SUP-3-20261026T090000Z")]["entity"] == "jira:SUP-3"


def test_only_the_selected_packs_subtypes_are_read():
    g = Graph(queue_rows())
    g.missing_subtypes = {"jira_issue"}  # the instance has no jira pack: okg would refuse the subtype
    scan_queue(g, NAME, ["redmine", "jira"], packs={"tickets"})
    assert g.cursors[(NAME, "jira:tracker_ticket@https://jira.example.org/browse/")]["updated"] == parse_time("2026-09-11T08:00:00Z")
    assert not any("jira_issue" in k[1] for k in g.cursors)  # SUP-2, the newest generic JIRA; no jira pack


def test_the_instance_queues_narrow_the_signals():
    tickets = [Ticket.from_row(_new("servicenow", "INC0010003", "a", NOW, group="")),
               Ticket.from_row(_new("redmine", "4820", "b", NOW))]
    fresh, _, _ = new_or_updated(tickets, parse_time("2026-10-01T00:00:00Z"), queues=["Cluster Ops"])
    assert [t.key for t in fresh] == ["4820"]


def test_a_source_that_publishes_late_with_older_times_is_not_skipped():
    """#746 finding 5: one cursor per system skipped, for good, a source that published after another one with
    older update times. Each source (subtype at a site) keeps its own cursor."""
    g = Graph(queue_rows())
    scan_queue(g, NAME, ["redmine", "jira"])
    g.publish([_new("jira", "OPS-15", "SRM_TIMEOUT after the upgrade", "2026-10-26T09:00:00.000+0000")])
    assert scan_queue(g, NAME, ["redmine", "jira"]) == [f"{NAME}/jira-OPS-15-20261026T090000Z"]
    # the tickets pack's generic JIRA reader publishes later, with a ticket older than OPS-15
    g.publish([to_row("redmine", arrival("redmine", "SUP-3", "Jobs held: home quota", "x", "2026-10-20T08:00:00Z",
                                         tracker="jira"))])
    assert scan_queue(g, NAME, ["redmine", "jira"]) == [f"{NAME}/jira-SUP-3-20261020T080000Z"]


def test_a_ticket_updated_in_the_same_second_as_the_cursor_is_triaged_once():
    g = Graph(queue_rows())
    scan_queue(g, NAME, ["redmine"])  # cursor 2026-10-10T14:00:00Z, at which 4811 was read (and not triaged)
    g.publish([_new("redmine", "4821", "Same second as the cursor", "2026-10-10T14:00:00Z")])
    assert scan_queue(g, NAME, ["redmine"]) == [f"{NAME}/redmine-4821-20261010T140000Z"]
    g.publish([])
    assert scan_queue(g, NAME, ["redmine"]) == []


def test_one_issue_in_two_packs_with_different_times_makes_one_signal_and_never_an_older_one():
    """#746 finding 6: the two JIRA paths were deduplicated only on identical update times."""
    g = Graph(queue_rows())
    scan_queue(g, NAME, ["redmine", "jira"])
    g.publish([_new("jira", "OPS-20", "In both packs", "2026-10-26T09:05:00.000+0000"),
               to_row("redmine", arrival("redmine", "OPS-20", "In both packs", "x", "2026-10-26T09:00:00Z", tracker="jira"))])
    assert scan_queue(g, NAME, ["redmine", "jira"]) == [f"{NAME}/jira-OPS-20-20261026T090500Z"]
    # the generic reader catches up with a version still older than the one signalled
    g.publish([to_row("redmine", arrival("redmine", "OPS-20", "In both packs", "x", "2026-10-26T09:03:00Z", tracker="jira"))])
    assert scan_queue(g, NAME, ["redmine", "jira"]) == []


def test_a_new_signal_supersedes_the_tickets_waiting_signals():
    """#746 finding 7: waiting signals were never collapsed to the latest per ticket."""
    g = Graph(queue_rows())
    scan_queue(g, NAME, ["redmine"])
    g.publish([_new("redmine", "4820", "Scratch full on node051", NOW)])
    scan_queue(g, NAME, ["redmine"])
    g.publish([_new("redmine", "4820", "Scratch full on node051", "2026-10-26T09:30:00Z")])
    assert scan_queue(g, NAME, ["redmine"]) == [f"{NAME}/redmine-4820-20261026T093000Z"]
    status = {sid: s["status"] for (_, sid), s in g.signals.items()}
    assert status == {"redmine-4820-20261026T090000Z": "superseded", "redmine-4820-20261026T093000Z": "new"}




def test_the_scanner_reads_only_ticket_queue_automations_and_a_failure_stays_with_its_automation():
    from types import SimpleNamespace

    class Auto:
        def __init__(self, trigger):
            self.trigger = trigger

    g = Graph(queue_rows())
    g.missing_subtypes = {"jira_issue"}
    autos = {"tt": Auto({"type": "ticket_queue", "system": ["jira"], "queue": "*"}),
             "ok": Auto({"type": "ticket_queue", "system": ["redmine"], "queue": "*"}),
             "health": Auto({"type": "schedule", "cron": "7 * * * *"})}
    settings = {n: SimpleNamespace(proposer={}, queues=()) for n in autos}
    said = []
    rt = SimpleNamespace(settings=settings, automation=lambda n: autos[n])
    assert add_ticket_signals(rt, g, said.append) == []
    assert ("ok", "redmine/*") in g.cursors and ("tt", "jira/*") not in g.cursors
    assert not any(a == "health" for a, _ in g.cursors)
    assert any("tt: cannot read its ticket queue" in s for s in said)


# --- the run's read ----------------------------------------------------------------------


def precedent(entity, rid="inc-old"):
    return {"node_id": f"ops_incident:{rid}", "subtype": "ops_incident",
            "attrs": {"record_id": rid, "entity": entity, "title": "triage of an earlier scratch ticket",
                      "status": "resolved", "detected_at": "2026-10-03T10:00:00Z", "automation": NAME}}


def graph_with_arrivals():
    g = Graph(queue_rows() + [precedent("redmine:4799")])
    g.publish([_new("redmine", "4820", "Scratch filesystem full on node051", NOW),
               _new("jira", "OPS-15", "Transfers failing with SRM_TIMEOUT after the upgrade", NOW),
               _new("servicenow", "INC0010003", "scratch filesystem full on node051", NOW)])
    return g


def test_the_read_returns_the_ticket_related_tickets_docs_and_precedents():
    g = graph_with_arrivals()
    g.missing_subtypes = {"documentation_page"}
    d = read_ticket_context(g, "redmine:4820", g.generation)
    ctx = d["context"]
    assert ctx["ticket"]["key"] == "4820" and ctx["ticket"]["entity"] == "redmine:4820"
    related = [r["key"] for r in ctx["related"]]
    assert related[:2] in (["4790", "4799"], ["4799", "4790"]) and "4820" not in related
    assert [p["node_id"] for p in ctx["precedents"]] == ["ops_incident:inc-old"]
    assert [r["node_id"] for r in d["rows"]] == ["ops_incident:inc-old"]  # carried with a proposal that cites it
    assert [doc["node_id"] for doc in ctx["docs"]] == ["servicenow_kb_article:KB0000010"]
    assert [s["what"] for s in ctx["skipped"]] == ["docs:documentation_page"]  # this instance has no such subtype
    assert set(d["node_ids"]) >= {"tracker_ticket:redmine:4820", "tracker_ticket:redmine:4799", "ops_incident:inc-old"}


def test_a_generic_jira_ticket_reads_its_related_jira_tickets_only():
    g = Graph(queue_rows())
    g.publish([to_row("redmine", arrival("redmine", "SUP-3", "Jobs held: home quota exceeded",
                                         "Jobs stay held, home quota exceeded.", NOW, tracker="jira"))])
    g.missing_subtypes = {"jira_issue"}
    d = read_ticket_context(g, "jira:SUP-3", g.generation, {"doc_subtypes": []}, packs={"tickets"})
    assert d["context"]["ticket"]["node_id"] == "tracker_ticket:jira:SUP-3"
    assert [r["key"] for r in d["context"]["related"]] == ["SUP-2", "SUP-1"] or \
        [r["key"] for r in d["context"]["related"]] == ["SUP-1", "SUP-2"]  # no Redmine ticket, though both are tracker_tickets


def test_docs_come_from_the_named_subtypes():
    g = graph_with_arrivals()
    d = read_ticket_context(g, "servicenow:INC0010003", g.generation, {"doc_subtypes": ["servicenow_kb_article"]})
    assert [doc["node_id"] for doc in d["context"]["docs"]] == ["servicenow_kb_article:KB0000010"]
    assert "servicenow_kb_article:KB0000010" in d["node_ids"]


def test_a_ticket_missing_from_the_generation_is_an_error():
    g = Graph(queue_rows())
    with pytest.raises(ContextError, match="redmine:9999 is not in generation"):
        read_ticket_context(g, "redmine:9999", g.generation)


# --- the proposer ------------------------------------------------------------------------


def propose(entity, level, config=None, graph=None):
    g = graph or graph_with_arrivals()
    d = read_ticket_context(g, entity, g.generation, {"doc_subtypes": ["servicenow_kb_article"]})
    out = TriageProposer(config).propose(ProposeRequest(NAME, {"id": "s", "entity": entity}, d["context"], g.generation, level))
    return out, set(d["node_ids"])


def test_l1_drafts_a_cited_triage_note_and_reply():
    out, known = propose("redmine:4820", 1)
    p = out.proposal
    assert out.cost == 0.0 and p["action_class"] == "draft_reply"
    assert set(p["evidence_ids"]) <= known and "tracker_ticket:redmine:4820" in p["evidence_ids"]
    body = p["payload"]["body"]
    assert "Category: Support (from the ticket [1])" in body
    assert "Usually handled by: Grace Hopper (2 of 3 related tickets" in body
    assert "Precedents in the operational record:" in body and "ops_incident:inc-old" in body
    assert "Suggested reply:" in body and "Nothing was sent." in body
    # every [n] in the text resolves to a source line
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body.split("Sources:")[0])}
    sources = dict(re.findall(r"^\[(\d+)\] (\S+)$", body.split("Sources:")[1], re.M))
    assert cited and cited <= {int(k) for k in sources} and set(sources.values()) == set(p["evidence_ids"])
    assert 0 <= p["confidence"] <= 1


def test_l2_proposes_assigning_an_unassigned_ticket_to_whom_the_related_tickets_went():
    config = {"systems": {"redmine": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY"}}}
    p = propose("redmine:4820", 2, config)[0].proposal
    assert p["action_class"] == "assign_ticket"
    assert p["payload"] == {"system": "redmine", "ticket": "4820", "ticket_url": "https://redmine.example.org/issues/4820",
                            "field": "assignee", "value": "Grace Hopper", "expected": "", "note": p["payload"]["note"],
                            "endpoint": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY"}}
    assert "set assignee of redmine 4820 to Grace Hopper" in p["summary"] and "unassigned" in p["rollback"]


def test_l2_routes_a_servicenow_ticket_to_a_group():
    config = {"systems": {"servicenow": {"url": "https://example.service-now.com", "user_env": "SN_USER",
                                         "password_env": "SN_PASSWORD"}}}
    p = propose("servicenow:INC0010003", 2, config)[0].proposal
    assert p["action_class"] == "assign_ticket"
    assert {k: p["payload"][k] for k in ("field", "value", "table")} == {
        "field": "assignment_group", "value": "Cluster Operations", "table": "incident"}


def test_l2_moves_an_assigned_new_ticket_to_the_triaged_status():
    g = graph_with_arrivals()
    g.publish([to_row("redmine", {**cache_item("redmine", "4811"), "updated": NOW})])
    config = {"systems": {"redmine": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY",
                                      "triaged_status": "In Progress", "triage_from": ["New"]}}}
    p = propose("redmine:4811", 2, config, g)[0].proposal
    assert p["action_class"] == "set_ticket_status"
    assert {k: p["payload"][k] for k in ("status", "expected_status")} == {"status": "In Progress", "expected_status": "New"}


def test_l2_without_an_api_drafts_and_says_why():
    p = propose("redmine:4820", 2)[0].proposal
    assert p["action_class"] == "draft_reply" and "proposer.systems.redmine.url is not set" in p["summary"]


def test_a_context_without_a_ticket_is_a_proposer_error():
    with pytest.raises(ProposerError):
        TriageProposer().propose(ProposeRequest(NAME, {"id": "s"}, {"precedents": []}, "gen-1", 1))


# --- review findings (#746): the server a change goes to, rejections, what is stored --------------

JIRA_L2 = {"systems": {"jira": {"url": "https://jira.example.org", "token_env": "JIRA_TOKEN"}}}
OTHER = "https://jira.other.org"


def jira_pair(key, other_site=OTHER, both=True):
    """Two JIRA issues closed by Grace Hopper (so L2 would assign) and a new one, `key`, from the generic reader
    on `other_site` and, when `both`, also from the jira pack on the instance's own server."""
    g = Graph(queue_rows())
    rows = [to_row("redmine", arrival("redmine", key, "Jobs held: home quota exceeded", "home quota exceeded", NOW,
                                      tracker="jira"), site=other_site)]
    if both:
        rows.append(_new("jira", key, "Jobs held: home quota exceeded", "2026-10-26T09:00:00.000+0000"))
    g.publish(rows)
    return g


def test_a_key_on_two_servers_is_never_changed():
    """#746 blocker 2: with the jira pack and the generic reader on different servers, `jira:<KEY>` named two
    tickets, and an approved change could land on the wrong one."""
    g = jira_pair("OPS-30")
    d = read_ticket_context(g, "jira:OPS-30", g.generation, JIRA_L2)
    assert d["context"]["sites"] == ["https://jira.example.org/browse/", f"{OTHER}/browse/"]
    assert d["context"]["ticket"]["root"] == "https://jira.example.org/browse/"  # the copy on the server it writes to
    p = TriageProposer(JIRA_L2).propose(ProposeRequest(NAME, {"id": "s"}, d["context"], g.generation, 2)).proposal
    assert p["action_class"] == "draft_reply" and "is on more than one jira server" in p["summary"]


def test_a_ticket_from_another_server_than_the_endpoint_is_not_changed():
    g = jira_pair("SUP-30", both=False)
    p = propose("jira:SUP-30", 2, JIRA_L2, g)[0].proposal
    assert p["action_class"] == "draft_reply"
    assert f"link ({OTHER}/browse/SUP-30) is not under https://jira.example.org" in p["summary"]


def test_a_site_names_the_link_root_when_the_api_is_a_proxy():
    g = jira_pair("SUP-31", both=False)
    config = {"systems": {"jira": {"url": "http://127.0.0.1:8099", "site": OTHER, "token_env": "JIRA_TOKEN"}}}
    p = propose("jira:SUP-31", 2, config, g)[0].proposal
    assert p["action_class"] == "assign_ticket" and p["payload"]["ticket_url"] == f"{OTHER}/browse/SUP-31"
    assert p["payload"]["endpoint"] == {"url": "http://127.0.0.1:8099", "site": OTHER, "token_env": "JIRA_TOKEN"}


def rejected_change(entity, action_class, payload, rid="4820", verdict="rejected"):
    return [{"node_id": f"ops_incident:inc-{rid}", "subtype": "ops_incident",
             "attrs": {"record_id": f"inc-{rid}", "entity": entity, "title": "earlier triage", "status": "open",
                       "detected_at": "2026-10-26T08:00:00Z", "automation": NAME}},
            {"node_id": f"ops_proposal:prop-{rid}", "subtype": "ops_proposal",
             "attrs": {"record_id": f"prop-{rid}", "incident_id": f"inc-{rid}", "action_class": action_class,
                       "payload": json.dumps(payload, sort_keys=True), "summary": "x", "evidence_ids": ["x"]}},
            {"node_id": f"ops_decision:dec-{rid}", "subtype": "ops_decision",
             "attrs": {"record_id": f"dec-{rid}", "proposal_id": f"prop-{rid}", "verdict": verdict,
                       "actor_kind": "person"}}]


def test_a_change_a_person_rejected_is_not_proposed_again_on_the_next_comment():
    """#746 finding 7: a rejected proposal was proposed again on every comment."""
    g = graph_with_arrivals()
    g.publish(rejected_change("redmine:4820", "assign_ticket", {"value": "Grace Hopper", "field": "assignee"}))
    config = {"systems": {"redmine": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY",
                                      "triaged_status": "In Progress", "triage_from": ["New"]}}}
    out, _ = propose("redmine:4820", 2, config, g)
    p = out.proposal
    assert p["action_class"] == "set_ticket_status"  # the other change is still open to propose
    g.publish(rejected_change("redmine:4820", "set_ticket_status", {"status": "In Progress"}, rid="4820b"))
    p = propose("redmine:4820", 2, config, g)[0].proposal
    assert p["action_class"] == "draft_reply"
    assert "assigning it to Grace Hopper was rejected (ops_proposal:prop-4820) before" in p["summary"]
    assert "moving it to In Progress was rejected (ops_proposal:prop-4820b) before" in p["summary"]


def test_a_rejection_on_another_ticket_does_not_count():
    g = graph_with_arrivals()
    g.publish(rejected_change("redmine:4799", "assign_ticket", {"value": "Grace Hopper"}, rid="4799"))
    config = {"systems": {"redmine": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY"}}}
    assert propose("redmine:4820", 2, config, g)[0].proposal["action_class"] == "assign_ticket"


def test_the_note_on_the_ticket_names_related_tickets_by_key_not_node_id():
    """#746 finding 8: the note posted on the ticket showed internal node ids."""
    config = {"systems": {"redmine": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY"}}}
    note = propose("redmine:4820", 2, config)[0].proposal["payload"]["note"]
    assert "redmine 4799" in note and "tracker_ticket:" not in note


def test_the_read_step_output_holds_no_ticket_description_or_search_text():
    """#746 finding 8: the full ticket description was stored in the workflow step output (DBOS keeps it)."""
    g = graph_with_arrivals()
    ctx = read_ticket_context(g, "redmine:4820", g.generation)["context"]
    assert "description" not in ctx["ticket"] and all("description" not in r for r in ctx["related"])
    assert "query" not in ctx and "jobs fail with ENOSPC on /scratch" not in json.dumps(ctx)  # the arrival text


def test_ticket_text_has_email_addresses_removed():
    row = _new("servicenow", "INC0010009", "Mail ada.lovelace@example.org about it", "2026-10-26T09:00:00Z")
    row["attrs"]["description"] = "Contact grace@example.org"
    t = Ticket.from_row(row)
    assert "@" not in t.subject and "@" not in t.description


# --- second review (x4): what the first round of fixes got wrong -----------------------------------


def test_one_tracker_source_holding_two_systems_keeps_a_cursor_per_system():
    """A tickets source with no links holds Redmine and JIRA tickets under one subtype and site: the system is
    in the cursor key, so one system's newer times never move the other's cursor."""
    def bare(item):
        row = to_row("redmine", item)
        row["attrs"]["url"] = ""
        return row

    g = Graph([bare(i) for i in load_cache("redmine")])
    scan_queue(g, NAME, ["redmine", "jira"])
    assert {k[1] for k in g.cursors} >= {"redmine:tracker_ticket@-", "jira:tracker_ticket@-"}
    # a Redmine ticket far newer, and a JIRA one between JIRA's cursor (2026-09-11) and it
    g.publish([bare(arrival("redmine", "4830", "new redmine", "x", NOW)),
               bare(arrival("redmine", "SUP-9", "new jira", "x", "2026-10-01T08:00:00Z", tracker="jira"))])
    added = scan_queue(g, NAME, ["redmine", "jira"])
    assert sorted(added) == [f"{NAME}/jira-SUP-9-20261001T080000Z", f"{NAME}/redmine-4830-20261026T090000Z"]


def test_cursors_do_not_move_when_adding_a_signal_fails():
    g = Graph(queue_rows())
    scan_queue(g, NAME, ["redmine"])
    before = dict(g.cursors)
    g.publish([_new("redmine", "4820", "Scratch full on node051", NOW)])
    real = g.add_signal

    def broken(*a, **k):
        raise RuntimeError("database gone")

    g.add_signal = broken
    with pytest.raises(RuntimeError):
        scan_queue(g, NAME, ["redmine"])
    assert g.cursors == before
    g.add_signal = real
    assert scan_queue(g, NAME, ["redmine"]) == [f"{NAME}/redmine-4820-20261026T090000Z"]


def test_a_copy_with_no_link_makes_the_key_ambiguous():
    g = jira_pair("OPS-31")
    for row in g.rows.values():
        if row["node_id"] == "tracker_ticket:jira:OPS-31":
            row["attrs"]["url"] = ""
    d = read_ticket_context(g, "jira:OPS-31", g.generation, JIRA_L2)
    p = TriageProposer(JIRA_L2).propose(ProposeRequest(NAME, {"id": "s"}, d["context"], g.generation, 2)).proposal
    assert p["action_class"] == "draft_reply" and "(no link)" in p["summary"]


def test_two_servers_on_one_host_are_told_apart_by_path():
    g = jira_pair("OPS-32", other_site="https://jira.example.org/second")
    d = read_ticket_context(g, "jira:OPS-32", g.generation, JIRA_L2)
    assert len(d["context"]["sites"]) == 2


def test_a_site_other_than_the_url_is_allowed_only_behind_a_loopback_proxy():
    g = jira_pair("SUP-33", both=False)
    config = {"systems": {"jira": {"url": "https://jira.example.org", "site": OTHER, "token_env": "JIRA_TOKEN"}}}
    p = propose("jira:SUP-33", 2, config, g)[0].proposal
    assert p["action_class"] == "draft_reply" and "site differs from its url" in p["summary"]


def test_a_change_that_timed_out_is_not_proposed_again():
    g = graph_with_arrivals()
    g.publish(rejected_change("redmine:4820", "assign_ticket", {"value": "Grace Hopper"}, verdict="timed_out"))
    config = {"systems": {"redmine": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY"}}}
    p = propose("redmine:4820", 2, config, g)[0].proposal
    assert p["action_class"] == "draft_reply" and "was let time out (ops_proposal:prop-4820)" in p["summary"]


def test_compared_fields_are_not_redacted_only_shown_text_is():
    row = _new("redmine", "4840", "Ask bob@example.org", "2026-10-26T09:00:00Z", assignee="ada@example.org")
    t = Ticket.from_row(row)
    assert t.assignee == "ada@example.org" and "@" not in t.subject


def test_a_ticket_time_far_ahead_of_the_clock_does_not_move_the_cursor():
    """Second review: one row with a far-future `updated` set the source's cursor to that time, and every real
    update after it read as older and was skipped. It is now left out until the clock reaches it."""
    now = parse_time("2026-10-26T12:00:00Z")
    tickets = [Ticket.from_row(_new("redmine", "4820", "real", "2026-10-26T11:00:00Z")),
               Ticket.from_row(_new("redmine", "4821", "bad clock", "2036-01-01T00:00:00Z"))]
    fresh, mark, _ = new_or_updated(tickets, parse_time("2026-10-26T10:00:00Z"), now=now)
    assert [t.key for t in fresh] == ["4820"] and mark == parse_time("2026-10-26T11:00:00Z")
    later = parse_time("2036-01-01T00:30:00Z")
    fresh, mark, _ = new_or_updated(tickets, mark, now=later)
    assert [t.key for t in fresh] == ["4821"]


def test_an_email_shaped_assignee_is_proposed_as_is_but_never_written_into_text():
    """Second review: the value the tool sets stays raw, but the draft, the summary and the note posted on the
    ticket show no address (the module promise in archi.triage.tickets)."""
    g = graph_with_arrivals()
    d = read_ticket_context(g, "redmine:4820", g.generation, {"doc_subtypes": ["servicenow_kb_article"]})
    context = json.loads(json.dumps(d["context"]).replace("Grace Hopper", "grace@example.org"))
    config = {"systems": {"redmine": {"url": "https://redmine.example.org", "token_env": "REDMINE_API_KEY"}}}
    req = lambda level: ProposeRequest(NAME, {"id": "s", "entity": "redmine:4820"}, context, g.generation, level)  # noqa: E731
    draft = TriageProposer(None).propose(req(1)).proposal
    assert "Usually handled by:" in draft["payload"]["body"] and "grace@example.org" not in draft["payload"]["body"]
    p = TriageProposer(config).propose(req(2)).proposal
    assert p["action_class"] == "assign_ticket" and p["payload"]["value"] == "grace@example.org"
    assert "grace@example.org" not in p["payload"]["note"] and "grace@example.org" not in p["summary"]
