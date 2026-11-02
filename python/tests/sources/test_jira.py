"""req.w2.sources-parity — JiraIssueSource emission/probe/JQL, offline."""
import hashlib
import json
import time

import pytest

import archi.sources.jira as jira_mod

from okg.deployment import EdgeFact, NodeFact
from okg.deployment import MutableApiProbe

from archi.sources.jira import (
    JiraIssueSource,
    format_jira_datetime,
    issue_key_pattern,
    issues_jql,
    parse_jira_project_keys,
    quote_jql_string,
)

PROJECT_KEYS = ["CMSPROD", "CMSCOMPPR"]

# One flat record (the shape the cms fetch script cached) and one JIRA
# REST API shape ({"key": ..., "fields": {...}}); the parser accepts both.
FLAT_ISSUE = {
    "key": "CMSPROD-101",
    "summary": "Transfer failures at T2_US_MIT",
    "description": "Stuck transfers, see CMSPROD-100 and CMSSW_14_0_2.",
    "project": "CMSPROD",
    "status": "Open",
    "priority": "Critical",
    "issue_type": "Bug",
    "assignee": "Ada Lovelace",
    "reporter": "Grace Hopper",
    "created": "2026-01-05T10:00:00.000+0000",
    "updated": "2026-02-01T09:30:00.000+0000",
    "environment": "prod",
    "resolution": "",
    "parent_key": "",
    "comment_count": 1,
    "components": ["transfers"],
    "labels": ["ops"],
    "fix_versions": [],
    "issue_links": ["relates to CMSPROD-100"],
    "subtasks": [],
    "recent_comments": [
        {
            "author": {"displayName": "Ada Lovelace"},
            "created": "2026-01-06T08:00:00.000+0000",
            "body": "Retrying via cmsweb.cern.ch",
        }
    ],
}
API_ISSUE = {
    "key": "CMSPROD-100",
    "fields": {
        "summary": "Baseline transfer issue",
        "description": "Original report.",
        "project": {"key": "CMSPROD", "name": "CMS Production"},
        "status": {"name": "Closed"},
        "priority": {"name": "Major"},
        "issuetype": {"name": "Task"},
        "assignee": {"displayName": "Ada Lovelace"},
        "reporter": {"displayName": "Enrico Fermi"},
        "created": "2025-12-01T10:00:00.000+0000",
        "updated": "2025-12-20T10:00:00.000+0000",
        "subtasks": [{"key": "CMSPROD-102"}, {"key": "OTHER-9"}],
        "comment": {"comments": []},
    },
}


def _write_caches(tmp_path, records=None, with_targets=False, **source_kwargs):
    (tmp_path / "data" / "jira").mkdir(parents=True, exist_ok=True)
    if records is not None:
        (tmp_path / "data" / "jira" / "records.json").write_text(
            json.dumps(records)
        )
    # A clean fixture carries a record_count that matches the records
    # cache; run() cross-checks the two and treats a mismatch as a
    # truncated cache (see the mismatch regression tests below). The
    # cache-missing fixtures keep the original 72000 that the preflight
    # tests report on.
    meta_count = 72000 if records is None else len(records)
    (tmp_path / "data" / "jira" / "meta.json").write_text(
        json.dumps({"record_count": meta_count})
    )
    kwargs = {}
    if with_targets:
        (tmp_path / "data" / "cric").mkdir(parents=True)
        (tmp_path / "data" / "cric" / "sites.json").write_text(
            json.dumps({"T2_US_MIT": {}})
        )
        (tmp_path / "data" / "releases").mkdir(parents=True)
        (tmp_path / "data" / "releases" / "records.json").write_text(
            json.dumps([{"label": "CMSSW_14_0_2"}])
        )
        (tmp_path / "data" / "services").mkdir(parents=True)
        (tmp_path / "data" / "services" / "services.json").write_text(
            json.dumps(
                {"reqmgr2": {"endpoint": "https://cmsweb.cern.ch/reqmgr2"}}
            )
        )
        kwargs = {
            "sites_path": "data/cric/sites.json",
            "releases_path": "data/releases/records.json",
            "services_path": "data/services/services.json",
        }
    return JiraIssueSource(
        records_path="data/jira/records.json",
        meta_path="data/jira/meta.json",
        project_keys=PROJECT_KEYS,
        base=str(tmp_path),
        **kwargs,
        **source_kwargs,
    )


def _run_facts(source, mode="cursor"):
    run = source.run("run-1", mode=mode)
    return run, list(run.facts)


def _nodes(facts, subtype):
    return [f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype]


def _edges(facts, edge_type):
    return [f for f in facts if isinstance(f, EdgeFact) and f.edge_type == edge_type]


# --- emission ---------------------------------------------------------------

def test_issue_nodes_ids_subtypes_and_attrs(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE, API_ISSUE])
    _, facts = _run_facts(source)
    issues = {n.node_id: n for n in _nodes(facts, "jira_issue")}
    assert set(issues) == {"jira:CMSPROD-101", "jira:CMSPROD-100"}
    node = issues["jira:CMSPROD-101"]
    assert node.attrs["issue_key"] == "CMSPROD-101"
    assert node.attrs["label"] == "CMSPROD-101"
    assert node.attrs["status"] == "Open"
    assert node.attrs["url"] == "https://its.cern.ch/jira/browse/CMSPROD-101"
    assert node.source_record_id == {"issue_key": "CMSPROD-101"}
    assert node.source_revision["run_id"] == "run-1"
    # API-shaped record: nested fields flattened via name/displayName.
    api = issues["jira:CMSPROD-100"]
    assert api.attrs["project"] == "CMS Production"
    assert api.attrs["status"] == "Closed"
    assert api.attrs["assignee"] == "Ada Lovelace"


def test_person_nodes_and_role_edges(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE, API_ISSUE])
    _, facts = _run_facts(source)
    people = _nodes(facts, "person")
    # Ada appears twice (assignee of both) but is emitted once.
    assert len(people) == 3
    expected_id = (
        "person:"
        + hashlib.sha256("ada lovelace".encode("utf-8")).hexdigest()[:16]
    )
    assert expected_id in {p.node_id for p in people}
    assigned = _edges(facts, "assigned_to")
    reported = _edges(facts, "reported_by")
    assert {e.src for e in assigned} == {"jira:CMSPROD-101", "jira:CMSPROD-100"}
    assert all(e.dst == expected_id for e in assigned)
    assert {e.source_record_id["role"] for e in reported} == {"reporter"}


def test_chunk_ids_use_parity_separator(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE])
    _, facts = _run_facts(source)
    chunks = _nodes(facts, "document_chunk")
    assert len(chunks) == 1
    chunk = chunks[0]
    text = chunk.attrs["text"]
    # The cms original hashes with a literal backslash-zero separator.
    expected = "chunk:" + hashlib.sha256(
        f"jira:CMSPROD-101\\0{0}\\0{text}".encode("utf-8")
    ).hexdigest()[:16]
    assert chunk.node_id == expected
    assert chunk.attrs["chunker_name"] == "cms_jira_window_v1"
    assert chunk.attrs["heading_path"] == "CMSPROD-101"
    contains = _edges(facts, "contains")
    assert [(e.src, e.dst) for e in contains] == [
        ("jira:CMSPROD-101", chunk.node_id)
    ]
    assert contains[0].attrs == {"chunk_index": 0}


def test_chunk_text_carries_link_surface_and_extracted_keys(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE, API_ISSUE])
    _, facts = _run_facts(source)
    by_issue = {
        c.source_record_id["issue_key"]: c.attrs["text"]
        for c in _nodes(facts, "document_chunk")
    }
    assert "issue_links: CMSPROD-100" in by_issue["CMSPROD-101"]
    assert "recent_comments: Ada Lovelace" in by_issue["CMSPROD-101"]
    # Subtask extraction is restricted to the configured project keys:
    # OTHER-9 is dropped, CMSPROD-102 kept.
    assert "subtasks: CMSPROD-102" in by_issue["CMSPROD-100"]
    assert "OTHER-9" not in by_issue["CMSPROD-100"]


def test_reference_edges_from_chunks(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE, API_ISSUE], with_targets=True)
    _, facts = _run_facts(source)
    refs = _edges(facts, "references")
    by_type = {}
    for e in refs:
        by_type.setdefault(e.attrs["match_type"], set()).add((e.src, e.dst))
    chunk_101 = next(
        c.node_id
        for c in _nodes(facts, "document_chunk")
        if c.source_record_id["issue_key"] == "CMSPROD-101"
    )
    assert (chunk_101, "site:T2_US_MIT") in by_type["cms_site"]
    assert (chunk_101, "cmssw_release:CMSSW_14_0_2") in by_type["cmssw_release"]
    # Self-family references come from this source's own records cache.
    assert (chunk_101, "jira:CMSPROD-100") in by_type["jira_issue"]
    assert (chunk_101, "svc:reqmgr2") in by_type["infrastructure_service"]
    assert all(e.provenance == "derived_deterministic" for e in refs)
    assert all(e.confidence == 0.95 for e in refs)


def test_omitted_project_keys_bound_extraction_to_cache_projects(tmp_path):
    """Without project_keys, regex extraction only accepts keys whose
    project prefix exists in the records cache: free text like COVID-19
    must not become an issue key, while in-family keys still resolve."""
    records = [
        {
            "key": "ARCHI-1",
            "summary": "cluster upgrade",
            "issue_links": ["relates to ARCHI-7 during COVID-19 response"],
        },
        {"key": "ARCHI-7", "summary": "baseline"},
    ]
    (tmp_path / "data" / "jira").mkdir(parents=True)
    (tmp_path / "data" / "jira" / "records.json").write_text(
        json.dumps(records)
    )
    source = JiraIssueSource(
        records_path="data/jira/records.json",
        meta_path="data/jira/meta.json",
        base=str(tmp_path),
    )
    _, facts = _run_facts(source)
    by_issue = {
        c.source_record_id["issue_key"]: c.attrs["text"]
        for c in _nodes(facts, "document_chunk")
    }
    assert "issue_links: ARCHI-7" in by_issue["ARCHI-1"]
    assert "COVID-19" not in by_issue["ARCHI-1"]


def test_no_reference_targets_configured_means_no_reference_edges(tmp_path):
    records = [dict(FLAT_ISSUE, issue_links=[])]
    source = _write_caches(tmp_path, records)
    source = JiraIssueSource(
        records_path="data/jira/records.json",
        meta_path="data/jira/meta.json",
        project_keys=PROJECT_KEYS,
        base=str(tmp_path),
    )
    _, facts = _run_facts(source)
    refs = _edges(facts, "references")
    # jira self-targets still resolve (own cache), nothing else does.
    assert {e.attrs["match_type"] for e in refs} <= {"jira_issue"}


# --- record-set / deletion semantics ---------------------------------------

def test_completed_scope_by_mode(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE])
    assert source.run("r", mode="cursor").completed_scope is False
    assert source.run("r", mode="scope_complete").completed_scope is True
    assert source.run("r", mode="reconcile").completed_scope is True


def test_missing_cache_run_emits_nothing_and_never_completes_scope(tmp_path):
    source = _write_caches(tmp_path, records=None)
    run = source.run("r", mode="reconcile")
    assert list(run.facts) == []
    assert run.completed_scope is False  # nothing may be deleted from a gap
    assert run.health.status == "cache_missing"
    assert run.health.credential_refs == ("CERN_JIRA_TOKEN",)
    assert run.health.alias_refs == {"CERN_JIRA_TOKEN": ("JIRA_CERN_TOKEN",)}


def test_run_health_ok_with_content_hash(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE, API_ISSUE])
    run = source.run("r")
    assert run.health.status == "ok"
    assert run.health.mode == "cache"
    assert run.health.record_count == 2
    assert run.health.content_hash


def test_meta_record_count_mismatch_never_claims_scope(tmp_path):
    """Adversarial-review finding 6: a truncated-but-valid records cache
    (meta.json reports 72000 records, the file parses 2) must emit what
    parsed but report the mismatch and never claim a completed scope —
    claiming it would retract every record the truncation dropped."""
    source = _write_caches(tmp_path, [FLAT_ISSUE, API_ISSUE])
    (tmp_path / "data" / "jira" / "meta.json").write_text(
        json.dumps({"record_count": 72000})
    )
    run = source.run("r", mode="scope_complete")
    facts = list(run.facts)
    # The parsed records are still emitted...
    assert {n.node_id for n in _nodes(facts, "jira_issue")} == {
        "jira:CMSPROD-101",
        "jira:CMSPROD-100",
    }
    # ...but the run is degraded and claims nothing.
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert run.health.record_count == 2
    assert "record_count=72000" in run.health.reason
    assert "parsed 2 records" in run.health.reason


def test_meta_without_record_count_stays_clean(tmp_path):
    """A meta.json that carries no record_count cannot cross-check the
    cache and must not degrade the run."""
    source = _write_caches(tmp_path, [FLAT_ISSUE])
    (tmp_path / "data" / "jira" / "meta.json").write_text(
        json.dumps({"fetched_at": "2026-08-11T00:00:00Z"})
    )
    run = source.run("r", mode="scope_complete")
    assert list(run.facts)
    assert run.completed_scope is True
    assert run.health.status == "ok"


def test_meta_record_count_match_claims_scope(tmp_path):
    """A matching record_count is a clean run (the fixture default)."""
    source = _write_caches(tmp_path, [FLAT_ISSUE, API_ISSUE])
    run = source.run("r", mode="scope_complete")
    assert len(_nodes(list(run.facts), "jira_issue")) == 2
    assert run.completed_scope is True
    assert run.health.status == "ok"


# --- preflight --------------------------------------------------------------

def test_preflight_cache_missing_reports_expected_count(tmp_path):
    source = _write_caches(tmp_path, records=None)
    result = source.preflight()
    assert result.status == "cache_missing"
    assert result.required is True
    assert "metadata reports 72000 records" in result.reason
    assert result.credential_refs == ("CERN_JIRA_TOKEN",)


def test_preflight_corrupt_meta_reports_instead_of_raising(tmp_path):
    source = _write_caches(tmp_path, records=None)
    (tmp_path / "data" / "jira" / "meta.json").write_text(
        '{"record_count": 72'  # truncated JSON
    )
    result = source.preflight()
    assert result.status == "cache_missing"
    assert "metadata reports" not in result.reason


def test_preflight_ok(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE])
    result = source.preflight()
    assert result.status == "ok"
    assert result.mode == "cache"
    assert result.record_count == 1
    assert result.content_hash


def test_custom_credential_ref(tmp_path):
    source = JiraIssueSource(
        credential_ref="MY_JIRA_TOKEN",
        credential_aliases=[],
        base=str(tmp_path),
    )
    result = source.preflight()
    assert result.credential_refs == ("MY_JIRA_TOKEN",)
    assert result.alias_refs == {}


# --- probe declarations -----------------------------------------------------

def test_probe_declarations(tmp_path):
    source = _write_caches(tmp_path, records=None)
    assert JiraIssueSource.profile == "mutable_api"
    assert JiraIssueSource.change_probe_kind == "mutable_api"
    assert isinstance(source.change_probe, MutableApiProbe)


def test_probe_forces_live_read_without_cache(tmp_path):
    source = _write_caches(tmp_path, records=None)
    first = source.change_probe.build_token()
    time.sleep(0.002)
    assert source.change_probe.build_token() != first


def test_probe_tracks_cache_content_when_present(tmp_path):
    source = _write_caches(tmp_path, [FLAT_ISSUE])
    first = source.change_probe.build_token()
    time.sleep(0.002)
    assert source.change_probe.build_token() == first
    (tmp_path / "data" / "jira" / "records.json").write_text(
        json.dumps([FLAT_ISSUE, API_ISSUE])
    )
    assert source.change_probe.build_token() != first


# --- JQL helpers ------------------------------------------------------------

def test_parse_jira_project_keys_valid():
    assert parse_jira_project_keys([" CMSPROD ", "A1_B"], "bad") == [
        "CMSPROD",
        "A1_B",
    ]


@pytest.mark.parametrize(
    "value",
    ["CMSPROD", ["cmsprod"], [""], [], [1], ["1ABC"], ["CMS PROD"]],
)
def test_parse_jira_project_keys_invalid(value):
    with pytest.raises(ValueError, match="bad"):
        parse_jira_project_keys(value, "bad")


def test_quote_jql_string_escapes():
    assert quote_jql_string('say "hi" \\ there') == '"say \\"hi\\" \\\\ there"'
    assert quote_jql_string("plain") == '"plain"'


def test_format_jira_datetime():
    assert format_jira_datetime("2024-01-02T03:04:05") == "2024/01/02 03:04"
    with pytest.raises(ValueError):
        format_jira_datetime("not-a-date")


def test_issues_jql():
    assert issues_jql("CMSPROD") == 'project = "CMSPROD"'
    assert issues_jql(
        "CMSPROD",
        created_after="2024-01-02T03:04:05",
        updated_after="2024-02-03T04:05:06",
    ) == (
        'project = "CMSPROD" and created >= "2024/01/02 03:04" '
        'and updated >= "2024/02/03 04:05"'
    )
    with pytest.raises(ValueError):
        issues_jql("lower-case")


def test_issue_key_pattern_generic_and_restricted():
    generic = issue_key_pattern(None)
    assert generic.findall("see ABC-1 and COVID-19") == ["ABC-1", "COVID-19"]
    restricted = issue_key_pattern(["ABC"])
    assert restricted.findall("see ABC-1 and COVID-19") == ["ABC-1"]
    with pytest.raises(ValueError):
        issue_key_pattern(["not valid"])


def test_api_parent_dict_yields_key_not_repr(tmp_path):
    """JIRA REST fields.parent is a dict; parent_key must be its key, never
    the dict repr polluting chunk text (scratch-ingest finding, 2026-08-11)."""
    issue = {
        "key": "CMSPROD-200",
        "fields": {
            **API_ISSUE["fields"],
            "parent": {
                "id": "10",
                "key": "CMSPROD-9",
                "fields": {"summary": "parent issue"},
            },
        },
    }
    source = _write_caches(tmp_path, records=[issue])
    run = source.run("run-parent-1", mode="scope_complete")
    facts = list(run.facts)
    chunks = [
        f for f in facts
        if getattr(f, "subtype", "") == "document_chunk"
    ]
    assert chunks, "the single fixture issue must emit a chunk"
    text = chunks[0].attrs["text"]
    assert "{'id'" not in text
    assert "parent: CMSPROD-9" in text


# --- email redaction (parity with okg-deployments cms jira, b25e36f06c) -----

# One address per text field the source reads, each with a distinct local
# part so a leak names the field it came from. Mixed forms: plain, +tag,
# quoted local part, HTML-encoded @, double-encoded @, and URL %40.
EMAIL_ISSUE = {
    "key": "CMSPROD-300",
    "summary": "Contact summary.owner+ops@cern.ch about run 381000",
    "description": 'Ask "descr author"@cern.ch or descrtwo&#64;cern.ch.',
    "project": "CMSPROD",
    "status": "Waiting for statusperson@cern.ch",
    "priority": "Major",
    "issue_type": "Bug",
    "assignee": {"displayName": "Ada Lovelace <assigneeaddr@cern.ch>"},
    "reporter": "reporteraddr@cern.ch",
    "created": "2026-01-05T10:00:00.000+0000",
    "updated": "2026-02-01T09:30:00.000+0000",
    "environment": "prod, owner envaddr&amp;#64;cern.ch",
    "components": ["transfers componentaddr@cern.ch"],
    "labels": ["labeladdr@cern.ch"],
    "fix_versions": ["v1 fixaddr@cern.ch"],
    "recent_comments": [
        {
            "author": {"displayName": "commentauthor@cern.ch"},
            "created": "2026-01-06T08:00:00.000+0000",
            "body": " ".join([
                "Retrying via cmsweb.cern.ch; see",
                "https://its.cern.ch/jira/secure/ViewProfile?mail=commentbody%40cern.ch",
                "for the owner",
            ]),
        }
    ],
}
EMAIL_LOCAL_PARTS = (
    "summary.owner",
    "descr author",
    "descrtwo",
    "statusperson",
    "assigneeaddr",
    "reporteraddr",
    "envaddr",
    "componentaddr",
    "labeladdr",
    "fixaddr",
    "commentauthor",
    "commentbody",
)


def test_email_addresses_removed_from_every_emitted_field(tmp_path):
    source = _write_caches(tmp_path, [EMAIL_ISSUE])
    run, facts = _run_facts(source, mode="scope_complete")
    emitted = json.dumps(
        [
            {
                "id": getattr(f, "node_id", None),
                "attrs": f.attrs,
                "record": f.source_record_id,
            }
            for f in facts
        ],
        sort_keys=True,
    ).casefold()
    leaked = [part for part in EMAIL_LOCAL_PARTS if part in emitted]
    assert leaked == []
    assert "@" not in emitted
    assert "%40" not in emitted
    assert "&#64;" not in emitted

    # Only the addresses go; the text around them survives, as in cms.
    (issue,) = _nodes(facts, "jira_issue")
    assert issue.attrs["summary"] == "Contact  about run 381000"
    assert issue.attrs["assignee"] == "Ada Lovelace <>"
    assert issue.attrs["reporter"] == ""
    chunk_text = "\n".join(
        c.attrs["text"] for c in _nodes(facts, "document_chunk")
    )
    assert "Retrying via cmsweb.cern.ch" in chunk_text
    # "=" is a local-part character, so the query key goes too.
    assert "ViewProfile? for the owner" in chunk_text
    assert "environment: prod, owner" in chunk_text
    assert run.health.status == "ok"


def test_email_redaction_covers_rest_api_shaped_comments(tmp_path):
    issue = {
        "key": "CMSPROD-301",
        "fields": {
            **API_ISSUE["fields"],
            "summary": "Mail apiowner@cern.ch",
            "comment": {
                "comments": [
                    {
                        "author": {"displayName": "Enrico Fermi"},
                        "body": "cc apicomment@cern.ch please",
                    }
                ]
            },
        },
    }
    source = _write_caches(tmp_path, [issue])
    _, facts = _run_facts(source)
    emitted = json.dumps([f.attrs for f in facts], sort_keys=True)
    assert "apiowner" not in emitted
    assert "apicomment" not in emitted
    assert "cc  please" in emitted


def test_text_without_an_address_is_emitted_verbatim(tmp_path):
    # Redaction decodes nothing: an issue with HTML entities and no
    # address keeps them, so its chunk text and chunk id do not change.
    issue = {
        **FLAT_ISSUE,
        "key": "CMSPROD-302",
        "summary": "R&amp;D link to AT&amp;T &#64; noon",
        "description": "Ticket &commat;ops, 100% &amp;amp; done",
    }
    source = _write_caches(tmp_path, [issue])
    _, facts = _run_facts(source)
    (node,) = _nodes(facts, "jira_issue")
    assert node.attrs["summary"] == "R&amp;D link to AT&amp;T &#64; noon"
    (chunk,) = _nodes(facts, "document_chunk")
    assert "R&amp;D link to AT&amp;T &#64; noon" in chunk.attrs["text"]
    assert "Ticket &commat;ops, 100% &amp;amp; done" in chunk.attrs["text"]


def test_non_ascii_and_padded_addresses_leave_no_fragment(tmp_path):
    issue = {
        **FLAT_ISSUE,
        "key": "CMSPROD-303",
        "summary": "Ask j\u00fcrgen.k\u00f6nig&#064;cern.ch or fwidthaddr\uff20cern.ch",
        "assignee": "\u00fcber.m\u00fcller@cern.ch",
        "reporter": "Grace Hopper",
        "recent_comments": [],
    }
    source = _write_caches(tmp_path, [issue])
    _, facts = _run_facts(source)
    emitted = json.dumps(
        [f.attrs for f in facts], sort_keys=True, ensure_ascii=False
    )
    for fragment in ("j\u00fcrgen", "k\u00f6nig", "\u00fcber", "m\u00fc", "fwidthaddr"):
        assert fragment not in emitted
    (node,) = _nodes(facts, "jira_issue")
    assert node.attrs["summary"] == "Ask  or "
    # An assignee that is only an address yields no person, as in cms.
    assert [p.attrs["display_name"] for p in _nodes(facts, "person")] == [
        "Grace Hopper"
    ]


def test_adjacent_addresses_in_table_markup_are_both_removed(tmp_path):
    issue = {
        **FLAT_ISSUE,
        "key": "CMSPROD-304",
        "description": "||owner||backup||\n|tablebob@cern.ch|tablealice@fnal.gov|",
        "recent_comments": [],
    }
    source = _write_caches(tmp_path, [issue])
    _, facts = _run_facts(source)
    emitted = json.dumps([f.attrs for f in facts], sort_keys=True)
    assert "tablebob" not in emitted
    assert "tablealice" not in emitted
    (node,) = _nodes(facts, "jira_issue")
    assert node.attrs["description"] == "||owner||backup||\n|"


def test_git_account_kept_and_next_address_still_removed(tmp_path):
    # Operator decision (2026-09-28), changing #5's behavior for JIRA too:
    # the literal git@ account of a git host is kept, so clone
    # instructions survive; an address right after it is still removed,
    # and ssh user@host logins still go.
    issue = {
        **FLAT_ISSUE,
        "key": "CMSPROD-305",
        "summary": "Clone git@gitlab.cern.ch:cms/y.git",
        "description": "\n".join([
            "git clone git@github.com:x/y.git by gitnextaddr@cern.ch",
            "then ssh sshloginaddr@lxplus.cern.ch",
        ]),
        "recent_comments": [],
    }
    source = _write_caches(tmp_path, [issue])
    _, facts = _run_facts(source)
    emitted = json.dumps([f.attrs for f in facts], sort_keys=True)
    assert "gitnextaddr" not in emitted
    assert "sshloginaddr" not in emitted
    (node,) = _nodes(facts, "jira_issue")
    assert node.attrs["summary"] == "Clone git@gitlab.cern.ch:cms/y.git"
    assert node.attrs["description"] == (
        "git clone git@github.com:x/y.git by \nthen ssh "
    )
    chunk_text = "\n".join(
        c.attrs["text"] for c in _nodes(facts, "document_chunk")
    )
    assert "git clone git@github.com:x/y.git by \n" in chunk_text
    assert "Clone git@gitlab.cern.ch:cms/y.git" in chunk_text


# --- issue links as edges ---------------------------------------------------

def _link(key, link_type, direction):
    return {"key": key, "type": link_type, "direction": direction}


def _linked(facts):
    return sorted(
        (e.src, e.dst, e.attrs["link_type"]) for e in _edges(facts, "linked_to")
    )


def test_issue_links_become_one_edge_per_pair_from_the_outward_side(tmp_path):
    # The downloader caches every link on both issues: outward on one,
    # inward on the other. Each linked pair becomes one edge, from the
    # outward side, whichever record is read first.
    records = [
        {"key": "CMSPROD-1", "issue_links": [
            _link("CMSPROD-2", "Child-Issue", "outward"),
            _link("CMSPROD-3", "Blocks", "inward"),
        ]},
        {"key": "CMSPROD-2", "issue_links": [
            _link("CMSPROD-1", "Child-Issue", "inward"),
        ]},
        {"key": "CMSPROD-3", "issue_links": [
            _link("CMSPROD-1", "Blocks", "outward"),
            # Same pair, another type: the graph holds one edge per
            # ordered pair, so both names ride on it.
            _link("CMSPROD-1", "Relates", "outward"),
        ]},
    ]
    source = _write_caches(tmp_path, records, issue_link_edges=True)
    _, facts = _run_facts(source)
    assert _linked(facts) == [
        ("jira:CMSPROD-1", "jira:CMSPROD-2", "Child-Issue"),
        ("jira:CMSPROD-3", "jira:CMSPROD-1", "Blocks; Relates"),
    ]
    both = next(
        e for e in _edges(facts, "linked_to") if e.src == "jira:CMSPROD-3"
    )
    assert both.attrs["link_kind"] == "blocks"
    child = next(
        e for e in _edges(facts, "linked_to") if e.attrs["link_type"] == "Child-Issue"
    )
    assert child.source_record_id == {
        "issue_key": "CMSPROD-1",
        "linked_issue_key": "CMSPROD-2",
    }
    assert child.source_revision["run_id"] == "run-1"


def test_links_to_issues_outside_the_cache_are_dropped(tmp_path):
    records = [
        {"key": "CMSPROD-1", "issue_links": [
            _link("CMSPROD-99", "Relates", "outward"),  # not cached
            _link("CMSPROD-1", "Relates", "outward"),   # self link
            _link("OTHER-5", "Relates", "outward"),     # not a project key
            {"key": "CMSPROD-2", "type": "Relates"},    # no direction
            "relates to CMSPROD-2",                     # legacy text shape
        ]},
        {"key": "CMSPROD-2"},
    ]
    source = _write_caches(tmp_path, records, issue_link_edges=True)
    _, facts = _run_facts(source)
    assert _edges(facts, "linked_to") == []
    # The legacy keys still reach the chunk text.
    text = next(
        c.attrs["text"]
        for c in _nodes(facts, "document_chunk")
        if c.source_record_id["issue_key"] == "CMSPROD-1"
    )
    assert "issue_links: CMSPROD-1 CMSPROD-2 CMSPROD-99" in text


def test_rest_api_issuelinks_shape(tmp_path):
    records = [
        {"key": "CMSPROD-10", "fields": {
            "summary": "parent",
            "issuelinks": [
                {"type": {"name": "Duplicate", "inward": "is duplicated by",
                          "outward": "duplicates"},
                 "outwardIssue": {"key": "CMSPROD-11"}},
                {"type": {"name": "Blocks"},
                 "inwardIssue": {"key": "CMSPROD-12", "fields": {}}},
            ],
        }},
        {"key": "CMSPROD-11"},
        {"key": "CMSPROD-12"},
    ]
    source = _write_caches(tmp_path, records, issue_link_edges=True)
    _, facts = _run_facts(source)
    assert _linked(facts) == [
        ("jira:CMSPROD-10", "jira:CMSPROD-11", "Duplicate"),
        ("jira:CMSPROD-12", "jira:CMSPROD-10", "Blocks"),
    ]


def test_subtasks_link_parent_to_child_once(tmp_path):
    # A sub-task names its parent and the parent lists its sub-tasks:
    # one Sub-task edge per pair, from the parent.
    records = [
        {"key": "CMSPROD-20", "subtasks": ["CMSPROD-21", "CMSPROD-22"]},
        {"key": "CMSPROD-21", "parent_key": "CMSPROD-20"},
        {"key": "CMSPROD-22"},
        {"key": "CMSPROD-23", "parent_key": "CMSPROD-77"},  # parent not cached
    ]
    source = _write_caches(tmp_path, records, issue_link_edges=True)
    _, facts = _run_facts(source)
    assert _linked(facts) == [
        ("jira:CMSPROD-20", "jira:CMSPROD-21", "Sub-task"),
        ("jira:CMSPROD-20", "jira:CMSPROD-22", "Sub-task"),
    ]


def test_issue_node_carries_resolution_labels_and_components(tmp_path):
    records = [
        dict(FLAT_ISSUE, resolution="Fixed", labels=["Late", "8001"]),
        API_ISSUE,
    ]
    source = _write_caches(tmp_path, records, issue_link_edges=True)
    _, facts = _run_facts(source)
    issues = {n.node_id: n for n in _nodes(facts, "jira_issue")}
    flat = issues["jira:CMSPROD-101"].attrs
    assert flat["resolution"] == "Fixed"
    assert flat["labels"] == ["Late", "8001"]
    assert flat["components"] == ["transfers"]
    api = issues["jira:CMSPROD-100"].attrs
    assert (api["resolution"], api["labels"], api["components"]) == ("", [], [])


def test_a_subtask_that_is_also_linked_is_one_child_edge(tmp_path):
    # Sub-task parent and a Relates link on the same ordered pair: one
    # edge, the more specific kind, both names kept. The reverse
    # direction is its own pair and its own edge.
    records = [
        {"key": "CMSPROD-30", "subtasks": ["CMSPROD-31"], "issue_links": [
            _link("CMSPROD-31", "Relates", "outward"),
            _link("CMSPROD-31", "Blocks", "inward"),
        ]},
        {"key": "CMSPROD-31", "parent_key": "CMSPROD-30"},
    ]
    source = _write_caches(tmp_path, records, issue_link_edges=True)
    _, facts = _run_facts(source)
    assert sorted(
        (e.src, e.dst, e.attrs["link_kind"], e.attrs["link_type"])
        for e in _edges(facts, "linked_to")
    ) == [
        ("jira:CMSPROD-30", "jira:CMSPROD-31", "child", "Relates; Sub-task"),
        ("jira:CMSPROD-31", "jira:CMSPROD-30", "blocks", "Blocks"),
    ]


def test_issue_link_option_changes_the_probe_token_only_when_on(tmp_path):
    def token(**kwargs):
        return _write_caches(
            tmp_path, [FLAT_ISSUE], **kwargs
        ).change_probe.build_token()

    assert token() == token(issue_link_edges=False)
    assert token(issue_link_edges=True) != token()


def test_issue_link_edges_are_off_unless_asked_for(tmp_path):
    # The edge is a new output a strict output_signature must list, so
    # a deployment that does not set the option emits none.
    records = [
        {"key": "CMSPROD-1", "issue_links": [
            _link("CMSPROD-2", "Blocks", "outward"),
        ], "subtasks": ["CMSPROD-2"]},
        {"key": "CMSPROD-2", "parent_key": "CMSPROD-1"},
    ]
    source = _write_caches(tmp_path, records)
    assert source.issue_link_edges is False
    _, facts = _run_facts(source)
    assert _edges(facts, "linked_to") == []


@pytest.mark.parametrize("raw, kind", [
    ("Child-Issue", "child"),
    ("parent", "child"),
    ("Sub-task", "child"),
    ("Blocks", "blocks"),
    ("Cloners", "clones"),
    ("Duplicate", "duplicates"),
    ("Duplibcate", "duplicates"),
    ("Finish-to-Start link (WBSGantt)", "gantt"),
    ("Hierarchy link (WBSGantt)", "gantt"),
    ("multi-level hierarchy [GANTT]", "gantt"),
    ("Relates", "relates"),
    ("Relation", "relates"),
    ("Causal", "relates"),
    ("Dependency", "relates"),
    ("some type nobody has seen", "relates"),
])
def test_link_kind_collapses_the_raw_type_and_keeps_it(tmp_path, raw, kind):
    assert jira_mod.link_kind(raw) == kind
    assert kind in jira_mod.LINK_KINDS
    records = [
        {"key": "CMSPROD-1", "issue_links": [_link("CMSPROD-2", raw, "outward")]},
        {"key": "CMSPROD-2"},
    ]
    source = _write_caches(tmp_path, records, issue_link_edges=True)
    _, facts = _run_facts(source)
    (edge,) = [e for e in _edges(facts, "linked_to")
               if e.attrs["link_type"] == raw]
    assert edge.attrs == {"link_kind": kind, "link_type": raw}
