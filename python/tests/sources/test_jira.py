"""req.w2.sources-parity — JiraIssueSource emission/probe/JQL, offline."""
import hashlib
import json
import time

import pytest

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


def _write_caches(tmp_path, records=None, with_targets=False):
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
