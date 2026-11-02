"""JIRA comments as jira_comment records (JiraIssueSource comment_records).

Each comment becomes a record of its own, linked to its issue by
``has_comment`` and to its author by ``authored_by``, with an id made
from a hash of its content (the records cache keeps no JIRA comment id).
The issue chunks must not change: a chunk id hashes the chunk text, so a
changed chunk would re-key every JIRA chunk in a graph.
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from okg.deployment import EdgeFact, NodeFact

import archi.sources.jira as jira_mod
from archi.sources.jira import JiraIssueSource, comment_ids

REPO = Path(__file__).resolve().parents[3]
SCHEMAS = REPO / "python" / "archi" / "schemas"
BUNDLE_SCHEMAS = REPO / "bundles" / "packs" / "jira" / "schemas"

ADA = {"displayName": "Ada Lovelace"}
LINUS = {"displayName": "Linus Pauling"}

# Three comments, the second and third exact repeats of each other, plus
# one with an empty body that every path drops. The first body has
# surrounding whitespace; ids and records use the stripped body.
ISSUE = {
    "key": "CMSPROD-7",
    "summary": "Stuck transfers",
    "description": "Transfers to T2 are stuck.",
    "project": "CMSPROD",
    "status": "Open",
    "priority": "Major",
    "issue_type": "Bug",
    "assignee": "Ada Lovelace",
    "reporter": "Grace Hopper",
    "created": "2026-01-05T10:00:00.000+0000",
    "updated": "2026-01-07T10:00:00.000+0000",
    "environment": "prod",
    "parent_key": "CMSPROD-1",
    "comment_count": 4,
    "components": ["transfers"],
    "labels": ["ops"],
    "fix_versions": ["v2"],
    "issue_links": ["relates to CMSPROD-2"],
    "subtasks": ["CMSPROD-8"],
    "recent_comments": [
        {"author": ADA, "created": "2026-01-06T08:00:00.000+0000",
         "body": "  Retrying now.\n"},
        {"author": LINUS, "created": "2026-01-06T09:00:00.000+0000",
         "body": "Same here"},
        {"author": LINUS, "created": "2026-01-06T09:00:00.000+0000",
         "body": "Same here"},
        {"author": ADA, "created": "2026-01-06T10:00:00.000+0000",
         "body": "   "},
    ],
}
QUIET = {"key": "CMSPROD-1", "summary": "Parent", "project": "CMSPROD"}

# The chunk text main's adapter produces for ISSUE, written out by hand.
EXPECTED_CHUNK_TEXT = (
    "summary: Stuck transfers\n\n"
    "description: Transfers to T2 are stuck.\n\n"
    "environment: prod\n\n"
    "status: Open\n\n"
    "priority: Major\n\n"
    "parent: CMSPROD-1\n\n"
    "components: transfers\n\n"
    "labels: ops\n\n"
    "fix_versions: v2\n\n"
    "issue_links: CMSPROD-2\n\n"
    "subtasks: CMSPROD-8\n\n"
    "recent_comments: Ada Lovelace 2026-01-06T08:00:00.000+0000: Retrying now.\n"
    "Linus Pauling 2026-01-06T09:00:00.000+0000: Same here\n"
    "Linus Pauling 2026-01-06T09:00:00.000+0000: Same here"
)


def _source(tmp_path, records, **kwargs):
    (tmp_path / "data" / "jira").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "jira" / "records.json").write_text(json.dumps(records))
    (tmp_path / "data" / "jira" / "meta.json").write_text(
        json.dumps({"record_count": len(records)})
    )
    return JiraIssueSource(
        records_path="data/jira/records.json",
        meta_path="data/jira/meta.json",
        project_keys=["CMSPROD"],
        base=str(tmp_path),
        **kwargs,
    )


def _facts(source):
    return list(source.run("run-1", mode="cursor").facts)


def _nodes(facts, subtype):
    return [f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype]


def _edges(facts, edge_type):
    return [f for f in facts if isinstance(f, EdgeFact) and f.edge_type == edge_type]


def _hash_id(created, author, body):
    return "c-" + hashlib.sha256(
        f"{created}|{author}|{body}".encode("utf-8")
    ).hexdigest()[:12]


def _person_id(name):
    return "person:" + hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:16]


def test_three_comments_two_identical_give_three_ids_and_edges(tmp_path):
    facts = _facts(_source(tmp_path, [ISSUE, QUIET], comment_records=True))
    comments = _nodes(facts, "jira_comment")
    first = _hash_id("2026-01-06T08:00:00.000000Z", "Ada Lovelace", "Retrying now.")
    same = _hash_id("2026-01-06T09:00:00.000000Z", "Linus Pauling", "Same here")
    assert [c.node_id for c in comments] == [
        f"jira:CMSPROD-7#{first}",
        f"jira:CMSPROD-7#{same}",
        f"jira:CMSPROD-7#{same}-1",
    ]
    has_comment = _edges(facts, "has_comment")
    assert [(e.src, e.dst, e.attrs) for e in has_comment] == [
        ("jira:CMSPROD-7", c.node_id, {"position": i})
        for i, c in enumerate(comments)
    ]
    authored = _edges(facts, "authored_by")
    assert [(e.src, e.dst) for e in authored] == [
        (comments[0].node_id, _person_id("Ada Lovelace")),
        (comments[1].node_id, _person_id("Linus Pauling")),
        (comments[2].node_id, _person_id("Linus Pauling")),
    ]


def test_comment_attrs_and_parent_lineage(tmp_path):
    facts = _facts(_source(tmp_path, [ISSUE, QUIET], comment_records=True))
    first = _nodes(facts, "jira_comment")[0]
    cid = _hash_id("2026-01-06T08:00:00.000000Z", "Ada Lovelace", "Retrying now.")
    # The raw time stays the attribute; only the id uses the UTC form. No
    # `text`: okg's full-text index reads `text`, and the chunks already
    # hold the comment's words.
    assert first.attrs == {
        "issue_key": "CMSPROD-7",
        "project": "CMSPROD",
        "position": 0,
        "author": "Ada Lovelace",
        "created": "2026-01-06T08:00:00.000+0000",
        "body": "Retrying now.",
        "id_basis": "content",
    }
    # The issue key leads the source record id and project is the issue's,
    # so a rule on the project or the source matches the comment. The
    # record ref differs from the issue's ({"issue_key"} alone), so a rule
    # on the issue's own record does not; okg's
    # test_comment_lineage_filter.py gates turning the option on.
    assert first.source_record_id == {"issue_key": "CMSPROD-7", "comment": cid}
    for fact in _edges(facts, "has_comment") + _edges(facts, "authored_by"):
        assert fact.source_record_id["issue_key"] == "CMSPROD-7"


def test_comment_hash_is_over_the_stripped_body():
    ids = comment_ids([("A", "t", "body")])
    assert ids == [_hash_id("t", "A", "body")]
    # Changing any of the three parts changes the id.
    assert comment_ids([("B", "t", "body")]) != ids
    assert comment_ids([("A", "u", "body")]) != ids
    assert comment_ids([("A", "t", "body!")]) != ids
    # A repeat is numbered in source order; distinct comments are not.
    assert comment_ids([("A", "t", "x"), ("A", "t", "x"), ("A", "t", "x")]) == [
        _hash_id("t", "A", "x"),
        _hash_id("t", "A", "x") + "-1",
        _hash_id("t", "A", "x") + "-2",
    ]


@pytest.mark.parametrize("created", [
    "2026-01-06T10:00:00.000+0200",
    "2026-01-06T08:00:00.000+0000",
    "2026-01-06T08:00:00.000Z",
    "2026-01-06T08:00:00Z",
    "2026-01-06T09:00:00.000+01:00",
    "2026-01-06T03:00:00.000-0500",
])
def test_same_instant_in_any_zone_gives_the_same_id(created):
    # JIRA renders times in the fetching account's zone; a refetch in
    # another zone must not re-key the comment.
    assert comment_ids([("A", created, "body")]) == [
        _hash_id("2026-01-06T08:00:00.000000Z", "A", "body")
    ]
    assert jira_mod._created_utc(created) == "2026-01-06T08:00:00.000000Z"


@pytest.mark.parametrize("created", [
    "", "yesterday", "2026-01-06T08:00:00.000", "2026-13-06T08:00:00.000+0000",
    "2026-01-06T08:00:00.000+9999",
])
def test_unparseable_time_hashes_the_raw_string_and_says_so(tmp_path, created):
    assert comment_ids([("A", created, "body")]) == [_hash_id(created, "A", "body")]
    record = {"key": "CMSPROD-11", "project": "CMSPROD", "recent_comments": [
        {"author": ADA, "created": created, "body": "body"},
    ]}
    facts = _facts(_source(tmp_path, [record], comment_records=True))
    (comment,) = _nodes(facts, "jira_comment")
    assert comment.attrs["id_basis"] == "content_raw_created"
    assert comment.attrs["created"] == created


def test_ids_are_stable_across_ingests_and_an_edit_rekeys_only_that_comment(tmp_path):
    before = [c.node_id for c in _nodes(
        _facts(_source(tmp_path / "a", [ISSUE, QUIET], comment_records=True)),
        "jira_comment")]
    again = [c.node_id for c in _nodes(
        _facts(_source(tmp_path / "b", [ISSUE, QUIET], comment_records=True)),
        "jira_comment")]
    assert again == before
    edited = json.loads(json.dumps(ISSUE))
    edited["recent_comments"][0]["body"] = "Retrying now, second try."
    # The same instant fetched in another zone is not an edit.
    edited["recent_comments"][1]["created"] = "2026-01-06T11:00:00.000+0200"
    after = [c.node_id for c in _nodes(
        _facts(_source(tmp_path / "c", [edited, QUIET], comment_records=True)),
        "jira_comment")]
    assert after[0] != before[0]
    assert after[1:] == before[1:]


def test_chunks_are_byte_identical_with_and_without_comment_records(tmp_path):
    off = _facts(_source(tmp_path / "off", [ISSUE, QUIET]))
    on = _facts(_source(tmp_path / "on", [ISSUE, QUIET], comment_records=True))

    def chunk_facts(facts):
        return [
            (f.node_id, f.attrs, f.source_record_id)
            for f in facts
            if (isinstance(f, NodeFact) and f.subtype == "document_chunk")
        ] + [
            (f.src, f.dst, f.edge_type, f.attrs)
            for f in facts
            if isinstance(f, EdgeFact) and f.edge_type in ("contains", "references")
        ]

    assert chunk_facts(on) == chunk_facts(off)
    chunk = [c for c in _nodes(on, "document_chunk")
             if c.source_record_id["issue_key"] == "CMSPROD-7"]
    assert [c.attrs["text"] for c in chunk] == [EXPECTED_CHUNK_TEXT]
    assert chunk[0].node_id == "chunk:" + hashlib.sha256(
        f"jira:CMSPROD-7\\0{0}\\0{EXPECTED_CHUNK_TEXT}".encode("utf-8")
    ).hexdigest()[:16]


def test_comment_count_is_the_number_of_comment_records(tmp_path):
    facts = _facts(_source(tmp_path, [ISSUE, QUIET], comment_records=True))
    issues = {n.node_id: n.attrs for n in _nodes(facts, "jira_issue")}
    # The cache says 4; one has an empty body and is not a record.
    assert issues["jira:CMSPROD-7"]["comment_count"] == 3
    # An issue with no comments says so, so it is read from its records.
    assert issues["jira:CMSPROD-1"]["comment_count"] == 0


def test_comment_records_are_off_unless_asked_for(tmp_path):
    source = _source(tmp_path, [ISSUE, QUIET])
    assert source.comment_records is False
    facts = _facts(source)
    assert _nodes(facts, "jira_comment") == []
    assert _edges(facts, "has_comment") == []
    assert _edges(facts, "authored_by") == []
    assert all("comment_count" not in n.attrs for n in _nodes(facts, "jira_issue"))
    # Comment authors who are not assignee or reporter get no person node.
    assert _person_id("Linus Pauling") not in {p.node_id for p in _nodes(facts, "person")}


def test_option_changes_the_probe_token_only_when_on(tmp_path):
    def token(**kwargs):
        return _source(tmp_path, [ISSUE], **kwargs).change_probe.build_token()

    assert token() == token(comment_records=False)
    assert token(comment_records=True) != token()


def test_each_person_is_emitted_once_and_before_its_first_edge(tmp_path):
    facts = _facts(_source(tmp_path, [ISSUE, QUIET], comment_records=True))
    people = [f.node_id for f in _nodes(facts, "person")]
    assert len(people) == len(set(people))
    # Ada (assignee) and Linus (comment author only) both have a node.
    assert _person_id("Ada Lovelace") in people
    assert _person_id("Linus Pauling") in people
    seen = set()
    for fact in facts:
        if isinstance(fact, NodeFact) and fact.subtype == "person":
            seen.add(fact.node_id)
        if isinstance(fact, EdgeFact) and fact.edge_type == "authored_by":
            assert fact.dst in seen


def test_bare_string_comment_has_no_author_edge(tmp_path):
    record = {"key": "CMSPROD-9", "project": "CMSPROD",
              "recent_comments": ["just text"]}
    facts = _facts(_source(tmp_path, [record], comment_records=True))
    comments = _nodes(facts, "jira_comment")
    assert [c.attrs["body"] for c in comments] == ["just text"]
    assert comments[0].node_id == "jira:CMSPROD-9#" + _hash_id("", "", "just text")
    assert _edges(facts, "authored_by") == []


def test_rest_api_comment_shape(tmp_path):
    record = {"key": "CMSPROD-10", "fields": {
        "project": {"key": "CMSPROD"},
        "comment": {"comments": [
            {"author": ADA, "created": "2026-02-01T00:00:00.000+0000",
             "body": "From the API"},
        ]},
    }}
    facts = _facts(_source(tmp_path, [record], comment_records=True))
    (comment,) = _nodes(facts, "jira_comment")
    assert comment.attrs["body"] == "From the API"
    assert comment.attrs["project"] == "CMSPROD"


def test_issue_node_carries_the_formerly_chunk_only_fields(tmp_path):
    facts = _facts(_source(tmp_path, [ISSUE, QUIET]))
    attrs = {n.node_id: n.attrs for n in _nodes(facts, "jira_issue")}["jira:CMSPROD-7"]
    assert attrs["environment"] == "prod"
    assert attrs["fix_versions"] == ["v2"]
    assert attrs["parent_key"] == "CMSPROD-1"
    assert attrs["issue_links"] == ["CMSPROD-2"]
    assert attrs["subtasks"] == ["CMSPROD-8"]


# --- schema ------------------------------------------------------------------

def _classes(schemas: Path):
    return yaml.safe_load((schemas / "sources.yaml").read_text())["classes"]


def test_schema_declares_every_emitted_attribute(tmp_path):
    facts = _facts(_source(tmp_path, [ISSUE, QUIET], comment_records=True))
    classes = _classes(SCHEMAS)
    comment_attrs = set(classes["JiraComment"]["attributes"])
    for node in _nodes(facts, "jira_comment"):
        assert set(node.attrs) <= comment_attrs
    issue_attrs = set(classes["JiraIssue"]["attributes"])
    # `label` predates this change and is a substrate-level field.
    for node in _nodes(facts, "jira_issue"):
        assert set(node.attrs) - {"label"} <= issue_attrs


def test_record_parts_names_declared_attributes():
    classes = _classes(SCHEMAS)
    parts = classes["JiraIssue"]["annotations"]["okg_record_parts"]
    issue = set(classes["JiraIssue"]["attributes"])
    comment = set(classes["JiraComment"]["attributes"])
    assert parts["edge_type"] == "has_comment"
    assert parts["child_subtype"] == "JiraComment"
    assert {parts["text_attr"], *parts["order_by"], *parts["label_attrs"]} <= comment
    assert {*parts["header_attrs"], parts["body_attr"], parts["count_attr"]} <= issue


def test_comments_stay_out_of_the_full_text_index(tmp_path):
    # okg's full-text index reads `text` and the declared searchable
    # fields; the issue chunks are the one search unit for comment text.
    comment = _classes(SCHEMAS)["JiraComment"]
    assert "okg_searchable_text_fields" not in comment["annotations"]
    assert "text" not in comment["attributes"]
    for field in ("issue_key", "author", "project", "created"):
        assert comment["attributes"][field]["annotations"]["filterable"] == "true"
    facts = _facts(_source(tmp_path, [ISSUE], comment_records=True))
    assert all("text" not in c.attrs for c in _nodes(facts, "jira_comment"))


def test_probe_fingerprint_covers_the_comment_id_code(tmp_path, monkeypatch):
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(jira_mod, "cache_or_forced_live_change_probe", capture)
    _source(tmp_path, [ISSUE])
    targets = seen["emit_targets"]
    for function in (jira_mod.comment_ids, jira_mod._created_utc,
                     jira_mod._comment_facts, jira_mod._comment_parts):
        assert function in targets


def test_bundle_copies_match_the_package_schema():
    for name in ("sources.yaml", "bridges/sources.yaml"):
        assert (BUNDLE_SCHEMAS / name).read_bytes() == (SCHEMAS / name).read_bytes()


def _compose(schemas: Path):
    pytest.importorskip("okg")
    from okg.substrate.catalog.modules_compose import compose_catalog

    tmp = Path(tempfile.mkdtemp())
    try:
        dep = tmp / "probe"
        (dep / "schemas" / "bridges").mkdir(parents=True)
        for name in ("operations.yaml", "sources.yaml"):
            shutil.copy(schemas / name, dep / "schemas" / name)
            shutil.copy(schemas / "bridges" / name, dep / "schemas" / "bridges" / name)
        (dep / "deployment.yaml").write_text(yaml.safe_dump({
            "name": "probe",
            "postgres": {"dsn": "postgresql://unused/probe"},
            "schema_dir": "schemas",
            "modules": ["document_starter", "person", "extraction"],
        }))
        return compose_catalog(dep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("schemas", [SCHEMAS, BUNDLE_SCHEMAS], ids=["package", "bundle"])
def test_okg_composes_the_comment_subtype_and_its_edges(schemas):
    composed = _compose(schemas)
    assert "jira_comment" in composed.subtypes
    triples = composed.narrowing_triples()
    assert ("jira_issue", "has_comment", "jira_comment") in triples
    assert ("jira_comment", "authored_by", "person") in triples


def test_okg_reads_the_record_parts_declaration_when_it_knows_it():
    import okg.substrate.catalog.modules_compose as compose

    if not hasattr(compose, "RECORD_PARTS_KEYS"):
        pytest.skip("this okg has no okg_record_parts support; it ignores the annotation")
    declared = _compose(SCHEMAS).subtype_metadata["jira_issue"]["record_parts"]
    assert declared["edge_type"] == "has_comment"
    assert declared["child_subtype"] == "jira_comment"
    assert declared["count_attr"] == "comment_count"
    assert declared["order_by"] == ["created", "position"]
