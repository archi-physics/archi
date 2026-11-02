"""task.w2.sources-twiki — parser core + TwikiEOSSource + TwikiCrawlSource,
offline and fixture-driven."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import pytest
import requests

from okg.deployment import EdgeFact, NodeFact
from okg.deployment import ContentHashProbe
from okg.deployment import MutableApiProbe

import archi.sources.twiki as twiki_mod
from archi.sources._twiki_parse import (
    canonical_twiki_url,
    extract_bare_wikiwords,
    extract_wiki_links,
    is_real_page,
    parse_meta,
    strip_twiki,
    topic_page_id,
    twiki_node_id,
    twiki_page_id_from_url,
)
from archi.sources.twiki import TwikiCrawlSource, TwikiEOSSource

_FAR_FUTURE = int(datetime(2035, 1, 1, tzinfo=timezone.utc).timestamp())
_PAST = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())

# --- parser core -------------------------------------------------------------

META_FIXTURE = (
    '%META:TOPICINFO{author="jdoe" date="1700000000" format="1.1" '
    'version="12"}%\n'
    '%META:TOPICPARENT{name="ParentTopic"}%\n'
    "---+!! Twiki Ops Guide\n"
    "---++ Transfers\n"
    "Use =xrdcp= for *fast* copies at T2_US_MIT.\n"
    '%TOC{title="Contents"}%\n'
    "%STARTINCLUDE%\n"
    "See [[CMS.PhedexDocs][the PhEDEx docs]] and [[WebHome]].\n"
    "<verbatim>\nraw block with CamelCase\n</verbatim>\n"
    "| *Site* | *Status* |\n"
    "| T2_US_MIT | ok |\n"
    '<img src="https://twiki.cern.ch/pub/x.png" alt="plot">\n'
    "Visit https://cern.ch/FooBar and DataOps daily.\n"
)
# Byte-for-byte output of the cms original's _strip_twiki on the fixture
# (verified against okg-deployments cms/cms_sources/twiki_eos.py@f33a9c4).
# The deliberate parser deviations (single-line =code= unwrap, single-line
# heading titles — see the _twiki_parse module docstring) do not affect
# this fixture, so the expectation is unchanged.
META_FIXTURE_CMS_STRIP = (
    "! Twiki Ops Guide Transfers Use xrdcp for fast copies at T2_US_MIT. "
    "title= Contents See the PhEDEx docs and WebHome. raw block with "
    "CamelCase | Site | Status | | T2_US_MIT | ok | Visit "
    "https://cern.ch/FooBar and DataOps daily."
)


def test_strip_twiki_cms_defaults_byte_parity():
    assert strip_twiki(META_FIXTURE) == META_FIXTURE_CMS_STRIP


def test_strip_twiki_keeps_web_link_urls_only_when_asked():
    raw = (
        "Tracked in [[https://its.cern.ch/jira/browse/CMSCOMPPR-2361][the "
        "ticket]] and [[https://its.cern.ch/jira/browse/CMSPROD-7]]; see "
        "[[CMS.PhedexDocs][the docs]], [[mailto:someone@cern.ch][Jane]] "
        "and [[https://cern.ch/x][https://cern.ch/x]].\n"
    )
    # Default: the label alone, as before the option existed.
    assert strip_twiki(raw) == (
        "Tracked in the ticket and https://its.cern.ch/jira/browse/CMSPROD-7; "
        "see the docs, Jane and https://cern.ch/x."
    )
    # keep: a labelled web link also writes its URL, so the key in it
    # reaches the text. Topic and mailto links keep the label alone; a
    # label that is the URL is written once.
    assert strip_twiki(raw, link_urls="keep") == (
        "Tracked in the ticket (https://its.cern.ch/jira/browse/"
        "CMSCOMPPR-2361) and https://its.cern.ch/jira/browse/CMSPROD-7; "
        "see the docs, Jane and https://cern.ch/x."
    )
    # The scheme is matched in any case.
    assert strip_twiki(
        "[[HTTPS://its.cern.ch/jira/browse/CMSDM-1][x]]", link_urls="keep"
    ) == "x (HTTPS://its.cern.ch/jira/browse/CMSDM-1)"
    with pytest.raises(ValueError, match="link_urls"):
        strip_twiki(raw, link_urls="yes")


def test_strip_twiki_markdown_preserve_keeps_structure():
    out = strip_twiki(
        "---++ Alpha\nLine one.   \nLine two.\n\n\n\n"
        "<table><tr><th>Site</th><th>Status</th></tr>"
        "<tr><td>T2</td><td>ok</td></tr></table>\n"
        '<img src="plots/rate.png" alt="rate plot">\n',
        heading_style="markdown",
        whitespace="preserve",
        preserve_tables=True,
        preserve_images=True,
    )
    assert out == (
        "## Alpha\nLine one.\nLine two.\n\n"
        "| Site | Status |\n| --- | --- |\n| T2 | ok |\n\n"
        "![rate plot](plots/rate.png)"
    )
    # The cms flavor on the same input: one line, no heading marker,
    # no table/image preservation beyond native pipe text.
    cms = strip_twiki(
        "---++ Alpha\nLine one.\nLine two.\n"
        '<img src="plots/rate.png" alt="rate plot">\n'
    )
    assert "\n" not in cms
    assert "#" not in cms
    assert "![" not in cms
    assert cms.startswith("Alpha Line one.")


def test_strip_twiki_inline_code_ignores_assignments():
    # Deliberate deviation from the cms original (documented in the
    # module docstring): its regex paired the '=' of separate
    # assignments, emitting 'MAX5 and MIN2' here.
    assert strip_twiki("MAX=5 and MIN=2") == "MAX=5 and MIN=2"
    # Assignments and real inline code coexist on one line.
    assert strip_twiki("Set MAX=5 then run =ls -la= now") == (
        "Set MAX=5 then run ls -la now"
    )
    # Inline-code '=' pairs never span lines.
    assert strip_twiki("A=one\ntwo=B", whitespace="preserve") == (
        "A=one\ntwo=B"
    )
    # The classic TWiki inline-code form still unwraps, including when
    # wrapped in punctuation.
    assert strip_twiki("Use =xrdcp= here") == "Use xrdcp here"
    assert strip_twiki("(=ls -la=)") == "(ls -la)"


def test_strip_twiki_bare_heading_marker_never_absorbs_next_line():
    raw = "Intro line.\n---++\nNext line stays.\n"
    # markdown mode: the bare marker line is dropped entirely — it is
    # never promoted to a heading made of the following line.
    md = strip_twiki(raw, heading_style="markdown", whitespace="preserve")
    assert md == "Intro line.\nNext line stays."
    assert "#" not in md
    # drop mode: same — the next line is not absorbed.
    assert strip_twiki(raw) == "Intro line. Next line stays."
    # Titled headings still convert / lose only their marker.
    assert strip_twiki(
        "---++ Alpha\nBody.\n", heading_style="markdown",
        whitespace="preserve",
    ) == "## Alpha\nBody."
    assert strip_twiki("---++ Alpha\nBody.\n") == "Alpha Body."


def test_strip_twiki_rejects_unknown_flags():
    with pytest.raises(ValueError, match="heading_style"):
        strip_twiki("x", heading_style="bogus")
    with pytest.raises(ValueError, match="whitespace"):
        strip_twiki("x", whitespace="bogus")


def test_parse_meta_from_meta_fixture():
    assert parse_meta(META_FIXTURE) == {
        "author": "jdoe",
        "last_modified": "2023-11-14T22:13:20Z",
        "version": "12",
        "parent_topic": "ParentTopic",
    }
    assert parse_meta("no meta here") == {
        "author": "",
        "last_modified": "",
        "version": "",
        "parent_topic": "",
    }


def test_wiki_link_and_bare_wikiword_extraction():
    assert extract_wiki_links(META_FIXTURE) == {"CMS/PhedexDocs", "WebHome"}
    # No links to URLs/mailto/anchors.
    assert extract_wiki_links(
        "[[https://cern.ch][x]] [[mailto:a@b.c][y]] [[#Anchor]]"
    ) == set()
    # WikiWords inside meta, macros, verbatim, links, and URLs are not
    # bare; DataOps and the link label's PhEDEx are.
    assert extract_bare_wikiwords(META_FIXTURE) == {"DataOps", "PhEDEx"}


def test_url_canonicalization_and_page_ids():
    url = (
        "https://twiki.cern.ch/twiki/bin/viewauth/CMS/DataOps"
        "?rev=4;raw=on#Section"
    )
    assert canonical_twiki_url(url) == (
        "https://twiki.cern.ch/twiki/bin/view/CMS/DataOps"
    )
    assert twiki_page_id_from_url(canonical_twiki_url(url)) == "CMS/DataOps"
    assert twiki_page_id_from_url("https://example.org/other") == ""
    assert twiki_node_id("CMS/Sub/Topic") == "twiki:CMS:Sub:Topic"
    # Target resolution: web_root prefixing (cms lineage semantics).
    assert topic_page_id("Bare", web_name="CMS/Sub", web_root="CMS") == (
        "CMS/Sub/Bare"
    )
    assert topic_page_id("Other/Topic", web_name="CMS", web_root="CMS") == (
        "CMS/Other/Topic"
    )
    assert topic_page_id("CMS/Other/Topic", web_name="CMS", web_root="CMS") == (
        "CMS/Other/Topic"
    )
    assert topic_page_id("Web.Topic", web_name="Ops", web_root="") == (
        "Web/Topic"
    )
    assert topic_page_id("Bare", web_name="Ops", web_root="") == "Ops/Bare"
    assert topic_page_id("", web_name="Ops") == ""


def test_default_skip_patterns():
    for name in (
        "12345.txt",
        "20240101Notes.txt",
        "1-2-2024.txt",
        "SomeTopic-replies.txt",
        "lowercase.txt",
        "WebChanges.txt",
        "WebLeftBar.txt",
        "SearchResults.txt",
    ):
        assert not is_real_page(name), name
    for name in ("DataOps.txt", "WebHome.txt", "TopicOne.txt"):
        assert is_real_page(name), name


def test_skip_patterns_are_immutable_and_overridable():
    from archi.sources._twiki_parse import DEFAULT_SKIP_PATTERNS

    # The default is an immutable tuple, not a mutable module-level
    # list that callers could poison for everyone else.
    assert isinstance(DEFAULT_SKIP_PATTERNS, tuple)
    assert twiki_mod.DEFAULT_SKIP_PATTERNS is DEFAULT_SKIP_PATTERNS
    custom = DEFAULT_SKIP_PATTERNS + (r"^SecretTopic\.txt$",)
    assert is_real_page("SecretTopic.txt") is True
    assert is_real_page("SecretTopic.txt", custom) is False
    # Lists are accepted too (compiled per unique tuple).
    assert is_real_page("SecretTopic.txt", list(custom)) is False


# --- TwikiEOSSource ----------------------------------------------------------

TOPIC_ONE = (
    '%META:TOPICINFO{author="jdoe" date="1700000000" version="12"}%\n'
    '%META:TOPICPARENT{name="TopicTwo"}%\n'
    "---++ Topic One\n"
    "See [[TopicTwo]] about CMSPROD-100 and MissingTopic mentions.\n"
)
TOPIC_TWO = "---+ Topic Two\nBody two with [[Sub.SubTopicPage]].\n"
SUB_TOPIC = "---+ Sub Topic\nGrandchild body.\n"


def _write_snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "TopicOne.txt").write_text(TOPIC_ONE)
    (root / "TopicTwo.txt").write_text(TOPIC_TWO)
    (root / "Sub").mkdir()
    (root / "Sub" / "SubTopicPage.txt").write_text(SUB_TOPIC)
    (root / "WebChanges.txt").write_text("structural page")
    (root / "12345.txt").write_text("junk")
    return root


def _nodes(facts, subtype):
    return [f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype]


def _edges(facts, edge_type):
    return [f for f in facts if isinstance(f, EdgeFact) and f.edge_type == edge_type]


def test_eos_page_nodes_meta_attrs_and_skip_patterns(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    pages = {n.node_id: n for n in _nodes(facts, "documentation_page")}
    # Skip patterns filter WebChanges.txt and 12345.txt.
    assert set(pages) == {
        "twiki:CMS:TopicOne",
        "twiki:CMS:TopicTwo",
        "twiki:CMS:Sub:SubTopicPage",
    }
    one = pages["twiki:CMS:TopicOne"]
    assert one.attrs["title"] == "TopicOne"
    assert one.attrs["author"] == "jdoe"
    assert one.attrs["version"] == "12"
    assert one.attrs["last_updated"] == "2023-11-14T22:13:20Z"
    assert one.attrs["parent_topic"] == "TopicTwo"
    assert one.attrs["web_name"] == "CMS"
    assert one.attrs["site_name"] == "twiki.cern.ch"
    assert one.attrs["url"] == (
        "https://twiki.cern.ch/twiki/bin/view/CMS/TopicOne"
    )
    assert one.attrs["source_repo"] == ""
    assert one.attrs["text"] == "TopicOne TopicTwo CMS"
    assert one.source_record_id == {"path": "TopicOne.txt"}
    sub = pages["twiki:CMS:Sub:SubTopicPage"]
    assert sub.attrs["web_name"] == "CMS/Sub"
    assert sub.source_record_id == {"path": "Sub/SubTopicPage.txt"}
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.mode == "filesystem"
    assert run.health.record_count == 3


def test_eos_chunks_parity_ids_and_chunker_name(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root))
    facts = list(source.run("r").facts)
    page = {n.node_id: n for n in _nodes(facts, "documentation_page")}[
        "twiki:CMS:TopicOne"
    ]
    chunks = [
        c for c in _nodes(facts, "document_chunk")
        if c.source_record_id["path"] == "TopicOne.txt"
    ]
    assert len(chunks) == 1
    chunk = chunks[0]
    # Parity wart: the chunk-id hash separator is a literal
    # backslash-zero (the cms original's f'..\\0..'), not a NUL byte.
    seed = f"{page.node_id}\\0{0}\\0{chunk.attrs['text']}"
    expected = "chunk:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    assert chunk.node_id == expected
    assert chunk.attrs["chunker_name"] == "cms_twiki_window_v1"
    assert chunk.attrs["heading_path"] == "TopicOne"
    # Chunk text is title + cms-stripped body (single line).
    assert chunk.attrs["text"].startswith("TopicOne ")
    assert "\n" not in chunk.attrs["text"]
    contains = [
        e for e in _edges(facts, "contains") if e.src == page.node_id
    ]
    assert [e.dst for e in contains] == [chunk.node_id]
    assert chunk.source_record_id == {"path": "TopicOne.txt", "chunk_index": 0}


def test_eos_page_references_filtered_to_known_ids(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root))
    facts = list(source.run("r").facts)
    page_ids = {n.node_id for n in _nodes(facts, "documentation_page")}
    refs = [
        e for e in _edges(facts, "references")
        if e.src.startswith("twiki:") and e.dst.startswith("twiki:")
    ]
    # TopicOne -> TopicTwo (TOPICPARENT wins the dedup over the
    # [[TopicTwo]] wiki link); TopicTwo -> Sub/SubTopicPage (wiki link).
    # TopicOne's bare "MissingTopic" resolves to twiki:CMS:MissingTopic,
    # which is not a known page id, so it is filtered out.
    assert {(e.src, e.dst, e.attrs["kind"]) for e in refs} == {
        ("twiki:CMS:TopicOne", "twiki:CMS:TopicTwo", "topic_parent"),
        ("twiki:CMS:TopicTwo", "twiki:CMS:Sub:SubTopicPage", "wiki_link"),
    }
    assert all(e.provenance == "derived_deterministic" for e in refs)
    assert all(e.dst in page_ids for e in refs)


def test_eos_chunk_entity_references(tmp_path):
    root = _write_snapshot(tmp_path)
    (tmp_path / "data" / "jira").mkdir(parents=True)
    (tmp_path / "data" / "jira" / "records.json").write_text(
        json.dumps([{"key": "CMSPROD-100"}])
    )
    source = TwikiEOSSource(
        eos_root=str(root),
        jira_records_path="data/jira/records.json",
        base=str(tmp_path),
    )
    facts = list(source.run("r").facts)
    jira_refs = [
        e for e in _edges(facts, "references") if e.dst == "jira:CMSPROD-100"
    ]
    assert len(jira_refs) == 1
    assert jira_refs[0].src.startswith("chunk:")
    assert jira_refs[0].attrs["match_type"] == "jira_issue"


def test_eos_configured_but_missing_reference_cache_raises(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(
        eos_root=str(root),
        sites_path="data/absent/sites.json",
        base=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="data/absent/sites.json"):
        source.run("r")


def test_eos_link_urls_option_reaches_chunk_text_and_the_probe(tmp_path):
    root = tmp_path / "snap"
    root.mkdir()
    (root / "MeetingNotes.txt").write_text(
        "---+ Notes\nDone: [[https://its.cern.ch/jira/browse/"
        "CMSCOMPPR-2361][reaper fix]].\n"
    )

    def chunk_text(**kwargs):
        source = TwikiEOSSource(eos_root=str(root), **kwargs)
        facts = list(source.run("run-1").facts)
        return " ".join(
            f.attrs["text"] for f in facts
            if isinstance(f, NodeFact) and f.subtype == "document_chunk"
        )

    assert "CMSCOMPPR-2361" not in chunk_text()
    assert "reaper fix (https://its.cern.ch/jira/browse/CMSCOMPPR-2361)" in (
        chunk_text(link_urls="keep")
    )
    # Unset and "drop" share one probe token (no re-ingest for a
    # deployment that never set it); "keep" changes it, because the
    # same snapshot now yields different text.
    def token(**kwargs):
        return TwikiEOSSource(
            eos_root=str(root), **kwargs
        ).change_probe.build_token()

    assert token() == token(link_urls="drop")
    assert token(link_urls="keep") != token()
    with pytest.raises(ValueError, match="link_urls"):
        TwikiEOSSource(eos_root=str(root), link_urls="both")


def test_eos_web_root_param(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root), web_root="Sandbox")
    facts = list(source.run("r").facts)
    pages = {n.node_id for n in _nodes(facts, "documentation_page")}
    assert "twiki:Sandbox:TopicOne" in pages
    assert "twiki:Sandbox:Sub:SubTopicPage" in pages


def test_eos_seed_topics_bound_the_walk(tmp_path):
    root = _write_snapshot(tmp_path)

    def page_ids(**kwargs):
        source = TwikiEOSSource(eos_root=str(root), **kwargs)
        facts = list(source.run("r").facts)
        return {n.node_id for n in _nodes(facts, "documentation_page")}

    # Default: the whole tree (cms behavior).
    assert len(page_ids()) == 3
    # Seeds only.
    assert page_ids(seed_topics=["TopicOne"], max_depth=0) == {
        "twiki:CMS:TopicOne"
    }
    # One hop: TopicOne's parent/wiki-link target TopicTwo; its bare
    # "MissingTopic" has no snapshot file and is skipped silently.
    assert page_ids(seed_topics=["TopicOne"], max_depth=1) == {
        "twiki:CMS:TopicOne",
        "twiki:CMS:TopicTwo",
    }
    # Two hops reach the subweb topic through TopicTwo's wiki link.
    assert page_ids(seed_topics=["TopicOne"], max_depth=2) == {
        "twiki:CMS:TopicOne",
        "twiki:CMS:TopicTwo",
        "twiki:CMS:Sub:SubTopicPage",
    }
    # Seeds accept web-root-qualified and dotted forms too.
    assert page_ids(seed_topics=["CMS/TopicTwo"], max_depth=1) == {
        "twiki:CMS:TopicTwo",
        "twiki:CMS:Sub:SubTopicPage",
    }
    assert page_ids(seed_topics=["Sub.SubTopicPage"], max_depth=0) == {
        "twiki:CMS:Sub:SubTopicPage"
    }


def test_eos_seed_validation():
    with pytest.raises(ValueError, match="non-empty"):
        TwikiEOSSource(eos_root="/nonexistent-any", seed_topics=[])
    with pytest.raises(ValueError, match="max_depth"):
        TwikiEOSSource(eos_root="/nonexistent-any", max_depth=-1)
    with pytest.raises(ValueError, match="web root"):
        TwikiEOSSource(eos_root="/nonexistent-any", seed_topics=["CMS"])
    with pytest.raises(ValueError, match="seed"):
        TwikiEOSSource(eos_root="/nonexistent-any", seed_topics=["  "])


def test_eos_max_files_validation(tmp_path):
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_files"):
            TwikiEOSSource(eos_root="/nonexistent-any", max_files=bad)
    # None stays "unlimited" and positive values still cap the listing.
    root = _write_snapshot(tmp_path)
    assert TwikiEOSSource(eos_root=str(root)).max_files is None
    capped = TwikiEOSSource(eos_root=str(root), max_files=1)
    assert len(list(capped.run("r").facts)) > 0
    assert capped.preflight().record_count == 1


def test_eos_missing_seed_fails_scope(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(
        eos_root=str(root),
        seed_topics=["TopicOne", "AbsentTopic"],
        max_depth=1,
    )
    run = source.run("r", mode="scope_complete")
    facts = list(run.facts)
    # The found seed's subtree is still emitted...
    assert {n.node_id for n in _nodes(facts, "documentation_page")} == {
        "twiki:CMS:TopicOne",
        "twiki:CMS:TopicTwo",
    }
    # ...but the absent seed forces an incomplete scope with a counted,
    # named health reason (a silent skip would let scope_complete
    # retract the whole missing subtree).
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "1/2" in run.health.reason
    assert "CMS/AbsentTopic" in run.health.reason
    assert "no complete scope claimed" in run.health.reason
    assert run.health.record_count == 2


def test_eos_all_seeds_missing_is_cache_missing(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(
        eos_root=str(root),
        seed_topics=["AbsentOne", "AbsentTwo"],
        max_depth=1,
    )
    run = source.run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert "2/2" in run.health.reason
    assert "CMS/AbsentOne" in run.health.reason
    assert "CMS/AbsentTwo" in run.health.reason


def test_eos_missing_followed_link_target_stays_silent(tmp_path):
    # Documented wisdqm-parity behavior: only operator-configured seeds
    # fail loudly; TopicOne's bare "MissingTopic" link target has no
    # snapshot file and is skipped without affecting scope.
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(
        eos_root=str(root), seed_topics=["TopicOne"], max_depth=1
    )
    run = source.run("r", mode="scope_complete")
    list(run.facts)
    assert run.completed_scope is True
    assert run.health.status == "ok"


def test_eos_seeded_preflight_reports_subtree_metrics(tmp_path):
    root = _write_snapshot(tmp_path)
    whole = TwikiEOSSource(eos_root=str(root)).preflight()
    assert whole.record_count == 3
    seeded = TwikiEOSSource(
        eos_root=str(root), seed_topics=["TopicOne"], max_depth=1
    ).preflight()
    # Preflight describes the seeded closure run() will emit (TopicOne
    # + TopicTwo), not the whole tree.
    assert seeded.status == "ok"
    assert seeded.record_count == 2
    assert seeded.content_hash != whole.content_hash
    assert "seeded closure" in seeded.reason
    # Missing seeds surface in the preflight reason too.
    part_missing = TwikiEOSSource(
        eos_root=str(root),
        seed_topics=["TopicOne", "AbsentTopic"],
        max_depth=0,
    ).preflight()
    assert part_missing.record_count == 1
    # Same verdict run() reaches: one of two seeds missing is a partial
    # read (endpoint_failed), not ok (PR #19 review item 1).
    assert part_missing.status == "endpoint_failed"
    assert "1/2 configured seed topics missing" in part_missing.reason


def test_eos_run_walks_the_tree_once(tmp_path, monkeypatch):
    # run() must not redo the preflight whole-tree walk on top of its
    # own listing.
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root))
    calls = {"n": 0}
    original = TwikiEOSSource._listing

    def counting(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(TwikiEOSSource, "_listing", counting)
    run = source.run("r", mode="scope_complete")
    list(run.facts)
    assert run.health.status == "ok"
    assert run.health.record_count == 3
    assert calls["n"] == 1


def test_eos_custom_skip_patterns_exclude_pages(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(
        eos_root=str(root),
        skip_patterns=twiki_mod.DEFAULT_SKIP_PATTERNS
        + (r"^TopicTwo\.txt$",),
    )
    run = source.run("r", mode="scope_complete")
    pages = {n.node_id for n in _nodes(list(run.facts), "documentation_page")}
    assert pages == {"twiki:CMS:TopicOne", "twiki:CMS:Sub:SubTopicPage"}
    assert source.preflight().record_count == 2


def test_eos_missing_root_is_safe(monkeypatch, tmp_path):
    monkeypatch.delenv("ARCHI_T_TWIKI_ROOT", raising=False)
    source = TwikiEOSSource(eos_root_env="ARCHI_T_TWIKI_ROOT")
    result = source.preflight()
    assert result.status == "missing_credential"
    assert result.credential_refs == ("ARCHI_T_TWIKI_ROOT",)
    run = source.run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "missing_credential"
    # And the env var route resolves the root.
    root = _write_snapshot(tmp_path)
    monkeypatch.setenv("ARCHI_T_TWIKI_ROOT", str(root))
    source = TwikiEOSSource(eos_root_env="ARCHI_T_TWIKI_ROOT")
    result = source.preflight()
    assert result.status == "ok"
    assert result.record_count == 3


@pytest.mark.parametrize("mode", ["scope_complete", "reconcile"])
def test_eos_empty_snapshot_dir_refuses_scope(tmp_path, mode):
    # A snapshot directory that exists but holds nothing (a failed sync,
    # an unmounted volume left as an empty mount point) is a missing
    # cache, not an empty wiki. Claiming a complete scope over it would
    # order the retraction of every page ingested before.
    root = tmp_path / "snapshot"
    root.mkdir()
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode=mode)
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert run.health.mode == "filesystem"
    assert run.health.record_count == 0
    assert run.health.cache_path == str(root)
    assert "no TWiki topics" in run.health.reason
    assert "no complete scope claimed" in run.health.reason
    preflight = source.preflight()
    assert preflight.status == "cache_missing"
    assert preflight.record_count == 0
    assert preflight.cache_path == str(root)


def test_eos_snapshot_of_only_skipped_pages_refuses_scope(tmp_path):
    # Structural pages and junk files are dropped before any record is
    # built, so a snapshot holding only those is as empty as no files.
    root = tmp_path / "snapshot"
    (root / "Sub").mkdir(parents=True)
    (root / "WebChanges.txt").write_text("structural page")
    (root / "12345.txt").write_text("junk")
    (root / "Sub" / "notes.md").write_text("not a topic")
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert source.preflight().status == "cache_missing"


def test_eos_one_topic_snapshot_still_claims_scope(tmp_path):
    # The guard is about zero topics, not few: one real page is a scope.
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "TopicOne.txt").write_text(TOPIC_ONE)
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert len(_nodes(list(run.facts), "documentation_page")) == 1
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.preflight().status == "ok"


def test_eos_completed_scope_by_mode_and_chunker_param(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root), chunker_name="my_twiki_v2")
    assert source.run("r", mode="cursor").completed_scope is False
    assert source.run("r", mode="reconcile").completed_scope is True
    chunk = _nodes(list(source.run("r").facts), "document_chunk")[0]
    assert chunk.attrs["chunker_name"] == "my_twiki_v2"


def test_eos_probe_declarations(tmp_path):
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root))
    assert TwikiEOSSource.profile == "discovery_crawl"
    assert TwikiEOSSource.change_probe_kind == "content_hash"
    assert isinstance(source.change_probe, ContentHashProbe)
    first = source.change_probe.build_token()
    assert source.change_probe.build_token() == first
    (root / "TopicOne.txt").write_text(TOPIC_ONE + "\nchanged\n")
    assert source.change_probe.build_token() != first


# --- TwikiCrawlSource --------------------------------------------------------

BASE_URL = "https://twiki.example.cern.ch/twiki"
COOKIE_ENV = "ARCHI_T_TWIKI_COOKIE_FILE"

START_RAW = (
    '%META:TOPICINFO{author="crawler" date="1700000000" version="3"}%\n'
    "---++ Start heading\n"
    "See [[LinkedTopic]] and [[Ops.BrokenTopic]] plus BareWord text.\n"
)
LINKED_RAW = "---+ Linked\nBody linking [[DeepTopic]].\n"
DEEP_RAW = "---+ Deep\nToo deep to fetch at depth 1.\n"


class _FakeResponse:
    def __init__(self, status_code=200, text="", url=""):
        self.status_code = status_code
        self.text = text
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _raw_url(page_id):
    return f"{BASE_URL}/bin/view/{page_id}?raw=all"


def _ok(page_id, raw):
    return _FakeResponse(200, raw, _raw_url(page_id))


def _default_responses():
    return {
        _raw_url("Ops/StartTopic"): _ok("Ops/StartTopic", START_RAW),
        _raw_url("Ops/LinkedTopic"): _ok("Ops/LinkedTopic", LINKED_RAW),
        _raw_url("Ops/DeepTopic"): _ok("Ops/DeepTopic", DEEP_RAW),
        _raw_url("Ops/BrokenTopic"): _FakeResponse(
            404, "gone", _raw_url("Ops/BrokenTopic")
        ),
    }


def _fake_sessions(monkeypatch, responses):
    sessions = []

    class _FakeSession:
        def __init__(self):
            self.cookies = None
            sessions.append(self)

        def get(self, url, **kwargs):
            return responses[url]

    monkeypatch.setattr(twiki_mod.requests, "Session", _FakeSession)
    return sessions


def _crawl_source(**overrides):
    kwargs = {
        "base_url": BASE_URL,
        "seed_topics": ["Ops/StartTopic"],
        "max_depth": 1,
    }
    kwargs.update(overrides)
    return TwikiCrawlSource(**kwargs)


def _write_cookie_file(path, *, expires=_FAR_FUTURE):
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".cern.ch\tTRUE\t/\tTRUE\t{expires}\tsessionid\ttest-value\n"
    )


def test_crawl_required_params_and_validation():
    with pytest.raises(TypeError):
        TwikiCrawlSource(base_url=BASE_URL)  # seeds + depth required
    with pytest.raises(ValueError, match="at least one seed"):
        _crawl_source(seed_topics=[])
    with pytest.raises(ValueError, match="Web/Topic"):
        _crawl_source(seed_topics=["TopicWithoutWeb"])
    with pytest.raises(ValueError, match="max_depth"):
        _crawl_source(max_depth=-1)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_pages"):
            _crawl_source(max_pages=bad)
    assert _crawl_source(max_pages=None).max_pages is None


def test_crawl_seeded_depth_bounding_and_emission(monkeypatch):
    responses = _default_responses()
    del responses[_raw_url("Ops/BrokenTopic")]  # keep this crawl clean
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "ok body")
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("run-7", mode="scope_complete")
    facts = list(run.facts)
    pages = {n.node_id: n for n in _nodes(facts, "documentation_page")}
    # Depth 1: seed + its two wiki links; DeepTopic (depth 2) is not
    # fetched, and the bare WikiWord "BareWord" is never fetched live.
    assert set(pages) == {
        "twiki:Ops:StartTopic",
        "twiki:Ops:LinkedTopic",
        "twiki:Ops:BrokenTopic",
    }
    start = pages["twiki:Ops:StartTopic"]
    assert start.attrs["title"] == "StartTopic"
    assert start.attrs["author"] == "crawler"
    assert start.attrs["web_name"] == "Ops"
    assert start.attrs["url"] == f"{BASE_URL}/bin/view/Ops/StartTopic"
    assert start.attrs["site_name"] == "twiki.example.cern.ch"
    assert start.source_record_id == {"path": "Ops/StartTopic"}
    # Crawl default fidelity is the wisdqm flavor: markdown headings
    # and preserved line structure.
    chunk = [
        c for c in _nodes(facts, "document_chunk")
        if c.source_record_id["path"] == "Ops/StartTopic"
    ][0]
    assert "## Start heading" in chunk.attrs["text"]
    assert "\n" in chunk.attrs["text"]
    assert chunk.attrs["chunker_name"] == "archi_twiki_crawl_v1"
    # Page-to-page references restricted to crawled ids.
    refs = {
        (e.src, e.dst, e.attrs["kind"])
        for e in _edges(facts, "references")
        if e.src.startswith("twiki:")
    }
    assert refs == {
        ("twiki:Ops:StartTopic", "twiki:Ops:LinkedTopic", "wiki_link"),
        ("twiki:Ops:StartTopic", "twiki:Ops:BrokenTopic", "wiki_link"),
    }
    # A clean crawl may claim the complete scope.
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.record_count == 3


def test_crawl_depth_zero_fetches_seeds_only(monkeypatch):
    _fake_sessions(monkeypatch, _default_responses())
    run = _crawl_source(max_depth=0).run("r", mode="scope_complete")
    pages = _nodes(list(run.facts), "documentation_page")
    assert [p.node_id for p in pages] == ["twiki:Ops:StartTopic"]
    assert run.completed_scope is True


def test_crawl_failure_never_claims_complete_scope(monkeypatch):
    responses = _default_responses()
    # A 5xx on a followed link is a real transport failure (unlike the
    # default 404, which is a dangling wiki link).
    responses[_raw_url("Ops/BrokenTopic")] = _FakeResponse(
        500, "boom", _raw_url("Ops/BrokenTopic")
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    facts = list(run.facts)
    # The successful topics are still emitted...
    assert {n.node_id for n in _nodes(facts, "documentation_page")} == {
        "twiki:Ops:StartTopic",
        "twiki:Ops:LinkedTopic",
    }
    # ...but the failed one forces an incomplete scope with a counted,
    # sampled health reason from the closed status vocabulary.
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "1/3" in run.health.reason
    assert f"{BASE_URL}/bin/view/Ops/BrokenTopic" in run.health.reason
    assert "no complete scope claimed" in run.health.reason
    assert run.health.record_count == 2


def test_crawl_dangling_link_404_is_not_a_failure(monkeypatch):
    # Default responses: BrokenTopic (a followed link, not a seed)
    # 404s. That is a dangling wiki link on the linking page, not a
    # crawl failure — the run stays ok and may claim complete scope,
    # and no garbage node is emitted for the missing target.
    _fake_sessions(monkeypatch, _default_responses())
    run = _crawl_source().run("r", mode="scope_complete")
    facts = list(run.facts)
    assert {n.node_id for n in _nodes(facts, "documentation_page")} == {
        "twiki:Ops:StartTopic",
        "twiki:Ops:LinkedTopic",
    }
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.record_count == 2
    assert "1 dangling wiki-link" in run.health.reason
    assert "missing_link_targets" in run.health.reason


def test_crawl_seed_404_is_a_failure(monkeypatch):
    # Seeds are operator configuration: an absent seed keeps strict
    # semantics.
    responses = _default_responses()
    responses[_raw_url("Ops/StartTopic")] = _FakeResponse(
        404, "gone", _raw_url("Ops/StartTopic")
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    assert _nodes(list(run.facts), "documentation_page") == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "1/1" in run.health.reason
    assert f"{BASE_URL}/bin/view/Ops/StartTopic" in run.health.reason


def test_crawl_oops_redirect_on_followed_link_is_dangling(monkeypatch):
    # Some TWiki deployments serve a missing topic as a 200 redirect to
    # /bin/oops/...?template=oopsmissing instead of a 404.
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _FakeResponse(
        200,
        "<html>Create the topic?</html>",
        f"{BASE_URL}/bin/oops/Ops/BrokenTopic?template=oopsmissing",
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    pages = {n.node_id for n in _nodes(list(run.facts), "documentation_page")}
    assert "twiki:Ops:BrokenTopic" not in pages
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert "1 dangling wiki-link" in run.health.reason


def test_crawl_oops_body_is_never_ingested(monkeypatch):
    # A 200 oops body (no %META:TOPICINFO, names the missing-topic
    # template) must not become a documentation_page.
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok(
        "Ops/BrokenTopic",
        "The topic does not exist (TopicDoesNotExist / oopsmissing).",
    )
    # A real topic that merely *mentions* the template markers carries
    # %META:TOPICINFO and is ingested normally.
    responses[_raw_url("Ops/LinkedTopic")] = _ok(
        "Ops/LinkedTopic",
        '%META:TOPICINFO{author="jdoe" date="1700000000" version="1"}%\n'
        "---+ Linked\nAbout the oopsmissing template.\n",
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    pages = {n.node_id for n in _nodes(list(run.facts), "documentation_page")}
    assert "twiki:Ops:BrokenTopic" not in pages
    assert "twiki:Ops:LinkedTopic" in pages
    assert run.completed_scope is True
    assert run.health.status == "ok"


def test_crawl_oops_page_on_seed_is_a_failure(monkeypatch):
    responses = _default_responses()
    responses[_raw_url("Ops/StartTopic")] = _FakeResponse(
        200,
        "<html>Create the topic?</html>",
        f"{BASE_URL}/bin/oops/Ops/StartTopic?template=oopsmissing",
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    assert _nodes(list(run.facts), "documentation_page") == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_crawl_login_bounce_counts_as_failure(monkeypatch):
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "fine")
    # Redirected to a real SSO host: a bounce.
    responses[_raw_url("Ops/LinkedTopic")] = _FakeResponse(
        200,
        "<html>Sign in to CERN</html>",
        "https://auth.cern.ch/auth/realms/cern",
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    facts = list(run.facts)
    assert "twiki:Ops:LinkedTopic" not in {
        n.node_id for n in _nodes(facts, "documentation_page")
    }
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "1/3" in run.health.reason


def test_crawl_login_lookalike_body_is_not_a_bounce(monkeypatch):
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "fine")
    # Non-redirected 200 that merely mentions auth.cern.ch: a real topic.
    responses[_raw_url("Ops/LinkedTopic")] = _ok(
        "Ops/LinkedTopic",
        "---+ Linked\nGet a token at auth.cern.ch first.\n",
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    pages = {n.node_id for n in _nodes(list(run.facts), "documentation_page")}
    assert "twiki:Ops:LinkedTopic" in pages
    assert run.completed_scope is True
    assert run.health.status == "ok"


def test_crawl_max_pages_cap(monkeypatch):
    _fake_sessions(monkeypatch, _default_responses())
    run = _crawl_source(max_pages=1).run("r")
    pages = _nodes(list(run.facts), "documentation_page")
    assert [p.node_id for p in pages] == ["twiki:Ops:StartTopic"]


def test_crawl_max_pages_truncation_never_claims_complete_scope(monkeypatch):
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "fine")
    _fake_sessions(monkeypatch, responses)
    # Every fetch succeeds, but max_pages stops the crawl with
    # LinkedTopic and BrokenTopic still queued: the scope was not
    # covered, so scope_complete mode must not claim it.
    run = _crawl_source(max_pages=1).run("r", mode="scope_complete")
    pages = _nodes(list(run.facts), "documentation_page")
    assert [p.node_id for p in pages] == ["twiki:Ops:StartTopic"]
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "crawl truncated at max_pages=1 with 2 queued" in run.health.reason
    assert "no complete scope claimed" in run.health.reason
    # An uncapped crawl of the same responses still claims it.
    run = _crawl_source().run("r", mode="scope_complete")
    list(run.facts)
    assert run.completed_scope is True


def test_crawl_missing_charset_defaults_to_utf8(monkeypatch):
    def _real_response(page_id, content, content_type):
        # A genuine requests.Response, prepared the way the adapter
        # would for this Content-Type (encoding from headers; text/*
        # without charset historically yields ISO-8859-1).
        resp = requests.Response()
        resp.status_code = 200
        resp._content = content
        resp.headers["Content-Type"] = content_type
        resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)
        resp.url = _raw_url(page_id)
        return resp

    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "fine")
    responses[_raw_url("Ops/LinkedTopic")] = _real_response(
        "Ops/LinkedTopic",
        "---+ Linked\nCafé résumé naïve\n".encode("utf-8"),
        "text/plain",  # no charset declared -> decode as UTF-8
    )
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source().run("r", mode="scope_complete")
    chunk = [
        c for c in _nodes(list(run.facts), "document_chunk")
        if c.source_record_id["path"] == "Ops/LinkedTopic"
    ][0]
    assert "Café résumé naïve" in chunk.attrs["text"]
    assert run.health.status == "ok"
    # A declared charset is respected, not overridden.
    responses[_raw_url("Ops/LinkedTopic")] = _real_response(
        "Ops/LinkedTopic",
        "---+ Linked\nDéjà vu\n".encode("iso-8859-1"),
        "text/plain; charset=iso-8859-1",
    )
    run = _crawl_source().run("r", mode="scope_complete")
    chunk = [
        c for c in _nodes(list(run.facts), "document_chunk")
        if c.source_record_id["path"] == "Ops/LinkedTopic"
    ][0]
    assert "Déjà vu" in chunk.attrs["text"]


def test_crawl_public_web_needs_no_cookie(monkeypatch):
    monkeypatch.delenv(COOKIE_ENV, raising=False)
    source = _crawl_source()  # cookie_file_env not configured
    result = source.preflight()
    assert result.status == "ok"
    assert result.credential_refs == ()
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "fine")
    _fake_sessions(monkeypatch, responses)
    run = source.run("r", mode="scope_complete")
    assert run.completed_scope is True
    assert run.health.credential_refs == ()


def test_crawl_missing_or_unparseable_cookie_is_auth_failed(
    monkeypatch, tmp_path
):
    # Env var configured but unset.
    monkeypatch.delenv(COOKIE_ENV, raising=False)
    source = _crawl_source(cookie_file_env=COOKIE_ENV)
    assert source.preflight().status == "missing_credential"
    run = source.run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "auth_failed"
    assert run.health.credential_refs == (COOKIE_ENV,)
    assert "no complete scope" in run.health.reason
    # Env var set but the file does not exist.
    monkeypatch.setenv(COOKIE_ENV, str(tmp_path / "absent.txt"))
    run = _crawl_source(cookie_file_env=COOKIE_ENV).run("r")
    assert run.completed_scope is False
    assert run.health.status == "auth_failed"
    # File exists but is not a Netscape cookie jar.
    bad = tmp_path / "bad.txt"
    bad.write_text("definitely not a cookie jar")
    monkeypatch.setenv(COOKIE_ENV, str(bad))
    run = _crawl_source(cookie_file_env=COOKIE_ENV).run("r")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "auth_failed"


def test_crawl_cookie_jar_is_wired_into_the_session(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "fine")
    sessions = _fake_sessions(monkeypatch, responses)
    source = _crawl_source(cookie_file_env=COOKIE_ENV)
    assert source.preflight().status == "ok"
    run = source.run("r", mode="scope_complete")
    list(run.facts)
    assert sessions and {c.name for c in sessions[0].cookies} == {"sessionid"}
    assert run.health.credential_refs == (COOKIE_ENV,)
    # Expired cookie file: preflight degrades to auth_failed.
    _write_cookie_file(cookie_path, expires=_PAST)
    assert source.preflight().status == "auth_failed"


def test_crawl_seed_urls_are_canonicalized(monkeypatch):
    responses = _default_responses()
    responses[_raw_url("Ops/BrokenTopic")] = _ok("Ops/BrokenTopic", "fine")
    _fake_sessions(monkeypatch, responses)
    source = _crawl_source(
        seed_topics=[
            "https://twiki.example.cern.ch/twiki/bin/viewauth/Ops/StartTopic"
            "?rev=2#Frag"
        ]
    )
    assert source.seed_topics == ("Ops/StartTopic",)
    run = source.run("r", mode="scope_complete")
    assert run.completed_scope is True


def test_crawl_probe_declarations():
    source = _crawl_source()
    assert TwikiCrawlSource.profile == "discovery_crawl"
    assert TwikiCrawlSource.change_probe_kind == "mutable_api"
    assert isinstance(source.change_probe, MutableApiProbe)
    first = source.change_probe.build_token()
    time.sleep(0.002)
    assert source.change_probe.build_token() != first


# --- email redaction (names kept; operator decision for cms-kb, 2026-09-28) --

# One address per field or markup form, each with a distinct local part so
# a leak names where it came from. Names (Main.JohnDoe, John Doe, Jane Roe,
# the link label) must survive.
EMAIL_TOPIC = (
    '%META:TOPICINFO{author="authoraddr@cern.ch" date="1700000000" '
    'version="3 versionaddr@cern.ch"}%\n'
    '%META:TOPICPARENT{name="parentaddr@cern.ch"}%\n'
    "---++ Contacts\n"
    "Contact Main.JohnDoe (John Doe, plainaddr@cern.ch) or "
    "[[mailto:linkaddr@cern.ch][Jane Roe]].\n"
    "Bare link [[mailto:barelinkaddr@cern.ch]], url pctaddr%40cern.ch, "
    "entity entityaddr&#64;cern.ch, tagged plus.tag+ops@fnal.gov.\n"
    '<a href="mailto:hrefaddr@cern.ch">Mail Alice Smith</a>\n'
    "| Contact | tableaddr@cern.ch |\n"
    "Spelled: bracketaddr[AT]cern.ch, parenaddr(at)cern(dot)ch, "
    "word.addr at cern.ch, glued.addr_at_cern.ch, "
    "spaced AT cern DOT ch, nospam.addr_NOSPAM_AT_cern.ch.\n"
)
EMAIL_TOPIC_LOCAL_PARTS = (
    "authoraddr",
    "versionaddr",
    "parentaddr",
    "plainaddr",
    "linkaddr",
    "barelinkaddr",
    "pctaddr",
    "entityaddr",
    "plus.tag",
    "hrefaddr",
    "tableaddr",
    "bracketaddr",
    "parenaddr",
    "glued.addr",
    "nospam.addr",
)


def _emitted(facts):
    return json.dumps(
        [
            {
                "id": getattr(f, "node_id", None),
                "attrs": f.attrs,
                "record": f.source_record_id,
            }
            for f in facts
        ],
        sort_keys=True,
        ensure_ascii=False,
    )


def _assert_no_address(emitted, local_parts):
    folded = emitted.casefold()
    assert [part for part in local_parts if part in folded] == []
    assert "@" not in emitted
    assert "%40" not in emitted
    assert "&#64;" not in emitted


def test_eos_email_addresses_removed_from_every_emitted_field(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "ContactsTopic.txt").write_text(EMAIL_TOPIC)
    run = TwikiEOSSource(eos_root=str(root)).run("r", mode="scope_complete")
    facts = list(run.facts)
    _assert_no_address(_emitted(facts), EMAIL_TOPIC_LOCAL_PARTS)

    (page,) = _nodes(facts, "documentation_page")
    # A field that held only an address is left empty, as in the JIRA source.
    assert page.attrs["author"] == ""
    assert page.attrs["parent_topic"] == ""
    assert page.attrs["version"] == "3 "
    assert page.attrs["title"] == "ContactsTopic"
    (chunk,) = _nodes(facts, "document_chunk")
    text = chunk.attrs["text"]
    # Names are kept; only the addresses go.
    assert "Contact Main.JohnDoe (John Doe, )" in text
    assert "or Jane Roe." in text
    assert "Mail Alice Smith" in text
    assert "Bare link mailto:, url , entity , tagged ." in text
    assert "| Contact |  |" in text
    # Bracketed, glued and NOSPAM forms go; the two free-prose forms stay
    # (operator rule, 2026-09-28).
    assert text.endswith(
        "Spelled: , , word.addr at cern.ch, , spaced AT cern DOT ch, ."
    )
    assert run.health.status == "ok"


def test_eos_email_redaction_covers_the_title_of_fixture_records():
    record = twiki_mod.TwikiRecord(
        page_id="CMS/OwnerPage",
        web_name="CMS",
        web_root="CMS",
        title="Owner titleaddr@cern.ch page",
        url="https://twiki.cern.ch/twiki/bin/view/CMS/OwnerPage",
        source_path="OwnerPage.txt",
        body="Ask bodyaddr@cern.ch or Main.AliceSmith.",
        author="AliceSmith",
        parent_topic="WebHome",
    )
    facts = list(TwikiEOSSource(records=[record]).run("r").facts)
    _assert_no_address(_emitted(facts), ("titleaddr", "bodyaddr"))
    (page,) = _nodes(facts, "documentation_page")
    assert page.attrs["title"] == "Owner  page"
    assert page.attrs["label"] == "Owner  page"
    assert page.attrs["author"] == "AliceSmith"
    assert page.attrs["text"] == "Owner  page WebHome CMS"
    (chunk,) = _nodes(facts, "document_chunk")
    assert chunk.attrs["heading_path"] == "Owner  page"
    assert chunk.attrs["text"] == "Owner  page Ask  or Main.AliceSmith."


def test_eos_address_free_topic_is_emitted_byte_identical(tmp_path):
    # Redaction decodes nothing: a topic with entities, names, a version
    # pin and "AT"-words but no address keeps its exact text, so its chunk
    # text and chunk id are the ones the parser alone would give.
    raw = (
        '%META:TOPICINFO{author="JohnDoe" date="1700000000" version="4"}%\n'
        "---++ Plain\n"
        "Main.JohnDoe and Jane Roe at CERN: AT&amp;T, R&amp;D &commat; ops, "
        "&#64; alone, pip install numpy@1.26.4, 100% done, served at "
        "cmsweb.cern.ch, run at 13.6 TeV, file x_2016_at_13TeV.root, "
        "remove NOSPAM.\n"
    )
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "PlainTopic.txt").write_text(raw)
    facts = list(TwikiEOSSource(eos_root=str(root)).run("r").facts)
    (page,) = _nodes(facts, "documentation_page")
    assert page.attrs["author"] == "JohnDoe"
    (chunk,) = _nodes(facts, "document_chunk")
    expected_text = "PlainTopic " + strip_twiki(raw)
    assert chunk.attrs["text"] == expected_text
    seed = f"twiki:CMS:PlainTopic\\0{0}\\0{expected_text}"
    expected_id = "chunk:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    assert chunk.node_id == expected_id


def test_eos_address_across_a_chunk_boundary_leaves_no_fragment(tmp_path):
    # Redaction runs before chunking, so an address that the 4,000-char
    # window would split is still removed whole from both chunks.
    # "LongTopic " + 1,990 "x " pairs puts the address at chars 3990-4020.
    raw = "x " * 1990 + "boundaryaddr.longlocal@cern.ch " + "tail text " * 30
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "LongTopic.txt").write_text(raw)
    facts = list(TwikiEOSSource(eos_root=str(root)).run("r").facts)
    chunks = _nodes(facts, "document_chunk")
    assert len(chunks) == 2
    joined = " ".join(c.attrs["text"] for c in chunks)
    assert "boundaryaddr" not in joined
    assert "longlocal" not in joined
    assert "@" not in joined
    assert "tail text" in joined


def test_change_probes_fingerprint_the_emission_and_redaction_code():
    # The class body alone does not hold the fact-building functions or
    # the redaction, so a change to them must still change the probe
    # token; otherwise an unchanged snapshot is skipped and already
    # stored text keeps its addresses.
    import sys

    from archi.enrichment import anonymizer

    for source_class in (TwikiEOSSource, TwikiCrawlSource):
        targets = twiki_mod._emit_targets(source_class)
        assert source_class in targets
        assert sys.modules["archi.sources.twiki"] in targets
        assert anonymizer in targets
        from archi.sources import _twiki_analysis_id

        assert _twiki_analysis_id in targets


def test_crawl_email_addresses_removed_and_names_kept(monkeypatch):
    raw = (
        '%META:TOPICINFO{author="crawlauthor@cern.ch" date="1700000000" '
        'version="2"}%\n'
        "---++ Owners\n"
        "Owner Main.JaneRoe (crawlbody@cern.ch).\n"
        "<table><tr><td>Jane Roe</td><td>crawlcell&#64;cern.ch</td></tr>"
        "</table>\n"
    )
    responses = {_raw_url("Ops/StartTopic"): _ok("Ops/StartTopic", raw)}
    _fake_sessions(monkeypatch, responses)
    run = _crawl_source(max_depth=0).run("r", mode="scope_complete")
    facts = list(run.facts)
    _assert_no_address(
        _emitted(facts), ("crawlauthor", "crawlbody", "crawlcell")
    )
    (chunk,) = _nodes(facts, "document_chunk")
    assert "Owner Main.JaneRoe ()." in chunk.attrs["text"]
    assert "| Jane Roe |  |" in chunk.attrs["text"]


# --- operator rule for spelled-out forms and git@ (Jason, 2026-09-28) -------


def _single_chunk_text(tmp_path, name, raw):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / f"{name}.txt").write_text(raw)
    facts = list(TwikiEOSSource(eos_root=str(root)).run("r").facts)
    (chunk,) = _nodes(facts, "document_chunk")
    return chunk


def test_eos_prose_forms_and_git_account_are_emitted_byte_identical(tmp_path):
    # Free-prose "at ... dot" forms and the git@ account are not removed,
    # so a topic holding only those keeps the parser's exact text and the
    # chunk id computed from it.
    raw = (
        '%META:TOPICINFO{author="JohnDoe" date="1700000000" version="2"}%\n'
        "---++ Setup\n"
        "Write to john.doe at cern.ch or jdoe AT cern DOT ch; the group is "
        "based at cern.ch.\n"
        "<verbatim>\ngit clone git@github.com:cms-sw/cmssw.git\n"
        "git remote add up git@gitlab.cern.ch:cms/y.git\n</verbatim>\n"
    )
    chunk = _single_chunk_text(tmp_path, "SetupTopic", raw)
    expected_text = "SetupTopic " + strip_twiki(raw)
    assert "john.doe at cern.ch" in expected_text
    assert "git clone git@github.com:cms-sw/cmssw.git" in expected_text
    assert chunk.attrs["text"] == expected_text
    seed = f"twiki:CMS:SetupTopic\\0{0}\\0{expected_text}"
    expected_id = "chunk:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    assert chunk.node_id == expected_id


def test_eos_bracketed_forms_and_ssh_logins_removed_git_kept(tmp_path):
    raw = (
        "---++ Access\n"
        "Mail bracketone[AT]cern.ch or brackettwo(at)cern(dot)ch or "
        "bracketthree at cern[DOT]ch.\n"
        "Log in with ssh sshuser@lxplus.cern.ch then run "
        "git clone git@github.com:x/y.git by gitnext@cern.ch today.\n"
    )
    chunk = _single_chunk_text(tmp_path, "AccessTopic", raw)
    text = chunk.attrs["text"]
    for part in ("bracketone", "brackettwo", "bracketthree", "sshuser", "gitnext"):
        assert part not in text
    assert "Mail  or  or ." in text
    assert "ssh  then run git clone git@github.com:x/y.git by  today." in text


# --- empty-cache-fails-loudly: PR #19 review follow-ups ----------------------
#
# Each case below claimed a complete scope (or disagreed between
# preflight and run) on main 8e056961. All snapshot content is made up.

_NOT_ROOT = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads mode-000 files and folders",
)


def _page_ids(run):
    return {n.node_id for n in _nodes(list(run.facts), "documentation_page")}


@pytest.mark.parametrize("mode", ["scope_complete", "reconcile"])
def test_eos_all_seeds_missing_preflight_agrees_with_run(tmp_path, mode):
    # Review item 1: preflight said ok with 0 records; run said
    # cache_missing.
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(
        eos_root=str(root), seed_topics=["AbsentOne", "AbsentTwo"]
    )
    run = source.run("r", mode=mode)
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert "2/2 configured seed topics missing" in run.health.reason
    preflight = source.preflight()
    assert preflight.status == "cache_missing"
    assert preflight.cache_path == str(root)
    assert "2/2 configured seed topics missing" in preflight.reason


def test_eos_one_seed_missing_preflight_agrees_with_run(tmp_path):
    # Review item 1: preflight said ok; run said endpoint_failed.
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(
        eos_root=str(root), seed_topics=["TopicOne", "AbsentTopic"],
        max_depth=0,
    )
    run = source.run("r", mode="scope_complete")
    assert _page_ids(run) == {"twiki:CMS:TopicOne"}
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert source.preflight().status == "endpoint_failed"


@_NOT_ROOT
@pytest.mark.parametrize("mode", ["scope_complete", "reconcile"])
def test_eos_unreadable_root_says_it_could_not_be_read(tmp_path, mode):
    # Review item 2: the refusal said "holds no TWiki topics".
    root = _write_snapshot(tmp_path)
    root.chmod(0)
    try:
        source = TwikiEOSSource(eos_root=str(root))
        run = source.run("r", mode=mode)
        assert list(run.facts) == []
        assert run.completed_scope is False
        assert run.health.status == "cache_missing"
        assert run.health.cache_path == str(root)
        assert "could not be read" in run.health.reason
        assert "holds no TWiki topics" not in run.health.reason
        preflight = source.preflight()
        assert preflight.status == "cache_missing"
        assert "could not be read" in preflight.reason
    finally:
        root.chmod(0o755)


@_NOT_ROOT
def test_eos_unreadable_subfolder_refuses_scope(tmp_path):
    # Review item 3: rglob hid the PermissionError and the run claimed
    # a complete scope, retracting every page under the folder.
    root = _write_snapshot(tmp_path)
    (root / "Sub").chmod(0)
    try:
        source = TwikiEOSSource(eos_root=str(root))
        run = source.run("r", mode="scope_complete")
        assert _page_ids(run) == {"twiki:CMS:TopicOne", "twiki:CMS:TopicTwo"}
        assert run.completed_scope is False
        assert run.health.status == "endpoint_failed"
        assert "could not be read, e.g. Sub" in run.health.reason
        preflight = source.preflight()
        assert preflight.status == "endpoint_failed"
        assert preflight.record_count == 2
    finally:
        (root / "Sub").chmod(0o755)


@_NOT_ROOT
def test_eos_unreadable_topic_file_refuses_scope(tmp_path):
    # main: read_text raised PermissionError out of run().
    root = _write_snapshot(tmp_path)
    (root / "TopicTwo.txt").chmod(0)
    try:
        source = TwikiEOSSource(eos_root=str(root))
        run = source.run("r", mode="scope_complete")
        assert "twiki:CMS:TopicTwo" not in _page_ids(run)
        assert run.completed_scope is False
        assert run.health.status == "endpoint_failed"
        assert "TopicTwo.txt" in run.health.reason
        assert source.preflight().status == "endpoint_failed"
    finally:
        (root / "TopicTwo.txt").chmod(0o644)


def test_eos_zero_byte_topic_is_skipped_and_claims_scope(tmp_path):
    # Review item 4 said a zero-byte topic is an interrupted sync and
    # refused the scope. The real CMS snapshot holds two topics that are
    # empty upstream, so that rule refused every real ingest
    # (twiki-empty-topics, 2026-09-29). A few empty topics are skipped
    # (never emitted: that would blank the page), named, and the scope
    # is still claimed, so the page is retracted as absent.
    root = _write_snapshot(tmp_path)
    (root / "TopicTwo.txt").write_bytes(b"")
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert _page_ids(run) == {
        "twiki:CMS:TopicOne", "twiki:CMS:Sub:SubTopicPage",
    }
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.record_count == 2
    assert "1 empty topic files skipped" in run.health.reason
    assert run.health.reason.endswith(": TopicTwo.txt")
    assert source.last_empty_topics == ("TopicTwo.txt",)
    preflight = source.preflight()
    assert preflight.status == "ok"
    assert preflight.record_count == 2
    assert "TopicTwo.txt" in preflight.reason


def test_eos_zero_byte_seed_topic_is_skipped_and_claims_scope(tmp_path):
    root = _write_snapshot(tmp_path)
    (root / "TopicTwo.txt").write_bytes(b"")
    source = TwikiEOSSource(
        eos_root=str(root), seed_topics=["TopicOne"], max_depth=1
    )
    run = source.run("r", mode="scope_complete")
    assert _page_ids(run) == {"twiki:CMS:TopicOne"}
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert "TopicTwo.txt" in run.health.reason
    preflight = source.preflight()
    assert preflight.status == "ok"
    assert "TopicTwo.txt" in preflight.reason


def _write_many_topics(root, count):
    for index in range(count):
        (root / f"Filler{index:03d}.txt").write_text(
            f"---+ Filler {index}\nBody {index}.\n"
        )


def test_eos_two_empty_topics_among_many_claim_scope_and_are_named(tmp_path):
    # The real case: two topics empty upstream in a large snapshot.
    root = _write_snapshot(tmp_path)
    _write_many_topics(root, 40)
    (root / "EmptyUpstream.txt").write_bytes(b"")
    (root / "Sub" / "Blank.txt").write_bytes(b" \n\x00\n")
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert len(_page_ids(run)) == 43
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.record_count == 43
    assert (
        "2 empty topic files skipped (zero bytes or only whitespace/NUL "
        "bytes; taken as empty upstream, at most 10 allowed among 45 "
        "topic files): EmptyUpstream.txt, Sub/Blank.txt"
    ) in run.health.reason
    assert source.last_empty_topics == ("EmptyUpstream.txt", "Sub/Blank.txt")
    preflight = source.preflight()
    assert preflight.status == "ok"
    assert preflight.record_count == 43
    assert "EmptyUpstream.txt, Sub/Blank.txt" in preflight.reason


def test_eos_empty_topics_at_the_limit_claim_scope(tmp_path):
    root = _write_snapshot(tmp_path)
    for index in range(10):
        (root / f"Empty{index:02d}.txt").write_bytes(b"")
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert "10 empty topic files skipped" in run.health.reason
    assert source.preflight().status == "ok"


@pytest.mark.parametrize("seeded", [False, True], ids=["tree", "seeded"])
def test_eos_many_empty_topics_refuse_scope(tmp_path, seeded):
    # Above max(10, 0.1%) empty topics the snapshot looks like an
    # interrupted sync: skip them, emit the rest, claim no scope.
    root = _write_snapshot(tmp_path)
    names = [f"Empty{index:02d}" for index in range(11)]
    for name in names:
        (root / f"{name}.txt").write_bytes(b"")
    kwargs = (
        {"seed_topics": ["TopicOne", *names], "max_depth": 0}
        if seeded else {}
    )
    source = TwikiEOSSource(eos_root=str(root), **kwargs)
    run = source.run("r", mode="scope_complete")
    assert "twiki:CMS:TopicOne" in _page_ids(run)
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "11 topic files are zero bytes" in run.health.reason
    assert "more than the 10 empty topics allowed" in run.health.reason
    assert "interrupted sync" in run.health.reason
    assert "no complete scope claimed" in run.health.reason
    assert len(source.last_empty_topics) == 11
    preflight = source.preflight()
    assert preflight.status == "endpoint_failed"
    assert preflight.reason == run.health.reason


def test_eos_tolerated_empty_topics_are_named_when_another_problem_refuses(
    tmp_path,
):
    root = _write_snapshot(tmp_path)
    (root / "TopicTwo.txt").write_bytes(b"")
    source = TwikiEOSSource(
        eos_root=str(root), seed_topics=["TopicOne", "NoSuchSeed"],
        max_depth=1,
    )
    run = source.run("r", mode="scope_complete")
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "NoSuchSeed" in run.health.reason
    assert "1 empty topic files skipped" in run.health.reason
    assert source.preflight().status == "endpoint_failed"


def test_eos_empty_topic_names_are_capped_in_the_reason(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(twiki_mod, "EMPTY_TOPIC_NAMES_SHOWN", 1)
    root = _write_snapshot(tmp_path)
    for index in range(3):
        (root / f"Empty{index:02d}.txt").write_bytes(b"")
    run = TwikiEOSSource(eos_root=str(root)).run("r", mode="scope_complete")
    assert run.completed_scope is True
    assert run.health.reason.endswith(": Empty00.txt, and 2 more")


@pytest.mark.parametrize(
    ("topic_files", "limit"),
    [(0, 10), (45, 10), (10999, 10), (11000, 11), (43746, 43)],
)
def test_empty_topic_limit_is_max_of_ten_and_a_tenth_of_a_percent(
    topic_files, limit
):
    assert twiki_mod.empty_topic_limit(topic_files) == limit
    assert twiki_mod.empty_topics_tolerated(limit, topic_files)
    assert not twiki_mod.empty_topics_tolerated(limit + 1, topic_files)


def test_eos_snapshot_of_only_zero_byte_topics_is_cache_missing(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "TopicOne.txt").write_bytes(b"")
    (root / "TopicTwo.txt").write_bytes(b"")
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert "2 topic files are zero bytes" in run.health.reason
    assert "and no readable topic" in run.health.reason
    assert "allowed" not in run.health.reason
    assert source.preflight().status == "cache_missing"


def test_eos_hidden_files_and_folders_are_not_topics(tmp_path):
    # Review item 5: .TopicA.txt and .sync/... were ingested as topics.
    root = _write_snapshot(tmp_path)
    (root / ".TopicA.txt").write_text(TOPIC_TWO)
    (root / ".sync").mkdir()
    (root / ".sync" / "TopicB.txt").write_text(TOPIC_TWO)
    (root / "Sub" / ".hidden").mkdir()
    (root / "Sub" / ".hidden" / "TopicC.txt").write_text(TOPIC_TWO)
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert _page_ids(run) == {
        "twiki:CMS:TopicOne", "twiki:CMS:TopicTwo",
        "twiki:CMS:Sub:SubTopicPage",
    }
    # Hidden files are not part of the scope, so skipping them is not a
    # partial read: the run still claims its scope.
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.preflight().record_count == 3


def test_eos_snapshot_of_only_hidden_topics_is_cache_missing(tmp_path):
    root = tmp_path / "snapshot"
    (root / ".sync").mkdir(parents=True)
    (root / ".sync" / "TopicOne.txt").write_text(TOPIC_ONE)
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.health.status == "cache_missing"
    assert source.preflight().status == "cache_missing"


# --- review of #21: max_files cap and blank topics ---------------------------


@pytest.mark.parametrize("mode", ["scope_complete", "reconcile"])
def test_eos_max_files_cut_claims_no_scope(tmp_path, mode):
    # The cap read 1 of 3 topics and still claimed a complete scope,
    # retracting the 2 pages past the cap.
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root), max_files=1)
    run = source.run("r", mode=mode)
    assert len(_page_ids(run)) == 1
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "max_files=1 left out 2 of 3 topics" in run.health.reason
    preflight = source.preflight()
    assert preflight.status == "ok"
    assert "no complete scope claimed" in preflight.reason


def test_eos_max_files_that_cuts_nothing_still_claims_scope(tmp_path):
    root = _write_snapshot(tmp_path)
    run = TwikiEOSSource(eos_root=str(root), max_files=3).run(
        "r", mode="scope_complete"
    )
    assert len(_page_ids(run)) == 3
    assert run.completed_scope is True


@pytest.mark.parametrize(
    "content", [b"   \n\t \r\n", b"\x00" * 70000, b"\n\x00 \x00\n"],
    ids=["whitespace", "nul-bytes", "mixed"],
)
def test_eos_blank_topic_is_skipped_and_claims_scope(tmp_path, content):
    # Whitespace-only or NUL-filled topics were ingested as real pages,
    # blanking their content. Treated like zero-byte files.
    root = _write_snapshot(tmp_path)
    (root / "TopicTwo.txt").write_bytes(content)
    source = TwikiEOSSource(eos_root=str(root))
    run = source.run("r", mode="scope_complete")
    assert "twiki:CMS:TopicTwo" not in _page_ids(run)
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert "TopicTwo.txt" in run.health.reason
    assert twiki_mod.is_blank_topic(content)
    preflight = source.preflight()
    assert preflight.status == "ok"
    assert "TopicTwo.txt" in preflight.reason


def test_eos_blank_seed_topic_is_skipped_and_claims_scope(tmp_path):
    root = _write_snapshot(tmp_path)
    (root / "TopicTwo.txt").write_bytes(b"\x00\x00\n")
    source = TwikiEOSSource(
        eos_root=str(root), seed_topics=["TopicOne"], max_depth=1
    )
    run = source.run("r", mode="scope_complete")
    assert _page_ids(run) == {"twiki:CMS:TopicOne"}
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.preflight().status == "ok"
