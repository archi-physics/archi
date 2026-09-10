"""task.w2.sources-twiki — parser core + TwikiEOSSource + TwikiCrawlSource,
offline and fixture-driven."""
import hashlib
import json
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
    assert "1 seeds missing" in part_missing.reason


def test_eos_run_walks_the_tree_once(tmp_path, monkeypatch):
    # run() must not redo the preflight whole-tree walk on top of its
    # own listing.
    root = _write_snapshot(tmp_path)
    source = TwikiEOSSource(eos_root=str(root))
    calls = {"n": 0}
    original = TwikiEOSSource._paths

    def counting(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(TwikiEOSSource, "_paths", counting)
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
