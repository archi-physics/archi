"""pact twiki-page-index — twiki_eos can record which pages it emitted.

A source that links INTO TWiki pages must know which pages the ingest
kept, because the physics filter drops most of a CMS snapshot and an
edge to a dropped page is an unresolvable endpoint that fails admission
for the whole emission. The index makes that set a by-product of the
run that produced the pages, so it cannot describe a different snapshot.

The first test here is the one that matters to every existing archi
consumer: unset, nothing is written and nothing changes.
"""
import json

from archi.sources.twiki import TwikiEOSSource


def _topic(body_parent: str = "") -> str:
    parent = (
        f'%META:TOPICPARENT{{name="{body_parent}"}}%\n' if body_parent else ""
    )
    return (
        '%META:TOPICINFO{author="jdoe" date="1700000000" version="3"}%\n'
        f"{parent}"
        "---+ Heading\nBody text.\n"
    )


def _snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "HIG-19-001.txt").write_text(_topic())
    (root / "CompOpsTransferTeam.txt").write_text(_topic())
    (root / "RandomUnrelatedPage.txt").write_text(_topic())
    return root


def _page_ids(facts):
    return sorted(
        f.node_id for f in facts
        if getattr(f, "subtype", None) == "documentation_page"
    )


# --- off by default ----------------------------------------------------------

def test_no_index_is_written_by_default(tmp_path):
    root = _snapshot(tmp_path)
    out = tmp_path / "index"
    out.mkdir()
    source = TwikiEOSSource(eos_root=str(root))
    list(source.run("run-1", mode="scope_complete").facts)
    assert source.page_index_path is None
    assert list(out.iterdir()) == []


# --- the opt-in path ---------------------------------------------------------

def test_index_equals_the_pages_the_run_emitted(tmp_path):
    root = _snapshot(tmp_path)
    index = tmp_path / "pages.json"
    run = TwikiEOSSource(
        eos_root=str(root), page_index_path=str(index)
    ).run("run-1", mode="scope_complete")
    emitted = _page_ids(list(run.facts))
    assert index.is_file()
    assert json.loads(index.read_text()) == emitted
    assert len(emitted) == 3


def test_index_holds_only_pages_the_physics_filter_kept(tmp_path):
    """The whole point: the kept set is not the snapshot listing."""
    root = _snapshot(tmp_path)
    index = tmp_path / "pages.json"
    run = TwikiEOSSource(
        eos_root=str(root), physics_filter=True, page_index_path=str(index)
    ).run("run-1", mode="scope_complete")
    emitted = _page_ids(list(run.facts))
    indexed = json.loads(index.read_text())
    assert indexed == emitted
    assert any("HIG-19-001" in page_id for page_id in indexed)
    assert not any("CompOpsTransferTeam" in page_id for page_id in indexed)


def test_index_is_written_into_a_directory_that_does_not_exist_yet(tmp_path):
    root = _snapshot(tmp_path)
    index = tmp_path / "nested" / "dir" / "pages.json"
    TwikiEOSSource(
        eos_root=str(root), page_index_path=str(index)
    ).run("run-1", mode="scope_complete")
    assert index.is_file()


def test_no_temp_file_is_left_behind(tmp_path):
    """The write is atomic, so a consumer never sees a partial list."""
    root = _snapshot(tmp_path)
    index = tmp_path / "pages.json"
    TwikiEOSSource(
        eos_root=str(root), page_index_path=str(index)
    ).run("run-1", mode="scope_complete")
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


# --- a refused scope must not damage an existing index ------------------------

def test_a_filter_that_keeps_nothing_leaves_the_old_index_alone(tmp_path):
    """A failed run must not hand the consumer a truncated list.

    A physics filter that keeps zero pages is a misconfigured mount, not
    an empty wiki; the run refuses its scope. If it rewrote the index it
    would erase a good one, and the next note ingest would silently lose
    every bridge edge instead of failing loudly.
    """
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "CompOpsTransferTeam.txt").write_text(_topic())
    (root / "RandomUnrelatedPage.txt").write_text(_topic())

    index = tmp_path / "pages.json"
    index.write_text(json.dumps(["twiki:CMS:KeptFromAGoodRun"]))

    run = TwikiEOSSource(
        eos_root=str(root), physics_filter=True, page_index_path=str(index)
    ).run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.health.status == "endpoint_failed"
    assert json.loads(index.read_text()) == ["twiki:CMS:KeptFromAGoodRun"]


def test_a_missing_root_leaves_the_old_index_alone(tmp_path):
    index = tmp_path / "pages.json"
    index.write_text(json.dumps(["twiki:CMS:KeptFromAGoodRun"]))
    run = TwikiEOSSource(
        eos_root=str(tmp_path / "absent"), page_index_path=str(index)
    ).run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert json.loads(index.read_text()) == ["twiki:CMS:KeptFromAGoodRun"]


# --- the index is not part of the content identity ---------------------------

def test_setting_the_path_does_not_change_the_content_hash(tmp_path):
    """Otherwise turning the index on would look like a corpus change
    and force a full re-ingest of an unchanged snapshot."""
    root = _snapshot(tmp_path)
    plain = TwikiEOSSource(eos_root=str(root)).run(
        "run-1", mode="scope_complete")
    indexed = TwikiEOSSource(
        eos_root=str(root), page_index_path=str(tmp_path / "pages.json")
    ).run("run-1", mode="scope_complete")
    assert plain.health.content_hash == indexed.health.content_hash
