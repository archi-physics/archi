"""Every emitted document_chunk carries exact source positions.

``char_offset`` is where the stored chunk text starts in the text that was
chunked, and ``char_end`` is ``char_offset + len(text)``. A reader (okg's
whole-record text read) cuts the overlap two neighbouring chunks repeat only
when both carry such exact positions; before this change the offset pointed
at the unstripped window and ``char_end`` was missing, so a whole TWiki page
came back with every 200-character overlap twice.
"""

import json

import pytest

from okg.deployment import NodeFact

import archi.sources.docs as docs_mod
import archi.sources.indico as indico_mod
from archi.sources.docs import DocumentationSource
from archi.sources.hypernews import HyperNewsSource
from archi.sources.indico import IndicoSource
from archi.sources.jira import JiraIssueSource
from archi.sources.twiki import TwikiEOSSource

# Single spaces every 13 characters: 3800 is 4 mod 13, so the window starts
# walk through every phase and some windows start (and end) on a space that
# ``strip`` removes.
BODY = "abcdefghijkl " * 4400


def _old_chunks(text, pg_text):
    """The chunker before this change, kept to prove ids do not move."""
    step = 4000 - 200
    idx = 0
    offset = 0
    while offset < len(text):
        chunk = pg_text(text[offset:offset + 4000].strip())
        if chunk:
            yield idx, offset, chunk
            idx += 1
        if offset + 4000 >= len(text):
            break
        offset += step


def _stitch(chunks):
    """Join chunks the way okg's text read does: cut a proven overlap only.

    Mirrors okg ``record_text._exact`` + ``trim_overlap``: both neighbours
    exact, offsets increasing and overlapping, and the previous text really
    ends with the characters cut. Fails the test if any pair is not exact.
    """
    out = chunks[0]["text"]
    for prev, cur in zip(chunks, chunks[1:]):
        for c in (prev, cur):
            assert c["char_end"] - c["char_offset"] == len(c["text"])
        assert prev["char_offset"] < cur["char_offset"] < prev["char_end"]
        overlap = prev["char_end"] - cur["char_offset"]
        assert prev["text"].endswith(cur["text"][:overlap])
        out += cur["text"][overlap:]
    return out


@pytest.mark.parametrize("chunker", [docs_mod._chunks, indico_mod._chunks])
def test_chunk_start_is_where_the_stored_text_starts(chunker):
    # Whitespace runs land on window starts (3800, 7600) and ends.
    text = (
        "x" * 3790 + " " * 30 + "y" * 3770 + "\n\t " * 20 + "z" * 5000 + "  "
    )
    got = list(chunker(text))
    assert len(got) == 4
    chunks = []
    for idx, start, chunk in got:
        assert text[start:start + len(chunk)] == chunk
        chunks.append(
            {"text": chunk, "char_offset": start, "char_end": start + len(chunk)}
        )
    assert [c["char_offset"] for c in chunks] != [0, 3800, 7600, 11400]
    assert _stitch(chunks) == text.strip()
    # Index and text (so chunk ids and content hashes) are unchanged.
    old = [(i, c) for i, _o, c in _old_chunks(text, docs_mod._pg_text)]
    assert [(i, c) for i, _s, c in got] == old


def test_nul_keeps_length_so_positions_stay_exact():
    text = "a\x00b " * 3000
    for idx, start, chunk in docs_mod._chunks(text):
        assert "\x00" not in chunk
        assert len(text[start:start + len(chunk)]) == len(chunk)
        assert text[start:start + len(chunk)].replace("\x00", " ") == chunk


def _chunk_attrs(facts):
    chunks = [
        f for f in facts
        if isinstance(f, NodeFact) and f.subtype == "document_chunk"
    ]
    chunks.sort(key=lambda f: f.source_record_id["chunk_index"])
    return [f.attrs for f in chunks]


def _assert_exact(chunks):
    assert len(chunks) > 10
    for c in chunks:
        assert c["char_end"] == c["char_offset"] + len(c["text"])
        assert c["char_length"] == len(c["text"])
    # The fixture really hits the strip case.
    assert any(c["char_offset"] % 3800 for c in chunks)
    return _stitch(chunks)


def test_docs_chunks_are_exact(tmp_path):
    (tmp_path / "data" / "docsite").mkdir(parents=True)
    (tmp_path / "data" / "docsite" / "records.json").write_text(json.dumps([{
        "url": "https://docs.example.cern.ch/long/",
        "title": "Long",
        "body": BODY,
        "site_name": "docs.example.cern.ch",
    }]))
    source = DocumentationSource(base=str(tmp_path))
    chunks = _chunk_attrs(source.run("r").facts)
    assert _assert_exact(chunks) == BODY.strip()


def test_hypernews_chunks_are_exact(tmp_path):
    root = tmp_path / "data" / "hypernews"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps([{
        "thread_id": "comp-ops/9",
        "title": "Long thread",
        "url": "https://hypernews.cern.ch/HyperNews/CMS/get/comp-ops/9.html",
        "forum_name": "comp-ops",
        "body": BODY,
        "author": "Ada Lovelace",
        "date": "2026-08-01",
        "reply_count": 0,
    }]))
    source = HyperNewsSource(base=str(tmp_path))
    chunks = _chunk_attrs(source.run("r", mode="scope_complete").facts)
    assert _assert_exact(chunks) == ("Long thread\n\n" + BODY).strip()


def test_indico_chunks_are_exact(tmp_path):
    root = tmp_path / "data" / "indico"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps([{
        "id": "100", "title": "A", "_pdf_texts": [{"text": BODY}],
    }]))
    source = IndicoSource(base=str(tmp_path))
    chunks = _chunk_attrs(source.run("r", mode="scope_complete").facts)
    assert _assert_exact(chunks) == BODY.strip()


def test_jira_chunks_are_exact(tmp_path):
    root = tmp_path / "data" / "jira"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps([{
        "key": "CMSPROD-7",
        "summary": "Long issue",
        "description": BODY,
        "project": "CMSPROD",
        "status": "Open",
    }]))
    (root / "meta.json").write_text(json.dumps({"record_count": 1}))
    source = JiraIssueSource(
        records_path="data/jira/records.json",
        meta_path="data/jira/meta.json",
        project_keys=["CMSPROD"],
        base=str(tmp_path),
    )
    chunks = _chunk_attrs(source.run("r", mode="cursor").facts)
    text = _assert_exact(chunks)
    assert text.startswith("summary: Long issue\n\ndescription: abcdefghijkl ")
    assert text.endswith("abcdefghijkl \n\nstatus: Open")


def test_twiki_chunks_rebuild_the_page(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "LongTopic.txt").write_text(
        '%META:TOPICINFO{author="JohnDoe" date="1700000000" version="2"}%\n'
        + BODY
        + "\n"
    )
    source = TwikiEOSSource(eos_root=str(root))
    chunks = _chunk_attrs(source.run("r").facts)
    text = _assert_exact(chunks)
    assert text.endswith("abcdefghijkl")
    assert len(text) == chunks[-1]["char_end"] - chunks[0]["char_offset"]
