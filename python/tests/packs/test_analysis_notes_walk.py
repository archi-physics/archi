"""The analysis-note walker, floor and bridge: edge cases (PACT r3-bundle-cms-physics).

Companion to test_analysis_notes.py. Each case is a synthetic note tree built
in `tmp_path`; the ids are invented.
"""
import hashlib

import pytest

from archi.sources import _an_parse
from archi.sources.analysis_notes import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    AnalysisNoteSource,
    an_node_id,
    section_node_id,
)
from archi.sources.twiki import TwikiEOSSource
from packs.physics_fixtures import (
    BRIDGE_PAGE_ID,
    LEGACY_NOTE,
    write_note_mirror,
    write_twiki_dump,
)


def _run(source, mode="scope_complete"):
    run = source.run("run-1", mode=mode)
    return run, list(run.facts)


def _of(facts, subtype):
    return [f for f in facts if getattr(f, "subtype", None) == subtype]


def _note(root, name, files, main=None):
    note = root / name
    note.mkdir(parents=True, exist_ok=True)
    (note / f"{name}.tex").write_text(
        main if main is not None else "\\title{Layout}\n", encoding="utf-8"
    )
    for rel, text in files.items():
        path = note / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return note


def _stems(facts, an_id):
    return {f.attrs["stem"] for f in _of(facts, "section") if f.attrs["an_id"] == an_id}


# --- the note floor ----------------------------------------------------------


@pytest.fixture
def mirror(tmp_path):
    return write_note_mirror(tmp_path / "mirror")


def test_a_mirror_below_the_floor_fails_preflight_and_emits_nothing(mirror):
    source = AnalysisNoteSource(an_root=str(mirror), min_notes=3)
    result = source.preflight()
    assert result.status == "cache_missing"
    assert "min_notes" in result.reason
    run = source.run("run-1", mode="scope_complete")
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert list(run.facts) == []
    assert AnalysisNoteSource(an_root=str(mirror), min_notes=2).preflight().status == "ok"


def test_the_floor_counts_directories_not_the_max_notes_bound(mirror):
    assert AnalysisNoteSource(an_root=str(mirror), min_notes=2, max_notes=1).preflight().status == "ok"


@pytest.mark.parametrize("bad", [-1, True, "2500", 1.5])
def test_a_floor_that_is_not_a_count_is_refused(mirror, bad):
    with pytest.raises(ValueError, match="min_notes"):
        AnalysisNoteSource(an_root=str(mirror), min_notes=bad)


# --- the bridge reads its page list from the dump ----------------------------


def test_the_bridge_reads_the_page_list_from_the_twiki_dump(mirror, tmp_path):
    dump = write_twiki_dump(tmp_path / "twiki")
    source = AnalysisNoteSource(an_root=str(mirror), twiki_eos_root=str(dump))
    _r, facts = _run(source)
    (edge,) = [f for f in facts if getattr(f, "attrs", {}).get("kind") == "bib_twiki_link"]
    assert (edge.src, edge.dst) == (an_node_id(LEGACY_NOTE), BRIDGE_PAGE_ID)
    assert source.last_walk_report["twiki_pages_known"] == 3


def test_the_list_is_exactly_what_the_twiki_source_emits(mirror, tmp_path):
    dump = write_twiki_dump(tmp_path / "twiki")
    for physics in (False, True):
        emitted = {
            f.node_id
            for f in TwikiEOSSource(eos_root=str(dump), physics_filter=physics)
            .run("r", mode="scope_complete").facts
            if getattr(f, "subtype", None) == "documentation_page"
        }
        source = AnalysisNoteSource(
            an_root=str(mirror), twiki_eos_root=str(dump), twiki_physics_filter=physics
        )
        assert source._twiki_pages() == emitted


def test_the_physics_filter_narrows_the_bridge_like_the_twiki_source(mirror, tmp_path):
    dump = write_twiki_dump(tmp_path / "twiki")
    source = AnalysisNoteSource(
        an_root=str(mirror), twiki_eos_root=str(dump), twiki_physics_filter=True
    )
    _r, facts = _run(source)
    # The bridge target is not a physics topic, so the filter drops it and no
    # edge may point at it.
    assert not [f for f in facts if getattr(f, "attrs", {}).get("kind") == "bib_twiki_link"]


def test_a_missing_dump_or_two_page_sources_fail_loudly(mirror, tmp_path):
    with pytest.raises(FileNotFoundError, match="twiki_eos_root"):
        AnalysisNoteSource(
            an_root=str(mirror), twiki_eos_root=str(tmp_path / "gone")
        ).run("run-1", mode="scope_complete")
    with pytest.raises(ValueError, match="not both"):
        AnalysisNoteSource(
            an_root=str(mirror), twiki_eos_root=str(tmp_path), twiki_pages_path="x.json"
        )


def test_the_probe_moves_when_the_dump_changes(mirror, tmp_path):
    dump = write_twiki_dump(tmp_path / "twiki")
    source = AnalysisNoteSource(an_root=str(mirror), twiki_eos_root=str(dump))
    before = dict(source._probe_content_items())["twiki_dump"]
    (dump / "NewTopic.txt").write_text("new", encoding="utf-8")
    assert dict(source._probe_content_items())["twiki_dump"] != before


@pytest.mark.parametrize("bad", ["ture", "true", 1, None])
def test_the_notes_source_refuses_a_physics_filter_that_is_not_a_boolean(mirror, bad):
    with pytest.raises(ValueError, match="true or false"):
        AnalysisNoteSource(an_root=str(mirror), twiki_physics_filter=bad)


@pytest.mark.parametrize("bad", ["ture", "true", 1])
def test_the_twiki_source_refuses_a_physics_filter_that_is_not_a_boolean(tmp_path, bad):
    with pytest.raises(ValueError, match="true or false"):
        TwikiEOSSource(eos_root=str(tmp_path), physics_filter=bad)


# --- layout and macro files ---------------------------------------------------


def test_unprefixed_files_are_sections_when_none_follows_a_convention(tmp_path):
    _note(tmp_path, "AN-2099-070", {"intro.tex": "i", "results.tex": "r"})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    assert _stems(facts, "AN-2099-070") == {"intro", "results"}


@pytest.mark.parametrize("defs", ["mydefs.tex", "my_defs.tex", "topic-defs.tex", "DEFS.tex"])
def test_macro_files_are_never_sections(tmp_path, defs):
    _note(tmp_path, "AN-2099-071", {defs: "\\newcommand{\\x}{y}", "01-Body.tex": "b"})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    assert _stems(facts, "AN-2099-071") == {"01-Body"}


def test_one_appendix_file_does_not_silence_the_unprefixed_files_beside_it(tmp_path):
    """Keeping only the structural files would hide every other file of the note."""
    _note(tmp_path, "AN-2099-072", {
        "App_pdf.tex": "a", "selection.tex": "s", "03-Results.tex": "r", "a_extra.tex": "e",
    })
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    assert _stems(facts, "AN-2099-072") == {"App_pdf", "selection", "03-Results", "a_extra"}


def test_structural_files_come_first_and_numbered_ones_carry_an_order(tmp_path):
    _note(tmp_path, "AN-2099-073", {"zzz.tex": "z", "02-B.tex": "b", "01-A.tex": "a"})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    sections = _of(facts, "section")
    assert [f.attrs["stem"] for f in sections] == ["01-A", "02-B", "zzz"]
    assert {f.attrs["stem"]: f.attrs.get("order") for f in sections} == {
        "01-A": 1, "02-B": 2, "zzz": None,
    }


def test_a_note_directory_with_no_tex_is_skipped(tmp_path):
    (tmp_path / "AN-2099-074").mkdir()
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    assert facts == []


# --- includes ----------------------------------------------------------------


def test_import_and_subimport_resolve_like_latex(tmp_path):
    note = _note(
        tmp_path, "AN-2099-075", {},
        main="\\import{parts/}{one}\n\\subimport{parts/}{two}\n",
    )
    (note / "parts").mkdir()
    (note / "parts" / "one.tex").write_text("one \\input{deeper}", encoding="utf-8")
    (note / "parts" / "deeper.tex").write_text("deeper", encoding="utf-8")
    (note / "parts" / "two.tex").write_text("two", encoding="utf-8")
    found = _an_parse.included_tex_files(note, note / "AN-2099-075.tex")
    assert [p.relative_to(note.resolve()).as_posix() for p in found] == [
        "parts/one.tex", "parts/deeper.tex", "parts/two.tex",
    ]
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    assert _stems(facts, "AN-2099-075") == {"parts/one", "parts/deeper", "parts/two"}


def test_an_include_loop_terminates(tmp_path):
    note = _note(tmp_path, "AN-2099-076", {}, main="\\input{a}")
    (note / "a.tex").write_text("\\input{b}", encoding="utf-8")
    (note / "b.tex").write_text("\\input{a}", encoding="utf-8")
    found = _an_parse.included_tex_files(note, note / "AN-2099-076.tex")
    assert sorted(p.name for p in found) == ["a.tex", "b.tex"]


# --- figures -----------------------------------------------------------------


def test_a_figure_without_a_label_gets_a_positional_id(tmp_path):
    figure = "\\begin{figure}\\caption{No label.}\\end{figure}\n"
    _note(tmp_path, "AN-2099-077", {"01-Body.tex": figure + figure})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    ids = sorted(f.node_id for f in _of(facts, "note_figure"))
    assert ids == ["an_figure:AN-2099-077:01-Body#1", "an_figure:AN-2099-077:01-Body#2"]


def test_a_label_reused_by_a_second_figure_falls_back_to_its_position(tmp_path):
    figure = "\\begin{figure}\\caption{C.}\\label{fig:same}\\end{figure}\n"
    _note(tmp_path, "AN-2099-078", {"01-Body.tex": figure + figure})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    ids = sorted(f.node_id for f in _of(facts, "note_figure"))
    assert ids == ["an_figure:AN-2099-078:01-Body#2", "an_figure:AN-2099-078:fig:same"]


def test_a_panel_outside_the_note_is_not_followed(tmp_path):
    mirror_root = tmp_path / "mirror"
    body = "\\begin{figure}\\includegraphics{../secret.png}\\caption{C.}\\label{fig:x}\\end{figure}"
    _note(mirror_root, "AN-2099-079", {"01-Body.tex": body})
    # The file exists beside the note, one level up: it is outside the note.
    (mirror_root / "secret.png").write_bytes(b"x")
    _r, facts = _run(AnalysisNoteSource(an_root=str(mirror_root)))
    (fig,) = _of(facts, "note_figure")
    assert fig.attrs["panels_missing"] == 1
    assert fig.attrs["images"] == []


def test_a_panel_without_an_extension_resolves_through_graphicspath(mirror):
    _r, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    (panel,) = [
        f for f in _of(facts, "note_figure") if f.attrs["an_id"] == LEGACY_NOTE
    ][0].attrs["panels"]
    assert panel["include"] == "masspeak" and panel["path"] == "figs/masspeak.png"
    assert panel["bytes"] > 0


# --- chunks ------------------------------------------------------------------


def test_chunks_cover_the_section_exactly_with_a_fixed_overlap(tmp_path):
    text = "".join(str(i % 10) for i in range(CHUNK_SIZE * 2 + 500))
    _note(tmp_path, "AN-2099-080", {"01-Body.tex": text})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    chunks = sorted(_of(facts, "document_chunk"), key=lambda f: f.attrs["char_offset"])
    assert [c.attrs["char_offset"] for c in chunks] == [0, CHUNK_SIZE - CHUNK_OVERLAP, 2 * (CHUNK_SIZE - CHUNK_OVERLAP)]
    for chunk in chunks:
        start = chunk.attrs["char_offset"]
        assert chunk.attrs["text"] == text[start:start + CHUNK_SIZE]
        assert chunk.attrs["char_end"] == start + len(chunk.attrs["text"])
    assert chunks[0].attrs["text"][-CHUNK_OVERLAP:] == chunks[1].attrs["text"][:CHUNK_OVERLAP]
    assert chunks[-1].attrs["char_end"] == len(text)


def test_a_chunk_id_hashes_section_index_and_text_and_not_the_span(tmp_path):
    _note(tmp_path, "AN-2099-081", {"01-Body.tex": "short body"})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    (chunk,) = _of(facts, "document_chunk")
    seed = f"{section_node_id('AN-2099-081', '01-Body')}\\0{0}\\0short body"
    assert chunk.node_id == "chunk:" + hashlib.sha256(seed.encode()).hexdigest()[:16]
    assert chunk.attrs["char_offset"] == 0 and chunk.attrs["char_end"] == len("short body")


def test_a_chunk_that_moves_in_the_section_keeps_its_id_only_if_text_and_index_agree(tmp_path):
    """The id has no offset in it: prepending text to a section changes the
    text of every window, so ids change with the content, not with a span."""
    body = "x" * (CHUNK_SIZE + 10)
    _note(tmp_path, "AN-2099-083", {"01-Body.tex": body})
    first = {f.node_id for f in _of(_run(AnalysisNoteSource(an_root=str(tmp_path)))[1], "document_chunk")}
    again = {f.node_id for f in _of(_run(AnalysisNoteSource(an_root=str(tmp_path)))[1], "document_chunk")}
    assert first == again and len(first) == 2


def test_whitespace_only_windows_make_no_chunk(tmp_path):
    _note(tmp_path, "AN-2099-082", {"01-Body.tex": "   \n  "})
    _r, facts = _run(AnalysisNoteSource(an_root=str(tmp_path)))
    assert _of(facts, "document_chunk") == []
