"""The analysis-notes pack (PACT r3-bundle-cms-physics, acc.r3-bundle-cms-physics.notes).

Fixture note trees yield notes, sections, figures and references, and the
bibliography-to-TWiki bridge. Every input is invented (see physics_fixtures).
"""
import ast
import inspect
import json
from pathlib import Path

import pytest
from packs.pack_support import PACKS_DIR, invariants, manifest, templates
from packs.physics_fixtures import (
    BRIDGE_PAGE_ID,
    CITED_CODE,
    LEGACY_NOTE,
    PAG_NOTE,
    write_note_mirror,
)

from archi.sources import _an_parse
from archi.sources.analysis_notes import (
    AnalysisNoteAdapter,
    AnalysisNoteSource,
    an_node_id,
    figure_node_id,
    pag_hub_id,
    reference_node_id,
    section_node_id,
)


@pytest.fixture
def mirror(tmp_path):
    return write_note_mirror(tmp_path / "mirror")


def _run(source, mode="scope_complete"):
    run = source.run("run-1", mode=mode)
    return run, list(run.facts)


def _nodes(facts, subtype):
    return {f.node_id: f for f in facts if getattr(f, "subtype", None) == subtype}


def _edges(facts, edge_type, kind=None):
    return [
        f for f in facts
        if hasattr(f, "edge_type") and f.edge_type == edge_type
        and (kind is None or f.attrs.get("kind") == kind)
    ]


# --- notes ---------------------------------------------------------------


def test_a_mirror_yields_one_note_per_note_directory(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    notes = _nodes(facts, "analysis_note")
    assert set(notes) == {an_node_id(LEGACY_NOTE), an_node_id(PAG_NOTE)}
    legacy = notes[an_node_id(LEGACY_NOTE)].attrs
    assert legacy["directory_format"] == "legacy_an"
    assert legacy["title"] == "Synthetic search for a fictional resonance"
    assert "pag_code" not in legacy
    pag = notes[an_node_id(PAG_NOTE)].attrs
    assert pag["directory_format"] == "pag_coded"
    assert pag["pag_code"] == "HIG"


def test_a_directory_that_is_not_a_note_is_ignored(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    assert not any("README" in f.node_id for f in facts if hasattr(f, "node_id"))


def test_max_notes_bounds_the_walk(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror), max_notes=1))
    assert len(_nodes(facts, "analysis_note")) == 1
    with pytest.raises(ValueError):
        AnalysisNoteSource(an_root=str(mirror), max_notes=0)


# --- sections ------------------------------------------------------------


def test_sections_are_the_prose_files_and_the_included_chapters(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    sections = _nodes(facts, "section")
    legacy = {
        node_id for node_id, f in sections.items() if f.attrs["an_id"] == LEGACY_NOTE
    }
    # mydefs.tex is a macros file and the main tex is not a section of a
    # note that has section files; chapters/results.tex is reached by \input.
    assert legacy == {
        section_node_id(LEGACY_NOTE, "01-Introduction"),
        section_node_id(LEGACY_NOTE, "02-Selection"),
        section_node_id(LEGACY_NOTE, "chapters/results"),
    }
    intro = sections[section_node_id(LEGACY_NOTE, "01-Introduction")]
    assert intro.attrs["order"] == 1
    assert "b tagging" in intro.attrs["body"]
    note = _nodes(facts, "analysis_note")[an_node_id(LEGACY_NOTE)]
    assert note.attrs["n_sections"] == 3
    contained = {e.dst for e in _edges(facts, "contains") if e.src == note.node_id}
    assert legacy <= contained


def test_a_note_with_no_section_files_is_one_section(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    sections = _nodes(facts, "section")
    assert section_node_id(PAG_NOTE, PAG_NOTE) in sections
    only = [f for f in sections.values() if f.attrs["an_id"] == PAG_NOTE]
    assert len(only) == 1
    assert "toy W boson" in only[0].attrs["body"]


def test_sections_are_chunked_and_chunks_belong_to_their_section(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    chunks = _nodes(facts, "document_chunk")
    assert chunks
    section_ids = set(_nodes(facts, "section"))
    parents = {e.src for e in _edges(facts, "contains") if e.dst in chunks}
    assert parents <= section_ids
    for chunk in chunks.values():
        assert chunk.attrs["char_end"] - chunk.attrs["char_offset"] == len(chunk.attrs["text"])


def test_a_very_long_section_is_cut_at_the_body_cap(tmp_path):
    root = tmp_path / "mirror"
    note = root / "AN-2099-050"
    note.mkdir(parents=True)
    (note / "AN-2099-050.tex").write_text("\\title{Long}\n", encoding="utf-8")
    (note / "01-Body.tex").write_text("x" * (_an_parse.BODY_CAP + 500), encoding="utf-8")
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(root)))
    (section,) = _nodes(facts, "section").values()
    assert section.attrs["char_length"] == _an_parse.BODY_CAP


# --- figures -------------------------------------------------------------


def test_a_figure_has_its_caption_label_and_panel_on_disk(mirror):
    run, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    figures = _nodes(facts, "note_figure")
    fig = figures[figure_node_id(LEGACY_NOTE, "fig:massplot")]
    assert fig.attrs["caption"] == "Fictional dijet mass spectrum with a toy signal."
    assert fig.attrs["latex_label"] == "fig:massplot"
    assert fig.attrs["panel_count"] == 1
    assert fig.attrs["panels_missing"] == 0
    (panel,) = fig.attrs["panels"]
    assert panel["path"] == "figs/masspeak.png"
    assert panel["format"] == "png" and panel["exists"] is True
    # The image is named relative to the mirror root, for okg's ingest to render.
    assert fig.attrs["images"] == [
        {"okg_image": "file", "path": f"{LEGACY_NOTE}/figs/masspeak.png"}
    ]
    assert run.image_root == str(mirror)
    host = [e.src for e in _edges(facts, "contains") if e.dst == fig.node_id]
    assert host == [section_node_id(LEGACY_NOTE, "chapters/results")]


def test_a_figure_whose_image_is_not_in_the_mirror_is_counted_not_dropped(mirror):
    source = AnalysisNoteSource(an_root=str(mirror))
    _run_result, facts = _run(source)
    fig = _nodes(facts, "note_figure")[figure_node_id(PAG_NOTE, "fig:missing")]
    assert fig.attrs["panel_count"] == 1
    assert fig.attrs["panels_missing"] == 1
    assert fig.attrs["images"] == []
    assert source.last_walk_report["figure_panels_missing"] == 1
    assert source.last_walk_report["figures"] == 2


def test_prose_that_refs_a_figure_label_is_linked_to_the_figure(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    fig_id = figure_node_id(LEGACY_NOTE, "fig:massplot")
    chunks = _nodes(facts, "document_chunk")
    linked = [e for e in _edges(facts, "references", "latex_ref") if e.dst == fig_id]
    assert len(linked) == 1
    assert linked[0].src in chunks
    assert "Figure~\\ref{fig:massplot}" in chunks[linked[0].src].attrs["text"]


# --- references and the bibliography bridge --------------------------------


def test_bibliography_entries_become_references_with_their_analysis_code(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    refs = _nodes(facts, "reference")
    assert len(refs) == 4
    by_key = {f.attrs["bib_key"]: f for f in refs.values()}
    cited = by_key["synth_search"]
    assert cited.node_id == reference_node_id("synth_search")
    assert cited.attrs["title"] == "A fictional resonance search"
    assert cited.attrs["report_numbers"] == ["CMS-TOP-99-777", "CERN-EP-2099-001"]
    assert cited.attrs["analysis_code"] == CITED_CODE
    # A conference record names no analysis, so it must not join to one.
    assert by_key["synth_conference"].attrs["analysis_code"] == ""
    cites = _edges(facts, "references", "cites")
    assert {e.dst for e in cites} == set(refs)
    assert {e.src for e in cites} == {an_node_id(LEGACY_NOTE)}
    note = _nodes(facts, "analysis_note")[an_node_id(LEGACY_NOTE)]
    assert note.attrs["n_references"] == 4


def test_a_reference_shared_by_two_notes_is_one_node(mirror):
    shared = mirror / "AN-2099-002"
    shared.mkdir()
    (shared / "AN-2099-002.tex").write_text("\\title{Second}\n", encoding="utf-8")
    (shared / "01-Intro.tex").write_text("Text.", encoding="utf-8")
    (shared / "AN-2099-002.bib").write_text(
        (mirror / LEGACY_NOTE / f"{LEGACY_NOTE}.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    refs = [f for f in facts if getattr(f, "subtype", None) == "reference"]
    assert len(refs) == 4
    assert len({f.node_id for f in refs}) == 4
    assert {e.src for e in _edges(facts, "references", "cites")} == {
        an_node_id(LEGACY_NOTE), an_node_id("AN-2099-002"),
    }


def test_the_pag_coded_note_references_its_pag_hub(mirror):
    _run_result, facts = _run(AnalysisNoteSource(an_root=str(mirror)))
    hubs = _nodes(facts, "physics_analysis_group")
    assert set(hubs) == {pag_hub_id("HIG")}
    (edge,) = _edges(facts, "references", "belongs_to_pag")
    assert (edge.src, edge.dst) == (an_node_id(PAG_NOTE), pag_hub_id("HIG"))


def _pages_file(tmp_path, page_ids):
    path = tmp_path / "twiki-page-index" / "pages.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(page_ids), encoding="utf-8")
    return path


def test_the_bridge_links_a_note_to_a_twiki_page_that_was_ingested(mirror, tmp_path):
    pages = _pages_file(tmp_path, [BRIDGE_PAGE_ID, "twiki:CMS:SomethingElse"])
    source = AnalysisNoteSource(an_root=str(mirror), twiki_pages_path=str(pages))
    _run_result, facts = _run(source)
    (edge,) = _edges(facts, "references", "bib_twiki_link")
    assert (edge.src, edge.dst) == (an_node_id(LEGACY_NOTE), BRIDGE_PAGE_ID)
    report = source.last_walk_report
    assert report["bridge_candidates"] == 2
    assert report["bridge_emitted"] == 1
    # The second URL names a page the TWiki ingest did not keep: no edge to
    # a node that is not in the graph.
    assert report["bridge_dropped_unknown_page"] == 1


def test_without_a_page_list_the_bridge_emits_nothing_and_says_so(mirror):
    source = AnalysisNoteSource(an_root=str(mirror))
    _run_result, facts = _run(source)
    assert _edges(facts, "references", "bib_twiki_link") == []
    assert source.last_walk_report["bridge_candidates"] == 2
    assert source.last_walk_report["bridge_emitted"] == 0
    assert source.last_walk_report["twiki_pages_known"] == 0


def test_a_configured_page_list_that_is_missing_fails_loudly(mirror, tmp_path):
    source = AnalysisNoteSource(
        an_root=str(mirror), twiki_pages_path=str(tmp_path / "absent.json")
    )
    with pytest.raises(FileNotFoundError, match="twiki_pages_path"):
        source.run("run-1", mode="scope_complete")


def test_a_page_list_that_is_not_a_list_is_refused(mirror, tmp_path):
    path = tmp_path / "pages.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        AnalysisNoteSource(
            an_root=str(mirror), twiki_pages_path=str(path)
        ).run("run-1", mode="scope_complete")


def test_the_bridge_mints_the_node_id_the_twiki_source_mints():
    from archi.sources._twiki_parse import twiki_node_id as twiki_source_node_id

    assert _an_parse.twiki_node_id("CMS", "TopicTwo") == twiki_source_node_id("CMS/TopicTwo")


# --- parsing helpers -------------------------------------------------------


def test_include_resolution_stays_inside_the_note(tmp_path):
    note = tmp_path / "AN-2099-060"
    note.mkdir()
    outside = tmp_path / "outside.tex"
    outside.write_text("secret", encoding="utf-8")
    (note / "AN-2099-060.tex").write_text(
        "\\input{../outside}\n\\input{sub/inside}\n", encoding="utf-8"
    )
    (note / "sub").mkdir()
    (note / "sub" / "inside.tex").write_text("inside", encoding="utf-8")
    found = _an_parse.included_tex_files(note, note / "AN-2099-060.tex")
    assert [p.name for p in found] == ["inside.tex"]


@pytest.mark.parametrize(
    "name,kind",
    [
        ("HIG-99-001", "pag_coded"),
        ("AN-2099-001", "legacy_an"),
        ("AN-99-001", "legacy_an"),
        ("ABC-99-001", None),
        ("HIG-99-1", None),
        ("notes", None),
    ],
)
def test_directory_names(name, kind):
    assert _an_parse.classify_an_directory(name) == kind


# --- lifecycle -------------------------------------------------------------


def test_preflight_counts_the_notes_in_the_mirror(mirror):
    result = AnalysisNoteSource(an_root=str(mirror)).preflight()
    assert result.status == "ok"
    assert result.record_count == 2


@pytest.mark.parametrize("root", ["", "/nonexistent/analysis-notes-mirror"])
def test_a_missing_mirror_fails_preflight_and_run_emits_nothing(root):
    source = AnalysisNoteSource(an_root=root)
    result = source.preflight()
    assert result.status == "cache_missing"
    assert result.required is True
    assert "tdr/notes" in result.reason
    run = source.run("run-1", mode="scope_complete")
    assert run.health.status == "cache_missing"
    assert run.completed_scope is False
    assert list(run.facts) == []


def test_only_a_complete_run_mode_claims_the_scope(mirror):
    source = AnalysisNoteSource(an_root=str(mirror))
    assert source.run("run-1", mode="scope_complete").completed_scope is True
    assert source.run("run-1", mode="cursor").completed_scope is False


def test_the_content_hash_moves_when_a_section_is_edited(mirror):
    before = AnalysisNoteSource(an_root=str(mirror)).run("r", mode="scope_complete").health.content_hash
    (mirror / LEGACY_NOTE / "02-Selection.tex").write_text("\\section{Selection}\nChanged.", encoding="utf-8")
    after = AnalysisNoteSource(an_root=str(mirror)).run("r", mode="scope_complete").health.content_hash
    assert before != after


def test_the_probe_reads_every_file_the_walk_reads(mirror, tmp_path):
    pages = _pages_file(tmp_path, [BRIDGE_PAGE_ID])
    source = AnalysisNoteSource(an_root=str(mirror), twiki_pages_path=str(pages))
    keys = {key for key, _item in source._probe_content_items()}
    assert f"{LEGACY_NOTE}/chapters/results.tex" in keys
    assert f"{LEGACY_NOTE}/{LEGACY_NOTE}.bib" in keys
    assert f"{PAG_NOTE}/{PAG_NOTE}.tex" in keys
    assert "twiki_pages" in keys


# --- adapter ---------------------------------------------------------------


def test_adapter_binds_and_rejects_typos(mirror):
    adapter = AnalysisNoteAdapter(an_root=str(mirror), required=False)
    assert adapter.reader.an_root == str(mirror)
    with pytest.raises(TypeError):
        AnalysisNoteAdapter(an_roots=str(mirror))


def test_adapter_declares_literal_profile_and_probe_kind():
    tree = ast.parse(inspect.getsource(AnalysisNoteAdapter))
    literals = {
        node.targets[0].id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert literals["profile"] == AnalysisNoteSource.profile
    assert literals["change_probe_kind"] == AnalysisNoteSource.change_probe_kind


# --- the pack ----------------------------------------------------------------


def test_the_template_binds_to_the_adapter_and_follows_the_twiki_source():
    entry = templates("analysis-notes")["analysis_notes"]
    params = set(entry["params"])
    assert params <= set(inspect.signature(AnalysisNoteSource).parameters)
    twiki = templates("twiki")["twiki_eos"]
    # The bridge's page list comes from the same dump, web and filter the
    # twiki source reads, and the notes source is ordered after it.
    assert entry["params"]["twiki_eos_root"] == "${twiki_eos_root}"
    assert twiki["params"]["eos_root"] == "${eos_root}"
    assert entry["params"]["twiki_web_root"] == twiki["params"]["web_root"]
    assert entry["params"]["twiki_physics_filter"] == "${twiki_physics_filter}"
    assert twiki["params"]["physics_filter"] == "${physics_filter}"
    assert "page_index_path" not in twiki["params"]
    assert entry["params"]["min_notes"] == "${min_notes}"
    assert entry["predecessor_source"] == twiki["ownership_id"]
    assert entry["admission_policy"]["trust_label"] == "untrusted"
    assert entry["params"]["required"] is True


def test_the_pack_asks_for_the_mirror_and_says_it_needs_credentials():
    pack = manifest("analysis-notes")
    assert pack["needs_credentials"] is True
    assert pack["credentials"] == ["CERN_GITLAB_TOKEN"]
    questions = (PACKS_DIR / "analysis-notes" / "questions.yaml").read_text(encoding="utf-8")
    assert "id: mirror_root" in questions
    assert "required: true" in questions


def test_the_readme_documents_the_mirror_fetch():
    readme = (PACKS_DIR / "analysis-notes" / "README.md").read_text(encoding="utf-8")
    assert "gitlab.cern.ch/tdr/notes" in readme
    assert "CERN_GITLAB_TOKEN" in readme
    assert "git" in readme and "clone" in readme
    # The token goes in a header, never in a URL or a file.
    assert "oauth2:" not in readme


def test_the_coverage_invariant_does_not_require_a_bibliography():
    (invariant,) = invariants("analysis-notes")
    assert invariant["name"] == "analysis_notes_coverage"
    assert "'analysis_note'" in invariant["sql"] and "'section'" in invariant["sql"]
    # A mirror with no .bib files must still publish.
    assert "'reference'" not in invariant["sql"]


def test_the_note_floor_is_an_install_question_the_bundle_raises():
    import yaml

    questions = yaml.safe_load(
        (PACKS_DIR / "analysis-notes" / "questions.yaml").read_text(encoding="utf-8")
    )["init_questions"]
    (floor,) = [q for q in questions if q["id"] == "min_notes"]
    assert floor["type"] == "integer" and floor["default"] == 1
    bundle = yaml.safe_load(
        (PACKS_DIR.parent / "cms-physics" / "bundle.yaml").read_text(encoding="utf-8")
    )
    assert bundle["answers"]["analysis_notes_min_notes"] == 2500


def test_the_bundle_answers_reach_the_rendered_questions():
    from archi.bundles.compose import compose
    from archi.bundles.resolve import resolve
    from packs.pack_support import shipped_catalog

    comp = compose(resolve(shipped_catalog(), ["cms-physics"]))
    defaults = {q["id"]: q.get("default") for q in comp.questions}
    assert defaults["analysis_notes_min_notes"] == 2500
    assert defaults["twiki_physics_filter"] == "true"


def test_a_bundle_answer_no_pack_asks_is_refused(tmp_path):
    from archi.bundles.compose import compose
    from archi.bundles.resolve import CompositionError, resolve
    from packs.pack_support import shipped_catalog

    catalog = shipped_catalog()
    bundle = catalog.bundles["cms-physics"]
    object.__setattr__(
        bundle, "manifest", {**bundle.manifest, "answers": {"nobody_asks_this": 1}}
    )
    with pytest.raises(CompositionError, match="no selected pack asks"):
        compose(resolve(catalog, ["cms-physics"]))
