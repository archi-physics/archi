"""The cms-analyses pack (PACT r3-bundle-cms-physics, acc.r3-bundle-cms-physics.links).

A fixture corpus - invented notes, an invented TWiki dump, invented CDS
records - goes through the pack's real sources and its real cross-link rules,
and yields linked analysis records. The rules run in okg's own
``DeclarativeLinker`` matchers against an in-memory graph, so a rule that
stops matching its sources' attribute names fails here, not at publish.
"""
import pathlib
import re
from collections import defaultdict

import pytest
import yaml
from packs.pack_support import PACKS_DIR, invariants, manifest, templates
from packs.physics_fixtures import (
    CITED_CODE,
    LEGACY_NOTE,
    PAG_NOTE,
    synthetic_cds_records,
    write_note_mirror,
    write_twiki_dump,
)

from archi.enrichment.physics_cross_links import (
    PhysicsCrossLinker,
    cross_link_rules_files,
)
from archi.sources import _an_parse
from archi.sources import _physics_entities_vocabulary as vocabulary
from archi.sources.analysis_notes import (
    AnalysisNoteSource,
    an_node_id,
    reference_node_id,
)
from archi.sources.cds import CDSSource
from archi.sources.physics_entities import PhysicsEntityAdapter, PhysicsEntitySource
from archi.sources.physics_vocabulary import (
    PAGS,
    POGS,
    PhysicsVocabularyAdapter,
    PhysicsVocabularySource,
    pag_hub_id,
    pog_hub_id,
)
from archi.sources.twiki import TwikiEOSSource


@pytest.fixture
def corpus(tmp_path):
    return {
        "mirror": write_note_mirror(tmp_path / "mirror"),
        "twiki": write_twiki_dump(tmp_path / "twiki"),
        "cache": tmp_path / "matches.json",
    }


def _facts(run):
    return list(run.facts)


# --- the vocabulary ----------------------------------------------------------


def test_the_vocabulary_declares_every_pag_and_pog():
    run = PhysicsVocabularySource().run("run-1", mode="release_new")
    facts = _facts(run)
    pags = {f.node_id for f in facts if f.subtype == "physics_analysis_group"}
    pogs = {f.node_id for f in facts if f.subtype == "physics_object_group"}
    assert len(pags) == 17 and len(pogs) == 8
    assert pag_hub_id("HIG") in pags and pog_hub_id("MUON") in pogs
    assert run.completed_scope is True
    assert PhysicsVocabularySource().run("run-1", mode="cursor").completed_scope is False


def test_the_pag_codes_are_the_codes_the_note_walker_and_the_rules_read():
    codes = {code for code, _name in PAGS}
    assert codes == set(_an_parse.PAG_CODES)
    rule = _rules()["cms_analyses_documentation_page_in_pag"]
    assert set(re.match(r"\((.*?)\)", rule["dst"]["derive_from_path"]["regex"]).group(1).split("|")) == codes


def test_a_pog_that_shares_a_group_with_a_pag_carries_its_code():
    paired = {code: pag for code, _name, _roots, pag in POGS if pag}
    assert paired == {
        "MUON": "MUO", "EGAMMA": "EGM", "JETMET": "JME",
        "BTAG": "BTV", "TAUPOG": "TAU", "TRACKING": "TRK",
    }
    assert set(paired.values()) <= {code for code, _name in PAGS}
    facts = {f.node_id: f for f in _facts(PhysicsVocabularySource().run("r", mode="release_new"))}
    btag = facts[pog_hub_id("BTAG")].attrs
    # Scalars as well as the list: the declarative matcher reads a scalar.
    assert (btag["title_root"], btag["title_root_alt"], btag["pag_code"]) == ("BTag", "BTV", "BTV")
    assert "pag_code" not in facts[pog_hub_id("TRIGGER")].attrs


def test_the_vocabulary_adapter_declares_its_profile_and_binds():
    assert PhysicsVocabularyAdapter.profile == "reference_catalog"
    assert PhysicsVocabularyAdapter(required=True).reader.required is True
    with pytest.raises(TypeError):
        PhysicsVocabularyAdapter(requird=True)


# --- the entity vocabulary ---------------------------------------------------


def test_the_entity_vocabulary_is_well_formed():
    assert [len(vocabulary.TECHNIQUES), len(vocabulary.CONCEPTS), len(vocabulary.TOPICS)] == [81, 58, 41]
    assert len(vocabulary.ALL) == 180
    codes = [(kind, code) for kind, code, _label, _aliases in vocabulary.ALL]
    assert len(set(codes)) == 180
    for kind, code, label, aliases in vocabulary.ALL:
        assert code and label and aliases, (kind, code)
        for alias in aliases:
            re.compile(alias)
    assert vocabulary.entity_node_id("technique", "b_tagging") == "physics_technique:b_tagging"


@pytest.mark.parametrize(
    "code,text",
    [
        ("b_tagging", "events are selected with b-tagging"),
        ("b_tagging", "the b jet identification working point"),
        ("tag_and_probe", "efficiencies from the tag and probe method"),
        ("higgs_boson", "A Higgs boson candidate"),
        ("w_boson", "the W boson mass"),
        ("jet_energy_corrections", "the jet energy scale is applied"),
    ],
)
def test_an_alias_matches_the_text_physicists_write(code, text):
    aliases = next(a for _kind, c, _label, a in vocabulary.ALL if c == code)
    assert any(re.search(alias, text, re.I) for alias in aliases)


def test_an_alias_does_not_match_a_lookalike():
    aliases = next(a for _kind, c, _label, a in vocabulary.ALL if c == "b_tagging")
    assert not any(re.search(alias, "the tagging of photographs", re.I) for alias in aliases)


# --- the entity source, on the fixture corpus ---------------------------------


def _entities(corpus, **kwargs):
    return PhysicsEntitySource(
        eos_root=str(corpus["twiki"]), an_root=str(corpus["mirror"]),
        physics_filter=False, **kwargs,
    )


def _entity_edges(facts):
    out = defaultdict(set)
    for f in facts:
        if hasattr(f, "edge_type"):
            out[f.src].add(f.dst.split(":", 1)[1])
    return out


def test_documents_are_linked_to_the_entities_their_text_names(corpus):
    source = _entities(corpus)
    run = source.run("run-1", mode="release_new")
    facts = _facts(run)
    nodes = {f.node_id: f for f in facts if hasattr(f, "subtype")}
    assert len(nodes) == 180
    edges = _entity_edges(facts)
    # The legacy note's prose names b tagging, tag and probe and a Higgs boson.
    assert {"b_tagging", "tag_and_probe", "higgs_boson"} <= edges[an_node_id(LEGACY_NOTE)]
    # The PAG-coded note names a W boson and jet energy corrections, not b tagging.
    assert {"w_boson", "jet_energy_corrections"} <= edges[an_node_id(PAG_NOTE)]
    assert "b_tagging" not in edges[an_node_id(PAG_NOTE)]
    # A TWiki page is a document too.
    assert {"b_tagging", "tag_and_probe"} <= edges["twiki:CMS:HIG99001Review"]
    assert source.last_link_report["notes"] == 2
    assert source.last_link_report["pages"] == 3


def test_an_entity_counts_the_documents_that_reach_it(corpus):
    run = _entities(corpus).run("run-1", mode="release_new")
    nodes = {f.node_id: f for f in _facts(run) if hasattr(f, "subtype")}
    b_tagging = nodes["physics_technique:b_tagging"].attrs
    assert (b_tagging["notes_reached"], b_tagging["pages_reached"]) == (1, 1)
    assert b_tagging["documents_reached"] == 2
    assert b_tagging["selectivity"] == "rare"
    assert nodes["physics_concept:neutrino"].attrs["documents_reached"] == 0


def test_the_match_cache_reproduces_the_edges_without_rematching(corpus):
    cold = _entities(corpus, match_cache_path=str(corpus["cache"]))
    cold_facts = _facts(cold.run("run-1", mode="release_new"))
    assert cold.last_link_report["matched"] == 5 and cold.last_link_report["reused"] == 0
    warm = _entities(corpus, match_cache_path=str(corpus["cache"]))
    warm_facts = _facts(warm.run("run-2", mode="release_new"))
    assert warm.last_link_report["matched"] == 0 and warm.last_link_report["reused"] == 5
    assert _entity_edges(cold_facts) == _entity_edges(warm_facts)


def test_an_edited_document_is_rematched(corpus):
    _facts(_entities(corpus, match_cache_path=str(corpus["cache"])).run("r", mode="release_new"))
    (corpus["twiki"] / "MuonPOGRecipes.txt").write_text("---+ Muon\nNow it names unfolding.\n", encoding="utf-8")
    again = _entities(corpus, match_cache_path=str(corpus["cache"]))
    facts = _facts(again.run("r2", mode="release_new"))
    assert again.last_link_report["matched"] == 1
    assert "unfolding" in _entity_edges(facts)["twiki:CMS:MuonPOGRecipes"]


def test_the_scope_is_claimed_only_while_the_mirrors_are_mounted(corpus, tmp_path):
    assert _entities(corpus).run("r", mode="release_new").completed_scope is True
    unmounted = PhysicsEntitySource(
        eos_root=str(tmp_path / "gone"), an_root=str(corpus["mirror"]), physics_filter=False
    )
    assert unmounted.run("r", mode="release_new").completed_scope is False


def test_the_entity_edges_point_at_nodes_the_other_sources_emit(corpus):
    twiki_pages = {
        f.node_id for f in TwikiEOSSource(eos_root=str(corpus["twiki"])).run("r", mode="scope_complete").facts
        if getattr(f, "subtype", None) == "documentation_page"
    }
    notes = {
        f.node_id for f in AnalysisNoteSource(an_root=str(corpus["mirror"])).run("r", mode="scope_complete").facts
        if getattr(f, "subtype", None) == "analysis_note"
    }
    sources = {src for src in _entity_edges(_facts(_entities(corpus).run("r", mode="release_new")))}
    assert sources <= twiki_pages | notes


def test_the_entity_adapter_binds_and_rejects_typos(corpus):
    assert PhysicsEntityAdapter.profile == "reference_catalog"
    adapter = PhysicsEntityAdapter(an_root=str(corpus["mirror"]), physics_filter=False)
    assert adapter.reader.physics_filter is False
    with pytest.raises(TypeError):
        PhysicsEntityAdapter(an_roots=str(corpus["mirror"]))


# --- the cross-link rules -----------------------------------------------------


def _rules():
    out = {}
    for path in cross_link_rules_files():
        for rule in yaml.safe_load(open(path, encoding="utf-8"))["rules"]:
            assert rule["id"] not in out
            out[rule["id"]] = rule
    return out


def test_the_pack_ships_the_rules_the_linker_loads():
    shipped = sorted(p.name for p in (PACKS_DIR / "cms-analyses" / "cross_links").glob("*.yaml"))
    assert shipped == [
        "bibliography_bridge.yaml", "cds_analysis_notes.yaml",
        "pag_pog_bridge.yaml", "twiki_physics_groups.yaml",
    ]
    rules_dirs = {pathlib.Path(path).parent for path in cross_link_rules_files()}
    assert rules_dirs == {(PACKS_DIR / "cms-analyses" / "cross_links").resolve()}
    assert set(_rules()) == {
        "cms_analyses_reference_names_cds_record",
        "cms_analyses_cds_record_documents_analysis_note",
        "cms_analyses_cds_record_belongs_to_pag",
        "cms_analyses_pag_pog_same_group",
        "cms_analyses_documentation_page_in_pag",
        "cms_analyses_documentation_page_in_pog",
        "cms_analyses_documentation_page_in_pog_alt",
    }


@pytest.fixture
def graph(corpus, monkeypatch):
    """The fixture corpus as the sources emit it, queryable by okg's linker.

    Replaces only the database read (`_chronos.query`) with an in-memory
    table of node rows; the linker's own field reading and matching run.
    """
    from okg.substrate.library.linkers import _chronos

    nodes = {}

    def add(facts):
        for fact in facts:
            if hasattr(fact, "subtype"):
                nodes[fact.node_id] = (fact.subtype, dict(fact.attrs))

    add(CDSSource(records=synthetic_cds_records()).run("r", mode="scope_complete").facts)
    add(AnalysisNoteSource(an_root=str(corpus["mirror"])).run("r", mode="scope_complete").facts)
    add(TwikiEOSSource(eos_root=str(corpus["twiki"])).run("r", mode="scope_complete").facts)
    add(PhysicsVocabularySource().run("r", mode="release_new").facts)

    def query(cur, sql, params=None):
        params = params or {}
        assert "okg.graph_nodes" in sql and "ANY(:subtypes)" in sql, sql
        wanted = set(params["subtypes"])
        only = set(params["node_ids"]) if "node_ids" in params else None
        return [
            {"node_id": node_id, "attrs": attrs}
            for node_id, (subtype, attrs) in nodes.items()
            if subtype in wanted and (only is None or node_id in only)
        ]

    monkeypatch.setattr(_chronos, "query", query)
    return nodes


def _pairs(rule_id):
    linker = PhysicsCrossLinker()
    (rule,) = [r for r in linker.rules if r.id == rule_id]
    return linker._dispatch_matcher(
        None, rule=rule, cv=0, dirty_node_ids=None, dirty_edge_ids=frozenset(),
        dirty_node_subtypes=None, dirty_node_attr_changes_by_subtype=None,
        endpoint_cache={}, shape_skips=[],
    )


def test_the_linker_loads_only_this_packs_rules():
    linker = PhysicsCrossLinker()
    assert {r.id for r in linker.rules} == set(_rules())
    assert linker.name == "cms_analyses_cross_links"
    # None for the rule sources must not fall through to okg's whole rule set.
    assert {r.id for r in PhysicsCrossLinker(rules_files=None).rules} == set(_rules())


def test_a_cds_summary_is_linked_to_the_note_it_was_written_from(graph):
    assert _pairs("cms_analyses_cds_record_documents_analysis_note") == {
        ("cds:9990001", an_node_id(PAG_NOTE))
    }
    (rule,) = [r for r in PhysicsCrossLinker().rules if r.id == "cms_analyses_cds_record_documents_analysis_note"]
    # "What was published from this note" and "which note backs this
    # summary" are both asked, so the edge runs both ways.
    assert rule.bidirectional is True


def test_a_legacy_note_reaches_the_published_analysis_it_cites(graph):
    cites = _pairs("cms_analyses_reference_names_cds_record")
    assert cites == {(reference_node_id("synth_search"), "cds:9990002")}
    assert graph[reference_node_id("synth_search")][1]["analysis_code"] == CITED_CODE
    # The conference record carries no analysis code and so finds no partner.
    assert reference_node_id("synth_conference") not in {src for src, _dst in cites}


def test_cds_summaries_and_twiki_pages_join_the_pag_hubs(graph):
    assert _pairs("cms_analyses_cds_record_belongs_to_pag") == {
        ("cds:9990001", pag_hub_id("HIG")), ("cds:9990002", pag_hub_id("TOP")),
    }
    assert _pairs("cms_analyses_documentation_page_in_pag") == {
        ("twiki:CMS:HIG99001Review", pag_hub_id("HIG"))
    }


def test_twiki_pages_join_the_pog_hubs_by_title_root(graph):
    assert _pairs("cms_analyses_documentation_page_in_pog") == {
        ("twiki:CMS:MuonPOGRecipes", pog_hub_id("MUON"))
    }
    assert _pairs("cms_analyses_documentation_page_in_pog_alt") == set()


def test_the_same_group_under_its_two_names_is_joined(graph):
    assert _pairs("cms_analyses_pag_pog_same_group") == {
        (pag_hub_id("MUO"), pog_hub_id("MUON")),
        (pag_hub_id("EGM"), pog_hub_id("EGAMMA")),
        (pag_hub_id("JME"), pog_hub_id("JETMET")),
        (pag_hub_id("BTV"), pog_hub_id("BTAG")),
        (pag_hub_id("TAU"), pog_hub_id("TAUPOG")),
        (pag_hub_id("TRK"), pog_hub_id("TRACKING")),
    }


def test_one_analysis_is_reachable_across_cds_notes_and_twiki(graph, corpus):
    """The point of the pack: start from the note and reach the CDS summary,
    the PAG hub, the wiki page and the other note's cited analysis."""
    links = defaultdict(set)
    rules = _rules()
    for rule_id, rule in rules.items():
        for src, dst in _pairs(rule_id):
            links[src].add(dst)
            if rule["bidirectional"]:
                links[dst].add(src)
    hig_note = an_node_id(PAG_NOTE)
    assert "cds:9990001" in links[hig_note]
    hub = pag_hub_id("HIG")
    assert hub in links["cds:9990001"]
    assert hub in links["twiki:CMS:HIG99001Review"]
    # The legacy note shares no id with a CDS number: two hops through its citation.
    cited = {dst for ref in links if ref.startswith("an_ref:") for dst in links[ref]}
    assert cited == {"cds:9990002"}


def test_every_edge_the_rules_write_is_declared_in_the_schema_bridge():
    bridge = yaml.safe_load(
        (PACKS_DIR / "cms-analyses" / "schemas" / "bridges" / "cms_analyses.yaml").read_text(encoding="utf-8")
    )
    attributes = bridge["classes"]["EdgeNarrowings"]["attributes"]
    declared = {
        (a["annotations"]["src_subtypes"], a["annotations"]["edge_archetype"], a["annotations"]["dst_subtypes"])
        for a in attributes.values()
    }
    classes = {
        "analysis_note": "AnalysisNote", "cds_record": "CDSRecord", "reference": "Reference",
        "physics_analysis_group": "PhysicsAnalysisGroup", "physics_object_group": "PhysicsObjectGroup",
        "documentation_page": "DocumentationPage",
    }
    linker = PhysicsCrossLinker()
    for src, edge, dst in linker.requires_narrowings:
        assert (classes[src], edge, classes[dst]) in declared, (src, edge, dst)


# --- the pack ----------------------------------------------------------------


def test_the_pack_requires_the_packs_whose_records_it_links():
    pack = manifest("cms-analyses")
    assert pack["requires"]["packs"] == ["twiki", "cds", "analysis-notes"]
    assert pack["needs_credentials"] is True


def test_the_entity_template_reads_the_answers_of_the_packs_it_requires():
    entry = templates("cms-analyses")["physics_entities"]
    params = entry["params"]
    assert params["eos_root"] == "${twiki_eos_root}"
    assert params["an_root"] == "${analysis_notes_mirror_root}"
    assert params["physics_filter"] == "${twiki_physics_filter}"
    # Ordered after the notes source, which is itself ordered after twiki.
    assert entry["predecessor_source"] == templates("analysis-notes")["analysis_notes"]["ownership_id"]
    assert entry["admission_policy"]["trust_label"] == "untrusted"


def test_the_coverage_invariant_requires_every_subtype_the_pack_emits():
    invariant, enricher = invariants("cms-analyses")
    # The second invariant only warns when the hand-wired enricher is missing.
    assert enricher["name"] == "cms_analyses_enricher_edges"
    assert enricher["severity"] == "warn"
    assert "physics_object_group" in enricher["sql"] and "cds_record" in enricher["sql"]
    for subtype in (
        "physics_analysis_group", "physics_object_group",
        "physics_technique", "physics_concept", "physics_topic",
    ):
        assert f"'{subtype}'" in invariant["sql"]


def test_the_playbook_is_the_library_skill():
    link = PACKS_DIR / "cms-analyses" / "playbooks" / "analysis_lookup.md"
    assert link.is_symlink() and link.resolve().is_file()
    text = link.read_text(encoding="utf-8")
    assert "analysis_code" in text and "physics_analysis_group" in text
