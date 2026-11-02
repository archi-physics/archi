"""cms-analyses: selectivity, the match cache's failure modes, and the linker's
write policy (PACT r3-bundle-cms-physics). Synthetic corpus throughout."""
import json
from types import SimpleNamespace

import pytest

from archi.enrichment import physics_cross_links
from archi.enrichment.physics_cross_links import TRUST_LABEL, PhysicsCrossLinker
from archi.sources.physics_entities import PhysicsEntitySource
from packs.physics_fixtures import write_note_mirror, write_twiki_dump


@pytest.fixture
def corpus(tmp_path):
    return {
        "mirror": write_note_mirror(tmp_path / "mirror"),
        "twiki": write_twiki_dump(tmp_path / "twiki"),
        "cache": tmp_path / "cache" / "matches.json",
    }


def _source(corpus, **kwargs):
    return PhysicsEntitySource(
        eos_root=str(corpus["twiki"]), an_root=str(corpus["mirror"]),
        physics_filter=False, **kwargs,
    )


def _edges(source):
    run = source.run("run-1", mode="release_new")
    return {
        (f.src, f.dst) for f in run.facts if hasattr(f, "edge_type")
    }


# --- selectivity ---------------------------------------------------------------


@pytest.mark.parametrize(
    "reached,band",
    [(0, "rare"), (150, "rare"), (151, "useful"), (999, "useful"), (1000, "broad"), (5000, "broad")],
)
def test_the_selectivity_band_follows_the_documents_reached(reached, band):
    assert PhysicsEntitySource._band(reached) == band


# --- the match cache -------------------------------------------------------------


def test_a_damaged_cache_is_ignored_and_the_edges_are_the_same(corpus):
    cold = _edges(_source(corpus))
    for damage in ("", "not json {", "[1, 2]", json.dumps({"format": 99, "documents": {}})):
        corpus["cache"].parent.mkdir(parents=True, exist_ok=True)
        corpus["cache"].write_text(damage, encoding="utf-8")
        source = _source(corpus, match_cache_path=str(corpus["cache"]))
        assert _edges(source) == cold
        assert source.last_link_report["reused"] == 0


def test_a_cache_written_under_another_vocabulary_is_discarded(corpus):
    source = _source(corpus, match_cache_path=str(corpus["cache"]))
    _edges(source)
    payload = json.loads(corpus["cache"].read_text(encoding="utf-8"))
    payload["vocabulary"] = "an older vocabulary"
    corpus["cache"].write_text(json.dumps(payload), encoding="utf-8")
    again = _source(corpus, match_cache_path=str(corpus["cache"]))
    _edges(again)
    assert again.last_link_report["reused"] == 0 and again.last_link_report["matched"] == 5


def test_an_unwritable_cache_costs_time_not_the_ingest(corpus, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    unwritable = blocker / "matches.json"  # its parent is a file
    cold = _edges(_source(corpus))
    source = _source(corpus, match_cache_path=str(unwritable))
    assert _edges(source) == cold
    assert not unwritable.exists()


def test_a_vanished_document_leaves_no_cache_entry_and_no_edges(corpus):
    source = _source(corpus, match_cache_path=str(corpus["cache"]))
    _edges(source)
    cached = json.loads(corpus["cache"].read_text(encoding="utf-8"))["documents"]
    assert "twiki:CMS:MuonPOGRecipes" in cached
    (corpus["twiki"] / "MuonPOGRecipes.txt").unlink()
    again = _source(corpus, match_cache_path=str(corpus["cache"]))
    edges = _edges(again)
    assert not any(src == "twiki:CMS:MuonPOGRecipes" for src, _dst in edges)
    rewritten = json.loads(corpus["cache"].read_text(encoding="utf-8"))["documents"]
    assert "twiki:CMS:MuonPOGRecipes" not in rewritten


def test_a_document_with_no_entities_is_remembered_too(corpus):
    source = _source(corpus, match_cache_path=str(corpus["cache"]))
    _edges(source)
    cached = json.loads(corpus["cache"].read_text(encoding="utf-8"))["documents"]
    empty = [doc for doc, entry in cached.items() if entry["e"] == []]
    assert empty, "the bridge-target page names no entity and must still be cached"


def test_the_cache_path_is_not_part_of_the_source_identity(corpus, tmp_path):
    plain = _source(corpus)
    cached = _source(corpus, match_cache_path=str(tmp_path / "elsewhere.json"))
    assert plain._probe_content_items() == cached._probe_content_items()
    assert plain.change_probe._config_hash == cached.change_probe._config_hash


def test_the_vocabulary_is_part_of_the_source_identity(corpus):
    assert dict(_source(corpus)._probe_content_items())["vocabulary"].startswith(b"technique:b_tagging:")


# --- the linker's write policy -----------------------------------------------------


def test_the_linker_policy_is_untrusted_and_allows_only_the_rules_edges():
    linker = PhysicsCrossLinker()
    policy = linker.producer_policy
    assert TRUST_LABEL == "untrusted"
    assert policy["trust_label"] == "untrusted"
    assert policy["producer_kind"] == "enricher"
    assert policy["producer_id"] == "_enricher:cms_analyses_cross_links"
    assert policy["authority_scope"]["enricher_name"] == "cms_analyses_cross_links"
    assert policy["allowed_record_kinds"] == ["edge"]
    allowed = {tuple(c["triple"]) for c in policy["policy_clauses"]}
    assert all(c["effect"] == "allow" and c["record_kind"] == "edge" for c in policy["policy_clauses"])
    # Exactly the triples the rules need, no more.
    assert allowed == set(linker.requires_narrowings)
    assert {tuple(e) for e in policy["output_scope_summary"]["edges"]} == allowed


def test_the_policy_keeps_okgs_signature_and_changes_only_the_label(monkeypatch):
    seen = {}

    def fake_envelope(linker, producer_id):
        seen["producer_id"] = producer_id
        return SimpleNamespace(
            signature={
                "nodes": [{"clause_id": "n1", "subtype": "reference"}],
                "edges": [{
                    "clause_id": "e1", "src_subtype": "reference",
                    "edge_type": "references", "dst_subtype": "cds_record",
                }],
            },
            unbounded=(),
            summary=lambda: {"fake": True},
        )

    monkeypatch.setattr(physics_cross_links, "derive_write_envelope", fake_envelope)
    policy = PhysicsCrossLinker().producer_policy
    assert seen["producer_id"] == "_enricher:cms_analyses_cross_links"
    assert policy["allowed_record_kinds"] == ["node", "edge"]
    assert policy["output_signature"]["edges"][0]["clause_id"] == "e1"
    assert {c["id"] for c in policy["policy_clauses"]} == {"n1", "e1"}
    assert policy["output_scope_summary"]["write_envelope"] == {"fake": True}
    assert policy["trust_label"] == "untrusted"


def test_an_unbounded_write_surface_is_refused_not_narrowed(monkeypatch):
    monkeypatch.setattr(
        physics_cross_links, "derive_write_envelope",
        lambda linker, producer_id: SimpleNamespace(
            signature={"nodes": [], "edges": []}, unbounded=("edges",), summary=dict
        ),
    )
    with pytest.raises(ValueError, match="unbounded"):
        PhysicsCrossLinker().producer_policy


def test_explicit_rules_replace_the_packs_rules():
    only = PhysicsCrossLinker(include_rule_ids=["cms_analyses_pag_pog_same_group"])
    assert [r.id for r in only.rules] == ["cms_analyses_pag_pog_same_group"]
    # An empty rule list means "not provided", not "no rules".
    assert len(PhysicsCrossLinker(rules=[]).rules) == 7
