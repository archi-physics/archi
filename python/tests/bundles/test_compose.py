"""acc.r3-bundle-format.rules: one test per ADR 0002 section 3.3 rule.

The fixture catalog has two packs: `alpha` (credential-free, every optional
part) and `beta` (needs a token, requires alpha). Tests that need a third
pack clone alpha as `gamma` and change one thing, so each conflict test
isolates one rule.
"""
import pytest
import yaml
from packaging.specifiers import SpecifierSet

from archi.bundles.compose import compose, digest
from archi.bundles.resolve import CompositionError, resolve
from archi.bundles.versions import OkgRangeError, canonical, intersect, is_empty

from bundles.bundle_support import CATALOG, clone_pack, edit_yaml, load_catalog, write_bundle


def _conflict(fn, code):
    with pytest.raises(CompositionError) as excinfo:
        fn()
    assert excinfo.value.code == code, (excinfo.value.code, str(excinfo.value))
    return excinfo.value


@pytest.fixture
def with_gamma(catalog_copy):
    """catalog_copy plus `gamma`, an alpha clone, and a bundle defaulting to both."""
    clone_pack(catalog_copy, "alpha", "gamma")
    write_bundle(catalog_copy, "pair", packs={"default": ["alpha", "gamma"]})
    return catalog_copy


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_extends_closure_puts_parents_first():
    r = resolve(load_catalog(CATALOG), ["child-bundle"])
    assert [b.name for b in r.bundles] == ["base-bundle", "child-bundle"]
    assert r.pack_names == ("alpha", "beta")


def test_extends_closure_is_transitive(catalog_copy):
    write_bundle(catalog_copy, "grandchild", extends=["child-bundle"], packs={"default": []})
    r = resolve(load_catalog(catalog_copy), ["grandchild"])
    assert [b.name for b in r.bundles] == ["base-bundle", "child-bundle", "grandchild"]
    assert r.pack_names == ("alpha", "beta")


def test_requires_closure_puts_requirements_first(catalog_copy):
    """beta requires alpha: listing beta first still resolves alpha before it."""
    write_bundle(catalog_copy, "rev", packs={"default": ["beta", "alpha"]})
    assert resolve(load_catalog(catalog_copy), ["rev"]).pack_names == ("alpha", "beta")


def test_requires_closure_pulls_in_unlisted_packs(catalog_copy):
    clone_pack(catalog_copy, "alpha", "gamma", lambda d: d.update(requires={"packs": ["alpha"]}))
    write_bundle(catalog_copy, "g", packs={"default": ["gamma"]})
    assert resolve(load_catalog(catalog_copy), ["g"]).pack_names == ("alpha", "gamma")


def test_requires_cycle_is_refused(catalog_copy):
    edit_yaml(catalog_copy / "packs" / "alpha" / "pack.yaml",
              lambda d: d["requires"].update(packs=["beta"]))
    write_bundle(catalog_copy, "c", packs={"default": ["alpha"]})
    from archi.bundles import discovery

    catalog = discovery.from_roots([catalog_copy / "packs"], [catalog_copy])
    _conflict(lambda: resolve(catalog, ["c"]), "requires_cycle")


def test_with_adds_a_pack():
    r = resolve(load_catalog(CATALOG), ["base-bundle"], with_packs=["beta"])
    assert r.pack_names == ("alpha", "beta")


def test_with_unknown_pack_is_refused():
    _conflict(lambda: resolve(load_catalog(CATALOG), ["base-bundle"], with_packs=["nope"]),
              "unknown_pack")


def test_with_does_not_close_requires(with_gamma):
    """The formula closes `requires` over defaults only; a --with pack whose
    requirement is missing is refused, naming the --with that fixes it."""
    write_bundle(with_gamma, "g", packs={"default": ["gamma"]})
    err = _conflict(lambda: resolve(load_catalog(with_gamma), ["g"], with_packs=["beta"]),
                    "requires_unmet")
    assert "--with alpha" in str(err)


def test_without_removes_a_default():
    r = resolve(load_catalog(CATALOG), ["child-bundle"], without_packs=["beta"])
    assert r.pack_names == ("alpha",)


def test_without_a_requirement_is_refused():
    err = _conflict(
        lambda: resolve(load_catalog(CATALOG), ["child-bundle"], without_packs=["alpha"]),
        "requires_unmet",
    )
    assert "do not remove ['alpha']" in str(err)


def test_without_an_unselected_pack_is_refused():
    _conflict(lambda: resolve(load_catalog(CATALOG), ["base-bundle"], without_packs=["beta"]),
              "without_not_selected")


def test_unknown_bundle_is_refused():
    _conflict(lambda: resolve(load_catalog(CATALOG), ["nope"]), "unknown_bundle")


def test_a_pack_selected_twice_installs_once(catalog_copy):
    write_bundle(catalog_copy, "other", packs={"default": ["alpha"]})
    r = resolve(load_catalog(catalog_copy), ["base-bundle", "other"], with_packs=["alpha"])
    assert r.pack_names == ("alpha",)
    assert list(compose(r).sources) == ["alpha.pages"]


# ---------------------------------------------------------------------------
# Merge rules
# ---------------------------------------------------------------------------


def test_modules_are_a_set_union(with_gamma):
    edit_yaml(with_gamma / "packs" / "gamma" / "pack.yaml",
              lambda d: d["requires"].update(modules=["extraction", "git_graph"]))
    comp = compose(resolve(load_catalog(with_gamma), ["pair"]))
    assert comp.modules == ["document_starter", "extraction", "git_graph"]


def test_identical_schema_paths_merge(with_gamma):
    comp = compose(resolve(load_catalog(with_gamma), ["pair"]))
    assert list(comp.schemas) == ["alpha.yaml"]
    assert comp.owners["schema:alpha.yaml"] == ["alpha", "gamma"]


def test_differing_schema_paths_stop_with_both_digests(with_gamma):
    path = with_gamma / "packs" / "gamma" / "schemas" / "alpha.yaml"
    original = (CATALOG / "packs" / "alpha" / "schemas" / "alpha.yaml").read_bytes()
    path.write_text("id: https://example.org/other\nname: other\n")
    err = _conflict(lambda: compose(resolve(load_catalog(with_gamma), ["pair"])),
                    "schema_conflict")
    assert digest(original) in str(err) and digest(path.read_bytes()) in str(err)
    assert err.data["packs"] == ["alpha", "gamma"]


def test_source_ids_are_pack_qualified():
    comp = compose(resolve(load_catalog(CATALOG), ["child-bundle"]))
    assert list(comp.sources) == ["alpha.pages", "beta.issues"]
    body = yaml.safe_load(comp.sources["alpha.pages"])
    assert list(body) == ["alpha.pages"]


def test_same_source_name_in_two_packs_does_not_collide(with_gamma):
    comp = compose(resolve(load_catalog(with_gamma), ["pair"]))
    assert list(comp.sources) == ["alpha.pages", "gamma.pages"]


def test_question_ids_are_pack_namespaced():
    comp = compose(resolve(load_catalog(CATALOG), ["base-bundle"]))
    ids = [q["id"] for q in comp.questions]
    assert ids[:5] == ["deployment_name", "postgres_dsn", "archi_data_root",
                       "chat_site_name", "chat_model"]
    assert "alpha_site_url" in ids and "site_url" not in ids
    body = yaml.safe_load(comp.sources["alpha.pages"])["alpha.pages"]
    assert body["params"]["base_url"] == "${alpha_site_url}"
    assert body["params"]["records_path"] == "${archi_data_root}/alpha/records.json"


def test_identical_questions_merge(catalog_copy):
    """`alpha-site` + `url` and `alpha` + `site_url` both render alpha_site_url."""
    def change(d):
        d["init_questions"][0]["id"] = "url"
    clone_pack(catalog_copy, "alpha", "alpha-site")
    edit_yaml(catalog_copy / "packs" / "alpha-site" / "questions.yaml", change)
    (catalog_copy / "packs" / "alpha-site" / "sources" / "pages.yaml").write_text(
        (catalog_copy / "packs" / "alpha-site" / "sources" / "pages.yaml")
        .read_text().replace("${site_url}", "${url}")
    )
    write_bundle(catalog_copy, "two", packs={"default": ["alpha", "alpha-site"]})
    comp = compose(resolve(load_catalog(catalog_copy), ["two"]))
    assert [q["id"] for q in comp.questions].count("alpha_site_url") == 1
    assert comp.owners["question:alpha_site_url"] == ["alpha", "alpha-site"]


def test_one_question_id_two_definitions_is_refused(catalog_copy):
    def change(d):
        d["init_questions"][0].update(id="url", default="https://other.example.org")
    clone_pack(catalog_copy, "alpha", "alpha-site")
    edit_yaml(catalog_copy / "packs" / "alpha-site" / "questions.yaml", change)
    (catalog_copy / "packs" / "alpha-site" / "sources" / "pages.yaml").write_text(
        (catalog_copy / "packs" / "alpha-site" / "sources" / "pages.yaml")
        .read_text().replace("${site_url}", "${url}")
    )
    write_bundle(catalog_copy, "two", packs={"default": ["alpha", "alpha-site"]})
    _conflict(lambda: compose(resolve(load_catalog(catalog_copy), ["two"])), "question_conflict")


def test_invariants_belong_to_their_pack(catalog_copy):
    clone_pack(catalog_copy, "alpha", "gamma")
    edit_yaml(catalog_copy / "packs" / "gamma" / "invariants.yaml",
              lambda d: d["invariants"][0].update(name="gamma_floor"))
    write_bundle(catalog_copy, "pair", packs={"default": ["alpha", "gamma"]})
    catalog = load_catalog(catalog_copy)
    both = compose(resolve(catalog, ["pair"]))
    assert [i["name"] for i in both.invariants] == ["alpha_pages_floor", "gamma_floor"]
    without = compose(resolve(catalog, ["pair"], without_packs=["gamma"]))
    assert [i["name"] for i in without.invariants] == ["alpha_pages_floor"]


def test_one_invariant_name_two_bodies_is_refused(with_gamma):
    edit_yaml(with_gamma / "packs" / "gamma" / "invariants.yaml",
              lambda d: d["invariants"][0].update(severity="warning"))
    _conflict(lambda: compose(resolve(load_catalog(with_gamma), ["pair"])), "invariant_conflict")


def test_freshness_belongs_to_its_pack(with_gamma):
    catalog = load_catalog(with_gamma)
    assert set(compose(resolve(catalog, ["pair"])).freshness) == {"alpha.pages", "gamma.pages"}
    assert set(compose(resolve(catalog, ["pair"], without_packs=["gamma"])).freshness) == {
        "alpha.pages"
    }


def test_search_subtypes_are_a_union(with_gamma):
    edit_yaml(with_gamma / "packs" / "gamma" / "pack.yaml",
              lambda d: d.update(search_subtypes=["document_chunk", "documentation_page"]))
    comp = compose(resolve(load_catalog(with_gamma), ["pair"]))
    assert comp.search_subtypes == ["document_chunk", "documentation_page"]


def test_identical_playbooks_and_triggers_merge(with_gamma):
    comp = compose(resolve(load_catalog(with_gamma), ["pair"]))
    assert list(comp.playbooks) == ["alpha.md"]
    assert list(comp.triggers["intents"]) == ["alpha_pages"]


def test_one_playbook_name_two_contents_is_refused(with_gamma):
    (with_gamma / "packs" / "gamma" / "playbooks" / "alpha.md").write_text("# Other\n")
    _conflict(lambda: compose(resolve(load_catalog(with_gamma), ["pair"])), "playbook_conflict")


def test_one_trigger_id_two_definitions_is_refused(with_gamma):
    edit_yaml(with_gamma / "packs" / "gamma" / "playbooks" / "skill-triggers.yaml",
              lambda d: d["intents"]["alpha_pages"].update(description="Different."))
    _conflict(lambda: compose(resolve(load_catalog(with_gamma), ["pair"])), "trigger_conflict")


def test_a_playbook_may_not_replace_the_rendered_prompt(with_gamma):
    (with_gamma / "packs" / "gamma" / "playbooks" / "chat-system-prompt.md").write_text("# x\n")
    _conflict(lambda: compose(resolve(load_catalog(with_gamma), ["pair"])), "playbook_reserved")


def _add_action_classes(root, pack, body):
    (root / "packs" / pack / "actions.yaml").write_text(yaml.safe_dump({"action_classes": body}))
    edit_yaml(root / "packs" / pack / "pack.yaml", lambda d: d.update(action_classes="actions.yaml"))


def test_action_classes_are_a_union_by_id(with_gamma):
    _add_action_classes(with_gamma, "alpha", {"drain_node": {"max_level": "L2"}})
    _add_action_classes(with_gamma, "gamma", {"drain_node": {"max_level": "L2"},
                                              "requeue_job": {"max_level": "L1"}})
    comp = compose(resolve(load_catalog(with_gamma), ["pair"]))
    assert list(comp.action_classes) == ["drain_node", "requeue_job"]


def test_one_action_class_two_definitions_is_refused(with_gamma):
    _add_action_classes(with_gamma, "alpha", {"drain_node": {"max_level": "L2"}})
    _add_action_classes(with_gamma, "gamma", {"drain_node": {"max_level": "L1"}})
    _conflict(lambda: compose(resolve(load_catalog(with_gamma), ["pair"])),
              "action_class_conflict")


def test_files_union_and_conflict(with_gamma):
    for pack, text in (("alpha", "a: 1\n"), ("gamma", "a: 1\n")):
        (with_gamma / "packs" / pack / "extractors.yaml").write_text(text)
        edit_yaml(with_gamma / "packs" / pack / "pack.yaml",
                  lambda d: d.update(files=["extractors.yaml"]))
    catalog = load_catalog(with_gamma)
    assert list(compose(resolve(catalog, ["pair"])).files) == ["extractors.yaml"]
    (with_gamma / "packs" / "gamma" / "extractors.yaml").write_text("a: 2\n")
    _conflict(lambda: compose(resolve(catalog, ["pair"])), "file_conflict")


def test_prompt_is_base_then_fragments_in_resolution_order(with_gamma):
    (with_gamma / "packs" / "gamma" / "prompt.md").write_text("Gamma fragment.\n")
    catalog = load_catalog(with_gamma)
    prompt = compose(resolve(catalog, ["pair"])).prompt
    assert prompt.startswith("# Knowledge-graph chat assistant")
    assert prompt.index("Alpha pages") < prompt.index("Gamma fragment")
    write_bundle(with_gamma, "rpair", packs={"default": ["gamma", "alpha"]})
    prompt = compose(resolve(load_catalog(with_gamma), ["rpair"])).prompt
    assert prompt.index("Gamma fragment") < prompt.index("Alpha pages")


def test_chat_scalars_come_from_the_last_bundle_given():
    catalog = load_catalog(CATALOG)
    r = resolve(catalog, ["child-bundle"])
    assert r.chat["site_name"] == "Child chat"  # a child overrides its parent
    assert r.notes == ("chat.site_name: 'child-bundle' sets 'Child chat'; "
                       "'base-bundle' sets 'Base chat', which is not used",)
    comp = compose(r)
    site = next(q for q in comp.questions if q["id"] == "chat_site_name")
    assert site["default"] == "Child chat"


def test_last_bundle_given_wins_over_an_earlier_child(catalog_copy):
    write_bundle(catalog_copy, "later", packs={"default": ["alpha"]},
                 chat={"site_name": "Later chat", "suggestions": ["Later?"]})
    r = resolve(load_catalog(catalog_copy), ["child-bundle", "later"])
    assert r.chat["site_name"] == "Later chat"
    assert len(r.notes) == 2  # Child chat and Base chat both reported
    assert r.chat["suggestions"] == ["What is in alpha?", "Later?"]


def test_okg_ranges_intersect():
    r = resolve(load_catalog(CATALOG), ["child-bundle"])
    assert canonical(r.okg_range) == "<0.5,>=0.4.0"


def test_empty_okg_intersection_stops(catalog_copy):
    write_bundle(catalog_copy, "future", packs={"default": ["alpha"]}, okg=">=0.5")
    err = _conflict(lambda: resolve(load_catalog(catalog_copy), ["base-bundle", "future"]),
                    "okg_range_empty")
    assert err.data["ranges"] == {"base-bundle": ">=0.4.0,<0.5", "future": ">=0.5"}


@pytest.mark.parametrize(
    "spec, empty",
    [
        (">=0.2.0,<0.3", False),
        (">=0.3,<0.3", True),
        (">0.2,<=0.2", True),
        ("==0.2.0,!=0.2.0", True),
        ("==0.2.*,>=0.3", True),
        ("~=0.2.1,<0.2.1", True),
        ("~=0.2,<0.4", False),
        ("<=0.2.0,>=0.2.0", False),
    ],
)
def test_range_emptiness(spec, empty):
    assert is_empty(SpecifierSet(spec)) is empty


def test_intersect_reports_every_range():
    with pytest.raises(OkgRangeError) as excinfo:
        intersect([">=0.2.0,<0.3", ">=0.3"])
    assert "'>=0.2.0,<0.3'" in str(excinfo.value) and "'>=0.3'" in str(excinfo.value)


def test_search_profiles_union_and_conflict(with_gamma):
    profile = {"description": "Pages only.", "subtypes": ["document_chunk"]}
    for pack in ("alpha", "gamma"):
        edit_yaml(with_gamma / "packs" / pack / "pack.yaml",
                  lambda d: d.update(search_profiles={"pages": dict(profile)}))
    catalog = load_catalog(with_gamma)
    comp = compose(resolve(catalog, ["pair"]))
    assert comp.search_profiles == {"pages": profile}
    assert comp.owners["search_profile:pages"] == ["alpha", "gamma"]
    edit_yaml(with_gamma / "packs" / "gamma" / "pack.yaml",
              lambda d: d["search_profiles"]["pages"].update(subtypes=["documentation_page"]))
    _conflict(lambda: compose(resolve(load_catalog(with_gamma), ["pair"])), "search_profile_conflict")


def test_a_pack_cannot_redefine_the_content_profile(catalog_copy):
    from archi.bundles.manifest import validate_pack

    pack = catalog_copy / "packs" / "alpha"
    edit_yaml(pack / "pack.yaml", lambda d: d.update(
        search_profiles={"content": {"description": "x", "subtypes": ["document_chunk"]}}))
    assert any("content" in str(p) for p in validate_pack(pack))
