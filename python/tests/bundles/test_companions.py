"""Companion packs (ADR 0002 section 3.3, amended 2026-10-31; PACT r3-bundle-companions).

A bundle may name packs that come on whenever another pack is selected:
`packs.companions: {<pack>: [<companion>, ...]}`. cms-comp-ops uses it so that
selecting git-code also brings code-text. These tests hold the rule on the
fixture catalog (alpha, beta requiring alpha, and clones), then on the shipped
cms-comp-ops bundle, then through `archi add` on an installed instance.
"""
import pytest

from archi.bundles.resolve import CompositionError, resolve

from bundles.bundle_support import CATALOG, clone_pack, load_catalog, write_bundle


def _conflict(fn, code):
    with pytest.raises(CompositionError) as excinfo:
        fn()
    assert excinfo.value.code == code, (excinfo.value.code, str(excinfo.value))
    return excinfo.value


@pytest.fixture
def companions(catalog_copy):
    """`gamma` (an alpha clone) is beta's companion in bundle `comp`, which
    defaults to alpha and offers beta; `delta` is gamma's own companion."""
    clone_pack(catalog_copy, "alpha", "gamma")
    clone_pack(catalog_copy, "alpha", "delta")
    write_bundle(catalog_copy, "comp", packs={
        "default": ["alpha"],
        "optional": ["beta"],
        "companions": {"beta": ["gamma"], "gamma": ["delta"]},
    })
    return catalog_copy


def test_no_trigger_selected_means_no_companion(companions):
    r = resolve(load_catalog(companions), ["comp"])
    assert r.pack_names == ("alpha",)
    assert not [n for n in r.notes if "comes with" in n]


def test_with_the_trigger_brings_its_companions(companions):
    """--with beta brings gamma, and gamma brings delta: companions chain."""
    r = resolve(load_catalog(companions), ["comp"], with_packs=["beta"])
    assert r.pack_names == ("alpha", "beta", "gamma", "delta")
    assert "gamma comes with beta (opt out with --without gamma)" in r.notes


def test_without_a_companion_opts_out(companions):
    """Removing a companion is allowed although it was never selected, and
    removes only that companion and what only it would have brought."""
    r = resolve(load_catalog(companions), ["comp"], with_packs=["beta"], without_packs=["gamma"])
    assert r.pack_names == ("alpha", "beta")


def test_a_default_trigger_brings_its_companion(companions):
    write_bundle(companions, "comp2", packs={"default": ["alpha", "beta"], "companions": {"beta": ["gamma"]}})
    r = resolve(load_catalog(companions), ["comp2"])
    assert r.pack_names == ("alpha", "beta", "gamma")


def test_removing_the_trigger_removes_its_companions(companions):
    write_bundle(companions, "comp2", packs={"default": ["alpha", "beta"], "companions": {"beta": ["gamma"]}})
    r = resolve(load_catalog(companions), ["comp2"], without_packs=["beta"])
    assert r.pack_names == ("alpha",)


def test_without_an_unselected_pack_that_is_no_companion_is_still_refused(companions):
    _conflict(lambda: resolve(load_catalog(companions), ["comp"], without_packs=["beta"]),
              "without_not_selected")


def test_an_opt_out_can_be_recorded_before_its_trigger_is_selected(companions):
    r = resolve(load_catalog(companions), ["comp"], without_packs=["gamma"])
    assert r.pack_names == ("alpha",)


def test_a_companion_brings_the_packs_it_requires(companions):
    """beta requires alpha; as a companion of gamma in a bundle without alpha,
    it comes with alpha."""
    write_bundle(companions, "comp3", packs={"default": ["gamma"], "companions": {"gamma": ["beta"]}})
    r = resolve(load_catalog(companions), ["comp3"])
    assert r.pack_names == ("gamma", "alpha", "beta")


def test_companions_merge_over_extends(companions):
    write_bundle(companions, "child", extends=["comp"], packs={"default": [], "companions": {"beta": ["delta"]}})
    r = resolve(load_catalog(companions), ["child"], with_packs=["beta"], without_packs=["gamma"])
    assert r.pack_names == ("alpha", "beta", "delta")


def test_an_unknown_companion_is_a_clear_validation_error(companions):
    from archi.bundles import discovery

    write_bundle(companions, "broken", packs={"default": ["alpha"], "companions": {"alpha": ["nope"]}})
    catalog = discovery.from_roots([companions / "packs"], [companions])
    messages = [str(p) for p in catalog.problems]
    assert any("packs.companions.alpha: unknown companion pack 'nope'" in m for m in messages), messages


def test_an_unknown_companion_stops_resolution(companions):
    """A catalog assembled without validation (a third-party root missing the
    pack) still stops at resolve, naming both packs."""
    from archi.bundles import discovery

    write_bundle(companions, "broken", packs={"default": ["alpha"], "companions": {"alpha": ["nope"]}})
    catalog = discovery.from_roots([companions / "packs"], [companions])
    if "broken" not in catalog.bundles:
        pytest.skip("this catalog drops a bundle that fails validation; the validation test covers it")
    err = _conflict(lambda: resolve(catalog, ["broken"]), "unknown_companion")
    assert "'alpha'" in str(err) and "'nope'" in str(err)


@pytest.mark.parametrize("bundle", ["base-bundle", "child-bundle"])
@pytest.mark.parametrize("with_packs, without_packs", [((), ()), (("beta",), ()), ((), ("beta",))])
def test_a_bundle_with_no_companions_resolves_as_before(bundle, with_packs, without_packs):
    """The fixture bundles name no companions: their resolution is the plain
    formula, closure(requires, defaults) ∪ with − without."""
    catalog = load_catalog(CATALOG)
    if bundle == "base-bundle" and without_packs:
        pytest.skip("beta is not selected in base-bundle")
    r = resolve(catalog, [bundle], with_packs=list(with_packs), without_packs=list(without_packs))
    expected = {
        ("base-bundle", (), ()): ("alpha",),
        ("base-bundle", ("beta",), ()): ("alpha", "beta"),
        ("child-bundle", (), ()): ("alpha", "beta"),
        ("child-bundle", ("beta",), ()): ("alpha", "beta"),
        ("child-bundle", (), ("beta",)): ("alpha",),
    }[(bundle, with_packs, without_packs)]
    assert r.pack_names == expected
    assert not [n for n in r.notes if "comes with" in n]


# ---------------------------------------------------------------------------
# The shipped cms-comp-ops bundle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shipped():
    from packs.pack_support import shipped_catalog

    return shipped_catalog()


def test_shipped_bundles_without_companions_resolve_as_before(shipped):
    """Only cms-comp-ops names companions; every other shipped bundle's bare
    and every-optional resolutions contain no companion note."""
    for name, bundle in shipped.bundles.items():
        if name == "cms-comp-ops":
            continue
        assert bundle.companions == {}, name
        for with_packs in ([], list(bundle.optional_packs)):
            r = resolve(shipped, [name], with_packs=with_packs)
            assert not [n for n in r.notes if "comes with" in n], (name, r.notes)


def test_comp_ops_without_git_code_gets_no_repository_packs(shipped):
    names = resolve(shipped, ["cms-comp-ops"]).pack_names
    assert "git-code" not in names and "code-text" not in names


def test_comp_ops_init_with_git_code_brings_code_text(shipped):
    r = resolve(shipped, ["cms-comp-ops"], with_packs=["git-code"])
    names = r.pack_names
    assert names.index("git-code") < names.index("code-text")
    assert "code-text comes with git-code (opt out with --without code-text)" in r.notes


def test_comp_ops_init_with_git_code_without_code_text(shipped):
    names = resolve(shipped, ["cms-comp-ops"], with_packs=["git-code"], without_packs=["code-text"]).pack_names
    assert "git-code" in names and "code-text" not in names


def test_comp_ops_render_with_git_code_carries_code_text(shipped, tmp_path):
    from archi.bundles.render import render

    rendered = render(resolve(shipped, ["cms-comp-ops"], with_packs=["git-code"]), tmp_path)
    assert {"git-code.repo", "git-code.git_history", "code-text.code_text"} <= set(rendered.composition.sources)
    assert (rendered.profile_dir / "extractors.yaml").is_file()


# ---------------------------------------------------------------------------
# archi add on an installed instance
# ---------------------------------------------------------------------------


def test_archi_add_git_code_brings_code_text(tmp_path):
    """An installed cms-comp-ops-shaped instance (fixture packs stand in for the
    catalogs) adds git-code; code-text's source and extractors.yaml come too,
    and the lock records the selection, not the companion."""
    import yaml
    from bundles.bundle_support import build_cern_team_catalog
    from bundles.test_lock import BASE_ANSWERS, _scaffold

    from archi.bundles import discovery
    from archi.bundles.lock import add, read_lock, render_and_lock

    root = build_cern_team_catalog(tmp_path / "catalog")
    write_bundle(root, "ops-like", extends=["cern-team"],
                 packs={"default": [], "optional": ["git-code"], "companions": {"git-code": ["code-text"]}})
    catalog = discovery.from_roots([root / "packs"], [root])
    assert catalog.problems == [], [str(p) for p in catalog.problems]

    instance = tmp_path / "instance"
    render_and_lock(catalog, instance, ["ops-like"])
    deployment = _scaffold(instance, "ops-like", BASE_ANSWERS)
    add(instance, catalog, "git-code", {"git_code_repo_url": "https://example.org/x.git"},
        deployment_dir=deployment, run_okg=False, echo=lambda *_: None)
    registry = yaml.safe_load((deployment / "source_registry.yaml").read_text())["sources"]
    assert {"git-code.repo", "git-code.git_history", "code-text.code_text"} <= set(registry)
    assert (deployment / "extractors.yaml").is_file()
    lock = read_lock(instance)
    assert lock["selection"]["with"] == ["git-code"]
    assert {"git-code", "code-text"} <= set(lock["packs"])


def test_archi_add_git_code_respects_an_earlier_opt_out(tmp_path):
    import yaml
    from bundles.bundle_support import build_cern_team_catalog
    from bundles.test_lock import BASE_ANSWERS, _scaffold

    from archi.bundles import discovery
    from archi.bundles.lock import add, read_lock, render_and_lock

    root = build_cern_team_catalog(tmp_path / "catalog")
    write_bundle(root, "ops-like", extends=["cern-team"],
                 packs={"default": [], "optional": ["git-code"], "companions": {"git-code": ["code-text"]}})
    catalog = discovery.from_roots([root / "packs"], [root])
    instance = tmp_path / "instance"
    render_and_lock(catalog, instance, ["ops-like"], without_packs=["code-text"])
    deployment = _scaffold(instance, "ops-like", BASE_ANSWERS)
    add(instance, catalog, "git-code", {"git_code_repo_url": "https://example.org/x.git"},
        deployment_dir=deployment, run_okg=False, echo=lambda *_: None)
    registry = yaml.safe_load((deployment / "source_registry.yaml").read_text())["sources"]
    assert "git-code.repo" in registry and "code-text.code_text" not in registry
    assert "code-text" not in read_lock(instance)["packs"]
