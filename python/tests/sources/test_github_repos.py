"""req.w2.sources-catalogs — GitHubRepoSource emission, offline."""

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.sources.github_repos import DEFAULT_REPOS, GitHubRepoSource


def test_repo_identity_emission():
    source = GitHubRepoSource(repos=["dmwm/WMCore", "cms-sw/cmssw"])
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    assert set(nodes) == {
        "software_repository:dmwm/WMCore",
        "repo:dmwm/WMCore",
        "software_repository:cms-sw/cmssw",
        "repo:cms-sw/cmssw",
    }
    sw = nodes["software_repository:dmwm/WMCore"]
    assert sw.subtype == "software_repository"
    assert sw.attrs["name"] == "WMCore"
    assert sw.attrs["full_name"] == "dmwm/WMCore"
    assert sw.attrs["url"] == "https://github.com/dmwm/WMCore"
    repo = nodes["repo:dmwm/WMCore"]
    assert repo.subtype == "repo"
    assert repo.attrs["slug"] == "dmwm/WMCore"
    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    assert edges == {
        ("software_repository:dmwm/WMCore", "references", "repo:dmwm/WMCore"),
        (
            "software_repository:cms-sw/cmssw",
            "references",
            "repo:cms-sw/cmssw",
        ),
    }


def test_defaults_dedupe_and_invalid_slugs():
    source = GitHubRepoSource(
        repos=["dmwm/WMCore", "dmwm/WMCore", "not-a-slug", ""]
    )
    facts = list(source.run("run-1").facts)
    nodes = [f for f in facts if isinstance(f, NodeFact)]
    assert len(nodes) == 2  # one pair; dup and invalid dropped
    default = GitHubRepoSource()
    assert default.repos == DEFAULT_REPOS
    result = default.preflight()
    assert result.status == "ok"
    assert result.mode == "registry_seed"
    assert result.record_count == len(DEFAULT_REPOS)


# --- reference-catalog-complete-scope regressions ---
# okg runs a reference_catalog source in release_new (chosen automatically)
# or, when a mode is set explicitly, release_unchanged; both complete the
# scope. The source reads no cache; it claims the scope only when at least
# one configured slug is valid.


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_reference_catalog_modes_claim_a_complete_scope(mode):
    run = GitHubRepoSource(repos=["dmwm/WMCore"]).run("run-1", mode=mode)
    assert run.completed_scope is True
    assert run.run_mode == mode


def test_cursor_mode_still_claims_no_scope():
    run = GitHubRepoSource(repos=["dmwm/WMCore"]).run("run-1", mode="cursor")
    assert run.completed_scope is False


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_all_invalid_slugs_claim_no_scope(mode):
    # Zero records under a complete scope would retract every repository
    # the source ever emitted (deletion_semantics: missing_from_completed_scope).
    run = GitHubRepoSource(repos=["not-a-slug", " "]).run("run-1", mode=mode)
    assert list(run.facts) == []
    assert run.completed_scope is False


def test_explicit_empty_repos_is_refused():
    # An explicit empty list used to fall back to DEFAULT_REPOS silently; a
    # registry that names no repository is a configuration error.
    with pytest.raises(ValueError, match="repos"):
        GitHubRepoSource(repos=[])


def test_missing_repos_keeps_the_defaults():
    assert GitHubRepoSource(repos=None).repos == DEFAULT_REPOS
