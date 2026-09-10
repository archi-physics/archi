"""req.w2.sources-catalogs — GitHubRepoSource emission, offline."""
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
