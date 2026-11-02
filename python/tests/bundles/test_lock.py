"""req.r3-bundle-format.lock: archi.lock, archi diff, archi add.

The deployment side runs against okg's own scaffold (build_plan +
planned_writes), which needs no database. The published round trip, `archi
add` running claim, catalog load, ingest and publish against Postgres, is the
executed test recorded in the PACT (scratch-instance run).
- acc.r3-bundle-format.diff: a hand edit to the registry shows in archi diff.
"""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.cli import main as archi_main
from archi.bundles.lock import (
    LOCK_FILE,
    LockError,
    add,
    diff,
    read_lock,
    render_and_lock,
    tree_digest,
)

from bundles.bundle_support import CATALOG, build_cern_team_catalog, load_catalog

THIRD_PARTY_PACKS = (
    Path(__file__).resolve().parent / "fixtures" / "thirdparty" / "archi_fixture_packs" / "packs"
)


@pytest.fixture(scope="module")
def cern_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("cern-team-packs")
    build_cern_team_catalog(root)
    return root


@pytest.fixture
def cern(cern_root):
    catalog = discovery.from_roots([cern_root / "packs", THIRD_PARTY_PACKS], [cern_root])
    assert catalog.problems == []
    return catalog


def _scaffold(instance: Path, name: str, answers: dict) -> Path:
    """What `okg install` would write, without a database."""
    from okg.substrate.deployment_bootstrap.profile_init import (
        build_plan,
        discover_profile,
        planned_writes,
    )

    plan = build_plan(
        profile=discover_profile(name, profiles_root=instance / "profiles"),
        deployment_name="qt",
        deployments_root=instance / "deployments",
        raw_answers=answers,
    )
    for write in planned_writes(plan):
        write.path.parent.mkdir(parents=True, exist_ok=True)
        if write.data is not None:
            write.path.write_bytes(write.data)
        else:
            write.path.write_text(write.text, encoding="utf-8")
    return plan.deployment_dir


BASE_ANSWERS = {
    "deployment_name": "qt",
    "postgres_dsn": "postgresql://localhost:1/qt",
    "archi_data_root": "/srv/data",
}


@pytest.fixture
def installed(cern, tmp_path):
    """A cern-team instance, rendered, locked and scaffolded."""
    instance = tmp_path / "instance"
    render_and_lock(cern, instance, ["cern-team"])
    deployment = _scaffold(instance, "cern-team", BASE_ANSWERS)
    return instance, deployment


# ---------------------------------------------------------------------------
# archi.lock
# ---------------------------------------------------------------------------


def test_render_writes_the_lock(tmp_path):
    catalog = load_catalog(CATALOG)
    render_and_lock(catalog, tmp_path, ["child-bundle"])
    lock = read_lock(tmp_path)
    assert lock["selection"] == {"bundles": ["child-bundle"], "with": [], "without": []}
    assert lock["okg"]["range"] == "<0.5,>=0.4.0"
    assert lock["okg"]["rendered_with"]
    assert {k: v["version"] for k, v in lock["bundles"].items()} == {
        "base-bundle": "0.1.0", "child-bundle": "0.1.0"
    }
    assert {k: v["version"] for k, v in lock["packs"].items()} == {"alpha": "0.1.0", "beta": "0.2.0"}
    assert lock["packs"]["alpha"]["digest"] == tree_digest(CATALOG / "packs" / "alpha")
    assert lock["packs"]["alpha"]["digest"].startswith("sha256:")
    files = lock["profile"]["files"]
    assert "profile.yaml" in files and "source-defaults/alpha.pages.yaml" in files
    assert lock["composed"]["freshness"] == {
        "alpha.pages": {"max_staleness": "24h"},
        "beta.issues": {"max_staleness": "24h"},
    }
    assert lock["composed"]["sources"] == {"alpha.pages": "alpha", "beta.issues": "beta"}
    assert lock["deployment"] is None


def test_lock_is_deterministic(tmp_path):
    catalog = load_catalog(CATALOG)
    render_and_lock(catalog, tmp_path / "a", ["child-bundle"])
    render_and_lock(catalog, tmp_path / "b", ["child-bundle"])
    a, b = read_lock(tmp_path / "a"), read_lock(tmp_path / "b")
    assert a == b


def test_render_command_writes_the_lock(tmp_path, capsys):
    rc = archi_main(["render", "--bundle", "base-bundle", "--instance", str(tmp_path),
                     "--pack-root", str(CATALOG / "packs"), "--bundle-root", str(CATALOG)])
    assert rc == 0
    assert (tmp_path / LOCK_FILE).is_file()


# ---------------------------------------------------------------------------
# archi diff
# ---------------------------------------------------------------------------


def test_diff_is_clean_after_render_and_install(cern, installed):
    instance, deployment = installed
    drift = diff(instance, cern, deployment)
    assert drift.clean, drift.as_dict()


def test_a_hand_edit_to_the_registry_shows_in_diff(cern, installed):
    instance, deployment = installed
    registry = deployment / "source_registry.yaml"
    data = yaml.safe_load(registry.read_text())
    data["sources"]["docsite.docsite"]["params"]["required"] = False
    registry.write_text(yaml.safe_dump(data, sort_keys=False))
    drift = diff(instance, cern, deployment)
    assert drift.deployment == [
        "source docsite.docsite: registry differs from the profile in ['params']"
    ]


def test_a_source_removed_from_or_added_to_the_registry_shows_in_diff(cern, installed):
    instance, deployment = installed
    registry = deployment / "source_registry.yaml"
    data = yaml.safe_load(registry.read_text())
    entry = data["sources"].pop("docsite.docsite")
    data["sources"]["handmade"] = entry
    registry.write_text(yaml.safe_dump(data, sort_keys=False))
    drift = diff(instance, cern, deployment)
    assert drift.deployment == [
        "source docsite.docsite: in the profile, not in the registry",
        "source handmade: in the registry, not in the profile",
    ]


def test_a_hand_edit_to_the_rendered_profile_shows_in_diff(cern, installed):
    instance, _ = installed
    target = instance / "profiles" / "cern-team" / "invariants.yaml"
    target.write_text(target.read_text() + "\n# local tweak\n")
    (instance / "profiles" / "cern-team" / "extra.yaml").write_text("x: 1\n")
    drift = diff(instance, cern)
    assert drift.profile == [
        "profiles/cern-team/extra.yaml: not rendered by archi",
        "profiles/cern-team/invariants.yaml: edited since render",
    ]


def test_an_upstream_pack_change_shows_in_diff(cern_root, installed, tmp_path):
    instance, _ = installed
    changed = tmp_path / "changed"
    shutil.copytree(cern_root, changed)
    manifest = changed / "packs" / "docsite" / "pack.yaml"
    manifest.write_text(manifest.read_text().replace("version: 0.2.0", "version: 9.9.9"))
    playbook = next((changed / "packs" / "core-playbooks" / "playbooks").glob("*.md"))
    playbook.write_text(playbook.read_text() + "\nmore\n")
    catalog = discovery.from_roots([changed / "packs", THIRD_PARTY_PACKS], [changed])
    drift = diff(instance, catalog)
    assert drift.upstream == [
        "pack docsite: version 0.2.0 locked, 9.9.9 installed",
        "pack core-playbooks: content changed at version 0.1.0",
    ]


def test_diff_command_exit_codes(cern_root, installed, capsys):
    instance, deployment = installed
    roots = ["--pack-root", str(cern_root / "packs"), "--pack-root", str(THIRD_PARTY_PACKS),
             "--bundle-root", str(cern_root)]
    assert archi_main(["diff", "--instance", str(instance), "--deployment", str(deployment), *roots]) == 0
    (deployment / "source_registry.yaml").write_text(
        (deployment / "source_registry.yaml").read_text().replace(
            "/docsite/records.json", "/docsite/other.json"
        )
    )
    capsys.readouterr()
    rc = archi_main(["diff", "--instance", str(instance), "--deployment", str(deployment),
                     "--json", *roots])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["deployment"]
    assert archi_main(["diff", "--instance", str(instance / "nowhere"), *roots]) == 2


# ---------------------------------------------------------------------------
# archi add (deployment edits; the okg steps are the executed scratch run)
# ---------------------------------------------------------------------------


def test_add_extends_an_installed_deployment(cern, installed):
    instance, deployment = installed
    result = add(instance, cern, "repo-mirror", {"repo_mirror_url": "file:///srv/demo.git"},
                 deployment_dir=deployment, run_okg=False, echo=lambda *_: None)
    assert result.added_sources == ["repo-mirror.repo"]

    registry = yaml.safe_load((deployment / "source_registry.yaml").read_text())["sources"]
    assert sorted(registry) == ["docsite.docsite", "indico.indico_live", "repo-mirror.repo"]
    assert registry["repo-mirror.repo"]["params"]["url"] == "file:///srv/demo.git"
    assert registry["repo-mirror.repo"]["params"]["repo_root"] == "/srv/data/repos/mirror"
    manifest = yaml.safe_load((deployment / "deployment.yaml").read_text())
    assert "git_graph" in manifest["modules"]
    assert "source_file" in manifest["search"]["profiles"]["content"]["subtypes"]
    assert manifest["profile"]["init_answers"]["repo_mirror_url"] == "file:///srv/demo.git"

    lock = read_lock(instance)
    assert lock["selection"]["with"] == ["repo-mirror"]
    assert "repo-mirror" in lock["packs"]
    assert lock["deployment"] == str(deployment.resolve())
    # The lock now names the deployment, so diff needs no --deployment.
    assert diff(instance, cern).clean, diff(instance, cern).as_dict()


def test_add_copies_new_schema_slices_and_files(cern, tmp_path):
    instance = tmp_path / "instance"
    render_and_lock(cern, instance, ["cern-team"], without_packs=["docsite", "indico"],
                    with_packs=["repo-mirror"])
    deployment = _scaffold(instance, "cern-team",
                           {**BASE_ANSWERS, "repo_mirror_url": "file:///srv/demo.git"})
    assert not (deployment / "schemas" / "sources.yaml").exists()
    # code-text brings git-code, which it requires, and extractors.yaml.
    add(instance, cern, "code-text", {"git_code_repo_url": "https://example.org/x.git"},
        deployment_dir=deployment, run_okg=False, echo=lambda *_: None)
    assert (deployment / "schemas" / "sources.yaml").is_file()
    assert (deployment / "extractors.yaml").is_file()
    registry = yaml.safe_load((deployment / "source_registry.yaml").read_text())["sources"]
    assert {"git-code.repo", "git-code.git_history", "code-text.code_text"} <= set(registry)


def test_add_refuses_a_pack_already_installed(cern, installed):
    instance, deployment = installed
    with pytest.raises(LockError, match="already in this instance"):
        add(instance, cern, "docsite", {}, deployment_dir=deployment, run_okg=False)


def test_add_refuses_to_overwrite_a_diverged_file(cern, tmp_path):
    instance = tmp_path / "instance"
    render_and_lock(cern, instance, ["cern-team"])
    deployment = _scaffold(instance, "cern-team", BASE_ANSWERS)
    # jira ships the same shared slice docsite and indico installed.
    (deployment / "schemas" / "sources.yaml").write_text("locally: changed\n")
    with pytest.raises(LockError, match="sources.yaml differs"):
        add(instance, cern, "jira", {}, deployment_dir=deployment, run_okg=False,
            echo=lambda *_: None)


def test_add_needs_a_deployment(cern, tmp_path):
    instance = tmp_path / "instance"
    render_and_lock(cern, instance, ["cern-team"])
    with pytest.raises(LockError, match="pass --deployment"):
        add(instance, cern, "repo-mirror", {}, run_okg=False)
