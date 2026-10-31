"""cms-comp-ops is a bundle of packs (ADR 0002 section 5; PACT r3-bundle-cms-comp-ops).

The packs are checked in tests/packs/. These tests hold the bundle: its
default and optional packs, what a render selects, where the comp-ops
playbooks live, and the chat site. The install proof is the bundle case in
tests/packs/test_pack_isolation.py.
"""
from pathlib import Path

import pytest
import yaml

from archi.bundles.render import render
from archi.bundles.resolve import resolve

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "bundles" / "cms-comp-ops"

OPTIONAL = ["monit-live", "dqm", "hypernews", "jira", "git-code"]
PLAYBOOKS = {
    "rucio_command": "rucio_command",
    "rucio_dataset_hosting": "rucio_dataset_hosting",
    "site_profile": "cms_site_profile",
    "monitoring_condor_snapshot": "cms_condor_monitoring_snapshot",
    "incident_root_cause": "cms_incident_root_cause",
}


def _bundle():
    return yaml.safe_load((BUNDLE / "bundle.yaml").read_text())


@pytest.fixture(scope="module")
def catalog():
    from packs.pack_support import shipped_catalog

    return shipped_catalog()


@pytest.fixture(scope="module")
def bare(catalog, tmp_path_factory):
    return render(resolve(catalog, ["cms-comp-ops"]), tmp_path_factory.mktemp("bare"))


@pytest.fixture(scope="module")
def full(catalog, tmp_path_factory):
    return render(resolve(catalog, ["cms-comp-ops"], with_packs=OPTIONAL), tmp_path_factory.mktemp("full"))


def test_the_packs_are_the_pact_selection():
    manifest = _bundle()
    assert manifest["extends"] == ["cern-team"]
    assert manifest["packs"]["default"] == ["cms-catalogs", "cmssst", "cmssw-releases", "ops-record"]
    assert manifest["packs"]["optional"] == OPTIONAL


def test_the_bundle_is_only_a_manifest_and_a_readme():
    assert sorted(p.name for p in BUNDLE.iterdir()) == ["README.md", "bundle.yaml"]


def test_the_bundle_validates(catalog):
    assert "cms-comp-ops" in catalog.bundles


def test_a_bare_render_adds_the_defaults_to_cern_teams(bare):
    names = set(bare.composition.sources)
    assert {"docsite.docsite", "indico.indico_live"} <= names
    for prefix in ("cms-catalogs.", "cmssst.", "cmssw-releases."):
        assert any(n.startswith(prefix) for n in names), prefix
    assert not any(n.startswith(("monit-live.", "dqm.", "hypernews.", "jira.")) for n in names)


def test_every_optional_pack_renders(full):
    names = set(full.composition.sources)
    for prefix in ("monit-live.", "dqm.", "hypernews.", "jira."):
        assert any(n.startswith(prefix) for n in names), prefix


def test_the_comp_ops_playbooks_ride_with_their_packs(full):
    skills = full.profile_dir / "skills"
    triggers = yaml.safe_load((skills / "skill-triggers.yaml").read_text())["intents"]
    library = yaml.safe_load((ROOT / "skills" / "skill-triggers.yaml").read_text())["intents"]
    for skill, intent in PLAYBOOKS.items():
        assert (skills / f"{skill}.md").is_file(), skill
        assert triggers[intent] == library[intent], intent


def test_a_playbook_is_absent_without_its_pack(bare):
    skills = bare.profile_dir / "skills"
    assert (skills / "rucio_command.md").is_file()
    assert not (skills / "monitoring_condor_snapshot.md").exists()


def test_the_chat_site_is_comp_ops(bare):
    site = next(q for q in bare.composition.questions if q["id"] == "chat_site_name")
    assert site["default"] == "CMS computing operations chat"
    defaults = yaml.safe_load((bare.profile_dir / "deployment-defaults.yaml").read_text())
    preset = defaults["chat"]["preset"]
    assert "computing operations" in preset["description"]
    # cern-team's suggestions come first, then the bundle's own.
    assert [s for s in preset["suggestions"] if "storage endpoints" in s or "downtimes" in s]


def test_the_incident_playbook_is_not_in_ops_record():
    """ops-record is generic (cluster-manager defaults to it too), so the CMS
    incident trigger lives with the CMS catalogs it reads."""
    assert not (ROOT / "bundles" / "packs" / "ops-record" / "skills" / "incident_root_cause.md").exists()
    assert (ROOT / "bundles" / "packs" / "cms-catalogs" / "playbooks" / "incident_root_cause.md").is_symlink()
