"""req.r3-ops-runtime.pack: the automation reaches the rendered instance and archi.lock,
so the runtime has something to read; `archi add` of an automation pack carries it too."""
import shutil
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.lock import add, diff, read_lock, render_and_lock
from archi.ops.levels import effective_level

from bundles.bundle_support import build_cern_team_catalog
from bundles.test_lock import BASE_ANSWERS, THIRD_PARTY_PACKS, _scaffold

LIBRARY = Path(__file__).resolve().parent / "fixtures" / "library"
ALPHA = Path(__file__).resolve().parents[1] / "bundles" / "fixtures" / "library" / "packs" / "alpha"


@pytest.fixture
def team(tmp_path):
    """A bundle of a context pack and fixture-triage (with its tool pack), bounded at L1."""
    root = tmp_path / "lib"
    shutil.copytree(LIBRARY, root)
    shutil.copytree(ALPHA, root / "packs" / "alpha")
    bdir = root / "bundles" / "triage-team"
    bdir.mkdir(parents=True)
    (bdir / "bundle.yaml").write_text(yaml.safe_dump({
        "name": "triage-team", "version": "0.1.0", "description": "t",
        "packs": {"default": ["alpha", "fixture-triage"]},
        "automations": {"fixture-triage": {"max_level": "L1"}},
        "okg": ">=0.4.0,<0.5",
    }, sort_keys=False))
    catalog = discovery.from_roots([root / "packs"], [root / "bundles"])
    assert catalog.problems == []
    return catalog


def test_render_writes_the_automation_for_the_runtime(team, tmp_path):
    _, rendered, lock = render_and_lock(team, tmp_path / "inst", ["triage-team"])
    written = yaml.safe_load((tmp_path / "inst" / "automations" / "fixture-triage.yaml").read_text())
    source = yaml.safe_load((LIBRARY / "packs" / "fixture-triage" / "pack.yaml").read_text())["automation"]

    assert written["pack"] == "fixture-triage"
    for key in ("trigger", "context_gate", "action_classes", "proposal_schema", "budget", "default_level"):
        assert written[key] == source[key]
    assert written["pack_max_level"] == "L2"
    assert written["bundle_max_level"] == "L1"
    # The playbook is where the renderer put it in the profile.
    assert written["playbook"] == "skills/triage.md"
    assert (rendered.profile_dir / written["playbook"]).is_file()
    assert sorted(written["action_class_definitions"]) == ["draft_reply", "set_ticket_status"]
    assert written["action_class_definitions"]["set_ticket_status"]["mediator"] == "archi.tools.fixture:set_ticket_status"


def test_lock_carries_the_automation_and_the_tool_it_needs(team, tmp_path):
    render_and_lock(team, tmp_path / "inst", ["triage-team"])
    lock = read_lock(tmp_path / "inst")
    auto = lock["composed"]["automations"]["fixture-triage"]
    assert auto["bundle_max_level"] == "L1"
    assert auto["action_classes"] == ["draft_reply", "set_ticket_status"]
    assert "archi.tools.fixture:set_ticket_status" in lock["composed"]["live_tools"]
    assert lock["packs"]["fixture-triage"]["kind"] == "automation"
    assert sorted(lock["composed"]["action_classes"]) == ["draft_reply", "set_ticket_status"]


def test_runtime_resolves_the_level_from_the_lock_and_the_instance(team, tmp_path):
    render_and_lock(team, tmp_path / "inst", ["triage-team"])
    auto = read_lock(tmp_path / "inst")["composed"]["automations"]["fixture-triage"]

    def level(instance_level):
        return effective_level(
            instance_level=instance_level, default_level=auto["default_level"],
            bundle_max=auto["bundle_max_level"], pack_max=auto["pack_max_level"],
        )

    assert level(None) == "L1"
    assert level("L2") == "L1"  # the bundle bound wins
    assert level("L0") == "L0"


def test_render_is_deterministic_and_drops_stale_files(team, tmp_path):
    render_and_lock(team, tmp_path / "a", ["triage-team"])
    render_and_lock(team, tmp_path / "b", ["triage-team"])
    assert read_lock(tmp_path / "a")["composed"] == read_lock(tmp_path / "b")["composed"]
    stale = tmp_path / "a" / "automations" / "gone.yaml"
    stale.write_text("x: 1\n")
    render_and_lock(team, tmp_path / "a", ["triage-team"])
    assert not stale.exists()


def test_two_bundles_take_the_lower_bound(team, tmp_path):
    root = Path(team.bundles["triage-team"].dir).parent
    other = root / "tight"
    other.mkdir()
    (other / "bundle.yaml").write_text(yaml.safe_dump({
        "name": "tight", "version": "0.1.0", "description": "t",
        "packs": {"default": ["alpha"], "optional": ["fixture-triage"]},
        "automations": {"fixture-triage": {"max_level": "L0"}},
        "okg": ">=0.4.0,<0.5",
    }))
    catalog = discovery.from_roots([root.parent / "packs"], [root])
    assert catalog.problems == []
    render_and_lock(catalog, tmp_path / "inst", ["triage-team", "tight"])
    auto = read_lock(tmp_path / "inst")["composed"]["automations"]["fixture-triage"]
    assert auto["bundle_max_level"] == "L0"


def test_a_pack_that_is_not_selected_leaves_no_automation(team, tmp_path):
    render_and_lock(team, tmp_path / "inst", ["triage-team"], without_packs=["fixture-triage"])
    assert read_lock(tmp_path / "inst")["composed"]["automations"] == {}
    assert not (tmp_path / "inst" / "automations").exists()


# ---- archi add of an automation pack ----------------------------------------


@pytest.fixture
def cern_with_ops(tmp_path):
    cern_root = tmp_path / "cern"
    cern_root.mkdir()
    build_cern_team_catalog(cern_root)
    catalog = discovery.from_roots(
        [cern_root / "packs", THIRD_PARTY_PACKS, LIBRARY / "packs"], [cern_root]
    )
    assert catalog.problems == []
    return catalog


def test_add_an_automation_pack_extends_the_instance_and_the_lock(cern_with_ops, tmp_path):
    instance = tmp_path / "instance"
    render_and_lock(cern_with_ops, instance, ["cern-team"])
    deployment = _scaffold(instance, "cern-team", BASE_ANSWERS)
    assert not (instance / "automations").exists()

    result = add(instance, cern_with_ops, "fixture-ops-watch", {}, deployment_dir=deployment,
                 run_okg=False, echo=lambda *_: None)
    # The pack has no sources; its required tool pack (monit-live) comes with its own.
    assert "monit-live.monit_sam" in result.added_sources
    assert not [s for s in result.added_sources if s.startswith("fixture-ops-watch")]

    lock = read_lock(instance)
    assert "fixture-ops-watch" in lock["selection"]["with"]
    auto = lock["composed"]["automations"]["fixture-ops-watch"]
    assert auto["action_classes"] == ["look_up_condor"]
    assert (instance / "automations" / "fixture-ops-watch.yaml").is_file()
    assert "archi.tools.monit:archi_monit_condor_search" in lock["composed"]["live_tools"]
    assert diff(instance, cern_with_ops).clean, diff(instance, cern_with_ops).as_dict()
