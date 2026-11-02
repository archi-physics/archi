"""acc.r3-ops-runtime.no-l3 (validate half): no place that can set a level accepts
L3 or an unknown level, and the effective level is the lowest of the inputs
(ADR 0002 sections 7.3, D13). Also pins the workflow-step registry."""
import shutil
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.manifest import validate_bundle
from archi.ops import steps
from archi.ops.levels import (
    LevelError,
    effective_level,
    min_level,
    resolve_level,
    settings_problems,
)
from archi.stack.instance import Instance, InstanceError

LIBRARY = Path(__file__).resolve().parent / "fixtures" / "library"


# ---- effective_level -------------------------------------------------------


def eff(**kw):
    kw.setdefault("default_level", "L1")
    kw.setdefault("pack_max", "L2")
    return effective_level(**kw)


def test_instance_level_applies_when_nothing_bounds_it():
    assert eff(instance_level="L2") == "L2"


def test_pack_default_stands_in_for_an_instance_with_no_level():
    assert eff() == "L1"


def test_bundle_bound_lowers_the_instance_level():
    assert eff(instance_level="L2", bundle_max="L1") == "L1"
    assert resolve_level(instance_level="L2", default_level="L1", bundle_max="L1", pack_max="L2").limited_by == "bundle"


def test_pack_maximum_lowers_the_instance_level():
    assert eff(instance_level="L2", bundle_max="L2", pack_max="L0", default_level="L0") == "L0"
    assert resolve_level(instance_level="L2", default_level="L0", pack_max="L0").limited_by == "pack"


def test_effective_level_is_the_minimum_of_all_inputs():
    levels = ("L0", "L1", "L2")
    for i in levels:
        for b in levels:
            for p in levels:
                assert eff(instance_level=i, bundle_max=b, pack_max=p, default_level="L0") == min_level(i, b, p)


@pytest.mark.parametrize("where", ["instance_level", "bundle_max", "pack_max", "default_level"])
@pytest.mark.parametrize("bad", ["L3", "L4", "l1", ""])
def test_effective_level_refuses_l3_and_unknown_levels_everywhere(where, bad):
    with pytest.raises(LevelError, match="ADR 0002 D13"):
        eff(**{where: bad})


def test_l3_is_refused_even_when_a_lower_bound_would_hide_it():
    with pytest.raises(LevelError, match="L3 is refused"):
        eff(instance_level="L3", bundle_max="L0")


# ---- bundle.yaml: automations.<name>.max_level -----------------------------


def library_with_bundle(tmp_path, bounds):
    root = tmp_path / "lib"
    shutil.copytree(LIBRARY, root)
    bdir = root / "bundles" / "triage-team"
    bdir.mkdir(parents=True)
    # a context pack so the bundle can publish
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "bundles" / "fixtures" / "library" / "packs" / "alpha",
        root / "packs" / "alpha",
    )
    (bdir / "bundle.yaml").write_text(yaml.safe_dump({
        "name": "triage-team", "version": "0.1.0", "description": "t",
        "packs": {"default": ["alpha", "fixture-tools", "fixture-triage"]},
        "okg": ">=0.4.0,<0.5",
        "automations": bounds,
    }, sort_keys=False))
    return root


def bundle_messages(root):
    catalog = discovery.from_roots([root / "packs"], [root / "bundles"])
    problems = [p.message for p in catalog.problems]
    if "triage-team" in catalog.bundles:
        problems += [p.message for p in validate_bundle(catalog.bundles["triage-team"].dir, catalog.packs, catalog.bundles)]
    return problems


def test_bundle_bound_is_accepted(tmp_path):
    root = library_with_bundle(tmp_path, {"fixture-triage": {"max_level": "L1"}})
    assert bundle_messages(root) == []
    catalog = discovery.from_roots([root / "packs"], [root / "bundles"])
    assert catalog.bundles["triage-team"].automation_bounds == {"fixture-triage": "L1"}


@pytest.mark.parametrize("level", ["L3", "L4", "high"])
def test_bundle_bound_above_l2_is_refused_citing_d13(tmp_path, level):
    root = library_with_bundle(tmp_path, {"fixture-triage": {"max_level": level}})
    found = bundle_messages(root)
    assert len(found) == 1 and "ADR 0002 D13" in found[0] and "automations.fixture-triage.max_level" in found[0]


def test_bundle_bound_for_a_non_automation_pack_is_refused(tmp_path):
    root = library_with_bundle(tmp_path, {"alpha": {"max_level": "L1"}})
    assert "pack 'alpha' is not an automation pack" in "; ".join(bundle_messages(root))


def test_bundle_bound_for_an_unknown_pack_is_refused(tmp_path):
    root = library_with_bundle(tmp_path, {"nothing": {"max_level": "L1"}})
    assert "unknown automation pack 'nothing'" in "; ".join(bundle_messages(root))


def test_bundle_bound_needs_max_level_only(tmp_path):
    root = library_with_bundle(tmp_path, {"fixture-triage": {"level": "L1"}})
    assert any("max_level" in m for m in bundle_messages(root))


# ---- archi.yaml: automations.<name>.level ----------------------------------


def write_instance(tmp_path, automations):
    data = {
        "instance_version": 1, "name": "demo", "bundles": ["cern-team"],
        "data_root": str(tmp_path / "data"), "runtime": "docker",
        "ports": {"graph_db": 1, "mcp": 2, "webui": 3},
    }
    if automations is not None:
        data["automations"] = automations
    (tmp_path / "archi.yaml").write_text(yaml.safe_dump(data))
    return tmp_path


def test_instance_level_is_read_and_saved(tmp_path):
    inst = Instance.load(write_instance(tmp_path, {"ticket-triage": {"level": "L1"}}))
    assert inst.automations == {"ticket-triage": {"level": "L1"}}
    inst.save()
    assert Instance.load(tmp_path).automations == {"ticket-triage": {"level": "L1"}}


def test_instance_without_automations_does_not_write_the_key(tmp_path):
    inst = Instance.load(write_instance(tmp_path, None))
    assert inst.automations == {}
    assert "automations" not in inst.to_dict()


@pytest.mark.parametrize("level", ["L3", "L4", "yes"])
def test_instance_level_above_l2_is_refused_citing_d13(tmp_path, level):
    write_instance(tmp_path, {"ticket-triage": {"level": level}})
    with pytest.raises(InstanceError, match="ADR 0002 D13"):
        Instance.load(tmp_path)


def test_instance_automation_setting_must_be_level_only(tmp_path):
    write_instance(tmp_path, {"ticket-triage": {"max_level": "L1"}})
    with pytest.raises(InstanceError, match="must be exactly"):
        Instance.load(tmp_path)


def test_settings_problems_names_the_path():
    assert settings_problems({"x": {"level": "L3"}}, "level", "automations") == [
        "automations.x.level: L3 is refused (ADR 0002 D13: 3.0 stops at L2, L3 is reserved and refused)"
    ]


# ---- the workflow-step registry --------------------------------------------


def test_step_registry_is_pinned():
    """A new workflow step is a decision: it must have no external effect.
    Change this list only together with ADR 0002 section 7.2."""
    assert {n: steps.get_step(n).max_level for n in steps.step_names()} == {
        "record_finding": "L0",
        "draft_reply": "L1",
        # PACT r3-ops-self-monitoring: a draft issue file in the instance repository; no network, no forge
        "draft_issue": "L1",
    }


def test_only_steps_that_stay_inside_the_instance_repository_have_an_implementation():
    """A step with an implementation runs at L1 with no escalation, so each one is a decision."""
    assert {n for n in steps.step_names() if steps.get_step(n).impl} == {"draft_issue"}
    assert steps.local_step("draft_issue").impl == "archi.health.steps:draft_issue"
    assert steps.local_step("draft_reply") is None
    assert steps.local_step("archi.tools.instance_health:sync_forge_issues") is None


def test_registering_a_step_above_l1_is_refused():
    with pytest.raises(ValueError, match="no external effect"):
        steps.register_step("restart_node", "L2", "x")
    assert "restart_node" not in steps.step_names()


def test_registering_a_step_twice_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        steps.register_step("draft_reply", "L1", "x")


def test_registering_a_step_with_l3_is_refused():
    with pytest.raises(ValueError, match="ADR 0002 D13"):
        steps.register_step("x", "L3", "x")


# ---- instance names against the selected packs ------------------------------


def rendered_instance(tmp_path, automations):
    from archi.bundles.lock import render_and_lock

    root = library_with_bundle(tmp_path, {"fixture-triage": {"max_level": "L1"}})
    catalog = discovery.from_roots([root / "packs"], [root / "bundles"])
    inst = tmp_path / "inst"
    render_and_lock(catalog, inst, ["triage-team"])
    data = {
        "instance_version": 1, "name": "demo", "bundles": ["triage-team"],
        "data_root": str(tmp_path / "data"), "runtime": "docker",
        "ports": {"graph_db": 1, "mcp": 2, "webui": 3},
        "automations": automations,
    }
    (inst / "archi.yaml").write_text(yaml.safe_dump(data))
    return inst


def test_instance_levels_resolve_for_the_runtime(tmp_path):
    from archi.ops.levels import instance_levels

    inst = rendered_instance(tmp_path, {"fixture-triage": {"level": "L2"}})
    got = instance_levels(inst)["fixture-triage"]
    assert (got.level, got.limited_by) == ("L1", "bundle")


def test_instance_with_a_misspelled_automation_is_refused_for_the_runtime(tmp_path):
    from archi.ops.levels import instance_levels

    inst = rendered_instance(tmp_path, {"fixture-triaje": {"level": "L1"}})
    with pytest.raises(InstanceError, match="'fixture-triaje', which no selected automation pack declares"):
        instance_levels(inst)


def test_duplicate_key_in_archi_yaml_is_refused(tmp_path):
    write_instance(tmp_path, None)
    path = tmp_path / "archi.yaml"
    path.write_text(path.read_text() + "runtime: podman\n")
    with pytest.raises(InstanceError, match="duplicate key 'runtime'"):
        Instance.load(tmp_path)


def test_duplicate_key_in_archi_lock_is_refused(tmp_path):
    from archi.bundles.lock import LockError, read_lock

    inst = rendered_instance(tmp_path, {})
    path = inst / "archi.lock"
    path.write_text(path.read_text() + "lock_version: 1\n")
    with pytest.raises(LockError, match="duplicate key 'lock_version'"):
        read_lock(inst)
