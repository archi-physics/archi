"""The loop's configuration comes from the rendered automation block: render the
fixture-triage pack, load `<instance>/automations/fixture-triage.yaml`, and run
the loop on it."""
import shutil
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.lock import render_and_lock
from archi.ops.loop import Deps, Signal, load_automation, make_run_id, run_once
from archi.ops.proposers import StubProposer
from loop_fakes import PASS, Budget, Clock, Decider, Gate, Halt, Mediator, Outcome, Reader, Recorder

LIBRARY = Path(__file__).resolve().parent / "fixtures" / "library"
ALPHA = Path(__file__).resolve().parents[1] / "bundles" / "fixtures" / "library" / "packs" / "alpha"


@pytest.fixture
def instance(tmp_path):
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
    render_and_lock(catalog, tmp_path / "inst", ["triage-team"])
    return tmp_path / "inst"


def test_config_is_built_from_the_rendered_block(instance):
    source = yaml.safe_load((LIBRARY / "packs" / "fixture-triage" / "pack.yaml").read_text())["automation"]
    a = load_automation(instance, "fixture-triage", instance_level="L2", mode="live")
    assert a.name == "fixture-triage" and a.mode == "live"
    assert (a.instance_level, a.bundle_max_level, a.pack_max_level, a.default_level) == ("L2", "L1", "L2", "L1")
    assert a.proposal_schema == source["proposal_schema"]
    assert a.gate.sources == ("fixture-tools.status",) and a.gate.min_context_eval_score == 0.8
    assert (a.cost_per_run, a.cost_per_day) == (0.5, 10)
    assert sorted(a.action_classes) == ["draft_reply", "set_ticket_status"]
    assert a.action_classes["set_ticket_status"].max_level == "L2"
    assert a.action_classes["set_ticket_status"].mediator == "archi.tools.fixture:set_ticket_status"
    assert a.action_classes["draft_reply"].payload_schema["required"] == ["ticket_key", "body"]


def run(instance, proposal, level="L2"):
    automation = load_automation(instance, "fixture-triage", instance_level=level, mode="live")
    clock = Clock()
    mediator = Mediator()
    deps = Deps(Gate(PASS), Reader(), StubProposer({"default": proposal}), Decider(clock, None), mediator,
                Outcome(), Recorder(), Halt(False), clock, Budget())
    result = run_once(automation, Signal("sig-1", "T2_US_MIT", "Site down"), deps,
                      run_id=make_run_id("fixture-triage", "sig-1", 1))
    return result, deps, mediator


FULL = {
    "summary": "reply to the ticket", "evidence_ids": ["jira:A-1"], "action_class": "draft_reply",
    "payload": {"ticket_key": "A-1", "body": "We looked."}, "confidence": 0.7, "rollback": "delete the draft",
}


def test_the_bundle_bound_holds_the_run_at_l1_and_the_draft_is_recorded(instance):
    result, deps, mediator = run(instance, FULL, level="L2")
    assert (result.level, result.status) == ("L1", "drafted")
    assert mediator.calls == []
    assert [c[0] for c in deps.recorder.calls] == ["incident", "proposal", "decision"]


def test_the_rendered_payload_schema_is_enforced(instance):
    bad = {**FULL, "payload": {"ticket_key": "A-1"}}
    result, _, mediator = run(instance, bad)
    assert result.status == "invalid_proposal" and "'body' is a required property" in result.reason
    assert mediator.calls == []


def test_the_rendered_proposal_schema_is_enforced(instance):
    result, _, _ = run(instance, {k: v for k, v in FULL.items() if k != "rollback"})
    assert result.status == "invalid_proposal" and "'rollback' is a required property" in result.reason
