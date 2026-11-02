"""req.r3-ops-runtime.pack: automation pack manifests.

acc.r3-ops-runtime.schema: valid and invalid automation packs. Each invalid
case copies the fixture pack, breaks one thing and asserts that problem.
acc.r3-ops-runtime.no-l3: validate refuses any level above L2 (ADR 0002 D13).
"""
import shutil
from pathlib import Path

import pytest
import yaml

from archi.bundles.cli import main as archi_main
from archi.bundles.manifest import load_pack, validate_pack

LIBRARY = Path(__file__).resolve().parent / "fixtures" / "library"
SHIPPED = Path(__file__).resolve().parents[3] / "bundles" / "packs"


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "library"
    shutil.copytree(LIBRARY, root)
    return root


def edit(root, change, *, pack="fixture-triage", file="pack.yaml"):
    path = root / "packs" / pack / file
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def messages(root, pack="fixture-triage"):
    """Problems with a pack, resolved against the packs next to it as a catalog."""
    packs = {
        d.name: load_pack(d, defer_requires=True)
        for d in (root / "packs").iterdir()
        if d.name != pack and not validate_pack(d, defer_requires=True)
    }
    return [p.message for p in validate_pack(root / "packs" / pack, packs=packs)]


def auto(change):
    return lambda d: change(d["automation"])


def classes(change):
    return lambda d: change(d["action_classes"])


def only(found, needle):
    hits = [m for m in found if needle in m]
    assert len(hits) == 1, found
    return hits[0]


def test_fixture_packs_are_valid(library):
    assert messages(library) == []
    assert messages(library, "fixture-tools") == []


@pytest.mark.parametrize(
    "trigger",
    [
        {"type": "schedule", "cron": "*/15 * * * *"},
        {"type": "signal", "source": "fixture-tools.status", "condition": "error rate above 5%"},
    ],
)
def test_other_trigger_types_are_valid(library, trigger):
    edit(library, auto(lambda a: a.update(trigger=trigger)))
    assert messages(library) == []


def test_shipped_packs_still_validate():
    packs = sorted(SHIPPED.glob("*/pack.yaml"))
    assert packs
    for pack in packs:
        assert validate_pack(pack.parent) == [], pack


def test_cli_validates_the_fixture_pack(library, capsys):
    assert archi_main(["bundles", "validate", str(library / "packs" / "fixture-triage"), "--pack-root", str(library / "packs")]) == 0


def test_cli_reports_a_problem(library, capsys):
    edit(library, auto(lambda a: a.update(default_level="L3")))
    assert archi_main(["bundles", "validate", str(library / "packs" / "fixture-triage"), "--pack-root", str(library / "packs")]) == 1
    assert "D13" in capsys.readouterr().err


# ---- shape ----------------------------------------------------------------


def test_automation_block_is_required(library):
    edit(library, lambda d: d.pop("automation"))
    assert "must declare an `automation:` block" in only(messages(library), "automation")


def test_other_kinds_may_not_declare_an_automation(library):
    block = yaml.safe_load((library / "packs" / "fixture-triage" / "pack.yaml").read_text())["automation"]
    edit(library, lambda d: d.update(automation=block), pack="fixture-tools")
    assert "only a pack of kind `automation`" in only(messages(library, "fixture-tools"), "automation")


@pytest.mark.parametrize(
    "key", ["trigger", "context_gate", "playbook", "action_classes", "proposal_schema", "budget", "default_level", "max_level"]
)
def test_each_automation_key_is_required(library, key):
    edit(library, auto(lambda a: a.pop(key)))
    assert f"'{key}' is a required property" in only(messages(library), key)


def test_unknown_automation_key_is_refused(library):
    edit(library, auto(lambda a: a.update(color="red")))
    assert "Additional properties are not allowed" in only(messages(library), "color")


# ---- trigger --------------------------------------------------------------


def test_schedule_needs_cron(library):
    edit(library, auto(lambda a: a.update(trigger={"type": "schedule"})))
    assert "a schedule trigger needs `cron`" in only(messages(library), "trigger")


def test_cron_must_have_five_fields(library):
    edit(library, auto(lambda a: a.update(trigger={"type": "schedule", "cron": "every day"})))
    assert "not a five-field cron expression" in only(messages(library), "cron")


def test_ticket_queue_needs_queue(library):
    edit(library, auto(lambda a: a["trigger"].pop("queue")))
    assert "a ticket_queue trigger needs `queue`" in only(messages(library), "trigger")


def test_trigger_key_of_another_type_is_refused(library):
    edit(library, auto(lambda a: a["trigger"].update(cron="* * * * *")))
    assert "`cron` does not belong to a ticket_queue trigger" in only(messages(library), "cron")


def test_unknown_ticket_system_is_refused(library):
    edit(library, auto(lambda a: a["trigger"].update(system="trello")))
    assert "'trello' is not one of" in only(messages(library), "trello")


def test_unknown_trigger_type_is_refused(library):
    edit(library, auto(lambda a: a.update(trigger={"type": "webhook"})))
    assert "'webhook' is not one of" in only(messages(library), "webhook")


# ---- context gate ---------------------------------------------------------


def test_unknown_gate_key_is_refused(library):
    edit(library, auto(lambda a: a["context_gate"].update(freshness="1h")))
    assert "unknown gate keys ['freshness']" in only(messages(library), "gate")


def test_bad_gate_staleness_is_refused(library):
    edit(library, auto(lambda a: a["context_gate"].update(max_staleness="soon")))
    assert "max_staleness" in only(messages(library), "context_gate")


def test_gate_score_out_of_range_is_refused(library):
    edit(library, auto(lambda a: a["context_gate"].update(min_context_eval_score=1.5)))
    assert "must be a number from 0 to 1" in only(messages(library), "context_gate")


def test_empty_gate_is_refused(library):
    edit(library, auto(lambda a: a.update(context_gate={})))
    assert "declares no check" in only(messages(library), "context_gate")


# ---- playbook -------------------------------------------------------------


def test_missing_playbook_file_is_refused(library):
    (library / "packs" / "fixture-triage" / "playbooks" / "triage.md").unlink()
    assert "'triage.md' is not a file under 'playbooks/'" in only(messages(library), "playbook")


def test_playbook_cannot_leave_the_pack(library):
    edit(library, auto(lambda a: a.update(playbook="../pack.yaml")))
    assert "should not be valid under" in only(messages(library), "playbook")


def test_playbooks_directory_is_required(library):
    edit(library, lambda d: d.pop("playbooks"))
    assert "must declare a `playbooks:` directory" in only(messages(library), "playbooks")


# ---- action classes -------------------------------------------------------


def test_duplicate_action_class_id_is_refused(library):
    path = library / "packs" / "fixture-triage" / "action_classes.yaml"
    path.write_text(path.read_text() + "  draft_reply:\n    mediator: draft_reply\n")
    assert "duplicate key 'draft_reply'" in only(messages(library), "duplicate key")


def test_action_class_listed_twice_is_refused(library):
    edit(library, auto(lambda a: a["action_classes"].append("draft_reply")))
    assert "'draft_reply' is listed more than once" in only(messages(library), "listed more than once")


def test_undefined_action_class_is_refused(library):
    edit(library, auto(lambda a: a["action_classes"].append("restart_node")))
    assert "'restart_node' is not defined in action_classes.yaml" in only(messages(library), "restart_node")


def test_bad_action_class_id_is_refused(library):
    edit(library, classes(lambda c: c.update({"Bad-Id": c.pop("draft_reply")})), file="action_classes.yaml")
    edit(library, auto(lambda a: a.update(action_classes=["set_ticket_status"])))
    assert "action class id 'Bad-Id' must match" in only(messages(library), "Bad-Id")


def test_action_class_missing_a_field_is_refused(library):
    edit(library, classes(lambda c: c["draft_reply"].pop("blast_radius")), file="action_classes.yaml")
    assert "'blast_radius' is a required property" in only(messages(library), "blast_radius")


def test_reversible_must_be_boolean(library):
    edit(library, classes(lambda c: c["draft_reply"].update(reversible="yes")), file="action_classes.yaml")
    assert "reversible: 'yes' is not of type 'boolean'" in only(messages(library), "reversible")


def test_invalid_payload_schema_is_refused(library):
    edit(library, classes(lambda c: c["draft_reply"]["payload_schema"].update(type="strng")), file="action_classes.yaml")
    assert "payload_schema: not a valid JSON Schema" in only(messages(library), "payload_schema")


def test_payload_schema_must_be_an_object_schema(library):
    edit(library, classes(lambda c: c["draft_reply"].update(payload_schema={"type": "string"})), file="action_classes.yaml")
    assert "payload_schema: must be an object schema" in only(messages(library), "payload_schema")


def test_mediator_tool_must_be_declared(library):
    edit(library, classes(lambda c: c["set_ticket_status"].update(mediator="archi.tools.fixture:rm_rf")), file="action_classes.yaml")
    assert "is not a live tool of this pack or of a pack it requires" in only(messages(library), "rm_rf")


def test_mediator_tool_of_a_pack_not_required_is_refused(library):
    edit(library, lambda d: d.update(requires={"packs": []}))
    assert "is not a live tool of this pack or of a pack it requires" in only(messages(library), "set_ticket_status")


def test_mediator_tool_of_the_pack_itself_is_accepted(library):
    tool = "archi.tools.fixture:set_ticket_status"
    edit(library, lambda d: d.update(requires={"packs": []}, live_tools=[tool], idempotent_tools=[tool]))
    assert messages(library) == []


def test_mediator_must_be_a_known_workflow_step(library):
    edit(library, classes(lambda c: c["draft_reply"].update(mediator="shell")), file="action_classes.yaml")
    assert "neither a live tool entry (archi.tools.x:fn) nor a registered workflow step" in only(messages(library), "'shell'")


# ---- proposal schema ------------------------------------------------------


def test_proposal_schema_must_be_valid_json_schema(library):
    edit(library, auto(lambda a: a["proposal_schema"].update(type="strng")))
    assert "proposal_schema: not a valid JSON Schema" in only(messages(library), "proposal_schema")


def test_proposal_schema_must_require_the_ops_proposal_fields(library):
    edit(library, auto(lambda a: a["proposal_schema"]["required"].remove("rollback")))
    assert "`required` must list rollback" in only(messages(library), "proposal_schema")


def test_proposal_schema_may_not_invent_fields(library):
    edit(library, auto(lambda a: a["proposal_schema"]["properties"].update(mood={"type": "string"})))
    assert "property 'mood' is not an ops_proposal field" in only(messages(library), "mood")


# ---- budget ---------------------------------------------------------------


@pytest.mark.parametrize("key,value", [("cost_per_run", 0), ("cost_per_day", -1), ("runs_per_hour", 0)])
def test_budget_must_be_positive(library, key, value):
    edit(library, auto(lambda a: a["budget"].update({key: value})))
    assert f"automation.budget.{key}: must be greater than zero" in only(messages(library), key)


def test_budget_key_is_required(library):
    edit(library, auto(lambda a: a["budget"].pop("runs_per_hour")))
    assert "'runs_per_hour' is a required property" in only(messages(library), "runs_per_hour")


def test_run_cost_may_not_exceed_day_cost(library):
    edit(library, auto(lambda a: a["budget"].update(cost_per_run=20)))
    assert "cost_per_run exceeds cost_per_day" in only(messages(library), "exceeds")


# ---- levels: acc.r3-ops-runtime.no-l3 -------------------------------------


@pytest.mark.parametrize("level", ["L0", "L1", "L2"])
def test_levels_up_to_l2_are_accepted(library, level):
    edit(library, auto(lambda a: a.update(default_level=level, max_level="L2")))
    assert messages(library) == []


@pytest.mark.parametrize("key", ["default_level", "max_level"])
def test_automation_l3_is_refused_citing_d13(library, key):
    edit(library, auto(lambda a: a.update({key: "L3"})))
    found = messages(library)
    assert "L3 is refused" in only(found, f"automation.{key}")
    assert "ADR 0002 D13" in only(found, f"automation.{key}")


def test_action_class_l3_is_refused_citing_d13(library):
    edit(library, classes(lambda c: c["set_ticket_status"].update(max_level="L3")), file="action_classes.yaml")
    found = messages(library)
    assert "max_level: L3 is refused (ADR 0002 D13" in only(found, "set_ticket_status")


@pytest.mark.parametrize("level", ["L4", "l2", "high", ""])
def test_unknown_level_is_refused_citing_d13(library, level):
    edit(library, auto(lambda a: a.update(max_level=level)))
    assert "unknown level" in only(messages(library), "max_level")
    assert "ADR 0002 D13" in only(messages(library), "max_level")


def test_unknown_action_class_level_is_refused(library):
    edit(library, classes(lambda c: c["draft_reply"].update(max_level="L9")), file="action_classes.yaml")
    assert "unknown level 'L9' is refused" in only(messages(library), "L9")


def test_default_level_may_not_exceed_max_level(library):
    edit(library, auto(lambda a: a.update(default_level="L2", max_level="L1")))
    assert "default_level L2 is above max_level L1" in only(messages(library), "above max_level")


# ---- workflow steps are not external actions ------------------------------


def test_workflow_step_cannot_carry_l2(library):
    """A step has no external effect, so an action that runs after approval is a live tool."""
    edit(library, classes(lambda c: c["draft_reply"].update(max_level="L2")), file="action_classes.yaml")
    found = only(messages(library), "draft_reply")
    assert "max_level L2 is above L1, the highest level of workflow step 'draft_reply'" in found


def test_record_finding_is_l0_only(library):
    edit(library, classes(lambda c: c["draft_reply"].update(mediator="record_finding", max_level="L1")), file="action_classes.yaml")
    assert "above L0, the highest level of workflow step 'record_finding'" in only(messages(library), "record_finding")


def test_external_change_is_not_a_workflow_step(library):
    edit(library, classes(lambda c: c["draft_reply"].update(mediator="update_ticket")), file="action_classes.yaml")
    found = only(messages(library), "update_ticket")
    assert "must be a live tool of a tool pack" in found


# ---- cron against the run budget ------------------------------------------


@pytest.mark.parametrize(
    "cron,fires",
    [
        ("0 * * * *", 1),
        ("*/15 * * * *", 4),
        ("5,35 9-17 * * 1-5", 2),
        ("* * * * *", 60),
        ("0-10/5 3 * * *", 3),
        ("30 */6 1 * *", 1),
    ],
)
def test_cron_runs_per_hour(cron, fires):
    from archi.bundles.automation import cron_runs_per_hour

    assert cron_runs_per_hour(cron) == fires


@pytest.mark.parametrize("cron", ["@daily", "0 0 * * MON-FRI extra", "MON * * * *", "61 * * * *", "*/0 * * * *", "30-10 * * * *"])
def test_cron_forms_not_handled_are_refused(cron):
    from archi.bundles.automation import cron_runs_per_hour

    with pytest.raises(ValueError):
        cron_runs_per_hour(cron)


def test_cron_faster_than_the_budget_is_refused(library):
    edit(library, auto(lambda a: a.update(trigger={"type": "schedule", "cron": "*/5 * * * *"})))
    assert "can fire 12 times an hour, over budget.runs_per_hour 20" not in messages(library)
    edit(library, auto(lambda a: a["budget"].update(runs_per_hour=10)))
    assert "can fire 12 times an hour, over budget.runs_per_hour 10" in only(messages(library), "cron")


def test_cron_within_the_budget_is_accepted(library):
    edit(library, auto(lambda a: a.update(trigger={"type": "schedule", "cron": "*/5 * * * *"})))
    assert messages(library) == []


def test_cron_with_names_is_refused_as_not_handled(library):
    edit(library, auto(lambda a: a.update(trigger={"type": "schedule", "cron": "0 MON * * *"})))
    assert "must be numbers, ranges, lists or steps" in only(messages(library), "cron")


# ---- proposal schema field types ------------------------------------------


@pytest.mark.parametrize(
    "field,patch,needle",
    [
        ("evidence_ids", {"type": "string"}, "'evidence_ids' must be an array of strings"),
        ("evidence_ids", {"type": "array", "items": {"type": "integer"}}, "'evidence_ids' must be an array of strings"),
        ("confidence", {"type": "string"}, "'confidence' must be a number within [0, 1]"),
        ("confidence", {"type": "number"}, "'confidence' must be a number within [0, 1]"),
        ("confidence", {"type": "number", "minimum": 0, "maximum": 5}, "'confidence' must be a number within [0, 1]"),
        ("payload", {"type": "string"}, "'payload' must have `type: object`"),
        ("summary", {"minLength": 1}, "'summary' must have `type: string`"),
    ],
)
def test_proposal_field_types_are_required(library, field, patch, needle):
    edit(library, auto(lambda a: a["proposal_schema"]["properties"].update({field: patch})))
    assert needle in only(messages(library), "proposal_schema")


def test_proposal_field_must_be_declared(library):
    edit(library, auto(lambda a: a["proposal_schema"]["properties"].pop("rollback")))
    assert "property 'rollback' must have `type: string`" in only(messages(library), "rollback")


# ---- duplicate keys in pack.yaml ------------------------------------------


def test_duplicate_key_in_the_automation_block_is_refused(library):
    path = library / "packs" / "fixture-triage" / "pack.yaml"
    path.write_text(path.read_text().replace("  default_level: L1\n", "  default_level: L1\n  default_level: L0\n"))
    found = messages(library)
    assert len(found) == 1 and "duplicate key 'default_level'" in found[0]


def test_duplicate_top_level_key_in_pack_yaml_is_refused(library):
    path = library / "packs" / "fixture-triage" / "pack.yaml"
    path.write_text(path.read_text() + "version: 0.2.0\n")
    assert "duplicate key 'version'" in only(messages(library), "duplicate key")


# ---- required packs resolve through the catalog (any root) ----------------

from archi.bundles import discovery  # noqa: E402

THIRD_PARTY_PACKS = (
    Path(__file__).resolve().parents[1] / "bundles" / "fixtures" / "thirdparty" / "archi_fixture_packs" / "packs"
)


def test_third_party_automation_requires_a_shipped_pack():
    """fixture-ops-watch lives in the third-party package and uses a tool of
    the shipped monit-live pack: different roots, so no sibling directory."""
    catalog = discovery.from_roots([SHIPPED, THIRD_PARTY_PACKS])
    assert catalog.problems == []
    assert "fixture-ops-watch" in catalog.packs
    assert not (THIRD_PARTY_PACKS / "monit-live").exists()


def test_cross_root_pack_is_refused_when_the_required_pack_is_absent():
    catalog = discovery.from_roots([THIRD_PARTY_PACKS])
    assert "fixture-ops-watch" not in catalog.packs
    message = only([p.message for p in catalog.problems], "archi_monit_condor_search")
    assert "required pack(s) not found: monit-live" in message


def test_cross_root_tool_that_the_required_pack_lacks_is_refused(tmp_path):
    root = tmp_path / "third"
    shutil.copytree(THIRD_PARTY_PACKS / "fixture-ops-watch", root / "fixture-ops-watch")
    path = root / "fixture-ops-watch" / "action_classes.yaml"
    path.write_text(path.read_text().replace("archi_monit_condor_search", "archi_monit_nothing"))
    catalog = discovery.from_roots([SHIPPED, root])
    assert "is not a live tool of this pack or of a pack it requires" in only(
        [p.message for p in catalog.problems], "archi_monit_nothing"
    )


def test_cli_validates_a_path_against_the_catalog(capsys):
    pack = THIRD_PARTY_PACKS / "fixture-ops-watch"
    assert archi_main(["bundles", "validate", str(pack), "--pack-root", str(SHIPPED)]) == 0
    assert not (pack.parent / "monit-live").exists()


def test_without_a_catalog_a_required_tool_is_not_guessed_from_siblings(library):
    """fixture-tools sits next to fixture-triage, yet validate_pack(dir) with no
    catalog does not look: it says what it cannot resolve."""
    found = [p.message for p in validate_pack(library / "packs" / "fixture-triage")]
    assert "required pack not resolvable without a catalog (fixture-tools)" in only(found, "set_ticket_status")


def test_without_a_catalog_the_packs_own_tool_still_resolves(library):
    tool = "archi.tools.fixture:set_ticket_status"
    edit(library, lambda d: d.update(requires={"packs": []}, live_tools=[tool], idempotent_tools=[tool]))
    assert [p.message for p in validate_pack(library / "packs" / "fixture-triage")] == []


# --- idempotent tools (a mediator that can run at L2 must honour the idempotency key) ---


def test_an_l2_mediator_tool_must_be_declared_idempotent(library):
    edit(library, lambda d: d.pop("idempotent_tools"), pack="fixture-tools")
    found = messages(library)
    assert "idempotent_tools" in only(found, "can run at L2")
    assert "set_ticket_status" in only(found, "can run at L2")


def test_an_l1_class_needs_no_idempotent_declaration(library):
    edit(library, lambda d: d.pop("idempotent_tools"), pack="fixture-tools")
    edit(library, classes(lambda c: c["set_ticket_status"].update(max_level="L1")), file="action_classes.yaml")
    assert messages(library) == []


def test_idempotent_tools_must_be_live_tools_of_the_pack(library):
    edit(library, lambda d: d["idempotent_tools"].append("archi.tools.fixture:other"), pack="fixture-tools")
    assert "not one of this pack's live_tools" in only(messages(library, "fixture-tools"), "other")
