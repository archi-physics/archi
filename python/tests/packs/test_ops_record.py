"""PACT r3-ops-record: the ops-record pack and its schema slice.

acc.r3-ops-record.load: the catalog loads with the slice composed on
cern-team and on cluster-manager. cern-team is the shipped bundle and its
packs, ops-record included (copied by bundle_support), and cluster-manager, which PACT r3-bundle-cluster-manager ships later, is its
ADR 0002 §5 shape with the packs that exist today (docsite + ops-record,
jira optional). okg itself composes the catalog (`okg catalog load`, no
--apply), from the deployment okg's own planner scaffolds from the rendered
profile.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from archi.bundles.manifest import validate_pack
from archi.bundles.render import PROFILES_DIR, render
from archi.bundles.resolve import resolve

from bundles.bundle_support import REPO, build_cern_team_catalog, load_catalog, write_bundle

PACK = REPO / "bundles" / "packs" / "ops-record"
OKG = str(Path(sys.executable).with_name("okg"))

OPS_SUBTYPES = ["ops_incident", "ops_proposal", "ops_decision", "ops_action", "ops_outcome"]


def _schema(name):
    return yaml.safe_load((PACK / "schemas" / name).read_text(encoding="utf-8"))


def _narrowings():
    return _schema("bridges/ops_record.yaml")["classes"]["EdgeNarrowings"]["attributes"]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    root = build_cern_team_catalog(tmp_path_factory.mktemp("ops-record-catalog"))
    write_bundle(
        root,
        "cluster-manager",
        packs={"default": ["docsite", "ops-record"], "optional": ["jira"]},
    )
    return load_catalog(root)


# ---------------------------------------------------------------------------
# The slice
# ---------------------------------------------------------------------------


def test_pack_validates():
    assert [str(p) for p in validate_pack(PACK)] == []


def test_slice_declares_the_five_subtypes():
    classes = _schema("ops_record.yaml")["classes"]
    subtypes = {
        "ops_" + name[3:].lower(): body
        for name, body in classes.items()
        if "archetype" in body.get("annotations", {})
    }
    assert sorted(subtypes) == sorted(OPS_SUBTYPES)
    for subtype, body in subtypes.items():
        attrs = body["attributes"]
        # Every record names its actor and the generation it read
        # (req.r3-ops-record.schema).
        for field in ("record_id", "actor_kind", "actor", "generation_read"):
            assert field in attrs, (subtype, field)
        assert attrs["record_id"]["required"] is True
        ann = body["annotations"]
        searchable = ann["okg_searchable_text_fields"].split(",")
        assert set(ann["okg_identifier_fields"].split(",")) <= set(searchable), subtype


def test_slice_carries_evidence_escalation_and_precedent_fields():
    classes = _schema("ops_record.yaml")["classes"]
    proposal = classes["OpsProposal"]["attributes"]
    assert proposal["evidence_ids"]["required"] and proposal["evidence_ids"]["multivalued"]
    assert proposal["generation_read"]["required"] is True
    assert "escalation_id" in classes["OpsDecision"]["attributes"]
    # Precedent lookup filters incidents on the entity that broke.
    entity = classes["OpsIncident"]["attributes"]["entity"]
    assert entity["required"] and entity["annotations"]["filterable"] == "true"


def test_edges_join_ops_records_only():
    """Edges to other packs' nodes wait on okg (see the bridge's header);
    those references are attributes, and an outcome's ticket is filterable."""
    loop = {"OpsIncident", "OpsProposal", "OpsDecision", "OpsAction", "OpsOutcome"}
    for name, body in _narrowings().items():
        ann = body["annotations"]
        assert {ann["src_subtypes"], ann["dst_subtypes"]} <= loop, name
        assert "optional_when_subtypes_missing" not in ann, name
    classes = _schema("ops_record.yaml")["classes"]
    assert classes["OpsOutcome"]["attributes"]["ticket_key"]["annotations"]["filterable"] == "true"


# ---------------------------------------------------------------------------
# acc.r3-ops-record.load: okg composes the catalog on both bundles
# ---------------------------------------------------------------------------


def _okg(args, cwd):
    env = {k: v for k, v in os.environ.items() if k != "OKG_PROFILES_DIR"}
    env["HOME"] = str(cwd)
    return subprocess.run(
        [OKG, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=180
    )


def _catalog_load(catalog, bundle, tmp_path, with_packs=(), without_packs=()):
    """Render, scaffold with okg's planner, and run `okg catalog load` (dry run)."""
    from okg.substrate.deployment_bootstrap.profile_init import (
        build_plan,
        discover_profile,
        planned_writes,
    )

    instance = tmp_path / "instance"
    resolution = resolve(catalog, [bundle], with_packs=with_packs, without_packs=without_packs)
    rendered = render(resolution, instance)
    answers = {
        "deployment_name": "qt",
        "postgres_dsn": "postgresql://localhost:1/qt",
        "archi_data_root": "/srv/archi-data",
    }
    for q in rendered.composition.questions:
        if q["id"] not in answers and q.get("required") and "default" not in q:
            answers[q["id"]] = "https://example.org/demo"
    plan = build_plan(
        profile=discover_profile(rendered.name, profiles_root=instance / PROFILES_DIR),
        deployment_name="qt",
        deployments_root=tmp_path / "deployments",
        raw_answers=answers,
    )
    for write in planned_writes(plan):
        write.path.parent.mkdir(parents=True, exist_ok=True)
        if write.data is not None:
            write.path.write_bytes(write.data)
        else:
            write.path.write_text(write.text, encoding="utf-8")
    result = _okg(["catalog", "load", "--deployment", str(plan.deployment_dir), "--json"], tmp_path)
    assert result.returncode == 0, result.stderr[-3000:] + result.stdout[-3000:]
    return [p.name for p in resolution.packs], json.loads(result.stdout)


#: bundle -> (selection with ops-record, the same selection without it)
SELECTIONS = {
    "cern-team": ({"with_packs": ["ops-record"]}, {}),
    "cluster-manager": (
        {"with_packs": ["jira"]},
        {"with_packs": ["jira"], "without_packs": ["ops-record"]},
    ),
}


@pytest.mark.parametrize("bundle", sorted(SELECTIONS))
def test_catalog_loads_with_the_slice(catalog, tmp_path, bundle):
    with_ops, without_ops = SELECTIONS[bundle]
    packs, report = _catalog_load(catalog, bundle, tmp_path / "with", **with_ops)
    base_packs, base = _catalog_load(catalog, bundle, tmp_path / "without", **without_ops)
    assert "ops-record" in packs and "ops-record" not in base_packs

    # okg composed all five subtypes and every ops-record narrowing.
    assert report["subtype_count"] - base["subtype_count"] == len(OPS_SUBTYPES), (report, base)
    assert report["narrowings_by_origin"]["bridge:ops_record.yaml"] == len(_narrowings()), report
    assert "bridge:ops_record.yaml" not in base["narrowings_by_origin"]


def test_rendered_profile_carries_the_precedent_profile_and_playbook(catalog, tmp_path):
    rendered = render(resolve(catalog, ["cern-team"], with_packs=["ops-record"]), tmp_path)
    defaults = yaml.safe_load((rendered.profile_dir / "deployment-defaults.yaml").read_text())
    profiles = defaults["search"]["profiles"]
    assert profiles["precedents"]["subtypes"] == OPS_SUBTYPES
    assert {"ops_incident", "ops_outcome"} <= set(profiles["content"]["subtypes"])
    assert (rendered.profile_dir / "skills" / "ops_precedents.md").is_file()
    triggers = yaml.safe_load((rendered.profile_dir / "skills" / "skill-triggers.yaml").read_text())
    assert triggers["intents"]["ops_precedents"]["skill"] == "ops_precedents"
