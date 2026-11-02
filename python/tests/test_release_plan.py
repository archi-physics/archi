"""The 3.0 release plan and the r3-* PACT manifests name the same tasks."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load((ROOT / "pact" / "plan" / "archi-3.0.yaml").read_text())


def test_every_planned_pact_has_a_manifest_with_the_same_tasks():
    for pid, entry in PLAN["pacts"].items():
        manifest = yaml.safe_load(
            (ROOT / "pact" / "changes" / pid / "pact.yaml").read_text()
        )
        assert manifest["id"] == pid
        assert [t["id"] for t in manifest["tasks"]] == list(entry["tasks"]), pid


def test_every_r3_manifest_is_planned():
    on_disk = {p.name for p in (ROOT / "pact" / "changes").glob("r3-*")}
    assert on_disk == set(PLAN["pacts"])


def test_milestones_and_labels_resolve():
    titles = {m["title"] for m in PLAN["milestones"]}
    for pid, entry in PLAN["pacts"].items():
        assert entry["milestone"] in titles, pid
        for label in entry.get("labels", []):
            assert label in PLAN["labels"], (pid, label)


def test_issue_titles_are_unique():
    titles = [t for e in PLAN["pacts"].values() for t in e["tasks"].values()]
    assert len(titles) == len(set(titles))
