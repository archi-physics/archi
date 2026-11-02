"""Archi types declare their catalog handle fields.

okg reads a record's title, creation date and last-modified date from the
attribute a class names in `okg_title_from`, `okg_created_at_from` and
`okg_last_modified_at_from`. okg refuses a declaration that names an
attribute the class does not have, so a misspelt name would stop compose.

The YAML tests need no okg: they pin the exact 27 declarations and check each
names an attribute of its class. The compose test reads them back through
okg's own composer, on an okg that knows `okg_title_from`.
"""
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = REPO / "python" / "archi" / "schemas"
FILES = ("operations.yaml", "sources.yaml")

KEYS = {
    "okg_title_from": "title_from",
    "okg_created_at_from": "created_at_from",
    "okg_last_modified_at_from": "last_modified_at_from",
}

EXPECTED = {
    "JiraIssue": {
        "okg_title_from": "summary",
        "okg_created_at_from": "created",
        "okg_last_modified_at_from": "updated",
    },
    # A comment has no title of its own; its id names its issue.
    "JiraComment": {"okg_created_at_from": "created"},
    "Workflow": {"okg_title_from": "workflow_name", "okg_created_at_from": "created_at"},
    "CMSSWRelease": {"okg_title_from": "release"},
    "GlobalTag": {"okg_title_from": "tag_name"},
    "DocumentationPage": {"okg_title_from": "title"},
    "InfrastructureService": {"okg_title_from": "name"},
    "Downtime": {"okg_title_from": "description"},
    "Operator": {"okg_title_from": "name"},
    "MeetingMinutes": {"okg_title_from": "title"},
    "Site": {"okg_title_from": "name"},
    "StorageEndpoint": {"okg_title_from": "name"},
    "DataCertification": {"okg_title_from": "certification_id"},
    "Facility": {"okg_title_from": "name"},
    "Run": {"okg_title_from": "run_number"},
    "ComputeEndpoint": {"okg_title_from": "name"},
    "Federation": {"okg_title_from": "name"},
    "SoftwareRepository": {"okg_title_from": "full_name"},
    "SiteConfig": {"okg_title_from": "site_name"},
    "ForumThread": {"okg_title_from": "title"},
    "SAMTest": {"okg_title_from": "test_id"},
    "MonitoringSnapshot": {"okg_title_from": "subject"},
    "TransferJob": {"okg_title_from": "transfer_id"},
    "CDSRecord": {"okg_title_from": "title"},
}


def _classes(schemas: Path = SCHEMAS) -> dict:
    classes: dict = {}
    for name in FILES:
        classes.update(yaml.safe_load((schemas / name).read_text())["classes"])
    return classes


def _declarations(classes: dict) -> dict:
    found: dict = {}
    for cls, body in classes.items():
        annotations = body.get("annotations") or {}
        picked = {k: annotations[k] for k in KEYS if k in annotations}
        if picked:
            found[cls] = picked
    return found


def _unknown_attributes(classes: dict) -> list:
    bad = []
    for cls, picked in _declarations(classes).items():
        attributes = set((classes[cls].get("attributes") or {}).keys())
        bad += [(cls, key, value) for key, value in picked.items() if value not in attributes]
    return bad


def test_declarations_are_exactly_the_reviewed_set():
    found = _declarations(_classes())
    assert found == EXPECTED
    assert sum(len(v) for v in found.values()) == 27


def test_every_declaration_names_an_attribute_of_its_class():
    assert _unknown_attributes(_classes()) == []


def test_a_misspelt_attribute_is_caught():
    classes = _classes()
    classes["Workflow"]["annotations"]["okg_created_at_from"] = "created"
    assert _unknown_attributes(classes) == [("Workflow", "okg_created_at_from", "created")]


def test_composer_reads_every_declaration():
    pytest.importorskip("okg")
    from okg.substrate.catalog import modules_compose

    if "okg_title_from" not in getattr(modules_compose, "CATALOG_FIELD_ANNOTATIONS", {}):
        pytest.skip("this okg predates okg_title_from (read-model slice 1)")
    tmp = Path(tempfile.mkdtemp())
    try:
        dep = tmp / "probe"
        (dep / "schemas" / "bridges").mkdir(parents=True)
        for name in FILES:
            shutil.copy(SCHEMAS / name, dep / "schemas" / name)
            shutil.copy(SCHEMAS / "bridges" / name, dep / "schemas" / "bridges" / name)
        (dep / "deployment.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "probe",
                    "postgres": {"dsn": "postgresql://unused/probe"},
                    "schema_dir": "schemas",
                    "modules": ["document_starter", "person", "extraction", "dataset", "repo_starter"],
                }
            )
        )
        composed = modules_compose.compose_catalog(dep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    by_class = {
        cls: {
            KEYS[k]: composed.subtype_metadata.get(_snake(cls), {}).get(KEYS[k]) for k in picked
        }
        for cls, picked in EXPECTED.items()
    }
    assert by_class == {
        cls: {KEYS[k]: v for k, v in picked.items()} for cls, picked in EXPECTED.items()
    }


def _snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and (not name[i - 1].isupper() or (i + 1 < len(name) and name[i + 1].islower())):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
