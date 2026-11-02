"""Record types declare the fields that hold an id a person types.

An exact-id search ranks the record that bears the id first, using
the class annotation `okg_identifier_fields` (comma-separated fields). okg
refuses a declaration whose field is not also in `okg_searchable_text_fields`,
and refuses `text`, `okg_producer_text` and any `okg_*` name.

The YAML tests read the schema files directly, so they run on any okg. The
compose test runs only on an okg that knows the annotation.
"""
import inspect
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = REPO / "python" / "archi" / "schemas"
# The slices a pack installs (every pack links the same package files).
BUNDLE_SCHEMAS = REPO / "bundles" / "packs" / "jira" / "schemas"

EXPECTED = {
    "JiraIssue": ["issue_key"],
    "DocumentationPage": ["analysis_id"],
    "CDSRecord": ["report_number", "analysis_code"],
    "CMSSWRelease": ["release"],
}
MAX_FIELDS = 16


def _classes(schemas: Path) -> dict:
    classes: dict = {}
    for name in ("sources.yaml", "operations.yaml"):
        classes.update(yaml.safe_load((schemas / name).read_text())["classes"])
    return classes


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _declared(classes: dict) -> dict[str, list[str]]:
    out = {}
    for cls, body in classes.items():
        raw = (body.get("annotations") or {}).get("okg_identifier_fields")
        if raw is not None:
            out[cls] = _split(str(raw))
    return out


@pytest.mark.parametrize("schemas", [SCHEMAS, BUNDLE_SCHEMAS], ids=["package", "bundle"])
def test_expected_classes_declare_their_identifier_fields(schemas):
    assert _declared(_classes(schemas)) == EXPECTED


@pytest.mark.parametrize("schemas", [SCHEMAS, BUNDLE_SCHEMAS], ids=["package", "bundle"])
def test_every_identifier_field_is_searchable_and_an_attribute(schemas):
    for cls, fields in _declared(_classes(schemas)).items():
        body = _classes(schemas)[cls]
        searchable = _split(body["annotations"]["okg_searchable_text_fields"])
        attributes = set(body.get("attributes") or {})
        assert 0 < len(fields) <= MAX_FIELDS
        for field in fields:
            assert field in searchable, f"{cls}.{field} is not a searchable text field"
            assert field in attributes, f"{cls}.{field} is not an attribute of {cls}"


@pytest.mark.parametrize("schemas", [SCHEMAS, BUNDLE_SCHEMAS], ids=["package", "bundle"])
def test_no_text_or_okg_field_is_declared(schemas):
    for cls, fields in _declared(_classes(schemas)).items():
        for field in fields:
            assert field not in {"text", "okg_producer_text"}, f"{cls}.{field}"
            assert not field.startswith("okg_"), f"{cls}.{field}"


def test_bundle_and_package_copies_declare_the_same_fields():
    assert _declared(_classes(SCHEMAS)) == _declared(_classes(BUNDLE_SCHEMAS))


def _okg_knows_identifier_fields() -> bool:
    try:
        from okg.substrate.catalog import modules_compose
    except ImportError:
        return False
    return "okg_identifier_fields" in inspect.getsource(modules_compose)


@pytest.mark.skipif(
    not _okg_knows_identifier_fields(),
    reason="this okg does not read okg_identifier_fields yet",
)
def test_okg_composes_the_declarations():
    from okg.substrate.catalog.modules_compose import compose_catalog

    tmp = Path(tempfile.mkdtemp())
    try:
        dep = tmp / "probe"
        (dep / "schemas" / "bridges").mkdir(parents=True)
        for name in ("operations.yaml", "sources.yaml"):
            shutil.copy(BUNDLE_SCHEMAS / name, dep / "schemas" / name)
            shutil.copy(
                BUNDLE_SCHEMAS / "bridges" / name, dep / "schemas" / "bridges" / name
            )
        (dep / "deployment.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "probe",
                    "postgres": {"dsn": "postgresql://unused/probe"},
                    "schema_dir": "schemas",
                    "modules": ["document_starter", "person", "extraction"],
                }
            )
        )
        meta = compose_catalog(dep).subtype_metadata
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert meta["jira_issue"]["identifier_fields"] == ["issue_key"]
    assert meta["documentation_page"]["identifier_fields"] == ["analysis_id"]
    assert meta["cds_record"]["identifier_fields"] == ["report_number", "analysis_code"]
    assert meta["cmssw_release"]["identifier_fields"] == ["release"]
