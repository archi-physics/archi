"""Doc pages and JIRA issues declare their chunk relation and title splitting.

archi-physics/cms-kb#6: TWiki and docs-site pages were hard to find. Their
text lives on `document_chunk` rows reached by `contains`, and a TWiki title
is one WikiWord the search index reads as a single token. Two catalog
declarations fix the read side:

- `okg_passage_rollup` on DocumentationPage and JiraIssue, so a passage names
  its page or issue as parent and passage-ranked search rolls chunk scores up
  to it (okg resolves the parent by the subtype the edge comes from);
- `okg_searchable_text_path_fields: title` on DocumentationPage, so the title
  also contributes its component words.

These tests read the declarations through okg's own composer, for both real
consumers' module sets, so a declaration okg does not accept fails here.
"""
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("okg")
from okg.substrate.catalog.modules_compose import compose_catalog  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = REPO / "python" / "archi" / "schemas"
# The slices a pack installs (every pack links the same package files).
BUNDLE_SCHEMAS = REPO / "bundles" / "packs" / "jira" / "schemas"

CERN_TEAM_MODULES = ["document_starter", "person", "extraction"]
COMPOPS_MODULES = ["document_starter", "person", "extraction", "dataset", "repo_starter"]

CHUNK_ROLLUP = {"edge_type": "contains", "child_subtype": "document_chunk"}


def _compose(modules: list[str], schemas: Path = SCHEMAS):
    tmp = Path(tempfile.mkdtemp())
    try:
        dep = tmp / "probe"
        (dep / "schemas" / "bridges").mkdir(parents=True)
        for name in ("operations.yaml", "sources.yaml"):
            shutil.copy(schemas / name, dep / "schemas" / name)
            shutil.copy(schemas / "bridges" / name, dep / "schemas" / "bridges" / name)
        (dep / "deployment.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "probe",
                    "postgres": {"dsn": "postgresql://unused/probe"},
                    "schema_dir": "schemas",
                    "modules": modules,
                }
            )
        )
        return compose_catalog(dep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize(
    "modules,label",
    [(CERN_TEAM_MODULES, "cern-team"), (COMPOPS_MODULES, "cms-compops")],
)
@pytest.mark.parametrize("parent", ["documentation_page", "jira_issue"])
def test_parent_declares_chunk_rollup(modules, label, parent):
    composed = _compose(modules)
    meta = composed.subtype_metadata.get(parent, {})
    assert meta.get("passage_rollup") == CHUNK_ROLLUP, (
        f"{label}: {parent} does not declare its chunk relation; got "
        f"{meta.get('passage_rollup')!r}"
    )
    # The composer already refuses a rollup with no matching narrowing; this
    # states the relation the sources actually emit.
    assert (parent, "contains", "document_chunk") in composed.narrowing_triples()


@pytest.mark.parametrize(
    "modules,label",
    [(CERN_TEAM_MODULES, "cern-team"), (COMPOPS_MODULES, "cms-compops")],
)
def test_documentation_page_title_is_a_path_field(modules, label):
    meta = _compose(modules).subtype_metadata.get("documentation_page", {})
    assert meta.get("searchable_text_path_fields") == ["title"], (
        f"{label}: documentation_page title is not split into words; got "
        f"{meta.get('searchable_text_path_fields')!r}"
    )
    # A path field must also be a plain searchable field, or it stamps nothing.
    assert "title" in meta.get("searchable_text_fields", [])


def test_bundle_copy_matches_the_package_schema():
    """Each pack ships sources.yaml (a link to the package file); a declaration
    added to one copy and not the other would reach only one consumer."""
    assert (BUNDLE_SCHEMAS / "sources.yaml").read_bytes() == (
        SCHEMAS / "sources.yaml"
    ).read_bytes()
    bundle = _compose(CERN_TEAM_MODULES, BUNDLE_SCHEMAS).subtype_metadata
    for parent in ("documentation_page", "jira_issue"):
        assert bundle[parent]["passage_rollup"] == CHUNK_ROLLUP
