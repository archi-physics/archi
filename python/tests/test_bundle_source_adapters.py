"""The bundle's sources must run through the substrate's adapter contract.

Since the connector migration the readers return ``ConnectorRun``, which the
substrate runner cannot consume: it reads ``next_cursor``, a field that result
deliberately omits. A registry that names a reader directly fails every run
with ``AttributeError: 'ConnectorRun' object has no attribute 'next_cursor'``,
which is what an installed instance did until the bundle named adapters.

These tests hold that line: every source this bundle ships names a
``ConnectorAdapter`` whose declared profile matches the registry entry, and the
three cache-backed sources run from their shipped parameters and return a
result carrying the substrate's own fields.
"""

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest
import yaml
from okg.deployment import ConnectorAdapter, EdgeFact, NodeFact

SOURCE_DEFAULTS = (
    Path(__file__).resolve().parents[2] / "bundles" / "cern-team" / "source-defaults"
)


def _bundle_entries():
    for path in sorted(SOURCE_DEFAULTS.iterdir()):
        if not path.is_file():
            continue
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, entry in body.items():
            if isinstance(entry, dict) and str(entry.get("module", "")).startswith(
                "archi."
            ):
                yield path.name, name, entry


ENTRIES = list(_bundle_entries())
ENTRY_IDS = [f"{filename}-{name}" for filename, name, _ in ENTRIES]


def _adapter_class(entry):
    module = importlib.import_module(entry["module"])
    return getattr(module, entry["class"])


@pytest.mark.parametrize(("filename", "name", "entry"), ENTRIES, ids=ENTRY_IDS)
def test_every_bundle_source_names_a_runnable_adapter(filename, name, entry):
    cls = _adapter_class(entry)
    assert issubclass(cls, ConnectorAdapter), (
        f"{filename}: {name} names {entry['class']}, which the substrate runner "
        "cannot drive; register the reader's ConnectorAdapter instead"
    )
    # The substrate reads both off the class, without constructing it.
    assert inspect.getattr_static(cls, "profile") == entry["source_class"]
    assert isinstance(inspect.getattr_static(cls, "change_probe_kind"), str)


def _entry(source_name, filename):
    """One bundle entry, named by its file so the live and frozen CMSSW
    templates (both `cmssw_releases`) are never confused for each other."""
    for candidate_file, name, entry in ENTRIES:
        if name == source_name and candidate_file == filename:
            return entry
    raise AssertionError(f"{source_name} is not declared in {filename}")


def _params(source_name, filename, data_root, extra=None):
    """The shipped parameters, with this bundle's placeholders filled in."""
    params = dict(_entry(source_name, filename)["params"])
    replacements = {"${archi_data_root}": str(data_root), "${deployment_name}": "test"}
    for key, value in list(params.items()):
        if isinstance(value, str):
            for placeholder, actual in {**replacements, **(extra or {})}.items():
                value = value.replace(placeholder, actual)
            params[key] = value
    return params


def _run(adapter, mode="scope_complete"):
    run = adapter.run("run-1", mode=mode)
    # `next_cursor` is the substrate-only field the runner reads off every run.
    # A `ConnectorRun` has no such attribute, which is the defect these
    # adapters exist to prevent.
    assert hasattr(run, "next_cursor"), (
        "the run result must carry the substrate's fields, not the SDK's"
    )
    return run, list(run.facts)


def test_documentation_source_runs_from_its_shipped_parameters(tmp_path):
    (tmp_path / "docsite").mkdir()
    (tmp_path / "docsite" / "records.json").write_text(
        json.dumps(
            [
                {
                    "url": "https://docs.example.org/guide",
                    "title": "Transfer guide",
                    "body": "How the team runs transfers.",
                    "site_name": "docs.example.org",
                }
            ]
        )
    )
    (tmp_path / "jira").mkdir()
    (tmp_path / "jira" / "records.json").write_text(json.dumps([]))
    entry = _entry("docsite", "docsite.yaml.example")
    adapter = _adapter_class(entry)(**_params("docsite", "docsite.yaml.example", tmp_path))
    run, facts = _run(adapter)
    assert run.completed_scope is True
    assert [f for f in facts if isinstance(f, NodeFact) and f.subtype == "documentation_page"]


def test_jira_source_runs_from_its_shipped_parameters(tmp_path):
    (tmp_path / "jira").mkdir()
    (tmp_path / "jira" / "records.json").write_text(
        json.dumps(
            [
                {
                    "key": "CMSPROD-1",
                    "summary": "Transfers stalled",
                    "description": "Queue drained overnight.",
                    "status": "Closed",
                    "updated": "2026-01-02T03:04:05.000+0000",
                }
            ]
        )
    )
    (tmp_path / "jira" / "meta.json").write_text(
        json.dumps(
            {
                "record_count": 1,
                "fetched_at": "2026-01-02T03:04:05+00:00",
                "projects": ["CMSPROD"],
            }
        )
    )
    entry = _entry("jira", "jira.yaml.example")
    adapter = _adapter_class(entry)(**_params("jira", "jira.yaml.example", tmp_path))
    _run_result, facts = _run(adapter)
    assert [f for f in facts if isinstance(f, NodeFact) and f.subtype == "jira_issue"]


def test_cmssw_source_runs_from_its_shipped_frozen_parameters(tmp_path):
    catalogue = tmp_path / "cmssw" / "releases.map"
    catalogue.parent.mkdir()
    catalogue.write_text(
        "architecture=el8_amd64_gcc12;label=CMSSW_14_0_1;type=Production;state=Announced;\n"
        "architecture=el8_amd64_gcc12;label=CMSSW_14_0_2;type=Production;state=Announced;\n"
    )
    digest = "sha256:" + hashlib.sha256(catalogue.read_bytes()).hexdigest()
    # The frozen example, not the live default: that one fetches the real
    # catalogue over the network, which no test here may do.
    frozen = "cmssw_releases_frozen.yaml.example"
    params = _params(
        "cmssw_releases",
        frozen,
        tmp_path,
        extra={"${cmssw_map_path}": str(catalogue), "${cmssw_map_digest}": digest},
    )
    assert params["fetch"] is False
    adapter = _adapter_class(_entry("cmssw_releases", frozen))(**params)
    run, facts = _run(adapter)
    assert run.completed_scope is True
    releases = [f for f in facts if isinstance(f, NodeFact) and f.subtype == "cmssw_release"]
    # The reader emits one node per release plus the family node they share.
    assert {node.attrs["label"] for node in releases} == {
        "CMSSW_14_0_1",
        "CMSSW_14_0_2",
        "CMSSW_14_0_X",
    }
    assert [f for f in facts if isinstance(f, EdgeFact) and f.edge_type == "supersedes"]
