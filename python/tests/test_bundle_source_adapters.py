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

import ast
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


def _all_bundle_entries():
    """Every source this bundle ships, whoever owns the adapter.

    Unlike `_bundle_entries` this does NOT filter to `archi.` modules: the
    profile-tuple contract is the substrate's, so it binds the entries that
    name a substrate class (github_repo, gitlab_repo) exactly as it binds
    ours. `.yaml.example` files are included — an operator enables one by
    renaming it, and an illegal tuple blocks their install just the same.
    """
    for path in sorted(SOURCE_DEFAULTS.iterdir()):
        if not path.is_file():
            continue
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, entry in body.items():
            if isinstance(entry, dict) and entry.get("module"):
                yield path.name, name, entry


ALL_ENTRIES = list(_all_bundle_entries())
ALL_ENTRY_IDS = [f"{filename}-{name}" for filename, name, _ in ALL_ENTRIES]


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


def _class_level_constant(source_path, class_name, attr):
    """Return a class-level ``attr`` only when it is a string LITERAL.

    This is the substrate's own acceptance rule, reimplemented so the test
    proves it without depending on a private okg symbol: okg's
    ``substrate/deployment_lint.py`` reads ``change_probe_kind`` by parsing
    this file (``_class_level_str_attr``), and accepts only an
    ``ast.Constant`` whose value is a ``str``. Anything else -- notably
    ``profile = SomeReader.profile``, which parses as an ``ast.Attribute`` --
    reads as absent and fails the source-registry lint with
    ``deployment.source_registry.probe_missing``.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for stmt in node.body:
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                targets = [stmt.target]
            if not any(t.id == attr for t in targets):
                continue
            value = stmt.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            raise AssertionError(
                f"{class_name}.{attr} is {ast.dump(value)}, not a string literal; "
                "the substrate parses this file and accepts only a literal, so a "
                "reference to the reader's attribute reads as absent"
            )
    raise AssertionError(f"{class_name} declares no class-level {attr}")


@pytest.mark.parametrize(("filename", "name", "entry"), ENTRIES, ids=ENTRY_IDS)
def test_adapter_authority_is_a_literal_that_matches_its_reader(filename, name, entry):
    """Both attributes must be literals, and must not drift from the reader.

    A literal cannot be computed from the reader, so nothing but this test
    keeps the two in step: if a reader's profile or probe kind changes and the
    adapter's literal does not, the registry entry and the reader disagree and
    the substrate refuses the source (``source_class_profile_mismatch``) or
    probes the wrong way.
    """
    cls = _adapter_class(entry)
    source_path = Path(importlib.import_module(entry["module"]).__file__)
    reader = inspect.getattr_static(cls, "reader_class")
    for attr in ("profile", "change_probe_kind"):
        literal = _class_level_constant(source_path, entry["class"], attr)
        assert literal == getattr(reader, attr), (
            f"{filename}: {entry['class']}.{attr} is {literal!r} but "
            f"{reader.__name__}.{attr} is {getattr(reader, attr)!r}"
        )
        # And the literal is what the class actually exposes.
        assert inspect.getattr_static(cls, attr) == literal


def test_every_shipped_source_default_declares_a_legal_profile_tuple():
    """No source-default may ship a tuple the substrate would refuse.

    `indico.yaml.example` shipped `source_class: discovery_crawl` with
    `record_identity_kind: remote_id` — a crossbreed of jira's identity half
    and docsite's revision half. The substrate allows `remote_id` only under
    `mutable_api`, so `okg install` refused with
    `deployment.source_registry.profile_invalid` and an operator who enabled
    Indico could not install at all. Nothing caught it because the tuple only
    reaches lint once the file has been renamed to `.yaml`.

    This calls the substrate's own `validate_profile_tuple` — the same
    function `deployment_lint` calls — rather than restating the matrix, so
    the test tracks the SDK instead of drifting from it. It is checked in one
    body rather than parametrized so a bad tuple reports every offender at
    once, not just the first.
    """
    from okg.substrate.sources.profiles import (
        PROFILE_NAMES,
        validate_profile_tuple,
    )

    failures = []
    for filename, name, entry in ALL_ENTRIES:
        source_class = entry.get("source_class")
        try:
            if source_class not in PROFILE_NAMES:
                raise ValueError(
                    f"unknown source_class {source_class!r}; "
                    f"valid profiles: {sorted(PROFILE_NAMES)}"
                )
            validate_profile_tuple(
                source_class=str(source_class),
                record_identity_kind=entry.get("record_identity_kind"),
                source_revision_kind=entry.get("source_revision_kind"),
                deletion_semantics=entry.get("deletion_semantics"),
                publication_mode=entry.get("publication_mode"),
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{filename} [{name}]: {exc}")
    assert not failures, (
        "these source-defaults would fail `okg install` with "
        "deployment.source_registry.profile_invalid:\n\n"
        + "\n\n".join(failures)
    )


def test_the_indico_record_key_is_the_locator_it_declares():
    """The identity kind and the emitted record key must agree.

    Declaring `scoped_locator` while emitting the upstream event id would
    fix the lint and keep the lie. The reader emits a scope-relative path,
    so this holds the two together.
    """
    from archi.sources.indico import IndicoEventRecord, _meeting_node

    record = IndicoEventRecord(
        event_id="654321", title="Weekly ops", url="", description="",
        date="", end_date="", event_type="", category="", category_id=None,
    )
    node = _meeting_node(record, {"run_id": "r1"})
    assert node.source_record_id == {"path": "event/654321"}
    # The node id is NOT keyed by the locator: it stays event-id shaped, so
    # this change does not re-identify any graph node.
    assert node.node_id == "meeting_minutes:654321"


def test_the_adapter_forwards_nothing_the_readers_do_not_all_define():
    """``cache_paths`` is a reader detail the substrate never reads.

    Two wrapped readers (``TwikiCrawlSource``, ``TwikiEOSSource``) do not
    define it, so a forwarding property on the shared base raised
    ``AttributeError`` for them. Nothing in the substrate reads the attribute.
    """
    from archi.sources._sdk_adapter import ReaderAdapter

    assert not hasattr(ReaderAdapter, "cache_paths")


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


@pytest.mark.parametrize(("filename", "name", "entry"), ENTRIES, ids=ENTRY_IDS)
def test_every_entry_binds_against_its_adapter_signature(filename, name, entry):
    """Each adapter must expose the parameters its registry entry authors.

    The adapter takes ``**params``, which the substrate refuses under strict
    admission -- ``source_param_unconsumed``, "**kwargs is not proof of
    consumption" -- and which also stops a misspelled parameter being caught
    when the adapter is bound. The adapters therefore publish the wrapped
    reader's signature, and this binds the shipped parameters against it.
    """
    cls = _adapter_class(entry)
    signature = inspect.signature(cls)
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ), f"{filename}: {entry['class']} still exposes **kwargs to the substrate"
    # Shipped params only; placeholders are strings either way, and binding
    # does not read the values.
    signature.bind(**(entry.get("params") or {}))


def test_a_misspelled_parameter_fails_when_the_adapter_is_bound():
    from archi.sources.jira import JiraIssueAdapter

    with pytest.raises(TypeError) as excinfo:
        JiraIssueAdapter(records_pathh="/tmp/records.json")
    assert "records_pathh" in str(excinfo.value)


#: Entries that strict admission still refuses, and why. NOT about the
#: adapter signature: these two author `source_name` in `params`, which is a
#: substrate-owned key (`RESERVED_ADAPTER_PARAM_NAMES`), so the substrate
#: refuses them with `source_param_reserved` before it ever looks at the
#: constructor. Pre-existing and dormant -- cern-team ships on the legacy
#: admission contract -- and fixing it changes which name the readers emit,
#: so it is tracked separately rather than bundled in here. Remove an entry
#: from this set when it is fixed; the test will tell you if you forget.
STRICT_PATH_KNOWN_REFUSALS = {
    ("docsite.yaml.example", "docsite"): "source_param_reserved",
    ("twiki_crawl.yaml.example", "twiki_crawl"): "source_param_reserved",
}


def test_the_substrate_strict_admission_check_consumes_every_parameter():
    """Exercise okg's own refusal, not a restatement of it.

    `source_adapter_init_params` is the function that raises
    `source_param_unconsumed`. It takes the strict path only for an entry whose
    admission contract is `strict_v1`, so the entry is relabelled here; nothing
    else about it changes and no database is touched. cern-team ships on the
    legacy contract today, which is why the refusal was dormant rather than
    absent.

    The assertion is specifically that no entry fails for an unconsumed
    parameter. Entries with a different known refusal are listed above with
    their reason, so this cannot quietly become a test that asserts nothing.
    """
    import dataclasses

    from okg.substrate.ingest.adapter_factory import source_adapter_init_params
    from okg.substrate.sources.registry import (
        STRICT_ADMISSION_CONTRACT,
        admit_source_registry_document,
    )

    for filename, name, raw_entry in ENTRIES:
        admission = admit_source_registry_document(
            {"sources": {name: raw_entry}}, registry_path=Path(filename)
        )
        strict = dataclasses.replace(
            admission.entries[name], admission_contract=STRICT_ADMISSION_CONTRACT
        )
        expected = STRICT_PATH_KNOWN_REFUSALS.get((filename, name))
        try:
            params = source_adapter_init_params(
                strict,
                dsn="postgresql://localhost/unused",
                deployment="unused",
                adapter_class=_adapter_class(raw_entry),
            )
        except ValueError as exc:
            code = str(exc).split(":", 1)[0]
            assert code != "source_param_unconsumed", (
                f"{filename}: {name} — {exc}"
            )
            assert code == expected, (
                f"{filename}: {name} hits an unlisted strict refusal {code!r}: "
                f"{exc}"
            )
            continue
        assert expected is None, (
            f"{filename}: {name} no longer fails with {expected!r}; remove it "
            "from STRICT_PATH_KNOWN_REFUSALS"
        )
        assert set(raw_entry.get("params") or {}) <= set(params), (
            f"{filename}: {name} lost an authored parameter on the strict path"
        )
