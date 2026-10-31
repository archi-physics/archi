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
import os
import re
import textwrap
from pathlib import Path

import pytest
import yaml
from okg.deployment import ConnectorAdapter, EdgeFact, NodeFact

PACKS = Path(__file__).resolve().parents[2] / "bundles" / "packs"


def _pack_templates():
    """(pack/file, source name, entry) for every source template a pack ships."""
    for path in sorted(PACKS.glob("*/sources/*.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, entry in body.items():
            yield f"{path.parent.parent.name}/{path.name}", name, entry


def _bundle_entries():
    for filename, name, entry in _pack_templates():
        if isinstance(entry, dict) and str(entry.get("module", "")).startswith("archi."):
            yield filename, name, entry


ENTRIES = list(_bundle_entries())
ENTRY_IDS = [f"{filename}-{name}" for filename, name, _ in ENTRIES]


def _all_bundle_entries():
    """Every source the packs ship, whoever owns the adapter.

    Unlike `_bundle_entries` this does NOT filter to `archi.` modules: the
    profile-tuple contract is the substrate's, so it binds the entries that
    name a substrate class (git-code's repo and git_history) exactly as it
    binds ours.
    """
    for filename, name, entry in _pack_templates():
        if isinstance(entry, dict) and entry.get("module"):
            yield filename, name, entry


ALL_ENTRIES = list(_all_bundle_entries())
ALL_ENTRY_IDS = [f"{filename}-{name}" for filename, name, _ in ALL_ENTRIES]
FROZEN_CMSSW = PACKS / "cmssw-releases" / "frozen" / "cmssw_releases.yaml"


def _adapter_class(entry):
    module = importlib.import_module(entry["module"])
    return getattr(module, entry["class"])


def _substrate_native(cls):
    """The okg source class `cls` extends directly, when it is one.

    A reader written on the public SDK returns ``ConnectorRun`` and needs a
    ``ConnectorAdapter``. A subclass of a substrate source (code-text's
    ``SecretScreenedDocCorpusSource`` extends okg's ``DocCorpusSource``)
    already returns the substrate's own result, so the runner drives it as it
    drives the parent; the pack-isolation install test proves that.
    """
    from okg.substrate.library.sources.doc_corpus import DocCorpusSource

    from archi.sources.code_text import SecretScreenedDocCorpusSource

    # An allowlist, not any subclass: each entry overrides only how files are
    # listed and keeps DocCorpusSource's run(), which the install test runs.
    native = {SecretScreenedDocCorpusSource: DocCorpusSource}
    return native.get(cls)


@pytest.mark.parametrize(("filename", "name", "entry"), ENTRIES, ids=ENTRY_IDS)
def test_every_bundle_source_names_a_runnable_adapter(filename, name, entry):
    cls = _adapter_class(entry)
    assert issubclass(cls, ConnectorAdapter) or _substrate_native(cls), (
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
    reader = _substrate_native(cls) or inspect.getattr_static(cls, "reader_class")
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

    The cern-team Indico template once shipped `source_class:
    discovery_crawl` with `record_identity_kind: remote_id` — a crossbreed of
    jira's identity half and docsite's revision half. The substrate allows
    `remote_id` only under `mutable_api`, so `okg install` refused with
    `deployment.source_registry.profile_invalid` and an operator who enabled
    Indico could not install at all.

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
    frozen = yaml.safe_load(FROZEN_CMSSW.read_text(encoding="utf-8"))
    for filename, name, entry in ALL_ENTRIES + [
        ("cmssw-releases/frozen/cmssw_releases.yaml", n, e) for n, e in frozen.items()
    ]:
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


def test_every_placeholder_has_an_install_answer_behind_it(tmp_path):
    """A `${...}` with no init_question makes its source uninstallable.

    `build_plan` interpolates every source template against the validated
    answers, and `interpolate` refuses an unknown reference outright (okg
    `substrate/catalog/profile.py` `replace_match`) rather than leaving it
    verbatim. So a placeholder nothing asks about aborts the whole install
    with `ProfileError: undefined variable reference`. The cern-team frozen
    CMSSW template once did exactly that.

    Each pack is rendered alone (with the packs it requires) and handed to
    okg's own `build_plan`, so every reference must resolve against the
    questions the rendered profile really asks.
    """
    from packs.pack_support import context_packs, okg_plan, render_alone

    failures = {}
    for pack in context_packs():
        rendered = render_alone(tmp_path / pack, pack)
        answers = {
            q["id"]: "x" for q in rendered.composition.questions
            if q.get("required") and q["id"] not in ("deployment_name", "postgres_dsn")
        }
        try:
            okg_plan(rendered, tmp_path / pack, answers)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures[pack] = str(exc)
    assert not failures, failures


def test_the_frozen_cmssw_template_replaces_the_live_entry():
    """The frozen template is a replacement for the installed live entry.

    It is not a pack source (two templates for one source would install
    twice), so its key is the rendered id of the live entry it replaces, and
    its header says how. Frozen mode means reviewed bytes, never a fetch.
    """
    live = yaml.safe_load(
        (PACKS / "cmssw-releases" / "sources" / "cmssw_releases.yaml").read_text(encoding="utf-8")
    )
    frozen_text = FROZEN_CMSSW.read_text(encoding="utf-8")
    frozen = yaml.safe_load(frozen_text)
    assert set(live) == {"cmssw_releases"}
    assert set(frozen) == {"cmssw-releases.cmssw_releases"}
    assert "replace the `cmssw-releases.cmssw_releases` entry" in frozen_text
    assert "${" not in frozen_text, "a pasted entry gets no placeholder filled"
    assert frozen["cmssw-releases.cmssw_releases"]["params"]["fetch"] is False
    assert live["cmssw_releases"]["params"]["fetch"] is True
    # Everything but the parameters is the live entry's, so the swap changes
    # only where the bytes come from.
    def rest(entry):
        text = yaml.safe_dump({k: v for k, v in entry.items() if k != "params"})
        return text.replace("${deployment_name}", "<deployment>")

    assert rest(frozen["cmssw-releases.cmssw_releases"]) == rest(live["cmssw_releases"])


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
    """One pack template, named by `<pack>/<file>` as well as its source."""
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
    entry = _entry("docsite", "docsite/docsite.yaml")
    adapter = _adapter_class(entry)(**_params("docsite", "docsite/docsite.yaml", tmp_path))
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
    entry = _entry("jira", "jira/jira.yaml")
    adapter = _adapter_class(entry)(**_params("jira", "jira/jira.yaml", tmp_path))
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
    # The frozen template, not the live default: that one fetches the real
    # catalogue over the network, which no test here may do.
    entry = yaml.safe_load(FROZEN_CMSSW.read_text(encoding="utf-8"))["cmssw-releases.cmssw_releases"]
    params = dict(entry["params"], map_cache_path=str(catalogue), map_cache_digest=digest)
    assert params["fetch"] is False
    adapter = _adapter_class(entry)(**params)
    run, facts = _run(adapter)
    assert run.completed_scope is True
    releases = [f for f in facts if isinstance(f, NodeFact) and f.subtype == "cmssw_release"]
    # The reader emits one node per release plus the family node they share
    # (`cmssw_release_family:CMSSW_14_0`, labelled `CMSSW_14_0`).
    assert {node.attrs["label"] for node in releases} == {
        "CMSSW_14_0_1",
        "CMSSW_14_0_2",
        "CMSSW_14_0",
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


#: Entries that strict admission still refuses, and why. Empty since the
#: docsite template stopped authoring the substrate-reserved `source_name`
#: (`RESERVED_ADAPTER_PARAM_NAMES`; the reader's default is the same name).
#: Add an entry with its refusal code if one is ever knowingly shipped; the
#: test will tell you when it is fixed.
STRICT_PATH_KNOWN_REFUSALS: dict = {}


def test_the_substrate_strict_admission_check_consumes_every_parameter():
    """Exercise okg's own refusal, not a restatement of it.

    `source_adapter_init_params` is the function that raises
    `source_param_unconsumed`. It takes the strict path only for an entry whose
    admission contract is `strict_v1`, so the entry is relabelled here; nothing
    else about it changes and no database is touched. The packs ship on the
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


@pytest.mark.parametrize(
    "answer",
    [
        r"C:\reviewed\twiki-dump",
        "/a path/with spaces/twiki-dump",
        "/plain/twiki-dump",
    ],
    ids=["backslashes", "spaces", "plain"],
)
def test_an_install_answer_reaches_the_source_byte_for_byte(tmp_path, answer):
    """Answers are substituted as text, so the quoting style is load-bearing.

    `build_plan` interpolates each source template's TEXT and parses the
    result as YAML, and the substitution escapes nothing (okg
    `substrate/catalog/profile.py`, `replace_match` returns `str(v)` as-is).
    A double-quoted placeholder therefore hands the answer to YAML's escape
    rules: a Windows path came back with a carriage return where its `\\r`
    had been, and the only symptom was an unexplained cache_missing. Single
    quotes take the bytes literally. Checked on the twiki pack's `eos_root`,
    rendered and planned the way an install does.
    """
    from packs.pack_support import okg_plan, render_alone

    rendered = render_alone(tmp_path, "twiki")
    plan = okg_plan(rendered, tmp_path, {"twiki_eos_root": answer})
    body = yaml.safe_load(plan.source_defaults["twiki.twiki_eos"])
    assert body["twiki.twiki_eos"]["params"]["eos_root"] == answer


# --- TWiki comes from an EOS dump, and only from one -------------------------

#: A tiny synthetic dump: one topic per file, raw markup with %META lines, as
#: the EOS export lays them out. Invented text only; no real wiki content.
_DUMP_TOPICS = {
    "DemoHome": (
        '%META:TOPICINFO{author="jdoe" date="1700000000" version="3"}%\n'
        "---+ Demo home\n"
        "Welcome. See [[DemoTransfers][the transfer notes]].\n"
    ),
    "DemoTransfers": (
        '%META:TOPICINFO{author="jdoe" date="1700000000" version="1"}%\n'
        '%META:TOPICPARENT{name="DemoHome"}%\n'
        "---+ Demo transfers\n"
        "Transfers are drained nightly by the demo team.\n"
    ),
}


def test_twiki_in_the_bundle_is_the_eos_dump_reader_and_nothing_else():
    """The decision: one TWiki parser path, no live crawl, no web fallback.

    Every bundle entry that names a TWiki reader must be ``twiki_eos``, and its
    ``eos_root`` must come from the install answer so an operator points it at
    their dump rather than editing the file.
    """
    twiki_entries = [
        (filename, name, entry)
        for filename, name, entry in ALL_ENTRIES
        if entry.get("module") == "archi.sources.twiki"
    ]
    assert [(f, n, e["class"]) for f, n, e in twiki_entries] == [
        ("twiki/twiki_eos.yaml", "twiki_eos", "TwikiEOSAdapter")
    ]
    params = twiki_entries[0][2]["params"]
    assert params["eos_root"] == "${eos_root}"
    assert params["required"] is True

    questions = yaml.safe_load((PACKS / "twiki" / "questions.yaml").read_text(encoding="utf-8"))
    ids = {q["id"] for q in questions["init_questions"]}
    assert ids == {"eos_root", "physics_filter"}


@pytest.mark.parametrize(
    "dump_name", ["dump", "a dump with spaces"], ids=["plain", "spaces"]
)
def test_the_twiki_eos_root_answer_reaches_the_reader_and_reads_the_dump(
    tmp_path, monkeypatch, dump_name
):
    """Answer ``--twiki-eos-root`` and the reader reads the dump.

    Renders the twiki pack and goes through okg's own ``build_plan`` so the
    placeholder and the quoting are the ones an installer hits, then runs the
    adapter on the resolved parameters against a synthetic dump.
    """
    from packs.pack_support import okg_plan, render_alone

    dump = tmp_path / dump_name
    dump.mkdir()
    for topic, text in _DUMP_TOPICS.items():
        (dump / f"{topic}.txt").write_text(text, encoding="utf-8")

    rendered = render_alone(tmp_path / "render", "twiki")
    plan = okg_plan(rendered, tmp_path / "render", {"twiki_eos_root": str(dump)})
    entry = yaml.safe_load(plan.source_defaults["twiki.twiki_eos"])["twiki.twiki_eos"]
    assert entry["params"]["eos_root"] == str(dump)

    adapter = _adapter_class(entry)(**entry["params"])
    run, facts = _run(adapter)
    assert run.completed_scope is True
    pages = {
        f.node_id for f in facts
        if isinstance(f, NodeFact) and f.subtype == "documentation_page"
    }
    assert pages == {"twiki:CMS:DemoHome", "twiki:CMS:DemoTransfers"}


def test_the_twiki_eos_reader_refuses_a_missing_dump_instead_of_shrinking(tmp_path):
    entry = _entry("twiki_eos", "twiki/twiki_eos.yaml")
    params = _params(
        "twiki_eos",
        "twiki/twiki_eos.yaml",
        tmp_path,
        extra={"${eos_root}": str(tmp_path / "no-such-dump")},
    )
    # The install answer is YAML true/false once substituted; the reader
    # refuses the raw placeholder (or any typo) instead of reading it as true.
    with pytest.raises(ValueError, match="true or false"):
        _adapter_class(entry)(**params)
    params["physics_filter"] = False
    run = _adapter_class(entry)(**params).run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False


# --- comp-ops readers with no bundle entry ---------------------------------
#
# These readers ship no source-default: an operator registers one by copying
# the registry-entry template out of the reader module's docstring. Before
# they had adapters that template named the bare reader, so every run failed
# on `next_cursor` exactly as the bundle's own sources once did. The template
# is what an operator copies, so the test reads it rather than restating it.

#: source id -> the files its template's params name, as a tiny valid cache.
#: The shapes follow the readers' own tests under tests/sources/.
_COMPOPS_CACHES = {
    "cric": {
        "sites_path": {
            "T2_US_MIT": {
                "tier_level": 2,
                "facility": "MIT",
                "sitedb_title": "MIT Bates",
                "computeunits": {"MIT-CE1": {}},
            }
        },
        "storage_units_path": {
            "MIT-SE": {"type": "DISK", "site": {"name": "T2_US_MIT"}}
        },
        "compute_units_path": {"MIT-CE1": {"corepower": 11.0}},
        "facilities_path": {"MIT": {"cmssites": [{"name": "T2_US_MIT"}]}},
        "responsibilities_path": {
            "result": [["adalove", "MIT Bates", "Site Executive"]]
        },
    },
    "cric_core": {
        "services_path": {
            "reqmgr2-cmsweb.cern.ch": {
                "type": "webservice",
                "endpoint": "cmsweb.cern.ch/reqmgr2",
                "rcsite": "CERN-PROD",
            }
        },
        "rcsites_path": {
            "CERN-PROD": {"sites": [{"name": "T0_CH_CERN", "vo_name": "cms"}]}
        },
        "federations_path": {
            "CH-CERN": {"vos": ["cms"], "rcsites": ["CERN-PROD"], "pledges": {}}
        },
    },
    "dqm": {
        "records_path": [
            {
                "filename": "Cert_Collisions2024_378981_385194_Golden.json",
                "cert_name": "Cert_Collisions2024_378981_385194_Golden",
                "run_range": [378981, 385194],
                "datasets": ["/Muon0/Run2024C-PromptReco-v1/DQMIO"],
            }
        ],
    },
    "gocdb_downtimes": {
        "records_path": [
            {
                "downtime_id": 101,
                "severity": "OUTAGE",
                "classification": "SCHEDULED",
                "start_date": "2026-08-01T06:00:00",
                "end_date": "2026-08-01T18:00:00",
                "hosted_by": "T2_US_MIT",
                "service_type": "CE",
                "hostname": "ce01.cmsaf.mit.edu",
            }
        ],
        "sites_path": {"T2_US_MIT": {}},
        "services_path": {"reqmgr2": {"endpoint": "cmsweb.cern.ch/reqmgr2"}},
    },
    "conddb_global_tags": {
        "records_path": [{"name": "140X_dataRun3_v2", "release": "CMSSW_14_0_X"}],
        "cmssw_records_path": [{"label": "CMSSW_14_0_1"}],
    },
    "dbs_datasets": {
        "records_path": [
            {
                "dataset": "/TTto2L2Nu/Run3Summer23-v1/AODSIM",
                "data_tier_name": "AODSIM",
                "primary_ds_name": "TTto2L2Nu",
                "processed_ds_name": "Run3Summer23-v1",
            }
        ],
    },
    "wmstats_workflows": {
        "records_path": [
            {
                "RequestName": "pdmvserv_task_TOP-Run3Summer23-00001",
                "RequestType": "TaskChain",
                "RequestStatus": "running-open",
                "OutputDatasets": ["/TTto2L2Nu/Run3Summer23-v1/AODSIM"],
            }
        ],
    },
    # Reads no cache: its template authors no parameters, and the reader
    # falls back to its configured default repository list.
    "github_repos": {},
}

#: source id -> (module, a subtype a successful run must emit).
_COMPOPS_READERS = {
    "cric": ("archi.sources.cric", "site"),
    "cric_core": ("archi.sources.cric", "infrastructure_service"),
    "dqm": ("archi.sources.dqm", "data_certification"),
    "gocdb_downtimes": ("archi.sources.gocdb", "downtime"),
    "conddb_global_tags": ("archi.sources.conddb", "global_tag"),
    "dbs_datasets": ("archi.sources.dbs", "dataset"),
    "wmstats_workflows": ("archi.sources.wmstats", "workflow"),
    "github_repos": ("archi.sources.github_repos", "software_repository"),
}


def _docstring_template(module_name, source_id):
    """The registry entry for ``source_id`` in the module's docstring.

    Each reader module ends its docstring with a literal YAML block after a
    ``::`` line; that block is what an operator pastes into a registry.
    """
    module = importlib.import_module(module_name)
    _head, sep, block = (module.__doc__ or "").rpartition("::\n")
    assert sep, f"{module_name} has no registry-entry template"
    templates = yaml.safe_load(textwrap.dedent(block))
    assert source_id in templates, f"{module_name} template has no {source_id}"
    entry = templates[source_id]
    assert entry["module"] == module_name
    return entry


@pytest.mark.parametrize("source_id", sorted(_COMPOPS_READERS))
def test_compops_template_names_an_adapter_that_runs(tmp_path, source_id):
    module_name, subtype = _COMPOPS_READERS[source_id]
    entry = _docstring_template(module_name, source_id)
    module = importlib.import_module(module_name)
    cls = getattr(module, entry["class"], None)
    assert cls is not None, f"{module_name} defines no {entry['class']}"
    assert inspect.isclass(cls) and issubclass(cls, ConnectorAdapter), (
        f"the {source_id} template names {entry['class']}, which the substrate "
        "runner cannot drive; it must name the reader's ConnectorAdapter"
    )

    # Authority: string literals the substrate can read, equal to the
    # reader's own values and to what the template declares.
    reader = inspect.getattr_static(cls, "reader_class")
    source_path = Path(module.__file__)
    for attr in ("profile", "change_probe_kind"):
        literal = _class_level_constant(source_path, entry["class"], attr)
        assert literal == getattr(reader, attr)
        assert inspect.getattr_static(cls, attr) == literal
    assert inspect.getattr_static(cls, "profile") == entry["source_class"]

    # The template's own parameters bind against the adapter's signature.
    signature = inspect.signature(cls)
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    params = dict(entry.get("params") or {})
    signature.bind(**params)

    # And it runs: write a tiny cache at the template's paths, then drive the
    # adapter the way the substrate runner does.
    caches = _COMPOPS_CACHES[source_id]
    assert set(caches) == set(params), "the fixture must cover every template path"
    for key, payload in caches.items():
        path = tmp_path / params[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    if "base" in signature.parameters:
        params["base"] = str(tmp_path)
    adapter = cls(**params)
    assert adapter.preflight().status == "ok"
    run, facts = _run(adapter)
    assert run.completed_scope is True
    primary = [f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype]
    assert primary
    # The declared identity fields are the keys the reader actually emits for
    # its primary records, so the declaration describes the real record key.
    for fact in primary:
        assert set(fact.source_record_id) == set(entry["record_identity_fields"]), (
            f"{source_id}: {fact.node_id} is keyed by {sorted(fact.source_record_id)}"
            f" but the template declares {entry['record_identity_fields']}"
        )


def test_compops_template_profile_tuples_are_all_accepted():
    """Call the substrate's own validator on each template, as install does.

    Each template once declared a combination `okg install` refused with
    `deployment.source_registry.profile_invalid`: `record_identity_kind:
    remote_id` under discovery_crawl or reference_catalog, and `content_hash`
    revisions under mutable_api for WMStats. The fix is in the declaration
    only: the readers' emitted record keys and node ids are unchanged, and
    match what the okg-deployments cms registry declared for the same code.
    """
    from okg.substrate.sources.profiles import validate_profile_tuple

    refused = {}
    for source_id, (module_name, _subtype) in sorted(_COMPOPS_READERS.items()):
        entry = _docstring_template(module_name, source_id)
        try:
            validate_profile_tuple(
                source_class=entry["source_class"],
                record_identity_kind=entry.get("record_identity_kind"),
                source_revision_kind=entry.get("source_revision_kind"),
                deletion_semantics=entry.get("deletion_semantics"),
                publication_mode=entry.get("publication_mode"),
            )
        except ValueError as exc:
            refused[source_id] = str(exc)
    assert not refused, (
        "these templates would fail `okg install` with "
        "deployment.source_registry.profile_invalid:\n\n"
        + "\n\n".join(f"{name}: {why}" for name, why in sorted(refused.items()))
    )


@pytest.mark.parametrize("source_id", sorted(_COMPOPS_READERS))
def test_compops_template_passes_strict_registry_admission(source_id):
    """Admit the pasted template the way a registry load does, then bind it.

    Uses okg's own admission and `source_adapter_init_params` on the strict
    contract, as the bundle test above does. This is what caught the GitHub
    template's bare `params:` (null), which strict admission refuses with
    `source_params_not_mapping`.
    """
    import dataclasses

    from okg.substrate.ingest.adapter_factory import source_adapter_init_params
    from okg.substrate.sources.registry import (
        STRICT_ADMISSION_CONTRACT,
        admit_source_registry_document,
    )

    module_name, _subtype = _COMPOPS_READERS[source_id]
    entry = _docstring_template(module_name, source_id)
    admission = admit_source_registry_document(
        {"sources": {source_id: entry}}, registry_path=Path("registry.yaml")
    )
    strict = dataclasses.replace(
        admission.entries[source_id], admission_contract=STRICT_ADMISSION_CONTRACT
    )
    params = source_adapter_init_params(
        strict,
        dsn="postgresql://localhost/unused",
        deployment="unused",
        adapter_class=getattr(importlib.import_module(module_name), entry["class"]),
    )
    assert set(entry["params"]) <= set(params)


def test_a_bare_compops_reader_lacks_the_field_the_runner_reads(tmp_path):
    """The defect the adapters exist for, shown on one reader.

    If a reader's run result ever grows ``next_cursor`` the adapters become
    redundant; this test says so rather than letting them rot silently.
    """
    from archi.sources.dqm import DQMSource

    path = tmp_path / "data" / "dqm" / "records.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_COMPOPS_CACHES["dqm"]["records_path"]))
    run = DQMSource(base=str(tmp_path)).run("run-1", mode="scope_complete")
    assert not hasattr(run, "next_cursor")
