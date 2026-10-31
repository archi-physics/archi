"""cluster-manager is a bundle of packs with no CERN dependency (ADR 0002 section 5; PACT r3-bundle-cluster-manager).

Offline checks of the bundle, the five new packs and their templates:

- acc.r3-bundle-cluster-manager.admission: each template passes the substrate's profile check and strict
  admission, binds its parameters against its adapter, and runs from its shipped parameters over the
  recorded fixtures; every rendered profile passes `okg install --dry-run` and lints with no blocker.
- The install proof against a real Postgres is tests/packs/test_pack_isolation.py.
"""
import dataclasses
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from okg.deployment import ConnectorAdapter, NodeFact

from archi.bundles.render import PROFILES_DIR, render
from archi.bundles.resolve import resolve
from packs import cluster_fixtures
from packs.pack_support import PACKS_DIR, FIXTURES, manifest, shipped_catalog, templates

from okg.substrate.deployment_bootstrap.profile_init import build_plan, discover_profile

from bundles.test_render_okg import _write_scaffold

PYTHON_DIR = str(Path(__file__).resolve().parents[1])
OKG = str(Path(sys.executable).with_name("okg"))
BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "cluster-manager"
NEW_PACKS = ["servicenow", "tickets", "slurm", "htcondor", "node-health"]
#: Every pack that arrived from the CERN bundles: none may be a dependency of this bundle.
CERN_PACKS = {
    "indico", "twiki", "cds", "jira", "code-text", "code-symbols", "core-playbooks", "cms-catalogs", "cmssst",
    "cmssw-releases", "dqm", "hypernews", "monit-live", "analysis-notes", "cms-analyses", "instance-health",
}
ANSWERS = {
    "deployment_name": "qt",
    "postgres_dsn": "postgresql://localhost:1/qt",
    "archi_data_root": "/srv/archi-data",
    "git_code_repo_url": "https://example.org/demo.git",
    "servicenow_instance_url": "https://example.service-now.com",
    "tickets_base_url": "https://redmine.example.org",
}
#: source id -> (primary subtype, adapter fixture writer)
SOURCES = {
    "servicenow": ("servicenow_incident", cluster_fixtures.write_servicenow_cache),
    "tickets": ("tracker_ticket", cluster_fixtures.write_tickets_cache),
    "slurm": ("slurm_node", cluster_fixtures.write_slurm_cache),
    "htcondor": ("condor_machine", cluster_fixtures.write_condor_cache),
    "node_health": ("compute_node", cluster_fixtures.write_node_health_cache),
}


def _okg(args, cwd):
    """Run okg with this working tree's `archi` first on its path, so the subprocess loads the adapters
    under test and not whichever archi the venv has installed (a stock venv may hold another checkout)."""
    env = {k: v for k, v in os.environ.items() if k != "OKG_PROFILES_DIR"}
    env["HOME"] = str(cwd)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (PYTHON_DIR, os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([OKG, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


def _blockers(deployment_dir, cwd):
    report = json.loads(_okg(["deployment", "lint", str(deployment_dir), "--json"], cwd=cwd).stdout)
    return {(f["location"].get("pointer", ""), f["code"]) for f in report["findings"]
            if f["severity"] in ("blocker", "error")}


def _bundle():
    return yaml.safe_load((BUNDLE / "bundle.yaml").read_text())


@pytest.fixture(scope="module")
def catalog():
    return shipped_catalog()


def _closure(catalog, names):
    out = []

    def visit(name):
        for dep in catalog.packs[name].requires_packs:
            visit(dep)
        if name not in out:
            out.append(name)

    for name in names:
        visit(name)
    return out


# --- the bundle -----------------------------------------------------------------------------


def test_the_packs_are_the_adr_selection():
    packs = _bundle()["packs"]
    assert packs["default"] == ["docsite", "tickets", "ops-record"]
    assert sorted(packs["optional"]) == sorted(["servicenow", "slurm", "htcondor", "git-code", "node-health", "ticket-triage"])


def test_the_bundle_depends_on_no_cern_pack(catalog):
    packs = _bundle()["packs"]
    everything = _closure(catalog, packs["default"] + packs["optional"])
    assert not set(everything) & CERN_PACKS, sorted(set(everything) & CERN_PACKS)
    assert "extends" not in _bundle(), "it extends no bundle: cern-team's defaults are CERN packs"


def test_nothing_in_the_new_packs_or_the_bundle_manifest_names_cern():
    files = [p for pack in NEW_PACKS for p in (PACKS_DIR / pack).rglob("*") if p.is_file()]
    files += [BUNDLE / "bundle.yaml"]
    assert files
    for path in files:
        assert not re.search(r"\bcern\b", path.read_text(encoding="utf-8"), re.I), path


def test_the_bundle_is_a_manifest_and_a_readme():
    # eval/ holds the smoke questions (PACT r3-bundle-evals), not profile files.
    assert sorted(p.name for p in BUNDLE.iterdir()) == ["README.md", "bundle.yaml", "eval"]


def test_the_bundle_sets_the_documentation_to_the_schedulers_own_manuals():
    urls = _bundle()["answers"]["docsite_urls"]
    assert "slurm.schedmd.com" in urls and "htcondor" in urls and "cern" not in urls.lower()


def test_the_default_install_needs_no_credential(catalog):
    for name in _bundle()["packs"]["default"]:
        assert catalog.packs[name].needs_credentials is False, name


def test_servicenow_says_it_needs_a_credential_and_names_each_by_variable():
    pack = manifest("servicenow")
    assert pack["needs_credentials"] is True
    assert set(pack["credentials"]) == {
        "SERVICENOW_CLIENT_ID", "SERVICENOW_CLIENT_SECRET", "SERVICENOW_USERNAME", "SERVICENOW_PASSWORD",
    }
    entry = templates("servicenow")["servicenow"]
    assert set(entry["credential_refs"]) == set(pack["credentials"])
    assert not re.search(r"(secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9]{8,}", yaml.safe_dump(entry), re.I)


def test_the_credential_free_packs_say_so():
    for name in ("tickets", "slurm", "htcondor", "node-health"):
        assert manifest(name)["needs_credentials"] is False, name


def test_every_new_context_pack_has_fixtures_a_coverage_invariant_and_init_questions():
    for name in NEW_PACKS:
        assert name in FIXTURES, name
        pack = manifest(name)
        assert pack["kind"] == "context" and pack["init_questions"] and pack["invariants"] and pack["freshness"]
        body = yaml.safe_load((PACKS_DIR / name / pack["invariants"]).read_text())["invariants"]
        assert [i["name"] for i in body] == [name.replace("-", "_") + "_coverage"]
        assert body[0]["severity"] == "error"
        # the schema slice and its narrowings are the pack's own
        assert (PACKS_DIR / name / "schemas" / f"{name.replace('-', '_')}.yaml").is_file()
        assert (PACKS_DIR / name / "schemas" / "bridges" / f"{name.replace('-', '_')}.yaml").is_file()


# --- every template: profile check, strict admission, binding, and a run over the fixtures ----------


def _template(source_id):
    pack = "node-health" if source_id == "node_health" else source_id
    return pack, templates(pack)[source_id]


def _interpolate(value, tmp_path, pack):
    """Fill `${...}` the way an install does: base answers, then the pack's question defaults and the fixture answers."""
    questions = {q["id"]: q for q in yaml.safe_load((PACKS_DIR / pack / "questions.yaml").read_text())["init_questions"]}
    given = FIXTURES[pack](tmp_path, _NoServer())
    answers = {"deployment_name": "qt", "archi_data_root": str(tmp_path)}
    answers.update({qid: q.get("default") for qid, q in questions.items() if q.get("default") is not None})
    answers.update(given)

    def fill(item):
        if isinstance(item, str):
            return re.sub(r"\$\{([A-Za-z_]\w*)\}", lambda m: str(answers[m.group(1)]), item)
        if isinstance(item, list):
            return [fill(i) for i in item]
        if isinstance(item, dict):
            return {k: fill(v) for k, v in item.items()}
        return item

    return fill(value)


class _NoServer:
    """The new packs fetch nothing live: a fixture that tried to register a route would fail here."""

    def add(self, *args, **kwargs):
        raise AssertionError("a cluster-manager fixture registered an HTTP route")


@pytest.mark.parametrize("source_id", sorted(SOURCES))
def test_the_template_names_an_adapter_the_substrate_can_drive(source_id, tmp_path):
    pack, entry = _template(source_id)
    entry = _interpolate(entry, tmp_path, pack)
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])
    assert inspect.isclass(cls) and issubclass(cls, ConnectorAdapter)
    reader = inspect.getattr_static(cls, "reader_class")
    # the substrate reads these two by parsing the module, so they are literals equal to the reader's own
    for attr in ("profile", "change_probe_kind"):
        assert inspect.getattr_static(cls, attr) == getattr(reader, attr)
    assert entry["source_class"] == cls.profile
    params = dict(entry["params"])
    inspect.signature(cls).bind(**params)
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(cls).parameters.values())


@pytest.mark.parametrize("source_id", sorted(SOURCES))
def test_the_template_passes_the_profile_check_and_strict_registry_admission(source_id, tmp_path):
    from okg.substrate.ingest.adapter_factory import source_adapter_init_params
    from okg.substrate.sources.profiles import validate_profile_tuple
    from okg.substrate.sources.registry import STRICT_ADMISSION_CONTRACT, admit_source_registry_document

    pack, entry = _template(source_id)
    entry = _interpolate(entry, tmp_path, pack)
    validate_profile_tuple(
        source_class=entry["source_class"],
        record_identity_kind=entry["record_identity_kind"],
        source_revision_kind=entry["source_revision_kind"],
        deletion_semantics=entry["deletion_semantics"],
        publication_mode=entry["publication_mode"],
    )
    admission = admit_source_registry_document({"sources": {source_id: entry}}, registry_path=Path("registry.yaml"))
    strict = dataclasses.replace(admission.entries[source_id], admission_contract=STRICT_ADMISSION_CONTRACT)
    params = source_adapter_init_params(
        strict, dsn="postgresql://localhost/unused", deployment="qt",
        adapter_class=getattr(importlib.import_module(entry["module"]), entry["class"]),
    )
    assert set(entry["params"]) <= set(params)


@pytest.mark.parametrize("source_id", sorted(SOURCES))
def test_the_fixture_input_yields_the_records_the_template_declares(source_id, tmp_path):
    """Run each source from its shipped parameters over its fixture, as the substrate runner does."""
    pack, entry = _template(source_id)
    subtype, _ = SOURCES[source_id]
    entry = _interpolate(entry, tmp_path, pack)
    cls = getattr(importlib.import_module(entry["module"]), entry["class"])
    adapter = cls(**dict(entry["params"]), base=str(tmp_path))
    assert adapter.preflight().status == "ok"
    run = adapter.reader.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    assert run.completed_scope is True

    declared = {n["subtype"] for n in entry["admission_policy"]["output_signature"]["nodes"]}
    emitted = {f.subtype for f in facts if isinstance(f, NodeFact)}
    assert emitted == declared, "the output signature lists exactly what the source emits"
    edges = {(e.src.split(":")[0], e.edge_type, e.dst.split(":")[0]) for e in facts if not isinstance(e, NodeFact)}
    assert edges == {(e["src_subtype"], e["edge_type"], e["dst_subtype"])
                     for e in entry["admission_policy"]["output_signature"]["edges"]}
    primary = [f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype]
    assert primary
    for fact in primary:
        assert set(fact.source_record_id) == set(entry["record_identity_fields"]), fact.node_id


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_the_coverage_invariant_requires_subtypes_the_fixture_emits(pack, tmp_path):
    from packs.pack_support import emitted_subtypes, invariants, required_subtypes

    required = required_subtypes(invariants(pack)[0])
    assert required and set(required) <= emitted_subtypes(pack)


# --- okg accepts every rendered profile ------------------------------------------------------------


def _matrix():
    optional = _bundle()["packs"]["optional"]
    return [[]] + [[p] for p in optional] + [optional]


@pytest.mark.parametrize("with_packs", _matrix(), ids=lambda m: "+".join(m) or "defaults")
def test_okg_accepts_the_rendered_profile_and_lints_clean(catalog, tmp_path, with_packs):
    instance = tmp_path / "instance"
    rendered = render(resolve(catalog, ["cluster-manager"], with_packs=with_packs), instance)
    asked = {q["id"] for q in rendered.composition.questions}
    answers = {qid: value for qid, value in ANSWERS.items() if qid in asked}
    missing = [q["id"] for q in rendered.composition.questions
               if q.get("required") and q["id"] not in answers and q["id"] != "deployment_name"]
    assert not missing, f"a required question the test does not answer: {missing}"

    flags = []
    for qid, value in answers.items():
        if qid != "deployment_name":
            flags += [f"--{qid.replace('_', '-')}", value]
    result = _okg(
        ["install", "--profile", rendered.name, "--deployment-name", "qt",
         "--deployments-root", str(tmp_path / "deployments"), "--dry-run", "--json", "--non-interactive", *flags],
        cwd=instance,
    )
    assert result.returncode == 0, result.stderr[-2000:] + result.stdout[-2000:]
    report = json.loads(result.stdout)
    assert report["status"] == "dry_run_ok" and report["blockers"] == [], report

    plan = build_plan(
        profile=discover_profile(rendered.name, profiles_root=instance / PROFILES_DIR),
        deployment_name="qt",
        deployments_root=tmp_path / "deployments-rendered",
        raw_answers=answers,
    )
    _write_scaffold(plan)
    # An automation pack turns okg's runtime on (archi.bundles.compose), and okg 0.2.0's lint then wants
    # `postgres.database`. okg's installer writes `postgres: {dsn: ...}` itself, replacing any `postgres:`
    # block a profile's defaults carry, so no profile can declare it. That one finding is okg's, and is
    # expected for a profile with an automation; every other blocker still fails the test.
    automation = {("/postgres/database", "deployment.runtime.database_undeclared")}
    expected = automation if rendered.composition.automations else set()
    assert _blockers(plan.deployment_dir, instance) == expected


def test_a_bare_render_selects_the_default_packs_only(catalog, tmp_path):
    rendered = render(resolve(catalog, ["cluster-manager"]), tmp_path / "instance")
    assert sorted(rendered.composition.sources) == ["docsite.docsite", "tickets.tickets"]
    names = [i["name"] for i in rendered.composition.invariants]
    assert [n for n in names if n.endswith("_coverage")] == ["docsite_coverage", "tickets_coverage"]
    assert [n for n in names if not n.endswith("_coverage")] == ["tickets_link_ticket_author"]
    for sid, text in rendered.composition.sources.items():
        assert not yaml.safe_load(text)[sid].get("credential_refs", []) or sid == "tickets.tickets"


def test_the_bundle_answer_replaces_the_cern_documentation_default(catalog, tmp_path):
    rendered = render(resolve(catalog, ["cluster-manager"]), tmp_path / "instance")
    defaults = {q["id"]: q.get("default") for q in rendered.composition.questions}
    assert "slurm.schedmd.com" in defaults["docsite_urls"] and "rucio" not in defaults["docsite_urls"]


def test_the_chat_prompt_names_each_selected_packs_fragment(catalog, tmp_path):
    rendered = render(
        resolve(catalog, ["cluster-manager"], with_packs=_bundle()["packs"]["optional"]), tmp_path / "instance")
    prompt = (rendered.profile_dir / "skills" / "chat-system-prompt.md").read_text(encoding="utf-8")
    for heading in ("## Tickets", "## ServiceNow", "## Slurm", "## HTCondor", "## Node health"):
        assert heading in prompt
    assert "cern" not in prompt.lower()


def test_tickets_asks_for_its_tracker_url_instead_of_defaulting_to_a_placeholder():
    """ADR 0002 section 3.2: a bundle with nothing configured publishes when a default pack is credential-free
    (docsite, which downloads public pages at init) or a required init question says what is missing. The ADR
    and the PACT make tickets a default, and no tracker URL is a safe default, so its URL is a required question."""
    questions = {q["id"]: q for q in yaml.safe_load((PACKS_DIR / "tickets" / "questions.yaml").read_text())["init_questions"]}
    assert questions["base_url"]["required"] is True and "default" not in questions["base_url"]
    assert manifest("docsite")["needs_credentials"] is False and manifest("docsite").get("prepare")


def test_the_node_health_page_says_what_the_exporter_must_write():
    description = manifest("node-health")["description"]
    for word in ("node-health/records.json", "node", "cluster", "observed_at", "checks", "status", "detail"):
        assert word in description


def test_an_automation_profile_lints_clean_once_archi_declares_the_runtime_database(catalog, tmp_path):
    # An automation pack turns okg's runtime on (archi.bundles.compose), and okg 0.2.0's lint then
    # blocks until postgres.database is declared, which `okg install` never writes. `archi init`
    # declares it right after the install (archi.cli.init.declare_runtime_database); this is the same
    # step on the same scaffold okg writes, then okg's own lint.
    from archi.cli.init import declare_runtime_database

    instance = tmp_path / "instance"
    rendered = render(resolve(catalog, ["cluster-manager"], with_packs=["instance-health"]), instance)
    assert rendered.composition.automations
    asked = {q["id"] for q in rendered.composition.questions}
    answers = {qid: value for qid, value in ANSWERS.items() if qid in asked}
    plan = build_plan(
        profile=discover_profile(rendered.name, profiles_root=instance / PROFILES_DIR),
        deployment_name="qt",
        deployments_root=tmp_path / "deployments-rendered",
        raw_answers=answers,
    )
    _write_scaffold(plan)
    undeclared = ("/postgres/database", "deployment.runtime.database_undeclared")
    assert undeclared in _blockers(plan.deployment_dir, instance)
    assert declare_runtime_database(plan.deployment_dir, "qt") is True
    assert _blockers(plan.deployment_dir, instance) == set()
