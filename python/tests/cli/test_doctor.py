"""req.r3-golden-path.doctor: `archi doctor` (archi.cli.doctor, archi.stack.checks).

- acc.r3-golden-path.checks: each footgun check has a failing fixture.

A healthy instance (written by `archi init --no-okg`, then given what okg
and `archi up` would add) passes every check with fake probes. Each fixture
below breaks exactly one thing and asserts that its check fails with a
remediation line. `test_every_check_has_a_failing_fixture` keeps the two
lists equal, and `test_every_check_is_in_the_ledger` ties each check to its
entry in docs/docs/operating/footguns.md.
"""
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.cli.doctor import doctor
from archi.cli.init import commit_all, init
from archi.stack.checks import CHECKS, FAIL, PASS, WARN, Context, Probes, run_checks
from archi.stack.instance import Instance, read_env, write_env
from archi.stack.okg import okg_version

REPO = Path(__file__).resolve().parents[3]
LEDGER = REPO / "docs" / "docs" / "operating" / "footguns.md"


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for key, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.org"),
                       ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.org")):
        monkeypatch.setenv(key, value)


@pytest.fixture(scope="module")
def shipped():
    return discovery.discover([], [])


@pytest.fixture
def instance(tmp_path, shipped) -> Instance:
    """An instance as `archi init` + `archi up` leave it, without okg or containers."""
    inst = init("myteam", bundles=["cern-team"], catalog=shipped, directory=tmp_path / "myteam",
                data_root=tmp_path / "data", run_okg=False, echo=lambda _: None)
    dep = inst.deployment_dir
    dep.mkdir(parents=True)
    (dep / "deployment.yaml").write_text(yaml.safe_dump({
        "name": "myteam",
        "profile": {"init_answers": {
            "archi_data_root": str(inst.data_root),
            "twiki_eos_root": str(inst.data_root / "twiki-snapshot"),
            "code_extractors_dir": "${deployment_dir}",
        }},
        "chat": {"ui": {"site_name": "Team chat"}, "mcp": {"port": 8765}},
    }))
    (dep / "source_registry.yaml").write_text(yaml.safe_dump({"sources": {
        "cmssw_releases": {"params": {"cache_root": str(inst.data_root / "cmssw")}},
    }}))
    secrets = read_env(inst.env_path)
    secrets["ARCHI_OKG_MCP_PASSWORD"] = "mcp-role-pw"
    write_env(inst.env_path, secrets)
    inst.role_secrets_path.parent.mkdir(parents=True, exist_ok=True)
    inst.role_secrets_path.write_text(json.dumps({"roles": {
        "okg_mcp": f"postgresql://okg_mcp:mcp-role-pw@127.0.0.1:{inst.ports.graph_db}/myteam",
    }}))
    from archi.stack import compose as stack_compose

    stack_compose.write_stack_files(inst)
    commit_all(inst.dir, "deployment")
    return inst


def healthy_db(dsn, sql):
    if "shared_preload_libraries" in sql:
        return [("timescaledb,pg_cron,pg_textsearch",)]
    if "cron.database_name" in sql:
        return [("myteam",)]
    if "current_user" in sql:
        return [(dsn.split("://", 1)[1].split(":", 1)[0],)]
    raise AssertionError(sql)


def shipped_source_ids() -> list[str]:
    """Every `<pack>.<source>` id the shipped packs give a max_staleness."""
    ids = []
    for manifest in sorted((REPO / "bundles" / "packs").glob("*/pack.yaml")):
        ids += list((yaml.safe_load(manifest.read_text()) or {}).get("freshness", {}))
    return ids


def watermarks(age_hours=1, **late_hours):
    """okg's `doctor --check sources` report: each shipped source last reconciled
    `age_hours` ago, except those named in `late_hours` (source id with `.` as `__`)."""
    now = datetime.now(timezone.utc)
    sources = []
    for sid in shipped_source_ids():
        hours = late_hours.get(sid.replace(".", "__"), age_hours)
        stamp = None if hours is None else (now - timedelta(hours=hours)).isoformat()
        sources.append({"name": sid, "ownership_id": f"myteam.{sid.split('.', 1)[1]}",
                        "coverage": {"watermark": {"last_reconcile_at": stamp} if stamp else {}}})
    return {"returncode": 0, "data": {"ok": True, "runtime_available": True, "runtime_error": None,
                                      "sources": sources, "errors": []}}


def healthy_okg(*args):
    if args[:3] == ("doctor", "--check", "sources"):
        return watermarks()
    if args[0] == "status":
        return {"returncode": 0, "data": {"publishing": {
            "latest_published_generation": "okg:myteam:gen:1",
            "latest_published_node_count": 2481, "latest_published_edge_count": 2117}}}
    if args[:2] == ("deployment", "lint"):
        return {"returncode": 0, "data": {"findings": [{"severity": "skipped", "message": "x"}]}}
    if args[0] == "doctor":
        return {"returncode": 0, "data": {"ok": True, "errors": []}}
    raise AssertionError(args)


def healthy_probes(**overrides) -> Probes:
    probes = dict(
        db_query=healthy_db,
        http_status=lambda url, **kw: 200,
        services=lambda: {s: {"state": "running", "health": "healthy" if s != "mcp" else ""}
                          for s in ("graph-db", "chat-db", "mcp", "webui")},
        okg_json=healthy_okg,
        mcp=lambda url, token: {"tools": ["inspect", "search"], "is_error": False, "text": "ok"},
        environ={},
        installed_okg=okg_version,  # the instance records the okg it was made with
        okg_distributions=lambda: {"akmon": okg_version() or ""},  # one engine package
    )
    probes.update(overrides)
    return Probes(**probes)


def results(instance, probes=None):
    inst = Instance.load(instance.dir)
    return {r.id: r for r in run_checks(Context(inst, read_env(inst.env_path), probes or healthy_probes()))}


def test_a_healthy_instance_passes_every_check(instance):
    found = results(instance)
    bad = {k: (r.status, r.message) for k, r in found.items() if r.status != PASS}
    assert bad == {}


def test_doctor_report_is_json_for_agents(instance):
    report = doctor(Instance.load(instance.dir), healthy_probes())
    assert report["ok"] is True
    json.dumps(report)
    assert {c["id"] for c in report["checks"]} >= {"okg-lint", "okg-doctor", "published"}
    assert all(c["footgun"] for c in report["checks"]
               if c["id"] not in ("okg-lint", "okg-doctor", "source-freshness", "automation-names"))


# ---------------------------------------------------------------------------
# One failing fixture per check
# ---------------------------------------------------------------------------


def _edit_yaml(path: Path, change):
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data))


def break_instance_git(inst):
    subprocess.run(["rm", "-rf", str(inst.dir / ".git")], check=True)


def break_deployment_committed(inst):
    (inst.deployment_dir / "new-source.yaml").write_text("x: 1\n")


def break_env_secrets(inst):
    os.chmod(inst.env_path, 0o644)


def break_secrets_not_committed(inst):
    subprocess.run(["git", "-C", str(inst.dir), "add", "-f", ".env"], check=True)
    subprocess.run(["git", "-C", str(inst.dir), "commit", "-qm", "oops"], check=True)


def break_data_root(inst):
    _edit_yaml(inst.dir / "archi.yaml", lambda d: d.update(data_root=str(inst.dir / "data")))


def break_absolute_paths(inst):
    _edit_yaml(inst.deployment_dir / "source_registry.yaml",
               lambda d: d["sources"]["cmssw_releases"]["params"].update(cache_root="data/cmssw"))


def break_profiles_env(inst):
    return {"environ": {"OKG_PROFILES_DIR": "/elsewhere"}}


def break_okg_version(inst):
    return {"installed_okg": lambda: "0.1.9"}


def break_graph_db_shm(inst):
    _edit_yaml(inst.dir / "compose.yaml", lambda d: d["services"]["graph-db"].pop("shm_size"))


def break_ports_loopback(inst):
    _edit_yaml(inst.dir / "compose.yaml",
               lambda d: d["services"]["webui"].update(ports=["0.0.0.0:8080:8080"]))


def break_chat_db_distinct(inst):
    _edit_yaml(inst.dir / "compose.yaml", lambda d: d["services"]["webui"]["environment"].update(
        DATABASE_URL="postgresql://postgres:x@graph-db:5432/myteam"))


def break_mcp_port(inst):
    _edit_yaml(inst.deployment_dir / "deployment.yaml", lambda d: d["chat"]["mcp"].update(port=9999))


def break_selected_sources_fed(inst):
    _edit_yaml(inst.deployment_dir / "source_registry.yaml",
               lambda d: d["sources"].update(github_repo={"params": {"url": "", "repo": "x"}}))


def break_graph_db_preload(inst):
    def db(dsn, sql):
        if "shared_preload_libraries" in sql:
            return [("timescaledb",)]
        return healthy_db(dsn, sql)

    return {"db_query": db}


def break_role_passwords(inst):
    inst.role_secrets_path.unlink()


def break_published(inst):
    def okg(*args):
        if args[0] == "status":
            return {"returncode": 0, "data": {"publishing": {"latest_published_generation": None}}}
        return healthy_okg(*args)

    return {"okg_json": okg}


def break_services(inst):
    return {"services": lambda: {"graph-db": {"state": "running", "health": "healthy"},
                                 "webui": {"state": "exited", "health": ""}}}


def break_mcp_answers(inst):
    def mcp(url, token):
        raise ConnectionError("All connection attempts failed")

    return {"mcp": mcp}


def break_webui_up(inst):
    return {"http_status": lambda url, **kw: 0}


def break_okg_lint(inst):
    def okg(*args):
        if args[:2] == ("deployment", "lint"):
            return {"returncode": 1, "data": {"findings": [
                {"severity": "error", "message": "unknown module", "next_action": "fix modules"}]}}
        return healthy_okg(*args)

    return {"okg_json": okg}


def break_source_freshness(inst):
    def okg(*args):
        if args[:3] == ("doctor", "--check", "sources"):
            return watermarks(docsite__docsite=24 * 30)  # docsite allows 7d
        return healthy_okg(*args)

    return {"okg_json": okg}


def break_okg_doctor(inst):
    def okg(*args):
        if args[0] == "doctor":
            return {"returncode": 1, "data": {"ok": False, "errors": [
                {"message": "extension pg_cron missing", "fix_hint": "preload pg_cron"}]}}
        return healthy_okg(*args)

    return {"okg_json": okg}


def break_automation_names(inst):
    inst.automations = {"ticket-triaje": {"level": "L1"}}
    inst.save()


FIXTURES = {
    "instance-git": break_instance_git,
    "deployment-committed": break_deployment_committed,
    "env-secrets": break_env_secrets,
    "secrets-not-committed": break_secrets_not_committed,
    "data-root": break_data_root,
    "absolute-paths": break_absolute_paths,
    "profiles-env": break_profiles_env,
    "okg-version": break_okg_version,
    "graph-db-shm": break_graph_db_shm,
    "ports-loopback": break_ports_loopback,
    "chat-db-distinct": break_chat_db_distinct,
    "mcp-port": break_mcp_port,
    "selected-sources-fed": break_selected_sources_fed,
    "graph-db-preload": break_graph_db_preload,
    "role-passwords": break_role_passwords,
    "published": break_published,
    "services": break_services,
    "mcp-answers": break_mcp_answers,
    "webui-up": break_webui_up,
    "okg-lint": break_okg_lint,
    "okg-doctor": break_okg_doctor,
    "source-freshness": break_source_freshness,
    "automation-names": break_automation_names,
}

#: A check that warns rather than fails: the setting does not break archi,
#: only okg run by hand.
WARNS = {"profiles-env"}


def test_okg_version_warns_when_okg_and_akmon_are_both_installed(instance):
    # AKMON 0.6.0 renamed the distribution okg -> akmon. Upgrading without
    # `pip uninstall okg` leaves both installed over the same files.
    installed = okg_version()
    result = results(instance, healthy_probes(
        okg_distributions=lambda: {"akmon": installed, "okg": "0.4.0"}))["okg-version"]
    assert result.status == WARN, result
    assert "akmon" in result.message and "okg 0.4.0" in result.message
    assert "pip uninstall -y okg akmon" in result.remediation
    assert f'pip install "akmon=={installed}"' in result.remediation
    assert "\n" not in result.remediation


def test_an_instance_made_on_okg_never_gets_an_okg_wheel_fix_in_an_akmon_environment(instance):
    # A 3.0.1 instance (okg 0.4.0) after `pip install archi-kb==3.0.2` in the
    # same environment: akmon 0.6.0 sits over okg 0.4.0. Installing the okg
    # wheel again would overwrite akmon's files, so the fix must not say that.
    _edit_yaml(instance.dir / "archi.yaml", lambda d: d.update(okg_version="0.4.0"))
    result = results(instance, healthy_probes(
        installed_okg=lambda: "0.6.0",
        okg_distributions=lambda: {"akmon": "0.6.0", "okg": "0.4.0"}))["okg-version"]
    assert result.status == FAIL, result
    assert "run 0.4.0" in result.message and "two engine packages" in result.message
    assert "pip install" not in result.remediation
    assert "separate environment" in result.remediation and "archi init" in result.remediation
    assert "\n" not in result.remediation


def test_okg_version_fixes_name_the_package_of_the_instances_engine(instance):
    _edit_yaml(instance.dir / "archi.yaml", lambda d: d.update(okg_version="0.6.0"))
    missing = results(instance, healthy_probes(installed_okg=lambda: None))["okg-version"]
    assert missing.status == FAIL and missing.remediation == 'pip install "akmon==0.6.0"'
    newer = results(instance, healthy_probes(
        installed_okg=lambda: "0.6.1", okg_distributions=lambda: {"akmon": "0.6.1"}))["okg-version"]
    assert newer.status == FAIL
    assert newer.remediation == 'pip install "akmon==0.6.0"  (or re-init on 0.6.1)'


@pytest.mark.parametrize("check_id", sorted(FIXTURES))
def test_each_check_has_a_failing_fixture(instance, check_id):
    overrides = FIXTURES[check_id](instance) or {}
    result = results(instance, healthy_probes(**overrides))[check_id]
    assert result.status == (WARN if check_id in WARNS else FAIL), result
    assert result.remediation.strip()
    assert "\n" not in result.remediation


def test_every_check_has_a_failing_fixture():
    registered = {check_id for check_id, _, _ in CHECKS} | {"okg-lint", "okg-doctor", "source-freshness", "automation-names"}
    assert registered == set(FIXTURES)


def test_every_check_is_in_the_ledger():
    ledger = LEDGER.read_text(encoding="utf-8")
    entries = set(re.findall(r"^\| (FG-\d\d) \|", ledger, flags=re.M))
    for check_id, footgun, _ in CHECKS:
        assert footgun in entries, f"{check_id} cites {footgun}, which the ledger lacks"
        assert f"`{check_id}`" in ledger, f"the ledger does not name check {check_id}"


def test_the_ledger_leaves_no_trap_unresolved():
    rows = re.findall(r"^\| (FG-\d\d) \|(.*)$", LEDGER.read_text(encoding="utf-8"), flags=re.M)
    assert len(rows) >= 30
    for fid, rest in rows:
        status = rest.split("|")[1].strip().lower()
        assert status.startswith(("absorbed", "filed")), f"{fid}: status {status!r}"
