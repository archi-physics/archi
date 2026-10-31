"""`archi init` and `archi up` with an automation pack selected need no hand step: the DBOS schema is
bootstrapped when provision stops at its ingest stage (okg upstream ask O12), and `up` repeats the
bootstrap, which is idempotent."""
import subprocess

import pytest

from archi.cli import init as init_mod
from archi.cli import up as up_mod
from archi.stack.okg import OkgError

from .test_stack import _runner, instance  # noqa: F401  (the fixture and the fake compose runner)

DBOS_MISSING = 'relation "dbos.workflow_status" does not exist'


class Okg:
    """Stands in for `Okg.run`: provision fails with the DBOS error until the runtime is bootstrapped."""

    def __init__(self, bootstrapped=False, provision_error=DBOS_MISSING):
        self.calls, self.bootstrapped, self.provision_error = [], bootstrapped, provision_error

    def run(self, *args, ok_codes=(0,), **kw):
        self.calls.append(args)
        code, err = 0, ""
        if args[0] == "provision" and not self.bootstrapped:
            code, err = 1, self.provision_error
        if args[:2] == ("runtime", "bootstrap"):
            self.bootstrapped = True
        result = subprocess.CompletedProcess(args, code, "", err)
        if code not in ok_codes:
            raise OkgError(f"`okg {' '.join(args)}` failed ({code}):\n{err}", result)
        return result


def test_provision_bootstraps_the_runtime_then_ingests_in_process_and_publishes_when_dbos_is_missing():
    okg, said = Okg(), []
    result = init_mod.provision(okg, "dep", runtime=True, echo=said.append)
    assert result.returncode == 0
    assert okg.calls == [("provision", "--deployment", "dep"), ("runtime", "bootstrap", "--deployment", "dep"),
                         ("ingest", "--deployment", "dep", "--inline"),
                         ("run", "--once", "--deployment", "dep", "--apply", "--inline")]
    assert any("bootstrapping the runtime" in line for line in said)


def test_provision_without_the_runtime_is_one_call_and_any_failure_is_raised():
    okg = Okg()
    with pytest.raises(OkgError):
        init_mod.provision(okg, "dep", runtime=False, echo=lambda _: None)
    assert okg.calls == [("provision", "--deployment", "dep")]


def test_a_provision_failure_that_is_not_the_dbos_schema_is_raised_not_bootstrapped_around():
    okg = Okg(provision_error="publish blocked: invariant docsite_coverage failed")
    with pytest.raises(OkgError, match="docsite_coverage"):
        init_mod.provision(okg, "dep", runtime=True, echo=lambda _: None)
    assert [c[0] for c in okg.calls] == ["provision"]


def test_provision_that_works_is_followed_by_the_idempotent_bootstrap_when_the_runtime_is_on():
    okg = Okg(bootstrapped=True)
    init_mod.provision(okg, "dep", runtime=True, echo=lambda _: None)
    assert okg.calls == [("provision", "--deployment", "dep"), ("runtime", "bootstrap", "--deployment", "dep")]


def test_has_automations_reads_the_lock(tmp_path, monkeypatch):
    from archi.bundles import lock

    assert init_mod.has_automations(tmp_path) is False  # no lock: nothing rendered
    monkeypatch.setattr(lock, "read_lock", lambda d: {"composed": {"automations": {"triage": {}}}})
    assert init_mod.has_automations(tmp_path) is True
    monkeypatch.setattr(lock, "read_lock", lambda d: {"composed": {"automations": {}}})
    assert init_mod.has_automations(tmp_path) is False


def test_up_bootstraps_the_runtime_once_the_database_is_ready_when_an_automation_is_selected(monkeypatch, instance):
    from archi.stack import compose as stack_compose
    from archi.stack import webui

    monkeypatch.setattr(up_mod, "commit_all", lambda d, m: False)
    monkeypatch.setattr(up_mod, "has_automations", lambda d: True)
    tables = []
    monkeypatch.setattr(up_mod, "ensure_ops_tables", lambda inst, secrets: tables.append(inst.name))
    monkeypatch.setattr(stack_compose, "wait_for", lambda *a, **k: None)
    monkeypatch.setattr(stack_compose, "http_status", lambda url, **k: 0)
    monkeypatch.setattr(webui, "admin_token", lambda *a, **k: "tok")
    calls, lines = [], []

    def okg(cmd, cwd=None, env=None, **kw):
        calls.append(cmd[1:3])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    up_mod.up(instance, echo=lines.append, compose_runner=_runner({}), okg_runner=okg)
    assert calls[0] == ["runtime", "bootstrap"] and ["chat", "sync"] in calls
    assert tables == [instance.name]  # created again after the bootstrap, idempotently
    assert any("no runtime worker yet" in line for line in lines)


def test_up_without_an_automation_does_not_touch_the_runtime(monkeypatch, instance):
    from archi.stack import compose as stack_compose
    from archi.stack import webui

    monkeypatch.setattr(up_mod, "commit_all", lambda d, m: False)
    monkeypatch.setattr(up_mod, "has_automations", lambda d: False)
    monkeypatch.setattr(stack_compose, "wait_for", lambda *a, **k: None)
    monkeypatch.setattr(stack_compose, "http_status", lambda url, **k: 0)
    monkeypatch.setattr(webui, "admin_token", lambda *a, **k: "tok")
    calls = []
    up_mod.up(instance, echo=lambda _: None, compose_runner=_runner({}),
              okg_runner=lambda cmd, **kw: calls.append(cmd[1:3]) or subprocess.CompletedProcess(cmd, 0, "", ""))
    assert ["runtime", "bootstrap"] not in calls


def test_install_and_publish_creates_the_archi_ops_tables_after_the_bootstrap_when_an_automation_is_selected(
        monkeypatch, instance):
    from archi.stack.instance import read_env

    monkeypatch.setattr(init_mod, "commit_all", lambda d, m: False)
    monkeypatch.setattr(init_mod.stack_compose, "write_stack_files", lambda i: None)
    monkeypatch.setattr(init_mod, "role_password", lambda path, role: "pw")
    monkeypatch.setattr(init_mod, "write_env", lambda path, values: None)
    monkeypatch.setattr(init_mod, "has_automations", lambda d: True)
    order = []
    okg = Okg(bootstrapped=True)
    real_run = okg.run
    okg.run = lambda *a, **k: order.append(a[0] if a[0] != "runtime" else "runtime bootstrap") or real_run(*a, **k)
    monkeypatch.setattr(init_mod, "ensure_ops_tables", lambda inst, secrets: order.append("ops tables"))
    init_mod.install_and_publish(instance, read_env(instance.env_path), okg, "prof", {}, echo=lambda _: None)
    assert order == ["install", "provision", "runtime bootstrap", "ops tables", "db"]


def test_install_and_publish_without_an_automation_makes_no_ops_tables(monkeypatch, instance):
    from archi.stack.instance import read_env

    monkeypatch.setattr(init_mod, "commit_all", lambda d, m: False)
    monkeypatch.setattr(init_mod.stack_compose, "write_stack_files", lambda i: None)
    monkeypatch.setattr(init_mod, "role_password", lambda path, role: "pw")
    monkeypatch.setattr(init_mod, "write_env", lambda path, values: None)
    monkeypatch.setattr(init_mod, "has_automations", lambda d: False)
    monkeypatch.setattr(init_mod, "ensure_ops_tables", lambda *a: pytest.fail("no automation, no tables"))
    init_mod.install_and_publish(instance, read_env(instance.env_path), Okg(bootstrapped=True), "prof", {},
                                 echo=lambda _: None)


def test_ensure_ops_tables_runs_the_ddl_against_the_instance_database(monkeypatch, instance):
    from archi.ops import okg_io
    from archi.stack.instance import graph_dsn, read_env

    seen = {}
    monkeypatch.setattr(okg_io.OpsIO, "ensure_tables", lambda self: seen.update(dsn=self.dsn, name=self.deployment))
    secrets = read_env(instance.env_path)
    init_mod.ensure_ops_tables(instance, secrets)
    assert seen == {"dsn": graph_dsn(instance, secrets), "name": instance.name}


def _manifest(path, runtime, postgres=None):
    import yaml

    data = {"name": "dep", "postgres": postgres or {"dsn": "${OKG_DSN}"}}
    if runtime is not None:
        data["runtime"] = {"enabled": runtime}
    (path / "deployment.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def _read(path):
    import yaml

    return yaml.safe_load((path / "deployment.yaml").read_text())


def test_the_instance_database_is_declared_when_the_runtime_is_enabled(tmp_path):
    _manifest(tmp_path, runtime=True)
    assert init_mod.declare_runtime_database(tmp_path, "my_team") is True
    assert _read(tmp_path)["postgres"] == {"dsn": "${OKG_DSN}", "database": "my_team"}
    # Idempotent: a second call changes nothing.
    assert init_mod.declare_runtime_database(tmp_path, "my_team") is False


def test_no_runtime_and_no_install_change_nothing(tmp_path):
    _manifest(tmp_path, runtime=None)
    assert init_mod.declare_runtime_database(tmp_path, "my_team") is False
    assert "database" not in _read(tmp_path)["postgres"]
    _manifest(tmp_path, runtime=False)
    assert init_mod.declare_runtime_database(tmp_path, "my_team") is False
    assert init_mod.declare_runtime_database(tmp_path / "missing", "my_team") is False


def test_a_different_declared_database_is_refused_not_overwritten(tmp_path):
    _manifest(tmp_path, runtime=True, postgres={"dsn": "${OKG_DSN}", "database": "other"})
    with pytest.raises(init_mod.InitError, match="other"):
        init_mod.declare_runtime_database(tmp_path, "my_team")
    assert _read(tmp_path)["postgres"]["database"] == "other"


def test_install_and_publish_declares_the_database_right_after_the_install(monkeypatch, instance):
    from archi.stack.instance import DEPLOYMENTS_DIR, read_env

    deployment = instance.dir / DEPLOYMENTS_DIR / instance.name
    order = []

    def commit(directory, message):
        # okg publishes only committed configuration: the database must be declared by now.
        order.append(("commit", _read(deployment)["postgres"].get("database")))
        return False

    monkeypatch.setattr(init_mod, "commit_all", commit)
    monkeypatch.setattr(init_mod.stack_compose, "write_stack_files", lambda i: None)
    monkeypatch.setattr(init_mod, "role_password", lambda path, role: "pw")
    monkeypatch.setattr(init_mod, "write_env", lambda path, values: None)
    monkeypatch.setattr(init_mod, "has_automations", lambda d: True)
    monkeypatch.setattr(init_mod, "ensure_ops_tables", lambda inst, secrets: None)
    okg = Okg(bootstrapped=True)
    real_run = okg.run

    def run(*args, **kw):
        if args[0] == "install":  # what okg install leaves: a runtime-enabled manifest with no database
            deployment.mkdir(parents=True, exist_ok=True)
            _manifest(deployment, runtime=True)
        if args[0] == "provision":
            order.append(("provision", _read(deployment)["postgres"]["database"]))
        return real_run(*args, **kw)

    okg.run = run
    init_mod.install_and_publish(instance, read_env(instance.env_path), okg, "prof", {}, echo=lambda _: None)
    assert order == [("commit", instance.database), ("provision", instance.database)]


def test_an_empty_postgres_block_or_blank_database_is_filled_in(tmp_path):
    import yaml

    (tmp_path / "deployment.yaml").write_text(yaml.safe_dump({"postgres": None, "runtime": {"enabled": True}}))
    assert init_mod.declare_runtime_database(tmp_path, "my_team") is True
    assert _read(tmp_path)["postgres"] == {"database": "my_team"}
    _manifest(tmp_path, runtime=True, postgres={"dsn": "${OKG_DSN}", "database": "  "})
    assert init_mod.declare_runtime_database(tmp_path, "my_team") is True
    assert _read(tmp_path)["postgres"]["database"] == "my_team"


def test_up_declares_the_runtime_database_of_an_instance_made_before_the_fix(monkeypatch, instance):
    from archi.stack import compose as stack_compose
    from archi.stack import webui

    instance.deployment_dir.mkdir(parents=True, exist_ok=True)
    _manifest(instance.deployment_dir, runtime=True)  # what an older `archi init` left: no database
    committed = []
    monkeypatch.setattr(up_mod, "commit_all",
                        lambda d, m: committed.append(_read(instance.deployment_dir)["postgres"].get("database")))
    monkeypatch.setattr(up_mod, "has_automations", lambda d: True)
    monkeypatch.setattr(up_mod, "ensure_ops_tables", lambda inst, secrets: None)
    monkeypatch.setattr(stack_compose, "wait_for", lambda *a, **k: None)
    monkeypatch.setattr(stack_compose, "http_status", lambda url, **k: 0)
    monkeypatch.setattr(webui, "admin_token", lambda *a, **k: "tok")
    up_mod.up(instance, echo=lambda _: None, compose_runner=_runner({}),
              okg_runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    assert committed[0] == instance.database
