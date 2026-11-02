"""req.r3-golden-path.up: the compose file, the runtime, and `archi up` / `down`.

The running stack is the executed test in the PACT (golden_path_run.sh).
These cover what archi renders and decides without a container runtime.
"""
import json
import subprocess

import pytest
import yaml

from archi.cli import up as up_mod
from archi.stack import compose as stack_compose
from archi.stack import webui
from archi.stack.compose import ComposeError, Compose, detect_runtime, render_compose
from archi.stack.instance import Instance, Ports, generate_secrets, write_env


@pytest.fixture
def instance(tmp_path) -> Instance:
    inst = Instance(
        dir=tmp_path / "myteam", name="my-team", bundles=["cern-team"],
        data_root=tmp_path / "data", runtime="docker",
        ports=Ports(graph_db=15432, mcp=18765, webui=18080), profile="cern-team",
        okg_version="0.2.0", admin_email="admin@my-team.archi.invalid",
    )
    dep = inst.deployment_dir
    dep.mkdir(parents=True)
    (dep / "deployment.yaml").write_text(yaml.safe_dump({
        "chat": {"ui": {"site_name": "Team chat"}, "mcp": {"port": 8765}},
    }))
    secrets = generate_secrets()
    secrets["ARCHI_OKG_MCP_PASSWORD"] = "role-pw"
    write_env(inst.env_path, secrets)
    inst.save()
    return inst


def test_compose_holds_no_secret_value(instance):
    stack_compose.write_stack_files(instance)
    text = (instance.dir / "compose.yaml").read_text()
    for value in (instance.dir / ".env").read_text().splitlines()[1:]:
        assert value.split("=", 1)[1] not in text
    assert "${ARCHI_GRAPH_DB_PASSWORD}" in text and "${OKG_CHAT_WEBUI_SECRET_KEY}" in text


def test_compose_graph_db_has_preload_cron_and_shm(instance):
    db = render_compose(instance)["services"]["graph-db"]
    command = " ".join(db["command"])
    assert "shared_preload_libraries=timescaledb,pg_cron,pg_textsearch" in command
    assert "cron.database_name=my_team" in command
    assert db["shm_size"] == "2g"
    assert db["image"] == "ghcr.io/mitdbg/akmon-postgres:v0.2.0"
    assert db["environment"]["POSTGRES_DB"] == "my_team"


def test_graph_volume_holds_the_images_own_pgdata(instance):
    """The AKMON image sets PGDATA=/home/postgres/pgdata/data; the volume must
    cover it or the graph lives in the container layer and dies with it."""
    # ghcr.io/mitdbg/akmon-postgres:v0.2.0 sets PGDATA=/home/postgres/pgdata/data
    # (GRAPH_DB_PGDATA); the volume target is its parent.
    assert stack_compose.GRAPH_DB_IMAGE.endswith(":v{okg_version}")
    assert stack_compose.GRAPH_DB_PGDATA == "/home/postgres/pgdata/data"
    db = render_compose(instance)["services"]["graph-db"]
    assert db["volumes"] == ["graph-data:/home/postgres/pgdata"]
    assert stack_compose.GRAPH_DB_PGDATA.startswith(db["volumes"][0].split(":")[1] + "/")


def test_compose_services_reach_each_other_by_name(instance):
    services = render_compose(instance)["services"]
    mcp, webui_service = services["mcp"], services["webui"]
    assert "@graph-db:5432/my_team" in mcp["environment"]["OKG_DSN"]
    assert mcp["environment"]["OKG_MCP_RO_DSN"].startswith("postgresql://okg_mcp:${ARCHI_OKG_MCP_PASSWORD}@graph-db:5432/")
    assert "@chat-db:5432/okg_chat_app" in webui_service["environment"]["DATABASE_URL"]
    assert webui_service["environment"]["WEBUI_NAME"] == "Team chat"
    # Every published port is loopback-only; chat-db is not published at all.
    for name, service in services.items():
        for spec in service.get("ports") or []:
            assert spec.startswith("127.0.0.1:"), (name, spec)
    assert "ports" not in services["chat-db"]
    # MCP serves on chat.mcp.port, behind the bearer token.
    assert mcp["ports"] == ["127.0.0.1:18765:8765"]
    assert "--auth-token-env" in mcp["command"] and "8765" in mcp["command"]
    # The instance is mounted at its own path, read-only.
    assert mcp["volumes"] == [f"{instance.dir}:{instance.dir}:ro"]


def test_mcp_dockerfile_pins_the_host_okg_and_installs_git(instance):
    text = stack_compose.render_dockerfile("0.2.0")
    assert "okg-0.2.0-py3-none-any.whl" in text
    assert "install -y --no-install-recommends git" in text
    assert "safe.directory" in text


def _runner(results):
    calls = []
    kwargs = []

    def run(cmd, **kw):
        calls.append(cmd)
        kwargs.append(kw)
        return subprocess.CompletedProcess(cmd, results.get(tuple(cmd[:2]), 0), "", "")

    run.calls = calls
    run.kwargs = kwargs
    return run


def test_detect_runtime_prefers_a_working_docker(monkeypatch):
    monkeypatch.setattr(stack_compose.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert detect_runtime(runner=_runner({})) == "docker"


def test_detect_runtime_falls_back_to_podman(monkeypatch):
    monkeypatch.setattr(stack_compose.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert detect_runtime(runner=_runner({("docker", "info"): 1})) == "podman"


def test_detect_runtime_names_every_reason(monkeypatch):
    monkeypatch.setattr(stack_compose.shutil, "which",
                        lambda name: "/usr/bin/podman" if name == "podman" else None)
    with pytest.raises(ComposeError, match="docker: not installed; podman: installed, but"):
        detect_runtime(runner=_runner({("podman", "info"): 125}))


def test_compose_commands_name_the_project_and_env_file(instance):
    run = _runner({})
    Compose(instance, runner=run).up("graph-db")
    cmd = run.calls[-1]
    assert cmd[:2] == ["docker", "compose"]
    assert cmd[cmd.index("-p") + 1] == "archi-my-team"
    assert cmd[cmd.index("--env-file") + 1] == str(instance.dir / ".env")
    assert cmd[-3:] == ["up", "-d", "graph-db"]


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_compose_argv_is_the_same_for_both_runtimes_and_has_no_project_directory(instance, runtime):
    """podman-compose 1.4 rejects --project-directory; both runtimes default the
    project directory to the first -f file's, so only -f, --env-file and -p are
    passed, and the process runs in the instance directory."""
    instance.runtime = runtime
    run = _runner({})
    Compose(instance, runner=run).up("graph-db")
    assert run.calls[-1] == [
        runtime, "compose",
        "-f", str(instance.dir / "compose.yaml"),
        "--env-file", str(instance.dir / ".env"),
        "-p", instance.project,
        "up", "-d", "graph-db",
    ]
    assert "--project-directory" not in run.calls[-1]
    assert run.kwargs[-1]["cwd"] == instance.dir


def _ps_runner(compose_out=None, runtime_out=None):
    """compose ps answers compose_out (None: unsupported, exit 2); `<runtime> ps` runtime_out."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "compose":
            return subprocess.CompletedProcess(cmd, 2 if compose_out is None else 0, compose_out or "", "")
        return subprocess.CompletedProcess(cmd, 0, runtime_out or "", "")

    run.calls = calls
    return run


def test_services_reads_compose_ps_json(instance):
    out = json.dumps([{"Service": "graph-db", "State": "running", "Health": "healthy"}])
    assert Compose(instance, runner=_ps_runner(out)).services() == {
        "graph-db": {"state": "running", "health": "healthy"}
    }


def test_services_falls_back_to_the_runtime_when_compose_ps_has_no_json(instance):
    """podman-compose 1.4 rejects `ps --format json`; podman lists labels as a dict."""
    out = json.dumps([
        {"State": "running", "Status": "Up 3 minutes (healthy)",
         "Labels": {"com.docker.compose.service": "graph-db"}},
        {"State": "running", "Status": "Up 3 minutes", "Labels": {"com.docker.compose.service": "mcp"}},
    ])
    run = _ps_runner(None, out)
    assert Compose(instance, runner=run).services() == {
        "graph-db": {"state": "running", "health": "healthy"},
        "mcp": {"state": "running", "health": ""},
    }
    assert run.calls[-1][-2:] == ["--filter", f"label=com.docker.compose.project={instance.project}"]


def test_services_fallback_reads_docker_label_strings(instance):
    row = {"State": "running", "Status": "Up 1 second (unhealthy)",
           "Labels": "a=b,com.docker.compose.service=webui"}
    assert Compose(instance, runner=_ps_runner(None, json.dumps(row))).services() == {
        "webui": {"state": "running", "health": "unhealthy"}
    }


def test_down_keeps_volumes_and_purge_deletes_them(instance):
    run = _runner({})
    up_mod.down(instance, compose_runner=run, echo=lambda _: None)
    assert run.calls[-1][-1] == "down"
    up_mod.down(instance, purge=True, compose_runner=run, echo=lambda _: None)
    assert run.calls[-1][-2:] == ["down", "--volumes"]


def test_down_purge_deletes_volumes(instance):
    """The CLI refuses --purge without --yes."""
    from archi.bundles.cli import main

    assert main(["down", "--instance", str(instance.dir), "--purge"]) == 1


# ---------------------------------------------------------------------------
# Open WebUI calls
# ---------------------------------------------------------------------------


def test_admin_token_signs_in_first(monkeypatch):
    calls = []

    def post(url, body, token=None, timeout=30):
        calls.append(url)
        return 200, {"token": "t-signin"}

    monkeypatch.setattr(webui, "_post", post)
    assert webui.admin_token("http://x", "a@b", "pw") == "t-signin"
    assert calls == ["http://x/api/v1/auths/signin"]


def test_admin_token_signs_up_on_a_fresh_site(monkeypatch):
    def post(url, body, token=None, timeout=30):
        if url.endswith("signin"):
            return 400, {"detail": "The email or password provided is incorrect."}
        return 200, {"token": "t-signup", "role": "admin"}

    monkeypatch.setattr(webui, "_post", post)
    assert webui.admin_token("http://x", "a@b", "pw") == "t-signup"


def test_admin_token_refuses_a_non_admin_signup(monkeypatch):
    def post(url, body, token=None, timeout=30):
        if url.endswith("signin"):
            return 400, {}
        return 200, {"token": "t", "role": "pending"}

    monkeypatch.setattr(webui, "_post", post)
    with pytest.raises(webui.WebUIError, match="not the admin"):
        webui.admin_token("http://x", "a@b", "pw")


def test_configure_ollama_sends_the_container_url(monkeypatch):
    sent = {}

    def post(url, body, token=None, timeout=30):
        sent.update(url=url, body=body, token=token)
        return 200, {}

    monkeypatch.setattr(webui, "_post", post)
    webui.configure_ollama("http://x", "tok", "http://host.docker.internal:11434")
    assert sent["url"] == "http://x/ollama/config/update"
    assert sent["body"]["OLLAMA_BASE_URLS"] == ["http://host.docker.internal:11434"]
    assert sent["token"] == "tok"


def test_configure_ollama_failure_is_reported(monkeypatch):
    monkeypatch.setattr(webui, "_post", lambda *a, **k: (422, {"detail": "x"}))
    with pytest.raises(webui.WebUIError, match="422"):
        webui.configure_ollama("http://x", "tok", "http://h:1")


# ---------------------------------------------------------------------------
# archi up
# ---------------------------------------------------------------------------


def _fake_up(monkeypatch, instance, environ):
    monkeypatch.setattr(up_mod, "commit_all", lambda d, m: False)
    monkeypatch.setattr(stack_compose, "wait_for", lambda *a, **k: None)
    monkeypatch.setattr(stack_compose, "http_status", lambda url, **k: 0)
    monkeypatch.setattr(webui, "admin_token", lambda *a, **k: "tok")
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    okg_calls = []

    def okg(cmd, cwd=None, env=None, **kw):
        okg_calls.append((cmd[1:], env))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    lines = []
    up_mod.up(instance, echo=lines.append, compose_runner=_runner({}), okg_runner=okg)
    return lines, okg_calls


def test_up_runs_chat_sync_against_the_compose_site(monkeypatch, instance):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    lines, okg_calls = _fake_up(monkeypatch, instance, {})
    (args, env), = okg_calls
    assert args[:2] == ["chat", "sync"]
    assert args[args.index("--mcp-probe-url") + 1] == "http://127.0.0.1:18765/mcp"
    assert args[args.index("--mcp-instance-url") + 1] == "http://mcp:8765/mcp"
    assert env["OKG_CHAT_INSTANCE_URL"] == "http://127.0.0.1:18080"
    assert env["OKG_CHAT_ADMIN_TOKEN"] == "tok"
    assert not any("ssh -L" in line for line in lines)


def test_up_prints_the_tunnel_over_ssh(monkeypatch, instance):
    lines, _ = _fake_up(monkeypatch, instance, {"SSH_CONNECTION": "10.0.0.1 5000 10.0.0.2 22"})
    assert any("ssh -L 18080:127.0.0.1:18080" in line for line in lines)


def test_up_refuses_an_unfinished_init(instance):
    secrets = generate_secrets()
    write_env(instance.env_path, secrets)
    from archi.stack.instance import InstanceError

    with pytest.raises(InstanceError, match="did not finish"):
        up_mod.up(instance, echo=lambda _: None, compose_runner=_runner({}))
