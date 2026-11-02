"""req.r3-mcp-clients.external and the scripted call of each recipe (recipe_check, no LLM)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from archi.bundles.cli import main as archi_main
from archi.stack.instance import Instance, InstanceError, mcp_server_problems

from connect import recipe_check
from connect.conftest import TOKEN, make_instance
from connect.fixtures.echo_server import serve


# --- mcp_servers validation ----------------------------------------------------------------


GOOD = [{"name": "notes", "url": "https://notes.example.org/mcp", "headers": {"Authorization": "NOTES_AUTH"}}]


def test_valid_blocks_pass():
    assert mcp_server_problems(None) == []
    assert mcp_server_problems(GOOD) == []
    assert mcp_server_problems([{"name": "a", "url": "http://127.0.0.1:1/mcp"}]) == []


@pytest.mark.parametrize("raw, fragment", [
    ({"name": "x"}, "must be a list"),
    (["x"], "must be a mapping"),
    ([{"name": "Notes", "url": "http://h/mcp"}], "name must be"),
    ([{"name": "n", "url": "ftp://h/x"}], "http(s) URL"),
    ([{"name": "n", "url": "http://user:pw@h/mcp"}], "credentials"),
    ([{"name": "n", "url": "http://h/mcp?token=abc"}], "credentials or a query"),
    ([{"name": "n", "url": "http://h/mcp", "headers": {"Authorization": "Bearer abc def"}}], "environment variable"),
    ([{"name": "n", "url": "http://h/mcp", "headers": ["A"]}], "headers must map"),
    ([{"name": "n", "url": "http://h/mcp", "headers": {"bad header": "X"}}], "not a header name"),
    ([{"name": "n", "url": "http://h/mcp", "token": "abc"}], "unknown key"),
    ([{"name": "n", "url": "http://h/mcp"}, {"name": "n", "url": "http://h/mcp"}], "duplicate"),
])
def test_invalid_blocks_are_refused(raw, fragment):
    assert any(fragment in p for p in mcp_server_problems(raw)), mcp_server_problems(raw)


def test_the_instance_servers_own_name_is_reserved():
    assert any("own MCP server" in p for p in mcp_server_problems(
        [{"name": "archi-demo", "url": "http://h/mcp"}], reserved="archi-demo"))


def test_instance_load_round_trips_and_refuses_bad_blocks(tmp_path):
    instance = make_instance(tmp_path, external=[])
    assert "mcp_servers" not in yaml.safe_load((instance.dir / "archi.yaml").read_text())
    data = yaml.safe_load((instance.dir / "archi.yaml").read_text())
    data["mcp_servers"] = GOOD
    (instance.dir / "archi.yaml").write_text(yaml.safe_dump(data))
    loaded = Instance.load(instance.dir)
    assert [(s.name, s.headers) for s in loaded.mcp_servers] == [("notes", {"Authorization": "NOTES_AUTH"})]
    loaded.save()
    assert Instance.load(instance.dir).mcp_servers == loaded.mcp_servers

    data["mcp_servers"] = [{"name": "n", "url": "http://h/mcp", "headers": {"Authorization": "Bearer abc"}}]
    (instance.dir / "archi.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(InstanceError, match="environment variable"):
        Instance.load(instance.dir)


def test_a_secret_value_never_reaches_a_written_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NOTES_API_KEY", "super-secret-value")
    from archi.stack.instance import McpServer

    instance = make_instance(tmp_path, external=[
        McpServer("notes", "http://127.0.0.1:1/mcp", {"X-Api-Key": "NOTES_API_KEY"})])
    target = tmp_path / ".mcp.json"
    archi_main(["connect", "claude-code", "--instance", str(instance.dir), "--write", str(target)])
    assert "super-secret-value" not in target.read_text()
    assert "${NOTES_API_KEY}" in target.read_text()


# --- the external server fixture ---------------------------------------------------------------


def test_the_external_fixture_is_reachable_and_echoes():
    with serve("external") as url:
        out = _run(recipe_check.call_server(url, {}, [("echo", {"text": "hi"})]))
    assert out["tools"] == ["echo"]
    assert out["calls"] == [{"tool": "echo", "ok": True, "text": "echo: hi"}]


def test_the_fixture_refuses_a_missing_secret():
    with serve("external", "X-Api-Key", "k1") as url:
        with pytest.raises(BaseException):
            _run(recipe_check.call_server(url, {}, [("echo", {"text": "hi"})]))
        out = _run(recipe_check.call_server(url, {"X-Api-Key": "k1"}, [("echo", {"text": "hi"})]))
    assert out["calls"][0]["ok"]


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# --- recipe_check against a written config, per client -----------------------------------------


@pytest.mark.parametrize("client", ["claude-code", "codex", "openclaw", "mcp"])
def test_recipe_check_calls_search_and_echo_from_a_written_config(client, tmp_path, monkeypatch, capsys):
    from archi.stack.instance import McpServer

    monkeypatch.setenv("NOTES_API_KEY", "ext-key")
    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", TOKEN)  # Codex reads the token from the environment
    with serve("instance", "Authorization", f"Bearer {TOKEN}") as inst_url, \
            serve("external", "X-Api-Key", "ext-key") as ext_url:
        port = int(inst_url.split(":")[2].split("/")[0])
        instance = make_instance(tmp_path, mcp_port=port,
                                 external=[McpServer("notes", ext_url, {"X-Api-Key": "NOTES_API_KEY"})])
        target = tmp_path / ("config.toml" if client == "codex" else "config.json")
        assert archi_main(["connect", client, "--instance", str(instance.dir), "--write", str(target)]) == 0
        capsys.readouterr()
        code = recipe_check.main(["--client", client, "--config", str(target)])
        report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["ok"], report
    inst = report["servers"]["archi-demo"]
    assert inst["role"] == "instance" and inst["calls"][0]["tool"] == "search"
    assert "search ok" in inst["calls"][0]["text"]
    assert inst["citations"] == [{"rank": 1, "node_id": "doc:fixture:readme", "subtype": "document"}]
    ext = report["servers"]["notes"]
    assert ext["role"] == "external" and ext["calls"][0]["text"] == "echo: archi-connect"
    assert TOKEN not in json.dumps(report)


def test_recipe_check_uses_the_headers_as_written_and_fails_on_a_wrong_token(tmp_path, capsys):
    with serve("instance", "Authorization", f"Bearer {TOKEN}") as url:
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {"archi-demo": {
            "type": "http", "url": url, "headers": {"Authorization": "Bearer wrong"}}}}))
        code = recipe_check.main(["--client", "claude-code", "--config", str(config)])
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and not report["ok"] and "error" in report["servers"]["archi-demo"]


def test_recipe_check_reads_references_from_the_env_file_and_fails_when_unset(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OKG_CHAT_MCP_TOKEN", raising=False)
    with serve("instance", "Authorization", f"Bearer {TOKEN}") as url:
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {"archi-demo": {
            "type": "http", "url": url, "headers": {"Authorization": "Bearer ${OKG_CHAT_MCP_TOKEN}"}}}}))
        assert recipe_check.main(["--client", "claude-code", "--config", str(config)]) == 1
        assert "OKG_CHAT_MCP_TOKEN} is not set" in json.loads(capsys.readouterr().out)["error"]
        env = tmp_path / ".env"
        env.write_text(f"OKG_CHAT_MCP_TOKEN={TOKEN}\n")
        assert recipe_check.main(["--client", "claude-code", "--config", str(config), "--env-file", str(env)]) == 0


def test_the_fixture_runs_as_a_script():
    script = Path(__file__).parent / "fixtures" / "echo_server.py"
    out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--port" in out.stdout
