"""Open WebUI's scripted search call (webui_check): the in-container client, run here as a local process."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from connect import webui_check
from connect.fixtures.echo_server import serve

KEY = "registered-key-123"


def test_the_in_container_client_calls_search_over_streamable_http_and_finds_cited_rows():
    with serve("instance", "Authorization", f"Bearer {KEY}") as url:
        result = webui_check.search_in_container([], url, KEY, "readme", python=sys.executable)
    assert result["ok"] and result["tools"] == ["search"]
    assert result["citations"] == [{"rank": 1, "node_id": "doc:fixture:readme", "subtype": "document"}]


def test_a_rejected_key_fails_and_the_key_is_not_in_the_message():
    with serve("instance", "Authorization", f"Bearer {KEY}") as url:
        with pytest.raises(webui_check.CheckError) as err:
            webui_check.search_in_container([], url, "wrong-key", "readme", python=sys.executable)
    assert "401" in str(err.value) and "wrong-key" not in str(err.value)


def test_an_answer_without_cited_rows_is_not_ok():
    fake = SimpleNamespace(returncode=0, stderr="", stdout='{"tools": ["search"], "is_error": false, "text": "search ok"}\n')
    result = webui_check.search_in_container(["x"], "http://u/mcp", KEY, "q", runner=lambda *a, **k: fake)
    assert result["ok"] is False and result["citations"] == []


def test_the_command_runs_python_in_the_named_container_and_passes_the_key_on_stdin():
    seen = {}

    def runner(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        return SimpleNamespace(returncode=0, stderr="", stdout='{"tools": [], "is_error": false, "text": ""}')

    webui_check.search_in_container(["podman", "exec", "-i", "archi-gp_webui_1"], "http://mcp:8765/mcp", KEY, "q",
                                    runner=runner)
    assert seen["cmd"][:4] == ["podman", "exec", "-i", "archi-gp_webui_1"] and seen["cmd"][4] == "python"
    assert KEY not in " ".join(seen["cmd"]) and KEY in seen["input"] and "http://mcp:8765/mcp" in seen["input"]


def test_container_selection_needs_exactly_one_webui_of_this_instance():
    instance = SimpleNamespace(project="archi-gp")

    def runner_for(names):
        return lambda *a, **k: SimpleNamespace(stdout="\n".join(names))

    one = ["archi-gp_mcp_1", "archi-gp_webui_1", "archi-other_webui_1"]
    assert webui_check.webui_container(instance, "podman", runner_for(one)) == "archi-gp_webui_1"
    with pytest.raises(webui_check.CheckError):
        webui_check.webui_container(instance, "podman", runner_for(["archi-gp_mcp_1"]))


def test_python_subprocess_is_available():  # the in-container script needs only the standard library
    assert subprocess.run([sys.executable, "-c", "import urllib.request, json"]).returncode == 0
