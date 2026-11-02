"""`archi eval` end to end through the real parser, with the model and MCP faked."""
import json
from contextlib import asynccontextmanager

import pytest

from archi.bundles.cli import main
from archi.evaluation import agent as agent_mod
from archi.evaluation.agent import ChatReply, ToolSpec

SEARCH = ToolSpec("search", "Search", {"type": "object"})


class NeverCites:
    name = "fake"

    def __init__(self, base_url, model, **kw):
        self.name = model
        self.kw = kw

    def complete(self, messages, tools):
        return ChatReply("I do not know.", (), 5, 1)


class Tools:
    specs = [SEARCH]

    def __init__(self, url, token):
        self.url, self.token = url, token

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def list_tools(self):
        return list(self.specs)

    async def call(self, name, arguments):
        return "", False


class NoTools(Tools):
    specs = []


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(agent_mod, "OpenAIChat", NeverCites)
    monkeypatch.setattr(agent_mod, "McpToolBox", Tools)
    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", "t0ken")


def eval_args(tmp_path, *extra):
    return ["eval", "--bundle", "cern-team", "--mcp-url", "http://127.0.0.1:9/mcp", "--instance",
            str(tmp_path / "none"), "--model", "m:1b", "--out", str(tmp_path / "out"), *extra]


def test_below_the_threshold_exits_1_and_writes_both_reports(fakes, tmp_path, capsys):
    assert main(eval_args(tmp_path)) == 1
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["bundle"] == "cern-team" and data["status"] == "fail" and data["passed"] == 0
    assert data["total"] >= 10 and data["model"] == "m:1b" and data["tools"] == ["search"]
    assert (tmp_path / "out" / "report.md").is_file()
    assert "fail: 0/" in capsys.readouterr().out


def test_zero_mcp_tools_exits_2_as_execution_failed(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(agent_mod, "McpToolBox", NoTools)
    assert main(eval_args(tmp_path)) == 2
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["status"] == "execution_failed"
    assert {q["status"] for q in data["questions"]} == {"execution_failed"}


def test_a_missing_token_is_an_error_before_anything_runs(fakes, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OKG_CHAT_MCP_TOKEN")
    assert main(eval_args(tmp_path)) == 2
    assert "no MCP bearer token" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_a_paid_model_endpoint_is_refused(fakes, tmp_path, capsys):
    assert main(eval_args(tmp_path, "--model-url", "https://api.openai.com/v1")) == 2
    assert "no paid calls" in capsys.readouterr().err


def test_a_bundle_without_a_smoke_set_is_named(fakes, tmp_path, capsys):
    args = eval_args(tmp_path)
    args[2] = "cms-comp-ops-nope"
    assert main(args) == 2
    assert "no bundle named" in capsys.readouterr().err


def test_a_bundle_with_no_default_model_needs_one(fakes, tmp_path, capsys):
    args = eval_args(tmp_path)
    args[2] = "cms-physics"
    del args[args.index("--model"): args.index("--model") + 2]
    assert main(args) == 2
    assert "pass --model" in capsys.readouterr().err


def test_the_default_model_is_the_bundles_chat_model(fakes, tmp_path):
    args = eval_args(tmp_path)
    del args[args.index("--model"): args.index("--model") + 2]
    main(args)
    assert json.loads((tmp_path / "out" / "report.json").read_text())["model"] == "llama3.1:8b"


def test_with_mcp_url_no_instance_is_read_and_the_environment_token_is_sent(fakes, monkeypatch, tmp_path):
    from archi.stack.instance import Instance

    def no_instance(*a, **kw):
        raise AssertionError("an instance was read although --mcp-url names another endpoint")

    seen = []

    class Recording(Tools):
        def __init__(self, url, token):
            super().__init__(url, token)
            seen.append((url, token))

    monkeypatch.setattr(Instance, "load", no_instance)
    monkeypatch.setattr(agent_mod, "McpToolBox", Recording)
    assert main(eval_args(tmp_path)) == 1
    assert seen == [("http://127.0.0.1:9/mcp", "t0ken")]


def test_an_unreadable_chat_prompt_exits_2(fakes, tmp_path, capsys):
    assert main(eval_args(tmp_path, "--chat-prompt", str(tmp_path / "missing.md"))) == 2
    assert "missing.md" in capsys.readouterr().err


def test_eval_is_listed_in_the_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "eval" in capsys.readouterr().out
