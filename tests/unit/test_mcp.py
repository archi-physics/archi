import asyncio
from types import SimpleNamespace

from src.archi.pipelines.agents import base_react
from src.archi.pipelines.agents.tools import mcp


def _runtime_config():
    return {
        "mcp_servers": {
            "okg_cms": {
                "transport": "stdio",
                "command": "/opt/okg/bin/python",
                "args": ["/opt/okg/server.py"],
                "env": {"OKG_DEPLOYMENTS_DIR": "/opt/okg/deployments"},
                "skill": "cms_okg",
            }
        }
    }


def test_initialize_mcp_client_uses_explicit_runtime_config(monkeypatch):
    observed = {}
    tool = SimpleNamespace(
        name="search",
        description="Search the OKG",
        handle_tool_error=False,
    )

    class FakeClient:
        def __init__(self, connections):
            observed["connections"] = connections

        async def get_tools(self, server_name):
            observed["server_name"] = server_name
            return [tool]

    config = _runtime_config()
    monkeypatch.setattr(mcp, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(
        mcp,
        "get_full_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("explicit runtime config must not read global config")
        ),
    )
    monkeypatch.setattr(
        mcp,
        "load_skill",
        lambda name, loaded_config: (
            observed.update(skill=(name, loaded_config)) or "CMS instructions"
        ),
    )

    client, tools, skills_text = asyncio.run(mcp.initialize_mcp_client(config))

    assert isinstance(client, FakeClient)
    assert tools == [tool]
    assert tool.handle_tool_error is True
    assert observed["server_name"] == "okg_cms"
    assert observed["connections"]["okg_cms"]["command"] == "/opt/okg/bin/python"
    assert (
        observed["connections"]["okg_cms"]["env"]["OKG_DEPLOYMENTS_DIR"]
        == "/opt/okg/deployments"
    )
    assert observed["skill"] == ("cms_okg", config)
    assert "CMS instructions" in skills_text


def test_initialize_mcp_client_retains_global_config_fallback(monkeypatch):
    observed = {}
    config = _runtime_config()

    class FakeClient:
        def __init__(self, connections):
            observed["connections"] = connections

        async def get_tools(self, server_name):
            return []

    monkeypatch.setattr(mcp, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(mcp, "get_full_config", lambda: config)
    monkeypatch.setattr(mcp, "load_skill", lambda name, loaded_config: "")

    asyncio.run(mcp.initialize_mcp_client())

    assert observed["connections"]["okg_cms"]["command"] == "/opt/okg/bin/python"


def test_base_agent_passes_its_runtime_config_to_mcp_initializer(monkeypatch):
    observed = {}
    config = _runtime_config()

    async def initialize(runtime_config):
        observed["config"] = runtime_config
        return None, [], ""

    runner = SimpleNamespace(run=lambda coroutine: asyncio.run(coroutine))
    monkeypatch.setattr(base_react, "initialize_mcp_client", initialize)
    monkeypatch.setattr(
        base_react.AsyncLoopThread,
        "get_instance",
        classmethod(lambda cls: runner),
    )
    agent = base_react.BaseReActAgent.__new__(base_react.BaseReActAgent)
    agent.config = config

    assert agent._build_mcp_tools() is None
    assert observed["config"] is config


def test_initialize_mcp_client_applies_allowed_tools(monkeypatch):
    observed = {}
    names = ["inspect", "search", "expand", "filter", "map", "aggregate", "query"]
    all_tools = [SimpleNamespace(name=n, description=n, handle_tool_error=False) for n in names]

    class FakeClient:
        def __init__(self, connections):
            observed["connections"] = connections

        async def get_tools(self, server_name):
            return list(all_tools)

    config = {
        "mcp_servers": {
            "graph_server": {
                "transport": "streamable_http",
                "url": "http://172.17.0.1:8767/mcp",
                "headers": {"Authorization": "Bearer test"},
                "allowed_tools": names[:6],
            }
        }
    }
    monkeypatch.setattr(mcp, "MultiServerMCPClient", FakeClient)

    _, tools, _ = asyncio.run(mcp.initialize_mcp_client(config))

    assert [t.name for t in tools] == names[:6]
    # archi-only field is not passed to the MCP client; transport fields are
    assert "allowed_tools" not in observed["connections"]["graph_server"]
    assert observed["connections"]["graph_server"]["headers"] == {"Authorization": "Bearer test"}
