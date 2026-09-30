from __future__ import annotations
import os
import httpx
import base64
from typing import List, Any, Tuple, Optional, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.tools import BaseTool

from src.utils.config_access import get_full_config
from src.utils.logging import get_logger
from src.archi.pipelines.agents.utils.skill_utils import load_skill
from src.utils.env import read_secret

logger = get_logger(__name__)

def mcp_http_client_factory(**kwargs) -> httpx.AsyncClient:
    if "verify" in kwargs:
        kwargs.setdefault("verify", kwargs["verify"])
    user = read_secret("MCP_HTTP_USERNAME") or None
    password = read_secret("MCP_HTTP_PASSWORD") or None
    if user and password:
        auth_bytes = f"{user}:{password}".encode("utf-8")
        auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")
        if kwargs.get("headers"):
            kwargs["headers"].update({"Authorization":f"Basic {auth_b64}"})
        else:
            kwargs["headers"] = {"Authorization":f"Basic {auth_b64}"}
    return httpx.AsyncClient(**kwargs)


async def initialize_mcp_client(config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[MultiServerMCPClient], List[BaseTool], str]:
    """
    Initializes the MCP client and fetches tool definitions.
    Returns:
        client: The active client instance (must be kept alive by the caller).
        tools: The list of LangChain-compatible tools.
        skills_text: Concatenated skill content from all MCP servers that declare
            a `skill`. Empty string if no server has a skill. The caller is
            responsible for appending this to the agent's system prompt — we inject
            here only once per agent rather than into each tool description, so
            the content doesn't multiply by tool count.
    """

    full_config = config if config is not None else get_full_config()
    mcp_servers = full_config.get("mcp_servers", {})
    if not mcp_servers:
        logger.info("No MCP servers configured.")
        return None, [], ""

    # Strip archi-only fields that langchain-mcp-adapters doesn't understand.
    # These are consumed by the compose template (sidecars), the legacy stdio
    # install path, or post-load tool customization — the MCP client itself only
    # knows about transport-specific fields.
    _archi_only_fields = {
        "env_from_secrets", "host_file_mounts", "build_context", "image", "path", "skill",
        "shared_volume", "allowed_tools",
    }
    client_configs: dict[str, dict] = {}
    server_skills: dict[str, str] = {}
    for name, server_cfg in mcp_servers.items():
        # Load any declared skill so we can append it to this server's tool descriptions.
        skill_name = server_cfg.get("skill")
        if skill_name:
            skill_content = load_skill(skill_name, full_config)
            if skill_content:
                server_skills[name] = skill_content

        cfg = {k: v for k, v in server_cfg.items() if k not in _archi_only_fields}
        transport = cfg.get("transport")
        if transport == "stdio":
            # stdio subprocesses inherit nothing by default (mcp.client.stdio uses
            # an empty env). Forward the parent process env so stdio MCP servers see
            # what they need.
            cfg["env"] = {**os.environ, **(cfg.get("env") or {})}
        else:
            # For HTTP-based transports, `env` is for the sidecar container (compose),
            # not the MCP client connection — drop it here.
            server_env = server_cfg.get("env") or {}
            if server_env.get("httpx_client_factory"):
                host_mounts = server_cfg.get("host_file_mounts") or []
                verify = host_mounts[0] if host_mounts else True
                # Bind per-server values as defaults: a plain closure over the
                # loop variable would resolve to the LAST server for every
                # factory created in this loop.
                cfg["httpx_client_factory"] = (
                    lambda _verify=verify, **kwargs: mcp_http_client_factory(
                        verify=_verify, **kwargs
                    )
                )
            cfg.pop("env", None)
        client_configs[name] = cfg

    logger.info(f"Configuring MCP client with servers: {list(client_configs.keys())}")
    client = MultiServerMCPClient(client_configs)

    all_tools: List[BaseTool] = []
    failed_servers: dict[str, str] = {}

    for name in client_configs.keys():
        try:
            tools = await client.get_tools(server_name=name)
            # Optional per-server allow-list: expose only the listed tools of this
            # server (e.g. to hide a raw-query or write tool).
            allowed = mcp_servers[name].get("allowed_tools")
            if allowed:
                dropped = [t.name for t in tools if t.name not in allowed]
                tools = [t for t in tools if t.name in allowed]
                if dropped:
                    logger.info(f"MCP server '{name}': tools not in allowed_tools dropped: {dropped}")
            for tool in tools:
                # Return error messages to the LLM instead of crashing the agent chain.
                tool.handle_tool_error = True
                logger.info(f"Loaded tool from MCP server '{name}': {tool.name} - {tool.description}")
            all_tools.extend(tools)
        except Exception as e:
            logger.error(f"Failed to fetch tools from MCP server '{name}': {e}", exc_info=e)
            failed_servers[name] = str(e)

    logger.info(f"Active MCP servers: {[n for n in client_configs if n not in failed_servers]}")
    logger.warning(f"Failed MCP servers: {list(failed_servers.keys())}")

    # Build a single combined skills block keyed by server name — this is appended
    # to the agent's system prompt once, rather than duplicated across every tool.
    skills_parts: List[str] = []
    for name, skill_content in server_skills.items():
        if name not in failed_servers:
            skills_parts.append(
                f"\n--- {name} MCP Server Domain Knowledge ---\n{skill_content}"
            )
    skills_text = "".join(skills_parts)

    return client, all_tools, skills_text
