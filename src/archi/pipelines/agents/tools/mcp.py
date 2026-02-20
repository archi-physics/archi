from __future__ import annotations
from typing import List, Any, Tuple, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.tools import BaseTool

from src.utils.config_access import get_archi_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def initialize_mcp_client() -> Tuple[Optional[MultiServerMCPClient], List[BaseTool]]:
    """
    Initializes the MCP client and fetches tool definitions.
    Returns:
        client: The active client instance (must be kept alive by the caller).
        tools: The list of LangChain-compatible tools.
    """

    archi_cfg = get_archi_config()
    mcp_servers = archi_cfg.get("mcp_servers") or {}
    client = MultiServerMCPClient(mcp_servers)

    all_tools: List[BaseTool] = []
    failed_servers: dict[str, str] = {}

    for name in mcp_servers.keys():
        try:
            tools = await client.get_tools(server_name=name)
            all_tools.extend(tools)
        except Exception as e:
            failed_servers[name] = str(e)

    logger.info(f"Active MCP servers: {[n for n in mcp_servers if n not in failed_servers]}")
    logger.warning(f"Failed MCP servers: {list(failed_servers.keys())}")

    return client, all_tools
