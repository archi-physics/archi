"""Deterministic MCP server used by evaluator transport tests."""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

server = FastMCP(
    "qa-evaluator-test",
    host="127.0.0.1",
    port=int(os.environ.get("QA_FAKE_MCP_PORT", "8000")),
    stateless_http=True,
    json_response=True,
)


class CapacityResult(BaseModel):
    service: str
    available: int
    revision: str


@server.tool(structured_output=True)
def current_capacity(service: str) -> CapacityResult:
    value_path = os.environ.get("QA_FAKE_MCP_VALUE_FILE")
    available = (
        int(Path(value_path).read_text(encoding="utf-8").strip()) if value_path else 7
    )
    return CapacityResult(
        service=service,
        available=available,
        revision=f"fixture-r{available}" if value_path else "fixture-r1",
    )


if __name__ == "__main__":
    server.run(transport=os.environ.get("QA_FAKE_MCP_TRANSPORT", "stdio"))
