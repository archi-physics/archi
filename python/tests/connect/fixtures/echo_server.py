"""A tiny MCP server over streamable HTTP, for the `archi connect` tests and lab runs.

    python echo_server.py --port 8765 [--token T] [--header-name X-Api-Key] [--kind external|instance]

`external` (the default) exposes one tool, `echo`. `instance` stands in for an
instance's MCP endpoint and exposes `search`, which returns rows in the shape
the real server uses (`search ok · ... · source <name>` then `[1] node:id (subtype)`).
With --token the server answers 401 to any request that does not carry the secret
in the named header (`Authorization: Bearer T` by default).
"""
from __future__ import annotations

import argparse
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import PlainTextResponse


def build_mcp(kind: str = "external") -> FastMCP:
    mcp = FastMCP(
        f"fixture-{kind}",
        stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    if kind == "external":
        @mcp.tool()
        def echo(text: str) -> str:
            """Return the text unchanged."""
            return f"echo: {text}"
    else:
        @mcp.tool()
        def search(query: str) -> str:
            """A canned search result."""
            return (
                f"search ok · 1 rows · source fixture-source\n"
                f"[1] doc:fixture:readme (document) matched {query!r}\n"
            )
    return mcp


class RequireHeader:
    """ASGI middleware: 401 unless `header` equals `value`."""

    def __init__(self, app, header: str, value: str):
        self.app, self.header, self.value = app, header.lower().encode(), value.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            got = dict(scope["headers"]).get(self.header)
            if got != self.value:
                await PlainTextResponse("unauthorized", status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(kind: str = "external", header: Optional[str] = None, value: Optional[str] = None):
    app = build_mcp(kind).streamable_http_app()
    if header and value:
        app = RequireHeader(app, header, value)
    return app


@contextmanager
def serve(kind: str = "external", header: Optional[str] = None, value: Optional[str] = None,
          port: int = 0) -> Iterator[str]:
    """Run the server in a thread; yield its `/mcp` URL."""
    config = uvicorn.Config(build_app(kind, header, value), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("fixture MCP server did not start")
        time.sleep(0.02)
    bound = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{bound}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--kind", choices=("external", "instance"), default="external")
    parser.add_argument("--token")
    parser.add_argument("--header-name", default="Authorization")
    args = parser.parse_args(argv)
    value = None
    if args.token:
        value = args.token if args.header_name.lower() != "authorization" else f"Bearer {args.token}"
    uvicorn.run(build_app(args.kind, args.header_name, value), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
