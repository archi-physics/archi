"""Check a written client config by calling its MCP servers, with no LLM.

    python recipe_check.py --client claude-code --config .mcp.json \\
        [--instance-server archi-demo] [--query "CMSSW_14_0"] [--env-file DIR/.env]

Reads each server's URL and headers exactly as the client config holds them,
the way the client would: `${VAR}` in a header value is read from the
environment (then --env-file); Codex's `bearer_token_env_var` and
`env_http_headers` are read the same way. A variable that is not set is an error,
never an empty header. Then, with the `mcp` client library:

  * the instance server (--instance-server, default: the first server named
    `archi-*`): lists tools and calls `search`;
  * every other server: lists tools and calls `echo` when it has one.

Prints one JSON object and exits 0 only when every call returned without error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping, Optional

JSON_CLIENTS = ("claude-code", "openclaw", "mcp")
ROW = re.compile(r"^\[(\d+)\] (\S+:\S+) \(([^)]+)\)", re.M)
_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class CheckError(Exception):
    pass


def read_env_file(path: Optional[Path]) -> dict[str, str]:
    values: dict[str, str] = {}
    if path and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def _expand(text: str, env: Mapping[str, str], where: str) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1) not in env:
            raise CheckError(f"{where}: ${{{m.group(1)}}} is not set in the environment or --env-file")
        return env[m.group(1)]

    return _REF.sub(sub, text)


def _var(name: str, env: Mapping[str, str], where: str) -> str:
    if name not in env:
        raise CheckError(f"{where}: environment variable {name} is not set (environment or --env-file)")
    return env[name]


def read_servers(client: str, config: Path, env: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """{name: {"url": ..., "headers": {...}}}, exactly as the client would send them."""
    text = config.read_text(encoding="utf-8")
    servers: dict[str, dict[str, Any]] = {}
    if client == "codex":
        raw = tomllib.loads(text).get("mcp_servers") or {}
        for name, body in raw.items():
            if "url" not in body:
                continue  # a stdio server, not ours
            where = f"{config} [mcp_servers.{name}]"
            headers = {k: _expand(str(v), env, where) for k, v in (body.get("http_headers") or {}).items()}
            for k, var in (body.get("env_http_headers") or {}).items():
                headers[k] = _var(var, env, where)
            if body.get("bearer_token_env_var"):
                headers["Authorization"] = f"Bearer {_var(body['bearer_token_env_var'], env, where)}"
            servers[name] = {"url": body["url"], "headers": headers}
        return servers
    doc = json.loads(text)
    raw = (doc.get("mcp") or {}).get("servers") if client == "openclaw" else doc.get("mcpServers")
    for name, body in (raw or {}).items():
        if "url" not in body:
            continue
        where = f"{config} {name}"
        servers[name] = {
            "url": body["url"],
            "headers": {k: _expand(str(v), env, where) for k, v in (body.get("headers") or {}).items()},
        }
    return servers


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


async def call_server(url: str, headers: dict[str, str], calls: list[tuple[str, dict]]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    out: dict[str, Any] = {"url": url, "tools": [], "calls": []}
    async with streamablehttp_client(url, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            out["tools"] = tools
            for tool, arguments in calls:
                if tool not in tools:
                    out["calls"].append({"tool": tool, "ok": False, "text": f"server has no tool {tool!r}"})
                    continue
                result = await session.call_tool(tool, arguments)
                out["calls"].append({"tool": tool, "ok": not result.isError, "text": _text(result)[:2000]})
    return out


async def check(client: str, servers: dict[str, dict[str, Any]], instance_server: Optional[str],
                query: str, echo_text: str) -> dict[str, Any]:
    if not servers:
        raise CheckError("the config holds no HTTP MCP servers")
    primary = instance_server or next((n for n in servers if n.startswith("archi-")), None)
    if primary not in servers:
        raise CheckError(f"instance server {primary!r} is not in the config (has: {', '.join(servers)})")
    report: dict[str, Any] = {"client": client, "servers": {}, "ok": True}
    for name, spec in servers.items():
        calls = [("search", {"query": query})] if name == primary else [("echo", {"text": echo_text})]
        try:
            result = await call_server(spec["url"], spec["headers"], calls)
        except Exception as exc:  # noqa: BLE001 - the report names the failure
            result = {"url": spec["url"], "error": f"{type(exc).__name__}: {exc}"}
        result["role"] = "instance" if name == primary else "external"
        ok = "error" not in result and all(c["ok"] for c in result["calls"])
        if name == primary and ok:
            # search must return cited rows: `[n] <node id> (<subtype>)`
            rows = ROW.findall(result["calls"][0]["text"]) if result["calls"] else []
            result["citations"] = [{"rank": int(n), "node_id": nid, "subtype": sub} for n, nid, sub in rows]
            ok = bool(rows)
        if name != primary and "error" not in result and "echo" not in result.get("tools", []):
            # an external server without an echo tool is only connected to, not called
            result["calls"] = []
            ok = True
        result["ok"] = ok
        report["ok"] = report["ok"] and ok
        report["servers"][name] = result
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=("claude-code", "codex", "openclaw", "mcp"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instance-server")
    parser.add_argument("--query", default="CMSSW")
    parser.add_argument("--echo-text", default="archi-connect")
    parser.add_argument("--env-file", type=Path, help="KEY=VALUE file read after the environment (an instance .env)")
    args = parser.parse_args(argv)
    env = {**read_env_file(args.env_file), **os.environ}
    try:
        servers = read_servers(args.client, args.config, env)
        report = asyncio.run(check(args.client, servers, args.instance_server, args.query, args.echo_text))
    except CheckError as exc:
        print(json.dumps({"client": args.client, "ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
