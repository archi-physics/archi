"""Open WebUI's scripted search call: the registered tool server, from inside the compose network.

    python webui_check.py --instance DIR [--query "CMSSW release"]

Open WebUI needs no client config (`archi up` registers the MCP server with `okg chat sync`), so this
checks what the site holds. It signs in to the site as its admin, reads the tool server the site
registered (URL and key), then runs a small MCP client *inside the webui container*
(`podman|docker exec -i <webui container> python -c ...`): initialize, tools/call `search`, over the
compose network at the URL the site registered, not the host port. Exit 0 only if the answer cites
rows (`[n] <node id> (<subtype>)`). The key is passed on stdin and never printed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

ROW = re.compile(r"^\[(\d+)\] (\S+:\S+) \(([^)]+)\)", re.M)

#: Runs inside the container with the container's own Python (urllib only: no mcp package needed).
IN_CONTAINER = r'''
import json, sys, urllib.request

req = json.load(sys.stdin)
base = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
        "Authorization": "Bearer " + req["key"]}

def post(body, session=None):
    headers = dict(base)
    if session:
        headers["Mcp-Session-Id"] = session
    r = urllib.request.Request(req["url"], data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(r, timeout=60) as resp:
        raw = resp.read().decode()
        sid = resp.headers.get("Mcp-Session-Id")
        kind = resp.headers.get("Content-Type", "")
    if "text/event-stream" in kind:
        events = [json.loads(l[5:].strip()) for l in raw.splitlines() if l.startswith("data:") and l[5:].strip()]
        raw = events[-1] if events else None
        return sid, raw
    return sid, json.loads(raw) if raw.strip() else None

sid, init = post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "archi-webui-check", "version": "1"}}})
post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
_, tools = post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, sid)
_, call = post({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": req["query"]}}}, sid)
result = call["result"]
text = "".join(c.get("text", "") for c in result.get("content", []))
print(json.dumps({"tools": [t["name"] for t in tools["result"]["tools"]], "is_error": bool(result.get("isError")),
                  "text": text}))
'''


class CheckError(Exception):
    pass


def registered_server(instance) -> dict[str, str]:
    """The MCP tool server the site registered: {"url", "key", "name"}. Needs the site running."""
    from archi.stack import webui
    from archi.stack.instance import read_env

    secrets = read_env(instance.env_path)
    base = f"http://127.0.0.1:{instance.ports.webui}"
    token = webui.admin_token(base, instance.admin_email, secrets["ARCHI_ADMIN_PASSWORD"])
    req = urllib.request.Request(f"{base}/api/v1/configs/tool_servers", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        connections = json.load(resp).get("TOOL_SERVER_CONNECTIONS", [])
    mcp = [c for c in connections if c.get("type") == "mcp"]
    if not mcp:
        raise CheckError("the site has no registered MCP tool server; run `archi up` (okg chat sync)")
    c = mcp[0]
    if not c.get("key"):
        raise CheckError("the registered tool server has no bearer key")
    return {"url": c["url"], "key": c["key"], "name": (c.get("info") or {}).get("name", "")}


def webui_container(instance, runtime: str, runner: Callable[..., Any] = subprocess.run) -> str:
    out = runner([runtime, "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True).stdout
    names = [n for n in out.split() if n.startswith(instance.project) and "webui" in n]
    if len(names) != 1:
        raise CheckError(f"expected one running webui container for {instance.project}, found {names}")
    return names[0]


def search_in_container(exec_prefix: Sequence[str], url: str, key: str, query: str, *,
                        python: str = "python", runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Run the MCP client in the container; `exec_prefix` is e.g. ["podman", "exec", "-i", NAME]."""
    done = runner([*exec_prefix, python, "-c", IN_CONTAINER], input=json.dumps({"url": url, "key": key, "query": query}),
                  capture_output=True, text=True, timeout=180)
    if done.returncode != 0:
        raise CheckError(f"the call inside the container failed ({done.returncode}): "
                         + (done.stderr or done.stdout).replace(key, "<key>")[-800:])
    result = json.loads(done.stdout.strip().splitlines()[-1])
    rows = [{"rank": int(n), "node_id": nid, "subtype": sub} for n, nid, sub in ROW.findall(result["text"])]
    return {"tools": result["tools"], "is_error": result["is_error"], "citations": rows,
            "text": result["text"][:1500], "ok": bool(rows) and not result["is_error"]}


def main(argv: Optional[list[str]] = None) -> int:
    from archi.stack.instance import Instance

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("."))
    parser.add_argument("--query", default="CMSSW release")
    args = parser.parse_args(argv)
    instance = Instance.load(args.instance)
    try:
        reg = registered_server(instance)
        container = webui_container(instance, instance.runtime)
        result = search_in_container([instance.runtime, "exec", "-i", container], reg["url"], reg["key"], args.query)
    except (CheckError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"client": "open-webui", "ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"client": "open-webui", "container": container, "registered_url": reg["url"],
                      "tool_server": reg["name"], **result}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
