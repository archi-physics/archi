"""Ask an instance's MCP endpoint a question; succeed only on a cited answer.

    python mcp_cited_answer.py --instance <dir> "Which CMSSW_14_0 releases are in the graph?"

Runs `search` over the instance's streamable-HTTP MCP endpoint with its
bearer token, the way any MCP client would (ADR 0002 D5: the MCP endpoint is
the supported surface). Prints one JSON object with the rows' node ids and
the source that wrote them, and exits 0 only when at least one row names a
node id and every cited row names its source: in the result header when one
source wrote them all, otherwise on each row.

The nightly workflow's last user command (req.r3-golden-path.nightly).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROW = re.compile(r"^\[(\d+)\] (\S+:\S+) \(([^)]+)\)(?:.*? · source (\S+))?.*$", re.M)
HEADER = re.compile(r"^search ok · .*? · source (\S+)", re.M)


async def ask(url: str, token: str, question: str) -> str:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("search", {"query": question})
            if result.isError:
                raise RuntimeError("".join(getattr(c, "text", "") for c in result.content))
            return "".join(getattr(c, "text", "") for c in result.content)


def main(argv=None) -> int:
    from archi.stack.instance import Instance, read_env

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=Path("."))
    parser.add_argument("question")
    args = parser.parse_args(argv)
    instance = Instance.load(args.instance)
    token = read_env(instance.env_path)["OKG_CHAT_MCP_TOKEN"]
    url = f"http://127.0.0.1:{instance.ports.mcp}/mcp"
    text = asyncio.run(ask(url, token, args.question))
    header = HEADER.search(text)
    rows = [{"rank": int(n), "node_id": nid, "subtype": sub,
             "source": src or (header.group(1) if header else None)}
            for n, nid, sub, src in ROW.findall(text)]
    answer = {
        "question": args.question,
        "endpoint": url,
        "sources": sorted({r["source"] for r in rows if r["source"]}),
        # Kept for the nightly workflow's summary step: the single source, or all of them.
        "source": ", ".join(sorted({r["source"] for r in rows if r["source"]})) or None,
        "citations": rows[:10],
        "excerpt": text[:1200],
    }
    print(json.dumps(answer, indent=2))
    if not rows or any(not r["source"] for r in rows):
        print("no cited answer", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
