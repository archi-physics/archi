"""Fakes for the agent-arm tests (imported by module name): a scripted model, a fake MCP
server with `search` and `inspect`, and a small hand-written raw corpus. No network, no
paid call: every model here is a Python function."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

from archi.evaluation.agent import ChatReply, ModelError, ToolCall, ToolSpec
from archi.evaluation.raw_corpus import CORPUS_SCHEMA, INFO, KEYS, MANIFEST, OPS_RECORD, OPS_RECORD_REASON, RECORDS_DIR

SEARCH = ToolSpec("search", "Find records by words", {"type": "object", "properties": {"query": {"type": "string"}}})
INSPECT = ToolSpec("inspect", "Read one named thing", {"type": "object"})

#: The fixture corpus: source -> natural key -> (file under records/, content, node id).
RECORDS = {
    "jira": {
        "DEMO-1": ("jira/DEMO-1.json", '{"key": "DEMO-1", "description": "Transfer failures at the test site"}\n',
                   "jira:DEMO-1"),
        "DEMO-2": ("jira/DEMO-2.json", '{"key": "DEMO-2", "description": "Upgrade the storage servers"}\n',
                   "jira:DEMO-2"),
    },
    "docsite": {
        "https://docs.example.org/start.html": (
            "docsite/https%3A%2F%2Fdocs.example.org%2Fstart.html.json",
            '{"url": "https://docs.example.org/start.html", "body": "Install the client, then run the setup script now."}\n',
            "documentation_page:abc123"),
    },
    "slurm": {
        "gpu": ("slurm/partitions/gpu.json", '{"name": "gpu"}\n', "slurm_partition:submit:gpu"),
    },
}
#: inspect's answer per node id: (subtype, producer source, attrs)
NODES = {
    "jira:DEMO-1": ("jira_issue", "demo.jira", {"key": "DEMO-1"}),
    "jira:DEMO-2": ("jira_issue", "demo.jira", {"key": "DEMO-2"}),
    "documentation_page:abc123": ("documentation_page", "demo.docsite",
                                  {"url": "https://docs.example.org/start.html"}),
    "slurm_partition:submit:gpu": ("slurm_partition", "demo.slurm", {"name": "gpu"}),
    "ops_incident:inc-1": ("ops_incident", "workflow:archi-ops", {"title": "an incident"}),
    "hn:42": ("hypernews_thread", "demo.hypernews", {}),
    "mystery:1": ("thing", "demo.unknown_source", {}),
}


def write_corpus(root: Path, bundle: str = "demo", generation: Optional[str] = "g-1") -> Path:
    """A corpus laid out as `archi eval --raw-corpus` lays one out."""
    path = root / "corpus"
    records = path / RECORDS_DIR
    manifest = []
    keys: dict = {}
    for source, table in RECORDS.items():
        lookup, natural = {}, {}
        for key, (rel, content, node_id) in table.items():
            target = records / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            manifest.append((rel, hashlib.sha256(content.encode()).hexdigest()))
            lookup[key] = rel
            lookup[node_id] = rel
            natural[rel] = key
        keys[source] = {"group": source, "lookup": lookup, "ambiguous": [], "natural_key": natural}
    manifest.sort()
    text = "".join(f"{rel}\t{sha}\n" for rel, sha in manifest)
    (path / MANIFEST).write_text(text, encoding="utf-8")
    left_out = {OPS_RECORD: OPS_RECORD_REASON, "hypernews": "crawled live"}
    (path / KEYS).write_text(json.dumps({"schema": CORPUS_SCHEMA, "sources": keys, "left_out": left_out}))
    info = {"schema": CORPUS_SCHEMA, "bundle": bundle, "digest": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            "lock_sha256": "0" * 64, "built_at": "2026-10-27T00:00:00Z", "generation": generation,
            "coverage": f"{bundle}: jira, docsite, slurm", "files": len(manifest),
            "sources": {s: {"group": s, "files": len(t)} for s, t in RECORDS.items()}, "left_out": left_out}
    (path / INFO).write_text(json.dumps(info))
    return path


#: What a search row's excerpt shows of each fixture record (its summary or body).
EXCERPTS = {
    "jira:DEMO-1": "Transfer failures at the test site",
    "jira:DEMO-2": "Upgrade the storage servers",
    "documentation_page:abc123": "Install the client, then run the setup script now.",
}


def search_rows(*ids: str, excerpts: bool = True) -> str:
    """okg's text rows, each with its record's excerpt (so the record is read), or, with
    `excerpts=False`, its id alone (so the record is only listed)."""
    lines = []
    for i, nid in enumerate(ids, 1):
        lines.append(f"[{i}] {nid} (thing) · source demo")
        if excerpts and nid in EXCERPTS:
            lines.append(f"    {EXCERPTS[nid]}")
    return "search ok · rows · generation g-1\n\n" + "\n".join(lines) + "\n"


class FakeMcp:
    """`search` answers from `results` (query -> text); `inspect` from NODES."""

    instructions = None

    def __init__(self, results: Optional[dict] = None, nodes: Optional[dict] = None) -> None:
        self.results = dict(results or {})
        self.nodes = dict(NODES if nodes is None else nodes)
        self.calls: list = []

    async def list_tools(self):
        return [SEARCH, INSPECT]

    async def call(self, name, arguments, *, meta=None):
        self.calls.append((name, dict(arguments), meta))
        if name == "inspect" and arguments["target"].get("kind") == "graph":
            return json.dumps({"pinned_generation_id": "g-1", "counts": {}}), False
        if name == "inspect":
            nid = arguments["target"]["node_id"]
            if nid not in self.nodes:
                return f"error · inspect node: no visible record has node_id {nid}", True
            subtype, source, attrs = self.nodes[nid]
            return json.dumps({"node_id": nid, "subtype": subtype, "attrs": attrs,
                               "provenance": {"source": source, "trust": "trusted"},
                               "generation_id": arguments.get("generation_id", "g-1")}), False
        if name == "search":
            text = self.results.get(arguments.get("query"))
            if text is None:
                return "error · search: nothing", True
            return text, False
        return f"error: no tool {name}", True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


class ScriptedModel:
    """Replies from `script(messages, turn)`; counts calls. Never touches the network."""

    def __init__(self, script: Callable[[list, int], ChatReply], name: str = "fake-model") -> None:
        self.script, self.name, self.calls = script, name, 0

    def complete(self, messages, tools):
        self.calls += 1
        return self.script(messages, self.calls)


def reply(content: str = "", *calls: tuple[str, dict], prompt: int = 1000, completion: int = 100) -> ChatReply:
    return ChatReply(content, tuple(ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(calls)),
                     prompt_tokens=prompt, completion_tokens=completion)


def failing(messages, turn):
    raise ModelError("http://model: HTTP 500: boom", status=500, sent=True)


def http_error(status: int, body: str = "boom") -> ModelError:
    return ModelError(f"http://model: HTTP {status}: {body}", status=status, sent=True)
