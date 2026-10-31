"""
MCP SSE endpoint – exposes archi's RAG capabilities as MCP tools over HTTP+SSE.

AI assistants in VS Code (GitHub Copilot), Cursor, Claude Desktop, Claude Code,
and any other MCP-compatible client can connect with just a URL:

    http://<host>:<port>/mcp/sse

No local installation required on the client side.

VS Code  (.vscode/mcp.json):
    {
      "servers": {
        "archi": { "type": "sse", "url": "http://localhost:7861/mcp/sse" }
      }
    }

Cursor  (~/.cursor/mcp.json):
    {
      "mcpServers": {
        "archi": { "url": "http://localhost:7861/mcp/sse" }
      }
    }

Claude Desktop  (~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
                "archi": {
                    "command": "npx",
                    "args": [
                        "mcp-remote",
                        "http://localhost:7861/mcp/sse",
                        "--header",
                        "Authorization: Bearer <TOKEN>"
                    ]
                }
      }
    }

Claude Code  (run once in terminal):
    claude mcp add --transport sse archi http://localhost:7861/mcp/sse

    Or add to .mcp.json in your project root:
    {
      "mcpServers": {
        "archi": { "type": "sse", "url": "http://localhost:7861/mcp/sse" }
      }
    }

Implements the MCP SSE transport (JSON-RPC 2.0 over Server-Sent Events)
directly in Flask using thread-safe queues — no external ``mcp`` package needed.
"""

from __future__ import annotations

import json
import queue
import re
import shlex
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import BoundedSemaphore, Lock
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psycopg2
import yaml
from flask import Blueprint, Response, jsonify, request, stream_with_context

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MCP_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "archi", "version": "1.0.0"}
_KEEPALIVE_TIMEOUT = 30  # seconds between keepalive pings
_MCP_DISPATCH_MAX_WORKERS = 48
_MCP_DISPATCH_MAX_INFLIGHT = 512

# ---------------------------------------------------------------------------
# Session registry  (session_id → {"queue": Queue, "user_id": str|None})
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict] = {}
_sessions_lock = Lock()
_dispatch_executor = ThreadPoolExecutor(
    max_workers=_MCP_DISPATCH_MAX_WORKERS,
    thread_name_prefix="mcp-dispatch",
)
_dispatch_slots = BoundedSemaphore(_MCP_DISPATCH_MAX_INFLIGHT)


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------


def _validate_mcp_token(token: str, pg_config: Optional[dict]) -> Optional[str]:
    """Validate an MCP bearer token and return the user_id, or None if invalid."""
    if not token or not pg_config:
        return None
    try:
        conn = psycopg2.connect(**pg_config)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT user_id FROM mcp_tokens
                       WHERE token = %s
                         AND (expires_at IS NULL OR expires_at > NOW())""",
                    (token,),
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "UPDATE mcp_tokens SET last_used_at = NOW() WHERE token = %s",
                        (token,),
                    )
                    conn.commit()
                    return row[0]
        finally:
            conn.close()
    except Exception:
        logger.exception("Error validating MCP token")
    return None


def _extract_bearer_token(req) -> Optional[str]:
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None

# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "archi_query",
        "description": textwrap.dedent("""\
            Ask a question to the archi RAG (Retrieval-Augmented Generation) system.

            archi retrieves relevant documents from its knowledge base and uses an LLM
            to compose a grounded answer.  Use this tool when you need information that
            is stored in the connected archi deployment (documentation, tickets, wiki
            pages, research papers, course material, etc.).

            You may continue a conversation by passing the conversation_id returned by
            a previous call.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or request to send to archi.",
                },
                "conversation_id": {
                    "type": "integer",
                    "description": (
                        "Optional. Pass the conversation_id from a previous archi_query "
                        "call to continue the same conversation thread."
                    ),
                },
                "provider": {
                    "type": "string",
                    "description": "Optional. Override the LLM provider (e.g. 'openai', 'anthropic').",
                },
                "model": {
                    "type": "string",
                    "description": "Optional. Override the specific model (e.g. 'gpt-4o').",
                },
                "config_name": {
                    "type": "string",
                    "description": "Optional. The deployment config name to use (e.g. 'comp_ops'). Defaults to the active config.",
                },
                "client_timeout": {
                    "type": "number",
                    "description": "Optional. Request timeout in milliseconds (default 18000000 = 5 hours).",
                    "default": 18000000,
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "archi_list_documents",
        "description": textwrap.dedent("""\
            List the documents that have been indexed into archi's knowledge base.

            Returns a paginated list of document metadata (filename, source type,
            URL, enabled state, ingestion status, etc.).  Use this tool to discover
            what information archi has access to before querying it, or to find a
            specific document's hash for use with archi_get_document_content.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "integer",
                    "description": (
                        "Optional. Filter enabled/disabled state for a specific "
                        "conversation."
                    ),
                },
                "search": {
                    "type": "string",
                    "description": "Optional keyword to filter documents by name or URL.",
                },
                "source_type": {
                    "type": "string",
                    "description": "Optional. Filter by source type: 'web', 'git', 'local', 'jira', etc.",
                },
                "enabled": {
                    "type": "string",
                    "description": (
                        "Optional. Filter by enabled state: 'enabled', 'disabled', "
                        "or 'all' (default)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 50, max 500).",
                    "default": 50,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default 0).",
                    "default": 0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "archi_get_document_content",
        "description": textwrap.dedent("""\
            Retrieve the full text content of a document indexed in archi.

            Use archi_list_documents first to obtain a document's hash, then pass
            it here to read the raw source text that archi ingested.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_hash": {
                    "type": "string",
                    "description": "The document hash returned by archi_list_documents.",
                },
                "max_size": {
                    "type": "integer",
                    "description": (
                        "Optional maximum number of bytes/chars to return "
                        "(default 100000, max 1000000)."
                    ),
                    "default": 100000,
                },
            },
            "required": ["document_hash"],
        },
    },
    {
        "name": "archi_search_document_metadata",
        "description": textwrap.dedent("""\
            Search the indexed document catalog by metadata, paths, URLs, ticket IDs,
            and other stored document attributes.

            Supports free text plus exact `key:value` filters. Multiple filter groups
            can be OR-ed with the literal token `OR`, matching the same metadata-query
            syntax used by archi's built-in agent tools.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Metadata query string, e.g. "
                        "`source_type:git relative_path:docs/README.md`."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10, max 100).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "archi_list_metadata_schema",
        "description": textwrap.dedent("""\
            List the metadata filter keys and common values supported by
            archi_search_document_metadata.

            Use this tool before metadata searches when you do not know which
            fields exist in the catalog.
        """),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "archi_search_document_content",
        "description": textwrap.dedent("""\
            Search indexed document contents for an exact phrase or regex pattern.

            This is a grep-like content search intended for logs, error messages,
            code snippets, and other exact-text lookups. Optionally pre-filter the
            candidate documents with a metadata query.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Phrase or regex pattern to search for.",
                },
                "metadata_query": {
                    "type": "string",
                    "description": (
                        "Optional metadata pre-filter using the same syntax as "
                        "archi_search_document_metadata."
                    ),
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat `query` as a regular expression.",
                    "default": False,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Perform a case-sensitive match.",
                    "default": False,
                },
                "before": {
                    "type": "integer",
                    "description": "Number of context lines before each match.",
                    "default": 0,
                },
                "after": {
                    "type": "integer",
                    "description": "Number of context lines after each match.",
                    "default": 0,
                },
                "max_matches_per_document": {
                    "type": "integer",
                    "description": "Maximum matches to show per document.",
                    "default": 3,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum documents to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "archi_get_document_chunks",
        "description": textwrap.dedent("""\
            Inspect the stored chunks for a document as they exist in archi's
            vectorized corpus.

            Useful for debugging chunk boundaries, truncation, and ingestion issues.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_hash": {
                    "type": "string",
                    "description": "The document hash returned by archi_list_documents.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Chunk offset to start from (default 0).",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum chunks to return (default 20, max 100).",
                    "default": 20,
                },
                "max_chars_per_chunk": {
                    "type": "integer",
                    "description": "Maximum characters to show per chunk (default 600).",
                    "default": 600,
                },
            },
            "required": ["document_hash"],
        },
    },
    {
        "name": "archi_get_data_stats",
        "description": textwrap.dedent("""\
            Return corpus-level statistics for the connected archi deployment.

            Includes total documents, total chunks, enabled/disabled counts,
            ingestion status counts, bytes stored, and a breakdown by source type.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "integer",
                    "description": (
                        "Optional. Compute enabled/disabled counts for a specific "
                        "conversation."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "archi_get_deployment_info",
        "description": textwrap.dedent("""\
            Return configuration and status information about the connected archi
            deployment.

            Includes the active LLM pipeline and model, retrieval settings (number of
            documents retrieved, hybrid search weights), embedding model, and the list
            of available pipelines.  Useful for understanding how archi is configured
            before issuing queries.
        """),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "archi_list_agents",
        "description": textwrap.dedent("""\
            Return the agent configurations (agent specs) available in this archi
            deployment.

            Each agent spec defines a name, a system prompt, and the set of tools
            (retriever, MCP servers, local file search, etc.) that agent can use.
        """),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "archi_get_agent_spec",
        "description": textwrap.dedent("""\
            Retrieve the full agent spec markdown for a named archi agent.

            Use archi_list_agents first to discover available agent names, then call
            this tool to inspect the exact tools and prompt configured for that agent.
        """),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "The agent name returned by archi_list_agents.",
                },
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "archi_health",
        "description": (
            "Check whether the archi deployment is reachable and its database is healthy."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]

# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _ok(result: Any, rpc_id: Any) -> Dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _err(code: int, message: str, rpc_id: Any) -> Dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def _text(text: str) -> Dict:
    """Wrap a string as an MCP tool result."""
    return {"content": [{"type": "text", "text": str(text)}]}


_METADATA_ALIAS_MAP = {
    "resource_type": "source_type",
    "resource_id": "ticket_id",
}

_METADATA_FILTER_KEYS = [
    "path",
    "file_path",
    "display_name",
    "source_type",
    "url",
    "ticket_id",
    "suffix",
    "size_bytes",
    "original_path",
    "base_path",
    "relative_path",
    "created_at",
    "modified_at",
    "file_modified_at",
    "ingested_at",
]


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _truncate_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _parse_metadata_query(query: str) -> tuple[Dict[str, str] | list[Dict[str, str]], str]:
    filter_groups: list[Dict[str, str]] = []
    current_group: Dict[str, str] = {}
    free_tokens: list[str] = []

    try:
        tokens = shlex.split(query)
    except ValueError as exc:
        # Fall back to whitespace tokenization for malformed quoted input.
        logger.warning("Invalid metadata query syntax; using fallback tokenization: %s", exc)
        tokens = query.split()

    for token in tokens:
        if token.upper() == "OR":
            if current_group:
                filter_groups.append(current_group)
                current_group = {}
            continue
        if ":" in token:
            key, value = token.split(":", 1)
            key = _METADATA_ALIAS_MAP.get(key.strip(), key.strip())
            value = value.strip()
            if key and value:
                current_group[key] = value
                continue
        free_tokens.append(token)

    if current_group:
        filter_groups.append(current_group)

    if not filter_groups:
        filters: Dict[str, str] | list[Dict[str, str]] = {}
    elif len(filter_groups) == 1:
        filters = filter_groups[0]
    else:
        filters = filter_groups

    return filters, " ".join(free_tokens)


def _compile_query_pattern(query: str, *, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = query if regex else re.escape(query)
    return re.compile(pattern, flags)


def _grep_text_lines(
    text: str,
    pattern: re.Pattern[str],
    *,
    before: int = 0,
    after: int = 0,
    max_matches: int = 3,
) -> list[Dict[str, Any]]:
    if max_matches <= 0:
        return []
    lines = text.splitlines()
    matches: list[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not pattern.search(line):
            continue
        matches.append(
            {
                "line": idx + 1,
                "text": line,
                "before": lines[max(0, idx - before):idx] if before else [],
                "after": lines[idx + 1: idx + 1 + after] if after else [],
            }
        )
        if len(matches) >= max_matches:
            break
    return matches


def _document_display_name(doc: Dict[str, Any]) -> str:
    return (
        doc.get("display_name")
        or doc.get("filename")
        or doc.get("url")
        or doc.get("hash")
        or doc.get("id")
        or "unknown"
    )


def _parse_agent_frontmatter(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "---":
        return None

    idx += 1
    frontmatter_lines: list[str] = []
    while idx < len(lines):
        if lines[idx].strip() == "---":
            idx += 1
            break
        frontmatter_lines.append(lines[idx])
        idx += 1
    else:
        return None

    try:
        frontmatter = yaml.safe_load("\n".join(frontmatter_lines)) or {}
    except Exception:
        return None

    if not isinstance(frontmatter, dict):
        return None

    name = frontmatter.get("name")
    tools = frontmatter.get("tools")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(tools, list) or not all(isinstance(tool, str) and tool.strip() for tool in tools):
        return None

    return {
        "name": name.strip(),
        "tools": [tool.strip() for tool in tools],
        "path": path,
        "content": text,
    }


def _list_agent_specs(agents_dir: Path) -> list[Dict[str, Any]]:
    if not agents_dir.exists() or not agents_dir.is_dir():
        return []

    specs: list[Dict[str, Any]] = []
    for path in sorted(agents_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        spec = _parse_agent_frontmatter(path)
        if spec is not None:
            specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# Tool handlers (run inside the Flask process – no HTTP round-trip)
# ---------------------------------------------------------------------------


def _call_tool(
    name: str,
    arguments: Dict[str, Any],
    wrapper,
    user_id: Optional[str] = None,
    notify=None,
) -> Dict:
    """Dispatch a tools/call request to the appropriate archi internals.

    ``notify`` is an optional callable(message, progress, total) that sends a
    ``notifications/progress`` event back to the MCP client over the SSE stream.
    It is only provided when the client included ``_meta.progressToken`` in the
    tools/call request.
    """
    try:
        if name == "archi_query":
            return _tool_query(arguments, wrapper, user_id, notify=notify)
        elif name == "archi_list_documents":
            return _tool_list_documents(arguments, wrapper)
        elif name == "archi_get_document_content":
            return _tool_get_document_content(arguments, wrapper)
        elif name == "archi_search_document_metadata":
            return _tool_search_document_metadata(arguments, wrapper)
        elif name == "archi_list_metadata_schema":
            return _tool_list_metadata_schema(wrapper)
        elif name == "archi_search_document_content":
            return _tool_search_document_content(arguments, wrapper)
        elif name == "archi_get_document_chunks":
            return _tool_get_document_chunks(arguments, wrapper)
        elif name == "archi_get_data_stats":
            return _tool_get_data_stats(arguments, wrapper)
        elif name == "archi_get_deployment_info":
            return _tool_deployment_info(wrapper)
        elif name == "archi_list_agents":
            return _tool_list_agents(wrapper)
        elif name == "archi_get_agent_spec":
            return _tool_get_agent_spec(arguments, wrapper)
        elif name == "archi_health":
            return _text("status: OK\ndatabase: OK")
        else:
            return _text(f"ERROR: Unknown tool '{name}'.")
    except Exception as exc:
        logger.exception("MCP tool %s raised an exception", name)
        return _text(f"ERROR: {exc}")


def _tool_query(
    arguments: Dict[str, Any],
    wrapper,
    user_id: Optional[str] = None,
    notify=None,
) -> Dict:
    question = (arguments.get("question") or "").strip()
    if not question:
        return _text("ERROR: 'question' is required.")

    conversation_id = arguments.get("conversation_id")
    provider = arguments.get("provider") or None
    model = arguments.get("model") or None
    config_name = arguments.get("config_name") or None
    default_timeout_ms = 30000
    try:
        chat_cfg = (wrapper.config or {}).get("services", {}).get("chat_app", {})
        default_timeout_ms = int(float(chat_cfg.get("client_timeout_seconds", 30)) * 1000)
    except Exception:
        pass
    # client_timeout is in milliseconds (matching UI convention); convert to seconds
    client_timeout_ms = arguments.get("client_timeout", default_timeout_ms)
    try:
        client_timeout = max(float(client_timeout_ms) / 1000.0, 1.0)
    except (TypeError, ValueError):
        client_timeout = max(float(default_timeout_ms) / 1000.0, 1.0)
    client_id = f"mcp-sse-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    # Chunk events carry accumulated text (not deltas) — keep only the last one.
    answer: str = ""
    new_conv_id = None

    for event in wrapper.chat.stream(
        [["User", question]],
        conversation_id,
        client_id,
        False,           # is_refresh
        now,             # server_received_msg_ts
        now.timestamp(), # client_sent_msg_ts
        client_timeout,  # client_timeout (seconds, converted from ms)
        config_name,     # config_name (e.g. 'comp_ops', or None for active config)
        provider=provider,
        model=model,
        user_id=user_id,
    ):
        etype = event.get("type", "")

        if etype == "error":
            return _text(f"ERROR: {event.get('message', 'unknown error')}")

        elif etype == "thinking_start":
            if notify:
                notify("Thinking…")

        elif etype == "thinking_end":
            if notify:
                thinking = event.get("thinking_content", "")
                if thinking:
                    preview = thinking[:120].replace("\n", " ")
                    notify(f"Thought: {preview}{'…' if len(thinking) > 120 else ''}")

        elif etype == "tool_start":
            if notify:
                tool_name = event.get("tool_name", "tool")
                tool_args = event.get("tool_args") or {}
                if tool_args:
                    args_preview = ", ".join(
                        f"{k}={str(v)[:40]}" for k, v in (tool_args if isinstance(tool_args, dict) else {}).items()
                    )
                    notify(f"Calling {tool_name}({args_preview})")
                else:
                    notify(f"Calling {tool_name}()")

        elif etype == "tool_output":
            if notify:
                notify(f"Got result from {event.get('tool_name', 'tool')}")

        elif etype == "chunk":
            content = event.get("content", "")
            if content:
                answer = content
                if notify:
                    notify("Generating answer…")

        elif etype == "final":
            conv_id = event.get("conversation_id")
            if conv_id is not None:
                new_conv_id = conv_id
            response = event.get("response")
            final_answer = getattr(response, "answer", None) if response is not None else None
            if final_answer:
                answer = final_answer
    parts = [answer]
    if new_conv_id is not None:
        parts.append(
            f"\n\n---\n_conversation_id: {new_conv_id} "
            "(pass this to archi_query to continue the conversation)_"
        )
    return _text("".join(parts))


def _tool_list_documents(arguments: Dict[str, Any], wrapper) -> Dict:
    limit = _clamp_int(arguments.get("limit", 50), default=50, minimum=1, maximum=500)
    offset = _clamp_int(arguments.get("offset", 0), default=0, minimum=0, maximum=1_000_000)
    conversation_id = arguments.get("conversation_id")
    search: Optional[str] = arguments.get("search") or None
    source_type: Optional[str] = arguments.get("source_type") or None
    enabled_filter = (arguments.get("enabled") or "").strip().lower() or None
    if enabled_filter not in {None, "enabled", "disabled", "all"}:
        return _text("ERROR: 'enabled' must be one of: enabled, disabled, all.")

    result = wrapper.chat.data_viewer.list_documents(
        conversation_id=conversation_id,
        source_type=source_type,
        search=search,
        enabled_filter=None if enabled_filter in {None, "all"} else enabled_filter,
        limit=limit,
        offset=offset,
    )
    docs = result.get("documents", result.get("items", []))
    total = result.get("total", len(docs))

    lines = [f"Found {total} document(s) (offset={offset}, limit={limit}):\n"]
    for doc in docs:
        display = _document_display_name(doc)
        source = doc.get("source_type", doc.get("type", ""))
        doc_hash = doc.get("hash", doc.get("id", ""))
        status = doc.get("ingestion_status", "unknown")
        enabled = doc.get("enabled")
        extra: list[str] = [source] if source else []
        if status:
            extra.append(f"status={status}")
        if enabled is not None:
            extra.append(f"enabled={'yes' if enabled else 'no'}")
        lines.append(f"  • {display}  [{' | '.join(extra)}]  hash={doc_hash}")

    lines.append(
        "\nUse archi_get_document_content(document_hash=<hash>) to read a document."
    )
    return _text("\n".join(lines))


def _tool_get_document_content(arguments: Dict[str, Any], wrapper) -> Dict:
    doc_hash = (arguments.get("document_hash") or "").strip()
    if not doc_hash:
        return _text("ERROR: 'document_hash' is required.")

    max_size = _clamp_int(arguments.get("max_size", 100000), default=100000, minimum=1000, maximum=1_000_000)
    result = wrapper.chat.data_viewer.get_document_content(doc_hash, max_size)
    if result is None:
        return _text(f"ERROR: Document not found: {doc_hash}")

    content = result.get("content", result.get("text", json.dumps(result, indent=2)))
    if result.get("truncated"):
        content = f"{content}\n\n---\n(truncated at {max_size} bytes/chars)"
    return _text(content)


def _tool_search_document_metadata(arguments: Dict[str, Any], wrapper) -> Dict:
    query = (arguments.get("query") or "").strip()
    if not query:
        return _text("ERROR: 'query' is required.")

    limit = _clamp_int(arguments.get("limit", 10), default=10, minimum=1, maximum=100)
    filters, free_query = _parse_metadata_query(query)
    catalog = wrapper.chat.data_viewer.catalog
    results = catalog.search_metadata(
        free_query,
        limit=limit,
        filters=filters or None,
    )

    if not results:
        return _text("No documents matched that metadata query.")

    lines = [f"Found {len(results)} metadata match(es):\n"]
    for item in results:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        path = item.get("path")
        display = (
            metadata.get("display_name")
            or metadata.get("file_name")
            or metadata.get("title")
            or metadata.get("url")
            or item.get("hash")
            or "unknown"
        )
        lines.append(f"  • {display}  hash={item.get('hash')}")
        lines.append(f"    Path: {path}")
        if metadata.get("source_type"):
            lines.append(f"    Source: {metadata.get('source_type')}")
        if metadata.get("ticket_id"):
            lines.append(f"    Ticket: {metadata.get('ticket_id')}")
        if metadata.get("relative_path"):
            lines.append(f"    Relative path: {metadata.get('relative_path')}")
        if metadata.get("url"):
            lines.append(f"    URL: {_truncate_text(metadata.get('url'), max_chars=180)}")

    lines.append(
        "\nUse archi_get_document_content(document_hash=<hash>) to inspect a result."
    )
    return _text("\n".join(lines))


def _tool_list_metadata_schema(wrapper) -> Dict:
    catalog = wrapper.chat.data_viewer.catalog
    distinct = catalog.get_distinct_metadata(["source_type", "suffix"])
    keys = sorted(_METADATA_FILTER_KEYS)
    source_types = distinct.get("source_type", [])
    suffixes = distinct.get("suffix", [])

    lines = [
        "Supported metadata keys: " + (", ".join(keys) or "none"),
        "source_type values: " + (", ".join(source_types) or "none"),
        "suffix values: " + (", ".join(suffixes) or "none"),
        "",
        "Examples:",
        "  source_type:git relative_path:docs/README.md",
        "  ticket_id:CMSPROD-1234",
        "  source_type:web OR source_type:git",
        "  url:github.com/org/repo",
        "",
        "Legacy aliases: resource_type -> source_type, resource_id -> ticket_id",
    ]
    return _text("\n".join(lines))


def _tool_search_document_content(arguments: Dict[str, Any], wrapper) -> Dict:
    from src.data_manager.vectorstore.loader_utils import load_text_from_path

    query = (arguments.get("query") or "").strip()
    if not query:
        return _text("ERROR: 'query' is required.")

    regex = bool(arguments.get("regex", False))
    case_sensitive = bool(arguments.get("case_sensitive", False))
    before = _clamp_int(arguments.get("before", 0), default=0, minimum=0, maximum=20)
    after = _clamp_int(arguments.get("after", 0), default=0, minimum=0, maximum=20)
    max_matches_per_document = _clamp_int(
        arguments.get("max_matches_per_document", 3),
        default=3,
        minimum=1,
        maximum=20,
    )
    limit = _clamp_int(arguments.get("limit", 5), default=5, minimum=1, maximum=20)
    metadata_query = (arguments.get("metadata_query") or "").strip()

    try:
        pattern = _compile_query_pattern(query, regex=regex, case_sensitive=case_sensitive)
    except re.error as exc:
        return _text(f"ERROR: invalid regex: {exc}")

    catalog = wrapper.chat.data_viewer.catalog
    candidate_metadata: Dict[str, Dict[str, Any]] = {}
    if metadata_query:
        filters, free_query = _parse_metadata_query(metadata_query)
        candidates = catalog.search_metadata(
            free_query,
            limit=None,
            filters=filters or None,
        )
        iterable = []
        for item in candidates:
            resource_hash = item.get("hash")
            if not resource_hash:
                continue
            path = catalog.get_filepath_for_hash(resource_hash)
            if path:
                iterable.append((resource_hash, path))
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                candidate_metadata[resource_hash] = metadata
    else:
        iterable = list(catalog.iter_files())

    hits: list[Dict[str, Any]] = []
    for resource_hash, path in iterable:
        metadata = candidate_metadata.get(resource_hash) or catalog.get_metadata_for_hash(resource_hash) or {}
        text = load_text_from_path(path) or ""
        if not text:
            continue
        matches = _grep_text_lines(
            text,
            pattern,
            before=before,
            after=after,
            max_matches=max_matches_per_document,
        )
        if not matches:
            continue
        hits.append(
            {
                "hash": resource_hash,
                "path": path,
                "metadata": metadata,
                "matches": matches,
            }
        )
        if len(hits) >= limit:
            break

    if not hits:
        return _text("No indexed document contents matched that search query.")

    lines = [f"Found {len(hits)} matching document(s):\n"]
    for item in hits:
        metadata = item["metadata"] if isinstance(item["metadata"], dict) else {}
        display = (
            metadata.get("display_name")
            or metadata.get("file_name")
            or metadata.get("title")
            or str(item["path"])
        )
        source = metadata.get("source_type") or "unknown"
        lines.append(f"  • {display}  [{source}]  hash={item['hash']}")
        lines.append(f"    Path: {item['path']}")
        for match in item["matches"]:
            before_lines = match.get("before") or []
            after_lines = match.get("after") or []
            for ctx in before_lines:
                lines.append(f"    B: {_truncate_text(ctx, max_chars=240)}")
            lines.append(f"    L{match.get('line', '?')}: {_truncate_text(match.get('text'), max_chars=240)}")
            for ctx in after_lines:
                lines.append(f"    A: {_truncate_text(ctx, max_chars=240)}")

    return _text("\n".join(lines))


def _tool_get_document_chunks(arguments: Dict[str, Any], wrapper) -> Dict:
    doc_hash = (arguments.get("document_hash") or "").strip()
    if not doc_hash:
        return _text("ERROR: 'document_hash' is required.")

    offset = _clamp_int(arguments.get("offset", 0), default=0, minimum=0, maximum=1_000_000)
    limit = _clamp_int(arguments.get("limit", 20), default=20, minimum=1, maximum=100)
    max_chars_per_chunk = _clamp_int(
        arguments.get("max_chars_per_chunk", 600),
        default=600,
        minimum=80,
        maximum=5000,
    )

    chunks = wrapper.chat.data_viewer.get_document_chunks(doc_hash)
    if not chunks:
        return _text(f"No stored chunks found for document: {doc_hash}")

    selected = chunks[offset: offset + limit]
    lines = [
        f"Document {doc_hash} has {len(chunks)} chunk(s); showing {len(selected)} from offset {offset}:\n"
    ]
    for chunk in selected:
        start_char = chunk.get("start_char")
        end_char = chunk.get("end_char")
        lines.append(
            f"  • chunk {chunk.get('index')}  chars={start_char}-{end_char}\n"
            f"    {_truncate_text(chunk.get('text'), max_chars=max_chars_per_chunk)}"
        )
    return _text("\n".join(lines))


def _tool_get_data_stats(arguments: Dict[str, Any], wrapper) -> Dict:
    conversation_id = arguments.get("conversation_id")
    stats = wrapper.chat.data_viewer.get_stats(conversation_id)
    by_source_type = stats.get("by_source_type") or {}
    status_counts = stats.get("status_counts") or {}

    lines = [
        "Corpus statistics:",
        f"  Total documents:      {stats.get('total_documents', 0)}",
        f"  Total chunks:         {stats.get('total_chunks', 0)}",
        f"  Enabled documents:    {stats.get('enabled_documents', 0)}",
        f"  Disabled documents:   {stats.get('disabled_documents', 0)}",
        f"  Total size (bytes):   {stats.get('total_size_bytes', 0)}",
        f"  Last sync:            {stats.get('last_sync') or 'n/a'}",
        "",
        "Ingestion status:",
        f"  pending={status_counts.get('pending', 0)}",
        f"  embedding={status_counts.get('embedding', 0)}",
        f"  embedded={status_counts.get('embedded', 0)}",
        f"  failed={status_counts.get('failed', 0)}",
    ]

    if by_source_type:
        lines.append("")
        lines.append("By source type:")
        for source_type, counts in sorted(by_source_type.items()):
            total = counts.get("total", 0) if isinstance(counts, dict) else counts
            enabled = counts.get("enabled", total) if isinstance(counts, dict) else total
            lines.append(f"  {source_type}: total={total}, enabled={enabled}")

    return _text("\n".join(lines))


def _tool_deployment_info(wrapper) -> Dict:
    from src.utils.config_access import get_dynamic_config, get_full_config, get_static_config

    config = get_full_config() or {}
    static = get_static_config()
    services = config.get("services", {})
    chat_cfg = services.get("chat_app", {})
    dm_cfg = services.get("data_manager", {})
    mcp_servers = config.get("mcp_servers", {}) or {}

    try:
        dynamic = get_dynamic_config()
    except Exception:
        dynamic = None

    lines = [
        f"# archi Deployment: {config.get('name', 'unknown')}",
        "",
        "## Active configuration",
        f"  Pipeline:              {chat_cfg.get('pipeline', 'n/a')}",
        f"  Agent class:           {chat_cfg.get('agent_class', chat_cfg.get('pipeline', 'n/a'))}",
    ]
    if dynamic:
        lines += [
            f"  Active agent:          {dynamic.active_agent_name or getattr(getattr(wrapper.chat, 'agent_spec', None), 'name', 'n/a')}",
            f"  Model:                 {dynamic.active_model}",
            f"  Temperature:           {dynamic.temperature}",
            f"  Max tokens:            {dynamic.max_tokens}",
            f"  Docs retrieved (k):    {dynamic.num_documents_to_retrieve}",
            f"  Hybrid search:         {dynamic.use_hybrid_search}",
            f"    BM25 weight:         {dynamic.bm25_weight}",
            f"    Semantic weight:     {dynamic.semantic_weight}",
        ]
    else:
        lines += [
            f"  Active agent:          {getattr(getattr(wrapper.chat, 'agent_spec', None), 'name', 'n/a')}",
        ]

    embedding_cfg = dm_cfg.get("embedding", {})
    lines += [
        "",
        "## Embedding",
        f"  Model:                 {embedding_cfg.get('model', 'n/a')}",
        f"  Chunk size:            {embedding_cfg.get('chunk_size', 'n/a')}",
        f"  Chunk overlap:         {embedding_cfg.get('chunk_overlap', 'n/a')}",
        "",
        "## Runtime",
        f"  Available providers:   {', '.join(static.available_providers or []) or 'n/a'}",
        f"  Available pipelines:   {', '.join(static.available_pipelines or []) or 'n/a'}",
        f"  MCP servers:           {', '.join(sorted(mcp_servers.keys())) or 'none'}",
        f"  MCP endpoint enabled:  {services.get('mcp_server', {}).get('enabled', False)}",
    ]
    return _text("\n".join(lines))


def _tool_list_agents(wrapper) -> Dict:
    agents_dir = wrapper._get_agents_dir()
    lines = []
    for spec in _list_agent_specs(agents_dir):
        tools_str = ", ".join(spec.get("tools", []) or []) or "none"
        path = spec["path"]
        lines.append(f"  • {spec['name']}  ({path.name})")
        lines.append(f"    Tools: {tools_str}")

    if not lines:
        return _text("No agent specs found in this deployment.")
    return _text("Available agents:\n" + "\n".join(lines))


def _tool_get_agent_spec(arguments: Dict[str, Any], wrapper) -> Dict:
    agent_name = (arguments.get("agent_name") or "").strip()
    if not agent_name:
        return _text("ERROR: 'agent_name' is required.")

    agents_dir = wrapper._get_agents_dir()
    for spec in _list_agent_specs(agents_dir):
        if spec["name"] == agent_name:
            return _text(spec["content"])
    return _text(f"ERROR: Agent not found: {agent_name}")


# ---------------------------------------------------------------------------
# JSON-RPC dispatcher
# ---------------------------------------------------------------------------


def _dispatch(body: Dict, session_queue: queue.Queue, wrapper, user_id: Optional[str] = None) -> None:
    """Process one incoming JSON-RPC message and enqueue the response if needed."""
    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params") or {}

    # Notifications have no id – no response expected.
    if rpc_id is None:
        return

    if method == "initialize":
        response = _ok(
            {
                "protocolVersion": _MCP_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": _SERVER_INFO,
            },
            rpc_id,
        )
    elif method == "tools/list":
        response = _ok({"tools": _TOOLS}, rpc_id)
    elif method == "tools/call":
        # Extract optional progress token from _meta so we can stream status
        # events back to the client while archi works.
        meta = params.get("_meta") or {}
        progress_token = meta.get("progressToken")

        logger.info(
            "tools/call %s – progressToken=%s",
            params.get("name", "?"),
            progress_token if progress_token is not None else "<none: non-streaming>",
        )

        notify_fn: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None
        if progress_token is not None:
            _progress_counter = [0]

            def _notify(
                message: str,
                progress: Optional[int] = None,
                total: Optional[int] = None,
            ) -> None:
                _progress_counter[0] += 1
                p = progress if progress is not None else _progress_counter[0]
                notification: Dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": progress_token,
                        "progress": p,
                        "message": message,
                    },
                }
                if total is not None:
                    notification["params"]["total"] = total
                session_queue.put(notification)

            notify_fn = _notify

        result = _call_tool(
            params.get("name", ""),
            params.get("arguments") or {},
            wrapper,
            user_id,
            notify=notify_fn,
        )
        response = _ok(result, rpc_id)
    elif method == "ping":
        response = _ok({}, rpc_id)
    else:
        response = _err(-32601, f"Method not found: {method}", rpc_id)

    session_queue.put(response)


def _dispatch_and_release(
    body: Dict,
    session_queue: queue.Queue,
    wrapper,
    user_id: Optional[str] = None,
) -> None:
    try:
        _dispatch(body, session_queue, wrapper, user_id)
    finally:
        _dispatch_slots.release()


# ---------------------------------------------------------------------------
# Blueprint factory
# ---------------------------------------------------------------------------


def register_mcp_sse(
    app,
    wrapper,
    pg_config: Optional[dict] = None,
    auth_enabled: bool = False,
    public_url: Optional[str] = None,
) -> None:
    """Register the MCP SSE endpoints on a Flask app.

    ``public_url``: externally reachable base URL (e.g. ``https://example.com``).
    When set, the ``endpoint`` SSE event uses it to build the absolute POST URL
    instead of inferring from request headers.
    """
    mcp = Blueprint("mcp_sse", __name__)

    def _auth_check():
        """Return (user_id, error_response) tuple.  error_response is None on success."""
        if not auth_enabled:
            return None, None
        token = _extract_bearer_token(request)
        if not token:
            resp = jsonify({
                "error": "unauthorized",
                "message": "MCP access requires a bearer token. "
                           "Visit /mcp/auth to generate one after logging in.",
                "login_url": "/mcp/auth",
            })
            resp.status_code = 401
            return None, resp
        user_id = _validate_mcp_token(token, pg_config)
        if not user_id:
            resp = jsonify({
                "error": "invalid_token",
                "message": "The bearer token is invalid or has expired. "
                           "Visit /mcp/auth to generate a new token.",
                "login_url": "/mcp/auth",
            })
            resp.status_code = 401
            return None, resp
        return user_id, None

    @mcp.route("/mcp/sse")
    def sse():
        """Open an SSE stream for one MCP client session."""
        user_id, err = _auth_check()
        if err is not None:
            return err

        session_id = uuid.uuid4().hex
        q: queue.Queue = queue.Queue()
        # Resolve base URL now — generators run outside request context.
        if public_url:
            _base = public_url.rstrip("/")
        else:
            fwd_proto = request.headers.get("X-Forwarded-Proto") or request.scheme
            fwd_host = request.headers.get("X-Forwarded-Host") or request.host
            _base = f"{fwd_proto}://{fwd_host}"
        post_url = f"{_base}/mcp/messages?session_id={session_id}"
        with _sessions_lock:
            _sessions[session_id] = {"queue": q, "user_id": user_id}
        logger.info("MCP SSE session %s opened (user=%s)", session_id, user_id)

        def generate():
            yield f"event: endpoint\ndata: {post_url}\n\n"
            try:
                while True:
                    try:
                        msg = q.get(timeout=_KEEPALIVE_TIMEOUT)
                        if msg is None:
                            break
                        yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                with _sessions_lock:
                    _sessions.pop(session_id, None)

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @mcp.route("/mcp/messages", methods=["POST"])
    def messages():
        """Receive a JSON-RPC message from an MCP client."""
        # Auth was verified when the SSE stream was opened; session_id is the proof.
        session_id = request.args.get("session_id", "")
        with _sessions_lock:
            session_entry = _sessions.get(session_id)
        if session_entry is None:
            logger.warning("MCP /mcp/messages: unknown session_id=%r (active sessions: %s)",
                           session_id, list(_sessions.keys()))
            return {"error": "unknown or expired session_id"}, 404

        q = session_entry["queue"]
        user_id = session_entry.get("user_id")

        body = request.get_json(silent=True)
        if not body:
            return {"error": "request body must be valid JSON"}, 400

        logger.info("MCP /mcp/messages session=%s method=%s id=%s",
                    session_id, body.get("method", "?"), body.get("id"))
        # Use a bounded dispatch pool to prevent thread storms under high parallel load.
        if not _dispatch_slots.acquire(blocking=False):
            logger.warning(
                "MCP dispatch overloaded: rejecting request method=%s id=%s",
                body.get("method", "?"),
                body.get("id"),
            )
            rpc_id = body.get("id")
            if rpc_id is not None:
                q.put(_err(-32001, "Server is busy. Please retry shortly.", rpc_id))
            return "", 202

        try:
            _dispatch_executor.submit(_dispatch_and_release, body, q, wrapper, user_id)
        except Exception:
            _dispatch_slots.release()
            raise
        return "", 202

    app.register_blueprint(mcp)
    if auth_enabled:
        logger.info("Registered MCP SSE endpoint at /mcp/sse (auth required – Bearer token)")
    else:
        logger.info("Registered MCP SSE endpoint at /mcp/sse (no auth)")
