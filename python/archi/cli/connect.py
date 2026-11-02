"""`archi connect open-webui|claude-code|codex|openclaw|mcp` (PACT r3-mcp-clients req.connect).

    archi connect claude-code [--instance DIR] [--print | --write [PATH]] [--show-token] [--embed-token]

Writes the config another agent needs to use the instance's MCP endpoint
(`http://127.0.0.1:<ports.mcp>/mcp`, streamable HTTP, bearer token
`OKG_CHAT_MCP_TOKEN` from the instance `.env`) and every external server the
instance's `mcp_servers:` block names (archi.yaml).

Each client's real format (checked 2026-10; see docs/docs/connecting-an-agent.md):

    claude-code  .mcp.json            {"mcpServers": {name: {"type": "http", "url", "headers"}}}
                                      (what `claude mcp add --transport http --header` writes)
    codex        ~/.codex/config.toml [mcp_servers.<name>] url / bearer_token_env_var / env_http_headers.
                                      Codex speaks streamable HTTP itself: no stdio bridge.
    openclaw     ~/.openclaw/openclaw.json   mcp.servers.<name> {url, transport: "streamable-http", headers}
                                      from OpenClaw's configuration docs; not run against a live OpenClaw
    mcp          generic {"mcpServers": ...} JSON for any client that reads that shape
    open-webui   nothing to write: `archi up` registers the server (okg chat sync); prints what it did

The token is never printed unless `--show-token`. Printed and written configs carry
`${OKG_CHAT_MCP_TOKEN}` (Claude Code, OpenClaw and most MCP clients expand it from the
environment; Codex names the variable). `--write --embed-token` writes the literal token
(mode 0600) instead, and only to a file git ignores. A client config that
already exists is never replaced: archi's entries are merged into it, and
running the command again changes nothing.

External servers' secrets are environment-variable names in archi.yaml; they are
written as references (`${NAME}`, or Codex's `env_http_headers`), never as values.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from archi.stack.instance import Instance, InstanceError, McpServer, read_env

CLIENTS = ("open-webui", "claude-code", "codex", "openclaw", "mcp")
TOKEN_ENV = "OKG_CHAT_MCP_TOKEN"
DEFAULT_PATHS = {
    "claude-code": "./.mcp.json",
    "codex": "~/.codex/config.toml",
    "openclaw": "~/.openclaw/openclaw.json",
    "mcp": "./archi-mcp.json",
}
OPENCLAW_NOTE = (
    "OpenClaw's format is taken from its documentation (mcp.servers, transport "
    "\"streamable-http\"); archi has not run it against a live OpenClaw. Check it with "
    "`openclaw mcp show`."
)


class ConnectError(Exception):
    """A message for the operator, printed as `error: ...`."""


@dataclass
class Entry:
    """One MCP server as a client config holds it. `headers` values are literal or `${VAR}` text."""

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    #: set for the instance server: the header value is `Bearer <token env var>`
    token_env: Optional[str] = None
    #: external servers: {header: env var name} (the variable holds the whole header value)
    env_headers: dict[str, str] = field(default_factory=dict)


def server_name(instance: Instance) -> str:
    return f"archi-{instance.name}"


def mcp_url(instance: Instance) -> str:
    return f"http://127.0.0.1:{instance.ports.mcp}/mcp"


def entries(instance: Instance, token: str, *, literal: bool) -> list[Entry]:
    """The instance server first, then the external ones. `literal` puts the token in."""
    bearer = f"Bearer {token}" if literal else f"Bearer ${{{TOKEN_ENV}}}"
    out = [Entry(server_name(instance), mcp_url(instance), {"Authorization": bearer}, token_env=TOKEN_ENV)]
    for ext in instance.mcp_servers:
        out.append(Entry(
            ext.name, ext.url, {h: f"${{{var}}}" for h, var in ext.headers.items()},
            env_headers=dict(ext.headers),
        ))
    return out


# --- renderers ------------------------------------------------------------------------------


def json_servers(entry_list: list[Entry], *, openclaw: bool) -> dict[str, dict[str, Any]]:
    servers: dict[str, dict[str, Any]] = {}
    for e in entry_list:
        if openclaw:
            body: dict[str, Any] = {"url": e.url, "transport": "streamable-http"}
        else:
            body = {"type": "http", "url": e.url}
        if e.headers:
            body["headers"] = dict(e.headers)
        servers[e.name] = body
    return servers


def json_document(client: str, entry_list: list[Entry]) -> dict[str, Any]:
    if client == "openclaw":
        return {"mcp": {"servers": json_servers(entry_list, openclaw=True)}}
    return {"mcpServers": json_servers(entry_list, openclaw=False)}


def _q(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def toml_block(entry: Entry) -> str:
    """One `[mcp_servers.<name>]` table. Codex reads secrets from the environment by name."""
    lines = [f"[mcp_servers.{entry.name}]", f"url = {_q(entry.url)}"]
    if entry.token_env:
        lines.append(f"bearer_token_env_var = {_q(entry.token_env)}")
    if entry.env_headers:
        pairs = ", ".join(f"{_q(h)} = {_q(v)}" for h, v in entry.env_headers.items())
        lines.append(f"env_http_headers = {{ {pairs} }}")
    return "\n".join(lines) + "\n"


def toml_document(entry_list: list[Entry]) -> str:
    return "\n".join(toml_block(e) for e in entry_list)


def claude_add_commands(entry_list: list[Entry]) -> list[str]:
    cmds = []
    for e in entry_list:
        parts = ["claude mcp add --transport http", e.name, e.url]
        parts += [f'--header "{h}: {v}"' for h, v in e.headers.items()]
        cmds.append(" ".join(parts))
    return cmds


# --- merge ----------------------------------------------------------------------------------


def merge_json(existing: Optional[str], client: str, entry_list: list[Entry], source: str) -> tuple[str, str]:
    """Merge archi's servers into a JSON client config. Returns (text, "added"|"updated"|"unchanged")."""
    wanted = json_servers(entry_list, openclaw=client == "openclaw")
    if existing is None or not existing.strip():
        doc: Any = {}
    else:
        try:
            doc = json.loads(existing)
        except json.JSONDecodeError as exc:
            raise ConnectError(
                f"{source} is not strict JSON ({exc}); archi will not rewrite it. Merge this by hand "
                f"(`archi connect {client} --print`)"
                + (f". {OPENCLAW_NOTE}" if client == "openclaw" else "")
            ) from exc
        if not isinstance(doc, dict):
            raise ConnectError(f"{source}: top level is not an object")
    if client == "openclaw":
        mcp = doc.setdefault("mcp", {})
        if not isinstance(mcp, dict):
            raise ConnectError(f"{source}: `mcp` is not an object")
        servers = mcp.setdefault("servers", {})
    else:
        servers = doc.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ConnectError(f"{source}: the MCP servers entry is not an object")
    before = {name: servers.get(name) for name in wanted}
    existed = any(v is not None for v in before.values())
    servers.update(wanted)
    status = "unchanged" if before == wanted else ("updated" if existed else "added")
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n", status


#: What archi writes in a Codex `[mcp_servers.<name>]` table. Any other key there is the user's
#: (startup_timeout_sec, tool_timeout_sec, enabled, ...) and is kept.
TOML_OWNED_KEYS = ("url", "bearer_token_env_var", "env_http_headers")
_TABLE_HEADER = re.compile(r"^\s*\[(?!\[)\s*(.*?)\s*\]\s*(#.*)?$")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else _q(key)


def toml_value(value: Any) -> str:
    """A TOML value for what tomllib can read back (strings, bools, numbers, arrays, tables)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _q(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {toml_value(v)}" for k, v in value.items()) + " }" if value else "{}"
    raise ConnectError(f"cannot write a {type(value).__name__} value into a TOML config")


def _table_lines(name: str, table: dict[str, Any]) -> str:
    lines = [f"[mcp_servers.{_toml_key(name)}]"]
    lines += [f"{_toml_key(k)} = {toml_value(v)}" for k, v in table.items()]
    return "\n".join(lines) + "\n"


def _header_names(header: str) -> Optional[list[str]]:
    """The dotted key path in a `[a.b."c"]` header, or None when it is not simple enough to read."""
    parts: list[str] = []
    rest = header.strip()
    while rest:
        if rest[0] == '"':
            m = re.match(r'"((?:[^"\\]|\\.)*)"\s*', rest)
            if not m:
                return None
            parts.append(json.loads(f'"{m.group(1)}"'))
        else:
            m = re.match(r"([A-Za-z0-9_-]+)\s*", rest)
            if not m:
                return None
            parts.append(m.group(1))
        rest = rest[m.end():]
        if rest:
            if rest[0] != ".":
                return None
            rest = rest[1:].lstrip()
    return parts


def merge_toml(existing: Optional[str], entry_list: list[Entry], source: str) -> tuple[str, str]:
    """Merge archi's servers into a Codex config.

    The file is read with tomllib. Each archi table keeps the user's own keys and only archi's keys
    (TOML_OWNED_KEYS) are set; a stale owned key (a bearer token variable on a server that no longer
    uses one) is dropped. Everything outside the table's own lines (comments, other tables, the
    table's sub-tables) is kept byte for byte. The result is parsed again and must equal the old
    document with only those tables changed, or nothing is written.
    """
    text = existing or ""
    try:
        old = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConnectError(f"{source} is not valid TOML ({exc}); fix it first, archi will not rewrite it") from exc
    servers = old.get("mcp_servers", {})
    if not isinstance(servers, dict):
        raise ConnectError(f"{source}: `mcp_servers` is not a table")
    expected = {k: v for k, v in old.items()}
    expected["mcp_servers"] = dict(servers)
    lines = text.splitlines(keepends=True)
    # keys the user wrote as their own sub-table header, [mcp_servers.<name>.<key>]: those lines stay
    subtables: dict[str, set[str]] = {}
    for line in lines:
        m = _TABLE_HEADER.match(line)
        names = _header_names(m.group(1)) if m else None
        if names and len(names) >= 3 and names[0] == "mcp_servers":
            subtables.setdefault(names[1], set()).add(names[2])
    merged: dict[str, dict[str, Any]] = {}
    for e in entry_list:
        current = servers.get(e.name, {})
        if not isinstance(current, dict):
            raise ConnectError(f"{source}: mcp_servers.{e.name} is not a table")
        owned: dict[str, Any] = {"url": e.url}
        if e.token_env:
            owned["bearer_token_env_var"] = e.token_env
        if e.env_headers:
            owned["env_http_headers"] = dict(e.env_headers)
        mine = {k: v for k, v in current.items() if k not in TOML_OWNED_KEYS}
        # archi's keys first, then the user's own; sub-table keys are not rewritten
        merged[e.name] = {**owned, **{k: v for k, v in mine.items() if k not in subtables.get(e.name, set())}}
        expected["mcp_servers"][e.name] = {**owned, **mine}

    out: list[str] = []
    done: set[str] = set()
    i = 0
    while i < len(lines):
        m = _TABLE_HEADER.match(lines[i])
        names = _header_names(m.group(1)) if m else None
        if names and len(names) == 2 and names[0] == "mcp_servers" and names[1] in merged:
            j = i + 1
            while j < len(lines) and not _TABLE_HEADER.match(lines[j]) and not lines[j].lstrip().startswith("[["):
                j += 1
            block = _table_lines(names[1], merged[names[1]])
            if j < len(lines):
                block += "\n"
            out.append(block)
            done.add(names[1])
            i = j
            continue
        out.append(lines[i])
        i += 1
    body = "".join(out)
    for e in entry_list:
        if e.name not in done:
            body = (body.rstrip("\n") + "\n\n" if body.strip() else "") + _table_lines(e.name, merged[e.name])
    try:
        new_doc = tomllib.loads(body)
    except tomllib.TOMLDecodeError as exc:
        raise ConnectError(f"merging into {source} would produce invalid TOML ({exc}); nothing written") from exc
    if new_doc != expected:
        raise ConnectError(f"merging into {source} would change more than archi's own tables; nothing written. "
                           f"Merge by hand (`archi connect codex --print`)")
    if body == text:
        status = "unchanged"
    else:
        status = "updated" if any(e.name in servers for e in entry_list) else "added"
    return body, status


# --- files ----------------------------------------------------------------------------------


def _git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)


def refuse_unignored_secret_file(path: Path) -> None:
    """A file that will hold the token may sit inside a git work tree only if git ignores it."""
    path = path.expanduser().resolve()
    directory = next((p for p in [path.parent, *path.parent.parents] if p.is_dir()), None)
    if directory is None or _git(directory, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return  # not in a repository
    if _git(directory, "check-ignore", "-q", "--no-index", str(path)).returncode != 0:
        raise ConnectError(
            f"{path} is inside a git repository and is not git-ignored; refusing to write the token "
            f"into it. Add it to .gitignore, or drop --embed-token to write a ${{{TOKEN_ENV}}} reference"
        )


def write_file(path: Path, text: str, *, secret: bool) -> list[str]:
    """Write atomically. A symlinked config is written through to its target. A new file is 0600; an
    existing one keeps its mode unless it now holds a secret and is readable by others, which is
    tightened to 0600. Returns notes for the operator."""
    notes: list[str] = []
    path = Path(os.path.realpath(path.expanduser()))
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o600
    if path.exists():
        mode = path.stat().st_mode & 0o777
        if secret and mode & 0o077:
            notes.append(f"{path} was mode {mode:o}; set to 600 because it now holds the token")
            mode = 0o600
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return notes


# --- commands -------------------------------------------------------------------------------


def instance_token(instance: Instance) -> str:
    secrets = read_env(instance.env_path)
    if not secrets.get(TOKEN_ENV):
        raise ConnectError(f"{instance.env_path} has no {TOKEN_ENV}; `archi init` did not finish")
    return secrets[TOKEN_ENV]


def print_config(client: str, entry_list: list[Entry], token: str, show_token: bool) -> str:
    if client == "codex":
        out = toml_document(entry_list)
        hint = f"# Codex reads the token from the environment: export {TOKEN_ENV}=" + (
            token if show_token else "<token from the instance .env>")
        for e in entry_list[1:]:
            for var in e.env_headers.values():
                hint += f"\n# and {var} for {e.name}"
        return out + "\n" + hint + "\n"
    doc = json.dumps(json_document(client, entry_list), indent=2, ensure_ascii=False) + "\n"
    if client == "claude-code":
        doc += "\n# or, with the claude CLI:\n" + "\n".join(f"# {c}" for c in claude_add_commands(entry_list)) + "\n"
    if client == "openclaw":
        doc += f"\n# {OPENCLAW_NOTE}\n"
    return doc


def open_webui_report(instance: Instance) -> str:
    from archi.stack import compose as stack_compose

    if not (instance.deployment_dir / "deployment.yaml").is_file():
        raise ConnectError(f"{instance.deployment_dir}/deployment.yaml does not exist; run `archi init` first")
    port = stack_compose.mcp_port(instance.deployment_dir)
    lines = [
        "Open WebUI needs no config file: `archi up` registers the MCP server through `okg chat sync`.",
        f"  site            http://127.0.0.1:{instance.ports.webui}",
        f"  registered URL  http://mcp:{port}/mcp   (the site reaches the server by its service name)",
        f"  from the host   {mcp_url(instance)}",
        f"  auth            bearer; the token is {TOKEN_ENV} in {instance.env_path}",
        "  where           the preset named after the site in the model picker has the graph tools bound",
        "If `archi up` has not run (or ran with --no-sync), run it; it syncs again each time.",
    ]
    if instance.mcp_servers:
        lines.append("External servers in archi.yaml are not synced to Open WebUI. Add them by hand under "
                     "Settings > Admin > External Tools:")
        for ext in instance.mcp_servers:
            names = ", ".join(f"{h} <- ${v}" for h, v in ext.headers.items()) or "no headers"
            lines.append(f"  {ext.name}  {ext.url}  ({names})")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace, out=None) -> int:
    out = out or sys.stdout
    instance = Instance.load(args.instance)
    client = args.client
    if client == "open-webui":
        if args.write is not None:
            raise ConnectError("open-webui has no config file to write; `archi up` registers the server")
        out.write(open_webui_report(instance))
        return 0
    token = instance_token(instance)
    writing = args.write is not None
    # Every config carries a ${OKG_CHAT_MCP_TOKEN} reference (Claude Code, OpenClaw and most MCP
    # clients expand it from the environment). The literal token goes in only when asked:
    # --show-token to print it, --embed-token to write it (and then only to a git-ignored file).
    # Codex never holds the token: it names the environment variable.
    literal = bool(args.embed_token) if writing else bool(args.show_token)
    entry_list = entries(instance, token, literal=literal)
    if not writing:
        out.write(print_config(client, entry_list, token, args.show_token))
        return 0
    path = Path(args.write if args.write != "" else DEFAULT_PATHS[client]).expanduser()
    if literal and client != "codex":
        refuse_unignored_secret_file(path)
    existing = path.read_text(encoding="utf-8") if path.is_file() else None
    if client == "codex":
        text, status = merge_toml(existing, entry_list, str(path))
    else:
        text, status = merge_json(existing, client, entry_list, str(path))
    holds_token = literal and client != "codex"
    if status != "unchanged" or (holds_token and path.is_file()):
        notes = write_file(path, text, secret=holds_token)
    else:
        notes = []
    names = ", ".join(e.name for e in entry_list)
    out.write(f"{status}: {names} in {path}" + (" (mode 600)" if holds_token else "") + "\n")
    for note in notes:
        out.write(f"note: {note}\n")
    if client == "codex":
        out.write(f"note: Codex reads the token from the environment: export {TOKEN_ENV}=<token from "
                  f"{instance.env_path}>\n")
    if client == "openclaw":
        out.write(f"note: {OPENCLAW_NOTE}\n")
    if holds_token:
        out.write("note: this file holds the token; do not commit it\n")
    if args.show_token:
        out.write(f"token: {token}\n")
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    try:
        return run(args)
    except (ConnectError, InstanceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def add_parser(sub) -> None:
    p = sub.add_parser("connect", help="write another agent's MCP config for this instance")
    p.add_argument("client", choices=CLIENTS)
    p.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--print", dest="print_", action="store_true", help="print the config (the default)")
    mode.add_argument("--write", nargs="?", const="", default=None, metavar="PATH",
                      help="merge into this client config (default per client: "
                           + ", ".join(f"{k} {v}" for k, v in DEFAULT_PATHS.items()) + ")")
    p.add_argument("--show-token", action="store_true", help="include the bearer token in what is printed")
    p.add_argument("--embed-token", action="store_true",
                   help="with --write: write the literal token (mode 0600) instead of a ${%s} reference; "
                        "refused unless the file is git-ignored" % TOKEN_ENV)
    p.set_defaults(func=cmd_connect)
