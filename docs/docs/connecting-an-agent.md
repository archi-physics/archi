# Connecting other agents (`archi connect`)

Open WebUI is one client of an instance. Any agent that speaks MCP over
streamable HTTP can use the same graph tools. The endpoint is
`http://127.0.0.1:<ports.mcp>/mcp`; the bearer token is `OKG_CHAT_MCP_TOKEN` in
the instance `.env`.

## Who can read the graph

In 3.0, access is all or nothing. Every client of an instance uses the same
token, and anyone who holds it can read the whole published graph. There is
no per-user, per-group or per-record filtering, and no single sign-on for the
MCP endpoint. A request with no token or a wrong one gets HTTP 401.

Clients connected with `archi connect` also get `query`, the raw SQL read
tool. The Open WebUI preset leaves it out, but the server offers it to every
client that holds the token. It runs as the read-only `okg_mcp` database
role and refuses anything but `SELECT` and `WITH`. Give the token only to
people who may read everything in the instance.

The ports listen on 127.0.0.1 only. To reach the instance from another
machine, use an SSH tunnel.

## Configure a client

```console
archi connect claude-code [--instance DIR] [--print | --write [PATH]] [--show-token] [--embed-token]
```

`--print` (the default) shows the config. `--write` merges it into the client's
config file (default path below). A config file that exists is never replaced:
archi adds or updates its own entries (`archi-<instance name>` and the external
servers) and keeps everything else. Running the same command again changes
nothing. A file that archi cannot parse is left alone and the command stops.

The token is not printed unless you pass `--show-token`. Printed and written configs carry
`${OKG_CHAT_MCP_TOKEN}`, which Claude Code and OpenClaw expand from the environment, so
export the variable (from the instance `.env`) in the shell that starts the client. A new
file is mode 0600. `--write --embed-token` writes the literal token instead; archi refuses
unless the file is outside any git repository or git ignores it (`git check-ignore`), and
tightens an existing file to 0600. A symlinked config is written through to its target.

For Codex, archi sets only the keys it owns in `[mcp_servers.<name>]` (`url`,
`bearer_token_env_var`, `env_http_headers`). Your own keys there (`startup_timeout_sec`,
`tool_timeout_sec`, `enabled`, sub-tables) and the rest of the file are kept. The file is
read with a TOML parser, and the result is checked to differ from the old file only in
those tables; otherwise nothing is written.

| Client | Default file | Format | Checked how |
|---|---|---|---|
| `claude-code` | `./.mcp.json` | `mcpServers.<name>` with `type: "http"`, `url`, `headers` | Matches what `claude mcp add --transport http NAME URL --header "H: V"` writes (run against the installed CLI). |
| `codex` | `~/.codex/config.toml` | `[mcp_servers.<name>]` with `url`, `bearer_token_env_var`, `env_http_headers` | Codex's MCP documentation. Codex speaks streamable HTTP itself; no stdio bridge such as `mcp-remote` is needed. Codex reads the token from the environment: `export OKG_CHAT_MCP_TOKEN=...`. Not run against a Codex binary. |
| `openclaw` | `~/.openclaw/openclaw.json` | `mcp.servers.<name>` with `url`, `transport: "streamable-http"`, `headers` | OpenClaw's configuration documentation only; **not run against a live OpenClaw**. The file is JSON5: archi merges only strict JSON and otherwise tells you to merge by hand. Check with `openclaw mcp show`. |
| `mcp` | `./archi-mcp.json` | `mcpServers.<name>` JSON | The shape most MCP clients read; for any client not listed. |
| `open-webui` | none | none | `archi up` already registered the server (`okg chat sync`); the command prints the URL the site uses (`http://mcp:<port>/mcp`), the site URL and where the token is. |

## External MCP servers

Declare them in the instance's `archi.yaml`:

```yaml
mcp_servers:
  - name: notes                       # lowercase letters, digits, - _
    url: https://notes.example.org/mcp
    headers:
      Authorization: NOTES_AUTH       # header name -> NAME of an environment variable
```

The variable holds the whole header value (for example `Bearer abc`). The file is
committed, so it never holds a secret; a value that is not a variable name, a URL
with credentials or a query string, an unknown key, a duplicate name or the
instance's own server name stops `archi connect` with the reason.

`archi connect` writes these entries next to the instance's own, as `${NOTES_AUTH}`
references (Codex: `env_http_headers`). Export the variable before starting the
client. Open WebUI is not synced from this block; the command lists them to add
under Settings > Admin > External Tools.

## Checking a recipe without a model

`python/tests/connect/recipe_check.py` reads a written config, connects exactly as
the client would (URL and headers as written; `${VAR}` and Codex's variables come
from the environment or `--env-file DIR/.env`; an unset variable is an error),
calls `search` on the instance server and `echo` on any external server that has
one, and prints a JSON result. Exit 0 means every call succeeded.

```console
archi connect claude-code --write /tmp/probe/.mcp.json
python python/tests/connect/recipe_check.py --client claude-code \
    --config /tmp/probe/.mcp.json --env-file DIR/.env --query "CMSSW"
```

For Open WebUI, `python/tests/connect/webui_check.py --instance DIR` signs in as the site
admin, reads the tool server the site registered, and runs an MCP client inside the webui
container (`podman|docker exec`) that calls `search` at that registered URL over the compose
network; it exits 0 only if the answer cites rows.

`python/tests/connect/fixtures/echo_server.py` is a small external server (one
`echo` tool) for trying the `mcp_servers:` path.
