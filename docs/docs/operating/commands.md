# What each command does

The commands behind the [quickstart](../quickstart.md), in more detail. All of
them take `--instance DIR` (default: the current directory). The generated
[command reference](../reference/cli.md) lists every option.

`archi up` prints the chat address, the admin account and the MCP endpoint.
The passwords and the MCP bearer token are in `myteam/.env`.

**`archi init <name> [--bundle B] [--with P] [--answer ID=VALUE]`** creates
`./<name>/` as a git repository and commits a baseline. It renders the
bundle into `profiles/`, generates every secret into `.env` (mode 0600,
git-ignored), and writes `archi.yaml` and `compose.yaml`. Then it starts the
graph database and runs `okg install`, commits the deployment, publishes
with `okg provision`, and sets okg's role passwords. Every path it passes to
okg is absolute. The data root, where connector caches and repository
checkouts go, is `~/.local/share/archi/<name>` (`--data-root` to change it)
and must lie outside the instance. The default bundle is `cern-team`. Its default
packs publish with no answer and no credential. docsite downloads two small
public documentation sites (the FTS3 and Rucio docs) into its records cache
before the first ingest; indico reads one public event, "Knowledge Bases and
Agents for CERN", live. Read your own instead with
`--answer docsite_urls=<url>[,<url>...]` and
`--answer indico_event_ids='["<id>"]'` (or `--answer indico_category_ids=...`).
The public CMSSW release catalog is no longer a default; add it with
`--with cmssw-releases`. If a documentation page cannot be fetched, init stops
and names it.

**`archi up`** starts one compose project, `archi-<name>`, on one network:

| Service | What | Published on |
|---|---|---|
| `graph-db` | AKMON Postgres, with okg's preload libraries | 127.0.0.1, for okg on the host |
| `chat-db` | Open WebUI's own Postgres | not published |
| `mcp` | `okg mcp-serve`, read-only, bearer token | 127.0.0.1 |
| `webui` | Open WebUI v0.11.4 | 127.0.0.1 |

It then signs in to the site as its admin (creating the account on first
start), points it at a local ollama if one answers on port 11434
(`--ollama-url` for another), and runs `okg chat sync`, which registers the
graph tools and the preset and proves both. In the chat, pick the preset
named after the site: the graph tools are bound to it, not to the raw model.

**`archi doctor [--json]`** runs `okg deployment lint`, `okg doctor`, a
`source-freshness` check (each source's age from okg's watermarks against the
`max_staleness` its pack declares), and one check for each quickstart trap
archi absorbs, and prints a fix line for each failure. `--json` prints one object for agents. Exit code 0 means no
check failed.

**`archi eval [--bundle B] [--model M] [--model-url URL] [--out DIR]`** asks the
bundle's smoke questions of the instance through its MCP endpoint, with a local
model that can call tools, and writes `report.md` and `report.json`. It exits 1
below the bundle's pass threshold. Start ollama with `OLLAMA_CONTEXT_LENGTH=16384`
or more first (see [A local model](#a-local-model)). See [archi eval](evals.md).

**`archi down`** stops the project and keeps every volume. `archi down
--purge --yes` also deletes the graph, chat and webui volumes; that is the
reset (okg keeps its ownership claim in the database, so deleting the
folder alone is not one). Your data persists across `archi down` and
`archi up` (the published graph is the same generation), but not across
`--purge`. An instance created before 2026-10-26 kept its graph only inside
the container (the volume was mounted at the wrong path), so after upgrading
archi, re-run `archi init` for it rather than `archi up`.

## Adding sources

`archi add <pack> --answer ID=VALUE` adds a pack to a running instance and
publishes again. See [Adding a source](../adding-a-source.md). If the bundle
names companions for that pack (cms-comp-ops: `git-code` brings `code-text`),
they come too and a note says so; `--without <companion>` leaves one out.

## A local model

The chat needs a model server. With ollama on the same machine, `archi up`
finds it. Two things to know:

- Set the model at init: `archi init myteam --answer chat_model=<a model
  ollama list shows>`. The bundle default is `llama3.1:8b`.
- ollama's default context window (4096 tokens) is smaller than the preset's
  system prompt plus the graph tools' descriptions, so ollama cuts the prompt
  and the model never sees its tools. Start ollama with
  `OLLAMA_CONTEXT_LENGTH=16384` or more, or set `num_ctx` in the chat's
  controls.

Any MCP client can use the graph without a model in the chat: point it at the
MCP endpoint `archi up` printed, with `OKG_CHAT_MCP_TOKEN` from `.env` as
the bearer token.

## A remote host

Every port is bound to 127.0.0.1. On a remote machine, `archi up` prints the
`ssh -L` command that tunnels the chat to your own browser.

## The long quickstart

The older long cern-team quickstart is kept in the repository as reference
material (`docs/cern-team-quickstart.md`). Each of its traps is listed in the
[footgun ledger](footguns.md), as absorbed by archi or filed with AKMON.
