# cern-team quickstart

From nothing to a knowledge graph you can query, holding a GitHub
repository and a GitLab repository, with a chat frontend in front of it.

Follow it top to bottom. The commands are real and runnable as written.
Two things are yours to choose and are marked where they appear: the
repositories you want indexed, and a password for the chat database.

**What is proven.** Steps 1–6 were executed end to end against an empty
database on 2026-08-27, and their results are quoted inline. Step 8 (TWiki)
is the one part not verified, and says so where it appears.

**You need:** a machine with `podman` (or `docker`), Python 3.12+, `git`,
and read access to the private `mitdbg/okg` repository. No root required.

---

## 1. Get the code

```bash
mkdir -p ~/archi-quickstart && cd ~/archi-quickstart

python3.12 -m venv okg-venv          # 3.12+; archi refuses anything older

git clone git@github.com:mitdbg/okg.git
okg-venv/bin/pip install ./okg       # add -e only if you plan to edit okg

okg-venv/bin/pip install \
  "archi @ git+https://github.com/archi-physics/archi@archi_v3#subdirectory=python"
```

**Why a clone for okg but not for archi.** okg is a private repository and
is not on PyPI, so pip cannot fetch it for you. archi is public, so pip
takes it straight from git — no clone, no wheel build.

**Do not `pip install archi`.** That name on PyPI belongs to an unrelated
archive library. Ours is unpublished; the git URL above is the only correct
source.

*If pip is slow and noisy with connection retries, that is site networking
rather than anything here — it recovers on retry. Check whether the install
actually failed before assuming it did.*

## 2. Start a Postgres

okg needs six extensions, three of which must be loaded at server start, so
a stock Postgres cannot be adapted afterwards. okg ships an image with them.

**One Postgres server per deployment**, not one database on a shared
server — `pg_cron` is single-database.

```bash
( cd okg/ops/pg && podman build -t okg-pg17:local . )

podman run -d --name myteam-pg --shm-size 2g \
  -e POSTGRES_PASSWORD=okg \
  -e POSTGRES_DB=myteam \
  -e POSTGRES_INITDB_ARGS="--locale=C --encoding=UTF8" \
  -p 127.0.0.1:5433:5432 \
  okg-pg17:local \
  postgres -c shared_preload_libraries=timescaledb,pg_cron,pg_textsearch \
           -c cron.database_name=myteam \
           -c max_connections=100
```

The build pulls a large TimescaleDB base image, so it takes a while the
first time. `--shm-size 2g` is not optional: the 64 MB default is too small
for the index builds and fails partway through.

## 3. Install the bundle

```bash
export OKG_PROFILES_DIR="$(okg-venv/bin/archi-profiles-dir)"
export OKG_DEPLOYMENTS_DIR="$PWD/deployments"
export ARCHI_DATA_ROOT="$PWD/deployments/myteam"
export OKG_DSN='postgresql://postgres:okg@127.0.0.1:5433/myteam'

okg-venv/bin/okg install --profile cern-team \
  --deployment-name myteam \
  --postgres-dsn "$OKG_DSN" \
  --github-repo-name click --github-repo-url https://github.com/pallets/click \
  --gitlab-repo-name ci-example \
  --gitlab-repo-url https://gitlab.cern.ch/gitlabci-examples/build_docker_image.git
```

**That one command is the whole install.** It creates the extensions,
migrates, materialises the distribution's schemas, loads the catalog,
clones both repositories, ingests, and publishes.

Swap in your own repositories. `--*-repo-name` is just a short label for
the graph; `--*-repo-url` is what gets cloned.

Optionally add `--chat-site-name 'my team chat'` and `--chat-model
'<a model your provider serves>'`. Both have defaults, and both can be
changed afterwards in `deployments/myteam/deployment.yaml` under `chat:`.

**Give the repository URLs here.** They look like noise, but the bundle
selects both repository connectors by default, and a *selected* source that
cannot run blocks the publish for every other source too — so an install
without them ends with nothing published. Step 4 is how to fill them in
afterwards if you would rather not decide now, and how to change them
later.

**You never run `git clone`** — the connectors own their checkouts,
cloning on first run and fast-forwarding afterwards.

**Expected result:** `first publish complete`, roughly 2,586 nodes / 2,279
edges for the two repositories above.

*If the catalog load fails on authentication, the migration created the
loopback roles without passwords. Set them, then re-run the ingest (not the
install — see step 4):*

```bash
podman exec myteam-pg psql -U postgres -d myteam \
  -c "ALTER ROLE okg_mcp PASSWORD 'okg_mcp'"
podman exec myteam-pg psql -U postgres -d myteam \
  -c "ALTER ROLE app_rw PASSWORD 'okg_rw'"
podman exec myteam-pg psql -U postgres -d myteam \
  -c "ALTER ROLE app_ro PASSWORD 'okg_ro'"
```

## 4. Changing or adding repositories later

**`okg install` will not re-run over an existing deployment** — it refuses
rather than overwrite, and `--force --yes` rebuilds the directory. So this
is the route for any later change, and the recovery if you installed
without URLs.

Edit `deployments/myteam/source_registry.yaml`, and under `github_repo` and
`gitlab_repo` set the two values:

```yaml
    params:
      repo: click
      url: https://github.com/pallets/click
```

Then apply and publish:

```bash
cd deployments/myteam
okg-venv/bin/okg catalog ownership claim --deployment . --json
okg-venv/bin/okg catalog load --deployment . --apply --json
okg-venv/bin/okg ingest --deployment myteam --progress
cd -
```

**Not interested in repositories at all?** Install without the URLs — that
install will end with the publish blocked, which is expected — then exclude
the two connectors and ingest again. The CMSSW catalog alone publishes
fine:

```bash
okg-venv/bin/okg ingest --deployment myteam --progress \
  --exclude github_repo,gitlab_repo
```

## 5. Confirm it published

```bash
okg-venv/bin/okg status --deployment myteam --json
okg-venv/bin/okg search --deployment myteam --query "docker image build"
```

`latest_published_status` must be `published` and
`latest_published_generation` must not be null. If the generation is null,
nothing was published and search will refuse — see step 7.

The search returns real files from the GitLab repository, pinned to that
generation. Direct SQL against the graph tables is refused by design.

## 6. Start the chat frontend

The bundle already declares the chat site, so there is nothing to write.
The chat app needs **its own** database, separate from the graph; okg
checks they really are distinct and refuses if not.

**Read the note under the block before running it** — the `0.0.0.0` is
deliberate and the reason matters.

```bash
export CHAT_DB_PASSWORD='pick-your-own-here'

podman run -d --name myteam-chat-pg \
  -e POSTGRES_PASSWORD="$CHAT_DB_PASSWORD" -e POSTGRES_DB=okg_chat_app \
  -p 0.0.0.0:5458:5432 docker.io/library/postgres:16

export OKG_CHAT_APP_DATABASE_URL="postgresql://postgres:$CHAT_DB_PASSWORD@127.0.0.1:5458/okg_chat_app"
export OKG_CHAT_WEBUI_SECRET_KEY='choose-anything'
export OKG_CHAT_MCP_TOKEN='choose-anything'

# Prove the credentials work before handing them to okg. If this fails,
# fix it here — a wrong password shows up much later as a crashed
# container and a command that appears to hang.
sleep 5
okg-venv/bin/python -c "
import os, psycopg
psycopg.connect(os.environ['OKG_CHAT_APP_DATABASE_URL']).close()
print('chat database reachable with these credentials')"

okg-venv/bin/okg chat-instance up --deployment myteam \
  --container-runtime podman --instance-port 8099 --ready-timeout 300
okg-venv/bin/okg chat-instance status --deployment myteam
```

**Set `CHAT_DB_PASSWORD` and create the database in the same shell**, and do
not change it afterwards. The container bakes the password in at creation,
so a value that changes between `podman run` and the DSN gives
`password authentication failed for user "postgres"` — which surfaces as a
crashed chat container while `chat-instance up` polls on, looking like a
hang. The check above catches it immediately instead.

**Why `0.0.0.0` here and nowhere else.** This database is read from
*inside* a container, and under rootless podman a loopback-only published
port is not reachable from one. Get this wrong and the chat container
crashes on startup — *"server closed the connection unexpectedly"* — while
`chat-instance up` keeps polling until its timeout, which looks exactly
like a hang. Binding beyond loopback means other machines can reach it, so
choose a real password. The graph database is read from the host and stays
loopback-only.

**The DSN still says `127.0.0.1`, and that is correct.** okg validates the
database from the host, then rewrites the container's copy to reach back
through the container gateway, reporting the substitution rather than doing
it quietly. A container-only name fails host-side; the machine's own
network address fails container-side.

**`--ready-timeout 300` matters.** On first start Open WebUI loads a
sentence-embedding model, which overruns the 120-second default; okg then
tears the container down as never-started. Expect a quiet minute or two
here — that one really is just slow.

**`--container-runtime podman`** is needed because the default is docker,
and it is accepted by `up` only — `status` rejects it.

**Retrying after a failed attempt? Remove BOTH containers, not just the
database.** The chat container bakes its connection string in at creation,
and `chat-instance up` will reuse an existing one — so a chat container
built against the old password keeps failing with
`password authentication failed` no matter how correct the database and DSN
now are. Neither container has a volume, so removing them loses nothing:

```bash
podman rm -f okg-chat-myteam myteam-chat-pg
# then re-run the block above from the top, in one shell
```

This is the single most likely reason a second attempt fails the same way
as the first.

Expect `status` to report the tools endpoint dead. It is telling the truth:
the site is up, but the graph tools are a separate process. **Take it
seriously — the site can be up and every tool call still fail.**

### Connect the graph tools

**The site being up does not mean the assistant can reach your graph.** Ask
it something now and it will search Open WebUI's own built-in "knowledge
bases" — which are unrelated and empty — and tell you it has no access. Two
more steps wire the graph in.

**First, serve the tools.** This is a long-running process; give it its own
terminal. One server per deployment, on one branch — which is why it is not
folded into `chat-instance up`.

```bash
export OKG_DSN='postgresql://postgres:okg@127.0.0.1:5433/myteam'
export OKG_CHAT_MCP_TOKEN='choose-anything'   # the value you used above

okg-venv/bin/okg mcp-serve --deployment myteam \
  --transport streamable-http \
  --host 0.0.0.0 --port 8765 \
  --auth-token-env OKG_CHAT_MCP_TOKEN
```

`--host 0.0.0.0` for the same reason as the chat database: the chat
container reaches this from inside, where a loopback-only port is not
reachable.

**The port must be 8765**, because that is what the bundle declares in
`chat.mcp.port`. `chat sync` looks for the endpoint there and nowhere else —
serve on a different port and it fails with `mcp_unreachable` while the
server is running perfectly well somewhere you did not tell it about.

**Then wire the chat to it**, from your first terminal:

```bash
export OKG_CHAT_INSTANCE_URL=http://127.0.0.1:8099
export OKG_CHAT_ADMIN_TOKEN='<see below>'
okg-venv/bin/okg chat sync --deployment myteam
```

**Getting that token without touching a browser.** The first account
created on a fresh instance becomes the admin, and signing up returns a
token `chat sync` accepts directly — so this is one scriptable command, not
a round trip through the UI:

```bash
export OKG_CHAT_ADMIN_TOKEN=$(curl -s -X POST "$OKG_CHAT_INSTANCE_URL/api/v1/auths/signup" \
  -H 'Content-Type: application/json' \
  -d '{"name":"admin","email":"you@example.org","password":"pick-a-password"}' \
  | okg-venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["token"])')
```

On an instance where that account already exists, swap `signup` for
`signin` and drop the `name` field. Keep the chat app database and the
account persists, so this is a one-liner on every later run rather than a
fresh sign-up.

*(If you would rather use the UI: sign up in the browser, then
**Settings → Account → API Keys → Create new key**. Same result, more
clicks.)*

`chat sync` is the step that makes this automatic rather than clicked
together: it renders the site's appearance from the bundle, creates the MCP
connection, applies the model preset with the graph tools bound, and then
**proves it works** — an `initialize` and a `tools/list` through the
registered credential. A 401, an empty tool list, or a bare TCP connect is
treated as failure, not success. It reads everything back and compares
against the manifest, and a partial application is a failure.

**The bundle ships the assistant's system prompt**, at
`skills/chat-system-prompt.md` in your deployment. It tells the model it has
graph tools, names them, and tells it to ground answers and to say when the
graph does not contain something. This is required, not decorative: `chat
sync` refuses a deployment that declares no prompt, because Open WebUI never
shows the model the MCP server's own instructions — without it the assistant
would have tools registered and no idea it had them. Edit that file to
change the assistant's behaviour, then re-run `chat sync`.

### Opening it, and giving it a model

**If the machine is remote, tunnel to it.** The site binds to loopback on
purpose, so it is not reachable across the network. From your own machine:

```bash
ssh -L 8099:127.0.0.1:8099 <that-host>
```

Then open `http://127.0.0.1:8099` in your own browser.

**Point it at a model provider.** A local ollama needs no API key at all, and
with the admin token you already exported this is one command rather than a
trip through the settings screens:

```bash
curl -s -X POST "$OKG_CHAT_INSTANCE_URL/ollama/config/update" \
  -H "Authorization: Bearer $OKG_CHAT_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"ENABLE_OLLAMA_API":true,"OLLAMA_BASE_URLS":["http://host.docker.internal:<port>"]}'
```

*(You can do the same thing by hand under **Settings → Admin → Connections**
if you prefer clicking.)*

Two things about that URL, both verified the hard way:

- **Use the gateway name, not loopback or the hostname.** If ollama runs on
  the same machine as the chat container, the URL is
  `http://host.docker.internal:<port>`. `127.0.0.1` is the *container's* own
  loopback, and a rootless container cannot route to its host's network
  address — only the gateway alias reaches it.
- **Check the port.** `OLLAMA_HOST` is often set to something other than the
  default `11434`; `systemctl show ollama -p Environment` will tell you.

### Then use it — and pick the right model

**Select the deployment's preset in the model picker — the pre-selected model
is the wrong one.** This is the single most misleading step, and it is not a
matter of carelessness: the model the site opens on is a raw ollama model,
because the manifest field that names the preset's underlying model is also
the field that sets the site default. The graph tools are bound to the
*preset* `chat sync` created (named after your deployment), so unless you
change the picker you are talking to a plain language model with no access to
anything you indexed and no instructions about this deployment. It will
answer CMS questions from whatever it already knows, fluently, with nothing
indicating the graph was never consulted.

A good first question, with a checkable answer:

> What repositories are indexed in this graph? Name three actual files from
> them.

It should name the repositories you installed and files that really exist in
them. If it names things you never indexed — plausible-sounding projects from
the same domain — it is not reaching the graph. Check that you selected the
preset before concluding anything is broken.

The same trap applies to calling the API directly rather than using the
browser: Open WebUI deliberately does **not** attach a preset's tools for API
callers (*"API callers don't expect hidden tools; they can explicitly request
tools via `tool_ids`"*), so a raw completions call gets a toolless model that
confabulates. Pass `tool_ids` explicitly if you script against it.

Ollama on a *different* machine is simpler — an ordinary
`http://<other-host>:<port>` works, because outbound networking from a
container is unrestricted. It is only reaching back to its own host that
needs the gateway name.

## 7. If the publish is blocked

Every *selected* source must finish or none of them publish — including the
ones that worked. So a single source you cannot feed blocks everything.

The bundle selects only three by default and none needs a credential. If
you enabled others, exclude what you cannot feed and re-run:

```bash
okg-venv/bin/okg ingest --deployment myteam --progress \
  --exclude docsite,jira,twiki_eos,twiki_crawl
```

## 8. Adding a TWiki

> **Not verified** — no CERN SSO cookie was available where this was
> written. The requirement below is evidenced; the working configuration is
> not.

**A CERN SSO cookie is required even for a public wiki page.** The
connector reads a topic's source through `?raw=all`, and CERN redirects
that to its login page even where the rendered page opens for anyone. The
connector detects the login bounce and refuses to ingest it — correctly: a
login page stored as wiki text is worse than a clean failure.

Run the CERN SSO login helper (it prompts for a code from your
authenticator), then enable the connector and reinstall into a fresh
deployment:

```bash
export CERN_TWIKI_COOKIE_FILE=/path/to/cookies.txt
mv "$OKG_PROFILES_DIR/cern-team/source-defaults/twiki_crawl.yaml.example" \
   "$OKG_PROFILES_DIR/cern-team/source-defaults/twiki_crawl.yaml"
```

Leave the crawl depth at `0`. At depth 1 it follows every link on the seed
page, and one unreachable topic makes the whole run refuse to claim it saw
everything.

## Adding other sources

The bundle ships JIRA, documentation sites, Indico and both TWiki readers
as `.yaml.example`. Rename one to `.yaml`, supply its credential by
environment variable, and reinstall.

For something the bundle does not ship, `okg add <source> --deployment
myteam` writes a registry entry — deliberately an incomplete one, reporting
`activation_blocked` rather than activating a source it cannot fully
describe. You finish it in `source_registry.yaml`, then re-claim and
re-load.

## Starting over

The ownership claim lives in the **database**, not the deployment folder, so
deleting the folder alone is not a reset — the next publish will refuse with
`catalog_ownership_mismatch` because the registry no longer matches what was
claimed. Remove the container to clear it.

```bash
okg-venv/bin/okg chat-instance down --deployment myteam --container-runtime podman
podman rm -f myteam-pg myteam-chat-pg
rm -rf deployments
```

Keep `okg-venv` and the okg clone — nothing is wrong with them, and
reinstalling costs you the slow part again.

**If you hit `catalog_ownership_mismatch` and would rather not start over,**
re-running the claim is enough — it re-records the current registry as the
owner:

```bash
cd deployments/myteam
okg-venv/bin/okg catalog ownership claim --deployment . --json
okg-venv/bin/okg catalog load --deployment . --apply --json
okg-venv/bin/okg ingest --deployment myteam --progress
cd -
```

**`okg install` is not a retry.** It refuses to run over an existing
deployment directory, and where it does run it cannot fix a stale claim —
it will re-ingest happily and still fail to publish.

## What still needs a person

- **`OKG_PROFILES_DIR`** — okg cannot find bundles inside an installed
  package, so the variable is required even though the wheel carries them.
- **The role passwords**, when the server authenticates by password.
- **The TWiki cookie** — the CERN SSO login is interactive by design.
