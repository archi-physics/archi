# cern-team quickstart

> **Superseded. Do not install from this page.** To install archi today,
> follow [the quickstart](docs/quickstart.md) (`archi init` / `archi up` /
> `archi doctor`); the nightly test checks that path. This page is kept as
> reference: it shows the manual okg steps underneath, for a source the
> golden path does not cover yet, and every trap in it is in the
> [footgun ledger](docs/operating/footguns.md), as absorbed by archi or filed
> with AKMON. Its commands are not tested and may no longer match the
> current CLI.

From nothing to a knowledge graph you can query, holding an Indico event and
a GitHub repository (its files, history, function bodies and symbols), with a
chat frontend in front of it.

It was written to be followed top to bottom, before `archi init` existed.
Two things are yours to choose and are marked where they appear: the
repository you want indexed, and a password for the chat database.

**The bundle is a selection of packs** (PACT r3-packs-split). Each pack brings
its sources, schema slices, invariant and playbooks; `archi render --with` and
`--without` pick them, and `archi add` adds one later. Earlier versions of
this page renamed `.yaml.example` files inside the profile instead; that
layout is gone.

**What is proven.** Steps 3–5 were run on the pack layout on 2026-10-25,
against an empty database on AKMON v0.2.0 (okg 0.2.0), with the commands as
written; their results are quoted inline. Every pack was also installed alone
from fixture inputs and published, by `python/tests/packs/test_pack_isolation.py`.
Step 6 (chat) was proven on 2026-08-27 on the earlier layout; what it reads
(`deployment.yaml`'s `chat:` block and the system prompt) is rendered the same
way. Step 8 (TWiki) was run on v0.2.0 over a real dump on the earlier layout,
on the reinstall route; on the pack layout the twiki pack is covered by the
isolation test on a synthetic dump, and step 8 says which is which.

**A note on okg versions.** This document pins AKMON v0.2.0, the public
release of okg (`github.com/mitdbg/akmon`). The package and the `okg`
command keep the name `okg`. If you move to a newer release, a command here
may change — check the release notes before assuming you did something wrong.

**You need:** a machine with `podman` (or `docker`), Python 3.12+, and
`git`. All code and images are public; no GitHub account or key is needed. No root required.

---

## 1. Get the code

```bash
mkdir -p ~/archi-quickstart && cd ~/archi-quickstart

python3.12 -m venv okg-venv          # 3.12+; archi refuses anything older

# The clone is named okg/ so later steps read the same as before.
git clone --branch v0.2.0 --depth 1 https://github.com/mitdbg/akmon.git okg
okg-venv/bin/pip install -e ./okg    # -e: the only path proven with this bundle

okg-venv/bin/pip install \
  "archi-kb @ git+https://github.com/archi-physics/archi@archi_v3#subdirectory=python"
```

If you will run the SSO cookie helper (`archi.downloaders.sso_login`), also
run `okg-venv/bin/pip install requests-kerberos`. It needs the Kerberos
development headers (`libkrb5-dev` or `krb5-devel`) to build. okg's old `cern`
extra used to install this package. The JIRA downloader needs `jira` instead.
From an archi checkout, `pip install -e "./python[cern]"` does the same.

Until `archi-kb` is released this installs the `archi_v3` branch, which moves.
For a repeatable install, replace `archi_v3` with a commit of that branch.

**Why `-e`.** On okg `1475c87d5`, okg stamps every published generation with the Git
revision of the code that produced it, and it finds that revision by looking
for a Git checkout around its own installed module. A plain
`pip install ./okg` copied the code into `site-packages`, outside any
checkout, and okg then refused to publish — with five failed authority
checks and a message about `release.pin.code_repository` that did not
mention the cause. `-e` keeps the running code inside the clone, where it
can see its own revision. Leave the clone in place afterwards; it is now
part of the installation, not a build artefact.

**Why a clone for okg but not for archi.** okg is not on PyPI, and the clone + `-e` path is the one last proven. v0.2.0
can also identify itself from an installed wheel when no Git checkout exists;
that path has not been tried with this bundle, so clone the tagged release. archi needs no checkout, so pip takes it
straight from git — no clone, no wheel build.

**Do not `pip install archi`.** That name on PyPI belongs to an unrelated
archive library. Ours will be published as `archi-kb` (ADR 0002 D2); until
then the git URL above is the only correct source.

*If pip is slow and noisy with connection retries, that is site networking
rather than anything here — it recovers on retry. Check whether the install
actually failed before assuming it did.*

## 2. Start a Postgres

okg needs six extensions, three of which must be loaded at server start, so
a stock Postgres cannot be adapted afterwards. AKMON publishes an image with
them.

**One Postgres server per deployment**, not one database on a shared
server — `pg_cron` is single-database.

```bash
podman pull ghcr.io/mitdbg/akmon-postgres:v0.2.0

podman run -d --name myteam-pg --init --shm-size 2g \
  -e POSTGRES_PASSWORD=okg \
  -e POSTGRES_DB=myteam \
  -e POSTGRES_INITDB_ARGS="--locale=C --encoding=UTF8" \
  -p 127.0.0.1:5433:5432 \
  ghcr.io/mitdbg/akmon-postgres:v0.2.0 \
  postgres -c shared_preload_libraries=timescaledb,pg_cron,pg_textsearch \
           -c cron.database_name=myteam \
           -c max_connections=100 \
           -c max_worker_processes=32
```

The image is large (it carries TimescaleDB), so the first pull takes a while.
It is public and needs no login. `--shm-size 2g` is not optional: the 64 MB default is too small
for the index builds and fails partway through.

## 3. Install the bundle

**First, make the deployments directory a Git repository.** This is not
optional and not bookkeeping: okg records the exact deployment configuration
that produced each published generation, and it refuses to publish from a
directory it cannot resolve to a commit.

```bash
mkdir -p deployments
git -C deployments init
git -C deployments commit --allow-empty -m "deployment repository baseline"
```

Then install:

```bash
# Compose the bundle's packs into ./profiles/cern-team, which okg finds on its
# own when run from this directory. No OKG_PROFILES_DIR.
#   --without docsite: docsite reads a records cache you have not made yet
#   ("Adding other sources" shows how); leave it out until you have one.
#   --with ...: one repository's files and history, its function bodies, and
#   its symbols.
okg-venv/bin/archi render --bundle cern-team --without docsite \
  --with git-code,code-text,code-symbols --instance .
export OKG_DEPLOYMENTS_DIR="$PWD/deployments"
# Keep the data root OUTSIDE deployments/: reinstalling with --force deletes
# everything under the deployment directory, caches and checkouts included.
export ARCHI_DATA_ROOT="$PWD/archi-data"
export OKG_DSN='postgresql://postgres:okg@127.0.0.1:5433/myteam'

okg-venv/bin/okg install --profile cern-team \
  --deployment-name myteam \
  --postgres-dsn "$OKG_DSN" \
  --indico-event-ids '["1716863"]' \
  --git-code-repo-name click --git-code-repo-url https://github.com/pallets/click \
  --archi-data-root "$ARCHI_DATA_ROOT/data"
```

**Each pack asks its own questions.** A pack's question `<id>` is the flag
`--<pack>-<id>`: `--indico-event-ids`, `--git-code-repo-url`. The Indico
event above is a public CERN workshop; give your own event or category ids
(`--indico-category-ids '["<id>"]'`) as YAML lists. A selected pack whose
input is missing blocks the publish for every source, so `archi render` is
where you decide what the instance holds: select only what you can feed.

**Where the repository checkout goes.** It lands at
`<archi-data-root>/repos/<name>`. The default answer is the relative path
`data`, which means "under whichever directory you run okg from", and
exporting `ARCHI_DATA_ROOT` does not change that. So pass an absolute
`--archi-data-root`, as above, and keep it outside `deployments/`.

Run `okg install` from the directory holding `profiles/`: okg looks for
`./profiles/<name>/` relative to where it runs.

**This command will fail, and that is expected.** It creates the extensions,
migrates the database, materialises the packs' schemas, loads the
catalog and may ingest some sources — then stops at the final
publish with `repository provenance inputs are not committed`, listing the
files it just wrote. It exits non-zero. Nothing is wrong: okg requires the
deployment's configuration to be committed before it will publish a
generation from it, and those files did not exist until this command created
them. You are the one who commits them.

```bash
git -C deployments add -A
git -C deployments commit -m "myteam deployment config"

okg-venv/bin/okg ingest --deployment myteam --progress
okg-venv/bin/okg run --once --deployment myteam --apply
```

`okg ingest` reads every source and stages the result; the repository is
cloned by the first ingest after the commit, not by the install. The symbol
and code-text sources run after the repository source, which clones it
(`predecessor_source`), so one ingest is enough. `okg run --once` does not
read sources: it publishes what ingest staged, and is the command that must
exit zero.

*If the install fails for some other reason, do not commit and re-run — see
"Starting over". A half-finished deployment stays wedged behind `staged
source work was not published`, and the recovery is a teardown, not a retry.
One case is different: a reinstall over a deployment that already published
can fail at the ingest stage with its data staged (step 8 says how to
check).*

Swap in your own repository. `--git-code-repo-name` is just a short label
for the graph; `--git-code-repo-url` is what gets cloned. The git-code pack
indexes one repository; a second one is a hand-written registry entry (step
4).

Optionally add `--chat-site-name 'my team chat'` and `--chat-model
'<a model your provider serves>'`. Both have defaults, and both can be
changed afterwards in `deployments/myteam/deployment.yaml` under `chat:`.

**You never run `git clone`** — the connector owns its checkout, cloning on
first run and fast-forwarding afterwards.

**Expected result:** a published generation of roughly 19,000 records and
43,000 links. In the run on 2026-10-25 it was 19,057 / 42,972: 3,384 commits,
5,438 per-file diffs, 330 files, 2,116 symbols, 91 code documents and 1,315
text chunks from `click`, and the Indico event. The install took under a
minute, and the ingest and publish about 40 s. Confirm with `okg status
--deployment myteam`: `latest_published_status: published` and a
`latest_published_generation` id.

Do not treat those numbers as exact: the repository moves upstream. Counts
that drift by tens are the sources moving; counts that differ by an order of
magnitude, or a `latest_published_generation` of `None`, mean something
failed.

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

**Before `inspect`, `filter` or the MCP server,** set the read-role passwords
okg generates for itself. Without this they fail with
`mcp_role_cannot_login`. `okg search` works without it.

```bash
okg-venv/bin/okg db role-passwords --apply --deployment myteam
```

## 4. Changing the repository, or adding packs later

**`okg install` will not re-run over an existing deployment** — it refuses
rather than overwrite, and `--force --yes` rebuilds the directory. So these
are the routes for any later change.

**Adding a pack: `archi add`.** It re-renders the instance with the pack
(and any pack it requires that the instance lacks), adds the pack's registry
entries, schema slices, files, playbooks and invariant to the installed
deployment, commits the deployment, and runs the claim, catalog load, ingest
and publish okg needs. New questions are answered with `--answer`:

```bash
okg-venv/bin/archi add twiki --instance . --deployment deployments/myteam \
  --answer twiki_eos_root=/path/to/your/twiki-eos-dump
```

`--no-okg` stops after the edit and the commit. `archi diff` reports what an
instance changed since it was rendered.

**Pointing git-code at a different repository.** Edit
`deployments/myteam/source_registry.yaml`, and under `git-code.repo` set the
two values (`repo` and `repo_root` must match under `git-code.git_history`,
`code-text.code_text` and `code-symbols.code_structure` too):

```yaml
    params:
      repo: click
      url: https://github.com/pallets/click
```

**Commit the edit before applying it.** The same rule as step 3: okg will not
publish from a deployment directory with uncommitted changes, and you have
just changed one.

```bash
git -C deployments add -A
git -C deployments commit -m "point git-code at a different repository"

cd deployments/myteam
okg-venv/bin/okg catalog ownership claim --deployment . --json
okg-venv/bin/okg catalog load --deployment . --apply --json
okg-venv/bin/okg ingest --deployment myteam --progress
cd -
```

If you forget, the ingest fails with `repository provenance inputs are not
committed` and names the file you edited. Commit and re-run.

**A second repository.** A pack selected twice installs once, so git-code
holds one repository. For another, copy the `git-code.repo` entry under a new
id with its own `ownership_id`, `repo`, `url` and `repo_root`, commit, then
claim, load and ingest as above. `archi diff` will list it as added outside
archi, which is what it is.

**Not interested in repositories at all?** Leave out `--with git-code,...`
when you render. If you installed a pack and cannot feed it, exclude its
sources and ingest again (source ids are `<pack>.<source>`):

```bash
okg-venv/bin/okg ingest --deployment myteam --progress \
  --exclude git-code.repo,git-code.git_history
```

### Git history, not only the current files

The git-code pack's `git_history` source gives the repository's commits,
tags, authors and per-file diffs, for "who last changed this file", "what
changed between two tags", "which commit added this line". It reads the
checkout the `repo` source makes and clones nothing itself. The pack adds
`git_commit`, `git_file_diff` and `git_tag` to the `content` search profile,
so plain search and the chat find them.

Measured on the earlier layout, on a repository with 1,928 commits ingested
(1,675 on the main branch) and 403 tags: ingest and publish took 78 s and
added 15,686 links. "Which tag first contains this commit" needs the SQL
`query` tool, which the chat does not offer. The header of
`bundles/packs/git-code/sources/git_history.yaml` gives the bounds for a very
large repository and the other caveats.

### Code symbols, not only file paths

The git-code pack gives the repository's **files**: an agent can find that
`src/click/core.py` exists and read it, but not search the functions and
classes in it. The code-symbols pack adds them (`code_symbol`, with
`contains`, `member_of` and `calls` links). It requires okg's `code_graph`
module and the four it composes with (`agent_sessions`, `ci`, `compute_env`,
`openspec`); the pack declares them, so nothing is edited by hand. Select it
when you render (`--with code-symbols`, as in step 3) or add it later with
`archi add code-symbols`.

Check it: `okg-venv/bin/okg search --deployment myteam --query "Context"`
should return `symbol:click:...` nodes next to the `file:click:...` ones.

### Function bodies, not only signatures

Symbols give a function's name, signature, docstring and line numbers, but
not its body, so "what does this function do when the name is empty" cannot
be answered from them. The code-text pack stores the code itself: each code
file becomes a `doc:<repo>:<path>` document linked to its `file:` node, cut
into one chunk per Python function or class with its line range. Select it
when you render (`--with code-text`) or add it with `archi add code-text`.
It installs `extractors.yaml` (its parsers and chunk sizes) in the deployment
directory; leave the `--code-text-extractors-dir` answer at its default,
which okg expands to the deployment directory each time it reads the
registry.

Measured on the earlier layout, on a 99-file repository with 77 Python files,
this source added 77 documents and 613 chunks.

Check it: search for a word that appears only inside a function body; the
hit is a `document_chunk` whose `symbol` names the function and whose
`start_line`-`end_line` range covers it. The range starts at the blank or
comment lines above the function, so it can begin a line or two before the
`def`: read the first line that holds code (the decorator or `def`) as the
line where the function is defined.

## 5. Confirm it published

```bash
okg-venv/bin/okg status --deployment myteam --json
okg-venv/bin/okg search --deployment myteam --query "Knowledge Bases and Agents"
```

`latest_published_status` must be `published` and
`latest_published_generation` must not be null. If the generation is null,
nothing was published and search will refuse — see step 7.

The search returns the Indico event (`meeting_minutes:1716863`), pinned to that
generation. Direct SQL against the graph tables is refused by design.

## 6. Start the chat frontend

The bundle already declares the chat site, so there is nothing to write.
The chat app needs **its own** database, separate from the graph; okg
checks they really are distinct and refuses if not.

**Read the note under the block before running it** — the published
addresses are deliberate and the reason matters.

```bash
export CHAT_DB_PASSWORD='pick-your-own-here'

# The address the chat container reaches this host at. Publishing there
# instead of on every address keeps the database off your network.
export CONTAINER_GATEWAY="$(podman network inspect podman \
  --format '{{(index .Subnets 0).Gateway}}')"

podman run -d --name myteam-chat-pg \
  -e POSTGRES_PASSWORD="$CHAT_DB_PASSWORD" -e POSTGRES_DB=okg_chat_app \
  -p 127.0.0.1:5458:5432 -p "$CONTAINER_GATEWAY":5458:5432 \
  docker.io/library/postgres:16

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

**Why two addresses here and nowhere else.** This database is read from
two places, and one address cannot serve both:

* `127.0.0.1` — you, on this host: the credential check above, `psql`,
  backups.
* the container gateway — the chat container. Inside a container
  `127.0.0.1` means *that container*, so a loopback-only published port is
  invisible to it. Get this wrong and the chat container crashes on startup
  — *"server closed the connection unexpectedly"* — while
  `chat-instance up` keeps polling until its timeout, which looks exactly
  like a hang.

Neither address is reachable from another machine. Under docker the
gateway is usually `172.17.0.1`; read it with
`docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'`.

**Do not publish this on `0.0.0.0`.** That is every address this host has,
including the one your institution's network can reach — and this database
holds every conversation and, once single sign-on is wired up, each
person's cached provider tokens. Only the database password would stand
between a stranger and all of it. Prove it is closed, from a different
machine:

```bash
nc -zv <this-host> 5458     # must fail
```

If your runtime offers no gateway address the host can publish on, and
`0.0.0.0` is the only way the container can reach the database, block the
port at the host firewall and re-run that check.

**Already running an installation published on `0.0.0.0`?** Re-publish it
without losing anything. `podman rm` keeps the data volume, so pass the
same volume to the new container:

```bash
VOLUME="$(podman inspect myteam-chat-pg --format '{{(index .Mounts 0).Name}}')"
podman rm -f myteam-chat-pg
podman run -d --name myteam-chat-pg \
  -e POSTGRES_PASSWORD="$CHAT_DB_PASSWORD" -e POSTGRES_DB=okg_chat_app \
  -v "$VOLUME":/var/lib/postgresql/data \
  -p 127.0.0.1:5458:5432 -p "$CONTAINER_GATEWAY":5458:5432 \
  docker.io/library/postgres:16
```

The graph database is read from the host only, which is why okg publishes
it on `127.0.0.1` itself.

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
  --host "$CONTAINER_GATEWAY" --port 8765 \
  --auth-token-env OKG_CHAT_MCP_TOKEN
```

**Bind the gateway, not `0.0.0.0`**, for the same reason as the chat
database: the chat container reaches this from inside, where a
loopback-only port is not reachable — but every other machine has no
business reaching it. `$CONTAINER_GATEWAY` is the value exported above; in
a new terminal, export it again. The bearer token is required either way,
so this is defence in depth rather than the only lock.

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

**The render writes the assistant's system prompt**, at
`skills/chat-system-prompt.md` in your deployment: archi's base prompt
followed by a short section from each selected pack. It tells the model it has
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

Each pack's coverage invariant blocks the publish too, when the pack's
sources produced none of the records it requires (the message names the
invariant, `<pack>_coverage`, and the missing subtype). Either feed the pack,
or exclude what you cannot feed and re-run; source ids are `<pack>.<source>`:

```bash
okg-venv/bin/okg ingest --deployment myteam --progress \
  --exclude docsite.docsite,jira.jira,twiki.twiki_eos
```

A pack you will never feed is better left out of the render
(`archi render --without <pack>`), since its invariant stays with it.

## 8. Adding a TWiki

> **What was run.** On the earlier layout, a reinstall over a deployment that
> had already published was run once on AKMON v0.2.0 over a real dump of
> 44,627 files. It published; the notes marked "in the run" come from that
> run. On the pack layout neither route was run on a real dump: the twiki
> pack, installed alone on a synthetic dump, publishes in
> `python/tests/packs/test_pack_isolation.py`.

TWiki reaches the instance from an **EOS dump only**, through the twiki pack.
There is no live crawl and no anonymous web fallback, so no login or cookie is
involved: you point the pack at a directory of files you already have.

**What the dump must look like.** One raw-markup file per topic, named
`<Topic>.txt`, each including its `%META:` lines. Subdirectories are
subwebs. Topics of the web named by `web_root` (`CMS` in the shipped template)
sit directly in the dump directory. Do not wrap them in an extra `CMS/`
folder, or every page id is double-prefixed (`twiki:CMS:CMS:<Topic>`).

**Point `eos_root` at the dump.** The pack asks one install question,
`--twiki-eos-root`, and it is required. Use an absolute path, because a
relative one resolves against the directory you run okg from.

```bash
export TWIKI_DUMP=/path/to/your/twiki-eos-dump    # holds <Topic>.txt files
```

*First install.* Add `twiki` to the `--with` list of `archi render` in
step 3, and add `--twiki-eos-root "$TWIKI_DUMP"` to the `okg install`
command there. Keep the `--archi-data-root "$ARCHI_DATA_ROOT/data"` flag
too: `twiki_eos_root` does not follow it, so it must be given on its own.

*Already did step 3?* `archi add` adds the pack to the installed deployment
and runs the claim, load, ingest and publish (step 4):

```bash
okg-venv/bin/archi add twiki --instance . --deployment deployments/myteam \
  --answer twiki_eos_root="$TWIKI_DUMP"
```

That route was not run over a real dump. The run quoted below used the
earlier layout's reinstall route: the TWiki file enabled, then `okg install
... --force --yes` repeating every step-3 flag. If you reinstall that way
instead of using `archi add`, the notes below apply.

Keeping the data root outside `deployments/` (step 3) is what makes this
safe: `--force` deletes the deployment directory, not your caches or dump.

**`--force` also resets your hand edits in `deployment.yaml`.** The whole
file is rewritten from the bundle, so the `chat:` block (model, port,
image, provider) goes back to bundle defaults. Before reinstalling, commit
the deployment (`git -C deployments add -A && git -C deployments commit -m
"before reinstall"`) and keep a copy of the directory. After reinstalling,
run `git -C deployments diff` and put your `chat:` settings back, for
example by re-applying the commit that made them (`git -C deployments
show <commit> | git -C deployments apply`). Check the result with the same
`diff` before you commit.

**The reinstall reads every source itself, and can end red.** `okg install`
runs an ingest as its last stage. With a large dump that read is long (in
the run, about an hour), and it can print `install failed at stage ingest`
and exit 1, with only a cut-off tail of output. In the run the TWiki data
had been staged anyway. This is the one case where the step 3 rule ("do not
commit and re-run; start over") does not apply. Tell the two cases apart
like this:

- The message says `failed at stage ingest` (not another stage), **and**
  `okg status --deployment myteam` lists the TWiki source (`twiki-eos`) with
  a large `n_node_facts`, not zero. Then continue with the commands below.
  (The cause of the red exit was not found; the run only shows that the
  data was staged.)
- Any other stage, or the TWiki source is missing or empty: it is a real
  failure. Stop and see "Starting over".

*First install.* Finish as in step 3, unchanged: commit the config, run
`okg ingest` (it clones the repository), then `okg run --once --apply`. No ownership claim is needed. The
database has no earlier claim to disagree with, but this was not run.

*Reinstall (the route that was run).* Commit the deployment config, then
**claim the deployment again**. The reinstall changes the
`source_registry` block, so the catalog the database holds no longer matches
the directory, and the next `okg ingest` is refused with
`catalog_manifest_drifted` until you claim it. Then publish what the install
staged:

```bash
git -C deployments add -A
git -C deployments commit -m "myteam deployment config"

cd deployments/myteam
okg-venv/bin/okg catalog ownership claim --deployment . --json
cd -
okg-venv/bin/okg run --once --deployment myteam --apply
```

(In the run the claim was given the deployment directory, as above. The
command also accepts a deployment name; that form was not run.) In the run,
a second `okg ingest --deployment myteam --progress` after the publish read
every source again, found nothing new and exited zero; it is a useful check
that nothing was missed, not a required step.

`okg ingest` reads every source, the dump included, and stages the
result. `okg run --once` does not read sources: it publishes what ingest
staged.

**The first publish of a full TWiki is slow.** In the run, the publish took
about 22 minutes (1,352 s) and produced a generation of about 242,000
records, of which about 233,000 were new; during that time it holds the
database's publish lock. The command finished with a `lock_headroom` warning:
the publish ran longer than the database's 15-minute (900 s) statement
timeout. In the run the generation was `published` regardless, so read the
warning as "little headroom", not "failed"; confirm with step 5. As a
precaution (not something the run tested), do not start a second publish
while the first is running. Whether later publishes are shorter
was not measured.

The answer lands in the source as `eos_root: <your path>`. For physics topics
only, install with `--twiki-physics-filter true` (the default is `false`; any
other value is refused when the source starts). To read part of the dump, edit
the `twiki.twiki_eos` entry in the deployment's `source_registry.yaml` (then
commit, claim, load and ingest as in step 4): `seed_topics` and `max_depth` for
a subtree, and `page_index_path` to write the list of kept page ids. If the
dump root is a web other than `CMS`, set `web_root` to its name.

The source is `required`: a missing or empty dump blocks the publish
instead of shrinking the graph without telling you.

**Check it.** Count the topics you handed over, then confirm the graph
returns one of them:

```bash
find "$TWIKI_DUMP" -name '*.txt' | wc -l        # topics in the dump
okg-venv/bin/okg search --deployment myteam --query "<a word from one of your topics>"
```

The search should return a `twiki:<Web>:<Topic>` page whose text is that
topic. A count of `0` means `eos_root` points at the wrong directory.

**Expect fewer pages than files.** The reader skips topics by name, with
its built-in skip patterns (`DEFAULT_SKIP_PATTERNS`, imported by
`python/archi/sources/twiki.py` from okg core; the shipped template does not set
them): pages named by number or date, `-replies` pages, lowercase names,
`Web*` infrastructure pages, `LastViewedTopics` and similar. Empty files are
skipped too. In the run, 44,627 files gave 43,783 pages (842 skipped by
name, 2 empty), so the graph count was under 2% lower than `find | wc -l`.

## Adding other sources

Sources come in packs (`archi bundles list` lists them; `archi bundles show
<pack>` shows one). cern-team offers `twiki`, `cds`, `jira`, `git-code`,
`code-text` and `code-symbols` besides its defaults, and any other pack,
`cmssw-releases` for instance, can be selected too. Select one when you render
(`--with`) or add it to an installed instance with `archi add <pack>`
(step 4). A pack's credential is read from the environment variable its
manifest names (`credentials:`), never from a file.

**Indico talk images.** With `contributions: true` and `attachment_images: true`
in the `indico.indico_live` entry, each talk record also lists its PDF, PNG and JPEG
attachments as images that an agent can open with the MCP `inspect` tool
(kind `image`; a PDF shows page 1). Nothing is downloaded at ingest. The
deployment must allow the host, or okg refuses every fetch: add an
`agent_surface` block to `deployment.yaml`. The block, with the prefix set to your
Indico base URL, and what adding it changes (the served tool list, write
launches, the old top-level `mcp:` keys) are in the comments of
`bundles/packs/indico/sources/indico_live.yaml`.
Then, from the deployment folder:

```bash
git add -A && git commit -m "Indico talk images"   # an uncommitted edit makes publishes refuse
okg catalog ownership claim --deployment . --json
okg catalog load --deployment . --apply --json
okg ingest --deployment myteam --include indico.indico_live --progress
```

Restart `okg mcp-serve` (or reconnect your agent) so it reads the new block.

For something no pack ships, `okg add <source> --deployment myteam` only
works for sources whose entry is already known to the deployment. For
anything else it reports `activation_blocked` and writes nothing, so write
the entry in `source_registry.yaml` yourself (copy a pack's template and fill
it in), commit, then re-claim and re-load.

**Replacing or removing a source does not remove its old records.** Records
leave the graph only when the source that wrote them runs again and no longer
sees them. A source you removed from the deployment never runs, so its old
records stay and can show up next to the new ones (an old crawl's TWiki
pages next to the EOS reader's, for instance). They live in the database, not in
the deployment folder, so reinstalling the folder does not clear them; the
run that was checked went away only with a fresh, empty database. So decide
your sources before you start using the graph, and if you must change them
later, start again on an empty database (see "Starting over").

**Documentation sites.** The docsite pack (a cern-team default, left out in
step 3) reads a cache file; the shipped downloader fills it from a manifest.
Once the cache exists, `archi add docsite` adds the pack. For a site that has
no sitemap, write your own manifest that lists the page URLs (public pages only)
and point the downloader at it:

```yaml
# my-docs-manifest.yaml
targets:
  public_docs:
    urls:
      - name: cmssw_coding_rules
        status: ready
        url: https://cms-sw.github.io/cms_coding_rules.html
      - name: cmssw_build_release
        status: ready
        url: https://cms-sw.github.io/build-release.html
```

```bash
ARCHI_DATA_ROOT="$ARCHI_DATA_ROOT" okg-venv/bin/python -m archi.downloaders.static_docs \
  --manifest my-docs-manifest.yaml --skip-gitlab
```

It writes `$ARCHI_DATA_ROOT/data/docsite/records.json`, which is where the
`docsite` source looks when `--archi-data-root` is `$ARCHI_DATA_ROOT/data`.
Pages are listed one by one; nothing is crawled. Entries whose status is not `ready` are skipped without a message; pages that
are not HTML are skipped and listed in the summary the command prints.

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

`rm -rf deployments` removes the Git repository along with the configuration,
so step 3 starts again from `git init`. The data root (`archi-data/`) sits outside `deployments/` on purpose, so it survives: keep it to avoid re-cloning, or remove it for a fully clean start. That is intended — the deployment
repository records one deployment's history, and you are discarding the
deployment.

Keep `okg-venv` and the okg clone — nothing is wrong with them, and
reinstalling costs you the slow part again. **Do not delete the okg clone**:
with `-e` it is where the running code lives, not a build leftover.

**When a run fails partway, this is the recovery.** okg refuses to start new
source writes while an earlier run's work sits unpublished — `staged source
work was not published; refusing to start new source writes`. Fixing the
original problem and re-running does not clear it, and neither does
`okg up`, which fails differently at its backup phase. Tear down and start
again.

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

- **The role passwords**, when the server authenticates by password.
