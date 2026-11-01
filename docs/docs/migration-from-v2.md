# Moving from v2 to 3.0

**Short version.** 3.0 is a new install, not an upgrade. You do not convert a
v2 config file or a v2 database. You back v2 up, start a 3.0 instance beside
it, point the new instance at the same sources, compare answers, and move your
users when you are satisfied. v2 keeps running the whole time, and you can
stop at any step and lose nothing.

This page is for an operator who runs a v2 deployment from the `main` branch
(`archi create ...`). The date the v3 line becomes `main` is the
[cutover](operating/cutover.md). Until then `main` receives only v2 bug
fixes. After it, v2 lives on the `v2` branch with its `v2.x` tags, and
nothing in a running v2 deployment changes by itself. The v2 documentation
stays on this site under [v2 (legacy)](v2/index.md).

## What changes

| In v2 | In 3.0 |
|---|---|
| A RAG chatbot you configure with one large YAML file and deploy with `archi create`. | A knowledge graph you install from a bundle with `archi init` and `archi up`. Settings live in `archi.yaml` and in the packs. |
| Documents are chunked into a vector store the app owns. | OKG owns retrieval. Every answer cites records from one published generation. |
| The chat app, uploader and data viewer are Archi's own web apps. | Open WebUI is one built-in client. Any MCP client can read the graph. See [Connecting an agent](connecting-an-agent.md). |
| You pick a model provider (OpenAI, Anthropic, Gemini, local) in the Archi config. | Archi stores no model. The agent is configured with its own. |
| Sources are listed under `data_manager.sources` (links, git, Indico, JIRA, Redmine). | Sources come in packs: `docsite`, `indico`, `jira`, `tickets` (Redmine and JIRA), `git-code` and more. See the [library](library/index.md). |
| Queue bots (Redmine, JIRA, Mattermost) draft replies. | Automations under autonomy levels, with a shadow period before going live. See [the operations layer](operations-layer.md). |
| Quality is checked with `archi evaluate` (Ragas benchmark). | `archi eval` (smoke sets with deterministic checks) and `archi eval --context` (retrieval scored against gold evidence). See [archi eval](operating/evals.md). |
| Redeploy with `archi restart`. | `archi up`, `archi down`, `archi add`, `archi diff`. See [Operating](operating/index.md). |

## What has no equivalent yet

Check this list before you plan a move. Each item is a v2 feature that 3.0.0
does not have.

- **Resolution emails and mail intake.** At SubMIT, `ticket-triage` replaces
  the v2 Redmine bot at the cutover, with no shadow trial first (board d717).
  The v2 service that posted the bot's answers also sent the resolution email,
  and it stops at the cutover, so **SubMIT users get no resolution emails from
  the cutover until 3.1** (board d719). The repository owner tells the SubMIT
  A2rchi owner and the SubMIT users before the cutover. Mail intake (emails
  turned into tickets) stays on v2 until 3.1 (board d704). The other v2 bots
  (JIRA ticket responder, Mattermost) keep serving on a tagged `v2.x`.
- **The Ragas benchmark.** `archi evaluate` scored answers with an LLM judge.
  `archi eval` replaces it with deterministic checks (cited records and required
  strings) and a retrieval score; there is no LLM-judged score.
- **The Piazza bot.** It is retired and will not be ported.
- **The grader app, the A/B tools and the playbook store.** The design drops
  them. The `v2.x` tags keep them.
- **Document upload through the chat UI, and files copied into the data
  folder.** There is no upload source. To get a document in, publish it
  somewhere a pack reads (a documentation site, a Git repository) or write a
  pack.
- **The service status board and alert banners.** Not ported.
- **Web pages behind a login (`sso-` links).** The `docsite` pack reads public
  pages only. `hypernews` reads an SSO cookie file, which is interactive to
  make. There is no general SSO page scraper.
- **More than one Git repository per `git-code` pack,** and the v2 "MkDocs
  repository" mode. `git-code` reads one repository's tracked files and
  history.
- **Restricted Indico events.** The `indico` pack reads public events and
  asks for no credential.
- **Helm deployment.** 3.0 runs one compose project with podman or docker.
- **SSO login and admin roles for the chat (`services.chat_app.auth`,
  `auth_roles`).** Open WebUI keeps its own accounts, but every chat user and
  every connected agent reads the graph through one shared token, with no
  per-user or per-role filtering. See
  [who can read the graph](connecting-an-agent.md#who-can-read-the-graph).
- **Live instances rebuilt from the bundles.** All four bundles ship
  (`cern-team`, `cms-physics`, `cms-comp-ops`, `cluster-manager`), but the
  live comp-ops and SubMIT instances have not been rebuilt from them yet.

## Before you start: back up v2

Nothing here changes v2. Do it once, before anything else, and keep the
backup until you have retired the v2 deployment.

v2 keeps three things you care about:

- the **deployment folder**, `~/.archi/archi-<name>` (or `$ARCHI_DIR/archi-<name>`):
  the rendered compose file, your config and the secrets;
- the **Postgres volume** `archi-pg-<name>`: conversations, document records,
  and the data manager's catalog;
- the **data volume** `archi-data-<name>`: the scraped and uploaded documents.

Set `RT` to `docker` or `podman`, whichever v2 runs on, and `NAME` to the
deployment name that `archi list-deployments` prints.

```bash
RT=docker                  # or podman
NAME=my-archi
BK=~/archi-v2-backup/$NAME && mkdir -p "$BK" && cd "$BK"

archi list-deployments                                  # confirm the name
cp -a ~/.archi/archi-$NAME ./deployment                 # config, compose file, secrets
$RT exec postgres-$NAME pg_dump -U archi -d archi-db -Fc > archi-db.dump
$RT run --rm -v archi-data-$NAME:/data -v "$PWD":/backup alpine \
    tar czf /backup/archi-data.tgz -C /data .
ls -l                                                    # three non-empty items
```

Checks: `archi-db.dump` is not empty, and `pg_restore --list archi-db.dump |
head` (if you have `pg_restore` on the host) prints a table of contents.
`archi-data.tgz` lists with `tar tzf archi-data.tgz | head`. The deployment
folder holds secrets, so keep the backup folder private.

Write down which v2 sources and services you run: the `data_manager.sources`
block of your config, and the `--services` you passed to `archi create`. You
need them in the next sections.

## Stand up 3.0 beside v2

1. **Use a new Python environment.** The v2 and 3.0 command-line programs are
   both called `archi`. Do not install 3.0 into the environment that runs v2.

   ```bash
   python3 -m venv ~/archi3-venv
   . ~/archi3-venv/bin/activate
   pip install "okg @ https://github.com/mitdbg/akmon/releases/download/v0.2.0/okg-0.2.0-py3-none-any.whl#sha256=72bac3b67cf29cccd27e77e58ad309104f19fcce255a04a4a6f23b8318b88e12" archi-kb==3.0.0
   archi --help
   ```

   If `pip` reports no matching distribution for `archi-kb`, the PyPI release
   is not out yet. Use the git install in the [quickstart](quickstart.md)
   instead. Check that `archi --help` lists `init` and `up`. If it lists
   `create`, you are still in the v2 environment.

2. **Make the instance under another name.** Pick a name that v2 does not use
   (here `NAME3`). Ports are chosen free at `init` time and bound to
   127.0.0.1, so they do not collide with v2. The instance lives in its own
   directory and its own containers and volumes.

   ```bash
   NAME3=${NAME}-v3
   archi init $NAME3 --bundle cern-team --dir ~/archi3/$NAME3
   cd ~/archi3/$NAME3
   archi up
   archi doctor
   ```

   `archi doctor` exits 0 when no check failed. `archi up` prints the chat
   address and the MCP endpoint.

3. **Add your sources** (next section), then run `archi doctor` again. Its
   `source-freshness` check shows each source's age against its limit.

4. **Ask your real questions.** Use the questions your users ask v2, in the
   chat or through [an agent](connecting-an-agent.md). Compare the cited
   answers with v2's. Record which questions are worse and why.

5. **Move your users** when 3.0 answers well and nothing in the list above
   blocks you. Keep v2 running as long as you need.

## Move each source

`archi init --with <pack>` chooses packs at the start. `archi add <pack>` adds
one to a running instance and publishes again. `--answer ID=VALUE` answers
one of the pack's questions; the id is the pack name with dashes as
underscores, then `_` and the question id. Credentials never go on the command
line: put the value in `.env` in the instance, under the name the pack lists.

| v2 source | 3.0 pack | How to move it |
|---|---|---|
| `links`: public web pages from `.list` files | `docsite` | `archi add docsite --answer docsite_urls=<url>[,<url>...]`. Give the documentation sites or pages to read. It downloads the pages first, then ingests. A page that cannot be fetched stops the add before it edits anything. |
| `links` behind SSO (`sso-` prefix) | none | No equivalent. Leave these on v2 or publish the content where a pack can read it. |
| `git` (an MkDocs repository, `git-` prefix) | `git-code` | `archi add git-code --answer git_code_repo_url=<url> --answer git_code_repo_name=<short name>`. It reads every tracked file and the history, not only the Markdown. One repository per instance. |
| `indico` | `indico` | `archi add indico --answer indico_event_ids='["<id>"]'` or `--answer indico_category_ids='["<id>"]'`. Public events only. With no answer it reads the CERN workshop event 1716863. |
| `jira` | `jira` | Archi reads a cache that a downloader writes, and `archi add jira` stops if the cache is missing. So download first: `python -m archi.downloaders.jira --output-dir <data root>/jira --project <KEY>` (repeat `--project`; set `CERN_JIRA_TOKEN` in the environment if the projects are not public), then `archi add jira`. `<data root>` is the `data_root:` value in `archi.yaml`. The downloader needs the `jira` Python package (`pip install jira`). See `python -m archi.downloaders.jira --help`. Refreshing the cache is yours to schedule: the pack expects data no older than 24 hours, and `archi doctor` reports it when it is older. |
| `redmine` | `tickets` | Like `jira`, Archi reads a cache that a downloader writes. Download first: `python -m archi.downloaders.tickets --output-dir <data root>/tickets --redmine-url <url>` (set `REDMINE_API_KEY` in the environment), then `archi add tickets --answer tickets_base_url=<url>`. The same pack reads JIRA Server or Data Center with `--jira-url`. See `python -m archi.downloaders.tickets --help`. |
| Uploaded files, files copied into the data folder | none | No equivalent. |
| Mattermost, Piazza, grader, Grafana service board | none | Not ported. See above. |

[Adding a source](adding-a-source.md) says more about packs, and
[connector caches](reference/connector-caches.md) says what each downloader
writes. The [library](library/index.md) lists every pack with its
questions and credentials.

## Roll back

Each stage has its own way back. v2 is never edited by the move, so the first
two cost nothing.

- **Abandon the 3.0 instance.** v2 is untouched. From the instance directory:

  ```bash
  archi down --purge --yes     # stops the stack and deletes its volumes
  cd ~ && rm -rf ~/archi3/$NAME3
  ```

  Delete `~/.local/share/archi/$NAME3` too if you want the connector caches
  gone (that is the default data root).

- **Pause a move.** Run `archi down` (no `--purge`). It keeps every volume.
  `archi up` brings the instance back with the same graph.

- **Return your users to v2.** If you pointed users or a proxy at 3.0, point
  them back. v2 never stopped.

- **Restore v2 itself** (only if v2 was damaged, for example after the cutover
  moved its branch). Use the v2 line, not 3.0. In a v2 environment:

  ```bash
  git checkout v2.5.0           # or the newest v2.x tag your site pinned
  pip install -e .
  archi delete -n $NAME         # only if a broken deployment is still there
  archi create -n $NAME -c "$BK/deployment/<your config>.yaml" \
      -e "$BK/deployment/<your secrets>.env" --services <your services>
  $RT exec -i postgres-$NAME pg_restore -U archi -d archi-db --clean --if-exists < "$BK/archi-db.dump"
  $RT run --rm -v archi-data-$NAME:/data -v "$BK":/backup alpine \
      tar xzf /backup/archi-data.tgz -C /data
  ```

  The config and secrets are the files you used the first time. The copy of
  the deployment folder in the backup shows which they were. `archi delete`
  without `--rmv` keeps the volumes, and the restore then overwrites them.

## Tell the maintainers

If you run a v2 deployment, tell the maintainers. The
[cutover](operating/cutover.md) waits until every live deployment has a named
owner and has either moved or chosen to stay on a tagged `v2.x` release. A
deployment that stays on v2 should pin the tag it runs, so a later fix on
`main` cannot change it.
