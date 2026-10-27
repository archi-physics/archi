# Moving from v2 to 3.0

**Short version.** v2 keeps running on the `main` branch until the cutover.
Nothing in a running v2 deployment changes when 3.0 is released. 3.0 is a new
install, not an upgrade: you do not convert a v2 config file, you start a
3.0 instance and point it at the same sources.

At the cutover the 3.0 line becomes `main`, and v2 is kept as the `v2`
branch with its `v2.x` tags (ADR 0002 D1). Until then `main` receives only v2
bug fixes. The v2 documentation stays on this site under
[v2 (legacy)](v2/index.md).

## What changes

| In v2 | In 3.0 |
|---|---|
| A RAG chatbot you configure with one large YAML file and deploy with `archi create`. | A knowledge graph you install from a bundle with `archi init` and `archi up`. Settings live in `archi.yaml` and in the packs. |
| Documents are chunked into a vector store the app owns. | OKG owns retrieval. Every answer cites records from one published generation. |
| The chat app, uploader and data viewer are Archi's own web apps. | Open WebUI is one built-in client. Any MCP client can read the graph. See [Connecting an agent](connecting-an-agent.md). |
| You pick a model provider (OpenAI, Anthropic, Gemini, local) in the Archi config. | Archi stores no model. The agent is configured with its own. |
| Sources are listed under `data_manager.sources` (links, git, Indico, JIRA, Redmine). | Sources come in packs: `docsite`, `indico`, `jira`, `git-code` and more. See the [library](library/index.md). |
| Queue bots (Redmine, JIRA, Mattermost) draft replies. | Automations under autonomy levels, with a shadow period before going live. See [the operations layer](operations-layer.md). |
| Quality is checked with `archi evaluate` (Ragas benchmark). | `archi eval` (smoke sets) and `archi eval --context` (retrieval scored against gold evidence). See [archi eval](operating/evals.md). |
| Redeploy with `archi restart`. | `archi up`, `archi down`, `archi add`, `archi diff`. See [Operating](operating/index.md). |

## What has no equivalent yet

Check this list before you plan a move. These v2 features are not in this
branch of 3.0:

- **Queue bots in live mode.** The 3.0 reference automation, `ticket-triage`,
  is not in this branch. When it ships it runs in shadow first. Promotion to
  live is a later goal. The v2 bots keep serving until then.
- **`archi eval`.** The v2 benchmark has no replacement yet. See
  [archi eval](operating/evals.md).
- **The Piazza bot.** It is retired and will not be ported.
- **The grader app, the A/B tools and the playbook store.** The design drops
  them.
- **Document upload through the chat UI.** The design points to an OKG upload
  source. Archi does not ship one in this branch.
- **Helm deployment.** 3.0 runs one compose project with podman or docker.
- **The `cluster-manager` bundle.** Planned for 3.0, not in this branch.
  The `cms-comp-ops` bundle is here; the live comp-ops instance has not been
  rebuilt from it yet.

## How to move

1. Keep v2 running. Do not touch `main`.
2. Install 3.0 beside it with the [quickstart](quickstart.md), on another
   port and with another instance name.
3. Pick the bundle closest to your use and add the packs for your sources
   (`archi init --with`, or [`archi add`](adding-a-source.md)). Answer each
   pack's questions with the values from your v2 `data_manager.sources`
   block. Credentials go in the environment, by the names the pack lists.
4. Run `archi doctor` and ask your real questions through the chat or an
   agent. Compare the answers with v2.
5. When 3.0 answers well and nothing on the list above blocks you, move your
   users. v2 can keep running as long as you need until the cutover.

If you run a v2 deployment, tell the maintainers. The cutover waits until
every live deployment has an owner and has either moved or chosen to stay on a
tagged `v2.x` release.
