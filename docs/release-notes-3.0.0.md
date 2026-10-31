# archi 3.0.0

Archi 3.0 is built to automate operations for scientific computing. It turns
a facility's own knowledge (documentation, meetings, wikis, tickets, code,
catalogs, past incidents) into a context layer that any agent can read and
cite, and adds an operations layer whose automations propose actions that a
person approves. The package is `archi-kb` on PyPI. You import it as `archi`.

3.0 is a new product, not an upgrade of v2. v2 lives on in the `v2` branch
and its `v2.x` tags.

## What you get

- **One command from an empty machine to a cited answer.** `archi init`,
  `archi up`, `archi doctor`. The graph survives `archi down` and `archi up`.
  A nightly build proves the path on an empty runner.
- **Bundles and packs.** Pick what goes in the graph. Four bundles ship:
  `cern-team` (documentation and meetings, no credential needed), `cms-physics`
  (physics TWiki, CDS, CMSSW releases, analysis notes and the analysis ids that
  link them), `cms-comp-ops` (sites, storage, datasets, workflows, site status,
  past incidents) and `cluster-manager` (documentation, Redmine or JIRA
  tickets, operational memory, and optionally ServiceNow, Slurm, HTCondor and
  node health, for any site). Add a source later with `archi add <pack>`.
  Write your own pack in a separate pip package.
- **Repository text.** The `code-text` pack stores the text of every text file
  in a Git repository (code, Markdown, YAML, shell scripts, configuration),
  next to git-code's file and history records, and leaves out any file that
  holds a literal secret (private keys, tokens, passwords in URLs or config).
- **Evals.** `archi eval --bundle <name>` checks that a bundle answers its
  smoke set with cited records; `archi eval --context` scores retrieval with no
  model, and with `--agents` compares a model agent with bash over the raw
  files against the same model on the graph's MCP tools. `instance-health` raises an issue when a smoke-set run fails or
  drops. `cms-comp-ops` has no smoke set.
- **A context layer that reports its own health.** Each pack declares how stale
  its sources may be, and which links and checks must hold. `archi doctor`
  reports them. The `ops-record` pack keeps incidents, proposals, decisions,
  actions and outcomes as records.
- **An operations layer.** Automations run at levels L0 to L2 (L3 is refused),
  start in shadow, and are promoted by an approved decision. `archi ops
  status`, `pending`, `approve`, `reject`, `promote`, `halt`, `resume`. The
  first automation, `instance-health`, watches the instance itself. No bundle
  includes it: select it when you create the instance, with
  `archi init <name> --bundle <bundle> --with instance-health`.
- **Bring your own agent.** `archi connect` writes the MCP configuration for
  Claude Code, codex, openclaw, Open WebUI and generic MCP over HTTP.
- **Rewritten documentation**, with pages for every bundle, pack and command
  generated from the code.

## What is not in 3.0.0

- The `ticket-triage` automation and its shadow report: planned after 3.0.
  The v2 queue bots keep serving on a tagged `v2.x` until then.
- Instances rebuilt from the bundles: the live comp-ops instance and SubMIT.
- The context-eval acceptance run on a live instance, and a smoke-set pass on
  a local model (the sets pass on GPT-6 Luna).
- Adding an automation pack to an existing instance. `archi add` does not
  turn on the automation runtime, so select automations at `archi init`.
- Level L3. Document upload in the chat UI. SSO-page and Helm support.

The full list, and what each missing piece means for a v2 deployment, is in
[Moving from v2 to 3.0](https://github.com/archi-physics/archi/blob/main/docs/docs/migration-from-v2.md). Every item is
tracked in the [changelog](https://github.com/archi-physics/archi/blob/main/CHANGELOG.md).

Built and tested against okg 0.2.0 (AKMON). The asks filed with AKMON for what
Archi works around are listed in the changelog.

## Install

You need `docker` or `podman` (with `compose`), Python 3.12 or newer, and
`git`. Use a virtual environment.

```bash
pip install "okg @ https://github.com/mitdbg/akmon/releases/download/v0.2.0/okg-0.2.0-py3-none-any.whl#sha256=72bac3b67cf29cccd27e77e58ad309104f19fcce255a04a4a6f23b8318b88e12" archi-kb==3.0.0
```

`okg` is the graph engine. It is not on PyPI yet, so it comes from the AKMON
release. Do not `pip install archi`: that name belongs to an unrelated package.

## Quickstart

```bash
archi init myteam --bundle cern-team
cd myteam
archi up
archi doctor --json
```

Then ask a question in the chat that `archi up` printed, or connect an agent
with `archi connect claude-code`. The [quickstart](https://github.com/archi-physics/archi/blob/main/docs/docs/quickstart.md)
explains each command, including the check that an answer cites graph records.

## Upgrade from v2

There is no in-place upgrade. Back up v2, stand 3.0 up beside it under another
name, move each source, compare answers, and move your users when ready. v2 is
never touched, so you can stop at any step.

- [Moving from v2 to 3.0](https://github.com/archi-physics/archi/blob/main/docs/docs/migration-from-v2.md): what to back up,
  how to run both, how each source moves, what has no equivalent, how to roll
  back.
- If you run a v2 deployment, tell the maintainers. The
  [cutover runbook](https://github.com/archi-physics/archi/blob/main/docs/docs/operating/cutover.md) waits for every live
  deployment to have an owner and a decision.
