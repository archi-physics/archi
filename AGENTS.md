# archi (v3 line) — working rules

Archi **automates operations for scientific computing** (ADR 0002 §1):
a context layer that holds what a facility knows, kept solid and cited,
and an operations layer whose automations propose, get approval for, and
record actions. It is built on OKG (AKMON) and ships connectors,
schemas, enrichers, live tools, playbooks, packs, bundles and the
automation runtime as the pip package `archi-kb` (import name `archi`),
built from [`python/`](python/). It contains no credentials, no site
config and no running services. It is installed onto an OKG deployment
through a bundle; instances live in their own repositories and carry
the configuration and secrets. Do not describe it as
"the HEP distribution of OKG": that was the v3 program's first framing
(ADR 0001) and undersells it.

This is `archi-physics/archi`, branch `archi_v3`. The v3 line was
developed in `archi-physics/archi-okg` until 2026-10-25, when its `main`
became this branch (ADR 0002 D1, PACT `r3-release-home`; the old tip of
this branch is tag `archi_v3-pre-merge`). archi-okg takes no new commits
and is being archived.

| | |
|---|---|
| `python/archi/` | The wheel: sources, enrichers, schemas. |
| `python/tests/` | The suite — around 1,700 tests, run by CI on every PR. |
| `bundles/` | Install bundles (e.g. `cern-team`). |
| `skills/` | Playbooks for agents operating over the graph. |
| `pact/changes/` | The change record. |
| `docs/okg-alignment.md` | The okg-side program's window into this repo. |

> **This file is the one place working rules live.** `CLAUDE.md` is a
> single `@AGENTS.md` import, so Claude Code loads this file; Codex reads
> it directly. Never put rules in `CLAUDE.md`. The old one was deleted
> because it had drifted into describing a layout that no longer
> existed (see "What the old file got wrong" at the end).

## Where work lands

**PRs target `archi_v3`.** `pact/project.yaml` (`pr_policy.base`)
says the same.

`main` is the v2 application, the RAG chatbot. It serves live v2
deployments, is the repository default, and takes only v2 fixes. Never
open v3 work against it. At cutover the v3 line becomes `main` and v2
moves to a `v2` branch (ADR 0002 D1, PACT `r3-release`). This section
changes then, in the same PR as the cutover.

Do not open PRs against archi-okg either. Its `main` is frozen at
`011f5d64`, the commit where this branch took the line over.

## Standing rules

- **This repository and the okg substrate (AKMON) are both public.**
  Nothing here may carry a credential, a site hostname or an instance's
  configuration. The tests import `okg`, so CI installs the AKMON
  release wheel, pinned by version and sha256 in
  `.github/workflows/ci.yml`. No secret is involved, so fork PRs run
  the full suite.
- **A consumer's variant is an option here, off by default.** When a
  deployment needs a source to behave differently, add the option to
  the source with a default that preserves existing behaviour, and let
  the deployment turn it on in its profile. `physics_filter` and
  `page_index_path` on `TwikiEOSSource` are both this shape. A second
  copy of a source in a downstream repo is what this distribution
  exists to prevent.
- **Connectors are kept line-comparable** with the frozen canonical
  `okg-deployments/cms` copies, so an upstream bug report cites matching
  lines. That is why the format check is report-only: enforcing `black`
  would reformat 58 files and destroy the correspondence.
- **`docs/okg-alignment.md` is a contract, not a status page.** Any
  change that touches `python/archi`'s okg imports, bumps the okg
  commit tested against, or moves program state must update its
  "Current state" section and its pins **in the same PR**.
  `python/tests/test_alignment_page.py` enforces the import-surface
  half mechanically; the prose half is on you. The AKMON release in
  `.github/workflows/ci.yml` must equal the release in that page's
  *Last updated* line, or the page describes a build that never ran.

## Running the suite

```bash
pip install -e "./python[dev]"
python -m pytest python/tests -q
```

That is what CI runs. From a checkout that already has `okg` importable
(a cms-kb venv, for instance), `cd python && PYTHONPATH=. python -m
pytest tests -q` does the same thing without installing.

## PACT

Changes are recorded in `pact/changes/<id>/pact.yaml`, edited directly —
the CLI is reactive. The corpus here is **v5**, git-native, and takes
the git toplevel as its root:

```bash
okg pact <verb>        # from any venv carrying okg
```

Verbs: `new`, `view`, `evidence attach`, `check`, `approve`, `digest`.
**`check` is the one gate** — it validates, binds receipts, verifies
approval and computes the tier in one call. `approve` records approval
as a `Pact-Approved` git commit trailer, not a ledger row.

**Pass `--base origin/archi_v3` to `check`.** Without it, `check`
diffs against `origin/HEAD`, which is v2 `main` here. That diff holds
the whole v3 line, so the tier and the evidence-head checks describe
the wrong change.

```bash
okg pact check <id> --base origin/archi_v3
```

Two things that will otherwise waste your time:

- **A v5 task has no status field** — only `id`, `requirement` and
  `verification`. Completion is derived from receipts bound to a
  commit, so evidence whose `head` predates the base reads as
  `evidence_head_before_base` even when it is otherwise correct. Attach
  evidence *after* committing, and rebind it if you amend.
- **`pact/project.yaml` declares `pact_version: 4` on purpose.** The v4
  compatibility reader still uses it and v5 never reads it. Do not
  "fix" it.

Every change gets a manifest, before the branch has commits on it. The
check that catches a missing one costs a command:

```bash
git diff --name-only origin/archi_v3...HEAD | grep '^pact/changes/' || echo "NO PACT"
```

## Reporting

PR bodies and closeouts, in order: (1) what this changes, one plain
sentence a reader who is not a substrate expert can follow;
(2) behaviour before → after; (3) findings, ordered by severity, each
with a `file:line`; (4) how I verified — the command and its result.

No narration of how the work went, no invented nouns, no undefined
jargon unless glossed on first use. A correct diff with an
unintelligible description is not done.

## What the old file got wrong

The old `CLAUDE.md` was deleted rather than edited because most of what
it pointed at was gone. Recorded here so nobody restores it from memory:

| it said | actually |
|---|---|
| PRs target `archi_v3`, never `main` | false in archi-okg, where `main` was the only branch; true again here since 2026-10-25 (see "Where work lands") |
| `docs/runbooks/agent-*.md` are the backing contracts | there is no `docs/runbooks/` |
| `pact/AGENTS.md` holds the PACT contract | no such file |
| `pact/project.md` is human context | no such file |
| `.github/PULL_REQUEST_TEMPLATE.md` hard-codes the section order | no such file |
| `docs/runbooks/plain-language-reporting.md` is the how | no such file |
| skills are installed under `.claude/skills/` from `src/okg/substrate/pact/skills/` | skills live in `skills/`; neither path exists |
| use `uv run okg` | there is no `uv run okg` here |
| `okg pact gate` / `decide` / `hooks` close a task | suppressed in v5; `check` is the gate |

Most of those came from a managed `okg-pact-install` block — okg's
generic boilerplate, written for okg's own layout. It is **not**
reproduced here: a block of dead paths is worse than no block. If
anyone runs `okg pact install` in this repo, check its output against
the table above before trusting it.
