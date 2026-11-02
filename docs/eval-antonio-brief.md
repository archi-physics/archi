# Brief for Antonio: the eval framework and v3

Antonio — this is context on what changed under your evaluation work while v3
was being assembled, what we built, what we got wrong, and the decisions we
think are yours to weigh in on. Nothing is being merged while we wait.

## What changed under you

- **The v2 `src/` tree is gone on `archi_v3`** (PR #611, merged 2026-08-21).
  `src/evaluation/`, `src/interfaces/chat_app/`, the v2 build — all deleted on
  that branch line. `main` still carries it and still serves the live v2
  deployments, so nothing of yours is lost; but on `archi_v3` there is no chat
  app left to host the `/evaluations` console, and no `src.archi.providers` to
  build a judge model from.
- **Archi v3 is a pip package installed onto OKG**, not an application: it ships
  connectors, enrichers, schemas, playbooks, bundles and the evaluation suite,
  and holds "no credentials, no site config, no running services" (ADR 0001).
  Chat belongs to OKG now.
- ADR 0001 §W5 planned for your engine to be kept and its runtime adapter
  re-pointed from the in-process v2 agent to MCP-client agents, and it planned
  for the browser console to "die with the chat app… a console is revisited
  post-first-draft (candidate home: OKG operator console)".

## What we built, and why

`python/archi/eval/` (PR #619, now frozen as draft) is a port of #596 + #608
restructured around an **arm** — one answering configuration. The engine runs
atoms × arms so a single run compares, say, the OpenWebUI chatbot against an
agent driving the graph against a bare LLM, on identical questions. That was
the §W5(b) adapter generalized, and it is the one part we think is genuinely
additive: it also brought deterministic `exact`/`contains`/`regex` checks (a
free, reproducible scoring path), expected values pulled from live oracle state
by JSON pointer, token/cost/latency rollups reconcilable against OKG's
`okg.llm_calls`, and OKG generation pinning with cross-arm conflict detection.

## What we got wrong

**We re-implemented your live work.** #608 has commits from today (your merges
of Austin's three fixes). We forked a moving two-person branch without checking
whether it was active, and the port was authored as our own commits. We have
added `Co-Authored-By` trailers and named you in `docs/eval.md`, but the deeper
problem is that we should have talked to you first. Hence the freeze.

**We dropped capability we shouldn't have.** Our port kept your judge
*interface* and shipped no implementation, so gold-fact atoms silently come
back `ungraded` — even though the ADR says per-atom judging is kept. We also
have no equivalent of `extract_gold`, which is the thing that would make the
comp-ops golden sets judgeable at all. And the porting brief dismissed
`phases.py`, `artifacts.py`, `jobs.py`, `tool_traces.py` and your
bounded-memory aggregation as "v2 scaffolding" when they import nothing from
the v2 tree — they are portable library code and we were wrong about them. Full
accounting, with line references: `docs/eval-transition-audit.md`.

## Two findings you may want regardless of what we do

1. **A differential test says your scoring math survived the port intact** —
   8 of 9 cases identical, running your real `score_attempt` against ours. The
   two divergences are both ours to fix: we score `unjudgeable` as 0 (you raise
   → `evaluation_failed`, excluded from quality denominators, re-judgeable via
   `retry` without re-running the agent — your design is better), and we accept
   an empty `rationale` where you require one.

2. **Evaluation logic lives in `evaluations.js` and would be lost by any
   backend-only port — including yours.** The clearest case: the **Atoms-recall
   metric** (`macroMeanScoredAttemptAtomRecall`, `evaluations.js:1121`) is
   computed only in the browser; only the *required*-atom variant is persisted.
   Also browser-only: the question→attempt→judgment join across the three
   artifact streams (`:1072`), the tool-time vs other-agent-time attribution
   rules including the deliberate "unattributed" remainder (`:1129-1251`), the
   atom-review validation rules (`:892`), the launchability rule (`:128`), and
   the dataset action state machine (`:216`). If the console is ever rehosted or
   replaced, those rules want to be in the library first.

## Decisions we'd like your view on

1. **Who ports, and how.** Our preference is that we contribute the arm layer
   as a PR into *your* branch and you land the whole thing — engine, judge, UI —
   on `archi_v3` as its author. Alternatives: an explicit split (you own engine
   + UI, we own arms + the OKG-MCP integration), or you tell us to carry on.
2. **Where the console lives in v3**, given the no-running-services rule: a
   self-contained static HTML report; a local operator-run `archi eval serve`
   (needs an ADR amendment); OKG's operator console (OKG-side work, would go upstream);
   or CLI + `report.md` as the ADR currently says.
3. **The MCP oracle.** Your `oracle_config.py` has transports, four auth modes,
   timeouts, tool discovery and secret redaction. Ours has an injected invoker
   and none of that. Should `mcp`/`anyio`/`httpx` become dependencies of an
   eval extra? The same adapter is what our `okg-mcp` arm needs — and we'd like
   to know which tool a deployment should expose as its answering entry point.
4. **Atom generation** — does it come along? It gates the golden sets.
5. **Where the arm abstraction belongs** — inside your engine, or as a layer
   above it.

Everything of ours is on the frozen branch `w5-eval` (PR #619) with 350 passing
tests, if it is useful as a reference for the arm design. If you'd rather it be
closed, say so and we'll close it.
