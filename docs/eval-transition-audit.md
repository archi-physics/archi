# Audit: what the v3 evaluation port kept, dropped, and changed

**Status:** audit only. The v3 port (`w5-eval`, PR #619) is **frozen as draft**
pending a conversation with Antonio Battaglia. Nothing here is a decision.

**Why this exists.** The v3 evaluation package (`python/archi/eval/`) was
produced by porting Antonio's two open PRs — #596 (`feat/archi-eval-command`)
and #608 (`feat/live-eval`) — into the v3 distribution. The port dropped
capability without flagging it, so this document establishes what actually
happened, separates the losses that the program spec sanctioned from the ones
nobody sanctioned, and puts the open decisions in one place.

**Method.** Three independent read-only explorations of both sides (his code
read from `origin/feat/live-eval`, never checked out), plus a differential test
that runs **his real scoring code** against ours on identical inputs. No
changes were made to either side.

---

## 1. Headline numbers

| | Antonio (side A) | v3 port (side B) |
|---|---|---|
| Eval backend | **9,979 lines**, 23 modules (`src/evaluation/qa/` + `src/cli/qa_eval.py`) | **1,865 lines**, 5 modules (`python/archi/eval/`) |
| Tests | 8,488 lines | 1,337 lines |
| UI | 2,976 lines shipped (Flask routes + template + JS + CSS) + 25 Playwright tests | none |
| Concerns with a counterpart | — | **5 of his 23 modules** |

The port is 18.7% of his backend by line count. Of his 23 modules, 18 have no
counterpart at all: jobs, worker, workspace, workflow phases, catalog, history,
console, live-check artifacts, oracle transport, profiles, agent runtime, tool
traces, schema versioning, artifacts/digests, parallel phases.

## 2. Accountability: which losses were sanctioned

**Sanctioned by ADR 0001 (approved program spec):**

- **The browser console.** §W5(c) and the disposition table (line 284): the
  `/evaluations` console "dies with the chat app — CLI + `report.md` remain the
  surface, a console is revisited post-first-draft (candidate home: OKG
  operator console)". The v2 chat app that hosted it was deleted in PR #611.
- **The v2 agent runtime** (`runtime.py`'s `ArchiAgentRuntime`, vectorstore,
  pipeline lookup). §W5(b) explicitly calls for re-pointing the runtime adapter
  at MCP-client agents. The arm registry is that adapter, generalized.

**Not sanctioned — unexamined choices in the porting brief I wrote:**

- **The judge.** §W5 lists "per-atom judging (pass rate, atom score,
  required-atom recall)" among what #596 brings and is *kept*. His judge is
  real and portable: `LangChainEvaluatorRuntime.compare`
  (`src/evaluation/qa/runtime.py:244`), two version-pinned,
  injection-hardened prompts (`constants.py:42-87`), a provider-agnostic
  `model_factory` **injection point** (`runtime.py:200-217`), strict
  re-validation of the model's output (`validation.py:52`). The port shipped
  the interface with no implementation, so every prose question returns
  `ungraded`.
- **LLM atom generation** (`extract_gold`, `runtime.py:236`, prompt at
  `constants.py:42`). Turns a canonical answer into required/optional
  obligations. This is the mechanism that would make the comp-ops golden sets
  judgeable at all. The port has no equivalent and the omission was never
  raised as a consequence.
- **Everything operational**: jobs/cancel/resume, parallel workers, run
  history, per-run artifacts with digests, frozen-input verification, retry
  into a successor run, tool-call traces. My brief said to drop "the
  workspace/manifest/phase machinery" as v2 scaffolding. That was wrong as
  stated: `phases.py` (parallel workers), `artifacts.py` (atomic writes +
  digests), `jobs.py`, `scoring.py`'s bounded-memory aggregation and
  `tool_traces.py` are **stdlib-only and import nothing from the v2 tree**.
  They are portable library code, not app scaffolding.

**Nothing was destroyed.** His branch is intact and *ahead* of ours: #608 has
commits from 2026-08-24 (Antonio merging three fixes from Austin Swinney).

## 3. Differential test: did the port preserve his semantics?

His `score_attempt` and `validate_judgments` were loaded verbatim from
`origin/feat/live-eval` (only his dataset-only imports stubbed) and run against
our `score_gold_facts` / `_validate_judgments` on identical inputs.

**Scoring math: 8 of 9 cases byte-identical.** All-entailed, required
contradicted, required not-mentioned, optional missed, optional contradicted,
the `max(0, …)` floor, single-fact, and a mixed three-fact case all produce the
same `atom_score`, recall and `passed`. The port did faithfully carry his math.

**Two divergences, both real:**

| Case | His behavior | Our behavior | Consequence |
|---|---|---|---|
| A judge returns `unjudgeable` | Raises before scoring (`phases.py:123`) → attempt recorded `evaluation_failed`, **excluded from quality denominators**, and re-judgeable later via `retry` *without re-running the agent* | Scores it 0 → `atom_score 0.5, recall 0.5, passed False` | A judge that couldn't decide is silently counted as the *arm's* failure to mention a fact. His design is better and we should adopt it. |
| Judge returns an empty `rationale` | Rejected (`validate_nonempty_string`) | **Accepted** | We admit judgments with no stated reason; his contract requires one. Small, but it is our regression, found only by running both. |

Judge-contract validation otherwise matches: both reject missing judgments,
unknown fact ids, duplicates, and invalid outcomes.

Harness: `<scratch>/parity/` (scratch, not
committed).

## 4. Capability ledger

Verdicts are **proposals for Antonio and the maintainer**, not decisions.

| Capability | His side | Ours | Portability | Proposed verdict |
|---|---|---|---|---|
| Judge (`compare`) | `runtime.py:244` + prompts `constants.py:61` | interface only | `model_factory` injectable; needs `langchain_core`, not the chat app | **Restore — blocking.** Without it the framework cannot grade prose. |
| Atom generation (`extract_gold`) | `runtime.py:236`, `preparation.py:268` | none | same injection point | **Restore — blocking for golden sets.** |
| `unjudgeable` → `evaluation_failed` + retry | `phases.py:123`, `workflow.py:1116` | scores 0 | pure logic | **Adopt his semantics.** |
| Non-empty rationale enforced | `validation.py` | not enforced | pure logic | **Fix ours.** |
| Parallel workers, per-thread runtime reuse | `phases.py:74-184` | sequential | **stdlib only** | Restore — cheap, and large datasets need it. |
| Repeated attempts per item + pass@k | `workflow.py:539`, `scoring.py:172` | one answer per pair | pure logic | Restore — non-determinism is the point of k. |
| Artifacts: atomic writes, digests, manifest | `artifacts.py`, `workflow.py:257` | one `--output` file | **stdlib only** | Restore. |
| Frozen-input digest verification | `workflow.py:303-324`, Austin's fixes | none | stdlib | Restore with the above. |
| Bounded-memory aggregation (sqlite/ijson) | `scoring.py:53`, `history.py` | in-memory | stdlib + `ijson` | Later; fine at current scale. |
| Schema versioning + legacy readers | `schema.py`, `constants.py:12` | no version string | stdlib | Restore the version pin at least. |
| Job lifecycle, cancel, restart recovery | `jobs.py` | none | stdlib, POSIX signals | Only meaningful with a resident process — tie to the UI decision. |
| Subprocess worker | `jobs.py:217`, `worker.py` | none | stdlib | Same. |
| Pause on live drift → human continue | `workflow.py:462-503` | quarantines and continues | logic + oracle | Needs decision: his gate is stricter and safer. |
| Retry into successor run + lineage | `workflow.py:884`, `workspace.py:136` | none | stdlib + sqlite | Restore after artifacts. |
| Run history + trends | `history.py` (726 lines) | none | stdlib + `ijson` | Tie to the UI decision. |
| Dataset catalog, child datasets, approval | `catalog.py` (1,231 lines) | `load_dataset` only | stdlib + yaml | Tie to the UI decision. |
| Streaming dataset codecs, JSONL, strict JSON | `dataset.py` | JSON/YAML, no streaming | needs `ijson` | Later. |
| Oracle over real MCP: transports, auth, timeouts, redaction | `oracle_config.py` (514 lines) | injected invoker only | needs `mcp`, `anyio`, `httpx` | Decide: dependency vs injection. This is also what the `okg-mcp` arm needs. |
| Pre/post live drift check | `live_checks.py` | **equivalent present** | — | Kept. |
| Tool-call traces | `tool_traces.py`, `runtime.py:100` | none | trace model is stdlib; capture is LangChain-shaped | Restore the model; capture per arm. |
| Evaluator profiles (per-component model pins) | `profile.py` | arm configs partly overlap | stdlib + yaml | Restore — provenance needs the judge model pinned. |
| Phase CLI (prepare/run/score, `--overwrite`) | `src/cli/qa_eval.py` | `list-arms` / `run` | click vs argparse | Merge surfaces. |
| Deployment staging of oracle config | `templates_manager.py:532` | none | v2 deployment templating | Drop — v3 instances own their config. |
| RBAC (`evaluations:view/run/manage`) | `evaluation_routes.py:86` | none | chat-app coupled | Tie to the UI decision. |
| **~15 rules living only in `evaluations.js`** | `:1121` Atoms-recall metric (computed nowhere else), `:1072` question→attempt→judgment join, `:1129-1251` tool-time attribution, `:892` atom-review validation, `:128` launchability, `:216` draft state machine | none | pure logic, wrong layer | **Lift into the library regardless of who ports** — a backend-only port loses these, including his own. |

### What the port adds that he does not have

Genuinely additive, and the reason to keep the branch as a reference:

- **Arm registry** — atoms × arms in one run; he hard-wired a single tested
  agent. This is ADR §W5(b) realized.
- **Deterministic checks** (`exact`/`contains`/`regex`) — a free, reproducible
  scoring path; his scoring was LLM-judged only.
- **`value_from`** — a check's expected value pulled from live oracle state.
- **Token/cost/latency rollups** and **`okg.llm_calls`** reconciliation; he
  recorded only `duration_ms`.
- **OKG generation pinning** with cross-arm conflict detection.
- **`NotConfiguredError` that fails the run** instead of scoring zeros.
- YAML datasets, `module:attr` injection, `list-arms` discoverability.

## 5. Deviations from the approved ADR, independent of Antonio's work

1. **Judging dropped** — contra §W5 (above).
2. **No raw-document baseline arm.** §W5 practice, tracing to the Cooper A/B at
   line 37 where *the graph arm lost to a raw-note baseline*, says retrieval
   benchmarks keep a raw-document baseline "where practical". Our `raw-llm` is
   a **no-context** baseline — a weaker and more flattering comparison. The arm
   the program actually asks for (documents in context, no graph) does not exist.
3. **`queries.json` question sets not migrated** — §W5(d).
4. **CMS deployment benchmark assets not folded in as datasets** — §W5(e).

## 6. Open decisions

For the maintainer and Antonio, in rough dependency order:

1. **Who ports, and how.** Contribute the arm layer into his branch and let him
   land the whole thing on `archi_v3`? Split (he owns engine + UI, we own arms
   + OKG integration)? Something else? His branch is live, so this decides
   whether further work converges or diverges.
2. **Where the UI lives**, under "Archi holds no running services" (ADR line
   45/394): a self-contained HTML report artifact; a local, explicitly
   operator-run `archi eval serve`; OKG's operator console (the ADR's named
   candidate, OKG-side work); or CLI + markdown only as the ADR currently says.
   Note that whichever is chosen, the `evaluations.js` logic above should move
   into the library.
3. **`unjudgeable` semantics** — adopt his `evaluation_failed` + retry.
4. **Does atom generation come along?** It gates whether the comp-ops golden
   sets can be used at all.
5. **MCP oracle**: take `mcp`/`anyio`/`httpx` as dependencies of the eval extra,
   or keep the injected-invoker seam and lose transports/auth/timeouts?
6. **Where the arm abstraction belongs** — inside his engine, or as a layer
   above it.
7. **What happens to `w5-eval`** — reference implementation, closed, or
   foundation to build his work back onto.
