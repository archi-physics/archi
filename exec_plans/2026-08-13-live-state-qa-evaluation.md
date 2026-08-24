# Implement static and live-state QA evaluation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Edge Cases & Unspecified Behavior`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

After this change, Archi operators can evaluate fixed questions and time-sensitive questions whose correct answers come from evaluator-owned, read-only MCP tools. Authors store portable inline recipes in explicit Dataset V2 files; administrators separately configure evaluator MCP connections and environment-held credentials. Operators resolve and review live truth before approving an immutable child dataset, while each run checks that truth before and after agent attempts and scores only answers whose live baseline remained unchanged.

The observable result is a `qa-v2` CLI or console run with a materialized live answer, ordered `live_checks.jsonl` evidence, ordinary free-form agent answers, terminal live-validation failures where current truth is unsafe to score, and compatible history for existing `qa-v0` and `qa-v1` runs. The complete behavior is normative in `docs/prds/live-state-qa-evaluation.md`; this plan is an execution and evidence record, never an alternative specification.

## Progress

- [x] (2026-08-13 10:00Z) Created `feat/live-eval` from the current `feat/archi-eval-command` commit while preserving the dirty worktree and avoiding the reflog.
- [x] (2026-08-13 10:15Z) Read repository guidance, implementation workflow references, OpenSpec change records, installed MCP 1.27.2 interfaces, and the complete normative PRD.
- [x] (2026-08-13 10:20Z) Replaced obsolete evaluation ExecPlans and reset misleading OpenSpec implementation completion marks.
- [x] (2026-08-13 13:30Z) Implemented and tested the version-aware bounded-memory Dataset V1/V2 gateway and canonical semantic models.
- [x] (2026-08-13 14:20Z) Implemented and tested strict inline recipes, canonical JSON/pointers, diagnostics, evaluator MCP registry, and lazy stdio/HTTP invocation.
- [x] (2026-08-13 15:10Z) Implemented and tested live materialization, immutable catalog children, static-only scope, refresh, add-live, and lineage.
- [x] (2026-08-13 16:30Z) Implemented and tested `qa-v2` preparation, pre/post checks, global score barrier, artifacts, retries, summaries, and compatibility readers.
- [x] (2026-08-13 17:15Z) Implemented CLI flags and console job/API/UI/RBAC behavior and updated operator and configuration documentation.
- [x] (2026-08-13 17:45Z) Passed the initial full unit/browser verification and deterministic real stdio/HTTP fake-MCP integration.
- [x] (2026-08-13 20:45Z) Completed PRD, code-shape, and repeated independent intent-alignment reviews; fixed security, integrity, compatibility, streaming, retry, permission, and compact-projection findings until both reviews were clean.
- [x] (2026-08-13 21:00Z) Passed 858 unit tests, 24 Playwright cases on a clean isolated server, the manager-without-run browser regression, scoped formatting/import checks, JavaScript syntax, OpenSpec, strict docs, CLI help, and diff checks.
- [x] (2026-08-13 21:20Z) Completed the separate standard code review; fixed legal empty metadata selection and multi-live retry ordering, added regressions, and finished with no findings.
- [x] (2026-08-13 21:25Z) Committed the feature-only changes on `feat/live-eval`; preserved all unrelated dirty worktree files and recorded final verification evidence.

## Surprises & Discoveries

- Observation: the current worktree contains Python bytecode caches naming live-evaluation modules and tests, but the corresponding source and permanent tests are absent and Git reports no tracked deletion.
  Evidence: `find src/evaluation/qa tests/unit/evaluation/qa -name '*.pyc'` includes `dataset`, `oracle`, `oracle_config`, `live_checks`, and live tests, while the matching `.py` files do not exist.
- Observation: `openspec/changes/add-live-state-qa-evaluation/tasks.md` claimed 18 of 19 tasks complete even though those source and test files are absent.
  Evidence: the only unchecked item was review task 6.4 before this plan reset.
- Observation: the installed local MCP SDK is version 1.27.2 and exposes the standard `ClientSession`, `stdio_client`, `streamable_http_client`, `CallToolResult`, and output-schema validation needed by the PRD.
  Evidence: local `.venv` package metadata and inspected signatures; no online contract was substituted for the installed version.
- Observation: a complete repository-wide Black check is not a usable feature gate because 126 pre-existing Python files outside this implementation fail the current formatter.
  Evidence: `uv run black --check src ...` reported 126 unrelated files; the feature-owned Python files pass, except `src/interfaces/chat_app/app.py`, whose surrounding pre-existing file formatting fails while the added six-line block is Black-compatible.
- Observation: the first reused browser-test server carried catalog state from an earlier run and caused the mutable full-flow fixture to deduplicate its import.
  Evidence: the clean isolated server rooted at `/tmp/archi-eval-ui.1kjKLe` passed all 23 Playwright cases; no product defect reproduced with isolated state.
- Observation: independent frontend review found that a catalog refresh replaced manager-only attention evidence with the run-only projection.
  Evidence: the polling path now restores the authorized detailed job after catalog refresh, with a permanent Playwright regression that opens the approved/current evidence and exercises continuation.

## Edge Cases & Unspecified Behavior

The PRD specifies the feature's material edge cases. Any genuinely unspecified behavior discovered during implementation must be recorded here with its chosen handling and exact permanent test before the related progress item can be completed.

## Decision Log

- Decision: branch in the current worktree instead of creating Treehouse isolation.
  Rationale: the user explicitly requested a branch from the current branch and named it `feat/live-eval`; branching in place preserves the exact dirty state as the base.
  Date/Author: 2026-08-13 / Codex.
- Decision: ignore bytecode caches and prior completion claims as behavioral authority.
  Rationale: the user explicitly designated the PRD as the source of truth, source/tests are missing, and cache recovery would not prove conformance.
  Date/Author: 2026-08-13 / Codex.
- Decision: retain the approved OpenSpec proposal/design but reset implementation tasks rather than create a duplicate change.
  Rationale: `add-live-state-qa-evaluation` already expresses the same PRD-backed capability and passed its approval gate; only its progress state was inaccurate.
  Date/Author: 2026-08-13 / Codex.
- Decision: use the installed MCP Python SDK directly and validate all file/provider input into frozen dataclasses and closed enums at the boundary.
  Rationale: this matches the PRD, repository edge-validation rule, and installed runtime without adapting tested-agent MCP configuration.
  Date/Author: 2026-08-13 / Codex.
- Decision: preserve the existing console's 25 MB bounded upload/review contract while making the dataset gateway, workflow artifacts, validation passes, history aggregation, and MCP resolution streaming.
  Rationale: the browser atom editor intentionally presents one bounded review draft, while evaluator execution and child/run artifact processing avoid unbounded whole-file reads and retain only explicitly bounded or indexed state.
  Date/Author: 2026-08-13 / Codex.
- Decision: treat `attention_required` as a persisted pause, not an active evaluation, and show the exact live-precheck state while an evaluation process is queued or running.
  Rationale: this makes worker ownership and the no-agent-before-precheck guarantee observable, and prevents run-only operators from being directed to a manager-only action.
  Date/Author: 2026-08-13 / Codex.

## Outcomes & Retrospective

Implementation and all review loops are complete pending commit. The branch now contains strict Dataset V2/oracle/MCP boundaries, immutable live/static children, the staged and composite live lifecycle, ordered and hash-verified evidence, live-aware scoring/retry/history, authorized CLI flags, RBAC-safe console actions, operator documentation, and permanent unit/browser coverage. Existing Dataset V1, pre-integrity catalog entries, historical inline drafts, and `qa-v0`/`qa-v1` readers remain covered by the full suite.

Verification evidence: 858 unit tests passed with five dependency/runtime warnings before the final two code-review regressions; the final feature-focused suite passed 318 tests and the reviewer independently passed all 272 QA tests. All 24 Playwright cases passed on a clean isolated server, including the production `qa-v2` stdio-MCP materialize/approve/pre-check/continue/refresh/retry flow; the changed manager-without-run permission case passed separately. Scoped Black and isort, JavaScript syntax, `git diff --check`, strict OpenSpec validation, strict MkDocs build, and CLI option help passed. The docs build reports pre-existing broken-anchor INFO messages but succeeds under `--strict`. Repository-wide Black remains skipped as a feature gate because of the documented 126-file baseline drift.

## Context and Orientation

The existing static evaluator lives under `src/evaluation/qa/`. `validation.py` currently parses only the historical headerless dataset; `preparation.py` freezes static gold; `workflow.py` owns prepare, run, score, composite, and retry phases; `schema.py` and `workspace.py` validate run artifacts; `catalog.py`, `jobs.py`, `console.py`, and `history.py` support the Flask operator console. `src/cli/qa_eval.py` exposes staged and composite commands. `src/interfaces/chat_app/evaluation_routes.py`, `templates/evaluations.html`, the evaluation CSS/JavaScript, and deployment configuration expose the browser surface.

Dataset V1 means the historical headerless JSON array or JSONL row stream. Dataset V2 means an explicit `qa-dataset-v2` envelope/header whose canonical items are static, unresolved live, or materialized live. A definition parent is an immutable unresolved V2 dataset. A materialized child is an immutable locally approved complete or static-only snapshot. A live check is one pre-run or post-run observation of a materialized live item. A terminal live-validation failure is an attempt slot that was deliberately not quality-scored because current live truth could not safely be proven equal to the approved answer.

The new ownership boundaries are deliberate. A single dataset gateway owns physical codecs, version resolution, row readers, row numbering, and unique IDs. `oracle.py` owns recipes, selection, normalization, canonical hashes, bounded diagnostics, and resolution. `oracle_config.py` owns strict evaluator-only connection configuration and lazy clients. `live_checks.py` owns validated observations and failure projections. Preparation and workflow consume those typed boundaries; tested-agent runtime never sees any evaluator registry, recipe, truth, atoms, metadata, or call output.

## Plan of Work

First replace the direct dataset parser with a facade whose JSON and JSONL codecs stream raw rows, whose closed resolver accepts only exact historical V1 shapes or explicit V2, and whose V1/V2 readers produce a shared frozen semantic item. Keep one unique-ID set and use two reads of an immutable staged or trusted hash-verified snapshot around any external calls. Preserve exact V1 IDs, atoms, visible skips, and public errors.

Then add strict frozen recipe and connection models. Implement RFC 6901 selection, strict duplicate-key/non-finite JSON parsing, canonical serialization and SHA-256, selected answer/metadata aggregation, 4,096-scalar redacted diagnostics, and exactly ordered per-call evidence. Build lazy stdio and streamable-HTTP sessions directly on MCP 1.27.2, with environment-only bearer/basic/OAuth credentials, exact tool discovery, one 120-second timeout, and no retries or agent-config fallback.

Extend preparation records and catalog drafts so V2 unresolved live items resolve and atomize, trusted materialized children carry approved answer/atoms, and `--skip-live` creates run-scoped static membership without altering input. Publish complete/static-only children atomically with hashes and lineage; refresh/add-live must validate parent/static carry-forward before MCP, resolve every parent live row, show prior/current state, and create a sibling only after approval.

Upgrade new workspaces to `qa-v2`. Run every pre-check in dataset order before agent calls, persist resolved mismatches rather than discarding them, schedule only admitted membership, complete the global post-check barrier before comparison, and emit one terminal row per slot. Keep answers only for actual executions. Update retries so any live failure reruns the whole question lifecycle while unrelated terminal evidence is copied unchanged. Extend summaries, reports, manifests, history, and projections while continuing to read `qa-v0` and `qa-v1` without mutation.

Finally wire only the PRD-authorized CLI options and console actions. Persist `attention_required` without holding the provider lock, recheck before delayed continuation, enforce run/manage disclosure boundaries, add refresh/add-live/advisory status UI, and document dataset authoring plus evaluator connection deployment. Permanent tests must cover each PRD test-map row, including deterministic local stdio and streamable-HTTP fake servers and real browser flows.

## Concrete Steps

All commands run from `/home/antonio/GSOC/archi` on branch `feat/live-eval`.

Run the tightest contract tests while implementing:

    uv run pytest tests/unit/evaluation/qa/test_dataset_gateway.py -v --tb=short
    uv run pytest tests/unit/evaluation/qa/test_oracle_config.py tests/unit/evaluation/qa/test_oracle_results.py tests/unit/evaluation/qa/test_oracle_stdio_integration.py -v --tb=short
    uv run pytest tests/unit/evaluation/qa/test_preparation.py tests/unit/evaluation/qa/test_live_workflow.py -v --tb=short
    uv run pytest tests/unit/evaluation/qa tests/unit/test_evaluation_routes.py -v --tb=short

Then run the full and static verification:

    uv run pytest tests/unit/ -v --tb=short
    uv run black --check src/evaluation/qa src/cli/qa_eval.py tests/unit/evaluation/qa tests/unit/test_evaluation_routes.py
    uv run isort --check src/evaluation/qa src/cli/qa_eval.py tests/unit/evaluation/qa tests/unit/test_evaluation_routes.py
    openspec validate add-live-state-qa-evaluation --strict --no-interactive
    uv run mkdocs build --strict
    uv run archi eval qa --help
    uv run archi eval qa prepare --help
    uv run archi eval qa run --help
    uv run archi eval qa score --help

The deterministic smoke and browser commands will be recorded here when their fixture entrypoints are confirmed from the current source. They must start the actual application locally, run materialize/approve/pre-check/continue/refresh/retry/history flows, and produce inspected desktop/mobile screenshots plus console/network/accessibility evidence.

## Validation and Acceptance

Acceptance is exactly the PRD completion list. V1 behavior and old run readers must not regress. Every supported V2 shape and failure must have permanent tests. Staged and composite artifacts must be semantically equivalent. A completed run must re-read and rescore offline without MCP. Live truth that is stale, changed, unavailable, malformed, or secret-bearing must never be counted as agent quality or existing technical failure. Parent datasets and completed artifacts must remain byte-identical across child materialization, evaluation, and retry. Deterministic local MCP smoke and console browser flows must pass with no unresolved functional or accessibility finding.

## Idempotence and Recovery

Dataset and catalog writes use temporary siblings, validation, hashing, and atomic rename. Phase overwrite removes only named evaluator-owned files. Interrupted phases are rerun, not resumed. Existing immutable parents, children, completed artifacts, unrelated dirty files, and unknown workspace files remain untouched. If a test or review fails, update this plan, apply the smallest fix, and rerun the affected verification/review loop before proceeding.

## Artifacts and Notes

The normative contract is `docs/prds/live-state-qa-evaluation.md`. The approved change record is `openspec/changes/add-live-state-qa-evaluation/`. The current branch intentionally also contains pre-existing uncommitted research, PRD, configuration, lockfile, and local-example files; preserve them unless they are explicitly required by this feature. The two prior evaluation ExecPlans were removed at the user's request because they described superseded static/console milestones and conflicted with the single current PRD.

## Interfaces and Dependencies

Use existing Python/Click/Flask/Jinja/vanilla-JavaScript patterns and existing dependencies. The local environment provides MCP 1.27.2, Pydantic 2.11.7, ijson 3.5.1, httpx, anyio, and PyYAML. No new runtime dependency is currently justified.

`DatasetGateway.read(path)` must return a descriptor and bounded iterator of canonical `DatasetItem` values. `OracleResolver.resolve(item.oracle)` must return normalized answer, answer hash, metadata, and ordered call evidence. `EvaluatorMCPRegistry` must load strict YAML without connecting and resolve aliases lazily. `PreparationRecord` must represent V1 skip, preparation failure, static prepared, and materialized-live prepared shapes. `QAWorkflow.prepare`, `run`, `score`, `composite`, and `retry` must retain their public staged ownership while adding only the PRD-authorized MCP and live arguments. Console catalog/job/history services and the Blueprint must continue using server-generated IDs and typed service methods, never client paths.

Revision note (2026-08-13): created from the normative live-state PRD, removed the two superseded ExecPlans, reset inaccurate implementation progress, and recorded the current source/cache discrepancy without consulting the reflog.
