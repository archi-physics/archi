# CMS Answer Synthesis

Use this skill after retrieval and before writing the final answer. The goal is
to turn graph evidence into an operator-useful answer, not an evidence
inventory.

Before writing prose, classify the question into an operator knowledge type
from `operator_knowledge.md` and assemble typed evidence. The final answer
should satisfy that type's completeness contract. If the graph only supports an
inferred answer, say so explicitly instead of turning adjacent evidence into an
unstated definition or procedure.

## Operator Answer Contract

For operator-facing answers, include these answer elements in compact prose:

- `Answerability`: answered from the pinned OKG, partial from pinned OKG,
  live source required, insufficient evidence, or out of CMS OKG scope.
- `Supported finding`: the direct OKG-backed conclusion.
- `Operator action path`: exact next checks, commands, or source systems to
  inspect.
- `Decision criteria`: how outcomes change the diagnosis or next step.
- `Source authority`: which source family owns the current truth.
- `Remaining gap`: what the pinned generation could not prove.

If live data is required, do not leave the operator at a dead end. Give the
bounded source-backed check sequence that would complete the answer, and say
which result would confirm or reject the likely diagnosis.

## Source Authority

Rank source families before answering:

1. Current deployment-maintained documentation or procedure pages for normal
   operator procedure.
2. Current graph facts, monitoring snapshots, catalog/RSE facts, and generated
   views for observed state.
3. Newer operational tickets for time-sensitive status, exceptions, incidents,
   rollout state, and source conflicts.
4. Older Twiki pages and older tickets as history unless newer evidence still
   confirms them.

If a newer ticket contradicts current documentation, state the best current
answer and the conflict. Do not silently average the sources.

## Temporal Triage

Use source dates whenever the question asks for a current procedure, latest
status, deployment path, incident state, or operational recommendation.

- For JIRA evidence, use `updated` for recency and keep `created` separate.
- For documentation pages and Twiki pages, use `last_updated` when present.
- For monitoring and graph-state overlays, use `observed_at` or the snapshot
  date.
- For workflow nodes, use `updated_at` when present and `created_at` only as a
  fallback.
- When candidate sources lack timestamps, state that the generation contains
  the source but not a usable source date. Do not treat an undated source as
  newer than a dated current procedure or ticket.

Reasonable timestamp filters are part of retrieval, not post-hoc decoration:
for current/procedural/status questions, run a bounded timestamp-ranked check
over the relevant `okg.v_nodes` candidates before selecting the lead source.

## Answer Shapes

### Out Of OKG Scope

- If the question asks about this assistant, this chat, memory, platform
  logging, or session behavior, do not answer from CMS operational records.
- Say the question is outside CMS OKG scope and answer only from the runtime
  boundary available in the prompt.
- Do not search for CMS chat pages, meeting minutes, or tickets merely because
  the word `chat` appears.

### Procedure

- Start with the recommended current path.
- Include prerequisites, concrete commands/paths, where to run them, expected
  checks, and warnings.
- Put historical, site-specific, or alternate procedures after the current
  path and label them.
- If only old evidence exists, say that no current procedure was found in the
  pinned generation.

### Config Parameter

- Define the parameter directly.
- Name the consumer or system that reads it when the graph supports that.
- Include the default, typical value, allowed value, or explicitly say it was
  not found in the pinned generation.
- Separate documented semantics from inferred examples.

### Operational Term Or Process

- Define the term or process first.
- Then describe where it lives, how operators inspect/use it, inputs/outputs or
  metrics, and lifecycle/cleanup when relevant.
- Include concrete commands or paths if the source contains them; otherwise
  name that gap.

### Timeline Or Migration

- State the scope before the chronology.
- Use dated timeline events only when directly evidenced.
- Separate commissioning/testing, production readiness, cleanup, and current
  status. Do not conflate adjacent ticket families.

### Incident Or Root Cause

- Identify scope, symptoms, timeline, root cause, mitigation, current status,
  and unresolved uncertainty.
- For follow-up questions, use the prior sanitized user turns to keep the same
  incident target.
- Separate confirmed cause from suspected cause and mitigation.

### Policy Or Exception

- State the current best answer first.
- Name the documentation side and ticket/current-state side when they disagree.
- Say `source conflict` only when the pinned generation does not contain enough
  evidence to resolve the disagreement.

### Exploratory Analysis

- Provide an executive summary, ranked clusters, representative evidence,
  operational impact, and concrete next checks.
- Do not answer with a list of matching nodes or counts alone.

## Hard Fails

Do not finish with phrases like `strongest OKG evidence`, `canonical OKG
parents`, or `top hits` unless the question only asked for evidence discovery.
Those are retrieval notes, not final CMS operator answers.

Do not present a deployment-specific procedure as a generic procedure. For
example, Tier0 PyPI WMAgent deployment is not the generic WMAgent Docker
deployment unless the question explicitly asks about Tier0/T0. For generic
WMAgent deployment questions, lead with the current generic-source status; if
the pinned generation lacks a current generic Docker/container procedure, say
that before mentioning Tier0-specific or legacy procedures.

Do not answer broad WMAgent overview questions from old installation pages
alone. Start with the current operational role, then cite current Tier0,
WMCore, or Rucio-integration evidence. Label older WMAgentIntegration-style
pages as historical context when they are the only overview-like source.
