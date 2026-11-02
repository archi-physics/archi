# CMS Operator Knowledge Modeling

Use this skill before answer synthesis. Its purpose is to turn retrieved CMS
evidence into typed operator knowledge, so the final answer is not improvised
from chunks.

## Knowledge Types

Classify each question into one primary type before writing the answer:

- `procedure`: an operator workflow with ordered actions.
- `command`: executable syntax with placeholders and policy-dependent flags.
- `config_parameter`: a named field, option, or campaign parameter.
- `operational_term`: a CMS term, service, probe, component, role, or account.
- `operational_process`: a lifecycle/process such as production output,
  deletion, subscriptions, rule creation, cleanup, or monitoring flow.
- `timeline`: migration, commissioning, rollout, cleanup, or transition.
- `policy_rule`: documented policy, exception, or conflict.
- `incident`: symptoms, root cause, mitigation, status, and uncertainty.
- `definition`: stable factual lookup.
- `out_of_scope`: assistant/session/meta questions.

## Typed Evidence Assembly

For each useful source, extract compact typed facts before composing prose.
Record these facts in `typed_evidence` when the output schema allows it.

Each typed fact should include:

- `type`: `procedure_step`, `command_example`, `config_parameter`,
  `timeline_event`, `policy_rule`, `term_definition`, `warning`,
  `validation_check`, `source_authority`, or `gap`.
- `claim`: one short source-backed statement.
- `source_id`: a real OKG ID, preferably a canonical parent.
- `status`: `current`, `historical`, `inferred`, `conflict`, or `unknown`.

Do not treat a chunk match as a typed fact until the parent source has been
inspected or the parent cannot be found.

## Completeness Contracts

Use the contract for the selected type. If a required field is missing, do one
more targeted retrieval pass. If it remains missing, state the gap instead of
filling it from general memory.

### Procedure

Required fields:

- purpose or scope;
- prerequisites;
- ordered steps;
- exact commands or paths when available;
- where to run them;
- validation/check after running;
- warnings and destructive-action caveats;
- source authority and current-vs-historical status.

### Command

Required fields:

- canonical syntax;
- placeholders and what to replace;
- optional/policy-dependent flags;
- one source-backed CMS example when available;
- verification or inspection command;
- documentation source and example-ticket source separated.

### Config Parameter

Required fields:

- definition;
- consumer or code/system that reads it;
- default, typical value, or explicit `not found`;
- operational effect;
- allowed values or observed examples;
- current authoritative source;
- whether any part is inferred.

### Operational Term

Required fields:

- concise definition;
- service/system where it lives;
- how operators use or inspect it;
- inputs/outputs or emitted metrics when relevant;
- current source and historical context if older evidence is used.

### Operational Process

Required fields:

- what initiates the process;
- actors/accounts/services involved;
- creation path and lifecycle;
- disk/tape/storage destination or state transition when relevant;
- cleanup/removal path;
- policy exceptions and current caveats.

### Timeline

Required fields:

- scope of the migration/rollout;
- dated timeline events only when directly evidenced;
- blockers and mitigations;
- cleanup or rollback state;
- current status at the pinned generation;
- unresolved gaps.

## Source Authority

Exact source beats adjacent evidence:

- If the question names an exact term, parameter, service, or process, first
  prove that exact source exists in the pinned generation.
- If only adjacent tickets or loosely related docs exist, label the answer as
  inferred and lower confidence.
- Do not let a newer adjacent ticket silently replace a current documentation
  definition unless the ticket directly updates that definition.

## Final Self-Check

Before returning JSON, answer these checks:

- Does the final answer contain the concrete command, path, parameter meaning,
  or timeline the question asked for?
- Are current, historical, inferred, and missing facts labeled?
- Are source IDs canonical where possible?
- Did any source conflict or missing exact documentation get surfaced plainly?

If the self-check fails, revise once. If it still fails, keep the answer short,
state the missing typed evidence, and lower confidence.
