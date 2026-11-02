# CMS Condor Monitoring Snapshot

Use this skill for questions about jobs, workflow execution, failures, CPU,
memory, wall clock, queueing, sites, or Condor monitoring.

## OKG Mapping

The CMS OKG does not expose live HTCondor OpenSearch. It exposes
generation-pinned graph evidence such as `monitoring_snapshot:*`,
`workflow:*`, `site:*`, JIRA tickets, documentation pages, and read-side
views. Treat those as historical or snapshot evidence for the queried
generation.

## Retrieval Strategy

1. Search exact workflow names, campaign names, site names, job-type terms, and
   error phrases from the question.
2. Use `inspect` for `monitoring_snapshot`, `workflow`, and `site`
   when unsure which attributes or edge routes exist.
3. Use aggregation or bounded `query` for count, distribution, top-error, or
   "how many" questions.
4. Use `expand` from workflow or site candidates to incident tickets,
   monitoring snapshots, and related datasets.
5. If the question asks what a monitoring field means, how to query it, or how
   to interpret an operator runbook, route through source/document exploration:
   search docs, inspect parent source metadata, and fetch nearby/ordered chunks.
6. Cite snapshot IDs when using monitoring evidence and tickets/docs when using
   operational explanations.

## Answer Discipline

- State whether evidence is a historical snapshot, ticket report, or catalog
  fact.
- Do not present generation-pinned evidence as a live reading.
- For current/live-state questions, give the best generation-pinned snapshot
  answer and explicitly say it is not a live OpenSearch reading.
- If root cause or current status is not explicit, say the graph does not
  contain that conclusion.
