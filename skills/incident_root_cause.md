# CMS Incident Root-Cause Triage

Use this skill for questions asking why something happened, what caused an
incident, or what the root cause was.

## Retrieval Strategy

1. Search the literal incident terms, affected site/RSE/workflow/dataset, and
   error phrases.
2. Search root-cause language: `root cause`, `caused by`, `due to`, `reason`,
   `suspected`, `resolved`, `mitigated`, `workaround`, and `postmortem`.
3. Inspect the best ticket or documentation parent, then expand to linked
   tickets, affected sites, workflows, datasets, and monitoring snapshots.
4. If several candidate incidents match, cluster evidence by canonical ticket
   or affected entity before answering.
5. Use bounded SQL to confirm candidate IDs, timestamps, statuses,
   source-family coverage, and surrounding chunks when the root-cause statement
   appears inside a long ticket or documentation page.

## Answer Discipline

- Distinguish confirmed cause, suspected cause, mitigation, and symptom.
- If the graph has multiple plausible incidents, say which one best matches and
  why.
- If tickets say the cause was unknown, preserve that conclusion rather than
  inventing a cause.
- Cite the ticket or document that explicitly states the cause; do not infer
  root cause from monitoring symptoms alone.
