# CMS Policy Exception Arbitration

Use this skill when documentation policy and operational tickets may disagree,
or when the question asks about exceptions, special campaigns, copy counts,
retention, custodial placement, or current policy status.

## Evidence Ranking

1. Current documentation or policy page for the general rule.
2. Newer tickets for exceptions, rollout status, or incomplete cleanup.
3. Catalog, site, RSE, dataset, or monitoring graph facts for current
   generation state.
4. Older tickets only as history unless confirmed by newer evidence.

## Arbitration Pattern

- Search for the policy term and the named exception separately.
- Inspect both the policy document parent and exception ticket parent.
- If either side is only a chunk hit, use bounded SQL or neighbor traversal to
  fetch parent source metadata and ordered/nearby chunks before comparing.
- Compare timestamps: creation, update, and operator-stated effective dates.
- Use neighbors or bounded SQL to check whether graph state confirms the
  exception.
- Return `source conflict` when sources disagree and no authoritative
  resolution exists in the pinned generation.

## Answer Discipline

- Do not collapse "planned", "in progress", and "completed" into the same
  status.
- Name the source family that supports each side.
- Prefer "the graph shows X, but does not confirm completion of Y" over a
  forced yes/no answer.
