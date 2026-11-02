# CMS Procedure Extraction

Use this skill for procedural questions asking what command, option, policy
step, or operator workflow should be used.

## Retrieval Strategy

1. Search for exact command fragments and option names first.
2. Search documentation pages before relying on ticket examples.
3. Use tickets to recover CMS-specific examples, approvals, comments, account
   names, lifetimes, activities, and exceptional cases.
4. Use `inspect` on the documentation or ticket parent, then expand with
   `expand` and bounded `query` to supporting chunks that contain the
   exact syntax.
5. If a chunk contains the relevant syntax, fetch the parent source metadata
   and ordered/nearby chunks before deciding the procedure shape. A one-chunk
   hit is usually not enough for prerequisites, where to run the command,
   validation checks, and warnings.
6. If command syntax is spread across evidence, separate canonical syntax from
   examples and policy-dependent values.

## Answer Discipline

- Return the command or procedure directly, then explain required placeholders.
- Do not merge incompatible examples into one invented command.
- Mark optional or policy-dependent flags explicitly.
- Cite documentation IDs for syntax and ticket IDs for CMS examples.
- Include source date/status when answering current deployment procedures.
- If only ticket examples exist, state that no canonical documentation page was
  found in the pinned generation.
