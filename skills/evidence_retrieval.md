# CMS Evidence Retrieval

Use this skill for CMS operator questions where the answer should be grounded in
OKG nodes, edges, document chunks, tickets, views, or monitoring snapshots.

## Retrieval Loop

1. Pin the generation explicitly.
2. Use `search` for discovery.
3. Inspect canonical nodes with `inspect`.
4. Roll chunks up to their parent JIRA issue or documentation page with
   `expand`.
5. Traverse from operational entities to adjacent evidence: site, storage
   endpoint, dataset, workflow, monitoring snapshot, incident ticket, and
   documentation page.
6. Cite stable OKG IDs in the answer.

## Evidence Quality

- Prefer parent tickets and documentation pages over isolated chunks when
  giving the final citation list.
- Use chunks as supporting evidence when the answer depends on exact prose.
- Use `query` only for bounded read-only view-layer checks over `okg.v_*`.
- Distinguish "not found" from "found but not ranked in first-pass search".
