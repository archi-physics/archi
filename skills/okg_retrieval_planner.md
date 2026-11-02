# CMS OKG Retrieval Planner

Use this plan before answering:

1. Start with exact identifiers and short query variants. Use site names, RSEs,
   datasets, workflow names, command fragments, config keys, JIRA keys, service
   names, and aliases from retrieval hints.
2. If the first search is weak, do not repeat it. Try independent variants:
   exact phrase, acronym expansion, service/repo/path term, source-family term,
   ticket key, config key, command fragment, and timestamp/status verbs.
   Use `search(method="lexical")` and exact `search(method="alias")` by
   default. Semantic search is available only when the instance's MCP
   server explicitly has an embedder; do not call
   `search(method="semantic")` unless the instance provides one.
3. Inspect promising canonical candidates with `inspect`. When search returns
   chunks, roll them to parent issues or documentation pages before treating
   them as final evidence.
   When the question names a specific page, document, or ticket, or search
   surfaces one as the likely source, read its text with
   `inspect(target={"kind": "text", "node_id": "<page node id>"})` before
   answering or concluding the answer is absent. `inspect` with
   `kind: "node"` on a page returns metadata (title, URL) only. For a long
   page, repeat the call with the returned `continuation_token` in the target
   while `complete` is false. The text read needs an okg release from 2026-09-28
   or later; an older okg refuses it ("inspect target kind must be
   one of ..."). Then fall back to the ordered-chunk recipe (Parent To
   Ordered Chunks in `source_document_exploration.md`).
4. Expand compact hits into source context. For a promising chunk, use
   `expand` or bounded `query` to find its parent document/ticket, then read
   the parent's text with the text read above. Prefer it over the `LIMIT 12`
   ordered-chunk SQL recipe, which can stop partway through a long page; use
   that recipe only when the text read is refused.
5. Use bounded SQL as a retrieval primitive, not as final evidence. Keep SQL
   view rows in `evidence_notes`; cite real OKG node IDs from the rows.
6. Expand one promising candidate when evidence exists. Use one-hop `expand`,
   typed-path `expand`, `aggregate`, `search(method="identifier_mentions")`, or
   bounded `query` over `okg.v_*` views.
7. For count, impact, distribution, or high-cardinality RSE/dataset questions,
   aggregate before listing rows.
8. For current, latest, status, deploy, or how-to questions, run a reasonable
   timestamp-ranked check over candidate canonical nodes. Prefer
   `jira_issue.updated`, `documentation_page.last_updated`,
   `monitoring_snapshot.observed_at`, `workflow.updated_at`, then subtype
   `created` or `date` fields.

## SQL Document And Source Recipes

Use these patterns with `query` only as bounded read-only `SELECT` or `WITH`
queries over `okg.v_*` views. Replace IDs/terms with the current question's
values and keep `LIMIT` small.

The `generation_id` tool argument pins `okg.v_*` reads. Do not add
`generation_id = ...` predicates inside SQL over `okg.v_nodes` or
`okg.v_edges`; those views do not expose a `generation_id` column.

When selecting optional tool fields:

- `search(method="lexical")`: use `node_id`, `subtype`, `attrs`, `text_hash`,
  `score`, or omit `select`.
- `search(method="alias")`: use `alias`, `canonical`, `scope`, `match_kind`,
  `node`, or omit `select`; do not request `node_id`, `subtype`, or `attrs`.
- `expand`: use `edge_id`, `src`, `dst`, `edge_type`, `attrs`,
  `direction`, or omit `select`; do not request `target_id` or
  `target_subtype`.

Avoid broad whole-graph aggregates such as grouping all `okg.v_nodes` without a
selective predicate; they can exceed the SQL timeout. Prefer exact IDs, subtype
filters, small `LIMIT`s, or `aggregate` around a known node.

Find a chunk's parent source:

```sql
SELECT p.node_id AS parent_id,
       p.subtype AS parent_subtype,
       p.attrs->>'title' AS title,
       p.attrs->>'url' AS url,
       p.attrs->>'path' AS path,
       p.attrs->>'source_repo' AS source_repo,
       p.attrs->>'updated' AS updated,
       p.attrs->>'last_updated' AS last_updated
FROM okg.v_edges e
JOIN okg.v_nodes p ON p.node_id = e.src
WHERE e.edge_type = 'contains'
  AND e.dst = '<chunk_id>'
LIMIT 5
```

Fetch ordered chunks for a CMS documentation page, JIRA issue, or TWiki parent:

```sql
SELECT e.src AS parent_id,
       c.node_id AS chunk_id,
       e.attrs->>'chunk_index' AS chunk_index,
       c.attrs->>'heading_path' AS heading_path,
       c.attrs->>'text' AS text
FROM okg.v_edges e
JOIN okg.v_nodes c ON c.node_id = e.dst
WHERE e.edge_type = 'contains'
  AND e.src = '<parent_id>'
  AND c.subtype = 'document_chunk'
ORDER BY COALESCE((e.attrs->>'chunk_index')::int,
                  (c.attrs->>'char_offset')::int,
                  0),
         c.node_id
LIMIT 12
```

Search source metadata when URL/path/title/source family matters:

```sql
SELECT node_id,
       subtype,
       attrs->>'title' AS title,
       attrs->>'url' AS url,
       attrs->>'path' AS path,
       attrs->>'source_repo' AS source_repo,
       attrs->>'last_updated' AS last_updated,
       attrs->>'updated' AS updated
FROM okg.v_nodes
WHERE subtype IN ('documentation_page', 'jira_issue', 'document')
  AND (
    attrs->>'title' ILIKE '%<term>%'
    OR attrs->>'url' ILIKE '%<term>%'
    OR attrs->>'path' ILIKE '%<term>%'
    OR attrs->>'source_repo' ILIKE '%<term>%'
  )
ORDER BY COALESCE(attrs->>'last_updated', attrs->>'updated', attrs->>'created') DESC NULLS LAST
LIMIT 20
```

For generic doc-corpus sources, `okg.v_documents`, `okg.v_doc_chunks`, and
`okg.v_chunk_sequence` provide parent document metadata and previous/next chunk
IDs. Prefer those views when a hit has `document` / `document_chunk` membership;
prefer the `okg.v_edges` `contains` recipe for CMS `documentation_page`,
`jira_issue`, and `twiki:CMS:*` parents.

If retrieval finds only adjacent evidence, label the answer as inferred and
lower confidence. If no current source is found, state the missing current
source instead of filling it from older or context-specific evidence.

## Live Authority Checks

For current-state questions, first discover the graph-native live authority
surface before calling any external live tool:

1. Search for `tool_authority_profile` and `external_live_tool` using the
   source family and task terms, such as `Rucio live aggregate`,
   `Condor current failures`, or `MONIT transfer count`.
2. Inspect the profile/tool node with `inspect` and, when useful, traverse
   neighbors to the authority, service, or endpoint.
3. If live tools are enabled and the profile applies, cite the profile/tool ID
   before calling the matching `cms_monit_*` tool. The profile ID is the
   applicability proof; the tool ID is only capability support.
4. If the profile applies but the named live tool is unavailable at runtime,
   report the missing tool or credential condition. Do not treat the profile as
   a live result.
5. If live tools are not enabled, use the profile as graph evidence for the
   missing live source family and mark the answer `live_source_required`.

Use these authority families when choosing search terms:

- CRAB task state: CRAB task metadata, scheduler state, Condor/job evidence.
- Production workflow state: ReqMgr2, Unified, WMAgent, job reports, monitoring
  snapshots.
- Storage/quota/replica state: Rucio account, RSE, rule, quota, replica, or
  catalog facts.
- Incident status: current JIRA status/comments plus newer monitoring or
  workflow facts.

If the authoritative live family is absent, record it in `missing_fields` and
write the final answer as partial: the OKG-supported historical/context finding
plus the exact live checks and decision criteria needed to finish the diagnosis.
