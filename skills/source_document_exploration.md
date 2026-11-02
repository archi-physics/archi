# CMS Source Document Exploration

Use this skill when a question asks for a current procedure, source authority,
documentation-backed definition, repo path, URL, runbook shape, or when search
finds a promising chunk but not enough surrounding context.

## Exploration Posture

- Treat the OKG as the database to learn from. Inspect schema/source shape with
  `inspect`, `aggregate`, and bounded SQL instead of guessing
  which source family exists.
- Try independent query variants before declaring a gap: exact phrase, shorter
  noun phrase, command/config key, repo/path fragment, source family, service
  alias, ticket key, and timestamp/status verbs.
- Do not repeat an identical failed query. Change the terms, subtype filter,
  method, source metadata path, or expansion direction.
- Expand promising compact hits into broader source context before synthesis.
- When the question names a specific page, document, or ticket, or search
  surfaces one as the likely source, read its text with
  `inspect(target={"kind": "text", "node_id": "<page node id>"})` before
  answering or concluding the answer is absent. `inspect` with
  `kind: "node"` gives its metadata (title, URL) only. For a long page,
  repeat the call with the returned `continuation_token` in the target while
  `complete` is false. Prefer this read over the ordered-chunk SQL below.
  It needs an okg release from 2026-09-28 or later; an older okg refuses
  it ("inspect target kind must be one of ..."). Then fall back to Parent To
  Ordered Chunks.
- Give the best supported CMS operator answer from retrieved evidence, with
  explicit missing fields when the pinned generation lacks the current source.

## SQL Recipes

Use `query` only for bounded read-only `SELECT` or `WITH` queries over
`okg.v_*` views. Keep limits small and pass the pinned generation when the tool
accepts it. SQL view names are retrieval receipts, not final evidence IDs.

### Chunk To Parent

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

### Parent To Ordered Chunks

Fallback for okg without the text read (see Exploration Posture). `LIMIT 12`
can stop partway through a long page.

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

### Adjacent Generic Doc Chunks

Use this when the parent is a generic `document` node from doc-corpus sources:

```sql
SELECT s.doc_id,
       s.chunk_id,
       s.prev_chunk_id,
       s.next_chunk_id,
       s.chunk_index,
       s.doc_chunk_total
FROM okg.v_chunk_sequence s
WHERE s.chunk_id = '<chunk_id>'
LIMIT 1
```

Then fetch the parent or neighboring chunks through `okg.v_doc_chunks`:

```sql
SELECT chunk_id, doc_id, path, heading_path, text
FROM okg.v_doc_chunks
WHERE doc_id = '<doc_id>'
ORDER BY char_offset, chunk_id
LIMIT 12
```

### Source Metadata Search

Use this when the question names a URL/path/source family/title, or when
current-source authority matters:

```sql
SELECT node_id,
       subtype,
       attrs->>'title' AS title,
       attrs->>'url' AS url,
       attrs->>'path' AS path,
       attrs->>'source_repo' AS source_repo,
       attrs->>'last_updated' AS last_updated,
       attrs->>'updated' AS updated,
       attrs->>'created' AS created
FROM okg.v_nodes
WHERE subtype IN ('documentation_page', 'jira_issue', 'document')
  AND (
    attrs->>'title' ILIKE '%<term>%'
    OR attrs->>'url' ILIKE '%<term>%'
    OR attrs->>'path' ILIKE '%<term>%'
    OR attrs->>'source_repo' ILIKE '%<term>%'
  )
ORDER BY COALESCE(attrs->>'last_updated',
                  attrs->>'updated',
                  attrs->>'created') DESC NULLS LAST
LIMIT 20
```

## Answer Discipline

- For runbooks/procedures, read the page's text (or, on older okg, enough
  ordered chunks) to identify
  prerequisites, where to run commands, ordered steps, validation checks,
  warnings, and source date/status.
- For current-source questions, compare documentation date/status against newer
  ticket or monitoring evidence before choosing the authority.
- If only stale or context-specific evidence is found, say that clearly and
  answer only the part supported by the pinned generation.
