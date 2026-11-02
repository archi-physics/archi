# CMS CLI Agent Benchmark Mode

Use this skill when a CMS benchmark question should be answered by a real
external CLI agent connected to the CMS OKG MCP server.

## Leakage Boundary

- Accept only sanitized fields: `question_id`, `category`, and `question`.
- Do not read, request, summarize, or infer from comparator answers,
  comparator sources, baseline answer files, or private evaluator reports.
- Keep comparator access in the separate redacted evaluator only.

## Agent Harness

- The harness launches a real provider CLI, not an in-process traversal policy.
- Supported providers are Claude Code CLI and Codex CLI; tests use a fake CLI
  with the same stdin/stdout JSON contract.
- Live provider runs require operator acknowledgement because prompts and MCP
  query results can leave the local machine through the provider.
- The MCP server is named `cms_okg` and must be pinned to the requested
  generation whenever a tool accepts `generation_id`.

## Answer Contract

For every question with OKG evidence:

1. Run multiple query variants unless an exact canonical answer is found and
   verified.
2. Inspect canonical parents for chunk hits. When the question names a
   specific page, document, or ticket, or search surfaces one as the likely
   source, read its text with
   `inspect(target={"kind": "text", "node_id": "<page node id>"})` before
   answering or concluding the answer is absent; `kind: "node"` gives only
   its metadata. Follow `continuation_token` while `complete` is false.
   The text read needs an okg release from 2026-09-28 or later; an
   older okg refuses it ("inspect target kind must be one of ..."). Then
   fall back to the ordered-chunk recipe (Parent To Ordered Chunks in
   `source_document_exploration.md`).
3. Perform at least one expansion action: neighbor traversal, path traversal,
   alias/entity lookup, source-family search, or bounded `okg.v_*` SQL.
4. Rerank evidence clusters by canonical source ID, not by chunk rank alone.
5. Return real evidence IDs, traversal receipts, query variants, tool-call
   count, confidence, and any gaps.
6. Classify the operator knowledge type and assemble typed evidence before
   writing the final answer. Include `operator_knowledge_type`,
   `typed_evidence`, and `answer_completeness` when the harness requests them.

For current/latest/status/deployment questions, include one timestamp-ranked
retrieval pass over candidate canonical nodes. Use `updated`, `last_updated`,
`observed_at`, `updated_at`, `created`, or `date` according to subtype, and
record the source dates in evidence notes when they affect source authority.

Early stop is allowed only when the answer includes a verified canonical ID,
the requested value or command, and a note explaining why further traversal
would not change the answer.

## Evidence IDs

Prefer canonical IDs in final answers:

- `jira:*` for JIRA issues.
- `twiki:CMS:*` or other documentation page IDs for docs.
- `site:*`, `se:*`, `monitoring_snapshot:*`, `dataset:*`, and `workflow:*` for
  operational graph evidence.

Chunk IDs are supporting evidence. A chunk-only answer is weak unless no
parent/source ID exists.

`okg.v_*` SQL view names are not evidence IDs. Use them in traversal receipts
or evidence notes to describe how rows were found, but cite the canonical rows
or chunk IDs that were actually observed as evidence.
