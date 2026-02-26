## Context

The current A/B system is entirely client-driven: `chat.js` picks two provider/model combos, streams both, and records the result. The server is a passive recorder. This design cannot support comparing agents that differ in prompt, tools, or retriever config because those are not client-selectable — they're determined by the pipeline/agent spec loaded server-side.

The new design introduces a **server-side pool** that the server samples from when a user opts in to A/B mode. The client's role simplifies to: request an A/B response, display two results, collect a vote.

### Key constraints
- Agent specs are markdown files loaded from disk at pipeline init time (`agent_spec.py`)
- The `archi` class creates a single pipeline instance at init, then reuses it
- Streaming responses go through `ChatWrapper.stream()` which takes `config_name`, `provider`, `model` overrides
- The `ab_comparisons` table already exists with `config_a_id`/`config_b_id` FK references to `configs`

## Goals / Non-Goals

**Goals:**
- Config-driven variant pool with champion designation
- Server-side champion/challenger sampling per A/B request
- Support variants differing in: agent spec, provider, model, retriever params (`num_documents_to_retrieve`), `recursion_limit`
- Per-variant win/loss/tie metrics updated atomically on vote
- Backward compatibility — if no `ab_testing` config section, behavior unchanged

**Non-Goals:**
- Automatic champion promotion (manual admin decision for now)
- Multi-arm bandit / Thompson sampling (simple random for now)
- Admin UI for pool management (config files only)
- ELO computation at runtime (offline analysis only)
- Deployment-wide forced A/B (per-user opt-in only)

## Decisions

### Decision 1: Variant identity is a name, not a config_id

Each variant is identified by its `name` string in config. The server resolves variant name → agent spec + provider + model + params at request time. We do NOT create a `configs` table row per variant — the existing configs table is for deployment configs, not A/B variants.

**Why:** Variants are ephemeral experiments. Config-table rows are long-lived deployment artifacts. Mixing them complicates the data model. Variant metadata is stored directly in the comparison record.

### Decision 2: Server builds two pipeline instances per A/B request

When handling an A/B streaming request, the server constructs two temporary pipeline instances — one for champion, one for challenger — with the variant-specific overrides (agent spec, provider, model, retriever params). These are not cached; they're created per-request.

**Why:** Agent specs, models, and retriever params all affect pipeline construction. Caching would require invalidation logic. Per-request construction is simple and correct. Performance cost is negligible compared to LLM inference time.

**Alternative considered:** Pre-build pipeline instances for all variants at startup. Rejected because: high memory cost for large pools, stale if config changes, and retriever params may need per-request vectorstore connections.

### Decision 3: Extend `ab_comparisons` rather than new table

Add `variant_a_name`, `variant_b_name`, `variant_a_meta` (JSONB), `variant_b_meta` (JSONB) columns to the existing `ab_comparisons` table. The JSONB columns store the full variant config snapshot (agent spec name, provider, model, retriever params) at comparison time.

**Why:** Keeps all comparison data in one table. JSONB captures arbitrary variant config without schema changes when new params are added. Snapshot ensures historical accuracy even if pool config changes.

### Decision 4: Separate `ab_variant_metrics` table for aggregates

A dedicated table with columns: `variant_name`, `wins`, `losses`, `ties`, `total_comparisons`, `last_updated`. Updated atomically via `UPDATE ... SET wins = wins + 1` on vote.

**Why:** Avoids expensive `COUNT(*)` queries over the comparisons table for dashboards. Simple schema. Easy to reset or recompute from raw data if needed.

### Decision 5: New `/api/ab/compare` endpoint instead of reusing stream

A new endpoint that accepts the user message and returns two NDJSON streams (interleaved, tagged with `arm: "a"` / `arm: "b"`). The server handles champion/challenger selection, randomization, and parallel execution.

**Why:** The current stream endpoint is single-pipeline. Multiplexing two pipelines into one response with arm tagging is cleaner than having the client make two separate stream requests with server-assigned variants.

**Alternative considered:** Two separate `/api/ab/stream_a` and `/api/ab/stream_b` endpoints. Rejected because: requires two HTTP connections, client needs to correlate them, and server can't atomically assign champion/challenger.

## Risks / Trade-offs

- **2x compute per A/B message** — Mitigated by per-user opt-in; only engaged users pay the cost.
- **Per-request pipeline construction** — Could be slow if agent spec loading is expensive. Mitigate by profiling; add caching later if needed.
- **JSONB variant metadata** — Not queryable with simple SQL. Mitigate by keeping `variant_a_name`/`variant_b_name` as indexed top-level columns for common queries; use JSONB only for full audit trail.
- **Pool config changes don't migrate history** — If a variant is renamed or removed from the pool, historical comparison records still reference the old name/config via snapshots. This is by design (immutable history).

## Open Questions

- None — all decisions resolved based on user input.

## Decision 6: Shared PipelineEventFormatter class

Extract a stateful `PipelineEventFormatter` class into `src/interfaces/chat_app/event_formatter.py`. Both `Chat.stream()` and `_stream_arm()` instantiate a formatter per request and call `formatter.process(output)` for each `PipelineOutput`, yielding the events it returns.

**Key design choices:**
- **Deferred tool_start emission**: On `tool_start` events, the formatter parses and remembers tool calls but does NOT yield events. When `tool_output` arrives, it yields a `tool_start` + `tool_output` pair. This prevents orphaned tool spinners and is already the pattern used by `Chat.stream()`.
- **Caller decorates events**: The formatter yields "bare" events (just `type` + type-specific fields). Callers add context: `Chat.stream()` adds `conversation_id`, `timestamp`, and appends to `trace_events`; `_stream_arm()` adds `arm` and pushes to the queue.
- **Progressive tool-call merging**: The formatter tracks all seen tool calls in `_calls` dict, progressively merging info from `tool_calls`, `additional_kwargs`, `tool_call_chunks`, and `tool_inputs_by_id` so that tool_start events have the best-available name and args.
- **Class-per-request, not singleton**: Each streaming request creates a new formatter instance. No thread-safety concerns; no stale state.

**Why not keep inline?** The tool_start extraction alone is ~80 lines of logic (parsing partial chunks from GPT-5, falling back through multiple sources). Having it in one class means new event types or new LLM provider quirks only need changes in one place.

On the JS side, extract `_renderStreamEvent(messageId, event)` on the `Chat` object — a switch over event type that calls the appropriate `UI.render*` method. Both `streamResponse` and `sendABMessage` use it, ensuring new event types are rendered consistently.

## Decision 7: Audit-driven redundancy elimination

A comprehensive code audit identified 9 categories of duplicated logic in `app.py` and `chat.js` that create maintenance burden and bug-risk. Each is addressed with a minimal extraction:

1. **`_error_event(error_code)`**: Static helper replacing 3 copies of error-code→human-message mapping. Returns `{"type": "error", "status": code, "message": text}`.

2. **`_ab_comparison_from_row(row)`**: Static helper replacing 3 identical 17-field positional row→dict conversions for AB comparisons. Single source of truth for column ordering.

3. **`_trace_from_row(row)`**: Static helper replacing 3 trace row→dict conversions (2 full 16-field, 1 subset). Uses `len(row)` to handle both full and subset results.

4. **`_get_agent_class_name(chat_cfg)`**: Static helper replacing 5 inline `chat_cfg.get("agent_class") or chat_cfg.get("pipeline")` calls. Added to ChatWrapper; FlaskAppWrapper already has this but 2 sites weren't using it.

5. **`_ndjson_response(event_iter)`**: Method on FlaskAppWrapper replacing 2 identical NDJSON response construction patterns (padding, headers, json.dumps loop).

6. **`_delete_source_documents(source_type, ...)`**: Shared helper on FlaskAppWrapper merging `_delete_git_repo()` and `_delete_jira_project()`. The two differed only in source_type, match column, and match value.

7. **`_readNDJSON(response)`**: Shared async generator in chat.js replacing 2 NDJSON reader loops. Also fixes a latent bug: `streamResponse`'s reader didn't flush the final buffer after the read loop, silently dropping events in the last incomplete chunk. `streamABComparison` already had the fix.

8. **`_toggle_reaction(reaction_type)`**: Merges `like()` and `dislike()` which were 85% identical (same lock/cleanup/toggle pattern, different field population).

9. **`_format_score_str(score)`**: Static helper deduplicating score→string logic shared between `_entry_html()` (HTML format_links) and `_format_source_entry()` (markdown format_links).

**Why not a DB context manager?** The audit also noted 20+ raw `psycopg2.connect()` / `cursor.close()` / `conn.close()` calls. While a context manager would reduce boilerplate, it's a larger refactor that touches every DB method and changes error-handling semantics. Deferred to a separate change for safety.
