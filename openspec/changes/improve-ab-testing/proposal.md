# Proposal: Improve A/B Testing with Champion/Challenger Agent Pools

## Why

The current A/B testing system is half-functional: users manually select two provider+model combos to compare. This has several limitations:

1. **No agent pool** — only two manually selected provider/model pairs; no support for comparing agents that differ in prompt, tools, or retriever config.
2. **No champion concept** — there's no way to guarantee that the production-best agent is always one of the two being compared. Users could end up comparing two poor candidates, leading to frustration.
3. **Client-side only** — all A/B logic lives in `chat.js`; the server just records results. There's no server-side sampling strategy or config-driven setup.
4. **No metrics tracking** — comparisons are stored but there's no structured win/loss/tie tracking per variant to support analysis (e.g., ELO computation after the fact).
5. **Duplicated streaming logic** — the A/B streaming path (`_stream_arm`) re-implements event formatting that already exists in `Chat.stream()`, causing recurring parity bugs (tool events, thinking indicators, etc. break independently).

## What Changes

### 1. Config-driven agent variant pool
Add an `ab_testing` section to `config.yaml` that defines a pool of agent variants. Each variant specifies a name plus any combination of: agent spec (prompt + tools), provider, model, and retriever parameters. The pool is loaded at startup by the server.

### 2. Champion/Challenger sampling
One variant is designated the **champion** in config. When A/B mode is active for a user, every comparison includes the champion as one arm and a randomly sampled **challenger** from the rest of the pool as the other arm. Positions are randomized to avoid bias.

### 3. Per-user opt-in with forced voting
A/B mode remains opt-in per user (settings toggle). When enabled, users MUST vote (A, B, or tie) before sending the next message — matching current behavior but now backed by server-side pool logic.

### 4. Variant metadata in comparison records
Extend the `ab_comparisons` table to store full variant metadata (variant name, agent spec, provider, model, retriever config) so results are self-describing for later analysis.

### 5. Basic metrics tracking
Add an `ab_variant_metrics` table tracking per-variant win/loss/tie counts, updated on each vote. This supports dashboard display and later ELO computation without needing to reprocess raw comparison rows.

### 6. Server-side pool endpoint
Add a `/api/ab/pool` endpoint that returns the pool configuration (variant names and metadata) so the client knows A/B is available and can display variant info after voting.

### 7. Unified streaming event formatter (refactor)
Extract a shared `PipelineEventFormatter` class that converts `PipelineOutput` objects into structured JSON events. Both `Chat.stream()` (regular chat) and `_stream_arm()` (A/B arms) use the same formatter, eliminating the duplicated event‐formatting code that caused parity bugs. The formatter:
- Handles all event types: `tool_start`, `tool_output`, `tool_end`, `thinking_start`, `thinking_end`, `text`, `final`
- Manages mutable state (emitted tool IDs, pending tool IDs, progressive tool‐call merging) in one place
- Defers `tool_start` emission until the corresponding `tool_output` arrives, preventing orphaned spinners
- Extracts tool info from all sources: `message.tool_calls`, `additional_kwargs.tool_calls`, `tool_call_chunks`, and `metadata.tool_inputs_by_id`

On the JS side, extract a shared `_renderStreamEvent(messageId, event)` helper so both `streamResponse` and `sendABMessage` dispatch trace/tool/thinking events through one code path.

### 8. Code redundancy elimination (audit-driven)
A comprehensive audit identified 9 categories of duplicated logic across `app.py` and `chat.js`. These are eliminated in Phase 8:

- **Error-code→message mapping** (3 sites) → `_error_event(error_code)` static helper
- **AB comparison row→dict** (3 sites) → `_ab_comparison_from_row(row)` static helper
- **Agent trace row→dict** (3 sites) → `_trace_from_row(row)` static helper
- **agent_class resolution** (5 inline sites) → consolidated to `_get_agent_class_name()` on both `ChatWrapper` and `FlaskAppWrapper`
- **NDJSON response wrapper** (2 sites) → `_ndjson_response(event_iter)` on `FlaskAppWrapper`
- **Delete source documents** (2 structural clones) → `_delete_source_documents()` shared helper
- **JS NDJSON reader** (2 sites, 1 with latent bug) → shared `_readNDJSON()` async generator, also fixes a buffer-drain bug where `streamResponse` silently drops events in the final chunk
- **Like/dislike toggle** (85% identical) → `_toggle_reaction()` helper
- **format_links score formatting** (duplicated in HTML and markdown) → `_entry_html` reuses `_format_score_str()` static helper

## Impact

- **Affected specs**: None (no existing specs for A/B testing)
- **Affected code**:
  - `configs/config.yaml` — new `ab_testing` section
  - `src/utils/config_loader.py` / `src/utils/config_access.py` — load pool config
  - `src/interfaces/chat_app/app.py` (`ChatWrapper`) — server-side champion/challenger selection, new streaming logic for pool-based A/B, new `/api/ab/pool` endpoint
  - `src/interfaces/chat_app/event_formatter.py` — **new file**: `PipelineEventFormatter` class
  - `src/interfaces/chat_app/static/chat.js` — adapt `sendABMessage` to use server-side pool instead of client-side config selects; shared `_renderStreamEvent` helper
  - `src/cli/templates/init.sql` — extend `ab_comparisons` table, add `ab_variant_metrics` table
  - `src/utils/sql.py` — new SQL constants
  - `src/utils/conversation_service.py` — variant metrics update logic
  - `src/archi/archi.py` — support constructing pipelines with per-variant overrides
- **Breaking changes**: None — existing manual A/B mode continues to work as fallback if no pool is configured

## Design Decisions

### Why config-based pool (not UI-based)?
- Pool definition is an admin/deployment concern, not an end-user concern
- Config files are version-controllable and reviewable
- Consistent with how agent specs, providers, and models are already configured
- Simpler implementation; admin UI can be added later

### Why always include the champion?
- Guarantees every user always sees at least one high-quality response
- Generates direct champion-vs-challenger win rates, which are the most actionable metric
- Simpler sampling logic than multi-arm bandit approaches
- Matches the user's explicit goal of avoiding user frustration from two poor results

### Why basic metrics instead of ELO?
- Win/loss/tie counts are simple to maintain (atomic increment on vote)
- ELO requires ordering assumptions and tuning (K-factor, initial rating)
- Counts are sufficient raw data to compute ELO, Bradley-Terry, or any other model offline
- Keeps the runtime system simple; analysis happens separately

### Why per-user opt-in (not deployment-wide forced)?
- 2x API cost per message is significant — shouldn't be forced on all users
- Users who need fast answers shouldn't be blocked by voting requirements
- Opt-in users self-select for engagement, producing higher-quality preference data
- Admin can still encourage opt-in via documentation/onboarding

### Why a shared event formatter instead of keeping two code paths?
- The two paths diverged and caused a production bug where A/B tool/thinking events had a different structure than regular streaming
- Every new event type (thinking indicators, tool_end, etc.) had to be implemented twice
- The formatter centralizes the complex tool-call merging logic (partial chunks, additional_kwargs, memory fallback) in one class
- Deferred tool_start emission (emit start only when output arrives) is cleaner UX and prevents orphaned spinners
