# Tasks: Improve A/B Testing

## Phase 1: Config Schema & Loading

- [x] 1.1 Define `ab_testing` config schema in `config.yaml` with pool, champion, and variant definitions
- [x] 1.2 Add config validation in `config_manager.py` — validate variant names are unique, champion exists in pool, required fields present
- [x] 1.3 Add `load_ab_pool()` utility to parse config into typed variant objects (dataclass or dict)
- [x] 1.4 Add example `ab_testing` config block to `examples/defaults/` or inline in `configs/config.yaml`

## Phase 2: Database Schema Changes

- [x] 2.1 Add `variant_a_name`, `variant_b_name` (VARCHAR), `variant_a_meta`, `variant_b_meta` (JSONB) columns to `ab_comparisons` table in `init.sql`
- [x] 2.2 Add index on `variant_a_name` and `variant_b_name` columns
- [x] 2.3 Create `ab_variant_metrics` table: `variant_name` (PK), `wins`, `losses`, `ties`, `total_comparisons`, `last_updated`
- [x] 2.4 Add SQL constants to `src/utils/sql.py` for variant metrics CRUD (insert, update wins/losses/ties, select)
- [x] 2.5 Update `SQL_INSERT_AB_COMPARISON` to include variant name and meta columns

## Phase 3: Server-Side Pool & Sampling

- [x] 3.1 Add champion/challenger selection logic: given pool config, return (champion_variant, random_challenger_variant) with randomized position
- [x] 3.2 Extend `archi.py` or add factory function to create a pipeline instance from a variant definition (agent spec path, provider, model, retriever params, recursion_limit)
- [x] 3.3 Add `/api/ab/pool` GET endpoint returning pool info (variant names, champion name, pool size) for client
- [x] 3.4 Add `/api/ab/compare` POST endpoint that accepts user message and streams two responses (champion + challenger) as interleaved NDJSON with `arm` tags

## Phase 4: Metrics Tracking

- [x] 4.1 Add `update_variant_metrics(winner_name, loser_name)` and `update_variant_tie(name_a, name_b)` methods to `ConversationService`
- [x] 4.2 Update `ab_submit_preference` endpoint to call metrics update after recording the raw preference
- [x] 4.3 Add `/api/ab/metrics` GET endpoint returning per-variant win/loss/tie/total counts
- [x] 4.4 Ensure metrics updates are atomic (single UPDATE with `wins = wins + 1`)

## Phase 5: Client-Side Adaptation

- [x] 5.1 On `Chat.init()`, fetch `/api/ab/pool` — if pool is configured, show A/B toggle; if not, hide or show legacy manual mode
- [x] 5.2 When A/B is enabled and pool exists, `sendABMessage` calls `/api/ab/compare` instead of making two separate stream requests
- [x] 5.3 Parse interleaved NDJSON stream, routing events to panel A or B based on `arm` tag
- [x] 5.4 After vote, submit preference via existing `/api/ab/preference` (which now also updates metrics)
- [x] 5.5 Remove or deprecate manual provider-B selection UI when pool mode is active (keep as fallback when no pool configured)

## Phase 6: Streaming Event Formatter Refactor

- [x] 6.1 Create `PipelineEventFormatter` class in `src/interfaces/chat_app/event_formatter.py` — stateful converter from `PipelineOutput` to structured event dicts
- [x] 6.2 Implement tool_start handling with deferred emission (remember on tool_start, emit on tool_output) and full extraction from `tool_calls`, `additional_kwargs`, `tool_call_chunks`, `tool_inputs_by_id`
- [x] 6.3 Implement tool_output, tool_end, thinking_start, thinking_end, text, and legacy fallback handlers
- [x] 6.4 Refactor `Chat.stream()` to use `PipelineEventFormatter.process()`, replacing inline tool-extraction and event-formatting logic (~200 lines)
- [x] 6.5 Refactor `_stream_arm()` to use `PipelineEventFormatter.process()`, replacing the manually-duplicated event formatting (~120 lines)
- [x] 6.6 Extract `_renderStreamEvent(messageId, event)` JS helper for shared UI dispatch of tool/thinking events
- [x] 6.7 Simplify `sendABMessage` and `streamResponse` JS handlers to use `_renderStreamEvent`
- [x] 6.8 Deploy to submit76, verify A/B streaming + regular streaming both work identically
- [x] 6.9 Verify backward compatibility — regular chat, A/B mode, thinking indicators, tool rendering all unchanged

## Phase 7: Documentation & Testing

- [ ] 7.1 Add `ab_testing` config section to `docs/docs/configuration.md`
- [ ] 7.2 Add A/B testing usage section to `docs/docs/user_guide.md` or new `docs/docs/ab_testing.md`
- [ ] 7.3 Document variant config fields: `name`, `agent_spec`, `provider`, `model`, `num_documents_to_retrieve`, `recursion_limit`
- [ ] 7.4 Add unit tests for pool loading, champion/challenger selection, and metrics update logic
- [ ] 7.5 Verify backward compatibility — no `ab_testing` config section means existing behavior unchanged
- [ ] 7.6 Test end-to-end: enable A/B, send message, receive two streams, vote, verify metrics updated

## Verification Checklist

After implementation, verify:
- [x] Config with `ab_testing.pool` loads and validates correctly
- [x] Champion is always one of the two arms in every comparison
- [x] Challenger is randomly sampled from non-champion variants
- [x] Variant metadata (agent spec, provider, model, params) is snapshotted in comparison record
- [x] `ab_variant_metrics` table updates atomically on each vote
- [x] Client correctly routes interleaved streams to A/B panels
- [x] Voting is required before next message (tie allowed)
- [x] No `ab_testing` config = existing manual A/B behavior preserved
- [x] Regular streaming and A/B streaming use the same event formatter (no duplicate code)
- [x] New event types only need to be added in one place (the formatter)
