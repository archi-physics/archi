## Context
`ChatWrapper.create_conversation` currently inserts a title derived from the first 20 characters of the first message. The title is persisted in `conversation_metadata`; user renames update the same field. The chat wrapper already knows the configured provider and model and the app uses daemon threads for background work.

## Goals / Non-Goals
- Goals: provide a concise, meaningful title without delaying the first reply; retain the initial title when generation fails; never overwrite a user rename; allow deployments to disable the feature.
- Non-Goals: add a database migration, change the sidebar UI, or add a separately configurable title model in this change.

## Decisions
- Decision: use the configured chat provider and model through the existing model construction path. This avoids another model setting and keeps provider behavior consistent. It does mean title generation uses the chat model's resource/cost profile.
- Decision: generate the title in a background task after inserting the conversation. Failure is logged and must not affect the request serving the chat.
- Decision: make `auto_title` enabled by default, including when omitted, so new installations get the requested behavior; setting it to `false` preserves current behavior. The setting must be rendered by the CLI config template and read with the same default at runtime.
- Decision: normalize model output by keeping the first non-empty line, trimming surrounding whitespace and quote characters, removing trailing sentence punctuation, and capping the result at 100 characters. Empty output is treated as generation failure.
- Decision: update the database with a compare-and-set predicate on conversation ID and original title. A zero-row update means the title changed or the conversation is gone, so the generated value is discarded.
- Decision: use the existing conversation ownership model when updating metadata, including the client/user ownership constraints where applicable.

## Risks / Trade-offs
- A daemon-thread task may be interrupted if the service stops before generation finishes. The initial title remains usable, and no request is blocked.
- A generation call adds model usage for each new conversation when enabled. Deployments can disable it; a separate cheaper model can be considered later if maintainers want that control.
- Multiple web workers may run independent background tasks, but each conversation is created once and the conditional update limits stale writes.

## Migration Plan
No database migration is required. Add a defaulted configuration value; existing deployments that omit it receive the enabled behavior. Setting it to `false` disables generation.

## Open Questions
- Confirm with maintainers whether enabled-by-default is acceptable given the additional model usage.
- Confirm that using the configured chat model is preferred over adding a separate `title_model` setting.
- Confirm the existing daemon-thread pattern is acceptable for this non-critical task.
