# Change: Generate descriptive titles for new chats

## Why
New chat titles currently copy only the first 20 characters of the first user message. This makes the sidebar hard to scan because titles are often truncated versions of generic openings rather than descriptions of the conversation.

## What Changes
- Generate a concise title from the first message asynchronously after creating a conversation.
- Replace the initial title only while it still matches the generated conversation's original truncated title, preserving user renames.
- Add `services.chat_app.auto_title` to enable or disable generation, defaulting to enabled for new deployments and treating omitted settings as enabled.
- Document the setting and add unit coverage for cleanup, generation failures, races with renames, and disabled mode.

## Impact
- Affected specs: `chat-titles` (new capability)
- Affected code: `src/interfaces/chat_app/app.py`, `src/utils/sql.py`, chat title generation helper, `src/cli/templates/base-config.yaml`, `docs/docs/configuration.md`
- No schema migration or frontend changes are expected.
