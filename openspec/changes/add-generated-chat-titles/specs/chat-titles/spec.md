## ADDED Requirements
### Requirement: Generate descriptive titles for new conversations
When `services.chat_app.auto_title` is enabled, the system SHALL generate a concise title from the first user message after creating a conversation, without delaying the chat response. The system SHALL retain the initial title if generation fails or produces no usable text. The setting SHALL default to enabled when omitted and SHALL allow operators to disable generation.

#### Scenario: Title generation succeeds
- **WHEN** a new conversation is created and title generation returns usable text
- **THEN** the conversation title is updated to the normalized generated title asynchronously

#### Scenario: Generation fails
- **WHEN** title generation raises an error or returns no usable text
- **THEN** the initial title remains and the failure is logged without interrupting chat

#### Scenario: Operator disables title generation
- **WHEN** `services.chat_app.auto_title` is `false`
- **THEN** the system does not start title generation and retains current title behavior

### Requirement: Preserve user-selected conversation titles
The system SHALL update a generated title only if the conversation title still equals the initial title created from the first message.

#### Scenario: User renames before generation completes
- **WHEN** a user renames a conversation before background title generation finishes
- **THEN** the generated title is discarded and the user's title remains

#### Scenario: Conversation title has not changed
- **WHEN** generation finishes while the conversation title still equals its initial title
- **THEN** the system updates the title to the generated value
