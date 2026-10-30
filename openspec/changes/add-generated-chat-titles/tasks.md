## 1. Implementation
- [x] 1.1 Add a title-generation helper that requests a concise title from the configured chat model and normalizes the response to one line and at most 100 characters.
- [x] 1.2 Add a conditional SQL update that changes a conversation title only if it still equals the initial title.
- [x] 1.3 Start title generation asynchronously after conversation creation when `services.chat_app.auto_title` is enabled; log failures without affecting chat.
- [x] 1.4 Add and document the `auto_title` configuration setting.
- [x] 1.5 Add unit tests for title cleanup, model failures, concurrent user renames, and disabled configuration.
- [ ] 1.6 Validate the OpenSpec change and perform the repository-required deployment validation after implementation.
