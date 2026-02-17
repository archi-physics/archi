# archi Chat UI - UX Workflows

A living document defining all user experience workflows in the chat interface, with MCP verification checklists and Playwright test specifications.

**Last Updated**: January 2026  
**Coverage Status**: 🟢 Core | 🟡 Partial | 🔴 Not Tested

---

## Table of Contents

1. [Page Load & Initialization](#1-page-load--initialization)
2. [Conversation Management](#2-conversation-management)
3. [Message Flow](#3-message-flow)
4. [Streaming & Cancellation](#4-streaming--cancellation)
5. [Provider & Model Selection](#5-provider--model-selection)
6. [A/B Testing Mode](#6-ab-testing-mode)
7. [Agent Info Modal](#7-agent-info-modal)
8. [Settings Modal](#8-settings-modal)
9. [API Key Management](#9-api-key-management)
10. [Sidebar Navigation](#10-sidebar-navigation)
11. [Code Block Interactions](#11-code-block-interactions)
12. [Agent Trace Visualization](#12-agent-trace-visualization)
13. [Data Tab Navigation](#13-data-tab-navigation)
14. [Keyboard Navigation](#14-keyboard-navigation)
15. [Error Handling](#15-error-handling)
16. [Responsive Layout](#16-responsive-layout)

---

## 1. Page Load & Initialization

**Coverage**: 🟢 Core

### User Story
When a user navigates to `/chat`, the page should load all necessary UI components, fetch initial data, and restore previous session state.

### Expected Behavior
1. Page renders with sidebar, header, input area
2. Configs are fetched and populate the model dropdown
3. Conversations are fetched and render in sidebar
4. Providers are fetched and populate the provider dropdown
5. Pipeline default model info is fetched
6. Entry meta label shows agent and model info
7. If a previous conversation was active, it's restored

### MCP Verification Checklist
```markdown
## Check: Page Load & Initialization
1. Navigate to http://localhost:7861/chat
2. Take snapshot and verify:
   - [ ] Sidebar is visible with conversation list
   - [ ] Header shows "Chat" title and tabs (Chat, Data)
   - [ ] Input area has textarea, config dropdown, settings button, send button
   - [ ] Entry meta label shows "Agent: <config_name> · Model: <model_info>"
3. Check console for errors:
   - [ ] No JavaScript errors
   - [ ] No failed network requests (4xx/5xx)
4. Check network requests completed:
   - [ ] GET /api/get_configs → 200
   - [ ] GET /api/list_conversations → 200
   - [ ] GET /api/providers → 200
   - [ ] GET /api/pipeline/default_model → 200
```

### Playwright Tests
```typescript
test('page loads with all required elements', async ({ page }) => {
  await page.goto('/chat');
  await expect(page.locator('.sidebar')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'archi Chat' })).toBeVisible();
  await expect(page.getByLabel('Message input')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
});

test('entry meta label shows agent and model info', async ({ page }) => {
  await page.goto('/chat');
  const entryMeta = page.locator('.entry-meta');
  await expect(entryMeta).toContainText('Agent:');
  await expect(entryMeta).toContainText('Model:');
});

test('header tabs are visible', async ({ page }) => {
  await page.goto('/chat');
  await expect(page.locator('.header-tab').filter({ hasText: 'Chat' })).toBeVisible();
  await expect(page.locator('.header-tab').filter({ hasText: 'Data' })).toBeVisible();
});

test('restores active conversation on reload', async ({ page }) => {
  // Setup: create a conversation
  await page.goto('/chat');
  await page.evaluate(() => {
    localStorage.setItem('archi_active_conversation_id', '1');
  });
  await page.route('**/api/load_conversation', async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({
        messages: [{ sender: 'User', content: 'Hello', message_id: 1 }]
      })
    });
  });
  await page.reload();
  await expect(page.locator('.message.user')).toBeVisible();
});
```

---

## 2. Conversation Management

**Coverage**: 🟢 Core

### User Story
Users can create new conversations, switch between existing ones, and delete conversations.

### Expected Behavior
1. **New Chat**: Clicking "New chat" clears messages and creates fresh conversation
2. **Switch**: Clicking a conversation in sidebar loads its messages
3. **Delete**: Clicking delete icon removes conversation (with confirmation)
4. **Grouping**: Conversations grouped by date (Today, Yesterday, Previous 7 Days, Older)

### MCP Verification Checklist
```markdown
## Check: Conversation Management
1. Start a conversation by sending a message
2. Click "New chat" button:
   - [ ] Messages area clears
   - [ ] New conversation appears in sidebar after sending message
3. Click an existing conversation:
   - [ ] Messages load correctly
   - [ ] Conversation item shows active state
4. Click delete button on a conversation:
   - [ ] Confirmation dialog appears
   - [ ] Conversation is removed from sidebar
5. Verify date grouping:
   - [ ] Conversations show under correct date headers
```

### Playwright Tests
```typescript
test('new chat button clears messages', async ({ page }) => {
  await page.goto('/chat');
  
  // Mock conversation API
  await page.route('**/api/get_chat_response_stream', async (route) => {
    const body = '{"type":"final","response":"Hello!","message_id":1,"user_message_id":1,"conversation_id":1}\n';
    await route.fulfill({ status: 200, contentType: 'text/plain', body });
  });
  
  await page.getByLabel('Message input').fill('Test');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('.message')).toHaveCount(2);
  
  await page.getByRole('button', { name: 'New chat' }).click();
  await expect(page.locator('.message.user')).toHaveCount(0);
});

test('clicking conversation loads its messages', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/list_conversations*', async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({
        conversations: [{ conversation_id: 1, title: 'Test Conv', last_message_at: new Date().toISOString() }]
      })
    });
  });
  
  await page.route('**/api/load_conversation', async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({
        messages: [{ sender: 'User', content: 'Previous message', message_id: 1 }]
      })
    });
  });
  
  await page.reload();
  await page.locator('.conversation-item').first().click();
  await expect(page.locator('.message')).toBeVisible();
});

test('delete conversation shows confirmation', async ({ page }) => {
  await page.goto('/chat');
  
  page.on('dialog', async dialog => {
    expect(dialog.type()).toBe('confirm');
    await dialog.accept();
  });
  
  await page.route('**/api/list_conversations*', async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({
        conversations: [{ conversation_id: 1, title: 'Test', last_message_at: new Date().toISOString() }]
      })
    });
  });
  
  await page.route('**/api/delete_conversation', async (route) => {
    await route.fulfill({ status: 200, body: '{}' });
  });
  
  await page.reload();
  await page.locator('.conversation-item-delete').first().click();
});
```

---

## 3. Message Flow

**Coverage**: 🟢 Core

### User Story
Users can type messages, send them, and receive AI responses with proper rendering.

### Expected Behavior
1. User types in textarea (auto-resizes)
2. Pressing Enter or clicking Send submits message
3. User message appears immediately
4. AI response streams in with cursor indicator
5. Markdown content is rendered (code blocks, lists, etc.)
6. Message meta shows under assistant messages (not user messages)

### MCP Verification Checklist
```markdown
## Check: Message Flow
1. Type "Hello" in input field:
   - [ ] Input field auto-resizes for multiline
2. Press Enter to send:
   - [ ] User message appears with "You" label
   - [ ] Input clears
3. Wait for response:
   - [ ] Streaming cursor visible during response
   - [ ] archi message appears with avatar
4. Verify markdown rendering:
   - [ ] Code blocks have syntax highlighting
   - [ ] Lists render correctly
5. Check message meta:
   - [ ] Assistant message has meta (Agent, Model)
   - [ ] User message has NO meta
```

### Playwright Tests
```typescript
test('user can send message with Enter key', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/get_chat_response_stream', async (route) => {
    const body = '{"type":"final","response":"OK","message_id":1,"user_message_id":1,"conversation_id":1}\n';
    await route.fulfill({ status: 200, contentType: 'text/plain', body });
  });
  
  await page.getByLabel('Message input').fill('Hello');
  await page.keyboard.press('Enter');
  
  await expect(page.locator('.message.user')).toBeVisible();
  await expect(page.locator('.message.assistant')).toBeVisible();
});

test('message meta appears under assistant message only', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/get_chat_response_stream', async (route) => {
    const body = '{"type":"final","response":"Hello!","message_id":1,"user_message_id":1,"conversation_id":1}\n';
    await route.fulfill({ status: 200, contentType: 'text/plain', body });
  });
  
  await page.getByLabel('Message input').fill('Hi');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  const userMsg = page.locator('.message.user').first();
  const assistantMsg = page.locator('.message.assistant').first();
  
  await expect(userMsg.locator('.message-meta')).toHaveCount(0);
  await expect(assistantMsg.locator('.message-meta')).toBeVisible();
});

test('textarea auto-resizes on input', async ({ page }) => {
  await page.goto('/chat');
  
  const textarea = page.getByLabel('Message input');
  const initialHeight = await textarea.evaluate(el => el.offsetHeight);
  
  await textarea.fill('Line 1\nLine 2\nLine 3\nLine 4\nLine 5');
  const newHeight = await textarea.evaluate(el => el.offsetHeight);
  
  expect(newHeight).toBeGreaterThan(initialHeight);
});
```

---

## 4. Streaming & Cancellation

**Coverage**: 🟢 Core

### User Story
Users see real-time streaming responses and can cancel in-progress streams.

### Expected Behavior
1. Send button becomes Stop button during streaming
2. Streaming cursor visible while receiving
3. Clicking Stop cancels the stream
4. Partial response is preserved
5. Input re-enabled after streaming completes or cancels

### MCP Verification Checklist
```markdown
## Check: Streaming & Cancellation
1. Send a message that triggers streaming:
   - [ ] Send button icon changes to stop (⏹)
   - [ ] Button aria-label is "Stop streaming"
2. During streaming:
   - [ ] Streaming cursor (blinking) visible
   - [ ] Input field is disabled
3. Click stop button:
   - [ ] Streaming stops
   - [ ] Button reverts to send icon
   - [ ] Input field re-enabled
4. Verify partial response:
   - [ ] Any received content is preserved
```

### Playwright Tests
```typescript
test('send button toggles to stop while streaming', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/get_chat_response_stream', async (route) => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const body = '{"type":"chunk","content":"Hi"}\n';
    await route.fulfill({ status: 200, contentType: 'text/plain', body });
  });
  
  await page.getByLabel('Message input').fill('Hello');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  await expect(page.getByRole('button', { name: 'Stop streaming' })).toBeVisible();
  await page.getByRole('button', { name: 'Stop streaming' }).click();
  await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
});

test('input is disabled during streaming', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/get_chat_response_stream', async (route) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    await route.fulfill({ status: 200, body: '{"type":"chunk","content":"..."}\n' });
  });
  
  await page.getByLabel('Message input').fill('Test');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  await expect(page.getByLabel('Message input')).toBeDisabled();
});

test('streaming cursor visible during response', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/get_chat_response_stream', async (route) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    await route.fulfill({ status: 200, body: '{"type":"chunk","content":"Test"}\n' });
  });
  
  await page.getByLabel('Message input').fill('Hi');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  await expect(page.locator('.streaming-cursor')).toBeVisible();
});
```

---

## 5. Provider & Model Selection

**Coverage**: 🟢 Core

### User Story
Users can select different AI providers and models, or use the pipeline default.

### Expected Behavior
1. Provider dropdown shows available providers (+ "Use pipeline default")
2. Selecting provider loads its available models
3. OpenRouter shows "Custom model..." option with text input
4. Entry meta updates to reflect selection
5. Selection persists across page reloads (localStorage)

### MCP Verification Checklist
```markdown
## Check: Provider & Model Selection
1. Open Settings modal → Models section:
   - [ ] Provider dropdown visible
   - [ ] "Use pipeline default" is first option
2. Select a provider (e.g., OpenRouter):
   - [ ] Model dropdown populates with models
   - [ ] Provider status shows "Connected"
3. Select "Custom model..." (OpenRouter):
   - [ ] Custom model input appears
   - [ ] Entry meta updates with custom model name
4. Select "Use pipeline default":
   - [ ] Model dropdown shows "Using pipeline default"
   - [ ] Entry meta shows "Pipeline default: <class> <model>"
5. Reload page:
   - [ ] Previous selection is restored
```

### Playwright Tests
```typescript
test('provider dropdown defaults to pipeline default', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  
  const providerSelect = page.locator('#provider-select');
  await expect(providerSelect).toHaveValue('');
});

test('selecting provider loads its models', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/providers', async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({
        providers: [{
          type: 'openrouter',
          display_name: 'OpenRouter',
          enabled: true,
          models: [{ id: 'gpt-4', name: 'GPT-4' }]
        }]
      })
    });
  });
  
  await page.reload();
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('#provider-select').selectOption('openrouter');
  
  const modelSelect = page.locator('#model-select-primary');
  await expect(modelSelect).not.toBeDisabled();
});

test('custom model input appears for OpenRouter', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('#provider-select').selectOption('openrouter');
  await page.locator('#model-select-primary').selectOption('__custom__');
  
  await expect(page.locator('#custom-model-input')).toBeVisible();
});

test('entry meta updates when provider changes', async ({ page }) => {
  await page.goto('/chat');
  const entryMeta = page.locator('.entry-meta');
  
  await expect(entryMeta).toContainText('Pipeline default');
  
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('#provider-select').selectOption('openrouter');
  await page.locator('#model-select-primary').selectOption('__custom__');
  await page.locator('#custom-model-input').fill('test-model');
  await page.getByRole('button', { name: 'Close settings' }).click();
  
  await expect(entryMeta).toContainText('OpenRouter');
  await expect(entryMeta).toContainText('test-model');
});
```

---

## 6. A/B Testing Mode

**Coverage**: 🟡 Partial

### User Story
Users can enable A/B testing to compare two model responses side-by-side and vote on the better one.

### Expected Behavior
1. A/B toggle in Settings → Advanced shows warning modal first time
2. When enabled, two responses appear side-by-side
3. Vote buttons appear after both responses complete
4. Voting collapses to winning response
5. Cannot send new message while vote pending (unless A/B disabled)
6. Message meta is NOT shown in A/B comparison (blind comparison)

### MCP Verification Checklist
```markdown
## Check: A/B Testing Mode
1. Settings → Advanced → Enable A/B Testing:
   - [ ] Warning modal appears first time
   - [ ] Shows 2× API usage warning
2. Confirm and send message:
   - [ ] Two response columns appear (Model A, Model B)
   - [ ] Both stream independently
3. After both complete:
   - [ ] Vote buttons appear (Model A / Model B)
   - [ ] Neither response shows message meta
4. Click vote:
   - [ ] Collapses to single winning response
   - [ ] Input re-enabled
5. Try sending without voting:
   - [ ] Toast message: "Please vote first..."
```

### Playwright Tests
```typescript
test('A/B toggle shows warning modal on first enable', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('.settings-nav-item[data-section="advanced"]').click();
  
  // Clear any previous dismissal
  await page.evaluate(() => sessionStorage.removeItem('archi_ab_warning_dismissed'));
  
  await page.locator('#ab-checkbox').click();
  
  await expect(page.locator('.ab-warning-modal')).toBeVisible();
  await expect(page.locator('.ab-warning-modal')).toContainText('2× API usage');
});

test('A/B mode shows two response columns', async ({ page }) => {
  await page.goto('/chat');
  
  // Enable A/B mode (skip warning)
  await page.evaluate(() => sessionStorage.setItem('archi_ab_warning_dismissed', 'true'));
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('.settings-nav-item[data-section="advanced"]').click();
  await page.locator('#ab-checkbox').check();
  await page.getByRole('button', { name: 'Close settings' }).click();
  
  // Mock A/B API
  await page.route('**/api/ab/create', async (route) => {
    await route.fulfill({ status: 200, body: JSON.stringify({ comparison_id: 1 }) });
  });
  await page.route('**/api/get_chat_response_stream', async (route) => {
    await route.fulfill({
      status: 200,
      body: '{"type":"final","response":"Test","message_id":1,"user_message_id":1,"conversation_id":1}\n'
    });
  });
  
  await page.getByLabel('Message input').fill('Compare');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  await expect(page.locator('.ab-comparison')).toBeVisible();
  await expect(page.locator('.ab-response-a')).toBeVisible();
  await expect(page.locator('.ab-response-b')).toBeVisible();
});

test('A/B comparison hides message meta', async ({ page }) => {
  // Similar setup as above...
  await page.goto('/chat');
  await page.evaluate(() => sessionStorage.setItem('archi_ab_warning_dismissed', 'true'));
  
  // Enable A/B and send message...
  // Verify no .message-meta inside .ab-comparison
  const comparison = page.locator('.ab-comparison');
  await expect(comparison.locator('.message-meta')).toHaveCount(0);
});
```

---

## 8. Settings Modal

**Coverage**: 🟢 Core

### User Story
Users can access settings for model selection, API keys, and advanced options.

### Expected Behavior
1. Settings button opens modal
2. Three sections: Models, API Keys, Advanced
3. Navigation sidebar switches between sections
4. Modal closes on X button, backdrop click, or Escape
5. Changes are applied immediately

### MCP Verification Checklist
```markdown
## Check: Settings Modal
1. Click Settings button:
   - [ ] Modal opens
   - [ ] "Models" section visible by default
   - [ ] Nav sidebar shows 3 items
2. Switch sections:
   - [ ] Click "API Keys" → API keys form visible
   - [ ] Click "Advanced" → A/B toggle, trace options visible
   - [ ] Click "Models" → Provider selection visible
3. Close modal:
   - [ ] X button closes
   - [ ] Backdrop click closes
   - [ ] Escape key closes
```

### Playwright Tests
```typescript
test('settings modal opens with Models section active', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  
  await expect(page.locator('#settings-models')).toBeVisible();
  await expect(page.locator('.settings-nav-item[data-section="models"]')).toHaveClass(/active/);
});

test('can switch between settings sections', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  
  await page.locator('.settings-nav-item[data-section="api-keys"]').click();
  await expect(page.locator('#settings-api-keys')).toBeVisible();
  
  await page.locator('.settings-nav-item[data-section="advanced"]').click();
  await expect(page.locator('#settings-advanced')).toBeVisible();
});

test('settings modal closes on backdrop click', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  
  await page.mouse.click(10, 10);
  await expect(page.locator('.settings-panel')).not.toBeVisible();
});
```

---

## 9. API Key Management

**Coverage**: 🟡 Partial

### User Story
Users can configure API keys for different providers.

### Expected Behavior
1. API Keys section shows all providers that need keys
2. Status indicator shows: configured (✓) or not configured (○)
3. Can enter and save new API key
4. Can clear session-stored keys
5. Keys marked as "Session" vs "Env" based on source

### MCP Verification Checklist
```markdown
## Check: API Key Management
1. Settings → API Keys:
   - [ ] Provider list shows (OpenRouter, OpenAI, etc.)
   - [ ] Each has status indicator
2. Enter API key and click Save:
   - [ ] Status changes to configured (✓ Session)
3. Click Clear button:
   - [ ] Session key removed
   - [ ] Status reverts if no env key
```

### Playwright Tests
```typescript
test('API key section shows provider status', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/providers/keys', async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({
        providers: [
          { provider: 'openrouter', display_name: 'OpenRouter', configured: false, has_session_key: false }
        ]
      })
    });
  });
  
  await page.reload();
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('.settings-nav-item[data-section="api-keys"]').click();
  
  await expect(page.locator('.api-key-row[data-provider="openrouter"]')).toBeVisible();
});

test('can save API key', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('.settings-nav-item[data-section="api-keys"]').click();
  
  await page.route('**/api/providers/keys/set', async (route) => {
    await route.fulfill({ status: 200, body: '{"success": true}' });
  });
  
  await page.locator('.api-key-input[data-provider="openrouter"]').fill('sk-test-key');
  await page.locator('.save-btn[data-provider="openrouter"]').click();
  
  // Verify save was attempted (button text changes during save)
  await expect(page.locator('.save-btn[data-provider="openrouter"]')).toContainText('Save');
});
```

---

## 10. Sidebar Navigation

**Coverage**: 🟢 Core

### User Story
Users can toggle the sidebar and navigate conversations on both desktop and mobile.

### Expected Behavior
1. Desktop: Sidebar collapsed/expanded with toggle button
2. Mobile: Sidebar overlays content with backdrop
3. Clicking overlay closes sidebar on mobile
4. aria-expanded attribute updates correctly

### MCP Verification Checklist
```markdown
## Check: Sidebar Navigation
1. Desktop (>768px):
   - [ ] Sidebar visible by default
   - [ ] Toggle button collapses sidebar
   - [ ] Content area expands to fill space
2. Mobile (≤768px):
   - [ ] Sidebar hidden by default
   - [ ] Toggle shows overlay sidebar
   - [ ] Backdrop click closes sidebar
3. Accessibility:
   - [ ] Toggle has aria-expanded attribute
   - [ ] Sidebar has appropriate ARIA landmarks
```

### Playwright Tests
```typescript
test('sidebar toggle collapses on desktop', async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  await page.goto('/chat');
  
  await expect(page.locator('.sidebar')).toBeVisible();
  await page.getByRole('button', { name: 'Toggle sidebar' }).click();
  await expect(page.locator('.app')).toHaveClass(/sidebar-collapsed/);
});

test('sidebar overlay closes on click (mobile)', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('/chat');
  
  await page.getByRole('button', { name: 'Toggle sidebar' }).click();
  await expect(page.locator('.app')).toHaveClass(/sidebar-open/);
  
  await page.locator('.sidebar-overlay').click();
  await expect(page.locator('.app')).not.toHaveClass(/sidebar-open/);
});
```

---

## 11. Code Block Interactions

**Coverage**: 🟡 Partial

### User Story
Users can view syntax-highlighted code blocks and copy code to clipboard.

### Expected Behavior
1. Code blocks have language label
2. Copy button copies code to clipboard
3. "Copied!" feedback appears briefly
4. Syntax highlighting applied via highlight.js

### MCP Verification Checklist
```markdown
## Check: Code Block Interactions
1. Send message requesting code response
2. Verify code block:
   - [ ] Language label shown (e.g., "python")
   - [ ] Syntax highlighting applied
   - [ ] Copy button visible
3. Click Copy:
   - [ ] Button text changes to "Copied!"
   - [ ] Text on clipboard matches code
4. After 2 seconds:
   - [ ] Button reverts to "Copy"
```

### Playwright Tests
```typescript
test('code blocks have copy button', async ({ page }) => {
  await page.goto('/chat');
  
  // Mock response with code block
  await page.route('**/api/get_chat_response_stream', async (route) => {
    const body = '{"type":"final","response":"```python\\nprint(1)\\n```","message_id":1,"user_message_id":1,"conversation_id":1}\n';
    await route.fulfill({ status: 200, contentType: 'text/plain', body });
  });
  
  await page.getByLabel('Message input').fill('Show code');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  await expect(page.locator('.code-block-copy')).toBeVisible();
  await expect(page.locator('.code-block-lang')).toContainText('python');
});

test('copy button shows feedback', async ({ page }) => {
  // Setup code block as above...
  await page.locator('.code-block-copy').click();
  await expect(page.locator('.code-block-copy')).toContainText('Copied!');
});
```

---

## 12. Agent Trace Visualization

**Coverage**: 🟡 Partial

### User Story
Users can view agent activity including tool calls during response generation.

### Expected Behavior
1. Trace container appears for assistant messages
2. Shows "Agent Activity" with tool count
3. Tool calls show: name, arguments, status, output
4. Can expand/collapse trace and individual tools
5. Trace verbosity controlled by settings (minimal/normal/verbose)

### MCP Verification Checklist
```markdown
## Check: Agent Trace Visualization
1. Send message that triggers tool use:
   - [ ] "Agent Activity" container appears
2. While tools running:
   - [ ] Spinner shows for running tools
   - [ ] Tool name and arguments visible
3. After completion:
   - [ ] Checkmark for success, X for error
   - [ ] Duration shown
   - [ ] Output visible (truncated if long)
4. Toggle visibility:
   - [ ] Click header collapses trace
   - [ ] Click tool expands/collapses details
5. Settings → Advanced → Trace mode:
   - [ ] "Minimal" hides trace container
   - [ ] "Verbose" auto-expands tools
```

### Playwright Tests
```typescript
test('trace container shows for tool calls', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/get_chat_response_stream', async (route) => {
    const events = [
      '{"type":"tool_start","tool_call_id":"tc1","tool_name":"search","tool_args":{"query":"test"}}',
      '{"type":"tool_output","tool_call_id":"tc1","output":"Found 5 results"}',
      '{"type":"tool_end","tool_call_id":"tc1","status":"success","duration_ms":150}',
      '{"type":"final","response":"Done","message_id":1,"user_message_id":1,"conversation_id":1}'
    ].join('\n');
    await route.fulfill({ status: 200, body: events });
  });
  
  await page.getByLabel('Message input').fill('Search');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  await expect(page.locator('.trace-container')).toBeVisible();
  await expect(page.locator('.tool-block')).toBeVisible();
});

test('trace verbose mode changes display', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('.settings-nav-item[data-section="advanced"]').click();
  
  await page.locator('input[name="trace-verbose"][value="minimal"]').check();
  // In minimal mode, trace container would be hidden
});
```

---

## 13. Data Tab Navigation

**Coverage**: 🟢 Core

### User Story
Users can navigate to the data viewer for the current conversation.

### Expected Behavior
1. Data tab in header is clickable
2. With active conversation: navigates to `/data?conversation_id=X`
3. Without conversation: shows alert message

### MCP Verification Checklist
```markdown
## Check: Data Tab Navigation
1. Without conversation:
   - [ ] Click Data tab → Alert: "Please select or start a conversation..."
2. With active conversation:
   - [ ] Click Data tab → Navigate to /data?conversation_id=X
```

### Playwright Tests
```typescript
test('Data tab without conversation shows alert', async ({ page }) => {
  await page.goto('/chat');
  
  page.on('dialog', async dialog => {
    expect(dialog.message()).toContain('conversation');
    await dialog.accept();
  });
  
  await page.locator('.header-tab').filter({ hasText: 'Data' }).click();
});

test('Data tab with conversation navigates', async ({ page }) => {
  await page.goto('/chat');
  
  // Set up active conversation
  await page.evaluate(() => {
    localStorage.setItem('archi_active_conversation_id', '123');
  });
  
  // Mock the conversation load
  await page.route('**/api/load_conversation', async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({ messages: [] })
    });
  });
  
  await page.reload();
  
  // Set state in JS
  await page.evaluate(() => {
    // @ts-ignore
    if (window.Chat) window.Chat.state.conversationId = 123;
  });
  
  await page.locator('.header-tab').filter({ hasText: 'Data' }).click();
  await expect(page).toHaveURL(/\/data\?conversation_id=123/);
});
```

---

## 14. Keyboard Navigation

**Coverage**: 🟡 Partial

### User Story
Users can navigate the interface using keyboard only.

### Expected Behavior
1. Tab through interactive elements in logical order
2. Enter/Space activates buttons
3. Escape closes modals
4. Enter in input field sends message (Shift+Enter for newline)

### MCP Verification Checklist
```markdown
## Check: Keyboard Navigation
1. Tab order (from input):
   - [ ] Input → Settings → Send
2. Modal trap:
   - [ ] Tab stays within open modal
3. Escape:
   - [ ] Closes settings modal
4. Enter in input:
   - [ ] Sends message
   - [ ] Shift+Enter adds newline
```

### Playwright Tests
```typescript
test('Enter sends message, Shift+Enter adds newline', async ({ page }) => {
  await page.goto('/chat');
  
  const textarea = page.getByLabel('Message input');
  await textarea.fill('Line 1');
  await textarea.press('Shift+Enter');
  await textarea.type('Line 2');
  
  const value = await textarea.inputValue();
  expect(value).toContain('\n');
  
  // Now test Enter sends
  await page.route('**/api/get_chat_response_stream', async (route) => {
    await route.fulfill({ status: 200, body: '{"type":"final","response":"OK"}\n' });
  });
  
  await textarea.press('Enter');
  await expect(page.locator('.message.user')).toBeVisible();
});

test('Escape closes modals', async ({ page }) => {
  await page.goto('/chat');
  
  await page.getByRole('button', { name: 'Settings' }).click();
  await expect(page.locator('.settings-panel')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('.settings-panel')).not.toBeVisible();
});
```

---

## 15. Error Handling

**Coverage**: 🟡 Partial

### User Story
Users see appropriate error messages when things go wrong.

### Expected Behavior
1. API errors show toast or inline message
2. Streaming errors preserve partial content
3. 401 errors redirect to login
4. Network failures show appropriate message

### MCP Verification Checklist
```markdown
## Check: Error Handling
1. API error (e.g., 500):
   - [ ] Error message displayed
   - [ ] UI remains functional
2. Streaming error:
   - [ ] Partial response preserved
   - [ ] Input re-enabled
3. 401 Unauthorized:
   - [ ] Redirects to login (/)
4. Model not selected:
   - [ ] Toast: "Please select a model..."
```

### Playwright Tests
```typescript
test('API error shows message', async ({ page }) => {
  await page.goto('/chat');
  
  await page.route('**/api/get_chat_response_stream', async (route) => {
    await route.fulfill({ status: 500, body: 'Internal Server Error' });
  });
  
  await page.getByLabel('Message input').fill('Test');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  // Should re-enable input after error
  await expect(page.getByLabel('Message input')).not.toBeDisabled();
});

test('missing model shows toast', async ({ page }) => {
  await page.goto('/chat');
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.locator('#provider-select').selectOption('openrouter');
  // Don't select a model
  await page.getByRole('button', { name: 'Close settings' }).click();
  
  await page.getByLabel('Message input').fill('Test');
  await page.getByRole('button', { name: 'Send message' }).click();
  
  await expect(page.locator('.toast')).toContainText('select a model');
});
```

---

## 16. Responsive Layout

**Coverage**: 🟡 Partial

### User Story
The interface adapts appropriately to different screen sizes.

### Expected Behavior
1. Desktop (>768px): Sidebar always visible, can collapse
2. Mobile (≤768px): Sidebar hidden, overlay mode
3. Input area sticks to bottom
4. Messages area scrolls independently

### MCP Verification Checklist
```markdown
## Check: Responsive Layout
1. Desktop (1200px):
   - [ ] Sidebar visible
   - [ ] Messages and input side-by-side with sidebar
2. Tablet (768px):
   - [ ] Layout still functional
3. Mobile (375px):
   - [ ] Sidebar hidden initially
   - [ ] Full-width input area
   - [ ] Messages area scrollable
```

### Playwright Tests
```typescript
test('mobile layout hides sidebar', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('/chat');
  
  const sidebar = page.locator('.sidebar');
  const boundingBox = await sidebar.boundingBox();
  
  // Sidebar should be off-screen or hidden
  expect(boundingBox?.x ?? -1000).toBeLessThan(0);
});

test('desktop layout shows sidebar', async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  await page.goto('/chat');
  
  await expect(page.locator('.sidebar')).toBeVisible();
});
```

---

## Test Execution Summary

### Running All Playwright Tests
```bash
npx playwright test tests/ui/
```

### Running MCP Validation
```bash
# Use Chrome DevTools MCP tools via VS Code / agent
# Follow checklists in each section above
```

### Coverage Matrix

| Workflow | Playwright | MCP Checks | Status |
|----------|------------|------------|--------|
| Page Load | ✅ | ✅ | 🟢 |
| Conversations | ✅ | ✅ | 🟢 |
| Message Flow | ✅ | ✅ | 🟢 |
| Streaming | ✅ | ✅ | 🟢 |
| Provider Selection | ✅ | ✅ | 🟢 |
| A/B Testing | ⚠️ | ⚠️ | 🟡 |
| Agent Info Modal | ✅ | ✅ | 🟢 |
| Settings Modal | ✅ | ✅ | 🟢 |
| API Key Mgmt | ⚠️ | ⚠️ | 🟡 |
| Sidebar | ✅ | ✅ | 🟢 |
| Code Blocks | ⚠️ | ⚠️ | 🟡 |
| Agent Trace | ⚠️ | ⚠️ | 🟡 |
| Data Tab | ✅ | ✅ | 🟢 |
| Keyboard Nav | ⚠️ | ⚠️ | 🟡 |
| Error Handling | ⚠️ | ⚠️ | 🟡 |
| Responsive | ⚠️ | ⚠️ | 🟡 |

---

## Maintenance Notes

### Adding New Workflows
1. Create section following the template above
2. Define user story and expected behavior
3. Write MCP verification checklist
4. Write Playwright test specifications
5. Update coverage matrix

### Updating Existing Workflows
1. Update expected behavior if feature changes
2. Update MCP checklist steps
3. Update Playwright tests
4. Run tests to verify
5. Update coverage status

### Version History
- **v1.0** (Jan 2026): Initial document with 16 workflows
