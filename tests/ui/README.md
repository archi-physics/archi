# UI Testing Guide

This document describes how to run UI tests for the archi chat interface.

## Test Structure

```
tests/ui/
├── README.md                 # This file
├── fixtures.ts               # Shared test fixtures and utilities
├── ux-workflows.md           # Living UX workflows documentation
├── manual-checks.md          # MCP-executable visual checks
├── chat.spec.ts              # Legacy monolithic test file (18 tests)
└── workflows/                # Workflow-organized tests
    ├── 01-page-load.spec.ts      # Page initialization (10 tests)
    ├── 02-conversations.spec.ts  # Conversation management (6 tests)
    ├── 03-message-flow.spec.ts   # Message sending/rendering (14 tests)
    ├── 04-streaming.spec.ts      # Streaming & cancellation (8 tests)
    ├── 05-providers.spec.ts      # Provider/model selection (13 tests)
    ├── 06-ab-testing.spec.ts     # A/B testing mode (9 tests)
    ├── 08-settings.spec.ts       # Settings modal (9 tests)
    ├── 09-api-keys.spec.ts       # API key management (9 tests)
    ├── 10-sidebar.spec.ts        # Sidebar navigation (12 tests)
    ├── 11-code-blocks.spec.ts    # Code block rendering (9 tests)
    ├── 12-trace.spec.ts          # Agent trace visualization (10 tests)
    ├── 13-data-tab.spec.ts       # Data tab navigation (10 tests)
    ├── 14-keyboard.spec.ts       # Keyboard navigation (10 tests)
    ├── 15-errors.spec.ts         # Error handling (12 tests)
    └── 16-responsive.spec.ts     # Responsive layout (13 tests)
```

## Prerequisites

1. **Playwright installed**:
   ```bash
   npm install
   npx playwright install
   ```

2. **Local deployment running**:
   ```bash
   # Start test database
   cd tests/smoke && docker compose -f docker-compose.integration.yaml up -d && cd ../..
   
   # Start the chat app
   ./scripts/dev/run_chat_local.sh
   ```
   The tests expect the chat app at `http://localhost:2786` (configurable via `BASE_URL` env var).

## Running Playwright Tests

### All UI Tests
```bash
npx playwright test tests/ui/
```

### All Workflow Tests
```bash
npx playwright test tests/ui/workflows/
```

### Specific Workflow
```bash
npx playwright test tests/ui/workflows/01-page-load.spec.ts
npx playwright test tests/ui/workflows/04-streaming.spec.ts
```

### By Workflow Pattern
```bash
# All message-related tests
npx playwright test tests/ui/workflows/0[1-3]-*

# All modal tests
npx playwright test tests/ui/workflows/0[7-8]-*
```

### With UI (headed mode)
```bash
npx playwright test tests/ui/workflows/ --headed
```

### Debug Mode
```bash
npx playwright test tests/ui/workflows/03-message-flow.spec.ts --debug
```

## Test Categories by Workflow

| # | Workflow | Tests | Description |
|---|----------|-------|-------------|
| 01 | Page Load | 10 | Initial load, required elements, config dropdown |
| 02 | Conversations | 6 | New chat, switching, deleting conversations |
| 03 | Message Flow | 14 | Sending messages, rendering, markdown |
| 04 | Streaming | 8 | Real-time streaming, cancellation, cursor |
| 05 | Providers | 13 | Provider/model selection, custom models |
| 06 | A/B Testing | 9 | Dual panels, voting, blind comparison |
| 08 | Settings | 9 | Settings persistence, theme options |
| 09 | API Keys | 9 | Key management, validation |
| 10 | Sidebar | 12 | Tab switching, collapse/expand |
| 11 | Code Blocks | 9 | Syntax highlighting, copy functionality |
| 12 | Trace | 10 | Tool calls, trace visualization |
| 13 | Data Tab | 10 | Data sources, document browsing |
| 14 | Keyboard | 10 | Keyboard navigation, shortcuts |
| 15 | Errors | 12 | Error handling, recovery |
| 16 | Responsive | 13 | Desktop/tablet/mobile layouts |

**Total: ~144 tests across 15 workflow files**

## Using Fixtures

All workflow tests import from `fixtures.ts`:

```typescript
import { test, expect, setupBasicMocks, mockData } from './fixtures';

test.describe('My Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('my test', async ({ page }) => {
    await page.goto('/chat');
    // ...
  });
});
```

### Available Fixtures

- **`mockData`** - Pre-configured mock data for all API responses
- **`setupBasicMocks(page)`** - Sets up standard API mocks
- **`setupStreamMock(page, response, delay)`** - Mocks streaming responses
- **`createStreamResponse(content, options)`** - Generates stream response body
- **`createToolCallEvents(name, args, output)`** - Generates tool call events
- **`enableABMode(page)`** - Enables A/B testing mode
- **`clearStorage(page)`** - Clears localStorage/sessionStorage
- **`test`** - Extended test fixture with `chatPage`

## MCP Manual Validation

For additional visual and interaction quality checks, see:
- [Manual Checks Guide](manual-checks.md)
- [UX Workflows Document](ux-workflows.md)

These checks use Chrome DevTools MCP for browser automation to verify:
- Visual quality (styling, contrast, spacing)
- Interaction quality (tab order, focus states, modal behavior)
- Layout (responsive, scrolling, overflow)
- Console health (no JS errors, no failed requests)

## Adding New Tests

### Recommended: Add to Workflow File
```typescript
// In workflows/XX-category.spec.ts
import { test, expect, setupBasicMocks } from './fixtures';

test.describe('Category Name', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('new feature works correctly', async ({ page }) => {
    await page.goto('/chat');
    
    // Action
    await page.getByRole('button', { name: 'Action' }).click();
    
    // Assert
    await expect(page.locator('.result')).toBeVisible();
  });
});
```

### Mock Streaming Responses
```typescript
test('streaming works', async ({ page }) => {
  await page.route('**/api/get_chat_response_stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/plain',
      body: createStreamResponse('Hello world!'),
    });
  });
  // ...
});
```

### Test Pattern
  await expect(page.locator('.result')).toBeVisible();
});
```

### MCP Check Pattern
```markdown
## Check: Feature Name
1. Navigate to `/chat`
2. Perform action
3. Verify expected state
4. Check console for errors

### Pass Criteria
- [ ] Expected element visible
- [ ] No console errors
- [ ] Correct styling applied
```

## Test Configuration

Tests are configured in `playwright.config.ts`:
- Base URL: `http://localhost:7861`
- Browser: Chromium
- Timeout: 30 seconds per test
- Screenshots: On failure

## Troubleshooting

### Tests fail with "navigation timeout"
- Ensure the local deployment is running
- Check that port 7861 is accessible

### Tests fail with "element not found"
- Selectors may have changed; update locators
- Check if the element requires user interaction first

### Flaky tests
- Add explicit waits: `await expect(locator).toBeVisible()`
- Use network mocking to ensure consistent API responses
