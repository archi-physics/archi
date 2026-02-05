/**
 * Workflow 3: Message Flow Tests
 * 
 * Tests for sending messages, receiving responses, and message rendering.
 */
import { test, expect, setupBasicMocks, createStreamResponse } from '../fixtures';

test.describe('Message Flow', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('user can send message with Enter key', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('Hello!'),
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hello');
    await page.keyboard.press('Enter');
    
    await expect(page.locator('.message.user')).toBeVisible();
    await expect(page.locator('.message.assistant')).toBeVisible();
  });

  test('user can send message with Send button', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('Hello!'),
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hello');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.locator('.message.user')).toContainText('Hello');
  });

  test('Shift+Enter adds newline instead of sending', async ({ page }) => {
    await page.goto('/chat');
    
    const textarea = page.getByLabel('Message input');
    await textarea.fill('Line 1');
    await textarea.press('Shift+Enter');
    await textarea.type('Line 2');
    
    const value = await textarea.inputValue();
    expect(value).toContain('\n');
    expect(value).toContain('Line 1');
    expect(value).toContain('Line 2');
    
    // Message should NOT be sent
    await expect(page.locator('.message.user')).toHaveCount(0);
  });

  test('textarea auto-resizes on input', async ({ page }) => {
    await page.goto('/chat');
    
    const textarea = page.getByLabel('Message input');
    const initialHeight = await textarea.evaluate((el: HTMLElement) => el.offsetHeight);
    
    await textarea.fill('Line 1\nLine 2\nLine 3\nLine 4\nLine 5');
    const newHeight = await textarea.evaluate((el: HTMLElement) => el.offsetHeight);
    
    expect(newHeight).toBeGreaterThan(initialHeight);
  });

  test('input clears after sending', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('OK'),
      });
    });

    await page.goto('/chat');
    
    const textarea = page.getByLabel('Message input');
    await textarea.fill('Test message');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(textarea).toHaveValue('');
  });

  test('message meta appears under assistant message only', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('Hello back!'),
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hi');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Wait for response
    await expect(page.locator('.message.assistant')).toBeVisible();
    
    // User message has no meta
    const userMsg = page.locator('.message.user').first();
    await expect(userMsg.locator('.message-meta')).toHaveCount(0);
    
    // Assistant message has meta - format is "<agent> · <model>" without labels
    const assistantMsg = page.locator('.message.assistant').first();
    await expect(assistantMsg.locator('.message-meta')).toBeVisible();
    await expect(assistantMsg.locator('.message-meta')).toContainText('·');
  });

  test('user message shows sender as "You"', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('OK'),
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hello');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    const userMsg = page.locator('.message.user');
    await expect(userMsg.locator('.message-sender')).toContainText('You');
  });

  test('assistant message shows sender as "archi"', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('Hello!'),
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hi');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    const assistantMsg = page.locator('.message.assistant');
    await expect(assistantMsg.locator('.message-sender')).toContainText('archi');
  });

  test('markdown is rendered in responses', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('Here is **bold** and *italic* text'),
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test markdown');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    const content = page.locator('.message.assistant .message-content');
    await expect(content.locator('strong')).toContainText('bold');
    await expect(content.locator('em')).toContainText('italic');
  });

  test('code blocks are rendered with syntax highlighting', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: createStreamResponse('```python\nprint("Hello")\n```'),
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Show code');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.locator('pre code')).toBeVisible();
    await expect(page.locator('.code-block-lang')).toContainText('python');
    await expect(page.locator('.code-block-copy')).toBeVisible();
  });

  test('cannot send empty message', async ({ page }) => {
    let messageSent = false;
    
    await page.route('**/api/get_chat_response_stream', async (route) => {
      messageSent = true;
      await route.fulfill({ status: 200, body: '' });
    });

    await page.goto('/chat');
    
    // Try to send empty message
    await page.getByRole('button', { name: 'Send message' }).click();
    
    expect(messageSent).toBe(false);
    await expect(page.locator('.message.user')).toHaveCount(0);
  });

  test('cannot send whitespace-only message', async ({ page }) => {
    let messageSent = false;
    
    await page.route('**/api/get_chat_response_stream', async (route) => {
      messageSent = true;
      await route.fulfill({ status: 200, body: '' });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('   \n\t  ');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    expect(messageSent).toBe(false);
  });
});
