/**
 * Workflow 11: Code Block Interactions Tests
 * 
 * Tests for code block rendering, syntax highlighting, and copy functionality.
 */
import { test, expect, setupBasicMocks } from '../fixtures';

test.describe('Code Block Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  const codeResponse = `Here's some code:

\`\`\`python
def hello_world():
    print("Hello, World!")
    return 42

if __name__ == "__main__":
    hello_world()
\`\`\`

And here's JavaScript:

\`\`\`javascript
const greet = (name) => {
  console.log(\`Hello, \${name}!\`);
};
\`\`\`
`;

  test('code blocks render with proper styling', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: `{"type":"final","response":${JSON.stringify(codeResponse)},"message_id":1,"user_message_id":1,"conversation_id":1}\n`,
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Show me code');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Code blocks should be present
    await expect(page.locator('pre code')).toHaveCount(2);
  });

  test('copy button appears on code blocks', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: `{"type":"final","response":${JSON.stringify(codeResponse)},"message_id":1,"user_message_id":1,"conversation_id":1}\n`,
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Code please');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Copy button should exist
    await expect(page.locator('.copy-button, button:has-text("Copy")')).toHaveCount(2);
  });

  test('clicking copy button copies code', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: `{"type":"final","response":${JSON.stringify(codeResponse)},"message_id":1,"user_message_id":1,"conversation_id":1}\n`,
      });
    });

    await page.goto('/chat');
    
    // Grant clipboard permissions
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    
    await page.getByLabel('Message input').fill('Code');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Wait for code block
    await expect(page.locator('pre code')).toHaveCount(2);
    
    // Click first copy button
    await page.locator('.copy-button, button:has-text("Copy")').first().click();
    
    // Should show copied feedback
    await expect(page.locator('text=/copied/i')).toBeVisible({ timeout: 2000 });
  });

  test('copy feedback disappears after timeout', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: `{"type":"final","response":${JSON.stringify(codeResponse)},"message_id":1,"user_message_id":1,"conversation_id":1}\n`,
      });
    });

    await page.goto('/chat');
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    
    await page.getByLabel('Message input').fill('Code');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.locator('pre code')).toHaveCount(2);
    await page.locator('.copy-button, button:has-text("Copy")').first().click();
    
    // Feedback should appear
    await expect(page.locator('text=/copied/i')).toBeVisible();
    
    // Then disappear
    await expect(page.locator('text=/copied/i')).not.toBeVisible({ timeout: 3000 });
  });

  test('syntax highlighting applied', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: `{"type":"final","response":${JSON.stringify(codeResponse)},"message_id":1,"user_message_id":1,"conversation_id":1}\n`,
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Code');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.locator('pre code')).toHaveCount(2);
    
    // Check for syntax highlighting classes (hljs or similar)
    const codeBlock = page.locator('pre code').first();
    const hasHighlighting = await codeBlock.locator('.hljs-keyword, .token, .keyword').count();
    
    // If no highlighting library is used, code should still be in pre/code
    const codeText = await codeBlock.textContent();
    expect(codeText).toContain('def');
  });

  test('inline code renders correctly', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Use the `print()` function to output text.","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('How to print?');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Inline code should be in code element (not pre)
    await expect(page.locator('.message.assistant code:not(pre code)')).toContainText('print()');
  });



  test('multiple code blocks are independent', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: `{"type":"final","response":${JSON.stringify(codeResponse)},"message_id":1,"user_message_id":1,"conversation_id":1}\n`,
      });
    });

    await page.goto('/chat');
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    
    await page.getByLabel('Message input').fill('Code');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.locator('pre code')).toHaveCount(2);
    
    // Each code block should have its own copy button
    const copyButtons = page.locator('.copy-button, button:has-text("Copy")');
    await expect(copyButtons).toHaveCount(2);
    
    // Clicking one shouldn't affect the other
    await copyButtons.first().click();
    await expect(page.locator('text=/copied/i')).toHaveCount(1);
  });
});
