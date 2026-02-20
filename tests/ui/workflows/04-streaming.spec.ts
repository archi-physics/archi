/**
 * Workflow 4: Streaming & Cancellation Tests
 * 
 * Tests for real-time streaming responses and stream cancellation.
 */
import { test, expect, setupBasicMocks } from '../fixtures';

test.describe('Streaming & Cancellation', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('send button toggles to stop while streaming', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      // Delay response to simulate streaming
      await new Promise(resolve => setTimeout(resolve, 1000));
      const body = '{"type":"chunk","content":"Hi"}\n';
      await route.fulfill({ status: 200, contentType: 'text/plain', body });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hello');
    await page.getByRole('button', { name: 'Send message' }).click();

    // Button should change to stop
    const stopBtn = page.getByRole('button', { name: 'Stop streaming' });
    await expect(stopBtn).toBeVisible();
    
    // Click stop
    await stopBtn.click();
    
    // Button should revert to send
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
  });

  test('input is disabled during streaming', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Done","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Input should be disabled during streaming
    await expect(page.getByLabel('Message input')).toBeDisabled();
    
    // Wait for streaming to complete
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
    
    // Input should be re-enabled
    await expect(page.getByLabel('Message input')).not.toBeDisabled();
  });

  test('streaming cursor visible during response', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"chunk","content":"Streaming..."}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hi');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Streaming cursor should appear
    await expect(page.locator('.streaming-cursor')).toBeVisible();
  });

  test('streaming cursor disappears after completion', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Done","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Hi');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Wait for completion
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
    
    // Streaming cursor should be gone
    await expect(page.locator('.streaming-cursor')).toHaveCount(0);
  });

  test('input re-enabled after streaming error', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({ status: 500, body: 'Internal Server Error' });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Wait for error to be handled
    await expect(page.getByLabel('Message input')).not.toBeDisabled();
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
  });

  test('input focused after streaming completes', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Done","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Wait for completion
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
    
    // Input should be focused
    await expect(page.getByLabel('Message input')).toBeFocused();
  });

  test('cannot send message during streaming', async ({ page }) => {
    let sendCount = 0;
    
    await page.route('**/api/get_chat_response_stream', async (route) => {
      sendCount++;
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Done","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    // Send first message
    await page.getByLabel('Message input').fill('First');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Try to send second message while streaming
    await page.evaluate(() => {
      const input = document.querySelector('.input-field') as HTMLTextAreaElement;
      if (input) {
        input.disabled = false;
        input.value = 'Second';
      }
    });
    
    // Button should be in stop mode, clicking should cancel not send
    await page.getByRole('button', { name: 'Stop streaming' }).click();
    
    // Only one request should have been made
    expect(sendCount).toBe(1);
  });
});
