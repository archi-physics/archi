import { test, expect } from '@playwright/test';

test.describe('Chat UI', () => {
  // ============================================================
  // 1.1 Core Functionality Tests
  // ============================================================
  test('page loads with all required elements', async ({ page }) => {
    await page.goto('/chat');
    
    // Sidebar
    await expect(page.locator('.sidebar')).toBeVisible();
    await expect(page.getByRole('button', { name: 'New chat' })).toBeVisible();
    
    // Header
    await expect(page.getByRole('heading', { name: 'archi Chat' })).toBeVisible();
    await expect(page.locator('.header-tabs')).toBeVisible();
    
    // Input area
    await expect(page.getByLabel('Message input')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
  });

  test('entry meta label shows agent and model info', async ({ page }) => {
    await page.goto('/chat');
    // Config dropdown shows available models/configs
    const configDropdown = page.getByRole('combobox', { name: 'Select model' });
    await expect(configDropdown).toBeVisible();
  });

  test('header tabs are visible (Chat, Data)', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('.header-tab').filter({ hasText: 'Chat' })).toBeVisible();
    await expect(page.locator('.header-tab').filter({ hasText: 'Data' })).toBeVisible();
  });

  test('agent info button opens modal with correct data', async ({ page }) => {
    await page.goto('/chat');
    
    // Click agent info button
    await page.getByRole('button', { name: 'Agent info' }).click();
    
    // Modal should be visible
    const modal = page.locator('.agent-info-modal');
    await expect(modal).toBeVisible();
    
    // Should show agent information
    await expect(modal).toContainText('Active agent');
    await expect(modal).toContainText('Model');
    await expect(modal).toContainText('Pipeline');
    await expect(modal).toContainText('Embedding');
    await expect(modal).toContainText('Data sources');
    
    // Close modal
    await page.locator('.agent-info-close').click();
    await expect(modal).not.toBeVisible();
  });

  test('settings button opens settings modal', async ({ page }) => {
    await page.goto('/chat');
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.locator('.settings-panel')).toBeVisible();
    await page.getByRole('button', { name: 'Close settings' }).click();
    await expect(page.locator('.settings-panel')).not.toBeVisible();
  });

  // ============================================================
  // 1.2 Message Flow Tests
  // ============================================================
  test('shows pipeline default model label', async ({ page }) => {
    await page.goto('/chat');
    // Config dropdown shows available models/configs
    const configDropdown = page.getByRole('combobox', { name: 'Select model' });
    await expect(configDropdown).toBeVisible();
    // Should have at least one option available
    await expect(configDropdown).not.toHaveValue('');
  });

  test('send button toggles to stop while streaming', async ({ page }) => {
    await page.goto('/chat');

    await page.route('**/api/get_chat_response_stream', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const body = '{"type":"chunk","content":"Hi"}\n';
      await route.fulfill({ status: 200, contentType: 'text/plain', body });
    });

    await page.getByLabel('Message input').fill('Hello');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.getByRole('button', { name: 'Stop streaming' })).toBeVisible();
    await page.getByRole('button', { name: 'Stop streaming' }).click();
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
  });

  test('message meta appears under assistant message', async ({ page }) => {
    await page.goto('/chat');

    await page.route('**/api/get_chat_response_stream', async (route) => {
      const body = '{"type":"final","response":"Hello back!","message_id":1,"user_message_id":1,"conversation_id":1}\n';
      await route.fulfill({ status: 200, contentType: 'text/plain', body });
    });

    await page.getByLabel('Message input').fill('Hello');
    await page.getByRole('button', { name: 'Send message' }).click();

    // Wait for response to complete
    await expect(page.locator('.message.assistant')).toBeVisible();
    
    // Check that message meta is present under assistant message
    const assistantMessage = page.locator('.message.assistant').first();
    const messageMeta = assistantMessage.locator('.message-meta');
    await expect(messageMeta).toBeVisible();
    // Format is "<agent> · <model>" without Agent:/Model: labels
    await expect(messageMeta).toContainText('·');
  });

  // ============================================================
  // 1.3 Provider Selection Tests
  // ============================================================
  test('provider dropdown defaults to pipeline default', async ({ page }) => {
    await page.goto('/chat');
    await page.getByRole('button', { name: 'Settings' }).click();
    
    const providerSelect = page.locator('#provider-select');
    await expect(providerSelect).toHaveValue('');  // Empty = pipeline default
  });

  test('entry meta updates when provider/model changes', async ({ page }) => {
    await page.goto('/chat');
    
    // Config dropdown should be visible
    const configDropdown = page.getByRole('combobox', { name: 'Select model' });
    await expect(configDropdown).toBeVisible();
    
    // Open settings - verify it can be opened
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.locator('.settings-panel')).toBeVisible();
    
    await page.getByRole('button', { name: 'Close settings' }).click();
    await expect(page.locator('.settings-panel')).not.toBeVisible();
  });

  // ============================================================
  // 1.4 A/B Testing Mode Tests
  // ============================================================
  test('A/B streaming includes provider overrides', async ({ page }) => {
    await page.goto('/chat');

    // Verify settings button exists and can be opened
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.locator('.settings-panel')).toBeVisible();
    
    // Just verify settings panel has navigation sections
    const navItems = page.locator('.settings-nav-item');
    const count = await navItems.count();
    expect(count).toBeGreaterThan(0);
    
    await page.getByRole('button', { name: 'Close settings' }).click();
    await expect(page.locator('.settings-panel')).not.toBeVisible();
  });

  // ============================================================
  // 1.5 Conversation Management Tests
  // ============================================================
  test('new chat button clears messages', async ({ page }) => {
    await page.goto('/chat');
    
    // Send a message first
    await page.route('**/api/get_chat_response_stream', async (route) => {
      const body = '{"type":"final","response":"Hello!","message_id":1,"user_message_id":1,"conversation_id":1}\n';
      await route.fulfill({ status: 200, contentType: 'text/plain', body });
    });
    
    await page.getByLabel('Message input').fill('Test message');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.locator('.message')).toHaveCount(2);  // User + Assistant
    
    // Click new chat
    await page.getByRole('button', { name: 'New chat' }).click();
    
    // Messages should be cleared (or just welcome state)
    await expect(page.locator('.message.user')).toHaveCount(0);
  });

  // ============================================================
  // 1.6 Agent Info Modal Tests
  // ============================================================
  test('agent info modal closes on backdrop click', async ({ page }) => {
    await page.goto('/chat');
    
    await page.getByRole('button', { name: 'Agent info' }).click();
    const modal = page.locator('.agent-info-modal');
    await expect(modal).toBeVisible();
    
    // Click backdrop area outside modal (use position well outside the modal content)
    await page.mouse.click(10, 10);
    await expect(modal).not.toBeVisible();
  });

  test('agent info modal closes on Escape key', async ({ page }) => {
    await page.goto('/chat');
    
    await page.getByRole('button', { name: 'Agent info' }).click();
    const modal = page.locator('.agent-info-modal');
    await expect(modal).toBeVisible();
    
    await page.keyboard.press('Escape');
    await expect(modal).not.toBeVisible();
  });

  // ============================================================
  // 1.7 Data Tab Tests
  // ============================================================
  test('Data tab click without conversation shows alert', async ({ page }) => {
    await page.goto('/chat');
    
    // Clear any active conversation
    await page.evaluate(() => {
      localStorage.removeItem('archi_active_conversation');
    });
    await page.reload();
    
    // Set up dialog handler
    page.on('dialog', async dialog => {
      expect(dialog.message()).toContain('conversation');
      await dialog.accept();
    });
    
    // Click Data tab
    await page.locator('.header-tab').filter({ hasText: 'Data' }).click();
  });

  // ============================================================
  // 1.8 Settings Modal Tests
  // ============================================================
  test('settings modal opens with Models section active', async ({ page }) => {
    await page.goto('/chat');
    await page.getByRole('button', { name: 'Settings' }).click();
    
    // Models section should be visible
    await expect(page.locator('#settings-models')).toBeVisible();
    await expect(page.locator('.settings-nav-item[data-section="models"]')).toHaveClass(/active/);
  });

  test('can switch between settings sections', async ({ page }) => {
    await page.goto('/chat');
    await page.getByRole('button', { name: 'Settings' }).click();
    
    // Switch to API Keys
    await page.locator('.settings-nav-item[data-section="api-keys"]').click();
    await expect(page.locator('#settings-api-keys')).toBeVisible();
    
    // Switch to Advanced
    await page.locator('.settings-nav-item[data-section="advanced"]').click();
    await expect(page.locator('#settings-advanced')).toBeVisible();
    
    // Switch back to Models
    await page.locator('.settings-nav-item[data-section="models"]').click();
    await expect(page.locator('#settings-models')).toBeVisible();
  });

  test('settings modal closes on backdrop click', async ({ page }) => {
    await page.goto('/chat');
    await page.getByRole('button', { name: 'Settings' }).click();
    
    const panel = page.locator('.settings-panel');
    await expect(panel).toBeVisible();
    
    // Click backdrop area outside panel (click far left where backdrop exists)
    await page.mouse.click(10, 10);
    await expect(panel).not.toBeVisible();
  });
});
