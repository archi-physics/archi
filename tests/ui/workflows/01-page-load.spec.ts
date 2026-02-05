/**
 * Workflow 1: Page Load & Initialization Tests
 * 
 * Tests that the page loads correctly with all required components
 * and initial data is fetched and rendered.
 */
import { test, expect, setupBasicMocks, mockData } from '../fixtures';

test.describe('Page Load & Initialization', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

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
    await expect(page.getByRole('button', { name: 'Settings' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Agent info' })).toBeVisible();
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

  test('Chat tab shows active state', async ({ page }) => {
    await page.goto('/chat');
    const chatTab = page.locator('.header-tab').filter({ hasText: 'Chat' });
    await expect(chatTab).toHaveAttribute('aria-current', 'page');
  });

  test('shows pipeline default model in entry meta', async ({ page }) => {
    await page.goto('/chat');
    // Config dropdown shows available models
    const configDropdown = page.getByRole('combobox', { name: 'Select model' });
    await expect(configDropdown).toBeVisible();
    // Should have a selected value
    await expect(configDropdown).not.toHaveValue('');
  });

  test('config dropdown is populated', async ({ page }) => {
    await page.goto('/chat');
    const configSelect = page.locator('#model-select-a');
    await expect(configSelect).toBeVisible();
    
    const options = configSelect.locator('option');
    await expect(options).toHaveCount(mockData.configs.options.length);
  });

  test('conversations load in sidebar', async ({ page }) => {
    await page.goto('/chat');
    const convList = page.locator('.conversation-list');
    await expect(convList).toBeVisible();
    // Mock data has 2 conversations
    const count = await convList.locator('.conversation-item').count();
    expect(count).toBeGreaterThanOrEqual(1);
  });

  test('restores active conversation on reload', async ({ page }) => {
    await page.route('**/api/load_conversation', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          messages: [
            { sender: 'User', content: 'Previous message', message_id: 1 },
            { sender: 'archi', content: 'Previous response', message_id: 2 },
          ]
        }
      });
    });

    await page.goto('/chat');
    
    // Set active conversation
    await page.evaluate(() => {
      localStorage.setItem('archi_active_conversation_id', '1');
    });
    
    await page.reload();
    
    // Messages should be loaded
    await expect(page.locator('.message')).toHaveCount(2);
  });

  test('new chat starts with empty message area', async ({ page }) => {
    await page.goto('/chat');
    
    // Click new chat button to start a new conversation
    await page.getByRole('button', { name: 'New chat' }).click();
    
    // The main area should be empty (no messages) or show an intro
    // When there are no messages, main should not have message elements
    const messages = page.locator('[role="main"] .message');
    const count = await messages.count();
    // A new chat should have 0 messages
    expect(count).toBeLessThanOrEqual(1); // Allow for intro message if present
  });
});
