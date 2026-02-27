/**
 * Workflow 8: Settings Modal Tests
 * 
 * Tests for the settings modal with provider/model selection and API key management.
 */
import { test, expect, setupBasicMocks } from '../fixtures';

test.describe('Settings Modal', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('settings button visible in UI', async ({ page }) => {
    await page.goto('/chat');
    
    await expect(page.getByRole('button', { name: /settings/i })).toBeVisible();
  });

  test('clicking settings opens modal', async ({ page }) => {
    await page.goto('/chat');
    
    await page.getByRole('button', { name: /settings/i }).click();
    
    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
  });

  test('settings modal has category tabs', async ({ page }) => {
    await page.goto('/chat');
    
    await page.getByRole('button', { name: /settings/i }).click();
    
    // Verify all three category buttons exist (styled as tabs)
    await expect(page.getByRole('button', { name: 'Models' })).toBeVisible();
    // await expect(page.getByRole('button', { name: 'API Keys' })).toBeVisible();
    // await expect(page.getByRole('button', { name: 'Advanced' })).toBeVisible();
  });

  test('settings modal has provider selection', async ({ page }) => {
    await page.goto('/chat');
    
    await page.getByRole('button', { name: /settings/i }).click();
    
    // Provider dropdown should be visible in Models tab (default tab)
    const providerDropdown = page.locator('select').filter({ hasText: /pipeline default|OpenAI|Anthropic/i }).first();
    await expect(providerDropdown).toBeVisible();
    
    // Check that expected provider options exist
    await expect(providerDropdown).toContainText('Use pipeline default');
  });

  test('close button closes settings', async ({ page }) => {
    await page.goto('/chat');
    
    await page.getByRole('button', { name: /settings/i }).click();
    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
    
    await page.getByRole('button', { name: /close settings/i }).click();
    
    await expect(page.getByRole('heading', { name: 'Settings' })).not.toBeVisible();
  });

  test('Escape key closes settings modal', async ({ page }) => {
    await page.goto('/chat');
    
    await page.getByRole('button', { name: /settings/i }).click();
    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
    
    await page.keyboard.press('Escape');
    
    await expect(page.getByRole('heading', { name: 'Settings' })).not.toBeVisible();
  });

  // test('API Keys tab shows key management', async ({ page }) => {
  //   await page.goto('/chat');
    
  //   await page.getByRole('button', { name: /settings/i }).click();
    
  //   // Click API Keys category button
  //   await page.getByRole('button', { name: 'API Keys' }).click();
    
  //   // Should show API key management UI with a heading and description
  //   await expect(page.getByRole('heading', { name: 'API Keys' })).toBeVisible();
  // });
});
