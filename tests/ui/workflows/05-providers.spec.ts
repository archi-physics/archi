/**
 * Workflow 5: Provider & Model Selection Tests
 * 
 * Tests for selecting providers and models via the Settings modal.
 * The provider/model selection is in Settings > Models tab.
 */
import { test, expect, setupBasicMocks } from '../fixtures';

test.describe('Provider & Model Selection', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('agent dropdown shows available agents', async ({ page }) => {
    await page.goto('/chat');
    
    const agentDropdown = page.locator('.agent-dropdown');
    await expect(agentDropdown).toBeVisible();

    await page.locator('.agent-dropdown-btn').click();
    const agentItems = page.locator('.agent-dropdown-item');
    const optionCount = await agentItems.count();
    expect(optionCount).toBeGreaterThanOrEqual(1);
  });

  test('settings modal opens and shows Models tab', async ({ page }) => {
    await page.goto('/chat');
    
    // Open settings
    await page.getByRole('button', { name: 'Settings' }).click();
    
    // Should show Settings modal with Models tab
    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Models' })).toBeVisible();
  });

  test('provider dropdown lists available providers', async ({ page }) => {
    await page.goto('/chat');
    
    // Open settings
    await page.getByRole('button', { name: 'Settings' }).click();
    
    const providerSelect = page.locator('#provider-select');
    await expect(providerSelect).toBeVisible();
    await page.waitForFunction(() => {
      const select = document.querySelector('#provider-select');
      return select && select.options.length > 1;
    });
    
    // Should have pipeline default plus providers (real UI has 5: default + 4 providers)
    const optionCount = await providerSelect.locator('option').count();
    expect(optionCount).toBeGreaterThanOrEqual(2); // At least default + one provider
  });

  test('selecting provider enables model dropdown', async ({ page }) => {
    await page.goto('/chat');
    
    // Open settings
    await page.getByRole('button', { name: 'Settings' }).click();
    
    const modelSelect = page.locator('#model-select-primary');
    
    // Model dropdown starts disabled when using pipeline default
    await expect(modelSelect).toBeDisabled();
    
    const providerSelect = page.locator('#provider-select');
    await page.waitForFunction(() => {
      const select = document.querySelector('#provider-select');
      return select && select.options.length > 1;
    });
    const providerValues = await providerSelect.evaluate((select) =>
      Array.from(select.options).map((option) => option.value),
    );
    const providerValue = providerValues.find((value) => value);
    if (providerValue) {
      await providerSelect.selectOption(providerValue);
    }
    
    // Model dropdown should now be enabled
    await expect(modelSelect).toBeEnabled();
  });

  test('model dropdown lists models for selected provider', async ({ page }) => {
    await page.goto('/chat');
    
    // Open settings
    await page.getByRole('button', { name: 'Settings' }).click();
    
    const providerSelect = page.locator('#provider-select');
    await page.waitForFunction(() => {
      const select = document.querySelector('#provider-select');
      return select && select.options.length > 1;
    });
    const providerValues = await providerSelect.evaluate((select) =>
      Array.from(select.options).map((option) => option.value),
    );
    const providerValue = providerValues.find((value) => value);
    if (providerValue) {
      await providerSelect.selectOption(providerValue);
    }
    
    // Model dropdown should have options
    const modelSelect = page.locator('#model-select-primary');
    const optionCount = await modelSelect.locator('option').count();
    expect(optionCount).toBeGreaterThan(0);
  });

  test('pipeline default option clears provider selection', async ({ page }) => {
    await page.goto('/chat');
    
    // Open settings
    await page.getByRole('button', { name: 'Settings' }).click();
    
    const providerSelect = page.locator('#provider-select');
    const modelSelect = page.locator('#model-select-primary');
    
    // Select a provider first
    await page.waitForFunction(() => {
      const select = document.querySelector('#provider-select');
      return select && select.options.length > 1;
    });
    const providerValues = await providerSelect.evaluate((select) =>
      Array.from(select.options).map((option) => option.value),
    );
    const providerValue = providerValues.find((value) => value);
    if (providerValue) {
      await providerSelect.selectOption(providerValue);
    }
    
    // Model should be enabled
    await expect(modelSelect).toBeEnabled();
    
    // Select pipeline default
    await providerSelect.selectOption('');
    
    // Model should be disabled again
    await expect(modelSelect).toBeDisabled();
  });

  // test('API Keys tab is accessible', async ({ page }) => {
  //   await page.goto('/chat');
    
  //   // Open settings
  //   await page.getByRole('button', { name: 'Settings' }).click();
    
  //   // Click API Keys tab
  //   await page.getByRole('button', { name: 'API Keys' }).click();
    
  //   // Should show API Keys content - modal should still be open
  //   await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  // });

  // test('Advanced tab is accessible', async ({ page }) => {
  //   await page.goto('/chat');
    
  //   // Open settings
  //   await page.getByRole('button', { name: 'Settings' }).click();
    
  //   // Click Advanced tab
  //   await page.getByRole('button', { name: 'Advanced' }).click();
    
  //   // Should show Advanced settings content
  //   // Just verify the tab was clickable and modal is still open
  //   await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  // });

  test('close button closes settings modal', async ({ page }) => {
    await page.goto('/chat');
    
    // Open settings
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
    
    // Close via button
    await page.getByRole('button', { name: 'Close settings' }).click();
    
    // Modal should be gone
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).not.toBeVisible();
  });
});
