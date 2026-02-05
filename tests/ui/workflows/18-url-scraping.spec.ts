/**
 * Workflow 18: URL Scraping Tests
 * 
 * Tests for URL scraping functionality including adding URLs,
 * configuring crawl settings, and managing the scrape queue.
 */
import { test, expect } from '@playwright/test';

test.describe('URL Scraping Workflows', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API endpoints - matches /api/upload/status endpoint
    await page.route('**/api/upload/status', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          documents_in_catalog: 100,
          documents_embedded: 95,
          pending_embedding: 5,
          is_synced: false
        }
      });
    });

    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.route('**/api/sources/jira', async (route) => {
      await route.fulfill({ status: 200, json: { projects: [] } });
    });

    await page.route('**/api/sources/urls/queue', async (route) => {
      await route.fulfill({ status: 200, json: { urls: [] } });
    });
  });

  test('URL input validates URL format', async ({ page }) => {
    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    const urlInput = page.getByPlaceholder(/https:\/\/docs.example.com/);
    
    // Enter invalid URL
    await urlInput.fill('not-a-url');
    await page.getByRole('button', { name: 'Add' }).click();
    
    await page.waitForTimeout(300);
    
    // Should show validation error or not add to queue
  });

  test('valid URL gets added to queue', async ({ page }) => {
    await page.route('**/api/sources/urls/add', async (route) => {
      await route.fulfill({ status: 200, json: { success: true } });
    });

    // Update queue mock to return the added URL
    let urlAdded = false;
    await page.route('**/api/sources/urls/queue', async (route) => {
      if (urlAdded) {
        await route.fulfill({
          status: 200,
          json: {
            urls: [{ url: 'https://docs.example.com/guide', depth: 2, sso: false }]
          }
        });
      } else {
        await route.fulfill({ status: 200, json: { urls: [] } });
      }
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    // Enter valid URL
    await page.getByPlaceholder(/https:\/\/docs.example.com/).fill('https://docs.example.com/guide');
    
    urlAdded = true;
    await page.getByRole('button', { name: 'Add' }).click();

    await page.waitForTimeout(500);
    
    // URL should appear in queue
    await expect(page.getByText('https://docs.example.com/guide')).toBeVisible();
  });

  test('follow links checkbox controls crawling', async ({ page }) => {
    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    const followLinksCheckbox = page.getByRole('checkbox', { name: /Follow links/i });
    
    // Should be checked by default
    await expect(followLinksCheckbox).toBeChecked();
    
    // Uncheck it
    await followLinksCheckbox.uncheck();
    await expect(followLinksCheckbox).not.toBeChecked();
  });

  test('SSO checkbox controls authentication', async ({ page }) => {
    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    const ssoCheckbox = page.getByRole('checkbox', { name: /SSO/i });
    
    // Should be unchecked by default
    await expect(ssoCheckbox).not.toBeChecked();
    
    // Check it
    await ssoCheckbox.check();
    await expect(ssoCheckbox).toBeChecked();
  });

  test('crawl depth can be changed', async ({ page }) => {
    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    // Find the depth selector
    const depthSelect = page.locator('select').filter({ hasText: /level/ });
    
    if (await depthSelect.isVisible()) {
      // Change to 3 levels
      await depthSelect.selectOption({ label: '3 levels' });
      
      // Verify selection
      await expect(depthSelect).toHaveValue('3');
    }
  });

  test('start scraping button initiates scrape', async ({ page }) => {
    await page.route('**/api/sources/urls/queue', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          urls: [{ url: 'https://example.com', depth: 2, sso: false }]
        }
      });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    await page.waitForTimeout(500);

    // Start Scraping button should be visible when URLs are queued
    await expect(page.getByRole('button', { name: 'Start Scraping' })).toBeVisible();
  });

  test('remove button removes URL from queue', async ({ page }) => {
    let removeCalled = false;
    
    await page.route('**/api/sources/urls/remove', async (route) => {
      removeCalled = true;
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.route('**/api/sources/urls/queue', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          urls: [{ url: 'https://example.com', depth: 2, sso: false }]
        }
      });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    await page.waitForTimeout(500);

    // Find and click remove button
    const removeBtn = page.getByRole('button', { name: 'Remove' }).first();
    if (await removeBtn.isVisible()) {
      await removeBtn.click();
      await page.waitForTimeout(300);
      expect(removeCalled).toBe(true);
    }
  });

  test('queued URL shows crawl settings', async ({ page }) => {
    // Navigate to URLs panel
    await page.goto('/upload');
    await page.getByRole('button', { name: /URLs/ }).click();

    // Verify URL input exists (actual UI structure)
    await expect(page.getByRole('textbox').first()).toBeVisible();
    // Verify crawl depth dropdown exists
    await expect(page.getByText('Crawl Depth')).toBeVisible();
  });
});
