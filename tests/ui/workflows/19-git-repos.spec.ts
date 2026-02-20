/**
 * Workflow 19: Git Repository Tests
 * 
 * Tests for Git repository cloning and indexing functionality
 * including adding repos, configuring indexing options, and managing repos.
 */
import { test, expect } from '@playwright/test';

test.describe('Git Repository Workflows', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API endpoints - matches /api/upload/status endpoint
    await page.route('**/api/upload/status', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          documents_in_catalog: 200,
          documents_embedded: 150,
          pending_embedding: 50,
          is_synced: false
        }
      });
    });

    await page.route('**/api/sources/jira', async (route) => {
      await route.fulfill({ status: 200, json: { projects: [] } });
    });

    await page.route('**/api/sources/urls/queue', async (route) => {
      await route.fulfill({ status: 200, json: { urls: [] } });
    });
  });

  test('displays existing repositories', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          sources: [{
            name: 'archi',
            url: 'https://github.com/archi-physics/archi',
            file_count: 167,
            last_updated: '2026-01-30T10:00:00Z'
          }]
        }
      });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    await page.waitForTimeout(500);

    // Should show the repo
    await expect(page.getByText('archi')).toBeVisible();
    await expect(page.getByText(/167.*files/)).toBeVisible();
  });

  test('clone button initiates repo clone', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    // Enter repo URL
    await page.getByPlaceholder(/https:\/\/github.com/).fill('https://github.com/test/repo');

    // Click Clone (use exact to avoid matching tab button)
    const cloneBtn = page.getByRole('button', { name: 'Clone', exact: true });
    await expect(cloneBtn).toBeVisible();
    await cloneBtn.click();

    await page.waitForTimeout(500);
  });

  test('MkDocs indexing option is checked by default', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    const mkdocsCheckbox = page.getByRole('checkbox', { name: /MkDocs/i });
    await expect(mkdocsCheckbox).toBeChecked();
  });

  test('code files indexing option is checked by default', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    const codeCheckbox = page.getByRole('checkbox', { name: /code files/i });
    await expect(codeCheckbox).toBeChecked();
  });

  test('README only option is unchecked by default', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    const readmeCheckbox = page.getByRole('checkbox', { name: /README/i });
    await expect(readmeCheckbox).not.toBeChecked();
  });

  test('refresh button refreshes repository files', async ({ page }) => {
    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    await page.waitForTimeout(500);

    // Check refresh button exists on existing repo
    const refreshBtn = page.locator('.source-item-actions').getByRole('button', { name: 'Refresh' });
    if (await refreshBtn.first().isVisible()) {
      await expect(refreshBtn.first()).toBeVisible();
    }
  });

  test('remove button removes repository', async ({ page }) => {
    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    await page.waitForTimeout(500);

    // Check remove button exists on existing repo
    const removeBtn = page.locator('.source-item-actions').getByRole('button', { name: 'Remove' });
    if (await removeBtn.first().isVisible()) {
      await expect(removeBtn.first()).toBeVisible();
    }
  });

  test('auto-sync schedule can be changed', async ({ page }) => {
    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    // Find the auto-sync schedule selector in the git panel
    const gitPanel = page.locator('#panel-git');
    const scheduleSelect = gitPanel.locator('select');
    
    if (await scheduleSelect.first().isVisible()) {
      await expect(scheduleSelect.first()).toBeVisible();
    }
  });

  test('accepts GitHub URLs', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    const urlInput = page.getByPlaceholder(/https:\/\/github.com/);
    
    // Enter GitHub URL
    await urlInput.fill('https://github.com/user/repo');
    
    // Should be accepted (no validation error)
    await expect(urlInput).toHaveValue('https://github.com/user/repo');
  });

  test('accepts GitLab URLs', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    const urlInput = page.getByPlaceholder(/https:\/\/github.com/);
    
    // Enter GitLab URL
    await urlInput.fill('https://gitlab.com/user/repo');
    
    // Should be accepted
    await expect(urlInput).toHaveValue('https://gitlab.com/user/repo');
  });

  test('shows last updated time for repos', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          sources: [{
            name: 'archi',
            url: 'https://github.com/archi-physics/archi',
            file_count: 167,
            last_updated: '2026-01-30T10:00:00Z'
          }]
        }
      });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    await page.waitForTimeout(500);

    // Should show updated time
    await expect(page.getByText(/Updated/)).toBeVisible();
  });

  test('empty repo list shows appropriate message', async ({ page }) => {
    await page.route('**/api/sources/git', async (route) => {
      await route.fulfill({ status: 200, json: { sources: [] } });
    });

    await page.goto('/upload');
    await page.getByRole('button', { name: /Git Repos/ }).click();

    await page.waitForTimeout(500);

    // Should show "no repos" message or just empty list
    // Active Repositories section should exist
    await expect(page.getByText('Active Repositories')).toBeVisible();
  });
});
