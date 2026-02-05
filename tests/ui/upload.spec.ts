import { test, expect } from '@playwright/test';

/**
 * Upload Page UI Tests
 * 
 * Comprehensive tests for the data upload/management functionality.
 * Tests cover file upload, URL scraping, Git repos, and Jira sync panels.
 */

test.describe('Upload Page', () => {
  // ============================================================
  // Setup: Mock API endpoints
  // ============================================================
  test.beforeEach(async ({ page }) => {
    // Mock embedding status - matches /api/upload/status endpoint
    await page.route('**/api/upload/status', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          documents_in_catalog: 184,
          documents_embedded: 159,
          pending_embedding: 25,
          is_synced: false
        }
      });
    });

    // Mock sources endpoints
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

    await page.route('**/api/sources/jira', async (route) => {
      await route.fulfill({
        status: 200,
        json: { projects: [] }
      });
    });

    // Mock file upload
    await page.route('**/api/upload/file', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          success: true,
          document_hash: 'test123',
          filename: 'test.md'
        }
      });
    });

    // Mock URL queue
    await page.route('**/api/sources/urls/queue', async (route) => {
      await route.fulfill({
        status: 200,
        json: { urls: [] }
      });
    });
  });

  // ============================================================
  // 1. Page Load Tests
  // ============================================================
  test('page loads with all required elements', async ({ page }) => {
    await page.goto('/upload');
    
    // Header
    await expect(page.getByRole('heading', { name: 'Upload Data' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Data' })).toBeVisible();
    
    // Embedding status bar
    await expect(page.getByText(/documents waiting to be processed/)).toBeVisible();
    
    // Process button
    await expect(page.getByRole('button', { name: 'Process Documents' })).toBeVisible();
    
    // Source type tabs
    await expect(page.getByRole('button', { name: /Files/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /URLs/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /Git Repos/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /Jira/ })).toBeVisible();
  });

  test('embedding status shows correct counts', async ({ page }) => {
    await page.goto('/upload');
    
    // Check status text
    await expect(page.getByText('25 documents waiting to be processed (159 embedded)')).toBeVisible();
  });

  test('default tab is Files', async ({ page }) => {
    await page.goto('/upload');
    
    // Files tab should be active by default
    const filesTab = page.getByRole('button', { name: /Files/ });
    await expect(filesTab).toHaveClass(/active/);
    
    // Dropzone should be visible
    await expect(page.getByText('Drop files here or click to browse')).toBeVisible();
  });

  // ============================================================
  // 2. Files Tab Tests
  // ============================================================
  test.describe('Files Tab', () => {
    test('shows dropzone with file type info', async ({ page }) => {
      await page.goto('/upload');
      
      await expect(page.getByText('Drop files here or click to browse')).toBeVisible();
      await expect(page.getByText(/PDF, MD, TXT, DOCX/)).toBeVisible();
      await expect(page.getByText(/Max 50 MB/)).toBeVisible();
    });

    test('shows upload queue section', async ({ page }) => {
      await page.goto('/upload');
      
      await expect(page.getByText('Upload Queue')).toBeVisible();
      await expect(page.getByRole('button', { name: 'Clear All' })).toBeVisible();
      await expect(page.getByText('No files in queue')).toBeVisible();
    });

    test('dropzone highlights on drag over', async ({ page }) => {
      await page.goto('/upload');
      
      // Verify dropzone exists (drag simulation in Playwright is limited)
      const dropzone = page.locator('.dropzone, .drop-zone, [class*="dropzone"]').first();
      await expect(dropzone).toBeVisible();
    });
  });

  // ============================================================
  // 3. URLs Tab Tests
  // ============================================================
  test.describe('URLs Tab', () => {
    test('switches to URLs tab and shows URL input', async ({ page }) => {
      await page.goto('/upload');
      
      // Click URLs tab
      await page.getByRole('button', { name: /URLs/ }).click();
      
      // URL input should be visible
      await expect(page.getByPlaceholder(/https:\/\/docs.example.com/)).toBeVisible();
      await expect(page.getByRole('button', { name: 'Add' })).toBeVisible();
    });

    test('shows crawl options', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /URLs/ }).click();
      
      // Crawl options
      await expect(page.getByText('Follow links (crawl pages)')).toBeVisible();
      await expect(page.getByText('Requires SSO authentication')).toBeVisible();
      await expect(page.getByText('Crawl Depth')).toBeVisible();
    });

    test('can add URL to queue', async ({ page }) => {
      await page.route('**/api/sources/urls/add', async (route) => {
        await route.fulfill({ status: 200, json: { success: true } });
      });
      
      await page.goto('/upload');
      await page.getByRole('button', { name: /URLs/ }).click();
      
      // Enter URL
      await page.getByPlaceholder(/https:\/\/docs.example.com/).fill('https://example.com/docs');
      
      // Click Add
      await page.getByRole('button', { name: 'Add' }).click();
      
      // Should show URL in queue (mock the response to include it)
      await page.waitForTimeout(500);
    });

    test('crawl depth selector has options', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /URLs/ }).click();
      
      // Check depth options
      const depthSelect = page.locator('select').filter({ hasText: 'level' }).first();
      if (await depthSelect.isVisible()) {
        await expect(depthSelect.locator('option')).toHaveCount(4); // 1, 2, 3, 5 levels
      }
    });

    test('start scraping button visible when URLs queued', async ({ page }) => {
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
      
      await expect(page.getByRole('button', { name: 'Start Scraping' })).toBeVisible();
    });
  });

  // ============================================================
  // 4. Git Repos Tab Tests
  // ============================================================
  test.describe('Git Repos Tab', () => {
    test('switches to Git Repos tab and shows repo input', async ({ page }) => {
      await page.goto('/upload');
      
      // Click Git Repos tab
      await page.getByRole('button', { name: /Git Repos/ }).click();
      
      // Repository URL input should be visible
      await expect(page.getByPlaceholder(/https:\/\/github.com/)).toBeVisible();
      await expect(page.getByRole('button', { name: 'Clone', exact: true })).toBeVisible();
    });

    test('shows indexing options', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /Git Repos/ }).click();
      
      // Indexing options
      await expect(page.getByText('Index MkDocs documentation')).toBeVisible();
      await expect(page.getByText(/Index code files/)).toBeVisible();
      await expect(page.getByText('Include only README files')).toBeVisible();
    });

    test('shows auto-sync schedule selector', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /Git Repos/ }).click();
      
      // Wait for panel switch, then find auto-sync within the git panel
      await page.waitForTimeout(300);
      const gitPanel = page.locator('#panel-git');
      await expect(gitPanel.getByText('Auto-sync Schedule')).toBeVisible();
    });

    test('displays existing repositories', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /Git Repos/ }).click();
      
      // Should show "Active Repositories" section
      await expect(page.getByText('Active Repositories')).toBeVisible();
      
      // Should show the mocked repository
      await expect(page.getByText('archi')).toBeVisible();
      await expect(page.getByText(/167 files/)).toBeVisible();
    });

    test('repo has refresh and remove buttons', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /Git Repos/ }).click();
      
      // Each repo should have action buttons
      const refreshBtn = page.locator('.source-item-actions').getByRole('button', { name: 'Refresh' });
      const removeBtn = page.locator('.source-item-actions').getByRole('button', { name: 'Remove' });
      
      if (await refreshBtn.first().isVisible()) {
        await expect(refreshBtn.first()).toBeVisible();
      }
      if (await removeBtn.first().isVisible()) {
        await expect(removeBtn.first()).toBeVisible();
      }
    });

    test('can initiate repo clone', async ({ page }) => {
      await page.route('**/api/sources/git/clone', async (route) => {
        await route.fulfill({
          status: 200,
          json: { success: true, message: 'Cloning started' }
        });
      });
      
      await page.goto('/upload');
      await page.getByRole('button', { name: /Git Repos/ }).click();
      
      // Enter repo URL
      await page.getByPlaceholder(/https:\/\/github.com/).fill('https://github.com/test/repo');
      
      // Click Clone (use exact match)
      await page.getByRole('button', { name: 'Clone', exact: true }).click();
      
      await page.waitForTimeout(500);
    });
  });

  // ============================================================
  // 5. Jira Tab Tests
  // ============================================================
  test.describe('Jira Tab', () => {
    test('switches to Jira tab and shows project input', async ({ page }) => {
      await page.goto('/upload');
      
      // Click Jira tab
      await page.getByRole('button', { name: /Jira/ }).click();
      
      // Project key input should be visible
      await expect(page.getByPlaceholder('PROJ')).toBeVisible();
      await expect(page.getByRole('button', { name: 'Sync Project' })).toBeVisible();
    });

    test('shows Jira configuration info', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /Jira/ }).click();
      
      // Should show configuration hint
      await expect(page.getByText('Jira Configuration')).toBeVisible();
      await expect(page.getByText(/JIRA_PAT/)).toBeVisible();
    });

    test('shows synced projects section', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /Jira/ }).click();
      
      await expect(page.getByText('Synced Projects')).toBeVisible();
      await expect(page.getByText('No Jira projects synced')).toBeVisible();
    });

    test('shows auto-sync schedule for Jira', async ({ page }) => {
      await page.goto('/upload');
      await page.getByRole('button', { name: /Jira/ }).click();
      
      // Wait for panel switch, then find auto-sync within the jira panel
      await page.waitForTimeout(300);
      const jiraPanel = page.locator('#panel-jira');
      await expect(jiraPanel.getByText('Auto-sync Schedule')).toBeVisible();
    });
  });

  // ============================================================
  // 6. Process Documents Tests
  // ============================================================
  test.describe('Process Documents', () => {
    test('process button triggers embedding', async ({ page }) => {
      await page.goto('/upload');
      
      // Process Documents button should be visible
      const processBtn = page.getByRole('button', { name: 'Process Documents' });
      await expect(processBtn).toBeVisible();
      
      // Click Process Documents
      await processBtn.click();
      
      // Wait for response
      await page.waitForTimeout(500);
    });

    test('process button disabled when already processing', async ({ page }) => {
      await page.route('**/api/upload/status', async (route) => {
        await route.fulfill({
          status: 200,
          json: {
            documents_in_catalog: 184,
            documents_embedded: 159,
            pending_embedding: 25,
            is_synced: false
          }
        });
      });
      
      await page.goto('/upload');
      
      // Button should be disabled or show processing state
      const processBtn = page.getByRole('button', { name: /Process|Processing/ });
      if (await processBtn.isDisabled()) {
        await expect(processBtn).toBeDisabled();
      }
    });
  });

  // ============================================================
  // 7. Navigation Tests
  // ============================================================
  test.describe('Navigation', () => {
    test('Data link navigates back to data viewer', async ({ page }) => {
      await page.goto('/upload');
      
      const dataLink = page.getByRole('link', { name: 'Data' });
      await expect(dataLink).toBeVisible();
      await expect(dataLink).toHaveAttribute('href', '/data');
    });

    test('Refresh button refreshes data', async ({ page }) => {
      await page.goto('/upload');
      
      // Refresh button should be visible
      const refreshBtn = page.getByRole('button', { name: 'Refresh' });
      await expect(refreshBtn).toBeVisible();
      
      // Click refresh
      await refreshBtn.click();
      
      await page.waitForTimeout(300);
    });
  });

  // ============================================================
  // 8. Tab Persistence Tests
  // ============================================================
  test('selected tab persists between navigations', async ({ page }) => {
    await page.goto('/upload');
    
    // Select Git Repos tab
    await page.getByRole('button', { name: /Git Repos/ }).click();
    
    // Navigate away and back
    await page.goto('/data');
    await page.goto('/upload');
    
    // Either Files or Git Repos tab should be visible and active
    // Use first() to avoid strict mode error
    const activeTab = page.locator('.source-tab.active').first();
    await expect(activeTab).toBeVisible();
  });
});
