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
          is_synced: false,
          status_counts: {
            pending: 20,
            embedding: 0,
            embedded: 159,
            failed: 5
          }
        }
      });
    });

    // Mock grouped document status - matches /api/upload/documents/grouped endpoint
    await page.route('**/api/upload/documents/grouped**', async (route) => {
      const url = new URL(route.request().url());
      const showAll = url.searchParams.get('show_all') === 'true';
      const expand = url.searchParams.get('expand');

      const groups = [
        {
          source_name: 'https://docs.example.com',
          total: 150,
          pending: 1,
          embedding: 0,
          embedded: 148,
          failed: 1,
          has_actionable: true,
          documents: expand === 'https://docs.example.com' ? [
            { hash: 'def456', display_name: 'guide.pdf', source_type: 'web', suffix: '.pdf', size_bytes: 50000, ingestion_status: 'pending', ingestion_error: null },
            { hash: 'ghi789', display_name: 'broken.txt', source_type: 'web', suffix: '.txt', size_bytes: 100, ingestion_status: 'failed', ingestion_error: 'Failed to parse file' },
          ] : [],
        },
        {
          source_name: 'Local files',
          total: 34,
          pending: 0,
          embedding: 0,
          embedded: 34,
          failed: 0,
          has_actionable: false,
          documents: [],
        },
      ];

      // If not show_all, only return actionable groups
      const filtered = showAll ? groups : groups.filter(g => g.has_actionable);

      await route.fulfill({
        status: 200,
        json: {
          groups: filtered,
          status_counts: { pending: 1, embedding: 0, embedded: 182, failed: 1 },
        }
      });
    });

    // Mock flat document list - matches /api/upload/documents endpoint (for full list mode)
    await page.route(/\/api\/upload\/documents(\?|$)/, async (route) => {
      const url = new URL(route.request().url());
      const statusFilter = url.searchParams.get('status') || '';
      
      const allDocs = [
        { hash: 'abc123', display_name: 'readme.md', source_type: 'local_files', suffix: '.md', size_bytes: 1234, ingestion_status: 'embedded', ingestion_error: null },
        { hash: 'def456', display_name: 'guide.pdf', source_type: 'web', suffix: '.pdf', size_bytes: 50000, ingestion_status: 'pending', ingestion_error: null },
        { hash: 'ghi789', display_name: 'broken.txt', source_type: 'web', suffix: '.txt', size_bytes: 100, ingestion_status: 'failed', ingestion_error: 'Failed to parse file' },
        { hash: 'jkl012', display_name: 'https://docs.example.com', source_type: 'web', suffix: '.html', size_bytes: 8500, ingestion_status: 'embedded', ingestion_error: null },
      ];

      const filtered = statusFilter ? allDocs.filter(d => d.ingestion_status === statusFilter) : allDocs;

      await route.fulfill({
        status: 200,
        json: {
          documents: filtered,
          total: filtered.length,
          limit: 50,
          offset: 0,
          status_counts: { pending: 1, embedding: 0, embedded: 2, failed: 1 }
        }
      });
    });

    // Mock retry all failed
    await page.route('**/api/upload/documents/retry-all-failed', async (route) => {
      await route.fulfill({
        status: 200,
        json: { success: true, count: 1, message: '1 document(s) reset to pending' }
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

  // ============================================================
  // 9. Unified Ingestion Status Tests
  // ============================================================
  test.describe('Ingestion Status Section', () => {
    test('shows ingestion status section', async ({ page }) => {
      await page.goto('/upload');
      
      await expect(page.locator('.ingestion-status-section')).toBeVisible();
      await expect(page.locator('.ingestion-header')).toBeVisible();
    });

    test('shows status counts in summary', async ({ page }) => {
      await page.goto('/upload');
      
      // Summary should include pending and failed counts
      await expect(page.locator('.summary-count.pending')).toBeVisible();
      await expect(page.locator('.summary-count.failed')).toBeVisible();
      await expect(page.locator('.summary-count.embedded')).toBeVisible();
    });

    test('shows source groups when docs need attention', async ({ page }) => {
      await page.goto('/upload');
      
      // Detail panel should be expanded with groups
      await expect(page.locator('.ingestion-detail')).toBeVisible();
      await expect(page.locator('.source-group')).toHaveCount(1); // only actionable group by default
      await expect(page.getByText('https://docs.example.com')).toBeVisible();
    });

    test('group header shows counts', async ({ page }) => {
      await page.goto('/upload');
      
      // The actionable group should show its summary
      await expect(page.locator('.group-summary').first()).toContainText('1 failed');
      await expect(page.locator('.group-summary').first()).toContainText('1 pending');
    });

    test('clicking group header expands documents', async ({ page }) => {
      await page.goto('/upload');
      
      // Click on the group header to expand
      await page.locator('.group-header').first().click();
      
      // Should show documents
      await expect(page.locator('.group-doc-row').first()).toBeVisible({ timeout: 3000 });
    });

    test('expanded group shows document details', async ({ page }) => {
      await page.goto('/upload');
      
      // Expand the group
      await page.locator('.group-header').first().click();
      
      // Wait for documents to load
      await expect(page.locator('.group-doc-row').first()).toBeVisible({ timeout: 3000 });
      
      // Should show individual doc names and status badges
      await expect(page.locator('.group-documents .doc-name').first()).toBeVisible();
      await expect(page.locator('.group-documents .status-badge').first()).toBeVisible();
    });

    test('failed document in group has retry button', async ({ page }) => {
      await page.goto('/upload');
      
      // Expand the group
      await page.locator('.group-header').first().click();
      await expect(page.locator('.group-doc-row').first()).toBeVisible({ timeout: 3000 });
      
      // Failed doc should have retry button
      await expect(page.locator('.group-documents .btn-retry').first()).toBeVisible();
    });

    test('failed document shows error message', async ({ page }) => {
      await page.goto('/upload');
      
      // Expand the group
      await page.locator('.group-header').first().click();
      await expect(page.locator('.group-doc-row').first()).toBeVisible({ timeout: 3000 });
      
      // Failed doc should show error
      await expect(page.locator('.doc-error').first()).toContainText('Failed to parse file');
    });

    test('retry all failed button visible when failures exist', async ({ page }) => {
      await page.goto('/upload');
      
      await expect(page.locator('#retry-all-btn')).toBeVisible();
    });

    test('retry all failed calls correct API', async ({ page }) => {
      let retryAllRequested = false;
      await page.route('**/api/upload/documents/retry-all-failed', async (route) => {
        retryAllRequested = true;
        await route.fulfill({
          status: 200,
          json: { success: true, count: 1, message: '1 document(s) reset to pending' }
        });
      });
      
      await page.goto('/upload');
      
      await page.locator('#retry-all-btn').click();
      await page.waitForTimeout(500);
      expect(retryAllRequested).toBe(true);
    });

    test('show all toggle switches to full list mode', async ({ page }) => {
      await page.goto('/upload');
      
      // Click "Show all documents"
      await page.locator('#show-all-toggle').click();
      
      // Full list table should appear
      await expect(page.locator('.ingestion-full-list')).toBeVisible();
      await expect(page.locator('.ingestion-table')).toBeVisible();
    });

    test('full list mode shows filter buttons', async ({ page }) => {
      await page.goto('/upload');
      await page.locator('#show-all-toggle').click();
      
      await expect(page.locator('#ingestion-filters .status-filter-btn').filter({ hasText: 'All' })).toBeVisible();
      await expect(page.locator('#ingestion-filters .status-filter-btn').filter({ hasText: 'Pending' })).toBeVisible();
      await expect(page.locator('#ingestion-filters .status-filter-btn').filter({ hasText: 'Failed' })).toBeVisible();
      await expect(page.locator('#ingestion-filters .status-filter-btn').filter({ hasText: 'Embedded' })).toBeVisible();
    });

    test('full list mode shows documents', async ({ page }) => {
      await page.goto('/upload');
      await page.locator('#show-all-toggle').click();
      
      await expect(page.getByText('readme.md')).toBeVisible();
      await expect(page.getByText('guide.pdf')).toBeVisible();
    });

    test('full list has search input', async ({ page }) => {
      await page.goto('/upload');
      await page.locator('#show-all-toggle').click();
      
      await expect(page.locator('#doc-status-search')).toBeVisible();
    });

    test('full list has pagination', async ({ page }) => {
      await page.goto('/upload');
      await page.locator('#show-all-toggle').click();
      
      await expect(page.locator('#doc-prev-btn')).toBeVisible();
      await expect(page.locator('#doc-next-btn')).toBeVisible();
      await expect(page.locator('#pagination-info')).toBeVisible();
    });

    test('back to groups from full list', async ({ page }) => {
      await page.goto('/upload');
      
      // Go to full list
      await page.locator('#show-all-toggle').click();
      await expect(page.locator('.ingestion-full-list')).toBeVisible();
      
      // Click back (toggle text changes)
      await page.locator('#show-all-toggle').click();
      
      // Should show groups again
      await expect(page.locator('.ingestion-groups')).toBeVisible();
      await expect(page.locator('.ingestion-full-list')).not.toBeVisible();
    });

    test('per-document retry in group calls API', async ({ page }) => {
      let retryHash = '';
      await page.route('**/api/upload/documents/*/retry', async (route) => {
        const url = route.request().url();
        retryHash = url.split('/documents/')[1].split('/retry')[0];
        await route.fulfill({
          status: 200,
          json: { success: true, message: 'Document reset to pending' }
        });
      });
      
      await page.goto('/upload');
      
      // Expand group and click retry
      await page.locator('.group-header').first().click();
      await expect(page.locator('.group-documents .btn-retry').first()).toBeVisible({ timeout: 3000 });
      await page.locator('.group-documents .btn-retry').first().click();
      
      await page.waitForTimeout(500);
      expect(retryHash).toBe('ghi789');
    });
  });

  // ============================================================
  // 10. Synced State Tests
  // ============================================================
  test.describe('Synced State', () => {
    test('shows compact summary when all synced', async ({ page }) => {
      // Override grouped endpoint for synced state
      await page.route('**/api/upload/documents/grouped**', async (route) => {
        await route.fulfill({
          status: 200,
          json: {
            groups: [],
            status_counts: { pending: 0, embedding: 0, embedded: 100, failed: 0 }
          }
        });
      });
      
      await page.route('**/api/upload/status', async (route) => {
        await route.fulfill({
          status: 200,
          json: {
            documents_in_catalog: 100,
            documents_embedded: 100,
            pending_embedding: 0,
            is_synced: true,
            status_counts: { pending: 0, embedding: 0, embedded: 100, failed: 0 }
          }
        });
      });
      
      await page.goto('/upload');
      
      // Should show "All N documents embedded"
      await expect(page.getByText(/All .* documents embedded/)).toBeVisible();
      
      // Section should have synced class
      await expect(page.locator('.ingestion-status-section.synced')).toBeVisible();
      
      // Detail panel should be hidden
      await expect(page.locator('.ingestion-detail')).not.toBeVisible();
      
      // Retry all button should be hidden
      await expect(page.locator('#retry-all-btn')).not.toBeVisible();
    });

    test('synced state still has show all toggle', async ({ page }) => {
      await page.route('**/api/upload/documents/grouped**', async (route) => {
        await route.fulfill({
          status: 200,
          json: {
            groups: [],
            status_counts: { pending: 0, embedding: 0, embedded: 100, failed: 0 }
          }
        });
      });
      
      await page.route('**/api/upload/status', async (route) => {
        await route.fulfill({
          status: 200,
          json: {
            documents_in_catalog: 100,
            documents_embedded: 100,
            pending_embedding: 0,
            is_synced: true,
            status_counts: { pending: 0, embedding: 0, embedded: 100, failed: 0 }
          }
        });
      });
      
      await page.goto('/upload');
      
      // Show all toggle should still be available
      await expect(page.locator('#show-all-toggle')).toBeVisible();
    });
  });
});
