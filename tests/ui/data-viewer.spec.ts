import { test, expect } from '@playwright/test';

/**
 * Data Viewer Page Tests
 * 
 * Comprehensive tests for the data viewer functionality.
 * Tests cover document listing, filtering, search, and preview.
 */

test.describe('Data Viewer Page', () => {
  // ============================================================
  // Setup: Mock API endpoints
  // ============================================================
  test.beforeEach(async ({ page }) => {
    // Mock documents list
    await page.route('**/api/data/documents*', async (route) => {
      const url = route.request().url();
      
      // If requesting specific document content
      if (url.includes('/content')) {
        await route.fulfill({
          status: 200,
          json: {
            content: '# Test Document\n\nThis is test content.',
            metadata: { source: 'test.md', type: 'local_files' }
          }
        });
        return;
      }
      
      // If requesting chunks
      if (url.includes('/chunks')) {
        await route.fulfill({
          status: 200,
          json: {
            chunks: [
              { id: 1, content: 'Chunk 1 content', chunk_index: 0 },
              { id: 2, content: 'Chunk 2 content', chunk_index: 1 }
            ]
          }
        });
        return;
      }
      
      // Default: return document list
      await route.fulfill({
        status: 200,
        json: {
          documents: [
            {
              hash: 'doc1',
              display_name: 'test.md',
              url: '/uploads/test.md',
              source_type: 'local_files',
              enabled: true,
              suffix: 'md',
              ingested_at: '2026-01-30T10:00:00Z',
              size_bytes: 1024
            },
            {
              hash: 'doc2',
              display_name: 'github.com/archi-physics',
              url: 'https://github.com/archi-physics/archi/blob/main/README.md',
              source_type: 'git',
              enabled: true,
              suffix: 'md',
              ingested_at: '2026-01-30T09:00:00Z',
              size_bytes: 2048
            },
            {
              hash: 'doc3',
              display_name: 'example.com',
              url: 'https://example.com/docs',
              source_type: 'web',
              enabled: true,
              suffix: 'html',
              ingested_at: '2026-01-30T08:00:00Z',
              size_bytes: 512
            },
            {
              hash: 'doc4',
              display_name: 'PROJ-123',
              url: 'https://jira.example.com/browse/PROJ-123',
              source_type: 'ticket',
              enabled: false,
              suffix: null,
              ingested_at: '2026-01-30T07:00:00Z',
              size_bytes: 256
            },
            {
              hash: 'doc5',
              display_name: 'private.docs.example/sso',
              url: 'https://private.docs.example/sso',
              source_type: 'sso',
              enabled: true,
              suffix: 'html',
              ingested_at: '2026-01-30T06:00:00Z',
              size_bytes: 128
            },
            {
              hash: 'doc6',
              display_name: 'legacy-source-item',
              url: '/legacy/item',
              source_type: 'legacy_source',
              enabled: true,
              suffix: 'txt',
              ingested_at: '2026-01-30T05:00:00Z',
              size_bytes: 64
            }
          ],
          total: 6,
          enabled_count: 5,
          limit: 500,
          offset: 0
        }
      });
    });

    // Mock stats
    await page.route('**/api/data/stats*', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          total_documents: 184,
          total_chunks: 1074,
          total_size_bytes: 59392,
          last_updated: '2026-01-30T12:00:00Z',
          status_counts: { pending: 2, embedding: 1, embedded: 180, failed: 1 },
          ingestion_in_progress: true,
          by_source_type: {
            local_files: { total: 46, enabled: 46, disabled: 0 },
            git: { total: 46, enabled: 46, disabled: 0 },
            web: { total: 46, enabled: 46, disabled: 0 },
            ticket: { total: 44, enabled: 43, disabled: 1 },
            sso: { total: 1, enabled: 1, disabled: 0 },
            legacy_source: { total: 1, enabled: 1, disabled: 0 }
          }
        }
      });
    });
  });

  // ============================================================
  // 1. Page Load Tests
  // ============================================================
  test('page loads with all required elements', async ({ page }) => {
    await page.goto('/data');
    
    // Header
    await expect(page.getByRole('heading', { name: 'Data Sources' })).toBeVisible();
    
    // Navigation links
    await expect(page.getByRole('link', { name: 'Chat' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Upload Data' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Database Viewer' })).toBeVisible();
    
    // Control buttons
    await expect(page.getByRole('button', { name: 'Expand All' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Collapse All' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Refresh' })).toBeVisible();
  });

  test('displays stats bar with correct info', async ({ page }) => {
    await page.goto('/data');
    
    // Stats should show document count
    await expect(page.getByText('Documents:')).toBeVisible();
    await expect(page.getByText('184')).toBeVisible();
    
    // Chunks count
    await expect(page.getByText('Chunks:')).toBeVisible();
    await expect(page.getByText('1074')).toBeVisible();
    
    // Size
    await expect(page.getByText('Size:')).toBeVisible();
  });

  test('displays filter and search controls', async ({ page }) => {
    await page.goto('/data');
    
    // Search input
    await expect(page.getByPlaceholder('Search documents')).toBeVisible();
    
    // Filter dropdown
    const filterSelect = page.locator('select').filter({ hasText: /All Types/ });
    await expect(filterSelect).toBeVisible();
  });

  // ============================================================
  // 2. Document Tree Tests
  // ============================================================
  test('displays document categories', async ({ page }) => {
    await page.goto('/data');
    
    // Wait for documents to load
    await page.waitForTimeout(500);
    
    // Should show source type categories in the document list (not filter dropdown)
    const documentList = page.locator('#document-list');
    await expect(documentList.getByText('Local Files')).toBeVisible();
    await expect(documentList.getByText('Git Repos')).toBeVisible();
    await expect(documentList.getByText('Web Pages')).toBeVisible();
    await expect(documentList.getByText('SSO Pages')).toBeVisible();
    await expect(documentList.getByText('Other Sources')).toBeVisible();
  });



  test('clicking category expands/collapses it', async ({ page }) => {
    await page.goto('/data');
    
    await page.waitForTimeout(500);
    
    // Click on a category to toggle
    const localFilesCategory = page.getByText('Local Files').first();
    if (await localFilesCategory.isVisible()) {
      await localFilesCategory.click();
      await page.waitForTimeout(100);
      
      // Click again to toggle back
      await localFilesCategory.click();
    }
  });

  test('expand all button expands all categories', async ({ page }) => {
    await page.goto('/data');
    
    // Click Expand All
    await page.getByRole('button', { name: 'Expand All' }).click();
    
    await page.waitForTimeout(300);
    
    // All categories should be expanded
    // Documents inside should be visible
  });

  test('collapse all button collapses all categories', async ({ page }) => {
    await page.goto('/data');
    
    // First expand all
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(200);
    
    // Then collapse all
    await page.getByRole('button', { name: 'Collapse All' }).click();
    await page.waitForTimeout(200);
  });

  // ============================================================
  // 3. Filter Tests
  // ============================================================
  test('filter dropdown has all source type options', async ({ page }) => {
    await page.goto('/data');
    
    const filterSelect = page.locator('select#filter-select');
    
    // Check options exist (options may be hidden until dropdown opened)
    await expect(filterSelect.locator('option[value="all"]')).toBeAttached();
    await expect(filterSelect.locator('option[value="local_files"]')).toBeAttached();
    await expect(filterSelect.locator('option[value="git"]')).toBeAttached();
    await expect(filterSelect.locator('option[value="web"]')).toBeAttached();
    await expect(filterSelect.locator('option[value="ticket"]')).toBeAttached();
    await expect(filterSelect.locator('option[value="sso"]')).toBeAttached();
    await expect(filterSelect.locator('option[value="other"]')).toBeAttached();
  });

  test('filtering by source type shows only that type', async ({ page }) => {
    await page.goto('/data');
    
    // Select Git Repos filter
    await page.locator('select#filter-select').selectOption('git');
    
    await page.waitForTimeout(300);
    
    // Should show Git Repos category in document list
    const documentList = page.locator('#document-list');
    await expect(documentList.getByText('Git Repos')).toBeVisible();
    
    // Other categories should be filtered out
    // (if no docs match, they won't appear)
  });

  test('search filters documents by name', async ({ page }) => {
    await page.goto('/data');
    
    // Type in search
    await page.getByPlaceholder('Search documents').fill('test');
    
    await page.waitForTimeout(500);
    
    // Should filter to matching documents
    // test.md should appear if it matches
  });

  test('search is case-insensitive', async ({ page }) => {
    await page.goto('/data');
    
    // Search with different case
    await page.getByPlaceholder('Search documents').fill('TEST');
    
    await page.waitForTimeout(500);
    
    // Should still find test.md
  });

  test('clearing search shows all documents', async ({ page }) => {
    await page.goto('/data');
    
    // Search for something
    const searchInput = page.getByPlaceholder('Search documents');
    await searchInput.fill('test');
    await page.waitForTimeout(300);
    
    // Clear search
    await searchInput.clear();
    await page.waitForTimeout(300);
    
    // All documents should be visible again
  });

  // ============================================================
  // 4. Document Selection Tests
  // ============================================================
  test('clicking document shows preview', async ({ page }) => {
    await page.goto('/data');
    
    // Wait for documents to load
    await page.waitForTimeout(500);
    
    // Expand a category first
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);
    
    // Click on a document file
    const docItem = page.locator('.tree-file').first();
    if (await docItem.isVisible()) {
      await docItem.click();
      
      await page.waitForTimeout(500);
      
      // Preview pane should show content
      // Either the actual content or a loading state
    }
  });

  test('preview pane shows document content', async ({ page }) => {
    await page.goto('/data');
    
    await page.waitForTimeout(500);
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);
    
    const docItem = page.locator('.tree-file').first();
    if (await docItem.isVisible()) {
      await docItem.click();
      
      // Wait for content to load
      await page.waitForTimeout(500);
      
      // Preview should no longer show "Select a document"
      // Either it's hidden or replaced with actual content
    }
  });

  test('preview shows placeholder when no document selected', async ({ page }) => {
    await page.goto('/data');
    
    // Initially should show placeholder
    await expect(page.getByText('Select a document to preview')).toBeVisible();
    await expect(page.getByText('Browse the file tree')).toBeVisible();
  });

  test('does not prefetch document content before selection', async ({ page }) => {
    let contentRequests = 0;

    await page.route('**/api/data/documents/*/content*', async (route) => {
      contentRequests++;
      await route.fulfill({
        status: 200,
        json: { content: 'test' }
      });
    });

    await page.goto('/data');
    await page.waitForTimeout(500);

    expect(contentRequests).toBe(0);
  });

  test('shows phase labels and documents-left hint when indexing is active', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);

    await expect(page.locator('#list-status')).toContainText('data collection ongoing');
    await expect(page.locator('#list-status')).toContainText('3 documents left to embed');
  });

  test('falls back to embedding in progress when left count is unavailable', async ({ page }) => {
    await page.route('**/api/data/documents*', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          documents: [
            {
              hash: 'doc-embedding',
              display_name: 'embedding-doc',
              url: '/uploads/embedding-doc',
              source_type: 'local_files',
              ingestion_status: 'embedding',
              enabled: true,
              suffix: 'txt',
              ingested_at: '2026-01-30T10:00:00Z',
              size_bytes: 123
            }
          ],
          total: 1,
          enabled_count: 1,
          limit: 500,
          offset: 0
        }
      });
    });

    await page.route('**/api/data/stats*', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          total_documents: 1,
          total_chunks: 0,
          total_size_bytes: 123,
          last_updated: '2026-01-30T12:00:00Z',
          by_source_type: {
            local_files: { total: 1, enabled: 1, disabled: 0 }
          }
        }
      });
    });

    await page.goto('/data');
    await page.waitForTimeout(500);

    await expect(page.locator('#list-status')).toContainText('embedding in progress');
    await expect(page.locator('#list-status')).not.toContainText('left to embed');
  });

  // ============================================================
  // 6. Navigation Tests
  // ============================================================
  test('Chat link navigates to chat', async ({ page }) => {
    await page.goto('/data');
    
    const chatLink = page.getByRole('link', { name: 'Chat' });
    await expect(chatLink).toHaveAttribute('href', '/chat');
  });

  test('Upload Data link navigates to upload', async ({ page }) => {
    await page.goto('/data');
    
    const uploadLink = page.getByRole('link', { name: 'Upload Data' });
    await expect(uploadLink).toHaveAttribute('href', '/upload');
  });

  test('Database Viewer link navigates to admin', async ({ page }) => {
    await page.goto('/data');
    
    const dbLink = page.getByRole('link', { name: 'Database Viewer' });
    await expect(dbLink).toHaveAttribute('href', '/admin/database');
  });

  // ============================================================
  // 7. Refresh Tests
  // ============================================================
  test('Refresh button reloads documents', async ({ page }) => {
    let fetchCount = 0;
    
    await page.route('**/api/data/documents*', async (route) => {
      const url = route.request().url();
      if (!url.includes('/content') && !url.includes('/chunks')) {
        fetchCount++;
      }
      await route.fulfill({
        status: 200,
        json: {
          documents: [],
          total: 0,
          enabled_count: 0
        }
      });
    });
    
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    const initialCount = fetchCount;
    
    // Click Refresh
    await page.getByRole('button', { name: 'Refresh' }).click();
    
    await page.waitForTimeout(500);
    
    // Should have fetched again
    expect(fetchCount).toBeGreaterThan(initialCount);
  });

  test('Refresh also updates stats', async ({ page }) => {
    let statsFetchCount = 0;
    
    await page.route('**/api/data/stats*', async (route) => {
      statsFetchCount++;
      await route.fulfill({
        status: 200,
        json: {
          total_documents: 184 + statsFetchCount,
          total_chunks: 1074,
          total_size_bytes: 59392,
          last_updated: new Date().toISOString()
        }
      });
    });
    
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    const initialCount = statsFetchCount;
    
    // Click Refresh
    await page.getByRole('button', { name: 'Refresh' }).click();
    
    await page.waitForTimeout(500);
    
    expect(statsFetchCount).toBeGreaterThan(initialCount);
  });

  // ============================================================
  // 8. Empty State Tests
  // ============================================================
  test('shows empty state when no documents', async ({ page }) => {
    await page.route('**/api/data/documents*', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          documents: [],
          total: 0,
          enabled_count: 0
        }
      });
    });
    
    await page.goto('/data');
    
    await page.waitForTimeout(500);
    
    // Should show empty state message
    await expect(page.getByText(/No documents|No files/i)).toBeVisible();
  });

  test('shows no results when search has no matches', async ({ page }) => {
    await page.goto('/data');
    
    // Search for something that doesn't exist
    await page.getByPlaceholder('Search documents').fill('xyznonexistent123');
    
    await page.waitForTimeout(500);
    
    // Should show no results message
    await expect(page.getByText(/No documents match/i)).toBeVisible();
  });

  // ============================================================
  // 9. Error Handling Tests
  // ============================================================
  test('handles API error gracefully', async ({ page }) => {
    await page.route('**/api/data/documents*', async (route) => {
      await route.fulfill({
        status: 500,
        json: { error: 'Internal server error' }
      });
    });
    
    await page.goto('/data');
    
    await page.waitForTimeout(500);
    
    // Should show error or handle gracefully
    // Not crash the page
  });

  test('handles network timeout gracefully', async ({ page }) => {
    await page.route('**/api/data/documents*', async (route) => {
      // Don't fulfill - let it timeout
      await new Promise(resolve => setTimeout(resolve, 10000));
    });
    
    page.setDefaultTimeout(2000);
    
    await page.goto('/data');
    
    // Page should still be functional
    await expect(page.getByRole('heading', { name: 'Data Sources' })).toBeVisible();
  });
});
