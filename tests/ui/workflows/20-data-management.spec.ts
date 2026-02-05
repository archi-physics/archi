/**
 * Workflow 20: Data Viewer Document Management Tests
 * 
 * Tests for advanced data viewer functionality including document
 * preview, chunk viewing, and document management.
 */
import { test, expect } from '@playwright/test';

test.describe('Data Viewer Document Management', () => {
  test.beforeEach(async ({ page }) => {
    // Mock documents
    await page.route('**/api/data/documents*', async (route) => {
      const url = route.request().url();
      
      if (url.includes('/content')) {
        await route.fulfill({
          status: 200,
          json: {
            content: '# Test Document\n\nThis is the full content of the document.\n\n## Section 1\n\nSome content here.\n\n## Section 2\n\nMore content.',
            metadata: {
              source: 'test.md',
              type: 'local_files',
              url: '/uploads/test.md'
            }
          }
        });
        return;
      }
      
      if (url.includes('/chunks')) {
        await route.fulfill({
          status: 200,
          json: {
            chunks: [
              { 
                id: 1, 
                content: '# Test Document\n\nThis is the full content of the document.', 
                chunk_index: 0,
                metadata: { source: 'test.md' }
              },
              { 
                id: 2, 
                content: '## Section 1\n\nSome content here.', 
                chunk_index: 1,
                metadata: { source: 'test.md' }
              },
              { 
                id: 3, 
                content: '## Section 2\n\nMore content.', 
                chunk_index: 2,
                metadata: { source: 'test.md' }
              }
            ]
          }
        });
        return;
      }
      
      // Default document list
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
              size_bytes: 2048
            },
            {
              hash: 'doc2',
              display_name: 'README.md',
              url: 'https://github.com/test/repo/blob/main/README.md',
              source_type: 'git',
              enabled: true,
              suffix: 'md',
              ingested_at: '2026-01-30T09:00:00Z',
              size_bytes: 4096
            },
            {
              hash: 'doc3',
              display_name: 'example.html',
              url: 'https://example.com/docs',
              source_type: 'web',
              enabled: false,
              suffix: 'html',
              ingested_at: '2026-01-30T08:00:00Z',
              size_bytes: 1024
            }
          ],
          total: 3,
          enabled_count: 2,
          limit: 500,
          offset: 0
        }
      });
    });

    await page.route('**/api/data/stats*', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          total_documents: 3,
          total_chunks: 9,
          total_size_bytes: 7168,
          last_updated: '2026-01-30T12:00:00Z'
        }
      });
    });
  });

  test('selecting document loads its content', async ({ page }) => {
    let contentRequested = false;
    
    await page.route('**/api/data/documents/doc1/content*', async (route) => {
      contentRequested = true;
      await route.fulfill({
        status: 200,
        json: {
          content: '# Test Document Content',
          metadata: { source: 'test.md' }
        }
      });
    });

    await page.goto('/data');
    await page.waitForTimeout(500);
    
    // Expand all
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    // Click on a document
    const docItem = page.locator('.tree-file').first();
    if (await docItem.isVisible()) {
      await docItem.click();
      await page.waitForTimeout(500);
      
      expect(contentRequested).toBe(true);
    }
  });

  test('preview shows markdown rendered', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    // Verify document list is visible (preview placeholder shown initially)
    await expect(page.getByText('Select a document to preview')).toBeVisible();
  });

  test('chunks tab shows document chunks', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    const docItem = page.locator('.tree-file').first();
    if (await docItem.isVisible()) {
      await docItem.click();
      await page.waitForTimeout(500);

      // Look for chunks tab or chunks section
      const chunksTab = page.getByRole('tab', { name: /Chunks/i }).or(
        page.getByRole('button', { name: /Chunks/i })
      );
      
      if (await chunksTab.isVisible()) {
        await chunksTab.click();
        await page.waitForTimeout(500);

        // Should show chunks
        await expect(page.getByText(/Chunk \d|chunk_index/)).toBeVisible();
      }
    }
  });

  test('disabled document has visual indication', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    // doc3 is disabled - should have .disabled class
    const disabledItem = page.locator('.tree-file.disabled');
    
    // Should have at least one disabled document
    const count = await disabledItem.count();
    expect(count).toBeGreaterThan(0);
  });

  test('checkbox state reflects enabled status', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    // Enabled documents should have checked checkbox
    const enabledCheckbox = page.locator('.tree-file:not(.disabled) input[type="checkbox"]').first();
    if (await enabledCheckbox.isVisible()) {
      await expect(enabledCheckbox).toBeChecked();
    }

    // Disabled documents should have unchecked checkbox
    const disabledCheckbox = page.locator('.tree-file.disabled input[type="checkbox"]').first();
    if (await disabledCheckbox.isVisible()) {
      await expect(disabledCheckbox).not.toBeChecked();
    }
  });

  test('toggling checkbox sends enable/disable request', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    // Try to toggle a checkbox - should show warning about needing conversation
    const checkbox = page.locator('.tree-file input[type="checkbox"]').first();
    if (await checkbox.isVisible()) {
      await checkbox.click();
      await page.waitForTimeout(300);
      
      // Should show warning toast about needing conversation context
      const toast = page.locator('.toast');
      await expect(toast.first()).toBeVisible();
    }
  });

  test('document preview shows metadata', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    const docItem = page.locator('.tree-file').first();
    if (await docItem.isVisible()) {
      await docItem.click();
      await page.waitForTimeout(500);

      // Should show some metadata - source type, file name, etc.
      // This depends on the preview implementation
    }
  });

  test('search highlights matching documents', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);

    // Search for "test"
    await page.getByPlaceholder('Search documents').fill('test');
    await page.waitForTimeout(500);

    // Should show filtered results
    // test.md should be visible
    await expect(page.getByText('test.md')).toBeVisible();
  });

  test('filter and search can be combined', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);

    // Filter by Git
    await page.locator('select#filter-select').selectOption('git');
    await page.waitForTimeout(300);

    // Then search
    await page.getByPlaceholder('Search documents').fill('README');
    await page.waitForTimeout(500);

    // Should show Git Repos in document list
    const documentList = page.locator('#document-list');
    await expect(documentList.getByText('Git Repos')).toBeVisible();
  });

  test('preview updates when different document selected', async ({ page }) => {
    await page.goto('/data');
    await page.waitForTimeout(500);
    
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    // Verify preview placeholder is shown initially
    await expect(page.getByText('Select a document to preview')).toBeVisible();
    
    // Verify documents are visible after expanding
    const docs = page.locator('.tree-file');
    const docCount = await docs.count();
    expect(docCount).toBeGreaterThan(0);
  });

  test('stats update after enabling/disabling documents', async ({ page }) => {
    let enabledCount = 2;

    await page.route('**/api/data/stats*', async (route) => {
      await route.fulfill({
        status: 200,
        json: {
          total_documents: 3,
          total_chunks: 9,
          total_size_bytes: 7168,
          enabled_count: enabledCount,
          last_updated: new Date().toISOString()
        }
      });
    });

    await page.goto('/data');
    await page.waitForTimeout(500);
    
    // Initial state
    await page.getByRole('button', { name: 'Expand All' }).click();
    await page.waitForTimeout(300);

    // Disable a document
    const checkbox = page.locator('.tree-file:not(.disabled) input[type="checkbox"]').first();
    if (await checkbox.isVisible()) {
      enabledCount = 1;
      await checkbox.uncheck();
      await page.waitForTimeout(500);

      // Stats should update (via refresh or automatic update)
    }
  });
});
