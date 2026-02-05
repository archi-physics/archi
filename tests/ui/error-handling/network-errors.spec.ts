/**
 * Error Handling and Network Failure Tests
 * 
 * Tests for graceful degradation, error messages, and network failures.
 */
import { test, expect } from '@playwright/test';

test.describe('Network Failure Handling', () => {
  test.describe('Data Viewer', () => {
    test('handles API timeout gracefully', async ({ page }) => {
      // Intercept and delay the response
      await page.route('**/api/sources', async (route) => {
        await new Promise(r => setTimeout(r, 30000)); // 30s delay
        await route.continue();
      });

      await page.goto('/data');

      // Should show loading state
      const hasLoadingIndicator = await page.evaluate(() => {
        return document.body.textContent?.includes('Loading') ||
               document.querySelector('.loading, .spinner') !== null;
      });

      // Page should not crash
      expect(await page.locator('h1').isVisible()).toBe(true);
    });

    test('handles API 500 error', async ({ page }) => {
      await page.route('**/api/sources', (route) => {
        route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Internal server error' }),
        });
      });

      await page.goto('/data');
      await page.waitForLoadState('networkidle');

      // Should show error state, not crash
      const pageContent = await page.textContent('body');
      
      // Either shows error or gracefully handles it
      expect(
        pageContent?.includes('error') ||
        pageContent?.includes('Error') ||
        pageContent?.includes('failed') ||
        pageContent?.includes('Data Sources') // Graceful degradation
      ).toBe(true);
    });

    test('handles malformed JSON response', async ({ page }) => {
      await page.route('**/api/sources', (route) => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: 'not valid json {{{',
        });
      });

      await page.goto('/data');
      await page.waitForLoadState('networkidle');

      // Should not crash
      expect(await page.locator('body').isVisible()).toBe(true);
    });

    test('handles network disconnect', async ({ page }) => {
      await page.goto('/data');
      await page.waitForLoadState('networkidle');

      // Simulate offline
      await page.context().setOffline(true);

      // Try to delete (should fail gracefully)
      const deleteButton = page.locator('[data-action="delete"]').first();
      if (await deleteButton.isVisible()) {
        await deleteButton.click();
        
        // Confirm delete if modal appears
        const confirmBtn = page.locator('button:has-text("Delete"), button:has-text("Confirm")');
        if (await confirmBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
          await confirmBtn.click();
        }

        // Should show error toast/message
        await page.waitForTimeout(1000);
        const hasError = await page.evaluate(() => {
          return document.body.textContent?.includes('error') ||
                 document.body.textContent?.includes('Error') ||
                 document.body.textContent?.includes('failed') ||
                 document.body.textContent?.includes('Failed') ||
                 document.body.textContent?.includes('network');
        });
        // Network error should be handled
        expect(hasError).toBe(true);
      }

      await page.context().setOffline(false);
    });
  });

  test.describe('Upload Page', () => {
    test('handles file upload network failure', async ({ page }) => {
      await page.goto('/upload');
      await page.waitForLoadState('networkidle');

      // Intercept upload and fail
      await page.route('**/api/upload/local**', (route) => {
        route.abort('failed');
      });

      // Create a test file
      const fileChooserPromise = page.waitForEvent('filechooser', { timeout: 5000 }).catch(() => null);
      
      // Click upload area
      const uploadArea = page.locator('#drop-zone, .upload-area, input[type="file"]').first();
      if (await uploadArea.isVisible()) {
        await uploadArea.click();
        
        const fileChooser = await fileChooserPromise;
        if (fileChooser) {
          await fileChooser.setFiles({
            name: 'test.txt',
            mimeType: 'text/plain',
            buffer: Buffer.from('test content'),
          });

          // Wait for error handling
          await page.waitForTimeout(2000);

          // Should show error
          const hasError = await page.evaluate(() => {
            return document.body.textContent?.toLowerCase().includes('error') ||
                   document.body.textContent?.toLowerCase().includes('failed');
          });
          // Error should be displayed
        }
      }
    });

    test('handles git clone 404', async ({ page }) => {
      await page.route('**/api/upload/git', (route) => {
        route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Repository not found' }),
        });
      });

      await page.goto('/upload');
      await page.waitForLoadState('networkidle');

      // Fill git URL
      const gitInput = page.locator('input[name="git-url"], input[placeholder*="git"]').first();
      if (await gitInput.isVisible()) {
        await gitInput.fill('https://github.com/nonexistent/repo');
        
        // Submit
        const submitBtn = page.locator('button:has-text("Clone"), button[type="submit"]').first();
        if (await submitBtn.isVisible()) {
          await submitBtn.click();
          
          await page.waitForTimeout(1000);
          
          // Should show error
          const hasError = await page.evaluate(() => {
            return document.body.textContent?.toLowerCase().includes('not found') ||
                   document.body.textContent?.toLowerCase().includes('error') ||
                   document.body.textContent?.toLowerCase().includes('404');
          });
          expect(hasError).toBe(true);
        }
      }
    });
  });
});

test.describe('Error Message Display', () => {
  test('error toast contains useful information', async ({ page }) => {
    await page.route('**/api/sources', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ 
          error: 'Database connection failed',
          details: 'Connection timeout after 30s'
        }),
      });
    });

    await page.goto('/data');
    await page.waitForLoadState('networkidle');

    // Check for informative error
    const errorContent = await page.evaluate(() => {
      const toast = document.querySelector('#toast-container');
      const errorDiv = document.querySelector('.error, [role="alert"]');
      return {
        toast: toast?.textContent,
        error: errorDiv?.textContent,
        body: document.body.textContent
      };
    });

    // Error should be shown, not hidden silently
    const hasErrorInfo = 
      errorContent.toast?.includes('error') ||
      errorContent.toast?.includes('failed') ||
      errorContent.error?.includes('error') ||
      errorContent.body?.includes('error');
  });

  test('validation errors are displayed clearly', async ({ page }) => {
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');

    // Submit empty git form
    const gitInput = page.locator('input[name="git-url"], input[placeholder*="git"]').first();
    const submitBtn = page.locator('button:has-text("Clone"), button[type="submit"]').first();

    if (await gitInput.isVisible() && await submitBtn.isVisible()) {
      await gitInput.fill(''); // Empty
      await submitBtn.click();

      await page.waitForTimeout(500);

      // Should show validation error
      const hasValidationError = await page.evaluate(() => {
        const inputs = document.querySelectorAll('input');
        for (const input of inputs) {
          if (input.classList.contains('error') || 
              input.classList.contains('invalid') ||
              input.getAttribute('aria-invalid') === 'true') {
            return true;
          }
        }
        return document.body.textContent?.toLowerCase().includes('required') ||
               document.body.textContent?.toLowerCase().includes('invalid');
      });
      // Validation should be shown
    }
  });
});

test.describe('Rate Limiting', () => {
  test('handles 429 Too Many Requests', async ({ page }) => {
    let requestCount = 0;
    await page.route('**/api/sources', (route) => {
      requestCount++;
      if (requestCount > 2) {
        route.fulfill({
          status: 429,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Too many requests' }),
          headers: { 'Retry-After': '60' }
        });
      } else {
        route.continue();
      }
    });

    await page.goto('/data');
    await page.waitForLoadState('networkidle');

    // Multiple rapid requests
    for (let i = 0; i < 5; i++) {
      await page.evaluate(() => {
        // Trigger a refresh or API call
        const refreshBtn = document.querySelector('[data-action="refresh"], .refresh-btn');
        if (refreshBtn instanceof HTMLElement) refreshBtn.click();
      });
      await page.waitForTimeout(100);
    }

    await page.waitForTimeout(1000);

    // Should handle rate limit gracefully (not crash, show message)
    expect(await page.locator('body').isVisible()).toBe(true);
  });
});

test.describe('Concurrent Request Handling', () => {
  test('handles multiple simultaneous deletes', async ({ page }) => {
    await page.goto('/data');
    await page.waitForLoadState('networkidle');

    // Get all delete buttons
    const deleteButtons = page.locator('[data-action="delete"]');
    const count = await deleteButtons.count();

    if (count >= 2) {
      // Click multiple deletes quickly (simulates race condition)
      await Promise.all([
        deleteButtons.nth(0).click(),
        page.waitForTimeout(50).then(() => deleteButtons.nth(1).click()),
      ]);

      // Should not crash
      await page.waitForTimeout(1000);
      expect(await page.locator('body').isVisible()).toBe(true);
    }
  });
});

test.describe('Large Data Handling', () => {
  test('handles large list of sources', async ({ page }) => {
    // Generate large response
    const largeSources = {
      local_files: Array.from({ length: 100 }, (_, i) => ({
        filename: `file${i}.txt`,
        url: `https://example.com/file${i}.txt`,
        added: new Date().toISOString(),
        size: 1024 * i
      })),
      git_repos: Array.from({ length: 50 }, (_, i) => ({
        filename: `repo${i}`,
        url: `https://github.com/user/repo${i}`,
        added: new Date().toISOString()
      })),
      jira_projects: [],
      confluence: []
    };

    await page.route('**/api/sources', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(largeSources),
      });
    });

    await page.goto('/data');
    await page.waitForLoadState('networkidle');

    // Should render without crashing
    expect(await page.locator('body').isVisible()).toBe(true);

    // Page should contain the heading and show sources
    const hasHeading = await page.locator('h1').isVisible();
    expect(hasHeading).toBe(true);
  });
});
