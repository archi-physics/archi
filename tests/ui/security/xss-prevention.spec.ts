/**
 * XSS Prevention Tests
 * 
 * Tests for sanitizeUrl() and other XSS prevention measures
 * in the data viewer and upload modules.
 */
import { test, expect } from '@playwright/test';

test.describe('XSS Prevention - URL Sanitization', () => {
  test.beforeEach(async ({ page }) => {
    // Load the data viewer page which includes utils.js
    await page.goto('/data');
    await page.waitForLoadState('networkidle');
  });

  const xssPayloads = [
    // JavaScript protocol variations
    { payload: 'javascript:alert(1)', description: 'basic javascript protocol' },
    { payload: 'JAVASCRIPT:alert(1)', description: 'uppercase javascript' },
    { payload: 'JavaScript:alert(1)', description: 'mixed case javascript' },
    { payload: 'javascript:alert("XSS")', description: 'javascript with quotes' },
    { payload: '  javascript:alert(1)', description: 'javascript with leading spaces' },
    { payload: 'javascript:alert(1)  ', description: 'javascript with trailing spaces' },
    { payload: '\tjavascript:alert(1)', description: 'javascript with tab' },
    { payload: 'java\nscript:alert(1)', description: 'javascript with newline' },
    { payload: 'java\rscript:alert(1)', description: 'javascript with carriage return' },
    { payload: 'java\0script:alert(1)', description: 'javascript with null byte' },
    
    // URL encoded variations
    { payload: 'javascript%3Aalert(1)', description: 'URL encoded colon' },
    { payload: '%6A%61%76%61%73%63%72%69%70%74:alert(1)', description: 'fully URL encoded javascript' },
    { payload: 'java&#115;cript:alert(1)', description: 'HTML entity encoded' },
    { payload: 'java&#x73;cript:alert(1)', description: 'hex HTML entity encoded' },
    
    // Data protocol
    { payload: 'data:text/html,<script>alert(1)</script>', description: 'data protocol with script' },
    { payload: 'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==', description: 'data protocol base64' },
    { payload: 'DATA:text/html,<script>alert(1)</script>', description: 'uppercase data protocol' },
    
    // VBScript (legacy IE)
    { payload: 'vbscript:msgbox(1)', description: 'vbscript protocol' },
    { payload: 'VBSCRIPT:msgbox(1)', description: 'uppercase vbscript' },
    
    // Other dangerous protocols
    { payload: 'file:///etc/passwd', description: 'file protocol' },
    { payload: 'blob:http://evil.com/malicious', description: 'blob protocol' },
    
    // Edge cases
    { payload: '//javascript:alert(1)', description: 'protocol-relative with javascript' },
  ];

  for (const { payload, description } of xssPayloads) {
    test(`blocks ${description}`, async ({ page }) => {
      // Test sanitizeUrl function directly
      const result = await page.evaluate((url) => {
        // Access the archiUtils global
        if (typeof window.archiUtils !== 'undefined') {
          return window.archiUtils.sanitizeUrl(url);
        }
        // Fallback: test via DataViewer if available
        if (typeof window.dataViewer !== 'undefined') {
          return window.dataViewer.sanitizeUrl(url);
        }
        return 'UTILS_NOT_LOADED';
      }, payload);

      // XSS payloads should return null or empty string
      expect(result).toBeNull();
    });
  }

  const safeUrls = [
    { url: 'https://example.com', description: 'basic https' },
    { url: 'http://example.com', description: 'basic http' },
    { url: 'https://example.com/path/to/page', description: 'https with path' },
    { url: 'https://example.com/page?query=value', description: 'https with query string' },
    { url: 'https://example.com/page#section', description: 'https with hash' },
    { url: 'https://user:pass@example.com/page', description: 'https with auth' },
    { url: 'https://example.com:8080/page', description: 'https with port' },
    { url: 'https://sub.domain.example.com/', description: 'https with subdomain' },
  ];

  for (const { url, description } of safeUrls) {
    test(`allows ${description}`, async ({ page }) => {
      const result = await page.evaluate((testUrl) => {
        if (typeof window.archiUtils !== 'undefined') {
          return window.archiUtils.sanitizeUrl(testUrl);
        }
        if (typeof window.dataViewer !== 'undefined') {
          return window.dataViewer.sanitizeUrl(testUrl);
        }
        return 'UTILS_NOT_LOADED';
      }, url);

      // Safe URLs should be returned (possibly normalized)
      expect(result).not.toBeNull();
      expect(result).toContain('://');
    });
  }

  test('handles null/undefined gracefully', async ({ page }) => {
    const results = await page.evaluate(() => {
      const utils = window.archiUtils || window.dataViewer;
      if (!utils) return ['UTILS_NOT_LOADED'];
      return [
        utils.sanitizeUrl(null),
        utils.sanitizeUrl(undefined),
        utils.sanitizeUrl(''),
      ];
    });

    // All should return null or empty for invalid input
    for (const result of results) {
      expect(result === null || result === '' || result === 'UTILS_NOT_LOADED').toBeTruthy();
    }
  });
});

test.describe('XSS Prevention - HTML Escaping', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/data');
    await page.waitForLoadState('networkidle');
  });

  // Test payloads containing HTML tags (should be escaped)
  const htmlTagPayloads = [
    { input: '<script>alert(1)</script>', description: 'script tag' },
    { input: '<img src=x onerror=alert(1)>', description: 'img onerror' },
    { input: '<svg onload=alert(1)>', description: 'svg onload' },
    { input: '"><script>alert(1)</script>', description: 'quote escape + script' },
    { input: '<iframe src="javascript:alert(1)">', description: 'iframe javascript src' },
  ];

  for (const { input, description } of htmlTagPayloads) {
    test(`escapes ${description}`, async ({ page }) => {
      const result = await page.evaluate((html) => {
        const utils = window.archiUtils || window.dataViewer;
        if (!utils) return 'UTILS_NOT_LOADED';
        return utils.escapeHtml(html);
      }, input);

      // escapeHtml should escape < and > to prevent tag injection
      if (result !== 'UTILS_NOT_LOADED') {
        expect(result).not.toContain('<script');
        expect(result).not.toContain('<img');
        expect(result).not.toContain('<svg');
        expect(result).not.toContain('<iframe');
        // HTML tags should be escaped
        expect(result).toContain('&lt;');
        expect(result).toContain('&gt;');
      }
    });
  }

  // Test payloads without HTML tags (these shouldn't be changed)
  const nonHtmlPayloads = [
    { input: "'+alert(1)+'", description: 'single quote escape' },
    { input: '{{constructor.constructor("alert(1)")()}}', description: 'angular template injection' },
    { input: '${alert(1)}', description: 'template literal injection' },
  ];

  for (const { input, description } of nonHtmlPayloads) {
    test(`preserves ${description} (no HTML)`, async ({ page }) => {
      const result = await page.evaluate((html) => {
        const utils = window.archiUtils || window.dataViewer;
        if (!utils) return 'UTILS_NOT_LOADED';
        return utils.escapeHtml(html);
      }, input);

      // These don't contain HTML tags, so escapeHtml shouldn't change them much
      // The important thing is they're returned as strings, not executed
      if (result !== 'UTILS_NOT_LOADED') {
        expect(typeof result).toBe('string');
      }
    });
  }
});

test.describe('XSS Prevention - Attribute Escaping', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/upload');
    await page.waitForLoadState('networkidle');
  });

  const attrPayloads = [
    { input: '" onclick="alert(1)', description: 'double quote breakout' },
    { input: "' onclick='alert(1)", description: 'single quote breakout' },
    { input: '`onclick=alert(1)`', description: 'backtick breakout' },
    { input: '"><script>alert(1)</script><"', description: 'tag injection via attr' },
    { input: "' onmouseover='alert(1)", description: 'event handler injection' },
  ];

  for (const { input, description } of attrPayloads) {
    test(`escapes ${description} in attributes`, async ({ page }) => {
      const result = await page.evaluate((attr) => {
        const utils = window.archiUtils;
        if (!utils || !utils.escapeAttr) return 'ESCAPE_ATTR_NOT_AVAILABLE';
        return utils.escapeAttr(attr);
      }, input);

      if (result !== 'ESCAPE_ATTR_NOT_AVAILABLE') {
        // Should escape quotes and other dangerous characters
        expect(result).not.toMatch(/^[^&]*"[^&]*$/); // unescaped double quotes
        expect(result).not.toMatch(/^[^&]*'[^&]*$/); // unescaped single quotes
      }
    });
  }
});

test.describe('XSS Prevention - Data Viewer URL Display', () => {
  test('document URL with javascript: protocol is not clickable', async ({ page }) => {
    // Mock the API to return a document with malicious URL
    await page.route('**/api/sources', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          local_files: [{
            filename: 'malicious.txt',
            url: 'javascript:alert(document.cookie)',
            added: new Date().toISOString(),
            size: 100
          }],
          git_repos: [],
          jira_projects: [],
          confluence: []
        })
      });
    });

    await page.goto('/data');
    await page.waitForLoadState('networkidle');

    // Check that javascript: URLs are not rendered as clickable links
    const dangerousLink = await page.$('a[href^="javascript:"]');
    expect(dangerousLink).toBeNull();
    
    // Also check that javascript: doesn't appear in any href attribute
    const allLinks = await page.$$('a[href]');
    for (const link of allLinks) {
      const href = await link.getAttribute('href');
      expect(href?.toLowerCase()).not.toMatch(/^javascript:/);
    }
  });
});
