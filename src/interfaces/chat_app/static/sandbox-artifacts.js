/**
 * Sandbox Artifacts Rendering Module
 * 
 * Provides enhanced rendering for sandbox-generated artifacts beyond what
 * marked.js does by default.  Images get lightbox-style click-to-zoom;
 * other file types get download buttons with file-type icons.
 * 
 * Usage:
 *   After rendering markdown, call `SandboxArtifacts.enhance(containerEl)`
 *   to upgrade any artifact links/images inside that element.
 * 
 * The module detects artifacts by their URL pattern:
 *   /api/sandbox-artifacts/<trace_id>/<filename>
 */

// eslint-disable-next-line no-unused-vars
const SandboxArtifacts = (function () {
  'use strict';

  const ARTIFACT_URL_RE = /^\/api\/sandbox-artifacts\/([0-9a-f-]{36})\/(.+)$/i;

  // File extension -> icon emoji (keep simple; no external icon lib needed).
  const FILE_ICONS = {
    csv: '📊',
    json: '📋',
    txt: '📄',
    md: '📝',
    py: '🐍',
    js: '📜',
    html: '🌐',
    pdf: '📕',
    zip: '📦',
    tar: '📦',
    gz: '📦',
  };

  /**
   * Check if a URL is a sandbox artifact.
   * @param {string} url
   * @returns {boolean}
   */
  function isArtifactUrl(url) {
    return ARTIFACT_URL_RE.test(url);
  }

  /**
   * Extract filename from an artifact URL.
   * @param {string} url
   * @returns {string|null}
   */
  function extractFilename(url) {
    const m = ARTIFACT_URL_RE.exec(url);
    return m ? decodeURIComponent(m[2]) : null;
  }

  /**
   * Get a suitable icon for a filename.
   * @param {string} filename
   * @returns {string}
   */
  function getFileIcon(filename) {
    const ext = (filename.split('.').pop() || '').toLowerCase();
    return FILE_ICONS[ext] || '📎';
  }

  /**
   * Enhance artifact images within a container.
   * 
   * Adds click-to-open-in-new-tab and a subtle border style.
   * @param {HTMLElement} container
   */
  function enhanceImages(container) {
    const imgs = container.querySelectorAll('img');
    imgs.forEach((img) => {
      const src = img.getAttribute('src') || '';
      if (!isArtifactUrl(src)) return;

      // Style for sandbox artifact images
      img.classList.add('sandbox-artifact-image');

      // Wrap in a link if not already wrapped
      if (img.parentElement.tagName !== 'A') {
        const link = document.createElement('a');
        link.href = src;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.title = 'Click to open full size';
        img.parentNode.insertBefore(link, img);
        link.appendChild(img);
      }
    });
  }

  /**
   * Enhance artifact download links.
   * 
   * Converts plain `[📎 file.csv](/api/sandbox-artifacts/...)` links into
   * styled download buttons.
   * @param {HTMLElement} container
   */
  function enhanceLinks(container) {
    const links = container.querySelectorAll('a');
    links.forEach((link) => {
      const href = link.getAttribute('href') || '';
      if (!isArtifactUrl(href)) return;

      // Skip image links (handled by enhanceImages)
      if (link.querySelector('img')) return;

      const filename = extractFilename(href);
      if (!filename) return;

      // Already enhanced?
      if (link.classList.contains('sandbox-artifact-link')) return;

      link.classList.add('sandbox-artifact-link');
      link.setAttribute('download', filename);
      link.title = `Download ${filename}`;

      // Rewrite content to include icon + filename
      const icon = getFileIcon(filename);
      link.innerHTML = `<span class="artifact-icon">${icon}</span> ${escapeHtml(filename)}`;
    });
  }

  /**
   * Simple HTML escape.
   * @param {string} str
   * @returns {string}
   */
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Main entry point.  Call after markdown rendering to enhance artifacts.
   * @param {HTMLElement} container
   */
  function enhance(container) {
    if (!container) return;
    enhanceImages(container);
    enhanceLinks(container);
  }

  /**
   * Inject default styles for artifact images and links.
   * Call once on page load.
   */
  function injectStyles() {
    if (document.getElementById('sandbox-artifact-styles')) return;

    const style = document.createElement('style');
    style.id = 'sandbox-artifact-styles';
    style.textContent = `
      .sandbox-artifact-image {
        max-width: 100%;
        border: 1px solid var(--border-color, #444);
        border-radius: 6px;
        cursor: pointer;
        transition: box-shadow 0.2s;
      }
      .sandbox-artifact-image:hover {
        box-shadow: 0 0 8px rgba(102, 179, 255, 0.4);
      }
      .sandbox-artifact-link {
        display: inline-flex;
        align-items: center;
        gap: 0.4em;
        padding: 0.3em 0.7em;
        background: var(--secondary-bg, #2d2d2d);
        border: 1px solid var(--border-color, #444);
        border-radius: 4px;
        color: var(--primary-text, #e0e0e0);
        text-decoration: none;
        font-size: 0.9em;
        transition: background 0.2s;
      }
      .sandbox-artifact-link:hover {
        background: var(--hover-bg, #3a3a3a);
        text-decoration: none;
      }
      .artifact-icon {
        font-size: 1.1em;
      }
    `;
    document.head.appendChild(style);
  }

  // Auto-inject styles when module loads
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', injectStyles);
    } else {
      injectStyles();
    }
  }

  return {
    enhance,
    isArtifactUrl,
    extractFilename,
    injectStyles,
  };
})();
