/**
 * Shared Utilities Module
 * 
 * Common utility functions used across the application.
 * Provides consistent formatting, escaping, and validation.
 */

const archiUtils = {
  /**
   * Escape HTML special characters to prevent XSS
   * @param {string} text - The text to escape
   * @returns {string} - Escaped HTML string
   */
  escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  /**
   * Escape a string for use in HTML attributes
   * Handles both HTML entities and quotes
   * @param {string} text - The text to escape
   * @returns {string} - Escaped attribute string
   */
  escapeAttr(text) {
    if (!text) return '';
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  },

  /**
   * Format file size in human-readable format
   * @param {number} bytes - Size in bytes
   * @returns {string} - Formatted size string
   */
  formatSize(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (bytes >= 1024 && i < units.length - 1) {
      bytes /= 1024;
      i++;
    }
    return `${bytes.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
  },

  /**
   * Format a date as relative time (e.g., "5 min ago")
   * @param {string|Date} dateInput - Date string or Date object
   * @returns {string} - Relative time string
   */
  formatRelativeTime(dateInput) {
    if (!dateInput) return 'Unknown';
    
    const date = dateInput instanceof Date ? dateInput : new Date(dateInput);
    if (isNaN(date.getTime())) return 'Invalid date';
    
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 0) return 'Just now'; // Future date
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  },

  /**
   * Validate URL format and optionally check scheme
   * @param {string} urlString - URL to validate
   * @param {Object} options - Validation options
   * @param {boolean} options.requireHttps - Require https:// scheme
   * @param {string[]} options.allowedSchemes - List of allowed schemes
   * @returns {boolean} - True if valid
   */
  isValidUrl(urlString, options = {}) {
    if (!urlString) return false;
    
    try {
      const url = new URL(urlString);
      
      // Check allowed schemes
      const allowedSchemes = options.allowedSchemes || ['http:', 'https:'];
      if (!allowedSchemes.includes(url.protocol)) {
        return false;
      }
      
      // Check https requirement
      if (options.requireHttps && url.protocol !== 'https:') {
        return false;
      }
      
      return true;
    } catch (_) {
      return false;
    }
  },

  /**
   * Sanitize a URL for safe use in href attributes
   * Returns null if URL is not safe
   * @param {string} urlString - URL to sanitize
   * @returns {string|null} - Sanitized URL or null if unsafe
   */
  sanitizeUrl(urlString) {
    if (!urlString) return null;
    
    try {
      const url = new URL(urlString);
      // Only allow http and https
      if (url.protocol === 'http:' || url.protocol === 'https:') {
        return url.href;
      }
      return null;
    } catch (_) {
      return null;
    }
  },

  /**
   * Debounce a function
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in milliseconds
   * @returns {Function} - Debounced function
   */
  debounce(func, wait = 300) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  /**
   * Get file extension from filename
   * @param {string} filename - Filename or path
   * @returns {string} - Lowercase extension without dot
   */
  getExtension(filename) {
    if (!filename) return '';
    const parts = filename.split('.');
    if (parts.length < 2) return '';
    return parts.pop().toLowerCase();
  },

  /**
   * Get file icon emoji based on extension
   * @param {string} filename - Filename or path
   * @returns {string} - Emoji icon
   */
  getFileIcon(filename) {
    const ext = this.getExtension(filename);
    const iconMap = {
      pdf: '📄',
      md: '📝',
      txt: '📄',
      docx: '📄',
      doc: '📄',
      html: '🌐',
      htm: '🌐',
      json: '{ }',
      yaml: '⚙️',
      yml: '⚙️',
      py: '🐍',
      js: '📜',
      ts: '📘',
      jsx: '⚛️',
      tsx: '⚛️',
      java: '☕',
      go: '🔵',
      rs: '🦀',
      rb: '💎',
      php: '🐘',
      c: '🔧',
      cpp: '🔧',
      h: '🔧',
      sh: '🖥️',
      bash: '🖥️',
      sql: '🗃️',
      css: '🎨',
      scss: '🎨',
    };
    return iconMap[ext] || '📄';
  },

  /**
   * Generate a unique ID
   * @param {string} prefix - Optional prefix
   * @returns {string} - Unique ID
   */
  generateId(prefix = 'id') {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  },

  /**
   * Deep clone an object
   * @param {Object} obj - Object to clone
   * @returns {Object} - Cloned object
   */
  deepClone(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    return JSON.parse(JSON.stringify(obj));
  },
};

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = archiUtils;
}
if (typeof window !== 'undefined') {
  window.archiUtils = archiUtils;
}
