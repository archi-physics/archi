/**
 * Toast Notification Module
 * 
 * Provides consistent toast notifications across the application.
 * Supports success, error, warning, and info messages.
 */

class ToastManager {
  constructor(options = {}) {
    this.options = {
      containerId: options.containerId || 'toast-container',
      position: options.position || 'bottom-right',
      defaultDuration: options.defaultDuration || 3000,
      maxToasts: options.maxToasts || 5,
      ...options
    };
    
    this.container = null;
    this.toasts = [];
    this.init();
  }

  /**
   * Initialize the toast container
   */
  init() {
    // Check if container already exists
    this.container = document.getElementById(this.options.containerId);
    
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = this.options.containerId;
      this.container.className = `toast-container toast-${this.options.position}`;
      document.body.appendChild(this.container);
    }
  }

  /**
   * Get icon for toast type
   * @param {string} type - Toast type
   * @returns {string} - Icon HTML
   */
  getIcon(type) {
    const icons = {
      success: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
      </svg>`,
      error: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
      </svg>`,
      warning: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>`,
      info: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>`
    };
    return icons[type] || icons.info;
  }

  /**
   * Show a toast notification
   * @param {string} message - Message to display
   * @param {string} type - Toast type (success, error, warning, info)
   * @param {Object} options - Additional options
   * @returns {HTMLElement} - Toast element
   */
  show(message, type = 'info', options = {}) {
    const duration = options.duration ?? this.options.defaultDuration;
    const dismissable = options.dismissable ?? true;
    
    // Remove oldest toast if at max
    if (this.toasts.length >= this.options.maxToasts) {
      this.dismiss(this.toasts[0]);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'polite');
    
    const escapeHtml = window.archiUtils?.escapeHtml || ((t) => {
      const div = document.createElement('div');
      div.textContent = t;
      return div.innerHTML;
    });
    
    toast.innerHTML = `
      <span class="toast-icon">${this.getIcon(type)}</span>
      <span class="toast-message">${escapeHtml(message)}</span>
      ${dismissable ? `<button class="toast-dismiss" aria-label="Dismiss">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>` : ''}
    `;
    
    // Add dismiss handler
    if (dismissable) {
      toast.querySelector('.toast-dismiss').addEventListener('click', () => {
        this.dismiss(toast);
      });
    }
    
    // Add to container
    this.container.appendChild(toast);
    this.toasts.push(toast);
    
    // Animate in
    requestAnimationFrame(() => {
      toast.classList.add('show');
    });
    
    // Auto-dismiss after duration
    if (duration > 0) {
      setTimeout(() => {
        this.dismiss(toast);
      }, duration);
    }
    
    return toast;
  }

  /**
   * Dismiss a toast
   * @param {HTMLElement} toast - Toast element to dismiss
   */
  dismiss(toast) {
    if (!toast || !toast.parentElement) return;
    
    toast.classList.remove('show');
    toast.classList.add('hide');
    
    // Remove from tracking array
    const index = this.toasts.indexOf(toast);
    if (index > -1) {
      this.toasts.splice(index, 1);
    }
    
    // Remove from DOM after animation
    setTimeout(() => {
      if (toast.parentElement) {
        toast.remove();
      }
    }, 200);
  }

  /**
   * Dismiss all toasts
   */
  dismissAll() {
    [...this.toasts].forEach(toast => this.dismiss(toast));
  }

  // Convenience methods
  success(message, options = {}) {
    return this.show(message, 'success', options);
  }

  error(message, options = {}) {
    return this.show(message, 'error', { duration: 5000, ...options });
  }

  warning(message, options = {}) {
    return this.show(message, 'warning', { duration: 4000, ...options });
  }

  info(message, options = {}) {
    return this.show(message, 'info', options);
  }
}

// Create singleton instance
const toast = new ToastManager();

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ToastManager, toast };
}
if (typeof window !== 'undefined') {
  window.ToastManager = ToastManager;
  window.toast = toast;
}
