/**
 * Theme Initializer
 * 
 * Reads the user's theme preference from localStorage and applies it.
 * Must be loaded early (before page renders) to prevent flash of wrong theme.
 * Shared across all pages: chat, data viewer, upload, database viewer.
 */
(function() {
  const theme = localStorage.getItem('archi_theme') || 'light';
  document.documentElement.setAttribute('data-theme', theme);
})();
