/**
 * DataViewer - Main Controller
 * 
 * Manages the data viewer UI, coordinating between FileTree and ContentRenderer modules.
 * Handles document loading, selection, filtering, and API communication.
 * 
 * Dependencies: utils.js, toast.js, file-tree.js, content-renderer.js
 */

class DataViewer {
  constructor() {
    this.documents = [];
    this.selectedDocument = null;
    this.selectedChunks = [];
    this.selectedContent = '';
    this.conversationId = null;
    this.searchQuery = '';
    this.filterType = 'all';
    this.showChunks = false; // Toggle for chunk view
    
    // Initialize modules
    this.fileTree = new FileTree({
      onSelect: (hash) => this.selectDocument(hash),
      onToggle: (pathOrHash, enabled) => {
        if (typeof enabled === 'boolean') {
          this.toggleDocument(pathOrHash, enabled);
        } else {
          this.renderDocuments();
        }
      }
    });
    
    this.contentRenderer = contentRenderer;
    
    // Get conversation ID from URL or session
    const urlParams = new URLSearchParams(window.location.search);
    this.conversationId = urlParams.get('conversation_id');
    
    this.init();
  }

  async init() {
    this.bindEvents();
    await Promise.all([
      this.loadDocuments(),
      this.loadStats()
    ]);
  }

  /**
   * Bind DOM event handlers
   */
  bindEvents() {
    // Search input
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.searchQuery = e.target.value;
        this.renderDocuments();
      });
    }
    
    // Filter select
    const filterSelect = document.getElementById('filter-select');
    if (filterSelect) {
      filterSelect.addEventListener('change', (e) => {
        this.filterType = e.target.value;
        this.renderDocuments();
      });
    }
    
    // Bulk actions
    const enableAllBtn = document.getElementById('enable-all-btn');
    if (enableAllBtn) {
      enableAllBtn.addEventListener('click', () => this.bulkEnable());
    }
    
    const disableAllBtn = document.getElementById('disable-all-btn');
    if (disableAllBtn) {
      disableAllBtn.addEventListener('click', () => this.bulkDisable());
    }
    
    // Refresh
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.refresh());
    }
    
    // Expand/collapse all
    const expandAllBtn = document.getElementById('expand-all-btn');
    if (expandAllBtn) {
      expandAllBtn.addEventListener('click', () => this.expandAll());
    }
    
    const collapseAllBtn = document.getElementById('collapse-all-btn');
    if (collapseAllBtn) {
      collapseAllBtn.addEventListener('click', () => this.collapseAll());
    }
  }

  /**
   * Load documents from API
   */
  async loadDocuments() {
    const listEl = document.getElementById('document-list');
    if (listEl) {
      listEl.innerHTML = '<div class="loading-state"><div class="spinner"></div><span>Loading documents...</span></div>';
    }
    
    try {
      const params = new URLSearchParams();
      if (this.conversationId) {
        params.set('conversation_id', this.conversationId);
      }
      // Request all documents up to the max limit
      params.set('limit', '500');
      
      const response = await fetch(`/api/data/documents?${params.toString()}`);
      if (!response.ok) throw new Error('Failed to load documents');
      
      const data = await response.json();
      this.documents = data.documents || [];
      this.renderDocuments();
    } catch (error) {
      console.error('Error loading documents:', error);
      this.showError('Failed to load documents. Please try again.');
    }
  }

  /**
   * Load stats from API
   */
  async loadStats() {
    try {
      const params = new URLSearchParams();
      if (this.conversationId) {
        params.set('conversation_id', this.conversationId);
      }
      
      const response = await fetch(`/api/data/stats?${params.toString()}`);
      if (!response.ok) return;
      
      const stats = await response.json();
      this.renderStats(stats);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  }

  /**
   * Render stats bar
   */
  renderStats(stats) {
    const elements = {
      'stat-documents': stats.total_documents || 0,
      'stat-chunks': stats.total_chunks || '--',
      'stat-size': this.formatSize(parseInt(stats.total_size_bytes) || 0),
    };
    
    for (const [id, value] of Object.entries(elements)) {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }
    
    const lastUpdatedEl = document.getElementById('stat-last-updated');
    if (lastUpdatedEl && stats.last_sync) {
      lastUpdatedEl.textContent = this.formatRelativeTime(stats.last_sync);
    }
  }

  /**
   * Render document list using FileTree
   */
  renderDocuments() {
    const listEl = document.getElementById('document-list');
    if (!listEl) return;
    
    // Filter documents
    const filtered = this.filterDocuments(this.documents);
    
    if (filtered.length === 0) {
      listEl.innerHTML = `
        <div class="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="12" y1="18" x2="12" y2="12"/>
            <line x1="9" y1="15" x2="15" y2="15"/>
          </svg>
          <span>${this.searchQuery ? 'No documents match your search' : 'No documents ingested yet'}</span>
        </div>
      `;
      return;
    }
    
    // Build trees
    const trees = this.fileTree.buildTrees(filtered);
    
    // Render each category
    let html = '';
    html += this.fileTree.renderCategory('local_files', trees.localFiles, this.selectedDocument?.hash);
    html += this.fileTree.renderCategory('git', trees.gitRepos, this.selectedDocument?.hash);
    html += this.fileTree.renderCategory('web', trees.webPages, this.selectedDocument?.hash);
    html += this.fileTree.renderCategory('ticket', trees.tickets, this.selectedDocument?.hash);
    
    listEl.innerHTML = html || '<div class="empty-state"><span>No documents to display</span></div>';
  }

  /**
   * Filter documents based on search and type
   */
  filterDocuments(documents) {
    return documents.filter(doc => {
      // Type filter
      if (this.filterType !== 'all' && doc.source_type !== this.filterType) {
        return false;
      }
      
      // Search filter
      if (this.searchQuery) {
        const query = this.searchQuery.toLowerCase();
        const searchFields = [
          doc.display_name,
          doc.url,
          doc.source_type
        ].filter(Boolean);
        
        return searchFields.some(field => 
          field.toLowerCase().includes(query)
        );
      }
      
      return true;
    });
  }

  /**
   * Select a document for preview
   */
  async selectDocument(hash) {
    const doc = this.documents.find(d => d.hash === hash);
    if (!doc) return;
    
    this.selectedDocument = doc;
    this.renderDocuments();
    
    // Show preview panel
    const emptyEl = document.getElementById('preview-empty');
    const contentEl = document.getElementById('preview-content');
    
    if (emptyEl) emptyEl.style.display = 'none';
    if (contentEl) contentEl.style.display = 'flex';
    
    // Update header
    this.updatePreviewHeader(doc);
    
    // Load and render content
    await this.loadDocumentContent(hash);
  }

  /**
   * Update preview header with document info
   */
  updatePreviewHeader(doc) {
    const nameEl = document.getElementById('preview-name');
    if (nameEl) nameEl.textContent = doc.display_name;
    
    const toggleEl = document.getElementById('preview-enabled');
    if (toggleEl) {
      toggleEl.checked = doc.enabled;
      toggleEl.onchange = () => this.toggleDocument(doc.hash, toggleEl.checked);
    }
    
    const toggleLabelEl = document.getElementById('toggle-label');
    if (toggleLabelEl) toggleLabelEl.textContent = doc.enabled ? 'Enabled' : 'Disabled';
    
    // Metadata
    const sourceNames = {
      'local_files': 'Local File',
      'web': 'Web Page',
      'ticket': 'Ticket'
    };
    
    const fields = {
      'preview-source': sourceNames[doc.source_type] || doc.source_type,
      'preview-size': this.formatSize(doc.size_bytes),
      'preview-date': doc.ingested_at ? new Date(doc.ingested_at).toLocaleString() : 'Never',
    };
    
    for (const [id, value] of Object.entries(fields)) {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }
    
    // URL field
    const urlEl = document.getElementById('preview-url');
    if (urlEl) {
      // Sanitize URL to prevent javascript: XSS attacks
      const sanitizedUrl = this.sanitizeUrl(doc.url);
      if (sanitizedUrl && doc.source_type === 'web') {
        urlEl.innerHTML = `<a href="${this.escapeHtml(sanitizedUrl)}" target="_blank" rel="noopener">${this.escapeHtml(doc.url)}</a>`;
        urlEl.parentElement.style.display = 'flex';
      } else {
        urlEl.parentElement.style.display = 'none';
      }
    }
    
    // Content type indicator
    const typeInfo = this.contentRenderer.detectContentType(doc.display_name);
    const typeEl = document.getElementById('preview-type');
    if (typeEl) {
      typeEl.innerHTML = `${typeInfo.icon} ${typeInfo.type}${typeInfo.language ? ` (${typeInfo.language})` : ''}`;
    }
  }

  /**
   * Load and render document content
   */
  async loadDocumentContent(hash) {
    const viewerEl = document.getElementById('content-viewer');
    const loadingEl = document.getElementById('content-loading');
    
    if (loadingEl) loadingEl.style.display = 'flex';
    if (viewerEl) viewerEl.innerHTML = '';
    
    try {
      const params = new URLSearchParams();
      if (this.conversationId) params.set('conversation_id', this.conversationId);
      
      const response = await fetch(`/api/data/documents/${hash}/content?${params.toString()}`);
      if (!response.ok) throw new Error('Failed to load content');
      
      const data = await response.json();
      const content = data.content || '';
      const truncated = data.truncated || false;
      
      // Also fetch chunks
      let chunks = [];
      try {
        const chunksResponse = await fetch(`/api/data/documents/${hash}/chunks`);
        if (chunksResponse.ok) {
          const chunksData = await chunksResponse.json();
          chunks = chunksData.chunks || [];
        }
      } catch (chunkError) {
        console.warn('Could not load chunks:', chunkError);
      }
      
      // Store for re-rendering when toggle changes
      this.selectedContent = content;
      this.selectedChunks = chunks;
      
      if (loadingEl) loadingEl.style.display = 'none';
      
      // Render content
      this.renderContent();
      
      // Update chunk count and show toggle if chunks exist
      const chunkCountEl = document.getElementById('preview-chunks');
      const chunkToggleEl = document.getElementById('chunk-toggle-container');
      
      if (chunkCountEl) {
        chunkCountEl.textContent = `${chunks.length} chunk${chunks.length !== 1 ? 's' : ''}`;
      }
      
      if (chunkToggleEl) {
        chunkToggleEl.style.display = chunks.length > 0 ? 'flex' : 'none';
      }
      
      // Show truncation warning if needed
      const truncatedEl = document.getElementById('content-truncated');
      if (truncatedEl) {
        truncatedEl.style.display = truncated ? 'flex' : 'none';
      }
    } catch (error) {
      console.error('Error loading content:', error);
      if (loadingEl) loadingEl.style.display = 'none';
      if (viewerEl) {
        viewerEl.innerHTML = '<div class="error-state">Error loading content</div>';
      }
    }
  }

  /**
   * Render content with current view mode (normal or chunks)
   */
  renderContent() {
    const viewerEl = document.getElementById('content-viewer');
    if (!viewerEl || !this.selectedDocument) return;
    
    const filename = this.selectedDocument.display_name || 'unknown';
    const rendered = this.contentRenderer.render(
      this.selectedContent, 
      filename, 
      { chunks: this.selectedChunks, showChunks: this.showChunks }
    );
    
    viewerEl.innerHTML = rendered.html;
  }

  /**
   * Toggle chunk view on/off
   */
  toggleChunkView(show) {
    this.showChunks = show;
    this.renderContent();
    
    // Update toggle button state
    const toggleBtn = document.getElementById('chunk-view-toggle');
    if (toggleBtn) {
      toggleBtn.classList.toggle('active', show);
    }
  }

  /**
   * Toggle document enabled state
   */
  async toggleDocument(hash, enabled) {
    if (!this.conversationId) {
      toast.warning('Cannot modify documents without a chat session');
      return;
    }
    
    const endpoint = enabled ? 'enable' : 'disable';
    
    try {
      const response = await fetch(`/api/data/documents/${hash}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: this.conversationId })
      });
      
      if (!response.ok) throw new Error(`Failed to ${endpoint} document`);
      
      // Update local state
      const doc = this.documents.find(d => d.hash === hash);
      if (doc) {
        doc.enabled = enabled;
        this.renderDocuments();
        
        if (this.selectedDocument?.hash === hash) {
          const toggleLabelEl = document.getElementById('toggle-label');
          if (toggleLabelEl) toggleLabelEl.textContent = enabled ? 'Enabled' : 'Disabled';
        }
      }
      
      toast.success(`Document ${enabled ? 'enabled' : 'disabled'}`);
      this.loadStats();
    } catch (error) {
      console.error(`Error ${endpoint}ing document:`, error);
      toast.error(`Failed to ${endpoint} document`);
      this.renderDocuments();
    }
  }

  /**
   * Bulk enable all documents
   */
  async bulkEnable() {
    if (!this.conversationId) {
      toast.warning('Cannot modify documents without a chat session');
      return;
    }
    
    const hashes = this.documents.map(d => d.hash);
    
    try {
      const response = await fetch('/api/data/bulk-enable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          conversation_id: this.conversationId,
          hashes: hashes
        })
      });
      
      if (!response.ok) throw new Error('Failed to enable all');
      
      this.documents.forEach(d => d.enabled = true);
      this.renderDocuments();
      toast.success('All documents enabled');
      this.loadStats();
    } catch (error) {
      console.error('Error enabling all:', error);
      toast.error('Failed to enable all documents');
    }
  }

  /**
   * Bulk disable all documents
   */
  async bulkDisable() {
    if (!this.conversationId) {
      toast.warning('Cannot modify documents without a chat session');
      return;
    }
    
    const hashes = this.documents.map(d => d.hash);
    
    try {
      const response = await fetch('/api/data/bulk-disable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          conversation_id: this.conversationId,
          hashes: hashes
        })
      });
      
      if (!response.ok) throw new Error('Failed to disable all');
      
      this.documents.forEach(d => d.enabled = false);
      this.renderDocuments();
      toast.success('All documents disabled');
      this.loadStats();
    } catch (error) {
      console.error('Error disabling all:', error);
      toast.error('Failed to disable all documents');
    }
  }

  /**
   * Expand all tree nodes
   */
  expandAll() {
    const trees = this.fileTree.buildTrees(this.documents);
    this.fileTree.expandAll(trees.localFiles, 'category-local_files');
    this.fileTree.expandAll(trees.gitRepos, 'category-git');
    this.fileTree.expandAll(trees.webPages, 'category-web');
    this.renderDocuments();
  }

  /**
   * Collapse all tree nodes
   */
  collapseAll() {
    const trees = this.fileTree.buildTrees(this.documents);
    this.fileTree.collapseAll(trees.localFiles, 'category-local_files');
    this.fileTree.collapseAll(trees.gitRepos, 'category-git');
    this.fileTree.collapseAll(trees.webPages, 'category-web');
    this.renderDocuments();
  }

  /**
   * Refresh documents and stats
   */
  async refresh() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) refreshBtn.classList.add('loading');
    
    await Promise.all([
      this.loadDocuments(),
      this.loadStats()
    ]);
    
    if (refreshBtn) refreshBtn.classList.remove('loading');
    toast.success('Data refreshed');
  }

  /**
   * Show error message
   */
  showError(message) {
    const listEl = document.getElementById('document-list');
    if (listEl) {
      listEl.innerHTML = `
        <div class="error-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>${this.escapeHtml(message)}</span>
        </div>
      `;
    }
  }

  /**
   * Utility Functions (kept as fallbacks when archiUtils is not loaded)
   */
  formatSize(bytes) {
    if (archiUtils?.formatSize) {
      return archiUtils.formatSize(bytes);
    }
    if (!bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (bytes >= 1024 && i < units.length - 1) {
      bytes /= 1024;
      i++;
    }
    return `${bytes.toFixed(1)} ${units[i]}`;
  }

  formatRelativeTime(dateString) {
    if (archiUtils?.formatRelativeTime) {
      return archiUtils.formatRelativeTime(dateString);
    }
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  }

  escapeHtml(text) {
    if (archiUtils?.escapeHtml) {
      return archiUtils.escapeHtml(text);
    }
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Sanitize URL to prevent XSS from javascript: and other dangerous schemes
   */
  sanitizeUrl(url) {
    if (archiUtils?.sanitizeUrl) {
      return archiUtils.sanitizeUrl(url);
    }
    if (!url) return '';
    
    // Only allow safe URL schemes
    const safeSchemes = ['http:', 'https:', 'mailto:', 'tel:'];
    try {
      const parsed = new URL(url);
      if (safeSchemes.includes(parsed.protocol)) {
        return url;
      }
    } catch (e) {
      // Invalid URL
    }
    return '';
  }
}

// Global instances for event handlers
let dataViewer;
let fileTree;

document.addEventListener('DOMContentLoaded', () => {
  dataViewer = new DataViewer();
  fileTree = dataViewer.fileTree;
});
