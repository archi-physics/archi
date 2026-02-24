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
    this.stats = null;
    this.totalDocuments = 0;
    this.pageSize = 500;
    this.hydrationInProgress = false;
    this._loadVersion = 0;
    
    // Initialize modules
    this.fileTree = new FileTree({
      onSelect: (hash) => this.selectDocument(hash),
      onToggle: (pathOrHash, enabled) => {
        this.renderDocuments();
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
    
    const loadVersion = ++this._loadVersion;
    this.documents = [];
    this.totalDocuments = 0;
    this.hydrationInProgress = false;
    this.updateListStatus();

    try {
      const firstPage = await this.fetchDocumentPage(this.pageSize, 0);
      if (loadVersion !== this._loadVersion) return;

      this.totalDocuments = firstPage.total || 0;
      this.mergeDocuments(firstPage.documents || []);
      this.renderDocuments();
      this.updateListStatus();

      if (firstPage.has_more) {
        this.hydrationInProgress = true;
        this.updateListStatus();
        await this.hydrateRemainingPages(firstPage.next_offset || this.documents.length, loadVersion);
      }
    } catch (error) {
      console.error('Error loading documents:', error);
      this.showError('Failed to load documents. Please try again.');
      this.updateListStatus('Failed to load documents');
    } finally {
      if (loadVersion === this._loadVersion) {
        this.hydrationInProgress = false;
        this.updateListStatus();
      }
    }
  }

  async fetchDocumentPage(limit, offset) {
    const params = new URLSearchParams();
    if (this.conversationId) {
      params.set('conversation_id', this.conversationId);
    }
    params.set('limit', String(limit));
    params.set('offset', String(offset));

    const response = await fetch(`/api/data/documents?${params.toString()}`);
    if (!response.ok) throw new Error('Failed to load documents');
    return response.json();
  }

  async hydrateRemainingPages(startOffset, loadVersion) {
    let nextOffset = startOffset;

    while (nextOffset != null && loadVersion === this._loadVersion) {
      const page = await this.fetchDocumentPage(this.pageSize, nextOffset);
      if (loadVersion !== this._loadVersion) return;

      this.mergeDocuments(page.documents || []);
      this.totalDocuments = page.total || this.totalDocuments;
      this.renderDocuments();
      this.updateListStatus();

      if (!page.has_more) break;
      nextOffset = page.next_offset;
    }
  }

  mergeDocuments(newDocuments) {
    if (!Array.isArray(newDocuments) || newDocuments.length === 0) return;

    const byHash = new Map(this.documents.map((doc) => [doc.hash, doc]));
    for (const doc of newDocuments) {
      if (doc && doc.hash) {
        byHash.set(doc.hash, doc);
      }
    }
    this.documents = Array.from(byHash.values());
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
      this.stats = stats;
      this.renderStats(stats);
      this.renderDocuments();
      this.updateListStatus();
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
      if (!this.searchQuery && this.filterType !== 'all') {
        const authoritativeCount = this.getCategoryCount(this.filterType);
        if (typeof authoritativeCount === 'number' && authoritativeCount > 0) {
          const emptyTrees = this.fileTree.buildTrees([]);
          const treeByType = {
            local_files: emptyTrees.localFiles,
            git: emptyTrees.gitRepos,
            web: emptyTrees.webPages,
            ticket: emptyTrees.tickets,
            sso: emptyTrees.ssoPages,
            other: emptyTrees.otherSources,
          };
          listEl.innerHTML = this.fileTree.renderCategory(
            this.filterType,
            treeByType[this.filterType],
            this.selectedDocument?.hash,
            {
              countOverride: authoritativeCount,
              hydrating: this.hydrationInProgress,
            }
          );
          this.updateListStatus();
          return;
        }
      }

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
      this.updateListStatus();
      return;
    }
    
    // Build trees
    const trees = this.fileTree.buildTrees(filtered);
    
    // Render categories
    const categories = this.filterType === 'all'
      ? ['local_files', 'git', 'web', 'ticket', 'sso', 'other']
      : [this.filterType];

    const loadingCategories = this.getLoadingCategories();
    let html = '';
    if (categories.includes('local_files')) {
      html += this.fileTree.renderCategory('local_files', trees.localFiles, this.selectedDocument?.hash, {
        countOverride: this.getCategoryCount('local_files'),
        hydrating: loadingCategories.has('local_files'),
      });
    }
    if (categories.includes('git')) {
      html += this.fileTree.renderCategory('git', trees.gitRepos, this.selectedDocument?.hash, {
        countOverride: this.getCategoryCount('git'),
        hydrating: loadingCategories.has('git'),
      });
    }
    if (categories.includes('web')) {
      html += this.fileTree.renderCategory('web', trees.webPages, this.selectedDocument?.hash, {
        countOverride: this.getCategoryCount('web'),
        hydrating: loadingCategories.has('web'),
      });
    }
    if (categories.includes('ticket')) {
      html += this.fileTree.renderCategory('ticket', trees.tickets, this.selectedDocument?.hash, {
        countOverride: this.getCategoryCount('ticket'),
        hydrating: loadingCategories.has('ticket'),
      });
    }
    if (categories.includes('sso')) {
      html += this.fileTree.renderCategory('sso', trees.ssoPages, this.selectedDocument?.hash, {
        countOverride: this.getCategoryCount('sso'),
        hydrating: loadingCategories.has('sso'),
      });
    }
    if (categories.includes('other')) {
      html += this.fileTree.renderCategory('other', trees.otherSources, this.selectedDocument?.hash, {
        countOverride: this.getCategoryCount('other'),
        hydrating: loadingCategories.has('other'),
      });
    }
    
    listEl.innerHTML = html || '<div class="empty-state"><span>No documents to display</span></div>';
    this.updateListStatus();
  }

  /**
   * Filter documents based on search and type
   */
  filterDocuments(documents) {
    return documents.filter(doc => {
      // Type filter
      const docCategory = this.getCategoryForSourceType(doc.source_type);
      if (this.filterType !== 'all' && docCategory !== this.filterType) {
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

  getCategoryCount(sourceType) {
    // During active search, category counts should reflect visible matches.
    if (this.searchQuery) {
      return undefined;
    }
    const bySource = this.stats?.by_source_type || {};
    if (sourceType === 'other') {
      let total = 0;
      let found = false;
      for (const [rawType, counts] of Object.entries(bySource)) {
        if (this.getCategoryForSourceType(rawType) === 'other' && typeof counts?.total === 'number') {
          total += counts.total;
          found = true;
        }
      }
      return found ? total : undefined;
    }

    const sourceStats = bySource[sourceType];
    if (!sourceStats || typeof sourceStats.total !== 'number') return undefined;
    return sourceStats.total;
  }

  getLoadingCategories() {
    const loading = new Set();
    if (!this.hydrationInProgress) return loading;
    if (this.searchQuery) return loading;

    const loadedCounts = this.documents.reduce((acc, doc) => {
      const category = this.getCategoryForSourceType(doc?.source_type);
      acc[category] = (acc[category] || 0) + 1;
      return acc;
    }, {});

    const categories = ['local_files', 'git', 'web', 'ticket', 'sso', 'other'];
    for (const category of categories) {
      const totalForCategory = this.getCategoryCount(category);
      if (typeof totalForCategory === 'number' && (loadedCounts[category] || 0) < totalForCategory) {
        loading.add(category);
      }
    }
    return loading;
  }

  getCategoryForSourceType(sourceType) {
    const explicit = new Set(['local_files', 'git', 'web', 'ticket', 'sso']);
    return explicit.has(sourceType) ? sourceType : 'other';
  }

  updateListStatus(errorText = '') {
    const statusEl = document.getElementById('list-status');
    if (!statusEl) return;

    if (errorText) {
      statusEl.textContent = errorText;
      statusEl.classList.add('error');
      return;
    }

    statusEl.classList.remove('error');
    const loaded = this.documents.length;
    const total = this.totalDocuments || this.stats?.total_documents || loaded;
    const ingestionSuffix = this.getIngestionStatusSuffix();

    if (total === 0) {
      statusEl.textContent = '';
      return;
    }

    if (this.searchQuery) {
      const filteredCount = this.filterDocuments(this.documents).length;
      statusEl.textContent = `Showing ${filteredCount} matching documents (${loaded} loaded of ${total} total)${ingestionSuffix}`;
      return;
    }

    if (loaded < total || this.hydrationInProgress) {
      statusEl.textContent = `Showing ${loaded} of ${total} documents (loading remaining...)${ingestionSuffix}`;
      return;
    }

    statusEl.textContent = `Showing all ${total} documents${ingestionSuffix}`;
  }

  getIngestionStatusSuffix() {
    const statusCounts = this.stats?.status_counts;
    if (statusCounts) {
      const collecting = Number(statusCounts.pending || 0);
      const embedding = Number(statusCounts.embedding || 0);
      const parts = [];

      if (collecting > 0) {
        parts.push('data collection ongoing');
      }
      if (embedding > 0) {
        const leftToEmbed = collecting + embedding;
        if (Number.isFinite(leftToEmbed) && leftToEmbed > 0) {
          const noun = leftToEmbed === 1 ? 'document' : 'documents';
          parts.push(`${leftToEmbed} ${noun} left to embed`);
        } else {
          parts.push('embedding in progress');
        }
      }

      if (parts.length > 0) {
        return ` (${parts.join(', ')})`;
      }
      return '';
    }

    // Fallback for partial stats payloads
    const hasPending = this.documents.some((doc) => doc?.ingestion_status === 'pending');
    const hasEmbedding = this.documents.some((doc) => doc?.ingestion_status === 'embedding');
    const fallbackParts = [];

    if (hasPending) {
      fallbackParts.push('data collection ongoing');
    }
    if (hasEmbedding) {
      fallbackParts.push('embedding in progress');
    }

    if (fallbackParts.length > 0) {
      return ` (${fallbackParts.join(', ')})`;
    }

    return '';
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
    
    // Metadata
    const sourceNames = {
      'local_files': 'Local File',
      'web': 'Web Page',
      'ticket': 'Ticket',
      'sso': 'SSO Page'
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
    
    // Ingestion status badge
    const statusEl = document.getElementById('preview-ingestion-status');
    if (statusEl) {
      const status = doc.ingestion_status || 'pending';
      const statusLabel = status.charAt(0).toUpperCase() + status.slice(1);
      statusEl.innerHTML = `<span class="status-badge ${status}"><span class="status-dot ${status}"></span>${statusLabel}</span>`;
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
   * Expand all tree nodes
   */
  expandAll() {
    const trees = this.fileTree.buildTrees(this.documents);
    this.fileTree.expandAll(trees.localFiles, 'category-local_files');
    this.fileTree.expandAll(trees.gitRepos, 'category-git');
    this.fileTree.expandAll(trees.webPages, 'category-web');
    this.fileTree.expandAll(trees.ssoPages, 'category-sso');
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
    this.fileTree.collapseAll(trees.ssoPages, 'category-sso');
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
