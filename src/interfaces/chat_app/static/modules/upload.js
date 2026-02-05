/**
 * DataUploader - Upload Module
 * 
 * Manages data upload UI for files, URLs, Git repos, and Jira projects.
 * Uses Dropzone.js for file uploads.
 * 
 * Dependencies: utils.js, toast.js, api-client.js
 */

class DataUploader {
  constructor() {
    this.dropzone = null;
    this.urlQueue = [];
    this.gitRepos = [];
    this.jiraProjects = [];
    this.isEmbedding = false;
    
    this.init();
  }

  init() {
    this.bindTabEvents();
    this.initDropzone();
    this.bindFormEvents();
    this.loadExistingSources();
    this.initEmbeddingStatus();
  }

  /**
   * Embedding Status Management
   */
  initEmbeddingStatus() {
    const embedBtn = document.getElementById('embed-btn');
    if (embedBtn) {
      embedBtn.addEventListener('click', () => this.triggerEmbedding());
    }
    
    // Load initial status
    this.refreshEmbeddingStatus();
  }

  async refreshEmbeddingStatus() {
    const statusBar = document.getElementById('embedding-status-bar');
    const statusText = document.getElementById('embedding-status-text');
    const embedBtn = document.getElementById('embed-btn');
    
    if (!statusBar || !statusText) return;
    
    try {
      const response = await fetch('/api/upload/status');
      if (!response.ok) throw new Error('Failed to fetch status');
      
      const data = await response.json();
      
      statusBar.classList.remove('synced', 'pending', 'processing');
      
      if (this.isEmbedding) {
        statusBar.classList.add('processing');
        statusText.textContent = 'Processing documents...';
        if (embedBtn) embedBtn.disabled = true;
      } else if (data.is_synced) {
        statusBar.classList.add('synced');
        statusText.textContent = `✓ All ${data.documents_embedded} documents are embedded and searchable`;
        if (embedBtn) embedBtn.disabled = true;
      } else {
        statusBar.classList.add('pending');
        statusText.textContent = `${data.pending_embedding} document${data.pending_embedding !== 1 ? 's' : ''} waiting to be processed (${data.documents_embedded} embedded)`;
        if (embedBtn) embedBtn.disabled = false;
      }
    } catch (err) {
      console.error('Failed to fetch embedding status:', err);
      statusText.textContent = 'Unable to fetch status';
      if (embedBtn) embedBtn.disabled = true;
    }
  }

  async triggerEmbedding() {
    if (this.isEmbedding) return;
    
    const embedBtn = document.getElementById('embed-btn');
    const statusBar = document.getElementById('embedding-status-bar');
    const statusText = document.getElementById('embedding-status-text');
    
    this.isEmbedding = true;
    statusBar.classList.remove('synced', 'pending');
    statusBar.classList.add('processing');
    statusText.textContent = 'Processing documents...';
    if (embedBtn) {
      embedBtn.disabled = true;
      embedBtn.querySelector('span').textContent = 'Processing...';
    }
    
    try {
      const response = await fetch('/api/upload/embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      if (response.ok && data.success) {
        toast.success('Documents processed successfully! They are now searchable.');
      } else {
        throw new Error(data.error || 'Embedding failed');
      }
    } catch (err) {
      console.error('Embedding error:', err);
      toast.error(err.message || 'Failed to process documents');
    } finally {
      this.isEmbedding = false;
      if (embedBtn) {
        embedBtn.querySelector('span').textContent = 'Process Documents';
      }
      this.refreshEmbeddingStatus();
    }
  }

  /**
   * Tab Switching
   */
  bindTabEvents() {
    const tabs = document.querySelectorAll('.source-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        // Update tab states
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update panel visibility
        const source = tab.dataset.source;
        document.querySelectorAll('.upload-panel').forEach(panel => {
          panel.classList.remove('active');
        });
        const targetPanel = document.getElementById(`panel-${source}`);
        if (targetPanel) {
          targetPanel.classList.add('active');
        }
      });
    });
  }

  /**
   * Dropzone Initialization
   */
  initDropzone() {
    const dropzoneEl = document.getElementById('file-dropzone');
    if (!dropzoneEl) return;

    // Check if Dropzone is available
    if (typeof Dropzone === 'undefined') {
      console.warn('Dropzone.js not loaded - file upload will use fallback');
      this.initFallbackUpload(dropzoneEl);
      return;
    }

    // Disable auto-discover
    Dropzone.autoDiscover = false;

    this.dropzone = new Dropzone('#file-dropzone', {
      url: '/api/upload/file',
      paramName: 'file',
      maxFilesize: 50, // MB
      acceptedFiles: '.pdf,.md,.txt,.docx,.html,.htm,.json,.yaml,.yml,.py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.c,.cpp,.h,.sh',
      parallelUploads: 2,
      autoProcessQueue: true,
      addRemoveLinks: false,
      createImageThumbnails: false,
      clickable: true, // Explicitly enable click-to-browse
      previewTemplate: '<div style="display:none"></div>', // Hide default preview
      
      init: function() {
        // Make the entire dropzone clickable
        this.element.addEventListener('click', (e) => {
          // Don't trigger if clicking on a button or link inside
          if (e.target.tagName !== 'BUTTON' && e.target.tagName !== 'A') {
            this.hiddenFileInput.click();
          }
        });
      }
    });

    // Event handlers
    this.dropzone.on('addedfile', (file) => this.onFileAdded(file));
    this.dropzone.on('uploadprogress', (file, progress) => this.onUploadProgress(file, progress));
    this.dropzone.on('success', (file, response) => this.onUploadSuccess(file, response));
    this.dropzone.on('error', (file, errorMessage) => this.onUploadError(file, errorMessage));

    // Clear queue button
    const clearBtn = document.getElementById('clear-queue-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clearUploadQueue());
    }
  }

  /**
   * Fallback Upload (when Dropzone is not available)
   */
  initFallbackUpload(dropzoneEl) {
    // Create a hidden file input
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.accept = '.pdf,.md,.txt,.docx,.html,.htm,.json,.yaml,.yml,.py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.c,.cpp,.h,.sh';
    fileInput.style.display = 'none';
    dropzoneEl.appendChild(fileInput);

    // Click on dropzone triggers file input
    dropzoneEl.addEventListener('click', (e) => {
      if (e.target !== fileInput) {
        fileInput.click();
      }
    });

    // Handle file selection
    fileInput.addEventListener('change', async () => {
      const files = fileInput.files;
      for (const file of files) {
        await this.uploadFileFallback(file);
      }
      fileInput.value = ''; // Reset for next selection
    });

    // Clear queue button
    const clearBtn = document.getElementById('clear-queue-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clearUploadQueue());
    }
  }

  async uploadFileFallback(file) {
    const queueList = document.getElementById('upload-queue-list');
    if (!queueList) return;

    // Remove empty message
    const emptyMsg = queueList.querySelector('.empty-queue-message');
    if (emptyMsg) emptyMsg.remove();

    // Create upload item
    const itemId = `upload-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const item = document.createElement('div');
    item.className = 'upload-item';
    item.id = itemId;
    item.innerHTML = `
      <div class="upload-item-icon">${this.getFileIcon(file.name)}</div>
      <div class="upload-item-info">
        <div class="upload-item-name">${this.escapeHtml(file.name)}</div>
        <div class="upload-item-meta"><span>${this.formatFileSize(file.size)}</span></div>
        <div class="progress-bar"><div class="progress-bar-fill" style="width: 0%"></div></div>
      </div>
      <div class="upload-item-status status-uploading">Uploading...</div>
    `;
    queueList.appendChild(item);

    // Upload the file
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload/file', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      const progressFill = item.querySelector('.progress-bar-fill');
      const statusEl = item.querySelector('.upload-item-status');
      
      if (response.ok) {
        progressFill.style.width = '100%';
        statusEl.textContent = 'Complete';
        statusEl.className = 'upload-item-status status-complete';
      } else {
        throw new Error(data.error || 'Upload failed');
      }
    } catch (err) {
      const statusEl = item.querySelector('.upload-item-status');
      statusEl.textContent = err.message;
      statusEl.className = 'upload-item-status status-error';
    }
  }

  /**
   * File Upload Events
   */
  onFileAdded(file) {
    const queueList = document.getElementById('upload-queue-list');
    if (!queueList) return;

    // Remove empty message
    const emptyMsg = queueList.querySelector('.empty-queue-message');
    if (emptyMsg) emptyMsg.remove();

    // Create upload item
    const item = document.createElement('div');
    item.className = 'upload-item';
    item.id = `upload-${file.upload.uuid}`;
    item.innerHTML = `
      <div class="upload-item-icon">
        ${this.getFileIcon(file.name)}
      </div>
      <div class="upload-item-info">
        <div class="upload-item-name">${this.escapeHtml(file.name)}</div>
        <div class="upload-item-meta">
          <span>${this.formatFileSize(file.size)}</span>
        </div>
        <div class="progress-bar">
          <div class="progress-bar-fill" style="width: 0%"></div>
        </div>
      </div>
      <div class="upload-item-status processing">
        <span class="spinner"></span>
        <span>Uploading...</span>
      </div>
    `;
    queueList.appendChild(item);
  }

  onUploadProgress(file, progress) {
    const item = document.getElementById(`upload-${file.upload.uuid}`);
    if (!item) return;

    const progressFill = item.querySelector('.progress-bar-fill');
    if (progressFill) {
      progressFill.style.width = `${progress}%`;
    }
  }

  onUploadSuccess(file, response) {
    const item = document.getElementById(`upload-${file.upload.uuid}`);
    if (!item) return;

    // Remove progress bar
    const progressBar = item.querySelector('.progress-bar');
    if (progressBar) progressBar.remove();

    // Update status
    const status = item.querySelector('.upload-item-status');
    if (status) {
      status.className = 'upload-item-status success';
      status.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        <span>Uploaded - Click Process to embed</span>
      `;
    }
    
    // Refresh embedding status to show new pending documents
    this.refreshEmbeddingStatus();
  }

  onUploadError(file, errorMessage) {
    const item = document.getElementById(`upload-${file.upload.uuid}`);
    if (!item) return;

    // Remove progress bar
    const progressBar = item.querySelector('.progress-bar');
    if (progressBar) progressBar.remove();

    // Update status
    const status = item.querySelector('.upload-item-status');
    if (status) {
      status.className = 'upload-item-status error';
      const msg = typeof errorMessage === 'object' ? (errorMessage.error || 'Upload failed') : errorMessage;
      status.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="15" y1="9" x2="9" y2="15"></line>
          <line x1="9" y1="9" x2="15" y2="15"></line>
        </svg>
        <span>${this.escapeHtml(msg)}</span>
      `;
    }
  }

  clearUploadQueue() {
    const queueList = document.getElementById('upload-queue-list');
    if (queueList) {
      queueList.innerHTML = '<div class="empty-queue-message">No files in queue</div>';
    }
    if (this.dropzone) {
      this.dropzone.removeAllFiles(true);
    }
  }

  /**
   * Form Event Bindings
   */
  bindFormEvents() {
    // URL form
    const addUrlBtn = document.getElementById('add-url-btn');
    if (addUrlBtn) {
      addUrlBtn.addEventListener('click', () => this.addUrl());
    }
    
    const urlInput = document.getElementById('url-input');
    if (urlInput) {
      urlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') this.addUrl();
      });
    }

    const scrapeBtn = document.getElementById('scrape-urls-btn');
    if (scrapeBtn) {
      scrapeBtn.addEventListener('click', () => this.scrapeUrls());
    }

    // Git form
    const cloneBtn = document.getElementById('clone-repo-btn');
    if (cloneBtn) {
      cloneBtn.addEventListener('click', () => this.cloneGitRepo());
    }
    
    const gitInput = document.getElementById('git-url-input');
    if (gitInput) {
      gitInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') this.cloneGitRepo();
      });
    }

    // Jira form
    const syncJiraBtn = document.getElementById('sync-jira-btn');
    if (syncJiraBtn) {
      syncJiraBtn.addEventListener('click', () => this.syncJiraProject());
    }
    
    const jiraInput = document.getElementById('jira-project-input');
    if (jiraInput) {
      jiraInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') this.syncJiraProject();
      });
    }

    // Schedule dropdowns
    const jiraScheduleSelect = document.getElementById('jira-schedule');
    if (jiraScheduleSelect) {
      jiraScheduleSelect.addEventListener('change', (e) => {
        this.updateSourceSchedule('jira', e.target.value);
      });
    }
    
    const gitScheduleSelect = document.getElementById('git-schedule');
    if (gitScheduleSelect) {
      gitScheduleSelect.addEventListener('change', (e) => {
        this.updateSourceSchedule('git', e.target.value);
      });
    }

    const linksScheduleSelect = document.getElementById('links-schedule');
    if (linksScheduleSelect) {
      linksScheduleSelect.addEventListener('change', (e) => {
        this.updateSourceSchedule('links', e.target.value);
      });
    }

    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.loadExistingSources());
    }

    // Event delegation for dynamically rendered action buttons
    this.bindActionDelegation();
  }

  /**
   * Event Delegation for Dynamic Action Buttons
   * Handles clicks on data-action buttons in rendered lists to avoid inline onclick handlers
   */
  bindActionDelegation() {
    // URL queue actions
    const urlQueueContainer = document.getElementById('url-queue');
    if (urlQueueContainer) {
      urlQueueContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;
        
        const action = btn.dataset.action;
        const index = parseInt(btn.dataset.index, 10);
        
        if (action === 'remove-url' && !isNaN(index)) {
          this.removeUrl(index);
        }
      });
    }

    // Git repos actions
    const gitReposContainer = document.getElementById('git-repos-list');
    if (gitReposContainer) {
      gitReposContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;
        
        const action = btn.dataset.action;
        const repo = btn.dataset.repo;
        
        if (action === 'refresh-git' && repo) {
          this.refreshGitRepo(repo);
        } else if (action === 'remove-git' && repo) {
          this.removeGitRepo(repo);
        }
      });
    }

    // Jira projects actions
    const jiraProjectsContainer = document.getElementById('jira-projects-list');
    if (jiraProjectsContainer) {
      jiraProjectsContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;
        
        const action = btn.dataset.action;
        const project = btn.dataset.project;
        
        if (action === 'refresh-jira' && project) {
          this.refreshJiraProject(project);
        } else if (action === 'remove-jira' && project) {
          this.removeJiraProject(project);
        }
      });
    }
  }

  /**
   * URL Management
   */
  addUrl() {
    const input = document.getElementById('url-input');
    const url = input?.value?.trim();
    
    if (!url) return;
    if (!this.isValidUrl(url)) {
      alert('Please enter a valid URL');
      return;
    }

    const followLinks = document.getElementById('url-follow-links')?.checked ?? true;
    const requiresSso = document.getElementById('url-requires-sso')?.checked ?? false;
    const depth = document.getElementById('url-crawl-depth')?.value ?? '2';

    this.urlQueue.push({ url, followLinks, requiresSso, depth });
    input.value = '';
    this.renderUrlQueue();
  }

  renderUrlQueue() {
    const container = document.getElementById('url-queue');
    if (!container) return;

    if (this.urlQueue.length === 0) {
      container.innerHTML = '<div class="empty-list-message">No URLs queued</div>';
      return;
    }

    const escapeHtml = archiUtils?.escapeHtml || this.escapeHtml.bind(this);
    
    container.innerHTML = this.urlQueue.map((item, idx) => `
      <div class="source-item">
        <div class="source-item-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
          </svg>
        </div>
        <div class="source-item-info">
          <div class="source-item-name">${escapeHtml(item.url)}</div>
          <div class="source-item-meta">Depth: ${item.depth} • ${item.requiresSso ? 'SSO required' : 'No SSO'}</div>
        </div>
        <div class="source-item-actions">
          <button class="btn-icon danger" data-action="remove-url" data-index="${idx}" title="Remove">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
      </div>
    `).join('');
  }

  removeUrl(idx) {
    this.urlQueue.splice(idx, 1);
    this.renderUrlQueue();
  }

  async scrapeUrls() {
    if (this.urlQueue.length === 0) return;

    const btn = document.getElementById('scrape-urls-btn');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Scraping...';
    }

    let successCount = 0;
    let failCount = 0;
    let totalResources = 0;
    const errors = [];

    try {
      for (const item of this.urlQueue) {
        const response = await fetch('/api/upload/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: item.url,
            depth: parseInt(item.depth),
            requires_sso: item.requiresSso
          })
        });

        const data = await response.json();
        if (data.success) {
          successCount++;
          totalResources += data.resources_scraped || 0;
        } else {
          failCount++;
          errors.push(`${item.url}: ${data.message || data.error || 'Unknown error'}`);
        }
      }

      this.urlQueue = [];
      this.renderUrlQueue();

      // Show appropriate notification based on results
      if (failCount === 0 && successCount > 0) {
        toast.success(`Successfully scraped ${totalResources} page(s) from ${successCount} URL(s)`);
      } else if (successCount > 0 && failCount > 0) {
        toast.warning(`Scraped ${totalResources} page(s) from ${successCount} URL(s). ${failCount} URL(s) failed.`);
        console.warn('Scrape errors:', errors);
      } else {
        toast.error(`Failed to scrape URLs: ${errors[0] || 'No content found'}`);
      }
    } catch (err) {
      console.error('Scrape error:', err);
      toast.error('Failed to scrape URLs');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Start Scraping';
      }
      // Refresh embedding status after scraping
      this.refreshEmbeddingStatus();
    }
  }

  /**
   * Git Repository Management
   */
  async cloneGitRepo() {
    const input = document.getElementById('git-url-input');
    const repoUrl = input?.value?.trim();
    
    if (!repoUrl) return;

    const btn = document.getElementById('clone-repo-btn');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Cloning...';
    }

    try {
      const response = await fetch('/api/upload/git', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl })
      });
      
      const data = await response.json();
      if (response.ok && data.success) {
        input.value = '';
        toast.success('Repository cloned successfully. Click Process to embed.');
        this.loadGitRepos();
      } else {
        throw new Error(data.error || data.message || 'Clone failed');
      }
    } catch (err) {
      console.error('Clone error:', err);
      toast.error(err.message || 'Failed to clone repository');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Clone';
      }
      // Refresh embedding status after cloning
      this.refreshEmbeddingStatus();
    }
  }

  async loadGitRepos() {
    const container = document.getElementById('git-repos-list');
    if (!container) return;

    try {
      const response = await fetch('/api/sources/git');
      if (response.ok) {
        const data = await response.json();
        this.gitRepos = data.sources || [];
        this.renderGitRepos();
      }
    } catch (err) {
      console.error('Failed to load git repos:', err);
      container.innerHTML = '<div class="empty-list-message">Failed to load repositories</div>';
    }
  }

  renderGitRepos() {
    const container = document.getElementById('git-repos-list');
    if (!container) return;

    if (!this.gitRepos || this.gitRepos.length === 0) {
      container.innerHTML = '<div class="empty-list-message">No repositories indexed</div>';
      return;
    }

    const escapeAttr = archiUtils?.escapeAttr || this.escapeHtml.bind(this);
    const escapeHtml = archiUtils?.escapeHtml || this.escapeHtml.bind(this);
    const formatDate = archiUtils?.formatRelativeTime || this.formatDate.bind(this);
    
    container.innerHTML = this.gitRepos.map(repo => {
      const repoId = escapeAttr(repo.url || repo.name);
      return `
      <div class="source-item">
        <div class="source-item-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="4"></circle>
            <line x1="1.05" y1="12" x2="7" y2="12"></line>
            <line x1="17.01" y1="12" x2="22.96" y2="12"></line>
          </svg>
        </div>
        <div class="source-item-info">
          <div class="source-item-name">${escapeHtml(repo.name || repo.repo_name)}</div>
          <div class="source-item-meta">${repo.file_count || 0} files • Updated ${formatDate(repo.last_updated || repo.updated_at)}</div>
        </div>
        <div class="source-item-actions">
          <button class="btn-icon" data-action="refresh-git" data-repo="${repoId}" title="Refresh">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 4 23 10 17 10"></polyline>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
          </button>
          <button class="btn-icon danger" data-action="remove-git" data-repo="${repoId}" title="Remove">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
      </div>
    `}).join('');
  }

  async refreshGitRepo(repoName) {
    try {
      const response = await fetch('/api/upload/git/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_name: repoName })
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Refresh failed');
      }
      toast.success('Repository refreshed');
      this.loadGitRepos();
    } catch (err) {
      console.error('Refresh git error:', err);
      toast.error(err.message || 'Failed to refresh repository');
    }
  }

  async removeGitRepo(repoName) {
    if (!confirm(`Remove repository "${repoName}" and all its indexed files?`)) return;
    
    try {
      const response = await fetch('/api/upload/git', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_name: repoName })
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Delete failed');
      }
      toast.success('Repository removed');
      this.loadGitRepos();
    } catch (err) {
      console.error('Remove git error:', err);
      toast.error(err.message || 'Failed to remove repository');
    }
  }

  /**
   * Jira Project Management
   */
  async syncJiraProject() {
    const input = document.getElementById('jira-project-input');
    const projectKey = input?.value?.trim().toUpperCase();
    
    if (!projectKey) return;

    const btn = document.getElementById('sync-jira-btn');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Syncing...';
    }

    try {
      const response = await fetch('/api/upload/jira', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_key: projectKey })
      });
      
      const data = await response.json();
      if (response.ok) {
        input.value = '';
        toast.success('Jira project synced successfully');
        this.loadJiraProjects();
      } else {
        throw new Error(data.error || 'Sync failed');
      }
    } catch (err) {
      console.error('Sync error:', err);
      toast.error(err.message || 'Failed to sync Jira project');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Sync Project';
      }
    }
  }

  async loadJiraProjects() {
    const container = document.getElementById('jira-projects-list');
    if (!container) return;

    try {
      const response = await fetch('/api/sources/jira');
      if (response.ok) {
        const data = await response.json();
        this.jiraProjects = data.sources || [];
        this.renderJiraProjects();
      }
    } catch (err) {
      console.error('Failed to load Jira projects:', err);
      container.innerHTML = '<div class="empty-list-message">Failed to load projects</div>';
    }
  }

  renderJiraProjects() {
    const container = document.getElementById('jira-projects-list');
    if (!container) return;

    if (!this.jiraProjects || this.jiraProjects.length === 0) {
      container.innerHTML = '<div class="empty-list-message">No Jira projects synced</div>';
      return;
    }

    const escapeAttr = archiUtils?.escapeAttr || this.escapeHtml.bind(this);
    const escapeHtml = archiUtils?.escapeHtml || this.escapeHtml.bind(this);
    const formatDate = archiUtils?.formatRelativeTime || this.formatDate.bind(this);
    
    container.innerHTML = this.jiraProjects.map(project => {
      const projectKey = escapeAttr(project.key);
      return `
      <div class="source-item">
        <div class="source-item-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="3" y1="9" x2="21" y2="9"></line>
            <line x1="9" y1="21" x2="9" y2="9"></line>
          </svg>
        </div>
        <div class="source-item-info">
          <div class="source-item-name">${escapeHtml(project.key)} - ${escapeHtml(project.name || 'Project')}</div>
          <div class="source-item-meta">${project.ticket_count || 0} tickets • Last sync: ${formatDate(project.last_sync)}</div>
        </div>
        <div class="source-item-actions">
          <button class="btn-icon" data-action="refresh-jira" data-project="${projectKey}" title="Sync now">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 4 23 10 17 10"></polyline>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
          </button>
          <button class="btn-icon danger" data-action="remove-jira" data-project="${projectKey}" title="Remove">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
      </div>
    `}).join('');
  }

  async refreshJiraProject(projectKey) {
    try {
      const response = await fetch('/api/upload/jira', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_key: projectKey })
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Sync failed');
      }
      toast.success('Project synced');
      this.loadJiraProjects();
    } catch (err) {
      console.error('Sync jira error:', err);
      toast.error(err.message || 'Failed to sync project');
    }
  }

  async removeJiraProject(projectKey) {
    if (!confirm(`Remove Jira project "${projectKey}" and all synced tickets?`)) return;
    
    try {
      const response = await fetch('/api/sources/jira', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_key: projectKey })
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Delete failed');
      }
      toast.success('Project removed');
      this.loadJiraProjects();
    } catch (err) {
      console.error('Remove jira error:', err);
      toast.error(err.message || 'Failed to remove project');
    }
  }

  /**
   * Load Existing Sources
   */
  loadExistingSources() {
    this.loadGitRepos();
    this.loadJiraProjects();
    this.loadSourceSchedules();
  }

  async loadSourceSchedules() {
    try {
      const response = await fetch('/api/sources/schedules');
      if (!response.ok) return;
      
      const data = await response.json();
      const schedules = data.schedules || {};
      
      // Update Jira schedule dropdown
      const jiraSchedule = document.getElementById('jira-schedule');
      if (jiraSchedule && schedules.jira) {
        jiraSchedule.value = schedules.jira.display || 'disabled';
      }
      
      // Update Git schedule dropdown
      const gitSchedule = document.getElementById('git-schedule');
      if (gitSchedule && schedules.git) {
        gitSchedule.value = schedules.git.display || 'disabled';
      }

      // Update Links/URLs schedule dropdown
      const linksSchedule = document.getElementById('links-schedule');
      if (linksSchedule && schedules.links) {
        linksSchedule.value = schedules.links.display || 'disabled';
      }
    } catch (err) {
      console.warn('Failed to load source schedules:', err);
    }
  }

  async updateSourceSchedule(source, schedule) {
    try {
      const response = await fetch('/api/sources/schedules', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source, schedule })
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to update schedule');
      }
      
      console.log(`Updated ${source} schedule to: ${schedule}`);
      return true;
    } catch (err) {
      console.error('Failed to update schedule:', err);
      toast.error(`Failed to update schedule: ${err.message}`);
      return false;
    }
  }

  /**
   * Utility Functions (kept as fallbacks when archiUtils is not loaded)
   */
  isValidUrl(string) {
    if (archiUtils?.isValidUrl) {
      return archiUtils.isValidUrl(string);
    }
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;
    }
  }

  formatFileSize(bytes) {
    if (archiUtils?.formatSize) {
      return archiUtils.formatSize(bytes);
    }
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  formatDate(dateStr) {
    if (archiUtils?.formatRelativeTime) {
      return archiUtils.formatRelativeTime(dateStr);
    }
    if (!dateStr) return 'Unknown';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return Math.floor(diff / 60000) + ' min ago';
    if (diff < 86400000) return Math.floor(diff / 3600000) + ' hours ago';
    if (diff < 604800000) return Math.floor(diff / 86400000) + ' days ago';
    
    return date.toLocaleDateString();
  }

  getFileIcon(filename) {
    if (archiUtils?.getFileIcon) {
      return archiUtils.getFileIcon(filename);
    }
    const ext = filename.split('.').pop()?.toLowerCase();
    const iconMap = {
      pdf: '📄',
      md: '📝',
      txt: '📄',
      docx: '📄',
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
    };
    return iconMap[ext] || '📄';
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
}

// Export for use
if (typeof window !== 'undefined') {
  window.DataUploader = DataUploader;
}
