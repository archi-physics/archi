/**
 * FileTree Module
 * 
 * Handles building and rendering hierarchical tree structures from flat document lists.
 * Supports local file trees, web page domain grouping, and ticket lists.
 */

class FileTree {
  constructor(options = {}) {
    this.options = {
      commonPrefixStrip: options.commonPrefixStrip ?? true,
      defaultExpanded: options.defaultExpanded ?? 1, // Expand first N levels
      persistState: options.persistState ?? true,
      storageKey: options.storageKey ?? 'archi-filetree-state',
      ...options
    };
    
    this.expandedState = this.loadState();
    this.onSelect = options.onSelect || (() => {});
    this.onToggle = options.onToggle || (() => {});
  }

  /**
   * Build tree structure from flat document list
   * @param {Array} documents - List of document objects
   * @returns {{ localFiles: Object, webPages: Object, tickets: Array, gitRepos: Object }}
   */
  buildTrees(documents) {
    const explicitTypes = new Set(['local_files', 'web', 'ticket', 'git', 'sso']);
    const localFiles = documents.filter(d => d.source_type === 'local_files');
    const webPages = documents.filter(d => d.source_type === 'web');
    const tickets = documents.filter(d => d.source_type === 'ticket');
    const gitFiles = documents.filter(d => d.source_type === 'git');
    const ssoPages = documents.filter(d => d.source_type === 'sso');
    const otherSources = documents.filter(d => !explicitTypes.has(d.source_type));
    
    return {
      localFiles: this.buildFileTree(localFiles),
      webPages: this.buildDomainTree(webPages),
      tickets: tickets,
      gitRepos: this.buildGitRepoTree(gitFiles),
      ssoPages: this.buildDomainTree(ssoPages),
      otherSources: otherSources
    };
  }

  /**
   * Build repository-grouped tree for git files
   */
  buildGitRepoTree(documents) {
    const tree = { name: 'root', children: {}, files: [], path: '' };
    
    for (const doc of documents) {
      try {
        const url = doc.url || '';
        // Extract repo name from GitHub/GitLab URLs
        // e.g., https://github.com/org/repo/blob/main/path/file.py -> org/repo
        const match = url.match(/(?:github\.com|gitlab\.com)\/([^\/]+\/[^\/]+)/);
        const repoName = match ? match[1] : 'unknown';
        
        if (!tree.children[repoName]) {
          tree.children[repoName] = {
            name: repoName,
            children: {},
            files: [],
            path: repoName
          };
        }
        
        // Extract file path within repo
        const pathMatch = url.match(/\/blob\/[^\/]+\/(.+)$/);
        const filePath = pathMatch ? pathMatch[1] : doc.display_name;
        
        tree.children[repoName].files.push({
          ...doc,
          name: filePath,
          repoName: repoName
        });
      } catch (e) {
        // Invalid URL, add to root
        tree.files.push({
          ...doc,
          name: doc.display_name || doc.url
        });
      }
    }
    
    return tree;
  }

  /**
   * Build hierarchical file tree from local file documents
   */
  buildFileTree(documents) {
    if (documents.length === 0) {
      return { name: 'root', children: {}, files: [], path: '' };
    }
    
    // Extract paths and find common prefix
    const paths = documents.map(doc => {
      const url = doc.url || doc.display_name;
      return url.replace(/^file:\/\//, '');
    });
    
    const commonPrefix = this.options.commonPrefixStrip 
      ? this.findCommonPrefix(paths) 
      : '';
    
    // Build tree
    const tree = { name: 'root', children: {}, files: [], path: '' };
    
    for (const doc of documents) {
      const fullPath = (doc.url || doc.display_name).replace(/^file:\/\//, '');
      const relativePath = fullPath.slice(commonPrefix.length).replace(/^\//, '');
      const parts = relativePath.split('/').filter(Boolean);
      
      let node = tree;
      let currentPath = '';
      
      for (let i = 0; i < parts.length - 1; i++) {
        const part = parts[i];
        currentPath += '/' + part;
        
        if (!node.children[part]) {
          node.children[part] = { 
            name: part, 
            children: {}, 
            files: [],
            path: currentPath 
          };
        }
        node = node.children[part];
      }
      
      const fileName = parts[parts.length - 1] || doc.display_name;
      node.files.push({
        ...doc,
        name: fileName,
        relativePath: relativePath
      });
    }
    
    return tree;
  }

  /**
   * Build domain-grouped tree for web pages
   */
  buildDomainTree(documents) {
    const tree = { name: 'root', children: {}, files: [], path: '' };
    
    for (const doc of documents) {
      try {
        const url = new URL(doc.url);
        const domain = url.hostname;
        const path = url.pathname;
        
        if (!tree.children[domain]) {
          tree.children[domain] = {
            name: domain,
            children: {},
            files: [],
            path: domain
          };
        }
        
        tree.children[domain].files.push({
          ...doc,
          name: path || '/',
          domain: domain
        });
      } catch (e) {
        // Invalid URL, add to root
        tree.files.push({
          ...doc,
          name: doc.display_name || doc.url
        });
      }
    }
    
    return tree;
  }

  /**
   * Find common prefix among paths
   */
  findCommonPrefix(paths) {
    if (paths.length === 0) return '';
    if (paths.length === 1) {
      // For single file, return directory part
      const parts = paths[0].split('/');
      parts.pop();
      return parts.join('/');
    }
    
    const sorted = paths.slice().sort();
    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    
    let i = 0;
    while (i < first.length && first[i] === last[i]) {
      i++;
    }
    
    // Find last slash to get directory boundary
    const prefix = first.slice(0, i);
    const lastSlash = prefix.lastIndexOf('/');
    return lastSlash > 0 ? prefix.slice(0, lastSlash) : '';
  }

  /**
   * Count total files in a tree node (recursive)
   */
  countFiles(node) {
    let count = node.files ? node.files.length : 0;
    if (node.children) {
      for (const child of Object.values(node.children)) {
        count += this.countFiles(child);
      }
    }
    return count;
  }

  /**
   * Get all file hashes in a tree node
   */
  getAllHashes(node) {
    let hashes = node.files ? node.files.map(f => f.hash) : [];
    if (node.children) {
      for (const child of Object.values(node.children)) {
        hashes = hashes.concat(this.getAllHashes(child));
      }
    }
    return hashes;
  }

  /**
   * Check if a node is expanded
   */
  isExpanded(path, depth = 0) {
    if (this.expandedState[path] !== undefined) {
      return this.expandedState[path];
    }
    return depth < this.options.defaultExpanded;
  }

  /**
   * Toggle expansion state
   */
  toggleExpanded(path) {
    this.expandedState[path] = !this.isExpanded(path);
    this.saveState();
  }

  /**
   * Expand all nodes
   */
  expandAll(tree, path = '') {
    this.expandedState[path] = true;
    if (tree.children) {
      for (const [name, child] of Object.entries(tree.children)) {
        const childPath = path + '/' + name;
        this.expandAll(child, childPath);
      }
    }
    this.saveState();
  }

  /**
   * Collapse all nodes
   */
  collapseAll(tree, path = '') {
    this.expandedState[path] = false;
    if (tree.children) {
      for (const [name, child] of Object.entries(tree.children)) {
        const childPath = path + '/' + name;
        this.collapseAll(child, childPath);
      }
    }
    this.saveState();
  }

  /**
   * Save expansion state to localStorage
   */
  saveState() {
    if (!this.options.persistState) return;
    try {
      localStorage.setItem(this.options.storageKey, JSON.stringify(this.expandedState));
    } catch (e) {
      console.warn('Failed to save tree state:', e);
    }
  }

  /**
   * Load expansion state from localStorage
   */
  loadState() {
    if (!this.options.persistState) return {};
    try {
      const saved = localStorage.getItem(this.options.storageKey);
      return saved ? JSON.parse(saved) : {};
    } catch (e) {
      return {};
    }
  }

  /**
   * Render a complete source category
   * @param {string} type - 'local_files', 'web', 'ticket', 'git', 'sso', or 'other'
   * @param {Object} tree - Tree structure for local/web/git, or array for tickets
   * @param {string} selectedHash - Currently selected document hash
   * @param {Object} options - Rendering options
   */
  renderCategory(type, tree, selectedHash = null, options = {}) {
    const configs = {
      'local_files': { 
        icon: this.icons.folder, 
        label: 'Local Files',
        emptyMessage: 'No local files ingested'
      },
      'web': { 
        icon: this.icons.globe, 
        label: 'Web Pages',
        emptyMessage: 'No web pages ingested'
      },
      'ticket': { 
        icon: this.icons.ticket, 
        label: 'Tickets',
        emptyMessage: 'No tickets ingested'
      },
      'git': { 
        icon: this.icons.git, 
        label: 'Git Repos',
        emptyMessage: 'No git repositories ingested'
      },
      'sso': {
        icon: this.icons.globe,
        label: 'SSO Pages',
        emptyMessage: 'No SSO pages ingested'
      },
      'other': {
        icon: this.icons.file,
        label: 'Other Sources',
        emptyMessage: 'No other sources ingested'
      }
    };
    
    const config = configs[type];
    const isFlatList = type === 'ticket' || type === 'other';
    const loadedCount = isFlatList ? tree.length : this.countFiles(tree);
    const count = Number.isFinite(options.countOverride) ? options.countOverride : loadedCount;
    const showHydrating = Boolean(options.hydrating) && loadedCount < count;
    
    if (count === 0) {
      return ''; // Don't render empty categories
    }
    
    const categoryPath = `category-${type}`;
    const isExpanded = this.isExpanded(categoryPath, 0);
    
    let contentHtml;
    if (isFlatList) {
      contentHtml = this.renderFlatList(tree, selectedHash, type);
    } else {
      contentHtml = this.renderTreeNode(tree, type, 1, selectedHash);
    }

    if (!contentHtml) {
      if (showHydrating) {
        contentHtml = '<div class="tree-empty">Loading documents...</div>';
      } else {
        contentHtml = `<div class="tree-empty">${config.emptyMessage}</div>`;
      }
    }
    
    return `
      <div class="tree-category" data-type="${type}">
        <div class="tree-category-header ${isExpanded ? 'expanded' : ''}" 
             onclick="fileTree.toggleCategory('${categoryPath}')">
          <span class="tree-toggle">${isExpanded ? this.icons.chevronDown : this.icons.chevronRight}</span>
          ${config.icon}
          <span class="tree-category-name">${config.label}</span>
          <span class="tree-category-count">${count}</span>
        </div>
        <div class="tree-category-content ${isExpanded ? '' : 'collapsed'}">
          ${contentHtml}
        </div>
      </div>
    `;
  }

  /**
   * Render a tree node (folder) and its children
   */
  renderTreeNode(node, type, depth = 0, selectedHash = null) {
    const hasChildren = node.children && Object.keys(node.children).length > 0;
    const hasFiles = node.files && node.files.length > 0;
    
    if (!hasChildren && !hasFiles) {
      return '';
    }
    
    let html = '';
    
    // Sort and render child folders
    if (hasChildren) {
      const sortedChildren = Object.entries(node.children).sort((a, b) => 
        a[0].localeCompare(b[0])
      );
      
      for (const [name, child] of sortedChildren) {
        const childPath = node.path + '/' + name;
        const isExpanded = this.isExpanded(childPath, depth);
        const fileCount = this.countFiles(child);
        
        html += `
          <div class="tree-folder" data-path="${this.escapeAttr(childPath)}">
            <div class="tree-folder-header" style="padding-left: ${depth * 16}px"
                 onclick="fileTree.toggleFolder('${this.escapeAttr(childPath)}')">
              <span class="tree-toggle">${isExpanded ? this.icons.chevronDown : this.icons.chevronRight}</span>
              ${this.icons.folderIcon}
              <span class="tree-folder-name">${this.escapeHtml(name)}</span>
              <span class="tree-folder-count">${fileCount}</span>
            </div>
            <div class="tree-folder-content ${isExpanded ? '' : 'collapsed'}">
              ${this.renderTreeNode(child, type, depth + 1, selectedHash)}
            </div>
          </div>
        `;
      }
    }
    
    // Render files in this folder
    if (hasFiles) {
      const sortedFiles = node.files.slice().sort((a, b) => 
        a.name.localeCompare(b.name)
      );
      
      for (const file of sortedFiles) {
        const isSelected = file.hash === selectedHash;
        const fileIcon = type === 'web' ? this.icons.link : this.getFileIcon(file.name);
        const statusClass = file.ingestion_status || 'pending';
        const statusTitle = statusClass.charAt(0).toUpperCase() + statusClass.slice(1);
        
        html += `
          <div class="tree-file ${isSelected ? 'selected' : ''}"
               data-hash="${file.hash}"
               style="padding-left: ${depth * 16}px"
               onclick="fileTree.selectFile('${file.hash}')">
            ${fileIcon}
            <span class="tree-file-name" title="${this.escapeAttr(file.name)}">${this.escapeHtml(file.name)}</span>
            <span class="tree-status-dot ${statusClass}" title="${statusTitle}"></span>
          </div>
        `;
      }
    }
    
    return html;
  }

  /**
   * Render ticket list (flat, not tree)
   */
  renderTicketList(tickets, selectedHash = null) {
    if (tickets.length === 0) {
      return '<div class="tree-empty">No tickets</div>';
    }
    
    return tickets.map(ticket => {
      const isSelected = ticket.hash === selectedHash;
      return `
        <div class="tree-file ticket ${isSelected ? 'selected' : ''}"
             data-hash="${ticket.hash}"
             onclick="fileTree.selectFile('${ticket.hash}')">
          ${this.icons.ticket}
          <span class="tree-file-name" title="${this.escapeAttr(ticket.display_name)}">
            ${this.escapeHtml(ticket.display_name)}
          </span>
        </div>
      `;
    }).join('');
  }

  /**
   * Render flat list for categories without hierarchy
   */
  renderFlatList(documents, selectedHash = null, type = 'other') {
    if (!documents || documents.length === 0) {
      return '<div class="tree-empty">No documents</div>';
    }

    const icon = type === 'ticket' ? this.icons.ticket : this.icons.file;
    const sorted = documents.slice().sort((a, b) =>
      (a.display_name || '').localeCompare(b.display_name || '')
    );

    return sorted.map((doc) => {
      const isSelected = doc.hash === selectedHash;
      const name = doc.display_name || doc.url || doc.hash;
      return `
        <div class="tree-file ${type} ${isSelected ? 'selected' : ''}"
             data-hash="${doc.hash}"
             onclick="fileTree.selectFile('${doc.hash}')">
          ${icon}
          <span class="tree-file-name" title="${this.escapeAttr(name)}">
            ${this.escapeHtml(name)}
          </span>
        </div>
      `;
    }).join('');
  }

  /**
   * Get appropriate icon for file type
   */
  getFileIcon(filename) {
    const ext = (filename.split('.').pop() || '').toLowerCase();
    
    const iconMap = {
      'md': this.icons.markdown,
      'markdown': this.icons.markdown,
      'py': this.icons.python,
      'js': this.icons.javascript,
      'ts': this.icons.javascript,
      'json': this.icons.json,
      'yaml': this.icons.yaml,
      'yml': this.icons.yaml,
      'html': this.icons.html,
      'css': this.icons.css,
      'sql': this.icons.database,
      'txt': this.icons.file,
    };
    
    return iconMap[ext] || this.icons.file;
  }

  /**
   * Toggle category expansion
   */
  toggleCategory(path) {
    this.toggleExpanded(path);
    this.onToggle(path);
  }

  /**
   * Toggle folder expansion
   */
  toggleFolder(path) {
    this.toggleExpanded(path);
    this.onToggle(path);
  }

  /**
   * Handle file selection
   */
  selectFile(hash) {
    this.onSelect(hash);
  }

  /**
   * Escape HTML
   */
  escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Escape attribute value
   */
  escapeAttr(text) {
    return this.escapeHtml(text).replace(/"/g, '&quot;');
  }

  /**
   * SVG Icons
   */
  get icons() {
    return {
      chevronRight: '<svg class="icon-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>',
      
      chevronDown: '<svg class="icon-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>',
      
      folder: '<svg class="icon-folder" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
      
      folderIcon: '<svg class="icon-folder-small" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
      
      globe: '<svg class="icon-globe" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
      
      ticket: '<svg class="icon-ticket" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 5v2m0 4v2m0 4v2M5 5a2 2 0 00-2 2v3a2 2 0 110 4v3a2 2 0 002 2h14a2 2 0 002-2v-3a2 2 0 110-4V7a2 2 0 00-2-2H5z"/></svg>',
      
      file: '<svg class="icon-file" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
      
      markdown: '<svg class="icon-markdown" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M7 13l2 2 4-4"/></svg>',
      
      python: '<svg class="icon-python" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6v4m0 4v4"/></svg>',
      
      javascript: '<svg class="icon-js" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 8v8m4-6v4a2 2 0 1 1-4 0"/></svg>',
      
      json: '<svg class="icon-json" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h4m-4 6h6m-6 6h4m12-12h-4m4 6h-6m6 6h-4"/></svg>',
      
      yaml: '<svg class="icon-yaml" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4l4 8v8m12-16l-4 8v8m-4-16v16"/></svg>',
      
      html: '<svg class="icon-html" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
      
      css: '<svg class="icon-css" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 3l1.5 15L12 21l6.5-3L20 3z"/></svg>',
      
      database: '<svg class="icon-db" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
      
      link: '<svg class="icon-link" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>',
      
      git: '<svg class="icon-git" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 3v6m0 6v6"/><path d="M3 12h6m6 0h6"/></svg>',
    };
  }
}

// Export - will be instantiated by DataViewer
