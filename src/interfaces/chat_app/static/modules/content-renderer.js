/**
 * ContentRenderer Module
 * 
 * Handles intelligent content rendering based on file type detection.
 * Supports markdown, code syntax highlighting, and plain text.
 */

class ContentRenderer {
  constructor(options = {}) {
    this.options = {
      lineNumbers: options.lineNumbers ?? true,
      wordWrap: options.wordWrap ?? false,
      maxPreviewLength: options.maxPreviewLength ?? 50000,
      ...options
    };
    
    // Code extension to language mapping
    this.codeExtensions = {
      'py': 'python',
      'js': 'javascript',
      'jsx': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'yaml': 'yaml',
      'yml': 'yaml',
      'json': 'json',
      'sh': 'bash',
      'bash': 'bash',
      'zsh': 'bash',
      'sql': 'sql',
      'html': 'xml',
      'htm': 'xml',
      'xml': 'xml',
      'css': 'css',
      'scss': 'scss',
      'less': 'less',
      'java': 'java',
      'go': 'go',
      'rs': 'rust',
      'rb': 'ruby',
      'php': 'php',
      'c': 'c',
      'cpp': 'cpp',
      'h': 'c',
      'hpp': 'cpp',
      'cs': 'csharp',
      'swift': 'swift',
      'kt': 'kotlin',
      'scala': 'scala',
      'r': 'r',
      'lua': 'lua',
      'pl': 'perl',
      'dockerfile': 'dockerfile',
      'makefile': 'makefile',
      'toml': 'toml',
      'ini': 'ini',
      'cfg': 'ini',
      'conf': 'ini',
      'env': 'bash',
      'gitignore': 'bash',
      'txt': 'plaintext',
    };
    
    this.configureMarked();
  }

  /**
   * Configure marked.js for GitHub Flavored Markdown
   */
  configureMarked() {
    if (typeof marked === 'undefined') {
      console.warn('marked.js not loaded');
      return;
    }

    marked.setOptions({
      gfm: true,
      breaks: true,
      highlight: (code, lang) => {
        if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
          try {
            return hljs.highlight(code, { language: lang }).value;
          } catch (e) {
            console.warn('Highlight error:', e);
          }
        }
        return this.escapeHtml(code);
      }
    });
  }

  /**
   * Detect content type from filename and content
   * @param {string} filename - The file name or path
   * @param {string} content - The file content (optional, for heuristics)
   * @returns {{ type: 'markdown'|'code'|'text', language?: string, icon: string }}
   */
  detectContentType(filename, content = '') {
    const ext = this.getExtension(filename);
    const baseName = filename.split('/').pop().toLowerCase();
    
    // Check for markdown
    if (ext === 'md' || ext === 'markdown' || ext === 'mdx') {
      return { type: 'markdown', icon: '📝' };
    }
    
    // Check for known code extensions
    if (this.codeExtensions[ext]) {
      return { 
        type: 'code', 
        language: this.codeExtensions[ext],
        icon: '💻'
      };
    }
    
    // Check for special filenames
    const specialFiles = {
      'dockerfile': 'dockerfile',
      'makefile': 'makefile',
      'gemfile': 'ruby',
      'rakefile': 'ruby',
      'procfile': 'yaml',
      'vagrantfile': 'ruby',
      '.gitignore': 'bash',
      '.dockerignore': 'bash',
      '.env': 'bash',
      '.env.example': 'bash',
    };
    
    if (specialFiles[baseName]) {
      return {
        type: 'code',
        language: specialFiles[baseName],
        icon: '⚙️'
      };
    }
    
    // Heuristic: content starts with # likely markdown
    if (content && content.trim().startsWith('#') && !content.trim().startsWith('#!')) {
      return { type: 'markdown', icon: '📝' };
    }
    
    // Default to plain text
    return { type: 'text', icon: '📄' };
  }

  /**
   * Get file extension from filename
   */
  getExtension(filename) {
    if (!filename) return '';
    const parts = filename.split('.');
    if (parts.length < 2) return '';
    return parts.pop().toLowerCase();
  }

  /**
   * Render content based on detected type
   * @param {string} content - The content to render
   * @param {string} filename - The filename for type detection
   * @param {Object} options - Additional options
   * @param {Array} options.chunks - Chunk data for visualization
   * @param {boolean} options.showChunks - Whether to show chunk boundaries (default: false)
   * @returns {{ html: string, type: string, language?: string }}
   */
  render(content, filename, options = {}) {
    const contentType = this.detectContentType(filename, content);
    const { chunks = [], showChunks = false } = options;
    let html;
    
    switch (contentType.type) {
      case 'markdown':
        html = this.renderMarkdown(content, showChunks ? chunks : null);
        break;
      case 'code':
        html = this.renderCode(content, contentType.language, showChunks ? chunks : null);
        break;
      default:
        html = this.renderText(content, showChunks ? chunks : null);
    }
    
    return {
      html,
      type: contentType.type,
      language: contentType.language,
      icon: contentType.icon
    };
  }

  /**
   * Scroll to a specific chunk
   */
  scrollToChunk(index) {
    const chunk = document.querySelector(`[data-chunk="${index}"]`);
    if (chunk) {
      chunk.scrollIntoView({ behavior: 'smooth', block: 'start' });
      chunk.classList.add('chunk-flash');
      setTimeout(() => chunk.classList.remove('chunk-flash'), 1000);
    }
  }

  /**
   * Render markdown content with optional chunk highlighting
   */
  renderMarkdown(content, chunks = null) {
    if (typeof marked === 'undefined') {
      return this.renderText(content, chunks);
    }
    
    try {
      // If chunks, render with background highlighting
      if (chunks && chunks.length > 0) {
        return this.renderMarkdownWithChunks(content, chunks);
      }
      
      const html = marked.parse(content);
      return `<div class="markdown-content">${html}</div>`;
    } catch (e) {
      console.error('Markdown render error:', e);
      return this.renderText(content, chunks);
    }
  }

  /**
   * Render markdown with chunk highlighting
   * Each chunk gets a colored background highlight
   */
  renderMarkdownWithChunks(content, chunks) {
    const colors = [
      { bg: 'rgba(59, 130, 246, 0.12)', border: '#3b82f6' },   // blue
      { bg: 'rgba(16, 185, 129, 0.12)', border: '#10b981' },   // green
      { bg: 'rgba(245, 158, 11, 0.12)', border: '#f59e0b' },   // amber
      { bg: 'rgba(239, 68, 68, 0.12)', border: '#ef4444' },    // red
      { bg: 'rgba(139, 92, 246, 0.12)', border: '#8b5cf6' },   // purple
      { bg: 'rgba(236, 72, 153, 0.12)', border: '#ec4899' },   // pink
      { bg: 'rgba(6, 182, 212, 0.12)', border: '#06b6d4' },    // cyan
      { bg: 'rgba(132, 204, 22, 0.12)', border: '#84cc16' },   // lime
    ];
    
    // Build chunk navigation bar
    let navHtml = '<div class="chunk-indicators">';
    for (let i = 0; i < chunks.length; i++) {
      const color = colors[i % colors.length];
      navHtml += `<button class="chunk-indicator" style="--c: ${color.border}" onclick="contentRenderer.scrollToChunk(${i})" title="Jump to Chunk ${i}">${i}</button>`;
    }
    navHtml += '</div>';
    
    // Render each chunk with highlighted background
    let chunksHtml = '';
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const color = colors[i % colors.length];
      const chunkText = chunk.text || '';
      
      // Render this chunk's markdown
      const renderedMarkdown = marked.parse(chunkText);
      
      // Wrap with highlighted section - background color shows the chunk
      chunksHtml += `<section class="chunk-section" data-chunk="${i}" style="--chunk-bg: ${color.bg}; --chunk-border: ${color.border}">
          <span class="chunk-badge">${i}</span>
          ${renderedMarkdown}
        </section>`;
    }
    
    return `<div class="content-with-chunks">${navHtml}<div class="markdown-content chunked">${chunksHtml}</div></div>`;
  }

  /**
   * Render code/text with chunk boundaries (simple version for non-markdown)
   */
  renderWithChunkBoundaries(content, chunks, type) {
    const colors = [
      { bg: 'rgba(59, 130, 246, 0.12)', border: '#3b82f6' },
      { bg: 'rgba(16, 185, 129, 0.12)', border: '#10b981' },
      { bg: 'rgba(245, 158, 11, 0.12)', border: '#f59e0b' },
      { bg: 'rgba(239, 68, 68, 0.12)', border: '#ef4444' },
      { bg: 'rgba(139, 92, 246, 0.12)', border: '#8b5cf6' },
      { bg: 'rgba(236, 72, 153, 0.12)', border: '#ec4899' },
      { bg: 'rgba(6, 182, 212, 0.12)', border: '#06b6d4' },
      { bg: 'rgba(132, 204, 22, 0.12)', border: '#84cc16' },
    ];

    let navHtml = '<div class="chunk-indicators">';
    for (let i = 0; i < chunks.length; i++) {
      const color = colors[i % colors.length];
      navHtml += `<button class="chunk-indicator" style="--c: ${color.border}" onclick="contentRenderer.scrollToChunk(${i})" title="Jump to Chunk ${i}">${i}</button>`;
    }
    navHtml += '</div>';

    let chunksHtml = '';
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const color = colors[i % colors.length];
      const chunkText = this.escapeHtml(chunk.text || '');
      
      chunksHtml += `<section class="chunk-section" data-chunk="${i}" style="--chunk-bg: ${color.bg}; --chunk-border: ${color.border}">
          <span class="chunk-badge">${i}</span>
          <pre>${chunkText}</pre>
        </section>`;
    }

    return `<div class="content-with-chunks">${navHtml}<div class="text-content chunked">${chunksHtml}</div></div>`;
  }

  /**
   * Render code with syntax highlighting
   */
  renderCode(content, language, chunks = null) {
    if (chunks && chunks.length > 0) {
      return this.renderWithChunkBoundaries(content, chunks, 'code');
    }
    
    let highlighted;
    
    if (typeof hljs !== 'undefined' && language && hljs.getLanguage(language)) {
      try {
        highlighted = hljs.highlight(content, { language }).value;
      } catch (e) {
        highlighted = this.escapeHtml(content);
      }
    } else {
      highlighted = this.escapeHtml(content);
    }
    
    const lines = highlighted.split('\n');
    const lineNumbersHtml = this.options.lineNumbers 
      ? lines.map((_, i) => `<span class="line-number">${i + 1}</span>`).join('\n')
      : '';
    
    const codeHtml = lines.map((line, i) => 
      `<span class="code-line" data-line="${i + 1}">${line || ' '}</span>`
    ).join('\n');
    
    const wrapClass = this.options.wordWrap ? 'word-wrap' : '';
    
    return `
      <div class="code-content ${wrapClass}">
        <div class="code-header">
          <span class="code-language">${language || 'text'}</span>
          <button class="copy-btn" onclick="contentRenderer.copyCode(this)" title="Copy code">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
        </div>
        <div class="code-body">
          ${this.options.lineNumbers ? `<div class="line-numbers">${lineNumbersHtml}</div>` : ''}
          <pre class="code-pre"><code class="hljs language-${language || 'text'}">${codeHtml}</code></pre>
        </div>
      </div>
    `;
  }

  /**
   * Render plain text with preserved whitespace
   */
  renderText(content, chunks = null) {
    if (chunks && chunks.length > 0) {
      return this.renderWithChunkBoundaries(content, chunks, 'text');
    }
    const escaped = this.escapeHtml(content);
    return `<div class="text-content"><pre class="text-pre">${escaped}</pre></div>`;
  }

  /**
   * Escape HTML special characters
   */
  escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Copy code to clipboard
   */
  async copyCode(button) {
    const codeBlock = button.closest('.code-content');
    const code = codeBlock.querySelector('code').textContent;
    
    try {
      await navigator.clipboard.writeText(code);
      button.classList.add('copied');
      button.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      `;
      
      setTimeout(() => {
        button.classList.remove('copied');
        button.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
        `;
      }, 2000);
    } catch (e) {
      console.error('Copy failed:', e);
    }
  }
}

// Export singleton instance
const contentRenderer = new ContentRenderer();
