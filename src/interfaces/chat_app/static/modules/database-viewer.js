/**
 * DatabaseViewer - Database Viewer Module
 * 
 * Manages the database viewer UI for inspecting PostgreSQL tables,
 * running read-only queries, and viewing results.
 */

class DatabaseViewer {
  constructor() {
    this.tables = [];
    this.selectedTable = null;
    this.currentPage = 1;
    this.pageSize = 100;
    this.totalRows = 0;
    this.lastQueryTime = 0;
    
    // Predefined queries
    this.quickQueries = {
      stats: `SELECT 
  (SELECT COUNT(*) FROM documents) as documents,
  (SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL) as active_documents,
  (SELECT COUNT(*) FROM document_chunks) as chunks,
  (SELECT COUNT(*) FROM conversations) as conversations,
  (SELECT COUNT(*) FROM feedback) as feedback_count;`,
      
      recent_docs: `SELECT id, display_name, source_type, 
       pg_size_pretty(size_bytes) as size, created_at
FROM documents 
ORDER BY created_at DESC 
LIMIT 20;`,
      
      embedding_coverage: `SELECT d.id, d.display_name, 
       COUNT(c.id) as chunk_count
FROM documents d
LEFT JOIN document_chunks c ON c.document_id = d.id
GROUP BY d.id, d.display_name
ORDER BY chunk_count ASC
LIMIT 20;`,
      
      recent_chats: `SELECT cm.conversation_id, cm.title, cm.user_id,
       cm.created_at, cm.last_message_at,
       (SELECT COUNT(*) FROM conversations c WHERE c.conversation_id = cm.conversation_id) as message_count
FROM conversation_metadata cm
ORDER BY cm.last_message_at DESC
LIMIT 20;`,
      
      orphans: `SELECT c.id, LEFT(c.chunk_text, 100) as content_preview
FROM document_chunks c
LEFT JOIN documents d ON c.document_id = d.id
WHERE d.id IS NULL
LIMIT 20;`
    };
    
    this.init();
  }

  async init() {
    this.bindEvents();
    await this.loadTables();
  }

  /**
   * Event Bindings
   */
  bindEvents() {
    // Run query button
    const runBtn = document.getElementById('run-query-btn');
    if (runBtn) {
      runBtn.addEventListener('click', () => this.runQuery());
    }

    // Keyboard shortcut (Ctrl/Cmd + Enter)
    const editor = document.getElementById('sql-editor');
    if (editor) {
      editor.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
          e.preventDefault();
          this.runQuery();
        }
      });
    }

    // Format SQL button
    const formatBtn = document.getElementById('format-sql-btn');
    if (formatBtn) {
      formatBtn.addEventListener('click', () => this.formatSql());
    }

    // Export CSV button
    const exportBtn = document.getElementById('export-csv-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportCsv());
    }

    // Quick action buttons
    document.querySelectorAll('.quick-action-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const queryKey = btn.dataset.query;
        if (queryKey && this.quickQueries[queryKey]) {
          this.setQuery(this.quickQueries[queryKey]);
          this.runQuery();
        }
      });
    });

    // Pagination
    const prevBtn = document.getElementById('prev-page-btn');
    if (prevBtn) {
      prevBtn.addEventListener('click', () => this.prevPage());
    }

    const nextBtn = document.getElementById('next-page-btn');
    if (nextBtn) {
      nextBtn.addEventListener('click', () => this.nextPage());
    }
  }

  /**
   * Load Tables
   */
  async loadTables() {
    const tableList = document.getElementById('table-list');
    if (!tableList) return;

    try {
      const response = await fetch('/api/admin/database/tables');
      if (!response.ok) throw new Error('Failed to load tables');
      
      const data = await response.json();
      this.tables = data.tables || [];
      this.renderTableList();
    } catch (err) {
      console.error('Failed to load tables:', err);
      tableList.innerHTML = `
        <li class="table-list-error">
          Failed to load tables
        </li>
      `;
    }
  }

  renderTableList() {
    const tableList = document.getElementById('table-list');
    if (!tableList) return;

    if (!this.tables || this.tables.length === 0) {
      tableList.innerHTML = `<li class="table-list-empty">No tables found</li>`;
      return;
    }

    tableList.innerHTML = this.tables.map(table => `
      <li class="table-item${this.selectedTable === table.name ? ' selected' : ''}" 
          data-table="${this.escapeHtml(table.name)}">
        <span class="table-item-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="3" y1="9" x2="21" y2="9"></line>
            <line x1="9" y1="21" x2="9" y2="9"></line>
          </svg>
        </span>
        <span class="table-item-name">${this.escapeHtml(table.name)}</span>
        <span class="table-item-count">${this.formatNumber(table.row_count)}</span>
      </li>
    `).join('');

    // Add click handlers
    tableList.querySelectorAll('.table-item').forEach(item => {
      item.addEventListener('click', () => {
        const tableName = item.dataset.table;
        this.selectTable(tableName);
      });
    });
  }

  selectTable(tableName) {
    this.selectedTable = tableName;
    this.renderTableList();
    
    // Set a default query for the table
    this.setQuery(`SELECT * FROM ${tableName} LIMIT 100;`);
    this.runQuery();
  }

  /**
   * Query Execution
   */
  setQuery(sql) {
    const editor = document.getElementById('sql-editor');
    if (editor) {
      editor.value = sql;
    }
  }

  getQuery() {
    const editor = document.getElementById('sql-editor');
    return editor?.value?.trim() || '';
  }

  async runQuery() {
    const query = this.getQuery();
    if (!query) return;

    const runBtn = document.getElementById('run-query-btn');
    const resultsBody = document.getElementById('results-body');
    
    // Update UI state
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.innerHTML = `<span class="spinner"></span> Running...`;
    }

    if (resultsBody) {
      resultsBody.innerHTML = `
        <div class="results-placeholder">
          <span class="spinner"></span>
          <p>Executing query...</p>
        </div>
      `;
    }

    const startTime = performance.now();

    try {
      const response = await fetch('/api/admin/database/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query, 
          limit: this.pageSize,
          offset: (this.currentPage - 1) * this.pageSize
        })
      });

      const data = await response.json();
      this.lastQueryTime = Math.round(performance.now() - startTime);

      if (!response.ok) {
        throw new Error(data.error || 'Query failed');
      }

      this.renderResults(data);
    } catch (err) {
      console.error('Query error:', err);
      this.renderError(err.message);
    } finally {
      if (runBtn) {
        runBtn.disabled = false;
        runBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          Run
        `;
      }
    }
  }

  /**
   * Results Rendering
   */
  renderResults(data) {
    const resultsBody = document.getElementById('results-body');
    const resultsCount = document.getElementById('results-count');
    const resultsTime = document.getElementById('results-time');
    const exportBtn = document.getElementById('export-csv-btn');
    const resultsFooter = document.getElementById('results-footer');

    if (!resultsBody) return;

    const { columns, rows, row_count } = data;
    this.lastResults = data;

    // Update meta
    if (resultsCount) {
      resultsCount.textContent = `${row_count} rows`;
    }
    if (resultsTime) {
      resultsTime.textContent = `${this.lastQueryTime}ms`;
    }
    if (exportBtn) {
      exportBtn.disabled = row_count === 0;
    }

    // No results
    if (!rows || rows.length === 0) {
      resultsBody.innerHTML = `
        <div class="results-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <p>No results returned</p>
        </div>
      `;
      if (resultsFooter) resultsFooter.style.display = 'none';
      return;
    }

    // Build table
    const tableHtml = `
      <table class="results-table">
        <thead>
          <tr>
            ${columns.map(col => `<th>${this.escapeHtml(col)}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              ${columns.map((col, idx) => {
                const val = row[idx];
                if (val === null || val === undefined) {
                  return `<td class="null">NULL</td>`;
                }
                return `<td title="${this.escapeHtml(String(val))}">${this.escapeHtml(this.formatValue(val))}</td>`;
              }).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;

    resultsBody.innerHTML = tableHtml;

    // Show pagination if needed
    if (resultsFooter) {
      if (row_count >= this.pageSize) {
        resultsFooter.style.display = 'block';
        this.updatePagination();
      } else {
        resultsFooter.style.display = 'none';
      }
    }
  }

  renderError(message) {
    const resultsBody = document.getElementById('results-body');
    const resultsCount = document.getElementById('results-count');
    const resultsTime = document.getElementById('results-time');
    const resultsFooter = document.getElementById('results-footer');

    if (resultsCount) resultsCount.textContent = 'Error';
    if (resultsTime) resultsTime.textContent = `${this.lastQueryTime}ms`;
    if (resultsFooter) resultsFooter.style.display = 'none';

    if (resultsBody) {
      resultsBody.innerHTML = `
        <div class="results-error">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="15" y1="9" x2="9" y2="15"></line>
            <line x1="9" y1="9" x2="15" y2="15"></line>
          </svg>
          <p>Query Error</p>
          <div class="results-error-message">${this.escapeHtml(message)}</div>
        </div>
      `;
    }
  }

  /**
   * Pagination
   */
  updatePagination() {
    const prevBtn = document.getElementById('prev-page-btn');
    const nextBtn = document.getElementById('next-page-btn');
    const pageInfo = document.getElementById('pagination-info');

    if (prevBtn) prevBtn.disabled = this.currentPage <= 1;
    if (nextBtn) nextBtn.disabled = (this.lastResults?.count || 0) < this.pageSize;
    if (pageInfo) pageInfo.textContent = `Page ${this.currentPage}`;
  }

  prevPage() {
    if (this.currentPage > 1) {
      this.currentPage--;
      this.runQuery();
    }
  }

  nextPage() {
    this.currentPage++;
    this.runQuery();
  }

  /**
   * SQL Formatting
   */
  formatSql() {
    const editor = document.getElementById('sql-editor');
    if (!editor) return;

    // Basic SQL formatting
    let sql = editor.value;
    
    // Keywords to uppercase
    const keywords = [
      'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'ORDER BY', 'GROUP BY',
      'HAVING', 'LIMIT', 'OFFSET', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN',
      'INNER JOIN', 'ON', 'AS', 'DISTINCT', 'COUNT', 'SUM', 'AVG',
      'MIN', 'MAX', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER',
      'DROP', 'TABLE', 'INDEX', 'NULL', 'NOT NULL', 'PRIMARY KEY',
      'FOREIGN KEY', 'REFERENCES', 'CASCADE', 'SET', 'VALUES', 'INTO'
    ];

    keywords.forEach(kw => {
      const regex = new RegExp(`\\b${kw}\\b`, 'gi');
      sql = sql.replace(regex, kw);
    });

    // Add newlines before major clauses
    sql = sql.replace(/\s+(FROM|WHERE|AND|OR|ORDER BY|GROUP BY|HAVING|LIMIT|JOIN|LEFT JOIN|RIGHT JOIN|INNER JOIN)\s+/gi, 
      (match, clause) => `\n${clause.toUpperCase()} `);

    editor.value = sql.trim();
  }

  /**
   * CSV Export
   */
  exportCsv() {
    if (!this.lastResults || !this.lastResults.rows) return;

    const { columns, rows } = this.lastResults;
    
    // Build CSV content
    const csvRows = [
      columns.join(','),
      ...rows.map(row => 
        columns.map(col => {
          const val = row[col];
          if (val === null || val === undefined) return '';
          const str = String(val);
          // Escape quotes and wrap in quotes if needed
          if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replace(/"/g, '""')}"`;
          }
          return str;
        }).join(',')
      )
    ];

    const csvContent = csvRows.join('\n');
    
    // Download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `query_results_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  /**
   * Utility Functions
   */
  formatValue(val) {
    if (typeof val === 'object') {
      return JSON.stringify(val);
    }
    const str = String(val);
    // Truncate long values for display
    if (str.length > 100) {
      return str.substring(0, 100) + '...';
    }
    return str;
  }

  formatNumber(num) {
    if (num === null || num === undefined) return '?';
    return new Intl.NumberFormat().format(num);
  }

  escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Export for use
if (typeof window !== 'undefined') {
  window.DatabaseViewer = DatabaseViewer;
}
