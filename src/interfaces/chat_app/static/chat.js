/* =============================================================================
   archi Chat UI - Professional AI Assistant Interface
   Version: 2.0.0
   
   Modular vanilla JavaScript chat application.
   No framework dependencies - just clean, readable code.
   ============================================================================= */

// =============================================================================
// Constants & Configuration
// =============================================================================

const CONFIG = {
  STORAGE_KEYS: {
    CLIENT_ID: 'archi_client_id',
    ACTIVE_CONVERSATION: 'archi_active_conversation_id',
    AB_WARNING_DISMISSED: 'archi_ab_warning_dismissed',
    TRACE_VERBOSE_MODE: 'archi_trace_verbose_mode',
    SELECTED_PROVIDER: 'archi_selected_provider',
    SELECTED_MODEL: 'archi_selected_model',
    SELECTED_MODEL_CUSTOM: 'archi_selected_model_custom',
  },
  ENDPOINTS: {
    STREAM: '/api/get_chat_response_stream',
    CONFIGS: '/api/get_configs',
    CONVERSATIONS: '/api/list_conversations',
    LOAD_CONVERSATION: '/api/load_conversation',
    NEW_CONVERSATION: '/api/new_conversation',
    DELETE_CONVERSATION: '/api/delete_conversation',
    AB_PREFERENCE: '/api/ab/preference',
    AB_PENDING: '/api/ab/pending',
    AB_POOL: '/api/ab/pool',
    AB_DECISION: '/api/ab/decision',
    AB_POOL_SET: '/api/ab/pool/set',
    AB_POOL_DISABLE: '/api/ab/pool/disable',
    AB_COMPARE: '/api/ab/compare',
    AB_METRICS: '/api/ab/metrics',
    TRACE_GET: '/api/trace',
    CANCEL_STREAM: '/api/cancel_stream',
    PROVIDERS: '/api/providers',
    PROVIDER_MODELS: '/api/providers/models',
    VALIDATE_PROVIDER: '/api/providers/validate',
    PROVIDER_KEYS: '/api/providers/keys',
    SET_PROVIDER_KEY: '/api/providers/keys/set',
    CLEAR_PROVIDER_KEY: '/api/providers/keys/clear',
    PIPELINE_DEFAULT_MODEL: '/api/pipeline/default_model',
    AGENT_INFO: '/api/agent/info',
    AGENT_TEMPLATE: '/api/agents/template',
    AGENT_SAVE: '/api/agents',
    AGENTS_LIST: '/api/agents/list',
    AGENT_SPEC: '/api/agents/spec',
    AGENT_ACTIVE: '/api/agents/active',
    USER_ME: '/api/users/me',
    USER_PREFERENCES: '/api/users/me/preferences',
    LIKE: '/api/like',
    DISLIKE: '/api/dislike',
    TEXT_FEEDBACK: '/api/text_feedback',
    PLAYBOOKS: '/api/playbooks',
  },
  STREAMING: {
    TIMEOUT: 600000, // 10 minutes
  },
  TRACE: {
    MAX_TOOL_OUTPUT_PREVIEW: 500,
    AUTO_COLLAPSE_TOOL_COUNT: 5,
  },
  MESSAGES: {
    CLIENT_TIMEOUT: "client timeout; the agent wasn't able to find satisfactory information to respond to the query within the time limit set by the administrator.",
  },
};

// =============================================================================
// Utility Functions
// =============================================================================

const Utils = {
  /**
   * Generate a UUID v4
   */
  generateId() {
    if (crypto?.randomUUID) {
      return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  },

  /**
   * Escape HTML to prevent XSS
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  normalizeAgentName(name) {
    if (!name) return name;
    return name.replace(/^name:\s*/i, '').trim();
  },

  /**
   * Escape a string for use inside an HTML attribute (e.g. onclick)
   */
  escapeAttr(text) {
    if (text == null) return '';
    return String(text).replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  },

  /**
   * Format a date for display
   */
  formatDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return '';
    
    const now = new Date();
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  },

  /**
   * Group conversations by date
   */
  groupByDate(conversations) {
    const groups = { Today: [], Yesterday: [], 'Previous 7 Days': [], Older: [] };
    const now = new Date();
    
    conversations.forEach((conv) => {
      const date = new Date(conv.last_message_at || conv.created_at);
      const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
      
      if (diffDays === 0) groups['Today'].push(conv);
      else if (diffDays === 1) groups['Yesterday'].push(conv);
      else if (diffDays < 7) groups['Previous 7 Days'].push(conv);
      else groups['Older'].push(conv);
    });
    
    return groups;
  },

  /**
   * Debounce function calls
   */
  debounce(fn, delay) {
    let timeout;
    return (...args) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => fn(...args), delay);
    };
  },

  /**
   * Format duration in ms to human-readable string
   * @param {number} ms - Duration in milliseconds
   * @returns {string} - Formatted duration (e.g., "850ms", "2.3s", "1m 5s")
   */
  formatDuration(ms) {
    if (ms == null || isNaN(ms)) return '';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.round((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  },
};

// =============================================================================
// Storage Manager
// =============================================================================

const Storage = {
  getClientId() {
    let id = localStorage.getItem(CONFIG.STORAGE_KEYS.CLIENT_ID);
    if (!id) {
      id = Utils.generateId();
      localStorage.setItem(CONFIG.STORAGE_KEYS.CLIENT_ID, id);
    }
    return id;
  },

  getActiveConversationId() {
    const stored = localStorage.getItem(CONFIG.STORAGE_KEYS.ACTIVE_CONVERSATION);
    return stored ? Number(stored) : null;
  },

  setActiveConversationId(id) {
    if (id === null || id === undefined) {
      localStorage.removeItem(CONFIG.STORAGE_KEYS.ACTIVE_CONVERSATION);
    } else {
      localStorage.setItem(CONFIG.STORAGE_KEYS.ACTIVE_CONVERSATION, String(id));
    }
  },
};

// =============================================================================
// API Client
// =============================================================================

const API = {
  clientId: Storage.getClientId(),

  async fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    
    if (response.status === 401) {
      window.location.href = '/';
      return null;
    }
    
    const data = await response.json().catch(() => null);
    
    if (!response.ok) {
      throw new Error(data?.error || `Request failed (${response.status})`);
    }
    
    return data;
  },

  /**
   * Shared NDJSON reader: reads a fetch Response body and yields parsed JSON objects.
   * Properly flushes any remaining buffer content after the stream ends.
   */
  async *_readNDJSON(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            yield JSON.parse(trimmed);
          } catch (e) {
            console.warn('Failed to parse NDJSON line:', trimmed);
          }
        }
      }
      // Flush remaining buffer after stream ends
      if (buffer.trim()) {
        try {
          yield JSON.parse(buffer.trim());
        } catch (e) {
          console.warn('Failed to parse final NDJSON line:', buffer.trim());
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  async getConfigs() {
    return this.fetchJson(CONFIG.ENDPOINTS.CONFIGS);
  },

  async getConversations(limit = 100) {
    const url = `${CONFIG.ENDPOINTS.CONVERSATIONS}?limit=${limit}&client_id=${encodeURIComponent(this.clientId)}`;
    return this.fetchJson(url);
  },

  async loadConversation(conversationId) {
    return this.fetchJson(CONFIG.ENDPOINTS.LOAD_CONVERSATION, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: conversationId,
        client_id: this.clientId,
      }),
    });
  },

  async newConversation() {
    return this.fetchJson(CONFIG.ENDPOINTS.NEW_CONVERSATION, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this.clientId }),
    });
  },

  async deleteConversation(conversationId) {
    return this.fetchJson(CONFIG.ENDPOINTS.DELETE_CONVERSATION, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: conversationId,
        client_id: this.clientId,
      }),
    });
  },

  async *streamResponse(history, conversationId, configName, signal = null, provider = null, model = null, playbookName = undefined) {
    // A/B mode passes the name explicitly (both arms must get the same input);
    // otherwise consume the pending one-shot /playbook selection.
    if (playbookName === undefined) {
      playbookName = this._pendingPlaybookName;
      this._pendingPlaybookName = null;
    }
    const response = await fetch(CONFIG.ENDPOINTS.STREAM, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        last_message: history.slice(-1),
        conversation_id: conversationId,
        config_name: configName,
        client_sent_msg_ts: Date.now(),
        client_timeout: CONFIG.STREAMING.TIMEOUT,
        client_id: this.clientId,
        include_agent_steps: true,
        include_tool_steps: true,
        provider: provider,
        model: model,
        playbook_name: playbookName || null,
      }),
      signal: signal,
    });

    if (response.status === 401) {
      window.location.href = '/';
      return;
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed (${response.status})`);
    }

    yield* this._readNDJSON(response);
  },

  // A/B Testing API methods
  async submitABPreference(comparisonId, preference) {
    return this.fetchJson(CONFIG.ENDPOINTS.AB_PREFERENCE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        comparison_id: comparisonId,
        preference: preference,
        client_id: this.clientId,
      }),
    });
  },

  async getPendingABComparison(conversationId) {
    const url = `${CONFIG.ENDPOINTS.AB_PENDING}?conversation_id=${conversationId}&client_id=${encodeURIComponent(this.clientId)}`;
    return this.fetchJson(url);
  },

  // Pool-based A/B testing API methods
  async getABPool() {
    return this.fetchJson(`${CONFIG.ENDPOINTS.AB_POOL}?client_id=${encodeURIComponent(this.clientId)}`);
  },

  async getABDecision(conversationId = null) {
    const params = new URLSearchParams({ client_id: this.clientId });
    if (conversationId != null) {
      params.set('conversation_id', String(conversationId));
    }
    return this.fetchJson(`${CONFIG.ENDPOINTS.AB_DECISION}?${params.toString()}`);
  },

  async getABMetrics() {
    return this.fetchJson(`${CONFIG.ENDPOINTS.AB_METRICS}?client_id=${encodeURIComponent(this.clientId)}`);
  },

  async saveABPool(payload) {
    return this.fetchJson(CONFIG.ENDPOINTS.AB_POOL_SET, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, client_id: this.clientId }),
    });
  },

  async disableABPool() {
    return this.fetchJson(CONFIG.ENDPOINTS.AB_POOL_DISABLE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this.clientId }),
    });
  },

  /**
   * Stream a pool-based A/B comparison. Returns an async iterator of NDJSON events.
   * Each event has an 'arm' field ('a' or 'b') plus 'type', 'content', etc.
   */
  async *streamABComparison(history, conversationId, configName, signal, provider = null, model = null, playbookName = null) {
    const streamOverride = window.__ARCHI_PLAYWRIGHT__?.ab?.streamOverride;
    if (typeof streamOverride === 'function') {
      yield* streamOverride({
        history,
        conversationId,
        configName,
        signal,
        provider,
        model,
        playbookName,
        clientId: this.clientId,
      });
      return;
    }

    const body = {
      last_message: history.slice(-1),
      conversation_id: conversationId,
      config_name: configName || null,
      client_id: this.clientId,
      client_sent_msg_ts: Date.now(),
      client_timeout: CONFIG.STREAMING.TIMEOUT,
      provider,
      model,
      playbook_name: playbookName || null,
    };

    const response = await fetch(CONFIG.ENDPOINTS.AB_COMPARE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`A/B compare failed: ${response.status} ${errText}`);
    }

    yield* this._readNDJSON(response);
  },

  // Provider API methods
  async getProviders() {
    return this.fetchJson(CONFIG.ENDPOINTS.PROVIDERS);
  },

  async getPipelineDefaultModel() {
    return this.fetchJson(CONFIG.ENDPOINTS.PIPELINE_DEFAULT_MODEL);
  },

  async getAgentInfo(configName = null) {
    const url = configName
      ? `${CONFIG.ENDPOINTS.AGENT_INFO}?config_name=${encodeURIComponent(configName)}`
      : CONFIG.ENDPOINTS.AGENT_INFO;
    return this.fetchJson(url);
  },

  async getAgentTemplate(name = null) {
    const url = name
      ? `${CONFIG.ENDPOINTS.AGENT_TEMPLATE}?name=${encodeURIComponent(name)}`
      : CONFIG.ENDPOINTS.AGENT_TEMPLATE;
    return this.fetchJson(url);
  },

  async getAgentsList() {
    return this.fetchJson(CONFIG.ENDPOINTS.AGENTS_LIST);
  },

  async getAgentSpec(name) {
    const url = `${CONFIG.ENDPOINTS.AGENT_SPEC}?name=${encodeURIComponent(name)}`;
    return this.fetchJson(url);
  },

  async getCurrentUser() {
    return this.fetchJson(CONFIG.ENDPOINTS.USER_ME);
  },

  async updateUserPreferences(payload) {
    return this.fetchJson(CONFIG.ENDPOINTS.USER_PREFERENCES, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  async setActiveAgent(name) {
    return this.fetchJson(CONFIG.ENDPOINTS.AGENT_ACTIVE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        client_id: this.clientId,
      }),
    });
  },

  async deleteAgent(name) {
    return this.fetchJson(CONFIG.ENDPOINTS.AGENT_SAVE, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        client_id: this.clientId,
      }),
    });
  },

  async saveAgentSpec(payload) {
    return this.fetchJson(CONFIG.ENDPOINTS.AGENT_SAVE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...payload,
        client_id: this.clientId,
      }),
    });
  },

  _pendingPlaybookName: null,

  async getPlaybooksList() {
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}?client_id=${encodeURIComponent(this.clientId)}`);
  },

  async enablePlaybook(id) {
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}/${id}/enable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this.clientId }),
    });
  },

  async disablePlaybook(id) {
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}/${id}/disable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this.clientId }),
    });
  },

  async getPlaybook(id) {
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}/${id}?client_id=${encodeURIComponent(this.clientId)}`);
  },

  async createPlaybook(payload) {
    return this.fetchJson(CONFIG.ENDPOINTS.PLAYBOOKS, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, client_id: this.clientId }),
    });
  },

  async updatePlaybook(id, payload) {
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, client_id: this.clientId }),
    });
  },

  async deletePlaybook(id) {
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}/${id}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this.clientId }),
    });
  },

  async exportPlaybooks() {
    // fetch + blob (not a bare link) so the session cookie/client_id path works the
    // same in SSO and anonymous deployments. The server sends a zip of
    // <name>/SKILL.md folders — the Agent Skills format claude.ai also accepts.
    const resp = await fetch(`${CONFIG.ENDPOINTS.PLAYBOOKS}/export?client_id=${encodeURIComponent(this.clientId)}`);
    if (!resp.ok) throw new Error(`export failed (${resp.status})`);
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `archi-playbooks-${new Date().toISOString().slice(0, 10)}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },

  // Legacy JSON import (pre-SKILL.md exports).
  async importPlaybooks(playbooks, onConflict) {
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this.clientId, playbooks, on_conflict: onConflict }),
    });
  },

  // SKILL.md import: a zip of <name>/SKILL.md folders or a single .md file.
  async importPlaybookFile(file, onConflict) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('client_id', this.clientId);
    fd.append('on_conflict', onConflict);
    return this.fetchJson(`${CONFIG.ENDPOINTS.PLAYBOOKS}/import`, { method: 'POST', body: fd });
  },

  async getProviderModels(providerType) {
    const url = `${CONFIG.ENDPOINTS.PROVIDER_MODELS}?provider=${encodeURIComponent(providerType)}`;
    return this.fetchJson(url);
  },

  async validateProvider(providerType) {
    return this.fetchJson(CONFIG.ENDPOINTS.VALIDATE_PROVIDER, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: providerType }),
    });
  },

  // API Key management methods
  async getProviderKeys() {
    return this.fetchJson(CONFIG.ENDPOINTS.PROVIDER_KEYS);
  },

  async setProviderKey(providerType, apiKey) {
    return this.fetchJson(CONFIG.ENDPOINTS.SET_PROVIDER_KEY, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: providerType, api_key: apiKey }),
    });
  },

  async clearProviderKey(providerType) {
    return this.fetchJson(CONFIG.ENDPOINTS.CLEAR_PROVIDER_KEY, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: providerType }),
    });
  },

  // Feedback methods
  async likeMessage(messageId) {
    return this.fetchJson(CONFIG.ENDPOINTS.LIKE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message_id: messageId }),
    });
  },

  async dislikeMessage(messageId, options = {}) {
    return this.fetchJson(CONFIG.ENDPOINTS.DISLIKE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message_id: messageId,
        feedback_msg: options.feedback_msg || '',
        incorrect: options.incorrect || false,
        unhelpful: options.unhelpful || false,
        inappropriate: options.inappropriate || false,
      }),
    });
  },

  async submitTextFeedback(messageId, text) {
    return this.fetchJson(CONFIG.ENDPOINTS.TEXT_FEEDBACK, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message_id: messageId,
        feedback_msg: text,
      }),
    });
  },
};

// =============================================================================
// Markdown Renderer
// =============================================================================

const Markdown = {
  init() {
    if (typeof marked !== 'undefined') {
      marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: (code, lang) => this.highlightCode(code, lang),
      });
    }
  },

  highlightCode(code, lang) {
    if (typeof hljs !== 'undefined') {
      try {
        if (lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
      } catch (e) {
        console.error('Highlight error:', e);
      }
    }
    return Utils.escapeHtml(code);
  },

  render(text) {
    if (!text) return '';
    
    if (typeof marked !== 'undefined') {
      try {
        let html = marked.parse(text);
        // Add copy buttons to code blocks
        html = this.addCodeBlockHeaders(html);
        return html;
      } catch (e) {
        console.error('Markdown render error:', e);
      }
    }
    
    return Utils.escapeHtml(text);
  },

  addCodeBlockHeaders(html) {
    // Match <pre><code class="language-xxx"> blocks
    return html.replace(
      /<pre><code class="language-(\w+)">/g,
      (match, lang) => `
        <pre>
          <div class="code-block-header">
            <span class="code-block-lang">${lang}</span>
            <button class="code-block-copy" onclick="Markdown.copyCode(this)">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              <span>Copy</span>
            </button>
          </div>
          <code class="language-${lang}">`
    ).replace(
      /<pre><code>/g,
      `<pre>
        <div class="code-block-header">
          <span class="code-block-lang">code</span>
          <button class="code-block-copy" onclick="Markdown.copyCode(this)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy</span>
          </button>
        </div>
        <code>`
    );
  },

  copyCode(button) {
    const pre = button.closest('pre');
    const code = pre.querySelector('code');
    const text = code.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
      button.classList.add('copied');
      button.querySelector('span').textContent = 'Copied!';
      
      setTimeout(() => {
        button.classList.remove('copied');
        button.querySelector('span').textContent = 'Copy';
      }, 2000);
    });
  },
};

// Make copyCode globally accessible for onclick handlers
window.Markdown = Markdown;

// =============================================================================
// UI Components
// =============================================================================

const UI = {
  elements: {},
  sendBtnDefaultHtml: null,
  traceTimerIntervals: new Map(),

  init() {
    this.elements = {
      app: document.querySelector('.app'),
      sidebar: document.querySelector('.sidebar'),
      sidebarToggle: document.querySelector('.sidebar-toggle'),
      sidebarOverlay: document.querySelector('.sidebar-overlay'),
      conversationList: document.querySelector('.conversation-list'),
      newChatBtn: document.querySelector('.new-chat-btn'),
      messagesContainer: document.querySelector('.messages'),
      messagesInner: document.querySelector('.messages-inner'),
      inputField: document.querySelector('.input-field'),
      sendBtn: document.querySelector('.send-btn'),
      modelSelectA: null,

      settingsBtn: document.querySelector('.settings-btn'),
      dataTab: document.getElementById('data-tab'),
      settingsModal: document.querySelector('.settings-modal'),
      settingsBackdrop: document.querySelector('.settings-backdrop'),
      settingsClose: document.querySelector('.settings-close'),
      traceVerboseOptions: document.querySelector('.trace-verbose-options'),
      agentDropdown: document.querySelector('.agent-dropdown'),
      agentDropdownBtn: document.querySelector('.agent-dropdown-btn'),
      agentDropdownMenu: document.querySelector('.agent-dropdown-menu'),
      agentDropdownLabel: document.querySelector('.agent-dropdown-label'),
      agentDropdownList: document.querySelector('.agent-dropdown-list'),
      agentDropdownAdd: document.querySelector('.agent-dropdown-add'),
      agentInfoModal: document.querySelector('.agent-info-modal'),
      agentInfoBackdrop: document.querySelector('.agent-info-backdrop'),
      agentInfoClose: document.querySelector('.agent-info-close'),
      agentInfoContent: document.getElementById('agent-info-content'),
      agentSpecModal: document.querySelector('.agent-spec-modal'),
      agentSpecBackdrop: document.querySelector('.agent-spec-backdrop'),
      agentSpecClose: document.querySelector('.agent-spec-close'),
      agentSpecTitle: document.getElementById('agent-spec-title'),
      agentSpecEditor: document.getElementById('agent-spec-editor'),
      agentSpecName: document.getElementById('agent-spec-name'),
      agentSpecPrompt: document.getElementById('agent-spec-prompt'),
      agentSpecStatus: document.getElementById('agent-spec-status'),
      agentSpecSave: document.querySelector('.agent-spec-save'),
      agentSpecReset: document.querySelector('.agent-spec-reset'),
      agentSpecToolsList: document.querySelector('.agent-spec-tools-list'),
      agentSpecResizeHandle: document.querySelector('.agent-spec-resize-handle'),
      agentSpecPanel: document.querySelector('.agent-spec-panel'),
      // Provider selection elements
      providerSelect: document.getElementById('provider-select'),
      modelSelectPrimary: document.getElementById('model-select-primary'),

      providerStatus: document.getElementById('provider-status'),
      // User profile elements
      userProfileWidget: document.getElementById('user-profile-widget'),
      userDisplayName: document.getElementById('user-display-name'),
      userEmail: document.getElementById('user-email'),
      userRolesToggle: document.getElementById('user-roles-toggle'),
      userRolesPanel: document.getElementById('user-roles-panel'),
      userRolesList: document.getElementById('user-roles-list'),
      userLogoutBtn: document.getElementById('user-logout-btn'),
      customModelInput: document.getElementById('custom-model-input'),
      customModelRow: document.getElementById('custom-model-row'),
      activeModelLabel: document.getElementById('active-model-label'),
      darkModeToggle: document.getElementById('dark-mode-toggle'),
      abSettingsNav: document.getElementById('ab-settings-nav'),
      abSettingsSection: document.getElementById('settings-ab-testing'),
      abParticipationGroup: document.getElementById('ab-participation-group'),
      abParticipationSlider: document.getElementById('ab-participation-slider'),
      abParticipationValue: document.getElementById('ab-participation-value'),
      abParticipationDefault: document.getElementById('ab-participation-default'),
      abParticipationNote: document.getElementById('ab-participation-note'),
      abParticipationInactive: document.getElementById('ab-participation-inactive'),
      abAdminLinkSection: document.getElementById('ab-settings-section'),
    };

    this.sendBtnDefaultHtml = this.elements.sendBtn?.innerHTML || '';

    if (this.elements.agentDropdownMenu) {
      this.elements.agentDropdownMenu.hidden = true;
    }
    if (this.elements.agentDropdownBtn) {
      this.elements.agentDropdownBtn.setAttribute('aria-expanded', 'false');
    }

    this.bindEvents();
    this.initTraceVerboseMode();
    this.initThemeToggle();
  },

  initThemeToggle() {
    if (!this.elements.darkModeToggle) return;
    const savedTheme = localStorage.getItem('archi_theme') || 'light';
    const isDark = savedTheme === 'dark';
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    this.elements.darkModeToggle.checked = isDark;
  },

  initTraceVerboseMode() {
    // Set the initial radio button based on stored preference
    const storedMode = localStorage.getItem(CONFIG.STORAGE_KEYS.TRACE_VERBOSE_MODE) || 'normal';
    const radio = document.querySelector(`input[name="trace-verbose"][value="${storedMode}"]`);
    if (radio) {
      radio.checked = true;
    }
  },

  bindEvents() {
    // Sidebar toggle
    this.elements.sidebarToggle?.addEventListener('click', () => this.toggleSidebar());
    
    // Sidebar overlay click to close (mobile)
    this.elements.sidebarOverlay?.addEventListener('click', () => this.closeSidebar());
    
    // New chat
    this.elements.newChatBtn?.addEventListener('click', () => Chat.newConversation());
    
    // Send message
    this.elements.sendBtn?.addEventListener('click', () => Chat.handleSendOrStop());
    this.elements.inputField?.addEventListener('keydown', (e) => {
      // When the playbook menu is open, the keyboard drives it (Tab/Enter complete it).
      if (PlaybookMenu.open) {
        if (e.key === 'ArrowDown') { e.preventDefault(); PlaybookMenu.move(1); return; }
        if (e.key === 'ArrowUp')   { e.preventDefault(); PlaybookMenu.move(-1); return; }
        if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
          e.preventDefault(); PlaybookMenu.selectActive(); return;
        }
        if (e.key === 'Escape')    { e.preventDefault(); PlaybookMenu.hide(); return; }
      }
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        Chat.handleSendOrStop();
      }
    });

    // Auto-resize textarea
    this.elements.inputField?.addEventListener('input', () => this.autoResizeInput());

    // Playbook quick-invoke autocomplete
    this.elements.inputField?.addEventListener('input', () => PlaybookMenu.maybeShow());

    // Settings modal
    this.elements.settingsBtn?.addEventListener('click', () => this.openSettings());
    this.elements.settingsBackdrop?.addEventListener('click', () => this.closeSettings());
    this.elements.settingsClose?.addEventListener('click', () => this.closeSettings());
    
    // Data viewer navigation
    this.elements.dataTab?.addEventListener('click', (e) => {
      e.preventDefault();
      const conversationId = Chat.state.conversationId;
      if (conversationId) {
        // Store conversation ID for the data viewer
        localStorage.setItem('currentConversationId', conversationId);
        window.location.href = `/data?conversation_id=${encodeURIComponent(conversationId)}`;
      } else {
        // Allow viewing all documents without a conversation
        window.location.href = '/data';
      }
    });

    this.elements.agentInfoBackdrop?.addEventListener('click', () => {
      this.closeAgentInfo();
    });
    this.elements.agentInfoClose?.addEventListener('click', () => {
      this.closeAgentInfo();
    });
    this.elements.agentDropdownBtn?.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.toggleAgentDropdown();
    });
    this.elements.agentDropdownAdd?.addEventListener('click', async () => {
      this.closeAgentDropdown();
      try {
        await this.openAgentSpecEditor({ mode: 'create' });
      } catch (e) {
        console.error('Failed to open agent spec editor:', e);
      }
    });
    this.elements.agentDropdownList?.addEventListener('click', (e) => {
      const target = e.target;
      const row = target.closest('.agent-dropdown-item');
      if (!row) return;
      e.preventDefault();
      e.stopPropagation();
      // Handle inline delete confirmation buttons
      if (target.closest('.agent-dropdown-confirm-yes')) {
        const name = row.dataset.agentName;
        this.doDeleteAgent(name);
        return;
      }
      if (target.closest('.agent-dropdown-confirm-no')) {
        // Cancel: re-render list to remove confirmation state
        this.renderAgentsList(Chat.state.allAgents || Chat.state.agents || [], Chat.state.activeAgentName);
        return;
      }
      if (target.closest('.agent-dropdown-clone')) {
        const name = row.dataset.agentName;
        this.closeAgentDropdown();
        this.openAgentSpecEditor({ mode: 'clone', name });
        return;
      }
      if (target.closest('.agent-dropdown-edit')) {
        const name = row.dataset.agentName;
        this.closeAgentDropdown();
        this.openAgentSpecEditor({ mode: 'edit', name });
        return;
      }
      if (target.closest('.agent-dropdown-delete')) {
        const name = row.dataset.agentName;
        this.showDeleteConfirmation(row, name);
        return;
      }
      if (row.dataset.agentName && !target.closest('.agent-dropdown-actions')) {
        this.closeAgentDropdown();
        Chat.setActiveAgent(row.dataset.agentName);
      }
    });
    this.elements.agentSpecBackdrop?.addEventListener('click', () => {
      this.closeAgentSpecEditor();
    });
    this.elements.agentSpecClose?.addEventListener('click', () => {
      this.closeAgentSpecEditor();
    });
    this.elements.agentSpecReset?.addEventListener('click', () => {
      this.resetAgentSpecForm();
    });
    this.elements.agentSpecSave?.addEventListener('click', () => {
      this.saveAgentSpec();
    });
    // Playbooks section bindings (hosted in the Settings modal)
    document.querySelector('.playbook-cancel')?.addEventListener('click', () => {
      Chat._editingPlaybookId = null;
      Chat.setPlaybookEditorReadOnly(false);
      Chat.showPlaybooksView('list');
    });
    document.querySelector('.playbooks-back')?.addEventListener('click', () => {
      Chat._editingPlaybookId = null;
      Chat.setPlaybookEditorReadOnly(false);
      Chat.showPlaybooksView('list');
    });
    document.querySelector('.playbook-save')?.addEventListener('click', () => Chat.savePlaybookFromPanel());
    document.querySelector('.playbooks-new')?.addEventListener('click', () => {
      Chat._editingPlaybookId = null;
      Chat.setPlaybookEditorReadOnly(false);
      Chat.setPlaybookEditorFields();  // empty fields, visibility back to private
      Chat.showPlaybooksView('editor', 'New playbook');
    });
    document.querySelector('.playbooks-export')?.addEventListener('click', async () => {
      try {
        await API.exportPlaybooks();
      } catch (err) {
        UI.showToast('Export failed: ' + (err?.message || 'error'));
      }
    });
    const playbooksImportInput = document.querySelector('.playbooks-import-file');
    document.querySelector('.playbooks-import')?.addEventListener('click', () => playbooksImportInput?.click());
    playbooksImportInput?.addEventListener('change', async () => {
      const file = playbooksImportInput.files?.[0];
      playbooksImportInput.value = '';  // allow re-selecting the same file later
      if (!file) return;
      const isJson = /\.json$/i.test(file.name || '');
      let items = null;  // only known up front for legacy JSON files
      if (isJson) {
        try {
          const parsed = JSON.parse(await file.text());
          items = Array.isArray(parsed) ? parsed : parsed?.playbooks;
          if (!Array.isArray(items) || !items.length) throw new Error('no playbooks found in the file');
        } catch (err) {
          UI.showToast('Import failed: ' + (err?.message || 'invalid file'));
          return;
        }
      }
      const what = items ? `${items.length} playbook(s)` : 'playbooks';
      if (!confirm(`Import ${what} from "${file.name}"? Imported playbooks stay private to you.`)) return;
      const doImport = (onConflict, subset) => (isJson
        ? API.importPlaybooks(subset || items, onConflict)
        : API.importPlaybookFile(file, onConflict));
      try {
        let res = await doImport('skip');
        if (res.skipped?.length) {
          const shown = res.skipped.slice(0, 5).join(', ') + (res.skipped.length > 5 ? '…' : '');
          if (confirm(`${res.skipped.length} name(s) already exist (${shown}). Overwrite them?`)) {
            // JSON can retry just the skipped items; a zip is re-sent whole, so
            // names imported in round one re-overwrite with identical content.
            const retry = isJson ? items.filter(it => res.skipped.includes(it?.name)) : null;
            const res2 = await doImport('overwrite', retry);
            const fresh = res.imported || [];
            res = {
              imported: fresh,
              overwritten: (res2.overwritten || []).filter(n => !fresh.includes(n)),
              skipped: res2.skipped,
              errors: isJson ? [...(res.errors || []), ...(res2.errors || [])] : (res2.errors || []),
              public_flags_ignored: isJson
                ? (res.public_flags_ignored || 0) + (res2.public_flags_ignored || 0)
                : (res2.public_flags_ignored || 0),
            };
          }
        }
        const bits = [];
        if (res.imported?.length) bits.push(`${res.imported.length} imported`);
        if (res.overwritten?.length) bits.push(`${res.overwritten.length} overwritten`);
        if (res.skipped?.length) bits.push(`${res.skipped.length} skipped`);
        if (res.errors?.length) bits.push(`${res.errors.length} failed`);
        let note = '';
        if (res.public_flags_ignored) {
          const verb = res.public_flags_ignored === 1 ? 'was' : 'were';
          note = ` — ${res.public_flags_ignored} marked "public" in the file ${verb} kept private (share from the editor)`;
        }
        UI.showToast('Import: ' + (bits.join(', ') || 'nothing to do') + note);
        await Chat.loadPlaybooksPanel();
        PlaybookMenu.playbooks = [];
      } catch (err) {
        UI.showToast('Import failed: ' + (err?.message || 'error'));
      }
    });
    document.querySelector('.playbooks-list')?.addEventListener('click', async (e) => {
      const editBtn = e.target.closest('.playbook-edit');
      const delBtn = e.target.closest('.playbook-delete');
      const viewBtn = e.target.closest('.playbook-view');
      const addBtn = e.target.closest('.playbook-add');
      const removeBtn = e.target.closest('.playbook-remove');
      if (editBtn || viewBtn) {
        try {
          const playbook = await API.getPlaybook((editBtn || viewBtn).dataset.id);
          Chat._editingPlaybookId = editBtn ? playbook.id : null;
          Chat.setPlaybookEditorFields(playbook);
          Chat.setPlaybookEditorReadOnly(!editBtn);  // View = read-only
          Chat.showPlaybooksView('editor', editBtn ? 'Edit playbook' : playbook.name);
        } catch (err) {
          UI.showToast('Could not open playbook: ' + (err?.message || 'error'));
        }
      } else if (delBtn) {
        const name = delBtn.dataset.name || 'this playbook';
        if (!confirm(`Delete ${name}? This can't be undone.`)) return;
        try {
          await API.deletePlaybook(delBtn.dataset.id);
          await Chat.loadPlaybooksPanel();
          PlaybookMenu.playbooks = [];
        } catch (err) {
          UI.showToast('Could not delete playbook: ' + (err?.message || 'error'));
        }
      } else if (addBtn) {
        try {
          await API.enablePlaybook(addBtn.dataset.id);
          await Chat.loadPlaybooksPanel();
          PlaybookMenu.playbooks = [];
        } catch (err) {
          UI.showToast('Could not add playbook: ' + (err?.message || 'error'));
        }
      } else if (removeBtn) {
        try {
          await API.disablePlaybook(removeBtn.dataset.id);
          await Chat.loadPlaybooksPanel();
          PlaybookMenu.playbooks = [];
        } catch (err) {
          UI.showToast('Could not remove playbook: ' + (err?.message || 'error'));
        }
      }
    });
    document.querySelector('.playbooks-search')?.addEventListener('input', () => Chat.renderPlaybooksPanel());
    document.querySelector('.playbooks-tabs')?.addEventListener('click', (e) => {
      const tabBtn = e.target.closest('.playbooks-tab');
      if (!tabBtn) return;
      Chat._panelTab = tabBtn.dataset.tab;
      Chat.renderPlaybooksPanel();
    });
    // Resize handle for agent spec modal
    this.initAgentSpecResize();
    
    // A/B pool editor — save & disable buttons
    document.getElementById('ab-pool-save')?.addEventListener('click', async () => {
      const sel = UI._getABPoolSelection();
      if (!sel || !sel.champion || sel.variants.length < 2) return;
      const saveBtn = document.getElementById('ab-pool-save');
      const msgEl = document.getElementById('ab-pool-message');
      const sampleRate = Number(document.getElementById('ab-sample-rate')?.value || 1);
      const disclosureMode = document.getElementById('ab-disclosure-mode')?.value || 'post_vote_reveal';
      const defaultTraceMode = document.getElementById('ab-trace-mode')?.value || 'hidden';
      saveBtn.disabled = true;
      saveBtn.textContent = 'Saving…';
      try {
        const result = await API.saveABPool({
          champion: sel.champion,
          variants: sel.variants,
          comparison_rate: sampleRate,
          variant_label_mode: disclosureMode,
          activity_panel_default_state: defaultTraceMode,
        });
        if (result?.success) {
          if (msgEl) { msgEl.textContent = 'Pool saved'; msgEl.className = 'ab-pool-message success'; }
          Chat.state.abPool = result;
          // Re-render to reflect saved state
          UI.updateABPoolUI(result);
        } else {
          if (msgEl) { msgEl.textContent = result?.error || 'Save failed'; msgEl.className = 'ab-pool-message error'; }
        }
      } catch (e) {
        if (msgEl) { msgEl.textContent = e.message || 'Save failed'; msgEl.className = 'ab-pool-message error'; }
      } finally {
        saveBtn.textContent = 'Save Pool';
        UI._updateABPoolSaveState();
      }
    });

    document.getElementById('ab-pool-disable')?.addEventListener('click', async () => {
      const disableBtn = document.getElementById('ab-pool-disable');
      const msgEl = document.getElementById('ab-pool-message');
      disableBtn.disabled = true;
      try {
        const result = await API.disableABPool();
        if (result?.success) {
          Chat.state.abPool = null;
          UI.updateABPoolUI({ enabled: false });
          if (msgEl) { msgEl.textContent = 'Pool disabled'; msgEl.className = 'ab-pool-message success'; }
          // If A/B mode was active in chat, deactivate
          if (Chat.state.abVotePending) Chat.cancelPendingABComparison();
        }
      } catch (e) {
        if (msgEl) { msgEl.textContent = e.message || 'Failed'; msgEl.className = 'ab-pool-message error'; }
      } finally {
        disableBtn.disabled = false;
      }
    });

    // Trace verbose mode radio buttons
    this.elements.traceVerboseOptions?.addEventListener('change', (e) => {
      if (e.target.name === 'trace-verbose') {
        Chat.setTraceVerboseMode(e.target.value);
      }
    });

    this.elements.darkModeToggle?.addEventListener('change', (e) => {
      const isDark = e.target.checked;
      document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
      localStorage.setItem('archi_theme', isDark ? 'dark' : 'light');
    });

    this.elements.abParticipationSlider?.addEventListener('input', (e) => {
      this.updateABParticipationPreview(Number(e.target.value));
    });

    this.elements.abParticipationSlider?.addEventListener('change', async (e) => {
      await Chat.saveABParticipationPreference(Number(e.target.value) / 100);
    });

    // Provider selection
    this.elements.providerSelect?.addEventListener('change', (e) => {
      Chat.handleProviderChange(e.target.value);
    });

    this.elements.modelSelectPrimary?.addEventListener('change', (e) => {
      Chat.handleModelChange(e.target.value);
    });

    this.elements.customModelInput?.addEventListener('input', (e) => {
      Chat.handleCustomModelChange(e.target.value);
    });

    // User profile widget interactions
    this.elements.userRolesToggle?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleUserRolesPanel();
    });

    this.elements.userProfileWidget?.addEventListener('click', () => {
      this.toggleUserRolesPanel();
    });

    this.elements.userLogoutBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      window.location.href = '/logout';
    });

    // Close modal on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.elements.settingsModal?.style.display !== 'none') {
        this.closeSettings();
      }
      if (e.key === 'Escape' && this.elements.agentSpecModal?.style.display !== 'none') {
        this.closeAgentSpecEditor();
      }
      if (e.key === 'Escape' && this.elements.agentDropdownMenu && !this.elements.agentDropdownMenu.hidden) {
        this.closeAgentDropdown();
      }
      if (e.key === 'Escape' && this.elements.agentInfoModal?.style.display !== 'none') {
        this.closeAgentInfo();
      }
    });

    document.addEventListener('click', (e) => {
      if (!this.elements.agentDropdownMenu || this.elements.agentDropdownMenu.hidden) return;
      if (!this.elements.agentDropdown?.contains(e.target)) {
        this.closeAgentDropdown();
      }
    });
    
    // Settings navigation
    document.querySelectorAll('.settings-nav-item').forEach(btn => {
      btn.addEventListener('click', (e) => this.switchSettingsSection(e.target.closest('.settings-nav-item')));
    });
  },

  openSettings() {
    if (this.elements.settingsModal) {
      this.elements.settingsModal.style.display = 'flex';
      // Reset to first section when opening
      const firstNavItem = document.querySelector('.settings-nav-item');
      if (firstNavItem) {
        this.switchSettingsSection(firstNavItem);
      }
    }
  },
  
  switchSettingsSection(navItem) {
    if (!navItem) return;
    
    const sectionId = navItem.dataset.section;
    
    // Update nav items
    document.querySelectorAll('.settings-nav-item').forEach(item => {
      item.classList.remove('active');
      item.setAttribute('aria-selected', 'false');
    });
    navItem.classList.add('active');
    navItem.setAttribute('aria-selected', 'true');
    
    // Update sections
    document.querySelectorAll('.settings-section').forEach(section => {
      section.classList.remove('active');
      section.hidden = true;
    });
    
    const targetSection = document.getElementById(`settings-${sectionId}`);
    if (targetSection) {
      targetSection.classList.add('active');
      targetSection.hidden = false;
    }

    if (sectionId === 'playbooks' && typeof Chat !== 'undefined') {
      Chat.enterPlaybooksSection();
    }
  },

  closeSettings() {
    if (this.elements.settingsModal) {
      this.elements.settingsModal.style.display = 'none';
    }
  },

  async openAgentInfo() {
    if (!this.elements.agentInfoModal) return;
    this.elements.agentInfoModal.style.display = 'flex';
    if (this.elements.agentInfoContent) {
      this.elements.agentInfoContent.innerHTML = '<p class="agent-info-loading">Loading agent info…</p>';
    }
    await this.loadAgentInfo();
  },

  closeAgentInfo() {
    if (this.elements.agentInfoModal) {
      this.elements.agentInfoModal.style.display = 'none';
    }
  },

  toggleUserRolesPanel() {
    this.elements.userProfileWidget?.classList.toggle('expanded');
  },

  async loadUserProfile() {
    try {
      const response = await fetch('/auth/user');
      if (!response.ok) return;
      
      const data = await response.json();
      
      if (!data.logged_in) {
        // User not logged in, hide the widget
        if (this.elements.userProfileWidget) {
          this.elements.userProfileWidget.style.display = 'none';
        }
        return;
      }
      
      // Show the widget
      if (this.elements.userProfileWidget) {
        this.elements.userProfileWidget.style.display = 'block';
      }
      
      // Extract name from email (before @)
      const email = data.email || 'User';
      const displayName = email.split('@')[0];
      
      // Update user info
      if (this.elements.userDisplayName) {
        this.elements.userDisplayName.textContent = displayName;
      }
      if (this.elements.userEmail) {
        this.elements.userEmail.textContent = email;
      }
      
      // Render roles
      this.renderUserRoles(data.roles || []);
      
    } catch (e) {
      console.error('Failed to load user profile:', e);
      // Hide widget on error
      if (this.elements.userProfileWidget) {
        this.elements.userProfileWidget.style.display = 'none';
      }
    }
  },

  renderUserRoles(roles) {
    if (!this.elements.userRolesList) return;
    
    if (!roles || roles.length === 0) {
      this.elements.userRolesList.innerHTML = '<p style="color: var(--text-tertiary); font-size: var(--text-xs); padding: 0 4px;">No roles assigned</p>';
      return;
    }
    
    const getRoleClass = (role) => {
      if (role.includes('admin')) return 'role-admin';
      if (role.includes('expert')) return 'role-expert';
      return '';
    };
    
    const roleIcon = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
      <circle cx="9" cy="7" r="4"></circle>
      <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
      <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
    </svg>`;
    
    this.elements.userRolesList.innerHTML = roles
      .map(role => `
        <div class="user-role-badge ${getRoleClass(role)}">
          ${roleIcon}
          ${Utils.escapeHtml(role)}
        </div>
      `)
      .join('');
  },

  async loadAgentInfo() {
    if (!this.elements.agentInfoContent) return;
    try {
      const configName = this.getSelectedConfig('A');
      const info = await API.getAgentInfo(configName);
      const agentLabel = Utils.normalizeAgentName(info?.agent_name || Chat.getAgentLabel());
      if (info?.agent_name && !Chat.state.activeAgentName) {
        Chat.state.activeAgentName = Utils.normalizeAgentName(info.agent_name);
        if (this.elements.agentDropdownLabel) {
          this.elements.agentDropdownLabel.textContent = Utils.normalizeAgentName(info.agent_name);
        }
      }
      const modelLabel = Chat.getCurrentModelLabel();
      const pipelineLabel = info?.pipeline || 'Unknown';
      const embeddingLabel = info?.embedding_name || 'Not specified';
      const sources = Array.isArray(info?.data_sources) ? info.data_sources : [];
      const tools = Array.isArray(info?.agent_tools) ? info.agent_tools : [];
      const prompt = info?.agent_prompt || '';

      const sourcesHtml = sources.length
        ? `<ul class="agent-info-list">${sources.map(source => `<li>${Utils.escapeHtml(source)}</li>`).join('')}</ul>`
        : '<p>No data sources configured.</p>';

      const toolsHtml = tools.length
        ? `<ul class="agent-info-list">${tools.map(tool => `<li>${Utils.escapeHtml(tool)}</li>`).join('')}</ul>`
        : '<p>No tools configured.</p>';

      this.elements.agentInfoContent.innerHTML = `
        <div class="agent-info-section">
          <h4>Active agent</h4>
          <p>${Utils.escapeHtml(agentLabel)}</p>
        </div>
        <div class="agent-info-section">
          <h4>Model</h4>
          <p>${Utils.escapeHtml(modelLabel)}</p>
        </div>
        <div class="agent-info-section">
          <h4>Pipeline</h4>
          <p>${Utils.escapeHtml(pipelineLabel)}</p>
        </div>
        <div class="agent-info-section">
          <h4>Embedding</h4>
          <p>${Utils.escapeHtml(embeddingLabel)}</p>
        </div>
        <div class="agent-info-section">
          <h4>Data sources</h4>
          ${sourcesHtml}
        </div>
        <div class="agent-info-section">
          <h4>Tools</h4>
          ${toolsHtml}
        </div>
        <div class="agent-info-section">
          <h4>Prompt</h4>
          <pre class="agent-info-prompt">${Utils.escapeHtml(prompt)}</pre>
        </div>`;
    } catch (e) {
      console.error('Failed to load agent info:', e);
      this.elements.agentInfoContent.innerHTML = `
        <p class="agent-info-loading">Unable to load agent info. Please try again.</p>`;
    }
  },

  toggleAgentDropdown() {
    if (!this.elements.agentDropdownMenu || !this.elements.agentDropdownBtn) return;
    if (this.elements.agentDropdownMenu.hidden) {
      this.openAgentDropdown();
    } else {
      this.closeAgentDropdown();
    }
  },

  openAgentDropdown() {
    if (!this.elements.agentDropdownMenu || !this.elements.agentDropdownBtn) return;
    this.elements.agentDropdownMenu.hidden = false;
    this.elements.agentDropdownBtn.setAttribute('aria-expanded', 'true');
  },

  closeAgentDropdown() {
    if (!this.elements.agentDropdownMenu || !this.elements.agentDropdownBtn) return;
    this.elements.agentDropdownMenu.hidden = true;
    this.elements.agentDropdownBtn.setAttribute('aria-expanded', 'false');
  },

  showDeleteConfirmation(row, name) {
    if (!row) return;
    row.classList.add('agent-dropdown-item-confirming');
    row.innerHTML = `
      <span class="agent-dropdown-confirm-text">Delete "${Utils.escapeHtml(name)}"?</span>
      <div class="agent-dropdown-confirm-actions">
        <button class="agent-dropdown-confirm-yes" type="button">Delete</button>
        <button class="agent-dropdown-confirm-no" type="button">Cancel</button>
      </div>`;
  },

  async doDeleteAgent(name) {
    if (!name) return;
    try {
      await API.deleteAgent(Utils.normalizeAgentName(name));
      await Chat.loadAgents();
    } catch (e) {
      console.error('Failed to delete agent:', e);
      this.setAgentSpecStatus(e.message || 'Unable to delete agent.', 'error');
    }
  },

  renderAgentsList(agents = [], activeName = null) {
    if (this.elements.agentDropdownLabel) {
      this.elements.agentDropdownLabel.textContent = Utils.normalizeAgentName(activeName) || 'Agent';
    }
    if (!this.elements.agentDropdownList) return;
    let activeMatched = false;
    const rows = agents.map((agent) => {
      const rawName = agent.name || agent.filename || 'Unknown';
      const name = Utils.normalizeAgentName(rawName);
      let isActive = false;
      if (!activeMatched && activeName && Utils.normalizeAgentName(activeName) === name) {
        isActive = true;
        activeMatched = true;
      }
      const checkmark = isActive ? '<svg class="agent-dropdown-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>' : '<span class="agent-dropdown-check-spacer"></span>';
      return `
        <div class="agent-dropdown-item${isActive ? ' active' : ''}" data-agent-name="${Utils.escapeHtml(name)}">
          <span class="agent-dropdown-name">${checkmark}${Utils.escapeHtml(name)}</span>
          <div class="agent-dropdown-actions">
            <button class="agent-dropdown-clone" type="button" title="Create variant">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            </button>
            <button class="agent-dropdown-edit" type="button" title="Edit">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
            </button>
            <button class="agent-dropdown-delete" type="button" title="Delete">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
          </div>
        </div>`;
    });
    this.elements.agentDropdownList.innerHTML = rows.length
      ? rows.join('')
      : '<div class="agent-dropdown-item">No agents found</div>';
  },

  async openAgentSpecEditor({ mode = 'create', name = null } = {}) {
    if (!this.elements.agentSpecModal) return;
    this.elements.agentSpecModal.style.display = 'flex';
    this.setAgentSpecStatus('');
    // Clone mode → load source spec, then switch to create for saving
    this.agentSpecMode = mode === 'clone' ? 'create' : mode;
    this.agentSpecName = mode === 'clone' ? null : name;
    this.agentSpecOriginalName = mode === 'edit' ? name : null;
    // Restore persisted size
    this.restoreAgentSpecSize();
    if (this.elements.agentSpecTitle) {
      if (mode === 'clone') {
        this.elements.agentSpecTitle.textContent = `New Variant of ${name || 'Agent'}`;
      } else if (mode === 'edit') {
        this.elements.agentSpecTitle.textContent = `Edit ${name || 'Agent'}`;
      } else {
        this.elements.agentSpecTitle.textContent = 'New Agent';
      }
    }
    // Update reset button label
    if (this.elements.agentSpecReset) {
      this.elements.agentSpecReset.textContent = mode === 'edit' ? 'Revert changes' : 'Reset template';
    }
    if (this.elements.agentSpecName) {
      this.elements.agentSpecName.readOnly = mode === 'edit';
      this.elements.agentSpecName.title = mode === 'edit'
        ? 'Agent name is fixed while editing. Clone or create a new agent to use a different name.'
        : '';
    }
    // Clear validation errors
    this.clearAgentSpecValidation();
    if (mode === 'clone' && name) {
      // Load tool palette first, then load source spec and modify name
      await this.loadAgentToolPalette();
      await this.loadAgentSpecByName(name);
      // Append " (variant)" to the name so user can tweak tools & save
      if (this.elements.agentSpecName) {
        this.elements.agentSpecName.value = `${name} (variant)`;
      }
      this.setAgentSpecStatus('Cloned — adjust tools and name, then save.', 'info');
      setTimeout(() => this.elements.agentSpecName?.select(), 100);
    } else if (mode === 'edit' && name) {
      await this.loadAgentToolPalette();
      await this.loadAgentSpecByName(name);
      this.setAgentSpecStatus('Editing updates this agent in place. Clone or create a new agent to use a different name.', 'info');
    } else {
      await this.loadAgentSpecTemplate();
    }
    // Auto-focus name in create mode
    if (mode === 'create' && !name) {
      setTimeout(() => this.elements.agentSpecName?.focus(), 100);
    }
  },

  closeAgentSpecEditor() {
    if (this.elements.agentSpecModal) {
      this.elements.agentSpecModal.style.display = 'none';
    }
  },

  clearAgentSpecValidation() {
    this.elements.agentSpecName?.classList.remove('field-error');
    this.elements.agentSpecPrompt?.classList.remove('field-error');
  },

  setAgentSpecStatus(message, type = '') {
    if (!this.elements.agentSpecStatus) return;
    this.elements.agentSpecStatus.textContent = message || '';
    this.elements.agentSpecStatus.classList.remove('error', 'success');
    if (type) {
      this.elements.agentSpecStatus.classList.add(type);
    }
  },

  /** Parse YAML frontmatter and prompt body from .md content */
  parseAgentSpec(content) {
    const match = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/);
    if (!match) return { name: '', tools: [], prompt: content.trim() };
    const yaml = match[1];
    const prompt = (match[2] || '').trim();
    const nameMatch = yaml.match(/^name:\s*(.+)$/m);
    const name = nameMatch ? nameMatch[1].trim() : '';
    const tools = [];
    const toolsMatch = yaml.match(/^tools:\s*\n((?:\s+-\s+.+\n?)*)/m);
    if (toolsMatch) {
      const lines = toolsMatch[1].split('\n');
      for (const line of lines) {
        const m = line.match(/^\s+-\s+(.+)$/);
        if (m) tools.push(m[1].trim());
      }
    }
    return { name, tools, prompt };
  },

  /** Serialise structured form fields back to .md format */
  serialiseAgentSpec(name, tools, prompt, { ab_only = false } = {}) {
    let yaml = `---\nname: ${name}\n`;
    if (ab_only) yaml += 'ab_only: true\n';
    if (tools.length) {
      yaml += 'tools:\n';
      for (const t of tools) yaml += `  - ${t}\n`;
    }
    yaml += '---\n\n';
    return yaml + prompt;
  },

  /** Populate structured form fields from parsed data */
  populateAgentSpecForm({ name = '', tools = [], prompt = '' } = {}) {
    if (this.elements.agentSpecName) this.elements.agentSpecName.value = name;
    if (this.elements.agentSpecPrompt) this.elements.agentSpecPrompt.value = prompt;
    // Update tool checkboxes
    const checkboxes = this.elements.agentSpecToolsList?.querySelectorAll('input[type="checkbox"]');
    if (checkboxes) {
      checkboxes.forEach((cb) => {
        cb.checked = tools.includes(cb.value);
      });
    }
  },

  /** Collect form fields into the hidden editor textarea for save */
  collectAgentSpecForm() {
    const name = this.elements.agentSpecName?.value.trim() || '';
    const prompt = this.elements.agentSpecPrompt?.value.trim() || '';
    const tools = [];
    const checkboxes = this.elements.agentSpecToolsList?.querySelectorAll('input[type="checkbox"]:checked');
    if (checkboxes) {
      checkboxes.forEach((cb) => tools.push(cb.value));
    }
    return { name, tools, prompt };
  },

  async loadAgentSpecTemplate() {
    this.setAgentSpecStatus('');
    try {
      const response = await API.getAgentTemplate();
      const template = response?.template || '';
      if (this.elements.agentSpecEditor) this.elements.agentSpecEditor.value = template;
      this._lastAvailableTools = response?.tools || [];
      this.renderAgentToolPalette(this._lastAvailableTools);
      const parsed = this.parseAgentSpec(template);
      this.populateAgentSpecForm(parsed);
    } catch (e) {
      console.error('Failed to load agent template:', e);
      if (this.elements.agentSpecEditor) this.elements.agentSpecEditor.value = '';
      this.populateAgentSpecForm();
      this.setAgentSpecStatus('Unable to load agent template.', 'error');
    }
  },

  async loadAgentToolPalette() {
    try {
      const response = await API.getAgentTemplate();
      this._lastAvailableTools = response?.tools || [];
      this.renderAgentToolPalette(this._lastAvailableTools);
    } catch (e) {
      console.error('Failed to load tool palette:', e);
      this._lastAvailableTools = [];
      this.renderAgentToolPalette([]);
    }
  },

  async loadAgentSpecByName(name) {
    this.setAgentSpecStatus('');
    try {
      const response = await API.getAgentSpec(name);
      const content = response?.content || '';
      if (this.elements.agentSpecEditor) this.elements.agentSpecEditor.value = content;
      const parsed = this.parseAgentSpec(content);
      this.populateAgentSpecForm(parsed);
    } catch (e) {
      console.error('Failed to load agent spec:', e);
      if (this.elements.agentSpecEditor) this.elements.agentSpecEditor.value = '';
      this.populateAgentSpecForm();
      this.setAgentSpecStatus('Unable to load agent spec.', 'error');
    }
  },

  resetAgentSpecForm() {
    this.clearAgentSpecValidation();
    this.setAgentSpecStatus('');
    if (this.agentSpecMode === 'edit' && this.agentSpecOriginalName) {
      // Revert to saved version
      this.loadAgentSpecByName(this.agentSpecOriginalName);
    } else {
      this.loadAgentSpecTemplate();
    }
  },

  renderAgentToolPalette(tools = []) {
    if (!this.elements.agentSpecToolsList) return;
    if (!tools.length) {
      this.elements.agentSpecToolsList.innerHTML = '<div class="agent-spec-tool-desc">No tools available.</div>';
      return;
    }
    // Get currently selected tools from the form
    const currentForm = this.collectAgentSpecForm();
    const selectedTools = currentForm.tools || [];
    const items = tools.map((tool) => {
      // Tolerate both shapes: an object {name, description} or a bare name string.
      const toolName = typeof tool === 'string' ? tool : (tool.name || '');
      const toolDesc = typeof tool === 'string' ? '' : (tool.description || '');
      const checked = selectedTools.includes(toolName) ? 'checked' : '';
      return `
      <label class="agent-spec-tool">
        <input type="checkbox" class="agent-spec-tool-checkbox" value="${Utils.escapeHtml(toolName)}" ${checked} />
        <div class="agent-spec-tool-info">
          <div class="agent-spec-tool-name">${Utils.escapeHtml(toolName)}</div>
          <div class="agent-spec-tool-desc">${Utils.escapeHtml(toolDesc)}</div>
        </div>
      </label>`;
    });
    this.elements.agentSpecToolsList.innerHTML = items.join('');
  },

  async saveAgentSpec() {
    this.clearAgentSpecValidation();
    const { name, tools, prompt } = this.collectAgentSpecForm();
    // Client-side validation
    let hasError = false;
    if (!name) {
      this.elements.agentSpecName?.classList.add('field-error');
      this.setAgentSpecStatus('Agent name is required.', 'error');
      hasError = true;
    }
    if (!prompt) {
      this.elements.agentSpecPrompt?.classList.add('field-error');
      if (!hasError) this.setAgentSpecStatus('Prompt is required.', 'error');
      hasError = true;
    }
    if (hasError) return;
    if (this.agentSpecMode === 'edit' && this.agentSpecOriginalName && name !== this.agentSpecOriginalName) {
      this.elements.agentSpecName?.classList.add('field-error');
      this.setAgentSpecStatus('Agent name cannot be changed in edit mode. Clone or create a new agent instead.', 'error');
      return;
    }
    // Serialise to .md format
    const content = this.serialiseAgentSpec(name, tools, prompt);
    if (this.elements.agentSpecEditor) this.elements.agentSpecEditor.value = content;
    if (this.elements.agentSpecSave) {
      this.elements.agentSpecSave.disabled = true;
    }
    this.setAgentSpecStatus('Saving...');
    try {
      const response = await API.saveAgentSpec({
        content,
        mode: this.agentSpecMode || 'create',
        existing_name: this.agentSpecOriginalName || this.agentSpecName || null,
      });
      if (this.agentSpecMode === 'edit') {
        const savedName = Utils.normalizeAgentName(response?.name || this.agentSpecOriginalName || this.agentSpecName || '');
        if (savedName) {
          this.agentSpecName = savedName;
          this.agentSpecOriginalName = savedName;
        }
        if (Utils.normalizeAgentName(Chat.state.activeAgentName) === Utils.normalizeAgentName(savedName)) {
          await Chat.setActiveAgent(savedName);
        }
      }
      this.setAgentSpecStatus('Saved agent spec.', 'success');
      await Chat.loadAgents();
    } catch (e) {
      console.error('Failed to save agent spec:', e);
      this.setAgentSpecStatus(e.message || 'Unable to save agent spec.', 'error');
    } finally {
      if (this.elements.agentSpecSave) {
        this.elements.agentSpecSave.disabled = false;
      }
    }
  },

  /** Resize handle logic for agent spec modal */
  initAgentSpecResize() {
    const handle = this.elements.agentSpecResizeHandle;
    const panel = this.elements.agentSpecPanel;
    if (!handle || !panel) return;
    let startX, startY, startW, startH;
    const onMouseMove = (e) => {
      const newW = Math.max(480, startW + (e.clientX - startX));
      const newH = Math.max(400, startH + (e.clientY - startY));
      panel.style.width = newW + 'px';
      panel.style.maxWidth = newW + 'px';
      panel.style.maxHeight = newH + 'px';
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      // Persist size
      localStorage.setItem('archi_agent_spec_width', panel.style.width);
      localStorage.setItem('archi_agent_spec_height', panel.style.maxHeight);
    };
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      startX = e.clientX;
      startY = e.clientY;
      startW = panel.offsetWidth;
      startH = panel.offsetHeight;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    });
  },

  restoreAgentSpecSize() {
    const panel = this.elements.agentSpecPanel;
    if (!panel) return;
    const w = localStorage.getItem('archi_agent_spec_width');
    const h = localStorage.getItem('archi_agent_spec_height');
    if (w) { panel.style.width = w; panel.style.maxWidth = w; }
    if (h) { panel.style.maxHeight = h; }
  },

  toggleSidebar() {
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
      // On mobile, toggle sidebar-open to show/hide the overlay sidebar
      this.elements.app?.classList.toggle('sidebar-open');
    } else {
      // On desktop, toggle sidebar-collapsed to collapse the sidebar
      this.elements.app?.classList.toggle('sidebar-collapsed');
    }
    // Update aria-expanded state
    const toggle = this.elements.sidebarToggle;
    if (toggle) {
      const isOpen = isMobile 
        ? this.elements.app?.classList.contains('sidebar-open')
        : !this.elements.app?.classList.contains('sidebar-collapsed');
      toggle.setAttribute('aria-expanded', isOpen);
    }
  },

  closeSidebar() {
    // Close the sidebar on mobile (called by overlay click)
    this.elements.app?.classList.remove('sidebar-open');
    const toggle = this.elements.sidebarToggle;
    if (toggle) {
      toggle.setAttribute('aria-expanded', 'false');
    }
  },

  isABEnabled() {
    // A/B mode is active when the server reports this user is eligible
    return Chat.state.abPool?.enabled === true;
  },

  getABDisclosureMode() {
    return this.normalizeABDisclosureMode(
      Chat.state.abPool?.variant_label_mode ?? Chat.state.abPool?.disclosure_mode
    );
  },

  getABTraceMode() {
    return this.normalizeABTraceMode(
      Chat.state.abPool?.activity_panel_default_state ?? Chat.state.abPool?.default_trace_mode
    );
  },

  normalizeABDisclosureMode(mode) {
    if (mode === 'reveal_after_vote') return 'post_vote_reveal';
    if (mode === 'show_during_streaming') return 'always_visible';
    return ['hidden', 'post_vote_reveal', 'always_visible'].includes(mode)
      ? mode
      : 'post_vote_reveal';
  },

  normalizeABTraceMode(mode) {
    return ['hidden', 'collapsed', 'expanded'].includes(mode)
      ? mode
      : 'hidden';
  },

  isTraceVisibleMode(mode) {
    return !['minimal', 'hidden'].includes(mode);
  },

  isTraceCollapsedMode(mode) {
    return ['normal', 'collapsed'].includes(mode);
  },

  isTraceExpandedMode(mode) {
    return ['verbose', 'expanded'].includes(mode);
  },

  shouldUseABForNextTurn() {
    if (!this.isABEnabled()) return false;
    return true;
  },

  autoResizeInput() {
    const field = this.elements.inputField;
    if (!field) return;
    field.style.height = 'auto';
    field.style.height = Math.min(field.scrollHeight, 200) + 'px';
  },

  getInputValue() {
    return this.elements.inputField?.value.trim() ?? '';
  },

  clearInput() {
    if (this.elements.inputField) {
      this.elements.inputField.value = '';
      this.elements.inputField.style.height = 'auto';
    }
  },

  setInputDisabled(disabled, options = {}) {
    const { disableSend = disabled } = options;
    if (this.elements.inputField) this.elements.inputField.disabled = disabled;
    if (this.elements.sendBtn) this.elements.sendBtn.disabled = disableSend;
  },

  setStreamingState(isStreaming) {
    const sendBtn = this.elements.sendBtn;
    if (!sendBtn) return;

    if (isStreaming) {
      sendBtn.classList.add('stop-mode');
      sendBtn.title = 'Stop streaming';
      sendBtn.setAttribute('aria-label', 'Stop streaming');
      sendBtn.innerHTML = '⏹';
    } else {
      sendBtn.classList.remove('stop-mode');
      sendBtn.title = 'Send message';
      sendBtn.setAttribute('aria-label', 'Send message');
      sendBtn.innerHTML = this.sendBtnDefaultHtml;
    }
  },

  showCustomModelInput(show) {
    if (!this.elements.customModelRow) return;
    this.elements.customModelRow.style.display = show ? 'flex' : 'none';
  },

  updateActiveModelLabel(text) {
    if (!this.elements.activeModelLabel) return;
    this.elements.activeModelLabel.textContent = text || '';
  },

  getSelectedConfig(which = 'A') {
    return Chat.state.configs?.[0]?.name || '';
  },

  renderConfigs(configs) {
    // Config selector removed from UI; keep configs in state only.
  },

  renderProviders(providers, selectedProvider = null) {
    const select = this.elements.providerSelect;
    if (!select) return;

    // Filter to only enabled providers
    const enabledProviders = providers.filter(p => p.enabled);
    
    if (enabledProviders.length === 0) {
      select.innerHTML = '<option value="">No providers available</option>';
      select.disabled = true;
      return;
    }

    select.disabled = false;
    select.innerHTML = '<option value="">Use pipeline default</option>' +
      enabledProviders
        .map(p => `<option value="${Utils.escapeHtml(p.type)}">${Utils.escapeHtml(p.display_name)}</option>`)
        .join('');

    // Restore selection if provided, otherwise default to pipeline config
    if (selectedProvider && enabledProviders.some(p => p.type === selectedProvider)) {
      select.value = selectedProvider;
    } else {
      select.value = '';
    }

  },

  renderProviderModels(models, selectedModel = null, providerType = null) {
    const select = this.elements.modelSelectPrimary;
    if (!select) return;

    if (!models || models.length === 0) {
      select.innerHTML = '<option value="">Using pipeline default</option>';
      select.disabled = true;
      this.showCustomModelInput(false);
      return;
    }

    select.disabled = false;
    const options = models
      .map(m => `<option value="${Utils.escapeHtml(m.id)}">${Utils.escapeHtml(m.display_name || m.name)}</option>`)
      .join('');
    const customOption = providerType === 'openrouter'
      ? '<option value="__custom__">Custom model…</option>'
      : '';
    select.innerHTML = options + customOption;

    // Restore selection if provided
    if (selectedModel === '__custom__' && providerType === 'openrouter') {
      select.value = '__custom__';
      this.showCustomModelInput(true);
    } else if (selectedModel && models.some(m => m.id === selectedModel)) {
      select.value = selectedModel;
      this.showCustomModelInput(false);
    } else {
      this.showCustomModelInput(false);
    }
  },



  updateProviderStatus(status, message) {
    const statusEl = this.elements.providerStatus;
    if (!statusEl) return;

    statusEl.className = `provider-status ${status}`;
    statusEl.style.display = 'flex';
    statusEl.querySelector('.status-text').textContent = message;
  },

  hideProviderStatus() {
    const statusEl = this.elements.providerStatus;
    if (statusEl) {
      statusEl.style.display = 'none';
    }
  },

  renderApiKeyStatus(providers) {
    const container = document.getElementById('api-keys-container');
    if (!container) return;

    if (!providers || providers.length === 0) {
      container.innerHTML = '<div class="api-key-loading">No providers requiring API keys</div>';
      return;
    }

    container.innerHTML = providers.map(p => {
      const statusClass = p.configured ? 'configured' : 'not-configured';
      const statusIcon = p.configured ? '✓' : '○';
      const statusText = p.configured 
        ? (p.has_session_key ? 'Session' : 'Env')
        : '';
      
      return `
        <div class="api-key-row" data-provider="${Utils.escapeHtml(p.provider)}">
          <div class="api-key-provider">${Utils.escapeHtml(p.display_name)}</div>
          <div class="api-key-status ${statusClass}" title="${p.configured ? (p.has_session_key ? 'Session key configured' : 'Environment key configured') : 'Not configured'}">
            <span class="status-dot">${statusIcon}</span>
            ${statusText ? `<span class="status-label">${statusText}</span>` : ''}
          </div>
          <input type="password" 
                 class="api-key-input" 
                 placeholder="${p.configured ? '••••••••' : 'sk-...'}" 
                 data-provider="${Utils.escapeHtml(p.provider)}"
                 autocomplete="off">
          <div class="api-key-actions">
            <button class="api-key-btn save-btn" 
                    data-provider="${Utils.escapeHtml(p.provider)}"
                    data-action="save"
                    title="Save API key">
              Save
            </button>
            ${p.has_session_key ? `
              <button class="api-key-btn clear-btn" 
                      data-provider="${Utils.escapeHtml(p.provider)}"
                      data-action="clear"
                      title="Clear session key">
                ✕
              </button>
            ` : ''}
          </div>
        </div>
      `;
    }).join('');

    // Add event listeners for save/clear buttons
    container.querySelectorAll('.api-key-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const provider = btn.dataset.provider;
        const action = btn.dataset.action;
        const row = btn.closest('.api-key-row');
        const input = row.querySelector('.api-key-input');

        if (action === 'save') {
          const apiKey = input.value.trim();
          if (!apiKey) {
            input.focus();
            return;
          }
          
          btn.disabled = true;
          btn.textContent = 'Saving...';
          
          try {
            await Chat.setApiKey(provider, apiKey);
            input.value = '';
          } catch (err) {
            alert(`Failed to save API key: ${err.message}`);
          } finally {
            btn.disabled = false;
            btn.textContent = 'Save';
          }
        } else if (action === 'clear') {
          if (confirm(`Clear API key for ${provider}?`)) {
            btn.disabled = true;
            btn.textContent = 'Clearing...';
            
            try {
              await Chat.clearApiKey(provider);
            } catch (err) {
              alert(`Failed to clear API key: ${err.message}`);
            } finally {
              btn.disabled = false;
              btn.textContent = 'Clear';
            }
          }
        }
      });
    });

    // Allow Enter key to save
    container.querySelectorAll('.api-key-input').forEach(input => {
      input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          const row = input.closest('.api-key-row');
          const saveBtn = row.querySelector('.api-key-btn.save-btn');
          if (saveBtn) saveBtn.click();
        }
      });
    });
  },

  renderConversations(conversations, activeId) {
    const list = this.elements.conversationList;
    if (!list) return;

    if (!conversations.length) {
      list.innerHTML = `
        <div class="conversation-item" style="color: var(--text-tertiary); cursor: default;">
          No conversations yet
        </div>`;
      return;
    }

    const groups = Utils.groupByDate(conversations);
    let html = '';

    for (const [label, items] of Object.entries(groups)) {
      if (!items.length) continue;
      
      html += `<div class="conversation-group">
        <div class="conversation-group-label">${label}</div>`;
      
      for (const conv of items) {
        const isActive = conv.conversation_id === activeId;
        const title = Utils.escapeHtml(conv.title || `Conversation ${conv.conversation_id}`);
        
        html += `
          <div class="conversation-item ${isActive ? 'active' : ''}" 
               data-id="${conv.conversation_id}">
            <svg class="conversation-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <span class="conversation-item-title">${title}</span>
            <button class="conversation-item-delete" data-id="${conv.conversation_id}" aria-label="Delete conversation" title="Delete conversation">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
          </div>`;
      }
      
      html += '</div>';
    }

    list.innerHTML = html;

    // Bind click events
    list.querySelectorAll('.conversation-item').forEach((item) => {
      item.addEventListener('click', (e) => {
        if (e.target.closest('.conversation-item-delete')) return;
        const id = Number(item.dataset.id);
        Chat.loadConversation(id);
      });
    });

    list.querySelectorAll('.conversation-item-delete').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = Number(btn.dataset.id);
        Chat.deleteConversation(id);
      });
    });
  },

  renderMessages(messages) {
    const container = this.elements.messagesInner;
    if (!container) return;

    if (!messages.length) {
      container.innerHTML = `
        <div class="messages-empty">
          <img class="messages-empty-logo" src="/static/images/archi-logo.png" alt="archi logo">
          <h2 class="messages-empty-title">How can I help you today?</h2>
          <p class="messages-empty-subtitle">Ask me anything about CMS Computing Operations. I'm here to assist you.</p>
        </div>`;
      return;
    }

    container.innerHTML = messages.map((msg) => this.createMessageHTML(msg)).join('');
    this.scrollToBottom();
  },

  createMessageHTML(msg) {
    const isUser = msg.sender === 'User';
    const avatar = isUser 
      ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>'
      : '<img class="assistant-logo" src="/static/images/archi-logo.png" alt="archi logo">';
    const senderName = isUser ? 'You' : 'archi';
    const roleClass = isUser ? 'user' : 'assistant';
    
    let labelHtml = '';
    if (msg.label) {
      labelHtml = `<span class="message-label">${Utils.escapeHtml(msg.label)}</span>`;
    }

    // Chip marking a user turn that applied a saved playbook (live + on reload).
    let playbookChipHtml = '';
    if (msg.playbookName) {
      playbookChipHtml = `<span class="message-playbook-chip" title="Applied playbook">⚡ ${Utils.escapeHtml(msg.playbookName)}</span>`;
    }

    const metaHtml = !isUser && msg.meta
      ? `<div class="message-meta">${Utils.escapeHtml(msg.meta)}</div>`
      : '';

    // Determine feedback state class
    let feedbackClass = '';
    if (msg.feedback === 'like') {
      feedbackClass = 'feedback-like-active';
    } else if (msg.feedback === 'dislike') {
      feedbackClass = 'feedback-dislike-active';
    }

    return `
      <div class="message ${roleClass}" data-id="${msg.id || ''}">
        <div class="message-inner">
          <div class="message-header">
            <div class="message-avatar">${avatar}</div>
            <span class="message-sender">${senderName}</span>
            ${labelHtml}
            ${playbookChipHtml}
          </div>
          <div class="message-content">${msg.html || ''}</div>
          ${metaHtml}
          ${!isUser ? `
          <div class="message-actions ${feedbackClass}">
            <button class="feedback-btn feedback-like" onclick="UI.handleFeedback(this, 'like')" aria-label="Helpful" title="Helpful">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
              </svg>
            </button>
            <button class="feedback-btn feedback-dislike" onclick="UI.handleFeedback(this, 'dislike')" aria-label="Not helpful" title="Not helpful">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
              </svg>
            </button>
            <button class="feedback-btn feedback-comment" onclick="UI.handleFeedback(this, 'comment')" aria-label="Add comment" title="Add comment">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              </svg>
            </button>
          </div>` : ''}
        </div>
      </div>`;
  },

  addMessage(msg) {
    // Remove empty state if present
    const empty = this.elements.messagesInner?.querySelector('.messages-empty');
    if (empty) empty.remove();

    const html = this.createMessageHTML(msg);
    this.elements.messagesInner?.insertAdjacentHTML('beforeend', html);
    this.scrollToBottom();
  },

  updateMessage(id, updates) {
    const msgEl = this.elements.messagesInner?.querySelector(`[data-id="${id}"]`);
    if (!msgEl) return;

    const contentEl = msgEl.querySelector('.message-content');
    if (contentEl && updates.html !== undefined) {
      contentEl.innerHTML = updates.html;
      if (updates.streaming) {
        contentEl.innerHTML += '<span class="streaming-cursor"></span>';
      }
    }

    if (updates.meta !== undefined) {
      const metaEl = msgEl.querySelector('.entry-meta');
      if (metaEl) metaEl.textContent = updates.meta;
    }

    this.scrollToBottom();
  },

  showTypingIndicator() {
    const html = `
      <div class="typing-indicator">
        <div class="typing-indicator-inner">
          <div class="typing-dots">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>`;
    this.elements.messagesInner?.insertAdjacentHTML('beforeend', html);
    this.scrollToBottom();
  },

  hideTypingIndicator() {
    this.elements.messagesInner?.querySelector('.typing-indicator')?.remove();
  },

  scrollToBottom() {
    const container = this.elements.messagesContainer;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  },

  // =========================================================================
  // A/B Testing UI Methods
  // =========================================================================

  setABSectionVisible(visible) {
    if (this.elements.abAdminLinkSection) {
      this.elements.abAdminLinkSection.style.display = visible ? '' : 'none';
    }
  },

  setABSettingsVisible(visible) {
    if (this.elements.abSettingsNav) {
      this.elements.abSettingsNav.style.display = visible ? '' : 'none';
    }
    if (this.elements.abSettingsSection && !visible) {
      this.elements.abSettingsSection.hidden = true;
      this.elements.abSettingsSection.classList.remove('active');
    }
  },

  updateABParticipationPreview(value) {
    if (this.elements.abParticipationValue) {
      this.elements.abParticipationValue.textContent = `${Math.round(value)}%`;
    }
  },

  updateABSettingsSection() {
    const abState = Chat.state.abPool || {};
    const capabilities = Chat.state.abCapabilities || {};
    const currentUser = Chat.state.currentUser || {};
    const preferenceSaveState = Chat.state.abPreferenceSaveState || null;
    const canParticipate = capabilities.canParticipate === true;
    const canViewAdmin = capabilities.canView === true;
    const shouldShow = canParticipate || canViewAdmin;

    this.setABSettingsVisible(shouldShow);
    this.setABSectionVisible(canViewAdmin);

    if (this.elements.abParticipationGroup) {
      this.elements.abParticipationGroup.style.display = canParticipate ? '' : 'none';
    }
    if (!canParticipate) {
      return;
    }

    const defaultRate = Number(
      abState.default_comparison_rate
      ?? abState.default_sample_rate
      ?? abState.comparison_rate
      ?? abState.sample_rate
      ?? 1
    );
    const usingDefault = currentUser.ab_participation_rate == null || Number.isNaN(Number(currentUser.ab_participation_rate));
    const effectiveRate = usingDefault ? defaultRate : Number(currentUser.ab_participation_rate);
    const percent = Math.max(0, Math.min(100, Math.round(effectiveRate * 100)));

    if (this.elements.abParticipationSlider) {
      this.elements.abParticipationSlider.value = String(percent);
    }
    this.updateABParticipationPreview(percent);

    if (this.elements.abParticipationDefault) {
      this.elements.abParticipationDefault.textContent = `Default: ${Math.round(defaultRate * 100)}%`;
    }
    if (this.elements.abParticipationNote) {
      if (preferenceSaveState?.type === 'error') {
        this.elements.abParticipationNote.textContent = preferenceSaveState.message || 'Your last change was not saved.';
        this.elements.abParticipationNote.classList.add('settings-inline-error');
      } else {
        this.elements.abParticipationNote.textContent = preferenceSaveState?.type === 'success'
          ? (preferenceSaveState.message || 'Saved for your account.')
          : (usingDefault
            ? 'Currently using the deployment default until you choose your own rate.'
            : 'Your saved setting applies only to your account.');
        this.elements.abParticipationNote.classList.remove('settings-inline-error');
      }
    }
    if (this.elements.abParticipationInactive) {
      const reason = String(abState.participant_reason || '');
      let inactiveMessage = '';
      if (reason === 'not_targeted') {
        inactiveMessage = 'The current experiment does not target your role or permissions. Your saved rate will apply automatically if a future experiment includes you.';
      } else if (reason === 'disabled') {
        inactiveMessage = 'Experiments are currently inactive. Your preference will be used again if A/B testing is enabled.';
      }
      this.elements.abParticipationInactive.textContent = inactiveMessage;
      this.elements.abParticipationInactive.style.display = inactiveMessage ? '' : 'none';
    }
  },

  updateABPoolUI(poolInfo) {
    // Render pool editor with current agents + pool state
    const agentList = document.getElementById('ab-pool-agent-list');
    const statusBadge = document.getElementById('ab-pool-status');
    const disableBtn = document.getElementById('ab-pool-disable');
    const sampleRateInput = document.getElementById('ab-sample-rate');
    const disclosureModeInput = document.getElementById('ab-disclosure-mode');
    const traceModeInput = document.getElementById('ab-trace-mode');
    if (!agentList) return;

    // Use allAgents so AB-only variants appear in the pool editor
    const agents = Chat.state.allAgents || Chat.state.agents || [];
    const poolEnabled = poolInfo?.enabled === true;
    const currentChampion = poolInfo?.champion || poolInfo?.control || null;
    const currentVariants = poolInfo?.variants || [];

    // Update status badge
    if (statusBadge) {
      statusBadge.textContent = poolEnabled ? 'Active' : 'Inactive';
      statusBadge.classList.toggle('active', poolEnabled);
    }

    // Show/hide disable button
    if (disableBtn) {
      disableBtn.style.display = poolEnabled ? '' : 'none';
    }
    if (sampleRateInput) {
      sampleRateInput.value = String(poolInfo?.comparison_rate ?? poolInfo?.sample_rate ?? 1);
    }
    if (disclosureModeInput) {
      disclosureModeInput.value = poolInfo?.variant_label_mode || poolInfo?.disclosure_mode || 'post_vote_reveal';
    }
    if (traceModeInput) {
      traceModeInput.value = poolInfo?.activity_panel_default_state || poolInfo?.default_trace_mode || 'hidden';
    }

    // Render agent rows
    agentList.innerHTML = agents.map(agent => {
      const inPool = currentVariants.includes(agent.name);
      const isChampion = agent.name === currentChampion;
      const selectedClass = inPool ? ' selected' : '';
      const championClass = isChampion ? ' champion' : '';
      const isABOnly = agent.ab_only === true;
      return `
        <label class="ab-pool-agent-row${selectedClass}${championClass}" data-agent="${Utils.escapeHtml(agent.name)}">
          <span class="ab-pool-agent-check">
            <input type="checkbox" ${inPool ? 'checked' : ''}>
          </span>
          <span class="ab-pool-agent-name">
            ${Utils.escapeHtml(agent.name)}
            ${isABOnly ? '<span class="ab-pool-ab-badge">AB</span>' : ''}
          </span>
          <span class="ab-pool-agent-actions">
            <button type="button" class="ab-pool-variant-btn" title="Create variant of this agent">+</button>
            <button type="button" class="ab-pool-champion-btn${isChampion ? ' is-champion' : ''}" title="Set as champion">
              ${isChampion ? '★ Champion' : '☆ Champion'}
            </button>
          </span>
        </label>`;
    }).join('');

    // Wire up events
    agentList.querySelectorAll('.ab-pool-agent-row').forEach(row => {
      const agentName = row.dataset.agent;
      const checkbox = row.querySelector('input[type="checkbox"]');
      const champBtn = row.querySelector('.ab-pool-champion-btn');
      const variantBtn = row.querySelector('.ab-pool-variant-btn');

      checkbox.addEventListener('change', () => {
        row.classList.toggle('selected', checkbox.checked);
        if (!checkbox.checked && row.classList.contains('champion')) {
          row.classList.remove('champion');
          champBtn.classList.remove('is-champion');
          champBtn.innerHTML = '☆ Champion';
        }
        this._updateABPoolSaveState();
      });

      champBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!checkbox.checked) return;
        // Clear previous champion
        agentList.querySelectorAll('.ab-pool-agent-row').forEach(r => {
          r.classList.remove('champion');
          const btn = r.querySelector('.ab-pool-champion-btn');
          btn.classList.remove('is-champion');
          btn.innerHTML = '☆ Champion';
        });
        row.classList.add('champion');
        champBtn.classList.add('is-champion');
        champBtn.innerHTML = '★ Champion';
        this._updateABPoolSaveState();
      });

      if (variantBtn) {
        variantBtn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          this._showQuickVariantPanel(agentName);
        });
      }
    });

    this._updateABPoolSaveState();
  },

  _updateABPoolSaveState() {
    const agentList = document.getElementById('ab-pool-agent-list');
    const saveBtn = document.getElementById('ab-pool-save');
    const msgEl = document.getElementById('ab-pool-message');
    if (!agentList || !saveBtn) return;

    const selected = agentList.querySelectorAll('.ab-pool-agent-row.selected');
    const hasChampion = !!agentList.querySelector('.ab-pool-agent-row.champion');
    const valid = selected.length >= 2 && hasChampion;
    saveBtn.disabled = !valid;

    if (msgEl) {
      if (selected.length < 2 && selected.length > 0) {
        msgEl.textContent = 'Select at least 2 agents';
        msgEl.className = 'ab-pool-message error';
      } else if (selected.length >= 2 && !hasChampion) {
        msgEl.textContent = 'Click "Champion" to designate the baseline variant';
        msgEl.className = 'ab-pool-message error';
      } else {
        msgEl.textContent = '';
        msgEl.className = 'ab-pool-message';
      }
    }
  },

  _getABPoolSelection() {
    const agentList = document.getElementById('ab-pool-agent-list');
    if (!agentList) return null;
    const variants = [];
    let champion = null;
    agentList.querySelectorAll('.ab-pool-agent-row.selected').forEach(row => {
      const name = row.dataset.agent;
      variants.push(name);
      if (row.classList.contains('champion')) champion = name;
    });
    return { champion, variants };
  },

  /**
   * Show an inline panel to quickly create a variant of an existing agent.
   * The variant is saved with ab_only: true so it only appears in the pool editor.
   */
  async _showQuickVariantPanel(sourceAgentName) {
    // Remove any existing panel
    document.querySelector('.ab-quick-variant-panel')?.remove();

    // Fetch source agent spec and tool palette in parallel
    let sourceSpec = null;
    let availableTools = [];
    try {
      const [specResp, templateResp] = await Promise.all([
        API.getAgentSpec(sourceAgentName),
        API.getAgentTemplate(),
      ]);
      sourceSpec = UI.parseAgentSpec(specResp?.content || '');
      availableTools = (templateResp?.tools || []).map(t => typeof t === 'string' ? t : t.name);
    } catch (e) {
      console.error('Failed to load source agent for variant:', e);
      return;
    }

    const sourceTools = sourceSpec?.tools || [];

    // Build panel HTML
    const panel = document.createElement('div');
    panel.className = 'ab-quick-variant-panel';
    panel.innerHTML = `
      <div class="ab-qv-header">
        <strong>New variant of "${Utils.escapeHtml(sourceAgentName)}"</strong>
        <button class="ab-qv-close" title="Cancel">&times;</button>
      </div>
      <label class="ab-qv-label">Variant name</label>
      <input class="ab-qv-name" type="text" value="${Utils.escapeHtml(sourceAgentName)}-v2" spellcheck="false">
      <label class="ab-qv-label">Tools</label>
      <div class="ab-qv-tools">
        ${availableTools.map(t => `
          <label class="ab-qv-tool">
            <input type="checkbox" value="${Utils.escapeHtml(t)}" ${sourceTools.includes(t) ? 'checked' : ''}>
            ${Utils.escapeHtml(t)}
          </label>
        `).join('')}
      </div>
      <div class="ab-qv-footer">
        <span class="ab-qv-msg"></span>
        <button class="ab-qv-save">Create Variant</button>
      </div>
    `;

    // Insert panel after the agent list
    const agentList = document.getElementById('ab-pool-agent-list');
    agentList?.parentElement?.insertBefore(panel, agentList.nextSibling);

    // References
    const nameInput = panel.querySelector('.ab-qv-name');
    const saveBtn = panel.querySelector('.ab-qv-save');
    const closeBtn = panel.querySelector('.ab-qv-close');
    const msgEl = panel.querySelector('.ab-qv-msg');

    closeBtn.addEventListener('click', () => panel.remove());

    nameInput.addEventListener('input', () => {
      if (msgEl) { msgEl.textContent = ''; msgEl.className = 'ab-qv-msg'; }
    });

    saveBtn.addEventListener('click', async () => {
      const variantName = (nameInput.value || '').trim();
      if (!variantName) {
        if (msgEl) { msgEl.textContent = 'Name is required'; msgEl.className = 'ab-qv-msg error'; }
        nameInput.focus();
        return;
      }

      // Client-side duplicate check
      const existingNames = (Chat.state.allAgents || []).map(a => a.name);
      if (existingNames.includes(variantName)) {
        if (msgEl) { msgEl.textContent = '"' + variantName + '" already exists \u2014 choose a different name'; msgEl.className = 'ab-qv-msg error'; }
        nameInput.focus();
        nameInput.select();
        return;
      }

      const selectedTools = [...panel.querySelectorAll('.ab-qv-tools input:checked')].map(cb => cb.value);
      const specContent = UI.serialiseAgentSpec(variantName, selectedTools, sourceSpec?.prompt || '', { ab_only: true });

      saveBtn.disabled = true;
      saveBtn.textContent = 'Saving\u2026';
      if (msgEl) { msgEl.textContent = ''; msgEl.className = 'ab-qv-msg'; }

      try {
        const result = await API.saveAgentSpec({ content: specContent, mode: 'create' });
        if (result?.success) {
          if (msgEl) { msgEl.textContent = 'Created!'; msgEl.className = 'ab-qv-msg success'; }
          // Refresh agent lists + re-render pool editor
          await Chat.loadAgents();
          UI.updateABPoolUI(Chat.state.abPool || {});
          // Panel replaced by re-render; remove just in case
          setTimeout(() => panel.remove(), 600);
        } else {
          if (msgEl) { msgEl.textContent = result?.error || 'Save failed'; msgEl.className = 'ab-qv-msg error'; }
          saveBtn.disabled = false;
          saveBtn.textContent = 'Create Variant';
        }
      } catch (e) {
        if (msgEl) { msgEl.textContent = e.message || 'Save failed'; msgEl.className = 'ab-qv-msg error'; }
        saveBtn.disabled = false;
        saveBtn.textContent = 'Create Variant';
      }
    });

    // Focus the name input
    nameInput.focus();
    nameInput.select();
  },

  showABWarningModal(onConfirm, onCancel) {
    // Prevent duplicate modals
    if (document.getElementById('ab-warning-modal')) {
      return;
    }
    
    const modalHtml = `
      <div class="ab-warning-modal-overlay" id="ab-warning-modal">
        <div class="ab-warning-modal">
          <div class="ab-warning-modal-header">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <h3>Enable A/B Testing Mode</h3>
          </div>
          <div class="ab-warning-modal-body">
            <p>This will compare two AI responses for each message.</p>
            <ul>
              <li><strong>2× API usage</strong> - Each message generates two responses</li>
              <li><strong>Voting required</strong> - Once the pending comparison limit is reached, you must resolve one before continuing</li>
              <li>You can disable A/B mode at any time to skip voting</li>
            </ul>
          </div>
          <div class="ab-warning-modal-actions">
            <button class="ab-warning-btn ab-warning-btn-cancel">Cancel</button>
            <button class="ab-warning-btn ab-warning-btn-confirm">Enable A/B Mode</button>
          </div>
        </div>
      </div>`;

    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = document.getElementById('ab-warning-modal');

    const closeModal = () => modal?.remove();

    modal.querySelector('.ab-warning-btn-cancel').addEventListener('click', () => {
      closeModal();
      onCancel?.();
    });

    modal.querySelector('.ab-warning-btn-confirm').addEventListener('click', () => {
      closeModal();
      onConfirm?.();
    });

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeModal();
        onCancel?.();
      }
    });
  },

  showToast(message, duration = 3000) {
    // Remove existing toast
    document.querySelector('.toast')?.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => toast.classList.add('show'));

    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, duration);
  },

  getTraceModeForMessage(messageId) {
    const container = document.querySelector(`.trace-container[data-message-id="${messageId}"]`);
    return container?.dataset.traceMode || Chat.state.traceVerboseMode || 'normal';
  },

  getTraceIconSvg() {
    return `<svg class="trace-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg>`;
  },

  getTraceLabelText(toolCount = 0) {
    return toolCount > 0
      ? `Agent Activity (${toolCount} tool${toolCount === 1 ? '' : 's'})`
      : 'Agent Activity';
  },

  bindTraceToggleHandlers(root = document) {
    if (!root?.querySelectorAll) return;
    root.querySelectorAll('[data-trace-toggle]').forEach(el => {
      if (!el._traceToggleBound) {
        el._traceToggleBound = true;
        el.addEventListener('click', () => UI.toggleTraceExpanded(el.dataset.traceToggle));
      }
    });
  },

  addABComparisonContainer(msgIdA, msgIdB, options = {}) {
    // Remove empty state if present
    const empty = this.elements.messagesInner?.querySelector('.messages-empty');
    if (empty) empty.remove();

    const traceMode = this.normalizeABTraceMode(options.traceMode || this.getABTraceMode());
    const showTrace = this.isTraceVisibleMode(traceMode);
    const traceIconSvg = this.getTraceIconSvg();
    const traceCollapsed = this.isTraceCollapsedMode(traceMode);
    const traceHtml = (id) => showTrace ? `
          <div class="trace-container ab-trace-container${traceCollapsed ? ' collapsed' : ''}" data-message-id="${id}" data-trace-mode="${traceMode}">
            <div class="trace-header">
              ${traceIconSvg}
              <span class="trace-label">${this.getTraceLabelText()}</span>
              <span class="trace-timer" data-start="${Date.now()}">0.0s</span>
              <button class="trace-toggle" data-trace-toggle="${id}" aria-label="Toggle agent activity details" title="Toggle agent activity">
                <span class="toggle-icon" aria-hidden="true">${traceCollapsed ? '&#9654;' : '&#9660;'}</span>
              </button>
            </div>
            <div class="trace-content">
              <div class="step-timeline"></div>
            </div>
          </div>` : '';

    // Use normal message structure for each arm — looks like two regular chat messages side by side
    const armHtml = (id, label) => `
        <div class="message assistant ab-arm" data-id="${id}">
          <div class="message-inner">
            <div class="message-header">
              <div class="message-avatar"><img class="assistant-logo" src="/static/images/archi-logo.png" alt="archi logo"></div>
              <div class="ab-arm-header-copy">
                <div class="ab-arm-title-row">
                  <span class="message-sender">archi</span>
                  <span class="message-label ab-arm-label">${label}</span>
                </div>
                <span class="ab-arm-variant-name" data-arm-id="${id}"></span>
              </div>
            </div>
            ${traceHtml(id)}
            <div class="message-content"></div>
            <div class="message-meta" style="display: none;"></div>
            <div class="message-actions">
              <button class="feedback-btn feedback-like" onclick="UI.handleFeedback(this, 'like')" aria-label="Helpful" title="Helpful">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
              </button>
              <button class="feedback-btn feedback-dislike" onclick="UI.handleFeedback(this, 'dislike')" aria-label="Not helpful" title="Not helpful">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>
              </button>
              <button class="feedback-btn feedback-comment" onclick="UI.handleFeedback(this, 'comment')" aria-label="Add comment" title="Add comment">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
              </button>
            </div>
          </div>
        </div>`;

    const comparisonKey = Utils.escapeAttr(options.comparisonKey || `${msgIdA}-${msgIdB}`);
    const html = `
      <div class="ab-comparison" data-comparison-key="${comparisonKey}">
        ${armHtml(msgIdA, 'Response A')}
        ${armHtml(msgIdB, 'Response B')}
      </div>`;

    this.elements.messagesInner?.insertAdjacentHTML('beforeend', html);
    // Bind trace toggle handlers (replacing inline onclick for CSP compliance)
    this.bindTraceToggleHandlers(this.elements.messagesInner || document);
    // Start timers for both A/B arms
    if (showTrace) {
      this.startTraceTimer(msgIdA);
      this.startTraceTimer(msgIdB);
    }
    this.scrollToBottom();
  },

  findABComparisonElement(comparisonState = null) {
    if (!comparisonState) return null;

    const comparisonId = comparisonState.comparisonId;
    if (comparisonId != null) {
      const byId = document.querySelector(`.ab-comparison[data-comparison-id="${comparisonId}"]`);
      if (byId) return byId;
    }

    const armIds = [
      comparisonState.responseAUiId,
      comparisonState.responseBUiId,
      comparisonState.responseAId,
      comparisonState.responseBId,
    ].filter(Boolean);

    for (const armId of armIds) {
      const arm = document.querySelector(`.ab-arm[data-id="${armId}"], .ab-response[data-id="${armId}"]`);
      if (arm) {
        const container = arm.closest('.ab-comparison');
        if (container) return container;
      }
    }
    return null;
  },

  setABComparisonId(comparisonState = null) {
    const comparison = this.findABComparisonElement(comparisonState);
    if (!comparison || !comparisonState?.comparisonId) return;
    comparison.dataset.comparisonId = String(comparisonState.comparisonId);
  },

  updateABVariantLabel(armId, variantName) {
    const labelEl = document.querySelector(`.ab-arm-variant-name[data-arm-id="${armId}"]`);
    if (labelEl) {
      labelEl.textContent = variantName || '';
    }
  },

  updateABArmMeta(armId, metaText, visible = true) {
    const container = document.querySelector(`.ab-arm[data-id="${armId}"], .ab-response[data-id="${armId}"]`);
    const metaEl = container?.querySelector('.message-meta');
    if (!metaEl) return;
    metaEl.textContent = metaText || '';
    metaEl.style.display = visible && metaText ? '' : 'none';
  },

  updateABArmPresentation(
    armId,
    { variantName = '', modelUsed = '' } = {},
    { disclosureMode = 'post_vote_reveal', reveal = false } = {},
  ) {
    const normalizedDisclosureMode = this.normalizeABDisclosureMode(disclosureMode);
    const showVariant = normalizedDisclosureMode === 'always_visible'
      || (normalizedDisclosureMode === 'post_vote_reveal' && reveal);
    this.updateABVariantLabel(armId, showVariant ? variantName : '');
    this.updateABArmMeta(armId, modelUsed, showVariant && !!modelUsed);
  },

  rekeyABArm(oldId, newId) {
    if (!oldId || !newId || String(oldId) === String(newId)) return;

    const arm = document.querySelector(`.ab-arm[data-id="${oldId}"], .ab-response[data-id="${oldId}"]`);
    if (arm) {
      arm.dataset.id = String(newId);
    }

    const variantLabel = document.querySelector(`.ab-arm-variant-name[data-arm-id="${oldId}"]`);
    if (variantLabel) {
      variantLabel.dataset.armId = String(newId);
    }

    const traceContainer = document.querySelector(`.trace-container[data-message-id="${oldId}"]`);
    if (traceContainer) {
      traceContainer.dataset.messageId = String(newId);
      const toggle = traceContainer.querySelector('[data-trace-toggle]');
      if (toggle) {
        toggle.dataset.traceToggle = String(newId);
      }
    }

    const activeInterval = this.traceTimerIntervals.get(String(oldId));
    if (activeInterval != null) {
      this.traceTimerIntervals.set(String(newId), activeInterval);
      this.traceTimerIntervals.delete(String(oldId));
    }
  },

  updateABResponse(responseId, html, streaming = false) {
    const container = document.querySelector(`.ab-arm[data-id="${responseId}"], .ab-response[data-id="${responseId}"]`);
    if (!container) return;

    const contentEl = container.querySelector('.message-content');
    if (contentEl) {
      contentEl.innerHTML = html;
      if (streaming) {
        contentEl.innerHTML += '<span class="streaming-cursor"></span>';
      }
    }
    this.scrollToBottom();
  },

  showABVoteButtons(comparisonState) {
    const comparison = this.findABComparisonElement(comparisonState);
    if (!comparison) return;

    this.hideABVoteButtons();

    const voteHtml = `
      <div class="ab-vote-container" data-comparison-id="${comparisonState.comparisonId}">
        <div class="ab-vote-prompt">Which response do you prefer?</div>
        <div class="ab-vote-buttons">
          <button class="ab-vote-btn ab-vote-btn-a" data-vote="a">
            <span class="ab-vote-icon">👈</span>
            <span>Response A</span>
          </button>
          <button class="ab-vote-btn ab-vote-btn-tie" data-vote="tie">
            <span class="ab-vote-icon">🤝</span>
            <span>Tie</span>
          </button>
          <button class="ab-vote-btn ab-vote-btn-b" data-vote="b">
            <span class="ab-vote-icon">👉</span>
            <span>Response B</span>
          </button>
        </div>
      </div>`;

    comparison.insertAdjacentHTML('afterend', voteHtml);

    // Bind vote button events
    comparison.nextElementSibling?.querySelectorAll('.ab-vote-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const vote = btn.dataset.vote;
        Chat.submitABPreference(vote);
      });
    });

    this.scrollToBottom();
  },

  hideABVoteButtons() {
    document.querySelector('.ab-vote-container')?.remove();
  },

  stopTraceTimersInElement(container) {
    if (!container) return;
    container.querySelectorAll('.trace-container[data-message-id]').forEach((traceEl) => {
      this.stopTraceTimer(traceEl.dataset.messageId);
    });
  },

  markABWinner(preference, comparisonState = null) {
    const comparisonEl = this.findABComparisonElement(comparisonState);
    if (!comparisonEl) return;

    if (comparisonState?.responseAId) {
      this.rekeyABArm(comparisonState.responseAUiId || comparisonState.responseAId, comparisonState.responseAId);
      this.updateABArmPresentation(comparisonState.responseAId, {
        variantName: comparisonState.variantA,
        modelUsed: comparisonState.responseAModelUsed,
      }, {
        disclosureMode: comparisonState.disclosureMode,
        reveal: true,
      });
    }
    if (comparisonState?.responseBId) {
      this.rekeyABArm(comparisonState.responseBUiId || comparisonState.responseBId, comparisonState.responseBId);
      this.updateABArmPresentation(comparisonState.responseBId, {
        variantName: comparisonState.variantB,
        modelUsed: comparisonState.responseBModelUsed,
      }, {
        disclosureMode: comparisonState.disclosureMode,
        reveal: true,
      });
    }

    this.stopTraceTimersInElement(comparisonEl);

    const arms = comparisonEl.querySelectorAll('.ab-arm');
    const armA = arms[0];
    const armB = arms[1];

    if (preference === 'tie') {
      // Tie — dim both equally and add a badge
      armA?.classList.add('ab-arm-tie');
      armB?.classList.add('ab-arm-tie');
      delete comparisonEl.dataset.comparisonId;
      return;
    }

    // Winner/loser — collapse to single message
    const winner = preference === 'a' ? armA : armB;

    if (winner) {
      // Remove the AB label
      winner.querySelector('.ab-arm-label')?.remove();
      winner.classList.remove('ab-arm');
    }

    // Move the live winner node out of the comparison container so its
    // finalized timer text and bound trace interactions are preserved.
    if (winner && comparisonEl.parentNode) {
      comparisonEl.parentNode.insertBefore(winner, comparisonEl);
      comparisonEl.remove();
    }
  },

  removeABComparisonContainer(comparisonState = null) {
    const comparisonEl = comparisonState
      ? this.findABComparisonElement(comparisonState)
      : document.querySelector('.ab-comparison:last-of-type');
    this.stopTraceTimersInElement(comparisonEl);
    comparisonEl?.remove();
    this.hideABVoteButtons();
  },

  showABError(message) {
    this.removeABComparisonContainer();
    const errorHtml = `
      <div class="message assistant ab-error-message">
        <div class="message-inner">
          <div class="message-header">
            <div class="message-avatar">⚠️</div>
            <span class="message-sender">A/B Comparison Failed</span>
          </div>
          <div class="message-content">
            <p style="color: var(--error-text);">${Utils.escapeHtml(message)}</p>
            <p>Continuing in single-response mode.</p>
          </div>
        </div>
      </div>`;
    this.elements.messagesInner?.insertAdjacentHTML('beforeend', errorHtml);
    this.scrollToBottom();
  },

  // =========================================================================
  // Agent Trace Rendering
  // =========================================================================

  createTraceContainer(messageId) {
    const msgEl = this.elements.messagesInner?.querySelector(`[data-id="${messageId}"]`);
    if (!msgEl) return;

    // Insert trace container before message content
    const inner = msgEl.querySelector('.message-inner');
    if (!inner) return;

    const existingTrace = inner.querySelector('.trace-container');
    if (existingTrace) return;

    const traceIconSvg = this.getTraceIconSvg();
    const traceHtml = `
      <div class="trace-container" data-message-id="${messageId}">
        <div class="trace-header">
          ${traceIconSvg}
          <span class="trace-label">${this.getTraceLabelText()}</span>
          <span class="trace-timer" data-start="${Date.now()}">0.0s</span>
          <button class="trace-toggle" aria-label="Toggle agent activity details" title="Toggle agent activity" onclick="UI.toggleTraceExpanded('${messageId}')">
            <span class="toggle-icon" aria-hidden="true">&#9660;</span>
          </button>
        </div>
        <div class="trace-content">
          <div class="context-meter" style="display: none;" title="LLM token usage for this response. Prompt = tokens sent to the model; Completion = tokens generated back.">
            <div class="meter-bar" title="Context window usage"><div class="meter-fill"></div></div>
            <span class="meter-label"></span>
          </div>
          <div class="step-timeline"></div>
        </div>
      </div>`;

    inner.insertAdjacentHTML('afterbegin', traceHtml);

    // Start collapsed in normal mode (user can expand on demand)
    if (Chat.state.traceVerboseMode === 'normal') {
      const tc = inner.querySelector('.trace-container');
      if (tc) {
        tc.classList.add('collapsed');
        const ti = tc.querySelector('.toggle-icon');
        if (ti) ti.innerHTML = '&#9654;';
      }
    }

    // Start elapsed timer
    this.startTraceTimer(messageId);
  },

  getTraceTimerElement(messageId) {
    return document.querySelector(`.trace-container[data-message-id="${messageId}"] .trace-timer`);
  },

  getTraceElapsedMs(messageId) {
    const timerEl = this.getTraceTimerElement(messageId);
    if (!timerEl?.dataset.start) return null;
    const startTime = Number.parseInt(timerEl.dataset.start, 10);
    if (!Number.isFinite(startTime)) return null;
    return Math.max(Date.now() - startTime, 0);
  },

  startTraceTimer(messageId) {
    const timerKey = String(messageId);
    const timerEl = this.getTraceTimerElement(messageId);
    if (!timerEl) return;

    this.stopTraceTimer(timerKey);
    delete timerEl.dataset.finalDurationMs;

    const startTime = Number.parseInt(timerEl.dataset.start, 10);
    if (!Number.isFinite(startTime)) {
      timerEl.dataset.start = String(Date.now());
    }
    const updateTimer = () => {
      const baseTime = Number.parseInt(timerEl.dataset.start, 10);
      const elapsed = (Date.now() - baseTime) / 1000;
      timerEl.textContent = elapsed.toFixed(1) + 's';
    };

    updateTimer();
    const intervalId = window.setInterval(updateTimer, 100);
    this.traceTimerIntervals.set(timerKey, intervalId);
  },

  stopTraceTimer(messageId, durationMs = null) {
    const timerKey = String(messageId);
    const intervalId = this.traceTimerIntervals.get(timerKey);
    if (intervalId != null) {
      clearInterval(intervalId);
      this.traceTimerIntervals.delete(timerKey);
    }

    const timerEl = this.getTraceTimerElement(messageId);
    if (!timerEl) return;

    let resolvedDurationMs = durationMs;
    if (resolvedDurationMs == null && timerEl.dataset.finalDurationMs) {
      const storedDurationMs = Number.parseInt(timerEl.dataset.finalDurationMs, 10);
      if (Number.isFinite(storedDurationMs)) {
        resolvedDurationMs = storedDurationMs;
      }
    }

    const elapsedMs = resolvedDurationMs ?? this.getTraceElapsedMs(messageId);
    if (elapsedMs != null) {
      timerEl.dataset.finalDurationMs = String(elapsedMs);
      timerEl.textContent = Utils.formatDuration(elapsedMs);
    }
  },

  toggleTraceExpanded(messageId) {
    const container = document.querySelector(`.trace-container[data-message-id="${messageId}"]`);
    if (!container) return;

    container.classList.toggle('collapsed');
    const toggleIcon = container.querySelector('.toggle-icon');
    if (toggleIcon) {
      toggleIcon.innerHTML = container.classList.contains('collapsed') ? '&#9654;' : '&#9660;';
    }
  },

  // =========================================================================
  // Thinking Step Rendering
  // =========================================================================

  renderThinkingStart(messageId, event) {
    const timeline = document.querySelector(`.trace-container[data-message-id="${messageId}"] .step-timeline`);
    if (!timeline) return;

    const stepHtml = `
      <div class="step thinking-step" data-step-id="${event.step_id}">
        <div class="step-connector">
          <span class="step-marker thinking-marker"></span>
          <div class="step-line"></div>
        </div>
        <div class="step-content">
          <div class="step-header" onclick="UI.toggleStepExpanded('${Utils.escapeAttr(event.step_id)}')">
            <span class="step-icon">...</span>
            <span class="step-label">Thinking</span>
            <span class="step-timer">
              <span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>
            </span>
            <button class="step-toggle" aria-label="Expand thinking details">&#9654;</button>
          </div>
          <div class="step-details" style="display: none;">
            <div class="section-label">Details</div>
            <pre><code>Processing...</code></pre>
          </div>
        </div>
      </div>`;

    timeline.insertAdjacentHTML('beforeend', stepHtml);
    this.scrollToBottom();
  },

  renderThinkingEnd(messageId, event) {
    const step = document.querySelector(`.trace-container[data-message-id="${messageId}"] .thinking-step[data-step-id="${event.step_id}"]`);
    if (!step) return;

    // If no thinking content, remove the step entirely - it's just noise
    if (!event.thinking_content || !event.thinking_content.trim()) {
      step.remove();
      return;
    }

    // Has actual thinking content - show it
    step.classList.add('completed');
    const timerEl = step.querySelector('.step-timer');
    if (timerEl && event.duration_ms != null) {
      timerEl.textContent = Utils.formatDuration(event.duration_ms);
    }
    
    const details = step.querySelector('.step-details pre code');
    if (details) {
      details.textContent = event.thinking_content.trim();
    }
    
    const marker = step.querySelector('.step-marker');
    if (marker) {
      marker.classList.remove('thinking-marker');
      marker.classList.add('completed-marker');
    }
  },

  renderNarrationStep(messageId, event) {
    const timeline = document.querySelector(`.trace-container[data-message-id="${messageId}"] .step-timeline`);
    if (!timeline) return;

    const stepHtml = `
      <div class="step thinking-step completed" data-step-id="${Utils.escapeAttr(event.step_id)}">
        <div class="step-connector">
          <span class="step-marker completed-marker"></span>
          <div class="step-line"></div>
        </div>
        <div class="step-content">
          <div class="step-header" onclick="UI.toggleStepExpanded('${Utils.escapeAttr(event.step_id)}')">
            <span class="step-icon">📝</span>
            <span class="step-label">Planning</span>
            <span class="step-timer"></span>
            <button class="step-toggle" aria-label="Expand planning details">&#9654;</button>
          </div>
          <div class="step-details" style="display: none;">
            <div class="section-label">Details</div>
            <pre><code>${Utils.escapeHtml(event.content || '')}</code></pre>
          </div>
        </div>
      </div>`;

    timeline.insertAdjacentHTML('beforeend', stepHtml);
    this.scrollToBottom();
  },

  // =========================================================================
  // Tool Step Rendering (Timeline Style)
  // =========================================================================

  // A distinct activity step for the playbook that shaped this turn — shown for BOTH
  // the /name path (body injected server-side, no tool call) and auto-pickup (the
  // Playbook loader, surfaced here instead of as a generic tool row). Styled apart from
  // tool steps and NOT counted as a tool: it explains WHY the tools below it ran.
  renderPlaybookApplied(messageId, event) {
    this.createTraceContainer(messageId);  // no-op if it already exists
    const timeline = document.querySelector(`.trace-container[data-message-id="${messageId}"] .step-timeline`);
    if (!timeline || !event.name) return;
    // Key on the playbook name, not the tool_call_id: a /name turn emits one applied
    // step server-side (no id) AND, if the model redundantly re-loads the same playbook,
    // another via the tool (with an id). Same playbook → one step.
    // Self-defending sink: build the id from the name reduced to [a-z0-9_-] (server
    // _NAME_RE already enforces that, but the id/selector/onclick must not rely on it).
    const stepId = `playbook-${String(event.name).replace(/[^a-z0-9_-]/gi, '')}`;
    const stepIdAttr = Utils.escapeAttr(stepId);
    if (timeline.querySelector(`[data-step-id="${stepIdAttr}"]`)) return;  // dedupe
    // The body (the loaded playbook text) makes the step expandable — same detail the
    // tool row used to show. Absent it, render a plain, non-clickable pill.
    const body = event.body != null ? String(event.body).trim() : '';
    const onclick = body ? ` onclick="UI.toggleStepExpanded('${stepIdAttr}')"` : '';
    const toggle = body ? '<button class="step-toggle" aria-label="Expand playbook details">&#9654;</button>' : '';
    const details = body ? `
          <div class="step-details" style="display: none;">
            <div class="section-label">Playbook</div>
            <pre><code>${Utils.escapeHtml(body)}</code></pre>
          </div>` : '';
    timeline.insertAdjacentHTML('beforeend', `
      <div class="step playbook-step" data-step-id="${stepIdAttr}">
        <div class="step-connector">
          <span class="step-marker playbook-marker"></span>
          <div class="step-line"></div>
        </div>
        <div class="step-content">
          <div class="step-header"${onclick}>
            <span class="step-icon playbook-icon-glyph" aria-hidden="true">📘</span>
            <span class="step-label">Playbook applied · ${Utils.escapeHtml(event.name)}</span>
            ${toggle}
          </div>${details}
        </div>
      </div>`);
  },

  renderToolStart(messageId, event) {
    const timeline = document.querySelector(`.trace-container[data-message-id="${messageId}"] .step-timeline`);
    if (!timeline) return;

    const existingStep = timeline.querySelector(`[data-tool-call-id="${event.tool_call_id}"]`);
    if (existingStep) {
      const labelEl = existingStep.querySelector('.step-label');
      if (labelEl && event.tool_name) {
        labelEl.textContent = event.tool_name;
      }
      const argsCode = existingStep.querySelector('.tool-args pre code');
      if (argsCode) {
        argsCode.textContent = this.formatToolArgs(event.tool_args);
      }
      return;
    }

    const toolHtml = `
      <div class="step tool-step tool-running" data-step-id="${event.tool_call_id}" data-tool-call-id="${event.tool_call_id}">
        <div class="step-connector">
          <span class="step-marker tool-marker"></span>
          <div class="step-line"></div>
        </div>
        <div class="step-content">
          <div class="step-header" onclick="UI.toggleStepExpanded('${Utils.escapeAttr(event.tool_call_id)}')">
            <span class="step-icon tool-icon-glyph">T</span>
            <span class="step-label">${Utils.escapeHtml(event.tool_name)}</span>
            <span class="step-status">
              <span class="spinner"></span>
            </span>
            <button class="step-toggle" aria-label="Expand tool details">&#9654;</button>
          </div>
          <div class="step-details" style="display: none;">
            <div class="tool-args">
              <div class="section-label">Arguments</div>
              <pre><code>${this.formatToolArgs(event.tool_args)}</code></pre>
            </div>
            <div class="tool-output-section" style="display: none;">
              <div class="section-label">Output</div>
              <pre><code class="tool-output-content"></code></pre>
            </div>
          </div>
        </div>
      </div>`;

    timeline.insertAdjacentHTML('beforeend', toolHtml);
    this.scrollToBottom();

    // Auto-expand if verbose mode
    if (this.isTraceExpandedMode(this.getTraceModeForMessage(messageId))) {
      const step = timeline.querySelector(`[data-step-id="${event.tool_call_id}"]`);
      step?.classList.add('expanded');
      const details = step?.querySelector('.step-details');
      if (details) details.style.display = 'block';
    }
  },

  toggleStepExpanded(stepId) {
    const step = document.querySelector(`.step[data-step-id="${stepId}"]`);
    if (!step) return;
    
    step.classList.toggle('expanded');
    const details = step.querySelector('.step-details');
    const toggle = step.querySelector('.step-toggle');
    
    if (details) {
      details.style.display = step.classList.contains('expanded') ? 'block' : 'none';
    }
    if (toggle) {
      toggle.innerHTML = step.classList.contains('expanded') ? '&#9660;' : '&#9654;';
    }
  },

  renderToolOutput(messageId, event) {
    const step = document.querySelector(`.trace-container[data-message-id="${messageId}"] .tool-step[data-tool-call-id="${event.tool_call_id}"]`);
    if (!step) return;

    const outputSection = step.querySelector('.tool-output-section');
    const outputContent = step.querySelector('.tool-output-content');
    
    if (outputSection) {
      outputSection.style.display = 'block';
    }
    
    if (outputContent) {
      let displayText = event.output || '';
      if (displayText.length > CONFIG.TRACE.MAX_TOOL_OUTPUT_PREVIEW) {
        displayText = displayText.slice(0, CONFIG.TRACE.MAX_TOOL_OUTPUT_PREVIEW) + '...';
      }
      outputContent.textContent = displayText;
      
      if (event.truncated && event.full_length) {
        const notice = document.createElement('div');
        notice.className = 'truncation-notice';
        notice.textContent = `Showing ${CONFIG.TRACE.MAX_TOOL_OUTPUT_PREVIEW} of ${event.full_length} chars`;
        outputSection.appendChild(notice);
      }
    }

    this.scrollToBottom();
  },

  renderToolEnd(messageId, event) {
    const step = document.querySelector(`.trace-container[data-message-id="${messageId}"] .tool-step[data-tool-call-id="${event.tool_call_id}"]`);
    if (!step) return;

    step.classList.remove('tool-running');
    step.classList.add(event.status === 'success' ? 'tool-success' : 'tool-error');

    const marker = step.querySelector('.step-marker');
    if (marker) {
      marker.classList.remove('tool-marker');
      marker.classList.add(event.status === 'success' ? 'success-marker' : 'error-marker');
    }

    const statusEl = step.querySelector('.step-status');
    if (statusEl) {
      if (event.status === 'success') {
        const durationText = event.duration_ms ? Utils.formatDuration(event.duration_ms) : '';
        statusEl.innerHTML = `<span class="checkmark">&#10003;</span> ${durationText}`;
      } else {
        statusEl.innerHTML = `<span class="error-icon">&#10007;</span>`;
      }
    }

    // Auto-collapse if many tools
    const timeline = step.closest('.step-timeline');
    const toolCount = timeline?.querySelectorAll('.tool-step').length || 0;
    if (this.isTraceCollapsedMode(this.getTraceModeForMessage(messageId)) && toolCount > CONFIG.TRACE.AUTO_COLLAPSE_TOOL_COUNT) {
      step.classList.remove('expanded');
      const details = step.querySelector('.step-details');
      if (details) details.style.display = 'none';
    }
  },

  // =========================================================================
  // Context Meter
  // =========================================================================

  updateContextMeter(messageId, usage) {
    const meter = document.querySelector(`.trace-container[data-message-id="${messageId}"] .context-meter`);
    if (!meter || !usage) return;

    meter.style.display = 'flex';
    
    const fill = meter.querySelector('.meter-fill');
    const label = meter.querySelector('.meter-label');
    
    const promptTokens = usage.prompt_tokens || 0;
    const completionTokens = usage.completion_tokens || 0;
    const totalTokens = usage.total_tokens || (promptTokens + completionTokens);
    
    // Prefer backend-provided context window, fall back to 128k
    const contextWindow = (usage.context_window && usage.context_window > 0)
      ? usage.context_window
      : 128000;
    const usagePercent = Math.min((promptTokens / contextWindow) * 100, 100);
    
    if (fill) {
      fill.style.width = usagePercent.toFixed(1) + '%';
      // Color based on usage
      if (usagePercent > 80) {
        fill.style.backgroundColor = 'var(--error-text, #dc3545)';
      } else if (usagePercent > 50) {
        fill.style.backgroundColor = 'var(--warning-text, #ffc107)';
      }
    }
    
    if (label) {
      label.textContent = `${promptTokens.toLocaleString()} prompt + ${completionTokens.toLocaleString()} completion = ${totalTokens.toLocaleString()} tokens`;
      label.title = `Prompt tokens (sent to LLM): ${promptTokens.toLocaleString()}\nCompletion tokens (generated by LLM): ${completionTokens.toLocaleString()}\nTotal: ${totalTokens.toLocaleString()}\nContext window: ${contextWindow.toLocaleString()}`;
    }
  },

  // =========================================================================
  // Finalize Trace
  // =========================================================================

  finalizeTrace(messageId, trace, finalEvent) {
    this.stopTraceTimer(messageId, finalEvent?.duration_ms ?? null);
    
    const container = document.querySelector(`.trace-container[data-message-id="${messageId}"]`);
    if (!container) return;

    const toolCalls = trace?.toolCalls;
    const toolCount = toolCalls instanceof Map
      ? toolCalls.size
      : Array.isArray(toolCalls)
        ? toolCalls.length
        : 0;
    const label = container.querySelector('.trace-label');
    if (label) {
      label.textContent = this.getTraceLabelText(toolCount);
    }
    
    // Update context meter if usage available
    if (finalEvent && finalEvent.usage) {
      this.updateContextMeter(messageId, finalEvent.usage);
    }

    // Auto-collapse in normal mode
    if (this.isTraceCollapsedMode(this.getTraceModeForMessage(messageId))) {
      container.classList.add('collapsed');
      const toggleIcon = container.querySelector('.toggle-icon');
      if (toggleIcon) toggleIcon.innerHTML = '&#9654;';
    }
  },

  formatToolArgs(args) {
    if (!args) return '';
    try {
      if (typeof args === 'string') {
        return Utils.escapeHtml(args);
      }
      return Utils.escapeHtml(JSON.stringify(args, null, 2));
    } catch {
      return Utils.escapeHtml(String(args));
    }
  },

  // =========================================================================
  // Historical Trace Rendering (for loaded conversations)
  // =========================================================================

  renderHistoricalTrace(messageId, trace) {
    if (!trace || !trace.events) return;

    const msgEl = this.elements.messagesInner?.querySelector(`[data-id="${messageId}"]`);
    if (!msgEl) return;

    const inner = msgEl.querySelector('.message-inner');
    if (!inner) return;

    // Don't add if already exists
    if (inner.querySelector('.trace-container')) return;

    const events = trace.events;
    if (!events || events.length === 0) return;

    // Count unique tool calls (tool_start updates may appear multiple times for same id)
    const toolCallIds = new Set(
      events
        .filter(e => (e.type === 'tool_start' || e.type === 'tool_use') && e.tool_call_id)
        .map(e => e.tool_call_id)
    );
    const toolCount = toolCallIds.size;

    // Calculate total duration
    const durationMs = trace.total_duration_ms || 0;
    const durationStr = Utils.formatDuration(durationMs);

    const traceIconSvg = this.getTraceIconSvg();
    
    // Build trace container with collapsed state
    const labelText = this.getTraceLabelText(toolCount);

    const traceHtml = `
      <div class="trace-container collapsed" data-message-id="${messageId}">
        <div class="trace-header">
          ${traceIconSvg}
          <span class="trace-label">${labelText}</span>
          <span class="trace-timer">${durationStr}</span>
          <button class="trace-toggle" aria-label="Toggle agent activity details" title="Toggle agent activity" onclick="UI.toggleTraceExpanded('${messageId}')">
            <span class="toggle-icon" aria-hidden="true">&#9654;</span>
          </button>
        </div>
        <div class="trace-content">
          <div class="context-meter" style="display: none;" title="LLM token usage for this response. Prompt = tokens sent to the model; Completion = tokens generated back.">
            <div class="meter-bar" title="Context window usage"><div class="meter-fill"></div></div>
            <span class="meter-label"></span>
          </div>
          <div class="step-timeline"></div>
        </div>
      </div>`;

    inner.insertAdjacentHTML('afterbegin', traceHtml);

    // Now populate the timeline with events
    const timeline = inner.querySelector('.step-timeline');
    if (!timeline) return;

    // Process events and add steps
    const toolStartEvents = {};
    const thinkingEvents = {};
    let usageData = null;
    // Text events are cumulative per-turn snapshots; the last one before a
    // tool call is that turn's narration. Trailing text is the final answer.
    let pendingNarration = '';
    let narrationCount = 0;

    for (const event of events) {
      if (event.type === 'text') {
        pendingNarration = event.content || '';
      } else if (event.type === 'thinking_start') {
        thinkingEvents[event.step_id] = event;
      } else if (event.type === 'thinking_end') {
        const startEvent = thinkingEvents[event.step_id];
        if (startEvent && event.thinking_content && event.thinking_content.trim()) {
          this.addHistoricalThinkingStep(timeline, event);
        }
      } else if (event.type === 'tool_start' || event.type === 'tool_use') {
        if (pendingNarration.trim()) {
          narrationCount += 1;
          this.addHistoricalNarrationStep(timeline, {
            step_id: `narration-${narrationCount}`,
            content: pendingNarration.trim(),
          });
          pendingNarration = '';
        }
        toolStartEvents[event.tool_call_id] = event;
        // Add the tool step immediately
        this.addHistoricalToolStep(timeline, event, null);
      } else if (event.type === 'tool_end' || event.type === 'tool_result' || event.type === 'tool_output') {
        const startEvent = toolStartEvents[event.tool_call_id];
        // Update the tool step with output
        this.updateHistoricalToolStep(timeline, event, startEvent);
      } else if (event.type === 'playbook_applied') {
        // A persisted auto-pickup step: rebuild the same expandable "Playbook applied"
        // row (renderPlaybookApplied re-queries this message's live timeline and dedupes).
        this.renderPlaybookApplied(messageId, event);
      } else if (event.type === 'usage') {
        usageData = event;
      }
    }

    // Populate context meter if usage data is available
    if (usageData) {
      this.updateContextMeter(messageId, usageData);
    }
  },

  addHistoricalThinkingStep(timeline, event) {
    const stepHtml = `
      <div class="step thinking-step completed" data-step-id="${event.step_id}">
        <div class="step-connector">
          <span class="step-marker completed-marker"></span>
          <div class="step-line"></div>
        </div>
        <div class="step-content">
          <div class="step-header" onclick="UI.toggleStepExpanded('${Utils.escapeAttr(event.step_id)}')">
            <span class="step-icon">💭</span>
            <span class="step-label">Thinking</span>
            <span class="step-timer">${event.duration_ms ? Utils.formatDuration(event.duration_ms) : ''}</span>
            <button class="step-toggle" aria-label="Expand thinking details">&#9654;</button>
          </div>
          <div class="step-details" style="display: none;">
            <div class="section-label">Details</div>
            <pre><code>${Utils.escapeHtml(event.thinking_content || '')}</code></pre>
          </div>
        </div>
      </div>`;
    timeline.insertAdjacentHTML('beforeend', stepHtml);
  },

  addHistoricalNarrationStep(timeline, event) {
    const stepHtml = `
      <div class="step thinking-step completed" data-step-id="${Utils.escapeAttr(event.step_id)}">
        <div class="step-connector">
          <span class="step-marker completed-marker"></span>
          <div class="step-line"></div>
        </div>
        <div class="step-content">
          <div class="step-header" onclick="UI.toggleStepExpanded('${Utils.escapeAttr(event.step_id)}')">
            <span class="step-icon">📝</span>
            <span class="step-label">Planning</span>
            <span class="step-timer"></span>
            <button class="step-toggle" aria-label="Expand planning details">&#9654;</button>
          </div>
          <div class="step-details" style="display: none;">
            <div class="section-label">Details</div>
            <pre><code>${Utils.escapeHtml(event.content || '')}</code></pre>
          </div>
        </div>
      </div>`;
    timeline.insertAdjacentHTML('beforeend', stepHtml);
  },

  addHistoricalToolStep(timeline, event, outputEvent) {
    const existingStep = timeline.querySelector(`[data-tool-call-id="${event.tool_call_id}"]`);
    if (existingStep) {
      const labelEl = existingStep.querySelector('.step-label');
      if (labelEl && event.tool_name) {
        labelEl.textContent = event.tool_name;
      }
      const argsCode = existingStep.querySelector('.tool-args pre code');
      if (argsCode) {
        argsCode.textContent = this.formatToolArgs(event.tool_args || event.arguments);
      }
      return;
    }

    const toolName = event.tool_name || 'Unknown Tool';
    const toolArgs = this.formatToolArgs(event.tool_args || event.arguments);
    
    const stepHtml = `
      <div class="step tool-step completed" data-step-id="${event.tool_call_id}" data-tool-call-id="${event.tool_call_id}">
        <div class="step-connector">
          <span class="step-marker completed-marker"></span>
          <div class="step-line"></div>
        </div>
        <div class="step-content">
          <div class="step-header" onclick="UI.toggleStepExpanded('${Utils.escapeAttr(event.tool_call_id)}')">
            <span class="step-icon tool-icon-glyph">T</span>
            <span class="step-label">${Utils.escapeHtml(toolName)}</span>
            <span class="step-status">✓</span>
            <button class="step-toggle" aria-label="Expand tool details">&#9654;</button>
          </div>
          <div class="step-details" style="display: none;">
            <div class="tool-args">
              <div class="section-label">Arguments</div>
              <pre><code>${toolArgs}</code></pre>
            </div>
            <div class="tool-output-section" style="display: none;">
              <div class="section-label">Output</div>
              <pre><code class="tool-output-content"></code></pre>
            </div>
          </div>
        </div>
      </div>`;
    timeline.insertAdjacentHTML('beforeend', stepHtml);
  },

  updateHistoricalToolStep(timeline, outputEvent, startEvent) {
    const step = timeline.querySelector(`[data-tool-call-id="${outputEvent.tool_call_id}"]`);
    if (!step) return;

    // Update duration if available
    if (outputEvent.duration_ms) {
      const statusEl = step.querySelector('.step-status');
      if (statusEl) {
        statusEl.textContent = Utils.formatDuration(outputEvent.duration_ms);
      }
    }

    // Update output if available
    const output = outputEvent.tool_output || outputEvent.result || outputEvent.output;
    if (output) {
      const outputSection = step.querySelector('.tool-output-section');
      const outputContent = step.querySelector('.tool-output-content');
      if (outputSection && outputContent) {
        outputSection.style.display = 'block';
        outputContent.textContent = typeof output === 'string' ? output : JSON.stringify(output, null, 2);
      }
    }
  },

  showCancelButton(messageId) {
    const msgEl = this.elements.messagesInner?.querySelector(`[data-id="${messageId}"]`);
    if (!msgEl) return;

    const existing = msgEl.querySelector('.cancel-stream-btn');
    if (existing) return;

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'cancel-stream-btn';
    cancelBtn.innerHTML = 'Stop';
    cancelBtn.onclick = () => Chat.cancelStream();

    msgEl.querySelector('.message-inner')?.appendChild(cancelBtn);
  },

  hideCancelButton(messageId) {
    const msgEl = this.elements.messagesInner?.querySelector(`[data-id="${messageId}"]`);
    msgEl?.querySelector('.cancel-stream-btn')?.remove();
  },

  // =========================================================================
  // Feedback Handlers
  // =========================================================================

  async handleFeedback(button, type) {
    const msgEl = button.closest('.message');
    if (!msgEl) return;

    const messageId = msgEl.dataset.id;
    if (!messageId || isNaN(Number(messageId))) {
      console.warn('Cannot submit feedback: invalid message id', messageId);
      return;
    }

    const actionsEl = msgEl.querySelector('.message-actions');
    
    // Disable buttons during request
    const buttons = actionsEl?.querySelectorAll('.feedback-btn');
    buttons?.forEach(btn => btn.disabled = true);

    try {
      if (type === 'like') {
        const result = await API.likeMessage(Number(messageId));
        this.updateFeedbackState(actionsEl, result.state);
      } else if (type === 'dislike') {
        const result = await API.dislikeMessage(Number(messageId));
        this.updateFeedbackState(actionsEl, result.state);
      } else if (type === 'comment') {
        this.showFeedbackModal(messageId);
      }
    } catch (e) {
      console.error(`Failed to submit ${type}:`, e);
    } finally {
      buttons?.forEach(btn => btn.disabled = false);
    }
  },

  updateFeedbackState(actionsEl, state) {
    if (!actionsEl) return;
    
    // Remove all active states
    actionsEl.classList.remove('feedback-like-active', 'feedback-dislike-active');
    
    // Apply new state
    if (state === 'like') {
      actionsEl.classList.add('feedback-like-active');
    } else if (state === 'dislike') {
      actionsEl.classList.add('feedback-dislike-active');
    }
  },

  showFeedbackModal(messageId) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('feedback-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'feedback-modal';
      modal.className = 'modal-overlay';
      modal.innerHTML = `
        <div class="modal-content feedback-modal-content">
          <div class="modal-header">
            <div class="modal-title-group">
              <svg class="modal-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              </svg>
              <h3>Send Feedback</h3>
            </div>
            <button class="modal-close" aria-label="Close">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
          <div class="modal-body">
            <p class="feedback-description">Help us improve by sharing your thoughts on this response.</p>
            <label class="feedback-label" for="feedback-text">Your feedback</label>
            <textarea id="feedback-text" placeholder="What could be improved? What was helpful or unhelpful?" rows="5"></textarea>
            <p class="feedback-hint">Your feedback helps us make the assistant better for everyone.</p>
          </div>
          <div class="modal-footer">
            <button class="btn btn-secondary" data-dismiss="modal">Cancel</button>
            <button class="btn btn-primary" id="submit-feedback-btn">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
              Submit Feedback
            </button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
    }

    const textarea = modal.querySelector('#feedback-text');
    const submitBtn = modal.querySelector('#submit-feedback-btn');
    const closeBtn = modal.querySelector('.modal-close');
    const cancelBtn = modal.querySelector('[data-dismiss="modal"]');

    // Reset
    textarea.value = '';
    modal.style.display = 'flex';
    modal.classList.add('modal-visible');
    setTimeout(() => textarea.focus(), 100);

    const closeModal = () => {
      modal.classList.remove('modal-visible');
      setTimeout(() => { modal.style.display = 'none'; }, 150);
    };

    const handleSubmit = async () => {
      const text = textarea.value.trim();
      if (!text) {
        closeModal();
        return;
      }
      
      // Show loading state
      submitBtn.disabled = true;
      submitBtn.innerHTML = `
        <svg class="spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
        </svg>
        Sending...
      `;
      
      try {
        await API.submitTextFeedback(Number(messageId), text);
      } catch (e) {
        console.error('Failed to submit feedback:', e);
        // Show error in the modal instead of silently closing
        const hint = modal.querySelector('.feedback-hint');
        if (hint) {
          hint.textContent = 'Failed to submit feedback. Please try again.';
          hint.style.color = 'var(--error-text, #f85149)';
        }
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
          Submit Feedback
        `;
        return; // Don't close modal on error
      }
      
      // Reset button
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
        Submit Feedback
      `;
      closeModal();
    };

    // Clean up old listeners by cloning nodes
    const newSubmitBtn = submitBtn.cloneNode(true);
    submitBtn.parentNode.replaceChild(newSubmitBtn, submitBtn);
    newSubmitBtn.onclick = handleSubmit;

    const newCloseBtn = closeBtn.cloneNode(true);
    closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
    newCloseBtn.onclick = closeModal;

    const newCancelBtn = cancelBtn.cloneNode(true);
    cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);
    newCancelBtn.onclick = closeModal;

    // Use a named handler for the backdrop click so we can remove it
    if (modal._backdropHandler) {
      modal.removeEventListener('click', modal._backdropHandler);
    }
    modal._backdropHandler = (e) => {
      if (e.target === modal) closeModal();
    };
    modal.addEventListener('click', modal._backdropHandler);
  },
};

// Make UI globally accessible for onclick handlers
window.UI = UI;

// =============================================================================
// Playbook Quick-Invoke Menu
// =============================================================================

const PlaybookMenu = {
  el: () => document.querySelector('.playbook-menu'),
  playbooks: [],
  open: false,
  locked: null,   // name already autocompleted into the input — don't re-open for it

  async maybeShow() {
    const field = UI.elements.inputField;
    if (!field) return;
    const value = field.value;
    if (!value.startsWith('/')) {
      // not a playbook invoke anymore — drop any pending selection so it can't leak
      if (this.locked) { this.locked = null; API._pendingPlaybookName = null; }
      this.hide();
      return;
    }
    // already completed to "/name " — leave it in the text, keep the menu closed
    if (this.locked && value.startsWith(`/${this.locked} `)) { this.hide(); return; }
    // user edited the name again — unlock and re-filter
    if (this.locked) { this.locked = null; API._pendingPlaybookName = null; }
    const query = value.slice(1).split(/\s/)[0].toLowerCase();
    if (!this.open) {
      try { this.playbooks = (await API.getPlaybooksList())?.playbooks || []; }
      catch (e) { this.playbooks = []; }
    }
    // Prefix match (like shell/command autocomplete): "/cond" completes names that START with
    // "cond", not ones that merely contain it mid-string (which would make Tab pick the wrong playbook).
    // Only your list (own + enabled) appears in the menu; unadded public ones are added from the panel.
    // NOTE: this.playbooks stays the FULL list — the send-time "Add & run?" guard relies on it.
    const matches = this.playbooks.filter(s => s.is_enabled !== false && s.name.toLowerCase().startsWith(query));
    this.render(matches);
  },

  render(matches) {
    const menu = this.el();
    if (!menu) return;
    if (!matches.length) { this.hide(); return; }
    menu.innerHTML = matches.map((s, i) => `
      <div class="playbook-menu-item${i === 0 ? ' active' : ''}" role="option" data-playbook="${Utils.escapeHtml(s.name)}">
        <span class="playbook-menu-name">/${Utils.escapeHtml(s.name)}${s.is_mine === false ? '<span class="playbook-badge">public</span>' : ''}</span>
        <span class="playbook-menu-desc">${Utils.escapeHtml(s.description || '')}</span>
      </div>`).join('');
    menu.hidden = false;
    this.open = true;
    menu.querySelectorAll('.playbook-menu-item').forEach(item => {
      item.addEventListener('mousedown', (e) => {
        e.preventDefault();
        this.select(item.dataset.playbook);
      });
    });
  },

  select(name) {
    API._pendingPlaybookName = name;            // applied by the next stream request
    const field = UI.elements.inputField;
    if (field) {
      // autocomplete the partial "/xyz" to the full "/name " — keep it IN the text box
      const rest = field.value.replace(/^\/\S*\s?/, '');
      field.value = `/${name} ${rest}`;
      field.focus();
      field.setSelectionRange(field.value.length, field.value.length);
    }
    this.locked = name;                      // completed — don't re-open the menu for it
    this.hide();
  },

  move(delta) {
    const menu = this.el();
    if (!menu) return;
    const items = [...menu.querySelectorAll('.playbook-menu-item')];
    if (!items.length) return;
    let i = items.findIndex(it => it.classList.contains('active'));
    if (i < 0) i = 0;
    items[i].classList.remove('active');
    i = (i + delta + items.length) % items.length;
    items[i].classList.add('active');
    items[i].scrollIntoView({ block: 'nearest' });
  },

  selectActive() {
    const menu = this.el();
    if (!menu) return;
    const active = menu.querySelector('.playbook-menu-item.active') || menu.querySelector('.playbook-menu-item');
    if (active) this.select(active.dataset.playbook);
  },

  hide() {
    const menu = this.el();
    if (menu) { menu.hidden = true; menu.innerHTML = ''; }
    this.open = false;
  },
};

// =============================================================================
// Chat Controller
// =============================================================================

const Chat = {
  state: {
    conversationId: null,
    messages: [],
    history: [], // [sender, content] pairs for API
    isStreaming: false,
    configs: [],
    // A/B Testing state
    activeABComparison: null,  // { comparisonId, responseAId, responseBId, variantA, variantB }
    pendingABComparisons: [],  // unresolved comparisons in creation order
    abVotePending: false,      // true when waiting for user vote
    abPool: null,              // null or { enabled, champion, variants: [...] } from /api/ab/pool
    abCapabilities: {
      canView: false,
      canManage: false,
      canViewMetrics: false,
      canParticipate: false,
    },
    abPreferenceSaveState: null,
    // Trace state
    activeTrace: null,         // { traceId, events: [], toolCalls: Map<toolCallId, toolData> }
    traceVerboseMode: localStorage.getItem(CONFIG.STORAGE_KEYS.TRACE_VERBOSE_MODE) || 'normal', // 'minimal' | 'normal' | 'verbose'
    abortController: null,     // AbortController for cancellation
    // Provider state
    providers: [],
    pipelineDefaultModel: null,
    selectedProvider: localStorage.getItem(CONFIG.STORAGE_KEYS.SELECTED_PROVIDER) || null,
    selectedModel: localStorage.getItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL) || null,
    selectedCustomModel: localStorage.getItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL_CUSTOM) || null,

    agents: [],
    allAgents: [],  // full list including ab_only variants, for pool editor
    activeAgentName: null,
    currentUser: null,
  },

  getABPendingLimit() {
    const configured = Number(
      this.state.abPool?.max_pending_comparisons_per_conversation
      ?? this.state.abPool?.max_pending_per_conversation
      ?? 1
    );
    return Number.isFinite(configured) && configured >= 1 ? configured : 1;
  },

  hasReachedABPendingLimit() {
    return this.state.pendingABComparisons.length >= this.getABPendingLimit();
  },

  syncABPendingState() {
    const pending = Array.isArray(this.state.pendingABComparisons)
      ? this.state.pendingABComparisons.filter(Boolean)
      : [];
    this.state.pendingABComparisons = pending;
    this.state.activeABComparison = pending.length ? pending[pending.length - 1] : null;
    this.state.abVotePending = pending.length > 0;

    UI.hideABVoteButtons();
    if (this.state.activeABComparison) {
      UI.setABComparisonId(this.state.activeABComparison);
      UI.showABVoteButtons(this.state.activeABComparison);
    }

    if (!this.state.isStreaming) {
      UI.setInputDisabled(this.hasReachedABPendingLimit());
    }
  },

  addPendingABComparison(comparisonState) {
    if (!comparisonState?.comparisonId) return;
    const remaining = this.state.pendingABComparisons.filter(
      (item) => item?.comparisonId !== comparisonState.comparisonId,
    );
    remaining.push(comparisonState);
    this.state.pendingABComparisons = remaining;
    this.syncABPendingState();
  },

  removePendingABComparison(comparisonId) {
    this.state.pendingABComparisons = this.state.pendingABComparisons.filter(
      (item) => item?.comparisonId !== comparisonId,
    );
    this.syncABPendingState();
  },

  async init() {
    Markdown.init();
    UI.init();

    // Load initial data
    await Promise.all([
      this.loadConfigs(),
      this.loadConversations(),
      this.loadProviders(),
      this.loadPipelineDefaultModel(),
      this.loadApiKeyStatus(),
      UI.loadUserProfile(),
      this.loadCurrentUser(),
      this.loadAgents(),
      this.loadABPool(),
    ]);

    // Update model label after all data is loaded (configs, providers, pipeline default)
    this.updateActiveModelLabel();

    // Load active conversation if any
    const activeId = Storage.getActiveConversationId();
    if (activeId) {
      await this.loadConversation(activeId);
    }
  },

  async loadConfigs() {
    try {
      const data = await API.getConfigs();
      this.state.configs = data?.options || [];
      const timeoutMs = Number(data?.client_timeout_ms);
      if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
        CONFIG.STREAMING.TIMEOUT = timeoutMs;
      }
      UI.renderConfigs(this.state.configs);
    } catch (e) {
      console.error('Failed to load configs:', e);
    }
  },

  async loadAgents() {
    try {
      const data = await API.getAgentsList();
      // Keep full list (including ab_only) for pool editor
      this.state.allAgents = data?.agents || [];
      // Filter out ab_only agents for main dropdown
      this.state.agents = this.state.allAgents.filter(a => !a.ab_only);
      const activeName = data?.active_name || this.state.agents[0]?.name || null;
      this.state.activeAgentName = Utils.normalizeAgentName(activeName);
      UI.renderAgentsList(this.state.agents, this.state.activeAgentName);
      this.updateActiveModelLabel();
    } catch (e) {
      console.error('Failed to load agents list:', e);
    }
  },

  async loadABPool() {
    try {
      const data = await API.getABPool();
      this.state.abCapabilities = {
        canView: data?.can_view === true,
        canManage: data?.can_manage === true,
        canViewMetrics: data?.can_view_metrics === true,
        canParticipate: data?.can_participate === true,
      };

      this.state.abPool = data || null;
      UI.updateABSettingsSection();
      if (this.state.abCapabilities.canManage) {
        UI.updateABPoolUI(data?.enabled ? data : { enabled: false });
      }
    } catch (e) {
      console.warn('Failed to load A/B pool (pool mode disabled):', e);
      this.state.abPool = null;
      this.state.abCapabilities = {
        canView: false,
        canManage: false,
        canViewMetrics: false,
        canParticipate: false,
      };
      UI.updateABSettingsSection();
    }
  },

  async loadCurrentUser() {
    try {
      const data = await API.getCurrentUser();
      this.state.currentUser = data || null;
      this.state.abPreferenceSaveState = null;
      UI.updateABSettingsSection();
    } catch (e) {
      console.warn('Failed to load current user preferences:', e);
      this.state.currentUser = null;
      UI.updateABSettingsSection();
    }
  },

  async saveABParticipationPreference(rate) {
    const bounded = Math.max(0, Math.min(1, Number(rate)));
    try {
      const updated = await API.updateUserPreferences({ ab_participation_rate: bounded });
      this.state.currentUser = {
        ...(this.state.currentUser || {}),
        ...updated,
      };
      this.state.abPreferenceSaveState = {
        type: 'success',
        message: 'Saved for your account.',
      };
      UI.updateABSettingsSection();
    } catch (e) {
      console.error('Failed to save A/B participation preference:', e);
      this.state.abPreferenceSaveState = {
        type: 'error',
        message: e.message || 'Unable to save A/B participation preference.',
      };
      UI.showToast(e.message || 'Unable to save A/B participation preference.');
      UI.updateABSettingsSection();
    }
  },

  async setActiveAgent(name) {
    if (!name) return;
    try {
      const response = await API.setActiveAgent(name);
      const activeName = response?.active_name || name;
      this.state.activeAgentName = Utils.normalizeAgentName(activeName);
      UI.renderAgentsList(this.state.agents, this.state.activeAgentName);
      this.updateActiveModelLabel();
    } catch (e) {
      console.error('Failed to set active agent:', e);
    }
  },

  async deleteAgent(name) {
    // Legacy path — now handled via inline confirmation in UI
    if (!name) return;
    await this.doDeleteAgent(name);
  },

  async doDeleteAgent(name) {
    if (!name) return;
    try {
      await API.deleteAgent(Utils.normalizeAgentName(name));
      await this.loadAgents();
    } catch (e) {
      console.error('Failed to delete agent:', e);
      alert(e.message || 'Unable to delete agent.');
    }
  },

  async loadProviders() {
    try {
      const data = await API.getProviders();
      this.state.providers = data?.providers || [];
      
      // Render providers dropdown
      UI.renderProviders(this.state.providers, this.state.selectedProvider);
      
      // If we have a selected provider, load its models; otherwise use pipeline default
      const currentProvider = this.state.selectedProvider;
      if (currentProvider) {
        await this.loadProviderModels(currentProvider);
      } else {
        UI.renderProviderModels([], null);
        this.showPipelineDefaultStatus();
      }
    } catch (e) {
      console.error('Failed to load providers:', e);
      // Show error status
      UI.updateProviderStatus('disconnected', 'Failed to load providers');
    }
  },

  async loadPipelineDefaultModel() {
    try {
      const data = await API.getPipelineDefaultModel();
      this.state.pipelineDefaultModel = data || null;
      if (!this.state.selectedProvider) {
        this.showPipelineDefaultStatus();
      }
    } catch (e) {
      console.error('Failed to load pipeline default model:', e);
    }
  },

  showPipelineDefaultStatus() {
    const info = this.state.pipelineDefaultModel;
    const labelParts = [];
    if (info?.model_class) {
      labelParts.push(info.model_class);
    }
    if (info?.model_name) {
      labelParts.push(`(${info.model_name})`);
    }
    const label = labelParts.length ? labelParts.join(' ') : 'Pipeline default model';
    UI.updateProviderStatus('connected', `Using pipeline default: ${label}`);
  },

  formatPipelineDefaultLabel() {
    const info = this.state.pipelineDefaultModel;
    // Show model_name if available, otherwise model_class
    if (info?.model_name) {
      return info.model_name;
    }
    if (info?.model_class) {
      return info.model_class;
    }
    return 'Default model';
  },

  getAgentLabel() {
    if (this.state.activeAgentName) {
      return this.state.activeAgentName;
    }
    return 'Default agent';
  },

  getCurrentModelLabel() {
    const provider = this.state.selectedProvider;
    if (!provider) {
      return this.formatPipelineDefaultLabel();
    }

    const model = this.getSelectedProviderAndModel().model;
    return model || 'Select model';
  },

  getEntryMetaLabel() {
    const agentLabel = this.getAgentLabel();
    const modelLabel = this.getCurrentModelLabel();
    return `${agentLabel} · ${modelLabel}`;
  },

  updateActiveModelLabel() {
    UI.updateActiveModelLabel(this.getEntryMetaLabel());
  },

  async loadProviderModels(providerType) {
    try {
      const provider = this.state.providers.find(p => p.type === providerType);
      if (!provider) return;

      // Use models from the provider data (already loaded)
      const models = provider.models || [];
      UI.renderProviderModels(models, this.state.selectedModel, providerType);
      if (providerType === 'openrouter' && this.state.selectedModel === '__custom__') {
        if (UI.elements.customModelInput) {
          UI.elements.customModelInput.value = this.state.selectedCustomModel || '';
        }
      }

      // Set default model if none selected
      if (!this.state.selectedModel || !models.some(m => m.id === this.state.selectedModel)) {
        if (providerType === 'openrouter' && this.state.selectedCustomModel) {
          this.state.selectedModel = '__custom__';
          localStorage.setItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL, '__custom__');
          if (UI.elements.modelSelectPrimary) {
            UI.elements.modelSelectPrimary.value = '__custom__';
          }
          UI.showCustomModelInput(true);
        } else {
        const defaultModel = provider.default_model || models[0]?.id;
        if (defaultModel) {
          this.state.selectedModel = defaultModel;
          localStorage.setItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL, defaultModel);
          if (UI.elements.modelSelectPrimary) {
            UI.elements.modelSelectPrimary.value = defaultModel;
          }
          UI.showCustomModelInput(false);
        }
        }
      }

      // Show connected status
      if (provider.enabled) {
        UI.updateProviderStatus('connected', `Connected to ${provider.display_name}`);
        setTimeout(() => UI.hideProviderStatus(), 2000);
      }
      this.updateActiveModelLabel();
    } catch (e) {
      console.error('Failed to load provider models:', e);
      UI.updateProviderStatus('disconnected', 'Failed to load models');
    }
  },

  async handleProviderChange(providerType) {
    if (!providerType) {
      this.state.selectedProvider = null;
      this.state.selectedModel = null;
      this.state.selectedCustomModel = null;
      localStorage.removeItem(CONFIG.STORAGE_KEYS.SELECTED_PROVIDER);
      localStorage.removeItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL);
      localStorage.removeItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL_CUSTOM);
      UI.renderProviderModels([], null);
      this.showPipelineDefaultStatus();
      this.updateActiveModelLabel();
      return;
    }

    this.state.selectedProvider = providerType;
    localStorage.setItem(CONFIG.STORAGE_KEYS.SELECTED_PROVIDER, providerType);
    
    // Clear model selection until new models load
    this.state.selectedModel = null;
    localStorage.removeItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL);
    this.state.selectedCustomModel = null;
    localStorage.removeItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL_CUSTOM);
    
    UI.updateProviderStatus('loading', 'Loading models...');
    await this.loadProviderModels(providerType);
    this.updateActiveModelLabel();
  },

  handleSendOrStop() {
    if (this.state.isStreaming) {
      this.cancelStream();
      return;
    }
    this.sendMessage();
  },

  handleModelChange(modelId) {
    if (!modelId) return;

    this.state.selectedModel = modelId;
    localStorage.setItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL, modelId);
    if (modelId === '__custom__' && this.state.selectedProvider === 'openrouter') {
      UI.showCustomModelInput(true);
    } else {
      UI.showCustomModelInput(false);
    }
    this.updateActiveModelLabel();
  },

  handleCustomModelChange(value) {
    const trimmed = value.trim();
    this.state.selectedCustomModel = trimmed || null;
    if (trimmed) {
      localStorage.setItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL_CUSTOM, trimmed);
    } else {
      localStorage.removeItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL_CUSTOM);
    }
    this.updateActiveModelLabel();
  },

  getSelectedProviderAndModel() {
    const provider = this.state.selectedProvider || null;
    if (!provider) {
      return { provider: null, model: null };
    }
    if (provider === 'openrouter' && this.state.selectedModel === '__custom__') {
      return { provider, model: this.state.selectedCustomModel || null };
    }
    return { provider, model: this.state.selectedModel };
  },

  // API Key Management
  async loadApiKeyStatus() {
    try {
      const data = await API.getProviderKeys();
      this.state.apiKeyStatus = data?.providers || [];
      UI.renderApiKeyStatus(this.state.apiKeyStatus);
    } catch (e) {
      console.error('Failed to load API key status:', e);
      UI.renderApiKeyStatus([]);
    }
  },

  async setApiKey(providerType, apiKey) {
    try {
      const result = await API.setProviderKey(providerType, apiKey);
      
      // Reload status and providers to reflect changes
      await Promise.all([
        this.loadApiKeyStatus(),
        this.loadProviders(),
      ]);
      
      return result;
    } catch (e) {
      console.error('Failed to set API key:', e);
      throw e;
    }
  },

  async clearApiKey(providerType) {
    try {
      const result = await API.clearProviderKey(providerType);
      
      // Reload status and providers to reflect changes
      await Promise.all([
        this.loadApiKeyStatus(),
        this.loadProviders(),
      ]);
      
      return result;
    } catch (e) {
      console.error('Failed to clear API key:', e);
      throw e;
    }
  },

  async loadConversations() {
    try {
      const data = await API.getConversations();
      UI.renderConversations(data?.conversations || [], this.state.conversationId);
    } catch (e) {
      console.error('Failed to load conversations:', e);
    }
  },

  async loadConversation(conversationId) {
    try {
      const data = await API.loadConversation(conversationId);
      if (!data) return;

      this.state.conversationId = conversationId;
      Storage.setActiveConversationId(conversationId);

      // Convert messages to display format
      this.state.messages = (data.messages || []).map((msg, idx) => {
        const isUser = msg.sender === 'User';
        return {
          id: msg.message_id || `${idx}-${isUser ? 'u' : 'a'}`,
          sender: msg.sender,
          html: isUser ? Utils.escapeHtml(msg.content) : Markdown.render(msg.content),
          meta: isUser ? null : (msg.model_used || this.getEntryMetaLabel()),
          feedback: msg.feedback || null,
          trace: msg.trace || null,  // Include trace data
          playbookName: msg.playbook_name || null,  // show a chip on /playbook-invoked user turns
        };
      });

      // Build history for API
      this.state.history = (data.messages || []).map((msg) => [msg.sender, msg.content]);
      this.state.activeABComparison = null;
      this.state.pendingABComparisons = [];
      this.state.abVotePending = false;

      UI.renderMessages(this.state.messages);
      
      // Render historical trace data for assistant messages
      for (const msg of this.state.messages) {
        if (msg.sender !== 'User' && msg.trace) {
          UI.renderHistoricalTrace(msg.id, msg.trace);
        }
      }

      if (Array.isArray(data.pending_ab_comparisons) && data.pending_ab_comparisons.length) {
        this.restorePendingABComparisons(data.pending_ab_comparisons);
      } else if (data.pending_ab_comparison) {
        this.restorePendingABComparisons([data.pending_ab_comparison]);
      } else {
        UI.hideABVoteButtons();
        UI.setInputDisabled(false);
      }
      
      await this.loadConversations(); // Refresh list to show active state
    } catch (e) {
      console.error('Failed to load conversation:', e);
      this.state.conversationId = null;
      this.state.messages = [];
      this.state.history = [];
      this.state.activeABComparison = null;
      this.state.pendingABComparisons = [];
      this.state.abVotePending = false;
      Storage.setActiveConversationId(null);
      UI.renderMessages([]);
      UI.hideABVoteButtons();
      UI.setInputDisabled(false);
      UI.showToast('Conversation not found. Starting a new chat.');
    }
  },

  restorePendingABComparison(comparison) {
    if (!comparison?.response_a || !comparison?.response_b) return;

    const traceMode = UI.normalizeABTraceMode(
      comparison.activity_panel_default_state ?? comparison.default_trace_mode
    );
    const disclosureMode = UI.normalizeABDisclosureMode(
      comparison.variant_label_mode ?? comparison.disclosure_mode
    );
    const responseAId = comparison.response_a.message_id;
    const responseBId = comparison.response_b.message_id;

    UI.addABComparisonContainer(responseAId, responseBId, {
      traceMode,
      comparisonKey: comparison.comparison_id || `${responseAId}-${responseBId}`,
    });
    UI.updateABResponse(responseAId, Markdown.render(comparison.response_a.content || ''), false);
    UI.updateABResponse(responseBId, Markdown.render(comparison.response_b.content || ''), false);
    UI.updateABArmPresentation(responseAId, {
      variantName: comparison.variant_a_name,
      modelUsed: comparison.response_a.model_used,
    }, {
      disclosureMode,
      reveal: false,
    });
    UI.updateABArmPresentation(responseBId, {
      variantName: comparison.variant_b_name,
      modelUsed: comparison.response_b.model_used,
    }, {
      disclosureMode,
      reveal: false,
    });

    if (UI.isTraceVisibleMode(traceMode) && comparison.response_a.trace) {
      document.querySelector(`.ab-arm[data-id="${responseAId}"] .trace-container`)?.remove();
      UI.stopTraceTimer(responseAId);
      UI.renderHistoricalTrace(responseAId, comparison.response_a.trace);
    } else {
      UI.stopTraceTimer(responseAId);
      document.querySelector(`.ab-arm[data-id="${responseAId}"] .trace-container`)?.remove();
    }
    if (UI.isTraceVisibleMode(traceMode) && comparison.response_b.trace) {
      document.querySelector(`.ab-arm[data-id="${responseBId}"] .trace-container`)?.remove();
      UI.stopTraceTimer(responseBId);
      UI.renderHistoricalTrace(responseBId, comparison.response_b.trace);
    } else {
      UI.stopTraceTimer(responseBId);
      document.querySelector(`.ab-arm[data-id="${responseBId}"] .trace-container`)?.remove();
    }

    const state = {
      comparisonId: comparison.comparison_id,
      responseAId,
      responseBId,
      responseAUiId: responseAId,
      responseBUiId: responseBId,
      responseAText: comparison.response_a.content || '',
      responseBText: comparison.response_b.content || '',
      responseAModelUsed: comparison.response_a.model_used || '',
      responseBModelUsed: comparison.response_b.model_used || '',
      variantA: comparison.variant_a_name || '',
      variantB: comparison.variant_b_name || '',
      disclosureMode,
      traceMode,
    };
    UI.setABComparisonId(state);
    return state;
  },

  restorePendingABComparisons(comparisons) {
    const restored = [];
    (comparisons || []).forEach((comparison) => {
      const state = this.restorePendingABComparison(comparison);
      if (state) restored.push(state);
    });
    this.state.pendingABComparisons = restored;
    this.syncABPendingState();
  },

  async newConversation() {
    try {
      await API.newConversation();
      this.state.conversationId = null;
      this.state.messages = [];
      this.state.history = [];
      this.state.activeABComparison = null;
      this.state.pendingABComparisons = [];
      this.state.abVotePending = false;
      Storage.setActiveConversationId(null);
      
      UI.renderMessages([]);
      UI.hideABVoteButtons();
      UI.setInputDisabled(false);
      await this.loadConversations();
    } catch (e) {
      console.error('Failed to create conversation:', e);
    }
  },

  async deleteConversation(conversationId) {
    if (!confirm('Delete this conversation?')) return;
    
    try {
      await API.deleteConversation(conversationId);
      
      if (this.state.conversationId === conversationId) {
        this.state.conversationId = null;
        this.state.messages = [];
        this.state.history = [];
        this.state.activeABComparison = null;
        this.state.pendingABComparisons = [];
        this.state.abVotePending = false;
        Storage.setActiveConversationId(null);
        UI.renderMessages([]);
        UI.hideABVoteButtons();
        UI.setInputDisabled(false);
      }
      
      await this.loadConversations();
    } catch (e) {
      console.error('Failed to delete conversation:', e);
    }
  },

  async sendMessage() {
    let text = UI.getInputValue();
    if (!text || this.state.isStreaming) return;

    const selected = this.getSelectedProviderAndModel();
    if (selected.provider && !selected.model) {
      UI.showToast('Please select a model for the chosen provider.');
      return;
    }

    // Block only when the unresolved-comparison limit has been reached
    if (this.hasReachedABPendingLimit()) {
      const limit = this.getABPendingLimit();
      UI.showToast(`Please resolve one of the pending comparisons before continuing (limit: ${limit}).`);
      return;
    }

    // Guard: if the pending playbook is a public one the user hasn't enabled yet,
    // confirm "add and run" before proceeding; abort cleanly on decline.
    {
      const guardName = API._pendingPlaybookName;
      if (guardName) {
        const list = PlaybookMenu.playbooks?.length
          ? PlaybookMenu.playbooks
          : ((await API.getPlaybooksList())?.playbooks || []);
        const pb = list.find(p => p.name === guardName);
        if (pb && pb.is_mine === false && !pb.is_enabled) {
          const ok = window.confirm(`"/${guardName}" is a public playbook. Add it to your list and run it?`);
          if (!ok) {
            API._pendingPlaybookName = null;
            return;
          }
          try { await API.enablePlaybook(pb.id); PlaybookMenu.playbooks = []; }
          catch (err) { API._pendingPlaybookName = null; return; }
        }
      }
    }

    // A /playbook invoke leaves "/name " in the input (autocomplete) — strip it so the stored
    // message, the bubble, and the conversation title stay clean. The playbook still applies via
    // the separately-sent playbook_name + the chip.
    const pendingPlaybook = API._pendingPlaybookName || null;
    if (pendingPlaybook && text.startsWith('/' + pendingPlaybook)) {
      text = text.slice(1 + pendingPlaybook.length).replace(/^\s+/, '');
      if (!text) {
        // Bare "/name": run the playbook with a clean stand-in request so the bubble
        // and conversation title aren't empty (playbooks that need input will ask for it).
        text = `Run the ${pendingPlaybook} playbook.`;
      }
    }

    // Add user message
    const userMsg = {
      id: `${Date.now()}-user`,
      sender: 'User',
      html: Utils.escapeHtml(text),
      playbookName: pendingPlaybook,  // chip; streamResponse clears _pendingPlaybookName next
    };
    this.state.messages.push(userMsg);
    this.state.history.push(['User', text]);
    UI.addMessage(userMsg);

    UI.clearInput();
    UI.setInputDisabled(true, { disableSend: false });
    UI.setStreamingState(true);
    this.state.isStreaming = true;

    // Determine which config to use
    const configA = UI.getSelectedConfig('A');
    let isAB = false;
    if (UI.shouldUseABForNextTurn()) {
      try {
        const decision = await API.getABDecision(this.state.conversationId);
        isAB = decision?.use_ab === true;
      } catch (e) {
        console.warn('Failed to get server-side A/B decision, falling back to single response mode:', e);
      }
    }

    if (isAB) {
      await this.sendABMessage(text, configA, pendingPlaybook);
    } else {
      await this.sendSingleMessage(configA);
    }
  },

  async sendSingleMessage(configName) {
    const msgId = `${Date.now()}-assistant`;
    const assistantMsg = {
      id: msgId,
      sender: 'archi',
      html: '',
      meta: this.getEntryMetaLabel(),
    };
    this.state.messages.push(assistantMsg);
    UI.addMessage(assistantMsg);

    try {
      await this.streamResponse(msgId, configName);
    } catch (e) {
      console.error('Streaming error:', e);
    } finally {
      this.state.isStreaming = false;
      UI.setInputDisabled(false);
      UI.setStreamingState(false);
      UI.elements.inputField?.focus();
      await this.loadConversations();
    }
  },

  async sendABMessage(userText, configA, playbookName = null) {
    if (!this.state.abPool) {
      UI.showToast('A/B pool is not configured on the server. Cannot run comparison.');
      this.state.isStreaming = false;
      this.syncABPendingState();
      UI.setStreamingState(false);
      return;
    }

    // The pending one-shot /playbook selection was captured by sendMessage and is passed in
    // explicitly so both arms get the same input; clear it so it can't leak into a later send.
    API._pendingPlaybookName = null;

    const msgIdA = `${Date.now()}-ab-a`;
    const msgIdB = `${Date.now()}-ab-b`;
    const traceMode = UI.getABTraceMode();
    const disclosureMode = UI.getABDisclosureMode();
    const { provider, model } = this.getSelectedProviderAndModel();

    // Create side-by-side container using normal message styling
    UI.addABComparisonContainer(msgIdA, msgIdB, {
      traceMode,
      comparisonKey: `${msgIdA}-${msgIdB}`,
    });

    const armTexts = { a: '', b: '' };
    const armTraces = {
      a: { toolCalls: new Map(), events: [] },
      b: { toolCalls: new Map(), events: [] },
    };
    const finalEvents = { a: null, b: null };
    let abMeta = null;

    try {
      this.state.abortController = new AbortController();

      for await (const event of API.streamABComparison(
        this.state.history,
        this.state.conversationId,
        configA,
        this.state.abortController?.signal,
        provider,
        model,
        playbookName,
      )) {
        if (event.type === 'meta' && event.event === 'stream_started') {
          continue; // padding event
        }

        if (event.type === 'error') {
          const errMsg = event.message || 'A/B stream error';
          UI.showABError(errMsg);
          this.state.isStreaming = false;
          this.syncABPendingState();
          UI.setStreamingState(false);
          this.state.abortController = null;
          await this.loadConversations();
          return;
        }

        if (event.type === 'ab_arms') {
          UI.updateABArmPresentation(msgIdA, { variantName: event.arm_a_name }, {
            disclosureMode: event.variant_label_mode || event.disclosure_mode || disclosureMode,
            reveal: false,
          });
          UI.updateABArmPresentation(msgIdB, { variantName: event.arm_b_name }, {
            disclosureMode: event.variant_label_mode || event.disclosure_mode || disclosureMode,
            reveal: false,
          });
          continue;
        }

        if (event.type === 'ab_meta') {
          abMeta = event;
          // Update conversation_id if server assigned one
          if (event.conversation_id != null) {
            this.state.conversationId = event.conversation_id;
            Storage.setActiveConversationId(event.conversation_id);
          }
          continue;
        }

        const arm = event.arm; // 'a' or 'b'
        if (!arm) continue;
        const targetId = arm === 'a' ? msgIdA : msgIdB;
        const traceState = armTraces[arm];

        if (event.type === 'text' || event.type === 'chunk') {
          const content = event.content || '';
          if (content) {
            armTexts[arm] = content; // Server sends accumulated text
            UI.updateABResponse(targetId, Markdown.render(armTexts[arm]), true);
          }
        } else if (event.type === 'final') {
          const finalText = event.response || armTexts[arm] || '';
          armTexts[arm] = finalText;
          finalEvents[arm] = event;
          UI.updateABResponse(targetId, Markdown.render(finalText), false);
          UI.finalizeTrace(targetId, traceState, event);
          if (abMeta) {
            UI.updateABArmPresentation(targetId, {
              variantName: arm === 'a' ? abMeta.arm_a_variant : abMeta.arm_b_variant,
              modelUsed: event.model_used || '',
            }, {
              disclosureMode: abMeta.variant_label_mode || abMeta.disclosure_mode || disclosureMode,
              reveal: false,
            });
          }
        } else if (event.type === 'step' && event.step_type === 'agent') {
          const content = event.content || '';
          if (content) {
            armTexts[arm] = content;
            UI.updateABResponse(targetId, Markdown.render(armTexts[arm]), true);
          }
        } else {
          if (event.type === 'tool_start') {
            traceState.toolCalls.set(event.tool_call_id, {
              name: event.tool_name,
              args: event.tool_args,
              status: 'running',
            });
            traceState.events.push(event);
          } else if (event.type === 'tool_output') {
            const toolData = traceState.toolCalls.get(event.tool_call_id);
            if (toolData) {
              toolData.output = event.output;
              toolData.status = 'success';
            }
            traceState.events.push(event);
          } else if (event.type === 'tool_end') {
            const toolData = traceState.toolCalls.get(event.tool_call_id);
            if (toolData) {
              toolData.status = event.status;
              toolData.duration = event.duration_ms;
            }
            traceState.events.push(event);
          } else if (event.type === 'thinking_start' || event.type === 'thinking_end') {
            traceState.events.push(event);
          }
          this._renderStreamEvent(targetId, event);
        }
      }

      // Finalize both arms (remove streaming cursor)
      UI.updateABResponse(msgIdA, Markdown.render(armTexts.a), false);
      UI.updateABResponse(msgIdB, Markdown.render(armTexts.b), false);

      // Set up voting if we got a comparison_id
      if (abMeta?.comparison_id) {
        if (abMeta.arm_a_message_id) {
          UI.rekeyABArm(msgIdA, abMeta.arm_a_message_id);
        }
        if (abMeta.arm_b_message_id) {
          UI.rekeyABArm(msgIdB, abMeta.arm_b_message_id);
        }
        const comparisonState = {
          comparisonId: abMeta.comparison_id,
          responseAId: abMeta.arm_a_message_id,
          responseBId: abMeta.arm_b_message_id,
          responseAUiId: abMeta.arm_a_message_id || msgIdA,
          responseBUiId: abMeta.arm_b_message_id || msgIdB,
          responseAText: armTexts.a,
          responseBText: armTexts.b,
          responseAModelUsed: finalEvents.a?.model_used || abMeta.arm_a_model_used || '',
          responseBModelUsed: finalEvents.b?.model_used || abMeta.arm_b_model_used || '',
          variantA: abMeta.arm_a_variant,
          variantB: abMeta.arm_b_variant,
          disclosureMode: abMeta.variant_label_mode || abMeta.disclosure_mode || disclosureMode,
          traceMode,
        };
        UI.setABComparisonId(comparisonState);
        this.addPendingABComparison(comparisonState);
      } else {
        UI.showToast('Comparison completed without a recorded vote state. Input has been re-enabled.');
        UI.setInputDisabled(false);
      }

      // Highlight code
      if (typeof hljs !== 'undefined') {
        setTimeout(() => hljs.highlightAll(), 0);
      }

    } catch (e) {
      console.error('A/B comparison error:', e);
      UI.stopTraceTimer(msgIdA);
      UI.stopTraceTimer(msgIdB);
      UI.showABError(e.message || 'Failed to create comparison');
      this.state.isStreaming = false;
      this.syncABPendingState();
      UI.setStreamingState(false);
      this.state.abortController = null;
      await this.loadConversations();
      return;
    }

    this.state.isStreaming = false;
    UI.setStreamingState(false);
    this.state.abortController = null;
    this.syncABPendingState();
    await this.loadConversations();
  },

  async submitABPreference(preference) {
    if (!this.state.activeABComparison) return;

    try {
      const activeComparison = this.state.activeABComparison;
      const result = await API.submitABPreference(activeComparison.comparisonId, preference);
      if (result?.updated === false) {
        console.info('A/B preference already recorded for comparison', activeComparison.comparisonId);
      }

      // Update UI to show result
      UI.markABWinner(preference, activeComparison);
      UI.hideABVoteButtons();

      // Add the chosen response to history for context
      let winningText;
      if (preference === 'tie') {
        // For ties, use response A (arbitrary)
        winningText = activeComparison.responseAText;
      } else if (preference === 'b') {
        winningText = activeComparison.responseBText;
      } else {
        winningText = activeComparison.responseAText;
      }
      this.state.history.push(['archi', winningText]);

      // Clear A/B state
      this.removePendingABComparison(activeComparison.comparisonId);
      UI.elements.inputField?.focus();
      await this.loadConversations();
    } catch (e) {
      console.error('Failed to submit preference:', e);
      UI.showToast('Failed to submit preference. Please try again.');
    }
  },

  cancelPendingABComparison() {
    // Called when user disables A/B mode while vote is pending
    if (!this.state.abVotePending) return;

    const activeComparison = this.state.activeABComparison;
    if (!activeComparison) return;

    // Submit 'tie' as a skip preference so the comparison is resolved in the DB
    if (activeComparison.comparisonId) {
      API.submitABPreference(activeComparison.comparisonId, 'tie')
        .catch(e => console.warn('Failed to submit skip preference:', e));
    }

    // Add response A to history as default
    if (activeComparison.responseAText) {
      this.state.history.push(['archi', activeComparison.responseAText]);
    }

    // Mark as tie/skipped visually
    UI.markABWinner('tie', activeComparison);
    UI.hideABVoteButtons();

    // Clear state
    this.removePendingABComparison(activeComparison.comparisonId);
    UI.showToast('A/B comparison skipped');
  },

  /**
   * Dispatch a single streaming event to the appropriate UI renderer.
   * Shared between regular streaming and A/B comparison streaming so that
   * tool/thinking rendering logic is defined in exactly one place.
   */
  _renderStreamEvent(messageId, event) {
    const showTrace = UI.isTraceVisibleMode(UI.getTraceModeForMessage(messageId));
    if (!showTrace) return;
    switch (event.type) {
      case 'playbook_applied':
        UI.renderPlaybookApplied(messageId, event);
        break;
      case 'tool_start':
        UI.renderToolStart(messageId, event);
        break;
      case 'tool_output':
        UI.renderToolOutput(messageId, event);
        UI.renderToolEnd(messageId, { tool_call_id: event.tool_call_id, status: 'success' });
        break;
      case 'tool_end':
        UI.renderToolEnd(messageId, event);
        break;
      case 'thinking_start':
        UI.renderThinkingStart(messageId, event);
        break;
      case 'thinking_end':
        UI.renderThinkingEnd(messageId, event);
        break;
      case 'narration':
        UI.renderNarrationStep(messageId, event);
        break;
    }
  },

  async streamResponse(messageId, configName) {
    let streamedText = '';
    
    // Initialize trace state for this stream
    this.state.activeTrace = {
      traceId: null,
      events: [],
      toolCalls: new Map(), // Map<toolCallId, { name, args, status, output, duration }>
    };

    // Create abort controller for cancellation
    this.state.abortController = new AbortController();
    let timeoutId = null;
    let timedOut = false;

    const resetTimeout = () => {
      if (!CONFIG.STREAMING.TIMEOUT) return;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      timeoutId = setTimeout(() => {
        timedOut = true;
        this.state.abortController?.abort();
      }, CONFIG.STREAMING.TIMEOUT);
    };

    // Create trace container if in verbose/normal mode
    const showTrace = this.state.traceVerboseMode !== 'minimal';
    if (showTrace) {
      UI.createTraceContainer(messageId);
    }

    try {
      // Get selected provider and model
      const { provider, model } = this.getSelectedProviderAndModel();

      resetTimeout();
      
      for await (const event of API.streamResponse(
        this.state.history,
        this.state.conversationId,
        configName,
        this.state.abortController.signal,
        provider,
        model
      )) {
        resetTimeout();
        // Handle trace events
        if (event.type === 'tool_start') {
          // Text streamed before a tool call is that turn's narration:
          // move it into the activity timeline before the tool step lands.
          if (streamedText.trim()) {
            const narration = {
              type: 'narration',
              step_id: `narration-${this.state.activeTrace.events.length}`,
              content: streamedText.trim(),
            };
            this.state.activeTrace.events.push(narration);
            this._renderStreamEvent(messageId, narration);
            streamedText = '';
          }
          this.state.activeTrace.toolCalls.set(event.tool_call_id, {
            name: event.tool_name,
            args: event.tool_args,
            status: 'running',
            output: null,
            duration: null,
          });
          this.state.activeTrace.events.push(event);
          this._renderStreamEvent(messageId, event);
        } else if (event.type === 'tool_output') {
          const toolData = this.state.activeTrace.toolCalls.get(event.tool_call_id);
          if (toolData) {
            toolData.output = event.output;
            toolData.status = 'success';
          }
          this.state.activeTrace.events.push(event);
          this._renderStreamEvent(messageId, event);
        } else if (event.type === 'tool_end') {
          const toolData = this.state.activeTrace.toolCalls.get(event.tool_call_id);
          if (toolData) {
            toolData.status = event.status;
            toolData.duration = event.duration_ms;
          }
          this.state.activeTrace.events.push(event);
          this._renderStreamEvent(messageId, event);
        } else if (event.type === 'thinking_start' || event.type === 'thinking_end') {
          this.state.activeTrace.events.push(event);
          this._renderStreamEvent(messageId, event);
        } else if (event.type === 'playbook_applied') {
          this.state.activeTrace.events.push(event);
          this._renderStreamEvent(messageId, event);
        } else if (event.type === 'chunk') {
          // Chunks may be accumulated or delta content
          if (event.accumulated) {
            streamedText = event.content || '';
          } else {
            streamedText += event.content || '';
          }
          UI.updateMessage(messageId, {
            html: Markdown.render(streamedText),
            streaming: true,
          });
        } else if (event.type === 'step' && event.step_type === 'agent') {
          // Agent steps may contain full accumulated content
          const content = event.content || '';
          if (content) {
            streamedText = content;
            UI.updateMessage(messageId, {
              html: Markdown.render(streamedText),
              streaming: true,
            });
          }
        } else if (event.type === 'final') {
          const finalText = event.response || streamedText;
          
          // Store trace ID
          if (event.trace_id) {
            this.state.activeTrace.traceId = event.trace_id;
          }
          
          // Finalize trace display with usage data
          if (showTrace) {
            UI.finalizeTrace(messageId, this.state.activeTrace, event);
          }
          
          UI.updateMessage(messageId, {
            html: Markdown.render(finalText),
            streaming: false,
          });
          
          // Update model label from actual model used
          if (event.model_used) {
            const msg = this.state.messages.find(m => m.id === messageId);
            if (msg) msg.meta = event.model_used;
            UI.updateMessage(messageId, { meta: event.model_used });
          }

          // Update message ID from backend so feedback works
          if (event.message_id != null) {
            const msg = this.state.messages.find(m => m.id === messageId);
            if (msg) msg.id = event.message_id;
            const msgEl = document.querySelector(`[data-id="${messageId}"]`);
            if (msgEl) msgEl.dataset.id = event.message_id;
          }

          if (event.conversation_id != null) {
            this.state.conversationId = event.conversation_id;
            Storage.setActiveConversationId(event.conversation_id);
          }
          
          this.state.history.push(['archi', finalText]);
          
          // Re-highlight code blocks
          if (typeof hljs !== 'undefined') {
            setTimeout(() => hljs.highlightAll(), 0);
          }
          return;
        } else if (event.type === 'error') {
          UI.updateMessage(messageId, {
            html: `<p style="color: var(--error-text);">${Utils.escapeHtml(event.message || 'An error occurred')}</p>`,
            streaming: false,
          });
          return;
        } else if (event.type === 'cancelled') {
          UI.updateMessage(messageId, {
            html: streamedText 
              ? Markdown.render(streamedText) + '<p class="cancelled-notice"><em>Response cancelled</em></p>'
              : '<p class="cancelled-notice"><em>Response cancelled</em></p>',
            streaming: false,
          });
          return;
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        UI.updateMessage(messageId, {
          html: timedOut
            ? `<p class="cancelled-notice"><em>${Utils.escapeHtml(CONFIG.MESSAGES.CLIENT_TIMEOUT)}</em></p>`
            : streamedText 
              ? Markdown.render(streamedText) + '<p class="cancelled-notice"><em>Response cancelled</em></p>'
              : '<p class="cancelled-notice"><em>Response cancelled</em></p>',
          streaming: false,
        });
        return;
      }
      console.error('Stream error:', e);
      UI.updateMessage(messageId, {
        html: `<p style="color: var(--error-text);">${Utils.escapeHtml(e.message || 'Streaming failed')}</p>`,
        streaming: false,
      });
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (showTrace) {
        UI.stopTraceTimer(messageId);
      }
      this.state.abortController = null;
      this.state.activeTrace = null;
    }
  },

  async cancelStream() {
    if (this.state.abortController) {
      this.state.abortController.abort();
      this.state.isStreaming = false;
      UI.setInputDisabled(false);
      UI.setStreamingState(false);
      this.state.abortController = null;
      
      // Also notify server
      if (this.state.conversationId) {
        try {
          await fetch(CONFIG.ENDPOINTS.CANCEL_STREAM, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              conversation_id: this.state.conversationId,
              client_id: Storage.getClientId(),
            }),
          });
        } catch (e) {
          console.error('Failed to notify server of cancellation:', e);
        }
      }
    }
  },

  setTraceVerboseMode(mode) {
    if (['minimal', 'normal', 'verbose'].includes(mode)) {
      this.state.traceVerboseMode = mode;
      localStorage.setItem(CONFIG.STORAGE_KEYS.TRACE_VERBOSE_MODE, mode);
    }
  },

  showPlaybooksView(view, title) {
    const panel = document.querySelector('.playbooks-panel');
    if (panel) panel.dataset.view = view;
    const back = document.querySelector('.playbooks-back');
    if (back) back.hidden = (view !== 'editor');
    const titleEl = document.querySelector('#playbooks-title');
    if (titleEl) titleEl.textContent = (view === 'editor' ? (title || 'Playbook') : 'Playbooks');
  },

  enterPlaybooksSection() {
    // Called whenever Settings switches to the Playbooks section: reset to the
    // Active tab, unfiltered, list view, and (re)load the catalog.
    Chat._panelTab = 'mine';
    const _search = document.querySelector('.playbooks-search');
    if (_search) _search.value = '';                  // start unfiltered each open
    Chat.showPlaybooksView('list');
    this.loadPlaybooksPanel();
  },

  async loadPlaybooksPanel() {
    if (!document.querySelector('.playbooks-list')) return;
    try {
      const data = await API.getPlaybooksList();
      Chat._panelPlaybooks = data?.playbooks || [];
    } catch (e) {
      Chat._panelPlaybooks = null;  // signals "could not load"
    }
    Chat.renderPlaybooksPanel();
  },

  renderPlaybooksPanel() {
    const list = document.querySelector('.playbooks-list');
    const tabsEl = document.querySelector('.playbooks-tabs');
    if (!list) return;
    if (Chat._panelPlaybooks === null) {
      if (tabsEl) tabsEl.innerHTML = '';
      list.innerHTML = '<div class="playbooks-status">Could not load playbooks.</div>';
      return;
    }
    const all = Chat._panelPlaybooks || [];
    if (!all.length) {
      if (tabsEl) tabsEl.innerHTML = '';
      list.innerHTML = '<div class="playbook-menu-desc">No playbooks yet. Use "New playbook", or ask the agent to save one.</div>';
      return;
    }
    const q = (document.querySelector('.playbooks-search')?.value || '').trim().toLowerCase();
    const matchesQuery = (s) => !q
      || s.name.toLowerCase().includes(q)
      || (s.description || '').toLowerCase().includes(q);
    const mine = all.filter(s => s.is_enabled !== false).filter(matchesQuery);
    const pub = all.filter(s => s.is_enabled === false).filter(matchesQuery);

    const tab = Chat._panelTab === 'public' ? 'public' : 'mine';  // default to Active
    if (tabsEl) {
      tabsEl.innerHTML =
        `<button class="playbooks-tab${tab === 'mine' ? ' active' : ''}" data-tab="mine" type="button" role="tab">Active (${mine.length})</button>`
        + `<button class="playbooks-tab${tab === 'public' ? ' active' : ''}" data-tab="public" type="button" role="tab">Add from public (${pub.length})</button>`;
    }

    const rowHtml = (s) => {
      const isMine = s.is_mine !== false;
      const badge = s.visibility === 'public'
        ? `<span class="playbook-badge">public${!isMine && s.owner ? ' · ' + Utils.escapeHtml(s.owner) : ''}</span>`
        : '';
      const actions = isMine
        ? `<button class="playbook-edit" data-id="${s.id}" type="button">Edit</button>
           <button class="playbook-delete" data-id="${s.id}" data-name="${Utils.escapeHtml(s.name)}" type="button">Delete</button>`
        : (s.is_enabled
            ? `<button class="playbook-view" data-id="${s.id}" type="button">View</button>
               <button class="playbook-remove" data-id="${s.id}" type="button">Remove</button>`
            : `<button class="playbook-view" data-id="${s.id}" type="button">View</button>
               <button class="playbook-add" data-id="${s.id}" type="button">Add</button>`);
      return `
          <div class="playbook-row" data-id="${s.id}">
            <div><strong>${Utils.escapeHtml(s.name)}</strong>${badge}
              <div class="playbook-menu-desc">${Utils.escapeHtml(s.description || '')}</div></div>
            <div class="playbook-row-actions">${actions}</div>
          </div>`;
    };

    const items = tab === 'mine' ? mine : pub;
    const emptyMsg = q ? 'No matches.' : (tab === 'mine' ? 'Nothing here yet.' : 'None to add.');
    if (!items.length) {
      list.innerHTML = `<div class="playbook-menu-desc">${emptyMsg}</div>`;
    } else if (tab === 'mine') {
      // The Active tab groups what you own and what you added from public.
      const yours = items.filter(s => s.is_mine !== false);
      const added = items.filter(s => s.is_mine === false);
      let html = '';
      if (yours.length) html += `<div class="playbooks-group-title">Yours</div>` + yours.map(rowHtml).join('');
      if (added.length) html += `<div class="playbooks-group-title">Added from public</div>` + added.map(rowHtml).join('');
      list.innerHTML = html;
    } else {
      list.innerHTML = items.map(rowHtml).join('');
    }
  },

  // The editor's input fields, keyed by selector — the single source used to
  // populate, reset and (un)lock them, so the call sites cannot drift.
  playbookEditorFields: {
    '#playbook-name': 'name',
    '#playbook-description': 'description',
    '#playbook-visibility': 'visibility',
    '#playbook-body': 'body',
  },

  setPlaybookEditorFields(playbook = {}) {
    Object.entries(this.playbookEditorFields).forEach(([sel, field]) => {
      const el = document.querySelector(sel);
      if (!el) return;
      if (field === 'visibility') el.value = playbook.visibility || 'private';
      else el.value = playbook[field] || '';
    });
    const status = document.querySelector('#playbooks-status');
    if (status) status.textContent = '';
  },

  setPlaybookEditorReadOnly(readOnly) {
    this._playbookEditorReadOnly = !!readOnly;
    Object.keys(this.playbookEditorFields).forEach(sel => {
      const el = document.querySelector(sel);
      if (el) el.disabled = this._playbookEditorReadOnly;
    });
    const save = document.querySelector('.playbook-save');
    if (save) save.hidden = this._playbookEditorReadOnly;
  },

  async savePlaybookFromPanel() {
    if (this._playbookEditorReadOnly) return;  // viewing a public playbook — nothing to save
    const name = document.querySelector('#playbook-name')?.value.trim();
    const description = document.querySelector('#playbook-description')?.value.trim();
    const visibility = document.querySelector('#playbook-visibility')?.value || 'private';
    const body = document.querySelector('#playbook-body')?.value;
    const status = document.querySelector('#playbooks-status');
    const editingId = this._editingPlaybookId || null;
    try {
      if (editingId) await API.updatePlaybook(editingId, { name, description, body, visibility });
      else await API.createPlaybook({ name, description, body, visibility });
      if (status) { status.textContent = 'Saved.'; }
      this._editingPlaybookId = null;
      Chat.showPlaybooksView('list');
      await this.loadPlaybooksPanel();
      PlaybookMenu.playbooks = [];
    } catch (e) {
      if (status) status.textContent = e.message || 'Could not save playbook.';
    }
  },
};

window.__ARCHI_PLAYWRIGHT__ = {
  ab: {
    streamOverride: null,

    setStreamOverride(override) {
      this.streamOverride = typeof override === 'function' ? override : null;
    },

    clearStreamOverride() {
      this.streamOverride = null;
    },

    patchPoolState(patch = {}) {
      Chat.state.abPool = {
        ...(Chat.state.abPool || {}),
        ...patch,
      };
      if (typeof UI.updateABPoolUI === 'function') {
        UI.updateABPoolUI(Chat.state.abPool || {});
      }
      return Chat.state.abPool;
    },

    reset() {
      this.streamOverride = null;
    },
  },
};

// =============================================================================
// Initialize on DOM ready
// =============================================================================

document.addEventListener('DOMContentLoaded', () => Chat.init());
