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
    SELECTED_PROVIDER_B: 'archi_selected_provider_b',
    SELECTED_MODEL_B: 'archi_selected_model_b',
  },
  ENDPOINTS: {
    STREAM: '/api/get_chat_response_stream',
    CONFIGS: '/api/get_configs',
    CONVERSATIONS: '/api/list_conversations',
    LOAD_CONVERSATION: '/api/load_conversation',
    NEW_CONVERSATION: '/api/new_conversation',
    DELETE_CONVERSATION: '/api/delete_conversation',
    AB_CREATE: '/api/ab/create',
    AB_PREFERENCE: '/api/ab/preference',
    AB_PENDING: '/api/ab/pending',
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
    LIKE: '/api/like',
    DISLIKE: '/api/dislike',
    TEXT_FEEDBACK: '/api/text_feedback',
  },
  STREAMING: {
    TIMEOUT: 300000, // 5 minutes
  },
  TRACE: {
    MAX_TOOL_OUTPUT_PREVIEW: 500,
    AUTO_COLLAPSE_TOOL_COUNT: 5,
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

  async *streamResponse(history, conversationId, configName, signal = null, provider = null, model = null) {
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
        include_agent_steps: true,  // Required for streaming chunks
        include_tool_steps: true,   // Enable tool step events for trace
        provider: provider,  // Provider-based model selection
        model: model,        // Model ID/name for the provider
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

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          
          try {
            yield JSON.parse(trimmed);
          } catch (e) {
            console.error('Failed to parse stream event:', e);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  // A/B Testing API methods
  async createABComparison(data) {
    return this.fetchJson(CONFIG.ENDPOINTS.AB_CREATE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...data,
        client_id: this.clientId,
      }),
    });
  },

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
      modelSelectB: document.querySelector('.model-select-b'),
      settingsBtn: document.querySelector('.settings-btn'),
      dataTab: document.getElementById('data-tab'),
      settingsModal: document.querySelector('.settings-modal'),
      settingsBackdrop: document.querySelector('.settings-backdrop'),
      settingsClose: document.querySelector('.settings-close'),
      abCheckbox: document.querySelector('.ab-checkbox'),
      abModelGroup: document.querySelector('.ab-model-group'),
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
      providerSelectB: document.getElementById('provider-select-b'),
      providerStatus: document.getElementById('provider-status'),
      customModelInput: document.getElementById('custom-model-input'),
      customModelRow: document.getElementById('custom-model-row'),
      activeModelLabel: document.getElementById('active-model-label'),
      darkModeToggle: document.getElementById('dark-mode-toggle'),
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
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        Chat.handleSendOrStop();
      }
    });
    
    // Auto-resize textarea
    this.elements.inputField?.addEventListener('input', () => this.autoResizeInput());
    
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
      // Handle inline delete confirmation buttons
      if (target.closest('.agent-dropdown-confirm-yes')) {
        const name = row.dataset.agentName;
        this.closeAgentDropdown();
        this.doDeleteAgent(name);
        return;
      }
      if (target.closest('.agent-dropdown-confirm-no')) {
        // Cancel: re-render list to remove confirmation state
        Chat.loadAgents();
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
    // Resize handle for agent spec modal
    this.initAgentSpecResize();
    
    // A/B toggle in settings
    this.elements.abCheckbox?.addEventListener('change', (e) => {
      const isEnabled = e.target.checked;
      if (isEnabled) {
        // Show warning modal before enabling
        const dismissed = sessionStorage.getItem(CONFIG.STORAGE_KEYS.AB_WARNING_DISMISSED);
        if (!dismissed) {
          e.target.checked = false; // Reset checkbox
          this.showABWarningModal(
            () => {
              // On confirm
              e.target.checked = true;
              if (this.elements.abModelGroup) {
                this.elements.abModelGroup.style.display = 'block';
              }
              sessionStorage.setItem(CONFIG.STORAGE_KEYS.AB_WARNING_DISMISSED, 'true');
            },
            () => {
              // On cancel
              e.target.checked = false;
            }
          );
          return;
        }
      }
      if (this.elements.abModelGroup) {
        this.elements.abModelGroup.style.display = isEnabled ? 'block' : 'none';
      }
      // If disabling A/B mode while vote is pending, re-enable input
      if (!isEnabled && Chat.state.abVotePending) {
        Chat.cancelPendingABComparison();
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

    this.elements.providerSelectB?.addEventListener('change', (e) => {
      Chat.handleProviderBChange(e.target.value);
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
    this.agentSpecMode = mode;
    this.agentSpecName = name;
    // Restore persisted size
    this.restoreAgentSpecSize();
    if (this.elements.agentSpecTitle) {
      this.elements.agentSpecTitle.textContent = mode === 'edit' ? `Edit ${name || 'Agent'}` : 'New Agent';
    }
    // Update reset button label
    if (this.elements.agentSpecReset) {
      this.elements.agentSpecReset.textContent = mode === 'edit' ? 'Revert changes' : 'Reset template';
    }
    // Clear validation errors
    this.clearAgentSpecValidation();
    if (mode === 'edit' && name) {
      await this.loadAgentToolPalette();
      await this.loadAgentSpecByName(name);
    } else {
      await this.loadAgentSpecTemplate();
    }
    // Auto-focus name in create mode
    if (mode === 'create') {
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
  serialiseAgentSpec(name, tools, prompt) {
    let yaml = `---\nname: ${name}\n`;
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
    if (this.agentSpecMode === 'edit' && this.agentSpecName) {
      // Revert to saved version
      this.loadAgentSpecByName(this.agentSpecName);
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
      const toolName = tool.name || '';
      const checked = selectedTools.includes(toolName) ? 'checked' : '';
      return `
      <label class="agent-spec-tool">
        <input type="checkbox" class="agent-spec-tool-checkbox" value="${Utils.escapeHtml(toolName)}" ${checked} />
        <div class="agent-spec-tool-info">
          <div class="agent-spec-tool-name">${Utils.escapeHtml(toolName)}</div>
          <div class="agent-spec-tool-desc">${Utils.escapeHtml(tool.description || '')}</div>
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
        existing_name: this.agentSpecName || null,
      });
      if (this.agentSpecMode === 'edit') {
        const savedName = Utils.normalizeAgentName(response?.name || this.agentSpecName || '');
        if (savedName) {
          this.agentSpecName = savedName;
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
    return this.elements.abCheckbox?.checked ?? false;
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

    // Also populate provider B select for A/B testing
    const selectB = this.elements.providerSelectB;
    if (selectB) {
      selectB.innerHTML = '<option value="">Same as primary</option>' +
        enabledProviders
          .map(p => `<option value="${Utils.escapeHtml(p.type)}">${Utils.escapeHtml(p.display_name)}</option>`)
          .join('');
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

  renderModelBOptions(models, selectedModel = null, providerType = null) {
    const select = this.elements.modelSelectB;
    if (!select) return;

    if (!models || models.length === 0) {
      select.innerHTML = '<option value="">No models available</option>';
      return;
    }

    const options = models
      .map(m => `<option value="${Utils.escapeHtml(m.id)}">${Utils.escapeHtml(m.display_name || m.name)}</option>`)
      .join('');
    const customOption = providerType === 'openrouter'
      ? '<option value="__custom__">Custom model…</option>'
      : '';
    select.innerHTML = options + customOption;

    if (selectedModel === '__custom__' && providerType === 'openrouter') {
      select.value = '__custom__';
    } else if (selectedModel && models.some(m => m.id === selectedModel)) {
      select.value = selectedModel;
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
              <li><strong>Voting required</strong> - You must choose the better response before continuing</li>
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

  addABComparisonContainer(msgIdA, msgIdB) {
    // Remove empty state if present
    const empty = this.elements.messagesInner?.querySelector('.messages-empty');
    if (empty) empty.remove();

    const showTrace = Chat.state.traceVerboseMode !== 'minimal';
    const traceIconSvg = `<svg class="trace-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>`;
    const traceHtml = (id) => showTrace ? `
          <div class="trace-container ab-trace-container" data-message-id="${id}">
            <div class="trace-header" onclick="UI.toggleTraceExpanded('${id}')">
              ${traceIconSvg}
              <span class="trace-label">Agent Activity</span>
              <span class="toggle-icon">▼</span>
            </div>
            <div class="trace-content"></div>
          </div>` : '';

    const html = `
      <div class="ab-comparison" id="ab-comparison-active">
        <div class="ab-response ab-response-a" data-id="${msgIdA}">
          <div class="ab-response-header">
            <span class="ab-response-label">Model A</span>
          </div>
          ${traceHtml(msgIdA)}
          <div class="ab-response-content message-content"></div>
        </div>
        <div class="ab-response ab-response-b" data-id="${msgIdB}">
          <div class="ab-response-header">
            <span class="ab-response-label">Model B</span>
          </div>
          ${traceHtml(msgIdB)}
          <div class="ab-response-content message-content"></div>
        </div>
      </div>`;

    this.elements.messagesInner?.insertAdjacentHTML('beforeend', html);
    this.scrollToBottom();
  },

  updateABResponse(responseId, html, streaming = false) {
    const container = document.querySelector(`.ab-response[data-id="${responseId}"]`);
    if (!container) return;

    const contentEl = container.querySelector('.ab-response-content');
    if (contentEl) {
      contentEl.innerHTML = html;
      if (streaming) {
        contentEl.innerHTML += '<span class="streaming-cursor"></span>';
      }
    }
    this.scrollToBottom();
  },

  showABVoteButtons(comparisonId) {
    const comparison = document.getElementById('ab-comparison-active');
    if (!comparison) return;

    const voteHtml = `
      <div class="ab-vote-container" data-comparison-id="${comparisonId}">
        <div class="ab-vote-prompt">Which response was better?</div>
        <div class="ab-vote-buttons">
          <button class="ab-vote-btn ab-vote-btn-a" data-vote="a">
            <span class="ab-vote-icon">👍</span>
            <span>Model A</span>
          </button>
          <button class="ab-vote-btn ab-vote-btn-b" data-vote="b">
            <span class="ab-vote-icon">👍</span>
            <span>Model B</span>
          </button>
        </div>
      </div>`;

    comparison.insertAdjacentHTML('afterend', voteHtml);

    // Bind vote button events
    document.querySelectorAll('.ab-vote-btn').forEach((btn) => {
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

  markABWinner(preference) {
    const comparison = document.getElementById('ab-comparison-active');
    if (!comparison) return;

    const responseA = comparison.querySelector('.ab-response-a');
    const responseB = comparison.querySelector('.ab-response-b');

    let winnerContent = '';
    let winnerTrace = '';
    if (preference === 'a') {
      winnerContent = responseA?.querySelector('.ab-response-content')?.innerHTML || '';
      winnerTrace = responseA?.querySelector('.trace-container')?.outerHTML || '';
    } else if (preference === 'b') {
      winnerContent = responseB?.querySelector('.ab-response-content')?.innerHTML || '';
      winnerTrace = responseB?.querySelector('.trace-container')?.outerHTML || '';
    } else {
      // Tie - keep both visible but mark them
      responseA?.classList.add('ab-response-tie');
      responseB?.classList.add('ab-response-tie');
      comparison.removeAttribute('id');
      return;
    }

    // Replace the entire comparison with a normal archi message (matching createMessageHTML format)
    // Include the trace container from the winning response
    const metaLabel = Chat.getEntryMetaLabel();
    const metaHtml = metaLabel
      ? `<div class="message-meta">${Utils.escapeHtml(metaLabel)}</div>`
      : '';

    const normalMessage = `
      <div class="message assistant" data-id="ab-winner-${Date.now()}">
        <div class="message-inner">
          <div class="message-header">
            <div class="message-avatar">✦</div>
            <span class="message-sender">archi</span>
          </div>
          ${winnerTrace}
          <div class="message-content">${winnerContent}</div>
          ${metaHtml}
        </div>
      </div>`;

    comparison.outerHTML = normalMessage;
  },

  removeABComparisonContainer() {
    document.getElementById('ab-comparison-active')?.remove();
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

    const traceIconSvg = `<svg class="trace-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg>`;
    const traceHtml = `
      <div class="trace-container" data-message-id="${messageId}">
        <div class="trace-header">
          ${traceIconSvg}
          <span class="trace-label">Agent Activity</span>
          <span class="trace-timer" data-start="${Date.now()}">0.0s</span>
          <button class="trace-toggle" aria-label="Toggle agent activity details" title="Toggle agent activity" onclick="UI.toggleTraceExpanded('${messageId}')">
            <span class="toggle-icon" aria-hidden="true">&#9660;</span>
          </button>
        </div>
        <div class="trace-content">
          <div class="context-meter" style="display: none;">
            <div class="meter-bar"><div class="meter-fill"></div></div>
            <span class="meter-label">Calculating...</span>
          </div>
          <div class="step-timeline"></div>
        </div>
      </div>`;

    inner.insertAdjacentHTML('afterbegin', traceHtml);
    
    // Start elapsed timer
    this.startTraceTimer(messageId);
  },

  startTraceTimer(messageId) {
    const timerEl = document.querySelector(`.trace-container[data-message-id="${messageId}"] .trace-timer`);
    if (!timerEl) return;
    
    const startTime = parseInt(timerEl.dataset.start, 10);
    const updateTimer = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      timerEl.textContent = elapsed.toFixed(1) + 's';
    };
    
    const intervalId = setInterval(updateTimer, 100);
    timerEl.dataset.intervalId = intervalId;
  },

  stopTraceTimer(messageId) {
    const timerEl = document.querySelector(`.trace-container[data-message-id="${messageId}"] .trace-timer`);
    if (!timerEl || !timerEl.dataset.intervalId) return;
    
    clearInterval(parseInt(timerEl.dataset.intervalId, 10));
    delete timerEl.dataset.intervalId;
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
    const step = document.querySelector(`.thinking-step[data-step-id="${event.step_id}"]`);
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

  // =========================================================================
  // Tool Step Rendering (Timeline Style)
  // =========================================================================

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
    if (Chat.state.traceVerboseMode === 'verbose') {
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
    const step = document.querySelector(`.tool-step[data-tool-call-id="${event.tool_call_id}"]`);
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
    const step = document.querySelector(`.tool-step[data-tool-call-id="${event.tool_call_id}"]`);
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
    const toolCount = document.querySelectorAll('.tool-step').length;
    if (Chat.state.traceVerboseMode === 'normal' && toolCount > CONFIG.TRACE.AUTO_COLLAPSE_TOOL_COUNT) {
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
    }
  },

  // =========================================================================
  // Finalize Trace
  // =========================================================================

  finalizeTrace(messageId, trace, finalEvent) {
    this.stopTraceTimer(messageId);
    
    const container = document.querySelector(`.trace-container[data-message-id="${messageId}"]`);
    if (!container) return;

    const toolCount = trace.toolCalls.size;
    const label = container.querySelector('.trace-label');
    if (label && toolCount > 0) {
      label.textContent = `Agent Activity (${toolCount} tool${toolCount === 1 ? '' : 's'})`;
    }
    
    // Update context meter if usage available
    if (finalEvent && finalEvent.usage) {
      this.updateContextMeter(messageId, finalEvent.usage);
    }

    // Auto-collapse in normal mode
    if (Chat.state.traceVerboseMode === 'normal') {
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

    const traceIconSvg = `<svg class="trace-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg>`;
    
    // Build trace container with collapsed state
    const labelText = toolCount > 0 
      ? `Agent Activity (${toolCount} tool${toolCount === 1 ? '' : 's'})` 
      : 'Agent Activity';

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
          <div class="context-meter" style="display: none;">
            <div class="meter-bar"><div class="meter-fill"></div></div>
            <span class="meter-label">Calculating...</span>
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

    for (const event of events) {
      if (event.type === 'thinking_start') {
        thinkingEvents[event.step_id] = event;
      } else if (event.type === 'thinking_end') {
        const startEvent = thinkingEvents[event.step_id];
        if (startEvent && event.thinking_content && event.thinking_content.trim()) {
          this.addHistoricalThinkingStep(timeline, event);
        }
      } else if (event.type === 'tool_start' || event.type === 'tool_use') {
        toolStartEvents[event.tool_call_id] = event;
        // Add the tool step immediately
        this.addHistoricalToolStep(timeline, event, null);
      } else if (event.type === 'tool_end' || event.type === 'tool_result' || event.type === 'tool_output') {
        const startEvent = toolStartEvents[event.tool_call_id];
        // Update the tool step with output
        this.updateHistoricalToolStep(timeline, event, startEvent);
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
    activeABComparison: null,  // { comparisonId, responseAId, responseBId, configAId, configBId, userPromptMid }
    abVotePending: false,      // true when waiting for user vote
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
    selectedProviderB: localStorage.getItem(CONFIG.STORAGE_KEYS.SELECTED_PROVIDER_B) || null,
    selectedModelB: localStorage.getItem(CONFIG.STORAGE_KEYS.SELECTED_MODEL_B) || null,
    agents: [],
    activeAgentName: null,
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
      this.loadAgents(),
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
      UI.renderConfigs(this.state.configs);
    } catch (e) {
      console.error('Failed to load configs:', e);
    }
  },

  async loadAgents() {
    try {
      const data = await API.getAgentsList();
      this.state.agents = data?.agents || [];
      const activeName = data?.active_name || this.state.agents[0]?.name || null;
      this.state.activeAgentName = Utils.normalizeAgentName(activeName);
      UI.renderAgentsList(this.state.agents, this.state.activeAgentName);
      this.updateActiveModelLabel();
    } catch (e) {
      console.error('Failed to load agents list:', e);
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

      // Also update Model B options if provider B is same as primary
      if (!this.state.selectedProviderB || this.state.selectedProviderB === providerType) {
        UI.renderModelBOptions(models, this.state.selectedModelB, providerType);
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

  async handleProviderBChange(providerType) {
    this.state.selectedProviderB = providerType || null;
    
    if (providerType) {
      localStorage.setItem(CONFIG.STORAGE_KEYS.SELECTED_PROVIDER_B, providerType);
      
      // Load models for provider B
      const provider = this.state.providers.find(p => p.type === providerType);
      if (provider) {
        UI.renderModelBOptions(provider.models || [], this.state.selectedModelB, providerType);
      }
    } else {
      localStorage.removeItem(CONFIG.STORAGE_KEYS.SELECTED_PROVIDER_B);
      // Use primary provider's models
      const primaryProvider = this.state.providers.find(p => p.type === this.state.selectedProvider);
      if (primaryProvider) {
        UI.renderModelBOptions(primaryProvider.models || [], this.state.selectedModelB, this.state.selectedProvider);
      }
    }
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

  getSelectedProviderAndModelB() {
    const providerB = this.state.selectedProviderB || this.state.selectedProvider;
    const modelB = UI.elements.modelSelectB?.value || this.state.selectedModelB;
    if (!providerB) {
      return { provider: null, model: null };
    }
    if (providerB === 'openrouter' && modelB === '__custom__') {
      return { provider: providerB, model: this.state.selectedCustomModel || null };
    }
    return {
      provider: providerB,
      model: modelB,
    };
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
          meta: isUser ? null : this.getEntryMetaLabel(),
          feedback: msg.feedback || null,
          trace: msg.trace || null,  // Include trace data
        };
      });

      // Build history for API
      this.state.history = (data.messages || []).map((msg) => [msg.sender, msg.content]);

      UI.renderMessages(this.state.messages);
      
      // Render historical trace data for assistant messages
      for (const msg of this.state.messages) {
        if (msg.sender !== 'User' && msg.trace) {
          UI.renderHistoricalTrace(msg.id, msg.trace);
        }
      }
      
      await this.loadConversations(); // Refresh list to show active state
    } catch (e) {
      console.error('Failed to load conversation:', e);
      this.state.conversationId = null;
      this.state.messages = [];
      this.state.history = [];
      Storage.setActiveConversationId(null);
      UI.renderMessages([]);
      UI.showToast('Conversation not found. Starting a new chat.');
    }
  },

  async newConversation() {
    try {
      await API.newConversation();
      this.state.conversationId = null;
      this.state.messages = [];
      this.state.history = [];
      Storage.setActiveConversationId(null);
      
      UI.renderMessages([]);
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
        Storage.setActiveConversationId(null);
        UI.renderMessages([]);
      }
      
      await this.loadConversations();
    } catch (e) {
      console.error('Failed to delete conversation:', e);
    }
  },

  async sendMessage() {
    const text = UI.getInputValue();
    if (!text || this.state.isStreaming) return;

    const selected = this.getSelectedProviderAndModel();
    if (selected.provider && !selected.model) {
      UI.showToast('Please select a model for the chosen provider.');
      return;
    }

    // Block if A/B vote is pending
    if (this.state.abVotePending) {
      UI.showToast('Please vote on the current comparison first, or disable A/B mode');
      return;
    }

    // Add user message
    const userMsg = {
      id: `${Date.now()}-user`,
      sender: 'User',
      html: Utils.escapeHtml(text),
    };
    this.state.messages.push(userMsg);
    this.state.history.push(['User', text]);
    UI.addMessage(userMsg);

    UI.clearInput();
    UI.setInputDisabled(true, { disableSend: false });
    UI.setStreamingState(true);
    this.state.isStreaming = true;

    // Determine which configs to use
    const configA = UI.getSelectedConfig('A');
    const configB = UI.getSelectedConfig('B') || configA;
    const isAB = UI.isABEnabled();

    if (isAB) {
      await this.sendABMessage(text, configA, configB);
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

  async sendABMessage(userText, configA, configB) {
        const selectedA = this.getSelectedProviderAndModel();
        const selectedB = this.getSelectedProviderAndModelB();
        if (selectedA.provider && !selectedA.model) {
          UI.showToast('Please select a model for Provider A.');
          this.state.isStreaming = false;
          UI.setInputDisabled(false);
          UI.setStreamingState(false);
          return;
        }
        if (selectedB.provider && !selectedB.model) {
          UI.showToast('Please select a model for Provider B.');
          this.state.isStreaming = false;
          UI.setInputDisabled(false);
          UI.setStreamingState(false);
          return;
        }
    // Randomize which config gets A vs B
    const shuffled = Math.random() < 0.5;
    const [actualConfigA, actualConfigB] = shuffled ? [configB, configA] : [configA, configB];

    const msgIdA = `${Date.now()}-ab-a`;
    const msgIdB = `${Date.now()}-ab-b`;

    // Create side-by-side container
    UI.addABComparisonContainer(msgIdA, msgIdB);

    // Track streaming results
    const results = {
      a: { text: '', messageId: null, configId: null, error: null },
      b: { text: '', messageId: null, configId: null, error: null },
    };

    try {
      this.state.abortController = new AbortController();
      // Stream both responses in parallel
      await Promise.all([
        this.streamABResponse(msgIdA, actualConfigA, results.a, selectedA.provider, selectedA.model),
        this.streamABResponse(msgIdB, actualConfigB, results.b, selectedB.provider, selectedB.model),
      ]);

      // Check for errors
      if (results.a.error || results.b.error) {
        const errorMsg = results.a.error || results.b.error;
        UI.showABError(errorMsg);
        this.state.isStreaming = false;
        UI.setInputDisabled(false);
        UI.setStreamingState(false);
        await this.loadConversations();
        return;
      }

      // Get config IDs
      const configAId = this.getConfigId(actualConfigA);
      const configBId = this.getConfigId(actualConfigB);

      // Create A/B comparison record
      const response = await API.createABComparison({
        conversation_id: this.state.conversationId,
        user_prompt_mid: results.a.userPromptMid || results.b.userPromptMid,
        response_a_mid: results.a.messageId,
        response_b_mid: results.b.messageId,
        config_a_id: configAId,
        config_b_id: configBId,
        is_config_a_first: !shuffled,
      });

      if (response?.comparison_id) {
        this.state.activeABComparison = {
          comparisonId: response.comparison_id,
          responseAId: results.a.messageId,
          responseBId: results.b.messageId,
          responseAText: results.a.text,
          responseBText: results.b.text,
          configAId: configAId,
          configBId: configBId,
        };
        this.state.abVotePending = true;

        // Show vote buttons
        UI.showABVoteButtons(response.comparison_id);
      }

    } catch (e) {
      console.error('A/B comparison error:', e);
      UI.showABError(e.message || 'Failed to create comparison');
      this.state.isStreaming = false;
      UI.setInputDisabled(false);
      UI.setStreamingState(false);
      this.state.abortController = null;
      await this.loadConversations();
      return;
    }

    this.state.isStreaming = false;
    UI.setStreamingState(false);
    this.state.abortController = null;
    // Keep input disabled until vote
    await this.loadConversations();
  },

  async streamABResponse(elementId, configName, result, provider = null, model = null) {
    let streamedText = '';
    const showTrace = this.state.traceVerboseMode !== 'minimal';
    const toolCalls = new Map(); // Track tool calls for this response

    try {
      for await (const event of API.streamResponse(
        this.state.history,
        this.state.conversationId,
        configName,
        this.state.abortController?.signal || null,
        provider,
        model
      )) {
        // Handle trace events
        if (event.type === 'tool_start') {
          toolCalls.set(event.tool_call_id, {
            name: event.tool_name,
            args: event.tool_args,
            status: 'running',
            output: null,
            duration: null,
          });
          if (showTrace) {
            UI.renderToolStart(elementId, event);
          }
        } else if (event.type === 'tool_output') {
          const toolData = toolCalls.get(event.tool_call_id);
          if (toolData) {
            toolData.output = event.output;
            toolData.status = 'success';
          }
          if (showTrace) {
            UI.renderToolOutput(elementId, event);
            UI.renderToolEnd(elementId, {
              tool_call_id: event.tool_call_id,
              status: 'success',
            });
          }
        } else if (event.type === 'tool_end') {
          const toolData = toolCalls.get(event.tool_call_id);
          if (toolData) {
            toolData.status = event.status;
            toolData.duration = event.duration_ms;
          }
          if (showTrace) {
            UI.renderToolEnd(elementId, event);
          }
        } else if (event.type === 'chunk') {
          if (event.accumulated) {
            streamedText = event.content || '';
          } else {
            streamedText += event.content || '';
          }
          UI.updateABResponse(elementId, Markdown.render(streamedText), true);
        } else if (event.type === 'step' && event.step_type === 'agent') {
          const content = event.content || '';
          if (content) {
            streamedText = content;
            UI.updateABResponse(elementId, Markdown.render(streamedText), true);
          }
        } else if (event.type === 'final') {
          const finalText = event.response || streamedText;
          
          // Finalize trace display
          if (showTrace) {
            UI.finalizeTrace(elementId, { toolCalls }, event);
          }
          
          UI.updateABResponse(elementId, Markdown.render(finalText), false);

          if (event.conversation_id != null) {
            this.state.conversationId = event.conversation_id;
            Storage.setActiveConversationId(event.conversation_id);
          }

          result.text = finalText;
          result.messageId = event.message_id;
          result.userPromptMid = event.user_message_id;

          // Re-highlight code blocks
          if (typeof hljs !== 'undefined') {
            setTimeout(() => hljs.highlightAll(), 0);
          }
          return;
        } else if (event.type === 'error') {
          result.error = event.message || 'Stream error';
          UI.updateABResponse(
            elementId,
            `<p style="color: var(--error-text);">${Utils.escapeHtml(result.error)}</p>`,
            false
          );
          return;
        }
      }
    } catch (e) {
      console.error('A/B stream error:', e);
      result.error = e.message || 'Streaming failed';
      UI.updateABResponse(
        elementId,
        `<p style="color: var(--error-text);">${Utils.escapeHtml(result.error)}</p>`,
        false
      );
    }
  },

  getConfigId(configName) {
    const config = this.state.configs.find((c) => c.name === configName);
    return config?.id || null;
  },

  async submitABPreference(preference) {
    if (!this.state.activeABComparison) return;

    try {
      await API.submitABPreference(this.state.activeABComparison.comparisonId, preference);

      // Update UI to show result
      UI.markABWinner(preference);
      UI.hideABVoteButtons();

      // Add the winning response to history for context
      const winningText =
        preference === 'b'
          ? this.state.activeABComparison.responseBText
          : this.state.activeABComparison.responseAText;
      this.state.history.push(['archi', winningText]);

      // Clear A/B state
      this.state.activeABComparison = null;
      this.state.abVotePending = false;
      UI.setInputDisabled(false);
      UI.elements.inputField?.focus();
    } catch (e) {
      console.error('Failed to submit preference:', e);
      UI.showToast('Failed to submit preference. Please try again.');
    }
  },

  cancelPendingABComparison() {
    // Called when user disables A/B mode while vote is pending
    if (!this.state.abVotePending) return;

    // Add response A to history as default
    if (this.state.activeABComparison?.responseAText) {
      this.state.history.push(['archi', this.state.activeABComparison.responseAText]);
    }

    // Mark as tie/skipped visually
    UI.markABWinner('tie');
    UI.hideABVoteButtons();

    // Clear state
    this.state.activeABComparison = null;
    this.state.abVotePending = false;
    UI.setInputDisabled(false);
    UI.showToast('A/B comparison skipped');
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
          this.state.activeTrace.toolCalls.set(event.tool_call_id, {
            name: event.tool_name,
            args: event.tool_args,
            status: 'running',
            output: null,
            duration: null,
          });
          this.state.activeTrace.events.push(event);
          if (showTrace) {
            UI.renderToolStart(messageId, event);
          }
        } else if (event.type === 'tool_output') {
          const toolData = this.state.activeTrace.toolCalls.get(event.tool_call_id);
          if (toolData) {
            toolData.output = event.output;
            toolData.status = 'success';
          }
          this.state.activeTrace.events.push(event);
          if (showTrace) {
            UI.renderToolOutput(messageId, event);
            UI.renderToolEnd(messageId, {
              tool_call_id: event.tool_call_id,
              status: 'success',
            });
          }
        } else if (event.type === 'tool_end') {
          const toolData = this.state.activeTrace.toolCalls.get(event.tool_call_id);
          if (toolData) {
            toolData.status = event.status;
            toolData.duration = event.duration_ms;
          }
          this.state.activeTrace.events.push(event);
          if (showTrace) {
            UI.renderToolEnd(messageId, event);
          }
        } else if (event.type === 'thinking_start') {
          this.state.activeTrace.events.push(event);
          if (showTrace) {
            UI.renderThinkingStart(messageId, event);
          }
        } else if (event.type === 'thinking_end') {
          this.state.activeTrace.events.push(event);
          if (showTrace) {
            UI.renderThinkingEnd(messageId, event);
          }
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
            ? '<p class="cancelled-notice"><em>Response timed out</em></p>'
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
};

// =============================================================================
// Initialize on DOM ready
// =============================================================================

document.addEventListener('DOMContentLoaded', () => Chat.init());
