const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");
const refreshButton = document.querySelector("#refresh-btn");
const popupForm = document.getElementById("popup-form");
const additionalThoughtsInput = document.getElementById("dislike-additional-thoughts");
const submitButton = document.getElementById("dislike-submit-button");
const closeButton = document.getElementById("dislike-close-button");
const correct_checkbox= document.getElementById("correct_checkbox");
const helpful_checkbox = document.getElementById("helpful_checkbox");
const appropriate_checkbox = document.getElementById("appropriate_checkbox");
const textFeedbackForm = document.getElementById("text-feedback-form");
const textFeedbackTextarea = document.getElementById("text-feedback-textarea");
const textFeedbackSubmitButton = document.getElementById("text-feedback-submit-button");
const textFeedbackCloseButton = document.getElementById("text-feedback-close-button");
const textFeedbackCountLabel = document.getElementById("text-feedback-count");
const configMenuBtn = document.getElementById("config-menu-btn");
const configMenuOverlay = document.getElementById("config-menu-overlay");
const configMenuCloseButton = document.getElementById("config-menu-close");
const configMenuOptions = document.getElementById("config-menu-options");
const configDropdown = document.getElementById("config-dropdown");
const newChatButton = document.getElementById("new-chat-btn");
const conversationsList = document.getElementById("conversations-list");
const toggleSidebarButton = document.getElementById("toggle-sidebar-btn");

popupForm.style.display = "none";
if (textFeedbackForm) {
    textFeedbackForm.style.display = "none";
}

// DEFINITIONS
let DEFAULT_TIMEOUT_SECS = 300
const USE_STREAMING = true;
const STREAM_INCLUDE_AGENT_STEPS = true;
const STREAM_INCLUDE_TOOL_STEPS = true;
const STREAM_ENDPOINT = "/api/get_chat_response_stream";

// Configure marked.js with highlight.js for syntax highlighting
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined') {
                try {
                    if (lang && hljs.getLanguage(lang)) {
                        return hljs.highlight(code, { language: lang }).value;
                    }
                    return hljs.highlightAuto(code).value;
                } catch (e) {
                    console.error('Highlight error:', e);
                    return code;
                }
            }
            return code;
        }
    });
}
let userText = null;
let conversation = []
let num_responses_since_last_rating = 0;
let last_response_is_feedback_request = false;
const CLIENT_ID_STORAGE_KEY = "archi_client_id";
const ACTIVE_CONVERSATION_STORAGE_KEY = "archi_active_conversation_id";
const CLEARED_CONVERSATIONS_KEY = "archi_cleared_conversation_ids";
const MAX_CONVERSATION_ID_KEY = "archi_max_conversation_id";

const generateClientId = () => {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return window.crypto.randomUUID();
    }
    // Fallback UUID v4 generator
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        const v = c === "x" ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
};

const getStoredClientId = () => {
    let existingId = localStorage.getItem(CLIENT_ID_STORAGE_KEY);
    if (!existingId) {
        existingId = generateClientId();
        localStorage.setItem(CLIENT_ID_STORAGE_KEY, existingId);
    }
    return existingId;
};

const parseConversationId = (value) => {
    if (value === null || value === undefined) return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
};

const getStoredConversationId = () => {
    const stored = localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
    return parseConversationId(stored);
};

const updateActiveConversationId = (newId) => {
    const normalized = parseConversationId(newId);
    if (normalized === null) {
        localStorage.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
    } else {
        localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, normalized);
    }
    conversation_id = normalized;
};

const clientId = getStoredClientId();
let conversation_id = getStoredConversationId();

const loadClearedConversations = () => {
    try {
        const raw = localStorage.getItem(CLEARED_CONVERSATIONS_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.warn("Failed to parse cleared conversation ids:", error);
        return [];
    }
};

const persistClearedConversations = () => {
    try {
        localStorage.setItem(CLEARED_CONVERSATIONS_KEY, JSON.stringify(Array.from(clearedConversationIds)));
    } catch (error) {
        console.warn("Failed to persist cleared conversation ids:", error);
    }
};

const loadMaxConversationId = () => {
    const stored = localStorage.getItem(MAX_CONVERSATION_ID_KEY);
    if (stored === null || stored === undefined) {
        return null;
    }
    const parsed = Number(stored);
    return Number.isFinite(parsed) ? parsed : null;
};

const persistMaxConversationId = (value) => {
    if (value === null || value === undefined) {
        localStorage.removeItem(MAX_CONVERSATION_ID_KEY);
        return;
    }
    localStorage.setItem(MAX_CONVERSATION_ID_KEY, value);
};

const clearedConversationIds = new Set(loadClearedConversations());
let maxConversationIdSeen = loadMaxConversationId();
let pendingTextFeedbackMessageId = null;
let pendingTextFeedbackChatElement = null;
const LIKE_ICON = "/static/images/thumbs_up.png";
const LIKE_ICON_FILLED = "/static/images/thumbs_up_filled.png";
const DISLIKE_ICON = "/static/images/thumbs_down.png";
const DISLIKE_ICON_FILLED = "/static/images/thumbs_down_filled.png";
const CONFIG_BUTTON_DEFAULT_LABEL = "Choose your agent";
const configOptionsMeta = new Map();

function updateFeedbackVisualState(chatElement, feedbackState) {
    if (!chatElement) return;
    const buttonImages = chatElement.querySelectorAll(".button-container button img");
    const likeImg = buttonImages[0];
    const dislikeImg = buttonImages[1];
    if (likeImg) {
        likeImg.src = feedbackState === "like" ? LIKE_ICON_FILLED : LIKE_ICON;
    }
    if (dislikeImg) {
        dislikeImg.src = feedbackState === "dislike" ? DISLIKE_ICON_FILLED : DISLIKE_ICON;
    }
    if (feedbackState) {
        chatElement.dataset.feedbackState = feedbackState;
    } else {
        delete chatElement.dataset.feedbackState;
    }
}

const formatCommentCount = (count) => {
    const safe = Number.isFinite(count) && count > 0 ? count : 0;
    return safe === 1 ? "1 comment submitted" : `${safe} comments submitted`;
};

const updateTextFeedbackCountLabel = (count) => {
    if (!textFeedbackCountLabel) return;
    const safe = Number.isFinite(count) && count >= 0 ? count : 0;
    textFeedbackCountLabel.textContent = formatCommentCount(safe);
};

const resetTextFeedbackCountLabel = () => updateTextFeedbackCountLabel(0);

resetTextFeedbackCountLabel();

const updateConfigMenuLabel = (value) => {
    if (!configMenuBtn) return;
    const details = value ? configOptionsMeta.get(value) : null;
    if (!details) {
        configMenuBtn.setAttribute("title", CONFIG_BUTTON_DEFAULT_LABEL);
        return;
    }
    const description = details.description ? ` — trained on ${details.description}` : "";
    configMenuBtn.setAttribute("title", `${details.name}${description}`);
};

const highlightConfigOption = (value) => {
    if (!configMenuOptions) return;
    const buttons = configMenuOptions.querySelectorAll(".config-option");
    buttons.forEach(button => {
        const buttonName = button.dataset.configName;
        if (value && buttonName === value) {
            button.classList.add("selected");
        } else {
            button.classList.remove("selected");
        }
    });
};

const closeConfigMenu = () => {
    if (configMenuOverlay) {
        configMenuOverlay.style.display = "none";
    }
};

const openConfigMenu = () => {
    if (!configMenuOverlay) return;
    if (!configMenuOptions || !configMenuOptions.childElementCount) {
        return;
    }
    configMenuOverlay.style.display = "block";
};

const selectConfigOption = (name) => {
    if (!configDropdown) return;
    const available = Array.from(configDropdown.options || []).some(opt => opt.value === name);
    if (!available) return;
    configDropdown.value = name;
    updateConfigMenuLabel(name);
    highlightConfigOption(name);
    closeConfigMenu();
};

function resetDislikeForm() {
    if (additionalThoughtsInput) {
        additionalThoughtsInput.value = "";
    }
    if (correct_checkbox) {
        correct_checkbox.checked = false;
    }
    if (helpful_checkbox) {
        helpful_checkbox.checked = false;
    }
    if (appropriate_checkbox) {
        appropriate_checkbox.checked = false;
    }
}

async function fetchWithTimeout(resource, options = {}) {
    // extracts `timeout` field from options dict;
    // will default to DEFAULT_TIMEOUT_SECS if no field is present
    const { timeout = DEFAULT_TIMEOUT_SECS * 1000 } = options;
    console.log(`Sending request to ${resource} with timeout: ${timeout} ms`);

    const controller = new AbortController();
    const id = setTimeout(() => {
        console.error("Fetch timed out! Aborting request...");
        controller.abort();
    }, timeout);

    try {
        const response = await fetch(resource, { ...options, signal: controller.signal });
        console.log("API response received");
        
        // Check for 401 Unauthorized response
        if (response.status === 401) {
            console.log("Unauthorized - redirecting to landing page");
            window.location.href = '/';
            throw new Error('Unauthorized');
        }
        
        return response;
    } catch (error) {
        console.error("Fetch error:", error);
        throw error;
    } finally {
        clearTimeout(id);
    }
}

async function populateDropdown() {
    const API_URL = "/api/get_configs"

    try {
        const response = await fetch(API_URL);
        
        // Check for 401 Unauthorized response
        if (response.status === 401) {
            console.log("Unauthorized - redirecting to landing page");
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();

        if (data.error) {
            console.error("API Error fetching configs", data.error);
            return;
        }

        // Preserve the currently selected value, if any
        const previousValue = configDropdown ? configDropdown.value : null;
        if (configDropdown) {
            configDropdown.innerHTML = '';
        }
        if (configMenuOptions) {
            configMenuOptions.innerHTML = '';
        }
        configOptionsMeta.clear();

        const options = Array.isArray(data.options) ? data.options : [];
        if (!options.length) {
            console.warn("No configs returned from /api/get_configs");
        }
        // Populate the dropdown with options from the API response
        options.forEach(optionData => {
            if (!optionData || !optionData.name) return;
            const normalizedDescription = optionData.description || "";
            configOptionsMeta.set(optionData.name, {
                name: optionData.name,
                description: normalizedDescription
            });
            if (configDropdown) {
                const option = document.createElement("option");
                option.textContent = optionData.name;
                option.value = optionData.name;
                configDropdown.appendChild(option);
            }
            if (configMenuOptions) {
                const optionBtn = document.createElement("button");
                optionBtn.type = "button";
                optionBtn.className = "config-option";
                optionBtn.dataset.configName = optionData.name;
                const nameSpan = document.createElement("span");
                nameSpan.className = "config-option-name";
                nameSpan.textContent = optionData.name;
                const descSpan = document.createElement("span");
                descSpan.className = "config-option-desc";
                const descText = normalizedDescription || "custom data";
                descSpan.textContent = `Trained on: ${descText}`;
                optionBtn.appendChild(nameSpan);
                optionBtn.appendChild(descSpan);
                optionBtn.addEventListener("click", () => selectConfigOption(optionData.name));
                configMenuOptions.appendChild(optionBtn);
            }
        });

        let appliedValue = null;
        if (configDropdown) {
            const optionsList = Array.from(configDropdown.options);
            const matchingOption = optionsList.find(opt => opt.value === previousValue);
            if (matchingOption) {
                configDropdown.value = previousValue;
                appliedValue = previousValue;
            } else if (optionsList.length) {
                configDropdown.selectedIndex = 0;
                appliedValue = configDropdown.value;
            }
        }
        updateConfigMenuLabel(appliedValue);
        highlightConfigOption(appliedValue);
    }catch (error) {
        console.error("Failed to fetch dropdown options:", error);
    }

}

async function loadConversations() {
    const API_URL = `/api/list_conversations?limit=100&client_id=${encodeURIComponent(clientId)}`;
    
    try {
        const response = await fetch(API_URL);
        
        // Check for 401 Unauthorized response
        if (response.status === 401) {
            console.log("Unauthorized - redirecting to landing page");
            window.location.href = '/';
            return;
        }
        
        const data = await response.json();
        
        if (data.error) {
            console.error("API Error fetching conversations", data.error);
            conversationsList.innerHTML = '<div class="loading-conversations">Error loading conversations</div>';
            return;
        }
        
        // clear the list
        conversationsList.innerHTML = '';
        
        // fill with conversations
        if (data.conversations.length === 0) {
            if (maxConversationIdSeen !== null) {
                maxConversationIdSeen = null;
                persistMaxConversationId(null);
            }
            if (clearedConversationIds.size > 0) {
                clearedConversationIds.clear();
                persistClearedConversations();
            }
            conversationsList.innerHTML = '<div class="loading-conversations">No conversations yet</div>';
            return;
        }
        const conversationIds = data.conversations.map(conv => conv.conversation_id);
        const currentMaxConversationId = conversationIds.length > 0 ? Math.max(...conversationIds) : null;
        if (
            clearedConversationIds.size > 0 &&
            maxConversationIdSeen !== null &&
            currentMaxConversationId !== null &&
            currentMaxConversationId < maxConversationIdSeen
        ) {
            clearedConversationIds.clear();
            persistClearedConversations();
        }
        if (currentMaxConversationId !== maxConversationIdSeen) {
            maxConversationIdSeen = currentMaxConversationId;
            persistMaxConversationId(maxConversationIdSeen);
        }
        let hasVisibleConversations = false;
        
        data.conversations.forEach(conv => {
            if (clearedConversationIds.has(conv.conversation_id)) {
                return;
            }
            hasVisibleConversations = true;
            const convElement = document.createElement("div");
            convElement.classList.add("conversation-item");
            convElement.dataset.conversationId = conv.conversation_id;
            
            // Mark active conversation meaning user's working conversation_id space
            if (conv.conversation_id === conversation_id) {
                convElement.classList.add("active");
            }
            
            const date = new Date(conv.last_message_at);
            const now = new Date();
            const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
            let dateStr;
            if (diffDays === 0) {
                dateStr = "Today";
            } else if (diffDays === 1) {
                dateStr = "Yesterday";
            } else if (diffDays < 7) {
                dateStr = `${diffDays} days ago`;
            } else {
                dateStr = date.toLocaleDateString();
            }
            
            convElement.innerHTML = `
                <div class="conversation-item-title">${conv.title}</div>
                <div class="conversation-item-date">${dateStr}</div>
            `;
            
            convElement.addEventListener("click", () => loadConversation(conv.conversation_id));
            conversationsList.appendChild(convElement);
        });
        if (!hasVisibleConversations) {
            conversationsList.innerHTML = '<div class="loading-conversations">No conversations yet</div>';
        }
        
    } catch (error) {
        console.error("Failed to load conversations:", error);
        conversationsList.innerHTML = '<div class="loading-conversations">Error loading conversations</div>';
    }
}

async function loadConversation(convId) {
    const API_URL = "/api/load_conversation";
    const clientId = getStoredClientId();
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ conversation_id: convId, client_id: clientId }),
        });
        
        // Check for 401 Unauthorized response
        if (response.status === 401) {
            console.log("Unauthorized - redirecting to landing page");
            window.location.href = '/';
            return;
        }
        if (response.error) {
            throw new Error(response.error);
        }
        const data = await response.json();
        
        if (data.error) {
            console.error("Error loading conversation:", data.error);
            return;
        }
        
        // clear current chat & conversation state & container
        conversation = [];
        num_responses_since_last_rating = 0;        
        chatContainer.innerHTML = '';
        
        // message format
        data.messages.forEach(msg => {
            const className = msg.sender === "User" ? "outgoing" : "incoming";
            const html = className === "outgoing" 
                ? `<div class="chat-content">
                        <div class="chat-details">
                            <img src="/static/images/user.svg" alt="user-img">
                            <p>${msg.content}</p>
                        </div>
                    </div>`
                : `<div class="chat-content">
                        <div class="chat-details">
                            <img src="/static/images/archi-logo.png" alt="chatbot-img">
                            <div>${msg.content}</div>
                        </div>
                        <div class="button-container">
                            <button onclick="likeResponse(this)" class="material-button">
                                <img src="/static/images/thumbs_up.png" alt="Like" width="30" height="30">
                            </button>
                            <button onclick="dislikeResponse(this)" class="material-button">
                                <img src="/static/images/thumbs_down.png" alt="Dislike" width="30" height="30">
                            </button>
                            <button onclick="openWrittenFeedback(this)" class="material-button">
                                <img src="/static/images/feedback.svg" alt="Feedback" width="30" height="30">
                            </button>
                        </div>
                    </div>`;
            
            const chatDiv = createChatElement(html, className);
            if (className === "incoming") {
                if (msg.message_id) {
                    chatDiv.dataset.messageId = msg.message_id;
                } else {
                    delete chatDiv.dataset.messageId;
                }
                const incomingCommentCount = Number(msg.comment_count ?? 0);
                if (Number.isFinite(incomingCommentCount) && incomingCommentCount >= 0) {
                    chatDiv.dataset.commentCount = incomingCommentCount;
                } else {
                    delete chatDiv.dataset.commentCount;
                }
                updateFeedbackVisualState(chatDiv, msg.feedback || null);
            }
            chatContainer.appendChild(chatDiv);
            conversation.push([msg.sender, msg.content]);
        });
        
        // show latest history first
        chatContainer.scrollTo(0, chatContainer.scrollHeight);
        
        updateActiveConversationId(convId);
        localStorage.setItem("all-chats", chatContainer.innerHTML);

        // update sidebar to remark active conversation
        loadConversations();
        return true;
    } catch (error) {
        console.error("Failed to execute loadConversation:", error);
        const storedId = getStoredConversationId();
        if (storedId === convId) {
            updateActiveConversationId(null);
        }
        return false;
    }
}

function startNewConversation(options = {}) {
    const { hideActiveConversation = false } = options;
    // reset conversation state
    const previousConversationId = conversation_id;
    conversation = [];
    updateActiveConversationId(null);
    num_responses_since_last_rating = 0;
    if (hideActiveConversation && previousConversationId !== null) {
        clearedConversationIds.add(previousConversationId);
        persistClearedConversations();
    }
    
    // clear localStorage and show default welcome text
    localStorage.removeItem("all-chats");
    const defaultText = `<div class="default-text">
                            <h1>archi</h1>
                            <p>Start a conversation and explore the power of archi, specially trained on Local development test agent.<br> 
                            Your chat history will be displayed here. <br> <br>
                            Remember, the more context you provide in your question, the better archi will be able to answer your question! <br> <br>
                            By using this website, you agree to the <a href="/terms">terms and conditions</a>.</p>
                        </div>`;
    chatContainer.innerHTML = defaultText;
    localStorage.setItem("all-chats", defaultText);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
    
    // update sidebar to deselect previous active conversation
    loadConversations();
}

const loadDataFromLocalstorage = () => {
    // Load saved chats and theme from local storage and apply/add on the page
    const themeColor = localStorage.getItem("themeColor");

    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";

    const defaultText = `<div class="default-text">
                            <h1>archi</h1>
                            <p>Start a conversation and explore the power of archi, specially trained on Local development test agent.<br> 
                            Your chat history will be displayed here. <br> <br>
                            Remember, the more context you provide in your question, the better archi will be able to answer your question! <br> <br>
                            By using this website, you agree to the <a href="/terms">terms and conditions</a>.</p>
                        </div>`

    chatContainer.innerHTML = localStorage.getItem("all-chats") || defaultText;
    chatContainer.scrollTo(0, chatContainer.scrollHeight); // Scroll to bottom of the chat container
}

const createChatElement = (content, className) => {
    // Create new div and apply chat, specified class and set html content of div
    const chatDiv = document.createElement("div");
    chatDiv.classList.add("chat", className);
    chatDiv.innerHTML = content;
    return chatDiv; // Return the created chat div
}

const ensureStreamStepsContainer = (incomingChatDiv) => {
    if (!incomingChatDiv) return null;
    let container = incomingChatDiv.querySelector("[data-stream-steps]");
    if (!container) {
        container = document.createElement("div");
        container.classList.add("stream-steps");
        container.dataset.streamSteps = "true";
        incomingChatDiv.querySelector(".chat-details")?.appendChild(container);
    }
    return container;
};

const formatStreamStep = (event) => {
    if (!event) return "";
    if (event.step_type === "tool_call") {
        const argsValue = event.tool_args || "";
        const args = typeof argsValue === "string" ? argsValue : JSON.stringify(argsValue);
        const argsText = args ? ` ${args}` : "";
        return `Tool call: ${event.tool_name || "unknown"}${argsText}`;
    }
    if (event.step_type === "tool_result") {
        return `Tool result:\n${event.content || ""}`;
    }
    if (event.step_type === "agent") {
        return event.content || "";
    }
    return event.content || "";
};

const appendStreamStep = (incomingChatDiv, event) => {
    const container = ensureStreamStepsContainer(incomingChatDiv);
    if (!container) return;
    const step = document.createElement("div");
    const stepType = event.step_type || "agent";
    container.classList.add("active");
    step.classList.add("stream-step", `stream-step-${stepType}`);
    step.textContent = formatStreamStep(event);
    container.appendChild(step);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
};

const flushToUi = () => new Promise(requestAnimationFrame);

const normalizeStreamChunk = (chunk) => {
    if (chunk === null || chunk === undefined) return "";
    return typeof chunk === "string" ? chunk : String(chunk);
};

const mergeStreamChunk = (current, next) => {
    const nextText = normalizeStreamChunk(next);
    if (!current) return nextText;
    if (!nextText) return current;
    if (nextText.startsWith(current)) return nextText;
    if (current.startsWith(nextText)) return current;
    return current + nextText;
};

const renderStreamingMarkdown = (markdownText) => {
    if (markdownText === null || markdownText === undefined) {
        return { html: "", isHtml: false };
    }
    if (typeof marked !== 'undefined') {
        try {
            return { html: marked.parse(markdownText), isHtml: true };
        } catch (error) {
            console.error("Error rendering streaming markdown:", error);
        }
    }
    return { html: String(markdownText), isHtml: false };
};

const finalizeIncomingChat = (incomingChatDiv, pElement, responseData) => {
    if (responseData && responseData.response !== undefined) {
        try {
            // Render markdown on finalization (response is raw markdown)
            if (typeof marked !== 'undefined') {
                pElement.innerHTML = marked.parse(responseData.response);
            } else {
                pElement.innerHTML = responseData.response;
            }
        } catch (error) {
            console.error("Error setting innerHTML:", error);
            pElement.textContent = responseData.response || "";
        }
        pElement.setAttribute("id", responseData.archi_msg_id?.toString() || "");
        pElement.classList.add(".default-text");
        conversation.push(["archi", responseData.response]);
        updateActiveConversationId(responseData.conversation_id);
        last_response_is_feedback_request = false;
        if (responseData.archi_msg_id !== undefined && responseData.archi_msg_id !== null) {
            incomingChatDiv.dataset.messageId = responseData.archi_msg_id.toString();
            incomingChatDiv.dataset.commentCount = "0";
        } else {
            delete incomingChatDiv.dataset.messageId;
            delete incomingChatDiv.dataset.commentCount;
        }
        updateFeedbackVisualState(incomingChatDiv, null);
    }

    incomingChatDiv.querySelector(".typing-animation")?.remove();
    incomingChatDiv.querySelector("#loading-text")?.remove();
    incomingChatDiv.querySelector(".chat-details").appendChild(pElement);
    localStorage.setItem("all-chats", chatContainer.innerHTML);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);

    setTimeout(() => loadConversations(), 500);
    setTimeout(showFeedbackRequest, 500);
};

const refreshChat = async () => {
    // if the conversation is empty; this is a no-op
    if (conversation.length == 0) {
        return;
    }

    // remove message to be regenerated from conversation
    conversation.pop();
    chatContainer.removeChild(chatContainer.lastChild);

    // if the last response generated a feedback request, make sure to remove it here as well
    console.log(last_response_is_feedback_request)
    if (last_response_is_feedback_request) {
        console.log("remove again")
        chatContainer.removeChild(chatContainer.lastChild);
    }

    // generate new response
    showTypingAnimation(isRefresh=true);
}

const getChatResponse = async (incomingChatDiv, isRefresh=false) => {
    if (USE_STREAMING && window.ReadableStream) {
        return getChatResponseStream(incomingChatDiv, isRefresh);
    }
    const API_URL = "/api/get_chat_response";
    const pElement = document.createElement("div");
    const configName = configDropdown ? configDropdown.value : null;

     // Define the properties and data for the API request
     const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            last_message: conversation.slice(-1),
            conversation_id: conversation_id,
            config_name: configName,
            is_refresh: isRefresh,
            client_sent_msg_ts: Date.now(),
            client_timeout: DEFAULT_TIMEOUT_SECS * 1000,
            client_id: clientId
        }),
        timeout: DEFAULT_TIMEOUT_SECS * 1000
    }

    // send POST request to Flask API, get response and set the response as paragraph element text, console logging nice for debugging
    let responseData = null;
    try {
        const response = await (await fetchWithTimeout(API_URL, requestOptions)).json();
        console.log("Full API Response:", response);

        // Check if response contains an error message
        if (response.error) {
            console.error("API Returned an Error:", response.error);
            throw new Error(response.error);
        }
        responseData = response;
    } catch (error) {
        console.error("Fetch Error:", error);
        pElement.classList.add("error");
        const errorText = (error && error.message ? error.message : String(error || "")).trim();
        const truncated = errorText ? errorText.slice(0, 160) : "";
        const suffix = truncated ? ` (${truncated}${errorText.length > truncated.length ? "..." : ""})` : "";
        pElement.textContent = `Oops! Something went wrong while retrieving the response. Please try again, and contact support if the issue persists.${suffix}`;
    }

    finalizeIncomingChat(incomingChatDiv, pElement, responseData);
}

const getChatResponseStream = async (incomingChatDiv, isRefresh=false) => {
    const pElement = document.createElement("div");
    const configName = configDropdown ? configDropdown.value : null;

    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            last_message: conversation.slice(-1),
            conversation_id: conversation_id,
            config_name: configName,
            is_refresh: isRefresh,
            client_sent_msg_ts: Date.now(),
            client_timeout: DEFAULT_TIMEOUT_SECS * 1000,
            client_id: clientId,
            include_agent_steps: STREAM_INCLUDE_AGENT_STEPS,
            include_tool_steps: STREAM_INCLUDE_TOOL_STEPS,
        }),
        timeout: DEFAULT_TIMEOUT_SECS * 1000,
    };

    let finalized = false;
    let streamedText = "";
    let streamingParagraph = null;
    let pendingStreamRender = null;

    const ensureStreamingParagraph = () => {
        if (!streamingParagraph) {
            streamingParagraph = document.createElement("p");
            streamingParagraph.classList.add("streaming-response");
            pElement.appendChild(streamingParagraph);
        }
        if (!pElement.isConnected) {
            incomingChatDiv.querySelector("#loading-text")?.remove();
            incomingChatDiv.querySelector(".typing-animation")?.remove();
            incomingChatDiv.querySelector(".chat-details")?.appendChild(pElement);
        }
    };

    const scheduleStreamRender = () => {
        if (pendingStreamRender) return;
        pendingStreamRender = setTimeout(() => {
            pendingStreamRender = null;
            ensureStreamingParagraph();
            const rendered = renderStreamingMarkdown(streamedText);
            if (rendered.isHtml) {
                streamingParagraph.innerHTML = rendered.html;
            } else {
                streamingParagraph.textContent = rendered.html;
            }
            chatContainer.scrollTo(0, chatContainer.scrollHeight);
        }, 60);
    };

    const applyStreamChunk = (chunk) => {
        const nextText = mergeStreamChunk(streamedText, chunk);
        if (nextText === streamedText) return false;
        streamedText = nextText;
        scheduleStreamRender();
        return true;
    };

    try {
        const response = await fetchWithTimeout(STREAM_ENDPOINT, requestOptions);
        if (!response.ok) {
            throw new Error(`Streaming failed with status ${response.status}`);
        }
        if (!response.body) {
            throw new Error("Streaming response body not available");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                let event;
                try {
                    event = JSON.parse(trimmed);
                } catch (error) {
                    console.error("Failed to parse stream event:", error);
                    continue;
                }

                if (event.type === "step") {
                    if (event.step_type === "agent") {
                        if (applyStreamChunk(event.content)) {
                            await flushToUi();
                        }
                    } else {
                        appendStreamStep(incomingChatDiv, event);
                        await flushToUi();
                    }
                } else if (event.type === "chunk") {
                    if (applyStreamChunk(event.content)) {
                        await flushToUi();
                    }
                } else if (event.type === "meta") {
                    continue;
                } else if (event.type === "final") {
                    finalizeIncomingChat(incomingChatDiv, pElement, event);
                    finalized = true;
                    reader.cancel();
                    break;
                } else if (event.type === "error") {
                    const message = event.message || "Oops! Something went wrong while retrieving the response. Please try again.";
                    pElement.classList.add("error");
                    pElement.textContent = message;
                    finalizeIncomingChat(incomingChatDiv, pElement, null);
                    finalized = true;
                    reader.cancel();
                    break;
                }
            }
            if (finalized) break;
        }
    } catch (error) {
        console.error("Stream Error:", error);
        pElement.classList.add("error");
        const errorText = (error && error.message ? error.message : String(error || "")).trim();
        const truncated = errorText ? errorText.slice(0, 160) : "";
        const suffix = truncated ? ` (${truncated}${errorText.length > truncated.length ? "..." : ""})` : "";
        pElement.textContent = `Oops! Something went wrong while retrieving the response. Please try again.${suffix}`;
        finalizeIncomingChat(incomingChatDiv, pElement, null);
    }
};

const copyCode = (copyCodeBtn) => {
    // Copy the text content of the response to the clipboard
    const reponseTextElement = copyCodeBtn.parentElement.parentElement.querySelector(".code-box-body");
    navigator.clipboard.writeText(reponseTextElement.innerText);
}

const copyResponse = (copyBtn) => {
    // Copy the text content of the response to the clipboard
    const reponseTextElement = copyBtn.parentElement.previousElementSibling.querySelector("p");
    navigator.clipboard.writeText(reponseTextElement.textContent);
}

const likeResponse = (likeBtn) => {
    num_responses_since_last_rating = 0;

    const chatElement = likeBtn.closest(".chat");
    const messageId = chatElement?.dataset?.messageId;
    if (!messageId) {
        console.warn("Cannot submit like feedback without a message id.");
        return;
    }
    updateFeedbackVisualState(chatElement, "like");

    const API_URL = "/api/like";

     // Send an API request with the chat content and discussion ID
     try {
        fetch(API_URL, {
            method: "POST", // You may need to adjust the HTTP method
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            body: JSON.stringify({
                message_id: Number(messageId),
            }),
        })
     } catch (error) {
        console.log("liked error message")
     }
}

const dislikeResponse = (dislikeBtn) => {
    num_responses_since_last_rating = 0;

    const chatElement = dislikeBtn.closest(".chat");
    const messageId = chatElement?.dataset?.messageId;
    if (!messageId) {
        console.warn("Cannot submit dislike feedback without a message id.");
        return;
    }
    updateFeedbackVisualState(chatElement, "dislike");

    const API_URL = "/api/dislike";

    // Show pop-up form
    resetDislikeForm();
    popupForm.style.display = "block";

    // Function which handles sending the information in the pop up form to the backend API
    function handleSubmitToAPI() {
        const additionalThoughts = additionalThoughtsInput.value;

        try {
            fetch(API_URL, {
                method: "POST", // You may need to adjust the HTTP method
                headers: {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                body: JSON.stringify({ 
                    message_id: Number(messageId),
                    feedback_msg: additionalThoughts,
                    incorrect: correct_checkbox.checked,
                    unhelpful: helpful_checkbox.checked,
                    inappropriate: appropriate_checkbox.checked,
                }),
            });
        } catch (error) {
            console.log("disliked error message")
        }

        //hide pop up form
        resetDislikeForm();
        popupForm.style.display = "none";
    }

    //wait for user to submit response or close the additional feedback window
    submitButton.addEventListener("click", handleSubmitToAPI, {once: true});
    closeButton.addEventListener("click", handleSubmitToAPI, {once: true});
}

const openWrittenFeedback = (feedbackBtn) => {
    if (!textFeedbackForm || !textFeedbackTextarea) return;
    const chatElement = feedbackBtn.closest(".chat");
    const messageId = chatElement?.dataset?.messageId;
    if (!messageId) {
        console.warn("Cannot submit written feedback without a message id.");
        return;
    }

    pendingTextFeedbackMessageId = Number(messageId);
    pendingTextFeedbackChatElement = chatElement;
    textFeedbackTextarea.value = "";
    const commentCount = Number(chatElement.dataset.commentCount || 0);
    updateTextFeedbackCountLabel(commentCount);
    textFeedbackForm.style.display = "block";
    textFeedbackTextarea.focus();
};

const closeTextFeedbackForm = () => {
    if (!textFeedbackForm) return;
    textFeedbackForm.style.display = "none";
    if (textFeedbackTextarea) {
        textFeedbackTextarea.value = "";
    }
    resetTextFeedbackCountLabel();
    pendingTextFeedbackMessageId = null;
    pendingTextFeedbackChatElement = null;
};

const submitTextFeedback = async () => {
    if (pendingTextFeedbackMessageId === null || !pendingTextFeedbackChatElement || !textFeedbackTextarea) {
        closeTextFeedbackForm();
        return;
    }

    const trimmed = textFeedbackTextarea.value.trim();
    if (!trimmed) {
        closeTextFeedbackForm();
        return;
    }

    try {
        const response = await fetch("/api/text_feedback", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            body: JSON.stringify({
                message_id: pendingTextFeedbackMessageId,
                feedback_msg: trimmed,
            }),
        });
        if (!response.ok) {
            console.error("Failed to submit written feedback:", response.statusText);
        } else {
            const currentCount = Number(pendingTextFeedbackChatElement.dataset.commentCount || 0) + 1;
            pendingTextFeedbackChatElement.dataset.commentCount = currentCount;
            updateTextFeedbackCountLabel(currentCount);
        }
    } catch (error) {
        console.error("Failed to submit written feedback:", error);
    } finally {
        closeTextFeedbackForm();
    }
};

if (textFeedbackSubmitButton) {
    textFeedbackSubmitButton.addEventListener("click", submitTextFeedback);
}
if (textFeedbackCloseButton) {
    textFeedbackCloseButton.addEventListener("click", closeTextFeedbackForm);
}
if (textFeedbackForm) {
    textFeedbackForm.addEventListener("click", (event) => {
        if (event.target === textFeedbackForm) {
            closeTextFeedbackForm();
        }
    });
}

if (configMenuBtn) {
    configMenuBtn.addEventListener("click", openConfigMenu);
}
if (configMenuCloseButton) {
    configMenuCloseButton.addEventListener("click", closeConfigMenu);
}
if (configMenuOverlay) {
    configMenuOverlay.addEventListener("click", (event) => {
        if (event.target === configMenuOverlay) {
            closeConfigMenu();
        }
    });
}

const closeFeedback = (closeBtn) => {
    //hide pop up form
    popupForm.style.display = "none";
}

const showTypingAnimation = (isRefresh = false) => {
    // Create the typing animation and append it to the chat container
    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="/static/images/archi-logo.png" alt="chatbot-img">
                        <p id="loading-text"><em class="loading-text">Processing request...</em></p>
                        <div class="stream-steps" data-stream-steps></div>
                    </div>
                    <div class="button-container">
                        <button onclick="likeResponse(this)" class="material-button">
                            <img src="/static/images/thumbs_up.png" alt="Like" width="30" height="30">
                        </button>
                        <button onclick="dislikeResponse(this)" class="material-button">
                            <img src="/static/images/thumbs_down.png" alt="Dislike" width="30" height="30">
                        </button>
                        <button onclick="openWrittenFeedback(this)" class="material-button">
                            <img src="/static/images/feedback.svg" alt="Feedback" width="30" height="30">
                        </button>
                    </div>
                </div>`;

    // Create an incoming chat div and append it
    const incomingChatDiv = createChatElement(html, "incoming");
    chatContainer.appendChild(incomingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);

    // Get reference to the text element inside the typing animation
    const loadingTextElement = document.getElementById("loading-text");

    const statusMessages = [
        "<em class=\"loading-text\">Searching through relevant documents...</em>",
        "<em class=\"loading-text\">Almost there... fetching more details.</em>",
        "<em class=\"loading-text\">Still processing... please wait a moment, open-source magic happening.</em>"
    ];
    
    let statusIndex = 0;
    
    const statusUpdateInterval = setInterval(() => {
        if (statusIndex < statusMessages.length) {
            loadingTextElement.innerHTML = `${statusMessages[statusIndex]}`;
            statusIndex++;
        }
    }, 10000); // Change message every 10 seconds

    // Call API to get response
    getChatResponse(incomingChatDiv, isRefresh)
        .then(() => {
            clearInterval(statusUpdateInterval); // Stop updates once response arrives
        })
        .catch(() => {
            clearInterval(statusUpdateInterval); // Stop updates on error
        });
};

const showFeedbackRequest = () => {
    // Display a message from archi to ask the user to give feedback

    num_responses_since_last_rating = num_responses_since_last_rating + 1;

    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="/static/images/archi-logo.png" alt="chatbot-img">
                        <div class=".default-text">
                            <p>I've noticed you haven't rated any of my responses in a while. Rating responses is crucial because it not only helps me improve, but it also ensures that this project remains open source and freely accessible for everyone. Your input is highly valuable in supporting the archi mission! </p>
                        </div>
                    </div>
                </div>`;
    
    // Create an incoming chat div with feedback request and append it to chat container
    if (num_responses_since_last_rating > 2) {
        const incomingChatDiv = createChatElement(html, "incoming");
        chatContainer.appendChild(incomingChatDiv);
        chatContainer.scrollTo(0, chatContainer.scrollHeight);
        num_responses_since_last_rating = 0;
        last_response_is_feedback_request = true;
    }
}

const handleOutgoingChat = () => {
    userText = chatInput.value.trim(); // Get chatInput value and remove extra spaces
    if(!userText) return; // If chatInput is empty return from here
    conversation.push(["User", userText])

    // Clear the input field and reset its height
    chatInput.value = "";
    chatInput.style.height = `${initialInputHeight}px`;

    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="/static/images/user.svg" alt="user-img">
                        <p>${userText}</p>
                    </div>
                </div>`;

    // Create an outgoing chat div with user's message and append it to chat container
    const outgoingChatDiv = createChatElement(html, "outgoing");
    chatContainer.querySelector(".default-text")?.remove();
    chatContainer.appendChild(outgoingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
    setTimeout(showTypingAnimation, 500);
}

deleteButton.addEventListener("click", () => {
    if(!confirm("Clear this chat window? archi will no longer be able to reference it for context.")) {
        return;
    }
    startNewConversation({ hideActiveConversation: true });
});

refreshButton.addEventListener("click", () => {
    refreshChat();
});

themeButton.addEventListener("click", () => {
    // Toggle body's class for the theme mode and save the updated theme to the local storage 
    document.body.classList.toggle("light-mode");
    localStorage.setItem("themeColor", themeButton.innerText);
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
});

const initialInputHeight = chatInput.scrollHeight;

chatInput.addEventListener("input", () => {   
    // Adjust the height of the input field dynamically based on its content
    chatInput.style.height =  `${initialInputHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    // If the Enter key is pressed without Shift and the window width is larger 
    // than 800 pixels, handle the outgoing chat
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleOutgoingChat();
    }
});

loadDataFromLocalstorage();
populateDropdown();
loadConversations();
loadUserInfo();
const initializeChatState = async () => {
    await loadConversations();
    const storedConversationId = getStoredConversationId();
    if (storedConversationId !== null) {
        const loaded = await loadConversation(storedConversationId);
        if (!loaded) {
            startNewConversation();
        }
    }
};
initializeChatState();

// Event listeners for new conversation
newChatButton.addEventListener("click", () => startNewConversation());

toggleSidebarButton.addEventListener("click", () => {
    const sidebar = document.querySelector(".sidebar");
    const icon = toggleSidebarButton.querySelector(".material-symbols-rounded");
    
    // On mobile (screen width <= 800px), toggle 'hidden' class
    // On desktop, toggle 'collapsed' class
    if (window.innerWidth <= 800) {
        sidebar.classList.toggle("hidden");
        icon.textContent = sidebar.classList.contains("hidden") ? "menu" : "close";
    } else {
        sidebar.classList.toggle("collapsed");
        icon.textContent = sidebar.classList.contains("collapsed") ? "menu" : "chevron_left";
    }
});

sendButton.addEventListener("click", handleOutgoingChat);

// User authentication functions
async function loadUserInfo() {
    try {
        const response = await fetch('/auth/user');
        const data = await response.json();
        
        const userSection = document.getElementById('user-section');
        const loginSection = document.getElementById('login-section');
        
        // Hide both sections if auth is not enabled
        if (!data.auth_enabled) {
            userSection.style.display = 'none';
            loginSection.style.display = 'none';
            return;
        }
        
        if (data.logged_in) {
            // Show user info
            document.getElementById('user-name').textContent = data.name || 'User';
            document.getElementById('user-email').textContent = data.email || '';
            userSection.style.display = 'flex';
            loginSection.style.display = 'none';
        } else {
            // Show login button
            userSection.style.display = 'none';
            loginSection.style.display = 'flex';
        }
    } catch (error) {
        console.error('Error loading user info:', error);
        // Default to hiding both sections on error
        document.getElementById('user-section').style.display = 'none';
        document.getElementById('login-section').style.display = 'none';
    }
}

// Logout handler
const logoutButton = document.getElementById('logout-btn');
if (logoutButton) {
    logoutButton.addEventListener('click', () => {
        window.location.href = '/logout';
    });
}
