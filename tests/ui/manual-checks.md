# MCP Manual Test Checklist

This file contains test procedures that can be executed using Chrome DevTools MCP tools.
Each section has a procedure and expected outcomes.

---

## 1. Page Load Verification

**Procedure:**
```
1. mcp_io_github_chr_navigate_page(url="http://localhost:7861/chat")
2. mcp_io_github_chr_take_snapshot()
```

**Expected Elements:**
- [ ] Sidebar with "archi" brand
- [ ] Header with "Chat" title
- [ ] Header tabs: "Chat" (active), "Data"
- [ ] Message input textbox
- [ ] Settings button
- [ ] Send button
- [ ] Entry meta label showing agent and model

---

## 2. Entry Meta Label Validation

**Procedure:**
```
1. mcp_io_github_chr_take_snapshot()
2. Look for element with "Agent:" and "Model:" text
```

**Expected:**
- [ ] Format: "Agent: {config_name} · Model: {provider} {model_name}"
- [ ] Shows actual config name (e.g., "cms_simple"), NOT "Default agent"
- [ ] Shows provider and model info from pipeline

---

## 3. Settings → Models

**Procedure:**
```
1. mcp_io_github_chr_click(uid for "Settings" button)
2. mcp_io_github_chr_take_snapshot()
```

**Expected:**
- [ ] Settings modal appears
- [ ] Models tab is visible
- [ ] Provider dropdown is visible
- [ ] Model dropdown is visible (enabled after provider selection)

---

## 4. Header Tab Styling

**Procedure:**
```
1. mcp_io_github_chr_take_screenshot()
```

**Visual Check:**
- [ ] "Chat" tab appears active (distinct background/color)
- [ ] "Data" tab appears inactive (different style)
- [ ] Tabs are horizontally aligned
- [ ] Proper spacing between tabs

---

## 5. Data Tab Navigation (With Conversation)

**Setup:**
```
1. Send a message first to create conversation
2. mcp_io_github_chr_wait_for(text="Send message")
```

**Procedure:**
```
1. mcp_io_github_chr_click(uid for "Data" button)
2. mcp_io_github_chr_take_snapshot()
```

**Expected:**
- [ ] Page navigates to /data
- [ ] Conversation ID is passed in URL

---

## 6. Data Tab Navigation (Without Conversation)

**Procedure:**
```
1. Navigate to fresh page: mcp_io_github_chr_navigate_page(url="http://localhost:7861/chat")
2. mcp_io_github_chr_click(uid for "Data" button)
3. Check for alert dialog or error message
```

**Expected:**
- [ ] Shows alert: "Please select or start a conversation first to manage its data."
- [ ] Does NOT crash or show error page

---

## 7. Message Streaming

**Procedure:**
```
1. mcp_io_github_chr_fill(uid for input, value="Hello, testing")
2. mcp_io_github_chr_click(uid for Send button)
3. mcp_io_github_chr_take_snapshot() immediately
```

**During Streaming Expected:**
- [ ] User message appears
- [ ] Assistant message bubble appears
- [ ] Stop button replaces Send button
- [ ] Input is disabled

**After Streaming:**
```
1. mcp_io_github_chr_wait_for(text="Send message", timeout=60000)
2. mcp_io_github_chr_take_snapshot()
```

- [ ] Send button is back
- [ ] Input is enabled
- [ ] Assistant response is complete
- [ ] Message meta appears under assistant message

---

## 8. Message Meta Display

**After sending a message, check snapshot:**

**Expected:**
- [ ] User message does NOT have meta label
- [ ] Assistant message HAS meta label
- [ ] Meta format: "Agent: {name} · Model: {info}"
- [ ] Meta is styled subtly (smaller, muted color)

---

## 9. A/B Testing Mode

**Enable A/B:**
```
1. mcp_io_github_chr_click(uid for Settings button)
2. mcp_io_github_chr_click(uid for Advanced nav item)
3. Handle warning modal if shown
4. Enable A/B checkbox
5. Close settings
```

**Send Message in A/B Mode:**
```
1. mcp_io_github_chr_fill(uid for input, value="Test A/B")
2. mcp_io_github_chr_click(uid for Send button)
3. mcp_io_github_chr_wait_for(text="Model A")
```

**Expected:**
- [ ] Two response columns appear
- [ ] Columns labeled "Model A" and "Model B"
- [ ] Both columns stream independently
- [ ] Vote buttons appear after completion
- [ ] Message meta is NOT shown in comparison columns

---

## 10. Settings Modal Navigation

**Procedure:**
```
1. mcp_io_github_chr_click(uid for Settings button)
2. mcp_io_github_chr_take_snapshot()
```

**Expected:**
- [ ] Modal opens
- [ ] "Models" section is active by default
- [ ] Provider dropdown visible
- [ ] Model dropdown visible

**Navigate to API Keys:**
```
1. mcp_io_github_chr_click(uid for "API Keys" nav item)
2. mcp_io_github_chr_take_snapshot()
```

- [ ] API Keys section shows
- [ ] Provider key status displayed

**Navigate to Advanced:**
```
1. mcp_io_github_chr_click(uid for "Advanced" nav item)
2. mcp_io_github_chr_take_snapshot()
```

- [ ] A/B Testing toggle visible
- [ ] Agent Transparency options visible

---

## 11. Provider Selection

**In Settings > Models:**
```
1. mcp_io_github_chr_fill(uid for provider-select, value="openrouter")
2. mcp_io_github_chr_take_snapshot()
```

**Expected:**
- [ ] Model dropdown populates with OpenRouter models
- [ ] Custom model option appears
- [ ] If "__custom__" selected, custom input appears

**Verify Entry Meta Updates:**
```
1. Close settings
2. mcp_io_github_chr_take_snapshot()
```

- [ ] Entry meta now shows "OpenRouter · {model}"

---

## 12. Console Health Check

**Procedure:**
```
1. mcp_io_github_chr_list_console_messages()
```

**Expected:**
- [ ] No "error" type messages
- [ ] No "warn" type messages related to app code
- [ ] No uncaught exceptions

**Network Check:**
```
1. mcp_io_github_chr_list_network_requests()
```

- [ ] No 4xx responses (except favicon)
- [ ] No 5xx responses

---

## 13. Sidebar Collapse

**Procedure:**
```
1. mcp_io_github_chr_click(uid for Toggle sidebar button)
2. mcp_io_github_chr_take_snapshot()
```

**Expected:**
- [ ] Sidebar collapses or hides
- [ ] Main content area expands
- [ ] Toggle button updates state

---

## 14. Conversation Management

**Create New:**
```
1. mcp_io_github_chr_click(uid for New chat button)
2. mcp_io_github_chr_take_snapshot()
```

- [ ] Messages area clears
- [ ] Empty state shows

**After Sending Message:**
- [ ] Conversation appears in sidebar
- [ ] Shows message preview

---

## 15. Keyboard Navigation

**Procedure:**
```
1. mcp_io_github_chr_press_key(key="Tab")
2. mcp_io_github_chr_take_snapshot()
```

**Repeat Tab and check:**
- [ ] Focus moves through: input → settings → send
- [ ] Focus is visible on each element
- [ ] No elements are skipped

---

## 16. Visual Polish Checklist

**Take full screenshot:**
```
1. mcp_io_github_chr_take_screenshot()
```

**Manual Visual Review:**
- [ ] Consistent spacing (8px grid)
- [ ] Consistent typography
- [ ] Colors match design system
- [ ] No visual glitches or overlaps
- [ ] Proper alignment of elements
- [ ] Readable text contrast
