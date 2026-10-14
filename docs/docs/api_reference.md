# API Reference

REST API endpoints for the Archi chat application. All endpoints are prefixed with `/api/`.

> **Note:** For the CLI reference, see [CLI Reference](cli_reference.md). For the configuration YAML schema, see [Configuration Reference](configuration.md).

## How to Read This Page

- Base URL is your running chat service (for example `http://localhost:7861`).
- Most `/api/*` endpoints require an authenticated session.
- Endpoints marked **Admin only** require an admin user.
- Authentication routes (`/login`, `/logout`, `/auth/user`) are not under `/api/`.

---

## Chat

### `POST /api/get_chat_response`

Send a message and receive a complete response.

### `POST /api/get_chat_response_stream`

Send a message and receive a streaming response via NDJSON (`application/x-ndjson`).

Each line is a JSON object with a `type` field. Event types:

| Type | Description |
|------|-------------|
| `meta` | Stream metadata (sent first, includes padding) |
| `text` | Response text delta |
| `tool_start` | Agent is invoking a tool |
| `tool_output` | Tool result |
| `thinking_start` | Reasoning model thinking begins |
| `thinking_end` | Reasoning model thinking ends |
| `final` | Final response with full message and metadata |
| `error` | Error occurred |

### `POST /api/cancel_stream`

Cancel an in-progress streaming response.

### `GET /api/trace/<trace_id>`

Retrieve the full trace of a previous request.

### `POST /api/ab/create`

Create an A/B comparison between two model responses (legacy manual mode).

### `GET /api/ab/pool`

Get the server-side A/B testing pool configuration. The response shape depends on RBAC:

- `ab:view` or `ab:manage`: full read-only experiment configuration
- `ab:participate`: participant-focused payload including the effective per-user sample rate and participant eligibility diagnostics
- otherwise: `enabled: false`

**Response (pool active):**
```json
{
  "success": true,
  "enabled": true,
  "can_view": true,
  "can_manage": false,
  "champion": "default",
  "variants": ["default", "creative", "concise"],
  "comparison_rate": 0.25,
  "default_comparison_rate": 0.25,
  "participant_eligible": true,
  "participant_reason": "eligible"
}
```

Participant payloads can also report `participant_reason: "not_targeted"` when the deployment has an active experiment but the current user's roles or permissions are not included in that experiment's target filters.

### `POST /api/ab/compare`

Stream a pool-based champion-vs-variant A/B comparison. The server randomly pairs the champion against another variant from the pool and streams interleaved NDJSON events tagged with `arm: "a"` or `arm: "b"`. A final `ab_meta` event carries the `comparison_id` and variant mapping.

**Request body:** Same as `/api/get_chat_response_stream`.

### `GET /api/ab/metrics`

Get per-variant aggregate metrics (wins, losses, ties, total comparisons).

**Response:**
```json
{
  "success": true,
  "metrics": [
    {
      "variant_name": "creative",
      "wins": 12,
      "losses": 5,
      "ties": 3,
      "total_comparisons": 20,
      "last_updated": "2025-01-15T10:30:00"
    }
  ]
}
```
## Conversations

Manage saved conversations shown in the chat sidebar. In anonymous mode the caller's `client_id` doubles as the ownership credential and is passed in the JSON body. With SSO enabled the verified session user is the owner.

### `POST /api/rename_conversation`

Rename a conversation. The new title is stored in the conversation's metadata, so it persists across page reloads.

**Request:**
```json
{
  "conversation_id": 42,
  "title": "CERN internship research",
  "client_id": "abc-123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversation_id` | integer | Yes | ID of the conversation to rename |
| `title` | string | Yes | New title. Leading and trailing whitespace is trimmed; must be 1–100 characters after trimming |
| `client_id` | string | Anonymous mode | Ownership credential. Not needed when a signed-in session user is present |

**Response (200):**
```json
{
  "success": true,
  "conversation_id": 42,
  "title": "CERN internship research"
}
```

**Errors:**

| Status | Meaning |
|--------|---------|
| `400` | `conversation_id` missing, `title` empty or longer than 100 characters, or neither a session user nor a `client_id` was supplied |
| `404` | Conversation not found, or not owned by the caller |
| `500` | Unexpected server error |

Error responses have the shape `{"error": "<message>"}`.




---

## Playbooks

Per-user reusable instruction packs (the chat-side analog of an Agent Skills `SKILL.md`). Invoked in chat with `/name`; managed from **Settings → Playbooks** or via this API. In anonymous mode the caller's `client_id` doubles as the owner credential — pass it as a query parameter (GET) or JSON body field (POST/PUT/DELETE). With SSO enabled the verified session identity is the owner and `client_id` is ignored.

### `GET /api/playbooks`

List the caller's playbooks plus public ones (no bodies). Each item carries `id`, `name`, `description`, `visibility`, `is_mine`, and `is_enabled` (whether a public playbook has been added to the caller's active list).

### `GET /api/playbooks/<id>`

Fetch one playbook, including its body — own or public.

### `POST /api/playbooks`

Create a playbook owned by the caller: `{name, description, body, visibility?, client_id?}`. Names are lowercase slugs (`a-z0-9` with single hyphens, max 64 chars). Returns `409` for a duplicate own name, `400` for validation errors.

### `PUT /api/playbooks/<id>`

Update fields of an owned playbook (same body shape; all fields optional).

### `DELETE /api/playbooks/<id>`

Delete an owned playbook.

### `POST /api/playbooks/<id>/enable` · `POST /api/playbooks/<id>/disable`

Add / remove a public playbook to / from the caller's active list (opt-in). Own playbooks are always active.

### `GET /api/playbooks/export`

Download the caller's **own** playbooks as a zip of `<name>/SKILL.md` folders (the Agent Skills layout, portable to claude.ai). Public playbooks owned by others are excluded.

### `POST /api/playbooks/import`

Import playbooks from a multipart upload (field `file`: a zip of `<name>/SKILL.md` folders or a single SKILL.md) or the legacy JSON body `{playbooks: [...], on_conflict?, client_id?}`. `on_conflict` is `skip` (default) or `overwrite`. Imports are always created **private** (a file's public flag never publishes deployment-wide); per-item validation errors are reported without aborting the batch.

---

## Authentication

Authentication routes are served at the application root (not under `/api/`).

### `GET|POST /login`

Authenticate with email and password. GET renders the login page; POST processes credentials.

### `GET /logout`

End the current session.

### `GET /auth/user`

Get the current authenticated user.

---

## User Management

### `GET /api/users/me`

Get or create the current user.

**Response:**
```json
{
  "id": "user_abc123",
  "display_name": "John Doe",
  "email": "john@example.com",
  "auth_provider": "basic",
  "theme": "dark",
  "preferred_model": "gpt-4o",
  "preferred_temperature": 0.7,
  "has_openrouter_key": true,
  "has_openai_key": false,
  "has_anthropic_key": false
}
```

### `PATCH /api/users/me/preferences`

Update user preferences (model, temperature, prompts, theme, and A/B participation override).

**Request:**
```json
{
  "theme": "light",
  "preferred_model": "claude-3-opus",
  "preferred_temperature": 0.5,
  "ab_participation_rate": 0.75
}
```

### `PUT /api/users/me/api-keys/{provider}`

Set a BYOK API key. Provider: `openrouter`, `openai`, `anthropic`.

### `DELETE /api/users/me/api-keys/{provider}`

Delete a BYOK API key.

---

## Provider Keys (BYOK)

### `GET /api/providers/keys`

Get status of all provider API keys.

### `POST /api/providers/keys/set`

Set a session API key (validates before storing).

### `POST /api/providers/keys/clear`

Clear a session API key.

---

## Configuration

### `GET /api/config/static`

Get static (deploy-time) configuration.

**Response:**
```json
{
  "deployment_name": "my-archi",
  "embedding_model": "text-embedding-3-small",
  "available_pipelines": ["QAPipeline", "CMSCompOpsAgent"],
  "available_models": ["gpt-4o", "claude-3-opus"],
  "auth_enabled": true,
  "prompts_path": "/root/archi/data/prompts/"
}
```

### `GET /api/config/dynamic`

Get dynamic (runtime) configuration.

**Response:**
```json
{
  "active_pipeline": "QAPipeline",
  "active_model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 4096,
  "top_p": 0.9,
  "top_k": 50,
  "num_documents_to_retrieve": 10,
  "verbosity": 3
}
```

### `PATCH /api/config/dynamic`

Update dynamic configuration. **Admin only.**

**Request:**
```json
{
  "active_model": "gpt-4o",
  "temperature": 0.8,
  "num_documents_to_retrieve": 5
}
```

### `GET /api/config/effective`

Get effective configuration for the current user (user preferences applied).

### `GET /api/config/audit`

Get configuration change audit log. **Admin only.**

**Query params:** `limit` (default: 100)

---

## Agents

### `GET /api/agents/list`

List all available agent specs.

### `GET /api/agents/spec`

Get a specific agent spec (name, tools, prompt). Pass `name` as a query parameter.

### `GET /api/agents/template`

Get the template for creating a new agent (available tools, defaults).

### `POST /api/agents`

Create or update an agent spec.

**Request:**
```json
{
  "name": "My Agent",
  "tools": ["search_vectorstore_hybrid", "fetch_catalog_document"],
  "prompt": "You are a helpful assistant..."
}
```

### `DELETE /api/agents`

Delete an agent spec. Pass `name` as a query parameter or in the request body.

### `POST /api/agents/active`

Set the active agent for the current session.

**Request:**
```json
{
  "agent_name": "CMS Comp Ops"
}
```

### `GET /api/ab/agents/list`

List the Postgres-backed A/B agent catalog for the A/B admin page. Requires A/B page access.

### `GET /api/ab/agents/template`

Get the A/B admin template payload with structured tool metadata. Requires `ab:manage`.

**Response:**
```json
{
  "name": "New A/B Agent",
  "prompt": "Write your system prompt here.",
  "tools": [
    {"name": "search_vectorstore_hybrid", "description": "Search indexed documents."}
  ],
  "scope": "ab"
}
```

### `POST /api/ab/agents`

Create a new Postgres-backed A/B agent spec from structured fields. Requires `ab:manage`.

**Request:**
```json
{
  "name": "A/B Candidate",
  "tools": ["search_vectorstore_hybrid"],
  "prompt": "You are a helpful A/B experiment agent."
}
```

---

## Prompts

### `GET /api/prompts`

List all available prompts by type.

**Response:**
```json
{
  "condense": ["default", "concise"],
  "chat": ["default", "formal", "technical"],
  "system": ["default", "helpful"]
}
```

### `GET /api/prompts/{type}`

List prompts for a specific type.

### `GET /api/prompts/{type}/{name}`

Get prompt content.

### `POST /api/prompts/reload`

Reload prompt cache from disk. **Admin only.**

---

## Document Selection

Three-tier document selection: conversation override → user default → system default.

### `GET /api/documents/selection`

Get enabled documents. Query param: `conversation_id`.

### `PUT /api/documents/user-defaults`

Set user's default for a document.

**Request:**
```json
{
  "document_id": 42,
  "enabled": false
}
```

### `PUT /api/documents/conversation-override`

Set conversation-specific override.

### `DELETE /api/documents/conversation-override`

Clear conversation override (fall back to user default).

---

## Data Viewer

### `GET /api/data/documents`

List ingested documents with pagination and filtering.

**Query params:** `limit` (default: 100), `offset`, `search`, `source_type`

**Response:**
```json
{
  "documents": [
    {
      "hash": "5e90ca54526f3e11",
      "file_name": "readme.md",
      "source_type": "links",
      "chunk_count": 5,
      "enabled": true,
      "ingested_at": "2025-01-29T10:30:00Z"
    }
  ],
  "total": 42
}
```

### `GET /api/data/documents/<hash>/content`

Get document content and chunks.

### `POST /api/data/documents/<hash>/enable`

Enable a document for retrieval.

### `POST /api/data/documents/<hash>/disable`

Disable a document from retrieval.

### `POST /api/data/bulk-enable`

Enable multiple documents.

**Request:**
```json
{
  "hashes": ["5e90ca54526f3e11", "a1b2c3d4e5f67890"]
}
```

### `POST /api/data/bulk-disable`

Disable multiple documents.

### `GET /api/data/stats`

Get document statistics (total, enabled, disabled, by source type).

---

## Analytics

### `GET /api/analytics/model-usage`

Get model usage statistics. Query params: `start_date`, `end_date`, `service`.

### `GET /api/analytics/ab-comparisons`

Get A/B comparison statistics with win rates. Query params: `model_a`, `model_b`, `start_date`, `end_date`.

---

## Data Manager

These endpoints are served by the Data Manager service (default port: 7871).

### `GET /api/ingestion/status`

Get current ingestion progress.

### `POST /api/reload-schedules`

Trigger schedule reload from database.

### `GET /api/schedules`

Get current schedule status.

---

## Health & Info

### `GET /api/health`

Health check with database connectivity status.

### `GET /api/info`

Get API version and available features.
