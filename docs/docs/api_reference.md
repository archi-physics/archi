# API Reference

## CLI

The Archi CLI provides commands to create, manage, and delete Archi deployments and services.

---

### Commands

#### 1. `create`

Create a new Archi deployment.

**Usage:**
```sh
archi create --name <deployment_name> --config <config.yaml> --env-file <secrets.env> [OPTIONS]
```

**Options:**

- `--name, -n` (str, required): Name of the deployment.
- `--config, -c` (str): Path to a YAML configuration file (repeat the flag to supply multiple files).
- `--config-dir, -cd` (str): Directory containing configuration files.
- `--env-file, -e` (str, required): Path to the secrets `.env` file.
- `--services, -s` (comma-separated, required): List of services to enable (e.g., `chatbot,uploader`).
- `--sources, -src` (comma-separated): Additional data sources to enable (e.g., `git,jira`). The `links` source is always available.
- `--podman, -p`: Use Podman instead of Docker.
- `--gpu-ids`: GPU configuration (`all` or comma-separated IDs).
- `--tag, -t` (str): Image tag for built containers (default: `2000`).
- `--hostmode`: Use host network mode.
- `--verbosity, -v` (int): Logging verbosity (0-4, default: 3).
- `--force, -f`: Overwrite existing deployment if it exists.
- `--dry, --dry-run`: Validate and show what would be created, but do not deploy.

---

#### 2. `delete`

Delete an existing Archi deployment.

**Usage:**
```sh
archi delete --name <deployment_name> [OPTIONS]
```

**Options:**

- `--name, -n` (str): Name of the deployment to delete.
- `--rmi`: Remove container images.
- `--rmv`: Remove volumes.
- `--keep-files`: Keep deployment files (do not remove directory).
- `--list`: List all available deployments.

---

#### 3. `restart`

Restart a specific service in an existing deployment without restarting the entire stack.

**Usage:**
```sh
archi restart --name <deployment_name> --service <service_name> [OPTIONS]
```

**Options:**

- `--name, -n` (str, required): Name of the existing deployment.
- `--service, -s` (str): Service to restart (default: `chatbot`).
- `--config, -c` (str): Path to updated YAML configuration file(s) (can be specified multiple times).
- `--config-dir, -cd` (str): Path to directory containing configuration files.
- `--env-file, -e` (str): Path to `.env` file with secrets.
- `--no-build`: Restart without rebuilding the container image.
- `--with-deps`: Also restart dependent services (by default, only the specified service is restarted).
- `--podman, -p`: Use Podman instead of Docker.
- `--verbosity, -v` (int): Logging verbosity level (0-4, default: 3).

**Notes:**

- **Configuration changes**: Restarting with `--no-build` will reflect changes to configuration files. If you've modified code, you must rebuild the image (omit the `--no-build` flag).
- **Updating configuration**: If you provide `--config` or `--config-dir`, the command will update the deployment's configuration before restarting the service.
- **Finding services**: Use `archi list-deployments` to see existing deployments. If you specify an invalid service name, the restart command will display the available services for that deployment.

**Examples:**

Quick config update without rebuilding:
```sh
archi restart -n mybot --service chatbot --no-build
```

Test new agent code (requires rebuild):
```sh
archi restart -n mybot --service chatbot -c updated_config.yaml
```

Restart with updated secrets:
```sh
archi restart -n mybot --service chatbot -e new_secrets.env --no-build
```

Restart data_manager to re-scrape sources:
```sh
archi restart -n mybot --service data_manager
```

---

#### 4. `list-services`

List all available Archi services and data sources.

**Usage:**
```sh
archi list-services
```

---

#### 5. `list-deployments`

List all existing Archi deployments.

**Usage:**
```sh
archi list-deployments
```

---

#### 6. `evaluate`

Launch the benchmarking runtime to evaluate one or more configurations against a set of questions/answers.

**Usage:**
```sh
archi evaluate --name <run_name> --env-file <secrets.env> --config <file.yaml> [OPTIONS]
```
Use `--config-dir` if you want to point to a directory of configs instead.

**Options:**

- Supports the same flags as `create` (`--sources`, `--podman`, `--gpu-ids`, `--tag`, `--hostmode`, `--verbosity`, `--force`).
- Reads configuration from one or more YAML files that should define the `services.benchmarking` section.

---

### Examples

**Create a deployment:**
```sh
archi create --name mybot --config my.yaml --env-file secrets.env --services chatbot,uploader
```

**Delete a deployment and remove images/volumes:**
```sh
archi delete --name mybot --rmi --rmv
```

**Restart a service without rebuilding:**
```sh
archi restart --name mybot --service chatbot --no-build
```

**List all deployments:**
```sh
archi list-deployments
```

**List all services:**
```sh
archi list-services
```

---

## Configuration YAML API Reference

The Archi configuration YAML file defines the deployment, services, data sources, pipelines, models, and interface settings for your Archi instance.

---

### Top-Level Fields

#### `name`

- **Type:** string
- **Description:** Name of the deployment.

#### `global`

- **DATA_PATH:** path for persisted data (defaults to `/root/data/`).
- **ACCOUNTS_PATH:** path for uploader/grader account data.
- **ACCEPTED_FILES:** list of extensions allowed for manual uploads.
- **LOGGING.input_output_filename:** log file that stores pipeline inputs/outputs.
- **verbosity:** default logging level for services (0-4).

---

### `services`

Holds configuration for every containerised service. Common keys include:

- **port / external_port:** internal versus host port mapping for web apps.
- **host / hostname:** network binding and public hostname for frontends.
- **volume/paths:** template or static asset paths expected by the service.

Key services:

- **chat_app:** Chat interface options (`trained_on`, ports, UI toggles).
- **uploader_app:** Document uploader settings (`verify_urls`, ports).
- **grader_app:** Grader-specific knobs (`num_problems`, rubric paths).
- **grafana:** Port configuration for the monitoring dashboard.
- **postgres:** Database credentials (`user`, `database`, `port`, `host`). Also used for vector storage via pgvector.
- **piazza**, **mattermost**, **redmine_mailbox**, **benchmarking**, ...: Service-specific options (see user guide sections above).

---

### `data_manager`

Controls ingestion sources and vector store behaviour.

- **sources.links.input_lists:** `.list` files with seed URLs.
- **sources.links.scraper:** Behaviour toggles for HTTP scraping (resetting data, URL verification, warning output).
- **sources.links.selenium_scraper:** Selenium configuration used for SSO scraping and optional link scraping.
- **sources.<name>.visible:** Mark whether documents harvested from a source should appear in chat citations and other user-facing listings (`true` by default).
- **sources.git.enabled / sources.sso.enabled / sources.jira.enabled / sources.redmine.enabled:** Toggle additional collectors when paired with `--sources`.
- **sources.jira.cutoff_date:** ISO-8601 date; JIRA tickets created before this are ignored.
- **embedding_name:** Embedding backend (`OpenAIEmbeddings`, `HuggingFaceEmbeddings`, ...).
- **embedding_class_map:** Backend specific parameters (model name, device, similarity threshold).
- **chunk_size / chunk_overlap:** Text splitter parameters.
- **reset_collection:** Whether to wipe the collection before re-populating.
- **num_documents_to_retrieve:** Top-k documents returned at query time.
- **distance_metric / use_hybrid_search / bm25_weight / semantic_weight:** Retrieval tuning knobs (BM25 `k1`/`b` are fixed when the PostgreSQL index is created).
- **utils.anonymizer** (legacy) / **data_manager.utils.anonymizer**: Redaction settings applied when ticket collectors anonymise content.

Source configuration is persisted to PostgreSQL `static_config.sources_config` at deployment time and used for runtime ingestion.

---

### `archi`

Defines pipelines and model routing.

- **pipelines:** List of pipeline names to load (e.g., `QAPipeline`).
- **pipeline_map:** Per-pipeline configuration of prompts, models, token limits, and ReAct recursion limits (`recursion_limit`, default `100`).
- **model_class_map:** Definitions for each model family (base model names, provider-specific kwargs).
- **chain_update_time:** Polling interval for hot-reloading chains.

---

### `utils`

Utility configuration for supporting components (mostly legacy fallbacks):

- **git:** Legacy toggle for Git scraping.
- **jira / redmine:** Compatibility settings for ticket integrations; prefer configuring these under `data_manager.sources`.

---

### Required Fields

Some fields are required depending on enabled services and pipelines. For example:

- `name`
- `data_manager.sources.links.input_lists` (or other source-specific configuration)
- `archi.pipelines` and matching `archi.pipeline_map` entries
- Service-specific fields (e.g., `services.piazza.network_id`, `services.grader_app.num_problems`)

See the [User Guide](user_guide.md) for more configuration examples and explanations.

---

### Example

```yaml
name: my_deployment
global:
  DATA_PATH: "/root/data/"
  ACCOUNTS_PATH: "/root/.accounts/"
  ACCEPTED_FILES: [".txt", ".pdf"]
  LOGGING:
    input_output_filename: "chain_input_output.log"
  verbosity: 3

data_manager:
  sources:
    links:
      input_lists:
        - examples/deployments/basic-gpu/miscellanea.list
      scraper:
        reset_data: true
        verify_urls: false
        enable_warnings: false
  utils:
    anonymizer:
      nlp_model: en_core_web_sm
  embedding_name: "OpenAIEmbeddings"
  chunk_size: 1000
  chunk_overlap: 0
  num_documents_to_retrieve: 5

services:
  chat_app:
    agent_class: CMSCompOpsAgent
    agents_dir: examples/agents
    provider: local
    model: llama3.2
    trained_on: "Course documentation"
    hostname: "example.mit.edu"
  postgres:
    port: 5432
    database: "archi"
```

---

**Tip:**
For a full template, see `src/cli/templates/base-config.yaml` in
the repository.

---

## V2 API (PostgreSQL-Consolidated)

The V2 API provides REST endpoints for the PostgreSQL-consolidated architecture, 
using PostgreSQL with pgvector for unified vector storage and metadata.

### Base URL

All endpoints are prefixed with `/api/`.

---

### Authentication

#### `POST /api/auth/login`

Authenticate with email and password.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secret123"
}
```

**Response:**
```json
{
  "success": true,
  "user": {
    "id": "user_abc123",
    "email": "user@example.com",
    "display_name": "John Doe",
    "is_admin": false
  },
  "session_token": "sess_..."
}
```

#### `POST /api/auth/logout`

End the current session.

**Response:**
```json
{
  "success": true
}
```

#### `GET /api/auth/me`

Get the current authenticated user.

**Response (authenticated):**
```json
{
  "authenticated": true,
  "user": {
    "id": "user_abc123",
    "email": "user@example.com",
    "display_name": "John Doe",
    "is_admin": false
  }
}
```

**Response (anonymous):**
```json
{
  "authenticated": false,
  "user": null
}
```

---

### User Management

#### `GET /api/users/me`

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
  "has_anthropic_key": false,
  "created_at": "2025-01-15T10:30:00Z"
}
```

#### `PATCH /api/users/me/preferences`

Update user preferences.

**Request:**
```json
{
  "theme": "light",
  "preferred_model": "claude-3-opus",
  "preferred_temperature": 0.5
}
```

#### `PUT /api/users/me/api-keys/{provider}`

Set BYOK API key (provider: `openrouter`, `openai`, `anthropic`).

**Request:**
```json
{
  "api_key": "sk-..."
}
```

#### `DELETE /api/users/me/api-keys/{provider}`

Delete BYOK API key.

---

### Configuration

#### `GET /api/config/static`

Get static (deploy-time) configuration.

**Response:**
```json
{
  "deployment_name": "my-archi",
  "embedding_model": "text-embedding-ada-002",
  "embedding_dimensions": 1536,
  "available_pipelines": ["QAPipeline", "AgentPipeline"],
  "available_models": ["gpt-4o", "claude-3-opus"],
  "auth_enabled": true,
  "prompts_path": "/root/archi/data/prompts/"
}
```

#### `GET /api/config/dynamic`

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
  "active_condense_prompt": "default",
  "active_chat_prompt": "default",
  "active_system_prompt": "default",
  "num_documents_to_retrieve": 10,
  "verbosity": 3
}
```

#### `PATCH /api/config/dynamic`

Update dynamic configuration. **Admin only.**

**Request:**
```json
{
  "active_model": "gpt-4o",
  "temperature": 0.8,
  "num_documents_to_retrieve": 5
}
```

**Response:** `403 Forbidden` if not admin.

#### `GET /api/config/effective`

Get effective configuration for the current user, with user preferences applied.

**Response:**
```json
{
  "active_model": "claude-3-opus",
  "temperature": 0.5,
  "max_tokens": 4096,
  "num_documents_to_retrieve": 10,
  "active_condense_prompt": "concise",
  "active_chat_prompt": "technical"
}
```

#### `GET /api/config/audit`

Get configuration change audit log. **Admin only.**

**Query params:**
- `limit`: Max entries to return (default: 100)

**Response:**
```json
{
  "entries": [
    {
      "id": 1,
      "user_id": "admin_user",
      "changed_at": "2025-01-20T15:30:00Z",
      "config_type": "dynamic",
      "field_name": "temperature",
      "old_value": "0.7",
      "new_value": "0.8"
    }
  ]
}
```

---

### Prompts

#### `GET /api/prompts`

List all available prompts by type.

**Response:**
```json
{
  "condense": ["default", "concise"],
  "chat": ["default", "formal", "technical"],
  "system": ["default", "helpful"]
}
```

#### `GET /api/prompts/{type}`

List prompts for a specific type.

**Response:**
```json
["default", "formal", "technical"]
```

#### `GET /api/prompts/{type}/{name}`

Get prompt content.

**Response:**
```json
{
  "type": "chat",
  "name": "default",
  "content": "You are a helpful AI assistant..."
}
```

#### `POST /api/prompts/reload`

Reload prompt cache from disk. **Admin only.**

**Response:**
```json
{
  "success": true,
  "message": "Reloaded 7 prompts"
}
```

---

### Document Selection (3-Tier)

The document selection system uses a 3-tier precedence:
1. **Conversation override** (highest priority)
2. **User default**
3. **System default** (all documents enabled)

#### `GET /api/documents/selection?conversation_id={id}`

Get enabled documents for a conversation.

#### `PUT /api/documents/user-defaults`

Set user's default for a document.

**Request:**
```json
{
  "document_id": 42,
  "enabled": false
}
```

#### `PUT /api/documents/conversation-override`

Set conversation-specific override.

**Request:**
```json
{
  "conversation_id": 123,
  "document_id": 42,
  "enabled": true
}
```

#### `DELETE /api/documents/conversation-override`

Clear conversation override (fall back to user default).

---

### Analytics

#### `GET /api/analytics/model-usage`

Get model usage statistics.

**Query params:**
- `start_date`: ISO date (optional)
- `end_date`: ISO date (optional)
- `service`: Filter by service (optional)

#### `GET /api/analytics/ab-comparisons`

Get A/B comparison statistics with win rates.

**Query params:**
- `model_a`: Filter by model A (optional)
- `model_b`: Filter by model B (optional)
- `start_date`: ISO date (optional)
- `end_date`: ISO date (optional)

---

### Data Viewer

Browse and manage ingested documents.

#### `GET /api/data/documents`

List all ingested documents.

**Query params:**
- `limit`: Max documents to return (default: 100)
- `offset`: Pagination offset (default: 0)
- `search`: Filter by document name
- `source_type`: Filter by source type (e.g., `links`, `git`, `ticket`)

Ticketing integrations normalize to `source_type: ticket` and record the provider in `metadata.ticket_provider` (e.g., `jira`, `redmine`).

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

#### `GET /api/data/documents/<hash>/content`

Get document content and chunks.

**Response:**
```json
{
  "hash": "5e90ca54526f3e11",
  "file_name": "readme.md",
  "content": "Full document text...",
  "chunks": [
    {
      "id": 1,
      "content": "Chunk text...",
      "metadata": {}
    }
  ]
}
```

#### `POST /api/data/documents/<hash>/enable`

Enable a document for retrieval.

**Response:**
```json
{
  "success": true,
  "hash": "5e90ca54526f3e11",
  "enabled": true
}
```

#### `POST /api/data/documents/<hash>/disable`

Disable a document from retrieval.

**Response:**
```json
{
  "success": true,
  "hash": "5e90ca54526f3e11",
  "enabled": false
}
```

#### `POST /api/data/bulk-enable`

Enable multiple documents.

**Request:**
```json
{
  "hashes": ["5e90ca54526f3e11", "a1b2c3d4e5f67890"]
}
```

#### `POST /api/data/bulk-disable`

Disable multiple documents.

**Request:**
```json
{
  "hashes": ["5e90ca54526f3e11", "a1b2c3d4e5f67890"]
}
```

#### `GET /api/data/stats`

Get document statistics.

**Response:**
```json
{
  "total_documents": 42,
  "enabled_documents": 40,
  "disabled_documents": 2,
  "total_chunks": 350,
  "by_source_type": {
    "links": 30,
    "ticket": 12
  }
}
```

---

### Health & Info

#### `GET /api/health`

Health check with database connectivity status.

#### `GET /api/info`

Get API version and available features.
