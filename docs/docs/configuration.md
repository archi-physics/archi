# Configuration Reference

Archi deployments are configured via YAML files passed to the CLI with `--config`. Any fields not specified are populated from the base template at `src/cli/templates/base-config.yaml`.

> **Tip:** Start from one of the example configs in `examples/deployments/` and customize from there.

---

## Top-Level Fields

### `name`

**Type:** string (required)

Name of your deployment. Used for container naming and directory structure.

```yaml
name: my_deployment
```

---

## `global`

Global settings shared across all services.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `DATA_PATH` | string | `/root/data/` | Path for persisted data inside containers |
| `ACCOUNTS_PATH` | string | `/root/.accounts/` | Path for uploader/grader account data |
| `ACCEPTED_FILES` | list | See below | File extensions allowed for manual uploads |
| `LOGGING.input_output_filename` | string | `chain_input_output.log` | Pipeline I/O log filename |
| `verbosity` | int | `3` | Default logging level for services (0-4) |

Default accepted files: `.pdf`, `.md`, `.txt`, `.docx`, `.html`, `.htm`, `.json`, `.yaml`, `.yml`, `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.sh`

---

## `services`

Configuration for containerized services. Each service has its own subsection.

### `services.chat_app`

The main chat interface.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent_class` | string | `CMSCompOpsAgent` | Pipeline class to run |
| `agents_dir` | string | — | Path to agent markdown files |
| `default_provider` | string | `local` | Default LLM provider |
| `default_model` | string | `llama3.2` | Default model |
| `client_timeout_seconds` | number | `600` | Chat request/stream timeout in seconds (sent to frontend as ms) |
| `tools` | dict | `{}` | Agent-class-specific tool settings (for example `tools.monit.url`) |
| `trained_on` | string | — | Description shown in the chat UI |
| `hostname` | string | `localhost` | Public hostname for the chat interface |
| `port` | int | `7861` | Internal container port |
| `external_port` | int | `7861` | Host-mapped port |
| `host` | string | `0.0.0.0` | Network binding |
| `num_responses_until_feedback` | int | `3` | Responses before prompting for feedback |
| `auth.enabled` | bool | `false` | Enable authentication |
| `alerts.managers` | list | `[]` | Usernames allowed to create and delete alerts |

#### `services.chat_app.alerts`

Controls access to the [Service Status Board & Alert Banners](services.md#service-status-board--alert-banners).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `alerts.managers` | list of strings | `[]` | Usernames granted alert manager access |

Access rules (evaluated in order):

1. **Auth disabled** → all users may manage alerts.
2. **Auth enabled** → a user is an alert manager if **either**:
    - their username is in the `alerts.managers` list, **or**
    - their session roles grant the `alerts:manage` permission.
3. **Auth enabled, no username match, no `alerts:manage` permission** → nobody may manage (safe default).

```yaml
# Option 1: explicit username list
services:
  chat_app:
    alerts:
      managers:
        - alice
        - bob

# Option 2: role-based via RBAC (can be combined with Option 1)
services:
  chat_app:
    auth:
      auth_roles:
        roles:
          ops-team:
            permissions:
              - alerts:manage
```

#### Chat-app evaluation configuration

The evaluation console is opt-in. Both `archi create` and the chat runtime
disable it when `enabled` is omitted or `false`. Set `enabled: true` explicitly
to expose the console and its APIs. When enabled, it persists catalogs,
atom-review drafts, jobs, and run artifacts.

```yaml
services:
  chat_app:
    agents_dir: ../configs/agents
    evaluations:
      enabled: true
      mcp_config_path: ../configs/qa_evaluation_mcp.yaml
```

- `agents_dir` points to the host directory of Markdown agent specs that the
  Console offers for selection.
- `evaluations.enabled` must be `true` to expose `/evaluations` and its APIs.
- `evaluations.mcp_config_path` is needed only for Dataset V2 live oracle
  items. It is an absolute host path or a path relative to this deployment YAML.
  Archi validates and stages the referenced evaluator MCP registry.

Top-level `mcp_servers` configures tools available to the tested agent; it is
not reused as the evaluator registry. See
[Chat-app evaluation configuration](evaluation.md#chat-app-evaluation-configuration)
and [Evaluator MCP registry](evaluation.md#evaluator-mcp-registry) for path
resolution, runtime reachability, authentication, and schema rules.

#### `services.chat_app.auth`

Authentication can be enabled with SSO or basic auth.

For RBAC-managed admin access, use SSO plus `auth_roles`. Basic auth supports identity-only login, but it does not assign RBAC roles.

```yaml
services:
  chat_app:
    auth:
      enabled: true
      basic:
        enabled: true
      auth_roles:
        default_role: base-user
        roles:
          base-user:
            permissions:
              - chat:query
              - ab:participate
          ab-reviewer:
            permissions:
              - documents:view
              - ab:view
              - ab:metrics
          ab-admin:
            permissions:
              - ab:manage
```

#### Provider Configuration

```yaml
services:
  chat_app:
    providers:
      local:
        enabled: true
        base_url: http://localhost:11434
        mode: ollama              # or openai_compat
        default_model: llama3.2
        models:
          - llama3.2
      gemini:
        enabled: true
```

### `services.postgres`

PostgreSQL database settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | string | `postgres` | Database hostname |
| `port` | int | `5432` | Database port |
| `user` | string | `archi` | Database user |
| `database` | string | `archi-db` | Database name |

### `services.vectorstore`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | string | `postgres` | Vector store backend (only `postgres` supported) |

### `services.data_manager`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `port` | int | `7871` | Internal port |
| `external_port` | int | `7871` | Host-mapped port |
| `host` | string | `0.0.0.0` | Network binding |
| `enabled` | bool | `true` | Enable data manager service |

### `services.grafana`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `port` | int | `3000` | Grafana port |
| `external_port` | int | `3000` | Host-mapped port |

### `services.grader_app`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `port` | int | `7861` | Internal port |
| `external_port` | int | `7862` | Host-mapped port |
| `provider` | string | — | Provider for grading pipelines |
| `model` | string | — | Model for grading pipelines |
| `num_problems` | int | — | Number of problems (must match rubric files) |
| `local_rubric_dir` | string | — | Path to rubric files |
| `local_users_csv_dir` | string | — | Path to users CSV |

### Other Services

- **`services.jira_ticket_responder`**: Requires `url`, `projects` (a non-empty list of Jira project keys using uppercase letters, digits, or underscores and starting with a letter), and `visible_to_role`. Optional fields are `poll_interval_minutes` (defaults to `1`), `lookback_days` (defaults to `7`), `eligible_statuses` (defaults to `["Open", "In Progress"]`), `agent_class`, `agents_dir`, `default_provider`, and `default_model`; `agent_class` defaults to `CMSCompOpsAgent`, while provider/model default from `services.chat_app`. Do not add `enabled`; run the process with `--services jira_ticket_responder`.
- **`services.piazza`**: Requires `network_id`, `agent_class`, `provider`, `model`
- **`services.mattermost`**: Requires `update_time`
- **`services.redmine_mailbox`**: Requires `url`, `project`, `redmine_update_time`, `mailbox_update_time`
- **`services.benchmarking`**: See [Benchmarking](benchmarking.md)

---

## `data_manager`

Controls data ingestion, vectorstore behaviour, and retrieval settings.

### Core Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `collection_name` | string | `default_collection` | Vector store collection name |
| `embedding_name` | string | `OpenAIEmbeddings` | Embedding backend |
| `chunk_size` | int | `1000` | Max characters per text chunk |
| `chunk_overlap` | int | `0` | Overlapping characters between chunks |
| `parallel_workers` | int | `32` | Parallel ingestion workers |
| `reset_collection` | bool | `true` | Wipe collection on startup |
| `distance_metric` | string | `cosine` | Similarity metric: `cosine`, `l2`, `ip` |

### Retrieval Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `retrievers.hybrid_retriever.num_documents_to_retrieve` | int | `5` | Top-k documents per query |
| `retrievers.hybrid_retriever.bm25_weight` | float | `0.6` | BM25 keyword score weight |
| `retrievers.hybrid_retriever.semantic_weight` | float | `0.4` | Semantic similarity weight |
| `stemming.enabled` | bool | `false` | Enable Porter Stemmer for improved matching |

> **Note:** `use_hybrid_search` is a dynamic runtime setting (managed via the configuration API), not a YAML config key.

### Sources

```yaml
data_manager:
  sources:
    links:
      input_lists:
        - miscellanea.list
      scraper:
        reset_data: true
        verify_urls: false
        enable_warnings: false
      selenium_scraper:
        enabled: false
    git:
      enabled: false
    sso:
      enabled: false
    jira:
      url: https://jira.example.com
      projects: []
      anonymize_data: true
      cutoff_date: null
    redmine:
      url: https://redmine.example.com
      project: null
      anonymize_data: true
```

The `visible` flag on any source (`sources.<name>.visible`) controls whether content appears in chat citations (default: `true`).

### Embedding Configuration

```yaml
data_manager:
  embedding_name: OpenAIEmbeddings
  embedding_class_map:
    OpenAIEmbeddings:
      class: OpenAIEmbeddings
      kwargs:
        model: text-embedding-3-small
      similarity_score_reference: 10
```

See [Models & Providers](models_providers.md#embedding-models) for all embedding options.

### Anonymizer

```yaml
data_manager:
  utils:
    anonymizer:
      nlp_model: en_core_web_sm
      excluded_words: []
      greeting_patterns: []
      signoff_patterns: []
      email_pattern: '[\w\.-]+@[\w\.-]+\.\w+'
      username_pattern: '\[~[^\]]+\]'
```

---

## A/B Testing Pool

Archi supports champion-vs-variant A/B testing via a server-side variant pool. When configured, the system automatically pairs the champion agent against a random variant for each comparison. Users vote on which response is better, and aggregate metrics are tracked per variant.

Configure A/B testing under `services.chat_app.ab_testing`:

```yaml
services:
  chat_app:
    ab_testing:
      enabled: true
      force_yaml_override: false
      comparison_rate: 0.25
      variant_label_mode: post_vote_reveal
      activity_panel_default_state: hidden
      max_pending_comparisons_per_conversation: 1
      eligible_roles: []
      eligible_permissions: []
      pool:
        champion: default
        variants:
          - label: default
            agent_spec: default.md
          - label: creative
            agent_spec: default.md
            provider: openai
            model: gpt-4o
            recursion_limit: 30
          - label: concise
            agent_spec: concise.md
            provider: anthropic
            model: claude-sonnet-4-20250514
            num_documents_to_retrieve: 3
```

`services.ab_testing` is deprecated and no longer loaded. Use `services.chat_app.ab_testing` only.

If `enabled: true` is set before the A/B pool is fully configured, Archi starts successfully but keeps A/B inactive until setup is completed in the admin UI. Missing champion/variant selections or unresolved A/B agent-spec records are surfaced as warnings instead of blocking startup.

The runtime source of truth for A/B agent specs is now PostgreSQL. The optional `ab_agents_dir` path is treated only as a legacy import source during reconciliation; runtime A/B loading never falls back to reading staged container markdown files directly.

New A/B specs created through the admin UI are stored in the same PostgreSQL-backed catalog. Existing A/B specs are not edited through the admin page; if you need a changed prompt or tool selection, create a new A/B spec and point the variant at that new catalog entry.

For `services.chat_app.ab_testing`, the persisted PostgreSQL-backed static config becomes authoritative after the first successful bootstrap. On later restarts or reseeds, the rendered YAML A/B block is used only if no persisted A/B block exists yet, unless `force_yaml_override: true` is set to intentionally replace the saved A/B state. This flag is bootstrap-only and defaults to `false`.

### Variant Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `label` | string | *required* | Unique human-facing variant label used in the UI and metrics |
| `agent_spec` | string | *required* | A/B agent-spec filename resolved from the database-backed A/B catalog |
| `provider` | string | `null` | Override LLM provider |
| `model` | string | `null` | Override LLM model |
| `num_documents_to_retrieve` | int | `null` | Override retriever document count |
| `recursion_limit` | int | `null` | Override agent recursion limit |

### Pool Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable the experiment pool |
| `ab_agents_dir` | string | `/root/archi/ab_agents` | Optional legacy import directory for migrating A/B markdown specs into the DB catalog |
| `force_yaml_override` | boolean | `false` | Bootstrap-only override that forces the rendered YAML A/B block to replace persisted A/B state on reseed/restart |
| `comparison_rate` | float | `1.0` | Fraction of eligible turns that should run A/B |
| `variant_label_mode` | string | `post_vote_reveal` | One of `hidden`, `post_vote_reveal`, `always_visible` |
| `activity_panel_default_state` | string | `hidden` | One of `hidden`, `collapsed`, `expanded` |
| `max_pending_comparisons_per_conversation` | int | `1` | Maximum unresolved comparisons per conversation |
| `eligible_roles` | list[string] | `[]` | Restrict participation to matching RBAC roles |
| `eligible_permissions` | list[string] | `[]` | Restrict participation to matching permissions |

### Config-To-UI Mapping

| Config Field | Config Value | UI Label | Runtime Meaning |
|--------------|--------------|----------|-----------------|
| `comparison_rate` | `0.0..1.0` | `Comparison Rate` | Fraction of eligible turns that become A/B comparisons |
| `variant_label_mode` | `hidden` | `Hidden` | Hide variant labels before and after the vote |
| `variant_label_mode` | `post_vote_reveal` | `Post-Vote Reveal` | Hide variant labels until the vote is submitted |
| `variant_label_mode` | `always_visible` | `Always Visible` | Show variant labels throughout the comparison |
| `activity_panel_default_state` | `hidden` | `Hidden` | Do not show the per-arm activity panel by default |
| `activity_panel_default_state` | `collapsed` | `Collapsed` | Show the activity panel in a collapsed state |
| `activity_panel_default_state` | `expanded` | `Expanded` | Show the activity panel expanded by default |
| `max_pending_comparisons_per_conversation` | integer >= 1 | `Max Pending Comparisons Per Conversation` | Limit unresolved comparisons before the user must vote |
| `pool.champion` | existing variant label | `Champion` | Baseline variant that always appears in each comparison |

The `champion` field must reference an existing variant `label`. At least two variants are required before the experiment becomes active. `name`-only variant config is not supported. When a user enables A/B mode in the chat UI, the pool takes over: the champion always appears in one arm, and a random variant is placed in the other. Arm positions (A vs B) are randomized per comparison.

### A/B RBAC and User Preference

Use RBAC to separate participation, read-only review, metrics access, and write access:

| Permission | Purpose |
|------------|---------|
| `ab:participate` | Makes a user eligible for A/B comparisons and shows the per-user sampling slider in chat settings |
| `ab:view` | Allows read-only access to the A/B admin page |
| `ab:metrics` | Allows access to aggregate A/B metrics |
| `ab:manage` | Allows editing variants, A/B agent specs, and experiment settings |

`services.chat_app.ab_testing.comparison_rate` remains the deployment default. Users with `ab:participate` can override that default per account with a `0..1` slider in chat settings.

Users with `ab:view`, `ab:metrics`, or `ab:manage` can open the dedicated A/B Testing page from the data viewer and from chat settings. Users with `ab:participate` but not A/B page access still get the personal sampling slider in chat settings.

Variant metrics (wins, losses, ties) are tracked in the `ab_variant_metrics` database table and available via `GET /api/ab/metrics`.

---

## Agent Configuration Model

Archi no longer uses a top-level `archi:` block in standard deployment YAML.

Agent behavior is defined by:

- `services.chat_app.agent_class`: which pipeline class runs (for example `CMSCompOpsAgent`)
- `services.chat_app.agents_dir`: where agent spec markdown files live
- agent specs (`*.md`): selected tool subset (`tools`) and system prompt body
- `services.chat_app.tools`: optional agent-class-specific tool settings

Example:

```yaml
services:
  chat_app:
    agent_class: CMSCompOpsAgent
    agents_dir: examples/agents
    tools:
      monit:
        url: https://monit-grafana.cern.ch
```

See [Agents & Tools](agents_tools.md) for agent spec format and tool selection.

---

## Complete Example

```yaml
name: my_deployment

global:
  DATA_PATH: "/root/data/"
  ACCEPTED_FILES: [".txt", ".pdf", ".md"]
  verbosity: 3

services:
  chat_app:
    agent_class: CMSCompOpsAgent
    agents_dir: examples/agents
    default_provider: local
    default_model: llama3.2
    trained_on: "Course documentation"
    hostname: "example.mit.edu"
    external_port: 7861
    providers:
      local:
        enabled: true
        base_url: http://localhost:11434
        mode: ollama
        models:
          - llama3.2
  postgres:
    port: 5432
    database: archi-db
  vectorstore:
    backend: postgres

data_manager:
  sources:
    links:
      input_lists:
        - examples/deployments/basic-gpu/miscellanea.list
      scraper:
        reset_data: true
        verify_urls: false
  embedding_name: OpenAIEmbeddings
  chunk_size: 1000
  chunk_overlap: 0
```

> **Tip:** For the full base template with all defaults, see `src/cli/templates/base-config.yaml` in the repository.
