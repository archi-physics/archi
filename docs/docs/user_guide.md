# User Guide

This guide covers the core concepts and features of Archi. Each topic has its own dedicated page for detailed reference.

## Overview

Archi is a retrieval-based assistant framework with four core parts:

- **Data sources**: Where knowledge comes from (links, git repos, JIRA/Redmine, uploaded files)
- **Vector store + retrievers**: Where ingested content is indexed and searched semantically/lexically
- **Agents + tools**: The reasoning layer that decides what to do and can call tools (search, fetch, MCP, etc.)
- **Services**: The apps users interact with (`chatbot`, `data_manager`, integrations, dashboards)

Why both a vector store and tools?

- The **vector store** is best for relevance-based retrieval across the indexed knowledge base.
- **Tools** let the agent do targeted operations (metadata lookup, full-document fetch, external system calls) that pure embedding search cannot do reliably.

Services are enabled at deployment via flags to `archi create`:

```bash
archi create [...] --services chatbot
```

Pipelines (agent classes) define runtime behavior. Agent specs define prompt + enabled tool subset. Models, embeddings, and retriever settings are configured in YAML.

## Data Sources

Data sources define what gets ingested into Archi's knowledge base for retrieval.
Archi supports several data ingestion methods:

- **Web link lists** (including SSO-protected pages)
- **Git scraping** for MkDocs-based repositories
- **JIRA** and **Redmine** ticketing systems
- **Manual document upload** via the Uploader service or direct file copy
- **Local documents**

Sources are configured under `data_manager.sources` in your config file.

**[Read more →](data_sources.md)**

---

## Services

Archi provides these deployable services:

| Service | Description | Default Port |
|---------|-------------|-------------|
| `chatbot` | Web-based chat interface | 7861 |
| `data_manager` | Data ingestion and vectorstore management | 7871 |
| `jira_ticket_responder` | Jira ticket responder service | — |
| `piazza` | Piazza forum integration with Slack | — |
| `redmine-mailer` | Redmine ticket responses via email | — |
| `mattermost` | Mattermost channel integration | — |
| `grafana` | Monitoring dashboard | 3000 |
| `grader` | Automated grading service | 7862 |

**[Read more →](services.md)**

---

## Agents & Tools

Agents are defined by **agent specs** — Markdown files with YAML frontmatter specifying name, tools, and system prompt. The agent specs directory is configured via `services.chat_app.agents_dir`.

**[Read more →](agents_tools.md)**

---

## Playbooks

Playbooks are per-user reusable instruction packs — the chat-side analog of editing a `SKILL.md` skill file, for users who have no filesystem access:

- **Invoke** one in chat by typing `/` and picking it from the menu (`/name arguments…`); the turn shows a playbook chip.
- **Manage** them under **Settings → Playbooks**: create, edit, delete, share (make public), and add public playbooks shared by other users to your active list.
- **Ask the agent**: the assistant can save, update, and delete your playbooks through its own tools (it always previews a draft and asks before saving, and asks before deleting).
- **Export/import** uses the Agent Skills `<name>/SKILL.md` zip layout, so playbooks are portable to and from claude.ai. Imports always arrive private.

Public playbooks from other users are read-only and their content is fenced before the agent sees it. See the [API reference](api_reference.md#playbooks) for the REST endpoints.

---

## Models & Providers

Archi supports five LLM provider types:

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4o, GPT-4, etc. |
| Anthropic | Claude 4, Claude 3.5 Sonnet, etc. |
| Google Gemini | Gemini 2.0 Flash, Gemini 1.5 Pro, etc. |
| OpenRouter | Access to 100+ models via a unified API |
| Local (Ollama/vLLM) | Any open-source model |

Users can also provide their own API keys at runtime via **Bring Your Own Key (BYOK)**.

**[Read more →](models_providers.md)**

---

## Configuration Management

Archi uses a three-tier configuration system:

1. **Static Configuration** (deploy-time, immutable): deployment name, embedding model, available pipelines
2. **Dynamic Configuration** (admin-controlled, runtime-modifiable): default model, temperature, retrieval parameters
3. **User Preferences** (per-user overrides): preferred model, temperature, prompt selections

Settings are resolved as: User Preference → Dynamic Config → Static Default.

See the [Configuration Reference](configuration.md) for the full YAML schema and the [API Reference](api_reference.md) for the configuration API.

---

## Secrets

Secrets are stored in a `.env` file passed via `--env-file`. Required secrets depend on your deployment:

| Secret | Required For |
|--------|-------------|
| `PG_PASSWORD` | All deployments |
| `OPENAI_API_KEY` | OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `GOOGLE_API_KEY` | Google Gemini provider |
| `OPENROUTER_API_KEY` | OpenRouter provider |
| `HUGGINGFACEHUB_API_TOKEN` | Private HuggingFace models |
| `GIT_USERNAME` / `GIT_TOKEN` | Git source |
| `JIRA_PAT` | JIRA source |
| `JIRA_TICKET_RESPONDER_PAT` | Jira ticket responder service |
| `REDMINE_USER` / `REDMINE_PW` | Redmine source |

See [Data Sources](data_sources.md) and [Services](services.md) for service-specific secrets.

---

## Benchmarking

Archi has benchmarking functionality via the `archi evaluate` CLI command:

- **SOURCES mode**: Checks if retrieved documents contain the correct sources
- **RAGAS mode**: Uses the Ragas evaluator for answer relevancy, faithfulness, context precision, and context relevancy

**[Read more →](benchmarking.md)**

For repeatable evaluation of complete agent answers against fixed, reviewable
answer obligations, use the separate `archi eval qa` workflow or browser
console.

**[Evaluation guide →](evaluation.md)**

---

## Alerts & Service Status Board

The **Service Status Board (SSB)** lets operators communicate service health, outages, maintenance windows, and general announcements to all users directly in the chat app.

### For all users

- **Alert banners** appear at the top of every page when active alerts exist. Up to 5 banners are shown; each can be dismissed individually.
- The banner colour indicates severity: red (`alarm`), amber (`warning`), blue (`news`), slate (`info`).
- Click **details** on any banner, or navigate to **Status** in the header, to view the full [Service Status Board](/ssb/status) with alert history.

### For alert managers

Navigate to `/ssb/status` and use the **Post New Alert** form to create an alert. Required fields are **Message** and **Severity**. Optionally add an extended **Description** (shown only on the status page) and set an **Expires at** datetime for time-bounded notices.

Delete alerts by clicking **Delete** on any alert card. Deletion is permanent; expired alerts remain in history until deleted.

To grant alert manager access, add usernames to `services.chat_app.alerts.managers` in your config:

```yaml
services:
  chat_app:
    alerts:
      managers:
        - alice
        - bob
```

If auth is disabled, all users can manage alerts. If auth is enabled and the managers list is absent or empty, nobody can manage alerts.

**[Read more →](services.md#service-status-board--alert-banners)**

---

## Admin Guide

### Becoming an Admin

Set admin status in PostgreSQL:

```sql
UPDATE users SET is_admin = true WHERE email = 'admin@example.com';
```

### Admin Capabilities

- Set deployment-wide defaults via the dynamic configuration API
- Manage prompts (add, edit, reload via API)
- View the configuration audit log
- Grant admin privileges to other users

### Audit Logging

All admin configuration changes are logged and queryable:

```
GET /api/config/audit?limit=50
```

See the [API Reference](api_reference.md#configuration) for full endpoint documentation.
