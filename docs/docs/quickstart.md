# Quickstart

Deploy your first instance of Archi and walk through the important concepts.

## Sources and Services

Archi can ingest data from a variety of **sources** and supports several **services**. List them with the CLI command below and decide which ones you want to use so that we can configure them.

```bash
archi list-services
```

Example output:

```
Available Archi services:

Application Services:
  chatbot              Interactive chat interface for users to communicate with the AI agent
  grafana              Monitoring dashboard for system and LLM performance metrics
  uploader             Admin interface for uploading and managing documents
  grader               Automated grading service for assignments with web interface

Integration Services:
  piazza               Integration service for Piazza posts and Slack notifications
  mattermost           Integration service for Mattermost channels
  redmine-mailer       Email processing and Cleo/Redmine ticket management

Data Sources:
  git                 Git repository scraping for MkDocs-based documentation
  jira                Jira issue tracking integration
  redmine             Redmine ticket integration
  sso                 SSO-backed web crawling
```

See the [User Guide](user_guide.md) for detailed information about each service and source.

## Pipelines

Archi supports several pipelines (agentic and not). The active agent class is configured per service, and the agent prompt/tools are defined in agent markdown files.

Example agent spec file (`examples/agents/default.md`):

```markdown
---
name: CMS CompOps Default
tools:
  - search_local_files
  - search_metadata_index
  - list_metadata_schema
  - fetch_catalog_document
  - search_vectorstore_hybrid
---

You are a CMS CompOps assistant.
Use tools to gather evidence before answering, and keep responses concise.
```

## Configuration

Once you have chosen the services, sources, and agent class you want to use, create a configuration file that specifies their settings. You can start from one of the example configuration files under `examples/deployments/`, or create your own from scratch. This file sets parameters; the selected services and sources are determined at deployment time.

> **Important:** The configuration file follows the format of `src/cli/templates/base-config.yaml`. Any fields not specified in your configuration will be populated with the defaults from this template.

Example configuration for the `chatbot` service using a local Ollama model and agent specs from `services.chat_app.agents_dir`:

```yaml
name: my_archi

services:
  chat_app:
    agent_class: CMSCompOpsAgent
    agents_dir: examples/agents
    default_provider: local
    default_model: llama3.2
    providers:
      local:
        base_url: http://localhost:11434
        mode: ollama
        models:
          - llama3.2
    trained_on: "My data"
    hostname: "<your-hostname>"
  vectorstore:
    backend: postgres  # Uses PostgreSQL with pgvector (default)

data_manager:
  sources:
    links:
      visible: true          # include scraped pages in the chat citations
      input_lists:
        - examples/deployments/basic-gpu/miscellanea.list
  embedding_name: HuggingFaceEmbeddings
  chunk_size: 1000
```

Agent specs are Markdown files (see `examples/agents/`) with YAML frontmatter for `name` and `tools`, and the prompt in the Markdown body.

> **Using OpenAI, Anthropic, Gemini, OpenRouter, or a non-Ollama local server?** This quickstart uses Ollama for the first deployment path. For provider-specific startup snippets (including required secrets and config), see [Models & Providers](models_providers.md#quick-start-by-provider).

<details markdown="1">
<summary>Explanation of configuration parameters</summary>

- `name`: Name of your Archi deployment.
- `data_manager`: Settings related to data ingestion and the vector store.
- `data_manager.sources.links.input_lists`: Lists of URLs to seed the deployment.
- `data_manager.sources.<source>.visible`: Controls whether content from a given source is surfaced to end users (defaults to `true`).
- `data_manager.embedding_name`: Embedding model used for vectorization.
- `data_manager.chunk_size`: Controls how documents are split prior to embedding.
- `services`: Settings for individual services/interfaces.
- `services.chat_app.agent_class`: Agent class to run (pipeline class name).
- `services.chat_app.agents_dir`: Local path to agent markdown files (copied into the deployment).
- `services.chat_app.default_provider` and `services.chat_app.default_model`: Default provider/model for chat when no UI override is set.
- `services.chat_app.providers.local`: Ollama/local provider configuration.
- `services.chat_app`: Chat interface configuration, including hostname and descriptive metadata.
- `services.vectorstore.backend`: Vector store backend (`postgres` with pgvector).

</details>

## Secrets

Secrets are sensitive values (passwords, API keys, etc.) that should not be stored directly in code or configuration files. Store them in a single `.env` file on your filesystem.

Minimal deployments (chatbot with open-source LLM and embeddings) require:

- `PG_PASSWORD`: password used to secure the database.

Create the secrets file with:

```bash
echo "PG_PASSWORD=my_strong_password" > ~/.secrets.env
```

If you are not using open-source models, supply the relevant API credentials:

- `OPENAI_API_KEY`: OpenAI API key.
- `OPENROUTER_API_KEY`: OpenRouter API key.
- `OPENROUTER_SITE_URL`: Optional site URL for OpenRouter attribution.
- `OPENROUTER_APP_NAME`: Optional app name for OpenRouter attribution.
- `ANTHROPIC_API_KEY`: Anthropic API key.
- `GOOGLE_API_KEY`: Google Gemini API key.
- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace access token (for private models or embeddings).

Other services may require additional secrets; see the [User Guide](user_guide.md) for details.

## Creating an Archi Deployment with Ollama

Create your deployment with the CLI. A deployment with a local Ollama model (make sure you specify in the `config.yaml` the URL of your Ollama instance):

```bash
archi create --name my-archi \
  --config examples/deployments/basic-ollama/config.yaml \
  --podman \
  --env-file .secrets.env \
  --services chatbot
```

| Flag | Description |
|------|-------------|
| `--name` / `-n` | Deployment name |
| `--config` / `-c` | Path to configuration file |
| `--env-file` / `-e` | Path to the secrets `.env` file |
| `--services` / `-s` | Comma-separated services to deploy |
| `--podman` | Use Podman instead of Docker |

Agent specs are loaded from `services.chat_app.agents_dir` in the config.

<details>
<summary>Example output</summary>

```bash
archi create --name my-archi --config examples/deployments/basic-ollama/config.yaml --podman --env-file .secrets.env --services chatbot
```

```
Starting Archi deployment process...
[archi] Creating deployment 'my-archi' with services: chatbot
[archi] Auto-enabling dependencies: postgres
[archi] Configuration validated successfully
[archi] You are using an embedding model from HuggingFace; make sure to include a HuggingFace token if required for usage, it won't be explicitly enforced
[archi] Required secrets validated: PG_PASSWORD
[archi] Volume 'archi-pg-my-archi' already exists. No action needed.
[archi] Volume 'archi-my-archi' already exists. No action needed.
[archi] Starting compose deployment from /path/to/my/.archi/archi-my-archi
[archi] Using compose file: /path/to/my/.archi/archi-my-archi/compose.yaml
[archi] (This might take a minute...)
[archi] Deployment started successfully
Archi deployment 'my-archi' created successfully!
Services running: chatbot, postgres
[archi] Chatbot: http://localhost:7861
```

</details>

The first deployment builds the container images from scratch (which may take a few minutes). Subsequent deployments reuse the images and complete much faster (roughly a minute).

> **Tip:** Having issues? Run the command with `-v 4` to enable DEBUG-level logging.

### Verifying a deployment

Run these checks after `archi create`:

**Step 1: Confirm deployment registration**

```bash
archi list-deployments
```

You should see your deployment name (for example `my-archi`).

**Step 2: Confirm services are running with your container runtime**

```bash
podman ps
# or: docker ps
```

You should see containers for at least `chatbot` and `postgres`.

**Step 3: Open the chat app URL printed by the CLI (default `http://localhost:7861`) and verify the UI loads.**


---

## Next Steps

Once your deployment is running:

- **Chat UI**: Open `http://localhost:7861` in your browser to start chatting.
- **Data Viewer**: Navigate to the `/data` page in the chat UI to browse ingested documents.
- **Upload Documents**: If you deployed the `uploader` service, access the upload interface at its configured port.

From here, explore the rest of the documentation:

- [User Guide](user_guide.md) — overview of all capabilities
- [Agents & Tools](agents_tools.md) — customize agent behavior and prompts
- [Models & Providers](models_providers.md) — switch to cloud LLMs (OpenAI, Anthropic, Gemini)
- [Configuration Reference](configuration.md) — full YAML config schema
- [CLI Reference](cli_reference.md) — all CLI commands and options
- [Troubleshooting](troubleshooting.md) — common issues and fixes
