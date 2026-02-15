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

Archi supports several pipelines (agent classes and classic pipelines). The active agent class is configured per service, and the agent prompt/tools are defined in agent markdown files.

## Configuration

Once you have chosen the services, sources, and agent class you want to use, create a configuration file that specifies their settings. You can start from one of the example configuration files under `examples/deployments/`, or create your own from scratch. This file sets parameters; the selected services and sources are determined at deployment time.

> **Important:** The configuration file follows the format of `src/cli/templates/base-config.yaml`. Any fields not specified in your configuration will be populated with the defaults from this template.

Example configuration for the `chatbot` service using a local Ollama model and an agent spec from `--agents`:

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

<details>
<summary>Explanation of configuration parameters</summary>

- `name`: Name of your Archi deployment.
- `data_manager`: Settings related to data ingestion and the vector store.
  - `sources.links.input_lists`: Lists of URLs to seed the deployment.
  - `sources.<name>.visible`: Controls whether content from a given source should be surfaced to end users (defaults to `true`).
  - `embedding_name`: Embedding model used for vectorization.
  - `chunk_size`: Controls how documents are split prior to embedding.
- `services`: Settings for individual services/interfaces.
  - `chat_app.agent_class`: Agent class to run (pipeline class name).
  - `chat_app.agents_dir`: Local path to agent markdown files (copied into the deployment).
  - `chat_app.default_provider`/`chat_app.default_model`: Default provider/model for chat when no UI override is set.
  - `chat_app.providers.local`: Ollama/local provider configuration.
  - `chat_app`: Chat interface configuration, including hostname and descriptive metadata.
  - `vectorstore.backend`: Vector store backend (`postgres` with pgvector).

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
- `OPENROUTER_API_KEY`: OpenRouter API key (for `OpenRouterLLM`).
- `OPENROUTER_SITE_URL`: Optional site URL for OpenRouter attribution.
- `OPENROUTER_APP_NAME`: Optional app name for OpenRouter attribution.
- `ANTHROPIC_API_KEY`: Anthropic API key.
- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace access token (for private models or embeddings).

Other services may require additional secrets; see the [Secrets Management](secrets.md) guide for a complete reference.

## Creating an Archi Deployment

Create your deployment with the CLI. A CPU-only deployment with a local Ollama model:

```bash
archi create -n my-archi \
  --config examples/deployments/basic-ollama/config.yaml \
  --env-file .secrets.env \
  --services chatbot \
  --agents examples/agents
```

To use GPU acceleration, add `--gpu-ids`:

```bash
archi create -n my-archi \
  --config examples/deployments/basic-ollama/config.yaml \
  --env-file .secrets.env \
  --services chatbot \
  --agents examples/agents \
  --gpu-ids all
```

| Flag | Description |
|------|-------------|
| `--name` / `-n` | Deployment name |
| `--config` / `-c` | Path to configuration file |
| `--env-file` / `-e` | Path to the secrets `.env` file |
| `--services` / `-s` | Comma-separated services to deploy |
| `--agents` / `-a` | **(Required)** Path to a directory of agent markdown files |
| `--gpu-ids` | GPU IDs to use (e.g., `0`, `0,1`, or `all`) |
| `--podman` | Use Podman instead of Docker |

> **Note:** By default the deployment uses only the link sources in your config's `data_manager.sources.links.input_lists`. To include other sources (git, JIRA, etc.), use the `--sources` flag and add their config under `data_manager.sources`.

<details>
<summary>Example output</summary>

```bash
archi create --name my-archi --config examples/deployments/basic-ollama/config.yaml --podman --env-file .secrets.env --services chatbot --gpu-ids all --agents examples/agents
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

### A note about multiple configurations

When multiple configuration files are passed, their `services` sections must remain consistent, otherwise the deployment fails. The current use cases for multiple configurations include swapping pipelines/prompts dynamically via the chat app and maintaining separate benchmarking configurations.

### Verifying a deployment

List running deployments with:

```bash
archi list-deployments
```

You should see output similar to:

```text
Existing deployments:
```

(Additional details will follow for each deployment.)

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
