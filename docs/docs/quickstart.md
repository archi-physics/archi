# Quickstart

Deploy your first instance of A2RCHI and walk through the important concepts.

## Sources and Services

A2RCHI can ingest data from a variety of **sources** and supports several **services**. List them with the CLI command below and decide which ones you want to use so that we can configure them.

```bash
a2rchi list-services
```

Example output:

```
Available A2RCHI services:

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

A2RCHI supports several pipelines—pre-defined sequences of operations that process user inputs and generate responses. Each service supports a subset of pipelines (see the [User Guide](user_guide.md) for details).

An example pipeline is `QAPipeline`, a question-answering pipeline that takes a user's question, retrieves relevant documents from the vector store, and generates an answer using a language model.

You specify which pipelines should be available in the configuration file.

## Configuration

Once you have chosen the services, sources, and pipelines you want to use, create a configuration file that specifies their settings. You can start from one of the example configuration files under `examples/deployments/`, or create your own from scratch. This file sets parameters; the selected services and sources are determined at deployment time.

> **Important:** Static configuration is rendered from `src/cli/templates/base-static-config.yaml`. A2RCHI runtime settings are rendered to a separate `configs/<name>.a2rchi-settings.yaml` file from `src/cli/templates/a2rchi-settings.yaml` and seeded into Postgres.

Example configuration (`examples/deployments/basic-gpu/config.yaml`) for the `chatbot` service using `QAPipeline` with a local VLLM model:

```yaml
name: my_a2rchi

data_manager:
  sources:
    links:
      visible: true          # include scraped pages in the chat citations
      input_lists:
        - examples/deployments/basic-gpu/miscellanea.list
  embedding_name: HuggingFaceEmbeddings
  chunk_size: 1000

a2rchi:
  pipelines:
    - QAPipeline
  pipeline_map:
    QAPipeline:
      prompts:
        required:
          condense_prompt: examples/deployments/basic-gpu/condense.prompt
          chat_prompt: examples/deployments/basic-gpu/qa.prompt
      models:
        required:
          chat_model: VLLM
          condense_model: VLLM
  model_class_map:
    VLLM:
      kwargs:
        base_model: 'Qwen/Qwen2.5-7B-Instruct-1M'
        quantization: True
        max_model_len: 32768
        tensor_parallel_size: 2
        repetition_penalty: 1.0
        gpu_memory_utilization: 0.5

services:
  chat_app:
    trained_on: "My data"
    hostname: "<your-hostname>"
  chromadb:
    chromadb_host: localhost
```

<details>
<summary>Explanation of configuration parameters</summary>

- `name`: Name of your A2RCHI deployment.
- `data_manager`: Settings related to data ingestion and the vector store.
  - `sources.links.input_lists`: Lists of URLs to seed the deployment.
  - `sources.<name>.visible`: Controls whether content from a given source should be surfaced to end users (defaults to `true`).
  - `embedding_name`: Embedding model used for vectorization.
  - `chunk_size`: Controls how documents are split prior to embedding.
- `a2rchi`: Core pipeline settings (now stored in `configs/<name>.a2rchi-settings.yaml`, seeded into Postgres, and read from there at runtime).
  - `pipelines`: Pipelines to use (e.g., `QAPipeline`).
  - `pipeline_map`: Configuration for each pipeline, including prompts and models.
  - `model_class_map`: Mapping of model names to their classes and parameters.
- `services`: Settings for individual services/interfaces.
  - `chat_app`: Chat interface configuration, including hostname and descriptive metadata.
  - `chromadb`: Connection details for the vector store container.

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
- `ANTHROPIC_API_KEY`: Anthropic API key.
- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace access token (for private models or embeddings).

Other services may require additional secrets; see the [User Guide](user_guide.md) for details.

## Creating an A2RCHI Deployment

Create your deployment with the CLI:

```bash
a2rchi create --name my-a2rchi --config examples/deployments/basic-gpu/config.yaml --podman --env-file .secrets.env --services chatbot --gpu-ids all
```

This command specifies:

- `--name`: Deployment name.
- `--config`: Path to the configuration file.
- `--podman`: Use Podman for container management (`docker` is the default).
- `--env-file`: Path to the secrets file.
- `--services`: Services to deploy (only the `chatbot` service in this example but others can be included separated by commas).

Note that this command will create a deployment using only the link sources specified in the `data_manager.sources.links.input_lists` by default, if other sources (such as git-based documentation or pages under sso) want to be included they must be included using the `--sources` flag and in the configuration file.

<details>
<summary>Example output</summary>

```bash
a2rchi create --name my-a2rchi --config examples/deployments/basic-gpu/config.yaml --podman --env-file .secrets.env --services chatbot --gpu-ids all
```

```
Starting A2RCHI deployment process...
[a2rchi] Creating deployment 'my-a2rchi' with services: chatbot
[a2rchi] Auto-enabling dependencies: postgres, chromadb
[a2rchi] Configuration validated successfully
[a2rchi] You are using an embedding model from HuggingFace; make sure to include a HuggingFace token if required for usage, it won't be explicitly enforced
[a2rchi] Required secrets validated: PG_PASSWORD
[a2rchi] Volume 'a2rchi-pg-my-a2rchi' already exists. No action needed.
[a2rchi] Volume 'a2rchi-my-a2rchi' already exists. No action needed.
[a2rchi] Starting compose deployment from /path/to/my/.a2rchi/a2rchi-my-a2rchi
[a2rchi] Using compose file: /path/to/my/.a2rchi/a2rchi-my-a2rchi/compose.yaml
[a2rchi] (This might take a minute...)
[a2rchi] Deployment started successfully
A2RCHI deployment 'my-a2rchi' created successfully!
Services running: chatbot, postgres, chromadb
[a2rchi] Chatbot: http://localhost:7861
```

</details>

The first deployment builds the container images from scratch (which may take a few minutes). Subsequent deployments reuse the images and complete much faster (roughly a minute).

> **Tip:** Having issues? Run the command with `-v 4` to enable DEBUG-level logging.

### A note about multiple configurations

When multiple configuration files are passed, their `services` sections must remain consistent, otherwise the deployment fails. The current use cases for multiple configurations include swapping pipelines/prompts dynamically via the chat app and maintaining separate benchmarking configurations.

### Verifying a deployment

List running deployments with:

```bash
a2rchi list-deployments
```

You should see output similar to:

```text
Existing deployments:
```

(Additional details will follow for each deployment.)
