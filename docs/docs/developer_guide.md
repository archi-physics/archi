# Developers Guide

Below is all the information developers may need to get started contributing to the Archi project.

## Architecture Overview

Archi is a containerized RAG framework with these core components:

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (archi create)                     │
│  Renders configs, builds images, launches containers     │
└─────────────┬───────────────────────────────┬───────────┘
              │                               │
   ┌──────────▼──────────┐       ┌────────────▼──────────┐
   │   Chat Service      │       │   Data Manager        │
   │   (Flask, port 7861)│       │   (Flask, port 7871)  │
   │   - ReAct agent     │       │   - Collectors        │
   │   - Provider layer  │       │   - PersistenceService│
   │   - Auth, BYOK      │       │   - VectorStoreManager│
   └──────────┬──────────┘       └────────────┬──────────┘
              │                               │
   ┌──────────▼───────────────────────────────▼──────────┐
   │              PostgreSQL (pgvector)                    │
   │  Tables: resources, document_chunks, conversations,  │
   │          conversation_metadata, feedback, timing,     │
   │          agent_tool_calls, users, keys                │
   └──────────────────────────────────────────────────────┘
```

Key source directories:

| Directory | Purpose |
|-----------|---------|
| `src/archi/` | Core orchestration — agent base class, pipelines, providers |
| `src/bin/` | Service entrypoints (Flask app factories) |
| `src/cli/` | CLI commands, service/source registries, deployment manager |
| `src/data_manager/` | Collectors, persistence, vectorstore indexing |
| `src/interfaces/` | Flask blueprints for chat, uploader, data viewer |
| `src/utils/` | Config loader, logging, shared utilities |

## Provider Architecture

All LLM providers extend `BaseProvider` (`src/archi/providers/base_provider.py`):

```python
class BaseProvider(ABC):
    @abstractmethod
    def get_model(self, model_name, **kwargs) -> BaseChatModel: ...
    @abstractmethod
    def get_embedding_model(self, model_name, **kwargs) -> Embeddings: ...
    @abstractmethod
    def list_models(self) -> list[str]: ...
```

Providers register themselves via `ProviderType` enum and are registered through `register_provider()`. Factory functions in `src/archi/providers/__init__.py`:

- `get_provider(provider_type)` — returns a provider instance
- `get_model(provider_type, model_name)` — returns a LangChain `BaseChatModel`
- `get_provider_with_api_key(provider_type, api_key)` — for BYOK

Built-in providers:

| Provider | Module | Models |
|----------|--------|--------|
| `OpenAIProvider` | `openai_provider.py` | gpt-4o, gpt-4o-mini, etc. |
| `AnthropicProvider` | `anthropic_provider.py` | claude-sonnet-4-20250514, claude-3.5-haiku, etc. |
| `GeminiProvider` | `gemini_provider.py` | gemini-2.0-flash, gemini-2.5-pro, etc. |
| `OpenRouterProvider` | `openrouter_provider.py` | Any model via OpenRouter API |
| `LocalProvider` | `local_provider.py` | Ollama or OpenAI-compatible local models |

### Adding a New Provider

1. Create `src/archi/providers/my_provider.py` extending `BaseProvider`.
2. Add an entry to the `ProviderType` enum.
3. Call `register_provider(ProviderType.MY_PROVIDER, MyProvider)` to register it.
4. Implement `get_model()`, `get_embedding_model()`, and `list_models()`.
5. Add config keys under `services.chat_app.providers.my_provider`.

## Agent & Pipeline Architecture

The agent system is built around `BaseReActAgent` (`src/archi/archi.py`, ~975 lines):

- Implements a ReAct (Reasoning + Acting) loop with tool calling
- Manages conversation history, streaming, and tool execution
- Loads agent specs from markdown files with YAML frontmatter

`AgentSpec` (`src/archi/pipelines/agents/agent_spec.py`) is a dataclass:

```python
@dataclass
class AgentSpec:
    name: str
    tools: list[str]
    prompt: str
    source_path: str
```

Agent specs are discovered via `list_agent_files()` and loaded via `load_agent_spec()`. The `select_agent_spec()` function picks the correct spec given a name.

### Pipeline Classes

- `CMSCompOpsAgent` — default ReAct agent with 6 built-in tools (search_local_files, search_metadata_index, list_metadata_schema, fetch_catalog_document, search_vectorstore_hybrid, mcp)
- `QAPipeline` — simpler retrieval-augmented QA without tool calling

### Adding a New Tool

1. Define the tool function in the agent class or as a LangChain `@tool`.
2. Register it in the tool mapping within the agent's `_build_tools()` method.
3. Add the tool name to agent spec YAML frontmatter `tools` list.

## Contribution Workflow

1. **Branch**: Create a feature branch from `main` (e.g., `dev/my-feature`).
2. **Develop**: Follow PEP 8, use `snake_case` for functions, `PascalCase` for classes.
3. **Test**: Run smoke tests locally (see below).
4. **Commit**: Use short, lowercase commit summaries (e.g., `add gemini provider`).
5. **PR**: Include a brief summary, test results, and documentation impact. Link related issues.

## Editing Documentation

Editing documentation requires the `mkdocs` Python package:

```bash
pip install mkdocs
```

To edit documentation, update the `.md` and `.yml` files in the `./docs` folder. To preview changes locally, run:

```bash
cd docs
mkdocs serve
```

Add the `-a IP:HOST` argument (default is `localhost:8000`) to specify the host and port.

Publish your changes with:

```bash
mkdocs gh-deploy
```

Always open a PR to merge documentation changes into `main`. Do not edit files directly in the `gh-pages` branch.

## Smoke Tests

If you want the full CI-like smoke run (create deployment, wait for readiness, run checks, and clean up) you can use the shared runner:

```bash
export ARCHI_DIR=~/.archi
export DEPLOYMENT_NAME=local-smoke
export USE_PODMAN=false
export SMOKE_FORCE_CREATE=true
export SMOKE_OLLAMA_MODEL=qwen3:4b
scripts/dev/run_smoke_preview.sh "${DEPLOYMENT_NAME}"
```

The shared runner performs these checks in order (ensuring the configured Ollama model is available via `ollama pull` before running the checks):

- Create a deployment from the preview config and wait for the chat app health endpoint.
- Wait for initial data ingestion to complete (5 minute timeout).
- Preflight checks: Postgres reachable, data-manager catalog searchable.
- Tool probes: catalog tools and vectorstore retriever (executed inside the chatbot container to match the agent runtime).
- ReAct agent smoke: stream response and observe at least one tool call.

The combined smoke workflow alone does not start Archi for you. Start a deployment first, then run the checks (it validates Postgres, data-manager catalog, Ollama model availability, ReAct streaming, and direct tool probes inside the chatbot container):

```bash
export Archi_CONFIG_PATH=~/.archi/archi-<deployment-name>/configs/<config-name>.yaml
export Archi_CONFIG_NAME=<config-name>
export Archi_PIPELINE_NAME=CMSCompOpsAgent
export USE_PODMAN=false
export OLLAMA_MODEL=<ollama-model-name>
export PGHOST=localhost
export PGPORT=<postgres-port>
export PGUSER=archi
export PGPASSWORD=<pg-password>
export PGDATABASE=archi-db
export BASE_URL=http://localhost:2786
export DM_BASE_URL=http://localhost:<data-manager-port>  # from your deployment config
export OLLAMA_URL=http://localhost:11434
./tests/smoke/combined_smoke.sh <deployment-name>
```

Optional environment variables for deterministic queries:

```bash
export REACT_SMOKE_PROMPT="Use the search_local_files tool to find ... and summarize."
export FILE_SEARCH_QUERY="first linux server installation"
export METADATA_SEARCH_QUERY="ppc.mit.edu"
export VECTORSTORE_QUERY="cms"
```

## CI / CD Architecture

All CI workflows run on GitHub-hosted `ubuntu-latest` runners with Docker (not Podman).

### PR Preview (`pr-preview.yml`)

Every pull request triggers four parallel/sequential jobs:

| Job | Runner | Purpose |
|-----|--------|---------|
| **lint** | `ubuntu-latest` | `black --check .` and `isort --check .` |
| **unit-tests** | `ubuntu-latest` | `pytest tests/unit/ -v --tb=short` |
| **build-base-images** | `ubuntu-latest` | Detects changes to base image inputs; builds if needed |
| **preview** | `ubuntu-latest` | Smoke deployment + Playwright UI tests |

The `preview` job:

1. Installs Ollama and pulls `qwen3:4b` (~2.6GB).
2. Builds and deploys the app via `archi create` with Docker.
3. Runs integration smoke tests (`combined_smoke.sh`).
4. Runs Playwright UI tests against `http://localhost:2786`.

### Release (`test-and-build-tag.yml`)

Manually dispatched; builds Docker base images, pushes to DockerHub, runs smoke tests, then tags and releases.

### Publish Base Images (`publish-base-images.yml`)

Triggered on push to `main`; rebuilds and pushes base images when requirements or Dockerfiles change.

### Docker Layer Caching

All workflows that build Docker images use `docker/setup-buildx-action` with `actions/cache` for layer caching, reducing rebuild times on cache hits.

### Local Development

For local smoke testing, Docker is the default container runtime (`USE_PODMAN=false`). To use Podman locally, set `USE_PODMAN=true` and use the `--podman` flag with `archi` CLI commands.

## Postgres Usage Overview

Archi relies on Postgres as the durable metadata store across services. Core usage falls into two buckets:

- **Ingestion catalog**: the `resources` table tracks persisted files and metadata for the data manager catalog (`CatalogService`).
- **Conversation history**: the `conversation_metadata` and `conversations` tables store chat/session metadata plus message history for interfaces like the chat app and ticketing integrations (e.g., Redmine mailer).

The `conversations` table tracks:
- `model_used` (string) - The model that generated the response (e.g., "openai/gpt-4o")
- `pipeline_used` (string) - The pipeline that processed the request (e.g., "QAPipeline")

Additional supporting tables store interaction telemetry and feedback:

- `feedback` captures like/dislike/comment feedback tied to conversation messages.
- `timing` tracks per-message latency milestones.
- `agent_tool_calls` stores tool call inputs/outputs extracted from agent messages.

When extending an interface that writes to `conversations`, make sure a matching `conversation_metadata` row exists (create or update before inserting messages) to satisfy foreign key constraints.

## DockerHub Images

Archi loads different base images hosted on Docker Hub. The Python base image is used when GPUs are not required; otherwise the PyTorch base image is used. The Dockerfiles for these base images live in `src/cli/templates/dockerfiles/base-X-image`.

Images are hosted at:

- Python: <https://hub.docker.com/r/a2rchi/a2rchi-python-base>
- PyTorch: <https://hub.docker.com/r/a2rchi/a2rchi-pytorch-base>

To rebuild a base image, navigate to the relevant `base-xxx-image` directory under `src/cli/templates/dockerfiles/`. Each directory contains the Dockerfile, requirements, and license information.

Regenerate the requirements files with:

```bash
# Python image
cat requirements/cpu-requirementsHEADER.txt requirements/requirements-base.txt > src/cli/templates/dockerfiles/base-python-image/requirements.txt

# PyTorch image
cat requirements/gpu-requirementsHEADER.txt requirements/requirements-base.txt > src/cli/templates/dockerfiles/base-pytorch-image/requirements.txt
```

Build the image:

```bash
podman build -t a2rchi/<image-name>:<tag> .
```

After verifying the image, log in to Docker Hub (ask a senior developer for credentials):

```bash
podman login docker.io
```

Push the image:

```bash
podman push a2rchi/<image-name>:<tag>
```

## Data Ingestion Architecture

Archi ingests content through **sources** which are collected by **collectors** (`data_manager/collectors`).
These documents are written to persistent, local files via the `PersistenceService`, which uses `Resource` objects as an abstraction for different content types, and `ResourceMetadata` for associated metadata.
A catalog of persisted files and metadata is maintained in Postgres via
`CatalogService` (table: `resources`).
Finally, the `VectorStoreManager` reads these files, splits them into chunks, generates embeddings, and indexes them in PostgreSQL with pgvector.

### Resources and `BaseResource`

Every collected artifact from the collectors is represented as a subclass of `BaseResource` (`src/data_manager/collectors/resource_base.py`). Subclasses must implement:

- `get_hash()`: a stable identifier used as the key in the filesystem catalog.
- `get_filename()`: the on-disk file name (including extension).
- `get_content()`: returns the textual or binary payload that should be persisted.

Resources may optionally override:

- `get_metadata()`: returns a metadata object (typically `ResourceMetadata`) describing the item. Keys should be serialisable strings and are flattened into the vector store metadata.
- `get_metadata_path()`: legacy helper for `.meta.yaml` paths (metadata is now stored in Postgres).

`ResourceMetadata` (`src/data_manager/collectors/utils/metadata.py`) enforces a required `file_name` and normalises the `extra` dictionary so all values become strings. Optional UI labels like `display_name` live in `extra`, alongside source-specific information such as URLs, ticket identifiers, or visibility flags.

The guiding philosophy is that **resources describe content**, but never write to disk themselves. This separation keeps collectors simple, testable, and ensures consistent validation when persisting different resource types.

### Persistence Service

`PersistenceService` (`src/data_manager/collectors/persistence.py`) centralises all filesystem writes for document content and metadata catalog updates. When `persist_resource()` is called it:

1. Resolves the target path under the configured `DATA_PATH`.
2. Validates and writes the resource content (rejecting empty payloads or unknown types).
3. Normalises metadata (if provided) for storage.
4. Upserts a row into the Postgres `resources` catalog with file and metadata fields.

Collectors only interact with `PersistenceService`; they should not touch the filesystem directly.

### Vector Database

The vector store lives under the `data_manager/vectorstore` package. `VectorStoreManager` reads the Postgres catalog and manages embeddings in PostgreSQL:

1. Loads the tracked files and metadata hashes from the Postgres catalog.
2. Splits documents into chunks, optional stemming, and builds embeddings via the configured model.
3. Adds chunks to the document_chunks table with embeddings and flattened metadata (including resource hash, filename, human-readable display fields, and any source-specific extras).
4. Deletes stale entries when the underlying files disappear or are superseded.

Because the manager defers to the catalog, any resource persisted through `PersistenceService` automatically becomes eligible for indexing—no extra plumbing is required.

### Catalog Verification Checklist

- Confirm the Postgres `resources` table exists and is reachable from the service containers.
- Ingest or upload a new document and verify a new row appears in `resources`.
- Verify `VectorStoreManager` can update the collection using the Postgres catalog.

## Extending the Stack

### Adding a New Data Source

When integrating a new source, create a collector under `data_manager/collectors`. Collectors should yield `Resource` objects. A new `Resource` subclass is only needed if the content type is not already represented (e.g., text, HTML, markdown, images, etc.), but it must implement the required methods described above.

When integrating a new collector, ensure that any per-source configuration is encoded in the resource metadata so downstream consumers—such as the chat app—can honour it.

### Adding a New Service

1. Create a Flask blueprint under `src/interfaces/`.
2. Register the service in `src/cli/service_registry.py` with its name, port, and dependencies.
3. Add a service entrypoint in `src/bin/`.
4. Add any service-specific config keys under `services.<name>` in the base config template.

### Extending Embeddings or Storage

When extending the embedding pipeline or storage schema, keep this flow in mind: collectors produce resources → `PersistenceService` writes files and updates the Postgres catalog → `VectorStoreManager` indexes embeddings in PostgreSQL. Keeping responsibilities narrowly scoped makes the ingestion stack easier to reason about and evolve.
