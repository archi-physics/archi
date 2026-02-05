# Developers Guide

Below is all the information developers may need to get started contributing to the Archi project.

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
export Archi_DIR=~/.archi
export DEPLOYMENT_NAME=local-smoke
export USE_PODMAN=true
export SMOKE_FORCE_CREATE=true
export SMOKE_OLLAMA_MODEL=gpt-oss:latest
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
export USE_PODMAN=true
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

- Python: <https://hub.docker.com/r/archi/archi-python-base>
- PyTorch: <https://hub.docker.com/r/archi/archi-pytorch-base>

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
podman build -t archi/<image-name>:<tag> .
```

After verifying the image, log in to Docker Hub (ask a senior developer for credentials):

```bash
podman login docker.io
```

Push the image:

```bash
podman push archi/<image-name>:<tag>
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

## Extending the stack

When integrating a new source, create a collector under `data_manager/collectors`. Collectors should yield `Resource` objects. A new `Resource` subclass is only needed if the content type is not already represented (e.g., text, HTML, markdown, images, etc.), but it must implement the required methods described above.

When integrating a new collector, ensure that any per-source configuration is encoded in the resource metadata so downstream consumers—such as the chat app—can honour it.

When extending the embedding pipeline or storage schema, keep this flow in mind: collectors produce resources → `PersistenceService` writes files and updates the Postgres catalog → `VectorStoreManager` indexes embeddings in PostgreSQL. Keeping responsibilities narrowly scoped makes the ingestion stack easier to reason about and evolve.
