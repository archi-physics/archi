# Local Development Testing Guide

This guide explains how to run Archi services for local testing and development.

## Architecture Overview

Archi is designed to run as a **containerized deployment** via Docker or Podman. The CLI (`archi`) handles:

1. Configuration merging (your config + base-config.yaml template)
2. Secrets management (from `.env` files → individual secret files)
3. Docker Compose generation and orchestration
4. Volume management for data persistence

**Important**: Running services directly with `python src/bin/service_chat.py` requires careful configuration setup (see "Direct Service Execution" below).

## Quick Start with Local Dev Script

For quick local testing against the integration test database:

```bash
# 1. Start the test database
cd tests/smoke
docker compose -f docker-compose.integration.yaml up -d
cd ../..

# 2. Run the chat app
./scripts/dev/run_chat_local.sh

# 3. Open http://localhost:2786 in your browser

# 4. When done, stop the database
cd tests/smoke
docker compose -f docker-compose.integration.yaml down
```

The local dev script automatically:
- Activates the `.venv` virtual environment
- Sets up the test database connection (localhost:5439)
- Creates a minimal configuration file
- Uses DumbLLM for testing (no API keys required)

## CLI Deployment (Recommended for Production-like Testing)

### 1. Create a deployment using the CLI

```bash
# Install Archi in editable mode
pip install -e .

# Create a deployment with the basic-ollama example
archi create \
  --name test-local \
  --config examples/deployments/basic-ollama/config.yaml \
  --env-file examples/deployments/basic-ollama/secrets.env \
  --services chatbot \
  --hostmode

# This creates ~/.archi/archi-test-local/ with:
#   - configs/           Rendered configuration files
#   - secrets/           Individual secret files
#   - docker-compose.yml Generated compose file
```

### 2. Access the services

After deployment, the chatbot is available at the configured port (default: http://localhost:7868).

### 3. Delete when done

```bash
archi delete --name test-local
```

## Configuration Structure

### Config Files

Configs are YAML files that override the base template (`src/cli/templates/base-config.yaml`). Key sections:

```yaml
name: my_deployment

global:
  DATA_PATH: /root/archi/output/     # Inside container
  ACCOUNTS_PATH: /root/archi/accounts/
  verbosity: 2                        # 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE

services:
  postgres:
    host: postgres                    # Service name in docker network
    port: 5432
    database: archi
    user: archi
  chat_app:
    host: "0.0.0.0"
    port: 2786
    external_port: 2786
    template_folder: /root/archi/src/interfaces/chat_app/templates
    static_folder: /root/archi/src/interfaces/chat_app/static
    auth:
      enabled: false
```

### Secrets (.env file)

Secrets are loaded from a `.env` file and written as individual files to `secrets/`:

```env
PG_PASSWORD=your-postgres-password
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FLASK_CHAT_APP_SECRET_KEY=random-secret-key
BYOK_ENCRYPTION_KEY=32-character-encryption-key
```

The `read_secret()` function looks for:
1. `{SECRET_NAME}_FILE` env var pointing to a file
2. `{SECRET_NAME}` env var directly
3. Returns empty string if not found

## Running Integration Tests

For testing the PostgreSQL services without full deployment:

```bash
# Start the test database
cd tests/smoke
docker compose -f docker-compose.integration.yaml up -d

# Run integration tests
python -m pytest tests/smoke/test_integration.py -v

# Stop when done
docker compose -f docker-compose.integration.yaml down
```

The test database runs on port 5439 with credentials:
- Host: localhost
- Port: 5439
- Database: archi
- User: archi
- Password: testpassword123

## Direct Service Execution (Advanced)

If you need to run services directly (not recommended for normal use):

### Required Environment Variables

```bash
export ARCHI_CONFIGS_PATH=/path/to/configs/directory/  # Must end with /
export PG_PASSWORD=your-password
export OPENAI_API_KEY=your-key
# ... other secrets as direct env vars
```

### Required Config Keys

The config file must include all keys from `base-config.yaml`:

```yaml
global:
  DATA_PATH: /tmp/archi-data
  ACCOUNTS_PATH: /tmp/archi-accounts
  verbosity: 2

services:
  postgres:
    host: localhost
    port: 5439
    database: archi
    user: archi
  chat_app:
    host: "0.0.0.0"
    hostname: localhost
    port: 2786
    external_port: 2786
    template_folder: /absolute/path/to/templates
    static_folder: /absolute/path/to/static
    num_responses_until_feedback: 5
    auth:
      enabled: false

archi:
  agent_description: "Test agent"
  pipelines: [...]
  pipeline_map: {...}
  model_class_map: {...}
```

### Why Direct Execution is Complicated

1. **Config merging**: The CLI merges your config with `base-config.yaml` Jinja template
2. **Path resolution**: Templates use `/root/archi/` paths (container paths)
3. **Secret injection**: Secrets are written as files, not env vars
4. **Service dependencies**: Services expect PostgreSQL, vectorstore, etc. to be available

## Smoke Test Infrastructure

The test database (`docker-compose.integration.yaml`) provides:

- PostgreSQL 17 with pgvector extension
- Pre-created schema from `init-test.sql`
- Health checks for readiness

This is sufficient for testing:
- UserService, ConversationService
- PostgresCatalogService (document ingestion)
- DataViewerService
- Vector similarity search
- Connection pooling

## File Locations

| Component | Container Path | Purpose |
|-----------|---------------|---------|
| Configs | `/root/archi/configs/` | YAML configuration files |
| Secrets | `/root/archi/secrets/` | Individual secret files |
| Data | `/root/archi/output/` | Ingested documents, indexes |
| Templates | `/root/archi/src/interfaces/chat_app/templates/` | HTML templates |
| Static | `/root/archi/src/interfaces/chat_app/static/` | JS, CSS, images |
