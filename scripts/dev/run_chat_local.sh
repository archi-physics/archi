#!/bin/bash
# Run the chat app locally against the test database
# 
# Prerequisites:
#   1. Test database running: cd tests/smoke && docker compose -f docker-compose.integration.yaml up -d
#   2. Python environment with dependencies installed
#
# Usage:
#   ./scripts/dev/run_chat_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "📦 Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    echo "📦 Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Configuration directory
CONFIG_DIR="${PROJECT_ROOT}/tests/smoke/local_dev_config"
mkdir -p "$CONFIG_DIR"

# Create config file if it doesn't exist
CONFIG_FILE="$CONFIG_DIR/local-dev.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    cat > "$CONFIG_FILE" << 'EOF'
# Local development config for running chat app against test database
name: local-dev

global:
  DATA_PATH: /tmp/archi-data/
  ACCOUNTS_PATH: /tmp/archi-accounts/
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
    pipeline: CMSCompOpsAgent
    # These will be filled in by the script
    template_folder: "PLACEHOLDER"
    static_folder: "PLACEHOLDER"
    num_responses_until_feedback: 5
    auth:
      enabled: false

data_manager:
  collection_name: local_dev
  embedding_name: all-MiniLM-L6-v2
  embedding_dimensions: 384
  chunk_size: 1000
  chunk_overlap: 150
  distance_metric: cosine
  embedding_class_map:
    all-MiniLM-L6-v2:
      class: HuggingFaceEmbeddings
      kwargs:
        model_name: all-MiniLM-L6-v2
      similarity_score_reference: 0.45
  sources: {}

archi:
  agent_description: "Local development test agent"
  providers:
    local:
      enabled: true
      base_url: http://localhost:11434
      mode: ollama
      default_model: "qwen3:4b"
      models:
        - "qwen3:4b"
        - "qwen2.5-coder:3b"
  pipelines:
    - CMSCompOpsAgent
  pipeline_map:
    CMSCompOpsAgent:
      recursion_limit: 100
      prompts:
        required:
          agent_prompt: examples/deployments/basic-ollama/agent.prompt
      models:
        required:
          agent_model: local/qwen3:4b
EOF
    echo "Created config file: $CONFIG_FILE"
fi

# Update template/static paths in config
sed -i.bak "s|template_folder:.*|template_folder: \"${PROJECT_ROOT}/src/interfaces/chat_app/templates\"|" "$CONFIG_FILE"
sed -i.bak "s|static_folder:.*|static_folder: \"${PROJECT_ROOT}/src/interfaces/chat_app/static\"|" "$CONFIG_FILE"
rm -f "$CONFIG_FILE.bak"

# Create directories
mkdir -p /tmp/archi-data /tmp/archi-accounts

# Check if test database is running
if ! docker ps --format "{{.Names}}" | grep -q "archi-test-postgres"; then
    echo "❌ Test database not running!"
    echo "   Start it with:"
    echo "   cd tests/smoke && docker compose -f docker-compose.integration.yaml up -d"
    exit 1
fi
echo "✅ Test database is running"

# Set environment variables for PostgresServiceFactory.from_env()
export ARCHI_CONFIGS_PATH="$CONFIG_DIR/"
export PG_PASSWORD=testpassword123
export PGHOST=localhost
export PGPORT=5439
export PGDATABASE=archi
export PGUSER=archi

# Check for optional API keys
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "✅ OPENAI_API_KEY is set"
else
    echo "⚠️  OPENAI_API_KEY not set - LLM features won't work"
fi

echo ""
echo "Starting chat app..."
echo "  Config: $CONFIG_FILE"
echo "  Database: localhost:5439"
echo "  Chat UI: http://localhost:2786"
echo "  Python: $(which python)"
echo ""

# Bootstrap config into Postgres (the service expects config in DB, not YAML)
echo "📝 Bootstrapping config into Postgres..."
python -c "
import json
import yaml
import psycopg2
import psycopg2.extras
import os

# Load YAML config
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)

# Extract nested config values
data_manager = config.get('data_manager', {})
services = config.get('services', {})
chat_app = services.get('chat_app', {})
auth = chat_app.get('auth', {})

# Connect to DB
conn = psycopg2.connect(
    host=os.environ['PGHOST'],
    port=os.environ['PGPORT'],
    database=os.environ['PGDATABASE'],
    user=os.environ['PGUSER'],
    password=os.environ['PG_PASSWORD']
)

with conn.cursor() as cur:
    # Use INSERT ON CONFLICT to upsert the config
    cur.execute('''
        INSERT INTO static_config (
            id, deployment_name, embedding_model, embedding_dimensions,
            chunk_size, chunk_overlap, auth_enabled,
            services_config, data_manager_config,
            archi_config, global_config, sources_config
        ) VALUES (
            1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            deployment_name = EXCLUDED.deployment_name,
            embedding_model = EXCLUDED.embedding_model,
            embedding_dimensions = EXCLUDED.embedding_dimensions,
            chunk_size = EXCLUDED.chunk_size,
            chunk_overlap = EXCLUDED.chunk_overlap,
            auth_enabled = EXCLUDED.auth_enabled,
            services_config = EXCLUDED.services_config,
            data_manager_config = EXCLUDED.data_manager_config,
            archi_config = EXCLUDED.archi_config,
            global_config = EXCLUDED.global_config,
            sources_config = EXCLUDED.sources_config
    ''', (
        config.get('name', 'local-dev'),
        data_manager.get('embedding_name', 'all-MiniLM-L6-v2'),
        data_manager.get('embedding_dimensions', 384),
        data_manager.get('chunk_size', 1000),
        data_manager.get('chunk_overlap', 150),
        auth.get('enabled', False),
        psycopg2.extras.Json(services),
        psycopg2.extras.Json(data_manager),
        psycopg2.extras.Json(config.get('archi', {})),
        psycopg2.extras.Json(config.get('global', {})),
        psycopg2.extras.Json(config.get('sources', {}))
    ))
    conn.commit()
conn.close()
print('✅ Config loaded into Postgres')
"

# Run the chat service
cd "$PROJECT_ROOT"
python src/bin/service_chat.py
