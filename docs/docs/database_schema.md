# Archi Database Schema

> PostgreSQL schema v2.0 — Consolidated from PostgreSQL (conversations), ChromaDB (vectors), and SQLite (catalog)  
> **Requires:** PostgreSQL 17+ with `pgvector`, `pgcrypto`, `pg_trgm` (and optionally `pg_textsearch` for BM25)

---

## Overview

The Archi database is organized into these major areas:

| Area | Tables | Purpose |
|------|--------|---------|
| **Authentication** | `users`, `sessions` | User identity, auth providers, preferences |
| **Configuration** | `static_config`, `dynamic_config`, `config_audit` | Deploy-time and runtime settings |
| **Documents** | `documents`, `document_chunks` | Vector store with embeddings |
| **Document Selection** | `user_document_defaults`, `conversation_document_overrides` | 3-tier document filtering |
| **Conversations** | `conversation_metadata`, `conversations`, `feedback`, `timing` | Chat history and feedback |
| **Agent Tracing** | `agent_traces`, `agent_tool_calls` | Tool call logging |
| **A/B Testing** | `ab_comparisons` | Model comparison tracking |
| **Migrations** | `migration_state` | Resumable migration checkpoints |
| **Legacy** | `configs` | Backward compatibility (to be removed) |

---

## Configuration Management

Archi uses a **two-tier configuration system** separating immutable deploy-time settings from runtime-adjustable parameters.

### Static Configuration (`static_config`)

Settings that cannot change without redeployment (affects vector dimensions, paths, etc.).

```sql
CREATE TABLE static_config (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Single row enforced
    
    -- Deployment identity
    deployment_name VARCHAR(100) NOT NULL,
    config_version VARCHAR(20) NOT NULL DEFAULT '2.0.0',
    
    -- Paths
    data_path TEXT NOT NULL DEFAULT '/root/data/',
    prompts_path TEXT NOT NULL DEFAULT '/root/archi/data/prompts/',
    
    -- Embedding configuration (affects vector dimensions)
    embedding_model VARCHAR(200) NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL DEFAULT 1000,
    chunk_overlap INTEGER NOT NULL DEFAULT 150,
    distance_metric VARCHAR(20) NOT NULL DEFAULT 'cosine',
    
    -- Available options
    available_pipelines TEXT[] NOT NULL DEFAULT '{}',
    available_models TEXT[] NOT NULL DEFAULT '{}',
    available_providers TEXT[] NOT NULL DEFAULT '{}',
    
    -- Auth configuration
    auth_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    session_lifetime_days INTEGER NOT NULL DEFAULT 30,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

**Key Fields:**

| Field | Description |
|-------|-------------|
| `deployment_name` | Human-readable name for this instance |
| `embedding_model` | Model used for vector embeddings (e.g., `all-MiniLM-L6-v2`) |
| `embedding_dimensions` | Vector dimension (must match `document_chunks.embedding`) |
| `chunk_size` / `chunk_overlap` | Document chunking parameters |
| `available_*` | Arrays of what's installed and available to use |

---

### Dynamic Configuration (`dynamic_config`)

Runtime-adjustable settings for model selection, retrieval, and prompts.

```sql
CREATE TABLE dynamic_config (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Single row enforced
    
    -- Model settings
    active_pipeline VARCHAR(100) NOT NULL DEFAULT 'QAPipeline',
    active_model VARCHAR(200) NOT NULL DEFAULT 'openai/gpt-4o',
    temperature NUMERIC(3,2) NOT NULL DEFAULT 0.7,
    max_tokens INTEGER NOT NULL DEFAULT 4096,
    system_prompt TEXT,  -- NULL = use pipeline default
    
    -- Additional generation params
    top_p NUMERIC(3,2) NOT NULL DEFAULT 0.9,
    top_k INTEGER NOT NULL DEFAULT 50,
    repetition_penalty NUMERIC(4,2) NOT NULL DEFAULT 1.0,
    
    -- Prompt selection (file names without extension)
    active_condense_prompt VARCHAR(100) NOT NULL DEFAULT 'default',
    active_chat_prompt VARCHAR(100) NOT NULL DEFAULT 'default',
    active_system_prompt VARCHAR(100) NOT NULL DEFAULT 'default',
    
    -- Retrieval settings
    num_documents_to_retrieve INTEGER NOT NULL DEFAULT 10,
    use_hybrid_search BOOLEAN NOT NULL DEFAULT TRUE,
    bm25_weight NUMERIC(3,2) NOT NULL DEFAULT 0.3,
    semantic_weight NUMERIC(3,2) NOT NULL DEFAULT 0.7,
    
    -- BM25 parameters
    bm25_k1 NUMERIC(4,2) NOT NULL DEFAULT 1.2,
    bm25_b NUMERIC(3,2) NOT NULL DEFAULT 0.75,
    
    -- Schedules
    ingestion_schedule VARCHAR(100) NOT NULL DEFAULT '',  -- Cron expression
    
    -- Logging
    verbosity INTEGER NOT NULL DEFAULT 3,
    
    -- Metadata
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(200)  -- user_id who made the change
);
```

**Key Field Categories:**

| Category | Fields | Description |
|----------|--------|-------------|
| **Model** | `active_pipeline`, `active_model`, `temperature`, `max_tokens` | Which model/pipeline to use |
| **Generation** | `top_p`, `top_k`, `repetition_penalty` | Fine-tuning generation behavior |
| **Prompts** | `active_*_prompt` | Which prompt files to use |
| **Retrieval** | `num_documents_to_retrieve`, `use_hybrid_search`, `*_weight` | RAG retrieval tuning |
| **BM25** | `bm25_k1`, `bm25_b` | BM25 algorithm parameters |

---

### Config Audit Log (`config_audit`)

Tracks all configuration changes for compliance and debugging.

```sql
CREATE TABLE config_audit (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(200) NOT NULL,
    changed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    config_type VARCHAR(20) NOT NULL,  -- 'dynamic', 'user_pref'
    field_name VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT
);
```

**Indexes:**
- `idx_config_audit_user` — Find changes by user
- `idx_config_audit_time` — Recent changes (DESC order)

---

## User Preferences

Users can override global settings. Preferences are stored directly on the `users` table.

```sql
-- User preference columns on users table
preferred_model VARCHAR(200),          -- Override active_model
preferred_temperature NUMERIC(3,2),    -- Override temperature
preferred_max_tokens INTEGER,          -- Override max_tokens
preferred_num_documents INTEGER,       -- Override retrieval count
preferred_condense_prompt VARCHAR(100),
preferred_chat_prompt VARCHAR(100),
preferred_system_prompt VARCHAR(100),
preferred_top_p NUMERIC(3,2),
preferred_top_k INTEGER,
```

### Configuration Resolution Order

When processing a request, settings resolve in this order:

1. **Conversation override** (if present)
2. **User preference** (if set)
3. **Dynamic config** (global default)
4. **Static config** (deploy-time defaults)

---

## Users & Authentication

### Users Table (`users`)

```sql
CREATE TABLE users (
    id VARCHAR(200) PRIMARY KEY,  -- From auth provider or generated
    
    -- Identity
    display_name TEXT,
    email TEXT,
    auth_provider VARCHAR(50) NOT NULL DEFAULT 'anonymous',
    
    -- Local auth
    password_hash VARCHAR(256),           -- Werkzeug pbkdf2
    
    -- GitHub OAuth
    github_id VARCHAR(100),
    github_username VARCHAR(100),
    
    -- Role
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Preferences (see above)
    theme VARCHAR(20) NOT NULL DEFAULT 'system',
    preferred_* ...
    
    -- BYOK API keys (encrypted)
    api_key_openrouter BYTEA,
    api_key_openai BYTEA,
    api_key_anthropic BYTEA,
    
    -- Tracking
    last_login_at TIMESTAMP,
    login_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

**Auth Providers:**
- `anonymous` — No auth, identified by client_id
- `local` — Username/password stored in `password_hash`
- `github` — GitHub OAuth, uses `github_id` and `github_username`

**BYOK Keys:**
- Encrypted using `pgp_sym_encrypt(key, encryption_key)`
- Encryption key from `BYOK_ENCRYPTION_KEY` environment variable

### Sessions Table (`sessions`)

```sql
CREATE TABLE sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(200) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    data JSONB,                    -- Additional session data
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);
```

---

## Documents & Vectors

### Documents Table (`documents`)

Replaces the old SQLite catalog. Stores document metadata.

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    resource_hash VARCHAR(64) UNIQUE NOT NULL,
    
    -- Location
    file_path TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- 'local_files', 'web', 'ticket', 'git'
    
    -- Source-specific
    url TEXT,                    -- For web sources
    ticket_id VARCHAR(100),      -- For ticket sources
    git_repo VARCHAR(200),       -- For git sources
    git_commit VARCHAR(64),
    
    -- File metadata
    suffix VARCHAR(20),
    size_bytes BIGINT,
    mime_type VARCHAR(100),
    
    -- Provenance
    original_path TEXT,
    base_path TEXT,
    relative_path TEXT,
    
    -- Extensible metadata
    extra_json JSONB,
    extra_text TEXT,             -- Searchable
    
    -- Timestamps
    file_modified_at TIMESTAMP,
    ingested_at TIMESTAMP,
    indexed_at TIMESTAMP,        -- When embeddings created
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Soft delete
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMP
);
```

**Source Types:** `local_files`, `web`, `ticket`, `git`, `unknown`

### Document Chunks (`document_chunks`)

Replaces ChromaDB. Stores chunked text with vector embeddings.

```sql
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    
    chunk_text TEXT NOT NULL,
    embedding vector(384),       -- Dimension must match static_config
    
    -- Chunk position
    start_char INTEGER,
    end_char INTEGER,
    metadata JSONB,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    UNIQUE(document_id, chunk_index)
);
```

**Indexes:**
- `idx_chunks_embedding` — HNSW vector index for cosine similarity
- `idx_chunks_bm25` — BM25 full-text (if pg_textsearch available)
- `idx_chunks_fts` — GIN tsvector fallback (if pg_textsearch unavailable)

---

## Document Selection (3-Tier System)

Users can customize which documents are used for RAG.

### Tier 1: Global Default
All documents enabled unless soft-deleted.

### Tier 2: User Defaults (`user_document_defaults`)

Power users can disable documents globally for themselves.

```sql
CREATE TABLE user_document_defaults (
    user_id VARCHAR(200) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,  -- FALSE = opted out
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, document_id)
);
```

### Tier 3: Conversation Overrides (`conversation_document_overrides`)

Override user default for a specific conversation.

```sql
CREATE TABLE conversation_document_overrides (
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id),
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL,  -- Explicit override
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversation_id, document_id)
);
```

---

## Conversations

### Conversation Metadata (`conversation_metadata`)

```sql
CREATE TABLE conversation_metadata (
    conversation_id SERIAL PRIMARY KEY,
    user_id VARCHAR(200) REFERENCES users(id) ON DELETE SET NULL,
    client_id TEXT,              -- Legacy, for backward compatibility
    title TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMP NOT NULL DEFAULT NOW(),
    archi_version VARCHAR(50)
);
```

### Messages (`conversations`)

```sql
CREATE TABLE conversations (
    message_id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id),
    
    archi_service TEXT NOT NULL,
    sender TEXT NOT NULL,
    content TEXT NOT NULL,
    
    -- Model tracking (replaces conf_id join)
    model_used VARCHAR(200),
    pipeline_used VARCHAR(100),
    
    -- RAG context
    link TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    
    ts TIMESTAMP NOT NULL,
    
    -- Legacy (to be dropped)
    conf_id INTEGER REFERENCES configs(config_id)
);
```

### Feedback (`feedback`)

```sql
CREATE TABLE feedback (
    mid INTEGER NOT NULL REFERENCES conversations(message_id),
    feedback_ts TIMESTAMP NOT NULL,
    feedback TEXT NOT NULL,           -- 'like', 'dislike', 'comment'
    feedback_msg TEXT,                -- Optional text
    incorrect BOOLEAN,
    unhelpful BOOLEAN,
    inappropriate BOOLEAN,
    PRIMARY KEY (mid, feedback_ts)
);
```

### Timing (`timing`)

Detailed timing metrics for each message.

```sql
CREATE TABLE timing (
    mid INTEGER PRIMARY KEY REFERENCES conversations(message_id),
    client_sent_msg_ts TIMESTAMP NOT NULL,
    server_received_msg_ts TIMESTAMP NOT NULL,
    lock_acquisition_ts TIMESTAMP NOT NULL,
    vectorstore_update_ts TIMESTAMP NOT NULL,
    query_convo_history_ts TIMESTAMP NOT NULL,
    chain_finished_ts TIMESTAMP NOT NULL,
    archi_message_ts TIMESTAMP NOT NULL,
    insert_convo_ts TIMESTAMP NOT NULL,
    finish_call_ts TIMESTAMP NOT NULL,
    server_response_msg_ts TIMESTAMP NOT NULL,
    msg_duration INTERVAL NOT NULL
);
```

---

## Agent Tracing

### Agent Traces (`agent_traces`)

```sql
CREATE TABLE agent_traces (
    trace_id UUID PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id),
    message_id INTEGER REFERENCES conversations(message_id),
    user_message_id INTEGER REFERENCES conversations(message_id),
    
    config_id VARCHAR(100),
    pipeline_name VARCHAR(100) NOT NULL,
    events JSONB NOT NULL DEFAULT '[]',
    
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'running',  -- running, completed, cancelled, failed
    
    total_tool_calls INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    total_duration_ms INTEGER,
    
    cancelled_by VARCHAR(100),
    cancellation_reason TEXT,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### Tool Calls (`agent_tool_calls`)

```sql
CREATE TABLE agent_tool_calls (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id),
    message_id INTEGER NOT NULL REFERENCES conversations(message_id),
    
    step_number INTEGER NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    tool_args JSONB,
    tool_result TEXT,
    
    ts TIMESTAMP NOT NULL DEFAULT NOW()
);
```

---

## A/B Testing

### A/B Comparisons (`ab_comparisons`)

```sql
CREATE TABLE ab_comparisons (
    comparison_id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id),
    user_prompt_mid INTEGER NOT NULL REFERENCES conversations(message_id),
    response_a_mid INTEGER NOT NULL REFERENCES conversations(message_id),
    response_b_mid INTEGER NOT NULL REFERENCES conversations(message_id),
    
    -- Model info
    model_a VARCHAR(200),
    model_b VARCHAR(200),
    pipeline_a VARCHAR(100),
    pipeline_b VARCHAR(100),
    
    -- Legacy
    config_a_id INTEGER REFERENCES configs(config_id),
    config_b_id INTEGER REFERENCES configs(config_id),
    
    is_config_a_first BOOLEAN NOT NULL,
    preference VARCHAR(10),        -- 'a', 'b', 'tie', NULL
    preference_ts TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

---

## Entity Relationship Diagram

```
┌─────────────┐     ┌─────────────────────┐     ┌───────────────┐
│   users     │────<│ conversation_metadata│>────│ conversations │
└─────────────┘     └─────────────────────┘     └───────────────┘
      │                      │                         │
      │                      │                         │
      v                      v                         v
┌─────────────┐     ┌─────────────────────┐     ┌───────────────┐
│  sessions   │     │ conv_doc_overrides  │     │   feedback    │
└─────────────┘     └─────────────────────┘     └───────────────┘
      │                      │                         │
      │                      │                         │
      v                      v                         v
┌─────────────┐     ┌─────────────────────┐     ┌───────────────┐
│user_doc_def │────<│     documents       │>────│    timing     │
└─────────────┘     └─────────────────────┘     └───────────────┘
                             │
                             v
                    ┌─────────────────────┐
                    │  document_chunks    │
                    │  (with embeddings)  │
                    └─────────────────────┘

┌─────────────┐     ┌─────────────────────┐
│static_config│     │   dynamic_config    │──────> config_audit
│ (singleton) │     │    (singleton)      │
└─────────────┘     └─────────────────────┘
```

---

## Required Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;        -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS pgcrypto;      -- For API key encryption
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- For fuzzy name matching
CREATE EXTENSION IF NOT EXISTS pg_textsearch; -- Optional: BM25 search
```

---

## Migration Notes

### From v1.x to v2.0

1. **`configs` table** kept for backward compatibility
2. **`conversations.conf_id`** kept but nullable; `model_used`/`pipeline_used` added
3. **`ab_comparisons`** has both `config_*_id` and `model_*` columns
4. After migration verified, run cleanup to drop legacy columns

### Grafana Query Migration

```sql
-- OLD
SELECT c.*, conf.config_name 
FROM conversations c 
JOIN configs conf ON c.conf_id = conf.config_id

-- NEW
SELECT c.*, c.model_used, c.pipeline_used 
FROM conversations c
```
