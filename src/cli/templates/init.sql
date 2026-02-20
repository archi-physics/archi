-- Archi PostgreSQL Schema v2.0
-- Unified database for conversations, vectors, and document catalog
-- Requires: PostgreSQL 17+ with pgvector, pg_textsearch (optional), pgcrypto, pg_trgm

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;        -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS pgcrypto;      -- For API key encryption
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- For fuzzy name matching

-- pg_textsearch is optional (requires PG17+, may not be GA yet)
-- If not available, hybrid search falls back to semantic-only
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_textsearch;
    RAISE NOTICE 'pg_textsearch extension enabled - BM25 search available';
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'pg_textsearch not available - BM25 search disabled, using semantic-only';
END $$;

-- ============================================================================
-- 1. USERS & AUTHENTICATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(200) PRIMARY KEY,  -- From auth provider or generated client_id
    
    -- Identity
    display_name TEXT,
    email TEXT,
    auth_provider VARCHAR(50) NOT NULL DEFAULT 'anonymous',  -- 'anonymous', 'local', 'github'
    
    -- Local auth
    password_hash VARCHAR(256),           -- For local accounts (Werkzeug pbkdf2)
    
    -- GitHub OAuth
    github_id VARCHAR(100),               -- GitHub user ID
    github_username VARCHAR(100),         -- GitHub username
    
    -- Role
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Preferences (explicit columns for known fields)
    theme VARCHAR(20) NOT NULL DEFAULT 'system',
    preferred_model VARCHAR(200),          -- Override global default
    preferred_temperature NUMERIC(3,2),    -- Override global default
    preferred_max_tokens INTEGER,          -- Override global default
    preferred_num_documents INTEGER,       -- Override retrieval count
    preferred_condense_prompt VARCHAR(100), -- Prompt selection
    preferred_chat_prompt VARCHAR(100),
    preferred_system_prompt VARCHAR(100),
    preferred_top_p NUMERIC(3,2),
    preferred_top_k INTEGER,
    
    -- BYOK API keys (encrypted with pgcrypto)
    -- Keys stored as: pgp_sym_encrypt(key, encryption_key)
    -- Encryption key comes from BYOK_ENCRYPTION_KEY env var
    api_key_openrouter BYTEA,      -- Encrypted
    api_key_openai BYTEA,          -- Encrypted  
    api_key_anthropic BYTEA,       -- Encrypted
    
    -- Session tracking
    last_login_at TIMESTAMP,
    login_count INTEGER NOT NULL DEFAULT 0,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_auth_provider ON users(auth_provider);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_github_id ON users(github_id) WHERE github_id IS NOT NULL;

-- ============================================================================
-- 1.1 SESSIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(200) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    data JSONB,                           -- Additional session data
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- ============================================================================
-- 2. STATIC CONFIGURATION (Deploy-Time)
-- ============================================================================

CREATE TABLE IF NOT EXISTS static_config (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Enforce single row
    
    -- Deployment identity
    deployment_name VARCHAR(100) NOT NULL,
    config_version VARCHAR(20) NOT NULL DEFAULT '2.0.0',
    
    -- Paths
    data_path TEXT NOT NULL DEFAULT '/root/data/',
    prompts_path TEXT NOT NULL DEFAULT '/root/archi/data/prompts/',
    
    -- Embedding configuration (affects vector dimensions - can't change at runtime)
    embedding_model VARCHAR(200) NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL DEFAULT 1000,
    chunk_overlap INTEGER NOT NULL DEFAULT 150,
    distance_metric VARCHAR(20) NOT NULL DEFAULT 'cosine',
    
    -- Available options (what's installed/configured)
    available_pipelines TEXT[] NOT NULL DEFAULT '{}',
    available_models TEXT[] NOT NULL DEFAULT '{}',
    available_providers TEXT[] NOT NULL DEFAULT '{}',
    
    -- Auth configuration
    auth_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    session_lifetime_days INTEGER NOT NULL DEFAULT 30,

    -- Source configuration (deploy-time)
    sources_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    services_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    data_manager_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    archi_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    global_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 3. DYNAMIC CONFIGURATION (Runtime)
-- ============================================================================

CREATE TABLE IF NOT EXISTS dynamic_config (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Enforce single row
    
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
    
    -- Schedules
    ingestion_schedule VARCHAR(100) NOT NULL DEFAULT '',  -- Cron expression
    source_schedules JSONB NOT NULL DEFAULT '{}'::jsonb,  -- Per-source schedules
    
    -- Logging
    verbosity INTEGER NOT NULL DEFAULT 3,
    
    -- Metadata
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(200)  -- user_id who made the change
);

-- Initialize dynamic_config with defaults
INSERT INTO dynamic_config (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ============================================================================
-- 3.1 CONFIG AUDIT LOG
-- ============================================================================

CREATE TABLE IF NOT EXISTS config_audit (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(200) NOT NULL,
    changed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    config_type VARCHAR(20) NOT NULL,  -- 'dynamic', 'user_pref'
    field_name VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT
);

CREATE INDEX IF NOT EXISTS idx_config_audit_user ON config_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_config_audit_time ON config_audit(changed_at DESC);

-- ============================================================================
-- 4. DOCUMENTS & VECTORS
-- ============================================================================

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    resource_hash VARCHAR(64) UNIQUE NOT NULL,
    
    -- File location (relative to data_path)
    file_path TEXT NOT NULL,
    
    -- Display info
    display_name TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- 'local_files', 'web', 'ticket', 'git'
    
    -- Source-specific fields
    url TEXT,                    -- For web sources
    ticket_id VARCHAR(100),      -- For ticket sources
    git_repo VARCHAR(200),       -- For git sources
    git_commit VARCHAR(64),      -- For git sources
    
    -- File metadata
    suffix VARCHAR(20),
    size_bytes BIGINT,
    mime_type VARCHAR(100),
    
    -- Provenance
    original_path TEXT,
    base_path TEXT,              -- For relative path reconstruction
    relative_path TEXT,          -- Path relative to base_path
    
    -- Extensible metadata (for source-specific fields not in columns)
    extra_json JSONB,            -- Structured extra metadata
    extra_text TEXT,             -- Searchable text representation
    
    -- Timestamps
    file_modified_at TIMESTAMP,
    ingested_at TIMESTAMP,
    indexed_at TIMESTAMP,        -- When embeddings were created
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Ingestion tracking
    ingestion_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    ingestion_error TEXT,
    
    -- Soft delete
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMP,
    
    CONSTRAINT valid_source CHECK (source_type IN ('local_files', 'web', 'ticket', 'git', 'sso', 'unknown')),
    CONSTRAINT valid_ingestion_status CHECK (ingestion_status IN ('pending', 'embedding', 'embedded', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(resource_hash);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_name ON documents USING gin (display_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_documents_active ON documents(is_deleted) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_documents_ingestion_status ON documents(ingestion_status);

-- Document chunks with embeddings
-- Note: Vector dimension ({{ embedding_dimensions }}) must match static_config.embedding_dimensions
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    
    -- Chunk content
    chunk_text TEXT NOT NULL,
    
    -- Vector embedding (dimension set at deploy time)
    -- Common dimensions: 384 (all-MiniLM-L6-v2), 1536 (text-embedding-ada-002)
    embedding vector({{ embedding_dimensions | default(384) }}),
    
    -- Chunk metadata
    start_char INTEGER,
    end_char INTEGER,
    metadata JSONB,              -- Original document metadata propagated to chunk
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);

-- Vector index (HNSW - default, good balance of speed/accuracy)
{% if vector_index_type | default('hnsw') == 'hnsw' -%}
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
    USING hnsw (embedding vector_cosine_ops) 
    WITH (m = {{ vector_index_hnsw_m | default(16) }}, ef_construction = {{ vector_index_hnsw_ef | default(64) }});
{% elif vector_index_type == 'ivfflat' -%}
-- IVFFlat index (create AFTER data is loaded for best results)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = {{ vector_index_ivfflat_lists | default(100) }});
{% else -%}
-- No vector index (exact search - for small datasets or testing)
{% endif %}

-- BM25 full-text search index (pg_textsearch) - created only if extension exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunks_bm25 ON document_chunks 
            USING bm25(chunk_text) WITH (text_config=''english'')';
        RAISE NOTICE 'BM25 index created on document_chunks';
    ELSE
        -- Fallback: create GIN index on tsvector for basic full-text search
        EXECUTE 'ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector(''english'', chunk_text)) STORED';
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks USING gin(chunk_tsv)';
        RAISE NOTICE 'Fallback GIN tsvector index created (pg_textsearch not available)';
    END IF;
END $$;

-- ============================================================================
-- 5. DOCUMENT SELECTIONS (3-Tier System)
-- ============================================================================

-- User defaults: power users can disable docs globally for themselves
CREATE TABLE IF NOT EXISTS user_document_defaults (
    user_id VARCHAR(200) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,  -- FALSE = opted out
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (user_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_user_doc_defaults_user ON user_document_defaults(user_id);

-- Conversation overrides: override user default for a specific conversation
CREATE TABLE IF NOT EXISTS conversation_doc_overrides (
    conversation_id INTEGER NOT NULL,  -- FK added after conversation_metadata created
    document_hash VARCHAR NOT NULL,  -- Hash reference to document (denormalized for simplicity)
    enabled BOOLEAN NOT NULL,  -- Explicit override value
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (conversation_id, document_hash)
);

CREATE INDEX IF NOT EXISTS idx_conv_doc_overrides_conv ON conversation_doc_overrides(conversation_id);

-- ============================================================================
-- 6. CONVERSATIONS & CHAT
-- ============================================================================

-- Legacy configs table - kept for reference but not actively used
CREATE TABLE IF NOT EXISTS configs (
    config_id SERIAL,
    config TEXT NOT NULL,
    config_name TEXT NOT NULL,
    PRIMARY KEY (config_id)
);

CREATE TABLE IF NOT EXISTS conversation_metadata (
    conversation_id SERIAL PRIMARY KEY,
    user_id VARCHAR(200) REFERENCES users(id) ON DELETE SET NULL,
    client_id TEXT,
    title TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMP NOT NULL DEFAULT NOW(),
    archi_version VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_conv_meta_user ON conversation_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_meta_client ON conversation_metadata(client_id);

-- Add FK to conversation_doc_overrides now that conversation_metadata exists
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'conversation_doc_overrides_conversation_id_fkey'
          AND conrelid = 'conversation_doc_overrides'::regclass
    ) THEN
        ALTER TABLE conversation_doc_overrides
            DROP CONSTRAINT conversation_doc_overrides_conversation_id_fkey;
    END IF;
END $$;
ALTER TABLE conversation_doc_overrides 
    ADD CONSTRAINT conversation_doc_overrides_conversation_id_fkey 
    FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS conversations (
    message_id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE,
    
    archi_service TEXT NOT NULL,
    sender TEXT NOT NULL,
    content TEXT NOT NULL,
    
    -- NEW: Capture what was actually used (replaces conf_id join)
    model_used VARCHAR(200),
    pipeline_used VARCHAR(100),
    
    -- RAG context
    link TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    
    ts TIMESTAMP NOT NULL,
    
    conf_id INTEGER REFERENCES configs(config_id)
);

CREATE INDEX IF NOT EXISTS idx_conversations_conv ON conversations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_ts ON conversations(ts);
CREATE INDEX IF NOT EXISTS idx_conversations_model ON conversations(model_used);

-- Feedback on messages
CREATE TABLE IF NOT EXISTS feedback (
    mid INTEGER NOT NULL REFERENCES conversations(message_id) ON DELETE CASCADE,
    feedback_ts TIMESTAMP NOT NULL,
    feedback TEXT NOT NULL,           -- 'like', 'dislike', 'comment'
    feedback_msg TEXT,                -- Optional text feedback/comment
    incorrect BOOLEAN,                -- Flag: response was factually incorrect
    unhelpful BOOLEAN,                -- Flag: response didn't help
    inappropriate BOOLEAN,            -- Flag: response was inappropriate
    
    PRIMARY KEY (mid, feedback_ts)
);

CREATE INDEX IF NOT EXISTS idx_feedback_mid ON feedback(mid);

-- Response timing metrics
CREATE TABLE IF NOT EXISTS timing (
    mid INTEGER PRIMARY KEY REFERENCES conversations(message_id) ON DELETE CASCADE,
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

-- ============================================================================
-- 7. AGENT TRACES
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_traces (
    trace_id UUID PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE,
    message_id INTEGER REFERENCES conversations(message_id) ON DELETE SET NULL,
    user_message_id INTEGER REFERENCES conversations(message_id) ON DELETE SET NULL,
    
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

CREATE INDEX IF NOT EXISTS idx_agent_traces_conv ON agent_traces(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_traces_status ON agent_traces(status);
CREATE INDEX IF NOT EXISTS idx_agent_traces_message ON agent_traces(message_id);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE,
    message_id INTEGER NOT NULL REFERENCES conversations(message_id) ON DELETE CASCADE,
    
    step_number INTEGER NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    tool_args JSONB,
    tool_result TEXT,
    
    ts TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_message ON agent_tool_calls(message_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conv ON agent_tool_calls(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON agent_tool_calls(tool_name);

-- ============================================================================
-- 8. A/B COMPARISON TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS ab_comparisons (
    comparison_id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE,
    user_prompt_mid INTEGER NOT NULL REFERENCES conversations(message_id) ON DELETE CASCADE,
    response_a_mid INTEGER NOT NULL REFERENCES conversations(message_id) ON DELETE CASCADE,
    response_b_mid INTEGER NOT NULL REFERENCES conversations(message_id) ON DELETE CASCADE,
    
    -- Model/pipeline info (optional - can be derived from config_*_id if not set)
    model_a VARCHAR(200),
    model_b VARCHAR(200),
    pipeline_a VARCHAR(100),
    pipeline_b VARCHAR(100),
    
    config_a_id INTEGER REFERENCES configs(config_id),
    config_b_id INTEGER REFERENCES configs(config_id),
    
    is_config_a_first BOOLEAN NOT NULL,
    preference VARCHAR(10),
    preference_ts TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ab_comparisons_conversation ON ab_comparisons(conversation_id);
CREATE INDEX IF NOT EXISTS idx_ab_comparisons_models ON ab_comparisons(model_a, model_b);
CREATE INDEX IF NOT EXISTS idx_ab_comparisons_preference ON ab_comparisons(preference) WHERE preference IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ab_comparisons_pending ON ab_comparisons(conversation_id) WHERE preference IS NULL;

-- ============================================================================
-- 9. MIGRATION STATE (for resumable migrations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS migration_state (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(100) NOT NULL UNIQUE,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    last_checkpoint JSONB,  -- {phase: str, last_id: int, count: int}
    status VARCHAR(20) NOT NULL DEFAULT 'in_progress',  -- 'in_progress', 'completed', 'failed'
    error_message TEXT
);

-- ============================================================================
-- 10. GRAFANA ACCESS
-- ============================================================================

{% if use_grafana -%}
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana') THEN
        CREATE USER grafana WITH PASSWORD '{{ grafana_pg_password }}';
    END IF;
END $$;

GRANT USAGE ON SCHEMA public TO grafana;
GRANT SELECT ON 
    users,
    static_config,
    dynamic_config,
    documents,
    document_chunks,
    user_document_defaults,
    conversation_doc_overrides,
    configs,
    conversation_metadata,
    conversations,
    feedback,
    timing,
    agent_tool_calls,
    ab_comparisons,
    migration_state
TO grafana;
{% endif %}

-- ============================================================================
-- NOTES
-- ============================================================================
-- 
-- Grafana queries use model_used and pipeline_used columns directly:
-- SELECT c.*, c.model_used, c.pipeline_used FROM conversations c
