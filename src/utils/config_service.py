"""
ConfigService - Manages static and dynamic configuration in PostgreSQL.

Implements the Configuration requirements from the consolidate-to-postgres spec:
- Static configuration (deploy-time, immutable at runtime)
- Dynamic configuration (runtime-modifiable via API)
- Validation of configuration values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
import json

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StaticConfig:
    """Deploy-time configuration (immutable at runtime)."""
    
    deployment_name: str
    config_version: str
    
    # Paths
    data_path: str
    
    # Embedding configuration
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    distance_metric: str
    
    # Paths with defaults
    prompts_path: str = "/root/archi/data/prompts/"
    
    # Available options
    available_pipelines: List[str] = field(default_factory=list)
    available_models: List[str] = field(default_factory=list)
    available_providers: List[str] = field(default_factory=list)
    
    # Auth
    auth_enabled: bool = False
    session_lifetime_days: int = 30

    # Config sections
    sources_config: Dict[str, Any] = field(default_factory=dict)
    services_config: Dict[str, Any] = field(default_factory=dict)
    data_manager_config: Dict[str, Any] = field(default_factory=dict)
    archi_config: Dict[str, Any] = field(default_factory=dict)
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    created_at: Optional[str] = None


@dataclass
class DynamicConfig:
    """Runtime-modifiable configuration."""
    
    # Model settings
    active_pipeline: str = "QAPipeline"
    active_model: str = "openai/gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
    
    # Additional generation params
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    # Prompt selection (file names without extension)
    active_condense_prompt: str = "default"
    active_chat_prompt: str = "default"
    active_system_prompt: str = "default"
    
    # Retrieval settings
    num_documents_to_retrieve: int = 10
    use_hybrid_search: bool = True
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    
    # Schedules
    ingestion_schedule: str = ""
    source_schedules: Dict[str, str] = field(default_factory=dict)  # source_name -> cron expression
    
    # Logging
    verbosity: int = 3
    
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class ConfigService:
    """
    Service for managing application configuration in PostgreSQL.
    
    Handles both static (deploy-time) and dynamic (runtime) configuration.
    Static config is cached in memory after initial load.
    
    Example:
        >>> service = ConfigService(pg_config={'host': 'localhost', ...})
        >>> static = service.get_static_config()
        >>> dynamic = service.get_dynamic_config()
        >>> service.update_dynamic_config(temperature=0.5, updated_by="admin")
    """
    
    def __init__(self, pg_config: Optional[Dict[str, Any]] = None, *, connection_pool=None):
        """
        Initialize ConfigService.
        
        Args:
            pg_config: PostgreSQL connection parameters (fallback)
            connection_pool: ConnectionPool instance (preferred)
        """
        self._pool = connection_pool
        self._pg_config = pg_config
        self._static_cache: Optional[StaticConfig] = None
        # Ensure supporting tables exist for full-config storage (best-effort)
        try:
            self._ensure_config_tables()
        except Exception as exc:
            logger.debug("Could not ensure config tables: %s", exc)
    
    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get a database connection."""
        if self._pool:
            return self._pool.get_connection_direct()
        elif self._pg_config:
            return psycopg2.connect(**self._pg_config)
        else:
            raise ValueError("No connection pool or pg_config provided")

    def _release_connection(self, conn) -> None:
        """Release connection back to pool or close it."""
        if self._pool:
            self._pool.release_connection(conn)
        else:
            conn.close()

    def _ensure_config_tables(self) -> None:
        """Create/extend config tables if missing."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Create base tables if they don't exist
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS static_config (
                        id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                        deployment_name VARCHAR(100) NOT NULL,
                        config_version VARCHAR(20) NOT NULL DEFAULT '2.0.0',
                        data_path TEXT NOT NULL DEFAULT '/root/data/',
                        prompts_path TEXT NOT NULL DEFAULT '/root/archi/data/prompts/',
                        embedding_model VARCHAR(200) NOT NULL,
                        embedding_dimensions INTEGER NOT NULL,
                        chunk_size INTEGER NOT NULL DEFAULT 1000,
                        chunk_overlap INTEGER NOT NULL DEFAULT 150,
                        distance_metric VARCHAR(20) NOT NULL DEFAULT 'cosine',
                        available_pipelines TEXT[] NOT NULL DEFAULT '{}',
                        available_models TEXT[] NOT NULL DEFAULT '{}',
                        available_providers TEXT[] NOT NULL DEFAULT '{}',
                        auth_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        session_lifetime_days INTEGER NOT NULL DEFAULT 30,
                        sources_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        services_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        data_manager_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        archi_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        global_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS dynamic_config (
                        id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                        active_pipeline VARCHAR(100) NOT NULL DEFAULT 'QAPipeline',
                        active_model VARCHAR(200) NOT NULL DEFAULT 'openai/gpt-4o',
                        temperature NUMERIC(3,2) NOT NULL DEFAULT 0.7,
                        max_tokens INTEGER NOT NULL DEFAULT 4096,
                        system_prompt TEXT,
                        top_p NUMERIC(3,2) NOT NULL DEFAULT 0.9,
                        top_k INTEGER NOT NULL DEFAULT 50,
                        repetition_penalty NUMERIC(4,2) NOT NULL DEFAULT 1.0,
                        active_condense_prompt VARCHAR(100) NOT NULL DEFAULT 'default',
                        active_chat_prompt VARCHAR(100) NOT NULL DEFAULT 'default',
                        active_system_prompt VARCHAR(100) NOT NULL DEFAULT 'default',
                        num_documents_to_retrieve INTEGER NOT NULL DEFAULT 10,
                        use_hybrid_search BOOLEAN NOT NULL DEFAULT TRUE,
                        bm25_weight NUMERIC(3,2) NOT NULL DEFAULT 0.3,
                        semantic_weight NUMERIC(3,2) NOT NULL DEFAULT 0.7,
                        ingestion_schedule VARCHAR(100) NOT NULL DEFAULT '',
                        verbosity INTEGER NOT NULL DEFAULT 3,
                        updated_at TIMESTAMP,
                        updated_by VARCHAR(100)
                    );
                    """
                )
                # Ensure JSONB columns present (idempotent)
                cursor.execute(
                    """
                    ALTER TABLE static_config
                    ADD COLUMN IF NOT EXISTS services_config JSONB DEFAULT '{}'::jsonb,
                    ADD COLUMN IF NOT EXISTS data_manager_config JSONB DEFAULT '{}'::jsonb,
                    ADD COLUMN IF NOT EXISTS archi_config JSONB DEFAULT '{}'::jsonb,
                    ADD COLUMN IF NOT EXISTS global_config JSONB DEFAULT '{}'::jsonb
                    """
                )
                conn.commit()
        except psycopg2.Error as e:
            logger.debug("Could not ensure config tables/columns: %s", e)
        finally:
            self._release_connection(conn)

    # =========================================================================
    # Raw config storage (full YAML as JSONB)
    # =========================================================================

    # raw_config removed; use explicit sections in static_config

    @staticmethod
    def _normalize_sources_config(sources_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(sources_config, dict):
            return {}
        normalized = dict(sources_config)
        for key, value in list(normalized.items()):
            if not isinstance(value, dict):
                normalized[key] = {}
        return normalized
    
    # =========================================================================
    # Static Configuration
    # =========================================================================
    
    def get_static_config(self, *, force_reload: bool = False) -> Optional[StaticConfig]:
        """
        Get static configuration.
        
        Implements: Static config loading at startup with caching
        
        Args:
            force_reload: If True, bypass cache and reload from database
            
        Returns:
            StaticConfig object, or None if not initialized
        """
        if self._static_cache is not None and not force_reload:
            return self._static_cache
        
        # Ensure schema exists before attempting to read
        self._ensure_config_tables()
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT deployment_name, config_version, data_path, prompts_path,
                           embedding_model, embedding_dimensions,
                           chunk_size, chunk_overlap, distance_metric,
                           available_pipelines, available_models, available_providers,
                           auth_enabled, session_lifetime_days, sources_config,
                           services_config, data_manager_config, archi_config, global_config,
                           created_at
                    FROM static_config
                    WHERE id = 1
                    """
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                self._static_cache = StaticConfig(
                    deployment_name=row["deployment_name"],
                    config_version=row["config_version"],
                    data_path=row["data_path"],
                    prompts_path=row.get("prompts_path", "/root/archi/data/prompts/"),
                    embedding_model=row["embedding_model"],
                    embedding_dimensions=row["embedding_dimensions"],
                    chunk_size=row["chunk_size"],
                    chunk_overlap=row["chunk_overlap"],
                    distance_metric=row["distance_metric"],
                    available_pipelines=row["available_pipelines"] or [],
                    available_models=row["available_models"] or [],
                    available_providers=row["available_providers"] or [],
                    auth_enabled=row["auth_enabled"],
                    session_lifetime_days=row.get("session_lifetime_days", 30),
                    sources_config=row.get("sources_config") or {},
                    services_config=row.get("services_config") or {},
                    data_manager_config=row.get("data_manager_config") or {},
                    archi_config=row.get("archi_config") or {},
                    global_config=row.get("global_config") or {},
                    created_at=str(row["created_at"]) if row["created_at"] else None,
                )
                
                return self._static_cache
        finally:
            self._release_connection(conn)
    
    def initialize_static_config(
        self,
        *,
        deployment_name: str,
        config_version: str = "2.0.0",
        data_path: str = "/root/data/",
        embedding_model: str,
        embedding_dimensions: int,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        distance_metric: str = "cosine",
        available_pipelines: Optional[List[str]] = None,
        available_models: Optional[List[str]] = None,
        available_providers: Optional[List[str]] = None,
        auth_enabled: bool = False,
        sources_config: Optional[Dict[str, Any]] = None,
        services_config: Optional[Dict[str, Any]] = None,
        data_manager_config: Optional[Dict[str, Any]] = None,
        archi_config: Optional[Dict[str, Any]] = None,
        global_config: Optional[Dict[str, Any]] = None,
    ) -> StaticConfig:
        """
        Initialize static configuration (typically called once at deployment).
        
        This should only be called during initial setup or redeployment.
        Subsequent calls will fail due to the unique constraint.
        
        Args:
            deployment_name: Name of this deployment
            config_version: Schema version
            data_path: Path to data directory
            embedding_model: Embedding model identifier
            embedding_dimensions: Vector dimensions for embeddings
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            distance_metric: Vector distance metric
            available_pipelines: List of available pipelines
            available_models: List of available models
            available_providers: List of available providers
            auth_enabled: Whether authentication is enabled
            
        Returns:
            Created StaticConfig
            
        Raises:
            psycopg2.IntegrityError: If static config already exists
        """
        normalized_sources = self._normalize_sources_config(sources_config)
        services_section = services_config or {}
        data_manager_section = data_manager_config or {}
        archi_section = archi_config or {}
        global_section = global_config or {}
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO static_config (
                        id, deployment_name, config_version, data_path,
                        embedding_model, embedding_dimensions,
                        chunk_size, chunk_overlap, distance_metric,
                        available_pipelines, available_models, available_providers,
                        auth_enabled, sources_config,
                        services_config, data_manager_config, archi_config, global_config
                    )
                    VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        deployment_name = EXCLUDED.deployment_name,
                        config_version = EXCLUDED.config_version,
                        data_path = EXCLUDED.data_path,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dimensions = EXCLUDED.embedding_dimensions,
                        chunk_size = EXCLUDED.chunk_size,
                        chunk_overlap = EXCLUDED.chunk_overlap,
                        distance_metric = EXCLUDED.distance_metric,
                        available_pipelines = EXCLUDED.available_pipelines,
                        available_models = EXCLUDED.available_models,
                        available_providers = EXCLUDED.available_providers,
                        auth_enabled = EXCLUDED.auth_enabled,
                        sources_config = EXCLUDED.sources_config,
                        services_config = EXCLUDED.services_config,
                        data_manager_config = EXCLUDED.data_manager_config,
                        archi_config = EXCLUDED.archi_config,
                        global_config = EXCLUDED.global_config
                    RETURNING deployment_name, config_version, data_path,
                              embedding_model, embedding_dimensions,
                              chunk_size, chunk_overlap, distance_metric,
                              available_pipelines, available_models, available_providers,
                              auth_enabled, sources_config,
                              services_config, data_manager_config, archi_config, global_config,
                              created_at
                    """,
                    (
                        deployment_name, config_version, data_path,
                        embedding_model, embedding_dimensions,
                        chunk_size, chunk_overlap, distance_metric,
                        available_pipelines or [],
                        available_models or [],
                        available_providers or [],
                        auth_enabled,
                        psycopg2.extras.Json(normalized_sources),
                        psycopg2.extras.Json(services_section),
                        psycopg2.extras.Json(data_manager_section),
                        psycopg2.extras.Json(archi_section),
                        psycopg2.extras.Json(global_section),
                    )
                )
                row = cursor.fetchone()
                conn.commit()
                
                self._static_cache = StaticConfig(
                    deployment_name=row["deployment_name"],
                    config_version=row["config_version"],
                    data_path=row["data_path"],
                    embedding_model=row["embedding_model"],
                    embedding_dimensions=row["embedding_dimensions"],
                    chunk_size=row["chunk_size"],
                    chunk_overlap=row["chunk_overlap"],
                    distance_metric=row["distance_metric"],
                    available_pipelines=row["available_pipelines"] or [],
                    available_models=row["available_models"] or [],
                    available_providers=row["available_providers"] or [],
                    auth_enabled=row["auth_enabled"],
                    sources_config=row.get("sources_config") or {},
                    services_config=row.get("services_config") or {},
                    data_manager_config=row.get("data_manager_config") or {},
                    archi_config=row.get("archi_config") or {},
                    global_config=row.get("global_config") or {},
                    created_at=str(row["created_at"]) if row["created_at"] else None,
                )
                
                logger.info(f"Initialized static config: {deployment_name}")
                return self._static_cache
        finally:
            self._release_connection(conn)

    # =========================================================================
    # Embedding helpers
    # =========================================================================

    @staticmethod
    def _resolve_embedding_classes(embedding_class_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve known embedding class names to callables.

        Currently supports: OpenAIEmbeddings, HuggingFaceEmbeddings.
        """
        if not embedding_class_map:
            return {}

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import OpenAIEmbeddings

        EMBEDDING_MAPPING = {
            "OpenAIEmbeddings": OpenAIEmbeddings,
            "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        }

        resolved: Dict[str, Any] = {}
        for name, cfg in embedding_class_map.items():
            entry = dict(cfg or {})
            cls_name = entry.get("class")
            if isinstance(cls_name, str) and cls_name in EMBEDDING_MAPPING:
                entry["class"] = EMBEDDING_MAPPING[cls_name]
            elif cls_name is None and name in EMBEDDING_MAPPING:
                entry["class"] = EMBEDDING_MAPPING[name]
            resolved[name] = entry
        return resolved

    def get_embedding_class_map(self, *, resolved: bool = False) -> Dict[str, Any]:
        """
        Return embedding_class_map from static config.

        Args:
            resolved: If True, map known class names to callables.
        """
        static = self.get_static_config()
        if not static or not static.data_manager_config:
            return {}
        embedding_class_map = static.data_manager_config.get("embedding_class_map", {}) or {}
        if resolved:
            return self._resolve_embedding_classes(embedding_class_map)
        return embedding_class_map

    # =========================================================================
    # Dynamic Configuration
    # =========================================================================
    
    def get_dynamic_config(self) -> DynamicConfig:
        """
        Get current dynamic configuration.
        
        Implements: Dynamic config read
        
        Returns:
            DynamicConfig object
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Ensure source_schedules column exists (migration)
                cursor.execute("""
                    ALTER TABLE dynamic_config 
                    ADD COLUMN IF NOT EXISTS source_schedules JSONB NOT NULL DEFAULT '{}'::jsonb
                """)
                conn.commit()
                
                cursor.execute(
                    """
                    SELECT active_pipeline, active_model, temperature, max_tokens,
                           system_prompt, top_p, top_k, repetition_penalty,
                           active_condense_prompt, active_chat_prompt, active_system_prompt,
                           num_documents_to_retrieve, use_hybrid_search, bm25_weight, semantic_weight,
                           bm25_k1, bm25_b, ingestion_schedule, source_schedules, verbosity, updated_at, updated_by
                    FROM dynamic_config
                    WHERE id = 1
                    """
                )
                row = cursor.fetchone()
                
                if row is None:
                    # Return defaults if not initialized
                    return DynamicConfig()
                
                # Parse source_schedules JSONB
                source_schedules = row.get("source_schedules") or {}
                if isinstance(source_schedules, str):
                    import json
                    source_schedules = json.loads(source_schedules)
                
                return DynamicConfig(
                    active_pipeline=row["active_pipeline"],
                    active_model=row["active_model"],
                    temperature=float(row["temperature"]),
                    max_tokens=row["max_tokens"],
                    system_prompt=row["system_prompt"],
                    top_p=float(row.get("top_p", 0.9)),
                    top_k=row.get("top_k", 50),
                    repetition_penalty=float(row.get("repetition_penalty", 1.0)),
                    active_condense_prompt=row.get("active_condense_prompt", "default"),
                    active_chat_prompt=row.get("active_chat_prompt", "default"),
                    active_system_prompt=row.get("active_system_prompt", "default"),
                    num_documents_to_retrieve=row["num_documents_to_retrieve"],
                    use_hybrid_search=row["use_hybrid_search"],
                    bm25_weight=float(row["bm25_weight"]),
                    semantic_weight=float(row["semantic_weight"]),
                    ingestion_schedule=row.get("ingestion_schedule", ""),
                    source_schedules=source_schedules,
                    verbosity=row.get("verbosity", 3),
                    updated_at=str(row["updated_at"]) if row["updated_at"] else None,
                    updated_by=row["updated_by"],
                )
        finally:
            self._release_connection(conn)
    
    def update_dynamic_config(
        self,
        *,
        active_pipeline: Optional[str] = None,
        active_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        num_documents_to_retrieve: Optional[int] = None,
        use_hybrid_search: Optional[bool] = None,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        updated_by: Optional[str] = None,
    ) -> DynamicConfig:
        """
        Update dynamic configuration.
        
        Implements:
        - Dynamic config update via API
        - Dynamic config validation error (raises ConfigValidationError)
        - Model selection validation
        
        Args:
            active_pipeline: New active pipeline
            active_model: New active model (must be in available_models)
            temperature: New temperature (0.0 - 2.0)
            max_tokens: New max tokens
            system_prompt: New system prompt (or None to clear)
            num_documents_to_retrieve: Number of documents for retrieval
            use_hybrid_search: Enable hybrid search
            bm25_weight: Weight for BM25 in hybrid search
            semantic_weight: Weight for semantic in hybrid search
            updated_by: User ID making the change
            
        Returns:
            Updated DynamicConfig
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate values
        self._validate_dynamic_config(
            active_pipeline=active_pipeline,
            active_model=active_model,
            temperature=temperature,
            max_tokens=max_tokens,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )
        
        updates = []
        params: List[Any] = []
        
        if active_pipeline is not None:
            updates.append("active_pipeline = %s")
            params.append(active_pipeline)
        
        if active_model is not None:
            updates.append("active_model = %s")
            params.append(active_model)
        
        if temperature is not None:
            updates.append("temperature = %s")
            params.append(temperature)
        
        if max_tokens is not None:
            updates.append("max_tokens = %s")
            params.append(max_tokens)
        
        if system_prompt is not None:
            updates.append("system_prompt = %s")
            params.append(system_prompt if system_prompt else None)
        
        if num_documents_to_retrieve is not None:
            updates.append("num_documents_to_retrieve = %s")
            params.append(num_documents_to_retrieve)
        
        if use_hybrid_search is not None:
            updates.append("use_hybrid_search = %s")
            params.append(use_hybrid_search)
        
        if bm25_weight is not None:
            updates.append("bm25_weight = %s")
            params.append(bm25_weight)
        
        if semantic_weight is not None:
            updates.append("semantic_weight = %s")
            params.append(semantic_weight)
        
        if not updates:
            return self.get_dynamic_config()
        
        updates.append("updated_at = NOW()")
        updates.append("updated_by = %s")
        params.append(updated_by)
        
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    UPDATE dynamic_config
                    SET {', '.join(updates)}
                    WHERE id = 1
                    RETURNING active_pipeline, active_model, temperature, max_tokens,
                              system_prompt, num_documents_to_retrieve,
                              use_hybrid_search, bm25_weight, semantic_weight,
                              updated_at, updated_by
                    """,
                    params
                )
                row = cursor.fetchone()
                conn.commit()
                
                if row is None:
                    # Initialize if not exists
                    cursor.execute(
                        "INSERT INTO dynamic_config (id) VALUES (1) ON CONFLICT DO NOTHING"
                    )
                    conn.commit()
                    return self.get_dynamic_config()
                
                logger.info(f"Updated dynamic config by {updated_by}")
                
                return DynamicConfig(
                    active_pipeline=row["active_pipeline"],
                    active_model=row["active_model"],
                    temperature=float(row["temperature"]),
                    max_tokens=row["max_tokens"],
                    system_prompt=row["system_prompt"],
                    num_documents_to_retrieve=row["num_documents_to_retrieve"],
                    use_hybrid_search=row["use_hybrid_search"],
                    bm25_weight=float(row["bm25_weight"]),
                    semantic_weight=float(row["semantic_weight"]),
                    updated_at=str(row["updated_at"]) if row["updated_at"] else None,
                    updated_by=row["updated_by"],
                )
        finally:
            self._release_connection(conn)
    
    def update_source_schedule(
        self,
        source_name: str,
        schedule: str,
        *,
        updated_by: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Update the schedule for a specific data source.
        
        Args:
            source_name: Name of the source (e.g., 'jira', 'git', 'links')
            schedule: Cron expression or schedule key (e.g., '0 */6 * * *', 'hourly', 'disabled')
            updated_by: User making the change
            
        Returns:
            Updated source_schedules dict
        """
        import json
        
        # Map UI-friendly values to cron expressions
        schedule_map = {
            'disabled': '',
            'hourly': '0 * * * *',
            'every_6h': '0 */6 * * *',
            'daily': '0 0 * * *',
        }
        cron_expr = schedule_map.get(schedule, schedule)
        
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Update the specific source in JSONB
                cursor.execute(
                    """
                    UPDATE dynamic_config
                    SET source_schedules = source_schedules || %s::jsonb,
                        updated_at = NOW(),
                        updated_by = %s
                    WHERE id = 1
                    RETURNING source_schedules
                    """,
                    (json.dumps({source_name: cron_expr}), updated_by)
                )
                row = cursor.fetchone()
                conn.commit()
                
                if row is None:
                    return {}
                
                logger.info(f"Updated schedule for {source_name} to '{cron_expr}' by {updated_by}")
                return row.get("source_schedules", {})
        finally:
            self._release_connection(conn)
    
    def get_source_schedules(self) -> Dict[str, str]:
        """
        Get all source schedules.
        
        Returns:
            Dict mapping source names to cron expressions
        """
        dynamic = self.get_dynamic_config()
        return dynamic.source_schedules if dynamic else {}

    def _validate_dynamic_config(
        self,
        *,
        active_pipeline: Optional[str] = None,
        active_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ) -> None:
        """
        Validate dynamic config values.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        static = self.get_static_config()
        
        if active_pipeline is not None and static:
            if static.available_pipelines and active_pipeline not in static.available_pipelines:
                raise ConfigValidationError(
                    "active_pipeline",
                    f"must be one of {static.available_pipelines}"
                )
        
        if active_model is not None and static:
            if static.available_models and active_model not in static.available_models:
                raise ConfigValidationError(
                    "active_model",
                    f"must be one of {static.available_models}"
                )
        
        if temperature is not None:
            if not (0.0 <= temperature <= 2.0):
                raise ConfigValidationError(
                    "temperature",
                    "must be between 0.0 and 2.0"
                )
        
        if max_tokens is not None:
            if max_tokens < 1:
                raise ConfigValidationError(
                    "max_tokens",
                    "must be at least 1"
                )
        
        if bm25_weight is not None:
            if not (0.0 <= bm25_weight <= 1.0):
                raise ConfigValidationError(
                    "bm25_weight",
                    "must be between 0.0 and 1.0"
                )
        
        if semantic_weight is not None:
            if not (0.0 <= semantic_weight <= 1.0):
                raise ConfigValidationError(
                    "semantic_weight",
                    "must be between 0.0 and 1.0"
                )
    
    # =========================================================================
    # Helper methods for config.yaml migration
    # =========================================================================
    
    @staticmethod
    def from_config_yaml(config: Dict[str, Any], pg_config: Dict[str, Any]) -> "ConfigService":
        """
        Create ConfigService and initialize from config.yaml.
        
        This is the main entry point for migrating from config.yaml to database.
        
        Args:
            config: Parsed config.yaml dictionary
            pg_config: PostgreSQL connection parameters
            
        Returns:
            Initialized ConfigService
        """
        service = ConfigService(pg_config)
        # Store full raw config for downstream runtime consumption
        try:
            service.set_raw_config(config)
        except Exception as exc:
            logger.warning("Failed to persist raw config to Postgres: %s", exc)
        
        data_manager = config.get("data_manager", {})
        embedding_class_map = data_manager.get("embedding_class_map", {})
        embedding_name = data_manager.get("embedding_name", "HuggingFaceEmbeddings")
        
        # Determine embedding dimensions
        default_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "OpenAIEmbeddings": 1536,
            "HuggingFaceEmbeddings": 384,
        }
        embedding_dimensions = default_dimensions.get(embedding_name, 384)
        if embedding_name in embedding_class_map:
            embedding_dimensions = embedding_class_map[embedding_name].get(
                "dimensions", embedding_dimensions
            )
        
        # Get available providers/models from providers section
        archi_cfg = config.get("archi", {}) or {}
        providers_cfg = archi_cfg.get("providers", {}) or {}
        available_models = []
        available_providers = list(providers_cfg.keys())
        for p_cfg in providers_cfg.values():
            models = p_cfg.get("models") or []
            for m in models:
                if m not in available_models:
                    available_models.append(m)

        # Get available pipelines
        pipelines_config = archi_cfg.get("pipeline_map", {})
        available_pipelines = list(pipelines_config.keys())
        
        # Initialize static config
        service.initialize_static_config(
            deployment_name=config.get("name", "default"),
            data_path=config.get("global", {}).get("DATA_PATH", "/root/data/"),
            embedding_model=embedding_name,
            embedding_dimensions=embedding_dimensions,
            chunk_size=data_manager.get("chunk_size", 1000),
            chunk_overlap=data_manager.get("chunk_overlap", 150),
            distance_metric=data_manager.get("distance_metric", "cosine"),
            available_pipelines=available_pipelines,
            available_models=available_models,
            available_providers=available_providers,
            auth_enabled=config.get("services", {}).get("chat_app", {}).get("auth", {}).get("enabled", False),
            sources_config=data_manager.get("sources", {}),
        )
        
        # Initialize dynamic config from data_manager settings
        retrievers = data_manager.get("retrievers", {})
        hybrid = retrievers.get("hybrid_retriever", {})
        
        service.update_dynamic_config(
            active_pipeline=config.get("services", {}).get("chat_app", {}).get("pipeline", "QAPipeline"),
            num_documents_to_retrieve=hybrid.get("num_documents_to_retrieve", 10),
            bm25_weight=hybrid.get("bm25_weight", 0.3),
            semantic_weight=hybrid.get("semantic_weight", 0.7),
            updated_by="system",
        )
        
        return service

    def initialize_from_yaml(self, config: Dict[str, Any]) -> None:
        """
        Initialize or sync static/dynamic config from YAML config dictionary.
        
        Call this on service startup to ensure database config matches YAML.
        Uses UPSERT semantics - will update existing config or create if missing.
        
        Args:
            config: Parsed config.yaml dictionary
        """
        # Persist raw config for runtime access
        try:
            self.set_raw_config(config)
        except Exception as exc:
            logger.warning("Failed to persist raw config to Postgres: %s", exc)

        data_manager = config.get("data_manager", {})
        embedding_class_map = data_manager.get("embedding_class_map", {})
        embedding_name = data_manager.get("embedding_name", "HuggingFaceEmbeddings")
        
        # Determine embedding dimensions
        default_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "OpenAIEmbeddings": 1536,
            "HuggingFaceEmbeddings": 384,
        }
        embedding_dimensions = default_dimensions.get(embedding_name, 384)
        if embedding_name in embedding_class_map:
            embedding_dimensions = embedding_class_map[embedding_name].get(
                "dimensions", embedding_dimensions
            )
        
        # Get available providers/models from providers section
        archi_cfg = config.get("archi", {}) or {}
        providers_cfg = archi_cfg.get("providers", {}) or {}
        available_models = []
        available_providers = list(providers_cfg.keys())
        for p_cfg in providers_cfg.values():
            models = p_cfg.get("models") or []
            for m in models:
                if m not in available_models:
                    available_models.append(m)

        # Get available pipelines
        available_pipelines = archi_cfg.get("pipeline_map", [])
        
        existing_static = self.get_static_config()
        sources_config = data_manager.get("sources", {})
        if existing_static and existing_static.sources_config:
            sources_config = existing_static.sources_config

        # Initialize static config (uses UPSERT - won't fail if exists)
        self.initialize_static_config(
            deployment_name=config.get("name", "default"),
            data_path=config.get("global", {}).get("DATA_PATH", "/root/data/"),
            embedding_model=embedding_name,
            embedding_dimensions=embedding_dimensions,
            chunk_size=data_manager.get("chunk_size", 1000),
            chunk_overlap=data_manager.get("chunk_overlap", 150),
            distance_metric=data_manager.get("distance_metric", "cosine"),
            available_pipelines=available_pipelines,
            available_models=available_models,
            available_providers=available_providers,
            auth_enabled=config.get("services", {}).get("chat_app", {}).get("auth", {}).get("enabled", False),
            sources_config=sources_config,
        )
        
        # Initialize dynamic config from data_manager settings
        retrievers = data_manager.get("retrievers", {})
        hybrid = retrievers.get("hybrid_retriever", {})
        
        # Only set dynamic config if not already present (avoid overwriting admin changes)
        existing_dynamic = self.get_dynamic_config()
        if existing_dynamic.updated_by is None:
            # First initialization - set from YAML
            self.update_dynamic_config(
                active_pipeline=config.get("services", {}).get("chat_app", {}).get("pipeline", "QAPipeline"),
                num_documents_to_retrieve=hybrid.get("num_documents_to_retrieve", 10),
                bm25_weight=hybrid.get("bm25_weight", 0.3),
                semantic_weight=hybrid.get("semantic_weight", 0.7),
                verbosity=config.get("global", {}).get("verbosity", 3),
                updated_by="system",
            )
            logger.info("Initialized dynamic config from YAML")
        else:
            logger.debug("Skipping dynamic config initialization - already configured by admin")

    # =========================================================================
    # User Preferences
    # =========================================================================
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get preferences for a specific user.
        
        Args:
            user_id: The user ID
            
        Returns:
            Dict of user preferences (non-null values only)
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT preferred_model, preferred_temperature, preferred_max_tokens,
                           preferred_num_documents, preferred_condense_prompt,
                           preferred_chat_prompt, preferred_system_prompt,
                           preferred_top_p, preferred_top_k, theme
                    FROM users
                    WHERE id = %s
                    """,
                    (user_id,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return {}
                
                # Return only non-null preferences
                return {k: v for k, v in dict(row).items() if v is not None}
        finally:
            self._release_connection(conn)
    
    def update_user_preferences(
        self,
        user_id: str,
        *,
        preferred_model: Optional[str] = None,
        preferred_temperature: Optional[float] = None,
        preferred_max_tokens: Optional[int] = None,
        preferred_num_documents: Optional[int] = None,
        preferred_condense_prompt: Optional[str] = None,
        preferred_chat_prompt: Optional[str] = None,
        preferred_system_prompt: Optional[str] = None,
        preferred_top_p: Optional[float] = None,
        preferred_top_k: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update user preferences.
        
        Args:
            user_id: The user ID
            **preferences: Preference values to update
            
        Returns:
            Updated preferences dict
        """
        updates = []
        params: List[Any] = []
        
        if preferred_model is not None:
            updates.append("preferred_model = %s")
            params.append(preferred_model if preferred_model else None)
        
        if preferred_temperature is not None:
            updates.append("preferred_temperature = %s")
            params.append(preferred_temperature)
        
        if preferred_max_tokens is not None:
            updates.append("preferred_max_tokens = %s")
            params.append(preferred_max_tokens)
        
        if preferred_num_documents is not None:
            updates.append("preferred_num_documents = %s")
            params.append(preferred_num_documents)
        
        if preferred_condense_prompt is not None:
            updates.append("preferred_condense_prompt = %s")
            params.append(preferred_condense_prompt if preferred_condense_prompt else None)
        
        if preferred_chat_prompt is not None:
            updates.append("preferred_chat_prompt = %s")
            params.append(preferred_chat_prompt if preferred_chat_prompt else None)
        
        if preferred_system_prompt is not None:
            updates.append("preferred_system_prompt = %s")
            params.append(preferred_system_prompt if preferred_system_prompt else None)
        
        if preferred_top_p is not None:
            updates.append("preferred_top_p = %s")
            params.append(preferred_top_p)
        
        if preferred_top_k is not None:
            updates.append("preferred_top_k = %s")
            params.append(preferred_top_k)
        
        if theme is not None:
            updates.append("theme = %s")
            params.append(theme)
        
        if not updates:
            return self.get_user_preferences(user_id)
        
        updates.append("updated_at = NOW()")
        params.append(user_id)
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE users
                    SET {', '.join(updates)}
                    WHERE id = %s
                    """,
                    params
                )
                conn.commit()
                
                return self.get_user_preferences(user_id)
        finally:
            self._release_connection(conn)
    
    # =========================================================================
    # Effective Configuration (User -> Dynamic -> Defaults)
    # =========================================================================
    
    # Mapping of effective field names to (dynamic_field, user_pref_field)
    _EFFECTIVE_FIELDS: Dict[str, tuple] = {
        "model": ("active_model", "preferred_model"),
        "active_model": ("active_model", "preferred_model"),
        "temperature": ("temperature", "preferred_temperature"),
        "max_tokens": ("max_tokens", "preferred_max_tokens"),
        "num_documents": ("num_documents_to_retrieve", "preferred_num_documents"),
        "num_documents_to_retrieve": ("num_documents_to_retrieve", "preferred_num_documents"),
        "condense_prompt": ("active_condense_prompt", "preferred_condense_prompt"),
        "chat_prompt": ("active_chat_prompt", "preferred_chat_prompt"),
        "system_prompt": ("active_system_prompt", "preferred_system_prompt"),
        "top_p": ("top_p", "preferred_top_p"),
        "top_k": ("top_k", "preferred_top_k"),
    }
    
    def get_effective(self, field: str, user_id: Optional[str] = None) -> Any:
        """
        Get the effective value for a configuration field.
        
        Resolution order:
        1. User preference (if user_id provided and preference set)
        2. Deployment dynamic config
        3. Default from DynamicConfig dataclass
        
        Args:
            field: Configuration field name (e.g., "temperature", "active_model")
            user_id: Optional user ID to check preferences
            
        Returns:
            The effective configuration value
            
        Raises:
            KeyError: If field is not recognized
        """
        if field not in self._EFFECTIVE_FIELDS:
            # For fields without user override, just return dynamic config value
            dynamic = self.get_dynamic_config()
            if hasattr(dynamic, field):
                return getattr(dynamic, field)
            raise KeyError(f"Unknown config field: {field}")
        
        dynamic_field, pref_field = self._EFFECTIVE_FIELDS[field]
        
        # Check user preference first
        if user_id:
            prefs = self.get_user_preferences(user_id)
            if pref_field in prefs and prefs[pref_field] is not None:
                return prefs[pref_field]
        
        # Fall back to dynamic config
        dynamic = self.get_dynamic_config()
        return getattr(dynamic, dynamic_field)
    
    def get_effective_config(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all effective configuration values for a user.
        
        Args:
            user_id: Optional user ID to include preferences
            
        Returns:
            Dict with effective values for all fields
        """
        dynamic = self.get_dynamic_config()
        prefs = self.get_user_preferences(user_id) if user_id else {}
        
        result = {
            "active_pipeline": dynamic.active_pipeline,
            "active_model": prefs.get("preferred_model") or dynamic.active_model,
            "temperature": prefs.get("preferred_temperature") if prefs.get("preferred_temperature") is not None else dynamic.temperature,
            "max_tokens": prefs.get("preferred_max_tokens") or dynamic.max_tokens,
            "top_p": prefs.get("preferred_top_p") if prefs.get("preferred_top_p") is not None else dynamic.top_p,
            "top_k": prefs.get("preferred_top_k") or dynamic.top_k,
            "repetition_penalty": dynamic.repetition_penalty,
            "system_prompt": dynamic.system_prompt,
            "condense_prompt": prefs.get("preferred_condense_prompt") or dynamic.active_condense_prompt,
            "chat_prompt": prefs.get("preferred_chat_prompt") or dynamic.active_chat_prompt,
            "num_documents_to_retrieve": prefs.get("preferred_num_documents") or dynamic.num_documents_to_retrieve,
            "use_hybrid_search": dynamic.use_hybrid_search,
            "bm25_weight": dynamic.bm25_weight,
            "semantic_weight": dynamic.semantic_weight,
            "verbosity": dynamic.verbosity,
        }
        
        return result
    
    # =========================================================================
    # Audit Logging
    # =========================================================================
    
    def _log_audit(
        self,
        user_id: str,
        config_type: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """
        Log a configuration change to the audit table.
        
        Args:
            user_id: User who made the change
            config_type: Type of config ('dynamic' or 'user_pref')
            field_name: Name of the field changed
            old_value: Previous value
            new_value: New value
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO config_audit (user_id, config_type, field_name, old_value, new_value)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, config_type, field_name, str(old_value) if old_value is not None else None, str(new_value) if new_value is not None else None)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log audit: {e}")
        finally:
            self._release_connection(conn)
    
    def get_audit_log(
        self,
        *,
        user_id: Optional[str] = None,
        config_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get configuration audit log entries.
        
        Args:
            user_id: Filter by user ID
            config_type: Filter by config type ('dynamic' or 'user_pref')
            limit: Maximum entries to return
            
        Returns:
            List of audit log entries
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                conditions = []
                params: List[Any] = []
                
                if user_id:
                    conditions.append("user_id = %s")
                    params.append(user_id)
                
                if config_type:
                    conditions.append("config_type = %s")
                    params.append(config_type)
                
                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                
                cursor.execute(
                    f"""
                    SELECT id, user_id, changed_at, config_type, field_name, old_value, new_value
                    FROM config_audit
                    {where_clause}
                    ORDER BY changed_at DESC
                    LIMIT %s
                    """,
                    params + [limit]
                )
                
                return [dict(row) for row in cursor.fetchall()]
        finally:
            self._release_connection(conn)
    
    # =========================================================================
    # Admin Check
    # =========================================================================
    
    def is_admin(self, user_id: str) -> bool:
        """
        Check if a user is an admin.
        
        Args:
            user_id: The user ID to check
            
        Returns:
            True if user is admin, False otherwise
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT is_admin FROM users WHERE id = %s",
                    (user_id,)
                )
                row = cursor.fetchone()
                return bool(row and row[0])
        finally:
            self._release_connection(conn)
