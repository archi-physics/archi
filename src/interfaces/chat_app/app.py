import json
import os
import random
import re
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional
from pathlib import Path
from urllib.parse import urlparse
from functools import wraps

import requests

import mistune as mt
import numpy as np
import psycopg2
import psycopg2.extras
import yaml
from authlib.integrations.flask_client import OAuth
from flask import jsonify, render_template, request, session, flash, redirect, url_for, Response, stream_with_context
from flask_cors import CORS
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import (BashLexer, CLexer, CppLexer, FortranLexer,
                             HtmlLexer, JavaLexer, JavascriptLexer, JuliaLexer,
                             MathematicaLexer, MatlabLexer, PythonLexer,
                             TypeScriptLexer)

from src.archi.archi import archi
from src.archi.pipelines.agents.agent_spec import (
    AgentSpec,
    AgentSpecError,
    list_agent_files,
    load_agent_spec,
    select_agent_spec,
    load_agent_spec_from_text,
    slugify_agent_name,
)
from src.archi.providers.base import ModelInfo, ProviderConfig, ProviderType
from src.archi.utils.output_dataclass import PipelineOutput
from src.evaluation.qa.console import EvaluationConsoleService
# from src.data_manager.data_manager import DataManager
from src.data_manager.data_viewer_service import DataViewerService
from src.data_manager.vectorstore.manager import VectorStoreManager
from src.utils.env import read_secret, read_or_create_persistent_secret
from src.utils.logging import get_logger
from src.utils.config_access import get_full_config, get_services_config, get_global_config, get_dynamic_config
from src.utils.config_service import ConfigService, StaticConfig
from src.utils.sql import (
    SQL_INSERT_CONVO, SQL_INSERT_FEEDBACK, SQL_INSERT_TIMING, SQL_QUERY_CONVO,
    SQL_CREATE_CONVERSATION, SQL_UPDATE_CONVERSATION_TIMESTAMP,
    SQL_LIST_CONVERSATIONS, SQL_GET_CONVERSATION_METADATA, SQL_DELETE_CONVERSATION,
    SQL_LIST_CONVERSATIONS_BY_USER, SQL_GET_CONVERSATION_METADATA_BY_USER,
    SQL_DELETE_CONVERSATION_BY_USER, SQL_UPDATE_CONVERSATION_TIMESTAMP_BY_USER,
    SQL_INSERT_TOOL_CALLS, SQL_QUERY_CONVO_WITH_FEEDBACK,
    SQL_QUERY_CONVO_WITH_FEEDBACK_NO_PLAYBOOKS, SQL_DELETE_REACTION_FEEDBACK,
    SQL_GET_REACTION_FEEDBACK,
    SQL_CREATE_AGENT_TRACE, SQL_UPDATE_AGENT_TRACE, SQL_GET_AGENT_TRACE,
    SQL_GET_TRACE_BY_MESSAGE, SQL_GET_ACTIVE_TRACE, SQL_CANCEL_ACTIVE_TRACES,
)
from src.utils.playbook_service import (
    PlaybookService, PlaybookNotFoundError,
    playbook_invocation_text, resolve_playbook_owner,
)
from src.archi.pipelines.agents.tools.playbook_tools import (
    set_playbook_owner, get_playbook_owner,
    set_pending_playbook, get_pending_playbook, clear_pending_playbook,
    classify_playbook_tool_result,
)
from src.interfaces.chat_app.document_utils import *
from src.interfaces.chat_app.evaluation_routes import register_evaluations
from src.interfaces.chat_app.playbook_routes import register_playbooks
from src.interfaces.chat_app.service_alerts import (
    register_service_alerts, get_active_banner_alerts, is_alert_manager,
)
from src.interfaces.chat_app.utils import collapse_assistant_sequences
from src.utils.ab_testing import (
    ABPool,
    ABPoolLoadState,
    ABVariant,
    ABPoolError,
    DEFAULT_DISCLOSURE_MODE,
    DEFAULT_TRACE_MODE,
    load_ab_pool_state,
    normalize_ab_disclosure_mode,
    normalize_ab_trace_mode,
    resolve_ab_agents_dir,
)
from src.interfaces.chat_app.event_formatter import (
    PipelineEventFormatter, build_playbook_applied_events, playbook_loads_from_events,
)
from src.utils.conversation_service import ConversationService
from src.utils.user_service import UserService
from src.utils.ab_agent_spec_service import ABAgentSpecService, ABAgentSpecRecord

# RBAC imports for role-based access control
from src.utils.rbac import (
    Permission,
    get_registry,
    get_user_roles,
    has_permission,
    get_user_permissions,
    require_permission,
    require_any_permission,
    require_authenticated,
)
from src.utils.rbac.permissions import get_permission_context, is_admin as rbac_is_admin
from src.utils.rbac.audit import log_authentication_event
from src.utils.rbac.jwt_parser import decode_jwt_claims


logger = get_logger(__name__)


def _static_config_to_full_config(
    static: StaticConfig,
    *,
    resolve_embeddings: bool = False,
    config_service: Optional[ConfigService] = None,
) -> Dict[str, Any]:
    """
    Build a full runtime config dict directly from a freshly loaded StaticConfig.

    This avoids routing post-write refreshes back through config_access helpers,
    which may read from a different cached ConfigService instance.
    """
    data_manager_config = dict(static.data_manager_config or {})
    if resolve_embeddings and config_service is not None:
        try:
            resolved_map = config_service.get_embedding_class_map(resolved=True)
            if resolved_map:
                data_manager_config["embedding_class_map"] = resolved_map
        except Exception:
            pass

    return {
        "name": static.deployment_name,
        "config_version": static.config_version,
        "global": static.global_config,
        "services": static.services_config,
        "data_manager": data_manager_config,
        "archi": static.archi_config,
        "sources": static.sources_config,
        "mcp_servers": static.mcp_servers_config or {},
        "available_pipelines": static.available_pipelines,
        "available_models": static.available_models,
        "available_providers": static.available_providers,
    }


def _build_provider_config_from_payload(config_payload: Dict[str, Any], provider_type: ProviderType) -> Optional[ProviderConfig]:
    """Helper to build ProviderConfig from loaded YAML for a provider."""
    services_cfg = config_payload.get("services", {}) or {}
    chat_cfg = services_cfg.get("chat_app", {}) or {}
    providers_cfg = chat_cfg.get("providers", {}) or {}
    cfg = providers_cfg.get(provider_type.value, {})
    if not cfg:
        return None

    models = [ModelInfo(id=m, name=m, display_name=m) for m in cfg.get("models", [])]
    extra = {}
    local_types = (ProviderType.LOCAL, ProviderType.OLLAMA, ProviderType.VLLM)
    if provider_type in local_types and cfg.get("mode"):
        extra["local_mode"] = cfg.get("mode")

    return ProviderConfig(
        provider_type=provider_type,
        enabled=cfg.get("enabled", True),
        base_url=cfg.get("base_url"),
        models=models,
        default_model=cfg.get("default_model"),
        extra_kwargs=extra,
    )


def _is_provider_enabled_in_config(
    config_payload: Dict[str, Any],
    provider_type: Optional[ProviderType] = None,
    provider_name: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Return whether a provider is explicitly enabled by chat_app config.

    Only explicit `enabled: false` inside `services.chat_app.providers.<provider>`
    disables request-time overrides. Missing provider blocks remain allowed for
    backward compatibility.

    Exactly one of `provider_type` or `provider_name` should be provided.
    Unknown provider names are treated as enabled here; other validation paths
    handle invalid provider types.
    """
    if provider_type is None and provider_name:
        try:
            provider_type = ProviderType(str(provider_name).lower())
        except ValueError:
            return True, None
    if provider_type is None:
        return True, None

    services_cfg = config_payload.get("services", {}) if isinstance(config_payload, dict) else {}
    chat_cfg = services_cfg.get("chat_app", {}) if isinstance(services_cfg, dict) else {}
    providers_cfg = chat_cfg.get("providers", {}) if isinstance(chat_cfg, dict) else {}
    provider_cfg = providers_cfg.get(provider_type.value, {})

    if isinstance(provider_cfg, dict) and provider_cfg.get("enabled") is False:
        return False, f"Provider '{provider_type.value}' is disabled in services.chat_app.providers.{provider_type.value}.enabled"
    return True, None


def _config_names():
    cfg = get_full_config()
    return [cfg.get("name", "default")]

# DEFINITIONS
QUERY_LIMIT = 10000 # max queries per conversation
MAIN_PROMPT_FILE = "/root/archi/main.prompt"
CONDENSE_PROMPT_FILE = "/root/archi/condense.prompt"
SUMMARY_PROMPT_FILE = "/root/archi/summary.prompt"
ARCHI_SENDER = "archi"
CLIENT_TIMEOUT_ERROR_MESSAGE = (
    "client timeout; the agent wasn't able to find satisfactory information "
    "to respond to the query within the time limit set by the administrator."
)


class AnswerRenderer(mt.HTMLRenderer):
    """
    Class for custom rendering of archi output. Child of mistune's HTMLRenderer, with custom overrides.
    Code blocks are structured and colored according to pygment lexers
    """
    RENDERING_LEXER_MAPPING = {
            "python": PythonLexer,
            "java": JavaLexer,
            "javascript": JavascriptLexer,
            "bash": BashLexer,
            "c++": CppLexer,
            "cpp": CppLexer,
            "c": CLexer,
            "typescript": TypeScriptLexer,
            "html": HtmlLexer,
            "fortran" : FortranLexer,
            "julia" : JuliaLexer,
            "mathematica" : MathematicaLexer,
            "matlab": MatlabLexer
        }

    def __init__(self):
        self.config = get_full_config()
        super().__init__()

    def block_code(self, code, info=None):
        # Handle code blocks (triple backticks)
        if info not in self.RENDERING_LEXER_MAPPING.keys(): info = 'bash' #defaults in bash
        code_block_highlighted = highlight(code.strip(), self.RENDERING_LEXER_MAPPING[info](stripall=True), HtmlFormatter())

        if self.config["services"]["chat_app"]["include_copy_button"]:
            button = """<button class="copy-code-btn" onclick="copyCode(this)"> Copy Code </button>"""
        else: button = ""

        return f"""<div class="code-box">
                <div class="code-box-header">
                <span>{info}</span>{button}
                </div>
                <div class="code-box-body">{code_block_highlighted}
                </div>
                </div>"""

    def codespan(self, text):
        # Handle inline code snippets (single backticks)
        return f"""<code class="code-snippet">{text}</code>"""


class ConversationAccessError(Exception):
    """Raised when a client attempts to access a conversation it does not own."""
    pass


@dataclass
class ChatRequestContext:
    sender: str
    content: str
    conversation_id: int
    history: List
    is_refresh: bool
    config_name: Optional[str] = None
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    pipeline_used: Optional[str] = None
    # Name of the user-invoked playbook applied to this turn (None if none). The clean
    # `content` is stored/titled; the playbook body is injected only into `history`.
    playbook_name: Optional[str] = None
    playbook_id: Optional[int] = None


def _pooled_playbook_service(pg_config) -> PlaybookService:
    """PlaybookService on the process-wide pooled factory when available.

    The agent path initializes PostgresServiceFactory at startup; reusing it
    avoids a fresh TCP connection per call. Falls back to direct connections
    when no factory exists (e.g. non-agent pipelines). Shared by ChatWrapper
    (chat-flow turn tracking) and FlaskAppWrapper (REST + staging) so neither
    class can lose the accessor again.
    """
    try:
        from src.utils.postgres_service_factory import PostgresServiceFactory
        factory = PostgresServiceFactory.get_instance()
        if factory is not None:
            return factory.playbook_service
    except Exception as exc:
        logger.debug("Pooled PlaybookService unavailable, using direct connections: %s", exc)
    return PlaybookService(pg_config=pg_config)


def _query_convo_history_rows(cursor, conversation_id):
    """History rows (sender, content, message_id, feedback, comment_count,
    model_used, playbook_name) for one conversation.

    Falls back to the no-playbooks variant when conversation_playbook_turns is
    missing (failed boot migration): chips degrade to absent instead of every
    conversation load returning 500.
    """
    try:
        cursor.execute(SQL_QUERY_CONVO_WITH_FEEDBACK, (conversation_id,))
    except psycopg2.errors.UndefinedTable:
        cursor.connection.rollback()  # leave the aborted transaction before retrying
        logger.warning(
            "conversation_playbook_turns missing; loading conversation %s without playbook chips",
            conversation_id,
        )
        cursor.execute(SQL_QUERY_CONVO_WITH_FEEDBACK_NO_PLAYBOOKS, (conversation_id,))
    return cursor.fetchall()


def _resolve_auto_playbook_invocation(tc):
    """Resolve (playbook_id, name, status) for a model-invoked Playbook tool call.

    Prefer the ToolMessage artifact: a successful load carries the resolved id +
    name (server-side only) and implies status 'ok'. Fall back to classifying the
    result string when no artifact rode along — every error return path of the tool
    yields artifact=None. `tc` is one entry from PipelineOutput.extract_tool_calls.
    """
    args = tc.get("args") or {}
    requested = args.get("playbook") if isinstance(args, dict) else None
    artifact = tc.get("artifact")
    if isinstance(artifact, dict) and artifact.get("kind") == "playbook":
        return artifact.get("playbook_id"), artifact.get("playbook_name") or requested, "ok"
    return None, requested, classify_playbook_tool_result(tc.get("result", ""))


class ChatWrapper:
    AUTO_SOURCE_SECTION_LABEL = "Retrieved documents"
    AUTO_SOURCE_SECTION_EXPLANATION = "These are the knowledge-base documents retrieved for this answer."
    _AUTO_SOURCE_SECTION_PATTERN = re.compile(
        r"(Show all sources|Retrieved documents|Sources cited in this answer)\s*\(\d+\)",
        flags=re.IGNORECASE,
    )

    """
    Wrapper which holds functionality for the chatbot
    """
    def __init__(self):
        # Threading lock for database operations
        self.lock = Lock()
        self._agent_refresh_lock = Lock()
        
        # load configs
        self.config = get_full_config()
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]
        self.data_path = self.global_config["DATA_PATH"]

        # store postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.config_service = ConfigService(pg_config=self.pg_config)


        # initialize data manager (ingestion handled by data-manager service)
        # self.data_manager = DataManager(run_ingestion=False)
        embedding_name = self.config["data_manager"]["embedding_name"]
        self.similarity_score_reference = self.config["data_manager"]["embedding_class_map"][embedding_name]["similarity_score_reference"]
        self.sources_config = self.config["data_manager"]["sources"]

        # initialize vectorstore manager for embedding uploads (needs class-mapped config)
        vectorstore_config = get_full_config(resolve_embeddings=True)
        self.vector_manager = VectorStoreManager(
            config=vectorstore_config,
            global_config=vectorstore_config["global"],
            data_path=self.data_path,
            pg_config=self.pg_config,
        )

        # initialize data viewer service for per-chat document selection
        self.data_viewer = DataViewerService(data_path=self.data_path, pg_config=self.pg_config)

        # shared conversation service for A/B comparisons & metrics
        self.conv_service = ConversationService(connection_params=self.pg_config)
        self.user_service = UserService(pg_config=self.pg_config)
        self.ab_agent_spec_service = ABAgentSpecService(pg_config=self.pg_config)

        self.conn = None
        self.cursor = None

        # initialize agent spec
        chat_cfg = self.services_config.get("chat_app", {})
        agents_dir = Path(chat_cfg.get("agents_dir", "/root/archi/agents"))
        self.current_agent_path = None
        self.current_agent_mtime = None
        try:
            dynamic = get_dynamic_config()
        except Exception:
            dynamic = None
        agent_name = getattr(dynamic, "active_agent_name", None) if dynamic else None
        try:
            self.agent_spec, self.current_agent_path = self._load_agent_spec_with_path(agents_dir, agent_name)
        except AgentSpecError as exc:
            logger.warning("Failed to load agent spec '%s': %s", agent_name, exc)
            self.agent_spec, self.current_agent_path = self._load_agent_spec_with_path(agents_dir, None)
        self.current_agent_name = getattr(self.agent_spec, "name", None)
        if self.current_agent_path and self.current_agent_path.exists():
            self.current_agent_mtime = self.current_agent_path.stat().st_mtime

        agent_class = self._get_agent_class_from_cfg(chat_cfg)
        if not agent_class:
            raise ValueError("services.chat_app.agent_class must be configured.")
        default_provider = chat_cfg.get("default_provider")
        is_enabled, disabled_reason = _is_provider_enabled_in_config(self.config, provider_name=default_provider)
        if not is_enabled:
            raise ValueError(
                f"services.chat_app.default_provider='{str(default_provider).lower()}' is invalid because it is disabled. "
                f"{disabled_reason}"
            )
        default_model = chat_cfg.get("default_model")
        prompt_overrides = chat_cfg.get("prompts", {})

        # initialize chain
        self.archi = archi(
            pipeline=agent_class,
            agent_spec=self.agent_spec,
            default_provider=default_provider,
            default_model=default_model,
            prompt_overrides=prompt_overrides,
        )
        self.number_of_queries = 0

        # track active config/model/pipeline state
        self.default_config_name = self.config.get("name")
        self.current_config_name = None
        self._config_cache = {}
        if self.default_config_name:
            self._config_cache[self.default_config_name] = self.config

        # activate default config
        if self.default_config_name:
            self.update_config(config_name=self.default_config_name)

        # A/B testing pool (loaded from config; None if not configured)
        self.refresh_ab_pool()

    def reload_static_state(self) -> None:
        """
        Reload static config snapshots used by the chat wrapper.

        This is primarily used after runtime updates to the persisted chat A/B
        configuration so the active process picks up the latest pool settings.
        """
        static = self.config_service.get_static_config(force_reload=True)
        if static is None:
            raise ValueError("Static config not initialized")
        self.config = _static_config_to_full_config(static, config_service=self.config_service)
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]
        self.data_path = self.global_config["DATA_PATH"]
        self.sources_config = self.config["data_manager"]["sources"]
        self.refresh_ab_pool()

    def refresh_ab_pool(self) -> None:
        import_diagnostics = self._sync_ab_agent_specs_from_filesystem()
        state = load_ab_pool_state(
            self.config,
            agent_spec_exists=self.ab_agent_spec_service.spec_exists,
        )
        warnings = list(import_diagnostics.get("warnings", []))
        warnings.extend(list(getattr(state, "warnings", []) or []))
        self.ab_agent_import_diagnostics = import_diagnostics
        self.ab_pool_state = ABPoolLoadState(
            pool=state.pool,
            warnings=warnings,
            enabled_requested=state.enabled_requested,
            agent_dir=state.agent_dir,
            agent_dir_configured=state.agent_dir_configured,
        )
        self.ab_pool = self.ab_pool_state.pool
        for warning in self.ab_pool_state.warnings:
            logger.warning("%s", warning)
        if self.ab_pool:
            logger.info(
                "A/B pool active: %d variants, champion='%s'",
                len(self.ab_pool.variants), self.ab_pool.champion_name,
            )

    def _get_ab_agents_dir(self) -> Path:
        chat_cfg = self.services_config.get("chat_app", {}) or {}
        path, _ = resolve_ab_agents_dir(chat_cfg)
        return path

    def _sync_ab_agent_specs_from_filesystem(self) -> Dict[str, Any]:
        """
        Import legacy A/B markdown specs into the DB-backed catalog.

        The database remains the runtime source of truth after import.
        """
        directory = self._get_ab_agents_dir()
        diagnostics: Dict[str, Any] = {
            "directory": str(directory),
            "source_exists": directory.exists() and directory.is_dir(),
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "conflicts": [],
            "staged_unresolved": [],
            "warnings": [],
        }
        try:
            result = self.ab_agent_spec_service.import_directory(
                directory,
                created_by="system",
            )
            diagnostics.update({
                "imported": int(result.get("imported", 0)),
                "updated": int(result.get("updated", 0)),
                "skipped": int(result.get("skipped", 0)),
                "conflicts": list(result.get("conflicts", []) or []),
            })
            if result["imported"] or result["updated"]:
                logger.info(
                    "Imported A/B agent specs into DB: imported=%d updated=%d skipped=%d",
                    result["imported"],
                    result["updated"],
                    result["skipped"],
                )
            for conflict in result["conflicts"]:
                logger.warning("A/B agent import conflict: %s", conflict)
        except Exception as exc:
            logger.warning("Failed to sync A/B agent specs from filesystem: %s", exc)
            diagnostics["conflicts"].append(str(exc))

        for conflict in diagnostics["conflicts"]:
            diagnostics["warnings"].append(f"A/B agent import conflict: {conflict}")

        chat_cfg = self.services_config.get("chat_app", {}) or {}
        ab_cfg = (chat_cfg.get("ab_testing") or {}) if isinstance(chat_cfg.get("ab_testing"), dict) else {}
        try:
            configured_pool = ABPool.from_config(ab_cfg) if ab_cfg.get("enabled") else None
        except ABPoolError:
            configured_pool = None

        if configured_pool:
            for variant in configured_pool.variants:
                if self.ab_agent_spec_service.spec_exists(variant.agent_spec):
                    continue
                disk_path = directory / variant.agent_spec
                if disk_path.exists():
                    diagnostics["staged_unresolved"].append(variant.agent_spec)

        if diagnostics["staged_unresolved"]:
            unresolved = sorted(set(diagnostics["staged_unresolved"]))
            diagnostics["warnings"].append(
                "A/B agent specs are present in the staged import directory but unresolved in PostgreSQL after import: "
                f"{unresolved}."
            )

        return diagnostics

    @staticmethod
    def _variant_with_spec_record(variant: "ABVariant", record: ABAgentSpecRecord) -> ABVariant:
        return ABVariant(
            label=variant.label,
            agent_spec=record.filename,
            provider=variant.provider,
            model=variant.model,
            num_documents_to_retrieve=variant.num_documents_to_retrieve,
            recursion_limit=variant.recursion_limit,
            agent_spec_id=record.spec_id,
            agent_spec_name=record.name,
            agent_spec_version_id=record.version_id,
            agent_spec_version_number=record.version_number,
            agent_spec_content_hash=record.content_hash,
            agent_spec_tools=list(record.tools),
            agent_spec_prompt_hash=record.prompt_hash,
        )

    def _resolve_runtime_ab_variant(self, variant: "ABVariant") -> tuple["ABVariant", AgentSpec]:
        record = self.ab_agent_spec_service.load_agent_spec(variant.agent_spec)
        resolved = self._variant_with_spec_record(variant, record)
        return resolved, record.to_agent_spec()

    def update_config(self, config_name=None):
        """
        Update the active config and apply it to the pipeline.
        Tracks model_used and pipeline_used for conversation storage.
        """
        target_config_name = config_name or self.current_config_name or self.default_config_name
        if not target_config_name:
            raise ValueError("Config name must be provided to update the chat configuration.")

        config_payload = self._get_config_payload(target_config_name)
        chat_cfg = config_payload["services"]["chat_app"]

        try:
            dynamic = get_dynamic_config()
        except Exception:
            dynamic = None
        desired_agent_name = getattr(dynamic, "active_agent_name", None) if dynamic else None
        agent_changed = False
        agents_dir = Path(chat_cfg.get("agents_dir", "/root/archi/agents"))
        with self._agent_refresh_lock:
            spec_path = self.current_agent_path
            spec_mtime = None
            if spec_path and spec_path.exists():
                spec_mtime = spec_path.stat().st_mtime
            needs_reload = spec_mtime and self.current_agent_mtime and spec_mtime != self.current_agent_mtime
            if desired_agent_name and desired_agent_name != self.current_agent_name:
                needs_reload = True
            if needs_reload or self.agent_spec is None:
                try:
                    self.agent_spec, self.current_agent_path = self._load_agent_spec_with_path(agents_dir, desired_agent_name)
                    self.current_agent_name = getattr(self.agent_spec, "name", None)
                    if self.current_agent_path and self.current_agent_path.exists():
                        self.current_agent_mtime = self.current_agent_path.stat().st_mtime
                    self.archi.pipeline_kwargs["agent_spec"] = self.agent_spec
                    agent_changed = True
                except AgentSpecError as exc:
                    logger.warning("Active agent '%s' not found: %s", desired_agent_name, exc)

        if self.current_config_name == target_config_name and not agent_changed:
            return

        agent_class = self._get_agent_class_from_cfg(chat_cfg)
        if not agent_class:
            raise ValueError("services.chat_app.agent_class must be configured.")
        is_enabled, disabled_reason = _is_provider_enabled_in_config(
            config_payload, provider_name=chat_cfg.get("default_provider")
        )
        if not is_enabled:
            default_provider = str(chat_cfg.get("default_provider")).lower()
            raise ValueError(
                f"services.chat_app.default_provider='{default_provider}' is invalid because it is disabled. "
                f"{disabled_reason}"
            )

        model_name = self._extract_model_name(config_payload)
        
        self.current_config_name = target_config_name
        self.archi.update(pipeline=agent_class, config_name=target_config_name)

    def _extract_model_name(self, config_payload):
        """Extract the primary model name from config for the chat service."""
        try:
            chat_cfg = config_payload.get("services", {}).get("chat_app", {})
            provider = chat_cfg.get("default_provider")
            model = chat_cfg.get("default_model")
            if provider and model:
                return f"{provider}/{model}"
        except Exception:
            pass
        return None

    def _get_config_payload(self, config_name):
        if config_name not in self._config_cache:
            self._config_cache[config_name] = get_full_config()
        return self._config_cache[config_name]

    def _load_agent_spec_with_path(self, agents_dir: Path, agent_name: Optional[str]):
        agent_files = list_agent_files(agents_dir)
        if not agent_files:
            raise AgentSpecError(f"No agent markdown files found in {agents_dir}")
        if agent_name:
            for path in agent_files:
                try:
                    spec = load_agent_spec(path)
                except AgentSpecError:
                    continue
                if spec.name == agent_name:
                    return spec, path
            raise AgentSpecError(f"Agent name '{agent_name}' not found in {agents_dir}")
        path = agent_files[0]
        for path in agent_files:
            try:
                return load_agent_spec(path), path
            except AgentSpecError:
                continue
        raise AgentSpecError(f"No valid agent specs found in {agents_dir}")

    @staticmethod
    def convert_to_app_history(history):
        """
        Input: the history in the form of a list of tuples, where the first entry of each tuple is
        the author of the text and the second entry is the text itself (native archi history format)

        Output: the history in the form of a list of lists, where the first entry of each tuple is
        the author of the text and the second entry is the text itself
        """
        return [list(entry) for entry in history]


    @staticmethod
    def format_code_in_text(text):
        """
        Takes in input plain text (the output from archi);
        Recognizes structures in canonical Markdown format, and processes according to the custom renderer;
        Returns it formatted in HTML
        """

        enabled_plugins = ['table']
        markdown = mt.create_markdown(renderer=AnswerRenderer(), plugins=enabled_plugins)
        try:
            return markdown(text)
        except:
             logger.info("Rendering error: markdown formatting failed")
             return text

    def get_top_sources(self, documents, scores):
        """
        Build a de-duplicated list of reference entries (link or ticket id).
        """
        if scores:
            sorted_indices = np.argsort(scores)
            scores = [scores[i] for i in sorted_indices]
            documents = [documents[i] for i in sorted_indices]

        top_sources = []
        seen_refs = set()
        pairs = zip(scores, documents) if scores else ((None, doc) for doc in documents)

        for score, document in pairs:
            # Skip threshold filtering for placeholder scores (-1)
            # Otherwise, filter out documents with score > threshold
            if score is not None and score != -1.0 and score > self.similarity_score_reference:
                logger.debug(f"Skipping document with score {score} above threshold {self.similarity_score_reference}")
                break

            metadata = document.metadata or {}

            display_name = self._get_display_name(metadata)
            if not display_name:
                continue

            if not self._get_doc_visibility(self, metadata):
                logger.debug(f"Document {display_name} marked as not visible; skipping.")
                continue

            link = self._extract_link(metadata)

            if display_name in seen_refs:
                continue
            seen_refs.add(display_name)

            top_sources.append(
                {
                    "link": link,
                    "display": display_name,
                    "score": score if score is not None else "N/A",
                }
            )

        logger.debug(f"Top sources: {top_sources}")
        return top_sources

    @staticmethod
    def _format_source_entry(entry):
        score_str = ChatWrapper._format_score_str(entry["score"])
        link = entry["link"]
        display_name = entry["display"]

        if link:
            return f"- [{display_name}]({link}){score_str}\n"
        return f"- {display_name}{score_str}\n"

    @staticmethod
    def format_links(top_sources):
        _output = ""
        if not top_sources:
            return _output

        _output += '''
        <div style="
            margin-top: 1.5em;
            padding-top: 0.5em;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.75em;
            color: #adb5bd;
            line-height: 1.3;
        ">
        '''

        def _entry_html(entry):
            score_str = ChatWrapper._format_score_str(entry["score"]).strip()
            link = entry["link"]
            display_name = entry["display"]

            if link:
                reference_html = f"<a href=\"{link}\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"color: #66b3ff; text-decoration: none;\" onmouseover=\"this.style.textDecoration='underline'\" onmouseout=\"this.style.textDecoration='none'\">{display_name}</a>"
            else:
                reference_html = f"<span style=\"color: #66b3ff;\">{display_name}</span>"

            return f'''
                <div style="margin: 0.15em 0; display: flex; align-items: center; gap: 0.4em;">
                    <span>•</span>
                    {reference_html}
                    <span style="color: #6c757d; font-size: 0.9em;">{score_str}</span>
                </div>
            '''

        _output += (
            f'<div style="margin: 0.4em 0 0.3em 0;">'
            f'{ChatWrapper.AUTO_SOURCE_SECTION_EXPLANATION}'
            f'</div>'
        )
        _output += (
            f'<details style="margin-top: 0.4em;"><summary style="cursor: pointer; color: #66b3ff; '
            f'font-weight: 700;">{ChatWrapper.AUTO_SOURCE_SECTION_LABEL} ({len(top_sources)})</summary>'
        )
        for entry in top_sources:
            _output += _entry_html(entry)
        _output += '</details>'

        _output += '</div>'
        return _output

    @staticmethod
    def format_links_markdown(top_sources):
        """Format source links as markdown (for client-side rendering)."""
        if not top_sources:
            return ""

        _output = (
            "\n\n---\n"
            f"*{ChatWrapper.AUTO_SOURCE_SECTION_EXPLANATION}*\n\n"
            f"<details><summary><strong>{ChatWrapper.AUTO_SOURCE_SECTION_LABEL} ({len(top_sources)})</strong></summary>\n\n"
        )
        for entry in top_sources:
            _output += ChatWrapper._format_source_entry(entry)
        _output += "\n</details>\n"

        return _output

    @classmethod
    def _contains_source_section(cls, output: str) -> bool:
        return bool(output and cls._AUTO_SOURCE_SECTION_PATTERN.search(output))

    @classmethod
    def append_source_section(cls, output: str, top_sources, *, render_markdown: bool) -> str:
        if not top_sources or cls._contains_source_section(output):
            return output
        if render_markdown:
            return output + cls.format_links(top_sources)
        return output + cls.format_links_markdown(top_sources)

    @staticmethod
    def _looks_like_url(value: str | None) -> bool:
        return isinstance(value, str) and value.startswith(("http://", "https://"))

    @staticmethod
    def _get_display_name(metadata: dict) -> str | None:
        display_name = metadata.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            return display_name.strip()
        else:
            logger.error("display_name is not a valid non-empty string in metadata")
            logger.error(f"Metadata content: {metadata}")
            return None

    @staticmethod
    def _get_title(metadata: dict) -> str | None:
        title = metadata.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
        else:
            logger.error("title is not a valid non-empty string in metadata")
            logger.error(f"Metadata content: {metadata}")
            return None

    @staticmethod
    def _get_doc_visibility(self, metadata: dict) -> bool:
        """
        From the metadata, check the source type.
        From the config, check if the source type is visible or not.
        """
        source_type = metadata.get("source_type")
        if not source_type:
            return True  # default to True if not specified

        if source_type not in self.sources_config:
            logger.error(f"Source type {source_type} not found in config, defaulting to visible")
            return True
        return bool(self.sources_config[source_type].get("visible", True))

    @staticmethod
    def _extract_link(metadata: dict) -> str | None:
        for key in ("url", "link", "href"):
            candidate = metadata.get(key)
            if ChatWrapper._looks_like_url(candidate):
                return candidate
        return None

    def insert_feedback(self, feedback):
        """
        Insert feedback from user for specific message into feedback table.
        """
        # construct insert_tup (mid, feedback_ts, feedback, feedback_msg, incorrect, unhelpful, inappropriate)
        insert_tup = (
            feedback['message_id'],
            feedback['feedback_ts'],
            feedback['feedback'],
            feedback['feedback_msg'],
            feedback['incorrect'],
            feedback['unhelpful'],
            feedback['inappropriate'],
        )

        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        self.cursor.execute(SQL_INSERT_FEEDBACK, insert_tup)
        self.conn.commit()

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

    def delete_reaction_feedback(self, message_id: int):
        """
        Remove existing like/dislike records for a message so only one reaction is stored.
        """
        if message_id is None:
            return
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        self.cursor.execute(SQL_DELETE_REACTION_FEEDBACK, (message_id,))
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

    def get_reaction_feedback(self, message_id: int):
        """
        Get the current reaction (like/dislike) for a message.
        Returns 'like', 'dislike', or None.
        """
        if message_id is None:
            return None
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        self.cursor.execute(SQL_GET_REACTION_FEEDBACK, (message_id,))
        row = self.cursor.fetchone()
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None
        return row[0] if row else None

    # =========================================================================
    # Agent Trace Methods
    # =========================================================================

    def create_agent_trace(
        self,
        conversation_id: int,
        user_message_id: int,
        config_id: Optional[int] = None,
        pipeline_name: Optional[str] = None,
    ) -> str:
        """
        Create a new agent trace record for tracking execution.
        
        Returns:
            The trace_id (UUID string) of the newly created trace
        """
        trace_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(
                SQL_CREATE_AGENT_TRACE,
                (trace_id, conversation_id, None, user_message_id,
                 config_id, pipeline_name, json.dumps([]), started_at, 'running')
            )
            conn.commit()
            logger.info(f"Created agent trace {trace_id} for conversation {conversation_id}")
            return trace_id
        finally:
            cursor.close()
            conn.close()

    def update_agent_trace(
        self,
        trace_id: str,
        events: List[Dict[str, Any]],
        status: str = 'running',
        message_id: Optional[int] = None,
        total_tool_calls: Optional[int] = None,
        total_duration_ms: Optional[int] = None,
        cancelled_by: Optional[str] = None,
        cancellation_reason: Optional[str] = None,
    ) -> None:
        """
        Update an agent trace with new events and/or status.
        """
        completed_at = datetime.now(timezone.utc) if status in ('completed', 'cancelled', 'error') else None
        
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(
                SQL_UPDATE_AGENT_TRACE,
                (json.dumps(events), completed_at, status, message_id,
                 total_tool_calls, total_duration_ms, cancelled_by, cancellation_reason,
                 trace_id)
            )
            conn.commit()
            logger.debug(f"Updated agent trace {trace_id}: status={status}")
        finally:
            cursor.close()
            conn.close()

    def get_agent_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an agent trace by ID.
        
        Returns:
            Dict with trace data or None if not found
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_GET_AGENT_TRACE, (trace_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._trace_from_row(row)
        finally:
            cursor.close()
            conn.close()

    def get_trace_by_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        """
        Get agent trace by the final message ID.
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_GET_TRACE_BY_MESSAGE, (message_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._trace_from_row(row)
        finally:
            cursor.close()
            conn.close()

    def get_active_trace(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the currently running trace for a conversation, if any.
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_GET_ACTIVE_TRACE, (conversation_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._trace_from_row(row)
        finally:
            cursor.close()
            conn.close()

    def cancel_active_traces(
        self,
        conversation_id: int,
        cancelled_by: str = 'user',
        cancellation_reason: Optional[str] = None,
    ) -> int:
        """
        Cancel all running traces for a conversation.
        
        Returns:
            Number of traces cancelled
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(
                SQL_CANCEL_ACTIVE_TRACES,
                (datetime.now(timezone.utc), cancelled_by, cancellation_reason, conversation_id)
            )
            count = cursor.rowcount
            conn.commit()
            if count > 0:
                logger.info(f"Cancelled {count} active traces for conversation {conversation_id}")
            return count
        finally:
            cursor.close()
            conn.close()


    def query_conversation_history(self, conversation_id, client_id, user_id: Optional[str] = None):
        """
        Return the conversation history as an ordered list of tuples. The order
        is determined by ascending message_id. Each tuple contains the sender and
        the message content
        """
        # create connection to database (use local vars for thread safety)
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()

        # ensure conversation belongs to user/client before querying
        if user_id:
            cursor.execute(SQL_GET_CONVERSATION_METADATA_BY_USER, (conversation_id, user_id, client_id))
        else:
            cursor.execute(SQL_GET_CONVERSATION_METADATA, (conversation_id, client_id))
        metadata = cursor.fetchone()
        if metadata is None:
            cursor.close()
            conn.close()
            raise ConversationAccessError("Conversation does not exist for this client")

        # query conversation history
        cursor.execute(SQL_QUERY_CONVO, (conversation_id,))
        history_rows = cursor.fetchall()
        comparisons = self.conv_service.get_conversation_ab_comparisons(str(conversation_id))
        suppressed_ids = self._suppressed_ab_message_ids(comparisons)
        if suppressed_ids:
            history_rows = [row for row in history_rows if row[2] not in suppressed_ids]
        history_rows = collapse_assistant_sequences(history_rows, sender_name=ARCHI_SENDER)
        history = [(row[0], row[1]) for row in history_rows]

        # clean up database connection state
        cursor.close()
        conn.close()

        return history

    def create_conversation(self, first_message: str, client_id: str, user_id: Optional[str] = None) -> int:
        """
        Gets first message (activates a new conversation), and generates a title w/ first msg.
        (TODO: commercial ones use one-sentence summarizer to make the title)

        Returns: Conversation ID.

        """
        service = "Chatbot"
        title = first_message[:20] + ("..." if len(first_message) > 20 else "")
        now = datetime.now(timezone.utc)
        
        version = os.getenv("APP_VERSION", "unknown")

        # title, created_at, last_message_at, client_id, version, user_id
        insert_tup = (title, now, now, client_id, version, user_id)

        # create connection to database (use local vars for thread safety)
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        cursor.execute(SQL_CREATE_CONVERSATION, insert_tup)
        conversation_id = cursor.fetchone()[0]
        conn.commit()

        # clean up database connection state
        cursor.close()
        conn.close()

        logger.info(f"Created new conversation with ID: {conversation_id}")
        return conversation_id

    def update_conversation_timestamp(self, conversation_id: int, client_id: str, user_id: Optional[str] = None):
        """
        Update the last_message_at timestamp for a conversation.
        last_message_at is used to reorder conversations in the UI (on vertical sidebar).
        """
        now = datetime.now(timezone.utc)

        # create connection to database (use local vars for thread safety)
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()

        # update timestamp
        if user_id:
            cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP_BY_USER, (now, conversation_id, user_id, client_id))
        else:
            cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (now, conversation_id, client_id))
        conn.commit()

        # clean up database connection state
        cursor.close()
        conn.close()

    def prepare_context_for_storage(self, source_documents, scores):
        scores = scores or []
        num_retrieved_docs = len(source_documents)
        context = ""
        if num_retrieved_docs > 0:
            for k in range(num_retrieved_docs):
                document = source_documents[k]
                metadata = document.metadata or {}
                link_k = self._extract_link(metadata)
                if not link_k:
                    link_k = (
                        self._get_display_name(metadata)
                        or self._get_title(metadata)
                        or "link not available"
                    )
                multiple_newlines = r'\n{2,}'
                content = re.sub(multiple_newlines, '\n', document.page_content)
                # Safely get the score, use "N/A" if index is out of range
                score_display = scores[k] if k < len(scores) else "N/A"
                context += f"SOURCE {k+1}: {metadata.get('title', 'No Title')} ({link_k})\nSIMILARITY SCORE: {score_display}\n\n{content}\n\n\n\n"

        return context

    def _playbook_svc(self) -> PlaybookService:
        """Playbook service for chat-flow turn tracking (pooled when possible).

        Regression guard: the chat flow calls this on ChatWrapper — it must
        exist HERE, not only on FlaskAppWrapper (review run 3, finding 1).
        """
        return _pooled_playbook_service(self.pg_config)

    def _insert_conversation_rows(self, insert_tups) -> List[int]:
        """execute_values SQL_INSERT_CONVO rows and return the new message_ids
        (one per row, in order). Shared by the normal and A/B store paths so
        the insert + id-extraction logic cannot drift between them."""
        # use local vars for thread safety; close even if the insert raises
        conn = psycopg2.connect(**self.pg_config)
        try:
            cursor = conn.cursor()
            psycopg2.extras.execute_values(cursor, SQL_INSERT_CONVO, insert_tups)
            conn.commit()
            message_ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
        finally:
            conn.close()
        return message_ids

    def _record_playbook_turn_best_effort(self, message_id, context) -> None:
        """Side-table write for a /name turn — must never break the message
        insert itself (the service raises; a missing side table degrades)."""
        if not (message_id and context.playbook_name):
            return
        try:
            self._playbook_svc().record_playbook_turn(
                message_id, context.playbook_name, context.playbook_id)
        except Exception as exc:
            logger.warning("Could not record playbook turn for message %s: %s", message_id, exc)
        # Unified ledger: also log this explicit /name use to playbook_invocations
        # (one honest row per use across both sources). Independent best-effort so a
        # ledger failure never affects the chip side-table write above. Regenerate
        # turns are excluded upstream (this runs only when not is_refresh), matching
        # the side table — a known, deliberate limitation.
        try:
            self._playbook_svc().record_invocation(
                getattr(context, "conversation_id", None), message_id,
                context.playbook_id, context.playbook_name, source="explicit", status="ok")
        except Exception as exc:
            logger.warning("Could not record playbook invocation for message %s: %s", message_id, exc)

    def insert_conversation(self, conversation_id, user_message, archi_message, link, archi_context, context:ChatRequestContext, is_refresh=False) -> List[int]:
        """
        """
        logger.debug("Entered insert_conversation.")

        def _sanitize(text: str) -> str:
            return text.replace("\x00", "") if isinstance(text, str) else text

        service = "Chatbot"
        # parse user message / archi message
        user_sender, user_content, user_msg_ts = user_message
        ARCHI_SENDER, archi_content, archi_msg_ts = archi_message

        user_content = _sanitize(user_content)
        archi_content = _sanitize(archi_content)
        link = _sanitize(link)
        model_provider = f"{context.provider_used}/{context.model_used}"
        pipeline_used = type(context.pipeline_used).__name__
        archi_context = _sanitize(archi_context)

        # construct insert_tups with model_used, pipeline_used only (9 shared columns).
        # Format: (service, conversation_id, sender, content, link, context, ts, model_used, pipeline_used)
        # Playbook tracking goes to the side table via PlaybookService.record_playbook_turn, not here.
        insert_tups = (
            [
                (service, conversation_id, user_sender, user_content, '', '', user_msg_ts, model_provider, pipeline_used),
                (service, conversation_id, ARCHI_SENDER, archi_content, link, archi_context, archi_msg_ts, model_provider, pipeline_used),
            ]
            if not is_refresh
            else [
                (service, conversation_id, ARCHI_SENDER, archi_content, link, archi_context, archi_msg_ts, model_provider, pipeline_used),
            ]
        )

        message_ids = self._insert_conversation_rows(insert_tups)

        if not is_refresh:
            self._record_playbook_turn_best_effort(
                message_ids[0] if message_ids else None, context)

        return message_ids

    def insert_timing(self, message_id, timestamps):
        """
        Store timing info to understand response profile.
        """
        logger.debug("Entered insert_timing.")

        # construct insert_tup
        insert_tup = (
            message_id,
            timestamps['client_sent_msg_ts'],
            timestamps['server_received_msg_ts'],
            timestamps['lock_acquisition_ts'],
            timestamps['vectorstore_update_ts'],
            timestamps['query_convo_history_ts'],
            timestamps['chain_finished_ts'],
            timestamps['archi_message_ts'],
            timestamps['insert_convo_ts'],
            timestamps['finish_call_ts'],
            timestamps['server_response_msg_ts'],
            timestamps['server_response_msg_ts'] - timestamps['server_received_msg_ts']
        )

        # create connection to database (use local vars for thread safety)
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        cursor.execute(SQL_INSERT_TIMING, insert_tup)
        conn.commit()

        # clean up database connection state
        cursor.close()
        conn.close()

    def insert_tool_calls_from_output(self, conversation_id: int, message_id: int, output: PipelineOutput) -> None:
        """
        Extract and store agent tool calls from the pipeline output.

        AIMessage with tool_calls contains the tool name, args, and timestamp.
        ToolMessage contains the result, matched by tool_call_id.
        """
        if not output or not output.messages:
            return

        tool_calls = output.extract_tool_calls()
        if not tool_calls:
            return

        tool_call_timestamps: Dict[str, datetime] = {}
        for msg in output.messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                response_metadata = getattr(msg, "response_metadata", {}) or {}
                created_at = response_metadata.get("created_at")
                if created_at:
                    try:
                        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        ts = datetime.now(timezone.utc)
                else:
                    ts = datetime.now(timezone.utc)

                for tc in msg.tool_calls:
                    tool_call_id = tc.get("id", "")
                    if tool_call_id and tool_call_id not in tool_call_timestamps:
                        tool_call_timestamps[tool_call_id] = ts

        insert_tups = []
        step_number = 0
        auto_invocations = []  # (playbook_id, name, status) for model-invoked Playbook loads
        for tc in tool_calls:
            step_number += 1
            tool_call_id = tc.get("id", "")
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})
            tool_result = tc.get("result", "")
            if tool_name == "Playbook":
                auto_invocations.append(_resolve_auto_playbook_invocation(tc))
            if len(tool_result) > 500:
                tool_result = tool_result[:500] + "..."
            ts = tool_call_timestamps.get(tool_call_id, datetime.now(timezone.utc))

            insert_tups.append((
                conversation_id,
                message_id,
                step_number,
                tool_name,
                json.dumps(tool_args) if tool_args else None,
                tool_result,
                ts,
            ))

        logger.debug("Inserting %d tool calls for message %d", len(insert_tups), message_id)

        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        psycopg2.extras.execute_values(cursor, SQL_INSERT_TOOL_CALLS, insert_tups)
        conn.commit()

        cursor.close()
        conn.close()

        # Unified ledger: one auto row per model-invoked Playbook load (best-effort,
        # after the agent_tool_calls write so a ledger failure can't lose tool rows).
        for pb_id, pb_name, pb_status in auto_invocations:
            try:
                self._playbook_svc().record_invocation(
                    conversation_id, message_id, pb_id, pb_name, source="auto", status=pb_status)
            except Exception as exc:
                logger.warning(
                    "Could not record auto playbook invocation for message %s: %s", message_id, exc)

    def _init_timestamps(self) -> Dict[str, datetime]:
        return {
            "lock_acquisition_ts": datetime.now(timezone.utc),
            "vectorstore_update_ts": datetime.now(timezone.utc),
        }

    def _resolve_config_name(self, config_name: Optional[str]) -> str:
        return config_name or self.current_config_name or self.default_config_name

    def _create_provider_llm(self, provider: str, model: str, api_key: str = None):
        """
        Create a LangChain chat model using the provider abstraction layer.
        
        Args:
            provider: Provider type (openai, anthropic, gemini, openrouter, local)
            model: Model ID/name to use
            api_key: Optional API key (overrides environment variable)
        
        Returns:
            A LangChain BaseChatModel instance, or None if creation fails
        """
        try:
            from src.archi.providers import get_provider

            provider_type = ProviderType(provider)
            is_enabled, disabled_reason = _is_provider_enabled_in_config(self.config, provider_type)
            if not is_enabled:
                raise ValueError(disabled_reason or f"Provider '{provider}' is disabled by configuration")

            # Build provider config from YAML so base_url/mode/default_model are respected
            cfg = _build_provider_config_from_payload(self.config, provider_type)
            provider_instance = get_provider(provider, config=cfg, use_cache=False) if cfg else get_provider(provider)
            if api_key:
                provider_instance.set_api_key(api_key)
            return provider_instance.get_chat_model(model)
        except ImportError as e:
            logger.warning(f"Providers module not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to create provider LLM {provider}/{model}: {e}")
            raise

    def _create_variant_archi(
        self,
        variant: "ABVariant",
        *,
        variant_agent_spec: Optional[AgentSpec] = None,
        request_provider: Optional[str] = None,
        request_model: Optional[str] = None,
        request_provider_api_key: Optional[str] = None,
    ) -> "archi":
        """
        Build a temporary archi instance configured for a specific A/B variant.

        Uses the deployment defaults for provider/model unless the variant overrides
        them, but always requires an explicit variant agent spec.
        """
        chat_cfg = self.services_config.get("chat_app", {})

        spec_name = (variant.agent_spec or "").strip()
        if not spec_name:
            raise ABPoolError(f"Variant '{variant.label}' is missing required agent_spec.")
        if Path(spec_name).name != spec_name:
            raise ABPoolError(
                f"Variant '{variant.label}' must use an agent_spec filename in the A/B catalog, got '{spec_name}'."
            )
        if variant_agent_spec is None:
            record = self.ab_agent_spec_service.load_agent_spec(spec_name)
            variant_agent_spec = record.to_agent_spec()

        agent_class = self._get_agent_class_from_cfg(chat_cfg)
        default_provider = variant.provider or request_provider or chat_cfg.get("default_provider")
        default_model = variant.model or request_model or chat_cfg.get("default_model")
        prompt_overrides = chat_cfg.get("prompts", {})

        variant_archi = archi(
            pipeline=agent_class,
            agent_spec=variant_agent_spec,
            default_provider=default_provider,
            default_model=default_model,
            prompt_overrides=prompt_overrides,
        )

        if (
            request_provider_api_key
            and default_provider
            and default_model
            and default_provider == request_provider
            and hasattr(variant_archi, 'pipeline')
        ):
            override_llm = self._create_provider_llm(
                default_provider,
                default_model,
                request_provider_api_key,
            )
            if override_llm and hasattr(variant_archi.pipeline, 'agent_llm'):
                variant_archi.pipeline.agent_llm = override_llm
                if hasattr(variant_archi.pipeline, 'refresh_agent'):
                    variant_archi.pipeline.refresh_agent(force=True)

        # Apply retriever overrides if specified
        if variant.num_documents_to_retrieve is not None and hasattr(variant_archi, 'pipeline'):
            pipeline = variant_archi.pipeline
            if hasattr(pipeline, 'pipeline_config'):
                pipeline.pipeline_config['num_documents_to_retrieve'] = variant.num_documents_to_retrieve

        if variant.recursion_limit is not None and hasattr(variant_archi, 'pipeline'):
            pipeline = variant_archi.pipeline
            if hasattr(pipeline, 'recursion_limit'):
                pipeline.recursion_limit = variant.recursion_limit
            elif hasattr(pipeline, 'pipeline_config'):
                pipeline.pipeline_config['recursion_limit'] = variant.recursion_limit

        return variant_archi

    @staticmethod
    def _comparison_canonical_message_id(comparison) -> Optional[int]:
        preference = getattr(comparison, "preference", None)
        if preference == "b":
            return getattr(comparison, "response_b_mid", None)
        if preference in ("a", "tie", "skip"):
            return getattr(comparison, "response_a_mid", None)
        return None

    @classmethod
    def _suppressed_ab_message_ids(cls, comparisons: List[Any]) -> set:
        suppressed: set = set()
        for comparison in comparisons or []:
            preference = getattr(comparison, "preference", None)
            a_mid = getattr(comparison, "response_a_mid", None)
            b_mid = getattr(comparison, "response_b_mid", None)
            if preference is None:
                if a_mid:
                    suppressed.add(a_mid)
                if b_mid:
                    suppressed.add(b_mid)
                continue

            canonical_mid = cls._comparison_canonical_message_id(comparison)
            for mid in (a_mid, b_mid):
                if mid and mid != canonical_mid:
                    suppressed.add(mid)
        return suppressed

    def _prepare_chat_context(
        self,
        message: List[str],
        conversation_id: Optional[str],
        client_id: str,
        is_refresh: bool,
        server_received_msg_ts: datetime,
        client_sent_msg_ts: float,
        client_timeout: float,
        timestamps: Dict[str, datetime],
        config_name: str,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        pipeline: Optional[str] = None
    ) -> tuple[Optional[ChatRequestContext], Optional[int]]:
        if not client_id:
            raise ValueError("client_id is required to process chat messages")
        sender, content = tuple(message[0])

        # `content` stays the clean user text — it titles the conversation and is what
        # we persist/display. A /name-invoked playbook turn expands only into
        # `agent_content` (the command block + playbook body), which feeds the agent via
        # history; the playbook name rides on the context so the stored message can show a chip.
        pending = get_pending_playbook()
        playbook_name = pending["name"] if pending else None
        playbook_id = pending["playbook_id"] if pending else None
        agent_content = (
            playbook_invocation_text(
                content, pending["name"], pending["body"], pending.get("foreign", False)
            )
            if pending else content
        )

        if conversation_id is None:
            conversation_id = self.create_conversation(content, client_id, user_id)
            history = []
        else:
            history = self.query_conversation_history(conversation_id, client_id, user_id)
            self.update_conversation_timestamp(conversation_id, client_id, user_id)

        timestamps["query_convo_history_ts"] = datetime.now(timezone.utc)

        if is_refresh:
            while history and history[-1][0] == ARCHI_SENDER:
                _ = history.pop(-1)
            # Re-apply the playbook that shaped the turn being regenerated: its name is
            # stored on the user row but the body only ever lives in-flight, so without
            # this the regenerated answer silently loses it.
            if history:
                last = history[-1]
                body = pending["body"] if pending else None
                foreign = pending.get("foreign", False) if pending else False
                if pending:
                    stored_name = pending["name"]
                else:
                    try:
                        stored_name = self._playbook_svc().last_playbook_name_for_sender(
                            conversation_id, sender)
                    except Exception as exc:
                        logger.warning("Could not read stored playbook name for refresh: %s", exc)
                        stored_name = None
                if body is None and stored_name:
                    owner = get_playbook_owner()
                    if owner:
                        try:
                            playbook = self._playbook_svc().resolve_invokable_playbook(
                                owner, stored_name
                            )
                            body = playbook.body
                            foreign = playbook.owner_id != owner
                        except PlaybookNotFoundError:
                            pass  # deleted since — regenerate without it
                        except Exception as exc:
                            logger.warning(
                                "Could not re-apply playbook '%s' on refresh: %s", stored_name, exc
                            )
                if body:
                    history[-1] = (
                        last[0], playbook_invocation_text(last[1], stored_name, body, foreign)
                    )

        if server_received_msg_ts.timestamp() - client_sent_msg_ts > client_timeout:
            return None, 408

        if not is_refresh:
            history = history + [(sender, agent_content)]

        if len(history) >= QUERY_LIMIT:
            return None, 500

        if model is None:
            logger.debug(f"Model for chat context is None. Setting to default.")
            chat_cfg = self.config.get("services", {}).get("chat_app", {})
            provider = chat_cfg.get("default_provider")
            model = chat_cfg.get("default_model")

        logger.debug(f"Preparing chat context with model {model} provider {provider}")
        return (
            ChatRequestContext(
                sender=sender,
                content=content,
                conversation_id=conversation_id,
                history=history,
                is_refresh=is_refresh,
                config_name=config_name,
                model_used=model,
                provider_used=provider,
                pipeline_used=pipeline,
                playbook_name=playbook_name,
                playbook_id=playbook_id,
            ),
            None,
        )

    def _message_content(self, message) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        return str(content)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars and len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    # =========================================================================
    # Shared Helpers (deduplicated from multiple call-sites)
    # =========================================================================

    @staticmethod
    def _error_event(error_code: int) -> Dict[str, Any]:
        """Map an error code to a structured error event dict."""
        if error_code == 408:
            message = CLIENT_TIMEOUT_ERROR_MESSAGE
        elif error_code == 403:
            message = "conversation not found"
        else:
            message = "server error; see chat logs for message"
        return {"type": "error", "status": error_code, "message": message}

    @staticmethod
    def _trace_from_row(row) -> Dict[str, Any]:
        """Convert a positional agent trace DB row to a dict.
        
        Handles both full rows (16 fields) and subset rows (9 fields from get_active_trace).
        """
        result = {
            'trace_id': row[0],
            'conversation_id': row[1],
            'message_id': row[2],
            'user_message_id': row[3],
            'config_id': row[4],
            'pipeline_name': row[5],
            'events': row[6],
            'started_at': row[7].isoformat() if row[7] else None,
        }
        if len(row) > 9:
            # Full row from get_agent_trace / get_trace_by_message
            result.update({
                'completed_at': row[8].isoformat() if row[8] else None,
                'status': row[9],
                'total_tool_calls': row[10],
                'total_tokens_used': row[11],
                'total_duration_ms': row[12],
                'cancelled_by': row[13],
                'cancellation_reason': row[14],
                'created_at': row[15].isoformat() if row[15] else None,
            })
        else:
            # Subset row from get_active_trace
            result['status'] = row[8]
        return result

    @staticmethod
    def _get_agent_class_from_cfg(chat_cfg: dict) -> Optional[str]:
        """Extract agent class name from a chat config dict."""
        return chat_cfg.get("agent_class") or chat_cfg.get("pipeline")

    @staticmethod
    def _format_score_str(score) -> str:
        """Format a source relevance score for display."""
        if score == -1.0 or score == "N/A":
            return ""
        return f" ({score:.2f})"

    # =========================================================================
    # Pool-based A/B Comparison Streaming
    # =========================================================================

    def stream_ab_comparison(
        self,
        message: List[str],
        conversation_id: Optional[str],
        client_id: str,
        is_refresh: bool,
        server_received_msg_ts: datetime,
        client_sent_msg_ts: float,
        client_timeout: float,
        config_name: str,
        *,
        user_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        provider_api_key: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream a champion-vs-variant A/B comparison.

        Yields interleaved NDJSON events tagged with ``arm: 'a'`` or ``arm: 'b'``
        in real-time as each arm's pipeline produces output.
        Each arm emits its own terminal ``final`` event when generation ends.
        A final ``ab_meta`` event carries the comparison_id and variant mapping.
        """
        import queue
        import threading

        if not self.ab_pool:
            yield {"type": "error", "message": "A/B pool not configured"}
            return

        try:
            requested_config = self._resolve_config_name(config_name)
            self.update_config(config_name=requested_config)

            # Sample matchup
            arm_a_variant, arm_b_variant, is_champion_first = self.ab_pool.sample_matchup()
            logger.info(
                "A/B matchup: arm_a='%s' arm_b='%s' champion_first=%s",
                arm_a_variant.name, arm_b_variant.name, is_champion_first,
            )

            # Prepare chat context (shared — same user message for both arms)
            timestamps = self._init_timestamps()
            context, error_code = self._prepare_chat_context(
                message,
                conversation_id,
                client_id,
                is_refresh,
                server_received_msg_ts,
                client_sent_msg_ts,
                client_timeout,
                timestamps,
                config_name,
                user_id=user_id,
            )
            if error_code is not None:
                yield self._error_event(error_code)
                return

            # A /name-invoked playbook has no in-arm Playbook tool call to surface, and the
            # A/B client drops any event lacking an `arm` tag — so emit the applied-playbook
            # step once per arm here (each flows through the arm-tagged client path into its
            # own timeline; renderPlaybookApplied dedupes if an arm later re-loads it).
            for _pb_event in build_playbook_applied_events(get_pending_playbook(), ("a", "b")):
                yield _pb_event

            # G2 (Layer 2): the owner ContextVar is set on this request thread by
            # _stage_playbook_for_request; capture it so each arm thread can re-establish it.
            # A bare threading.Thread does NOT inherit ContextVars.
            _playbook_owner = get_playbook_owner()

            # Build variant archis
            try:
                arm_a_variant, arm_a_agent_spec = self._resolve_runtime_ab_variant(arm_a_variant)
                arm_b_variant, arm_b_agent_spec = self._resolve_runtime_ab_variant(arm_b_variant)
                archi_a = self._create_variant_archi(
                    arm_a_variant,
                    variant_agent_spec=arm_a_agent_spec,
                    request_provider=provider,
                    request_model=model,
                    request_provider_api_key=provider_api_key,
                )
                archi_b = self._create_variant_archi(
                    arm_b_variant,
                    variant_agent_spec=arm_b_agent_spec,
                    request_provider=provider,
                    request_model=model,
                    request_provider_api_key=provider_api_key,
                )
            except Exception as exc:
                logger.error("Failed to create variant pipelines: %s", exc)
                yield {"type": "error", "message": f"Failed to initialise A/B variants: {exc}"}
                return

            # Shared queue for real-time interleaving
            event_queue: queue.Queue = queue.Queue()
            _SENTINEL = object()

            # Track final text per arm (mutated by threads).
            # Thread-safety note: each thread writes to its own key ("a" or "b")
            # which is safe under CPython's GIL.  The "final_text" value relies on
            # PipelineEventFormatter yielding *accumulated* content (not deltas);
            # the last write per arm is therefore the complete response text.
            arm_results = {
                "a": {
                    "final_text": "",
                    "error": None,
                    "final_emitted": False,
                    "duration_ms": None,
                    "playbook_events": [],  # in-arm playbook_applied events (auto loads)
                },
                "b": {
                    "final_text": "",
                    "error": None,
                    "final_emitted": False,
                    "duration_ms": None,
                    "playbook_events": [],
                },
            }
            arm_model_used = {
                "a": f"{arm_a_variant.provider or ''}/{arm_a_variant.model or ''}".strip("/"),
                "b": f"{arm_b_variant.provider or ''}/{arm_b_variant.model or ''}".strip("/"),
            }

            def _stream_arm(arm_archi, arm_label):
                """Run one arm's stream in a thread, pushing events to the shared queue."""
                set_playbook_owner(_playbook_owner)  # G2: ContextVars don't cross the thread boundary
                import time as _time
                formatter = PipelineEventFormatter(message_content_fn=self._message_content)
                t0 = _time.monotonic()
                first_event_logged = False
                try:
                    logger.info("A/B arm '%s' thread started (t+0.0s)", arm_label)
                    vs = self.archi.vs_connector.get_vectorstore()
                    logger.info(
                        "A/B arm '%s' vectorstore ready (t+%.1fs)",
                        arm_label, _time.monotonic() - t0,
                    )
                    for output in arm_archi.pipeline.stream(
                        history=context.history,
                        conversation_id=context.conversation_id,
                        vectorstore=vs,
                    ):
                        output_meta = output.metadata or {}
                        for event in formatter.process(output):
                            if not first_event_logged:
                                logger.info(
                                    "A/B arm '%s' first event (t+%.1fs): type=%s",
                                    arm_label, _time.monotonic() - t0, event.get("type"),
                                )
                                first_event_logged = True
                            event["arm"] = arm_label
                            if event["type"] == "text":
                                arm_results[arm_label]["final_text"] = event["content"]
                            elif event["type"] == "playbook_applied":
                                # Collected for the unified ledger; filtered to in-arm auto
                                # loads (tool_call_id-bearing) after the arm's message id is known.
                                arm_results[arm_label]["playbook_events"].append(event)
                            event_queue.put(event)
                        if output_meta.get("event_type") == "final" and not arm_results[arm_label]["final_emitted"]:
                            if not first_event_logged:
                                logger.info(
                                    "A/B arm '%s' first event (t+%.1fs): type=final",
                                    arm_label, _time.monotonic() - t0,
                                )
                                first_event_logged = True
                            final_text = getattr(output, "answer", "") or formatter.last_text or arm_results[arm_label]["final_text"]
                            arm_results[arm_label]["final_text"] = final_text
                            arm_results[arm_label]["final_emitted"] = True
                            duration_ms = int((_time.monotonic() - t0) * 1000)
                            arm_results[arm_label]["duration_ms"] = duration_ms
                            event_queue.put({
                                "type": "final",
                                "arm": arm_label,
                                "response": final_text,
                                "usage": output_meta.get("usage"),
                                "model": output_meta.get("model"),
                                "model_used": arm_model_used[arm_label],
                                "duration_ms": duration_ms,
                            })
                except Exception as exc:
                    arm_results[arm_label]["error"] = str(exc)
                    event_queue.put({"type": "error", "arm": arm_label, "message": str(exc)})
                finally:
                    logger.info(
                        "A/B arm '%s' finished (t+%.1fs)",
                        arm_label, _time.monotonic() - t0,
                    )
                    event_queue.put(_SENTINEL)

            # Yield arm labels early so the frontend can display variant names
            yield {
                "type": "ab_arms",
                "arm_a_name": arm_a_variant.name,
                "arm_b_name": arm_b_variant.name,
                "variant_label_mode": self.ab_pool.variant_label_mode,
            }

            # Start both arms in parallel threads
            thread_a = threading.Thread(target=_stream_arm, args=(archi_a, "a"), daemon=True)
            thread_b = threading.Thread(target=_stream_arm, args=(archi_b, "b"), daemon=True)
            thread_a.start()
            thread_b.start()

            # Drain the queue in real-time, yielding events as they arrive
            finished_count = 0
            while finished_count < 2:
                item = event_queue.get()
                if item is _SENTINEL:
                    finished_count += 1
                    continue
                yield item

            thread_a.join()
            thread_b.join()

            # Check for errors
            arm_a_error = arm_results["a"]["error"]
            arm_b_error = arm_results["b"]["error"]
            arm_a_final_text = arm_results["a"]["final_text"]
            arm_b_final_text = arm_results["b"]["final_text"]
            arm_a_duration_ms = arm_results["a"]["duration_ms"]
            arm_b_duration_ms = arm_results["b"]["duration_ms"]

            if arm_a_error and arm_b_error:
                yield {"type": "error", "message": "Both A/B arms failed",
                       "arm_a_error": arm_a_error, "arm_b_error": arm_b_error}
                return

            if arm_a_error or arm_b_error:
                yield {"type": "error", "message": "One A/B arm failed",
                       "failed_arm": "a" if arm_a_error else "b",
                       "error": arm_a_error or arm_b_error}
                return

            # Store user message first (normal chat stores it inline, AB must do so explicitly)
            user_prompt_mid = None
            if not is_refresh:
                try:
                    insert_tups = [
                        ("chat", context.conversation_id, context.sender, context.content,
                         "", "", datetime.now(), None, None),
                    ]
                    inserted_ids = self._insert_conversation_rows(insert_tups)
                    user_prompt_mid = inserted_ids[0] if inserted_ids else None
                    self._record_playbook_turn_best_effort(user_prompt_mid, context)
                except Exception as exc:
                    logger.error("Failed to store user message: %s", exc)

            # Store both responses as messages
            pipeline_used = ChatWrapper._get_agent_class_from_cfg(self.services_config.get("chat_app", {})) or ""
            arm_a_mid = self._store_assistant_message(
                context.conversation_id,
                arm_a_final_text,
                model_used=f"{arm_a_variant.provider or ''}/{arm_a_variant.model or ''}".strip("/"),
                pipeline_used=pipeline_used,
            )
            arm_b_mid = self._store_assistant_message(
                context.conversation_id,
                arm_b_final_text,
                model_used=f"{arm_b_variant.provider or ''}/{arm_b_variant.model or ''}".strip("/"),
                pipeline_used=pipeline_used,
            )

            # Persist per-arm latency for analysis by reusing the timing table keyed by message_id.
            self._persist_ab_arm_timing(arm_a_mid, arm_a_duration_ms)
            self._persist_ab_arm_timing(arm_b_mid, arm_b_duration_ms)

            # Unified ledger: one auto row per in-arm model-invoked Playbook load, keyed to
            # that arm's assistant message. The up-front /name per-arm events carry no
            # tool_call_id and were already logged as explicit via the shared turn-recorder;
            # they are skipped here. Failed in-arm loads produce no artifact hence no event —
            # an accepted v1 limitation.
            self._record_ab_playbook_loads(
                context.conversation_id, arm_a_mid, "a", arm_results["a"]["playbook_events"])
            self._record_ab_playbook_loads(
                context.conversation_id, arm_b_mid, "b", arm_results["b"]["playbook_events"])

            # Get user prompt message ID if not already stored above
            if not user_prompt_mid:
                user_prompt_mid = self._get_last_user_message_id(context.conversation_id)

            # Create comparison record (skip if we have no valid message IDs)
            comparison_id = None
            if user_prompt_mid and arm_a_mid and arm_b_mid:
                try:
                    comparison_id = self.conv_service.create_ab_comparison(
                        conversation_id=context.conversation_id,
                        user_prompt_mid=user_prompt_mid,
                        response_a_mid=arm_a_mid,
                        response_b_mid=arm_b_mid,
                        model_a=f"{arm_a_variant.provider or ''}/{arm_a_variant.model or ''}".strip("/"),
                        pipeline_a=pipeline_used,
                        model_b=f"{arm_b_variant.provider or ''}/{arm_b_variant.model or ''}".strip("/"),
                        pipeline_b=pipeline_used,
                        is_config_a_first=is_champion_first,
                        variant_a_name=arm_a_variant.name,
                        variant_b_name=arm_b_variant.name,
                        variant_a_meta=arm_a_variant.to_meta_json(),
                        variant_b_meta=arm_b_variant.to_meta_json(),
                    )
                except Exception as exc:
                    logger.error("Failed to create A/B comparison record: %s", exc)
                    comparison_id = None

            # Emit final metadata event
            yield {
                "type": "ab_meta",
                "comparison_id": comparison_id,
                "conversation_id": context.conversation_id,
                "arm_a_variant": arm_a_variant.name,
                "arm_b_variant": arm_b_variant.name,
                "arm_a_model_used": f"{arm_a_variant.provider or ''}/{arm_a_variant.model or ''}".strip("/"),
                "arm_b_model_used": f"{arm_b_variant.provider or ''}/{arm_b_variant.model or ''}".strip("/"),
                "is_champion_first": is_champion_first,
                "arm_a_message_id": arm_a_mid,
                "arm_b_message_id": arm_b_mid,
                "arm_a_duration_ms": arm_a_duration_ms,
                "arm_b_duration_ms": arm_b_duration_ms,
                "variant_label_mode": self.ab_pool.variant_label_mode,
            }
        except Exception:
            logger.error("A/B comparison failed", exc_info=True)
            yield {"type": "error", "message": "A/B comparison failed; see chat logs for details"}
            return

    def _store_assistant_message(self, conversation_id, content, model_used=None, pipeline_used=None):
        """Store an assistant message and return the message_id."""
        try:
            inserted_ids = self._insert_conversation_rows([
                ("chat", conversation_id, "archi", content, "", "", datetime.now(), model_used, pipeline_used),
            ])
            return inserted_ids[0] if inserted_ids else None
        except Exception as exc:
            logger.error("Failed to store assistant message: %s", exc)
            return None

    def _record_ab_playbook_loads(self, conversation_id, message_id, arm, events) -> None:
        """Best-effort unified-ledger write for one A/B arm's in-arm auto Playbook
        loads (the arm-tagged playbook_applied events that carry a tool_call_id).
        No-op without a stored arm message id. Never breaks the comparison."""
        if not message_id:
            return
        for load in playbook_loads_from_events(events):
            try:
                self._playbook_svc().record_invocation(
                    conversation_id, message_id, load.get("playbook_id"), load.get("name"),
                    source="auto", status="ok", arm=arm)
            except Exception as exc:
                logger.warning(
                    "Could not record A/B auto playbook invocation (arm %s): %s", arm, exc)

    def _record_stream_playbook_loads(
            self, conversation_id, archi_message_id, trace_events, recorded_ids) -> None:
        """Best-effort unified-ledger write for the stream path's in-turn auto Playbook
        loads. The stream finalizes on the LAST streamed PipelineOutput, whose .messages
        hold only the final answer — so insert_tool_calls_from_output extracts nothing and
        the auto ledger would miss stream-turn loads. Recover them from the persisted trace
        events (playbook_applied events carry tool_call_id/name/playbook_id), skipping any
        id already written as an auto row for this turn (``recorded_ids`` — non-empty only
        for pipelines whose final output DOES carry full messages) so nothing is
        double-counted. No-op without a stored archi message id or any events; never breaks
        the stream."""
        if not (archi_message_id and trace_events):
            return
        for load in playbook_loads_from_events(trace_events):
            if load.get("tool_call_id") in recorded_ids:
                continue
            try:
                self._playbook_svc().record_invocation(
                    conversation_id, archi_message_id, load.get("playbook_id"), load.get("name"),
                    source="auto", status="ok")
            except Exception as exc:
                logger.warning(
                    "Could not record stream auto playbook invocation for message %s: %s",
                    archi_message_id, exc)

    def _persist_ab_arm_timing(self, message_id: Optional[int], duration_ms: Optional[int]) -> None:
        """Persist A/B arm latency into the timing table for post-hoc analysis."""
        if not message_id or duration_ms is None:
            return

        safe_duration_ms = max(int(duration_ms), 0)
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(milliseconds=safe_duration_ms)

        synthetic_timestamps = {
            "client_sent_msg_ts": start_ts,
            "server_received_msg_ts": start_ts,
            "lock_acquisition_ts": start_ts,
            "vectorstore_update_ts": start_ts,
            "query_convo_history_ts": start_ts,
            "chain_finished_ts": end_ts,
            "archi_message_ts": end_ts,
            "insert_convo_ts": end_ts,
            "finish_call_ts": end_ts,
            "server_response_msg_ts": end_ts,
        }

        try:
            self.insert_timing(message_id, synthetic_timestamps)
        except Exception as exc:
            logger.warning("Failed to persist A/B timing for message %s: %s", message_id, exc)

    def _get_last_user_message_id(self, conversation_id):
        """Get the most recent user message_id for a conversation."""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message_id FROM conversations WHERE conversation_id = %s AND LOWER(sender) = 'user' ORDER BY ts DESC LIMIT 1",
                (conversation_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            return row[0] if row else None
        except Exception as exc:
            logger.error("Failed to get user message id: %s", exc)
            return None

    def _stream_events_from_output(
        self,
        output,
        *,
        include_agent_steps: bool,
        include_tool_steps: bool,
        conversation_id: int,
        max_chars: int = 800,
    ) -> List[Dict[str, Any]]:
        messages = getattr(output, "messages", []) or []
        if not messages:
            return []
        message = messages[-1]
        events: List[Dict[str, Any]] = []
        msg_type = str(getattr(message, "type", "")).lower()

        if include_tool_steps and hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                events.append(
                    {
                        "type": "step",
                        "step_type": "tool_call",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_call_id": tool_call.get("id", ""),
                        "content": self._truncate_text(f"{tool_name}({tool_args})", max_chars),
                        "conversation_id": conversation_id,
                    }
                )

        if include_tool_steps and getattr(message, "tool_call_id", None):
            events.append(
                {
                    "type": "step",
                    "step_type": "tool_result",
                    "tool_call_id": message.tool_call_id,
                    "content": self._truncate_text(self._message_content(message), max_chars),
                    "conversation_id": conversation_id,
                }
            )

        content = self._message_content(message) if msg_type in {"ai", "assistant"} else ""
        handled_tool_call = False
        if include_tool_steps and content:
            tool_match = re.match(r"^\s*([\w.-]+)\[ARGS\](.*)$", content, re.DOTALL)
            if tool_match:
                tool_name = tool_match.group(1)
                tool_args_raw = tool_match.group(2).strip()
                events.append(
                    {
                        "type": "step",
                        "step_type": "tool_call",
                        "tool_name": tool_name,
                        "tool_args": tool_args_raw,
                        "tool_call_id": "",
                        "content": self._truncate_text(content, max_chars),
                        "conversation_id": conversation_id,
                    }
                )
                handled_tool_call = True

        if include_agent_steps and content and not handled_tool_call:
            events.append(
                {
                    "type": "step",
                    "step_type": "agent",
                    "content": content,
                    "conversation_id": conversation_id,
                }
            )

        return events

    def _finalize_result(
        self,
        result,
        *,
        context: ChatRequestContext,
        server_received_msg_ts: datetime,
        timestamps: Dict[str, datetime],
        render_markdown: bool = True,
    ) -> tuple[str, List[int]]:
        # For streaming responses, return raw markdown (client renders with marked.js)
        # For non-streaming responses, render server-side with Mistune
        if render_markdown:
            output = self.format_code_in_text(result["answer"])
        else:
            output = result["answer"]

        documents = result.get("source_documents", [])
        scores = result.get("metadata", {}).get("retriever_scores", [])
        top_sources = self.get_top_sources(documents, scores)
        
        output = self.append_source_section(
            output,
            top_sources,
            render_markdown=render_markdown,
        )

        timestamps["archi_message_ts"] = datetime.now(timezone.utc)
        context_data = self.prepare_context_for_storage(documents, scores)

        best_reference = "Link unavailable"
        if top_sources:
            primary_source = top_sources[0]
            best_reference = primary_source["link"] or primary_source["display"]

        user_message = (context.sender, context.content, server_received_msg_ts)
        archi_message = (ARCHI_SENDER, output, timestamps["archi_message_ts"])
        message_ids = self.insert_conversation(
            context.conversation_id,
            user_message,
            archi_message,
            best_reference,
            context_data,
            context,
            context.is_refresh,
        )
        timestamps["insert_convo_ts"] = datetime.now(timezone.utc)
        context.history.append((ARCHI_SENDER, result["answer"]))

        agent_messages = getattr(result, "messages", []) or []
        if agent_messages:
            logger.debug("Agent messages count: %d", len(agent_messages))
            for i, msg in enumerate(agent_messages):
                msg_type = type(msg).__name__
                has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                has_tool_call_id = hasattr(msg, "tool_call_id") and msg.tool_call_id
                logger.debug(
                    "  Message %d: %s, tool_calls=%s, tool_call_id=%s",
                    i,
                    msg_type,
                    has_tool_calls,
                    has_tool_call_id,
                )
        if agent_messages and message_ids:
            archi_message_id = message_ids[-1]
            self.insert_tool_calls_from_output(context.conversation_id, archi_message_id, result)

        return output, message_ids

    def __call__(self, message: List[str], conversation_id: int|None, client_id: str, is_refresh: bool, server_received_msg_ts: datetime,  client_sent_msg_ts: float, client_timeout: float, config_name: str, user_id: Optional[str] = None):
        """
        Execute the chat functionality.
        """
        timestamps = self._init_timestamps()
        output = None
        message_ids = None
        context = None

        try:
            context, error_code = self._prepare_chat_context(
                message,
                conversation_id,
                client_id,
                is_refresh,
                server_received_msg_ts,
                client_sent_msg_ts,
                client_timeout,
                timestamps,
                config_name,
                user_id=user_id,
            )
            if error_code is not None:
                return None, None, None, timestamps, error_code

            requested_config = self._resolve_config_name(config_name)
            self.update_config(config_name=requested_config)

            result = self.archi(history=context.history, conversation_id=context.conversation_id)
            timestamps["chain_finished_ts"] = datetime.now(timezone.utc)

            # keep track of total number of queries and log this amount
            self.number_of_queries += 1
            logger.info(f"Number of queries is: {self.number_of_queries}")

            output, message_ids = self._finalize_result(
                result,
                context=context,
                server_received_msg_ts=server_received_msg_ts,
                timestamps=timestamps,
            )

        except ConversationAccessError as e:
            logger.warning(f"Unauthorized conversation access attempt: {e}")
            return None, None, None, timestamps, 403
        except Exception as e:
            # NOTE: we log the error message and return here
            logger.error(f"Failed to produce response: {e}", exc_info=True)
            return None, None, None, timestamps, 500

        finally:
            if self.cursor is not None:
                self.cursor.close()
            if self.conn is not None:
                self.conn.close()

        timestamps['finish_call_ts'] = datetime.now(timezone.utc)

        return output, context.conversation_id if context else None, message_ids, timestamps, None

    def stream(
        self,
        message: List[str],
        conversation_id: Optional[str],
        client_id: str,
        is_refresh: bool,
        server_received_msg_ts: datetime,
        client_sent_msg_ts: float,
        client_timeout: float,
        config_name: str,
        *,
        include_agent_steps: bool = True,
        include_tool_steps: bool = True,
        max_step_chars: int = 800,
        provider: str = None,
        model: str = None,
        provider_api_key: str = None,
        user_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        timestamps = self._init_timestamps()
        context = None
        last_output = None
        formatter = PipelineEventFormatter(
            message_content_fn=self._message_content,
            max_step_chars=max_step_chars,
        )
        last_streamed_text = ""
        trace_id = None
        trace_events: List[Dict[str, Any]] = []
        stream_start_time = time.time()

        try:
            context, error_code = self._prepare_chat_context(
                message,
                conversation_id,
                client_id,
                is_refresh,
                server_received_msg_ts,
                client_sent_msg_ts,
                client_timeout,
                timestamps,
                config_name,
                user_id=user_id,
                model=model,
                provider=provider,
                pipeline=self.archi.pipeline
            )
            if error_code is not None:
                yield self._error_event(error_code)
                return
            
            requested_config = self._resolve_config_name(config_name)
            self.update_config(config_name=requested_config)
            
            # If provider and model are specified in the context, override the pipeline's LLM
            provider = context.provider_used
            model = context.model_used
            if provider and model:
                try:
                    override_llm = self._create_provider_llm(provider, model, provider_api_key)
                    if override_llm and hasattr(self.archi, 'pipeline') and hasattr(self.archi.pipeline, 'agent_llm'):
                        original_llm = self.archi.pipeline.agent_llm
                        self.archi.pipeline.agent_llm = override_llm
                        # Force agent refresh to use new LLM
                        if hasattr(self.archi.pipeline, 'refresh_agent'):
                            self.archi.pipeline.refresh_agent(force=True)
                        logger.info(f"Overrode pipeline LLM with {provider}/{model}")
                except ValueError as e:
                    logger.warning(f"Failed to create provider LLM {provider}/{model}: {e}")
                    yield {"type": "error", "status": 400, "message": str(e)}
                    return
                except Exception as e:
                    logger.warning(f"Failed to create provider LLM {provider}/{model}: {e}")
                    yield {"type": "warning", "message": f"Using default model: {e}"}
            
            # Create trace for this streaming request
            trace_id = self.create_agent_trace(
                conversation_id=context.conversation_id,
                user_message_id=None,  # Will be updated at finalization
                config_id=None,  # Legacy field, no longer used
                pipeline_name=self.archi.pipeline_name if hasattr(self.archi, 'pipeline_name') else None,
            )

            for output in self.archi.stream(history=context.history, conversation_id=context.conversation_id,model=context.model_used):
                if client_timeout and time.time() - stream_start_time > client_timeout:
                    if trace_id:
                        total_duration_ms = int((time.time() - stream_start_time) * 1000)
                        self.update_agent_trace(
                            trace_id=trace_id,
                            events=trace_events,
                            status='error',
                            cancelled_by='system',
                            cancellation_reason='Client timeout',
                            total_duration_ms=total_duration_ms,
                        )
                    yield self._error_event(408)
                    return
                last_output = output
                
                # Use shared event formatter for structured event types
                event_type = output.metadata.get("event_type", "text") if output.metadata else "text"
                timestamp = datetime.now(timezone.utc).isoformat()

                if event_type == "final":
                    pass  # handled after the loop
                elif event_type not in ("tool_start", "tool_output", "tool_end",
                                        "thinking_start", "thinking_end", "text"):
                    # Legacy fallback for non-agent pipelines
                    if getattr(output, "final", False):
                        continue
                    for event in self._stream_events_from_output(
                        output,
                        include_agent_steps=False,
                        include_tool_steps=include_tool_steps,
                        conversation_id=context.conversation_id,
                        max_chars=max_step_chars,
                    ):
                        yield event
                    if include_agent_steps:
                        content = getattr(output, "answer", "") or ""
                        if content:
                            if content.startswith(last_streamed_text):
                                delta = content[len(last_streamed_text):]
                            else:
                                delta = content
                            last_streamed_text = content
                            chunk_size = 80
                            for i in range(0, len(delta), chunk_size):
                                yield {
                                    "type": "chunk",
                                    "content": delta[i:i + chunk_size],
                                    "conversation_id": context.conversation_id,
                                }
                else:
                    # Formatter handles tool_start/output/end, thinking, text
                    for event in formatter.process(output):
                        event["timestamp"] = timestamp
                        event["conversation_id"] = context.conversation_id
                        if event["type"] == "text":
                            # Map to "chunk" type for backward compat with JS client
                            if include_agent_steps:
                                last_streamed_text = event["content"]
                                yield {
                                    "type": "chunk",
                                    "content": event["content"],
                                    "accumulated": True,
                                    "conversation_id": context.conversation_id,
                                }
                            trace_events.append({
                                "type": "text",
                                "content": event["content"],
                                "timestamp": timestamp,
                            })
                        else:
                            trace_events.append(event)
                            if include_tool_steps:
                                yield event

            timestamps["chain_finished_ts"] = datetime.now(timezone.utc)

            if last_output is None:
                if trace_id:
                    self.update_agent_trace(
                        trace_id=trace_id,
                        events=trace_events,
                        status='error',
                        cancelled_by='system',
                        cancellation_reason='No output from pipeline',
                    )
                yield {"type": "error", "status": 500, "message": "server error; see chat logs for message"}
                return
                
            # keep track of total number of queries and log this amount
            self.number_of_queries += 1
            logger.info(f"Number of queries is: {self.number_of_queries}")

            output, message_ids = self._finalize_result(
                last_output,
                context=context,
                server_received_msg_ts=server_received_msg_ts,
                timestamps=timestamps,
                render_markdown=False,  # Client renders with marked.js
            )

            # Stream path only: _finalize_result ran insert_tool_calls_from_output on the
            # LAST streamed output, whose .messages hold only the final answer — so auto
            # Playbook loads that happened mid-stream never reached the unified ledger.
            # Recover them from the trace events, skipping any tool-call id the finalize
            # already wrote (dedupe for pipelines whose final output does carry messages).
            if message_ids:
                recorded_ids = {tc.get("id") for tc in last_output.extract_tool_calls()}
                self._record_stream_playbook_loads(
                    context.conversation_id, message_ids[-1], trace_events, recorded_ids)

            timestamps["finish_call_ts"] = datetime.now(timezone.utc)
            timestamps["server_received_msg_ts"] = server_received_msg_ts
            timestamps["client_sent_msg_ts"] = datetime.fromtimestamp(client_sent_msg_ts, tz=timezone.utc)
            timestamps["server_response_msg_ts"] = datetime.now(timezone.utc)

            if message_ids:
                self.insert_timing(message_ids[-1], timestamps)
                
            # Calculate total duration
            total_duration_ms = int((time.time() - stream_start_time) * 1000)
            
            # Extract usage and model from final output metadata
            usage = None
            model = None
            if last_output and last_output.metadata:
                usage = last_output.metadata.get("usage")
                model = last_output.metadata.get("model")
            
            # Append usage summary to trace events so it's available in historical views
            if usage:
                trace_events.append({
                    "type": "usage",
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "context_window": usage.get("context_window", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            # Update trace with final state
            if trace_id:
                user_message_id = message_ids[0] if message_ids and len(message_ids) > 1 else None
                self.update_agent_trace(
                    trace_id=trace_id,
                    events=trace_events,
                    status='completed',
                    message_id=message_ids[-1] if message_ids else None,
                    total_tool_calls=formatter.tool_call_count,
                    total_duration_ms=total_duration_ms,
                )

            yield {
                "type": "final",
                "response": output,
                "conversation_id": context.conversation_id,
                "archi_msg_id": message_ids[-1] if message_ids else None,
                "message_id": message_ids[-1] if message_ids else None,
                "user_message_id": message_ids[0] if message_ids and len(message_ids) > 1 else None,
                "trace_id": trace_id,
                "server_response_msg_ts": timestamps["server_response_msg_ts"].timestamp(),
                "final_response_msg_ts": datetime.now(timezone.utc).timestamp(),
                "usage": usage,
                "model_used": model or context.model_used,
                "provider_used": provider or context.provider_used,
                "pipeline_used": context.pipeline_used,
            }

        except GeneratorExit:
            # User cancelled the stream
            if trace_id:
                total_duration_ms = int((time.time() - stream_start_time) * 1000)
                self.update_agent_trace(
                    trace_id=trace_id,
                    events=trace_events,
                    status='cancelled',
                    total_tool_calls=formatter.tool_call_count,
                    total_duration_ms=total_duration_ms,
                    cancelled_by='user',
                    cancellation_reason='Stream cancelled by client',
                )
            raise
        except ConversationAccessError as exc:
            logger.warning("Unauthorized conversation access attempt: %s", exc)
            if trace_id:
                self.update_agent_trace(
                    trace_id=trace_id,
                    events=trace_events,
                    status='error',
                    cancelled_by='system',
                    cancellation_reason=str(exc),
                )
            yield {"type": "error", "status": 403, "message": "conversation not found"}
        except Exception as exc:
            logger.error("Failed to stream response: %s", exc, exc_info=True)
            if trace_id:
                self.update_agent_trace(
                    trace_id=trace_id,
                    events=trace_events,
                    status='error',
                    cancelled_by='system',
                    cancellation_reason=str(exc),
                )
            yield {"type": "error", "status": 500, "message": "server error; see chat logs for message"}
        finally:
            if self.cursor is not None:
                self.cursor.close()
            if self.conn is not None:
                self.conn.close()


class FlaskAppWrapper(object):

    def __init__(self, app, **configs):
        logger.info("Entering FlaskAppWrapper")
        self.app = app
        self.configs(**configs)
        self.config = get_full_config()
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]
        self.chat_app_config = self.config["services"]["chat_app"]
        self.data_path = self.global_config["DATA_PATH"]
        evaluations_config = self.chat_app_config.get("evaluations", {})
        self.evaluations_enabled = evaluations_config.get("enabled") is True
        self.evaluation_service = None
        if self.evaluations_enabled:
            evaluation_mcp_config_path = evaluations_config.get("mcp_config_path")
            self.evaluation_service = EvaluationConsoleService(
                Path(evaluations_config.get("root", "/root/archi/evaluations")),
                agent_config_path=Path(
                    evaluations_config.get(
                        "agent_config_path", "/root/archi/configs/config.yaml"
                    )
                ),
                agents_dir=Path(
                    self.chat_app_config.get("agents_dir") or "/root/archi/agents"
                ),
                mcp_config_path=(
                    Path(evaluation_mcp_config_path)
                    if evaluation_mcp_config_path
                    else None
                ),
            )
        self.salt = read_secret("UPLOADER_SALT")
        # Persist an auto-generated key in the DATA_PATH volume when none is configured, so signed
        # sessions survive a restart — a fresh random key per boot would log every user out on each
        # restart (an explicit FLASK_UPLOADER_APP_SECRET_KEY, if set, still takes precedence).
        self.app.secret_key = read_or_create_persistent_secret(
            "FLASK_UPLOADER_APP_SECRET_KEY", self.data_path
        )
        
        # Session cookie security settings (BYOK security hardening)
        self.app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
        self.app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
        # SESSION_COOKIE_SECURE should be True in production (HTTPS only)
        # Leave it False for local development to work over HTTP
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB upload limit
        
        self.app.config['ACCOUNTS_FOLDER'] = self.global_config["ACCOUNTS_PATH"]
        os.makedirs(self.app.config['ACCOUNTS_FOLDER'], exist_ok=True)

        # store postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        # Initialize config service for dynamic settings
        self.config_service = ConfigService(pg_config=self.pg_config)

        # Refresh the RBAC registry against the current deployment config.
        try:
            get_registry(force_reload=True)
        except Exception as exc:
            logger.warning("Failed to reload RBAC registry from current config: %s", exc)

        # Data manager service URL for upload proxy
        dm_config = self.services_config.get("data_manager", {})
        # Use 'hostname' for service discovery (Docker network name), fallback to 'host' for local dev
        dm_host = dm_config.get("hostname") or dm_config.get("host", "localhost")
        dm_port = dm_config.get("port", 5001)
        self.data_manager_url = f"http://{dm_host}:{dm_port}"
        # API token for service-to-service auth with data-manager
        dm_token = read_secret("DM_API_TOKEN") or None
        self._dm_headers = {"Authorization": f"Bearer {dm_token}"} if dm_token else {}
        logger.info(f"Data manager service URL: {self.data_manager_url}")

        # Initialize authentication methods
        self.oauth = None
        auth_config = self.chat_app_config.get('auth', {})
        self.auth_enabled = auth_config.get('enabled', False)
        self.sso_enabled = auth_config.get('sso', {}).get('enabled', False)
        self.basic_auth_enabled = auth_config.get('basic', {}).get('enabled', False)
        
        logger.info(f"Auth enabled: {self.auth_enabled}, SSO: {self.sso_enabled}, Basic: {self.basic_auth_enabled}")
        
        if self.sso_enabled:
            self._setup_sso()

        # create the chat from the wrapper and ensure default config is active
        self.chat = ChatWrapper()
        self.chat.update_config(config_name=self.config["name"])

        # enable CORS:
        CORS(self.app)

        # inject active alerts into every template context
        @self.app.context_processor
        def _inject_alerts():
            if not session.get('logged_in'):
                return dict(active_banner_alerts=[], is_alert_manager=False)
            alerts = get_active_banner_alerts()
            return dict(
                active_banner_alerts=alerts,
                is_alert_manager=is_alert_manager(),
            )

        # add endpoints for flask app
        # Public endpoints (no auth required)
        self.add_endpoint('/', 'landing', self.landing)
        self.add_endpoint('/api/health', 'health', self.health, methods=["GET"])
        
        # Protected endpoints (require auth when enabled)
        self.add_endpoint('/chat', 'index', self.require_auth(self.index))
        self.add_endpoint('/api/get_chat_response', 'get_chat_response', self.require_auth(self.get_chat_response), methods=["POST"])
        self.add_endpoint('/api/get_chat_response_stream', 'get_chat_response_stream', self.require_auth(self.get_chat_response_stream), methods=["POST"])
        self.add_endpoint('/terms', 'terms', self.require_auth(self.terms))
        self.add_endpoint('/api/like', 'like', self.require_auth(self.like),  methods=["POST"])
        self.add_endpoint('/api/dislike', 'dislike', self.require_auth(self.dislike),  methods=["POST"])
        # Config modification requires config:modify permission (archi-expert or archi-admins)
        self.add_endpoint('/api/update_config', 'update_config', self.require_perm(Permission.Config.MODIFY)(self.update_config), methods=["POST"])
        self.add_endpoint('/api/get_configs', 'get_configs', self.require_auth(self.get_configs), methods=["GET"])
        self.add_endpoint('/api/text_feedback', 'text_feedback', self.require_auth(self.text_feedback), methods=["POST"])

        # endpoints for conversations managing
        logger.info("Adding conversations management API endpoints")
        self.add_endpoint('/api/list_conversations', 'list_conversations', self.require_auth(self.list_conversations), methods=["GET"])
        self.add_endpoint('/api/load_conversation', 'load_conversation', self.require_auth(self.load_conversation), methods=["POST"])
        self.add_endpoint('/api/new_conversation', 'new_conversation', self.require_auth(self.new_conversation), methods=["POST"])
        self.add_endpoint('/api/delete_conversation', 'delete_conversation', self.require_auth(self.delete_conversation), methods=["POST"])

        # A/B testing endpoints
        logger.info("Adding A/B testing API endpoints")
        self.add_endpoint('/api/ab/preference', 'ab_preference', self.require_auth(self.ab_submit_preference), methods=["POST"])
        self.add_endpoint('/api/ab/pending', 'ab_pending', self.require_auth(self.ab_get_pending), methods=["GET"])
        self.add_endpoint('/api/ab/pool', 'ab_pool', self.require_auth(self.ab_get_pool), methods=["GET"])
        self.add_endpoint('/api/ab/decision', 'ab_decision', self.require_auth(self.ab_get_decision), methods=["GET"])
        self.add_endpoint('/api/ab/pool/set', 'ab_pool_set', self.require_auth(self.ab_set_pool), methods=["POST"])
        self.add_endpoint('/api/ab/pool/settings/set', 'ab_pool_settings_set', self.require_auth(self.ab_set_settings), methods=["POST"])
        self.add_endpoint('/api/ab/pool/variants/set', 'ab_pool_variants_set', self.require_auth(self.ab_set_variants), methods=["POST"])
        self.add_endpoint('/api/ab/pool/disable', 'ab_pool_disable', self.require_auth(self.ab_disable_pool), methods=["POST"])
        self.add_endpoint('/api/ab/compare', 'ab_compare', self.require_auth(self.ab_compare_stream), methods=["POST"])
        self.add_endpoint('/api/ab/metrics', 'ab_metrics', self.require_auth(self.ab_get_metrics), methods=["GET"])
        self.add_endpoint('/api/ab/agents/list', 'list_ab_agents', self.require_auth(self.list_ab_agents), methods=["GET"])
        self.add_endpoint('/api/ab/agents/template', 'get_ab_agent_template', self.require_auth(self.get_ab_agent_template), methods=["GET"])
        self.add_endpoint('/api/ab/agents', 'save_ab_agent_spec', self.require_auth(self.save_ab_agent_spec), methods=["POST"])

        # Agent trace endpoints
        logger.info("Adding agent trace API endpoints")
        self.add_endpoint('/api/trace/<trace_id>', 'get_trace', self.require_auth(self.get_trace), methods=["GET"])
        self.add_endpoint('/api/trace/message/<int:message_id>', 'get_trace_by_message', self.require_auth(self.get_trace_by_message), methods=["GET"])
        self.add_endpoint('/api/cancel_stream', 'cancel_stream', self.require_auth(self.cancel_stream), methods=["POST"])

        # Provider endpoints
        logger.info("Adding provider API endpoints")
        self.add_endpoint('/api/providers', 'get_providers', self.require_auth(self.get_providers), methods=["GET"])
        self.add_endpoint('/api/providers/models', 'get_provider_models', self.require_auth(self.get_provider_models), methods=["GET"])
        self.add_endpoint('/api/providers/validate', 'validate_provider', self.require_auth(self.validate_provider), methods=["POST"])
        self.add_endpoint('/api/providers/keys', 'get_provider_api_keys', self.require_auth(self.get_provider_api_keys), methods=["GET"])
        self.add_endpoint('/api/providers/keys/set', 'set_provider_api_key', self.require_auth(self.set_provider_api_key), methods=["POST"])
        self.add_endpoint('/api/providers/keys/clear', 'clear_provider_api_key', self.require_auth(self.clear_provider_api_key), methods=["POST"])
        self.add_endpoint('/api/pipeline/default_model', 'get_pipeline_default_model', self.require_auth(self.get_pipeline_default_model), methods=["GET"])
        self.add_endpoint('/api/agent/info', 'get_agent_info', self.require_auth(self.get_agent_info), methods=["GET"])
        self.add_endpoint('/api/agents/list', 'list_agents', self.require_auth(self.list_agents), methods=["GET"])
        self.add_endpoint('/api/agents/template', 'get_agent_template', self.require_auth(self.get_agent_template), methods=["GET"])
        self.add_endpoint('/api/agents/spec', 'get_agent_spec', self.require_auth(self.get_agent_spec), methods=["GET"])
        self.add_endpoint('/api/agents', 'save_agent_spec', self.require_auth(self.save_agent_spec), methods=["POST"])
        self.add_endpoint('/api/agents', 'delete_agent_spec', self.require_auth(self.delete_agent_spec), methods=["DELETE"])
        self.add_endpoint('/api/agents/active', 'set_active_agent', self.require_auth(self.set_active_agent), methods=["POST"])

        # Data viewer endpoints
        # View data page and list documents - requires documents:view permission
        # Enable/disable documents - requires documents:select permission
        logger.info("Adding data viewer API endpoints")
        self.add_endpoint('/data', 'data_viewer', self.require_perm(Permission.Documents.VIEW)(self.data_viewer_page))
        self.add_endpoint('/admin/ab-testing', 'ab_testing_admin_page', self.require_auth(self.ab_testing_admin_page))
        self.add_endpoint('/api/data/documents', 'list_data_documents', self.require_perm(Permission.Documents.VIEW)(self.list_data_documents), methods=["GET"])
        self.add_endpoint('/api/data/documents/<document_hash>/content', 'get_data_document_content', self.require_perm(Permission.Documents.VIEW)(self.get_data_document_content), methods=["GET"])
        self.add_endpoint('/api/data/documents/<document_hash>/chunks', 'get_data_document_chunks', self.require_perm(Permission.Documents.VIEW)(self.get_data_document_chunks), methods=["GET"])
        self.add_endpoint('/api/data/documents/<document_hash>/enable', 'enable_data_document', self.require_perm(Permission.Documents.SELECT)(self.enable_data_document), methods=["POST"])
        self.add_endpoint('/api/data/documents/<document_hash>/disable', 'disable_data_document', self.require_perm(Permission.Documents.SELECT)(self.disable_data_document), methods=["POST"])
        self.add_endpoint('/api/data/bulk-enable', 'bulk_enable_documents', self.require_perm(Permission.Documents.SELECT)(self.bulk_enable_documents), methods=["POST"])
        self.add_endpoint('/api/data/bulk-disable', 'bulk_disable_documents', self.require_perm(Permission.Documents.SELECT)(self.bulk_disable_documents), methods=["POST"])
        self.add_endpoint('/api/data/stats', 'get_data_stats', self.require_perm(Permission.Documents.VIEW)(self.get_data_stats), methods=["GET"])

        # Data uploader endpoints
        logger.info("Adding data uploader API endpoints")
        self.add_endpoint('/upload', 'upload_page', self.require_perm(Permission.Upload.PAGE)(self.upload_page))
        self.add_endpoint('/api/upload/file', 'upload_file', self.require_perm(Permission.Upload.FILE)(self.upload_file), methods=["POST"])
        self.add_endpoint('/api/upload/url', 'upload_url', self.require_perm(Permission.Upload.URL)(self.upload_url), methods=["POST"])
        self.add_endpoint('/api/upload/git', 'upload_git', self.require_perm(Permission.Upload.GIT)(self.upload_git), methods=["POST", "DELETE"])
        self.add_endpoint('/api/upload/git/refresh', 'refresh_git', self.require_perm(Permission.Upload.GIT)(self.refresh_git), methods=["POST"])
        self.add_endpoint('/api/upload/jira', 'upload_jira', self.require_perm(Permission.Upload.JIRA)(self.upload_jira), methods=["POST"])
        self.add_endpoint('/api/upload/embed', 'trigger_embedding', self.require_perm(Permission.Upload.EMBED)(self.trigger_embedding), methods=["POST"])
        self.add_endpoint('/api/upload/status', 'get_embedding_status', self.require_perm(Permission.Upload.EMBED)(self.get_embedding_status), methods=["GET"])
        self.add_endpoint('/api/upload/documents', 'list_upload_documents', self.require_perm(Permission.Documents.VIEW)(self.list_upload_documents), methods=["GET"])
        self.add_endpoint('/api/upload/documents/grouped', 'list_upload_documents_grouped', self.require_perm(Permission.Documents.VIEW)(self.list_upload_documents_grouped), methods=["GET"])
        self.add_endpoint('/api/upload/documents/<document_hash>/retry', 'retry_document', self.require_perm(Permission.Documents.SELECT)(self.retry_document), methods=["POST"])
        self.add_endpoint('/api/upload/documents/retry-all-failed', 'retry_all_failed', self.require_perm(Permission.Documents.SELECT)(self.retry_all_failed), methods=["POST"])
        self.add_endpoint('/api/sources/git', 'list_git_sources', self.require_perm(Permission.Sources.VIEW)(self.list_git_sources), methods=["GET"])
        self.add_endpoint('/api/sources/jira', 'list_jira_sources', self.require_perm(Permission.Sources.VIEW)(self.list_jira_sources), methods=["GET", "DELETE"])
        self.add_endpoint('/api/sources/schedules', 'source_schedules', self.require_perm(Permission.Sources.SELECT)(self.source_schedules_dispatch), methods=["GET", "PUT"])

        # Database viewer endpoints (admin only)
        logger.info("Adding database viewer API endpoints")
        self.add_endpoint('/admin/database', 'database_viewer_page', self.require_perm(Permission.Admin.DATABASE)(self.database_viewer_page))
        self.add_endpoint('/api/admin/database/tables', 'list_database_tables', self.require_perm(Permission.Admin.DATABASE)(self.list_database_tables), methods=["GET"])
        self.add_endpoint('/api/admin/database/query', 'run_database_query', self.require_perm(Permission.Admin.DATABASE)(self.run_database_query), methods=["POST"])

        # Playbook library endpoints (per-user reusable instruction packs,
        # registered via Blueprint)
        logger.info("Adding playbook API endpoints")
        register_playbooks(
            self.app,
            auth_enabled=self.auth_enabled,
            require_auth=self.require_auth,
            resolve_owner=self._resolve_playbook_owner,
            playbook_svc=self._playbook_svc,
        )

        # Service status board endpoints (registered via Blueprint)
        logger.info("Adding service status board endpoints")
        register_service_alerts(
            self.app,
            pg_config=self.pg_config,
            auth_enabled=self.auth_enabled,
            chat_app_config=self.chat_app_config,
            require_auth=self.require_auth,
        )

        if self.evaluation_service is not None:
            logger.info("Adding QA evaluation console endpoints")
            register_evaluations(
                self.app,
                authorize_request=self.authorize_request,
                service=self.evaluation_service,
            )

        # add unified auth endpoints
        if self.auth_enabled:
            logger.info("Adding unified authentication endpoints")
            self.add_endpoint('/login', 'login', self.login, methods=['GET', 'POST'])
            self.add_endpoint('/logout', 'logout', self.logout)
            self.add_endpoint('/auth/user', 'get_user', self.get_user, methods=['GET'])
            self.add_endpoint('/api/permissions', 'get_permissions', self.get_permissions, methods=['GET'])
            self.add_endpoint('/api/permissions/check', 'check_permission', self.check_permission_endpoint, methods=['POST'])

            
            if self.sso_enabled:
                self.add_endpoint('/redirect', 'sso_callback', self.sso_callback)

    def _set_user_session(self, email: str, name: str, username: str, user_id: str = '', auth_method: str = 'sso', roles: list = None):
        """Set user session with well-defined structure."""
        session['user'] = {
            'email': email,
            'name': name,
            'username': username,
            'id': user_id
        }
        session['logged_in'] = True
        session['auth_method'] = auth_method
        session['roles'] = roles if roles is not None else []

    def _get_session_user_email(self) -> str:
        """Get user email from session. Returns empty string if not logged in."""
        if not session.get('logged_in'):
            return ''
        return session['user']['email']

    def _get_session_roles(self) -> list:
        """Get user roles from session. Returns empty list if not logged in."""
        return session.get('roles', [])

    def _setup_sso(self):
        """Initialize OAuth client for SSO using OpenID Connect"""
        auth_config = self.chat_app_config.get('auth', {})
        sso_config = auth_config.get('sso', {})
        
        # Read client credentials from environment
        client_id = read_secret('SSO_CLIENT_ID')
        client_secret = read_secret('SSO_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            logger.error("SSO is enabled but SSO_CLIENT_ID or SSO_CLIENT_SECRET environment variables are not set")
            self.sso_enabled = False
            return
        
        # Initialize OAuth
        self.oauth = OAuth(self.app)
        
        # Get server metadata URL and client kwargs from config
        server_metadata_url = sso_config.get('server_metadata_url', '')
        authorize_url = sso_config.get('authorize_url', None)
        client_kwargs = sso_config.get('client_kwargs', {'scope': 'openid profile email'})
        
        # Register the OAuth provider
        self.oauth.register(
            name='sso',
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=server_metadata_url,
            authorize_url=authorize_url,
            client_kwargs=client_kwargs
        )
        
        logger.info(f"SSO configured with server: {server_metadata_url}")

    def login(self):
        """Unified login endpoint supporting multiple auth methods"""
        # If user is already logged in, redirect to index
        if session.get('logged_in'):
            return redirect(url_for('index'))
        
        # Handle SSO login initiation
        if request.args.get('method') == 'sso' and self.sso_enabled:
            if not self.oauth:
                return jsonify({'error': 'SSO not configured'}), 400
            redirect_uri = url_for('sso_callback', _external=True)
            logger.info(f"Initiating SSO login with redirect URI: {redirect_uri}")
            return self.oauth.sso.authorize_redirect(redirect_uri)
        
        # Handle basic auth login form submission
        if request.method == 'POST' and self.basic_auth_enabled:
            username = request.form.get('username')
            password = request.form.get('password')
            
            if check_credentials(username, password, self.salt, self.app.config['ACCOUNTS_FOLDER']):
                self._set_user_session(
                    email=username,
                    name=username,
                    username=username,
                    auth_method='basic',
                    roles=[]
                )
                logger.info(f"Basic auth login successful for user: {username}")
                return redirect(url_for('index'))
            else:
                flash('Invalid credentials')
        
        # Render login page with available auth methods
        return render_template('login.html', 
                             sso_enabled=self.sso_enabled, 
                             basic_auth_enabled=self.basic_auth_enabled)

    def logout(self):
        """Unified logout endpoint for all auth methods"""
        auth_method = session.get('auth_method', 'unknown')
        user_email = self._get_session_user_email() or 'unknown'
        user_roles = session.get('roles', [])
        
        # Clear all session data including roles
        session.pop('user', None)
        session.pop('logged_in', None)
        session.pop('auth_method', None)
        session.pop('roles', None)
        
        # Log logout event
        log_authentication_event(
            user=user_email,
            event_type='logout',
            success=True,
            method=auth_method,
            details=f"Previous roles: {user_roles}"
        )
        
        logger.info(f"User {user_email} logged out (method: {auth_method})")
        flash('You have been logged out successfully')
        return redirect(url_for('landing'))

    def sso_callback(self):
        """Handle OAuth callback from SSO provider with RBAC role extraction"""
        if not self.sso_enabled or not self.oauth:
            return jsonify({'error': 'SSO not enabled'}), 400
        
        try:
            # Get the token from the callback
            token = self.oauth.sso.authorize_access_token()
            
            # Parse the user info from the token
            user_info = token.get('userinfo')
            if not user_info:
                # If userinfo is not in token, fetch it
                user_info = self.oauth.sso.userinfo(token=token)
            
            user_email = user_info.get('email', user_info.get('preferred_username', 'unknown'))
            
            # Extract roles from JWT token using RBAC module
            # This handles role validation and default role assignment
            user_roles = get_user_roles(token, user_email)
            
            # Upsert the SSO user into the users table so that conversation_metadata
            # can reference user_id via the FK constraint.
            sso_user_id = user_info.get('sub', '')
            if sso_user_id:
                try:
                    user_service = UserService(pg_config=self.pg_config)
                    user_service.get_or_create_user(
                        user_id=sso_user_id,
                        auth_provider='sso',
                        display_name=user_info.get('name', user_info.get('preferred_username', '')),
                        email=user_info.get('email', ''),
                    )
                    # Track the login (login_count / last_login_at) — best-effort; the enclosing
                    # try means a tracking failure can never block the login itself.
                    user_service.record_login(sso_user_id)
                except Exception as ue:
                    logger.warning(f"Failed to upsert SSO user {sso_user_id} into users table: {ue}")

            # Store user information in session (normalized structure)
            self._set_user_session(
                email=user_info.get('email', ''),
                name=user_info.get('name', user_info.get('preferred_username', '')),
                username=user_info.get('preferred_username', user_info.get('email', '')),
                user_id=sso_user_id,
                auth_method='sso',
                roles=user_roles
            )
            
            # Log successful authentication
            log_authentication_event(
                user=user_email,
                event_type='login',
                success=True,
                method='sso',
                details=f"Roles: {user_roles}"
            )
            
            logger.info(f"SSO login successful for user: {user_email} with roles: {user_roles}")
            
            # Redirect to main page
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"SSO callback error: {str(e)}")
            log_authentication_event(
                user='unknown',
                event_type='login',
                success=False,
                method='sso',
                details=str(e)
            )
            flash(f"Authentication failed: {str(e)}")
            return redirect(url_for('login'))

    def get_user(self):
        """API endpoint to get current user information including roles and permissions"""
        if session.get('logged_in'):
            user = session.get('user', {})
            roles = session.get('roles', [])
            
            # Get permission context for the frontend
            permissions = get_permission_context()
            
            return jsonify({
                'logged_in': True,
                'email': user.get('email', ''),
                'name': user.get('name', ''),
                'auth_method': session.get('auth_method', 'unknown'),
                'auth_enabled': self.auth_enabled,
                'roles': roles,
                'permissions': permissions
            })
        return jsonify({
            'logged_in': False,
            'auth_enabled': self.auth_enabled,
            'roles': [],
            'permissions': get_permission_context()
        })

    def _authenticate_bearer_token(self):
        """Authenticate via Bearer token in Authorization header.

        If a valid Bearer token is present, decode the JWT claims,
        extract user info and roles, and populate the Flask session
        so downstream code works identically to SSO-cookie auth.

        Returns True if authentication succeeded, False otherwise.
        """
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return False

        token_string = auth_header[7:]
        claims = decode_jwt_claims(token_string)
        if not claims:
            return False

        # Check expiration
        import time as _time
        exp = claims.get('exp')
        if exp and _time.time() > exp:
            logger.warning("Bearer token has expired")
            return False

        # Extract user info from JWT claims (Keycloak / CERN SSO format)
        email = claims.get('email', claims.get('preferred_username', 'unknown'))
        name = claims.get('name', claims.get('preferred_username', ''))
        username = claims.get('preferred_username', claims.get('email', ''))
        user_id = claims.get('sub', '')

        # Extract roles via the existing RBAC helper
        user_roles = get_user_roles(claims, email)

        # Upsert user into the users table so that conversation_metadata
        # can reference user_id via the FK constraint.
        if user_id:
            try:
                user_service = UserService(pg_config=self.pg_config)
                user_service.get_or_create_user(
                    user_id=user_id,
                    auth_provider='sso',
                    display_name=name,
                    email=email,
                )
            except Exception as ue:
                logger.warning(f"Failed to upsert bearer user {user_id} into users table: {ue}")

        # Populate the session so all downstream code works unchanged
        self._set_user_session(
            email=email,
            name=name,
            username=username,
            user_id=user_id,
            auth_method='bearer',
            roles=user_roles
        )

        log_authentication_event(
            user=email,
            event_type='bearer_auth',
            success=True,
            method='bearer',
            details=f"Roles: {user_roles}"
        )
        return True

    def require_auth(self, f):
        """Decorator to require authentication for routes.

        When SSO is enabled and anonymous access is blocked (sso.allow_anonymous: false),
        unauthenticated users are redirected to SSO login instead of getting a 401 error.

        Also supports Bearer token authentication via the Authorization header,
        allowing programmatic / API access without a browser session.
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not self.auth_enabled:
                # If auth is not enabled, allow access
                return f(*args, **kwargs)

            if not session.get('logged_in'):
                # Try Bearer token authentication for API requests
                if self._authenticate_bearer_token():
                    return f(*args, **kwargs)

                # Check if SSO is enabled and anonymous access is blocked
                if self.sso_enabled:
                    registry = get_registry()
                    if not registry.allow_anonymous:
                        # Log the redirect attempt
                        log_authentication_event(
                            user='anonymous',
                            event_type='anonymous_redirect',
                            success=False,
                            method='web',
                            details=f"path={request.path}, method={request.method}"
                        )
                        # For API requests return 401 instead of redirect
                        if request.path.startswith('/api/'):
                            return jsonify({'error': 'Unauthorized', 'message': 'Authentication required'}), 401
                        # Redirect to login page which will trigger SSO
                        return redirect(url_for('login'))

                # Return 401 Unauthorized response for API requests
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Unauthorized', 'message': 'Authentication required'}), 401
                else:
                    return redirect(url_for('login'))

            return f(*args, **kwargs)
        return decorated_function

    def authorize_request(self, permission: str):
        """Return an error response unless the current request has ``permission``."""
        if not self.auth_enabled:
            return None

        if not session.get("logged_in"):
            if not self._authenticate_bearer_token():
                if self.sso_enabled:
                    registry = get_registry()
                    if not registry.allow_anonymous:
                        if request.path.startswith("/api/"):
                            return (
                                jsonify(
                                    {
                                        "error": "Unauthorized",
                                        "message": "Authentication required",
                                    }
                                ),
                                401,
                            )
                        return redirect(url_for("login"))
                return (
                    jsonify(
                        {"error": "Unauthorized", "message": "Authentication required"}
                    ),
                    401,
                )

        roles = session.get("roles", [])
        if not has_permission(permission, roles):
            user_email = session.get("user", {}).get("email", "unknown")
            logger.warning(
                f"Permission denied: user {user_email} with roles {roles} lacks '{permission}'"
            )
            from src.utils.rbac.audit import log_permission_check

            log_permission_check(
                permission=permission,
                granted=False,
                user=user_email,
                roles=roles,
                endpoint=request.path,
            )
            return (
                jsonify(
                    {
                        "error": "Forbidden",
                        "message": f"Permission denied: requires {permission}",
                        "required_permission": permission,
                    }
                ),
                403,
            )

        return None

    def require_perm(self, permission: str):
        """Decorate a route to require authentication and ``permission``."""

        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                error_response = self.authorize_request(permission)
                if error_response is not None:
                    return error_response
                return f(*args, **kwargs)

            return decorated_function

        return decorator

    def health(self):
        return jsonify({"status": "OK"}), 200

    def get_permissions(self):
        """API endpoint to get current user's permissions"""
        if not session.get('logged_in'):
            return jsonify({
                'logged_in': False,
                'permissions': get_permission_context()
            })
        
        permissions = get_permission_context()
        return jsonify({
            'logged_in': True,
            'roles': session.get('roles', []),
            'permissions': permissions
        })
    
    def check_permission_endpoint(self):
        """API endpoint to check if user has a specific permission"""
        if not session.get('logged_in'):
            return jsonify({
                'error': 'Authentication required',
                'has_permission': False
            }), 401
        
        data = request.get_json()
        if not data or 'permission' not in data:
            return jsonify({
                'error': 'Permission name required',
                'has_permission': False
            }), 400
        
        permission = data['permission']
        roles = session.get('roles', [])
        result = has_permission(permission, roles)
        
        # Get which roles would grant this permission
        registry = get_registry()
        roles_with_permission = registry.get_roles_with_permission(permission)
        
        return jsonify({
            'permission': permission,
            'has_permission': result,
            'user_roles': roles,
            'roles_with_permission': roles_with_permission
        })

    def configs(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value

    def add_endpoint(self, endpoint = None, endpoint_name = None, handler = None, methods = ['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods = methods, *args, **kwargs)

    def run(self, **kwargs):
        self.app.run(**kwargs)

    def _build_provider_config(self, provider_type: ProviderType) -> Optional[ProviderConfig]:
        """Legacy shim: build ProviderConfig from the currently loaded YAML."""
        return _build_provider_config_from_payload(self.config, provider_type)

    def update_config(self):
        """
        Updates the config used by archi for responding to messages.
        Reloads the config and updates the chat wrapper.
        """
        return jsonify({"error": "Config updates must be applied to Postgres; file-based updates are disabled."}), 400

    def get_configs(self):
        """
        Gets the names of configs loaded in archi.


        Returns:
            A json with a response list of the configs names
        """

        config_names = _config_names()
        options = []
        for name in config_names:
            description = ""
            try:
                agent_spec = getattr(self.chat, "agent_spec", None)
                if agent_spec is not None:
                    description = getattr(agent_spec, "name", "") or "No description provided"
                else:
                    description = "No description provided"
            except Exception as exc:
                logger.warning(f"Failed to load config {name} for description: {exc}")
            options.append({"name": name, "description": description})
        timeout_seconds = 600.0
        try:
            chat_cfg = (self.config.get("services", {}) or {}).get("chat_app", {}) or {}
            configured_timeout = chat_cfg.get("client_timeout_seconds", 600)
            if isinstance(configured_timeout, bool):
                raise ValueError("boolean is not allowed")
            parsed_timeout = float(configured_timeout)
            if parsed_timeout > 0:
                timeout_seconds = parsed_timeout
            else:
                raise ValueError("must be positive")
        except Exception as exc:
            logger.warning("Invalid services.chat_app.client_timeout_seconds; using default 600s: %s", exc)

        return jsonify({
            'options': options,
            'client_timeout_seconds': timeout_seconds,
            'client_timeout_ms': int(timeout_seconds * 1000),
        }), 200

    def get_providers(self):
        """
        Get list of all enabled providers and their available models.
        
        Returns:
            JSON with providers list, each containing:
            - type: Provider type (openai, anthropic, etc.)
            - display_name: Human-readable name
            - enabled: Whether the provider has valid credentials
            - models: List of available models
        """
        try:
            from src.archi.providers import (
                list_provider_types,
                get_provider,
                ProviderType,
            )

            providers_data = []
            for provider_type in list_provider_types():
                try:
                    cfg = _build_provider_config_from_payload(self.config, provider_type)
                    provider = get_provider(provider_type, config=cfg) if cfg else get_provider(provider_type)
                    models = provider.list_models()
                    providers_data.append({
                        'type': provider_type.value,
                        'display_name': provider.display_name,
                        'enabled': provider.is_enabled,
                        'default_model': provider.config.default_model,
                        'models': [
                            {
                                'id': m.id,
                                'name': m.name,
                                'display_name': m.display_name,
                                'context_window': m.context_window,
                                'supports_tools': m.supports_tools,
                                'supports_streaming': m.supports_streaming,
                                'supports_vision': m.supports_vision,
                            }
                            for m in models
                        ],
                    })
                except Exception as e:
                    logger.warning(f"Failed to get provider {provider_type}: {e}")
                    providers_data.append({
                        'type': provider_type.value,
                        'display_name': provider_type.value.title(),
                        'enabled': False,
                        'error': str(e),
                        'models': [],
                    })

            return jsonify({'providers': providers_data}), 200
        except ImportError as e:
            logger.error(f"Providers module not available: {e}")
            return jsonify({'error': 'Providers module not available', 'providers': []}), 200
        except Exception as e:
            logger.error(f"Error getting providers: {e}")
            return jsonify({'error': str(e)}), 500

    def get_pipeline_default_model(self):
        """
        Get the default model configured for the active chat pipeline.

        Returns:
            JSON with pipeline name and provider/model reference (if available).
        """
        try:
            chat_cfg = self.config.get("services", {}).get("chat_app", {})
            agent_class = ChatWrapper._get_agent_class_from_cfg(chat_cfg)
            provider = chat_cfg.get("default_provider")
            model = chat_cfg.get("default_model")
            model_name = f"{provider}/{model}" if provider and model else None
            return jsonify({
                "pipeline": agent_class,
                "provider": provider,
                "model": model,
                "model_class": provider,
                "model_name": model_name,
            }), 200
        except Exception as e:
            logger.error(f"Error getting pipeline default model: {e}")
            return jsonify({"error": str(e)}), 500

    def _get_agents_dir(self) -> Path:
        agents_dir = self.services_config.get("chat_app", {}).get("agents_dir") or "/root/archi/agents"
        return Path(agents_dir).expanduser()

    def _get_ab_agents_dir(self) -> Path:
        chat_cfg = self.services_config.get("chat_app", {}) or {}
        path, _ = resolve_ab_agents_dir(chat_cfg)
        return path

    def _get_agent_scope(self) -> str:
        scope = None
        if request.is_json and request.json:
            scope = request.json.get("scope")
        if not scope:
            scope = request.args.get("scope", "default")
        scope = str(scope or "default").strip().lower()
        return "ab" if scope == "ab" else "default"

    def _get_agent_dir_for_scope(self, scope: str, *, create: bool = False) -> Path:
        if scope == "ab":
            raise PermissionError("A/B agent specs are stored in the database")
        directory = self._get_agents_dir()
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _get_ab_agent_spec_service(self) -> ABAgentSpecService:
        if hasattr(self, "chat") and getattr(self.chat, "ab_agent_spec_service", None) is not None:
            return self.chat.ab_agent_spec_service
        if not hasattr(self, "_ab_agent_spec_service"):
            self._ab_agent_spec_service = ABAgentSpecService(pg_config=self.pg_config)
        return self._ab_agent_spec_service

    def _get_ab_runtime_defaults(self) -> Dict[str, Any]:
        chat_cfg = self.services_config.get("chat_app", {}) or {}
        data_cfg = self.config.get("data_manager", {}) or {}
        retrievers_cfg = data_cfg.get("retrievers", {}) or {}
        hybrid_cfg = retrievers_cfg.get("hybrid_retriever", {}) or {}
        return {
            "provider": chat_cfg.get("default_provider"),
            "model": chat_cfg.get("default_model"),
            "recursion_limit": int(chat_cfg.get("recursion_limit", 50) or 50),
            "num_documents_to_retrieve": int(hybrid_cfg.get("num_documents_to_retrieve", 5) or 5),
            "ab_catalog_source": "database",
        }

    @staticmethod
    def _normalize_ab_variant_details(raw_variants: Any) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        if not isinstance(raw_variants, list):
            return details
        for entry in raw_variants:
            if isinstance(entry, str):
                details.append({"label": entry.strip(), "agent_spec": ""})
                continue
            if not isinstance(entry, dict):
                details.append({"label": "", "agent_spec": ""})
                continue
            details.append({
                "label": str(entry.get("label") or entry.get("name") or "").strip(),
                "agent_spec": str(entry.get("agent_spec") or "").strip(),
                "provider": entry.get("provider") or None,
                "model": entry.get("model") or None,
                "num_documents_to_retrieve": entry.get("num_documents_to_retrieve"),
                "recursion_limit": entry.get("recursion_limit"),
                "agent_spec_id": entry.get("agent_spec_id"),
                "agent_spec_name": entry.get("agent_spec_name"),
                "agent_spec_version_id": entry.get("agent_spec_version_id"),
                "agent_spec_version_number": entry.get("agent_spec_version_number"),
                "agent_spec_content_hash": entry.get("agent_spec_content_hash"),
                "agent_spec_tools": entry.get("agent_spec_tools"),
                "agent_spec_prompt_hash": entry.get("agent_spec_prompt_hash"),
            })
        return details

    @staticmethod
    def _get_ab_setting(
        mapping: Dict[str, Any],
        canonical_key: str,
        legacy_key: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        if isinstance(mapping, dict):
            if canonical_key in mapping:
                return mapping.get(canonical_key)
            if legacy_key and legacy_key in mapping:
                return mapping.get(legacy_key)
        return default

    @classmethod
    def _get_ab_pool_champion(cls, raw_pool: Dict[str, Any]) -> str:
        return str(cls._get_ab_setting(raw_pool, "champion", "control", "") or "").strip()

    def _build_admin_ab_pool_payload(self) -> Dict[str, Any]:
        chat_cfg = self.services_config.get("chat_app", {}) or {}
        raw_ab_cfg = (chat_cfg.get("ab_testing") or {}) if isinstance(chat_cfg.get("ab_testing"), dict) else {}
        raw_pool = raw_ab_cfg.get("pool") or {}
        state = getattr(self.chat, "ab_pool_state", None)
        active_pool = getattr(self.chat, "ab_pool", None)
        participation = self._get_ab_participation_state()
        defaults = self._get_ab_runtime_defaults()

        variant_details = self._normalize_ab_variant_details(raw_pool.get("variants"))
        champion = self._get_ab_pool_champion(raw_pool)
        comparison_rate = self._get_ab_setting(raw_ab_cfg, "comparison_rate", "sample_rate", 1.0)
        variant_label_mode = normalize_ab_disclosure_mode(
            self._get_ab_setting(
                raw_ab_cfg, "variant_label_mode", "disclosure_mode", DEFAULT_DISCLOSURE_MODE
            )
        )
        activity_panel_default_state = normalize_ab_trace_mode(
            self._get_ab_setting(
                raw_ab_cfg,
                "activity_panel_default_state",
                "default_trace_mode",
                DEFAULT_TRACE_MODE,
            )
        )
        max_pending = self._get_ab_setting(
            raw_ab_cfg,
            "max_pending_comparisons_per_conversation",
            "max_pending_per_conversation",
            1,
        )

        if active_pool:
            variant_details = [variant.to_meta() for variant in active_pool.variants]
            champion = active_pool.champion_name
            comparison_rate = active_pool.comparison_rate
            variant_label_mode = active_pool.variant_label_mode
            activity_panel_default_state = active_pool.activity_panel_default_state
            max_pending = active_pool.max_pending_comparisons_per_conversation

        return {
            "success": True,
            "is_admin": self._is_admin_request(),
            "can_view": self._can_view_ab_testing(),
            "can_manage": self._can_manage_ab_testing(),
            "can_view_metrics": self._can_view_ab_metrics(),
            "can_participate": participation["can_participate"],
            "participant_eligible": participation["eligible"],
            "participant_reason": participation["reason"],
            "participant_targeted": participation["targeted"],
            "enabled": bool(active_pool and active_pool.enabled),
            "enabled_requested": bool(raw_ab_cfg.get("enabled", False)),
            "champion": champion,
            "variants": [variant.get("label", "") for variant in variant_details if variant.get("label")],
            "variant_details": variant_details,
            "variant_count": len(variant_details),
            "comparison_rate": comparison_rate,
            "default_comparison_rate": float(
                self._get_ab_setting(raw_ab_cfg, "comparison_rate", "sample_rate", comparison_rate or 1.0)
            ),
            "eligible_roles": list(self._get_ab_setting(raw_ab_cfg, "eligible_roles", "target_roles", []) or []),
            "eligible_permissions": list(
                self._get_ab_setting(raw_ab_cfg, "eligible_permissions", "target_permissions", []) or []
            ),
            "max_pending_comparisons_per_conversation": max_pending,
            "variant_label_mode": variant_label_mode,
            "activity_panel_default_state": activity_panel_default_state,
            "defaults": defaults,
            "warnings": list(getattr(state, "warnings", []) or []),
            "import_diagnostics": dict(getattr(self.chat, "ab_agent_import_diagnostics", {}) or {}),
        }

    def _resolve_ab_variants(
        self,
        variant_items: Any,
        *,
        existing_variants: Optional[Dict[str, ABVariant]] = None,
    ) -> tuple[list[ABVariant], list[str]]:
        if not isinstance(variant_items, list) or len(variant_items) < 2:
            raise ABPoolError("At least 2 variants are required")

        parsed_labels: List[str] = []
        for item in variant_items:
            if isinstance(item, str):
                label = item.strip()
            elif isinstance(item, dict):
                label = str(item.get('label') or item.get('name') or '').strip()
            else:
                label = ''
            if not label:
                raise ABPoolError("All variants must include a non-empty label")
            parsed_labels.append(label)

        if len(set(parsed_labels)) != len(parsed_labels):
            raise ABPoolError("Variant labels must be unique")

        ab_specs = self._get_ab_agent_spec_service()
        spec_records = ab_specs.list_specs()
        spec_map: Dict[str, ABAgentSpecRecord] = {record.name: record for record in spec_records}

        chat_cfg = self.chat.services_config.get("chat_app", {}) if hasattr(self, "chat") else self.services_config.get("chat_app", {})
        default_provider = chat_cfg.get("default_provider", "")
        default_model = chat_cfg.get("default_model", "")
        existing_variants = existing_variants or {}

        variants: List[ABVariant] = []
        for item, label in zip(variant_items, parsed_labels):
            item_cfg = item if isinstance(item, dict) else {}
            explicit_agent_spec = str(item_cfg.get('agent_spec') or '').strip()
            if explicit_agent_spec:
                if Path(explicit_agent_spec).name != explicit_agent_spec:
                    raise ABPoolError(
                        f"Variant '{label}' must use an A/B catalog filename"
                    )
                record = ab_specs.get_spec_by_filename(explicit_agent_spec)
                if record is None:
                    raise ABPoolError(
                        f"Variant '{label}' references missing agent_spec '{explicit_agent_spec}'"
                    )
            else:
                record = spec_map.get(label)
                if not record:
                    raise ABPoolError(
                        f"Agent '{label}' not found in the A/B catalog; provide agent_spec explicitly"
                    )

            existing = existing_variants.get(label)
            provider_override = item_cfg.get('provider')
            if provider_override is None and existing and existing.provider and existing.provider != default_provider:
                provider_override = existing.provider
            model_override = item_cfg.get('model')
            if model_override is None and existing and existing.model and existing.model != default_model:
                model_override = existing.model

            variants.append(ABVariant(
                label=label,
                agent_spec=record.filename,
                provider=provider_override or None,
                model=model_override or None,
                num_documents_to_retrieve=item_cfg.get('num_documents_to_retrieve') or (
                    existing.num_documents_to_retrieve if existing else None
                ),
                recursion_limit=item_cfg.get('recursion_limit') or (
                    existing.recursion_limit if existing else None
                ),
                agent_spec_id=record.spec_id,
                agent_spec_name=record.name,
                agent_spec_version_id=record.version_id,
                agent_spec_version_number=record.version_number,
                agent_spec_content_hash=record.content_hash,
                agent_spec_tools=list(record.tools),
                agent_spec_prompt_hash=record.prompt_hash,
            ))

        return variants, parsed_labels

    def _persist_ab_pool_config(
        self,
        *,
        enabled: bool,
        champion_name: str,
        variants: List[ABVariant],
        comparison_rate: float,
        variant_label_mode: str,
        activity_panel_default_state: str,
        max_pending_comparisons_per_conversation: int,
    ) -> None:
        self.config_service.update_services_config({
            "chat_app": {
                "ab_testing": {
                    "enabled": enabled,
                    "comparison_rate": comparison_rate,
                    "variant_label_mode": variant_label_mode,
                    "activity_panel_default_state": activity_panel_default_state,
                    "max_pending_comparisons_per_conversation": max_pending_comparisons_per_conversation,
                    "pool": {
                        "champion": champion_name,
                        "variants": [variant.to_meta() for variant in variants],
                    },
                }
            }
        })
        self._refresh_runtime_config()

    def _ndjson_response(self, event_iter) -> Response:
        """Wrap an event iterator as an NDJSON streaming Response with standard headers."""
        def _event_stream() -> Iterator[str]:
            padding = " " * 2048
            yield json.dumps({"type": "meta", "event": "stream_started", "padding": padding}) + "\n"
            for event in event_iter:
                yield json.dumps(event, default=str) + "\n"

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
            "Content-Type": "application/x-ndjson",
        }
        return Response(stream_with_context(_event_stream()), headers=headers)

    def _get_agent_class_name(self) -> Optional[str]:
        chat_cfg = self.services_config.get("chat_app", {})
        return ChatWrapper._get_agent_class_from_cfg(chat_cfg)

    def _get_agent_tool_registry(self) -> List[str]:
        agent_class = self._get_agent_class_name()
        if not agent_class:
            return []
        try:
            from src.archi import pipelines
            agent_cls = getattr(pipelines, agent_class, None)
        except Exception as exc:
            logger.warning("Failed to load pipeline class %s: %s", agent_class, exc)
            return []
        if not agent_cls or not hasattr(agent_cls, "get_tool_registry"):
            return []
        try:
            dummy = agent_cls.__new__(agent_cls)
            registry = agent_cls.get_tool_registry(dummy) or {}
            return sorted([name for name in registry.keys() if isinstance(name, str)])
        except Exception as exc:
            logger.warning("Failed to read tool registry for %s: %s", agent_class, exc)
            return []

    def _get_agent_tools(self) -> List[Dict[str, str]]:
        agent_class = self._get_agent_class_name()
        if not agent_class:
            return []
        try:
            from src.archi import pipelines
            agent_cls = getattr(pipelines, agent_class, None)
        except Exception as exc:
            logger.warning("Failed to load pipeline class %s: %s", agent_class, exc)
            return []
        if not agent_cls or not hasattr(agent_cls, "get_tool_registry"):
            return []
        try:
            dummy = agent_cls.__new__(agent_cls)
            registry = agent_cls.get_tool_registry(dummy) or {}
            descriptions = {}
            if hasattr(agent_cls, "get_tool_descriptions"):
                try:
                    descriptions = agent_cls.get_tool_descriptions(dummy) or {}
                except Exception:
                    descriptions = {}
            tools = []
            for name in sorted([n for n in registry.keys() if isinstance(n, str)]):
                tools.append({
                    "name": name,
                    "description": descriptions.get(name, ""),
                })
            return tools
        except Exception as exc:
            logger.warning("Failed to read tool registry for %s: %s", agent_class, exc)
            return []

    def _build_agent_template(self, name: str, tools: List[str]) -> str:
        tools_block = "\n".join(f"  - {tool}" for tool in tools) if tools else "  - <tool_name>"
        return (
            "---\n"
            f"name: {name}\n"
            "tools:\n"
            f"{tools_block}\n"
            "---\n\n"
            "Write your system prompt here.\n\n"
        )

    def _build_agent_template_payload(self, name: str, *, scope: str = "default") -> Dict[str, Any]:
        tool_items = self._get_agent_tools()
        tools = [tool["name"] for tool in tool_items]
        return {
            "name": name,
            "tools": tool_items,
            "prompt": "Write your system prompt here.",
            "template": self._build_agent_template(name, tools),
            "scope": scope,
        }

    def _build_ab_agent_content(self, name: str, tools: List[str], prompt: str) -> str:
        normalized_name = str(name or "").strip()
        normalized_prompt = str(prompt or "").strip()
        normalized_tools = [
            str(tool).strip()
            for tool in (tools or [])
            if isinstance(tool, str) and str(tool).strip()
        ]
        if not normalized_name:
            raise AgentSpecError("Agent name is required.")
        if not normalized_tools:
            raise AgentSpecError("At least one tool is required.")
        if not normalized_prompt:
            raise AgentSpecError("Prompt body is required.")
        frontmatter = yaml.safe_dump(
            {
                "name": normalized_name,
                "ab_only": True,
                "tools": normalized_tools,
            },
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        ).strip()
        return f"---\n{frontmatter}\n---\n\n{normalized_prompt}\n"

    def _list_ab_agent_catalog_payload(self) -> Dict[str, Any]:
        agents = []
        for record in self._get_ab_agent_spec_service().list_specs():
            agents.append({"name": record.name, "filename": record.filename, "ab_only": True})
        return {
            "agents": agents,
            "active_name": None,
            "scope": "ab",
            "directory": None,
        }

    def list_ab_agents(self):
        try:
            if not self._can_view_ab_testing():
                return jsonify({"error": "A/B agent visibility requires A/B page access"}), 403
            return jsonify(self._list_ab_agent_catalog_payload()), 200
        except Exception as exc:
            logger.error(f"Error listing A/B agents: {exc}")
            return jsonify({"error": str(exc)}), 500

    def get_ab_agent_template(self):
        try:
            if not self._can_manage_ab_testing():
                return jsonify({"error": "A/B agent management requires admin access"}), 403
            agent_name = request.args.get("name") or "New A/B Agent"
            return jsonify(self._build_agent_template_payload(agent_name, scope="ab")), 200
        except Exception as exc:
            logger.error(f"Error building A/B agent template: {exc}")
            return jsonify({"error": str(exc)}), 500

    def save_ab_agent_spec(self):
        try:
            if not self._can_manage_ab_testing():
                return jsonify({"error": "A/B agent management requires admin access"}), 403
            data = request.get_json(force=True) or {}
            name = data.get("name")
            tools = data.get("tools")
            prompt = data.get("prompt")
            if not isinstance(tools, list):
                return jsonify({"error": "tools must be a list"}), 400
            content = self._build_ab_agent_content(name, tools, prompt)
            created_by = (
                session.get("user", {}).get("email")
                or session.get("user", {}).get("id")
                or data.get("client_id")
                or "system"
            )
            record = self._get_ab_agent_spec_service().save_spec(
                content,
                created_by=created_by,
            )
            return jsonify({
                "success": True,
                "name": record.name,
                "filename": record.filename,
                "path": None,
                "scope": "ab",
            }), 200
        except AgentSpecError as exc:
            logger.error(f"Invalid A/B agent spec: {exc}")
            return jsonify({"error": f"Invalid agent spec: {exc}"}), 400
        except Exception as exc:
            logger.error(f"Error saving A/B agent spec: {exc}")
            return jsonify({"error": str(exc)}), 500

    def list_agents(self):
        """
        List available agent specs for the dropdown.
        """
        try:
            scope = self._get_agent_scope()
            if scope == "ab":
                if not self._can_view_ab_testing():
                    return jsonify({"error": "A/B agent visibility requires A/B view access"}), 403
                return jsonify(self._list_ab_agent_catalog_payload()), 200
            agents_dir = self._get_agent_dir_for_scope(scope, create=(scope == "ab"))
            agent_files = list_agent_files(agents_dir)
            agents = []
            for path in agent_files:
                try:
                    spec = load_agent_spec(path)
                    agents.append({"name": spec.name, "filename": path.name, "ab_only": spec.ab_only})
                except AgentSpecError as exc:
                    logger.warning("Skipping invalid agent spec %s: %s", path, exc)
            active_name = None
            if scope != "ab":
                try:
                    dynamic = get_dynamic_config()
                except Exception:
                    dynamic = None
                active_name = getattr(dynamic, "active_agent_name", None) if dynamic else None
                if not active_name:
                    active_spec = getattr(self.chat, "agent_spec", None)
                    active_name = getattr(active_spec, "name", None)
            return jsonify({
                "agents": agents,
                "active_name": active_name,
                "scope": scope,
                "directory": str(agents_dir),
            }), 200
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except Exception as exc:
            logger.error(f"Error listing agents: {exc}")
            return jsonify({"error": str(exc)}), 500

    def get_agent_spec(self):
        """
        Fetch a single agent spec by name.
        """
        try:
            scope = self._get_agent_scope()
            name = request.args.get("name")
            filename = request.args.get("filename")
            if not name and not filename:
                return jsonify({"error": "name or filename parameter required"}), 400
            if scope == "ab":
                if not self._can_view_ab_testing():
                    return jsonify({"error": "A/B agent visibility requires A/B view access"}), 403
                if filename:
                    record = self._get_ab_agent_spec_service().get_spec_by_filename(filename)
                else:
                    record = self._get_ab_agent_spec_service().get_spec_by_name(name)
                if record is None:
                    lookup = filename or name
                    return jsonify({"error": f"Agent '{lookup}' not found"}), 404
                return jsonify({
                    "name": record.name,
                    "filename": record.filename,
                    "content": record.content,
                    "tools": list(record.tools),
                    "prompt": record.prompt,
                    "scope": scope,
                }), 200
            agents_dir = self._get_agent_dir_for_scope(scope, create=(scope == "ab"))
            for path in list_agent_files(agents_dir):
                try:
                    spec = load_agent_spec(path)
                except AgentSpecError:
                    continue
                if spec.name == name:
                    return jsonify({
                        "name": spec.name,
                        "filename": path.name,
                        "content": path.read_text(),
                        "tools": list(getattr(spec, "tools", []) or []),
                        "prompt": getattr(spec, "prompt", ""),
                        "scope": scope,
                    }), 200
            return jsonify({"error": f"Agent '{name}' not found"}), 404
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except Exception as exc:
            logger.error(f"Error fetching agent spec: {exc}")
            return jsonify({"error": str(exc)}), 500

    def get_agent_template(self):
        """
        Return a prefilled agent spec template and available tools.
        """
        try:
            scope = self._get_agent_scope()
            if scope == "ab" and not self._can_manage_ab_testing():
                return jsonify({"error": "A/B agent management requires admin access"}), 403
            agent_name = request.args.get("name") or "New Agent"
            return jsonify(self._build_agent_template_payload(agent_name, scope=scope)), 200
        except Exception as exc:
            logger.error(f"Error building agent template: {exc}")
            return jsonify({'error': str(exc)}), 500

    def set_active_agent(self):
        """
        Persist the active agent name in dynamic config.
        """
        try:
            data = request.get_json() or {}
            name = data.get("name")
            client_id = data.get("client_id") or "system"
            if not name:
                return jsonify({"error": "name is required"}), 400

            agents_dir = self._get_agents_dir()
            exists = False
            for path in list_agent_files(agents_dir):
                try:
                    spec = load_agent_spec(path)
                except AgentSpecError:
                    continue
                if spec.name == name:
                    exists = True
                    break
            if not exists:
                return jsonify({"error": f"Agent '{name}' not found"}), 404

            cfg = ConfigService(pg_config=self.pg_config)
            cfg.update_dynamic_config(active_agent_name=name, updated_by=client_id)

            return jsonify({
                "success": True,
                "active_name": name,
            }), 200
        except Exception as exc:
            logger.error(f"Error setting active agent: {exc}")
            return jsonify({"error": str(exc)}), 500

    def save_agent_spec(self):
        """
        Create or update an agent spec by name.
        """
        try:
            data = request.get_json() or {}
            scope = self._get_agent_scope()
            content = data.get("content")
            mode = data.get("mode", "create")
            existing_name = data.get("existing_name")
            if not content or not isinstance(content, str):
                return jsonify({'error': 'Content is required'}), 400

            if scope == "ab":
                if not self._can_manage_ab_testing():
                    return jsonify({"error": "A/B agent management requires admin access"}), 403
                created_by = session.get("user", {}).get("email") or session.get("user", {}).get("id") or data.get("client_id") or "system"
                ab_service = self._get_ab_agent_spec_service()
                if mode == "edit" or existing_name:
                    return jsonify({
                        "error": "Editing A/B agent specs is not supported. Create a new A/B agent spec instead."
                    }), 400
                record = ab_service.save_spec(
                    content,
                    created_by=created_by,
                )
                return jsonify({
                    'success': True,
                    'name': record.name,
                    'filename': record.filename,
                    'path': None,
                    'scope': scope,
                }), 200

            agents_dir = self._get_agent_dir_for_scope(scope, create=True)

            if mode == "edit" or existing_name:
                if not existing_name:
                    return jsonify({'error': 'existing_name required for edit'}), 400
                target_path = None
                for path in list_agent_files(agents_dir):
                    try:
                        spec = load_agent_spec(path)
                    except AgentSpecError:
                        continue
                    if spec.name == existing_name:
                        target_path = path
                        break
                if not target_path:
                    return jsonify({'error': f"Agent '{existing_name}' not found"}), 404
                new_spec = load_agent_spec_from_text(content)
                if new_spec.name != existing_name:
                    return jsonify({
                        'error': 'Agent name cannot be changed in edit mode. Create or clone a new agent instead.'
                    }), 400
                for path in list_agent_files(agents_dir):
                    if path == target_path:
                        continue
                    try:
                        spec = load_agent_spec(path)
                    except AgentSpecError:
                        continue
                    if spec.name == new_spec.name:
                        return jsonify({'error': f"Agent name '{new_spec.name}' already exists"}), 409
                target_path.write_text(content)
                return jsonify({
                    'success': True,
                    'name': new_spec.name,
                    'filename': target_path.name,
                    'path': str(target_path),
                    'scope': scope,
                }), 200

            # create mode
            # derive name from content to build filename and enforce uniqueness
            spec = load_agent_spec_from_text(content)
            existing_names = []
            for path in list_agent_files(agents_dir):
                try:
                    existing = load_agent_spec(path)
                    existing_names.append(existing.name)
                except AgentSpecError:
                    continue
            if spec.name in existing_names:
                return jsonify({'error': f"Agent name '{spec.name}' already exists"}), 409
            filename = slugify_agent_name(spec.name)
            target_path = agents_dir / filename
            if target_path.exists():
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                counter = 2
                while True:
                    candidate = agents_dir / f"{stem}-{counter}{suffix}"
                    if not candidate.exists():
                        target_path = candidate
                        break
                    counter += 1
            target_path.write_text(content)
            return jsonify({
                'success': True,
                'name': spec.name,
                'filename': target_path.name,
                'path': str(target_path),
                'scope': scope,
            }), 200
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except AgentSpecError as exc:
            logger.error(f"Invalid agent spec: {exc}")
            return jsonify({'error': f'Invalid agent spec: {exc}'}), 400
        except Exception as exc:
            logger.error(f"Error saving agent spec: {exc}")
            return jsonify({'error': str(exc)}), 500

    def delete_agent_spec(self):
        """
        Delete an agent spec by name.
        """
        try:
            data = request.get_json() or {}
            scope = self._get_agent_scope()
            name = data.get("name")
            if not name:
                return jsonify({"error": "name is required"}), 400
            name = name.strip()
            if name.lower().startswith("name:"):
                name = name.split(":", 1)[1].strip()

            if scope == "ab":
                if not self._can_manage_ab_testing():
                    return jsonify({"error": "A/B agent management requires admin access"}), 403
                deleted = self._get_ab_agent_spec_service().delete_spec_by_name(name)
                if not deleted:
                    return jsonify({"error": f"Agent '{name}' not found"}), 404
                return jsonify({"success": True, "deleted": name}), 200

            agents_dir = self._get_agent_dir_for_scope(scope, create=(scope == "ab"))
            target_path = None
            for path in list_agent_files(agents_dir):
                try:
                    spec = load_agent_spec(path)
                except AgentSpecError:
                    continue
                if spec.name == name:
                    target_path = path
                    break
            if not target_path:
                return jsonify({"error": f"Agent '{name}' not found"}), 404

            target_path.unlink()
            if scope != "ab":
                try:
                    dynamic = get_dynamic_config()
                except Exception:
                    dynamic = None
                if dynamic and dynamic.active_agent_name == name:
                    cfg = ConfigService(pg_config=self.pg_config)
                    cfg.update_dynamic_config(active_agent_name=None, updated_by=data.get("client_id") or "system")
            return jsonify({"success": True, "deleted": name}), 200
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except Exception as exc:
            logger.error(f"Error deleting agent spec: {exc}")
            return jsonify({"error": str(exc)}), 500

    # ── Playbooks (per-user reusable instruction packs) ──────────────────────────
    def _resolve_playbook_owner(self, request_client_id):
        """Resolve the verified owner for a playbook operation.

        When auth is enabled and the user is logged in, the server-verified session
        identity is used and any request-supplied client_id is ignored (closes IDOR).
        In anonymous / auth-disabled deployments, falls back to the request client_id.
        Returns (owner_id, None) on success, or (None, flask_error_response) on failure.
        """
        owner, err = resolve_playbook_owner(
            self.auth_enabled, session.get("logged_in"), session.get("user"), request_client_id)
        if err:
            return None, (jsonify({"error": err}), 400)
        return owner, None

    def _playbook_svc(self) -> PlaybookService:
        """PlaybookService for the REST/staging paths (pooled when possible)."""
        return _pooled_playbook_service(self.pg_config)

    def _stage_playbook_for_request(self, client_id, playbook_name) -> None:
        """Stage playbook state for a chat request via per-request ContextVars.

        Sets the verified owner (tools fail closed on None) and, for a
        /name-invoked turn, the pending (name, body, foreign) triple. The clean
        message is what gets persisted; the expansion happens only in-flight (see
        _prepare_chat_context), and the name rides along for the chip.
        """
        playbook_owner, _ = self._resolve_playbook_owner(client_id)
        set_playbook_owner(playbook_owner)   # verified identity (auth on) or client_id (anon); None fails closed
        clear_pending_playbook()             # start clean so no prior request's playbook leaks via the ContextVar
        if not playbook_name or not playbook_owner:
            return
        try:
            playbook = self._playbook_svc().resolve_invokable_playbook(
                playbook_owner, playbook_name
            )
            set_pending_playbook(
                playbook.name, playbook.body, foreign=playbook.owner_id != playbook_owner,
                playbook_id=playbook.id
            )
        except PlaybookNotFoundError:
            # Deleted between menu load and send: proceed without it (no chip is shown).
            logger.info("Requested playbook '%s' not found; sending the turn without it", playbook_name)
            # Unified ledger: record the failed /name attempt (no conversation exists
            # yet, so ids are NULL). Best-effort — never break the turn.
            try:
                self._playbook_svc().record_invocation(
                    None, None, None, playbook_name, source="explicit", status="not_found")
            except Exception as exc:
                logger.warning("Could not record not_found playbook invocation: %s", exc)
        except Exception as exc:
            logger.warning("Playbook lookup failed: %s", exc)

    def get_agent_info(self):
        """
        Get high-level information about the active agent configuration.

        Query params:
            config_name: Optional config name to describe (defaults to active config).

        Returns:
            JSON with config name, pipeline name, embedding name, and data sources.
        """
        config_name = request.args.get("config_name") or self.chat.current_config_name or self.config.get("name")

        try:
            config_payload = self.chat._get_config_payload(config_name) if config_name else self.config
        except Exception as exc:
            logger.error(f"Error loading config '{config_name}': {exc}")
            config_payload = self.config

        chat_cfg = config_payload.get("services", {}).get("chat_app", {})
        agent_class = ChatWrapper._get_agent_class_from_cfg(chat_cfg)
        embedding_name = config_payload.get("data_manager", {}).get("embedding_name")
        sources = config_payload.get("data_manager", {}).get("sources", {})
        source_names = list(sources.keys()) if isinstance(sources, dict) else []
        agent_spec = getattr(self.chat, "agent_spec", None)

        return jsonify({
            "config_name": config_name,
            "pipeline": agent_class,
            "embedding_name": embedding_name,
            "data_sources": source_names,
            "agent_name": getattr(agent_spec, "name", None),
            "agent_tools": getattr(agent_spec, "tools", None),
            "agent_prompt": getattr(agent_spec, "prompt", None),
        }), 200

    def get_provider_models(self):
        """
        Get models for a specific provider.
        
        Query params:
            provider: Provider type (openai, anthropic, gemini, openrouter, local)
        
        Returns:
            JSON with models list
        """
        provider_type = request.args.get('provider')
        if not provider_type:
            return jsonify({'error': 'provider parameter required'}), 400
        
        try:
            from src.archi.providers import get_provider

            cfg = _build_provider_config_from_payload(self.config, ProviderType(provider_type))
            provider = get_provider(provider_type, config=cfg) if cfg else get_provider(provider_type)
            models = provider.list_models()
            
            return jsonify({
                'provider': provider_type,
                'display_name': provider.display_name,
                'enabled': provider.is_enabled,
                'default_model': provider.config.default_model,
                'models': [
                    {
                        'id': m.id,
                        'name': m.name,
                        'display_name': m.display_name,
                        'context_window': m.context_window,
                        'supports_tools': m.supports_tools,
                        'supports_streaming': m.supports_streaming,
                        'supports_vision': m.supports_vision,
                        'max_output_tokens': m.max_output_tokens,
                    }
                    for m in models
                ],
            }), 200
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except ImportError:
            return jsonify({'error': 'Providers module not available'}), 500
        except Exception as e:
            logger.error(f"Error getting provider models: {e}")
            return jsonify({'error': str(e)}), 500

    def validate_provider(self):
        """
        Validate a provider connection.
        
        Request body:
            provider: Provider type (openai, anthropic, etc.)
        
        Returns:
            JSON with validation result
        """
        payload = request.get_json(silent=True) or {}
        provider_type = payload.get('provider')
        
        if not provider_type:
            return jsonify({'error': 'provider field required'}), 400
        
        try:
            from src.archi.providers import get_provider
            
            provider = get_provider(provider_type)
            is_valid = provider.validate_connection()
            
            return jsonify({
                'provider': provider_type,
                'display_name': provider.display_name,
                'valid': is_valid,
                'enabled': provider.is_enabled,
            }), 200
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except ImportError:
            return jsonify({'error': 'Providers module not available'}), 500
        except Exception as e:
            logger.error(f"Error validating provider: {e}")
            return jsonify({'error': str(e)}), 500

    def set_provider_api_key(self):
        """
        Set an API key for a specific provider.
        
        The API key is stored in the user's session, not in environment variables
        or persistent storage. This provides security (keys are not logged or stored)
        while allowing runtime configuration.
        
        Request body:
            provider: Provider type (openai, anthropic, gemini, openrouter)
            api_key: The API key to set
        
        Returns:
            JSON with success status and provider validation result
        """
        payload = request.get_json(silent=True) or {}
        provider_type = payload.get('provider')
        api_key = payload.get('api_key')
        
        if not provider_type:
            return jsonify({'error': 'provider field required'}), 400
        if not api_key:
            return jsonify({'error': 'api_key field required'}), 400
        
        # Validate the provider type
        try:
            from src.archi.providers import ProviderType
            ptype = ProviderType(provider_type.lower())
        except ValueError:
            return jsonify({'error': f'Unknown provider type: {provider_type}'}), 400
        
        # Store the API key in session
        if 'provider_api_keys' not in session:
            session['provider_api_keys'] = {}
        session['provider_api_keys'][provider_type.lower()] = api_key
        session.modified = True
        
        # Validate the API key by testing the provider
        try:
            from src.archi.providers import get_provider_with_api_key
            
            provider = get_provider_with_api_key(provider_type, api_key)
            is_valid = provider.validate_connection()
            
            return jsonify({
                'success': True,
                'provider': provider_type,
                'display_name': provider.display_name,
                'valid': is_valid,
                'message': 'API key saved to session' + (' and validated' if is_valid else ' but validation failed'),
            }), 200
        except Exception as e:
            # Still save the key even if validation fails
            logger.warning(f"API key validation failed for {provider_type}: {e}")
            return jsonify({
                'success': True,
                'provider': provider_type,
                'valid': False,
                'message': f'API key saved but validation failed: {e}',
            }), 200

    def get_provider_api_keys(self):
        """
        Get a list of which providers have API keys configured.
        
        For security, this does NOT return the actual API keys, only which
        providers have keys set and whether they are valid.
        
        Returns:
            JSON with list of configured providers
        """
        session_keys = session.get('provider_api_keys', {})
        
        try:
            from src.archi.providers import (
                list_provider_types,
                get_provider,
                get_provider_with_api_key,
                ProviderType,
            )
            
            providers_status = []
            for provider_type in list_provider_types():
                # Skip local providers - no API key needed
                if provider_type in (ProviderType.LOCAL, ProviderType.OLLAMA, ProviderType.VLLM):
                    continue
                    
                ptype_str = provider_type.value
                has_session_key = ptype_str in session_keys
                has_env_key = False
                is_valid = False
                display_name = ptype_str.title()  # fallback
                
                try:
                    # Check if there's an env-based key
                    env_provider = get_provider(provider_type)
                    has_env_key = env_provider.is_configured
                    display_name = env_provider.display_name  # use proper display name
                    
                    # If we have a session key, test that one
                    if has_session_key:
                        test_provider = get_provider_with_api_key(
                            provider_type,
                            session_keys[ptype_str]
                        )
                        is_valid = test_provider.is_configured
                    else:
                        is_valid = has_env_key
                except Exception as e:
                    logger.debug(f"Error checking provider {ptype_str}: {e}")
                
                providers_status.append({
                    'provider': ptype_str,
                    'display_name': display_name,
                    'has_session_key': has_session_key,
                    'has_env_key': has_env_key,
                    'configured': has_session_key or has_env_key,
                    'valid': is_valid,
                    'masked_key': ('*' * 8 + session_keys[ptype_str][-4:]) if has_session_key else None,
                })
            
            return jsonify({
                'providers': providers_status,
            }), 200
        except ImportError as e:
            logger.error(f"Providers module not available: {e}")
            return jsonify({'error': 'Providers module not available'}), 500
        except Exception as e:
            logger.error(f"Error getting provider API keys status: {e}")
            return jsonify({'error': str(e)}), 500

    def clear_provider_api_key(self):
        """
        Clear the API key for a specific provider from the session.
        
        Request body:
            provider: Provider type to clear
        
        Returns:
            JSON with success status
        """
        payload = request.get_json(silent=True) or {}
        provider_type = payload.get('provider')
        
        if not provider_type:
            return jsonify({'error': 'provider field required'}), 400
        
        ptype_str = provider_type.lower()
        
        if 'provider_api_keys' in session:
            if ptype_str in session['provider_api_keys']:
                del session['provider_api_keys'][ptype_str]
                session.modified = True
                return jsonify({
                    'success': True,
                    'message': f'API key for {provider_type} cleared from session',
                }), 200
        
        return jsonify({
            'success': True,
            'message': f'No API key found for {provider_type}',
        }), 200

    def validate_provider_api_key(self):
        """
        Validate an API key for a provider without storing it.
        
        This endpoint allows testing a key before committing to save it.
        The key is NOT stored in the session.
        
        Request body:
            provider: Provider type (openai, anthropic, gemini, openrouter)
            api_key: The API key to validate
        
        Returns:
            JSON with validation result and available models
        """
        payload = request.get_json(silent=True) or {}
        provider_type = payload.get('provider')
        api_key = payload.get('api_key')
        
        if not provider_type:
            return jsonify({'error': 'provider field required'}), 400
        if not api_key:
            return jsonify({'error': 'api_key field required'}), 400
        
        # Validate the provider type
        try:
            from src.archi.providers import ProviderType, get_provider_with_api_key
            ptype = ProviderType(provider_type.lower())
        except ValueError:
            return jsonify({'error': f'Unknown provider type: {provider_type}'}), 400
        
        try:
            # Create provider with the test key (not cached, not stored)
            provider = get_provider_with_api_key(provider_type, api_key)
            is_valid = provider.validate_connection()
            
            # If valid, also get available models
            models = []
            if is_valid:
                try:
                    models = [m.to_dict() for m in provider.list_models()]
                except Exception:
                    pass  # Models list is optional
            
            return jsonify({
                'valid': is_valid,
                'provider': provider_type,
                'display_name': provider.display_name,
                'models_available': models,
            }), 200
        except Exception as e:
            logger.warning(f"API key validation failed for {provider_type}: {e}")
            return jsonify({
                'valid': False,
                'provider': provider_type,
                'error': str(e),
            }), 200

    def _get_request_client_id(self) -> str:
        """Extract client_id from the current request (JSON body or query params)."""
        if request.is_json and request.json:
            cid = request.json.get('client_id')
            if cid:
                return cid
        return request.args.get('client_id', '')

    def _is_admin_request(self) -> bool:
        """Return True when the current request is from an RBAC admin user."""
        try:
            return bool(rbac_is_admin())
        except Exception:
            return False

    def _can_view_ab_testing(self) -> bool:
        return (
            self._can_manage_ab_testing()
            or has_permission(Permission.AB.VIEW)
            or has_permission(Permission.AB.METRICS)
        )

    def _can_view_evaluations(self) -> bool:
        if not self.evaluations_enabled:
            return False
        return not self.auth_enabled or has_permission(Permission.Evaluations.VIEW)

    def _can_manage_ab_testing(self) -> bool:
        return self._is_admin_request() or has_permission(Permission.AB.MANAGE)

    def _can_view_ab_metrics(self) -> bool:
        return self._can_manage_ab_testing() or has_permission(Permission.AB.METRICS)

    def _current_request_roles(self) -> List[str]:
        return list(session.get('roles', []) or [])

    def _current_request_permissions(self) -> List[str]:
        return sorted(get_user_permissions(self._current_request_roles()))

    @staticmethod
    def _current_user_id() -> Optional[str]:
        user = session.get('user') or {}
        return user.get('id') or session.get('client_id') or None

    def _get_effective_ab_sample_rate(self, default_rate: float) -> float:
        effective_rate = float(default_rate)
        user_id = self._current_user_id()
        if not user_id:
            return effective_rate
        session_user = session.get('user') or {}
        try:
            user = self.chat.user_service.get_or_create_user(
                user_id=user_id,
                auth_provider=session_user.get('auth_method', 'anonymous') if session.get('logged_in') else 'anonymous',
                display_name=session_user.get('name'),
                email=session_user.get('email'),
            )
        except Exception:
            user = None
        if user is not None and user.ab_participation_rate is not None:
            effective_rate = float(user.ab_participation_rate)
        return min(max(effective_rate, 0.0), 1.0)

    def _get_ab_participation_state(self) -> Dict[str, Any]:
        pool = getattr(self.chat, "ab_pool", None)
        can_participate = has_permission(Permission.AB.PARTICIPATE)
        if not can_participate:
            return {
                "can_participate": False,
                "eligible": False,
                "reason": "not_participant",
                "targeted": False,
            }
        if not pool or not pool.enabled:
            return {
                "can_participate": True,
                "eligible": False,
                "reason": "disabled",
                "targeted": False,
            }
        targeted = pool.is_targeted_user(
            roles=self._current_request_roles(),
            permissions=self._current_request_permissions(),
        )
        if not targeted:
            return {
                "can_participate": True,
                "eligible": False,
                "reason": "not_targeted",
                "targeted": False,
            }
        return {
            "can_participate": True,
            "eligible": True,
            "reason": "eligible",
            "targeted": True,
        }

    def _can_use_ab_testing(self) -> bool:
        return bool(self._get_ab_participation_state()["eligible"])

    def _refresh_runtime_config(self) -> None:
        static = self.config_service.get_static_config(force_reload=True)
        if static is None:
            raise ValueError("Static config not initialized")
        self.config = _static_config_to_full_config(static, config_service=self.config_service)
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]
        self.chat_app_config = self.services_config["chat_app"]
        self.chat.reload_static_state()

    def _serialize_pending_ab_comparison(
        self,
        comparison,
    ) -> Optional[Dict[str, Any]]:
        if comparison is None:
            return None

        mids = [comparison.response_a_mid, comparison.response_b_mid]
        if not all(mids):
            return None

        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT message_id, sender, content, model_used
                FROM conversations
                WHERE message_id = ANY(%s)
                ORDER BY message_id ASC
                """,
                (mids,),
            )
            messages = {
                row[0]: {
                    "message_id": row[0],
                    "sender": row[1],
                    "content": row[2],
                    "model_used": row[3],
                    "trace": self.chat.get_trace_by_message(row[0]),
                }
                for row in cursor.fetchall()
            }
        finally:
            cursor.close()
            conn.close()

        response_a = messages.get(comparison.response_a_mid)
        response_b = messages.get(comparison.response_b_mid)
        if not response_a or not response_b:
            return None

        return {
            "comparison_id": comparison.comparison_id,
            "conversation_id": comparison.conversation_id,
            "created_at": comparison.created_at.isoformat() if comparison.created_at else None,
            "response_a": response_a,
            "response_b": response_b,
            "variant_a_name": comparison.variant_a_name,
            "variant_b_name": comparison.variant_b_name,
            "preference": comparison.preference,
            "variant_label_mode": self.chat.ab_pool.variant_label_mode if self.chat.ab_pool else DEFAULT_DISCLOSURE_MODE,
            "activity_panel_default_state": self.chat.ab_pool.activity_panel_default_state if self.chat.ab_pool else DEFAULT_TRACE_MODE,
        }

    def _serialize_pending_ab_comparisons(
        self,
        comparisons,
    ) -> List[Dict[str, Any]]:
        """Serialize unresolved comparisons in stable creation order."""
        serialized: List[Dict[str, Any]] = []
        for comparison in comparisons or []:
            payload = self._serialize_pending_ab_comparison(comparison)
            if payload is not None:
                serialized.append(payload)
        return serialized

    def _parse_chat_request(self) -> Dict[str, Any]:
        payload = request.get_json(silent=True) or {}

        client_sent_msg_ts = payload.get("client_sent_msg_ts")
        client_timeout = payload.get("client_timeout")
        client_sent_msg_ts = client_sent_msg_ts / 1000 if client_sent_msg_ts else 0
        client_timeout = client_timeout / 1000 if client_timeout else 0

        include_agent_steps = payload.get("include_agent_steps", True)
        include_tool_steps = payload.get("include_tool_steps", True)
        if isinstance(include_agent_steps, str):
            include_agent_steps = include_agent_steps.lower() == "true"
        if isinstance(include_tool_steps, str):
            include_tool_steps = include_tool_steps.lower() == "true"

        return {
            "message": payload.get("last_message"),
            "conversation_id": payload.get("conversation_id"),
            "config_name": payload.get("config_name"),
            "is_refresh": payload.get("is_refresh"),
            "client_sent_msg_ts": client_sent_msg_ts,
            "client_timeout": client_timeout,
            "client_id": payload.get("client_id"),
            "include_agent_steps": include_agent_steps,
            "include_tool_steps": include_tool_steps,
            # Provider-based model selection
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "pipeline": payload.get("pipeline"),
            # Name of a user-invoked playbook to apply to this turn (None if none).
            "playbook_name": payload.get("playbook_name"),
        }


    def get_chat_response(self):
        """
        Gets a response when prompted. Asks as an API to the main app, who's
        functionality is carried through by javascript and html. Input is a
        requestion with

            conversation_id: Either None or an integer
            last_message:    list of length 2, where the first element is "User"
                             and the second element contains their message.

        Returns:
            A json with a response (html formatted plain text string) and a
            discussion ID (either None or an integer)
        """
        # compute timestamp at which message was received by server
        start_time = time.time()
        server_received_msg_ts = datetime.now(timezone.utc)

        # get user input and conversation_id from the request
        request_data = self._parse_chat_request()
        message = request_data["message"]
        conversation_id = request_data["conversation_id"]
        config_name = request_data["config_name"]
        is_refresh = request_data["is_refresh"]
        client_sent_msg_ts = request_data["client_sent_msg_ts"]
        client_timeout = request_data["client_timeout"]
        client_id = request_data["client_id"]
        model = request_data["model"]

        if not client_id:
            return jsonify({'error': 'client_id missing'}), 400

        user_id = session.get('user', {}).get('id') or None

        # Stage playbook state (verified owner + pending /name invocation) on this
        # request context before the chat pipeline reads it in _prepare_chat_context.
        self._stage_playbook_for_request(client_id, request_data["playbook_name"])

        # query the chat and return the results.
        logger.debug("Calling the ChatWrapper()")
        response, conversation_id, message_ids, timestamps, error_code = self.chat(message, conversation_id, client_id, is_refresh, server_received_msg_ts, client_sent_msg_ts, client_timeout,config_name, user_id=user_id)

        # handle errors
        if error_code is not None:
            err = ChatWrapper._error_event(error_code)
            return jsonify({'error': err['message']}), error_code

        # compute timestamp at which message was returned to client
        timestamps['server_response_msg_ts'] = datetime.now(timezone.utc)

        # store timing info for this message
        timestamps['server_received_msg_ts'] = server_received_msg_ts
        timestamps['client_sent_msg_ts'] = datetime.fromtimestamp(client_sent_msg_ts, tz=timezone.utc)
        self.chat.insert_timing(message_ids[-1], timestamps)

        # otherwise return archi's response to client
        try:
            response_size = len(response) if isinstance(response, str) else 0
            logger.info(f"Generated Response Length: {response_size} characters")
            json.dumps({'response': response})  # Validate JSON formatting
        except Exception as e:
            logger.error(f"JSON Encoding Error: {e}")
            response = "Error processing response"

        response_data = {
            'response': response,
            'conversation_id': conversation_id,
            'archi_msg_id': message_ids[-1],
            'server_response_msg_ts': timestamps['server_response_msg_ts'].timestamp(),
            'model_used': model,
            'final_response_msg_ts': datetime.now(timezone.utc).timestamp(),
        }

        end_time = time.time()
        logger.info(f"API Response Time: {end_time - start_time:.2f} seconds")

        return jsonify(response_data)

    def get_chat_response_stream(self):
        """
        Streams agent updates and the final response as NDJSON.
        """
        server_received_msg_ts = datetime.now(timezone.utc)
        request_data = self._parse_chat_request()

        message = request_data["message"]
        conversation_id = request_data["conversation_id"]
        config_name = request_data["config_name"]
        is_refresh = request_data["is_refresh"]
        client_sent_msg_ts = request_data["client_sent_msg_ts"]
        client_timeout = request_data["client_timeout"]
        client_id = request_data["client_id"]
        include_agent_steps = request_data["include_agent_steps"]
        include_tool_steps = request_data["include_tool_steps"]
        provider = request_data["provider"]
        model = request_data["model"]

        if not client_id:
            return jsonify({"error": "client_id missing"}), 400

        user_id = session.get('user', {}).get('id') or None

        # Stage playbook state (verified owner + pending /name invocation) on this
        # request context before the chat pipeline reads it in _prepare_chat_context.
        # stream_with_context keeps these ContextVars live through generator iteration.
        self._stage_playbook_for_request(client_id, request_data["playbook_name"])

        # Get API key from session if available
        session_api_key = None
        if provider and 'provider_api_keys' in session:
            session_api_key = session.get('provider_api_keys', {}).get(provider.lower())

        def _event_stream() -> Iterator[str]:
            padding = " " * 2048
            yield json.dumps({"type": "meta", "event": "stream_started", "padding": padding}) + "\n"
            # A /name-invoked playbook has its body injected server-side (no Playbook tool
            # call to surface), so emit the applied-playbook activity step up front here.
            _pending = get_pending_playbook()
            if _pending and _pending.get("name"):
                yield json.dumps({"type": "playbook_applied", "name": _pending["name"],
                                  "body": _pending.get("body", "")}) + "\n"
            for event in self.chat.stream(
                message,
                conversation_id,
                client_id,
                is_refresh,
                server_received_msg_ts,
                client_sent_msg_ts,
                client_timeout,
                config_name,
                include_agent_steps=include_agent_steps,
                include_tool_steps=include_tool_steps,
                provider=provider,
                model=model,
                provider_api_key=session_api_key,
                user_id=user_id,
            ):
                yield json.dumps(event, default=str) + "\n"

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
            "Content-Type": "application/x-ndjson",
        }
        return Response(stream_with_context(_event_stream()), headers=headers)

    def landing(self):
        """Landing page for unauthenticated users"""
        # If user is already logged in, redirect to chat
        if session.get('logged_in'):
            return redirect(url_for('index'))
        
        # Render landing page with auth method information
        return render_template('landing.html',
                             sso_enabled=self.sso_enabled,
                             basic_auth_enabled=self.basic_auth_enabled)

    def index(self):
        return render_template(
            'index.html',
            can_view_evaluations=self._can_view_evaluations(),
        )

    def terms(self):
        return render_template('terms.html')

    def _with_feedback_lock(self, fn):
        """Run fn() under the feedback lock with proper cleanup."""
        self.chat.lock.acquire()
        logger.info("Acquired lock file")
        try:
            return fn()
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            self.chat.lock.release()
            logger.info("Released lock file")
            if self.chat.cursor is not None:
                self.chat.cursor.close()
            if self.chat.conn is not None:
                self.chat.conn.close()

    def _toggle_reaction(self, reaction_type):
        """Shared like/dislike toggle: remove if already set, else insert."""
        def _do():
            data = request.json
            message_id = data.get('message_id')
            if not message_id:
                logger.warning(f"{reaction_type.capitalize()} request missing message_id")
                return jsonify({'error': 'message_id is required'}), 400

            current_reaction = self.chat.get_reaction_feedback(message_id)
            self.chat.delete_reaction_feedback(message_id)

            if current_reaction == reaction_type:
                return jsonify({'message': 'Reaction removed', 'state': None}), 200

            feedback = {
                "message_id"   : message_id,
                "feedback"     : reaction_type,
                "feedback_ts"  : datetime.now(),
                "feedback_msg" : data.get('feedback_msg') if reaction_type == 'dislike' else None,
                "incorrect"    : data.get('incorrect') if reaction_type == 'dislike' else None,
                "unhelpful"    : data.get('unhelpful') if reaction_type == 'dislike' else None,
                "inappropriate": data.get('inappropriate') if reaction_type == 'dislike' else None,
            }
            self.chat.insert_feedback(feedback)

            label = f"{reaction_type.capitalize()}d"
            return jsonify({'message': label, 'state': reaction_type}), 200

        return self._with_feedback_lock(_do)

    def like(self):
        return self._toggle_reaction('like')

    def dislike(self):
        return self._toggle_reaction('dislike')

    def text_feedback(self):
        def _do():
            data = request.json
            message_id = data.get('message_id')
            feedback_msg = (data.get('feedback_msg') or '').strip()

            if message_id is None:
                return jsonify({'error': 'message_id missing'}), 400
            if not feedback_msg:
                return jsonify({'error': 'feedback_msg missing'}), 400
            try:
                message_id = int(message_id)
            except (TypeError, ValueError):
                return jsonify({'error': 'message_id must be an integer'}), 400

            feedback = {
                "message_id"   : message_id,
                "feedback"     : "comment",
                "feedback_ts"  : datetime.now(timezone.utc),
                "feedback_msg" : feedback_msg,
                "incorrect"    : None,
                "unhelpful"    : None,
                "inappropriate": None,
            }
            self.chat.insert_feedback(feedback)
            return jsonify({'message': 'Feedback submitted'}), 200

        return self._with_feedback_lock(_do)

    def list_conversations(self):
        """
        List all conversations, ordered by most recent first.

        Query parameters:
        - limit (optional): Number of conversations to return (default: 50, max: 500)

        Returns:
            JSON with list of conversations with fields: (conversation_id, title, created_at, last_message_at).
        """
        try:
            client_id = request.args.get('client_id')
            user_id = session.get('user', {}).get('id') or None
            if not user_id and not client_id:
                return jsonify({'error': 'client_id missing'}), 400
            limit = min(int(request.args.get('limit', 50)), 500)

            # create connection to database
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            if user_id:
                cursor.execute(SQL_LIST_CONVERSATIONS_BY_USER, (user_id, client_id, limit))
            else:
                cursor.execute(SQL_LIST_CONVERSATIONS, (client_id, limit))
            rows = cursor.fetchall()

            conversations = []
            for row in rows:
                conversations.append({
                    'conversation_id': row[0],
                    'title': row[1] or "New Chat",
                    'created_at': row[2].isoformat() if row[2] else None,
                    'last_message_at': row[3].isoformat() if row[3] else None,
                })

            # clean up database connection state
            cursor.close()
            conn.close()

            return jsonify({'conversations': conversations}), 200

        except ValueError as e:
            return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
        except Exception as e:
            print(f"ERROR in list_conversations: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def load_conversation(self):
        """
        Load a specific conversation's full history.

        POST body:
        - conversation_id: The ID of the conversation to load

        Returns:
            JSON with conversation metadata and full message history
        """
        conn = None
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            client_id = data.get('client_id')
            user_id = session.get('user', {}).get('id') or None

            if not conversation_id:
                return jsonify({'error': 'conversation_id missing'}), 400
            if not user_id and not client_id:
                return jsonify({'error': 'client_id missing'}), 400

            # create connection to database
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()

            # get conversation metadata
            if user_id:
                cursor.execute(SQL_GET_CONVERSATION_METADATA_BY_USER, (conversation_id, user_id, client_id))
            else:
                cursor.execute(SQL_GET_CONVERSATION_METADATA, (conversation_id, client_id))
            meta_row = cursor.fetchone()

            # if no metadata found, return error
            if not meta_row:
                cursor.close()
                conn.close()
                return jsonify({'error': 'conversation not found'}), 404

            # get history of the conversation along with latest feedback state
            # (degrades to chip-less rows if the side-table migration failed)
            history_rows = _query_convo_history_rows(cursor, conversation_id)
            comparisons = self.chat.conv_service.get_conversation_ab_comparisons(str(conversation_id))
            suppressed_ids = self.chat._suppressed_ab_message_ids(comparisons)
            if suppressed_ids:
                history_rows = [row for row in history_rows if row[2] not in suppressed_ids]
            history_rows = collapse_assistant_sequences(history_rows, sender_name=ARCHI_SENDER, sender_index=0)

            # Build messages list with trace data for assistant messages
            messages = []
            
            # Batch-fetch trace data for all assistant messages to avoid N+1 queries
            assistant_mids = [row[2] for row in history_rows if row[0] == ARCHI_SENDER and row[2]]
            trace_map = {}
            if assistant_mids:
                placeholders = ','.join(['%s'] * len(assistant_mids))
                cursor.execute(f"""
                    SELECT trace_id, conversation_id, message_id, user_message_id,
                           config_id, pipeline_name, events, started_at, completed_at,
                           status, total_tool_calls, total_tokens_used, total_duration_ms,
                           cancelled_by, cancellation_reason, created_at
                    FROM agent_traces
                    WHERE message_id IN ({placeholders})
                """, tuple(assistant_mids))
                for trace_row in cursor.fetchall():
                    trace_map[trace_row[2]] = trace_row
            
            for row in history_rows:
                msg = {
                    'sender': row[0],
                    'content': row[1],
                    'message_id': row[2],
                    'feedback': row[3],
                    'comment_count': row[4] if len(row) > 4 else 0,
                    'model_used': row[5] if len(row) > 5 else None,
                    'playbook_name': row[6] if len(row) > 6 else None,
                }
                
                # Attach trace data if present
                if row[0] == ARCHI_SENDER and row[2] and row[2] in trace_map:
                    trace_row = trace_map[row[2]]
                    msg['trace'] = {
                        'trace_id': trace_row[0],
                        'events': trace_row[6],  # events JSON
                        'status': trace_row[9],
                        'total_tool_calls': trace_row[10],
                        'total_duration_ms': trace_row[12],
                    }
                
                messages.append(msg)

            pending_comparisons = [c for c in comparisons if c.preference is None]
            serialized_pending = self._serialize_pending_ab_comparisons(pending_comparisons)

            conversation = {
                'conversation_id': meta_row[0],
                'title': meta_row[1] or "New Conversation",
                'created_at': meta_row[2].isoformat() if meta_row[2] else None,
                'last_message_at': meta_row[3].isoformat() if meta_row[3] else None,
                'messages': messages,
                'pending_ab_comparisons': serialized_pending,
                'pending_ab_comparison': serialized_pending[-1] if serialized_pending else None,
            }

            # clean up database connection state
            cursor.close()
            conn.close()

            return jsonify(conversation), 200

        except Exception as e:
            logger.error(f"Error in load_conversation: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # never leak the connection, whichever path returned
            if conn is not None and not conn.closed:
                conn.close()

    def new_conversation(self):
        """
        Start a new conversation without sending a message yet.
        This simply returns null(Conversation ID == None) to indicate that the frontend should
        reset its conversation_id, and a new one will be created on first message.

        Returns:
            JSON with conversation_id == None
        """
        try:
            # return null to indicate a new conversation
            # actual conversation will be created when the first message is sent
            return jsonify({'conversation_id': None}), 200

        except Exception as e:
            logger.error(f"Error in new_conversation: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def delete_conversation(self):
        """
        Delete a conversation and all its messages. (Using SQL CASCADE)

        POST body:
        - conversation_id: The ID of the conversation to delete

        Returns:
            JSON with success status
        """
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            client_id = data.get('client_id')
            user_id = session.get('user', {}).get('id') or None

            if not conversation_id:
                return jsonify({'error': 'conversation_id missing when deleting.'}), 400
            if not user_id and not client_id:
                return jsonify({'error': 'client_id missing when deleting.'}), 400

            # create connection to database
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()

            # Delete conversation metadata (SQL CASCADE will delete all child messages)
            if user_id:
                cursor.execute(SQL_DELETE_CONVERSATION_BY_USER, (conversation_id, user_id, client_id))
            else:
                cursor.execute(SQL_DELETE_CONVERSATION, (conversation_id, client_id))
            deleted_count = cursor.rowcount
            conn.commit()

            # clean up database connection state
            cursor.close()
            conn.close()

            if deleted_count == 0:
                return jsonify({'error': 'Conversation not found'}), 404

            logger.info(f"Deleted conversation {conversation_id}")
            return jsonify({'success': True, 'deleted_conversation_id': conversation_id}), 200

        except ValueError as e:
            return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
        except Exception as e:
            print(f"ERROR in delete_conversation: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # A/B Testing API Endpoints
    # =========================================================================

    def ab_submit_preference(self):
        """
        Submit user's preference for an A/B comparison.

        POST body:
        - comparison_id: The comparison ID
        - preference: 'a', 'b', or 'tie'
        - client_id: Client ID for authorization

        Returns:
            JSON with success status
        """
        try:
            data = request.json
            comparison_id = data.get('comparison_id')
            preference = data.get('preference')
            client_id = data.get('client_id')
            user_id = session.get('user', {}).get('id') or None

            if not comparison_id:
                return jsonify({'error': 'comparison_id is required'}), 400
            if not preference:
                return jsonify({'error': 'preference is required'}), 400
            if preference not in ('a', 'b', 'tie'):
                return jsonify({'error': 'preference must be "a", "b", or "tie"'}), 400
            if not client_id:
                return jsonify({'error': 'client_id is required'}), 400

            # Verify the comparison belongs to the requesting client
            comparison = self.chat.conv_service.get_ab_comparison(comparison_id)
            if not comparison:
                return jsonify({'error': 'Comparison not found'}), 404
            comp_conv_id = comparison.conversation_id if comparison else None
            if comp_conv_id:
                try:
                    self.chat.query_conversation_history(comp_conv_id, client_id, user_id)
                except ConversationAccessError:
                    return jsonify({'error': 'Not authorized for this comparison'}), 403

            result = self.chat.conv_service.submit_ab_preference(comparison_id, preference)

            return jsonify({
                'success': True,
                'comparison_id': comparison_id,
                'preference': preference,
                'updated': result.get('updated', False),
                'canonical_message_id': self.chat._comparison_canonical_message_id(result.get('comparison')),
            }), 200

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error submitting A/B preference: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def ab_get_pending(self):
        """
        Get the pending (unvoted) A/B comparison for a conversation.

        Query params:
        - conversation_id: The conversation ID
        - client_id: Client ID for authorization

        Returns:
            JSON with comparison data or null if none pending
        """
        try:
            conversation_id = request.args.get('conversation_id', type=int)
            client_id = request.args.get('client_id')
            user_id = session.get('user', {}).get('id') or None

            if not conversation_id:
                return jsonify({'error': 'conversation_id is required'}), 400
            if not client_id:
                return jsonify({'error': 'client_id is required'}), 400

            try:
                self.chat.query_conversation_history(conversation_id, client_id, user_id)
            except ConversationAccessError:
                return jsonify({'error': 'Not authorized for this conversation'}), 403

            comparisons = self.chat.conv_service.get_pending_ab_comparisons(conversation_id)
            serialized = self._serialize_pending_ab_comparisons(comparisons)
            return jsonify({
                'success': True,
                'comparison': serialized[-1] if serialized else None,
                'comparisons': serialized,
                'pending_count': len(serialized),
            }), 200

        except Exception as e:
            logger.error(f"Error getting pending A/B comparison: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def ab_get_pool(self):
        """
        Get A/B testing pool configuration.

        Returns:
            JSON with pool info (enabled, champion, variant names) or enabled=false.
            Only admins see the full pool info; non-admins get enabled=false.
        """
        try:
            pool = self.chat.ab_pool
            can_view = self._can_view_ab_testing()
            can_manage = self._can_manage_ab_testing()
            participation = self._get_ab_participation_state()
            can_use = participation["eligible"]
            can_participate = participation["can_participate"]
            raw_ab_cfg = ((self.services_config.get("chat_app", {}) or {}).get("ab_testing") or {})
            default_comparison_rate = float(
                self._get_ab_setting(
                    raw_ab_cfg,
                    "comparison_rate",
                    "sample_rate",
                    getattr(pool, "comparison_rate", getattr(pool, "sample_rate", 1.0) or 1.0),
                )
            )
            if can_view:
                return jsonify(self._build_admin_ab_pool_payload()), 200
            if pool and can_use:
                effective_rate = self._get_effective_ab_sample_rate(pool.sample_rate)
                return jsonify({
                    'success': True,
                    'is_admin': self._is_admin_request(),
                    'can_view': False,
                    'can_manage': False,
                    'can_view_metrics': False,
                    'can_participate': can_participate,
                    'participant_eligible': True,
                    'participant_reason': participation["reason"],
                    'participant_targeted': True,
                    **pool.participant_info(),
                    'comparison_rate': effective_rate,
                    'default_comparison_rate': default_comparison_rate,
                }), 200
            return jsonify({
                'success': True,
                'enabled': False,
                'enabled_requested': bool(raw_ab_cfg.get('enabled', False)),
                'is_admin': self._is_admin_request(),
                'can_view': can_view,
                'can_manage': can_manage,
                'can_view_metrics': self._can_view_ab_metrics(),
                'can_participate': can_participate,
                'participant_eligible': participation["eligible"],
                'participant_reason': participation["reason"],
                'participant_targeted': participation["targeted"],
                'comparison_rate': self._get_effective_ab_sample_rate(default_comparison_rate) if can_participate else default_comparison_rate,
                'default_comparison_rate': default_comparison_rate,
            }), 200
        except Exception as e:
            logger.error(f"Error getting A/B pool: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def ab_get_decision(self):
        """
        Decide on the server whether the next turn should use A/B comparison.

        This keeps sampling authoritative on the backend instead of relying on
        browser-side Math.random().
        """
        try:
            client_id = request.args.get('client_id', '')
            conversation_id = request.args.get('conversation_id', type=int)
            user_id = session.get('user', {}).get('id') or None
            pool = self.chat.ab_pool
            participation = self._get_ab_participation_state()

            if not pool or not pool.enabled:
                return jsonify({'success': True, 'enabled': False, 'use_ab': False, 'reason': 'disabled'}), 200
            if not participation["can_participate"]:
                return jsonify({'success': True, 'enabled': True, 'use_ab': False, 'reason': 'not_participant'}), 200
            if not participation["eligible"]:
                return jsonify({'success': True, 'enabled': True, 'use_ab': False, 'reason': participation["reason"]}), 200

            if conversation_id:
                try:
                    self.chat.query_conversation_history(conversation_id, client_id, user_id)
                except ConversationAccessError:
                    return jsonify({'error': 'Not authorized for this conversation'}), 403

                pending_count = self.chat.conv_service.count_pending_ab_comparisons(conversation_id)
                if pending_count >= int(pool.max_pending_per_conversation):
                    pending = self.chat.conv_service.get_pending_ab_comparison(conversation_id)
                    return jsonify({
                        'success': True,
                        'enabled': True,
                        'use_ab': False,
                        'reason': 'pending_vote',
                        'comparison_id': getattr(pending, 'comparison_id', None),
                        'pending_count': pending_count,
                        'max_pending_comparisons_per_conversation': pool.max_pending_comparisons_per_conversation,
                    }), 200

            sample_rate = self._get_effective_ab_sample_rate(pool.sample_rate)
            if sample_rate <= 0:
                use_ab = False
                roll = None
            elif sample_rate >= 1:
                use_ab = True
                roll = None
            else:
                roll = random.random()
                use_ab = roll < sample_rate

            logger.info(
                "A/B decision: use_ab=%s comparison_rate=%.3f roll=%s conversation_id=%s client_id=%s",
                use_ab,
                sample_rate,
                "forced" if roll is None else f"{roll:.5f}",
                conversation_id,
                client_id,
            )

            return jsonify({
                'success': True,
                'enabled': True,
                'use_ab': use_ab,
                'reason': 'sampled' if use_ab else 'not_sampled',
                'comparison_rate': sample_rate,
                'default_comparison_rate': float(pool.comparison_rate),
                'variant_label_mode': pool.variant_label_mode,
                'activity_panel_default_state': pool.activity_panel_default_state,
                'max_pending_comparisons_per_conversation': pool.max_pending_comparisons_per_conversation,
            }), 200
        except Exception as e:
            logger.error(f"Error deciding A/B sampling: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def ab_set_pool(self):
        """
        Set the A/B testing pool from the UI.
        Admin only. Accepts JSON with champion label plus at least two variants.
        """
        if not self._can_manage_ab_testing():
            return jsonify({'error': 'Admin access required'}), 403
        try:
            data = request.get_json(force=True)
            champion_name = str(data.get('champion') or data.get('control') or '').strip()
            comparison_rate = float(data.get('comparison_rate', data.get('sample_rate', 1.0)))
            variant_label_mode = normalize_ab_disclosure_mode(
                data.get('variant_label_mode') or data.get('disclosure_mode') or DEFAULT_DISCLOSURE_MODE
            )
            activity_panel_default_state = normalize_ab_trace_mode(
                data.get('activity_panel_default_state') or data.get('default_trace_mode') or DEFAULT_TRACE_MODE
            )
            max_pending = int(
                data.get('max_pending_comparisons_per_conversation', data.get('max_pending_per_conversation', 1))
            )
            if not champion_name:
                return jsonify({'error': 'champion is required'}), 400
            variant_items = data.get('variants') or []
            existing_variants = {
                variant.label: variant for variant in (self.chat.ab_pool.variants if self.chat.ab_pool else [])
            }
            variants, parsed_labels = self._resolve_ab_variants(
                variant_items,
                existing_variants=existing_variants,
            )
            if champion_name not in parsed_labels:
                return jsonify({'error': 'Champion must be one of the variants'}), 400

            pool = ABPool(
                variants=variants,
                champion_name=champion_name,
                enabled=True,
                sample_rate=comparison_rate,
                disclosure_mode=variant_label_mode,
                default_trace_mode=activity_panel_default_state,
                max_pending_per_conversation=max_pending,
            )
            self._persist_ab_pool_config(
                enabled=True,
                champion_name=champion_name,
                variants=variants,
                comparison_rate=pool.comparison_rate,
                variant_label_mode=pool.variant_label_mode,
                activity_panel_default_state=pool.activity_panel_default_state,
                max_pending_comparisons_per_conversation=pool.max_pending_comparisons_per_conversation,
            )
            logger.info("Persisted A/B pool update: champion='%s', variants=%s", champion_name, parsed_labels)
            return jsonify(self._build_admin_ab_pool_payload()), 200
        except ABPoolError as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception as exc:
            logger.error("Error setting A/B pool: %s", exc)
            return jsonify({'error': str(exc)}), 500

    def ab_set_settings(self):
        """Persist only the experiment-settings section of the A/B admin page."""
        if not self._can_manage_ab_testing():
            return jsonify({'error': 'Admin access required'}), 403
        try:
            data = request.get_json(force=True)
            chat_cfg = self.services_config.get("chat_app", {}) or {}
            raw_ab_cfg = (chat_cfg.get("ab_testing") or {}) if isinstance(chat_cfg.get("ab_testing"), dict) else {}
            raw_pool = raw_ab_cfg.get("pool") or {}
            champion_name = str(
                data.get('champion') or data.get('control') or self._get_ab_pool_champion(raw_pool) or ''
            ).strip()
            if not champion_name:
                return jsonify({'error': 'champion is required'}), 400

            existing_variants = {
                variant.label: variant for variant in (self.chat.ab_pool.variants if self.chat.ab_pool else [])
            }
            variant_items = self._normalize_ab_variant_details(raw_pool.get("variants"))
            variants, parsed_labels = self._resolve_ab_variants(
                variant_items,
                existing_variants=existing_variants,
            )
            if champion_name not in parsed_labels:
                return jsonify({'error': 'Champion must match one of the saved variants'}), 400

            pool = ABPool(
                variants=variants,
                champion_name=champion_name,
                enabled=True,
                sample_rate=float(
                    data.get(
                        'comparison_rate',
                        data.get(
                            'sample_rate',
                            self._get_ab_setting(raw_ab_cfg, 'comparison_rate', 'sample_rate', 1.0),
                        ),
                    )
                ),
                disclosure_mode=normalize_ab_disclosure_mode(
                    data.get('variant_label_mode')
                    or data.get('disclosure_mode')
                    or self._get_ab_setting(raw_ab_cfg, 'variant_label_mode', 'disclosure_mode', DEFAULT_DISCLOSURE_MODE)
                ),
                default_trace_mode=normalize_ab_trace_mode(
                    data.get('activity_panel_default_state')
                    or data.get('default_trace_mode')
                    or self._get_ab_setting(
                        raw_ab_cfg, 'activity_panel_default_state', 'default_trace_mode', DEFAULT_TRACE_MODE
                    )
                ),
                max_pending_per_conversation=int(
                    data.get(
                        'max_pending_comparisons_per_conversation',
                        data.get(
                            'max_pending_per_conversation',
                            self._get_ab_setting(
                                raw_ab_cfg,
                                'max_pending_comparisons_per_conversation',
                                'max_pending_per_conversation',
                                1,
                            ),
                        ),
                    )
                ),
            )
            self._persist_ab_pool_config(
                enabled=True,
                champion_name=champion_name,
                variants=variants,
                comparison_rate=pool.comparison_rate,
                variant_label_mode=pool.variant_label_mode,
                activity_panel_default_state=pool.activity_panel_default_state,
                max_pending_comparisons_per_conversation=pool.max_pending_comparisons_per_conversation,
            )
            logger.info("Persisted A/B settings update: champion='%s'", champion_name)
            return jsonify(self._build_admin_ab_pool_payload()), 200
        except ABPoolError as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception as exc:
            logger.error("Error setting A/B experiment settings: %s", exc)
            return jsonify({'error': str(exc)}), 500

    def ab_set_variants(self):
        """Persist only the variant-list section of the A/B admin page."""
        if not self._can_manage_ab_testing():
            return jsonify({'error': 'Admin access required'}), 403
        try:
            data = request.get_json(force=True)
            variant_items = data.get('variants') or []
            chat_cfg = self.services_config.get("chat_app", {}) or {}
            raw_ab_cfg = (chat_cfg.get("ab_testing") or {}) if isinstance(chat_cfg.get("ab_testing"), dict) else {}
            raw_pool = raw_ab_cfg.get("pool") or {}
            existing_variants = {
                variant.label: variant for variant in (self.chat.ab_pool.variants if self.chat.ab_pool else [])
            }
            variants, parsed_labels = self._resolve_ab_variants(
                variant_items,
                existing_variants=existing_variants,
            )

            champion_name = self._get_ab_pool_champion(raw_pool)
            if champion_name not in parsed_labels:
                champion_name = parsed_labels[0]

            comparison_rate = float(self._get_ab_setting(raw_ab_cfg, 'comparison_rate', 'sample_rate', 1.0))
            variant_label_mode = normalize_ab_disclosure_mode(
                self._get_ab_setting(raw_ab_cfg, 'variant_label_mode', 'disclosure_mode', DEFAULT_DISCLOSURE_MODE)
            )
            activity_panel_default_state = normalize_ab_trace_mode(
                self._get_ab_setting(
                    raw_ab_cfg, 'activity_panel_default_state', 'default_trace_mode', DEFAULT_TRACE_MODE
                )
            )
            max_pending = int(
                self._get_ab_setting(
                    raw_ab_cfg,
                    'max_pending_comparisons_per_conversation',
                    'max_pending_per_conversation',
                    1,
                )
            )
            enabled_requested = bool(raw_ab_cfg.get('enabled', False))

            # Validate the resulting pool shape even if currently disabled.
            ABPool(
                variants=variants,
                champion_name=champion_name,
                enabled=True,
                sample_rate=comparison_rate,
                disclosure_mode=variant_label_mode,
                default_trace_mode=activity_panel_default_state,
                max_pending_per_conversation=max_pending,
            )
            self._persist_ab_pool_config(
                enabled=enabled_requested,
                champion_name=champion_name,
                variants=variants,
                comparison_rate=comparison_rate,
                variant_label_mode=variant_label_mode,
                activity_panel_default_state=activity_panel_default_state,
                max_pending_comparisons_per_conversation=max_pending,
            )
            logger.info("Persisted A/B variants update: champion='%s', variants=%s", champion_name, parsed_labels)
            return jsonify(self._build_admin_ab_pool_payload()), 200
        except ABPoolError as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception as exc:
            logger.error("Error setting A/B variants: %s", exc)
            return jsonify({'error': str(exc)}), 500

    def ab_disable_pool(self):
        """
        Disable (clear) the A/B testing pool. Admin only.

        Note: Pool state is ephemeral (in-memory only). Changes made via the
        UI will be lost on server restart. The pool reverts to whatever is
        configured in config.yaml.
        """
        if not self._can_manage_ab_testing():
            return jsonify({'error': 'Admin access required'}), 403
        try:
            self.config_service.update_services_config({
                "chat_app": {
                    "ab_testing": {
                        "enabled": False,
                    }
                }
            })
            self._refresh_runtime_config()
            logger.info("Persisted A/B pool disable")
            return jsonify(self._build_admin_ab_pool_payload()), 200
        except Exception as exc:
            logger.error("Error disabling A/B pool: %s", exc)
            return jsonify({'error': str(exc)}), 500

    def ab_compare_stream(self):
        """
        Stream a pool-based A/B comparison (champion vs variant).

        POST body:
        - message: [sender, content] pair
        - conversation_id: The conversation ID (optional)
        - client_id: Client ID for authorization
        - config_name: Config name (optional)

        Returns:
            NDJSON stream with arm-tagged events.
        """
        if not self._can_use_ab_testing():
            return jsonify({'error': 'A/B testing is not enabled for this user'}), 403

        server_received_msg_ts = datetime.now()
        request_data = self._parse_chat_request()

        message = request_data["message"]
        conversation_id = request_data["conversation_id"]
        config_name = request_data["config_name"]
        is_refresh = request_data["is_refresh"]
        client_sent_msg_ts = request_data["client_sent_msg_ts"]
        client_timeout = request_data["client_timeout"]
        client_id = request_data["client_id"]
        provider = request_data["provider"]
        model = request_data["model"]
        user_id = session.get('user', {}).get('id') or None
        session_api_key = None

        if not client_id:
            return jsonify({"error": "client_id missing"}), 400

        # G2: stage playbook state (verified owner + pending /name invocation) on this
        # request context, mirroring the normal chat endpoints (get_chat_response[_stream]),
        # so stream_ab_comparison -> _prepare_chat_context injects the body into both arms'
        # shared history and the existing chip-insert fires. Also sets the owner ContextVar
        # that the arm threads re-establish below.
        self._stage_playbook_for_request(client_id, request_data["playbook_name"])

        if provider and 'provider_api_keys' in session:
            session_api_key = session.get('provider_api_keys', {}).get(provider.lower())

        if conversation_id:
            pending_count = self.chat.conv_service.count_pending_ab_comparisons(conversation_id)
            max_pending = (
                int(self.chat.ab_pool.max_pending_comparisons_per_conversation)
                if self.chat.ab_pool else 1
            )
            if pending_count >= max_pending:
                return jsonify({
                    'error': 'Resolve one of the pending comparisons before sending another message',
                    'pending_count': pending_count,
                    'max_pending_comparisons_per_conversation': max_pending,
                }), 409

        return self._ndjson_response(self.chat.stream_ab_comparison(
            message,
            conversation_id,
            client_id,
            is_refresh,
            server_received_msg_ts,
            client_sent_msg_ts,
            client_timeout,
            config_name,
            user_id=user_id,
            provider=provider,
            model=model,
            provider_api_key=session_api_key,
        ))

    def ab_get_metrics(self):
        """
        Get per-variant A/B testing metrics. Admin only.

        Returns:
            JSON with variant metrics (wins, losses, ties, total).
        """
        if not self._can_view_ab_metrics():
            return jsonify({'error': 'Admin access required'}), 403
        try:
            metrics = self.chat.conv_service.get_all_variant_metrics()
            return jsonify({'success': True, 'metrics': metrics}), 200
        except Exception as e:
            logger.error(f"Error getting A/B metrics: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Agent Trace Endpoints
    # =========================================================================

    def get_trace(self, trace_id: str):
        """
        Get an agent trace by ID.

        URL params:
        - trace_id: The trace UUID

        Returns:
            JSON with trace data
        """
        try:
            trace = self.chat.get_agent_trace(trace_id)
            if trace is None:
                return jsonify({'error': 'Trace not found'}), 404

            return jsonify({
                'success': True,
                'trace': trace,
            }), 200

        except Exception as e:
            logger.error(f"Error getting trace {trace_id}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def get_trace_by_message(self, message_id: int):
        """
        Get agent trace by the final message ID.

        URL params:
        - message_id: The message ID

        Returns:
            JSON with trace data
        """
        try:
            trace = self.chat.get_trace_by_message(message_id)
            if trace is None:
                return jsonify({'error': 'Trace not found for message'}), 404

            return jsonify({
                'success': True,
                'trace': trace,
            }), 200

        except Exception as e:
            logger.error(f"Error getting trace for message {message_id}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def cancel_stream(self):
        """
        Cancel an active streaming request for a conversation.

        POST body:
        - conversation_id: The conversation ID
        - client_id: Client ID for authorization

        Returns:
            JSON with cancellation status
        """
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            client_id = data.get('client_id')

            if not conversation_id:
                return jsonify({'error': 'conversation_id is required'}), 400
            if not client_id:
                return jsonify({'error': 'client_id is required'}), 400

            # Cancel any active traces for this conversation
            cancelled_count = self.chat.cancel_active_traces(
                conversation_id=conversation_id,
                cancelled_by='user',
                cancellation_reason='Cancelled by user request',
            )

            return jsonify({
                'success': True,
                'cancelled_count': cancelled_count,
            }), 200

        except Exception as e:
            logger.error(f"Error cancelling stream for conversation {conversation_id}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Data Viewer Endpoints
    # =========================================================================

    def data_viewer_page(self):
        """Render the data viewer page."""
        return render_template(
            'data.html',
            can_view_ab_testing=self._can_view_ab_testing(),
        )

    def ab_testing_admin_page(self):
        """Render the dedicated admin A/B testing management page."""
        can_view = self._can_view_ab_testing()
        if not can_view:
            return "Forbidden", 403
        return render_template(
            'ab_testing.html',
            can_manage_ab_testing=self._can_manage_ab_testing(),
            can_view_ab_metrics=self._can_view_ab_metrics(),
        )

    def list_data_documents(self):
        """
        List documents with per-chat enabled state.

        Query params:
        - conversation_id: Optional. The conversation ID for per-chat state.
                          If omitted, shows all documents as enabled.
        - source_type: Optional. Filter by "local", "web", "ticket", or "all".
        - search: Optional. Search query for display_name and url.
        - enabled: Optional. Filter by "all", "enabled", or "disabled".
        - limit: Optional. Max results (default 100), or "all" for full retrieval.
        - offset: Optional. Pagination offset (default 0).

        Returns:
            JSON with documents list, total, enabled_count, limit, offset,
            has_more, next_offset
        """
        try:
            conversation_id = request.args.get('conversation_id')  # Optional now

            source_type = request.args.get('source_type', 'all')
            search = request.args.get('search', '')
            enabled_filter = request.args.get('enabled', 'all')
            limit_param = request.args.get('limit', '100')
            offset = request.args.get('offset', 0, type=int)
            limit = None
            if str(limit_param).lower() != 'all':
                try:
                    parsed_limit = int(limit_param)
                except (TypeError, ValueError):
                    return jsonify({'error': 'limit must be an integer or "all"'}), 400
                # Clamp paged requests to keep payloads bounded
                limit = max(1, min(parsed_limit, 500))

            result = self.chat.data_viewer.list_documents(
                conversation_id=conversation_id,
                source_type=source_type if source_type != 'all' else None,
                search=search if search else None,
                enabled_filter=enabled_filter if enabled_filter != 'all' else None,
                limit=limit,
                offset=offset,
            )

            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error listing data documents: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def get_data_document_content(self, document_hash: str):
        """
        Get document content for preview.

        URL params:
        - document_hash: The document's SHA-256 hash

        Query params:
        - max_size: Optional. Max content size (default 100000).

        Returns:
            JSON with hash, display_name, content, content_type, size_bytes, truncated
        """
        try:
            max_size = request.args.get('max_size', 100000, type=int)
            max_size = max(1000, min(max_size, 1000000))  # Clamp between 1KB and 1MB

            result = self.chat.data_viewer.get_document_content(document_hash, max_size)
            if result is None:
                return jsonify({'error': 'Document not found'}), 404

            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error getting document content for {document_hash}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def get_data_document_chunks(self, document_hash: str):
        """
        Get chunks for a document.

        URL params:
        - document_hash: The document's SHA-256 hash

        Returns:
            JSON with hash, chunks (list of {index, text, start_char, end_char})
        """
        try:
            chunks = self.chat.data_viewer.get_document_chunks(document_hash)
            return jsonify({
                'hash': document_hash,
                'chunks': chunks,
                'total': len(chunks)
            }), 200

        except Exception as e:
            logger.error(f"Error getting chunks for {document_hash}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def enable_data_document(self, document_hash: str):
        """
        Enable a document for the current chat.

        URL params:
        - document_hash: The document's SHA-256 hash

        POST body:
        - conversation_id: The conversation ID

        Returns:
            JSON with success, hash, enabled
        """
        try:
            data = request.json or {}
            conversation_id = data.get('conversation_id')
            if not conversation_id:
                return jsonify({'error': 'conversation_id is required'}), 400

            result = self.chat.data_viewer.enable_document(conversation_id, document_hash)
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error enabling document {document_hash}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def disable_data_document(self, document_hash: str):
        """
        Disable a document for the current chat.

        URL params:
        - document_hash: The document's SHA-256 hash

        POST body:
        - conversation_id: The conversation ID

        Returns:
            JSON with success, hash, enabled
        """
        try:
            data = request.json or {}
            conversation_id = data.get('conversation_id')
            if not conversation_id:
                return jsonify({'error': 'conversation_id is required'}), 400

            result = self.chat.data_viewer.disable_document(conversation_id, document_hash)
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error disabling document {document_hash}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def bulk_enable_documents(self):
        """
        Enable multiple documents for the current chat.

        POST body:
        - conversation_id: The conversation ID
        - hashes: List of document hashes to enable

        Returns:
            JSON with success, enabled_count
        """
        try:
            data = request.json or {}
            conversation_id = data.get('conversation_id')
            hashes = data.get('hashes', [])

            if not conversation_id:
                return jsonify({'error': 'conversation_id is required'}), 400
            if not isinstance(hashes, list):
                return jsonify({'error': 'hashes must be a list'}), 400

            result = self.chat.data_viewer.bulk_enable(conversation_id, hashes)
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error bulk enabling documents: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def bulk_disable_documents(self):
        """
        Disable multiple documents for the current chat.

        POST body:
        - conversation_id: The conversation ID
        - hashes: List of document hashes to disable

        Returns:
            JSON with success, disabled_count
        """
        try:
            data = request.json or {}
            conversation_id = data.get('conversation_id')
            hashes = data.get('hashes', [])

            if not conversation_id:
                return jsonify({'error': 'conversation_id is required'}), 400
            if not isinstance(hashes, list):
                return jsonify({'error': 'hashes must be a list'}), 400

            result = self.chat.data_viewer.bulk_disable(conversation_id, hashes)
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error bulk disabling documents: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def get_data_stats(self):
        """
        Get statistics for the data viewer.

        Query params:
        - conversation_id: Optional. The conversation ID for per-chat stats.
                          If omitted, shows stats for all documents as enabled.

        Returns:
            JSON with total_documents, enabled_documents, disabled_documents,
            total_size_bytes, by_source_type, last_sync
        """
        try:
            conversation_id = request.args.get('conversation_id')  # Optional now

            result = self.chat.data_viewer.get_stats(conversation_id)
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error getting data stats: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # =========================================================================
    # Data Uploader Endpoints
    # =========================================================================

    def upload_page(self):
        """Render the data upload page."""
        return render_template('upload.html')

    def upload_file(self):
        """
        Handle file uploads via multipart form data.
        Proxies to data-manager service.
        """
        try:
            upload = request.files.get("file")
            if not upload:
                return jsonify({"error": "missing_file"}), 400

            # Read file into memory to avoid stream position / exhaustion issues
            file_bytes = upload.stream.read()
            filename = upload.filename or "upload"
            content_type = upload.content_type or "application/octet-stream"

            # Proxy to data-manager service (long timeout for large files)
            resp = requests.post(
                f"{self.data_manager_url}/document_index/upload",
                files={"file": (filename, file_bytes, content_type)},
                headers=self._dm_headers,
                timeout=600,
                allow_redirects=False,
            )

            # Detect auth redirect (data-manager returns 302 → login page)
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                logger.error("Data-manager rejected upload (auth redirect to %s)", resp.headers.get("Location"))
                return jsonify({"error": "Data manager authentication failed"}), 502

            # Safely parse the response — data-manager may return
            # an empty body or non-JSON on error (e.g. OOM, crash).
            try:
                data = resp.json()
            except ValueError:
                logger.error(
                    "Data-manager returned non-JSON response for upload "
                    "(status=%s, body=%r)",
                    resp.status_code,
                    resp.text[:500],
                )
                return jsonify({
                    "error": f"Data manager error (HTTP {resp.status_code})"
                }), 502

            if resp.status_code == 200 and data.get("status") == "ok":
                return jsonify({
                    "success": True,
                    "filename": filename,
                    "path": data.get("path", "")
                }), 200
            else:
                return jsonify({"error": data.get("error", "upload_failed")}), resp.status_code

        except requests.exceptions.ConnectionError:
            logger.error("Data manager service unavailable")
            return jsonify({"error": "data_manager_unavailable"}), 503
        except requests.exceptions.Timeout:
            logger.error("Data manager timed out processing upload")
            return jsonify({"error": "Upload timed out — file may be too large"}), 504
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def upload_url(self):
        """
        Scrape and ingest content from a URL.
        Proxies to data-manager service.
        """
        try:
            data = request.json or {}
            url = data.get("url", "").strip()
            depth = data.get("depth", None)

            if not url:
                return jsonify({"error": "missing_url"}), 400
            if depth is not None:
                try:
                    depth = int(depth)
                except (TypeError, ValueError):
                    return jsonify({"error": "invalid_depth"}), 400
                if depth < 0:
                    return jsonify({"error": "invalid_depth"}), 400

            # Proxy to data-manager service
            dm_payload = {"url": url}
            if depth is not None:
                dm_payload["depth"] = str(depth)
            resp = requests.post(
                f"{self.data_manager_url}/document_index/upload_url",
                data=dm_payload,
                headers=self._dm_headers,
                timeout=300,
                allow_redirects=False,
            )

            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                logger.error("Data-manager rejected upload_url (auth redirect)")
                return jsonify({"error": "Data manager authentication failed"}), 502

            try:
                dm_data = resp.json()
            except ValueError:
                logger.error(
                    "Data-manager returned non-JSON for upload_url (status=%s, body=%r)",
                    resp.status_code,
                    resp.text[:500],
                )
                return jsonify({"error": f"Data manager error (HTTP {resp.status_code})"}), 502

            if resp.status_code == 200 and dm_data.get("status") == "ok":
                return jsonify({
                    "success": True,
                    "url": url,
                    "resources_scraped": dm_data.get("resources_scraped", 1)
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": dm_data.get("error", "scrape_failed"),
                    "url": url
                }), resp.status_code if resp.status_code != 200 else 400

        except requests.exceptions.ConnectionError:
            logger.error("Data manager service unavailable")
            return jsonify({"error": "data_manager_unavailable"}), 503
        except Exception as e:
            logger.error(f"Error uploading URL: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def upload_git(self):
        """
        Clone and ingest a Git repository (POST), or delete a git repo (DELETE).
        Proxies to data-manager service.
        """
        try:
            if request.method == 'DELETE':
                return self._delete_git_repo()
            
            data = request.json or {}
            repo_url = data.get("repo_url", "").strip()

            if not repo_url:
                return jsonify({"error": "missing_repo_url"}), 400

            # Proxy to data-manager service
            resp = requests.post(
                f"{self.data_manager_url}/document_index/add_git_repo",
                data={"repo_url": repo_url},
                headers=self._dm_headers,
                timeout=300,  # Git clones can take a while
                allow_redirects=False,
            )

            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                logger.error("Data-manager rejected add_git_repo (auth redirect)")
                return jsonify({"error": "Data manager authentication failed"}), 502

            try:
                dm_data = resp.json()
            except ValueError:
                logger.error("Data-manager returned non-JSON for add_git_repo (status=%s)", resp.status_code)
                return jsonify({"error": f"Data manager error (HTTP {resp.status_code})"}), 502

            if resp.status_code == 200 and dm_data.get("status") == "ok":
                return jsonify({
                    "success": True,
                    "repo_url": repo_url,
                    "message": "Repository cloned. Documents will be embedded shortly."
                }), 200
            else:
                return jsonify({"error": dm_data.get("error", "git_clone_failed")}), resp.status_code

        except requests.exceptions.ConnectionError:
            logger.error("Data manager service unavailable")
            return jsonify({"error": "data_manager_unavailable"}), 503
        except Exception as e:
            logger.error(f"Error cloning Git repo: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def _delete_source_documents(self, source_type: str, where_clause: str, params: tuple, label: str):
        """
        Shared helper: mark documents as deleted and remove their chunks.

        Args:
            source_type: 'git' or 'jira'
            where_clause: SQL WHERE fragment after 'source_type = %s AND NOT is_deleted AND'
            params: bind-parameters for the WHERE clause
            label: human-readable label for log/response messages
        """
        conn = psycopg2.connect(**self.chat.pg_config)
        try:
            with conn.cursor() as cursor:
                # Get resource hashes of documents to delete
                cursor.execute(
                    f"SELECT resource_hash FROM documents WHERE source_type = %s AND NOT is_deleted AND {where_clause}",
                    (source_type, *params),
                )
                hashes_to_delete = [row[0] for row in cursor.fetchall()]

                if hashes_to_delete:
                    cursor.execute(
                        "DELETE FROM document_chunks WHERE metadata->>'resource_hash' = ANY(%s)",
                        (hashes_to_delete,),
                    )
                    logger.info(f"Deleted {cursor.rowcount} chunks for {len(hashes_to_delete)} {source_type} documents")

                cursor.execute(
                    f"UPDATE documents SET is_deleted = TRUE, deleted_at = NOW() WHERE source_type = %s AND NOT is_deleted AND {where_clause}",
                    (source_type, *params),
                )
                deleted_count = cursor.rowcount
                conn.commit()

            logger.info(f"Deleted {deleted_count} documents from {label}")
            return jsonify({
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Removed {deleted_count} documents from {label}",
            }), 200
        finally:
            conn.close()

    def _delete_git_repo(self):
        """
        Delete a Git repository and all its indexed documents.
        Marks documents as deleted in the database and removes their chunks.
        """
        try:
            data = request.json or {}
            repo_name = data.get("repo_name", "").strip()
            
            if not repo_name:
                return jsonify({"error": "missing_repo_name"}), 400
            
            return self._delete_source_documents(
                source_type='git',
                where_clause='(url LIKE %s OR url LIKE %s)',
                params=(f'{repo_name}/%', f'%/{repo_name}/%'),
                label=f"git repo: {repo_name}",
            )
        except Exception as e:
            logger.error(f"Error deleting Git repo: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def refresh_git(self):
        """
        Refresh (re-clone) a Git repository to get latest changes.
        Proxies to data-manager service.
        """
        try:
            # Handle JSON parsing errors gracefully
            try:
                data = request.json
            except Exception:
                return jsonify({"error": "invalid_json"}), 400
            
            if data is None:
                return jsonify({"error": "invalid_json"}), 400
            
            repo_name = data.get("repo_name")
            
            # Type validation: repo_name must be a string
            if repo_name is None or not isinstance(repo_name, str):
                return jsonify({"error": "invalid_repo_name_type"}), 400
            
            repo_name = repo_name.strip()

            if not repo_name:
                return jsonify({"error": "missing_repo_name"}), 400
            
            # Input validation: reject overly long inputs (max 500 chars for repo names/URLs)
            if len(repo_name) > 500:
                return jsonify({"error": "repo_name_too_long"}), 400

            # The repo_name might be a URL or just a name
            # Try to reconstruct the full URL if needed
            if repo_name.startswith('http'):
                repo_url = repo_name
            else:
                # Query the database to find the full URL
                try:
                    conn = psycopg2.connect(**self.chat.pg_config)
                except Exception as db_err:
                    logger.error(f"Database connection failed: {db_err}")
                    return jsonify({"error": "database_unavailable"}), 503
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT DISTINCT 
                                CASE 
                                    WHEN url LIKE 'https://github.com/%' THEN
                                        regexp_replace(url, '^(https://github.com/[^/]+/[^/]+).*', '\\1')
                                    WHEN url LIKE 'https://gitlab.com/%' THEN
                                        regexp_replace(url, '^(https://gitlab.com/[^/]+/[^/]+).*', '\\1')
                                    ELSE url
                                END as repo_url
                            FROM documents 
                            WHERE source_type = 'git' 
                              AND NOT is_deleted
                              AND url LIKE %s
                            LIMIT 1
                        """, (f'%/{repo_name}%',))
                        row = cursor.fetchone()
                        if not row:
                            return jsonify({"error": "repo_not_found"}), 404
                        repo_url = row[0]
                finally:
                    conn.close()

            # Proxy to data-manager service to re-clone
            resp = requests.post(
                f"{self.data_manager_url}/document_index/add_git_repo",
                data={"repo_url": repo_url},
                headers=self._dm_headers,
                timeout=300,
                allow_redirects=False,
            )

            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                logger.error("Data-manager rejected git refresh (auth redirect)")
                return jsonify({"error": "Data manager authentication failed"}), 502

            # Try to parse JSON response, handle non-JSON gracefully
            try:
                dm_data = resp.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                logger.warning(f"Data manager returned non-JSON response: {resp.status_code}")
                if resp.status_code >= 500:
                    return jsonify({"error": "data_manager_error"}), 503
                return jsonify({"error": "git_refresh_failed"}), resp.status_code or 400

            if resp.status_code == 200 and dm_data.get("status") == "ok":
                return jsonify({
                    "success": True,
                    "repo_url": repo_url,
                    "message": "Repository refreshed."
                }), 200
            else:
                # Return the data manager's status code but cap at 503 for server errors
                status = resp.status_code if resp.status_code < 500 else 503
                return jsonify({"error": dm_data.get("error", "git_refresh_failed")}), status

        except requests.exceptions.ConnectionError:
            logger.error("Data manager service unavailable")
            return jsonify({"error": "data_manager_unavailable"}), 503
        except requests.exceptions.Timeout:
            logger.error("Data manager request timed out")
            return jsonify({"error": "data_manager_timeout"}), 503
        except Exception as e:
            logger.error(f"Error refreshing Git repo: {str(e)}")
            return jsonify({"error": "internal_error"}), 503

    def upload_jira(self):
        """
        Sync issues from a Jira project.
        Proxies to data-manager service.
        """
        try:
            data = request.json or {}
            project_key = data.get("project_key", "").strip()

            if not project_key:
                return jsonify({"error": "missing_project_key"}), 400

            # Proxy to data-manager service
            resp = requests.post(
                f"{self.data_manager_url}/document_index/add_jira_project",
                data={"project_key": project_key},
                headers=self._dm_headers,
                timeout=300,
                allow_redirects=False,
            )

            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                logger.error("Data-manager rejected jira sync (auth redirect)")
                return jsonify({"error": "Data manager authentication failed"}), 502

            try:
                dm_data = resp.json()
            except ValueError:
                logger.error("Data-manager returned non-JSON for add_jira_project (status=%s)", resp.status_code)
                return jsonify({"error": f"Data manager error (HTTP {resp.status_code})"}), 502

            if resp.status_code == 200 and dm_data.get("status") == "ok":
                return jsonify({
                    "success": True,
                    "project_key": project_key
                }), 200
            else:
                return jsonify({"error": dm_data.get("error", "jira_sync_failed")}), resp.status_code

        except requests.exceptions.ConnectionError:
            logger.error("Data manager service unavailable")
            return jsonify({"error": "data_manager_unavailable"}), 503
        except Exception as e:
            logger.error(f"Error syncing Jira project: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def trigger_embedding(self):
        """
        Trigger embedding/vectorstore update for recently uploaded documents.

        This synchronizes the documents catalog with the vectorstore,
        creating embeddings for any new documents that haven't been processed yet.

        Returns:
            JSON with embedding status including any failures
        """
        try:
            logger.info("Triggering vectorstore update...")
            self.chat.vector_manager.update_vectorstore()
            logger.info("Vectorstore update completed")

            # Check for failed documents after processing
            failed_docs = []
            try:
                conn = psycopg2.connect(**self.chat.pg_config)
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT display_name, ingestion_error
                            FROM documents
                            WHERE NOT is_deleted AND ingestion_status = 'failed'
                            ORDER BY created_at DESC
                            LIMIT 20
                            """
                        )
                        failed_docs = [
                            {"file": row[0], "error": row[1] or "Unknown error"}
                            for row in cursor.fetchall()
                        ]
                finally:
                    conn.close()
            except Exception as db_err:
                logger.warning(f"Could not check for failed documents: {db_err}")

            if failed_docs:
                return jsonify({
                    "success": True,
                    "partial": True,
                    "message": f"{len(failed_docs)} document(s) failed to process.",
                    "failed": failed_docs,
                }), 200

            return jsonify({
                "success": True,
                "message": "Embedding complete. Documents are now searchable."
            }), 200

        except Exception as e:
            logger.error(f"Error triggering embedding: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def get_embedding_status(self):
        """
        Get the current embedding/ingestion status.

        Returns:
            JSON with counts of documents by ingestion status
        """
        try:
            conn = psycopg2.connect(**self.chat.pg_config)
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT ingestion_status, COUNT(*) as count
                        FROM documents
                        WHERE NOT is_deleted
                        GROUP BY ingestion_status
                        """
                    )
                    status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            finally:
                conn.close()

            pending = status_counts.get("pending", 0)
            embedding = status_counts.get("embedding", 0)
            embedded = status_counts.get("embedded", 0)
            failed = status_counts.get("failed", 0)
            total = pending + embedding + embedded + failed
            
            return jsonify({
                "documents_in_catalog": total,
                "documents_embedded": embedded,
                "pending_embedding": pending,
                "is_synced": pending == 0 and embedding == 0,
                "status_counts": {
                    "pending": pending,
                    "embedding": embedding,
                    "embedded": embedded,
                    "failed": failed,
                },
            }), 200

        except Exception as e:
            logger.error(f"Error getting embedding status: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def list_upload_documents(self):
        """
        List documents with their ingestion status for the upload page.

        Query params:
            status: Filter by ingestion status (pending, embedding, embedded, failed)
            source_type: Filter by source type
            search: Search by display name
            limit: Max results (default 50)
            offset: Pagination offset (default 0)
        
        Returns:
            JSON with documents, total, status_counts
        """
        try:
            from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService
            
            catalog = PostgresCatalogService(
                data_path=self.chat.data_path,
                pg_config=self.chat.pg_config,
            )
            
            result = catalog.list_documents_with_status(
                status_filter=request.args.get("status"),
                source_type=request.args.get("source_type"),
                search=request.args.get("search"),
                limit=int(request.args.get("limit", 50)),
                offset=int(request.args.get("offset", 0)),
            )
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error listing upload documents: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def retry_document(self, document_hash):
        """
        Reset a failed document back to pending so it can be retried.

        Args:
            document_hash: The resource_hash of the document to retry
        
        Returns:
            JSON with success status
        """
        try:
            from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService
            
            catalog = PostgresCatalogService(
                data_path=self.chat.data_path,
                pg_config=self.chat.pg_config,
            )
            
            reset = catalog.reset_failed_document(document_hash)
            if reset:
                return jsonify({"success": True, "message": "Document reset to pending"}), 200
            else:
                return jsonify({"error": "Document not found or not in failed state"}), 404
        except Exception as e:
            logger.error(f"Error retrying document: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def retry_all_failed(self):
        """
        Reset all failed documents back to pending so they can be retried.

        Returns:
            JSON with count of documents reset
        """
        try:
            from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService

            catalog = PostgresCatalogService(
                data_path=self.chat.data_path,
                pg_config=self.chat.pg_config,
            )

            count = catalog.reset_all_failed_documents()
            return jsonify({"success": True, "count": count, "message": f"{count} document(s) reset to pending"}), 200
        except Exception as e:
            logger.error(f"Error retrying all failed documents: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def list_upload_documents_grouped(self):
        """
        List documents grouped by source origin for the unified status section.

        Query params:
            show_all: If 'true', include all groups (not just actionable). Default false.
            expand: Source group name to load full document list for.

        Returns:
            JSON with groups and aggregate status_counts
        """
        try:
            from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService

            catalog = PostgresCatalogService(
                data_path=self.chat.data_path,
                pg_config=self.chat.pg_config,
            )

            result = catalog.list_documents_grouped(
                show_all=request.args.get("show_all", "false").lower() == "true",
                expand=request.args.get("expand"),
            )
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error listing grouped documents: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def list_git_sources(self):
        """
        List currently synced Git repositories.

        Returns:
            JSON with list of git sources
        """
        try:
            # Query unique git repos from the database directly
            conn = psycopg2.connect(**self.chat.pg_config)
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    # Get unique git repos by extracting the repo URL from document URLs
                    cursor.execute("""
                        SELECT DISTINCT 
                            CASE 
                                WHEN url LIKE 'https://github.com/%' THEN
                                    regexp_replace(url, '^(https://github.com/[^/]+/[^/]+).*', '\\1')
                                WHEN url LIKE 'https://gitlab.com/%' THEN
                                    regexp_replace(url, '^(https://gitlab.com/[^/]+/[^/]+).*', '\\1')
                                ELSE url
                            END as repo_url,
                            COUNT(*) as file_count,
                            MAX(indexed_at) as last_updated
                        FROM documents 
                        WHERE source_type = 'git' 
                          AND NOT is_deleted
                          AND url IS NOT NULL
                        GROUP BY 1
                        ORDER BY last_updated DESC NULLS LAST
                    """)
                    rows = cursor.fetchall()
            finally:
                conn.close()

            sources = []
            for row in rows:
                repo_url = row['repo_url']
                if repo_url:
                    # Extract repo name from URL
                    name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
                    sources.append({
                        'name': name,
                        'url': repo_url,
                        'file_count': row['file_count'],
                        'last_updated': row['last_updated'].isoformat() if row['last_updated'] else None
                    })

            return jsonify({"sources": sources}), 200

        except Exception as e:
            logger.error(f"Error listing Git sources: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def list_jira_sources(self):
        """
        List currently synced Jira projects (GET), or delete a project (DELETE).

        Returns:
            JSON with list of jira sources or deletion status
        """
        try:
            if request.method == 'DELETE':
                return self._delete_jira_project()
                
            sources = []
            seen_projects = set()

            result = self.chat.data_viewer.list_documents(source_type='ticket', limit=1000)

            for doc in result.get('documents', []):
                # Parse project key from display name or URL
                display_name = doc.get('display_name', '')
                url = doc.get('url', '')
                # Jira documents often have display_name like "PROJECT-123: Title"
                if display_name:
                    project_key = display_name.split('-')[0] if '-' in display_name else display_name
                    if project_key and project_key not in seen_projects:
                        seen_projects.add(project_key)
                        logger.debug(f"Adding project key: {project_key}, display_name: {display_name}")
                        sources.append({
                            'key': project_key,
                            'name': url.split('-')[0] if '-' in url else url,
                        })

            for project in sources:
                project_key = project['key']
                
                ticket_count = sum(1 for doc in result.get('documents', []) if doc.get('display_name', '').startswith(project_key + '-'))
                project['ticket_count'] = ticket_count if ticket_count else 0
                
                last_sync = max((doc.get('ingested_at')
                                for doc in result.get('documents', [])
                                if project_key in doc.get('display_name', '') and doc.get('ingested_at') is not None),
                                default=None)
                
                project['last_sync'] = last_sync if last_sync else None

            return jsonify({"sources": sources}), 200

        except Exception as e:
            logger.error(f"Error listing Jira sources: {str(e)}",exc_info=True)
            return jsonify({"error": str(e)}), 500

    def _delete_jira_project(self):
        """
        Delete a Jira project and all its synced tickets.
        Marks documents as deleted in the database and removes their chunks.
        """
        try:
            data = request.json or {}
            project_key = data.get("project_key", "").strip()
            
            if not project_key:
                return jsonify({"error": "missing_project_key"}), 400
            
            return self._delete_source_documents(
                source_type='jira',
                where_clause='display_name LIKE %s',
                params=(f'{project_key}-%',),
                label=f"Jira project: {project_key}",
            )
        except Exception as e:
            logger.error(f"Error deleting Jira project: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def source_schedules_dispatch(self):
        """Route /api/sources/schedules to GET or PUT handler."""
        if request.method == "PUT":
            return self.update_source_schedule()
        return self.get_source_schedules()

    def get_source_schedules(self):
        """
        Get all source sync schedules.

        Returns:
            JSON with source schedules
        """
        try:
            schedules = self.config_service.get_source_schedules()
            jobs_by_source = {}

            # Best-effort enrich with scheduler runtime metadata from data-manager.
            try:
                dm_response = requests.get(
                    f"{self.data_manager_url}/api/schedules",
                    headers=self._dm_headers,
                    timeout=10,
                    allow_redirects=False,
                )
                if dm_response.ok and not dm_response.is_redirect:
                    jobs = (dm_response.json() or {}).get("jobs", [])
                    jobs_by_source = {
                        (job.get("name") or ""): job
                        for job in jobs
                        if isinstance(job, dict)
                    }
            except requests.exceptions.RequestException as e:
                logger.warning(f"Could not fetch scheduler runtime status from data-manager: {e}")
            
            # Convert cron expressions to UI-friendly values
            schedule_display = {}
            cron_to_ui = {
                '': 'disabled',
                '0 * * * *': 'hourly',
                '0 */6 * * *': 'every_6h',
                '0 0 * * *': 'daily',
            }
            
            for source, cron in schedules.items():
                runtime = jobs_by_source.get(source, {})
                schedule_display[source] = {
                    'cron': cron,
                    'display': cron_to_ui.get(cron, 'custom'),
                    'next_run': runtime.get('next_run'),
                    'last_run': runtime.get('last_run'),
                }
            
            return jsonify({"schedules": schedule_display}), 200

        except Exception as e:
            logger.error(f"Error getting source schedules: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def update_source_schedule(self):
        """
        Update the schedule for a specific data source.

        PUT body (JSON):
        - source: Source name (e.g., 'jira', 'git', 'links')
        - schedule: Schedule value ('disabled', 'hourly', 'every_6h', 'daily', or cron expression)

        Returns:
            JSON with updated schedules
        """
        try:
            data = request.json or {}
            source = data.get("source", "").strip()
            schedule = data.get("schedule", "").strip()

            if not source:
                return jsonify({"error": "missing_source"}), 400
            
            valid_sources = ['jira', 'git', 'links', 'local_files', 'redmine', 'sso']
            if source not in valid_sources:
                return jsonify({"error": f"invalid_source, must be one of {valid_sources}"}), 400

            # Get current user for audit logging, if available
            user_id = None
            if session.get('logged_in'):
                user = session.get('user', {})
                user_id = user.get('username') or user.get('email') or 'anonymous'
            
            schedules = self.config_service.update_source_schedule(
                source, 
                schedule,
                updated_by=user_id
            )

            # Notify data-manager to reload schedules immediately
            reload_result = None
            try:
                response = requests.post(
                    f"{self.data_manager_url}/api/reload-schedules",
                    headers=self._dm_headers,
                    timeout=10,
                    allow_redirects=False,
                )
                if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                    logger.warning("Data-manager rejected schedule reload (auth redirect)")
                elif response.ok:
                    reload_result = response.json()
                    logger.info(f"Data-manager reloaded schedules: {reload_result}")
                else:
                    logger.warning(f"Data-manager schedule reload failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Could not notify data-manager to reload schedules: {e}")

            return jsonify({
                "success": True,
                "schedules": schedules,
                "reload_result": reload_result
            }), 200

        except Exception as e:
            logger.error(f"Error updating source schedule: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Database Viewer Endpoints
    # =========================================================================

    def database_viewer_page(self):
        """Render the database viewer page."""
        return render_template('database.html')

    def list_database_tables(self):
        """
        List all tables in the database.

        Returns:
            JSON with list of tables and their row counts
        """
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(
                host=self.pg_config.get("host", "postgres"),
                port=self.pg_config.get("port", 5432),
                database=self.pg_config.get("database", "archi"),
                user=self.pg_config.get("user", "archi"),
                password=self.pg_config.get("password"),
            )
            cursor = conn.cursor()

            # Get list of tables with row counts
            # Note: pg_stat_user_tables uses 'relname' not 'tablename' in some PostgreSQL versions
            cursor.execute("""
                SELECT 
                    schemaname,
                    relname as tablename,
                    n_live_tup as row_count
                FROM pg_stat_user_tables
                ORDER BY schemaname, relname
            """)

            tables = []
            for row in cursor.fetchall():
                tables.append({
                    'schema': row[0],
                    'name': row[1],
                    'row_count': row[2],
                })

            return jsonify({"tables": tables}), 200

        except Exception as e:
            logger.error(f"Error listing database tables: {str(e)}")
            return jsonify({"error": str(e)}), 500
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def run_database_query(self):
        """
        Execute a read-only SQL query.

        POST body (JSON):
        - query: The SQL query to execute

        Returns:
            JSON with columns and rows
        """
        conn = None
        cursor = None
        try:
            data = request.json or {}
            query = data.get("query", "").strip()

            if not query:
                return jsonify({"error": "missing_query"}), 400

            # Reject multiple statements (semicolon-separated)
            # Strip trailing semicolons+whitespace, then check for remaining semicolons
            query_stripped = query.rstrip('; \t\n')
            if ';' in query_stripped:
                return jsonify({"error": "only_single_statement", "message": "Only a single SQL statement is allowed"}), 400

            # Basic security: only allow SELECT statements
            query_upper = query_stripped.upper().strip()
            if not query_upper.startswith("SELECT"):
                return jsonify({"error": "only_select_allowed", "message": "Only SELECT queries are allowed"}), 400

            # Block dangerous patterns - check for keywords as separate tokens
            dangerous_keywords = [
                'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
                'TRUNCATE', 'GRANT', 'REVOKE', 'COPY', 'EXECUTE', 'EXEC',
                'INTO', 'CALL',
            ]
            # Split on non-word characters and check for exact keyword matches
            tokens = set(re.findall(r'\b\w+\b', query_upper))
            for keyword in dangerous_keywords:
                if keyword in tokens:
                    return jsonify({"error": "forbidden_operation", "message": f"Operation '{keyword}' is not allowed"}), 400

            # Block function calls that can read/write the filesystem or execute commands
            dangerous_functions = [
                'PG_READ_FILE', 'PG_READ_BINARY_FILE', 'PG_WRITE_FILE',
                'LO_IMPORT', 'LO_EXPORT', 'LO_GET', 'LO_PUT',
                'PG_LS_DIR', 'PG_STAT_FILE',
                'DBLINK', 'DBLINK_EXEC',
            ]
            for func in dangerous_functions:
                if func in tokens:
                    return jsonify({"error": "forbidden_function", "message": f"Function '{func}' is not allowed"}), 400

            conn = psycopg2.connect(
                host=self.pg_config.get("host", "postgres"),
                port=self.pg_config.get("port", 5432),
                database=self.pg_config.get("database", "archi"),
                user=self.pg_config.get("user", "archi"),
                password=self.pg_config.get("password"),
            )

            # Enforce read-only at the database level
            conn.set_session(readonly=True, autocommit=False)
            cursor = conn.cursor()

            # Set a statement timeout to prevent runaway queries (30 seconds)
            cursor.execute("SET statement_timeout = '30s'")

            # Add a LIMIT if not present to prevent runaway queries
            if "LIMIT" not in query_upper:
                query_stripped += " LIMIT 1000"

            cursor.execute(query_stripped)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            # Convert rows to list of dicts for JSON serialization
            result_rows = []
            for row in rows:
                result_rows.append([
                    str(cell) if cell is not None else None
                    for cell in row
                ])

            return jsonify({
                "columns": columns,
                "rows": result_rows,
                "row_count": len(result_rows),
            }), 200

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return jsonify({"error": str(e)}), 500
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def is_authenticated(self):
        """
        Keeps the state of the authentication.

        Returns true if there has been a correct login authentication and false otherwise.
        """
        return 'logged_in' in session and session['logged_in']
