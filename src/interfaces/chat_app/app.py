import json
import os
import re
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
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
    AgentSpecError,
    list_agent_files,
    load_agent_spec,
    select_agent_spec,
    load_agent_spec_from_text,
    slugify_agent_name,
)
from src.archi.providers.base import ModelInfo, ProviderConfig, ProviderType
from src.utils.config_service import ConfigService
from src.archi.utils.output_dataclass import PipelineOutput
# from src.data_manager.data_manager import DataManager
from src.data_manager.data_viewer_service import DataViewerService
from src.data_manager.vectorstore.manager import VectorStoreManager
from src.utils.env import read_secret
from src.utils.logging import get_logger
from src.utils.config_access import get_full_config, get_services_config, get_global_config, get_dynamic_config
from src.utils.config_service import ConfigService
from src.utils.sql import (
    SQL_INSERT_CONVO, SQL_INSERT_FEEDBACK, SQL_INSERT_TIMING, SQL_QUERY_CONVO,
    SQL_CREATE_CONVERSATION, SQL_UPDATE_CONVERSATION_TIMESTAMP,
    SQL_LIST_CONVERSATIONS, SQL_GET_CONVERSATION_METADATA, SQL_DELETE_CONVERSATION,
    SQL_INSERT_TOOL_CALLS, SQL_QUERY_CONVO_WITH_FEEDBACK, SQL_DELETE_REACTION_FEEDBACK,
    SQL_GET_REACTION_FEEDBACK,
    SQL_UPDATE_AB_PREFERENCE, SQL_GET_AB_COMPARISON,
    SQL_GET_PENDING_AB_COMPARISON,
    SQL_CREATE_AGENT_TRACE, SQL_UPDATE_AGENT_TRACE, SQL_GET_AGENT_TRACE,
    SQL_GET_TRACE_BY_MESSAGE, SQL_GET_ACTIVE_TRACE, SQL_CANCEL_ACTIVE_TRACES,
)
from src.interfaces.chat_app.document_utils import *
from src.interfaces.chat_app.utils import collapse_assistant_sequences
from src.utils.ab_testing import ABPool, ABVariant, ABPoolError, load_ab_pool


logger = get_logger(__name__)


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
    if provider_type == ProviderType.LOCAL and cfg.get("mode"):
        extra["local_mode"] = cfg.get("mode")

    return ProviderConfig(
        provider_type=provider_type,
        enabled=cfg.get("enabled", True),
        base_url=cfg.get("base_url"),
        models=models,
        default_model=cfg.get("default_model"),
        extra_kwargs=extra,
    )

def _config_names():
    cfg = get_full_config()
    return [cfg.get("name", "default")]

# DEFINITIONS
QUERY_LIMIT = 10000 # max queries per conversation
MAIN_PROMPT_FILE = "/root/archi/main.prompt"
CONDENSE_PROMPT_FILE = "/root/archi/condense.prompt"
SUMMARY_PROMPT_FILE = "/root/archi/summary.prompt"
ARCHI_SENDER = "archi"


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


class ChatWrapper:
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
        self.current_model_used = None
        self.current_pipeline_used = None
        self._config_cache = {}
        if self.default_config_name:
            self._config_cache[self.default_config_name] = self.config

        # activate default config
        if self.default_config_name:
            self.update_config(config_name=self.default_config_name)

        # A/B testing pool (loaded from config; None if not configured)
        try:
            self.ab_pool = load_ab_pool(self.config)
        except ABPoolError as exc:
            logger.error("Failed to load A/B testing pool: %s", exc)
            self.ab_pool = None
        if self.ab_pool:
            logger.info(
                "A/B pool active: %d variants, champion='%s'",
                len(self.ab_pool.variants), self.ab_pool.champion_name,
            )
            self._prewarm_ab_models()

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

        model_name = self._extract_model_name(config_payload)
        
        self.current_config_name = target_config_name
        self.current_pipeline_used = agent_class
        self.current_model_used = model_name
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

        visible_sources = top_sources[:5]
        hidden_sources = top_sources[5:]
        _output += '''
        <div style="
            margin-top: 1.5em;
            padding-top: 0.5em;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.75em;
            color: #adb5bd;
            line-height: 1.3;
        ">
            <div style="margin-bottom: 0.3em; font-weight: 500;">Sources:</div>
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

        for entry in visible_sources:
            _output += _entry_html(entry)

        if hidden_sources:
            _output += f'<details style="margin-top: 0.4em;"><summary style="cursor: pointer; color: #66b3ff;">Show all sources ({len(top_sources)})</summary>'
            for entry in hidden_sources:
                _output += _entry_html(entry)
            _output += '</details>'

        _output += '</div>'
        return _output

    @staticmethod
    def format_links_markdown(top_sources):
        """Format source links as markdown (for client-side rendering)."""
        if not top_sources:
            return ""

        visible_sources = top_sources[:5]
        hidden_sources = top_sources[5:]

        _output = "\n\n---\n**Sources:**\n"
        for entry in visible_sources:
            _output += ChatWrapper._format_source_entry(entry)

        if hidden_sources:
            _output += f"\n<details><summary>Show all sources ({len(top_sources)})</summary>\n\n"
            for entry in hidden_sources:
                _output += ChatWrapper._format_source_entry(entry)
            _output += "\n</details>\n"

        return _output

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
    # A/B Comparison Methods
    # =========================================================================

    def update_ab_preference(self, comparison_id: int, preference: str) -> None:
        """
        Record user's preference for an A/B comparison.
        
        Args:
            comparison_id: The comparison to update
            preference: 'a', 'b', or 'tie'
        """
        if preference not in ('a', 'b', 'tie'):
            raise ValueError(f"Invalid preference: {preference}")
            
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(
                SQL_UPDATE_AB_PREFERENCE,
                (preference, datetime.now(), comparison_id)
            )
            conn.commit()
            logger.info(f"Updated A/B comparison {comparison_id} with preference '{preference}'")
        finally:
            cursor.close()
            conn.close()

    def get_ab_comparison(self, comparison_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an A/B comparison by ID.
        
        Returns:
            Dict with comparison data or None if not found
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_GET_AB_COMPARISON, (comparison_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._ab_comparison_from_row(row)
        finally:
            cursor.close()
            conn.close()

    def get_pending_ab_comparison(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the most recent incomplete A/B comparison for a conversation.
        
        Returns:
            Dict with comparison data or None if no pending comparison
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_GET_PENDING_AB_COMPARISON, (conversation_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._ab_comparison_from_row(row)
        finally:
            cursor.close()
            conn.close()

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


    def query_conversation_history(self, conversation_id, client_id):
        """
        Return the conversation history as an ordered list of tuples. The order
        is determined by ascending message_id. Each tuple contains the sender and
        the message content
        """
        # create connection to database (use local vars for thread safety)
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()

        # ensure conversation belongs to client before querying
        cursor.execute(SQL_GET_CONVERSATION_METADATA, (conversation_id, client_id))
        metadata = cursor.fetchone()
        if metadata is None:
            cursor.close()
            conn.close()
            raise ConversationAccessError("Conversation does not exist for this client")

        # query conversation history
        cursor.execute(SQL_QUERY_CONVO, (conversation_id,))
        history = cursor.fetchall()
        history = collapse_assistant_sequences(history, sender_name=ARCHI_SENDER)

        # clean up database connection state
        cursor.close()
        conn.close()

        return history

    def create_conversation(self, first_message: str, client_id: str) -> int:
        """
        Gets first message (activates a new conversation), and generates a title w/ first msg.
        (TODO: commercial ones use one-sentence summarizer to make the title)

        Returns: Conversation ID.

        """
        service = "Chatbot"
        title = first_message[:20] + ("..." if len(first_message) > 20 else "")
        now = datetime.now()
        
        version = os.getenv("APP_VERSION", "unknown")

        # title, created_at, last_message_at, version
        insert_tup = (title, now, now, client_id, version)

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

    def update_conversation_timestamp(self, conversation_id: int, client_id: str):
        """
        Update the last_message_at timestamp for a conversation.
        last_message_at is used to reorder conversations in the UI (on vertical sidebar).
        """
        now = datetime.now()

        # create connection to database (use local vars for thread safety)
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()

        # update timestamp
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

    def insert_conversation(self, conversation_id, user_message, archi_message, link, archi_context, is_refresh=False) -> List[int]:
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
        archi_context = _sanitize(archi_context)

        # construct insert_tups with model_used and pipeline_used
        # Format: (service, conversation_id, sender, content, link, context, ts, model_used, pipeline_used)
        insert_tups = (
            [
                (service, conversation_id, user_sender, user_content, '', '', user_msg_ts, self.current_model_used, self.current_pipeline_used),
                (service, conversation_id, ARCHI_SENDER, archi_content, link, archi_context, archi_msg_ts, self.current_model_used, self.current_pipeline_used),
            ]
            if not is_refresh
            else [
                (service, conversation_id, ARCHI_SENDER, archi_content, link, archi_context, archi_msg_ts, self.current_model_used, self.current_pipeline_used),
            ]
        )

        # create connection to database (use local vars for thread safety)
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        psycopg2.extras.execute_values(cursor, SQL_INSERT_CONVO, insert_tups)
        conn.commit()
        message_ids = list(map(lambda tup: tup[0], cursor.fetchall()))

        # clean up database connection state
        cursor.close()
        conn.close()

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
                        ts = datetime.now()
                else:
                    ts = datetime.now()

                for tc in msg.tool_calls:
                    tool_call_id = tc.get("id", "")
                    if tool_call_id and tool_call_id not in tool_call_timestamps:
                        tool_call_timestamps[tool_call_id] = ts

        insert_tups = []
        step_number = 0
        for tc in tool_calls:
            step_number += 1
            tool_call_id = tc.get("id", "")
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})
            tool_result = tc.get("result", "")
            if len(tool_result) > 500:
                tool_result = tool_result[:500] + "..."
            ts = tool_call_timestamps.get(tool_call_id, datetime.now())

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

    def _init_timestamps(self) -> Dict[str, datetime]:
        return {
            "lock_acquisition_ts": datetime.now(),
            "vectorstore_update_ts": datetime.now(),
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

            # Build provider config from YAML so base_url/mode/default_model are respected
            cfg = _build_provider_config_from_payload(self.config, ProviderType(provider))
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

    def _prewarm_ab_models(self):
        """Pre-load all A/B variant models into Ollama so they're warm for comparisons."""
        import threading
        chat_cfg = self.services_config.get("chat_app", {})
        providers_cfg = chat_cfg.get("providers", {})
        local_cfg = providers_cfg.get("local", {})
        base_url = local_cfg.get("base_url")
        if not base_url:
            return  # not using local/Ollama

        seen_models = set()
        models_to_warm = []
        for variant in self.ab_pool.variants:
            model = variant.model
            provider = variant.provider or chat_cfg.get("default_provider")
            if provider == "local" and model and model not in seen_models:
                seen_models.add(model)
                models_to_warm.append(model)

        if not models_to_warm:
            return

        def _warm(model_name):
            import requests as _requests
            try:
                logger.info("Pre-warming Ollama model '%s' at %s ...", model_name, base_url)
                resp = _requests.post(
                    f"{base_url}/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": "24h", "stream": False},
                    timeout=300,
                )
                logger.info("Pre-warm '%s': HTTP %s", model_name, resp.status_code)
            except Exception as exc:
                logger.warning("Pre-warm '%s' failed: %s", model_name, exc)

        threads = [threading.Thread(target=_warm, args=(m,), daemon=True) for m in models_to_warm]
        for t in threads:
            t.start()
        logger.info("Started pre-warming %d model(s) in background: %s", len(models_to_warm), models_to_warm)

    def _create_variant_archi(self, variant: "ABVariant") -> "archi":
        """
        Build a temporary archi instance configured for a specific A/B variant.

        Falls back to the deployment defaults for any field the variant doesn't override.
        """
        chat_cfg = self.services_config.get("chat_app", {})
        agents_dir = Path(chat_cfg.get("agents_dir", "/root/archi/agents"))

        # Resolve agent spec
        variant_agent_spec = self.agent_spec  # default
        if variant.agent_spec:
            try:
                spec_path = agents_dir / variant.agent_spec
                if not spec_path.exists():
                    spec_path = Path(variant.agent_spec)
                variant_agent_spec = load_agent_spec(spec_path)
            except Exception as exc:
                logger.warning(
                    "Variant '%s': failed to load agent spec '%s', using default: %s",
                    variant.name, variant.agent_spec, exc,
                )

        agent_class = self._get_agent_class_from_cfg(chat_cfg)
        default_provider = variant.provider or chat_cfg.get("default_provider")
        default_model = variant.model or chat_cfg.get("default_model")
        prompt_overrides = chat_cfg.get("prompts", {})

        variant_archi = archi(
            pipeline=agent_class,
            agent_spec=variant_agent_spec,
            default_provider=default_provider,
            default_model=default_model,
            prompt_overrides=prompt_overrides,
        )

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

    def _prepare_chat_context(
        self,
        message: List[str],
        conversation_id: int | None,
        client_id: str,
        is_refresh: bool,
        server_received_msg_ts: datetime,
        client_sent_msg_ts: float,
        client_timeout: float,
        timestamps: Dict[str, datetime],
    ) -> tuple[Optional[ChatRequestContext], Optional[int]]:
        if not client_id:
            raise ValueError("client_id is required to process chat messages")
        sender, content = tuple(message[0])

        if conversation_id is None:
            conversation_id = self.create_conversation(content, client_id)
            history = []
        else:
            history = self.query_conversation_history(conversation_id, client_id)
            self.update_conversation_timestamp(conversation_id, client_id)

        timestamps["query_convo_history_ts"] = datetime.now()

        if is_refresh:
            while history and history[-1][0] == ARCHI_SENDER:
                _ = history.pop(-1)

        if server_received_msg_ts.timestamp() - client_sent_msg_ts > client_timeout:
            return None, 408

        if not is_refresh:
            history = history + [(sender, content)]

        if len(history) >= QUERY_LIMIT:
            return None, 500

        return (
            ChatRequestContext(
                sender=sender,
                content=content,
                conversation_id=conversation_id,
                history=history,
                is_refresh=is_refresh,
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
            message = "client timeout"
        elif error_code == 403:
            message = "conversation not found"
        else:
            message = "server error; see chat logs for message"
        return {"type": "error", "status": error_code, "message": message}

    @staticmethod
    def _ab_comparison_from_row(row) -> Dict[str, Any]:
        """Convert a positional AB comparison DB row to a dict."""
        return {
            'comparison_id': row[0],
            'conversation_id': row[1],
            'user_prompt_mid': row[2],
            'response_a_mid': row[3],
            'response_b_mid': row[4],
            'model_a': row[5],
            'pipeline_a': row[6],
            'model_b': row[7],
            'pipeline_b': row[8],
            'variant_a_name': row[9],
            'variant_b_name': row[10],
            'variant_a_meta': row[11],
            'variant_b_meta': row[12],
            'is_config_a_first': row[13],
            'preference': row[14],
            'preference_ts': row[15].isoformat() if row[15] else None,
            'created_at': row[16].isoformat() if row[16] else None,
        }

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
        conversation_id: int | None,
        client_id: str,
        is_refresh: bool,
        server_received_msg_ts: datetime,
        client_sent_msg_ts: float,
        client_timeout: float,
        config_name: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream a champion/challenger A/B comparison.

        Yields interleaved NDJSON events tagged with ``arm: 'a'`` or ``arm: 'b'``
        in real-time as each arm's pipeline produces output.
        A final ``ab_meta`` event carries the comparison_id and variant mapping.
        """
        import queue
        import threading

        if not self.ab_pool:
            yield {"type": "error", "message": "A/B pool not configured"}
            return

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
        )
        if error_code is not None:
            yield self._error_event(error_code)
            return

        # Build variant archis
        try:
            archi_a = self._create_variant_archi(arm_a_variant)
            archi_b = self._create_variant_archi(arm_b_variant)
        except Exception as exc:
            logger.error("Failed to create variant pipelines: %s", exc)
            yield {"type": "error", "message": f"Failed to initialise A/B variants: {exc}"}
            return

        # Shared queue for real-time interleaving
        event_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()

        # Track final text per arm (mutated by threads)
        arm_results = {
            "a": {"final_text": "", "error": None},
            "b": {"final_text": "", "error": None},
        }

        def _stream_arm(arm_archi, arm_label):
            """Run one arm's stream in a thread, pushing events to the shared queue."""
            import time as _time
            from src.interfaces.chat_app.event_formatter import PipelineEventFormatter
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
                        event_queue.put(event)
            except Exception as exc:
                arm_results[arm_label]["error"] = str(exc)
                event_queue.put({"type": "error", "arm": arm_label, "message": str(exc)})
            finally:
                logger.info(
                    "A/B arm '%s' finished (t+%.1fs)",
                    arm_label, _time.monotonic() - t0,
                )
                event_queue.put(_SENTINEL)

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
                conn = psycopg2.connect(**self.pg_config)
                cursor = conn.cursor()
                insert_tups = [
                    ("chat", context.conversation_id, context.sender, context.content,
                     "", "", datetime.now(), None, None),
                ]
                psycopg2.extras.execute_values(cursor, SQL_INSERT_CONVO, insert_tups)
                row = cursor.fetchone()
                user_prompt_mid = row[0] if row else None
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as exc:
                logger.error("Failed to store user message: %s", exc)

        # Store both responses as messages
        arm_a_mid = self._store_assistant_message(
            context.conversation_id,
            arm_a_final_text,
            model_used=f"{arm_a_variant.provider or ''}/{arm_a_variant.model or ''}".strip("/"),
            pipeline_used=self.current_pipeline_used,
        )
        arm_b_mid = self._store_assistant_message(
            context.conversation_id,
            arm_b_final_text,
            model_used=f"{arm_b_variant.provider or ''}/{arm_b_variant.model or ''}".strip("/"),
            pipeline_used=self.current_pipeline_used,
        )

        # Get user prompt message ID if not already stored above
        if not user_prompt_mid:
            user_prompt_mid = self._get_last_user_message_id(context.conversation_id)

        # Create comparison record (skip if we have no valid message IDs)
        comparison_id = None
        if user_prompt_mid and arm_a_mid and arm_b_mid:
            try:
                from src.utils.conversation_service import ConversationService
                conv_service = ConversationService(connection_params=self.pg_config)
                comparison_id = conv_service.create_ab_comparison(
                    conversation_id=context.conversation_id,
                    user_prompt_mid=user_prompt_mid,
                    response_a_mid=arm_a_mid,
                    response_b_mid=arm_b_mid,
                    model_a=f"{arm_a_variant.provider or ''}/{arm_a_variant.model or ''}".strip("/"),
                    pipeline_a=self.current_pipeline_used or "",
                    model_b=f"{arm_b_variant.provider or ''}/{arm_b_variant.model or ''}".strip("/"),
                    pipeline_b=self.current_pipeline_used or "",
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
            "is_champion_first": is_champion_first,
            "arm_a_message_id": arm_a_mid,
            "arm_b_message_id": arm_b_mid,
        }

    def _store_assistant_message(self, conversation_id, content, model_used=None, pipeline_used=None):
        """Store an assistant message and return the message_id."""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            insert_tups = [
                ("chat", conversation_id, "archi", content, "", "", datetime.now(), model_used, pipeline_used),
            ]
            psycopg2.extras.execute_values(cursor, SQL_INSERT_CONVO, insert_tups)
            row = cursor.fetchone()
            mid = row[0] if row else None
            conn.commit()
            cursor.close()
            conn.close()
            return mid
        except Exception as exc:
            logger.error("Failed to store assistant message: %s", exc)
            return None

    def _get_last_user_message_id(self, conversation_id):
        """Get the most recent user message_id for a conversation."""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message_id FROM conversations WHERE conversation_id = %s AND sender = 'User' ORDER BY ts DESC LIMIT 1",
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
        
        # Use markdown links for client-side rendering, HTML for server-side
        if render_markdown:
            output += self.format_links(top_sources)
        else:
            output += self.format_links_markdown(top_sources)

        timestamps["archi_message_ts"] = datetime.now()
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
            context.is_refresh,
        )
        timestamps["insert_convo_ts"] = datetime.now()
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

    def __call__(self, message: List[str], conversation_id: int|None, client_id: str, is_refresh: bool, server_received_msg_ts: datetime,  client_sent_msg_ts: float, client_timeout: float, config_name: str):
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
            )
            if error_code is not None:
                return None, None, None, timestamps, error_code

            requested_config = self._resolve_config_name(config_name)
            self.update_config(config_name=requested_config)

            result = self.archi(history=context.history, conversation_id=context.conversation_id)
            timestamps["chain_finished_ts"] = datetime.now()

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

        timestamps['finish_call_ts'] = datetime.now()

        return output, context.conversation_id if context else None, message_ids, timestamps, None

    def stream(
        self,
        message: List[str],
        conversation_id: int | None,
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
    ) -> Iterator[Dict[str, Any]]:
        timestamps = self._init_timestamps()
        context = None
        last_output = None
        from src.interfaces.chat_app.event_formatter import PipelineEventFormatter
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
            )
            if error_code is not None:
                yield self._error_event(error_code)
                return

            requested_config = self._resolve_config_name(config_name)
            self.update_config(config_name=requested_config)
            
            # If provider and model are specified, override the pipeline's LLM
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

            for output in self.archi.stream(history=context.history, conversation_id=context.conversation_id):
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
                    yield {"type": "error", "status": 408, "message": "client timeout"}
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

            timestamps["chain_finished_ts"] = datetime.now()

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

            timestamps["finish_call_ts"] = datetime.now()
            timestamps["server_received_msg_ts"] = server_received_msg_ts
            timestamps["client_sent_msg_ts"] = datetime.fromtimestamp(client_sent_msg_ts)
            timestamps["server_response_msg_ts"] = datetime.now()

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
                "final_response_msg_ts": datetime.now().timestamp(),
                "usage": usage,
                "model": model,
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
        self.salt = read_secret("UPLOADER_SALT")
        secret_key = read_secret("FLASK_UPLOADER_APP_SECRET_KEY")
        if not secret_key:
            logger.warning("FLASK_UPLOADER_APP_SECRET_KEY not found, generating a random secret key")
            import secrets
            secret_key = secrets.token_hex(32)
        self.app.secret_key = secret_key
        
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
        self.add_endpoint('/api/update_config', 'update_config', self.require_auth(self.update_config), methods=["POST"])
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
        self.add_endpoint('/api/ab/compare', 'ab_compare', self.require_auth(self.ab_compare_stream), methods=["POST"])
        self.add_endpoint('/api/ab/metrics', 'ab_metrics', self.require_auth(self.ab_get_metrics), methods=["GET"])

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
        logger.info("Adding data viewer API endpoints")
        self.add_endpoint('/data', 'data_viewer', self.require_auth(self.data_viewer_page))
        self.add_endpoint('/api/data/documents', 'list_data_documents', self.require_auth(self.list_data_documents), methods=["GET"])
        self.add_endpoint('/api/data/documents/<document_hash>/content', 'get_data_document_content', self.require_auth(self.get_data_document_content), methods=["GET"])
        self.add_endpoint('/api/data/documents/<document_hash>/chunks', 'get_data_document_chunks', self.require_auth(self.get_data_document_chunks), methods=["GET"])
        self.add_endpoint('/api/data/documents/<document_hash>/enable', 'enable_data_document', self.require_auth(self.enable_data_document), methods=["POST"])
        self.add_endpoint('/api/data/documents/<document_hash>/disable', 'disable_data_document', self.require_auth(self.disable_data_document), methods=["POST"])
        self.add_endpoint('/api/data/bulk-enable', 'bulk_enable_documents', self.require_auth(self.bulk_enable_documents), methods=["POST"])
        self.add_endpoint('/api/data/bulk-disable', 'bulk_disable_documents', self.require_auth(self.bulk_disable_documents), methods=["POST"])
        self.add_endpoint('/api/data/stats', 'get_data_stats', self.require_auth(self.get_data_stats), methods=["GET"])

        # Data uploader endpoints
        logger.info("Adding data uploader API endpoints")
        self.add_endpoint('/upload', 'upload_page', self.require_auth(self.upload_page))
        self.add_endpoint('/api/upload/file', 'upload_file', self.require_auth(self.upload_file), methods=["POST"])
        self.add_endpoint('/api/upload/url', 'upload_url', self.require_auth(self.upload_url), methods=["POST"])
        self.add_endpoint('/api/upload/git', 'upload_git', self.require_auth(self.upload_git), methods=["POST", "DELETE"])
        self.add_endpoint('/api/upload/git/refresh', 'refresh_git', self.require_auth(self.refresh_git), methods=["POST"])
        self.add_endpoint('/api/upload/jira', 'upload_jira', self.require_auth(self.upload_jira), methods=["POST"])
        self.add_endpoint('/api/upload/embed', 'trigger_embedding', self.require_auth(self.trigger_embedding), methods=["POST"])
        self.add_endpoint('/api/upload/status', 'get_embedding_status', self.require_auth(self.get_embedding_status), methods=["GET"])
        self.add_endpoint('/api/upload/documents', 'list_upload_documents', self.require_auth(self.list_upload_documents), methods=["GET"])
        self.add_endpoint('/api/upload/documents/grouped', 'list_upload_documents_grouped', self.require_auth(self.list_upload_documents_grouped), methods=["GET"])
        self.add_endpoint('/api/upload/documents/<document_hash>/retry', 'retry_document', self.require_auth(self.retry_document), methods=["POST"])
        self.add_endpoint('/api/upload/documents/retry-all-failed', 'retry_all_failed', self.require_auth(self.retry_all_failed), methods=["POST"])
        self.add_endpoint('/api/sources/git', 'list_git_sources', self.require_auth(self.list_git_sources), methods=["GET"])
        self.add_endpoint('/api/sources/jira', 'list_jira_sources', self.require_auth(self.list_jira_sources), methods=["GET", "DELETE"])
        self.add_endpoint('/api/sources/schedules', 'source_schedules', self.require_auth(self.source_schedules_dispatch), methods=["GET", "PUT"])

        # Database viewer endpoints (admin only)
        logger.info("Adding database viewer API endpoints")
        self.add_endpoint('/admin/database', 'database_viewer_page', self.require_auth(self.database_viewer_page))
        self.add_endpoint('/api/admin/database/tables', 'list_database_tables', self.require_auth(self.list_database_tables), methods=["GET"])
        self.add_endpoint('/api/admin/database/query', 'run_database_query', self.require_auth(self.run_database_query), methods=["POST"])

        # add unified auth endpoints
        if self.auth_enabled:
            logger.info("Adding unified authentication endpoints")
            self.add_endpoint('/login', 'login', self.login, methods=['GET', 'POST'])
            self.add_endpoint('/logout', 'logout', self.logout)
            self.add_endpoint('/auth/user', 'get_user', self.get_user, methods=['GET'])
            
            if self.sso_enabled:
                self.add_endpoint('/redirect', 'sso_callback', self.sso_callback)

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
                session['user'] = {
                    'email': username,
                    'name': username,
                    'username': username
                }
                session['logged_in'] = True
                session['auth_method'] = 'basic'
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
        session.pop('user', None)
        session.pop('logged_in', None)
        session.pop('auth_method', None)
        
        logger.info(f"User logged out (method: {auth_method})")
        flash('You have been logged out successfully')
        return redirect(url_for('landing'))

    def sso_callback(self):
        """Handle OAuth callback from SSO provider"""
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
            
            # Store user information in session (normalized structure)
            session['user'] = {
                'email': user_info.get('email', ''),
                'name': user_info.get('name', user_info.get('preferred_username', '')),
                'username': user_info.get('preferred_username', user_info.get('email', '')),
                'id': user_info.get('sub', '')
            }
            session['logged_in'] = True
            session['auth_method'] = 'sso'
            
            logger.info(f"SSO login successful for user: {user_info.get('email')}")
            
            # Redirect to main page
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"SSO callback error: {str(e)}")
            flash(f"Authentication failed: {str(e)}")
            return redirect(url_for('login'))

    def get_user(self):
        """API endpoint to get current user information"""
        if session.get('logged_in'):
            user = session.get('user', {})
            return jsonify({
                'logged_in': True,
                'email': user.get('email', ''),
                'name': user.get('name', ''),
                'auth_method': session.get('auth_method', 'unknown'),
                'auth_enabled': self.auth_enabled
            })
        return jsonify({
            'logged_in': False,
            'auth_enabled': self.auth_enabled
        })

    def require_auth(self, f):
        """Decorator to require authentication for routes"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not self.auth_enabled:
                # If auth is not enabled, allow access
                return f(*args, **kwargs)
            
            if not session.get('logged_in'):
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Unauthorized', 'message': 'Authentication required'}), 401
                else:   
                    return redirect(url_for('login'))
            
            return f(*args, **kwargs)
        return decorated_function

    def health(self):
        return jsonify({"status": "OK"}), 200

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
        return jsonify({'options': options}), 200

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
        return Path(agents_dir)

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
        except Exception as exc:
            logger.warning("Failed to import pipelines module: %s", exc)
            return []
        agent_cls = getattr(pipelines, agent_class, None)
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
        except Exception as exc:
            logger.warning("Failed to import pipelines module: %s", exc)
            return []
        agent_cls = getattr(pipelines, agent_class, None)
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

    def list_agents(self):
        """
        List available agent specs for the dropdown.
        """
        try:
            agents_dir = self._get_agents_dir()
            agent_files = list_agent_files(agents_dir)
            agents = []
            for path in agent_files:
                try:
                    spec = load_agent_spec(path)
                    agents.append({"name": spec.name, "filename": path.name})
                except AgentSpecError as exc:
                    logger.warning("Skipping invalid agent spec %s: %s", path, exc)
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
            }), 200
        except Exception as exc:
            logger.error(f"Error listing agents: {exc}")
            return jsonify({"error": str(exc)}), 500

    def get_agent_spec(self):
        """
        Fetch a single agent spec by name.
        """
        try:
            name = request.args.get("name")
            if not name:
                return jsonify({"error": "name parameter required"}), 400
            agents_dir = self._get_agents_dir()
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
                    }), 200
            return jsonify({"error": f"Agent '{name}' not found"}), 404
        except Exception as exc:
            logger.error(f"Error fetching agent spec: {exc}")
            return jsonify({"error": str(exc)}), 500

    def get_agent_template(self):
        """
        Return a prefilled agent spec template and available tools.
        """
        try:
            agent_name = request.args.get("name") or "New Agent"
            tool_items = self._get_agent_tools()
            tools = [tool["name"] for tool in tool_items]
            return jsonify({
                "name": agent_name,
                "tools": tool_items,
                "template": self._build_agent_template(agent_name, tools),
            }), 200
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
            content = data.get("content")
            mode = data.get("mode", "create")
            existing_name = data.get("existing_name")
            if not content or not isinstance(content, str):
                return jsonify({'error': 'Content is required'}), 400

            agents_dir = self._get_agents_dir()
            agents_dir.mkdir(parents=True, exist_ok=True)

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
                try:
                    dynamic = get_dynamic_config()
                except Exception:
                    dynamic = None
                if dynamic and dynamic.active_agent_name == existing_name and new_spec.name != existing_name:
                    cfg = ConfigService(pg_config=self.pg_config)
                    cfg.update_dynamic_config(active_agent_name=new_spec.name, updated_by=data.get("client_id") or "system")
                return jsonify({
                    'success': True,
                    'name': new_spec.name,
                    'filename': target_path.name,
                    'path': str(target_path),
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
            }), 200
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
            name = data.get("name")
            if not name:
                return jsonify({"error": "name is required"}), 400
            name = name.strip()
            if name.lower().startswith("name:"):
                name = name.split(":", 1)[1].strip()

            agents_dir = self._get_agents_dir()
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
            try:
                dynamic = get_dynamic_config()
            except Exception:
                dynamic = None
            if dynamic and dynamic.active_agent_name == name:
                cfg = ConfigService(pg_config=self.pg_config)
                cfg.update_dynamic_config(active_agent_name=None, updated_by=data.get("client_id") or "system")
            return jsonify({"success": True, "deleted": name}), 200
        except Exception as exc:
            logger.error(f"Error deleting agent spec: {exc}")
            return jsonify({"error": str(exc)}), 500

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
                # Skip local provider - no API key needed
                if provider_type == ProviderType.LOCAL:
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
        server_received_msg_ts = datetime.now()

        # get user input and conversation_id from the request
        request_data = self._parse_chat_request()
        message = request_data["message"]
        conversation_id = request_data["conversation_id"]
        config_name = request_data["config_name"]
        is_refresh = request_data["is_refresh"]
        client_sent_msg_ts = request_data["client_sent_msg_ts"]
        client_timeout = request_data["client_timeout"]
        client_id = request_data["client_id"]

        if not client_id:
            return jsonify({'error': 'client_id missing'}), 400

        # query the chat and return the results.
        logger.debug("Calling the ChatWrapper()")
        response, conversation_id, message_ids, timestamps, error_code = self.chat(message, conversation_id, client_id, is_refresh, server_received_msg_ts, client_sent_msg_ts, client_timeout,config_name)

        # handle errors
        if error_code is not None:
            err = ChatWrapper._error_event(error_code)
            return jsonify({'error': err['message']}), error_code

        # compute timestamp at which message was returned to client
        timestamps['server_response_msg_ts'] = datetime.now()

        # store timing info for this message
        timestamps['server_received_msg_ts'] = server_received_msg_ts
        timestamps['client_sent_msg_ts'] = datetime.fromtimestamp(client_sent_msg_ts)
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
            'final_response_msg_ts': datetime.now().timestamp(),
        }

        end_time = time.time()
        logger.info(f"API Response Time: {end_time - start_time:.2f} seconds")

        return jsonify(response_data)

    def get_chat_response_stream(self):
        """
        Streams agent updates and the final response as NDJSON.
        """
        server_received_msg_ts = datetime.now()
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

        # Get API key from session if available
        session_api_key = None
        if provider and 'provider_api_keys' in session:
            session_api_key = session.get('provider_api_keys', {}).get(provider.lower())

        return self._ndjson_response(self.chat.stream(
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
        ))

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
        return render_template('index.html')

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

            label = reaction_type.capitalize() + ('d' if reaction_type == 'like' else 'd')
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
                "feedback_ts"  : datetime.now(),
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
            if not client_id:
                return jsonify({'error': 'client_id missing'}), 400
            limit = min(int(request.args.get('limit', 50)), 500)

            # create connection to database
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
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
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            client_id = data.get('client_id')

            if not conversation_id:
                return jsonify({'error': 'conversation_id missing'}), 400
            if not client_id:
                return jsonify({'error': 'client_id missing'}), 400

            # create connection to database
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()

            # get conversation metadata
            cursor.execute(SQL_GET_CONVERSATION_METADATA, (conversation_id, client_id))
            meta_row = cursor.fetchone()

            # if no metadata found, return error
            if not meta_row:
                cursor.close()
                conn.close()
                return jsonify({'error': 'conversation not found'}), 404

            # get history of the conversation along with latest feedback state
            cursor.execute(SQL_QUERY_CONVO_WITH_FEEDBACK, (conversation_id, ))
            history_rows = cursor.fetchall()
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

            conversation = {
                'conversation_id': meta_row[0],
                'title': meta_row[1] or "New Conversation",
                'created_at': meta_row[2].isoformat() if meta_row[2] else None,
                'last_message_at': meta_row[3].isoformat() if meta_row[3] else None,
                'messages': messages
            }

            # clean up database connection state
            cursor.close()
            conn.close()

            return jsonify(conversation), 200

        except Exception as e:
            logger.error(f"Error in load_conversation: {str(e)}")
            return jsonify({'error': str(e)}), 500

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

            if not conversation_id:
                return jsonify({'error': 'conversation_id missing when deleting.'}), 400
            if not client_id:
                return jsonify({'error': 'client_id missing when deleting.'}), 400

            # create connection to database
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()

            # Delete conversation metadata (SQL CASCADE will delete all child messages)
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

            if not comparison_id:
                return jsonify({'error': 'comparison_id is required'}), 400
            if not preference:
                return jsonify({'error': 'preference is required'}), 400
            if preference not in ('a', 'b', 'tie'):
                return jsonify({'error': 'preference must be "a", "b", or "tie"'}), 400
            if not client_id:
                return jsonify({'error': 'client_id is required'}), 400

            # Update the preference
            self.chat.update_ab_preference(comparison_id, preference)

            # Update variant-level aggregate metrics (pool-based comparisons only)
            try:
                from src.utils.conversation_service import ConversationService
                comparison = self.chat.get_ab_comparison(comparison_id)
                if comparison:
                    va = comparison.get('variant_a_name')
                    vb = comparison.get('variant_b_name')
                    if va and vb:
                        conv_service = ConversationService(connection_params=self.chat.pg_config)
                        conv_service.update_variant_metrics_for_preference(va, vb, preference)
                        logger.info(f"Updated variant metrics for {va} vs {vb}, preference={preference}")
            except Exception as metrics_err:
                # Metrics are best-effort; don't fail the preference submission
                logger.warning(f"Failed to update variant metrics: {metrics_err}")

            return jsonify({
                'success': True,
                'comparison_id': comparison_id,
                'preference': preference,
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

            if not conversation_id:
                return jsonify({'error': 'conversation_id is required'}), 400
            if not client_id:
                return jsonify({'error': 'client_id is required'}), 400

            comparison = self.chat.get_pending_ab_comparison(conversation_id)

            return jsonify({
                'success': True,
                'comparison': comparison,
            }), 200

        except Exception as e:
            logger.error(f"Error getting pending A/B comparison: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def ab_get_pool(self):
        """
        Get A/B testing pool configuration.

        Returns:
            JSON with pool info (enabled, champion, variant names) or enabled=false.
        """
        try:
            pool = self.chat.ab_pool
            if pool:
                return jsonify({'success': True, **pool.pool_info()}), 200
            else:
                return jsonify({'success': True, 'enabled': False}), 200
        except Exception as e:
            logger.error(f"Error getting A/B pool: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def ab_compare_stream(self):
        """
        Stream a pool-based A/B comparison (champion vs challenger).

        POST body:
        - message: [sender, content] pair
        - conversation_id: The conversation ID (optional)
        - client_id: Client ID for authorization
        - config_name: Config name (optional)

        Returns:
            NDJSON stream with arm-tagged events.
        """
        server_received_msg_ts = datetime.now()
        request_data = self._parse_chat_request()

        message = request_data["message"]
        conversation_id = request_data["conversation_id"]
        config_name = request_data["config_name"]
        is_refresh = request_data["is_refresh"]
        client_sent_msg_ts = request_data["client_sent_msg_ts"]
        client_timeout = request_data["client_timeout"]
        client_id = request_data["client_id"]

        if not client_id:
            return jsonify({"error": "client_id missing"}), 400

        return self._ndjson_response(self.chat.stream_ab_comparison(
            message,
            conversation_id,
            client_id,
            is_refresh,
            server_received_msg_ts,
            client_sent_msg_ts,
            client_timeout,
            config_name,
        ))

    def ab_get_metrics(self):
        """
        Get per-variant A/B testing metrics.

        Returns:
            JSON with variant metrics (wins, losses, ties, total).
        """
        try:
            from src.utils.conversation_service import ConversationService
            conv_service = ConversationService(connection_params=self.chat.pg_config)
            metrics = conv_service.get_all_variant_metrics()
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
        return render_template('data.html')

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

            if not url:
                return jsonify({"error": "missing_url"}), 400

            # Proxy to data-manager service
            resp = requests.post(
                f"{self.data_manager_url}/document_index/upload_url",
                data={"url": url},
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
                    "resources_scraped": 1
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

            result = self.chat.data_viewer.list_documents(source_type='jira', limit=1000)
            for doc in result.get('documents', []):
                # Parse project key from display name or URL
                display_name = doc.get('display_name', '')
                # Jira documents often have display_name like "PROJECT-123: Title"
                if display_name:
                    project_key = display_name.split('-')[0] if '-' in display_name else display_name
                    if project_key and project_key not in seen_projects:
                        seen_projects.add(project_key)
                        sources.append({
                            'project_key': project_key,
                            'name': project_key,
                        })

            return jsonify({"sources": sources}), 200

        except Exception as e:
            logger.error(f"Error listing Jira sources: {str(e)}")
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
            
            # Convert cron expressions to UI-friendly values
            schedule_display = {}
            cron_to_ui = {
                '': 'disabled',
                '0 * * * *': 'hourly',
                '0 */6 * * *': 'every_6h',
                '0 0 * * *': 'daily',
            }
            
            for source, cron in schedules.items():
                schedule_display[source] = {
                    'cron': cron,
                    'display': cron_to_ui.get(cron, 'custom'),
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
