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
    SQL_INSERT_AB_COMPARISON, SQL_UPDATE_AB_PREFERENCE, SQL_GET_AB_COMPARISON,
    SQL_GET_PENDING_AB_COMPARISON, SQL_DELETE_AB_COMPARISON, SQL_GET_AB_COMPARISONS_BY_CONVERSATION,
    SQL_CREATE_AGENT_TRACE, SQL_UPDATE_AGENT_TRACE, SQL_GET_AGENT_TRACE,
    SQL_GET_TRACE_BY_MESSAGE, SQL_GET_ACTIVE_TRACE, SQL_CANCEL_ACTIVE_TRACES,
)
from src.interfaces.chat_app.document_utils import *
from src.interfaces.chat_app.utils import collapse_assistant_sequences


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

        agent_class = chat_cfg.get("agent_class") or chat_cfg.get("pipeline")
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

        agent_class = chat_cfg.get("agent_class") or chat_cfg.get("pipeline")
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
        score = entry["score"]
        link = entry["link"]
        display_name = entry["display"]

        if score == -1.0 or score == "N/A":
            score_str = ""
        else:
            score_str = f" ({score:.2f})"

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
            score = entry["score"]
            link = entry["link"]
            display_name = entry["display"]

            if score == -1.0 or score == "N/A":
                score_str = ""
            else:
                score_str = f"({score:.2f})"

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

    def create_ab_comparison(
        self,
        conversation_id: int,
        user_prompt_mid: int,
        response_a_mid: int,
        response_b_mid: int,
        config_a_id: int,
        config_b_id: int,
        is_config_a_first: bool,
    ) -> int:
        """
        Create an A/B comparison record linking two responses to the same user prompt.
        
        Args:
            conversation_id: The conversation this comparison belongs to
            user_prompt_mid: Message ID of the user's question
            response_a_mid: Message ID of response A
            response_b_mid: Message ID of response B
            config_a_id: Config ID used for response A
            config_b_id: Config ID used for response B
            is_config_a_first: True if config A was the "first" config before randomization
            
        Returns:
            The comparison_id of the newly created record
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(
                SQL_INSERT_AB_COMPARISON,
                (conversation_id, user_prompt_mid, response_a_mid, response_b_mid,
                 config_a_id, config_b_id, is_config_a_first)
            )
            comparison_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Created A/B comparison {comparison_id} for conversation {conversation_id}")
            return comparison_id
        finally:
            cursor.close()
            conn.close()

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
            return {
                'comparison_id': row[0],
                'conversation_id': row[1],
                'user_prompt_mid': row[2],
                'response_a_mid': row[3],
                'response_b_mid': row[4],
                'config_a_id': row[5],
                'config_b_id': row[6],
                'is_config_a_first': row[7],
                'preference': row[8],
                'preference_ts': row[9].isoformat() if row[9] else None,
                'created_at': row[10].isoformat() if row[10] else None,
            }
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
            return {
                'comparison_id': row[0],
                'conversation_id': row[1],
                'user_prompt_mid': row[2],
                'response_a_mid': row[3],
                'response_b_mid': row[4],
                'config_a_id': row[5],
                'config_b_id': row[6],
                'is_config_a_first': row[7],
                'preference': row[8],
                'preference_ts': row[9].isoformat() if row[9] else None,
                'created_at': row[10].isoformat() if row[10] else None,
            }
        finally:
            cursor.close()
            conn.close()

    def delete_ab_comparison(self, comparison_id: int) -> bool:
        """
        Delete an A/B comparison (e.g., on abort/failure).
        
        Returns:
            True if a record was deleted, False otherwise
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_DELETE_AB_COMPARISON, (comparison_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            if deleted:
                logger.info(f"Deleted A/B comparison {comparison_id}")
            return deleted
        finally:
            cursor.close()
            conn.close()

    def get_ab_comparisons_by_conversation(self, conversation_id: int) -> List[Dict[str, Any]]:
        """
        Get all A/B comparisons for a conversation.
        
        Returns:
            List of comparison dicts
        """
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_GET_AB_COMPARISONS_BY_CONVERSATION, (conversation_id,))
            rows = cursor.fetchall()
            return [
                {
                    'comparison_id': row[0],
                    'conversation_id': row[1],
                    'user_prompt_mid': row[2],
                    'response_a_mid': row[3],
                    'response_b_mid': row[4],
                    'config_a_id': row[5],
                    'config_b_id': row[6],
                    'is_config_a_first': row[7],
                    'preference': row[8],
                    'preference_ts': row[9].isoformat() if row[9] else None,
                    'created_at': row[10].isoformat() if row[10] else None,
                }
                for row in rows
            ]
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
            return {
                'trace_id': row[0],
                'conversation_id': row[1],
                'message_id': row[2],
                'user_message_id': row[3],
                'config_id': row[4],
                'pipeline_name': row[5],
                'events': row[6],  # Already JSON from JSONB
                'started_at': row[7].isoformat() if row[7] else None,
                'completed_at': row[8].isoformat() if row[8] else None,
                'status': row[9],
                'total_tool_calls': row[10],
                'total_tokens_used': row[11],
                'total_duration_ms': row[12],
                'cancelled_by': row[13],
                'cancellation_reason': row[14],
                'created_at': row[15].isoformat() if row[15] else None,
            }
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
            return {
                'trace_id': row[0],
                'conversation_id': row[1],
                'message_id': row[2],
                'user_message_id': row[3],
                'config_id': row[4],
                'pipeline_name': row[5],
                'events': row[6],
                'started_at': row[7].isoformat() if row[7] else None,
                'completed_at': row[8].isoformat() if row[8] else None,
                'status': row[9],
                'total_tool_calls': row[10],
                'total_tokens_used': row[11],
                'total_duration_ms': row[12],
                'cancelled_by': row[13],
                'cancellation_reason': row[14],
                'created_at': row[15].isoformat() if row[15] else None,
            }
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
            return {
                'trace_id': row[0],
                'conversation_id': row[1],
                'message_id': row[2],
                'user_message_id': row[3],
                'config_id': row[4],
                'pipeline_name': row[5],
                'events': row[6],
                'started_at': row[7].isoformat() if row[7] else None,
                'status': row[8],
            }
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
        last_streamed_text = ""
        trace_id = None
        trace_events: List[Dict[str, Any]] = []
        tool_call_count = 0
        stream_start_time = time.time()
        emitted_tool_call_ids = set()
        emitted_tool_start_ids = set()
        pending_tool_call_ids: List[str] = []
        tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
        synthetic_tool_counter = 0

        def _next_tool_call_id(tool_name: str) -> str:
            nonlocal synthetic_tool_counter
            synthetic_tool_counter += 1
            safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", (tool_name or "unknown")).strip("_") or "unknown"
            return f"synthetic_tool_{synthetic_tool_counter}_{safe_name}"

        def _is_empty_tool_args(tool_args: Any) -> bool:
            return tool_args in (None, "", {}, [])

        def _has_meaningful_tool_payload(tool_name: Any, tool_args: Any) -> bool:
            if isinstance(tool_name, str) and tool_name.strip() and tool_name.strip().lower() != "unknown":
                return True
            return not _is_empty_tool_args(tool_args)

        def _remember_tool_call(tool_call_id: str, tool_name: Any, tool_args: Any) -> None:
            if not tool_call_id:
                return
            current = tool_calls_by_id.get(tool_call_id, {})
            current_name = current.get("tool_name", "unknown")
            current_args = current.get("tool_args", {})
            merged_name = (
                tool_name
                if isinstance(tool_name, str)
                and tool_name.strip()
                and tool_name.strip().lower() != "unknown"
                else current_name
            )
            merged_args = tool_args if not _is_empty_tool_args(tool_args) else current_args
            tool_calls_by_id[tool_call_id] = {
                "tool_name": merged_name or "unknown",
                "tool_args": merged_args,
            }

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
                error_message = "server error; see chat logs for message"
                if error_code == 408:
                    error_message = "client timeout"
                elif error_code == 403:
                    error_message = "conversation not found"
                yield {"type": "error", "status": error_code, "message": error_message}
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
                
                # Extract event_type from metadata (new structured events from BaseReActAgent)
                event_type = output.metadata.get("event_type", "text") if output.metadata else "text"
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # Handle different event types
                if event_type == "tool_start":
                    tool_messages = getattr(output, "messages", []) or []
                    tool_message = tool_messages[0] if tool_messages else None
                    tool_calls = getattr(tool_message, "tool_calls", None) if tool_message else None
                    memory_args_by_id = {}
                    if output.metadata:
                        memory_args_by_id = output.metadata.get("tool_inputs_by_id", {}) or {}
                    raw_args_by_id: Dict[str, Any] = {}
                    raw_name_by_id: Dict[str, str] = {}
                    if tool_message is not None:
                        try:
                            additional = getattr(tool_message, "additional_kwargs", {}) or {}
                            raw_tool_calls = additional.get("tool_calls") or []
                            for raw_call in raw_tool_calls:
                                if not isinstance(raw_call, dict):
                                    continue
                                raw_id = raw_call.get("id")
                                function_obj = raw_call.get("function") or {}
                                raw_name = function_obj.get("name")
                                raw_arguments = function_obj.get("arguments")
                                parsed_args: Any = None
                                if isinstance(raw_arguments, str) and raw_arguments.strip():
                                    try:
                                        parsed_args = json.loads(raw_arguments)
                                    except Exception:
                                        parsed_args = {"_raw_arguments": raw_arguments}
                                elif isinstance(raw_arguments, dict):
                                    parsed_args = raw_arguments
                                if raw_id and parsed_args is not None:
                                    raw_args_by_id[raw_id] = parsed_args
                                if raw_id and isinstance(raw_name, str) and raw_name.strip():
                                    raw_name_by_id[raw_id] = raw_name.strip()

                            # Newer OpenAI/LangChain payloads may carry partial tool calls here.
                            for chunk in getattr(tool_message, "tool_call_chunks", []) or []:
                                if not isinstance(chunk, dict):
                                    continue
                                chunk_id = chunk.get("id")
                                chunk_name = chunk.get("name")
                                chunk_args = chunk.get("args")
                                parsed_chunk_args: Any = None
                                if isinstance(chunk_args, str) and chunk_args.strip():
                                    try:
                                        parsed_chunk_args = json.loads(chunk_args)
                                    except Exception:
                                        parsed_chunk_args = {"_raw_arguments": chunk_args}
                                elif isinstance(chunk_args, dict):
                                    parsed_chunk_args = chunk_args
                                if chunk_id and parsed_chunk_args is not None:
                                    raw_args_by_id[chunk_id] = parsed_chunk_args
                                if chunk_id and isinstance(chunk_name, str) and chunk_name.strip():
                                    raw_name_by_id[chunk_id] = chunk_name.strip()
                        except Exception:
                            pass
                    if tool_calls:
                        for tool_call in tool_calls:
                            tool_call_id = tool_call.get("id", "")
                            tool_args = tool_call.get("args", {})
                            if _is_empty_tool_args(tool_args):
                                tool_args = raw_args_by_id.get(tool_call_id, tool_args)
                            if _is_empty_tool_args(tool_args):
                                fallback = memory_args_by_id.get(tool_call_id, {})
                                if isinstance(fallback, dict):
                                    tool_args = fallback.get("tool_input", tool_args)
                            tool_name = tool_call.get("name", "unknown")
                            if (not tool_name or str(tool_name).strip().lower() == "unknown") and tool_call_id in raw_name_by_id:
                                tool_name = raw_name_by_id[tool_call_id]
                            if (not tool_name) and isinstance(memory_args_by_id.get(tool_call_id), dict):
                                tool_name = memory_args_by_id[tool_call_id].get("tool_name", "unknown")
                            if (not tool_call_id) and (not _has_meaningful_tool_payload(tool_name, tool_args)):
                                continue
                            if not tool_call_id:
                                tool_call_id = _next_tool_call_id(tool_name)
                            _remember_tool_call(tool_call_id, tool_name, tool_args)
                            if tool_call_id in emitted_tool_call_ids:
                                continue
                            emitted_tool_call_ids.add(tool_call_id)
                            pending_tool_call_ids.append(tool_call_id)
                            tool_call_count += 1
                    elif memory_args_by_id:
                        for memory_id, memory_call in memory_args_by_id.items():
                            if not isinstance(memory_call, dict):
                                continue
                            tool_name = memory_call.get("tool_name", "unknown")
                            tool_args = memory_call.get("tool_input", {})
                            if not _has_meaningful_tool_payload(tool_name, tool_args):
                                continue
                            tool_call_id = memory_id or _next_tool_call_id(tool_name)
                            if tool_call_id in emitted_tool_call_ids:
                                continue
                            emitted_tool_call_ids.add(tool_call_id)
                            pending_tool_call_ids.append(tool_call_id)
                            _remember_tool_call(tool_call_id, tool_name, tool_args)
                            tool_call_count += 1
                        
                elif event_type == "tool_output":
                    tool_messages = getattr(output, "messages", []) or []
                    tool_message = tool_messages[0] if tool_messages else None
                    tool_output = self._message_content(tool_message) if tool_message else ""
                    truncated = len(tool_output) > max_step_chars
                    full_length = len(tool_output) if truncated else None
                    display_output = self._truncate_text(tool_output, max_step_chars)
                    
                    output_tool_call_id = getattr(tool_message, "tool_call_id", "") if tool_message else ""
                    if not output_tool_call_id and pending_tool_call_ids:
                        output_tool_call_id = pending_tool_call_ids.pop(0)
                    elif output_tool_call_id in pending_tool_call_ids:
                        pending_tool_call_ids.remove(output_tool_call_id)

                    # Emit tool_start once, immediately before first output for stable ordering.
                    if output_tool_call_id and output_tool_call_id not in emitted_tool_start_ids:
                        memory_args_by_id = output.metadata.get("tool_inputs_by_id", {}) if output.metadata else {}
                        fallback = memory_args_by_id.get(output_tool_call_id, {})
                        fallback_name = "unknown"
                        fallback_args: Any = {}
                        if isinstance(fallback, dict):
                            fallback_name = fallback.get("tool_name", "unknown")
                            fallback_args = fallback.get("tool_input", {})
                        _remember_tool_call(output_tool_call_id, fallback_name, fallback_args)
                        call_info = tool_calls_by_id.get(output_tool_call_id, {})
                        start_event = {
                            "type": "tool_start",
                            "tool_call_id": output_tool_call_id,
                            "tool_name": call_info.get("tool_name", "unknown"),
                            "tool_args": call_info.get("tool_args", {}),
                            "timestamp": timestamp,
                            "conversation_id": context.conversation_id,
                        }
                        trace_events.append(start_event)
                        emitted_tool_start_ids.add(output_tool_call_id)
                        if include_tool_steps:
                            yield start_event

                    trace_event = {
                        "type": "tool_output",
                        "tool_call_id": output_tool_call_id,
                        "output": display_output,
                        "truncated": truncated,
                        "full_length": full_length,
                        "timestamp": timestamp,
                        "conversation_id": context.conversation_id,
                    }
                    trace_events.append(trace_event)
                    if include_tool_steps:
                        yield trace_event
                        
                elif event_type == "tool_end":
                    trace_event = {
                        "type": "tool_end",
                        "tool_call_id": output.metadata.get("tool_call_id", ""),
                        "status": output.metadata.get("status", "success"),
                        "duration_ms": output.metadata.get("duration_ms"),
                        "timestamp": timestamp,
                        "conversation_id": context.conversation_id,
                    }
                    trace_events.append(trace_event)
                    if include_tool_steps:
                        yield trace_event
                        
                elif event_type == "thinking_start":
                    trace_event = {
                        "type": "thinking_start",
                        "step_id": output.metadata.get("step_id", ""),
                        "timestamp": timestamp,
                        "conversation_id": context.conversation_id,
                    }
                    trace_events.append(trace_event)
                    if include_tool_steps:
                        yield trace_event
                        
                elif event_type == "thinking_end":
                    thinking_content = output.metadata.get("thinking_content", "")
                    trace_event = {
                        "type": "thinking_end",
                        "step_id": output.metadata.get("step_id", ""),
                        "duration_ms": output.metadata.get("duration_ms"),
                        "thinking_content": thinking_content,
                        "timestamp": timestamp,
                        "conversation_id": context.conversation_id,
                    }
                    trace_events.append(trace_event)
                    if include_tool_steps:
                        yield trace_event
                        
                elif event_type == "text":
                    # Stream text content
                    content = getattr(output, "answer", "") or ""
                    if content and include_agent_steps:
                        last_streamed_text = content
                        yield {
                            "type": "chunk",
                            "content": content,
                            "accumulated": True,
                            "conversation_id": context.conversation_id,
                        }
                    # Record text event in trace
                    if content:
                        trace_events.append({
                            "type": "text",
                            "content": content,
                            "timestamp": timestamp,
                        })
                        
                elif event_type == "final":
                    # Final event handled below after loop
                    pass
                else:
                    # Fallback: legacy event handling for non-agent pipelines
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

                # For providers like gpt-5, streamed tool chunks may carry empty args while
                # the final AI message contains full tool arguments. Backfill before final.
                try:
                    final_tool_calls = last_output.extract_tool_calls() if hasattr(last_output, "extract_tool_calls") else []
                    for tc in final_tool_calls:
                        tool_call_id = tc.get("id", "")
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("args", {})
                        if not tool_call_id or _is_empty_tool_args(tool_args):
                            continue
                        _remember_tool_call(tool_call_id, tool_name, tool_args)
                except Exception:
                    pass
                
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
                    total_tool_calls=tool_call_count,
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
                    total_tool_calls=tool_call_count,
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
        self.add_endpoint('/api/ab/create', 'ab_create', self.require_auth(self.ab_create_comparison), methods=["POST"])
        self.add_endpoint('/api/ab/preference', 'ab_preference', self.require_auth(self.ab_submit_preference), methods=["POST"])
        self.add_endpoint('/api/ab/pending', 'ab_pending', self.require_auth(self.ab_get_pending), methods=["GET"])

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
            agent_class = chat_cfg.get("agent_class") or chat_cfg.get("pipeline")
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

    def _get_agent_class_name(self) -> Optional[str]:
        chat_cfg = self.services_config.get("chat_app", {})
        return chat_cfg.get("agent_class") or chat_cfg.get("pipeline")

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
        agent_class = chat_cfg.get("agent_class") or chat_cfg.get("pipeline")
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
            if error_code == 408:
                output = jsonify({'error': 'client timeout'})
            elif error_code == 403:
                output = jsonify({'error': 'conversation not found'})
            else:
                output = jsonify({'error': 'server error; see chat logs for message'})
            return output, error_code

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

        def _event_stream() -> Iterator[str]:
            padding = " " * 2048
            yield json.dumps({"type": "meta", "event": "stream_started", "padding": padding}) + "\n"
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
        return render_template('index.html')

    def terms(self):
        return render_template('terms.html')

    def like(self):
        self.chat.lock.acquire()
        logger.info("Acquired lock file")
        try:
            data = request.json
            message_id = data.get('message_id')

            if not message_id:
                logger.warning("Like request missing message_id")
                return jsonify({'error': 'message_id is required'}), 400

            # Check current state for toggle behavior
            current_reaction = self.chat.get_reaction_feedback(message_id)
            
            # Always delete existing reaction first
            self.chat.delete_reaction_feedback(message_id)

            # If already liked, just remove (toggle off) - don't re-add
            if current_reaction == 'like':
                response = {'message': 'Reaction removed', 'state': None}
                return jsonify(response), 200

            # Otherwise, add the like
            feedback = {
                "message_id"   : message_id,
                "feedback"     : "like",
                "feedback_ts"  : datetime.now(),
                "feedback_msg" : None,
                "incorrect"    : None,
                "unhelpful"    : None,
                "inappropriate": None,
            }
            self.chat.insert_feedback(feedback)

            response = {'message': 'Liked', 'state': 'like'}
            return jsonify(response), 200

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

    def dislike(self):
        self.chat.lock.acquire()
        logger.info("Acquired lock file")
        try:
            data = request.json
            message_id = data.get('message_id')

            if not message_id:
                logger.warning("Dislike request missing message_id")
                return jsonify({'error': 'message_id is required'}), 400

            feedback_msg = data.get('feedback_msg')
            incorrect = data.get('incorrect')
            unhelpful = data.get('unhelpful')
            inappropriate = data.get('inappropriate')

            # Check current state for toggle behavior
            current_reaction = self.chat.get_reaction_feedback(message_id)
            
            # Always delete existing reaction first
            self.chat.delete_reaction_feedback(message_id)

            # If already disliked, just remove (toggle off) - don't re-add
            if current_reaction == 'dislike':
                response = {'message': 'Reaction removed', 'state': None}
                return jsonify(response), 200

            # Otherwise, add the dislike
            feedback = {
                "message_id"   : message_id,
                "feedback"     : "dislike",
                "feedback_ts"  : datetime.now(),
                "feedback_msg" : feedback_msg,
                "incorrect"    : incorrect,
                "unhelpful"    : unhelpful,
                "inappropriate": inappropriate,
            }
            self.chat.insert_feedback(feedback)

            response = {'message': 'Disliked', 'state': 'dislike'}
            return jsonify(response), 200

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

    def text_feedback(self):
        self.chat.lock.acquire()
        logger.info("Acquired lock file for text feedback")
        try:
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

            response = {'message': 'Feedback submitted'}
            return jsonify(response), 200

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

    def ab_create_comparison(self):
        """
        Create a new A/B comparison record linking two responses.

        POST body:
        - conversation_id: The conversation ID
        - user_prompt_mid: Message ID of the user's question
        - response_a_mid: Message ID of response A
        - response_b_mid: Message ID of response B
        - config_a_id: Config ID used for response A
        - config_b_id: Config ID used for response B
        - is_config_a_first: True if config A was the "first" config before randomization
        - client_id: Client ID for authorization

        Returns:
            JSON with comparison_id
        """
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            user_prompt_mid = data.get('user_prompt_mid')
            response_a_mid = data.get('response_a_mid')
            response_b_mid = data.get('response_b_mid')
            config_a_id = data.get('config_a_id')
            config_b_id = data.get('config_b_id')
            is_config_a_first = data.get('is_config_a_first', True)
            client_id = data.get('client_id')

            # Validate required fields
            missing = []
            if not conversation_id:
                missing.append('conversation_id')
            if not user_prompt_mid:
                missing.append('user_prompt_mid')
            if not response_a_mid:
                missing.append('response_a_mid')
            if not response_b_mid:
                missing.append('response_b_mid')
            if not config_a_id:
                missing.append('config_a_id')
            if not config_b_id:
                missing.append('config_b_id')
            if not client_id:
                missing.append('client_id')

            if missing:
                return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

            # Create the comparison
            comparison_id = self.chat.create_ab_comparison(
                conversation_id=conversation_id,
                user_prompt_mid=user_prompt_mid,
                response_a_mid=response_a_mid,
                response_b_mid=response_b_mid,
                config_a_id=config_a_id,
                config_b_id=config_b_id,
                is_config_a_first=is_config_a_first,
            )

            return jsonify({
                'success': True,
                'comparison_id': comparison_id,
            }), 200

        except Exception as e:
            logger.error(f"Error creating A/B comparison: {str(e)}")
            return jsonify({'error': str(e)}), 500

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
            
            # Build a pattern to match the repo URL
            # repo_name could be a URL (https://github.com/org/repo) or just a repo name (org/repo)
            # URLs in database are like: https://github.com/pallets/click/blob/main/file.py
            conn = psycopg2.connect(**self.chat.pg_config)
            try:
                with conn.cursor() as cursor:
                    # First, get the resource hashes of documents to delete
                    cursor.execute("""
                        SELECT resource_hash FROM documents 
                        WHERE source_type = 'git' 
                          AND NOT is_deleted
                          AND (
                              url LIKE %s
                              OR url LIKE %s
                          )
                    """, (f'{repo_name}/%', f'%/{repo_name}/%'))
                    hashes_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    if hashes_to_delete:
                        # Delete chunks for these documents
                        cursor.execute("""
                            DELETE FROM document_chunks 
                            WHERE metadata->>'resource_hash' = ANY(%s)
                        """, (hashes_to_delete,))
                        chunks_deleted = cursor.rowcount
                        logger.info(f"Deleted {chunks_deleted} chunks for {len(hashes_to_delete)} documents")
                    
                    # Mark documents as deleted
                    cursor.execute("""
                        UPDATE documents 
                        SET is_deleted = TRUE, deleted_at = NOW()
                        WHERE source_type = 'git' 
                          AND NOT is_deleted
                          AND (
                              url LIKE %s
                              OR url LIKE %s
                          )
                    """, (f'{repo_name}/%', f'%/{repo_name}/%'))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                logger.info(f"Deleted {deleted_count} documents from git repo: {repo_name}")
                return jsonify({
                    "success": True,
                    "deleted_count": deleted_count,
                    "message": f"Removed {deleted_count} documents from repository"
                }), 200
            finally:
                conn.close()
                
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
            
            conn = psycopg2.connect(**self.chat.pg_config)
            try:
                with conn.cursor() as cursor:
                    # First, get the resource hashes of documents to delete
                    cursor.execute("""
                        SELECT resource_hash FROM documents 
                        WHERE source_type = 'jira' 
                          AND NOT is_deleted
                          AND display_name LIKE %s
                    """, (f'{project_key}-%',))
                    hashes_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    if hashes_to_delete:
                        # Delete chunks for these documents
                        cursor.execute("""
                            DELETE FROM document_chunks 
                            WHERE metadata->>'resource_hash' = ANY(%s)
                        """, (hashes_to_delete,))
                        chunks_deleted = cursor.rowcount
                        logger.info(f"Deleted {chunks_deleted} chunks for {len(hashes_to_delete)} Jira documents")
                    
                    # Mark documents from this Jira project as deleted
                    cursor.execute("""
                        UPDATE documents 
                        SET is_deleted = TRUE, deleted_at = NOW()
                        WHERE source_type = 'jira' 
                          AND NOT is_deleted
                          AND display_name LIKE %s
                    """, (f'{project_key}-%',))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                logger.info(f"Deleted {deleted_count} documents from Jira project: {project_key}")
                return jsonify({
                    "success": True,
                    "deleted_count": deleted_count,
                    "message": f"Removed {deleted_count} tickets from project {project_key}"
                }), 200
            finally:
                conn.close()
                
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
