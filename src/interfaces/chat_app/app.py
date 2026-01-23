import json
import os
import re
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse
from functools import wraps

import chromadb
import mistune as mt
import numpy as np
import psycopg2
import psycopg2.extras
import yaml
from authlib.integrations.flask_client import OAuth
from chromadb.config import Settings
from flask import jsonify, render_template, request, session, flash, redirect, url_for, Response, stream_with_context
from flask_cors import CORS
from langchain_chroma.vectorstores import Chroma
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import (BashLexer, CLexer, CppLexer, FortranLexer,
                             HtmlLexer, JavaLexer, JavascriptLexer, JuliaLexer,
                             MathematicaLexer, MatlabLexer, PythonLexer,
                             TypeScriptLexer)

from src.a2rchi.a2rchi import A2rchi
# from src.data_manager.data_manager import DataManager
from src.utils.runtime_config_loader import get_runtime_config_names, load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import get_logger
from src.utils.sql import SQL_INSERT_CONVO, SQL_INSERT_FEEDBACK, SQL_INSERT_TIMING, SQL_QUERY_CONVO, SQL_INSERT_CONFIG, SQL_CREATE_CONVERSATION, SQL_UPDATE_CONVERSATION_TIMESTAMP, SQL_LIST_CONVERSATIONS, SQL_GET_CONVERSATION_METADATA, SQL_DELETE_CONVERSATION, SQL_INSERT_TOOL_CALLS, SQL_QUERY_CONVO_WITH_FEEDBACK, SQL_DELETE_REACTION_FEEDBACK
from src.interfaces.chat_app.document_utils import *
from src.interfaces.chat_app.utils import collapse_assistant_sequences


logger = get_logger(__name__)

# DEFINITIONS
QUERY_LIMIT = 10000 # max queries per conversation
MAIN_PROMPT_FILE = "/root/A2rchi/main.prompt"
CONDENSE_PROMPT_FILE = "/root/A2rchi/condense.prompt"
SUMMARY_PROMPT_FILE = "/root/A2rchi/summary.prompt"
A2RCHI_SENDER = "A2rchi"


class AnswerRenderer(mt.HTMLRenderer):
    """
    Class for custom rendering of A2rchi output. Child of mistune's HTMLRenderer, with custom overrides.
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
        self.config = load_runtime_config()
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
        # load configs
        self.config = load_runtime_config()
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]
        self.data_path = self.global_config["DATA_PATH"]

        # initialize data manager (ingestion handled by data-manager service)
        # self.data_manager = DataManager(run_ingestion=False)
        embedding_name = self.config["data_manager"]["embedding_name"]
        self.similarity_score_reference = self.config["data_manager"]["embedding_class_map"][embedding_name]["similarity_score_reference"]
        self.sources_config = self.config["data_manager"]["sources"]

        # store postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        # initialize chain
        self.a2rchi = A2rchi(pipeline=self.config["services"]["chat_app"]["pipeline"])
        self.number_of_queries = 0

        # track configs and active config state
        self.default_config_name = self.config.get("name")
        self.current_config_name = None
        self.config_id = None
        self.config_name_to_id = {}
        self._config_cache = {}
        if self.default_config_name:
            self._config_cache[self.default_config_name] = self.config

        # ensure all supplied configs are registered in Postgres and activate default
        self._store_config_ids()
        if self.default_config_name:
            self.update_config(config_name=self.default_config_name)

    def update_config(self, config_id=None, config_name=None):
        """
        Update the active config by ensuring it exists in thje postgres and applying it to the pipeline.
        """
        target_config_name = config_name or self.current_config_name or self.default_config_name
        if not target_config_name:
            raise ValueError("Config name must be provided to update the chat configuration.")

        config_payload = self._get_config_payload(target_config_name)
        if config_id is None:
            config_id = self._get_or_create_config_id(target_config_name, config_payload)
        else:
            self.config_name_to_id[target_config_name] = config_id

        if self.config_id == config_id and self.current_config_name == target_config_name:
            return

        pipeline_name = config_payload["services"]["chat_app"]["pipeline"]
        self.config_id = config_id
        self.current_config_name = target_config_name
        self.a2rchi.update(pipeline=pipeline_name, config_name=target_config_name)

    def _get_config_payload(self, config_name):
        if config_name not in self._config_cache:
            self._config_cache[config_name] = load_runtime_config(name=config_name)
        return self._config_cache[config_name]

    def _get_or_create_config_id(self, config_name, config_payload=None):
        if config_name in self.config_name_to_id:
            return self.config_name_to_id[config_name]

        payload = config_payload or self._get_config_payload(config_name)
        serialized = yaml.dump(payload)

        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT config_id FROM configs WHERE config_name = %s ORDER BY config_id DESC LIMIT 1", (config_name,))
            row = cursor.fetchone()
            if row:
                config_id = row[0]
            else:
                insert_tup = [(serialized, config_name)]
                psycopg2.extras.execute_values(cursor, SQL_INSERT_CONFIG, insert_tup)
                config_id = list(map(lambda tup: tup[0], cursor.fetchall()))[0]
            conn.commit()
            self.config_name_to_id[config_name] = config_id
            return config_id
        finally:
            cursor.close()
            conn.close()

    def _store_config_ids(self):
        for config_name in get_runtime_config_names():
            try:
                payload = self._get_config_payload(config_name)
                self._get_or_create_config_id(config_name, payload)
            except FileNotFoundError:
                logger.warning(f"Config file {config_name} missing.")
            except Exception as exc:
                logger.warning(f"Failed to register config {config_name}: {exc}")

    def get_config_id(self, config_name):
        """
        Helper for external callers needing the config_id for a given config name.
        """
        if config_name in self.config_name_to_id:
            return self.config_name_to_id[config_name]
        return self._get_or_create_config_id(config_name)

    @staticmethod
    def convert_to_app_history(history):
        """
        Input: the history in the form of a list of tuples, where the first entry of each tuple is
        the author of the text and the second entry is the text itself (native A2rchi history format)

        Output: the history in the form of a list of lists, where the first entry of each tuple is
        the author of the text and the second entry is the text itself
        """
        return [list(entry) for entry in history]


    @staticmethod
    def format_code_in_text(text):
        """
        Takes in input plain text (the output from A2rchi);
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
        Build a list of top reference entries (link or ticket id).
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

            if len(top_sources) >= 5:
                break

        logger.debug(f"Top sources: {top_sources}")
        return top_sources

    @staticmethod
    def format_links(top_sources):
        _output = ""

        if top_sources:
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

            for entry in top_sources:
                score = entry["score"]
                link = entry["link"]
                display_name = entry["display"]
                
                # Format score: show nothing for -1 (placeholder), otherwise show numeric value
                if score == -1.0 or score == "N/A":
                    score_str = ""
                else:
                    score_str = f"({score:.2f})"

                if link:
                    reference_html = f"<a href=\"{link}\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"color: #66b3ff; text-decoration: none;\" onmouseover=\"this.style.textDecoration='underline'\" onmouseout=\"this.style.textDecoration='none'\">{display_name}</a>"
                else:
                    reference_html = f"<span style=\"color: #66b3ff;\">{display_name}</span>"

                _output += f'''
                    <div style="margin: 0.15em 0; display: flex; align-items: center; gap: 0.4em;">
                        <span>•</span>
                        {reference_html}
                        <span style="color: #6c757d; font-size: 0.9em;">{score_str}</span>
                    </div>
                '''

            _output += '</div>'

        return _output

    @staticmethod
    def _looks_like_url(value: str | None) -> bool:
        return isinstance(value, str) and value.startswith(("http://", "https://"))

    @staticmethod
    def _get_display_name(metadata: dict) -> str | None:
        display_name = metadata.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            return display_name.strip()
        file_name = metadata.get("file_name")
        if isinstance(file_name, str) and file_name.strip():
            return file_name.strip()
        logger.error("display_name or file_name is not a valid non-empty string in metadata")
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


    def query_conversation_history(self, conversation_id, client_id):
        """
        Return the conversation history as an ordered list of tuples. The order
        is determined by ascending message_id. Each tuple contains the sender and
        the message content
        """
        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()

        # ensure conversation belongs to client before querying
        self.cursor.execute(SQL_GET_CONVERSATION_METADATA, (conversation_id, client_id))
        metadata = self.cursor.fetchone()
        if metadata is None:
            self.cursor.close()
            self.conn.close()
            self.cursor, self.conn = None, None
            raise ConversationAccessError("Conversation does not exist for this client")

        # query conversation history
        self.cursor.execute(SQL_QUERY_CONVO, (conversation_id,))
        history = self.cursor.fetchall()
        history = collapse_assistant_sequences(history, sender_name=A2RCHI_SENDER)

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

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

        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        self.cursor.execute(SQL_CREATE_CONVERSATION, insert_tup)
        conversation_id = self.cursor.fetchone()[0]
        self.conn.commit()

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

        logger.info(f"Created new conversation with ID: {conversation_id}")
        return conversation_id

    def update_conversation_timestamp(self, conversation_id: int, client_id: str):
        """
        Update the last_message_at timestamp for a conversation.
        last_message_at is used to reorder conversations in the UI (on vertical sidebar).
        """
        now = datetime.now()

        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()

        # update timestamp
        self.cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (now, conversation_id, client_id))
        self.conn.commit()

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

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

    def insert_conversation(self, conversation_id, user_message, a2rchi_message, link, a2rchi_context, is_refresh=False) -> List[int]:
        """
        """
        logger.debug("Entered insert_conversation.")

        def _sanitize(text: str) -> str:
            return text.replace("\x00", "") if isinstance(text, str) else text

        service = "Chatbot"
        # parse user message / a2rchi message
        user_sender, user_content, user_msg_ts = user_message
        a2rchi_sender, a2rchi_content, a2rchi_msg_ts = a2rchi_message

        user_content = _sanitize(user_content)
        a2rchi_content = _sanitize(a2rchi_content)
        link = _sanitize(link)
        a2rchi_context = _sanitize(a2rchi_context)

        # construct insert_tups
        insert_tups = (
            [
                # (service, conversation_id, sender, content, context, ts)
                (service, conversation_id, user_sender, user_content, '', '', user_msg_ts, self.config_id),
                (service, conversation_id, a2rchi_sender, a2rchi_content, link, a2rchi_context, a2rchi_msg_ts, self.config_id),
            ]
            if not is_refresh
            else [
                (service, conversation_id, a2rchi_sender, a2rchi_content, link, a2rchi_context, a2rchi_msg_ts, self.config_id),
            ]
        )

        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        psycopg2.extras.execute_values(self.cursor, SQL_INSERT_CONVO, insert_tups)
        self.conn.commit()
        message_ids = list(map(lambda tup: tup[0], self.cursor.fetchall()))

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

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
            timestamps['a2rchi_message_ts'],
            timestamps['insert_convo_ts'],
            timestamps['finish_call_ts'],
            timestamps['server_response_msg_ts'],
            timestamps['server_response_msg_ts'] - timestamps['server_received_msg_ts']
        )

        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        self.cursor.execute(SQL_INSERT_TIMING, insert_tup)
        self.conn.commit()

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

    def insert_tool_calls_from_messages(self, conversation_id: int, message_id: int, messages: List) -> None:
        """
        Extract and store agent tool calls from the messages list.
        
        AIMessage with tool_calls contains the tool name, args, and timestamp.
        ToolMessage contains the result, matched by tool_call_id.
        """
        if not messages:
            return
        
        tool_results = {}
        for msg in messages:
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                tool_results[msg.tool_call_id] = getattr(msg, 'content', '')
        
        # Extract tool calls from AIMessages
        insert_tups = []
        step_number = 0
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Get timestamp from response_metadata if available
                response_metadata = getattr(msg, 'response_metadata', {}) or {}
                created_at = response_metadata.get('created_at')
                if created_at:
                    try:
                        # Parse ISO format timestamp
                        ts = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        ts = datetime.now()
                else:
                    ts = datetime.now()
                
                for tc in msg.tool_calls:
                    step_number += 1
                    tool_call_id = tc.get('id', '')
                    tool_name = tc.get('name', 'unknown')
                    tool_args = tc.get('args', {})
                    tool_result = tool_results.get(tool_call_id, '')
                    # Truncate result for storage (max 500 chars)
                    if len(tool_result) > 500:
                        tool_result = tool_result[:500] + '...'
                    
                    insert_tups.append((
                        conversation_id,
                        message_id,
                        step_number,
                        tool_name,
                        json.dumps(tool_args) if tool_args else None,
                        tool_result,
                        ts,
                    ))
        
        if not insert_tups:
            return
            
        logger.debug("Inserting %d tool calls for message %d", len(insert_tups), message_id)

        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        psycopg2.extras.execute_values(self.cursor, SQL_INSERT_TOOL_CALLS, insert_tups)
        self.conn.commit()

        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

    def _init_timestamps(self) -> Dict[str, datetime]:
        return {
            "lock_acquisition_ts": datetime.now(),
            "vectorstore_update_ts": datetime.now(),
        }

    def _resolve_config_name(self, config_name: Optional[str]) -> str:
        return config_name or self.current_config_name or self.default_config_name

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
            while history and history[-1][0] == A2RCHI_SENDER:
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
    ) -> tuple[str, List[int]]:
        output = self.format_code_in_text(result["answer"])

        documents = result.get("source_documents", [])
        scores = result.get("metadata", {}).get("retriever_scores", [])
        top_sources = self.get_top_sources(documents, scores)
        output += self.format_links(top_sources)

        timestamps["a2rchi_message_ts"] = datetime.now()
        context_data = self.prepare_context_for_storage(documents, scores)

        best_reference = "Link unavailable"
        if top_sources:
            primary_source = top_sources[0]
            best_reference = primary_source["link"] or primary_source["display"]

        user_message = (context.sender, context.content, server_received_msg_ts)
        a2rchi_message = (A2RCHI_SENDER, output, timestamps["a2rchi_message_ts"])
        message_ids = self.insert_conversation(
            context.conversation_id,
            user_message,
            a2rchi_message,
            best_reference,
            context_data,
            context.is_refresh,
        )
        timestamps["insert_convo_ts"] = datetime.now()
        context.history.append((A2RCHI_SENDER, result["answer"]))

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
            a2rchi_message_id = message_ids[-1]
            self.insert_tool_calls_from_messages(context.conversation_id, a2rchi_message_id, agent_messages)

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

            result = self.a2rchi(history=context.history, conversation_id=context.conversation_id)
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
    ) -> Iterator[Dict[str, Any]]:
        timestamps = self._init_timestamps()
        context = None
        last_output = None

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

            for output in self.a2rchi.stream(history=context.history, conversation_id=context.conversation_id):
                logger.debug("Recieved streaming output chunk: %s", output)
                last_output = output
                if getattr(output, "final", False):
                    continue
                for event in self._stream_events_from_output(
                    output,
                    include_agent_steps=include_agent_steps,
                    include_tool_steps=include_tool_steps,
                    conversation_id=context.conversation_id,
                    max_chars=max_step_chars,
                ):
                    yield event

            timestamps["chain_finished_ts"] = datetime.now()

            if last_output is None:
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
            )

            timestamps["finish_call_ts"] = datetime.now()
            timestamps["server_received_msg_ts"] = server_received_msg_ts
            timestamps["client_sent_msg_ts"] = datetime.fromtimestamp(client_sent_msg_ts)
            timestamps["server_response_msg_ts"] = datetime.now()

            if message_ids:
                self.insert_timing(message_ids[-1], timestamps)

            yield {
                "type": "final",
                "response": output,
                "conversation_id": context.conversation_id,
                "a2rchi_msg_id": message_ids[-1] if message_ids else None,
                "server_response_msg_ts": timestamps["server_response_msg_ts"].timestamp(),
                "final_response_msg_ts": datetime.now().timestamp(),
            }

        except ConversationAccessError as exc:
            logger.warning("Unauthorized conversation access attempt: %s", exc)
            yield {"type": "error", "status": 403, "message": "conversation not found"}
        except Exception as exc:
            logger.error("Failed to stream response: %s", exc, exc_info=True)
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
        self.config = load_runtime_config()
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
        self.app.config['ACCOUNTS_FOLDER'] = self.global_config["ACCOUNTS_PATH"]
        os.makedirs(self.app.config['ACCOUNTS_FOLDER'], exist_ok=True)

        # store postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        # Initialize authentication methods
        self.oauth = None
        auth_config = self.chat_app_config.get('auth', {})
        self.auth_enabled = auth_config.get('enabled', False)
        self.sso_enabled = auth_config.get('sso', {}).get('enabled', False)
        self.basic_auth_enabled = auth_config.get('basic', {}).get('enabled', False)
        
        logger.info(f"Auth enabled: {self.auth_enabled}, SSO: {self.sso_enabled}, Basic: {self.basic_auth_enabled}")
        
        if self.sso_enabled:
            self._setup_sso()

        # insert config
        self.config_id = self.insert_config(self.config)

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

        # conditionally add ChromaDB endpoints based on config
        # if self.chat_app_config.get('enable_debug_chroma_endpoints', False):
        #     logger.info("Adding ChromaDB API endpoints (list_docs, search_docs)")
        #     self.add_endpoint('/api/list_docs', 'list_docs', self.require_auth(self.list_docs), methods=["GET"])
        #     self.add_endpoint('/api/search_docs', 'search_docs', self.require_auth(self.search_docs), methods=["POST"])
        # else:
        #     logger.info("ChromaDB API endpoints disabled by config")

        # endpoints for conversations managing
        logger.info("Adding conversations management API endpoints")
        self.add_endpoint('/api/list_conversations', 'list_conversations', self.require_auth(self.list_conversations), methods=["GET"])
        self.add_endpoint('/api/load_conversation', 'load_conversation', self.require_auth(self.load_conversation), methods=["POST"])
        self.add_endpoint('/api/new_conversation', 'new_conversation', self.require_auth(self.new_conversation), methods=["POST"])
        self.add_endpoint('/api/delete_conversation', 'delete_conversation', self.require_auth(self.delete_conversation), methods=["POST"])

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
                # Return 401 Unauthorized response instead of redirecting
                return jsonify({'error': 'Unauthorized', 'message': 'Authentication required'}), 401
            
            return f(*args, **kwargs)
        return decorated_function

    def health(self):
        return jsonify({"status": "OK"}, 200)

    def configs(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value

    def add_endpoint(self, endpoint = None, endpoint_name = None, handler = None, methods = ['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods = methods, *args, **kwargs)

    def run(self, **kwargs):
        self.app.run(**kwargs)

    def insert_config(self, config):
        # TODO: use config_name (and then hash of config string) to determine
        #       if config already exists; if so, don't push new config

        # parse config and config_name
        config_name = self.config["name"]
        config = yaml.dump(self.config)

        # construct insert_tup
        insert_tup = [
            (config, config_name),
        ]

        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        psycopg2.extras.execute_values(self.cursor, SQL_INSERT_CONFIG, insert_tup)
        self.conn.commit()
        config_id = list(map(lambda tup: tup[0], self.cursor.fetchall()))[0]

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

        return config_id

    def update_config(self):
        """
        Updates the config used by A2rchi for responding to messages. The config
        is parsed and inserted into the `configs` table. Finally, the chat wrapper's
        config_id is updated.
        """
        # parse config payload and persist it in Postgres
        config_payload = request.json.get('config')
        if isinstance(config_payload, str):
            config_payload = yaml.safe_load(config_payload)
        if not isinstance(config_payload, dict):
            return jsonify({'error': 'Invalid config payload'}), 400
        config_name = config_payload.get('name')
        if not config_name:
            return jsonify({'error': 'Config name is required'}), 400

        config_text = yaml.safe_dump(config_payload, sort_keys=False)
        insert_tup = [(config_text, config_name)]
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        psycopg2.extras.execute_values(self.cursor, SQL_INSERT_CONFIG, insert_tup)
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None

        # parse prompts and write them to their respective locations
        # TODO fix
        main_prompt = request.json.get('main_prompt')
        with open(MAIN_PROMPT_FILE, 'w') as f:
            f.write(main_prompt)

        condense_prompt = request.json.get('condense_prompt')
        with open(CONDENSE_PROMPT_FILE, 'w') as f:
            f.write(condense_prompt)

        summary_prompt = request.json.get('summary_prompt')
        with open(SUMMARY_PROMPT_FILE, 'w') as f:
            f.write(summary_prompt)

        # re-read config using load_config and update dependent variables
        self.config = load_runtime_config(name=config_name)
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]
        self.chat_app_config = self.config["services"]["chat_app"]
        self.data_path = self.global_config["DATA_PATH"]

        # store postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        # recreate chat wrapper so all dependent services reload the new config
        self.chat = ChatWrapper()
        self.chat.update_config(config_name=self.config["name"])
        new_config_id = self.chat.get_config_id(self.config["name"])

        return jsonify({'response': f'config updated successfully w/config_id: {new_config_id}'}), 200

    def get_configs(self):
        """
        Gets the names of configs loaded in A2rchi.


        Returns:
            A json with a response list of the configs names
        """

        config_names = get_runtime_config_names()
        options = []
        for name in config_names:
            description = ""
            try:
                payload = load_runtime_config(name=name)
                description = self._resolve_agent_description(payload)
            except Exception as exc:
                logger.warning(f"Failed to load config {name} for description: {exc}")
            options.append({"name": name, "description": description})
        return jsonify({'options': options}), 200

    def _resolve_agent_description(self, payload: Dict[str, Any]) -> str:
        a2rchi_cfg = payload.get("a2rchi", {}) if isinstance(payload, dict) else {}
        services_cfg = payload.get("services", {}) if isinstance(payload, dict) else {}
        pipeline_name = services_cfg.get("chat_app", {}).get("pipeline")
        if not pipeline_name:
            pipeline_name = (a2rchi_cfg.get("pipelines") or [None])[0]
        pipeline_cfg = a2rchi_cfg.get("pipeline_map", {}).get(pipeline_name, {}) if pipeline_name else {}
        return pipeline_cfg.get("agent_description", "No description provided")

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

        # otherwise return A2rchi's response to client
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
            'a2rchi_msg_id': message_ids[-1],
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

        if not client_id:
            return jsonify({"error": "client_id missing"}), 400

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
            # Get the JSON data from the request body
            data = request.json

            # Extract the HTML content and any other data you need
            message_id = data.get('message_id')

            self.chat.delete_reaction_feedback(message_id)

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

            response = {'message': 'Liked'}
            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return jsonify({'error': str(e)}), 500

        # According to the Python documentation: https://docs.python.org/3/tutorial/errors.html#defining-clean-up-actions
        # this will still execute, before the function returns in the try or except block.
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
            # Get the JSON data from the request body
            data = request.json

            # Extract the HTML content and any other data you need
            message_id = data.get('message_id')
            feedback_msg = data.get('feedback_msg')
            incorrect = data.get('incorrect')
            unhelpful = data.get('unhelpful')
            inappropriate = data.get('inappropriate')

            self.chat.delete_reaction_feedback(message_id)

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

            response = {'message': 'Disliked'}
            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return jsonify({'error': str(e)}), 500

        # According to the Python documentation: https://docs.python.org/3/tutorial/errors.html#defining-clean-up-actions
        # this will still execute, before the function returns in the try or except block.
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

    # def list_docs(self):
    #     """
    #     API endpoint to list all documents indexed in ChromaDB with pagination.
    #     Query parameters:
    #     - page: Page number (1-based, default: 1)
    #     - per_page: Documents per page (default: 50, max: 500)
    #     - content_length: Max content preview length (default: -1 for full content)
    #     Returns a JSON with paginated list of documents and their metadata.
    #     """
    #     # Check if ChromaDB endpoints are enabled
    #     if not self.chat_app_config.get('enable_debug_chroma_endpoints', False):
    #         return jsonify({'error': 'ChromaDB endpoints are disabled in configuration'}), 404

    #     try:
    #         # Get pagination parameters from query string
    #         page = int(request.args.get('page', 1))
    #         per_page = min(int(request.args.get('per_page', 50)), 500)  # Cap at 500
    #         content_length = int(request.args.get('content_length', -1))  # Default -1 for full content

    #         # Validate parameters
    #         if page < 1:
    #             return jsonify({'error': 'Page must be >= 1'}), 400
    #         if per_page < 1:
    #             return jsonify({'error': 'per_page must be >= 1'}), 400
    #         if content_length < -1 or content_length == 0:
    #             return jsonify({'error': 'content_length must be -1 (full content) or > 0'}), 400

    #         # Get the collection from ChromaDB
    #         collection = self.chat.data_manager.fetch_collection()

    #         # Get total count first
    #         total_documents = collection.count()

    #         # Calculate pagination
    #         offset = (page - 1) * per_page
    #         total_pages = (total_documents + per_page - 1) // per_page  # Ceiling division

    #         # Check if page is valid
    #         if page > total_pages and total_documents > 0:
    #             return jsonify({'error': f'Page {page} does not exist. Total pages: {total_pages}'}), 400

    #         # Get paginated documents from the collection
    #         result = collection.get(
    #             include=['documents', 'metadatas'],
    #             limit=per_page,
    #             offset=offset
    #         )

    #         # Format the response
    #         documents = []
    #         for i, doc in enumerate(result['documents']):
    #             # Truncate content based on content_length parameter (-1 means full content)
    #             if content_length == -1:
    #                 content = doc  # Return full content
    #             else:
    #                 content = doc[:content_length] + '...' if len(doc) > content_length else doc

    #             doc_info = {
    #                 'id': result['ids'][i],
    #                 'content': content,
    #                 'content_length': len(doc),  # Original content length
    #                 'metadata': result['metadatas'][i] if i < len(result['metadatas']) else {}
    #             }
    #             documents.append(doc_info)

    #         response_data = {
    #             'pagination': {
    #                 'page': page,
    #                 'per_page': per_page,
    #                 'total_documents': total_documents,
    #                 'total_pages': total_pages,
    #                 'has_next': page < total_pages,
    #                 'has_prev': page > 1,
    #                 'next_page': page + 1 if page < total_pages else None,
    #                 'prev_page': page - 1 if page > 1 else None
    #             },
    #             'documents': documents
    #         }

    #         return jsonify(response_data), 200

    #     except ValueError as e:
    #         return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
    #     except Exception as e:
    #         print(f"ERROR in list_docs: {str(e)}")
    #         return jsonify({'error': str(e)}), 500

    # # TODO should this call a2rchi rather than connect to db directly?
    # # in any case, code-duplication should be elminated here
    # def search_docs(self):
    #     """
    #     API endpoint to search for the nearest documents to a given query with pagination.
    #     Expects JSON input with:
    #     - query (required): Search query string
    #     - n_results (optional): Number of results to return (default: 5, max: 100)
    #     - content_length (optional): Max content length in response (default: -1 for full content, max: 5000)
    #     - include_full_content (optional): Whether to include full content (default: false)
    #     Returns the most similar documents with their similarity scores.
    #     """
    #     # Check if ChromaDB endpoints are enabled
    #     if not self.chat_app_config.get('enable_debug_chroma_endpoints', False):
    #         return jsonify({'error': 'ChromaDB endpoints are disabled in configuration'}), 404

    #     try:
    #         # Get the query from request
    #         data = request.json
    #         query = data.get('query')
    #         n_results = min(int(data.get('n_results', 5)), 100)  # Cap at 100
    #         content_length = min(int(data.get('content_length', -1)), 5000) if data.get('content_length', -1) != -1 else -1  # Default -1 for full content
    #         include_full_content = data.get('include_full_content', False)

    #         if not query:
    #             return jsonify({'error': 'Query parameter is required'}), 400

    #         if n_results < 1:
    #             return jsonify({'error': 'n_results must be >= 1'}), 400

    #         if content_length < -1 or content_length == 0:
    #             return jsonify({'error': 'content_length must be -1 (full content) or > 0'}), 400

    #         # Connect to ChromaDB and create vectorstore
    #         client = None
    #         if self.services_config["chromadb"]["use_HTTP_chromadb_client"]:
    #             client = chromadb.HttpClient(
    #                 host=self.services_config["chromadb"]["chromadb_host"],
    #                 port=self.services_config["chromadb"]["port"],
    #                 settings=Settings(allow_reset=True, anonymized_telemetry=False),
    #             )
    #         else:
    #             client = chromadb.PersistentClient(
    #                 path=self.global_config["LOCAL_VSTORE_PATH"],
    #                 settings=Settings(allow_reset=True, anonymized_telemetry=False),
    #             )

    #         # Get the collection name and embedding model from chat
    #         collection_name = self.chat.chain.collection_name
    #         embedding_model = self.chat.chain.embedding_model

    #         # Create vectorstore
    #         vectorstore = Chroma(
    #             client=client,
    #             collection_name=collection_name,
    #             embedding_function=embedding_model,
    #         )

    #         # Perform similarity search with scores
    #         results = vectorstore.similarity_search_with_score(query, k=n_results)

    #         # Format the response
    #         documents = []
    #         for doc, score in results:
    #             # Handle content length based on parameters
    #             if include_full_content or content_length == -1:
    #                 content = doc.page_content
    #             else:
    #                 content = (doc.page_content[:content_length] + '...'
    #                          if len(doc.page_content) > content_length
    #                          else doc.page_content)

    #             doc_info = {
    #                 'content': content,
    #                 'content_length': len(doc.page_content),  # Original content length
    #                 'metadata': doc.metadata,
    #                 'similarity_score': float(score)
    #             }
    #             documents.append(doc_info)

    #         response_data = {
    #             'query': query,
    #             'search_params': {
    #                 'n_results_requested': n_results,
    #                 'n_results_returned': len(documents),
    #                 'content_length': content_length,
    #                 'include_full_content': include_full_content
    #             },
    #             'documents': documents
    #         }

    #         # Clean up
    #         del vectorstore
    #         del client

    #         return jsonify(response_data), 200

    #     except ValueError as e:
    #         return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
    #     except Exception as e:
    #         print(f"ERROR in search_docs: {str(e)}")
    #         return jsonify({'error': str(e)}), 500

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
            history_rows = collapse_assistant_sequences(history_rows, sender_name=A2RCHI_SENDER, sender_index=0)

            conversation = {
                'conversation_id': meta_row[0],
                'title': meta_row[1] or "New Conversation",
                'created_at': meta_row[2].isoformat() if meta_row[2] else None,
                'last_message_at': meta_row[3].isoformat() if meta_row[3] else None,
                'messages': [
                    {
                        'sender': row[0],
                        'content': row[1],
                        'message_id': row[2],
                        'feedback': row[3],
                        'comment_count': row[4] if len(row) > 4 else 0,
                    }
                    for row in history_rows
                ]
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

    def is_authenticated(self):
        """
        Keeps the state of the authentication.

        Returns true if there has been a correct login authentication and false otherwise.
        """
        return 'logged_in' in session and session['logged_in']
