"""Unit tests for pin/unpin/archive/unarchive/rename conversation endpoints
and the updated list_conversations on FlaskAppWrapper.

Pattern mirrors test_ab_pending_limit.py:
  - Guard the import with importorskip (app.py pulls the full LLM stack)
  - Instantiate FlaskAppWrapper via object.__new__ to skip __init__
  - Stub only pg_config
  - Patch psycopg2.connect so no real DB is needed
  - Call each method directly inside app.test_request_context
  - Assert status codes and JSON response shapes
"""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

FlaskAppWrapper = pytest.importorskip(
    "src.interfaces.chat_app.app",
    reason="app.py requires the full LLM stack",
).FlaskAppWrapper

from flask import Flask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrapper():
    app = Flask(__name__)
    app.secret_key = "test-secret"
    wrapper = object.__new__(FlaskAppWrapper)
    wrapper.app = app
    wrapper.pg_config = {
        "host": "localhost", "port": 5432,
        "database": "archi", "user": "archi", "password": "test",
    }
    return app, wrapper


def _call_flag(wrapper, method_name, body, *, rowcount=1):
    """Call a flag-setter method (pin/unpin/archive/unarchive) with mocked DB."""
    app = wrapper.app
    mock_cursor = MagicMock()
    mock_cursor.rowcount = rowcount
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn):
        with app.test_request_context(
            f"/api/{method_name}",
            method="POST",
            data=json.dumps(body),
            content_type="application/json",
        ):
            method = getattr(FlaskAppWrapper, method_name)
            response, status = method(wrapper)

    return response.get_json(), status, mock_cursor


def _call_list(wrapper, query_string="?client_id=abc&limit=50", *, rows=None):
    """Call list_conversations with mocked DB rows."""
    app = wrapper.app
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows or []
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn):
        with app.test_request_context(
            f"/api/list_conversations{query_string}",
            method="GET",
        ):
            response, status = FlaskAppWrapper.list_conversations(wrapper)

    return response.get_json(), status, mock_cursor


def _make_row(conv_id=1, title="Chat", is_pinned=False, is_archived=False):
    now = datetime.now(timezone.utc)
    return (conv_id, title, now, now, is_pinned, is_archived)


# ---------------------------------------------------------------------------
# pin_conversation
# ---------------------------------------------------------------------------

class TestPinConversation:

    def test_happy_path_returns_200(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "pin_conversation",
            {"conversation_id": 1, "client_id": "abc"},
        )
        assert status == 200
        assert data["success"] is True
        assert data["conversation_id"] == 1

    def test_missing_conversation_id_returns_400(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "pin_conversation",
            {"client_id": "abc"},
        )
        assert status == 400
        assert "conversation_id" in data["error"].lower()

    def test_missing_client_id_returns_400(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "pin_conversation",
            {"conversation_id": 1},
        )
        assert status == 400
        assert "client_id" in data["error"].lower()

    def test_not_found_returns_404(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "pin_conversation",
            {"conversation_id": 999, "client_id": "abc"},
            rowcount=0,
        )
        assert status == 404
        assert "not found" in data["error"].lower()


# ---------------------------------------------------------------------------
# unpin_conversation
# ---------------------------------------------------------------------------

class TestUnpinConversation:

    def test_happy_path_returns_200(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "unpin_conversation",
            {"conversation_id": 2, "client_id": "abc"},
        )
        assert status == 200
        assert data["success"] is True

    def test_not_found_returns_404(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "unpin_conversation",
            {"conversation_id": 999, "client_id": "abc"},
            rowcount=0,
        )
        assert status == 404


# ---------------------------------------------------------------------------
# archive_conversation
# ---------------------------------------------------------------------------

class TestArchiveConversation:

    def test_happy_path_returns_200(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "archive_conversation",
            {"conversation_id": 3, "client_id": "abc"},
        )
        assert status == 200
        assert data["success"] is True
        assert data["conversation_id"] == 3

    def test_missing_conversation_id_returns_400(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "archive_conversation",
            {"client_id": "abc"},
        )
        assert status == 400

    def test_not_found_returns_404(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "archive_conversation",
            {"conversation_id": 999, "client_id": "abc"},
            rowcount=0,
        )
        assert status == 404

    def test_archive_uses_correct_sql_anon(self):
        """Anonymous path must execute SQL_ARCHIVE_CONVERSATION (clears pin too)."""
        app, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn), \
             patch("src.interfaces.chat_app.app.SQL_ARCHIVE_CONVERSATION", "<<ARCHIVE_ANON>>"), \
             patch("src.interfaces.chat_app.app.SQL_ARCHIVE_CONVERSATION_BY_USER", "<<ARCHIVE_USER>>"):
            with app.test_request_context(
                "/api/archive_conversation", method="POST",
                data=json.dumps({"conversation_id": 1, "client_id": "c1"}),
                content_type="application/json",
            ):
                FlaskAppWrapper.archive_conversation(wrapper)

        executed = mock_cursor.execute.call_args_list[0][0][0]
        assert executed == "<<ARCHIVE_ANON>>"


# ---------------------------------------------------------------------------
# unarchive_conversation
# ---------------------------------------------------------------------------

class TestUnarchiveConversation:

    def test_happy_path_returns_200(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "unarchive_conversation",
            {"conversation_id": 4, "client_id": "abc"},
        )
        assert status == 200
        assert data["success"] is True

    def test_not_found_returns_404(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_flag(
            wrapper, "unarchive_conversation",
            {"conversation_id": 999, "client_id": "abc"},
            rowcount=0,
        )
        assert status == 404


# ---------------------------------------------------------------------------
# rename_conversation
# ---------------------------------------------------------------------------

class TestRenameConversation:

    def test_happy_path_returns_200(self):
        _, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn):
            with wrapper.app.test_request_context(
                "/api/rename_conversation", method="POST",
                data=json.dumps({"conversation_id": 5, "title": "New Name", "client_id": "abc"}),
                content_type="application/json",
            ):
                response, status = FlaskAppWrapper.rename_conversation(wrapper)

        data = response.get_json()
        assert status == 200
        assert data["success"] is True
        assert data["title"] == "New Name"

    def test_empty_title_returns_400(self):
        _, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn):
            with wrapper.app.test_request_context(
                "/api/rename_conversation", method="POST",
                data=json.dumps({"conversation_id": 5, "title": "", "client_id": "abc"}),
                content_type="application/json",
            ):
                response, status = FlaskAppWrapper.rename_conversation(wrapper)

        assert status == 400
        assert "empty" in response.get_json()["error"].lower()

    def test_title_over_100_chars_returns_400(self):
        _, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn):
            with wrapper.app.test_request_context(
                "/api/rename_conversation", method="POST",
                data=json.dumps({"conversation_id": 5, "title": "x" * 101, "client_id": "abc"}),
                content_type="application/json",
            ):
                response, status = FlaskAppWrapper.rename_conversation(wrapper)

        assert status == 400
        assert "100" in response.get_json()["error"]

    def test_not_found_returns_404(self):
        _, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn):
            with wrapper.app.test_request_context(
                "/api/rename_conversation", method="POST",
                data=json.dumps({"conversation_id": 999, "title": "Ghost", "client_id": "abc"}),
                content_type="application/json",
            ):
                response, status = FlaskAppWrapper.rename_conversation(wrapper)

        assert status == 404


# ---------------------------------------------------------------------------
# list_conversations — new fields and archived param
# ---------------------------------------------------------------------------

class TestListConversations:

    def test_returns_is_pinned_and_is_archived_fields(self):
        _, wrapper = _make_wrapper()
        rows = [
            _make_row(conv_id=1, title="Pinned Chat", is_pinned=True,  is_archived=False),
            _make_row(conv_id=2, title="Normal Chat", is_pinned=False, is_archived=False),
        ]
        data, status, _ = _call_list(wrapper, "?client_id=abc", rows=rows)
        assert status == 200
        convs = data["conversations"]
        assert len(convs) == 2
        assert convs[0]["is_pinned"] is True
        assert convs[0]["is_archived"] is False
        assert convs[1]["is_pinned"] is False

    def test_missing_client_id_returns_400(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call_list(wrapper, "?limit=10")
        assert status == 400
        assert "client_id" in data["error"].lower()

    def test_archived_true_uses_archive_sql(self):
        """?archived=true must call the archive SQL variant, not the default."""
        app, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn), \
             patch("src.interfaces.chat_app.app.SQL_LIST_ARCHIVED_CONVERSATIONS",     "<<ARCH>>"), \
             patch("src.interfaces.chat_app.app.SQL_LIST_CONVERSATIONS",               "<<MAIN>>"):
            with app.test_request_context(
                "/api/list_conversations?client_id=abc&archived=true",
                method="GET",
            ):
                FlaskAppWrapper.list_conversations(wrapper)

        executed = mock_cursor.execute.call_args_list[0][0][0]
        assert executed == "<<ARCH>>"

    def test_archived_false_uses_main_sql(self):
        """Default (no archived param) must use the normal active-conversations SQL."""
        app, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn), \
             patch("src.interfaces.chat_app.app.SQL_LIST_ARCHIVED_CONVERSATIONS",     "<<ARCH>>"), \
             patch("src.interfaces.chat_app.app.SQL_LIST_CONVERSATIONS",               "<<MAIN>>"):
            with app.test_request_context(
                "/api/list_conversations?client_id=abc",
                method="GET",
            ):
                FlaskAppWrapper.list_conversations(wrapper)

        executed = mock_cursor.execute.call_args_list[0][0][0]
        assert executed == "<<MAIN>>"

    def test_limit_capped_at_500(self):
        """limit param above 500 must be silently capped."""
        _, wrapper = _make_wrapper()
        data, status, cursor = _call_list(
            wrapper, "?client_id=abc&limit=9999", rows=[]
        )
        assert status == 200
        # The SQL call's last positional arg must be <= 500
        call_args = cursor.execute.call_args_list[0][0][1]
        assert call_args[-1] <= 500
