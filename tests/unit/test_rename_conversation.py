"""Unit tests for the rename_conversation endpoint on FlaskAppWrapper.

Strategy: guard the import of FlaskAppWrapper with importorskip — the class
lives in app.py which pulls in the full LLM stack at module level.  When the
stack is present (CI, the chat image) these tests run in full; on a minimal dev
env they are skipped rather than erroring.

Test approach follows test_ab_pending_limit.py:
  - Instantiate FlaskAppWrapper with object.__new__ to skip __init__
  - Stub only the attributes rename_conversation() touches (pg_config)
  - Patch psycopg2.connect so no real DB is needed
  - Call the method directly inside app.test_request_context
  - Assert status codes and JSON response shapes
"""
import json
from unittest.mock import MagicMock, patch

import pytest

# Guard: skip the whole module if the LLM stack is not installed
FlaskAppWrapper = pytest.importorskip(
    "src.interfaces.chat_app.app",
    reason="app.py requires the full LLM stack",
).FlaskAppWrapper

from flask import Flask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrapper():
    """Minimal FlaskAppWrapper stub — only pg_config is set."""
    app = Flask(__name__)
    app.secret_key = "test-secret"
    wrapper = object.__new__(FlaskAppWrapper)
    wrapper.app = app
    wrapper.pg_config = {
        "host": "localhost", "port": 5432,
        "database": "archi", "user": "archi", "password": "test",
    }
    return app, wrapper


def _call(wrapper, body, *, rowcount=1):
    """
    Call rename_conversation() with psycopg2 patched.
    Returns (json_payload, http_status_code, mock_cursor).
    """
    app = wrapper.app
    mock_cursor = MagicMock()
    mock_cursor.rowcount = rowcount
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("src.interfaces.chat_app.app.psycopg2.connect",
               return_value=mock_conn):
        with app.test_request_context(
            "/api/rename_conversation",
            method="POST",
            data=json.dumps(body),
            content_type="application/json",
        ):
            response, status = FlaskAppWrapper.rename_conversation(wrapper)

    return response.get_json(), status, mock_cursor


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestRenameConversationHappyPath:

    def test_returns_200_with_success_payload(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper,
            {"conversation_id": 42, "title": "My Research Chat", "client_id": "abc-123"},
        )
        assert status == 200
        assert data["success"] is True
        assert data["conversation_id"] == 42
        assert data["title"] == "My Research Chat"

    def test_title_is_stripped_of_whitespace(self):
        """Leading/trailing whitespace is trimmed; trimmed value is returned."""
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper,
            {"conversation_id": 7, "title": "  Trimmed Name  ", "client_id": "abc"},
        )
        assert status == 200
        assert data["title"] == "Trimmed Name"

    def test_anonymous_path_uses_client_id_sql(self):
        """Without a session user_id the anonymous SQL variant is executed."""
        app, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect",
                   return_value=mock_conn), \
             patch("src.interfaces.chat_app.app.SQL_RENAME_CONVERSATION",
                   "<<ANON_SQL>>"), \
             patch("src.interfaces.chat_app.app.SQL_RENAME_CONVERSATION_BY_USER",
                   "<<USER_SQL>>"):
            with app.test_request_context(
                "/api/rename_conversation",
                method="POST",
                data=json.dumps({"conversation_id": 1,
                                 "title": "T", "client_id": "c1"}),
                content_type="application/json",
            ):
                FlaskAppWrapper.rename_conversation(wrapper)

        executed_sql = mock_cursor.execute.call_args_list[0][0][0]
        assert executed_sql == "<<ANON_SQL>>"

    def test_title_exactly_100_chars_is_accepted(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper,
            {"conversation_id": 1, "title": "a" * 100, "client_id": "abc"},
        )
        assert status == 200

    def test_authenticated_path_uses_user_sql(self):
        app, wrapper = _make_wrapper()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=mock_conn), \
             patch("src.interfaces.chat_app.app.SQL_RENAME_CONVERSATION", "<<ANON_SQL>>"), \
             patch("src.interfaces.chat_app.app.SQL_RENAME_CONVERSATION_BY_USER", "<<USER_SQL>>"):
             
            with app.test_request_context(
                "/api/rename_conversation", method="POST",
                data=json.dumps({"conversation_id": 1, "title": "T", "client_id": "c1"}),
                content_type="application/json",
            ):
                from flask import session
                session["user"] = {"id": "user-9"}
                FlaskAppWrapper.rename_conversation(wrapper)

        sql, params = mock_cursor.execute.call_args_list[0][0]
        assert sql == "<<USER_SQL>>"
        assert params == ("T", 1, "user-9", "c1")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestRenameConversationValidation:

    def test_missing_conversation_id_returns_400(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper, {"title": "Some Title", "client_id": "abc"},
        )
        assert status == 400
        assert "conversation_id" in data["error"].lower()

    def test_empty_title_returns_400(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper, {"conversation_id": 1, "title": "", "client_id": "abc"},
        )
        assert status == 400
        assert "empty" in data["error"].lower()

    def test_whitespace_only_title_returns_400(self):
        """A title that trims to empty is still rejected."""
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper, {"conversation_id": 1, "title": "   ", "client_id": "abc"},
        )
        assert status == 400
        assert "empty" in data["error"].lower()

    def test_title_over_100_chars_returns_400(self):
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper,
            {"conversation_id": 1, "title": "x" * 101, "client_id": "abc"},
        )
        assert status == 400
        assert "100" in data["error"]

    def test_missing_client_id_and_no_session_returns_400(self):
        """No ownership token at all must be rejected before hitting the DB."""
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper, {"conversation_id": 1, "title": "Title"},
        )
        assert status == 400
        assert "client_id" in data["error"].lower()


# ---------------------------------------------------------------------------
# Not-found test
# ---------------------------------------------------------------------------

class TestRenameConversationNotFound:

    def test_rowcount_zero_returns_404(self):
        """rowcount == 0 means the row doesn't exist or isn't owned by caller."""
        _, wrapper = _make_wrapper()
        data, status, _ = _call(
            wrapper,
            {"conversation_id": 999, "title": "Ghost Chat", "client_id": "abc"},
            rowcount=0,
        )
        assert status == 404
        assert "not found" in data["error"].lower()
