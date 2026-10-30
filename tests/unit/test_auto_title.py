from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.utils.title_generator import (clean_generated_title,
                                       generate_conversation_title)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('"Higgs mechanism and particle mass."', "Higgs mechanism and particle mass"),
        ("Higgs mechanism\nExtra explanation", "Higgs mechanism"),
        ("  'A useful title'  ", "A useful title"),
        ("x" * 120, "x" * 100),
        ("   \n   ", None),
    ],
)
def test_clean_generated_title(response, expected):
    assert clean_generated_title(response) == expected


def test_clean_generated_title_accepts_chat_model_message():
    response = SimpleNamespace(content="Short title.")
    assert clean_generated_title(response) == "Short title"


def test_generate_conversation_title_uses_model_response():
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(content="Higgs mechanism")

    assert (
        generate_conversation_title(model, "Explain the Higgs mechanism")
        == "Higgs mechanism"
    )
    model.invoke.assert_called_once()
    assert "Explain the Higgs mechanism" in model.invoke.call_args.args[0]


def _wrapper():
    from src.interfaces.chat_app.app import ChatWrapper

    wrapper = ChatWrapper.__new__(ChatWrapper)
    wrapper.config = {
        "services": {
            "chat_app": {"default_provider": "local", "default_model": "test-model"}
        }
    }
    wrapper.services_config = wrapper.config["services"]
    wrapper.pg_config = {"host": "unused-in-tests"}
    return wrapper


def test_model_failure_does_not_touch_database_or_raise():
    wrapper = _wrapper()
    wrapper._create_provider_llm = MagicMock(
        side_effect=RuntimeError("model unavailable")
    )

    with patch("src.interfaces.chat_app.app.psycopg2.connect") as connect:
        wrapper._generate_conversation_title(
            1, "first message...", "first message", "client", None
        )

    connect.assert_not_called()


def test_title_update_is_compare_and_set_and_preserves_renamed_title():
    wrapper = _wrapper()
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(content="Generated title")
    wrapper._create_provider_llm = MagicMock(return_value=model)
    cursor = MagicMock()
    cursor.rowcount = 0  # The title no longer matches because the user renamed it.
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=conn):
        wrapper._generate_conversation_title(
            1, "first message...", "first message", "client", None
        )

    statement, parameters = cursor.execute.call_args.args
    assert "AND title = %s" in statement
    assert parameters == ("Generated title", 1, "first message...", "client", None)
    conn.commit.assert_called_once()
    cursor.close.assert_called_once()
    conn.close.assert_called_once()


def test_auto_title_disabled_does_not_start_background_task():
    from src.interfaces.chat_app.app import ChatWrapper

    wrapper = ChatWrapper.__new__(ChatWrapper)
    wrapper.pg_config = {"host": "unused-in-tests"}
    wrapper.services_config = {"chat_app": {"auto_title": False}}
    cursor = MagicMock()
    cursor.fetchone.return_value = (42,)
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=conn), patch(
        "src.interfaces.chat_app.app.Thread"
    ) as thread:
        assert wrapper.create_conversation("first message", "client") == 42

    thread.assert_not_called()


def test_auto_title_defaults_to_enabled_and_starts_daemon_task():
    from src.interfaces.chat_app.app import ChatWrapper

    wrapper = ChatWrapper.__new__(ChatWrapper)
    wrapper.pg_config = {"host": "unused-in-tests"}
    wrapper.services_config = {"chat_app": {}}
    cursor = MagicMock()
    cursor.fetchone.return_value = (43,)
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("src.interfaces.chat_app.app.psycopg2.connect", return_value=conn), patch(
        "src.interfaces.chat_app.app.Thread"
    ) as thread:
        assert wrapper.create_conversation("first message", "client", "user") == 43

    thread.assert_called_once()
    assert thread.call_args.kwargs["daemon"] is True
    assert thread.call_args.kwargs["args"] == (
        43, "first message", "first message", "client", "user"
    )
