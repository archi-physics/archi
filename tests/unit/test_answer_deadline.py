import pytest

from src.evaluation.qa import runtime


def test_deadline_passes_within_budget():
    cb = runtime.AnswerDeadlineCallback(60)
    cb.on_tool_start({}, "x")
    cb.on_chat_model_start({}, [])


def test_deadline_raises_after_budget(monkeypatch):
    cb = runtime.AnswerDeadlineCallback(900)
    monkeypatch.setattr(runtime, "perf_counter", lambda: cb.started + 901)
    with pytest.raises(runtime.AnswerTimeLimitExceeded, match="900 s exceeded"):
        cb.on_tool_start({}, "x")
    with pytest.raises(runtime.AnswerTimeLimitExceeded):
        cb.on_chat_model_start({}, [])


def test_deadline_callback_propagates_errors():
    assert runtime.AnswerDeadlineCallback.raise_error is True
