"""Stand-ins for ``requests`` answers in downloader tests (no network)."""
from __future__ import annotations

import json
from typing import Any, Callable

import requests


class FakeResponse:
    """Just what :class:`archi.downloaders._catalog.CatalogClient` reads."""

    def __init__(self, status_code: int = 200, body: str = "", headers: dict | None = None):
        self.status_code = status_code
        self.text = body
        self.content = body.encode("utf-8")
        self.headers = headers or {}

    def json(self) -> Any:
        return json.loads(self.text)


class Attempts(list):
    """One answer per attempt, in order, for a URL that is retried."""


def patch_session_get(monkeypatch, answers: dict[str, Any]) -> Callable:
    """Answer ``requests.Session.get`` from a URL map; record every call.

    A value is returned as JSON with status 200, unless it is a
    :class:`FakeResponse` (returned as is), an exception (raised), or
    :class:`Attempts` of those (one per attempt, in order). An unknown URL
    answers 404.
    """
    calls: list[tuple[str, dict]] = []

    def fake_get(self, url, **kwargs):
        calls.append((url, kwargs))
        answer = answers.get(url, FakeResponse(404, "not found"))
        if isinstance(answer, Attempts):
            answer = answer.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, FakeResponse):
            return answer
        return FakeResponse(200, json.dumps(answer))

    monkeypatch.setattr(requests.Session, "get", fake_get)
    fake_get.answers = answers  # type: ignore[attr-defined]
    fake_get.calls = calls  # type: ignore[attr-defined]
    return fake_get
