"""The two Open WebUI calls a compose-run site needs before `okg chat sync`.

okg's `chat setup` makes both (admin account, model provider), but only for
an instance it launched itself with `okg chat-instance up`, and on okg
v0.2.0 that launch refuses every published image (its pinned image sits
under a placeholder registry). archi runs the site in compose, so it makes
these two calls and hands everything else to `okg chat sync`. Drafted AKMON
ask: let `okg chat setup` adopt an externally managed instance.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional


class WebUIError(Exception):
    pass


def _post(url: str, body: Mapping[str, Any], token: Optional[str] = None, timeout: float = 30) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode()
            return response.status, json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(text)
        except json.JSONDecodeError:
            return exc.code, text


def admin_token(base_url: str, email: str, password: str, name: str = "archi admin") -> str:
    """Sign in as the site admin, creating the account on a fresh site.

    The first account on a fresh Open WebUI becomes its admin, and signing up
    returns a token `okg chat sync` accepts, so no browser is involved.
    """
    status, body = _post(f"{base_url}/api/v1/auths/signin", {"email": email, "password": password})
    if status == 200 and isinstance(body, dict) and body.get("token"):
        return body["token"]
    status, body = _post(
        f"{base_url}/api/v1/auths/signup", {"name": name, "email": email, "password": password}
    )
    if status == 200 and isinstance(body, dict) and body.get("token"):
        if body.get("role") not in (None, "admin"):
            raise WebUIError(
                f"signed up {email} but the site made it a {body.get('role')!r}, not the admin: "
                "another account was created first. Sign in as that admin and pass its API key "
                "as OKG_CHAT_ADMIN_TOKEN"
            )
        return body["token"]
    raise WebUIError(f"could not sign in or sign up {email} at {base_url}: {status} {body}")


def configure_ollama(base_url: str, token: str, ollama_url: str) -> None:
    """Point the site at an ollama server, as the container reaches it."""
    status, body = _post(
        f"{base_url}/ollama/config/update",
        {"ENABLE_OLLAMA_API": True, "OLLAMA_BASE_URLS": [ollama_url], "OLLAMA_API_CONFIGS": {}},
        token=token,
    )
    if status != 200:
        raise WebUIError(f"could not set the ollama URL: {status} {body}")
