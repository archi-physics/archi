"""The shared helper for CMS catalog downloaders (issue #903), no network.

Every catalog downloader fetches JSON over HTTPS, sometimes with an X.509
proxy, and writes caches that a reader claims a complete scope over. These
tests pin the helper's contract: a failed fetch or an empty payload never
replaces a cache, and a group of files is replaced all together or not at
all.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from archi.downloaders import _catalog
from archi.downloaders._catalog import (
    DownloadError,
    data_root,
    get_json,
    require_proxy,
    write_group_atomic,
    write_json_atomic,
)


def _write_cert(path: Path, *, not_after: datetime) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "archi-proxy-test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_after - timedelta(days=1))
        .not_valid_after(not_after)
        .sign(key, hashes.SHA256())
    )
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


class _Response:
    def __init__(self, status_code=200, body="", headers=None):
        self.status_code = status_code
        self.text = body
        self.headers = headers or {}

    def json(self):
        return json.loads(self.text)


class _Session:
    """Records each GET and answers from a URL -> response map."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        answer = self.answers[url]
        if isinstance(answer, Exception):
            raise answer
        return answer


URL = "https://catalog.example/api/thing/query/?json"


# --- data_root ----------------------------------------------------------


def test_data_root_prefers_the_explicit_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "env"))
    assert data_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_data_root_reads_archi_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "env"))
    assert data_root(None) == tmp_path / "env"


def test_data_root_falls_back_to_the_current_directory(tmp_path, monkeypatch):
    # Same rule as every other archi downloader (cmssst, jira, ...).
    monkeypatch.delenv("ARCHI_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert data_root(None) == tmp_path


# --- write_json_atomic --------------------------------------------------


def test_write_json_atomic_writes_sorted_json(tmp_path):
    path = tmp_path / "deep" / "cache.json"
    write_json_atomic(path, {"b": 1, "a": {"y": 2, "x": 1}})
    assert path.read_text(encoding="utf-8") == '{"a": {"x": 1, "y": 2}, "b": 1}\n'
    assert [p.name for p in path.parent.iterdir()] == ["cache.json"]


@pytest.mark.parametrize("payload", [{}, [], None, "text"], ids=["dict", "list", "none", "str"])
def test_write_json_atomic_refuses_an_empty_or_non_container_payload(tmp_path, payload):
    path = tmp_path / "cache.json"
    path.write_text('{"old": 1}\n', encoding="utf-8")
    with pytest.raises(DownloadError, match="cache.json"):
        write_json_atomic(path, payload)
    assert path.read_text(encoding="utf-8") == '{"old": 1}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]


def test_write_json_atomic_leaves_the_old_file_when_serialising_fails(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text('{"old": 1}\n', encoding="utf-8")
    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": object()})
    assert path.read_text(encoding="utf-8") == '{"old": 1}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]


# --- write_group_atomic -------------------------------------------------


def test_write_group_atomic_writes_every_file(tmp_path):
    write_group_atomic(tmp_path / "group", {"a.json": {"k": 1}, "b.json": [1]})
    assert json.loads((tmp_path / "group" / "a.json").read_text()) == {"k": 1}
    assert json.loads((tmp_path / "group" / "b.json").read_text()) == [1]
    assert sorted(p.name for p in (tmp_path / "group").iterdir()) == ["a.json", "b.json"]


def test_write_group_atomic_replaces_nothing_when_one_payload_is_empty(tmp_path):
    (tmp_path / "a.json").write_text('{"old": "a"}\n')
    (tmp_path / "b.json").write_text('{"old": "b"}\n')
    with pytest.raises(DownloadError, match="b.json"):
        write_group_atomic(tmp_path, {"a.json": {"new": "a"}, "b.json": {}})
    assert json.loads((tmp_path / "a.json").read_text()) == {"old": "a"}
    assert json.loads((tmp_path / "b.json").read_text()) == {"old": "b"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.json"]


def test_write_group_atomic_replaces_nothing_when_one_payload_fails_to_serialise(tmp_path):
    (tmp_path / "a.json").write_text('{"old": "a"}\n')
    with pytest.raises(TypeError):
        write_group_atomic(tmp_path, {"a.json": {"new": "a"}, "b.json": {"x": object()}})
    assert json.loads((tmp_path / "a.json").read_text()) == {"old": "a"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json"]


def test_write_group_atomic_refuses_an_empty_group_and_a_nested_name(tmp_path):
    with pytest.raises(DownloadError, match="no files"):
        write_group_atomic(tmp_path, {})
    with pytest.raises(DownloadError, match="plain file name"):
        write_group_atomic(tmp_path, {"../escape.json": {"k": 1}})
    assert list(tmp_path.iterdir()) == []


# --- get_json -----------------------------------------------------------


def test_get_json_returns_the_parsed_body_without_a_credential(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path / "absent")
    session = _Session({URL: _Response(200, '{"T2_XX_Example": {}}')})
    assert get_json(URL, timeout=7, session=session) == {"T2_XX_Example": {}}
    [(url, kwargs)] = session.calls
    assert url == URL
    assert kwargs["timeout"] == 7
    assert kwargs["allow_redirects"] is False
    assert "cert" not in kwargs
    assert kwargs["verify"] is True


def test_get_json_sends_the_proxy_as_cert_and_key(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path / "absent")
    proxy = tmp_path / "x509up"
    session = _Session({URL: _Response(200, "[1]")})
    get_json(URL, timeout=7, proxy=proxy, session=session)
    [(_, kwargs)] = session.calls
    assert kwargs["cert"] == (str(proxy), str(proxy))


def test_get_json_verifies_with_the_grid_ca_directory_when_it_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path)
    session = _Session({URL: _Response(200, "[1]")})
    get_json(URL, timeout=7, session=session)
    assert session.calls[0][1]["verify"] == str(tmp_path)


def test_get_json_prefers_an_explicit_ca_path(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path)
    session = _Session({URL: _Response(200, "[1]")})
    get_json(URL, timeout=7, ca_path="/etc/pki/custom.pem", session=session)
    assert session.calls[0][1]["verify"] == "/etc/pki/custom.pem"


def test_get_json_uses_requests_get_when_no_session_is_given(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _Response(200, '{"k": 1}')

    monkeypatch.setattr(requests, "get", fake_get)
    assert get_json(URL, timeout=7) == {"k": 1}
    assert calls == [URL]


def test_get_json_names_the_url_on_an_http_error(tmp_path):
    proxy = tmp_path / "x509up_secret_name"
    session = _Session({URL: _Response(503, "Service Unavailable")})
    with pytest.raises(DownloadError) as info:
        get_json(URL, timeout=7, proxy=proxy, session=session)
    assert URL in str(info.value)
    assert "HTTP 503" in str(info.value)
    assert "x509up_secret_name" not in str(info.value)


def test_get_json_reports_a_redirect_to_login_as_needing_a_credential():
    session = _Session({
        URL: _Response(302, "", {"Location": "/accounts/login/?next=/api/thing/"}),
    })
    with pytest.raises(DownloadError) as info:
        get_json(URL, timeout=7, session=session)
    message = str(info.value)
    assert URL in message
    assert "HTTP 302" in message
    assert "/accounts/login/" in message


def test_get_json_refuses_a_non_json_body():
    session = _Session({URL: _Response(200, "<html>Sign in</html>")})
    with pytest.raises(DownloadError, match="not JSON") as info:
        get_json(URL, timeout=7, session=session)
    assert URL in str(info.value)


def test_get_json_wraps_a_transport_error():
    session = _Session({URL: requests.ConnectionError("connection refused")})
    with pytest.raises(DownloadError) as info:
        get_json(URL, timeout=7, session=session)
    assert URL in str(info.value)
    assert "ConnectionError" in str(info.value)


# --- require_proxy ------------------------------------------------------


def test_require_proxy_returns_a_valid_proxy_from_x509_user_proxy(tmp_path):
    proxy = tmp_path / "proxy.pem"
    _write_cert(proxy, not_after=datetime.now(timezone.utc) + timedelta(hours=12))
    assert require_proxy({"X509_USER_PROXY": str(proxy)}) == proxy


def test_require_proxy_defaults_to_the_voms_location(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "DEFAULT_PROXY_DIR", tmp_path)
    proxy = tmp_path / f"x509up_u{os.getuid()}"
    _write_cert(proxy, not_after=datetime.now(timezone.utc) + timedelta(hours=12))
    assert require_proxy({}) == proxy


def test_require_proxy_refuses_a_missing_proxy(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(DownloadError) as info:
        require_proxy({"X509_USER_PROXY": str(missing)})
    assert str(missing) in str(info.value)
    assert "voms-proxy-init -voms cms" in str(info.value)


def test_require_proxy_refuses_an_expired_proxy(tmp_path):
    proxy = tmp_path / "proxy.pem"
    _write_cert(proxy, not_after=datetime.now(timezone.utc) - timedelta(hours=1))
    with pytest.raises(DownloadError, match="expired") as info:
        require_proxy({"X509_USER_PROXY": str(proxy)})
    assert "voms-proxy-init -voms cms" in str(info.value)


def test_require_proxy_refuses_an_unreadable_proxy(tmp_path):
    proxy = tmp_path / "proxy.pem"
    proxy.write_text("not a certificate")
    with pytest.raises(DownloadError, match="could not be read") as info:
        require_proxy({"X509_USER_PROXY": str(proxy)})
    assert "voms-proxy-init -voms cms" in str(info.value)
