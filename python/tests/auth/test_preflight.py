"""req.w2.auth — CERNPreflightSource probe evaluation, offline."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from archi.auth import preflight as preflight_mod
from archi.auth.preflight import CERNPreflightSource

_FAR_FUTURE = int(datetime(2035, 1, 1, tzinfo=timezone.utc).timestamp())
_PAST = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())


def _write_cookie_file(path: Path, *, expires: int) -> None:
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".cern.ch\tTRUE\t/\tTRUE\t{expires}\tsessionid\ttest-value\n"
    )


def _write_cert(path: Path, *, not_after: datetime) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "archi-preflight-test")]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_after - timedelta(days=30))
        .not_valid_after(not_after)
        .sign(key, hashes.SHA256())
    )
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


class _FakeResponse:
    def __init__(self, status_code=200, text="", url=""):
        self.status_code = status_code
        self.text = text
        self.url = url


def _fake_http(monkeypatch, *, response=None, exc=None):
    calls = []

    class _FakeSession:
        def __init__(self):
            self.cookies = requests.cookies.RequestsCookieJar()

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            if exc is not None:
                raise exc
            return response

    monkeypatch.setattr(preflight_mod.requests, "Session", _FakeSession)
    return calls


def test_token_kind(monkeypatch):
    source = CERNPreflightSource(
        source_name="jira_token", kind="token", credential_ref="ARCHI_T_TOKEN"
    )
    monkeypatch.delenv("ARCHI_T_TOKEN", raising=False)
    assert source.preflight().status == "missing_credential"
    monkeypatch.setenv("ARCHI_T_TOKEN", "test-value")
    result = source.preflight()
    assert result.status == "ok"
    assert result.credential_refs == ("ARCHI_T_TOKEN",)


def test_file_kind(monkeypatch, tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text("x")
    source = CERNPreflightSource(
        source_name="ref_file", kind="file", credential_ref="ARCHI_T_FILE"
    )
    monkeypatch.setenv("ARCHI_T_FILE", str(tmp_path / "absent.txt"))
    assert source.preflight().status == "missing_credential"
    monkeypatch.setenv("ARCHI_T_FILE", str(path))
    assert source.preflight().status == "ok"


def test_all_env_kind(monkeypatch):
    source = CERNPreflightSource(
        source_name="amq_env",
        kind="all_env",
        credential_refs=["ARCHI_T_AMQ_USER", "ARCHI_T_AMQ_PASSWORD"],
    )
    monkeypatch.setenv("ARCHI_T_AMQ_USER", "test-value")
    monkeypatch.delenv("ARCHI_T_AMQ_PASSWORD", raising=False)
    result = source.preflight()
    assert result.status == "missing_credential"
    assert "ARCHI_T_AMQ_PASSWORD" in result.reason
    monkeypatch.setenv("ARCHI_T_AMQ_PASSWORD", "test-value")
    assert source.preflight().status == "ok"


def test_any_file_kind(monkeypatch, tmp_path):
    path = tmp_path / "b.pem"
    path.write_text("x")
    source = CERNPreflightSource(
        source_name="any_cred",
        kind="any_file",
        credential_refs=["ARCHI_T_A", "ARCHI_T_B"],
    )
    monkeypatch.delenv("ARCHI_T_A", raising=False)
    monkeypatch.delenv("ARCHI_T_B", raising=False)
    assert source.preflight().status == "missing_credential"
    monkeypatch.setenv("ARCHI_T_B", str(path))
    result = source.preflight()
    assert result.status == "ok"
    assert "ARCHI_T_B" in result.reason


def test_ca_bundle_missing_is_tls_failed(monkeypatch):
    source = CERNPreflightSource(
        source_name="ca", kind="ca_bundle", credential_ref="ARCHI_T_CA"
    )
    monkeypatch.delenv("ARCHI_T_CA", raising=False)
    result = source.preflight()
    assert result.status == "tls_failed"
    assert "refusing TLS bypass" in result.reason


def test_cache_kind(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([1, 2, 3]))
    source = CERNPreflightSource(
        source_name="cache",
        kind="cache",
        cache_paths=["a.json"],
        base=str(tmp_path),
    )
    result = source.preflight()
    assert result.status == "ok"
    assert result.record_count == 3
    missing = CERNPreflightSource(
        source_name="cache",
        kind="cache",
        cache_paths=["absent.json"],
        base=str(tmp_path),
    )
    assert missing.preflight().status == "cache_missing"


def test_sso_cookie_kind(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    source = CERNPreflightSource(
        source_name="sso",
        kind="sso_cookie",
        credential_ref="ARCHI_T_COOKIE",
        max_age_hours=24,
    )
    monkeypatch.delenv("ARCHI_T_COOKIE", raising=False)
    assert source.preflight().status == "missing_credential"
    monkeypatch.setenv("ARCHI_T_COOKIE", str(cookie_path))
    _write_cookie_file(cookie_path, expires=_FAR_FUTURE)
    result = source.preflight()
    assert result.status == "ok"
    assert result.credential_refs == ("ARCHI_T_COOKIE",)
    _write_cookie_file(cookie_path, expires=_PAST)
    result = source.preflight()
    assert result.status == "auth_failed"
    assert "expired" in result.reason


def test_x509_proxy_kind(monkeypatch, tmp_path):
    proxy_path = tmp_path / "proxy.pem"
    source = CERNPreflightSource(
        source_name="proxy", kind="x509_proxy", credential_ref="ARCHI_T_PROXY"
    )
    monkeypatch.setenv("ARCHI_T_PROXY", str(proxy_path))
    assert source.preflight().status == "missing_credential"
    _write_cert(
        proxy_path,
        not_after=datetime.now(timezone.utc) + timedelta(hours=12),
    )
    result = source.preflight()
    assert result.status == "ok"
    assert "valid until" in result.reason
    _write_cert(
        proxy_path,
        not_after=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    result = source.preflight()
    assert result.status == "auth_failed"
    assert "expired at" in result.reason


def test_x509_proxy_undecodable_is_auth_failed(monkeypatch, tmp_path):
    """An offline kind has no live-probe backstop, so an unreadable proxy
    must not pass as healthy (code-review finding, 2026-08-11)."""
    proxy_path = tmp_path / "proxy.pem"
    proxy_path.write_text("not a certificate")
    source = CERNPreflightSource(
        source_name="proxy", kind="x509_proxy", credential_ref="ARCHI_T_PROXY"
    )
    monkeypatch.setenv("ARCHI_T_PROXY", str(proxy_path))
    result = source.preflight()
    assert result.status == "auth_failed"
    assert "could not be decoded" in result.reason


def test_cern_sso_http_corrupt_cookie_is_auth_failed(monkeypatch, tmp_path):
    """A cookie file that fails to parse is a credential problem, not an
    endpoint problem — no request may be sent (code-review finding)."""
    cookie_path = tmp_path / "sso.txt"
    cookie_path.write_text("<html>this is not a cookie file</html>")
    monkeypatch.setenv("ARCHI_T_COOKIE", str(cookie_path))
    calls = _fake_http(monkeypatch, response=_FakeResponse(200, "x", "u"))
    source = CERNPreflightSource(
        source_name="sso_http",
        kind="cern_sso_http",
        credential_ref="ARCHI_T_COOKIE",
        endpoint="https://docs.example.cern.ch/",
    )
    result = source.preflight()
    assert result.status == "auth_failed"
    assert "no request was sent" in result.reason
    assert calls == []


def test_cern_sso_http_authenticated(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path, expires=_FAR_FUTURE)
    monkeypatch.setenv("ARCHI_T_COOKIE", str(cookie_path))
    calls = _fake_http(
        monkeypatch,
        response=_FakeResponse(
            200, "<h1>Protected docs</h1>", "https://docs.example.cern.ch/"
        ),
    )
    source = CERNPreflightSource(
        source_name="sso_http",
        kind="cern_sso_http",
        credential_ref="ARCHI_T_COOKIE",
        endpoint="https://docs.example.cern.ch/",
    )
    result = source.preflight()
    assert result.status == "ok"
    assert result.credential_refs == ("ARCHI_T_COOKIE",)
    assert calls and calls[0][0] == "https://docs.example.cern.ch/"


def test_cern_sso_http_login_page_is_auth_failed(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path, expires=_FAR_FUTURE)
    monkeypatch.setenv("ARCHI_T_COOKIE", str(cookie_path))
    _fake_http(
        monkeypatch,
        response=_FakeResponse(
            200,
            "<title>Sign in to CERN</title>",
            "https://auth.cern.ch/auth/realms/cern",
        ),
    )
    source = CERNPreflightSource(
        source_name="sso_http",
        kind="cern_sso_http",
        credential_ref="ARCHI_T_COOKIE",
        endpoint="https://docs.example.cern.ch/",
    )
    assert source.preflight().status == "auth_failed"


def test_cern_sso_http_error_statuses(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path, expires=_FAR_FUTURE)
    monkeypatch.setenv("ARCHI_T_COOKIE", str(cookie_path))
    source = CERNPreflightSource(
        source_name="sso_http",
        kind="cern_sso_http",
        credential_ref="ARCHI_T_COOKIE",
        endpoint="https://docs.example.cern.ch/",
    )
    _fake_http(
        monkeypatch,
        response=_FakeResponse(403, "", "https://docs.example.cern.ch/"),
    )
    assert source.preflight().status == "auth_failed"
    _fake_http(monkeypatch, exc=requests.exceptions.SSLError("handshake"))
    assert source.preflight().status == "tls_failed"
    _fake_http(
        monkeypatch, exc=requests.exceptions.ConnectionError("refused")
    )
    assert source.preflight().status == "endpoint_failed"


def test_cern_tls_kind(monkeypatch, tmp_path):
    ca_path = tmp_path / "ca.pem"
    ca_path.write_text("test-bundle")
    monkeypatch.setenv("ARCHI_T_CA", str(ca_path))
    calls = _fake_http(
        monkeypatch,
        response=_FakeResponse(
            200, "<h1>payloads</h1>", "https://svc.example.cern.ch/"
        ),
    )
    source = CERNPreflightSource(
        source_name="tls",
        kind="cern_tls",
        credential_ref="ARCHI_T_CA",
        endpoint="https://svc.example.cern.ch/",
    )
    result = source.preflight()
    assert result.status == "ok"
    assert "CA bundle" in result.reason
    assert calls[0][1]["verify"] == str(ca_path)


def test_x509_http_expired_proxy_skips_network(monkeypatch, tmp_path):
    proxy_path = tmp_path / "proxy.pem"
    _write_cert(
        proxy_path,
        not_after=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    monkeypatch.setenv("ARCHI_T_PROXY", str(proxy_path))
    calls = _fake_http(monkeypatch, response=_FakeResponse(200))
    source = CERNPreflightSource(
        source_name="x509",
        kind="x509_http",
        credential_ref="ARCHI_T_PROXY",
        endpoint="https://svc.example.cern.ch/",
    )
    result = source.preflight()
    assert result.status == "auth_failed"
    assert "expired at" in result.reason
    assert calls == []


def test_x509_http_valid_proxy(monkeypatch, tmp_path):
    proxy_path = tmp_path / "proxy.pem"
    _write_cert(
        proxy_path,
        not_after=datetime.now(timezone.utc) + timedelta(hours=12),
    )
    monkeypatch.setenv("ARCHI_T_PROXY", str(proxy_path))
    calls = _fake_http(
        monkeypatch,
        response=_FakeResponse(200, "{}", "https://svc.example.cern.ch/"),
    )
    source = CERNPreflightSource(
        source_name="x509",
        kind="x509_http",
        credential_ref="ARCHI_T_PROXY",
        endpoint="https://svc.example.cern.ch/",
    )
    assert source.preflight().status == "ok"
    assert calls[0][1]["cert"] == str(proxy_path)


def test_x509_http_ssl_error_vocabulary(monkeypatch, tmp_path):
    proxy_path = tmp_path / "proxy.pem"
    _write_cert(
        proxy_path,
        not_after=datetime.now(timezone.utc) + timedelta(hours=12),
    )
    monkeypatch.setenv("ARCHI_T_PROXY", str(proxy_path))
    source = CERNPreflightSource(
        source_name="x509",
        kind="x509_http",
        credential_ref="ARCHI_T_PROXY",
        endpoint="https://svc.example.cern.ch/",
    )
    _fake_http(
        monkeypatch,
        exc=requests.exceptions.SSLError("alert bad certificate"),
    )
    assert source.preflight().status == "auth_failed"
    _fake_http(
        monkeypatch,
        exc=requests.exceptions.SSLError("unable to get issuer cert"),
    )
    assert source.preflight().status == "tls_failed"


def test_https_kind(monkeypatch):
    source = CERNPreflightSource(
        source_name="reach",
        kind="https",
        endpoint="https://public.example.cern.ch/",
    )
    _fake_http(
        monkeypatch,
        response=_FakeResponse(
            200, "<h1>ok</h1>", "https://public.example.cern.ch/"
        ),
    )
    assert source.preflight().status == "ok"
    _fake_http(
        monkeypatch,
        response=_FakeResponse(
            200, "", "https://auth.cern.ch/auth/realms/cern"
        ),
    )
    assert source.preflight().status == "auth_failed"
    _fake_http(monkeypatch, response=_FakeResponse(500, "", ""))
    assert source.preflight().status == "endpoint_failed"


def test_unknown_kind_is_endpoint_failed():
    source = CERNPreflightSource(source_name="odd", kind="bogus")
    result = source.preflight()
    assert result.status == "endpoint_failed"
    assert "unknown preflight kind" in result.reason


def test_missing_params_raise():
    with pytest.raises(ValueError):
        CERNPreflightSource(source_name="t", kind="token").preflight()
    with pytest.raises(ValueError):
        CERNPreflightSource(source_name="h", kind="https").preflight()


def test_run_emits_no_facts(monkeypatch):
    monkeypatch.setenv("ARCHI_T_TOKEN", "test-value")
    source = CERNPreflightSource(
        source_name="jira_token", kind="token", credential_ref="ARCHI_T_TOKEN"
    )
    run = source.run("run-1")
    assert list(run.facts) == []
    assert run.health is not None
    assert run.health.status == "ok"


def test_profile_is_live_overlay():
    assert CERNPreflightSource.profile == "live_overlay"
