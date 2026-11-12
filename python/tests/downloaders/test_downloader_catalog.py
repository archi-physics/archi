"""The shared helper for CMS catalog downloaders (issue #903), no network.

Every catalog downloader fetches over HTTPS, sometimes with an X.509 proxy,
and writes caches that a reader claims a complete scope over. These tests
pin the helper's contract: a failed fetch, an empty payload or a payload the
reader refuses never replaces a cache, and a group of files is replaced
together or not at all.
"""
from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from support.responses import Attempts, FakeResponse, patch_session_get
from support.x509 import write_cert

from archi.downloaders import _catalog
from archi.downloaders._catalog import (
    CatalogClient,
    DownloadError,
    PartialWriteError,
    data_root,
    keyed_objects,
    reader_check,
    require_proxy,
    run_cli,
    write_group_atomic,
    write_json_atomic,
)

URL = "https://catalog.example/api/thing/query/?json"


@pytest.fixture(autouse=True)
def sleeps(tmp_path, monkeypatch):
    """No grid CA directory, and backoff sleeps recorded instead of slept."""
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path / "no-grid-ca")
    recorded: list[float] = []
    monkeypatch.setattr(_catalog.time, "sleep", recorded.append)
    return recorded


# --- data_root ----------------------------------------------------------


def test_data_root_prefers_the_explicit_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "env"))
    assert data_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_data_root_reads_archi_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "env"))
    assert data_root(None) == tmp_path / "env"


def test_data_root_falls_back_to_the_current_directory(tmp_path, monkeypatch):
    # Same rule as every other archi downloader and as the readers.
    monkeypatch.delenv("ARCHI_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert data_root(None) == tmp_path


# --- write_json_atomic / write_group_atomic -----------------------------


def test_write_json_atomic_writes_sorted_json(tmp_path):
    path = tmp_path / "deep" / "cache.json"
    write_json_atomic(path, {"b": 1, "a": {"y": 2, "x": 1}})
    assert path.read_text(encoding="utf-8") == '{"a": {"x": 1, "y": 2}, "b": 1}\n'
    assert [p.name for p in path.parent.iterdir()] == ["cache.json"]
    assert [p.name for p in tmp_path.iterdir()] == ["deep"]


def test_cache_files_are_world_readable_under_the_umask(tmp_path):
    old = os.umask(0o022)
    try:
        write_json_atomic(tmp_path / "a.json", {"k": 1})
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "a.json").stat().st_mode) == 0o644
    old = os.umask(0o027)
    try:
        write_json_atomic(tmp_path / "b.json", {"k": 1})
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "b.json").stat().st_mode) == 0o640


def test_each_staged_file_is_fsynced_before_any_is_moved(tmp_path, monkeypatch):
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("fsync")
        real_fsync(fd)

    def replace(src, dst):
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(_catalog.os, "fsync", fsync)
    monkeypatch.setattr(_catalog.os, "replace", replace)
    write_group_atomic(tmp_path / "g", {"a.json": {"k": 1}, "b.json": {"k": 2}})
    assert events == ["fsync", "fsync", "replace", "replace"]


@pytest.mark.parametrize("payload", [{}, [], None, "text"], ids=["dict", "list", "none", "str"])
def test_an_empty_or_non_container_payload_is_refused(tmp_path, payload):
    path = tmp_path / "cache.json"
    path.write_text('{"old": 1}\n', encoding="utf-8")
    with pytest.raises(DownloadError, match="cache.json"):
        write_json_atomic(path, payload)
    assert path.read_text(encoding="utf-8") == '{"old": 1}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]


@pytest.mark.parametrize(
    "payload, message",
    [({"bad": object()}, "not JSON-serialisable"), ({"bad": "\ud800"}, "not encodable")],
    ids=["object", "lone-surrogate"],
)
def test_a_payload_that_cannot_be_written_is_a_download_error(tmp_path, payload, message):
    target = tmp_path / "group"
    target.mkdir()
    (target / "a.json").write_text('{"old": "a"}\n')
    with pytest.raises(DownloadError, match=message) as info:
        write_group_atomic(target, {"a.json": {"new": "a"}, "b.json": payload})
    assert "b.json" in str(info.value)
    assert json.loads((target / "a.json").read_text()) == {"old": "a"}
    assert sorted(p.name for p in target.iterdir()) == ["a.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["group"]


def test_write_group_atomic_replaces_nothing_when_one_payload_is_empty(tmp_path):
    (tmp_path / "a.json").write_text('{"old": "a"}\n')
    (tmp_path / "b.json").write_text('{"old": "b"}\n')
    with pytest.raises(DownloadError, match="b.json"):
        write_group_atomic(tmp_path, {"a.json": {"new": "a"}, "b.json": {}})
    assert json.loads((tmp_path / "a.json").read_text()) == {"old": "a"}
    assert json.loads((tmp_path / "b.json").read_text()) == {"old": "b"}


def test_write_group_atomic_refuses_an_empty_group_and_a_nested_name(tmp_path):
    with pytest.raises(DownloadError, match="no files"):
        write_group_atomic(tmp_path / "g", {})
    with pytest.raises(DownloadError, match="plain file name"):
        write_group_atomic(tmp_path / "g", {"../escape.json": {"k": 1}})
    assert list(tmp_path.iterdir()) == []


def test_the_check_sees_every_file_under_its_real_name_before_any_is_moved(tmp_path):
    target = tmp_path / "group"
    target.mkdir()
    (target / "a.json").write_text('{"old": "a"}\n')
    seen = {}

    def check(staging: Path) -> int:
        assert staging.parent == tmp_path and staging != target
        seen.update({p.name: json.loads(p.read_text()) for p in staging.iterdir()})
        assert json.loads((target / "a.json").read_text()) == {"old": "a"}
        return 7

    assert write_group_atomic(target, {"a.json": {"new": "a"}, "b.json": [1]}, check=check) == 7
    assert seen == {"a.json": {"new": "a"}, "b.json": [1]}
    assert json.loads((target / "a.json").read_text()) == {"new": "a"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["group"]


def test_a_refusing_check_leaves_the_old_files_and_no_staging_directory(tmp_path):
    target = tmp_path / "group"
    target.mkdir()
    (target / "a.json").write_text('{"old": "a"}\n')
    staged = []

    def check(staging: Path) -> None:
        staged.append(staging)
        raise DownloadError(f"{staging / 'a.json'} is no good")

    with pytest.raises(DownloadError) as info:
        write_group_atomic(target, {"a.json": {"new": "a"}}, check=check)
    # The message points at the target, not at the deleted staging copy.
    assert str(info.value) == f"{target / 'a.json'} is no good"
    assert (target / "a.json").read_text() == '{"old": "a"}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["group"]
    assert not staged[0].exists()


def test_a_refused_check_on_a_new_target_creates_no_directory(tmp_path):
    def check(staging: Path) -> None:
        raise DownloadError("no")

    with pytest.raises(DownloadError):
        write_group_atomic(tmp_path / "new", {"a.json": {"k": 1}}, check=check)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_move_after_an_earlier_one_names_the_replaced_files(tmp_path, monkeypatch):
    target = tmp_path / "group"
    real_replace = os.replace
    moves = []

    def flaky_replace(src, dst):
        moves.append(dst)
        if len(moves) == 2:
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(_catalog.os, "replace", flaky_replace)
    with pytest.raises(PartialWriteError) as info:
        write_group_atomic(target, {"a.json": {"k": 1}, "b.json": {"k": 2}})
    message = str(info.value)
    assert "a.json" in message and "b.json" in message
    assert "mixed" in message
    assert sorted(p.name for p in tmp_path.iterdir()) == ["group"]


def test_a_failed_first_move_is_a_plain_download_error(tmp_path, monkeypatch):
    def failing_replace(src, dst):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(_catalog.os, "replace", failing_replace)
    with pytest.raises(OSError) as info:
        write_group_atomic(tmp_path / "group", {"a.json": {"k": 1}})
    assert not isinstance(info.value, PartialWriteError)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["group"]


# --- reader_check -------------------------------------------------------


def _reader(status="ok", completed=True, count=3, reason="", raises=None):
    def make(root: Path):
        def run(run_id, *, mode):
            assert mode == "scope_complete"
            if raises:
                raise raises
            return SimpleNamespace(
                facts=iter(()),
                completed_scope=completed,
                health=SimpleNamespace(status=status, reason=reason, record_count=count),
            )

        return SimpleNamespace(run=run)

    return make


def test_reader_check_returns_the_record_count_of_an_accepted_cache(tmp_path):
    assert reader_check(_reader(count=12))(tmp_path) == 12


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(status="endpoint_failed", reason="x.json holds nothing"), "health endpoint_failed"),
        (dict(completed=False, reason="1 skipped in x.json"), "completed_scope=False"),
        (dict(count=0), "0 records"),
        (dict(raises=KeyError("boom")), "KeyError"),
    ],
    ids=["unhealthy", "incomplete", "no-records", "crash"],
)
def test_reader_check_refuses_what_the_reader_does_not_accept(tmp_path, kwargs, expected):
    with pytest.raises(DownloadError, match=expected):
        reader_check(_reader(**kwargs))(tmp_path)


def test_reader_check_names_the_endpoint_behind_the_refused_file(tmp_path):
    check = reader_check(
        _reader(status="endpoint_failed", reason=f"{tmp_path}/x.json holds nothing"),
        origins={"x.json": "https://a.example/x", "y.json": "https://a.example/y"},
    )
    with pytest.raises(DownloadError) as info:
        check(tmp_path)
    assert "x.json came from https://a.example/x" in str(info.value)
    assert "https://a.example/y" not in str(info.value)


def test_reader_check_names_every_endpoint_when_no_file_is_named(tmp_path):
    check = reader_check(
        _reader(status="endpoint_failed", reason="drifted payload"),
        origins={"x.json": "https://a.example/x", "y.json": "https://a.example/y"},
    )
    with pytest.raises(DownloadError) as info:
        check(tmp_path)
    assert "https://a.example/x" in str(info.value)
    assert "https://a.example/y" in str(info.value)


# --- keyed_objects ------------------------------------------------------


def test_keyed_objects_accepts_a_name_to_object_answer():
    assert keyed_objects({"A": {"k": 1}}, URL) == {"A": {"k": 1}}


@pytest.mark.parametrize(
    "payload, message",
    [
        ([{"name": "A"}], "a list, not a JSON object"),
        ({"error": "internal"}, "API error body"),
        ({"errors": ["bad"]}, "API error body"),
        ({}, "empty"),
        ({"A": "text"}, "entry A is a str"),
    ],
    ids=["list", "error", "errors", "empty", "non-object"],
)
def test_keyed_objects_refuses_other_answers_naming_the_url(payload, message):
    with pytest.raises(DownloadError, match=message) as info:
        keyed_objects(payload, URL)
    assert URL in str(info.value)


# --- CatalogClient ------------------------------------------------------


def test_get_json_returns_the_parsed_body_without_a_credential(monkeypatch):
    upstream = patch_session_get(monkeypatch, {URL: {"T2_XX_Example": {}}})
    client = CatalogClient(timeout=7)
    assert client.get_json(URL) == {"T2_XX_Example": {}}
    [(url, kwargs)] = upstream.calls
    assert url == URL
    assert kwargs["timeout"] == 7
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"] == {"Accept": "application/json"}
    assert "cert" not in kwargs
    assert kwargs["verify"] is True


def test_get_returns_the_raw_response_for_non_json_upstreams(monkeypatch):
    xml = "<results><DOWNTIME/></results>"
    upstream = patch_session_get(monkeypatch, {URL: FakeResponse(200, xml)})
    response = CatalogClient(timeout=7).get(URL, accept="application/xml")
    assert response.content == xml.encode()
    assert upstream.calls[0][1]["headers"] == {"Accept": "application/xml"}


def test_one_session_serves_every_request(monkeypatch):
    upstream = patch_session_get(monkeypatch, {URL: [1], URL + "&b": [2]})
    sessions = []

    def recording_get(self, url, **kwargs):
        sessions.append(self)
        return upstream(self, url, **kwargs)

    monkeypatch.setattr(requests.Session, "get", recording_get)
    with CatalogClient(timeout=7) as client:
        assert client.get_json(URL) == [1]
        assert client.get_json(URL + "&b") == [2]
    assert len(sessions) == 2 and sessions[0] is sessions[1]


def test_the_proxy_is_sent_as_cert_and_key(tmp_path, monkeypatch):
    upstream = patch_session_get(monkeypatch, {URL: [1]})
    proxy = tmp_path / "x509up"
    CatalogClient(timeout=7, proxy=proxy).get_json(URL)
    assert upstream.calls[0][1]["cert"] == (str(proxy), str(proxy))


def test_the_grid_ca_directory_is_used_when_it_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path)
    upstream = patch_session_get(monkeypatch, {URL: [1]})
    CatalogClient(timeout=7).get_json(URL)
    assert upstream.calls[0][1]["verify"] == str(tmp_path)


def test_an_explicit_ca_path_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path)
    upstream = patch_session_get(monkeypatch, {URL: [1]})
    CatalogClient(timeout=7, ca_path="/etc/pki/custom.pem").get_json(URL)
    assert upstream.calls[0][1]["verify"] == "/etc/pki/custom.pem"


@pytest.mark.parametrize(
    "first",
    [
        lambda: FakeResponse(503, "Service Unavailable"),
        lambda: requests.ConnectionError("connection reset"),
        lambda: requests.ReadTimeout("read timed out"),
    ],
    ids=["http-503", "connection-error", "timeout"],
)
def test_a_transient_failure_is_retried_with_backoff(monkeypatch, sleeps, first):
    upstream = patch_session_get(
        monkeypatch, {URL: Attempts([first(), first(), FakeResponse(200, "[1]")])}
    )
    assert CatalogClient(timeout=7).get_json(URL) == [1]
    assert len(upstream.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_retries_are_bounded(monkeypatch, sleeps):
    upstream = patch_session_get(monkeypatch, {URL: FakeResponse(502, "Bad Gateway")})
    with pytest.raises(DownloadError, match="HTTP 502 after 3 attempts") as info:
        CatalogClient(timeout=7).get_json(URL)
    assert URL in str(info.value)
    assert len(upstream.calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize(
    "answer",
    [
        FakeResponse(302, "", {"Location": "/accounts/login/"}),
        FakeResponse(403, "Forbidden"),
        FakeResponse(404, "Not Found"),
        requests.exceptions.SSLError("certificate verify failed"),
    ],
    ids=["redirect", "forbidden", "not-found", "tls"],
)
def test_a_permanent_failure_is_not_retried(monkeypatch, sleeps, answer):
    upstream = patch_session_get(monkeypatch, {URL: answer})
    with pytest.raises(DownloadError):
        CatalogClient(timeout=7).get_json(URL)
    assert len(upstream.calls) == 1
    assert sleeps == []


def test_an_http_error_names_the_url_not_the_proxy(tmp_path, monkeypatch):
    patch_session_get(monkeypatch, {URL: FakeResponse(403, "Forbidden")})
    proxy = tmp_path / "x509up_secret_name"
    with pytest.raises(DownloadError) as info:
        CatalogClient(timeout=7, proxy=proxy).get_json(URL)
    assert URL in str(info.value)
    assert "HTTP 403" in str(info.value)
    assert "x509up_secret_name" not in str(info.value)


def test_a_redirect_to_login_is_reported_as_needing_a_credential(monkeypatch):
    patch_session_get(
        monkeypatch,
        {URL: FakeResponse(302, "", {"Location": "/accounts/login/?next=/api/thing/"})},
    )
    with pytest.raises(DownloadError) as info:
        CatalogClient(timeout=7).get_json(URL)
    message = str(info.value)
    assert URL in message and "HTTP 302" in message and "/accounts/login/" in message


def test_a_tls_failure_hints_at_the_ca_options(monkeypatch):
    patch_session_get(monkeypatch, {URL: requests.exceptions.SSLError("certificate verify failed")})
    with pytest.raises(DownloadError) as info:
        CatalogClient(timeout=7).get_json(URL)
    message = str(info.value)
    assert URL in message
    assert "--ca-path" in message and "REQUESTS_CA_BUNDLE" in message


def test_a_non_json_body_is_refused(monkeypatch):
    patch_session_get(monkeypatch, {URL: FakeResponse(200, "<html>Sign in</html>")})
    with pytest.raises(DownloadError, match="not JSON") as info:
        CatalogClient(timeout=7).get_json(URL)
    assert URL in str(info.value)


# --- require_proxy ------------------------------------------------------


def test_require_proxy_returns_a_valid_proxy_from_x509_user_proxy(tmp_path):
    proxy = tmp_path / "proxy.pem"
    write_cert(proxy, not_after=datetime.now(timezone.utc) + timedelta(hours=12))
    assert require_proxy({"X509_USER_PROXY": str(proxy)}) == proxy


def test_require_proxy_defaults_to_the_voms_location(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "DEFAULT_PROXY_DIR", tmp_path)
    proxy = tmp_path / f"x509up_u{os.getuid()}"
    write_cert(proxy, not_after=datetime.now(timezone.utc) + timedelta(hours=12))
    assert require_proxy({}) == proxy


def test_require_proxy_refuses_a_missing_proxy(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(DownloadError) as info:
        require_proxy({"X509_USER_PROXY": str(missing)})
    assert str(missing) in str(info.value)
    assert "voms-proxy-init -voms cms" in str(info.value)


def test_require_proxy_refuses_an_expired_proxy(tmp_path):
    proxy = tmp_path / "proxy.pem"
    write_cert(proxy, not_after=datetime.now(timezone.utc) - timedelta(hours=1))
    with pytest.raises(DownloadError, match="expired") as info:
        require_proxy({"X509_USER_PROXY": str(proxy)})
    assert "voms-proxy-init -voms cms" in str(info.value)


def test_require_proxy_refuses_an_unreadable_proxy(tmp_path):
    proxy = tmp_path / "proxy.pem"
    proxy.write_text("not a certificate")
    with pytest.raises(DownloadError, match="could not be read") as info:
        require_proxy({"X509_USER_PROXY": str(proxy)})
    assert "voms-proxy-init -voms cms" in str(info.value)


# --- run_cli ------------------------------------------------------------


def test_run_cli_prints_the_summary_and_exits_zero(capsys):
    assert run_cli("thing", lambda: {"output": "/x", "files": {"a.json": 2}}) == 0
    assert json.loads(capsys.readouterr().out) == {"files": {"a.json": 2}, "output": "/x"}


@pytest.mark.parametrize(
    "error, prefix",
    [
        (DownloadError("GET u returned HTTP 500"), "thing download failed, cache not written: "),
        (OSError(13, "Permission denied"), "thing download failed, cache not written: "),
        (PartialWriteError("replaced a.json"), "thing download failed part-way: "),
    ],
    ids=["download", "os", "partial"],
)
def test_run_cli_reports_a_failure_and_exits_one(capsys, error, prefix):
    def download():
        raise error

    assert run_cli("thing", download) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(prefix)
    assert str(error) in captured.err
