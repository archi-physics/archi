"""req.compops-snapshot-builder -- the moved SSO login script, no network.

The interactive Kerberos + TOTP flow needs CERN; these tests cover what runs
without it: where cookie files go, how saved cookies are filtered and
protected, and that ``--check`` fails closed when a cookie file is absent.
"""
from __future__ import annotations

import http.cookiejar
import stat

import pytest
import requests

from archi.auth.cookies import check_cookie_file
from archi.downloaders import sso_login


def test_cookie_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv(sso_login.COOKIE_DIR_ENV, raising=False)
    assert sso_login.cookie_dir() == sso_login.COOKIE_DIR
    monkeypatch.setenv(sso_login.COOKIE_DIR_ENV, str(tmp_path / "env"))
    assert sso_login.cookie_dir() == tmp_path / "env"
    assert sso_login.cookie_dir(str(tmp_path / "flag")) == tmp_path / "flag"


def _cookie(name, domain):
    return http.cookiejar.Cookie(
        0, name, "v", None, False, domain, True, domain.startswith("."), "/", True,
        True, None, False, None, None, {},
    )


def test_saved_cookies_keep_only_service_and_sso_domains(tmp_path):
    session = requests.Session()
    session.cookies.set_cookie(_cookie("svc", "hypernews.cern.ch"))
    session.cookies.set_cookie(_cookie("sso", "auth.cern.ch"))
    session.cookies.set_cookie(_cookie("other", "cmssi.docs.cern.ch"))
    total, svc = sso_login._save_service_cookies(
        session,
        "hypernews",
        sso_login.SERVICES["hypernews"]["url"],
        directory=tmp_path / "cookies",
    )
    assert (total, svc) == (2, 1)
    path = tmp_path / "cookies" / "hypernews.txt"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    jar = http.cookiejar.MozillaCookieJar(str(path))
    jar.load(ignore_discard=True, ignore_expires=True)
    assert sorted(c.name for c in jar) == ["sso", "svc"]
    # The file is in the format archi.auth.cookies reads.
    status = check_cookie_file(path, domain="hypernews.cern.ch")
    assert status.exists and status.cookie_count == 1


def test_check_without_cookie_files_exits_1(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        sso_login.main(["--check", "--cookie-dir", str(tmp_path), "hypernews", "cms_conddb"])
    assert info.value.code == 1
    out = capsys.readouterr().out
    assert "[hypernews] No cookie file" in out and "[cms_conddb] No cookie file" in out


def test_unknown_service_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as info:
        sso_login.main(["--check", "--cookie-dir", str(tmp_path), "cernbox"])
    assert info.value.code == 2
