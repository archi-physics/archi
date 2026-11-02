"""req.w2.auth — cookie-file parse/expiry helpers, offline."""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from archi.auth import cookies

_FAR_FUTURE = int(datetime(2035, 1, 1, tzinfo=timezone.utc).timestamp())
_PAST = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())


def _write_cookie_file(path: Path, rows: list[tuple[str, str, int | str]]):
    lines = ["# Netscape HTTP Cookie File"]
    for domain, name, expires in rows:
        lines.append(
            f"{domain}\tTRUE\t/\tTRUE\t{expires}\t{name}\ttest-value"
        )
    path.write_text("\n".join(lines) + "\n")


def test_load_cookie_jar_keeps_expired_cookies(tmp_path):
    path = tmp_path / "c.txt"
    _write_cookie_file(
        path,
        [(".cern.ch", "live", _FAR_FUTURE), (".cern.ch", "dead", _PAST)],
    )
    jar = cookies.load_cookie_jar(path)
    assert sorted(c.name for c in jar) == ["dead", "live"]


def test_check_missing_file(tmp_path):
    status = cookies.check_cookie_file(tmp_path / "absent.txt")
    assert not status.exists
    assert not status.fresh
    assert "missing" in status.reason


def test_check_unparseable_file(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("this is not a cookie file\n")
    status = cookies.check_cookie_file(path)
    assert status.exists
    assert not status.fresh
    assert "could not be parsed" in status.reason


def test_check_fresh_cookie_file(tmp_path):
    path = tmp_path / "c.txt"
    _write_cookie_file(
        path,
        [(".cern.ch", "live", _FAR_FUTURE), (".cern.ch", "dead", _PAST)],
    )
    status = cookies.check_cookie_file(path)
    assert status.fresh
    assert status.cookie_count == 2
    assert status.live_count == 1
    assert status.expired_count == 1
    assert status.earliest_expiry == datetime.fromtimestamp(
        _FAR_FUTURE, tz=timezone.utc
    )


def test_check_all_expired(tmp_path):
    path = tmp_path / "c.txt"
    _write_cookie_file(path, [(".cern.ch", "dead", _PAST)])
    status = cookies.check_cookie_file(path)
    assert not status.fresh
    assert "expired" in status.reason


def test_check_domain_filter(tmp_path):
    path = tmp_path / "c.txt"
    _write_cookie_file(path, [(".cern.ch", "live", _FAR_FUTURE)])
    status = cookies.check_cookie_file(path, domain="example.org")
    assert not status.fresh
    assert "no cookies" in status.reason


def test_check_max_age(tmp_path):
    path = tmp_path / "c.txt"
    _write_cookie_file(path, [(".cern.ch", "live", _FAR_FUTURE)])
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).timestamp()
    os.utime(path, (old, old))
    stale = cookies.check_cookie_file(path, max_age=timedelta(hours=12))
    assert not stale.fresh
    assert "older than max age" in stale.reason
    assert cookies.check_cookie_file(path, max_age=timedelta(hours=72)).fresh


def test_session_cookies_count_as_live(tmp_path):
    path = tmp_path / "c.txt"
    _write_cookie_file(path, [(".cern.ch", "session", "")])
    status = cookies.check_cookie_file(path)
    assert status.fresh
    assert status.session_count == 1
    assert status.earliest_expiry is None


def test_looks_like_login_url():
    assert cookies.looks_like_login_url(
        "https://auth.cern.ch/auth/realms/cern"
    )
    assert cookies.looks_like_login_url("https://example.cern.ch/login")
    assert cookies.looks_like_login_url("https://x.cern.ch/sso/redirect")
    assert not cookies.looks_like_login_url(
        "https://cms-conddb.cern.ch/cmsDbBrowser/"
    )


def test_looks_like_login_page():
    assert cookies.looks_like_login_page("<title>Sign in to CERN</title>")
    assert cookies.looks_like_login_page("redirecting to auth.cern.ch ...")
    assert not cookies.looks_like_login_page("<h1>CondDB payloads</h1>")


def test_acquisition_command_mentions_official_tool(tmp_path):
    command = cookies.sso_cookie_acquisition_command(
        "https://cms-conddb.cern.ch/", tmp_path / "conddb.txt"
    )
    assert command.startswith("auth-get-sso-cookie ")
    assert "https://cms-conddb.cern.ch/" in command
    assert "conddb.txt" in command
