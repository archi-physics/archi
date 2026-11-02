"""CERN SSO cookie-file helpers.

Rewritten from okg-deployments ``cms/cms_sources/preflight.py`` (cookie
loading, login-page detection) and ``cms/scripts/sso-login.py`` (cookie
save/verify flow) at ``main@f33a9c4``, plus archi
``dev@28b977d1:src/data_manager/collectors/scrapers/auth/`` (login-page
URL patterns and credential-freshness semantics from the Scrapy
``AuthProvider`` layer). Changes: the dev branch's Playwright login
provider and Scrapy middleware plumbing are not ported — v3 never runs
an interactive login in-process; the freshness/TTL idea from dev's
``Credentials`` object becomes :func:`check_cookie_file`.

Acquisition (documented, not executed here)
-------------------------------------------
Cookie files are Netscape/Mozilla format, produced out-of-band:

- CERN's official ``auth-get-sso-cookie`` tool (Kerberos-backed,
  available on lxplus)::

      kinit <user>@CERN.CH
      auth-get-sso-cookie -u <protected-url> -o <cookie-file>

- or the interactive Kerberos+TOTP flow in
  :mod:`archi.downloaders.sso_login` (moved from okg-deployments
  ``cms/scripts/sso-login.py``), which drives CERN
  Keycloak once per session and saves per-service cookie files.

Modules here only ever *read* cookie files; the file paths arrive via
environment-variable references, never as embedded values.
"""
from __future__ import annotations

import http.cookiejar
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOGIN_URL_PATTERNS: tuple[str, ...] = (
    r"auth\.cern\.ch",
    r"login\.cern\.ch",
    r"/login",
    r"/sso/",
    r"keycloak",
)
_LOGIN_URL_RE = re.compile("|".join(LOGIN_URL_PATTERNS), re.IGNORECASE)
_LOGIN_TEXT_MARKERS: tuple[str, ...] = (
    "sign in to cern",
    "auth.cern.ch",
    "login.cern.ch",
)


def looks_like_login_url(url: str) -> bool:
    """Return True if *url* matches known CERN SSO login-page patterns."""
    return bool(_LOGIN_URL_RE.search(url))


def looks_like_login_page(text: str) -> bool:
    """Return True if *text* (URL or page body) reads as a CERN login page."""
    lowered = text.lower()
    return any(marker in lowered for marker in _LOGIN_TEXT_MARKERS)


def sso_cookie_acquisition_command(url: str, cookie_file: str | Path) -> str:
    """The documented out-of-band command that produces *cookie_file*."""
    return f"auth-get-sso-cookie -u {url} -o {cookie_file}"


def load_cookie_jar(path: str | Path) -> http.cookiejar.MozillaCookieJar:
    """Load a Netscape/Mozilla cookie file, keeping expired/session cookies."""
    jar = http.cookiejar.MozillaCookieJar(str(Path(path).expanduser()))
    jar.load(ignore_discard=True, ignore_expires=True)
    return jar


def load_cookie_jar_from_env(
    cookie_file_env: str,
) -> http.cookiejar.MozillaCookieJar | None:
    """Cookie jar from the file referenced by env var *cookie_file_env*.

    ``None`` when the variable is unset or the file is missing or
    unparseable — sources report that as ``auth_failed`` rather than an
    empty-but-complete run. Promoted from
    ``archi.sources.docs.SSOCookieDocsSource`` (private cookie-session
    logic) now that the TWiki crawl source needs the same behavior.
    """
    cookie_file = os.environ.get(cookie_file_env, "")
    if not cookie_file or not Path(cookie_file).is_file():
        return None
    try:
        return load_cookie_jar(cookie_file)
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class CookieFileStatus:
    """Offline freshness verdict for one cookie file."""

    path: Path
    exists: bool
    cookie_count: int = 0
    live_count: int = 0
    expired_count: int = 0
    session_count: int = 0
    earliest_expiry: datetime | None = None
    age: timedelta | None = None
    fresh: bool = False
    reason: str = ""


def check_cookie_file(
    path: str | Path,
    *,
    domain: str | None = None,
    max_age: timedelta | None = None,
    now: datetime | None = None,
) -> CookieFileStatus:
    """Validate a cookie file without any network traffic.

    Fresh means: at least one unexpired (or session) cookie, optionally
    scoped to *domain*, and the file itself is no older than *max_age*.
    """
    p = Path(path).expanduser()
    moment = now or datetime.now(timezone.utc)
    if not p.is_file():
        return CookieFileStatus(
            path=p, exists=False, reason="cookie file is missing",
        )
    try:
        jar = load_cookie_jar(p)
    except Exception as exc:  # noqa: BLE001
        return CookieFileStatus(
            path=p,
            exists=True,
            reason=f"cookie file could not be parsed: {type(exc).__name__}",
        )
    cookies = [
        c for c in jar
        if domain is None or c.domain.endswith(domain)
    ]
    session_count = sum(1 for c in cookies if c.expires is None)
    expiries = [
        datetime.fromtimestamp(c.expires, tz=timezone.utc)
        for c in cookies
        if c.expires is not None
    ]
    expired_count = sum(1 for e in expiries if e <= moment)
    live_expiries = sorted(e for e in expiries if e > moment)
    live_count = session_count + len(live_expiries)
    age = moment - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    earliest = live_expiries[0] if live_expiries else None
    if not cookies:
        scope = f" for domain {domain}" if domain else ""
        fresh, reason = False, f"cookie file has no cookies{scope}"
    elif live_count == 0:
        fresh, reason = False, f"all {len(cookies)} cookies are expired"
    elif max_age is not None and age > max_age:
        fresh, reason = False, (
            f"cookie file is older than max age "
            f"({age.total_seconds() / 3600:.1f}h > "
            f"{max_age.total_seconds() / 3600:.1f}h)"
        )
    else:
        fresh, reason = True, f"{live_count} live cookies"
    return CookieFileStatus(
        path=p,
        exists=True,
        cookie_count=len(cookies),
        live_count=live_count,
        expired_count=expired_count,
        session_count=session_count,
        earliest_expiry=earliest,
        age=age,
        fresh=fresh,
        reason=reason,
    )
