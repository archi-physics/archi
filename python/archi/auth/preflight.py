"""CERN credential/reachability preflight source.

Rewritten from okg-deployments ``cms/cms_sources/preflight.py``
(``CMSPreflightSource``, 674 LOC) at ``main@f33a9c4``. Changes:

- De-CMS-ified: ``CERNPreflightSource``; the CMS default endpoints are
  gone — HTTP kinds require an explicit ``endpoint`` param, so the CMS
  probe set stays expressible via registry params.
- The ``jira`` live-probe kind is dropped; the live JIRA check belongs
  with the jira source port, and token presence stays expressible via
  kind ``token``.
- New offline kinds ``sso_cookie`` (cookie-file freshness/expiry via
  :mod:`archi.auth.cookies`) and ``x509_proxy`` (proxy presence +
  expiry, extracted from the original ``x509_http`` pre-check), plus a
  credential-free ``https`` reachability kind.
- ``run()`` reports health from a live-mode preflight (the original
  tagged it ``cache``).

Probes emit no NodeFact/EdgeFact; a run returns an empty fact stream
with a ConnectorHealth summarizing the probe outcome. The failure
vocabulary is unchanged: ``ok`` / ``missing_credential`` /
``auth_failed`` / ``tls_failed`` / ``endpoint_failed`` /
``cache_missing``. Credentials arrive as env-var references and file
paths only; no values are read into results.
"""
from __future__ import annotations

import json
import os
import ssl
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import requests

from okg.deployment import (
    PreflightResult,
    ConnectorRun,
)
from okg.deployment import (
    credential_preflight,
    file_preflight,
    http_preflight,
)
from okg.deployment import redact

from archi.auth.cache import resolve_repo_path
from archi.auth.cookies import (
    check_cookie_file,
    load_cookie_jar,
    looks_like_login_page,
    looks_like_login_url,
)


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _json_record_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("records", "items", "results", "data"):
            inner = value.get(key)
            if isinstance(inner, list):
                return len(inner)
        list_values = [v for v in value.values() if isinstance(v, list)]
        if list_values:
            return sum(len(v) for v in list_values)
        return 1 if value else 0
    return 1


def _credential_file_path(
    credential_ref: str,
    aliases: tuple[str, ...],
) -> Path | None:
    env = _env()
    for ref in (credential_ref,) + aliases:
        value = env.get(ref)
        if value and Path(value).expanduser().is_file():
            return Path(value).expanduser()
    return None


def _load_cookie_file(session: requests.Session, path: Path) -> None:
    session.cookies.update(load_cookie_jar(path))


def _with_credential_context(
    result: PreflightResult,
    *,
    credential_ref: str,
    alias_refs: Mapping[str, tuple[str, ...]],
) -> PreflightResult:
    data = result.as_dict()
    data["credential_refs"] = (credential_ref,)
    data["alias_refs"] = dict(alias_refs)
    return PreflightResult(**data)


def _login_page_detected(url: str, text: str) -> bool:
    return looks_like_login_url(url) or looks_like_login_page(text[:2000])


class CERNPreflightSource:
    """Registry-instantiable credential/reachability preflight adapter.

    One instance covers one probe; the probe list is name/params-driven
    from the source registry (``kind`` selects the check, everything
    else arrives as params). Emits no graph facts.
    """

    profile = "live_overlay"

    def __init__(
        self,
        *,
        source_name: str,
        kind: str,
        required: bool = False,
        credential_ref: str | None = None,
        aliases: list[str] | None = None,
        cache_paths: list[str] | None = None,
        credential_refs: list[str] | None = None,
        endpoint: str | None = None,
        timeout: float = 10.0,
        max_age_hours: float | None = None,
        base: str | None = None,
    ) -> None:
        self.name = source_name
        self.kind = kind
        self.required = required
        self.credential_ref = credential_ref
        self.aliases = tuple(aliases or ())
        self.cache_paths = tuple(cache_paths or ())
        self.credential_refs = tuple(credential_refs or ())
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.max_age_hours = (
            float(max_age_hours) if max_age_hours is not None else None
        )
        self.base = base

    def preflight(self, mode: str = "live") -> PreflightResult:
        if self.kind == "cache":
            return self._cache_preflight(mode=mode)
        if self.kind == "token":
            return credential_preflight(
                self.name,
                self._credential_ref(),
                aliases=self.aliases,
                required=self.required,
                mode=mode,
            )
        if self.kind == "file":
            return file_preflight(
                self.name,
                self._credential_ref(),
                aliases=self.aliases,
                required=self.required,
                mode=mode,
            )
        if self.kind == "ca_bundle":
            return self._ca_bundle_preflight(mode=mode)
        if self.kind == "sso_cookie":
            return self._sso_cookie_preflight(mode=mode)
        if self.kind == "x509_proxy":
            return self._x509_proxy_preflight(mode=mode)
        if self.kind == "cern_sso_http":
            return self._cern_sso_http_preflight(mode=mode)
        if self.kind == "cern_tls":
            return self._cern_tls_preflight(mode=mode)
        if self.kind == "x509_http":
            return self._x509_http_preflight(mode=mode)
        if self.kind == "https":
            return self._https_preflight(mode=mode)
        if self.kind == "any_file":
            return self._any_file_preflight(mode=mode)
        if self.kind == "all_env":
            return self._all_env_preflight(mode=mode)
        return PreflightResult(
            source_name=self.name,
            status="endpoint_failed",
            mode=mode,
            required=self.required,
            reason=f"unknown preflight kind {self.kind!r}",
            checked_at=_checked_at(),
        )

    def run(
        self,
        run_id: str,
        *,
        mode: str = "cursor",
        sync_scope: Optional[Mapping[str, Any]] = None,
    ) -> ConnectorRun:
        return ConnectorRun(facts=[], health=self.preflight())

    def _credential_ref(self) -> str:
        if not self.credential_ref:
            raise ValueError(
                f"{self.name}: credential_ref is required for {self.kind}"
            )
        return self.credential_ref

    def _endpoint(self) -> str:
        if not self.endpoint:
            raise ValueError(
                f"{self.name}: endpoint is required for {self.kind}"
            )
        return self.endpoint

    def _cache_preflight(self, *, mode: str) -> PreflightResult:
        paths = [
            resolve_repo_path(p, base=self.base) for p in self.cache_paths
        ]
        missing = [p for p in paths if not p.exists()]
        if missing:
            return PreflightResult(
                source_name=self.name,
                status="cache_missing",
                mode="cache",
                required=self.required,
                cache_path=", ".join(str(p) for p in missing),
                reason="one or more expected cache files are missing",
                checked_at=_checked_at(),
            )
        count = 0
        for path in paths:
            try:
                count += _json_record_count(path)
            except Exception as exc:  # noqa: BLE001
                return PreflightResult(
                    source_name=self.name,
                    status="cache_missing",
                    mode="cache",
                    required=self.required,
                    cache_path=str(path),
                    reason=(
                        f"failed to read cache: {type(exc).__name__}: {exc}"
                    ),
                    checked_at=_checked_at(),
                )
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode="cache",
            required=self.required,
            record_count=count,
            reason="local cache present",
            checked_at=_checked_at(),
        )

    def _ca_bundle_preflight(self, *, mode: str) -> PreflightResult:
        result = file_preflight(
            self.name,
            self._credential_ref(),
            aliases=self.aliases,
            required=self.required,
            mode=mode,
        )
        if result.status != "ok":
            data = result.as_dict()
            data["status"] = "tls_failed"
            data["reason"] = "CERN CA bundle is missing; refusing TLS bypass"
            return PreflightResult(**data)
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode=mode,
            required=self.required,
            credential_refs=(self._credential_ref(),),
            alias_refs=result.alias_refs,
            reason="CERN CA bundle present",
            checked_at=_checked_at(),
        )

    def _sso_cookie_preflight(self, *, mode: str) -> PreflightResult:
        result = file_preflight(
            self.name,
            self._credential_ref(),
            aliases=self.aliases,
            required=self.required,
            mode=mode,
        )
        if result.status != "ok":
            return result
        cookie_path = _credential_file_path(
            self._credential_ref(), self.aliases,
        )
        max_age = (
            timedelta(hours=self.max_age_hours)
            if self.max_age_hours is not None else None
        )
        status = check_cookie_file(cookie_path, max_age=max_age)
        return PreflightResult(
            source_name=self.name,
            status="ok" if status.fresh else "auth_failed",
            mode=mode,
            required=self.required,
            credential_refs=(self._credential_ref(),),
            alias_refs=result.alias_refs,
            reason=f"SSO cookie file: {status.reason}",
            checked_at=_checked_at(),
        )

    def _x509_proxy_preflight(self, *, mode: str) -> PreflightResult:
        result = file_preflight(
            self.name,
            self._credential_ref(),
            aliases=self.aliases,
            required=self.required,
            mode=mode,
        )
        if result.status != "ok":
            return result
        proxy_path = _credential_file_path(
            self._credential_ref(), self.aliases,
        )
        expiry = _x509_not_after(proxy_path)
        if expiry is not None and expiry <= datetime.now(timezone.utc):
            return PreflightResult(
                source_name=self.name,
                status="auth_failed",
                mode=mode,
                required=self.required,
                credential_refs=(self._credential_ref(),),
                alias_refs=result.alias_refs,
                reason=(
                    f"{self._credential_ref()} expired at {expiry.isoformat()}"
                ),
                checked_at=_checked_at(),
            )
        if expiry is None:
            return PreflightResult(
                source_name=self.name,
                status="auth_failed",
                mode=mode,
                required=self.required,
                credential_refs=(self._credential_ref(),),
                alias_refs=result.alias_refs,
                reason=(
                    "X.509 proxy file present but its certificate could not "
                    "be decoded; re-create the proxy (this offline kind has "
                    "no live-probe backstop, so an unreadable proxy must not "
                    "pass as healthy)"
                ),
                checked_at=_checked_at(),
            )
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode=mode,
            required=self.required,
            credential_refs=(self._credential_ref(),),
            alias_refs=result.alias_refs,
            reason=f"X.509 proxy valid until {expiry.isoformat()}",
            checked_at=_checked_at(),
        )

    def _cern_sso_http_preflight(self, *, mode: str) -> PreflightResult:
        endpoint = self._endpoint()
        file_result = file_preflight(
            self.name,
            self._credential_ref(),
            aliases=self.aliases,
            required=self.required,
            mode=mode,
        )
        if file_result.status != "ok":
            return file_result
        cookie_path = _credential_file_path(
            self._credential_ref(), self.aliases,
        )
        session = requests.Session()
        if cookie_path is not None:
            try:
                _load_cookie_file(session, cookie_path)
            except Exception as exc:  # noqa: BLE001
                return PreflightResult(
                    source_name=self.name,
                    status="auth_failed",
                    mode=mode,
                    required=self.required,
                    credential_refs=(self._credential_ref(),),
                    alias_refs=file_result.alias_refs,
                    endpoint=endpoint,
                    reason=redact(
                        f"SSO cookie file could not be parsed — re-acquire "
                        f"it (no request was sent): "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    checked_at=_checked_at(),
                )
        try:
            response = session.get(
                endpoint,
                timeout=self.timeout,
                allow_redirects=True,
            )
        except requests.exceptions.SSLError as exc:
            return PreflightResult(
                source_name=self.name,
                status="tls_failed",
                mode=mode,
                required=self.required,
                credential_refs=(self._credential_ref(),),
                alias_refs=file_result.alias_refs,
                endpoint=endpoint,
                reason=redact(
                    f"CERN SSO HTTP probe failed TLS: "
                    f"{type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode=mode,
                required=self.required,
                credential_refs=(self._credential_ref(),),
                alias_refs=file_result.alias_refs,
                endpoint=endpoint,
                reason=redact(
                    f"CERN SSO HTTP probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        text = getattr(response, "text", "") or ""
        url = getattr(response, "url", endpoint) or endpoint
        status_code = int(getattr(response, "status_code", 0) or 0)
        login_page = _login_page_detected(url, text)
        if status_code == 200 and login_page:
            reason = "CERN SSO cookie reached login page"
        elif status_code == 200:
            reason = "CERN SSO HTTP probe returned authenticated payload"
        else:
            reason = f"CERN SSO HTTP probe returned HTTP {status_code}"
        return _with_credential_context(
            http_preflight(
                self.name,
                ok=status_code == 200,
                endpoint=endpoint,
                required=self.required,
                mode=mode,
                reason=reason,
                login_page_detected=login_page or status_code in {401, 403},
            ),
            credential_ref=self._credential_ref(),
            alias_refs=file_result.alias_refs,
        )

    def _cern_tls_preflight(self, *, mode: str) -> PreflightResult:
        endpoint = self._endpoint()
        ca_result = self._ca_bundle_preflight(mode=mode)
        if ca_result.status != "ok":
            return ca_result
        ca_path = _credential_file_path(self._credential_ref(), self.aliases)
        try:
            response = requests.Session().get(
                endpoint,
                timeout=self.timeout,
                allow_redirects=True,
                verify=str(ca_path),
            )
        except requests.exceptions.SSLError as exc:
            return PreflightResult(
                source_name=self.name,
                status="tls_failed",
                mode=mode,
                required=self.required,
                credential_refs=(self._credential_ref(),),
                alias_refs=ca_result.alias_refs,
                endpoint=endpoint,
                reason=redact(
                    f"CERN TLS probe failed: {type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode=mode,
                required=self.required,
                credential_refs=(self._credential_ref(),),
                alias_refs=ca_result.alias_refs,
                endpoint=endpoint,
                reason=redact(
                    f"CERN TLS probe failed: {type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        status_code = int(getattr(response, "status_code", 0) or 0)
        text = getattr(response, "text", "") or ""
        url = getattr(response, "url", endpoint) or endpoint
        login_page = _login_page_detected(url, text)
        if status_code == 200 and login_page:
            reason = "CERN TLS probe reached login page"
        elif status_code == 200:
            reason = "CERN TLS probe succeeded with configured CA bundle"
        else:
            reason = f"CERN TLS probe returned HTTP {status_code}"
        return _with_credential_context(
            http_preflight(
                self.name,
                ok=status_code == 200,
                endpoint=endpoint,
                required=self.required,
                mode=mode,
                reason=reason,
                login_page_detected=login_page or status_code in {401, 403},
            ),
            credential_ref=self._credential_ref(),
            alias_refs=ca_result.alias_refs,
        )

    def _x509_http_preflight(self, *, mode: str) -> PreflightResult:
        endpoint = self._endpoint()
        refs = self.credential_refs or (self._credential_ref(),)
        proxy_ref = refs[0]
        ca_ref = refs[1] if len(refs) > 1 else None
        proxy_result = file_preflight(
            self.name,
            proxy_ref,
            required=self.required,
            mode=mode,
        )
        if proxy_result.status != "ok":
            return proxy_result
        proxy_path = _credential_file_path(proxy_ref, ())
        if proxy_path is None:
            return PreflightResult(
                source_name=self.name,
                status="missing_credential",
                mode=mode,
                required=self.required,
                credential_refs=tuple(refs),
                reason=f"{proxy_ref} file is not set or does not exist",
                checked_at=_checked_at(),
            )
        expiry = _x509_not_after(proxy_path)
        if expiry is not None and expiry <= datetime.now(timezone.utc):
            return PreflightResult(
                source_name=self.name,
                status="auth_failed",
                mode=mode,
                required=self.required,
                credential_refs=tuple(refs),
                endpoint=endpoint,
                reason=f"{proxy_ref} expired at {expiry.isoformat()}",
                checked_at=_checked_at(),
            )
        ca_path: Path | None = None
        if ca_ref:
            ca_result = file_preflight(
                self.name,
                ca_ref,
                required=self.required,
                mode=mode,
            )
            if ca_result.status != "ok":
                data = ca_result.as_dict()
                data["credential_refs"] = tuple(refs)
                data["status"] = "tls_failed"
                return PreflightResult(**data)
            ca_path = _credential_file_path(ca_ref, ())
        try:
            response = requests.Session().get(
                endpoint,
                timeout=self.timeout,
                allow_redirects=True,
                cert=str(proxy_path),
                verify=str(ca_path) if ca_path is not None else True,
            )
        except requests.exceptions.SSLError as exc:
            text = str(exc)
            return PreflightResult(
                source_name=self.name,
                status=(
                    "auth_failed"
                    if "bad certificate" in text.lower() else "tls_failed"
                ),
                mode=mode,
                required=self.required,
                credential_refs=tuple(refs),
                endpoint=endpoint,
                reason=redact(
                    f"X.509 HTTPS probe failed TLS: "
                    f"{type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode=mode,
                required=self.required,
                credential_refs=tuple(refs),
                endpoint=endpoint,
                reason=redact(
                    f"X.509 HTTPS probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        status_code = int(getattr(response, "status_code", 0) or 0)
        return _with_credential_context(
            http_preflight(
                self.name,
                ok=status_code == 200,
                endpoint=endpoint,
                required=self.required,
                mode=mode,
                reason=f"X.509 HTTPS probe returned HTTP {status_code}",
                login_page_detected=status_code in {401, 403},
            ),
            credential_ref=proxy_ref,
            alias_refs={},
        )

    def _https_preflight(self, *, mode: str) -> PreflightResult:
        endpoint = self._endpoint()
        try:
            response = requests.Session().get(
                endpoint,
                timeout=self.timeout,
                allow_redirects=True,
            )
        except requests.exceptions.SSLError as exc:
            return PreflightResult(
                source_name=self.name,
                status="tls_failed",
                mode=mode,
                required=self.required,
                endpoint=endpoint,
                reason=redact(
                    f"HTTPS probe failed TLS: {type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                source_name=self.name,
                status="endpoint_failed",
                mode=mode,
                required=self.required,
                endpoint=endpoint,
                reason=redact(
                    f"HTTPS probe failed: {type(exc).__name__}: {exc}"
                ),
                checked_at=_checked_at(),
            )
        status_code = int(getattr(response, "status_code", 0) or 0)
        text = getattr(response, "text", "") or ""
        url = getattr(response, "url", endpoint) or endpoint
        login_page = _login_page_detected(url, text)
        return http_preflight(
            self.name,
            ok=status_code == 200,
            endpoint=endpoint,
            required=self.required,
            mode=mode,
            reason=f"HTTPS probe returned HTTP {status_code}",
            login_page_detected=login_page or status_code in {401, 403},
        )

    def _any_file_preflight(self, *, mode: str) -> PreflightResult:
        refs = self.credential_refs or (self._credential_ref(),)
        env = _env()
        for ref in refs:
            value = env.get(ref)
            if value and Path(value).expanduser().is_file():
                return PreflightResult(
                    source_name=self.name,
                    status="ok",
                    mode=mode,
                    required=self.required,
                    credential_refs=tuple(refs),
                    reason=f"{ref} file exists",
                    checked_at=_checked_at(),
                )
        return PreflightResult(
            source_name=self.name,
            status="missing_credential",
            mode=mode,
            required=self.required,
            credential_refs=tuple(refs),
            reason="no acceptable credential file is set",
            checked_at=_checked_at(),
        )

    def _all_env_preflight(self, *, mode: str) -> PreflightResult:
        refs = self.credential_refs or (self._credential_ref(),)
        env = _env()
        missing = [ref for ref in refs if not env.get(ref)]
        if missing:
            return PreflightResult(
                source_name=self.name,
                status="missing_credential",
                mode=mode,
                required=self.required,
                credential_refs=tuple(refs),
                reason="missing env refs: " + ", ".join(missing),
                checked_at=_checked_at(),
            )
        return PreflightResult(
            source_name=self.name,
            status="ok",
            mode=mode,
            required=self.required,
            credential_refs=tuple(refs),
            reason="all required env refs are set",
            checked_at=_checked_at(),
        )


def _x509_not_after(path: Path | None) -> datetime | None:
    if path is None:
        return None
    try:
        info = ssl._ssl._test_decode_cert(str(path))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
    raw = info.get("notAfter")
    if not raw:
        return None
    parsed = parsedate_to_datetime(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
