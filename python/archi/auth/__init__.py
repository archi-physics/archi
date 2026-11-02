"""CERN auth plumbing: cache helpers, cookie files, credential preflight.

Consolidates the auth-adjacent utilities every CERN-facing source
needs: cache path/hash/probe helpers (:mod:`archi.auth.cache`), SSO
cookie-file handling (:mod:`archi.auth.cookies`), and the
credential/reachability preflight source (:mod:`archi.auth.preflight`).
Rewritten from okg-deployments ``main@f33a9c4`` (``cms/cms_sources/``,
``wisdqm/wisdqm_sources/``) and archi ``dev@28b977d1``; per-module
docstrings record what changed.
"""
from archi.auth.cache import (
    DATA_ROOT_ENV,
    cache_or_forced_live_change_probe,
    content_hash,
    content_hash_change_probe,
    data_root,
    load_json,
    resolve_repo_path,
)
from archi.auth.cookies import (
    CookieFileStatus,
    check_cookie_file,
    load_cookie_jar,
    load_cookie_jar_from_env,
    looks_like_login_page,
    looks_like_login_url,
    sso_cookie_acquisition_command,
)
from archi.auth.preflight import CERNPreflightSource

__all__ = [
    "CERNPreflightSource",
    "CookieFileStatus",
    "DATA_ROOT_ENV",
    "cache_or_forced_live_change_probe",
    "check_cookie_file",
    "content_hash",
    "content_hash_change_probe",
    "data_root",
    "load_cookie_jar",
    "load_cookie_jar_from_env",
    "load_json",
    "looks_like_login_page",
    "looks_like_login_url",
    "resolve_repo_path",
    "sso_cookie_acquisition_command",
]
