"""Shared plumbing for the CMS catalog downloaders (issue #903).

The cms-catalogs pack's readers (``cric``, ``cric_core``, ``gocdb_downtimes``,
``conddb_global_tags``, ``wmstats_workflows``, ``dbs_datasets``) read JSON
caches and claim a complete scope over them: a record missing from the
cache is retracted from the graph. So a downloader must never replace a good
cache with a partial, empty or error-shaped one. This module holds the five
pieces every such downloader needs, and nothing source-specific:

``data_root(explicit)``
    Where caches go: ``explicit``, else ``ARCHI_DATA_ROOT``, else the current
    directory. It is :func:`archi.auth.cache.data_root`, the function the
    readers resolve their relative cache paths with, so a downloader writes
    exactly where its reader reads.
``get_json(url, *, timeout, proxy=None, ca_path=None, session=None)``
    One HTTPS GET returning parsed JSON. Redirects are not followed (a CERN
    service that wants a login answers 302 to its login page, which must not
    be read as data). Any non-200 answer, transport error or non-JSON body
    raises :class:`DownloadError` naming the URL, never the credential.
``require_proxy(env)``
    The X.509 proxy path (``X509_USER_PROXY``, else ``/tmp/x509up_u<uid>``),
    checked present, readable and unexpired before any request is sent.
``write_json_atomic(path, payload)``
    One cache file, written to a temp file in the same directory and moved
    into place; an empty payload is refused and the old file stays.
``write_group_atomic(directory, {name: payload})``
    A set of cache files one reader reads together (CRIC's five). Every
    payload is checked and serialised to a temp file first; only then are
    the temp files moved into place, so a failure before that point leaves
    the old set whole.

A downloader therefore fetches everything, validates it, and only then
writes; a failure anywhere exits non-zero with the old caches untouched.

Known limit of ``write_group_atomic``: the final moves are one
``os.replace`` per file, not one filesystem operation, so a crash *between*
two of them (power loss, SIGKILL) can leave a mixed set. The readers'
content hash then changes and the next successful run replaces the set.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from archi.auth.cache import data_root as _reader_data_root
from archi.auth.preflight import x509_not_after

#: The IGTF CA directory on grid hosts; CERN services present certificates
#: from the CERN Grid CA, which it holds.
GRID_CA_DIR = Path("/etc/grid-security/certificates")
#: Where ``voms-proxy-init`` writes the proxy when X509_USER_PROXY is unset.
DEFAULT_PROXY_DIR = Path("/tmp")
PROXY_ENV = "X509_USER_PROXY"
PROXY_HINT = "create one with `voms-proxy-init -voms cms`"


class DownloadError(RuntimeError):
    """The download cannot produce a complete cache; nothing is written."""


def data_root(explicit: Path | None = None) -> Path:
    """``explicit``, else ``ARCHI_DATA_ROOT``, else the current directory."""
    return _reader_data_root(explicit)


def get_json(
    url: str,
    *,
    timeout: float,
    proxy: Path | None = None,
    ca_path: str | None = None,
    session: Any = None,
) -> Any:
    """GET ``url`` and return its parsed JSON body.

    ``proxy`` is sent as both client certificate and key (a VOMS proxy file
    holds both). ``verify`` is ``ca_path`` when given, else
    :data:`GRID_CA_DIR` when that directory exists, else requests' own CA
    bundle. ``session`` is anything with a requests-style ``get``; without
    one, :func:`requests.get` is used.
    """
    kwargs: dict[str, Any] = {
        "timeout": timeout,
        "allow_redirects": False,
        "headers": {"Accept": "application/json"},
        "verify": _verify(ca_path),
    }
    if proxy is not None:
        kwargs["cert"] = (str(proxy), str(proxy))
    getter = session.get if session is not None else requests.get
    try:
        response = getter(url, **kwargs)
    except requests.RequestException as exc:
        raise DownloadError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        detail = ""
        location = response.headers.get("Location")
        if 300 <= response.status_code < 400 and location:
            detail = (
                f" (redirect to {location}; the endpoint wants a login, "
                "so the request carried no accepted credential)"
            )
        raise DownloadError(f"GET {url} returned HTTP {response.status_code}{detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise DownloadError(f"GET {url} answered 200 but the body is not JSON") from exc


def _verify(ca_path: str | None) -> str | bool:
    if ca_path:
        return ca_path
    if GRID_CA_DIR.is_dir():
        return str(GRID_CA_DIR)
    return True


def require_proxy(env: Mapping[str, str] | None = None) -> Path:
    """The X.509 proxy to authenticate with, or :class:`DownloadError`.

    ``X509_USER_PROXY`` from ``env`` (default: the process environment),
    else ``/tmp/x509up_u<uid>``. A missing, unreadable or expired proxy is
    refused, with the command that makes a new one.
    """
    env = os.environ if env is None else env
    raw = env.get(PROXY_ENV)
    path = Path(raw).expanduser() if raw else DEFAULT_PROXY_DIR / f"x509up_u{os.getuid()}"
    if not path.is_file():
        raise DownloadError(f"no X.509 proxy at {path}; {PROXY_HINT}")
    expiry = x509_not_after(path)
    if expiry is None:
        raise DownloadError(
            f"the X.509 proxy at {path} could not be read as a certificate; "
            f"{PROXY_HINT}"
        )
    if expiry <= datetime.now(timezone.utc):
        raise DownloadError(
            f"the X.509 proxy at {path} expired at {expiry.isoformat()}; {PROXY_HINT}"
        )
    return path


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write one cache file atomically; refuse an empty or non-container one."""
    write_group_atomic(path.parent, {path.name: payload})


def write_group_atomic(directory: Path, files: Mapping[str, Any]) -> None:
    """Write every ``name -> payload`` in ``directory``, or none of them.

    Each payload must be a non-empty JSON object or list. JSON is written
    with sorted keys, so an unchanged upstream answer gives an unchanged
    file and the readers' content hash does not move.
    """
    if not files:
        raise DownloadError(f"no files to write in {directory}")
    for name, payload in files.items():
        if Path(name).name != name or name in {"", ".", ".."}:
            raise DownloadError(f"cache name {name!r} is not a plain file name")
        if not isinstance(payload, (dict, list)) or not payload:
            raise DownloadError(
                f"{directory / name}: refusing to write an empty or non-JSON-"
                f"container payload ({type(payload).__name__})"
            )
    directory.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[str, Path]] = []
    try:
        for name, payload in files.items():
            handle, temp_name = tempfile.mkstemp(prefix=f".{name}.", dir=str(directory))
            staged.append((temp_name, directory / name))
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump(payload, out, ensure_ascii=False, sort_keys=True)
                out.write("\n")
        for temp_name, target in staged:
            os.replace(temp_name, target)
    except BaseException:
        for temp_name, _ in staged:
            Path(temp_name).unlink(missing_ok=True)
        raise
