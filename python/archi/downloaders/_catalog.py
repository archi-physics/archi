"""Shared plumbing for the CMS catalog downloaders (issue #903).

The cms-catalogs pack's readers (``cric``, ``cric_core``, ``gocdb_downtimes``,
``conddb_global_tags``, ``wmstats_workflows``, ``dbs_datasets``) read local
caches and claim a complete scope over them: a record missing from the cache
is retracted from the graph. So a downloader must never replace a good cache
with one its reader would refuse. Every downloader here follows one shape::

    from archi.auth.cache import data_root

    def download() -> dict:
        with CatalogClient(timeout=args.timeout, proxy=proxy, ca_path=args.ca_path) as client:
            payloads = {name: client.get_json(url) for name, url in urls.items()}
        records = write_group_atomic(
            output, payloads, check=reader_check(make_reader, origins=urls)
        )
        return {"output": str(output), "files": ..., "records": records}

    return run_cli("thing", download)

1. Fetch every answer with :class:`CatalogClient` (one ``requests.Session``,
   an optional X.509 proxy, transient failures retried, login redirects and
   non-200 answers refused with the URL in the message).
2. Optionally pre-check each answer for a clearer message
   (:func:`keyed_objects` for CRIC-style ``name -> object`` answers). The
   reader is the authority; pre-checks only say earlier what it would say.
3. Hand all files to :func:`write_group_atomic` with
   ``check=reader_check(make_reader)``: the files are staged under their
   real names in a staging directory inside the target, the real reader runs
   over the staged copy, and only a run with health ``ok`` and a complete
   scope lets the files move into place.
4. Wrap it all in :func:`run_cli`, which prints a JSON summary or a one-line
   error and returns the exit status.

Caches go under :func:`archi.auth.cache.data_root` (``ARCHI_DATA_ROOT``, else
the current directory), the function the readers resolve their paths with.
:func:`require_proxy` checks the grid proxy before any request.

The target directory also holds ``.lock`` (never removed, so a failed first
run leaves the directory with only that file) and, while a write runs or
after a killed one, ``.staging.*`` entries; a tool that walks a cache
directory recursively and refuses dot entries (the snapshot builder's
``_check_tree``) must exclude them.

Known limit of :func:`write_group_atomic`: the final moves are one
``os.replace`` per file, not one filesystem operation. If a move fails after
an earlier one succeeded, :class:`PartialWriteError` names the files already
replaced; a crash between two moves (power loss, SIGKILL) can leave a mixed
set without that message. The next successful run replaces the whole set.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from archi.auth.cache import resolve_repo_path
from archi.auth.preflight import x509_not_after

#: The IGTF CA directory on grid hosts; it holds the CERN Grid CA that CERN
#: services such as CRIC present certificates from.
GRID_CA_DIR = Path("/etc/grid-security/certificates")
#: Where ``voms-proxy-init`` writes the proxy when X509_USER_PROXY is unset.
DEFAULT_PROXY_DIR = Path("/tmp")
PROXY_ENV = "X509_USER_PROXY"
PROXY_HINT = "create one with `voms-proxy-init -voms cms`"
#: Top-level keys that mark an API error body sent in place of data.
ERROR_BODY_KEYS = ("error", "errors")
#: Requests per URL before a transient failure is final, and the first wait
#: (doubled after each failed attempt).
ATTEMPTS = 3
BACKOFF_SECONDS = 1.0
CHECK_RUN_ID = "archi-download-check"
#: Inside the target directory: the lock file held for a whole group write,
#: and the prefix of staging directories.
LOCK_NAME = ".lock"
STAGING_PREFIX = ".staging."
#: A staging directory older than this was left by a killed run.
STALE_STAGING_SECONDS = 6 * 3600

#: Given the staging directory (files under their real names), raise
#: :class:`DownloadError` to refuse the group; the return value is passed on.
Check = Callable[[Path], Any]

_TLS_ALERT = re.compile(r"alert", re.IGNORECASE)


class DownloadError(RuntimeError):
    """The download cannot produce a complete cache; nothing is written."""


class PartialWriteError(DownloadError):
    """Some files of a group were replaced before a later move failed."""


def require_proxy(env: Mapping[str, str] | None = None) -> Path:
    """The X.509 proxy to authenticate with, or :class:`DownloadError`.

    ``X509_USER_PROXY`` from ``env`` (default: the process environment),
    else ``/tmp/x509up_u<uid>``. A missing, unreadable or expired proxy is
    refused, with its path and the command that makes a new one.
    """
    env = os.environ if env is None else env
    raw = env.get(PROXY_ENV)
    path = Path(raw).expanduser() if raw else DEFAULT_PROXY_DIR / f"x509up_u{os.getuid()}"
    if not path.is_file():
        raise DownloadError(f"no X.509 proxy at {path}; {PROXY_HINT}")
    expiry = x509_not_after(path)
    if expiry is None:
        raise DownloadError(
            f"the X.509 proxy at {path} could not be read as a certificate; {PROXY_HINT}"
        )
    if expiry <= datetime.now(timezone.utc):
        raise DownloadError(
            f"the X.509 proxy at {path} expired at {expiry.isoformat()}; {PROXY_HINT}"
        )
    return path


class CatalogClient:
    """HTTPS GETs for one download run, over one ``requests.Session``.

    ``proxy`` is sent as both client certificate and key (a VOMS proxy file
    holds both). TLS is verified against ``ca_path`` when given, else
    :data:`GRID_CA_DIR` when that directory exists, else requests' own bundle
    (which honours ``REQUESTS_CA_BUNDLE``). Redirects are not followed: a
    CERN service that wants a login answers 302 to its login page, which must
    not be read as data.

    A 5xx answer, a timeout or a connection error is retried, :data:`ATTEMPTS`
    times in all, waiting :data:`BACKOFF_SECONDS`, then twice that, and so on.
    A 3xx or 4xx answer and a TLS failure are final at once.
    """

    def __init__(
        self,
        *,
        timeout: float,
        proxy: Path | None = None,
        ca_path: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.proxy = proxy
        self.ca_path = ca_path
        self.session = requests.Session()

    def __enter__(self) -> "CatalogClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    def get(self, url: str, *, accept: str = "application/json") -> requests.Response:
        """The 200 answer for ``url``, or :class:`DownloadError` naming it."""
        verify = self._verify()
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "allow_redirects": False,
            "headers": {"Accept": accept},
            "verify": verify,
        }
        if self.proxy is not None:
            kwargs["cert"] = (str(self.proxy), str(self.proxy))
        for attempt in range(1, ATTEMPTS + 1):
            final = attempt == ATTEMPTS
            tries = f" after {attempt} attempts" if attempt > 1 else ""
            try:
                response = self.session.get(url, **kwargs)
            except requests.exceptions.SSLError as exc:
                raise DownloadError(
                    f"GET {url} failed TLS ({exc}); {self._tls_hint(exc, verify)}"
                ) from exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                if final:
                    raise DownloadError(
                        f"GET {url} failed{tries}: {type(exc).__name__}: {exc}"
                    ) from exc
            except requests.RequestException as exc:
                raise DownloadError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
            else:
                if response.status_code == 200:
                    return response
                if response.status_code < 500 or final:
                    raise DownloadError(
                        f"GET {url} returned HTTP {response.status_code}{tries}"
                        f"{_redirect_note(response)}"
                    )
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
        raise AssertionError("unreachable")  # pragma: no cover

    def get_json(self, url: str) -> Any:
        """:meth:`get` ``url`` and parse its body as JSON."""
        response = self.get(url)
        try:
            return response.json()
        except ValueError as exc:
            raise DownloadError(f"GET {url} answered 200 but the body is not JSON") from exc

    def _verify(self) -> str | bool:
        if self.ca_path:
            return self.ca_path
        if GRID_CA_DIR.is_dir():
            return str(GRID_CA_DIR)
        return True

    def _tls_hint(self, exc: Exception, verify: str | bool) -> str:
        if self.proxy is not None and _TLS_ALERT.search(str(exc)):
            # An alert is the server ending the handshake, typically over the
            # client certificate it was offered.
            return (
                "the server may have refused the X.509 proxy; check it with "
                "`voms-proxy-info` (expired, or not a CMS VOMS proxy)"
            )
        if _TLS_ALERT.search(str(exc)):
            return (
                "the server ended the handshake; the endpoint may require a "
                "client certificate, which this request did not send"
            )
        if verify is True:
            return (
                "pass --ca-path, or set REQUESTS_CA_BUNDLE, to a CA bundle that "
                "holds the server's CA"
            )
        return (
            f"it was verified with {verify}; pass --ca-path a CA bundle or "
            "directory that holds the server's CA"
        )


def _redirect_note(response: Any) -> str:
    location = response.headers.get("Location")
    if 300 <= response.status_code < 400 and location:
        return (
            f" (redirect to {location}; the endpoint wants a login, so the "
            "request carried no accepted credential)"
        )
    return ""


def keyed_objects(payload: Any, url: str) -> dict[str, dict[str, Any]]:
    """A ``name -> object`` answer (CRIC's shape), or :class:`DownloadError`.

    Refuses a non-object, an API error body (top-level ``error``/``errors``),
    an empty object, and an entry that is not an object.
    """
    if not isinstance(payload, dict):
        raise DownloadError(f"{url}: answer is a {type(payload).__name__}, not a JSON object")
    error_keys = [key for key in ERROR_BODY_KEYS if key in payload]
    if error_keys:
        raise DownloadError(f"{url}: answer is an API error body (top-level {error_keys[0]!r} key)")
    if not payload:
        raise DownloadError(f"{url}: answer is empty; refusing to write an empty cache")
    for name, value in payload.items():
        if not isinstance(value, dict):
            raise DownloadError(f"{url}: entry {name} is a {type(value).__name__}, not an object")
    return payload


def reader_check(
    make_reader: Callable[[Path], Any],
    *,
    origins: Mapping[str, str] | None = None,
) -> Check:
    """A :data:`Check` that runs the real reader over the staged files.

    ``make_reader(staging)`` builds the reader with its cache paths inside
    ``staging``. Every staged file must be one of the reader's
    ``cache_paths`` (resolved against its ``base``): a file the reader does
    not read would reach the cache unchecked. The reader may read more (the
    GOCDB reader also reads the CRIC caches). The check then runs the reader
    in ``scope_complete`` mode, drains its facts, and accepts only health
    ``ok``, a complete scope and at least one record; it returns the record
    count. A refusal carries the reader's own reason; ``origins`` (cache file
    name -> URL) adds which endpoint the named file came from (every
    endpoint, when the reason names no file).
    """
    origins = dict(origins or {})

    def check(staging: Path) -> int:
        reader = make_reader(staging)
        read = {
            resolve_repo_path(path, base=reader.base).resolve()
            for path in reader.cache_paths
        }
        unread = sorted(path.name for path in staging.iterdir() if path.resolve() not in read)
        if unread:
            raise DownloadError(
                f"the reader does not read {', '.join(unread)}; refusing to write "
                "a cache file no reader check has seen"
            )
        try:
            run = reader.run(CHECK_RUN_ID, mode="scope_complete")
            for _ in run.facts:
                pass
        except Exception as exc:  # the reader's own failure, reported verbatim
            raise DownloadError(
                f"the reader failed on the downloaded answers: {type(exc).__name__}: "
                f"{exc}{_origin_note(str(exc), origins)}"
            ) from exc
        health = run.health
        reason = health.reason or "no reason given"
        if health.status != "ok" or not run.completed_scope:
            raise DownloadError(
                "the reader refuses the downloaded answers (health "
                f"{health.status}, completed_scope={run.completed_scope}): "
                f"{reason}{_origin_note(reason, origins)}"
            )
        count = health.record_count
        if not isinstance(count, int) or count <= 0:
            raise DownloadError(
                f"the reader read {count} records from the downloaded answers"
                f"{_origin_note('', origins)}"
            )
        return count

    return check


def _origin_note(reason: str, origins: Mapping[str, str]) -> str:
    """Which endpoints the files named in ``reason`` came from. A file
    counts as named only as a whole path segment (``/sites.json``), so
    ``rcsites.json`` does not name ``sites.json``."""
    named = {
        name: url
        for name, url in origins.items()
        if re.search(rf"(?:^|/){re.escape(name)}(?![\w.-])", reason)
    }
    if named:
        return "".join(f"; {name} came from {url}" for name, url in named.items())
    if origins:
        return "; answers from " + ", ".join(origins.values())
    return ""


def write_json_atomic(path: Path, payload: Any, *, check: Check | None = None) -> Any:
    """Write one cache file the way :func:`write_group_atomic` writes a group."""
    return write_group_atomic(path.parent, {path.name: payload}, check=check)


def write_group_atomic(
    directory: Path,
    files: Mapping[str, Any],
    *,
    check: Check | None = None,
) -> Any:
    """Write every ``name -> payload`` in ``directory``, or none of them.

    Each payload must be a non-empty JSON object or list. ``directory`` is
    resolved (a symlink to another filesystem is followed) and created if
    needed, and a lock on ``directory/.lock`` is taken for the whole write: a
    second writer fails at once instead of waiting. Staging directories in it
    older than six hours (left by a killed run) are removed. The files are
    written, with sorted keys (an unchanged answer keeps the readers' content
    hash) and fsynced, under their real names into a new staging directory
    inside ``directory``, so the final moves never cross a filesystem.
    ``check`` then sees that staging directory; if it raises, the staging
    copy is deleted and the old files stay. Only then is each file moved into
    place (mode 0644 less the umask). Returns what ``check`` returned.

    ``.lock`` is never removed and the target directory is never deleted,
    even when this call created it and then failed: removing either would let
    a run holding an open handle on the old ``.lock`` and a run on a new one
    both take "the" lock. A failed first run leaves the target holding only
    ``.lock``, which the next run reuses.
    """
    if not files:
        raise DownloadError(f"no files to write in {directory}")
    for name, payload in files.items():
        if Path(name).name != name or name in {"", ".", "..", LOCK_NAME}:
            raise DownloadError(f"cache name {name!r} is not a plain file name")
        if not isinstance(payload, (dict, list)) or not payload:
            raise DownloadError(
                f"{directory / name}: refusing to write an empty or non-JSON-"
                f"container payload ({type(payload).__name__})"
            )
    target = directory.resolve()
    target.mkdir(parents=True, exist_ok=True)
    lock_path = target / LOCK_NAME
    with open(lock_path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DownloadError(
                f"another download is writing {target} (it holds {lock_path}); "
                "try again when it has finished"
            ) from exc
        except OSError as exc:
            raise DownloadError(
                f"cannot take a lock on {lock_path} ({exc}); refusing to write "
                "the cache without a lock"
            ) from exc
        _remove_stale_staging(target)
        staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=target))
        try:
            return _stage_check_and_move(staging, target, files, check)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _stage_check_and_move(
    staging: Path, target: Path, files: Mapping[str, Any], check: Check | None
) -> Any:
    for name, payload in files.items():
        _write_staged(staging / name, payload, target=target / name)
    result = None
    if check is not None:
        try:
            result = check(staging)
        except DownloadError as exc:
            raise type(exc)(str(exc).replace(str(staging), str(target))) from exc
    replaced: list[str] = []
    for name in files:
        try:
            os.replace(staging / name, target / name)
        except OSError as exc:
            if not replaced:
                raise
            raise PartialWriteError(
                f"replaced {', '.join(replaced)} in {target}, then moving {name} "
                f"failed ({exc}); the cache set there is mixed until a run succeeds"
            ) from exc
        replaced.append(name)
    return result


def _remove_stale_staging(target: Path) -> None:
    cutoff = time.time() - STALE_STAGING_SECONDS
    for entry in target.glob(f"{STAGING_PREFIX}*"):
        if entry.is_dir() and not entry.is_symlink() and entry.stat().st_mtime < cutoff:
            shutil.rmtree(entry, ignore_errors=True)


def _write_staged(path: Path, payload: Any, *, target: Path) -> None:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise DownloadError(f"{target}: payload is not JSON-serialisable ({exc})") from exc
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DownloadError(f"{target}: payload is not encodable as UTF-8 ({exc})") from exc
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(handle, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def _exit_on_sigterm(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


def run_cli(name: str, download: Callable[[], Mapping[str, Any]]) -> int:
    """Run ``download`` and print its summary as JSON; 0 on success, else 1.

    A :class:`DownloadError` or ``OSError`` prints one line on stderr: "cache
    not written", or "failed part-way" for a :class:`PartialWriteError`.
    While ``download`` runs, SIGTERM raises ``SystemExit(143)`` instead of
    killing the process outright, so staging directories are removed and the
    lock released on the way out.
    """
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        summary = download()
    except PartialWriteError as exc:
        print(f"{name} download failed part-way: {exc}", file=sys.stderr)
        return 1
    except (DownloadError, OSError) as exc:
        print(f"{name} download failed, cache not written: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(dict(summary), sort_keys=True))
    return 0
