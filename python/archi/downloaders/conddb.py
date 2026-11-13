#!/usr/bin/env python3
"""Download every CondDB global tag into the cache ``archi.sources.conddb`` reads.

    python -m archi.downloaders.conddb [--output-dir DIR] [--cookie-file PATH] [--ca-path PATH]

The source is the CMS CondDB web browser (default
``https://cms-conddb.cern.ch/cmsDbBrowser``), read with one request type
only: ``GET <base>/search/Prod/<term>``. The same web application also has
routes that change global tags; this downloader never calls them.

How a search answers, from the browser's own code: ``name ILIKE '%term%' OR
release ILIKE '%term%' LIMIT 3000``, in no order, with no paging. In the
term, ``_`` matches any one character and ``%`` any run (the browser does
not escape them), case-insensitively. So a search that returns exactly 3000
rows may have been cut short.

The listing assumes every global tag names its release as
``CMSSW_<major>_<minor>_...``, with a major of 1-99 and a minor of 0-99.
That held for all 16,390 global tags on 2026-11-13 (checked against
CMSSW's ``conddb listGTs``), but a run cannot verify it: the server gives no
total count, so a tag with another release form would be missed without a
trace. The checks below catch such a tag only when some search happens to
return it. The download lists them all this way:

1. one search per major release, ``CMSSW_<M>_`` for M in 1..99. The ``_``
   wildcards make each answer a superset (``CMSSW_1_`` also matches majors
   10-19); the answers are merged by global tag name;
2. a major whose search returned :data:`SEARCH_CAP` rows is searched again
   per minor, ``CMSSW_<M>_<m>_`` for m in 0..99;
3. the run fails, writing nothing, when a per-minor search still returns
   :data:`SEARCH_CAP` rows; when a returned release does not start
   ``CMSSW_<digits>_<digits>_``, or starts with no search term whose answer
   was complete (a major above 99, a capped major's minor above 99: tags
   like it could be missing); when one name comes back with a different
   release, description or insertion time (the values written; a type or
   snapshot that changes during the run is not a conflict, and the first
   row seen is kept); and when no global tag comes back at all.

Each answer is an HTML page whose JavaScript builds four tables from inline
JSON: ``new API.DataTables.Table({"data": [...], "headers": [...]},
"<name>")``. Only ``"gts"`` is read. It must appear exactly once, with the
headers ``["", "Global Tag", "Type", "Release", "Insertion Time", "Snapshot
Time", "Description"]``, and every row must hold seven strings, a ``Type`` of
``Global Tag``, ``Candidate`` or ``Queue``, and times written ``YYYY-MM-DD
HH:MM:SS (UTC)`` (``9999-12-31 23:59:59 (UTC)`` is "no snapshot"). Anything
else fails the run: the page changed and the parse can no longer be trusted.

``records.json`` is a list with one record per global tag, sorted by name,
with the keys the reader reads: ``name``, ``release``, ``description`` and
``created_at`` (the insertion time in ISO 8601 UTC, as the GOCDB downloader
writes its dates). The page has no ``scenario``, so records carry none.

Every 20 searches, and after the last, a progress line (searches done,
global tags so far, seconds elapsed) goes to stderr; stdout holds only the
JSON summary: the output directory, the record count, the count per type,
the number of searches and the majors searched per minor. A full run makes
about 100 searches, plus 100 per refined major: on 2026-11-13, 399 searches
(majors 1, 10 and 12 refined) at about 5 s each, one at a time, so about 30
minutes.

Authentication is a CERN SSO session cookie, never a password. Make one
out of band, on a host with a Kerberos ticket::

    kinit <username>@CERN.CH
    python -m archi.downloaders.sso_login cms_conddb   # asks for the CERN TOTP code

That writes ``cms_conddb.txt`` in ``--cookie-dir``, else ``ARCHI_COOKIE_DIR``,
else ``./.cookies``. This downloader reads ``--cookie-file``, else
``CONDDB_COOKIE_FILE``, else ``cms_conddb.txt`` in that same directory, and
stops before any request when the file is missing, unreadable, or holds no
unexpired cookie for the server. The SSO session lasts about 12 hours, so a
scheduled refresh needs a fresh login first. An expired session makes the
server redirect to the CERN login page; the run fails on it with the login
command in the message, and redirects are never followed.

The server's certificate is issued by the CERN Grid CA, an intermediate CA
signed by the CERN Root CA 2; neither is in the public bundle. Before Python
3.13, verification must end at a root CA, so the Grid CA alone is not
enough: ``--ca-path`` takes a PEM file holding both the CERN Grid CA and the
CERN Root CA 2, or a CA directory such as
``/etc/grid-security/certificates``; either is used as it is, as in the other
catalog downloaders. Without ``--ca-path``, verification uses
``/etc/grid-security/certificates`` when present, else requests' bundle.
Verification is never turned off.

The file goes to ``--output-dir``, default ``conddb-global-tags`` under
``ARCHI_DATA_ROOT`` (else the current directory): where the cms-catalogs
pack's ``conddb_global_tags`` source reads it. It is staged, the reader runs
over the staged copy, and only health ``ok`` with a complete scope moves it
into place; any failure exits 1 and leaves the old file untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlparse

from archi.auth.cache import data_root, resolve_repo_path
from archi.auth.cookies import check_cookie_file, load_cookie_jar, looks_like_login_page
from archi.downloaders import sso_login
from archi.downloaders._catalog import (
    CatalogClient,
    DownloadError,
    reader_check,
    run_cli,
    write_json_atomic,
)

DEFAULT_BASE_URL = "https://cms-conddb.cern.ch/cmsDbBrowser"
DEFAULT_TIMEOUT = 120
#: The pack's cache directory under the data root, and its one file.
PACK_DIR = "conddb-global-tags"
CACHE_NAME = "records.json"
#: The browser's ``LIMIT``: a search answering this many rows may be cut short.
SEARCH_CAP = 3000
MAJORS = range(1, 100)
MINORS = range(0, 100)
#: A progress line on stderr every this many searches.
PROGRESS_EVERY = 20
#: The sso_login service whose cookie file this reads.
COOKIE_SERVICE = "cms_conddb"
COOKIE_ENV = sso_login.ENV_VARS[COOKIE_SERVICE]
SSO_HINT = (
    "log in again with `kinit <username>@CERN.CH` then "
    f"`python -m archi.downloaders.sso_login {COOKIE_SERVICE}` (it asks for the "
    "CERN TOTP code; the session lasts about 12 hours)"
)
GTS_TABLE = "gts"
GTS_HEADERS = ["", "Global Tag", "Type", "Release", "Insertion Time", "Snapshot Time", "Description"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S (UTC)"
NO_SNAPSHOT = "9999-12-31 23:59:59 (UTC)"

_RELEASE = re.compile(r"^CMSSW_\d+_\d+_")
_TABLE_START = re.compile(r"new\s+API\.DataTables\.Table\(\s*")
_TABLE_NAME = re.compile(r'\s*,\s*"([^"\\]*)"\s*\)')
_JSON = json.JSONDecoder()

Fetch = Callable[[str], str]


class GlobalTagType(Enum):
    """The browser's ``Type`` column."""

    GLOBAL_TAG = "Global Tag"
    CANDIDATE = "Candidate"
    QUEUE = "Queue"


@dataclass(frozen=True)
class GlobalTag:
    """One row of the ``gts`` table, validated."""

    name: str
    type: GlobalTagType
    release: str
    inserted: datetime
    snapshot: datetime | None
    description: str

    @classmethod
    def from_row(cls, row: Any, url: str) -> "GlobalTag":
        if (
            not isinstance(row, list)
            or len(row) != len(GTS_HEADERS)
            or not all(isinstance(cell, str) for cell in row)
        ):
            raise DownloadError(
                f"{url}: a gts row is not {len(GTS_HEADERS)} strings ({_clip(row)}); "
                "the page layout changed"
            )
        _, name, kind, release, inserted, snapshot, description = row
        if not name.strip():
            raise DownloadError(f"{url}: a gts row has no global tag name ({_clip(row)})")
        try:
            type_ = GlobalTagType(kind)
        except ValueError:
            raise DownloadError(
                f"{url}: global tag {name} has type {kind!r}, not one of "
                f"{', '.join(member.value for member in GlobalTagType)}"
            ) from None
        return cls(
            name=name,
            type=type_,
            release=release,
            inserted=_time(inserted, name, "Insertion Time", url),
            snapshot=None if snapshot == NO_SNAPSHOT else _time(snapshot, name, "Snapshot Time", url),
            description=description,
        )

    def record(self) -> dict[str, str]:
        """The cache record: only the keys the reader reads."""
        return {
            "name": self.name,
            "release": self.release,
            "description": self.description,
            "created_at": self.inserted.isoformat(),
        }


def _time(text: str, name: str, column: str, url: str) -> datetime:
    try:
        return datetime.strptime(text, TIME_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise DownloadError(
            f"{url}: global tag {name} has {column} {text!r}, not 'YYYY-MM-DD HH:MM:SS (UTC)'"
        ) from None


def _clip(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 200 else text[:200] + "..."


def search_url(base_url: str, term: str) -> str:
    return f"{base_url.rstrip('/')}/search/Prod/{quote(term, safe='')}"


def parse_search_page(html: str, url: str) -> list[GlobalTag]:
    """The global tags in one search page, or :class:`DownloadError`."""
    tables: dict[str, list[Any]] = {}
    # One table after another: a table's JSON may itself quote the table code
    # (in a description), and that text must not be read as a table.
    pos = 0
    while (start := _TABLE_START.search(html, pos)) is not None:
        try:
            payload, end = _JSON.raw_decode(html, start.end())
        except ValueError:
            raise DownloadError(
                f"{url}: a DataTables table is not inline JSON; the page layout changed"
            ) from None
        name = _TABLE_NAME.match(html, end)
        if name is None or not isinstance(payload, dict):
            raise DownloadError(
                f"{url}: a DataTables table is not ({{...}}, \"<name>\"); the page layout changed"
            )
        tables.setdefault(name.group(1), []).append(payload)
        pos = name.end()
    found = tables.get(GTS_TABLE, [])
    if len(found) != 1:
        if not found and looks_like_login_page(html):
            raise DownloadError(f"{url} answered the CERN login page; {SSO_HINT}")
        raise DownloadError(
            f"{url}: the page holds {len(found)} {GTS_TABLE!r} tables, not 1; "
            "the page layout changed"
        )
    table = found[0]
    if table.get("headers") != GTS_HEADERS:
        raise DownloadError(
            f"{url}: the {GTS_TABLE!r} table has headers {_clip(table.get('headers'))}, "
            f"not {_clip(GTS_HEADERS)}; the page layout changed"
        )
    rows = table.get("data")
    if not isinstance(rows, list):
        raise DownloadError(f"{url}: the {GTS_TABLE!r} table's data is not a list")
    return [GlobalTag.from_row(row, url) for row in rows]


def collect(fetch: Fetch, *, base_url: str = DEFAULT_BASE_URL) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """``(records, report)`` for every global tag, or :class:`DownloadError`.

    ``fetch`` takes a URL and returns the page's text. The searches are the
    ones the module docstring lists, one at a time.
    """
    started = time.monotonic()
    found: dict[str, GlobalTag] = {}
    complete_terms: list[str] = []
    searches = 0

    def progress() -> None:
        print(
            f"conddb: {searches} searches, {len(found)} global tags, "
            f"{time.monotonic() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    def search(term: str) -> bool:
        """Merge one search's answer; True when it hit the cap."""
        nonlocal searches
        url = search_url(base_url, term)
        tags = parse_search_page(fetch(url), url)
        searches += 1
        for tag in tags:
            if not _RELEASE.match(tag.release):
                raise DownloadError(
                    f"{url}: global tag {tag.name} has release {tag.release!r}, not "
                    "CMSSW_<major>_<minor>_...; listing by release no longer finds "
                    "every global tag"
                )
            # Only the written values must agree: a candidate promoted or a
            # snapshot set while the run goes on changes neither.
            seen = found.setdefault(tag.name, tag)
            if seen.record() != tag.record():
                raise DownloadError(
                    f"{url}: global tag {tag.name} came back as two different rows "
                    f"({_clip(seen.record())} and {_clip(tag.record())})"
                )
        if searches % PROGRESS_EVERY == 0:
            progress()
        capped = len(tags) >= SEARCH_CAP
        if not capped:
            complete_terms.append(term)
        return capped

    refined = [major for major in MAJORS if search(f"CMSSW_{major}_")]
    for major in refined:
        for minor in MINORS:
            term = f"CMSSW_{major}_{minor}_"
            if search(term):
                raise DownloadError(
                    f"{search_url(base_url, term)} returned {SEARCH_CAP} rows, the "
                    "server's limit, for a single release series; the listing would "
                    "be incomplete"
                )
    if searches % PROGRESS_EVERY:
        progress()
    for tag in found.values():
        if not tag.release.startswith(tuple(complete_terms)):
            raise DownloadError(
                f"global tag {tag.name} has release {tag.release}, which no complete "
                f"search covers (majors 1-99, and minors 0-99 of a major whose search "
                f"hit the {SEARCH_CAP}-row limit); other tags like it could be missing"
            )
    if not found:
        raise DownloadError(
            f"no search at {base_url} returned a global tag; refusing to write an empty catalog"
        )
    records = [found[name].record() for name in sorted(found)]
    types = Counter(tag.type.value for tag in found.values())
    report = {
        "types": dict(sorted(types.items())),
        "searches": searches,
        "refined_majors": refined,
    }
    return records, report


def cookie_file_path(explicit: str | None, env: Mapping[str, str] | None = None) -> Path:
    """``--cookie-file``, else ``CONDDB_COOKIE_FILE``, else sso_login's file."""
    env = os.environ if env is None else env
    if explicit:
        return Path(explicit).expanduser()
    raw = env.get(COOKIE_ENV)
    if raw:
        return Path(raw).expanduser()
    return sso_login.cookie_dir() / f"{COOKIE_SERVICE}.txt"


def session_cookies(path: Path, host: str) -> Any:
    """The cookie jar in ``path``, or :class:`DownloadError` when it cannot
    authenticate (missing, unreadable, or no unexpired cookie for ``host``)."""
    status = check_cookie_file(path, domain=host)
    if not status.fresh:
        raise DownloadError(f"the CondDB cookie file {path} is unusable ({status.reason}); {SSO_HINT}")
    try:
        return load_cookie_jar(path)
    except Exception as exc:  # noqa: BLE001 -- the parser's own failure, by type
        raise DownloadError(
            f"the CondDB cookie file {path} could not be read ({type(exc).__name__}); {SSO_HINT}"
        ) from exc


def make_reader(directory: Path) -> Any:
    """The ``conddb_global_tags`` reader over ``directory/records.json``."""
    from archi.sources.conddb import CondDBGlobalTagSource

    return CondDBGlobalTagSource(
        records_path=str(directory / CACHE_NAME),
        cmssw_records_path=str(resolve_repo_path("cmssw-releases/records.json")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Default: {PACK_DIR} under ARCHI_DATA_ROOT (else the current directory).",
    )
    parser.add_argument(
        "--cookie-file",
        default=None,
        help=f"CERN SSO cookie file. Default: {COOKIE_ENV}, else {COOKIE_SERVICE}.txt in "
        f"ARCHI_COOKIE_DIR (else ./.cookies), where `python -m archi.downloaders.sso_login "
        f"{COOKIE_SERVICE}` writes it.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--ca-path",
        default=None,
        help="CA bundle or directory to verify cms-conddb with: a PEM file holding both "
        "the CERN Grid CA and the CERN Root CA 2 (the Grid CA is an intermediate "
        "CA), or a directory such as /etc/grid-security/certificates. "
        "Default: /etc/grid-security/certificates when present, else requests' bundle.",
    )
    args = parser.parse_args(argv)
    output = Path(args.output_dir) if args.output_dir else data_root() / PACK_DIR

    def download() -> dict[str, Any]:
        ca_path = str(Path(args.ca_path).expanduser()) if args.ca_path is not None else None
        if ca_path is not None and not Path(ca_path).exists():
            raise DownloadError(f"--ca-path {ca_path} is neither a file nor a directory")
        cookies = session_cookies(
            cookie_file_path(args.cookie_file), urlparse(args.base_url).hostname or ""
        )
        with CatalogClient(
            timeout=args.timeout, ca_path=ca_path, credential_hint=SSO_HINT
        ) as client:
            client.session.cookies.update(cookies)

            def fetch(url: str) -> str:
                body = client.get(url, accept="text/html").content
                try:
                    return body.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DownloadError(f"{url}: the page is not UTF-8 ({exc})") from exc

            records, report = collect(fetch, base_url=args.base_url)
        count = write_json_atomic(
            output / CACHE_NAME,
            records,
            check=reader_check(make_reader, origins={CACHE_NAME: search_url(args.base_url, "")}),
        )
        return {"output": str(output), "records": count, **report}

    return run_cli("conddb", download)


if __name__ == "__main__":
    raise SystemExit(main())
