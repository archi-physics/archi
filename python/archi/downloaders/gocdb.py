#!/usr/bin/env python3
"""Download GOCDB downtimes into the cache ``archi.sources.gocdb`` reads.

    python -m archi.downloaders.gocdb [--output-dir DIR] [--days-back N] [--days-ahead N]

One anonymous GET against the public GOCDB programmatic interface::

    https://goc.egi.eu/gocdbpi/public/?method=get_downtime&windowstart=YYYY-MM-DD&windowend=YYYY-MM-DD

GOCDB answers every downtime that overlaps the window (it ends after
``windowstart`` and starts before ``windowend``), in every scope, in one
XML document: the method takes no ``page`` parameter. The window runs from
``--days-back`` days before today (default 7) to ``--days-ahead`` days after
it (default 60), in UTC dates. There is no host or site filter: the reader
links only the hosts the CRIC caches know.

The answer has one ``DOWNTIME`` element per affected service, so a downtime
over three services is three rows. Each row becomes one object of the
reader's list: ``downtime_id`` (the ``ID`` attribute, an integer),
``primary_key``, ``classification``, ``severity``, ``description``,
``hosted_by`` (the GOCDB site), ``hostname``, ``service_type``, and
``start_date``/``end_date`` as ISO 8601 UTC times converted from GOCDB's
epoch seconds. The rows are sorted, so an unchanged answer keeps the
reader's content hash.

The file goes to ``--output-dir``/``records.json``, default
``gocdb-downtimes`` under ``ARCHI_DATA_ROOT`` (or under the current
directory when that variable is unset), which is where the cms-catalogs
pack's ``gocdb_downtimes`` source reads it.

Fail closed: the reader also needs the CRIC ``sites.json`` and CRIC-core
``services.json`` caches (``--sites-path``/``--services-path``, default
where the pack reads them), so a run without them stops before any request.
An answer that is not a GOCDB ``results`` document (an ``<error>`` body, a
plain-text error, malformed XML, a DTD), a row without an integer ``ID`` or
epoch dates, and a window with no downtimes all exit 1 with a message naming
the URL. Otherwise the file is staged and moved into place only if the
``gocdb_downtimes`` reader reads the staged copy, with the real CRIC caches,
with health ``ok`` and a complete scope (see
:mod:`archi.downloaders._catalog`). On any failure the previous cache stays.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from archi.auth.cache import data_root
from archi.downloaders._catalog import (
    CatalogClient,
    DownloadError,
    reader_check,
    run_cli,
    write_json_atomic,
)

DEFAULT_BASE_URL = "https://goc.egi.eu/gocdbpi/public/"
DEFAULT_TIMEOUT = 120
DEFAULT_DAYS_BACK = 7
DEFAULT_DAYS_AHEAD = 60
CACHE_NAME = "records.json"


def window(today: date, *, days_back: int, days_ahead: int) -> tuple[date, date]:
    """The (windowstart, windowend) dates around ``today``."""
    return today - timedelta(days=days_back), today + timedelta(days=days_ahead)


def downtime_url(base_url: str, start: date, end: date) -> str:
    """The ``get_downtime`` URL for downtimes overlapping [start, end]."""
    query = urlencode(
        {
            "method": "get_downtime",
            "windowstart": start.isoformat(),
            "windowend": end.isoformat(),
        }
    )
    return f"{base_url}?{query}"


def parse_downtimes(body: bytes, url: str) -> list[dict[str, Any]]:
    """The reader's rows from a ``get_downtime`` answer, or :class:`DownloadError`.

    An empty ``<results/>`` is refused too: the reader refuses an empty
    cache, so a window without downtimes keeps the previous one.
    """
    # The stdlib parser expands internal entities; GOCDB sends no DTD, so
    # any document that declares one is refused before parsing.
    if b"<!DOCTYPE" in body or b"<!ENTITY" in body:
        raise DownloadError(f"{url}: answer declares a DTD; refusing to parse it")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise DownloadError(f"{url}: answer is not XML ({exc}): {_head(body)}") from exc
    if root.tag != "results":
        raise DownloadError(
            f"{url}: answer is not a GOCDB results document (root <{root.tag}>): "
            f"{(root.text or '').strip()[:200]}"
        )
    elements = root.findall("DOWNTIME")
    if not elements:
        raise DownloadError(
            f"{url}: GOCDB lists no downtimes in this window; refusing to write an "
            "empty cache (the reader refuses one), so the previous cache stays"
        )
    rows = [_row(element, url) for element in elements]
    rows.sort(key=lambda row: (row["downtime_id"], row["hostname"], row["service_type"]))
    return rows


def _row(element: ET.Element, url: str) -> dict[str, Any]:
    raw_id = element.get("ID") or ""
    if not _is_decimal(raw_id):
        raise DownloadError(f"{url}: a DOWNTIME has ID {raw_id!r}, not an integer")
    return {
        "downtime_id": int(raw_id),
        "primary_key": element.get("PRIMARY_KEY") or element.findtext("PRIMARY_KEY", ""),
        "classification": element.get("CLASSIFICATION") or "",
        "severity": element.findtext("SEVERITY", ""),
        "description": element.findtext("DESCRIPTION", ""),
        "hosted_by": element.findtext("HOSTED_BY", ""),
        "hostname": element.findtext("HOSTNAME", ""),
        "service_type": element.findtext("SERVICE_TYPE", ""),
        "start_date": _iso(element, "START_DATE", raw_id, url),
        "end_date": _iso(element, "END_DATE", raw_id, url),
    }


def _iso(element: ET.Element, tag: str, raw_id: str, url: str) -> str:
    raw = (element.findtext(tag) or "").strip()
    try:
        if not _is_decimal(raw):
            raise ValueError(raw)
        return datetime.fromtimestamp(int(raw), timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError) as exc:
        raise DownloadError(
            f"{url}: DOWNTIME {raw_id} has {tag} {raw!r}, not epoch seconds"
        ) from exc


def _is_decimal(text: str) -> bool:
    return text.isascii() and text.isdigit()


def _head(body: bytes) -> str:
    return body[:200].decode("utf-8", errors="replace").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: gocdb-downtimes under ARCHI_DATA_ROOT (else the current directory).",
    )
    parser.add_argument("--days-back", type=_non_negative, default=DEFAULT_DAYS_BACK)
    parser.add_argument("--days-ahead", type=_non_negative, default=DEFAULT_DAYS_AHEAD)
    parser.add_argument(
        "--sites-path",
        default=None,
        help="The CRIC sites cache the reader matches against. "
        "Default: cric/sites.json under ARCHI_DATA_ROOT.",
    )
    parser.add_argument(
        "--services-path",
        default=None,
        help="The CRIC-core services cache the reader matches against. "
        "Default: cric-core/services.json under ARCHI_DATA_ROOT.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--ca-path",
        default=None,
        help="CA bundle or directory to verify GOCDB with. Default: "
        "/etc/grid-security/certificates when present, else requests' bundle.",
    )
    args = parser.parse_args(argv)
    root = data_root()
    output = Path(args.output_dir) if args.output_dir else root / "gocdb-downtimes"
    sites = Path(args.sites_path) if args.sites_path else root / "cric" / "sites.json"
    services = (
        Path(args.services_path) if args.services_path else root / "cric-core" / "services.json"
    )

    def make_reader(directory: Path) -> Any:
        from archi.sources.gocdb import GoCDBDowntimeSource

        return GoCDBDowntimeSource(
            records_path=str(directory / CACHE_NAME),
            sites_path=str(sites),
            services_path=str(services),
        )

    def download() -> dict[str, Any]:
        for name, path in (("CRIC sites", sites), ("CRIC-core services", services)):
            if not path.is_file():
                raise DownloadError(
                    f"the gocdb_downtimes reader needs the {name} cache, and {path} "
                    "is missing; run python -m archi.downloaders.cric and "
                    "python -m archi.downloaders.cric_core first"
                )
        start, end = window(
            datetime.now(timezone.utc).date(),
            days_back=args.days_back,
            days_ahead=args.days_ahead,
        )
        url = downtime_url(args.base_url, start, end)
        with CatalogClient(timeout=args.timeout, ca_path=args.ca_path) as client:
            body = client.get(url, accept="application/xml").content
        rows = parse_downtimes(body, url)
        records = write_json_atomic(
            output / CACHE_NAME,
            rows,
            check=reader_check(make_reader, origins={CACHE_NAME: url}),
        )
        return {
            "output": str(output / CACHE_NAME),
            "url": url,
            "window": [start.isoformat(), end.isoformat()],
            "rows": len(rows),
            "records": records,
        }

    return run_cli("gocdb", download)


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} is negative")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
