#!/usr/bin/env python3
"""Download the CMS CRIC topology into the caches ``archi.sources.cric`` reads.

    python -m archi.downloaders.cric [--output-dir DIR] [--proxy PATH]

Five GETs against the CMS CRIC API (``https://cms-cric.cern.ch``), each
answer kept as CRIC sends it:

=========================  ==========================================================
cache file                 endpoint
=========================  ==========================================================
``sites.json``             ``/api/cms/site/query/?json``
``storage_units.json``     ``/api/cms/storageunit/query/?json``
``compute_units.json``     ``/api/cms/computeunit/query/?json``
``facilities.json``        ``/api/cms/facility/query/?json``
``responsibilities.json``  ``/api/accounts/user/query/?json&preset=site-responsibilities``
=========================  ==========================================================

The first three answer anonymously; the facility and site-responsibilities
endpoints redirect an anonymous request to CRIC's login page. So the run
needs an X.509 proxy (``X509_USER_PROXY``, else ``/tmp/x509up_u<uid>``, or
``--proxy``), checked before any request; make one with
``voms-proxy-init -voms cms``. Every request carries it.

The files go to ``--output-dir``, default ``cric`` under ``ARCHI_DATA_ROOT``
(or under the current directory when that variable is unset), which is
where the cms-catalogs pack's ``cric`` source reads them.

All or nothing: the reader claims a complete scope over these five files
together and retracts any site, endpoint or operator missing from them. So
every answer is fetched and checked first, and only then are the five files
written, as one group (:func:`archi.downloaders._catalog.write_group_atomic`).
A failed request, a body that is not JSON, an empty or error-shaped answer,
a site without an integer ``tier_level`` (the reader would present it as
Tier-0), or responsibility rows none of which names a known site stops the
run with a message naming the endpoint, exit status 1, and the previous
caches untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from archi.downloaders._catalog import (
    DownloadError,
    data_root,
    get_json,
    require_proxy,
    write_group_atomic,
)

DEFAULT_BASE_URL = "https://cms-cric.cern.ch"
DEFAULT_TIMEOUT = 120

#: Cache file -> API path under the base URL.
ENDPOINTS = {
    "sites.json": "/api/cms/site/query/?json",
    "storage_units.json": "/api/cms/storageunit/query/?json",
    "compute_units.json": "/api/cms/computeunit/query/?json",
    "facilities.json": "/api/cms/facility/query/?json",
    "responsibilities.json": "/api/accounts/user/query/?json&preset=site-responsibilities",
}

#: The reader's CMS site-name rule (``archi.sources.cric._CMS_SITE_NAME``).
CMS_SITE_NAME = re.compile(r"^T[0-3]_[A-Z]{2}_\w+$")
RESPONSIBILITY_COLUMNS = ["username", "site_name", "role"]
_ERROR_BODY_KEYS = ("error", "errors")

Fetch = Callable[[str], Any]


def collect(fetch: Fetch, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Every cache file's payload, checked, or :class:`DownloadError`.

    ``fetch`` takes a URL and returns its parsed JSON (raising
    :class:`DownloadError` on failure).
    """
    base = base_url.rstrip("/")
    urls = {name: base + path for name, path in ENDPOINTS.items()}
    payloads = {name: fetch(url) for name, url in urls.items()}

    sites = _keyed_objects(payloads["sites.json"], urls["sites.json"])
    for name, site in sites.items():
        if not CMS_SITE_NAME.match(name):
            raise DownloadError(f"{urls['sites.json']}: not a CMS site name: {name!r}")
        tier = site.get("tier_level")
        if not isinstance(tier, int) or isinstance(tier, bool):
            raise DownloadError(
                f"{urls['sites.json']}: site {name} has no integer tier_level"
            )

    storage_units = _keyed_objects(
        payloads["storage_units.json"], urls["storage_units.json"]
    )
    for name, unit in storage_units.items():
        site = unit.get("site")
        if not isinstance(site, dict) or not isinstance(site.get("name"), str):
            raise DownloadError(
                f"{urls['storage_units.json']}: storage unit {name} has no "
                "'site' object with a name"
            )

    _keyed_objects(payloads["compute_units.json"], urls["compute_units.json"])

    facilities = _keyed_objects(payloads["facilities.json"], urls["facilities.json"])
    for name, facility in facilities.items():
        if not isinstance(facility.get("cmssites", []), list):
            raise DownloadError(
                f"{urls['facilities.json']}: facility {name} 'cmssites' is not a list"
            )

    _check_responsibilities(
        payloads["responsibilities.json"], urls["responsibilities.json"], sites
    )
    return payloads


def _keyed_objects(payload: Any, url: str) -> dict[str, dict[str, Any]]:
    """A CRIC ``name -> object`` answer, or :class:`DownloadError`."""
    if not isinstance(payload, dict):
        raise DownloadError(
            f"{url}: answer is a {type(payload).__name__}, not a JSON object"
        )
    error_keys = [key for key in _ERROR_BODY_KEYS if key in payload]
    if error_keys:
        raise DownloadError(
            f"{url}: answer is an API error body (top-level {error_keys[0]!r} key)"
        )
    if not payload:
        raise DownloadError(f"{url}: answer is empty; refusing to write an empty cache")
    for name, value in payload.items():
        if not isinstance(value, dict):
            raise DownloadError(
                f"{url}: entry {name} is a {type(value).__name__}, not an object"
            )
    return payload


def _check_responsibilities(payload: Any, url: str, sites: dict[str, Any]) -> None:
    """The site-responsibilities answer: ``result`` rows of username, site
    title, role, at least one naming a site in the sites answer (else the
    reader would emit no operator and refuse the whole cache)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
        raise DownloadError(f"{url}: answer has no 'result' list of responsibility rows")
    desc = payload.get("desc")
    columns = desc.get("columns") if isinstance(desc, dict) else None
    if columns is not None and columns != RESPONSIBILITY_COLUMNS:
        raise DownloadError(
            f"{url}: columns are {columns}, expected {RESPONSIBILITY_COLUMNS}"
        )
    rows = payload["result"]
    if not rows:
        raise DownloadError(
            f"{url}: answer has an empty 'result'; refusing to write zero operators"
        )
    for index, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not isinstance(row[2], str)
            or not (row[1] is None or isinstance(row[1], str))
        ):
            raise DownloadError(
                f"{url}: row {index} is not [username, site title or null, role]"
            )
    titles = set(sites)
    titles.update(site["sitedb_title"] for site in sites.values() if site.get("sitedb_title"))
    if not any(row[0] and row[1] in titles for row in rows):
        raise DownloadError(
            f"{url}: none of its {len(rows)} rows names a site title from the "
            f"sites answer ({len(sites)} sites); drifted titles or payload"
        )


def _count(payload: Any) -> int:
    if isinstance(payload, dict) and isinstance(payload.get("result"), list):
        return len(payload["result"])
    return len(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: cric under ARCHI_DATA_ROOT (else the current directory).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--ca-path",
        default=None,
        help="CA bundle or directory to verify CRIC with. Default: "
        "/etc/grid-security/certificates when present, else requests' bundle.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="X.509 proxy file. Default: X509_USER_PROXY, else /tmp/x509up_u<uid>.",
    )
    args = parser.parse_args(argv)
    output = Path(args.output_dir) if args.output_dir else data_root() / "cric"
    try:
        proxy = require_proxy(
            {"X509_USER_PROXY": args.proxy} if args.proxy else None
        )
        payloads = collect(
            lambda url: get_json(
                url, timeout=args.timeout, proxy=proxy, ca_path=args.ca_path
            ),
            base_url=args.base_url,
        )
        write_group_atomic(output, payloads)
    except (DownloadError, OSError) as exc:
        print(f"cric download failed, cache not written: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(
        {
            "output": str(output),
            "files": {name: _count(payload) for name, payload in payloads.items()},
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
