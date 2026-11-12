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
``voms-proxy-init -voms cms``. Every request carries it. CRIC presents a
CERN Grid CA certificate: off a grid host (no
``/etc/grid-security/certificates``), pass ``--ca-path`` a bundle holding
that CA.

The files go to ``--output-dir``, default ``cric`` under ``ARCHI_DATA_ROOT``
(or under the current directory when that variable is unset), which is
where the cms-catalogs pack's ``cric`` source reads them.

All or nothing: the reader claims a complete scope over these five files
together and retracts any site, endpoint or operator missing from them. So
every answer is fetched first; then the five files are staged, the ``cric``
reader runs over the staged copy, and only a run with health ``ok`` and a
complete scope moves them into place (see :mod:`archi.downloaders._catalog`).
Before that, two pre-checks give clearer messages: each answer must be a
non-empty ``name -> object`` map (the responsibilities answer a non-empty
``result`` of ``[username, site title, role]`` rows), and every site must
carry an integer ``tier_level``, which the reader would otherwise default to
0 and present as Tier-0. Any failure exits 1 with a message naming the
endpoint and leaves the previous caches untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from archi.downloaders._catalog import (
    CatalogClient,
    DownloadError,
    data_root,
    keyed_objects,
    reader_check,
    require_proxy,
    run_cli,
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

Fetch = Callable[[str], Any]


def endpoint_urls(base_url: str) -> dict[str, str]:
    """Cache file -> full URL."""
    base = base_url.rstrip("/")
    return {name: base + path for name, path in ENDPOINTS.items()}


def collect(fetch: Fetch, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Every cache file's payload, pre-checked, or :class:`DownloadError`.

    ``fetch`` takes a URL and returns its parsed JSON (raising
    :class:`DownloadError` on failure).
    """
    urls = endpoint_urls(base_url)
    payloads = {name: fetch(url) for name, url in urls.items()}
    for name in ("sites.json", "storage_units.json", "compute_units.json", "facilities.json"):
        keyed_objects(payloads[name], urls[name])
    for name, site in payloads["sites.json"].items():
        tier = site.get("tier_level")
        if not isinstance(tier, int) or isinstance(tier, bool):
            raise DownloadError(f"{urls['sites.json']}: site {name} has no integer tier_level")
    _check_responsibility_rows(payloads["responsibilities.json"], urls["responsibilities.json"])
    return payloads


def _check_responsibility_rows(payload: Any, url: str) -> None:
    """A non-empty ``result`` of ``[username or null, site title or null,
    role]`` rows; the reader drops rows with a null username or site."""
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
        raise DownloadError(f"{url}: answer has no 'result' list of responsibility rows")
    rows = payload["result"]
    if not rows:
        raise DownloadError(f"{url}: answer has an empty 'result'; refusing to write zero operators")
    for index, row in enumerate(rows):
        if not (
            isinstance(row, list)
            and len(row) == 3
            and (row[0] is None or isinstance(row[0], str))
            and (row[1] is None or isinstance(row[1], str))
            and isinstance(row[2], str)
        ):
            raise DownloadError(
                f"{url}: row {index} is not [username or null, site title or null, role]"
            )


def make_reader(directory: Path) -> Any:
    """The ``cric`` reader over the five files in ``directory``."""
    from archi.sources.cric import CRICSource

    return CRICSource(
        sites_path=str(directory / "sites.json"),
        storage_units_path=str(directory / "storage_units.json"),
        compute_units_path=str(directory / "compute_units.json"),
        facilities_path=str(directory / "facilities.json"),
        responsibilities_path=str(directory / "responsibilities.json"),
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

    def download() -> dict[str, Any]:
        proxy = require_proxy({"X509_USER_PROXY": args.proxy} if args.proxy else None)
        with CatalogClient(timeout=args.timeout, proxy=proxy, ca_path=args.ca_path) as client:
            payloads = collect(client.get_json, base_url=args.base_url)
        records = write_group_atomic(
            output,
            payloads,
            check=reader_check(make_reader, origins=endpoint_urls(args.base_url)),
        )
        return {
            "output": str(output),
            "files": {name: _count(payload) for name, payload in payloads.items()},
            "records": records,
        }

    return run_cli("cric", download)


if __name__ == "__main__":
    raise SystemExit(main())
