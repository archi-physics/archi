#!/usr/bin/env python3
"""Download WLCG CRIC core data into the caches ``archi.sources.cric`` (``cric_core``) reads.

    python -m archi.downloaders.cric_core [--output-dir DIR]

Three anonymous GETs against the WLCG CRIC core API
(``https://wlcg-cric.cern.ch``), each answer kept as CRIC sends it:

====================  ===================================
cache file            endpoint
====================  ===================================
``services.json``     ``/api/core/service/query/?json``
``rcsites.json``      ``/api/core/rcsite/query/?json``
``federations.json``  ``/api/core/federation/query/?json``
====================  ===================================

WLCG CRIC, not CMS CRIC: ``cms-cric.cern.ch`` serves the same core paths,
but its federations carry an empty ``vos`` list and no CMS pledge, so the
reader would find no CMS federation there and refuse the whole cache. Like
CMS CRIC it presents a CERN Grid CA certificate; off a grid host, pass
``--ca-path``.

The files go to ``--output-dir``, default ``cric-core`` under
``ARCHI_DATA_ROOT`` (or under the current directory when that variable is
unset), which is where the cms-catalogs pack's ``cric_core`` source reads
them.

All or nothing, like :mod:`archi.downloaders.cric`: the three answers are
fetched, pre-checked as non-empty ``name -> object`` maps, staged, and
moved into place only if the ``cric_core`` reader reads the staged copy with
health ``ok`` and a complete scope. So a federations answer in which no
federation serves CMS, or one whose pledges the reader cannot read, exits 1
with the reader's reason and the endpoint, and the previous caches stay.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from archi.downloaders._catalog import (
    CatalogClient,
    data_root,
    keyed_objects,
    reader_check,
    run_cli,
    write_group_atomic,
)

DEFAULT_BASE_URL = "https://wlcg-cric.cern.ch"
DEFAULT_TIMEOUT = 120

#: Cache file -> API path under the base URL.
ENDPOINTS = {
    "services.json": "/api/core/service/query/?json",
    "rcsites.json": "/api/core/rcsite/query/?json",
    "federations.json": "/api/core/federation/query/?json",
}

Fetch = Callable[[str], Any]


def endpoint_urls(base_url: str) -> dict[str, str]:
    """Cache file -> full URL."""
    base = base_url.rstrip("/")
    return {name: base + path for name, path in ENDPOINTS.items()}


def collect(fetch: Fetch, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Every cache file's payload, pre-checked, or :class:`DownloadError`."""
    urls = endpoint_urls(base_url)
    payloads = {name: fetch(url) for name, url in urls.items()}
    for name, url in urls.items():
        keyed_objects(payloads[name], url)
    return payloads


def make_reader(directory: Path) -> Any:
    """The ``cric_core`` reader over the three files in ``directory``."""
    from archi.sources.cric import CRICCoreSource

    return CRICCoreSource(
        services_path=str(directory / "services.json"),
        rcsites_path=str(directory / "rcsites.json"),
        federations_path=str(directory / "federations.json"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: cric-core under ARCHI_DATA_ROOT (else the current directory).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--ca-path",
        default=None,
        help="CA bundle or directory to verify CRIC with. Default: "
        "/etc/grid-security/certificates when present, else requests' bundle.",
    )
    args = parser.parse_args(argv)
    output = Path(args.output_dir) if args.output_dir else data_root() / "cric-core"

    def download() -> dict[str, Any]:
        with CatalogClient(timeout=args.timeout, ca_path=args.ca_path) as client:
            payloads = collect(client.get_json, base_url=args.base_url)
        records = write_group_atomic(
            output,
            payloads,
            check=reader_check(make_reader, origins=endpoint_urls(args.base_url)),
        )
        return {
            "output": str(output),
            "files": {name: len(payload) for name, payload in payloads.items()},
            "records": records,
        }

    return run_cli("cric_core", download)


if __name__ == "__main__":
    raise SystemExit(main())
