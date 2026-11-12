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
reader would find no CMS federation there and refuse the whole cache.

The files go to ``--output-dir``, default ``cric-core`` under
``ARCHI_DATA_ROOT`` (or under the current directory when that variable is
unset), which is where the cms-catalogs pack's ``cric_core`` source reads
them.

All or nothing, like :mod:`archi.downloaders.cric`: the three answers are
fetched and checked first and only then written as one group. A failed
request, a non-JSON, empty or error-shaped answer, a nested field of the
wrong type, or a federations answer in which no federation serves CMS (the
reader refuses that cache: a complete scope would retract every federation)
stops the run with a message naming the endpoint, exit status 1, and the
previous caches untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from archi.downloaders._catalog import DownloadError, data_root, get_json, write_group_atomic
from archi.downloaders.cric import keyed_objects

DEFAULT_BASE_URL = "https://wlcg-cric.cern.ch"
DEFAULT_TIMEOUT = 120

#: Cache file -> API path under the base URL.
ENDPOINTS = {
    "services.json": "/api/core/service/query/?json",
    "rcsites.json": "/api/core/rcsite/query/?json",
    "federations.json": "/api/core/federation/query/?json",
}

Fetch = Callable[[str], Any]


def collect(fetch: Fetch, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Every cache file's payload, checked, or :class:`DownloadError`."""
    base = base_url.rstrip("/")
    urls = {name: base + path for name, path in ENDPOINTS.items()}
    payloads = {name: fetch(url) for name, url in urls.items()}

    for name, service in keyed_objects(payloads["services.json"], urls["services.json"]).items():
        if not isinstance(service.get("rcsite"), str):
            raise DownloadError(f"{urls['services.json']}: service {name} has no 'rcsite' name")

    for name, rcsite in keyed_objects(payloads["rcsites.json"], urls["rcsites.json"]).items():
        if not isinstance(rcsite.get("sites", []), list):
            raise DownloadError(f"{urls['rcsites.json']}: rcsite {name} 'sites' is not a list")

    url = urls["federations.json"]
    federations = keyed_objects(payloads["federations.json"], url)
    for name, federation in federations.items():
        for field in ("vos", "rcsites"):
            if not isinstance(federation.get(field, []), list):
                raise DownloadError(f"{url}: federation {name} '{field}' is not a list")
        if not _pledges_shaped(federation.get("pledges", {})):
            raise DownloadError(
                f"{url}: federation {name} 'pledges' is not year -> quarter -> VO objects"
            )
    if not any(serves_cms(federation) for federation in federations.values()):
        raise DownloadError(
            f"{url}: none of its {len(federations)} federations serves CMS (no 'cms' "
            "VO or pledge); the reader would refuse this cache"
        )
    return payloads


def _pledges_shaped(pledges: Any) -> bool:
    return isinstance(pledges, dict) and all(
        isinstance(quarters, dict)
        and all(isinstance(vos, dict) for vos in quarters.values())
        for quarters in pledges.values()
    )


def serves_cms(federation: dict[str, Any]) -> bool:
    """The reader's rule: a ``cms`` VO, or a ``cms`` entry in any pledge quarter."""
    return "cms" in federation.get("vos", []) or any(
        "cms" in vos
        for quarters in federation.get("pledges", {}).values()
        for vos in quarters.values()
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
    try:
        payloads = collect(
            lambda url: get_json(url, timeout=args.timeout, ca_path=args.ca_path),
            base_url=args.base_url,
        )
        write_group_atomic(output, payloads)
    except (DownloadError, OSError) as exc:
        print(f"cric_core download failed, cache not written: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(
        {
            "output": str(output),
            "files": {name: len(payload) for name, payload in payloads.items()},
            "cms_federations": sum(
                serves_cms(federation) for federation in payloads["federations.json"].values()
            ),
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
