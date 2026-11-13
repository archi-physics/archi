#!/usr/bin/env python3
"""Download production workflows into the cache ``archi.sources.wmstats`` reads.

    python -m archi.downloaders.wmstats [--output-dir DIR] [--status S ...] [--proxy PATH]

One GET per workflow status against the ReqMgr2 request API on cmsweb,
``/reqmgr2/data/request?status=<status>``. Each answer is ReqMgr2's default
format, ``{"result": [{<RequestName>: <request document>}]}`` (``{"result":
[]}`` when no request is in that status), and is kept as sent. That is the
per-status answer of the WMStats collection the reader reads (pact
``wmstats-collection-shape``), so the cache is::

    records.json    {"cutoff_utc": "<when the first request was sent>",
                     "status_payloads": {"<status>": <answer>, ...}}
    authority.json  {"record_count": <workflows in records.json>,
                     "deliberate_merge_count": <duplicates removed>,
                     "statuses": [<statuses queried>]}

Why ReqMgr2 and not wmstatsserver's ``filtered_requests``: that API answers
a list of bare request rows, not ``{name: record}`` objects, which the
reader would not read. Why whole documents and no ``mask``: ReqMgr2 (and
wmstatsserver) apply a mask through ``RequestInfo.get``, which for a
TaskChain or StepChain request returns a *list* gathered over its tasks for
any field a task carries (``CMSSWVersion``, ``Campaign``, ``InputDataset``);
the reader would turn that list into a string id. The whole document keeps
the top-level values the reader is known to read correctly.

Default statuses, WMCore's active production states from ``assigned`` on:
assigned, staging, staged, acquired, running-open, running-closed,
force-complete, completed, closed-out. ``--status`` (repeatable) picks
others from WMCore's ``ACTIVE_STATUS``; they are always queried in that
list's order, which follows the request state machine. A request that
moves forward between two GETs therefore shows up in the later answer
(perhaps in both), never in neither. The exception is the state machine's
one backward step: with ``--status failed``, a request that moves from
failed back to assigned after the assigned GET and before the failed one
is in neither answer, and is missing until the next run. One seen twice
or more keeps only its last answer, its latest status: each earlier copy
is removed and counted in ``deliberate_merge_count``. The reader refuses a
repeated name as a contradiction, so the merge must happen here.

After each status, a progress line (the status, its place in the run, the
workflows it holds, seconds elapsed) goes to stderr; stdout holds only the
JSON summary.

Narrowing ``--status`` narrows the scope: the reader claims a complete
scope over the file and the graph retracts any workflow missing from it.
For a small trial run, pass ``--status`` with ``--output-dir`` elsewhere.

The run needs an X.509 proxy (``X509_USER_PROXY``, else
``/tmp/x509up_u<uid>``, or ``--proxy``), checked before any request; make
one with ``voms-proxy-init -voms cms``. cmsweb presents a CERN Grid CA
certificate: off a grid host, pass ``--ca-path`` a PEM file holding both
the CERN Grid CA and the CERN Root CA 2, or a CA directory.

The files go to ``--output-dir``, default ``wmstats-workflows`` under
``ARCHI_DATA_ROOT`` (else the current directory): where the cms-catalogs
pack's ``wmstats_workflows`` source reads them. Both are staged, the reader
runs over the staged copy (it checks ``record_count`` against the records),
and only health ``ok`` with a complete scope moves them into place; any
failure exits 1 naming the URL and leaves the old files untouched.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

from archi.auth.cache import data_root
from archi.downloaders._catalog import (
    CatalogClient,
    DownloadError,
    reader_check,
    require_proxy,
    run_cli,
    write_group_atomic,
)

DEFAULT_BASE_URL = "https://cmsweb.cern.ch"
DEFAULT_TIMEOUT = 300

#: WMCore ``ReqMgr/DataStructs/RequestStatus.py`` ``ACTIVE_STATUS``, in its
#: order (the state machine's forward order).
ACTIVE_STATUSES = (
    "new",
    "assignment-approved",
    "assigned",
    "staging",
    "staged",
    "acquired",
    "running-open",
    "running-closed",
    "failed",
    "force-complete",
    "completed",
    "closed-out",
    "announced",
    "aborted",
    "aborted-completed",
    "rejected",
)
DEFAULT_STATUSES = (
    "assigned",
    "staging",
    "staged",
    "acquired",
    "running-open",
    "running-closed",
    "force-complete",
    "completed",
    "closed-out",
)

Fetch = Callable[[str], Any]


def status_urls(base_url: str, statuses: Iterable[str]) -> dict[str, str]:
    """Status -> request URL, in :data:`ACTIVE_STATUSES` order."""
    base = base_url.rstrip("/")
    wanted = set(statuses)
    return {
        status: f"{base}/reqmgr2/data/request?status={quote(status)}"
        for status in ACTIVE_STATUSES
        if status in wanted
    }


def collect(fetch: Fetch, urls: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(records.json, authority.json)`` payloads, or :class:`DownloadError`.

    ``fetch`` takes a URL and returns its parsed JSON. ``urls`` maps each
    status to its URL in query order.
    """
    cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started = time.monotonic()
    answers: dict[str, Any] = {}
    for index, (status, url) in enumerate(urls.items(), start=1):
        answer = fetch(url)
        _check_answer(answer, url)
        answers[status] = answer
        found = sum(len(entry) for entry in answer["result"])
        print(
            f"wmstats: {status} {index}/{len(urls)}, {found} workflows, "
            f"{time.monotonic() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )
    merged = _merge_repeats(answers)
    count = sum(len(entry) for answer in answers.values() for entry in answer["result"])
    if not count:
        raise DownloadError(
            f"no workflow in any of the {len(urls)} statuses; refusing to write an "
            f"empty cache (answers from {', '.join(urls.values())})"
        )
    records = {"cutoff_utc": cutoff, "status_payloads": answers}
    authority = {
        "record_count": count,
        "deliberate_merge_count": merged,
        "statuses": list(urls),
    }
    return records, authority


def _check_answer(answer: Any, url: str) -> None:
    """``{"result": [{name: request object}]}``, else :class:`DownloadError`."""
    if not isinstance(answer, dict) or not isinstance(answer.get("result"), list):
        raise DownloadError(f"{url}: answer has no 'result' list (an error body or drifted API)")
    for index, entry in enumerate(answer["result"]):
        if not isinstance(entry, dict):
            raise DownloadError(f"{url}: result entry {index} is not a {{name: request}} object")
        for name, record in entry.items():
            if not isinstance(record, dict):
                raise DownloadError(f"{url}: request {name} is not an object")


def _merge_repeats(answers: dict[str, Any]) -> int:
    """Drop every request that a later answer holds again; return how many.

    ``answers`` is in query order, so the later copy is the newer state.
    """
    later: set[str] = set()
    dropped = 0
    for answer in reversed(list(answers.values())):
        for entry in answer["result"]:
            for name in [name for name in entry if name in later]:
                del entry[name]
                dropped += 1
            later.update(entry)
    for answer in answers.values():
        answer["result"] = [entry for entry in answer["result"] if entry]
    return dropped


def make_reader(directory: Path) -> Any:
    """The ``wmstats_workflows`` reader over ``directory/records.json``."""
    from archi.sources.wmstats import WMStatsWorkflowSource

    return WMStatsWorkflowSource(records_path=str(directory / "records.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: wmstats-workflows under ARCHI_DATA_ROOT (else the current directory).",
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=ACTIVE_STATUSES,
        default=None,
        metavar="STATUS",
        help="A workflow status to download (repeatable; one of WMCore's active "
        f"statuses). Default: {', '.join(DEFAULT_STATUSES)}.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--ca-path",
        default=None,
        help="CA bundle or directory to verify cmsweb with. Default: "
        "/etc/grid-security/certificates when present, else requests' bundle.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="X.509 proxy file. Default: X509_USER_PROXY, else /tmp/x509up_u<uid>.",
    )
    args = parser.parse_args(argv)
    output = Path(args.output_dir) if args.output_dir else data_root() / "wmstats-workflows"
    urls = status_urls(args.base_url, args.status or DEFAULT_STATUSES)

    def download() -> dict[str, Any]:
        proxy = require_proxy({"X509_USER_PROXY": args.proxy} if args.proxy else None)
        with CatalogClient(timeout=args.timeout, proxy=proxy, ca_path=args.ca_path) as client:
            records, authority = collect(client.get_json, urls)
        count = write_group_atomic(
            output,
            {"records.json": records, "authority.json": authority},
            check=reader_check(make_reader, origins=urls),
        )
        return {
            "output": str(output),
            "statuses": {
                status: sum(len(entry) for entry in answer["result"])
                for status, answer in records["status_payloads"].items()
            },
            "deliberate_merge_count": authority["deliberate_merge_count"],
            "records": count,
        }

    return run_cli("wmstats", download)


if __name__ == "__main__":
    raise SystemExit(main())
