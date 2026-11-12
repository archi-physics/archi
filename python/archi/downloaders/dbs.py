#!/usr/bin/env python3
"""Download DBS dataset metadata into the cache ``archi.sources.dbs`` reads.

    python -m archi.downloaders.dbs [--output-dir DIR] [--pattern P ...] [--proxy PATH]

Metadata only, never files. Which datasets:

- by default, every input and output dataset of the workflows in the WMStats
  cache (``wmstats-workflows/records.json`` under ``ARCHI_DATA_ROOT``, or
  ``--wmstats-records``), as the ``wmstats_workflows`` reader itself reads
  them; refresh it first with ``python -m archi.downloaders.wmstats``;
- plus, for each ``--pattern`` (repeatable, a DBS wildcard such as
  ``/*/Run2026*/AOD``), the VALID datasets DBS lists for it.

Requests, against the DBS global reader (default
``https://cmsweb.cern.ch/dbs/prod/global/DBSReader``), each dataset asked for
once, over one connection pool:

- ``datasets?dataset=<pattern>&detail=1``, one per pattern;
- ``datasets?dataset=<name>&dataset_access_type=*&detail=1``, one per
  workflow dataset no pattern already returned (any access type, since a
  running workflow's output is PRODUCTION, not VALID);
- ``filesummaries?dataset=<name>&validFileOnly=1``, one per dataset found:
  its valid files, events and bytes, as WMCore's ``DBS3Reader`` counts them.

A workflow dataset DBS does not know (an answer of ``[]``: a typo, or an
output not registered yet) is skipped and counted in the summary; the run
fails only when no dataset at all is found. A pattern that matches nothing
adds nothing and is reported with its count.

``records.json`` is a list with one record per dataset, sorted by name, with
only the keys the reader reads: ``dataset``, ``primary_ds_name``,
``processed_ds_name``, ``data_tier_name``, ``dataset_access_type``,
``physics_group_name``, ``creation_date`` (DBS's epoch seconds), ``nevents``,
``nfiles`` and ``dataset_size``. ``create_by`` and ``last_modified_by`` (a
person's DN) are not kept.

``--limit N`` looks up only the first N names in sorted order, for a short
trial. That is a partial catalog, and the reader claims a complete scope over
its file, so ``--limit`` needs an ``--output-dir`` other than the pack's
cache.

The run needs an X.509 proxy (``X509_USER_PROXY``, else
``/tmp/x509up_u<uid>``, or ``--proxy``), checked before any request; make
one with ``voms-proxy-init -voms cms``. Off a grid host, pass ``--ca-path`` a
bundle holding the CERN Grid CA. The file goes to ``--output-dir``, default
``dbs-datasets`` under ``ARCHI_DATA_ROOT``: where the cms-catalogs pack's
``dbs_datasets`` source reads it. It is staged, the reader runs over the
staged copy, and only health ``ok`` with a complete scope moves it into
place; any failure exits 1 naming the URL and leaves the old file untouched.
"""
from __future__ import annotations

import argparse
import re
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
    write_json_atomic,
)

DEFAULT_BASE_URL = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader"
DEFAULT_TIMEOUT = 120
WMSTATS_COMMAND = "`python -m archi.downloaders.wmstats`"
#: The dataset row keys the reader reads; DBS sends more.
METADATA_KEYS = (
    "dataset",
    "primary_ds_name",
    "processed_ds_name",
    "data_tier_name",
    "dataset_access_type",
    "physics_group_name",
    "creation_date",
)
UNRESOLVED_SAMPLE = 20
_DATASET_PATH = re.compile(r"^/[^/*?]+/[^/*?]+/[^/*?]+$")

Fetch = Callable[[str], Any]


def datasets_url(base_url: str, name: str) -> str:
    return f"{base_url.rstrip('/')}/datasets?dataset={quote(name)}&dataset_access_type=*&detail=1"


def pattern_url(base_url: str, pattern: str) -> str:
    return f"{base_url.rstrip('/')}/datasets?dataset={quote(pattern, safe='/*')}&detail=1"


def summary_url(base_url: str, name: str) -> str:
    return f"{base_url.rstrip('/')}/filesummaries?dataset={quote(name)}&validFileOnly=1"


def workflow_datasets(records_path: Path) -> set[str]:
    """Every dataset the WMStats cache names, read by its own reader.

    :class:`DownloadError` when the reader does not accept the cache whole
    (health ``ok`` and a complete scope): a partial workflow list would make
    a partial dataset catalog.
    """
    from okg.deployment import NodeFact

    from archi.sources.wmstats import WMStatsWorkflowSource

    run = WMStatsWorkflowSource(records_path=str(records_path)).run(
        "archi-dbs-scope", mode="scope_complete"
    )
    names = {
        fact.attrs["dataset_name"]
        for fact in run.facts
        if isinstance(fact, NodeFact) and fact.subtype == "dataset"
    }
    if run.health.status != "ok" or not run.completed_scope:
        raise DownloadError(
            f"the WMStats cache at {records_path} is not usable (health "
            f"{run.health.status}): {run.health.reason}; refresh it with {WMSTATS_COMMAND}"
        )
    return names


def collect(
    fetch: Fetch,
    *,
    base_url: str = DEFAULT_BASE_URL,
    names: Iterable[str] = (),
    patterns: Iterable[str] = (),
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(records, report)``, or :class:`DownloadError`.

    ``fetch`` takes a URL and returns its parsed JSON. ``names`` are exact
    dataset names, ``patterns`` DBS wildcards.
    """
    rows: dict[str, dict[str, Any]] = {}
    matched: dict[str, int] = {}
    for pattern in dict.fromkeys(patterns):
        url = pattern_url(base_url, pattern)
        found = _dataset_rows(fetch(url), url)
        matched[pattern] = len(found)
        for row in found:
            rows.setdefault(row["dataset"], row)
    names = set(names)
    wanted = sorted(names | set(rows))
    if limit is not None:
        wanted = wanted[:limit]
    unresolved: list[str] = []
    for name in wanted:
        if name in rows:
            continue
        if not _DATASET_PATH.match(name):
            unresolved.append(name)
            continue
        url = datasets_url(base_url, name)
        found = _dataset_rows(fetch(url), url)
        if not found:
            unresolved.append(name)
            continue
        if [row["dataset"] for row in found] != [name]:
            raise DownloadError(
                f"{url}: asked for {name}, DBS answered "
                f"{', '.join(row['dataset'] for row in found)}"
            )
        rows[name] = found[0]
    found_names = [name for name in wanted if name in rows]
    if not found_names:
        raise DownloadError(
            f"none of the {len(wanted)} datasets is known to DBS at {base_url}; "
            "refusing to write an empty catalog"
        )
    records = []
    for name in found_names:
        url = summary_url(base_url, name)
        counts = _summary_row(fetch(url), url)
        record = {key: rows[name].get(key) for key in METADATA_KEYS}
        record["nfiles"] = counts.get("num_file")
        record["nevents"] = counts.get("num_event")
        record["dataset_size"] = counts.get("file_size")
        records.append(record)
    report = {
        "workflow_datasets": len(names),
        "patterns": matched,
        "unresolved": len(unresolved),
        "unresolved_sample": unresolved[:UNRESOLVED_SAMPLE],
    }
    return records, report


def _dataset_rows(answer: Any, url: str) -> list[dict[str, Any]]:
    if not isinstance(answer, list) or not all(
        isinstance(row, dict) and isinstance(row.get("dataset"), str) and row["dataset"]
        for row in answer
    ):
        raise DownloadError(f"{url}: answer is not a list of dataset rows (an error body or drifted API)")
    return answer


def _summary_row(answer: Any, url: str) -> dict[str, Any]:
    """The one summary row; ``[]`` (no valid file yet) is all zeros."""
    if not isinstance(answer, list) or len(answer) > 1 or not all(isinstance(r, dict) for r in answer):
        raise DownloadError(f"{url}: answer is not a list of at most one summary row")
    if not answer:
        return {"num_file": 0, "num_event": 0, "file_size": 0}
    return answer[0]


def make_reader(directory: Path) -> Any:
    """The ``dbs_datasets`` reader over ``directory/records.json``."""
    from archi.sources.dbs import DBSDatasetSource

    return DBSDatasetSource(records_path=str(directory / "records.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: dbs-datasets under ARCHI_DATA_ROOT (else the current directory).",
    )
    parser.add_argument(
        "--wmstats-records",
        default=None,
        help="The WMStats cache whose datasets to look up. Default: "
        "wmstats-workflows/records.json under ARCHI_DATA_ROOT.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Also download the VALID datasets matching this DBS pattern (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Look up only the first N dataset names (a trial; needs --output-dir "
        "other than the pack's cache).",
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
    pack_cache = data_root() / "dbs-datasets"
    output = Path(args.output_dir) if args.output_dir else pack_cache
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        if output.resolve() == pack_cache.resolve():
            parser.error(
                "--limit writes a partial catalog, and the reader retracts every "
                f"dataset missing from {pack_cache}; pass --output-dir elsewhere"
            )
    wmstats_records = (
        Path(args.wmstats_records)
        if args.wmstats_records
        else data_root() / "wmstats-workflows" / "records.json"
    )

    def download() -> dict[str, Any]:
        if wmstats_records.is_file():
            names = workflow_datasets(wmstats_records)
        elif args.pattern:
            names = set()
        else:
            raise DownloadError(
                f"no WMStats cache at {wmstats_records}, so no workflow datasets to "
                f"look up; run {WMSTATS_COMMAND} first, or pass --pattern"
            )
        proxy = require_proxy({"X509_USER_PROXY": args.proxy} if args.proxy else None)
        requests = 0

        with CatalogClient(timeout=args.timeout, proxy=proxy, ca_path=args.ca_path) as client:

            def fetch(url: str) -> Any:
                nonlocal requests
                requests += 1
                return client.get_json(url)

            records, report = collect(
                fetch,
                base_url=args.base_url,
                names=names,
                patterns=args.pattern,
                limit=args.limit,
            )
        count = write_json_atomic(
            output / "records.json",
            records,
            check=reader_check(make_reader, origins={"records.json": args.base_url}),
        )
        summary = {"output": str(output), "records": count, "requests": requests, **report}
        if args.limit is not None:
            summary["limit"] = args.limit
        return summary

    return run_cli("dbs", download)


if __name__ == "__main__":
    raise SystemExit(main())
