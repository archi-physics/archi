#!/usr/bin/env python3
"""Download CMS Site Support Team site status into the cache ``archi.sources.cmssst`` reads.

The CMS SST site status detail page
(``https://cmssst.web.cern.ch/siteStatus/detail.html?site=<SITE>``) is an
empty shell: its script loads ``data/<SITE>.json`` and draws it. The list of
sites is the JavaScript file ``data/summary.js``
(``var siteStatusData = [ { site: "T0_CH_CERN", ... }, ... ]``). Both are
public and need no login; the data is refreshed every 900 seconds.

    python -m archi.downloaders.cmssst [--output <records.json>]

Without ``--output`` the cache goes to
``data/cmssst-site-status/records.json`` under ``ARCHI_DATA_ROOT`` (or under
the current directory when that variable is unset).

The cache is a JSON list, one object per site, carrying only the site-level
fields the reader reads: ``site``, ``time``, ``alert``, ``msg``, ``ggus`` and
``metrics``. The per-host ``elements`` list (most of each 190 KB file) is
dropped.

All or nothing: the reader claims a complete scope over the cache, and under
``missing_from_completed_scope`` a site missing from it is retracted from the
graph. So a summary.js without a site list, a site name that is not a CMS
site name, a failed or error-shaped site fetch, or an empty result stops the
run with an error and leaves any existing cache untouched.

Known limit: because of that rule, one site whose JSON is briefly
unavailable fails the whole refresh; the previous cache stays in place
until a later run succeeds.

summary.js is parsed only inside the ``siteStatusData`` array literal, with
``//`` and ``/* */`` comments removed, so a commented-out entry or a later
array in the same file is not read as a site.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
DEFAULT_BASE_URL = "https://cmssst.web.cern.ch/siteStatus/data/"
DEFAULT_OUTPUT = Path("data/cmssst-site-status/records.json")
DEFAULT_TIMEOUT = 60

#: The site-name check the detail page itself applies (detail.html), anchored.
SITE_NAME = re.compile(r"^T[0-9]_[A-Z]{2}_\w+$")
#: Fields kept per site; everything else (``elements``, ``url``, ``reload``)
#: is dropped.
KEPT_FIELDS = ("site", "time", "alert", "msg", "ggus", "metrics")

_DATA_START = re.compile(r"\bvar\s+siteStatusData\s*=\s*\[")
_SITE_ENTRY = re.compile(r"\{\s*site\s*:\s*\"([^\"]*)\"")

Fetch = Callable[[str, int], str]


class DownloadError(RuntimeError):
    """The download cannot produce a complete cache; nothing is written."""


def _data_root() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV)
    return Path(raw).expanduser() if raw else Path.cwd()


def parse_site_list(script: str) -> list[str]:
    """Site names from summary.js, in file order, duplicates removed.

    Raises :class:`DownloadError` when the ``siteStatusData`` array is
    missing, holds no site entry, or names a site that is not a CMS site
    name (an error page or a changed format, not an empty site list).
    """
    start = _DATA_START.search(script)
    if start is None:
        raise DownloadError(
            "summary.js has no 'var siteStatusData = [' array; "
            "an error page or a changed format"
        )
    names = _SITE_ENTRY.findall(_array_body(script, start.end()))
    if not names:
        raise DownloadError("summary.js siteStatusData holds no site entry")
    bad = [name for name in names if not SITE_NAME.match(name)]
    if bad:
        raise DownloadError(
            f"summary.js names {len(bad)} site(s) that are not CMS site "
            f"names, first {bad[0]!r}"
        )
    return list(dict.fromkeys(names))


def _array_body(script: str, begin: int) -> str:
    """The array literal's text from ``begin`` (just after its ``[``) to
    its matching ``]``, comments removed and string literals kept whole.

    Raises :class:`DownloadError` when the array is never closed (a
    truncated file).
    """
    out: list[str] = []
    depth = 1
    index = begin
    length = len(script)
    while index < length:
        char = script[index]
        if char in "\"'":
            end = index + 1
            while end < length and script[end] != char:
                end += 2 if script[end] == "\\" else 1
            out.append(script[index:end + 1])
            index = end + 1
            continue
        if script.startswith("//", index):
            newline = script.find("\n", index)
            index = length if newline < 0 else newline
            continue
        if script.startswith("/*", index):
            close = script.find("*/", index + 2)
            if close < 0:
                break
            out.append(" ")
            index = close + 2
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(char)
        index += 1
    raise DownloadError("summary.js siteStatusData array is never closed")


def site_record(payload: Any, site: str) -> dict[str, Any]:
    """The cached record for one site's JSON, or :class:`DownloadError`.

    The payload must be an object for ``site`` with an integer snapshot
    ``time``, a ``ggus`` list and a non-empty ``metrics`` object whose
    values are objects; anything else is an error-shaped body.
    """
    if not isinstance(payload, dict):
        raise DownloadError(
            f"{site}: site JSON is a {type(payload).__name__}, not an object"
        )
    if payload.get("site") != site:
        raise DownloadError(
            f"{site}: site JSON names site {payload.get('site')!r}"
        )
    snapshot = payload.get("time")
    if not isinstance(snapshot, int) or isinstance(snapshot, bool) or snapshot <= 0:
        raise DownloadError(f"{site}: site JSON has no positive integer 'time'")
    if not isinstance(payload.get("ggus", []), list):
        raise DownloadError(f"{site}: site JSON 'ggus' is not a list")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise DownloadError(f"{site}: site JSON has no 'metrics' object")
    if not all(isinstance(value, dict) for value in metrics.values()):
        raise DownloadError(f"{site}: a site metric is not an object")
    record = {field: payload[field] for field in KEPT_FIELDS if field in payload}
    record.setdefault("alert", "")
    record.setdefault("msg", "")
    record.setdefault("ggus", [])
    return record


def collect(
    fetch: Fetch,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    sites: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Every site's record, or :class:`DownloadError` on the first failure."""
    base = base_url if base_url.endswith("/") else base_url + "/"
    if sites is None:
        names = parse_site_list(fetch(base + "summary.js", timeout))
    else:
        names = list(dict.fromkeys(sites))
        bad = [name for name in names if not SITE_NAME.match(name)]
        if bad:
            raise DownloadError(f"not a CMS site name: {bad[0]!r}")
    records: list[dict[str, Any]] = []
    for name in names:
        url = f"{base}{name}.json"
        body = fetch(url, timeout)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise DownloadError(
                f"{name}: {url} is not JSON ({exc.msg} at line {exc.lineno})"
            ) from exc
        records.append(site_record(payload, name))
    if not records:
        raise DownloadError("no site records; refusing to write an empty cache")
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the cache atomically; refuse an empty one."""
    if not records:
        raise DownloadError("no site records; refusing to write an empty cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(records, out, ensure_ascii=False, sort_keys=True)
            out.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def http_fetch(url: str, timeout: int) -> str:
    """GET ``url`` with no credentials; a non-200 answer is an error."""
    import requests

    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise DownloadError(f"GET {url} returned HTTP {response.status_code}")
    response.encoding = response.encoding or "utf-8"
    return response.text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)
    output = Path(args.output) if args.output else _data_root() / DEFAULT_OUTPUT
    try:
        records = collect(http_fetch, base_url=args.base_url, timeout=args.timeout)
        write_records(output, records)
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        print(f"cmssst download failed, cache not written: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output), "sites": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
