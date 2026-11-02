"""Download ServiceNow incidents, requests and KB articles into the cache ``archi.sources.servicenow`` reads.

Run it from an operator environment::

    SERVICENOW_CLIENT_ID=... SERVICENOW_CLIENT_SECRET=... \\
        python -m archi.downloaders.servicenow --instance-url https://example.service-now.com --auth oauth
    SERVICENOW_USERNAME=... SERVICENOW_PASSWORD=... \\
        python -m archi.downloaders.servicenow --instance-url https://example.service-now.com --auth basic

The credentials are read from the environment variables named above and never
printed or written. ``--auth oauth`` asks the instance's ``/oauth_token.do`` for
a token with the client-credentials grant; ``--auth basic`` sends HTTP basic
auth. The download is read-only: it issues GETs against the Table API with
display values and without reference links, so a caller or an assignee arrives
as a name.

``records.json`` and ``meta.json`` are written atomically under ``--output-dir``
(default ``servicenow`` under ``ARCHI_DATA_ROOT``, where the pack reads it). A table the account
cannot read stops the run with the table and the HTTP status named.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from archi.sources._cluster_cache import write_cache
from archi.sources.servicenow import CREDENTIALS, TABLES

DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
TIMEOUT = 60

FIELDS: dict[str, tuple[str, ...]] = {
    "incident": (
        "number", "sys_id", "short_description", "description", "state", "priority", "urgency", "impact",
        "category", "assignment_group", "assigned_to", "opened_by", "caller_id", "cmdb_ci", "opened_at",
        "sys_updated_on", "resolved_at", "close_notes",
    ),
    "sc_request": (
        "number", "sys_id", "short_description", "description", "request_state", "state", "priority",
        "assignment_group", "assigned_to", "opened_by", "requested_for", "opened_at", "sys_updated_on",
    ),
    "kb_knowledge": (
        "number", "sys_id", "short_description", "text", "author", "kb_knowledge_base", "workflow_state",
        "sys_updated_on", "published",
    ),
}


class Response(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


class Session(Protocol):
    """The slice of ``requests.Session`` this downloader uses (tests pass a fake)."""

    def get(self, url: str, **kwargs: Any) -> Response: ...

    def post(self, url: str, **kwargs: Any) -> Response: ...


class DownloadError(RuntimeError):
    pass


def credentials(method: str, env: Mapping[str, str]) -> tuple[str, str]:
    """The two values for ``method``, or a ``DownloadError`` that names the missing variables."""
    names = CREDENTIALS[method]
    missing = [name for name in names if not env.get(name)]
    if missing:
        raise DownloadError(f"--auth {method} needs {', '.join(missing)} in the environment")
    return env[names[0]], env[names[1]]


def auth_headers(session: Session, instance_url: str, method: str, env: Mapping[str, str]) -> dict[str, Any]:
    """Request keyword arguments carrying the credentials (a bearer header or basic auth)."""
    first, second = credentials(method, env)
    if method == "basic":
        return {"auth": (first, second)}
    response = session.post(
        f"{instance_url}/oauth_token.do",
        data={"grant_type": "client_credentials", "client_id": first, "client_secret": second},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        raise DownloadError(f"ServiceNow OAuth token request failed (HTTP {response.status_code})")
    token = (response.json() or {}).get("access_token")
    if not token:
        raise DownloadError("ServiceNow OAuth answer holds no access_token")
    return {"headers": {"Authorization": f"Bearer {token}"}}


def fetch_table(
    session: Session,
    instance_url: str,
    table: str,
    auth: Mapping[str, Any],
    *,
    page_size: int = 200,
    max_records: int = 0,
    query: str = "ORDERBYsys_id",  # stable: sys_id never changes, so paging cannot skip or repeat a row
) -> list[dict[str, Any]]:
    """Every row of one table, a page at a time, tagged with its ``table``."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        size = page_size if not max_records else min(page_size, max_records - len(rows))
        if size <= 0:
            break
        response = session.get(
            f"{instance_url}/api/now/table/{table}",
            params={
                "sysparm_limit": size,
                "sysparm_offset": offset,
                "sysparm_display_value": "true",
                "sysparm_exclude_reference_link": "true",
                "sysparm_fields": ",".join(FIELDS[table]),
                "sysparm_query": query,
            },
            timeout=TIMEOUT,
            headers={"Accept": "application/json", **auth.get("headers", {})},
            **({"auth": auth["auth"]} if "auth" in auth else {}),
        )
        if response.status_code != 200:
            raise DownloadError(f"ServiceNow table {table} could not be read (HTTP {response.status_code})")
        page = (response.json() or {}).get("result") or []
        rows.extend({**row, "table": table} for row in page)
        offset += len(page)
        if len(page) < size:
            break
    return rows


def download(
    session: Session,
    instance_url: str,
    method: str,
    env: Mapping[str, str],
    output_dir: Path,
    *,
    tables: Sequence[str] = tuple(TABLES),
    page_size: int = 200,
    max_records: int = 0,
) -> dict[str, Any]:
    instance_url = instance_url.rstrip("/")
    if not instance_url.lower().startswith("https://"):
        # the credentials and the token travel in this URL's requests
        raise DownloadError(f"--instance-url must be https://, got {instance_url!r}")
    unknown = [t for t in tables if t not in TABLES]
    if unknown:
        raise DownloadError(f"unknown table(s) {unknown}; choose from {sorted(TABLES)}")
    auth = auth_headers(session, instance_url, method, env)
    records: list[dict[str, Any]] = []
    for table in tables:
        records.extend(fetch_table(session, instance_url, table, auth, page_size=page_size, max_records=max_records))
    if not records:
        raise DownloadError("ServiceNow returned no records for the chosen tables; not writing an empty cache")
    return write_cache(output_dir, records, len(records), instance_url=instance_url, tables=list(tables), auth_method=method)


def default_output_dir() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV)
    return (Path(raw).expanduser() if raw else Path.cwd()) / "servicenow"


def main(argv: Optional[Sequence[str]] = None, *, session: Optional[Session] = None, env: Optional[Mapping[str, str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instance-url", required=True, help="e.g. https://example.service-now.com")
    parser.add_argument("--auth", choices=sorted(CREDENTIALS), default="oauth")
    parser.add_argument("--table", dest="tables", action="append", choices=sorted(TABLES), help="repeatable; default all three")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-records", type=int, default=0, help="per-table cap for probes; 0 means no cap")
    args = parser.parse_args(argv)
    if session is None:
        import requests

        session = requests.Session()
    try:
        info = download(
            session, args.instance_url, args.auth, os.environ if env is None else env,
            Path(args.output_dir) if args.output_dir else default_output_dir(),
            tables=tuple(args.tables or TABLES), page_size=args.page_size, max_records=args.max_records,
        )
    except DownloadError as exc:
        print(f"servicenow download stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"record_count": info["record_count"], "content_hash": info["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
