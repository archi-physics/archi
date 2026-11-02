"""Download Redmine issues into the cache ``archi.sources.tickets`` reads.

Run it from an operator environment::

    python -m archi.downloaders.redmine --base-url https://redmine.example.org --project ops
    REDMINE_API_KEY=... python -m archi.downloaders.redmine --base-url https://redmine.example.org

The API key is optional. Without one the download is anonymous and sees only
what the Redmine allows the public to see; with one (``REDMINE_API_KEY``, sent
as the ``X-Redmine-API-Key`` header) it sees what that account sees. The key is
never printed or written. The download is read-only (GET ``/issues.json`` with
journals), and every record is tagged ``"system": "redmine"``.

``records.json`` and ``meta.json`` are written atomically under ``--output-dir``
(default ``tickets`` under ``ARCHI_DATA_ROOT``, where the pack reads it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from archi.downloaders.servicenow import DownloadError, Session
from archi.sources._cluster_cache import write_cache

DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
KEY_ENV = "REDMINE_API_KEY"
TIMEOUT = 60


def fetch_issues(
    session: Session,
    base_url: str,
    *,
    api_key: str = "",
    projects: Sequence[str] = (),
    page_size: int = 100,
    max_issues: int = 0,
) -> list[dict[str, Any]]:
    """Every issue (open and closed) of the projects, or of everything visible, newest first."""
    headers = {"Accept": "application/json", **({"X-Redmine-API-Key": api_key} if api_key else {})}
    issues: list[dict[str, Any]] = []
    for project in projects or (None,):
        offset = 0
        while True:
            size = page_size if not max_issues else min(page_size, max_issues - len(issues))
            if size <= 0:
                return issues
            params: dict[str, Any] = {
                "status_id": "*", "limit": size, "offset": offset, "sort": "updated_on:desc", "include": "journals",
            }
            if project:
                params["project_id"] = project
            response = session.get(f"{base_url}/issues.json", params=params, headers=headers, timeout=TIMEOUT)
            if response.status_code != 200:
                where = f"project {project}" if project else "the issue list"
                hint = "" if api_key else f" (anonymous; set {KEY_ENV} if the project is private)"
                raise DownloadError(f"Redmine {where} could not be read (HTTP {response.status_code}){hint}")
            body = response.json() or {}
            page = body.get("issues") or []
            issues.extend({**issue, "system": "redmine"} for issue in page)
            offset += len(page)
            if len(page) < size or offset >= int(body.get("total_count", offset)):
                break
    return issues


def download(
    session: Session,
    base_url: str,
    env: Mapping[str, str],
    output_dir: Path,
    *,
    projects: Sequence[str] = (),
    page_size: int = 100,
    max_issues: int = 0,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    issues = fetch_issues(
        session, base_url, api_key=env.get(KEY_ENV, ""), projects=projects, page_size=page_size, max_issues=max_issues
    )
    if not issues:
        raise DownloadError("Redmine returned no issues; not writing an empty cache")
    return write_cache(
        output_dir, issues, len(issues), base_url=base_url, projects=list(projects), authenticated=bool(env.get(KEY_ENV))
    )


def default_output_dir() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV)
    return (Path(raw).expanduser() if raw else Path.cwd()) / "tickets"


def main(argv: Optional[Sequence[str]] = None, *, session: Optional[Session] = None, env: Optional[Mapping[str, str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project", dest="projects", action="append", help="project identifier; repeatable")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-issues", type=int, default=0, help="cap for probes; 0 means no cap")
    args = parser.parse_args(argv)
    environment = os.environ if env is None else env
    if not environment.get(KEY_ENV):
        print(f"no {KEY_ENV} set; reading anonymously, so only public projects are visible", file=sys.stderr)
    if session is None:
        import requests

        session = requests.Session()
    try:
        info = download(
            session, args.base_url, environment, Path(args.output_dir) if args.output_dir else default_output_dir(),
            projects=tuple(args.projects or ()), page_size=args.page_size, max_issues=args.max_issues,
        )
    except DownloadError as exc:
        print(f"redmine download stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"record_count": info["record_count"], "content_hash": info["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
