"""Download CERN JIRA issues into the cache ``archi.sources.jira`` reads.

Moved from okg-deployments ``cms/scripts/download_jira.py`` (main
``32f2b4e3c2``). It writes ``records.json`` and ``meta.json`` atomically,
so the JIRA reader can ingest the cache without talking to the network.

Run it from an operator environment that has the ``jira`` package
(``pip install jira``)::

    python -m archi.downloaders.jira --project ADAPTYST --output-dir <dir>
    CERN_JIRA_TOKEN=... python -m archi.downloaders.jira --output-dir <dir>

The token is optional. Without one the download is anonymous and sees only
public projects; a project that is not visible stops the run with a message
that says so. The default project list is the CMS set, which needs a token.

Without ``--output-dir`` the cache goes to ``data/jira`` under
``ARCHI_DATA_ROOT`` (or under the current directory when that variable is
unset), the path the reader's defaults and ``docs/docs/reference/connector-caches.md`` name.

Changes from the okg-deployments copy: the default output directory follows
``ARCHI_DATA_ROOT`` instead of the old repository layout
(``<repo>/data/cms/jira``), a missing ``jira`` package stops with a
message that names it, the token is optional (anonymous when unset), and a
search JIRA refuses stops the run with a plain message (status and JIRA's
own text only). Records and metadata are written byte for byte as
before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
DEFAULT_SERVER = "https://its.cern.ch/jira"
DEFAULT_PROJECTS = (
    "CMSCOMPPR",
    "CMSPROD",
    "CMSRUCIO",
    "CMSALCA",
    "CMSDM",
    "CMSTRANSF",
    "CMSMONIT",
    "CMSVOC",
    "CMSTZ",
    "PRCAMPAIGNS",
)
FIELDS = (
    "summary,description,status,priority,issuetype,assignee,"
    "created,updated,labels,components,reporter,resolution,"
    "fixVersions,comment,issuelinks,subtasks,parent,environment,duedate"
)


@dataclass(frozen=True)
class DownloadStats:
    record_count: int
    content_hash: str
    started_at: float
    completed_at: float


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # The token is optional: a public project reads without one. With no
    # token the client runs anonymously, and a project that is not public
    # fails with a message that says so (see _project_error).
    token = _token(args.token_env)
    if not token:
        print(
            "no JIRA token set (checked: "
            f"{', '.join(args.token_env)}); reading anonymously, so only "
            "public projects are visible",
            file=sys.stderr,
            flush=True,
        )

    projects = tuple(args.projects or DEFAULT_PROJECTS)
    records = list(_fetch_records(
        server=args.server,
        token=token,
        projects=projects,
        page_size=args.page_size,
        max_issues=args.max_issues,
        include_comments=args.include_comments,
    ))
    if args.probe_only:
        print(json.dumps({
            "ok": True,
            "record_count": len(records),
            "first_key": records[0]["key"] if records else None,
            "projects": list(projects),
        }, sort_keys=True))
        return 0

    output_dir = Path(args.output_dir)
    stats = _write_cache(
        records,
        output_dir=output_dir,
        server=args.server,
        projects=projects,
        include_comments=args.include_comments,
        started_at=time.time(),
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "record_count": stats.record_count,
        "content_hash": stats.content_hash,
        "elapsed_s": round(stats.completed_at - stats.started_at, 3),
    }, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory that will receive records.json and meta.json. "
            "Default: data/jira under ARCHI_DATA_ROOT, or under the current "
            "directory when ARCHI_DATA_ROOT is unset."
        ),
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument(
        "--project",
        dest="projects",
        action="append",
        help=(
            "JIRA project key to fetch. Repeatable. Defaults to the CMS "
            "baseline project set."
        ),
    )
    parser.add_argument(
        "--token-env",
        action="append",
        default=["CERN_JIRA_TOKEN", "JIRA_CERN_TOKEN"],
        help="Credential env var to try. Repeatable.",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--max-issues",
        type=int,
        default=0,
        help="Optional total issue cap for probes. 0 means no cap.",
    )
    parser.add_argument(
        "--include-comments",
        action="store_true",
        help="Fetch all comments per issue. Slower; descriptions are always full.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Fetch according to the limit but do not write cache files.",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(default_output_dir())
    return args


def default_output_dir() -> Path:
    """``data/jira`` under ``ARCHI_DATA_ROOT`` (else the current directory).

    The same base ``archi.auth.cache.data_root`` resolves reader paths against.
    """
    raw = os.environ.get(DATA_ROOT_ENV)
    base = Path(raw).expanduser() if raw else Path.cwd()
    return base / "data" / "jira"


def _token(env_names: Iterable[str]) -> str:
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _fetch_records(
    *,
    server: str,
    token: str,
    projects: tuple[str, ...],
    page_size: int,
    max_issues: int,
    include_comments: bool,
) -> Iterable[dict[str, Any]]:
    try:
        from jira import JIRA
    except ImportError as exc:
        raise SystemExit(
            "the 'jira' Python package is not installed in this environment; "
            "install it with: pip install jira "
            "to download the JIRA cache"
        ) from exc

    if token:
        client = JIRA(server=server.rstrip("/"), token_auth=token)
    else:
        client = JIRA(server=server.rstrip("/"))
    remaining = max_issues if max_issues > 0 else None
    for project in projects:
        start = 0
        while True:
            batch_size = page_size
            if remaining is not None:
                if remaining <= 0:
                    return
                batch_size = min(batch_size, remaining)
            try:
                results = client.search_issues(
                    f"project={project} ORDER BY updated DESC",
                    startAt=start,
                    maxResults=batch_size,
                    fields=FIELDS,
                )
            except Exception as exc:  # jira.exceptions.JIRAError et al.
                raise SystemExit(_project_error(project, exc, bool(token))) from exc
            if not results:
                break
            for issue in results:
                yield _issue_record(
                    client,
                    issue,
                    project=project,
                    include_comments=include_comments,
                )
            fetched = len(results)
            start += fetched
            if remaining is not None:
                remaining -= fetched
            print(
                json.dumps({
                    "project": project,
                    "fetched": start,
                    "last_batch": fetched,
                }, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            if fetched < batch_size:
                break


def _project_error(project: str, exc: Exception, has_token: bool) -> str:
    """A plain message for a search that JIRA refused.

    JIRA answers an unreadable project with "The value 'X' does not exist
    for the field 'project'", the same text it gives a project that is not
    there. Without a token the likely cause is visibility, so say that.
    """
    # Keep only the status and JIRA's own message. str(exc) of a JIRAError
    # carries response headers (for example Set-Cookie), so never fall back
    # to it.
    status = getattr(exc, "status_code", None)
    message = str(getattr(exc, "text", None) or "").strip()
    said = f" JIRA said: {message}" if message else ""
    if status == 401:
        if has_token:
            return (
                f"JIRA rejected the request for project {project} (HTTP 401): "
                f"the token was rejected or lacks access.{said}"
            )
        return (
            f"JIRA project {project} needs a login (HTTP 401). Set "
            "CERN_JIRA_TOKEN to a personal access token that can read it, "
            f"or pass --project with a public project key.{said}"
        )
    if status in (400, 403, 404):
        if has_token:
            return (
                f"JIRA project {project} could not be read with the token "
                f"given (HTTP {status}): it may not exist, or the token may "
                f"not have access to it.{said}"
            )
        return (
            f"JIRA project {project} is not visible without a token (HTTP "
            f"{status}): it is either not public or does not exist. Set "
            "CERN_JIRA_TOKEN to a personal access token that can read it, "
            f"or pass --project with a public project key.{said}"
        )
    where = f"HTTP {status}" if status else type(exc).__name__
    return f"JIRA search for project {project} failed ({where}).{said}"


def _issue_record(
    client: Any,
    issue: Any,
    *,
    project: str,
    include_comments: bool,
) -> dict[str, Any]:
    fields = issue.fields
    comments = _comments(client, issue.key, fields, include_all=include_comments)
    return {
        "key": issue.key,
        "summary": fields.summary or "",
        "description": fields.description or "",
        "status": _string(fields.status),
        "priority": _string(fields.priority),
        "issue_type": _string(fields.issuetype),
        "assignee": _display_name(fields.assignee),
        "reporter": _display_name(fields.reporter),
        "created": fields.created or "",
        "updated": fields.updated or "",
        "project": project,
        "labels": list(fields.labels or []),
        "components": [
            component.name for component in (fields.components or [])
        ],
        "resolution": _string(fields.resolution),
        "fix_versions": [
            version.name for version in (fields.fixVersions or [])
        ],
        "comment_count": (
            fields.comment.total
            if hasattr(fields, "comment") and fields.comment else 0
        ),
        "recent_comments": comments,
        "issue_links": _issue_links(fields),
        "subtasks": [
            subtask.key for subtask in (getattr(fields, "subtasks", None) or [])
        ],
        "parent_key": _key(getattr(fields, "parent", None)),
        "environment": getattr(fields, "environment", "") or "",
        "duedate": getattr(fields, "duedate", "") or "",
    }


def _comments(
    client: Any,
    issue_key: str,
    fields: Any,
    *,
    include_all: bool,
) -> list[dict[str, str]]:
    if not hasattr(fields, "comment") or not fields.comment:
        return []
    comments = list(fields.comment.comments or [])
    if include_all:
        start = len(comments)
        total = fields.comment.total
        while start < total:
            page = client.comments(issue_key, startAt=start, maxResults=100)
            if not page:
                break
            comments.extend(page)
            start += len(page)
    return [
        {
            "author": _display_name(comment.author),
            "created": comment.created or "",
            "body": comment.body or "",
        }
        for comment in comments
        if comment.body
    ]


def _issue_links(fields: Any) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for link in (fields.issuelinks or []):
        outward = getattr(link, "outwardIssue", None)
        inward = getattr(link, "inwardIssue", None)
        target = outward or inward
        key = _key(target)
        if not key:
            continue
        links.append({
            "key": key,
            "type": _string(getattr(link, "type", None)),
            "direction": "outward" if outward else "inward",
        })
    return links


def _display_name(value: Any) -> str:
    return getattr(value, "displayName", "") if value else ""


def _key(value: Any) -> str:
    return getattr(value, "key", "") if value else ""


def _string(value: Any) -> str:
    return str(value) if value else ""


def _write_cache(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    server: str,
    projects: tuple[str, ...],
    include_comments: bool,
    started_at: float,
) -> DownloadStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_at = time.time()
    data = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    digest = hashlib.sha256(data).hexdigest()
    meta = {
        "fetched_at": completed_at,
        "record_count": len(records),
        "content_hash": digest,
        "server": server,
        "projects": list(projects),
        "fields": FIELDS,
        "include_comments": include_comments,
        "ttl_seconds": 3600,
    }
    _atomic_write(output_dir / "records.json", data)
    _atomic_write(
        output_dir / "meta.json",
        json.dumps(meta, indent=2, sort_keys=True).encode() + b"\n",
    )
    return DownloadStats(
        record_count=len(records),
        content_hash=digest,
        started_at=started_at,
        completed_at=completed_at,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
