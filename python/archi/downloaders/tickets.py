"""Download Redmine and/or JIRA issues into the cache ``archi.sources.tickets`` reads.

The tickets pack's downloader. Run it from an operator environment::

    python -m archi.downloaders.tickets --redmine-url https://redmine.example.org --redmine-project ops
    JIRA_API_TOKEN=... python -m archi.downloaders.tickets --jira-url https://jira.example.org --jira-project OPS
    JIRA_API_TOKEN=... python -m archi.downloaders.tickets --jira-url https://jira.example.org \\
        --jql 'project = OPS AND component = batch ORDER BY key ASC'

Give ``--redmine-url``, ``--jira-url`` or both; with both, one cache holds the
tickets of the two trackers. Each record carries ``system`` (``redmine`` or
``jira``), so ticket ids of the two never collide, and its own ``url``. A key
that arrives twice is written once, so ``meta.json``'s count is the count of
distinct tickets.

Redmine is read exactly as ``archi.downloaders.redmine`` reads it:
``REDMINE_API_KEY`` is optional, and without it only public projects are visible.

JIRA is read over the REST v2 search API (``GET <url>/rest/api/2/search``,
paged with ``startAt``) of any JIRA Server or Data Center. It needs a personal
access token in ``JIRA_API_TOKEN``, sent as ``Authorization: Bearer``; the run
stops before any request when it is unset, and ``--jira-url`` must be
``https://`` because the token travels in its requests. ``--jira-project``
(repeatable) or ``--jql`` chooses the issues. The default order is
``ORDER BY key ASC``: paging over an order that changes during the download
(``updated DESC``) repeats or skips issues. A ``--jql`` of your own should keep
a stable order for the same reason.

What is written about people and private text:

- People (reporter, assignee, comment author) are written as display names
  only; a JIRA user's login and email address fields are not written.
- ``[~login]`` mentions in descriptions and comments become ``[~user]``.
- A comment with a ``visibility`` restriction (a group or role) is skipped
  unless ``--include-restricted-comments`` is given.
- Email addresses inside free text (descriptions, comments) are written as
  JIRA sends them and removed when the reader reads the cache
  (``archi.sources._cluster_cache.clean``), as on the Redmine path, so none
  reaches a node.

No credential is printed or written. ``records.json`` and ``meta.json`` are
written atomically under ``--output-dir`` (default ``tickets`` under
``ARCHI_DATA_ROOT``, where the pack reads it).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from archi.downloaders import redmine
from archi.downloaders.servicenow import DownloadError, Session
from archi.sources._cluster_cache import write_cache

DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
JIRA_TOKEN_ENV = "JIRA_API_TOKEN"
TIMEOUT = 60
JIRA_FIELDS = (
    "summary,description,project,issuetype,status,priority,reporter,assignee,"
    "created,updated,resolutiondate,comment"
)
PROJECT_KEY = re.compile(r"[A-Z][A-Z0-9_]*")
# [~jdoe] and Cloud's [~accountid:...]: a user's login in wiki markup.
MENTION = re.compile(r"\[~[^\]\s]*\]")
MENTION_PLACEHOLDER = "[~user]"


def jira_jql(projects: Sequence[str] = (), jql: str = "") -> str:
    """The search: ``--jql`` as given, else the projects in key order (stable while paging)."""
    if jql and projects:
        raise DownloadError("give --jira-project or --jql, not both")
    if jql:
        return jql
    bad = [p for p in projects if not PROJECT_KEY.fullmatch(p)]
    if bad:
        raise DownloadError(f"not a JIRA project key: {bad} (expected e.g. OPS)")
    where = f"project in ({', '.join(projects)}) " if projects else ""
    return f"{where}ORDER BY key ASC"


def _display_name(user: Any) -> str:
    # Only the display name: a JIRA user object also carries a login
    # ("name") and often an email address, neither of which is written.
    return str(user.get("displayName") or "") if isinstance(user, dict) else ""


def _name(value: Any) -> str:
    return str(value.get("name") or "") if isinstance(value, dict) else ""


def _text(value: Any) -> str:
    # REST v2 sends wiki-markup strings; anything else (an ADF document) is dropped.
    return value if isinstance(value, str) else ""


def _prose(value: Any) -> str:
    """Free text with ``[~login]`` mentions replaced by a placeholder."""
    return MENTION.sub(MENTION_PLACEHOLDER, _text(value))


def jira_record(issue: Mapping[str, Any], base_url: str, *, include_restricted_comments: bool = False) -> dict[str, Any]:
    """One REST v2 issue in the flat shape ``archi.sources.tickets`` reads."""
    fields = issue.get("fields") or {}
    key = str(issue.get("key") or "")
    comments = (fields.get("comment") or {}).get("comments") or []
    return {
        "system": "jira",
        "key": key,
        "subject": _text(fields.get("summary")),
        "description": _prose(fields.get("description")),
        "project": _name(fields.get("project")),
        "tracker": _name(fields.get("issuetype")),
        "status": _name(fields.get("status")),
        "priority": _name(fields.get("priority")),
        "author": _display_name(fields.get("reporter")),
        "assignee": _display_name(fields.get("assignee")),
        "created": _text(fields.get("created")),
        "updated": _text(fields.get("updated")),
        "closed": _text(fields.get("resolutiondate")),
        "notes": [
            {"author": _display_name(c.get("author")), "created": _text(c.get("created")), "body": _prose(c.get("body"))}
            for c in comments
            # A comment restricted to a group or role is not for everyone who reads the graph.
            if isinstance(c, dict) and (include_restricted_comments or not c.get("visibility"))
        ],
        "url": f"{base_url}/browse/{key}" if key else "",
    }


def _total(body: Mapping[str, Any]) -> Optional[int]:
    value = body.get("total")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def fetch_jira_issues(
    session: Session,
    base_url: str,
    *,
    token: str,
    jql: str,
    page_size: int = 100,
    max_issues: int = 0,
    include_restricted_comments: bool = False,
) -> list[dict[str, Any]]:
    """Every issue the search matches, paged by ``startAt``.

    Paging stops at JIRA's ``total``, or, when the answer carries no usable
    ``total``, at the first empty page; never on a short page, since JIRA may
    cap the page size below what was asked.
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    records: list[dict[str, Any]] = []
    start = 0
    while True:
        size = page_size if not max_issues else min(page_size, max_issues - len(records))
        if size <= 0:
            break
        params = {"jql": jql, "startAt": start, "maxResults": size, "fields": JIRA_FIELDS}
        response = session.get(f"{base_url}/rest/api/2/search", params=params, headers=headers, timeout=TIMEOUT)
        if response.status_code != 200:
            hint = {
                400: "the JQL was refused, or a project in it does not exist or is not visible to the token",
                401: f"the token in {JIRA_TOKEN_ENV} was rejected",
                403: f"the token in {JIRA_TOKEN_ENV} may not search",
            }.get(response.status_code, "")
            raise DownloadError(
                f"JIRA search could not be read (HTTP {response.status_code})" + (f": {hint}" if hint else "")
            )
        try:
            body = response.json()
        except ValueError:
            raise DownloadError(
                "JIRA search answered HTTP 200 with something that is not JSON (a login page in front of "
                "the server?); check --jira-url"
            ) from None
        if not isinstance(body, dict):
            raise DownloadError("JIRA search answered with JSON that is not a search result")
        page = body.get("issues")
        if page is None:
            page = []
        if not isinstance(page, list):
            raise DownloadError("JIRA search answered with an 'issues' field that is not a list")
        # Never more than asked for, so --max-issues holds whatever the server sends.
        page = page[:size]
        records.extend(
            jira_record(issue, base_url, include_restricted_comments=include_restricted_comments)
            for issue in page if isinstance(issue, dict)
        )
        start += len(page)
        total = _total(body)
        if not page or (total is not None and start >= total):
            break
    return records


def _distinct(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first record of each (system, key); a page boundary can repeat one."""
    seen: set[tuple[str, str]] = set()
    out = []
    for record in records:
        ident = (str(record.get("system") or ""), str(record.get("key") or record.get("id") or ""))
        if ident in seen:
            continue
        seen.add(ident)
        out.append(record)
    return out


def _https(url: str, flag: str) -> str:
    if not url.lower().startswith("https://"):
        raise DownloadError(f"{flag} must be https://, got {url!r}")
    return url.rstrip("/")


def download(
    session: Session,
    env: Mapping[str, str],
    output_dir: Path,
    *,
    redmine_url: str = "",
    redmine_projects: Sequence[str] = (),
    jira_url: str = "",
    jira_projects: Sequence[str] = (),
    jql: str = "",
    page_size: int = 100,
    max_issues: int = 0,
    include_restricted_comments: bool = False,
) -> dict[str, Any]:
    """Fetch each tracker given and write one cache. ``max_issues`` caps each tracker."""
    if not redmine_url and not jira_url:
        raise DownloadError("give --redmine-url, --jira-url or both")
    sources: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    query = ""
    if jira_url:
        # Checked before any request, Redmine's included, so a bad JIRA setup costs nothing.
        jira_url = _https(jira_url, "--jira-url")
        query = jira_jql(jira_projects, jql)
        if not env.get(JIRA_TOKEN_ENV):
            raise DownloadError(f"{JIRA_TOKEN_ENV} is not set; the JIRA reader needs a personal access token")
    if redmine_url:
        redmine_url = redmine_url.rstrip("/")
        issues = redmine.fetch_issues(
            session, redmine_url, api_key=env.get(redmine.KEY_ENV, ""), projects=redmine_projects,
            page_size=page_size, max_issues=max_issues,
        )
        if not issues:
            raise DownloadError("Redmine returned no issues; not writing an empty cache")
        records.extend({**i, "url": f"{redmine_url}/issues/{i['id']}"} if i.get("id") else i for i in issues)
        sources.append({"system": "redmine", "base_url": redmine_url, "projects": list(redmine_projects),
                        "authenticated": bool(env.get(redmine.KEY_ENV))})
    if jira_url:
        issues = fetch_jira_issues(
            session, jira_url, token=env[JIRA_TOKEN_ENV], jql=query, page_size=page_size, max_issues=max_issues,
            include_restricted_comments=include_restricted_comments,
        )
        if not issues:
            raise DownloadError("JIRA returned no issues; not writing an empty cache")
        records.extend(issues)
        sources.append({"system": "jira", "base_url": jira_url, "jql": query, "authenticated": True,
                        "restricted_comments": include_restricted_comments})
    records = _distinct(records)
    return write_cache(output_dir, records, len(records), sources=sources)


def default_output_dir() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV)
    return (Path(raw).expanduser() if raw else Path.cwd()) / "tickets"


def main(argv: Optional[Sequence[str]] = None, *, session: Optional[Session] = None, env: Optional[Mapping[str, str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--redmine-url", default="")
    parser.add_argument("--redmine-project", dest="redmine_projects", action="append", help="repeatable")
    parser.add_argument("--jira-url", default="", help="the JIRA root, e.g. https://jira.example.org")
    parser.add_argument("--jira-project", dest="jira_projects", action="append", help="project key; repeatable")
    parser.add_argument("--jql", default="", help="a JQL search, instead of --jira-project; keep a stable order")
    parser.add_argument("--include-restricted-comments", action="store_true",
                        help="also write JIRA comments restricted to a group or role (default: skipped)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-issues", type=int, default=0, help="cap per tracker, for probes; 0 means no cap")
    args = parser.parse_args(argv)
    environment = os.environ if env is None else env
    if args.redmine_url and not environment.get(redmine.KEY_ENV):
        print(f"no {redmine.KEY_ENV} set; reading Redmine anonymously, so only public projects are visible",
              file=sys.stderr)
    if session is None:
        import requests

        session = requests.Session()
    try:
        info = download(
            session, environment, Path(args.output_dir) if args.output_dir else default_output_dir(),
            redmine_url=args.redmine_url, redmine_projects=tuple(args.redmine_projects or ()),
            jira_url=args.jira_url, jira_projects=tuple(args.jira_projects or ()), jql=args.jql,
            page_size=args.page_size, max_issues=args.max_issues,
            include_restricted_comments=args.include_restricted_comments,
        )
    except DownloadError as exc:
        print(f"tickets download stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"record_count": info["record_count"], "content_hash": info["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
