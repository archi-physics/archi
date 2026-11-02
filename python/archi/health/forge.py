"""GitLab and GitHub issue trackers, over their REST APIs.

These reach an external system, so ADR 0002 section 7.2 allows them only inside a live tool
(`archi.tools.instance_health`), which the runtime runs at L2 after an approval. Importing this module
does nothing; building a tracker reads the token from the environment variable the instance config
names (`token_env`). The token is a header value only: it is never logged, put in an error message, a
URL or a returned result. URLs must be https, and redirects are not followed (a redirect would carry
the token to another host); a 3xx answer is an error.

What an issue holds. The body carries the problem's text, which includes `archi doctor` messages and
okg errors (paths, hostnames, source names). Point the tracker at a private project only.

How a problem finds its issue. Every issue the automation makes carries the label `archi-health` and,
last in its body, a marker comment `<!-- archi-health key=<key> family=<family> [resolved-by=archi]
[noted=1] -->`. A run lists the issues with the label (all states, every page) once and matches by the
marker's key. An issue whose label a person removed is still found: a key the label listing lacks is looked
up by searching the body for the marker, and the label is put back when the issue is next updated. An
issue without the marker is not ours and is ignored. A closed issue whose problem returns is reopened only if
this automation closed it (`resolved-by=archi`); one a person closed gets one comment ("still found") and
stays closed.

The HTTP session is a seam: any object with `request(method, url, headers=, json=, params=, timeout=,
allow_redirects=)` returning a response with `status_code`, `json()` and `text` (a `requests.Session`
does). The unit tests pass a fake and never touch a network.
"""
from __future__ import annotations

import urllib.parse
from typing import Any, Mapping, Optional

from archi.health.problems import Problem
from archi.health.tracker import LABEL, MARKER, OPEN, RESOLVED, Issue, TrackerError, render_body

PAGE = 100
TIMEOUT = 30
MAX_PAGES = 50


def marker(key: str, family: str, *, resolved: bool = False, noted: bool = False) -> str:
    flags = (" resolved-by=archi" if resolved else "") + (" noted=1" if noted else "")
    return f"<!-- archi-health key={key} family={family}{flags} -->"


def with_marker(problem: Problem) -> str:
    return f"{render_body(problem).rstrip()}\n\n{marker(problem.key, problem.family)}\n"


def strip_marker(body: str) -> str:
    return MARKER.sub("", body or "").rstrip() + "\n"


def require_https(url: str, what: str) -> str:
    if not url.lower().startswith("https://"):
        raise TrackerError(f"{what} must be an https:// URL (the token is sent with every request); got {url!r}")
    return url.rstrip("/")


class _Forge:
    """Shared by the two forges: the listing cache and the request helper."""

    kind = ""

    def __init__(self, base: str, token: str, session: Any = None, *, timeout: int = TIMEOUT):
        if not token:
            raise TrackerError(f"the {self.kind} token is empty")
        if session is None:
            import requests

            session = requests.Session()
        self.base = require_https(base, f"the {self.kind} URL")
        self._token = token
        self.session = session
        self.timeout = timeout
        self._issues: Optional[dict[str, tuple[Issue, Any]]] = None  # key -> (issue, forge id)
        self._unlabelled: set[str] = set()  # keys found by their marker only: the label is put back on update

    # subclasses
    def headers(self) -> dict[str, str]: ...
    def list_url(self) -> str: ...
    def parse(self, item: Mapping[str, Any]) -> Optional[tuple[Issue, Any]]: ...
    def create_request(self, problem: Problem) -> tuple[str, dict]: ...
    def update_request(self, ident: Any, problem: Problem, reopen: bool, relabel: bool) -> tuple[str, str, dict]: ...
    def body_request(self, ident: Any, body: str, close: bool) -> tuple[str, str, dict]: ...
    def comment(self, ident: Any, text: str) -> None: ...
    def search(self, key: str) -> list: ...
    def after_update(self, ident: Any, relabel: bool) -> None: ...

    def request(self, method: str, url: str, *, json: Any = None, params: Any = None) -> Any:
        try:
            response = self.session.request(method, url, headers=self.headers(), json=json, params=params,
                                            timeout=self.timeout, allow_redirects=False)
        except Exception as exc:  # a network error: say which call, never include the headers
            raise TrackerError(f"{self.kind} {method} {self._redact(url)} failed: {type(exc).__name__}") from exc
        status = response.status_code
        if not 200 <= status < 300:
            snippet = self._redact((getattr(response, "text", "") or "")[:200]).replace("\n", " ")
            hint = " (redirects are not followed)" if 300 <= status < 400 else ""
            raise TrackerError(f"{self.kind} {method} {self._redact(url)} returned {status}{hint}: {snippet}")
        try:
            return response.json()
        except ValueError as exc:
            raise TrackerError(f"{self.kind} {method} {self._redact(url)} returned {status} but not JSON") from exc

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    # --- the listing ---------------------------------------------------------

    def _load(self) -> dict[str, tuple[Issue, Any]]:
        if self._issues is not None:
            return self._issues
        found: dict[str, tuple[Issue, Any]] = {}
        for page in range(1, MAX_PAGES + 1):
            items = self.request("GET", self.list_url(),
                                 params={"labels": LABEL, "state": "all", "per_page": PAGE, "page": page})
            if not isinstance(items, list):
                raise TrackerError(f"{self.kind} issue list was not a list")
            for item in items:
                self._add(found, item)
            if not items:  # a server may return fewer than asked without it being the last page
                break
        else:
            raise TrackerError(f"{self.kind}: more than {MAX_PAGES * PAGE} labelled issues; refusing to guess")
        self._issues = found
        return found

    def _add(self, found: dict, item: Mapping[str, Any]) -> None:
        parsed = self.parse(item)
        if parsed is None:
            return
        issue, ident = parsed
        if issue.key in found and found[issue.key][1] != ident:
            raise TrackerError(f"{self.kind}: two issues carry the key {issue.key!r} "
                               f"({found[issue.key][0].ref} and {issue.ref}); close one by hand")
        found[issue.key] = (issue, ident)

    def open_issues(self) -> list[Issue]:
        return [i for i, _ in self._load().values() if i.state == OPEN]

    def find(self, key: str) -> Optional[Issue]:
        issues = self._load()
        if key not in issues:
            # a person may have removed the label: look for the marker in the bodies before making a second issue
            found: dict[str, tuple[Issue, Any]] = {}
            for item in self.search(key):
                self._add(found, item)
            if key in found:
                issues[key] = found[key]
                self._unlabelled.add(key)
        entry = issues.get(key)
        return entry[0] if entry else None

    def create(self, problem: Problem) -> Issue:
        self._load()
        url, body = self.create_request(problem)
        parsed = self.parse(self.request("POST", url, json=body))
        if parsed is None:
            raise TrackerError(f"{self.kind} did not return the issue it created for {problem.key}")
        self._issues[problem.key] = parsed
        return parsed[0]

    def update(self, issue: Issue, problem: Problem) -> Issue:
        _, ident = self._load()[issue.key]
        relabel = issue.key in self._unlabelled
        method, url, body = self.update_request(ident, problem, issue.state != OPEN, relabel)
        parsed = self.parse(self.request(method, url, json=body))
        if parsed is None:
            raise TrackerError(f"{self.kind} did not return the issue it updated for {problem.key}")
        self.after_update(ident, relabel)
        self._unlabelled.discard(issue.key)
        self._issues[problem.key] = parsed
        return parsed[0]

    def _rewrite(self, issue: Issue, *, close: bool, resolved: bool, noted: bool) -> Issue:
        _, ident = self._load()[issue.key]
        body = f"{issue.body.rstrip()}\n\n{marker(issue.key, issue.family, resolved=resolved, noted=noted)}\n"
        method, url, payload = self.body_request(ident, body, close)
        parsed = self.parse(self.request(method, url, json=payload))
        if parsed is None:
            raise TrackerError(f"{self.kind} did not return the issue it changed for {issue.key}")
        self._issues[issue.key] = parsed
        return parsed[0]

    def resolve(self, issue: Issue, note: str) -> Issue:
        _, ident = self._load()[issue.key]
        self.comment(ident, note)
        return self._rewrite(issue, close=True, resolved=True, noted=False)

    def note_still_present(self, issue: Issue, text: str) -> Issue:
        _, ident = self._load()[issue.key]
        self.comment(ident, text)
        return self._rewrite(issue, close=False, resolved=issue.resolved_by_us, noted=True)

    def finish(self) -> None:
        return None


def _flags(raw: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in (raw or "").split())


class GitLabTracker(_Forge):
    """GitLab: `<url>/api/v4/projects/<id or path>/issues`, token in the `PRIVATE-TOKEN` header."""

    kind = "gitlab"

    def __init__(self, url: str, project: str, token: str, session: Any = None, **kw: Any):
        super().__init__(f"{require_https(url, 'the gitlab url')}/api/v4", token, session, **kw)
        self.project = urllib.parse.quote(str(project), safe="")

    def headers(self) -> dict[str, str]:
        return {"PRIVATE-TOKEN": self._token, "Content-Type": "application/json"}

    def list_url(self) -> str:
        return f"{self.base}/projects/{self.project}/issues"

    def parse(self, item: Mapping[str, Any]) -> Optional[tuple[Issue, Any]]:
        match = MARKER.search(item.get("description") or "")
        if match is None:
            return None
        flags = _flags(match.group(3))
        state = OPEN if item.get("state") == "opened" else RESOLVED
        return (Issue(match.group(1), match.group(2), item.get("title", ""), strip_marker(item.get("description")),
                      state, ref=item.get("web_url") or str(item.get("iid")),
                      resolved_by_us=flags.get("resolved-by") == "archi", noted=flags.get("noted") == "1"), item["iid"])

    def create_request(self, problem: Problem) -> tuple[str, dict]:
        return self.list_url(), {"title": problem.title, "description": with_marker(problem),
                                 "labels": f"{LABEL},{LABEL}:{problem.family}"}

    def update_request(self, ident: Any, problem: Problem, reopen: bool, relabel: bool) -> tuple[str, str, dict]:
        body: dict[str, Any] = {"title": problem.title, "description": with_marker(problem)}
        if reopen:
            body["state_event"] = "reopen"
        if relabel:
            body["add_labels"] = f"{LABEL},{LABEL}:{problem.family}"
        return "PUT", f"{self.list_url()}/{ident}", body

    def body_request(self, ident: Any, body: str, close: bool) -> tuple[str, str, dict]:
        payload: dict[str, Any] = {"description": body}
        if close:
            payload["state_event"] = "close"
        return "PUT", f"{self.list_url()}/{ident}", payload

    def comment(self, ident: Any, text: str) -> None:
        self.request("POST", f"{self.list_url()}/{ident}/notes", json={"body": text})

    def search(self, key: str) -> list:
        items = self.request("GET", self.list_url(), params={
            "search": f"archi-health key={key}", "in": "description", "state": "all", "per_page": PAGE})
        return items if isinstance(items, list) else []

    def after_update(self, ident: Any, relabel: bool) -> None:
        return None


class GitHubTracker(_Forge):
    """GitHub: `<api>/repos/<owner>/<name>/issues`, token as a Bearer header."""

    kind = "github"

    def __init__(self, repo: str, token: str, session: Any = None, *, api_url: str = "https://api.github.com", **kw: Any):
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise TrackerError(f"github repo {repo!r} must be owner/name")
        super().__init__(api_url, token, session, **kw)
        self.repo = repo

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def list_url(self) -> str:
        return f"{self.base}/repos/{self.repo}/issues"

    def parse(self, item: Mapping[str, Any]) -> Optional[tuple[Issue, Any]]:
        if "pull_request" in item:  # the issues endpoint also lists pull requests
            return None
        match = MARKER.search(item.get("body") or "")
        if match is None:
            return None
        flags = _flags(match.group(3))
        state = OPEN if item.get("state") == "open" else RESOLVED
        return (Issue(match.group(1), match.group(2), item.get("title", ""), strip_marker(item.get("body")), state,
                      ref=item.get("html_url") or str(item.get("number")),
                      resolved_by_us=flags.get("resolved-by") == "archi", noted=flags.get("noted") == "1"), item["number"])

    def create_request(self, problem: Problem) -> tuple[str, dict]:
        return self.list_url(), {"title": problem.title, "body": with_marker(problem),
                                 "labels": [LABEL, f"{LABEL}:{problem.family}"]}

    def update_request(self, ident: Any, problem: Problem, reopen: bool, relabel: bool) -> tuple[str, str, dict]:
        body: dict[str, Any] = {"title": problem.title, "body": with_marker(problem)}
        if reopen:
            body["state"] = "open"
        self._relabel_family = problem.family
        return "PATCH", f"{self.list_url()}/{ident}", body

    def body_request(self, ident: Any, body: str, close: bool) -> tuple[str, str, dict]:
        payload: dict[str, Any] = {"body": body}
        if close:
            payload.update(state="closed", state_reason="completed")
        return "PATCH", f"{self.list_url()}/{ident}", payload

    def comment(self, ident: Any, text: str) -> None:
        self.request("POST", f"{self.list_url()}/{ident}/comments", json={"body": text})

    def search(self, key: str) -> list:
        found = self.request("GET", f"{self.base}/search/issues", params={
            "q": f'repo:{self.repo} is:issue in:body "archi-health key={key} "', "per_page": PAGE})
        items = found.get("items") if isinstance(found, dict) else None
        return items if isinstance(items, list) else []

    def after_update(self, ident: Any, relabel: bool) -> None:
        if relabel:  # POST adds labels and leaves a person's other labels alone
            self.request("POST", f"{self.list_url()}/{ident}/labels",
                         json={"labels": [LABEL, f"{LABEL}:{self._relabel_family}"]})
