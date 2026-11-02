"""req.r3-ops-self-monitoring.health: the GitLab and GitHub trackers, against mocked HTTP only.

`FakeForge` is an in-memory forge behind the `session.request(...)` seam. Nothing here opens a socket."""
import re
from types import SimpleNamespace

import pytest

from archi.health import forge
from archi.health.config import TrackerSpec, build_tracker
from archi.health.problems import Problem
from archi.health.tracker import LABEL, OPEN, RESOLVED, TrackerError, reconcile
from archi.tools.instance_health import sync_forge_issues

TOKEN = "glpat-SECRET-TOKEN-VALUE"


def problem(subject="jira.jira", detail="stale since 2026-09-20"):
    return Problem(key=f"source-freshness/{subject}", family="doctor", title=f"Source {subject} is stale", detail=detail)


class Response(SimpleNamespace):
    def json(self):
        return self.body


class FakeForge:
    """Just enough of the GitLab v4 and GitHub v3 issue APIs. Records every call."""

    def __init__(self, flavour, page=100, seed=None, fail=None):
        self.flavour, self.page, self.calls, self.issues, self.next = flavour, page, [], list(seed or []), len(seed or []) + 1
        self.fail = fail or {}
        self.redirect = False
        self.notes = []

    def request(self, method, url, headers=None, json=None, params=None, timeout=None, allow_redirects=True):
        assert allow_redirects is False, "a tracker must never follow redirects"
        self.calls.append((method, url, dict(headers or {}), json, dict(params or {}), timeout))
        if self.redirect:
            return Response(status_code=302, body={}, text="moved")
        for needle, status in self.fail.items():
            if needle in f"{method} {url}":
                return Response(status_code=status, body={}, text=f"denied; token {TOKEN} was refused")
        gl = self.flavour == "gitlab"
        if url.endswith("/search/issues"):
            needle = re.search(r'"(archi-health key=[^"]+)"', params["q"]).group(1)
            return Response(status_code=200, body={"items": [i for i in self.issues if needle in (i.get("body") or "")]}, text="")
        lab = re.search(r"/issues/(\d+)/labels$", url)
        if lab and method == "POST":
            found = next(i for i in self.issues if i["id"] == int(lab.group(1)))
            found["labels"] += [{"name": l} for l in json["labels"] if l not in [x["name"] for x in found["labels"]]]
            return Response(status_code=200, body=found["labels"], text="")
        coll = re.search(r"/issues$", url)
        item = re.search(r"/issues/(\d+)$", url)
        sub = re.search(r"/issues/(\d+)/(notes|comments)$", url)
        if method == "GET" and coll:
            if params.get("search"):
                return Response(status_code=200, body=[i for i in self.issues if params["search"] in i["description"]], text="")
            labelled = [i for i in self.issues if LABEL in (i["labels"] if gl else [l["name"] for l in i["labels"]])]
            if params.get("state") != "all":
                labelled = [i for i in labelled if i["state"] in ("opened", "open")]
            size = min(params["per_page"], self.page)
            start = (params["page"] - 1) * size
            return Response(status_code=200, body=labelled[start:start + size], text="")
        if method == "POST" and coll:
            n = self.next
            self.next += 1
            made = self._make(n, json)
            self.issues.append(made)
            return Response(status_code=201, body=made, text="")
        if sub:
            self.notes.append((method, url, json["body"]))
            return Response(status_code=201, body={}, text="")
        if item and method in ("PUT", "PATCH"):
            found = next(i for i in self.issues if i["id"] == int(item.group(1)))
            for field in ("title", "description", "body"):
                if field in json:
                    found[field] = json[field]
            if "add_labels" in json:
                found["labels"] = sorted(set(found["labels"]) | set(json["add_labels"].split(",")))
            if json.get("state_event") == "close" or json.get("state") == "closed":
                found["state"] = "closed"
            if json.get("state_event") == "reopen" or json.get("state") == "open":
                found["state"] = "opened" if gl else "open"
            return Response(status_code=200, body=found, text="")
        raise AssertionError(f"unexpected call {method} {url}")

    def _make(self, n, data):
        if self.flavour == "gitlab":
            return {"id": n, "iid": n, "title": data["title"], "description": data["description"], "state": "opened",
                    "labels": data["labels"].split(","), "web_url": f"https://gl.example/p/-/issues/{n}"}
        return {"id": n, "number": n, "title": data["title"], "body": data["body"], "state": "open",
                "labels": [{"name": l} for l in data["labels"]], "html_url": f"https://github.com/o/r/issues/{n}"}

    def writes(self):
        return [c for c in self.calls if c[0] != "GET"]


def gitlab(fake):
    return forge.GitLabTracker("https://gl.example", "grp/proj", TOKEN, fake)


def github(fake):
    return forge.GitHubTracker("o/r", TOKEN, fake)


@pytest.fixture(params=["gitlab", "github"])
def pair(request):
    fake = FakeForge(request.param)
    return fake, (gitlab(fake) if request.param == "gitlab" else github(fake))


# --- dedup, update, close ---------------------------------------------------------------


def test_two_runs_over_the_same_problem_make_one_issue(pair):
    fake, _ = pair
    make = (lambda: gitlab(fake)) if fake.flavour == "gitlab" else (lambda: github(fake))  # a new tracker per run
    first = reconcile(make(), [problem()], ["doctor"])
    second = reconcile(make(), [problem()], ["doctor"])
    assert [c.action for c in first] == ["created"] and [c.action for c in second] == ["unchanged"]
    assert len(fake.issues) == 1 and len(fake.writes()) == 1
    created = fake.writes()[0]
    assert created[0] == "POST" and LABEL in str(created[3])
    assert "archi-health key=source-freshness/jira.jira family=doctor" in str(created[3])


def test_a_changed_problem_updates_the_same_issue(pair):
    fake, tracker = pair
    reconcile(tracker, [problem()], ["doctor"])
    (change,) = reconcile(tracker, [problem(detail="stale since 2026-09-20, and now no run at all")], ["doctor"])
    assert change.action == "updated" and len(fake.issues) == 1
    body = fake.issues[0].get("description") or fake.issues[0].get("body")
    assert "no run at all" in body and body.count("archi-health key=") == 1


def test_a_cleared_problem_is_commented_and_closed_and_a_return_reopens_it(pair):
    fake, tracker = pair
    reconcile(tracker, [problem()], ["doctor"])
    (change,) = reconcile(tracker, [], ["doctor"])
    assert change.action == "resolved" and fake.issues[0]["state"] == "closed"
    assert [c[0] for c in fake.writes()][-2:] == ["POST", "PUT" if fake.flavour == "gitlab" else "PATCH"]
    (again,) = reconcile(tracker, [problem()], ["doctor"])
    assert again.action == "reopened" and len(fake.issues) == 1 and fake.issues[0]["state"] in ("opened", "open")


def test_an_issue_of_a_family_that_was_not_read_is_left_open(pair):
    fake, tracker = pair
    reconcile(tracker, [problem()], ["doctor"])
    assert reconcile(tracker, [], ["ingest"]) == [] and fake.issues[0]["state"] in ("opened", "open")


# --- what the listing reads ---------------------------------------------------------------


def test_every_page_is_read_before_deciding_an_issue_does_not_exist():
    fake = FakeForge("gitlab", page=2)
    reconcile(gitlab(fake), [problem(s) for s in "abc"], ["doctor"])
    fake.calls.clear()
    changes = reconcile(gitlab(fake), [problem(s) for s in "abc"], ["doctor"])
    assert {c.action for c in changes} == {"unchanged"} and len(fake.issues) == 3
    pages = [c[4]["page"] for c in fake.calls if c[0] == "GET"]
    assert pages == [1, 2, 3]  # the forge returns 2, then 1, then nothing: only an empty page ends the listing


def test_a_labelled_issue_without_our_marker_is_not_ours():
    other = {"id": 1, "iid": 1, "title": "someone else's", "description": "no marker here", "state": "opened",
             "labels": [LABEL], "web_url": "u"}
    fake = FakeForge("gitlab", seed=[other])
    reconcile(gitlab(fake), [problem()], ["doctor"])
    assert len(fake.issues) == 2 and fake.issues[0] == other


def test_github_pull_requests_in_the_issue_list_are_ignored():
    pr = {"id": 1, "number": 1, "title": "pr", "body": "<!-- archi-health key=source-freshness/jira.jira family=doctor -->",
          "state": "open", "labels": [{"name": LABEL}], "pull_request": {"url": "x"}, "html_url": "u"}
    fake = FakeForge("github", seed=[pr])
    (change,) = reconcile(github(fake), [problem()], ["doctor"])
    assert change.action == "created" and len(fake.issues) == 2


def test_two_issues_with_one_key_stop_the_run():
    a = FakeForge("github")
    reconcile(github(a), [problem()], ["doctor"])
    dup = dict(a.issues[0], id=9, number=9)
    with pytest.raises(TrackerError, match="two issues carry the key"):
        reconcile(github(FakeForge("github", seed=[a.issues[0], dup])), [problem()], ["doctor"])


# --- requests and secrets -------------------------------------------------------------------


def test_gitlab_requests_use_the_private_token_header_and_an_encoded_project_path():
    fake = FakeForge("gitlab")
    reconcile(gitlab(fake), [problem()], ["doctor"])
    method, url, headers, body, params, timeout, *_ = fake.calls[0]
    assert (method, url) == ("GET", "https://gl.example/api/v4/projects/grp%2Fproj/issues")
    assert headers["PRIVATE-TOKEN"] == TOKEN and params["labels"] == LABEL and timeout == forge.TIMEOUT
    assert TOKEN not in url and TOKEN not in str(body)


def test_github_requests_use_a_bearer_token_and_the_repos_issues_path():
    fake = FakeForge("github")
    reconcile(github(fake), [problem()], ["doctor"])
    method, url, headers, *_ = fake.calls[0]
    assert url == "https://api.github.com/repos/o/r/issues" and headers["Authorization"] == f"Bearer {TOKEN}"


@pytest.mark.parametrize("flavour", ["gitlab", "github"])
def test_an_http_error_stops_the_run_and_never_prints_the_token(flavour):
    fake = FakeForge(flavour, fail={"POST": 403})
    tracker = gitlab(fake) if flavour == "gitlab" else github(fake)
    with pytest.raises(TrackerError) as err:
        reconcile(tracker, [problem()], ["doctor"])
    assert "403" in str(err.value) and TOKEN not in str(err.value) and "***" in str(err.value)


def test_a_network_failure_is_reported_without_the_headers():
    class Down:
        def request(self, *a, **k):
            raise ConnectionError(f"cannot reach, header was {TOKEN}")

    with pytest.raises(TrackerError) as err:
        reconcile(github(Down()), [problem()], ["doctor"])
    assert "ConnectionError" in str(err.value) and TOKEN not in str(err.value)


def test_an_empty_token_or_a_bad_repo_is_refused():
    with pytest.raises(TrackerError):
        forge.GitLabTracker("https://gl.example", "p", "", FakeForge("gitlab"))
    with pytest.raises(TrackerError, match="owner/name"):
        forge.GitHubTracker("justone", TOKEN, FakeForge("github"))


# --- configuration: only the NAME of the token variable ----------------------------------------


def test_the_config_takes_the_name_of_the_token_variable_and_refuses_a_token():
    spec = TrackerSpec.from_mapping({"kind": "github", "repo": "o/r", "token_env": "ARCHI_HEALTH_GITHUB_TOKEN"})
    assert spec.action_class == "sync_forge_issues" and TrackerSpec.from_mapping(None).action_class == "draft_issue"
    with pytest.raises(TrackerError, match="never the token"):
        TrackerSpec.from_mapping({"kind": "github", "repo": "o/r", "token_env": "ghp_abc123"})
    with pytest.raises(TrackerError, match="needs token_env"):
        TrackerSpec.from_mapping({"kind": "gitlab", "url": "https://gl", "project": "p"})
    with pytest.raises(TrackerError, match="gitlab needs url"):
        TrackerSpec.from_mapping({"kind": "gitlab", "token_env": "T"})
    with pytest.raises(TrackerError, match="local tracker takes no"):
        TrackerSpec.from_mapping({"kind": "local", "token_env": "T"})
    with pytest.raises(TrackerError, match="not one of"):
        TrackerSpec.from_mapping({"kind": "jira"})


def test_build_tracker_reads_the_token_from_the_named_variable_and_says_when_it_is_unset():
    spec = TrackerSpec.from_mapping({"kind": "gitlab", "url": "https://gl.example", "project": "p", "token_env": "MY_TOKEN"})
    assert isinstance(build_tracker(spec, env={"MY_TOKEN": TOKEN}, session=FakeForge("gitlab")), forge.GitLabTracker)
    with pytest.raises(TrackerError, match="MY_TOKEN is not set"):
        build_tracker(spec, env={}, session=FakeForge("gitlab"))


# --- the live tool ----------------------------------------------------------------------------------


def test_the_live_tool_syncs_a_payload_through_the_named_backend_and_is_idempotent():
    fake = FakeForge("github")
    payload = {"tracker": {"kind": "github", "repo": "o/r", "token_env": "MY_TOKEN"}, "observed_at": "2026-10-25T12:00:00Z",
               "problems": [problem().as_dict()], "scope": ["doctor"], "skipped": []}
    out = sync_forge_issues(payload=payload, context={}, session=fake, env={"MY_TOKEN": TOKEN})
    again = sync_forge_issues(payload=payload, context={}, session=fake, env={"MY_TOKEN": TOKEN})
    assert out["status"] == "succeeded" and "1 created" in out["result"] and "1 unchanged" in again["result"]
    assert len(fake.issues) == 1 and TOKEN not in out["result"]


def test_the_live_tool_refuses_the_local_tracker():
    with pytest.raises(TrackerError, match="draft_issue step"):
        sync_forge_issues(payload={"tracker": {"kind": "local"}, "observed_at": "2026-10-25T12:00:00Z", "problems": [],
                                   "scope": []}, context={})


# --- review fixes -----------------------------------------------------------------------------------------------------


def close_by_person(fake):
    issue = fake.issues[0]
    issue["state"] = "closed"
    return issue


def fresh(fake):
    return gitlab(fake) if fake.flavour == "gitlab" else github(fake)  # a run builds its tracker anew


def test_an_issue_a_person_closed_gets_one_comment_and_stays_closed(pair):
    fake, _ = pair
    reconcile(fresh(fake), [problem()], ["doctor"])
    close_by_person(fake)
    (first,) = reconcile(fresh(fake), [problem()], ["doctor"])
    writes = len(fake.writes())
    (second,) = reconcile(fresh(fake), [problem()], ["doctor"])
    assert first.action == "noted" and second.action == "unchanged"
    assert fake.issues[0]["state"] == "closed" and len(fake.notes) == 1 and "closed by a person" in fake.notes[0][2]
    assert len(fake.writes()) == writes  # nothing written the second time


def test_an_issue_the_automation_closed_is_reopened_when_the_problem_returns(pair):
    fake, _ = pair
    reconcile(fresh(fake), [problem()], ["doctor"])
    reconcile(fresh(fake), [], ["doctor"])
    body = fake.issues[0].get("description") or fake.issues[0].get("body")
    assert "resolved-by=archi" in body
    assert reconcile(fresh(fake), [problem()], ["doctor"])[0].action == "reopened"
    assert len(fake.issues) == 1


@pytest.mark.parametrize("flavour", ["gitlab", "github"])
def test_removing_the_label_does_not_make_a_duplicate_and_the_label_comes_back(flavour):
    fake = FakeForge(flavour)
    make = (lambda: gitlab(fake)) if flavour == "gitlab" else (lambda: github(fake))
    reconcile(make(), [problem()], ["doctor"])
    fake.issues[0]["labels"] = [] if flavour == "gitlab" else [{"name": "bug"}]  # a person removes the label
    (change,) = reconcile(make(), [problem(detail="changed text")], ["doctor"])
    assert change.action == "updated" and len(fake.issues) == 1
    names = fake.issues[0]["labels"] if flavour == "gitlab" else [l["name"] for l in fake.issues[0]["labels"]]
    assert LABEL in names and (flavour == "gitlab" or "bug" in names)


@pytest.mark.parametrize("make", [
    lambda s: forge.GitLabTracker("http://gl.example", "p", TOKEN, s),
    lambda s: forge.GitHubTracker("o/r", TOKEN, s, api_url="http://api.example"),
])
def test_a_plain_http_url_is_refused_before_any_request(make):
    fake = FakeForge("gitlab")
    with pytest.raises(TrackerError, match="https"):
        make(fake)
    assert fake.calls == []


def test_the_config_refuses_http_too():
    with pytest.raises(TrackerError, match="https"):
        TrackerSpec.from_mapping({"kind": "gitlab", "url": "http://gl.example", "project": "p", "token_env": "T"})
    with pytest.raises(TrackerError, match="https"):
        TrackerSpec.from_mapping({"kind": "github", "repo": "o/r", "token_env": "T", "api_url": "http://x"})


def test_a_redirect_is_an_error_and_is_never_followed():
    fake = FakeForge("github")
    fake.redirect = True
    with pytest.raises(TrackerError, match="302 \\(redirects are not followed\\)"):
        reconcile(github(fake), [problem()], ["doctor"])
    assert len(fake.calls) == 1  # asserts allow_redirects=False inside the fake
