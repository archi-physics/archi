"""The tickets downloader's JIRA reader (and Redmine beside it) over a fake HTTP session.

task.r3-bundle-cluster-manager.tickets. Recorded JIRA REST v2 search pages on
example.org hosts; no request leaves the process and no live JIRA is involved.
"""
import json

import pytest
import yaml

from archi.downloaders import tickets
from archi.sources.tickets import TicketSource
from packs.cluster_fixtures import JIRA, FakeResponse, FakeSession, jira_session, load
from packs.pack_support import PACKS_DIR

TOKEN = {"JIRA_API_TOKEN": "token-fixture"}
REDMINE = "https://redmine.example.org"


def _records(path):
    return json.loads((path / "records.json").read_text())


def test_jira_pages_by_start_at_until_the_total_even_when_jira_shortens_the_page(tmp_path):
    # Ask for 50 a page; the server caps it at 2, so a short page must not end the run.
    session = jira_session()
    info = tickets.download(session, TOKEN, tmp_path, jira_url=JIRA + "/", jira_projects=["OPS"], page_size=50)
    assert info["record_count"] == 3
    assert [r[1] for r in session.requests] == [f"{JIRA}/rest/api/2/search"] * 2
    params = [r[2]["params"] for r in session.requests]
    assert [p["startAt"] for p in params] == [0, 2] and params[0]["maxResults"] == 50
    assert params[0]["jql"] == "project in (OPS) ORDER BY key ASC"
    assert "comment" in params[0]["fields"].split(",")
    assert all(r[2]["headers"]["Authorization"] == "Bearer token-fixture" for r in session.requests)
    assert info["sources"] == [{"system": "jira", "base_url": JIRA, "jql": params[0]["jql"], "authenticated": True,
                                "restricted_comments": False}]


def test_an_empty_page_ends_the_paging(tmp_path):
    pages = iter([load("jira_search_page1.json"), {"startAt": 2, "total": 9, "issues": []}])
    session = FakeSession(gets=[lambda url, kw: FakeResponse(200, next(pages))])
    assert tickets.download(session, TOKEN, tmp_path, jira_url=JIRA)["record_count"] == 2
    assert len(session.requests) == 2


def test_the_default_order_is_by_key_so_paging_is_stable():
    assert tickets.jira_jql(["OPS", "ADM"]) == "project in (OPS, ADM) ORDER BY key ASC"
    assert tickets.jira_jql() == "ORDER BY key ASC"


def test_an_issue_repeated_across_pages_is_written_once_and_counted_once(tmp_path):
    # An issue that moves during the download can appear on two pages.
    first, second = load("jira_search_page1.json"), load("jira_search_page2.json")
    second["issues"].append(first["issues"][1])
    second["total"] = 4
    session = FakeSession(gets=[lambda url, kw: FakeResponse(200, first if kw["params"]["startAt"] == 0 else second)])
    info = tickets.download(session, TOKEN, tmp_path, jira_url=JIRA)
    keys = [r["key"] for r in _records(tmp_path)]
    assert sorted(keys) == ["OPS-12", "OPS-3", "OPS-9"]
    assert info["record_count"] == 3 == json.loads((tmp_path / "meta.json").read_text())["record_count"]


def _pages(*bodies):
    """A session answering each search with the next body, by startAt order."""
    answers = iter(bodies)
    return FakeSession(gets=[lambda url, kw: FakeResponse(200, next(answers))])


def _issues(*keys):
    return [{"key": k, "fields": {"summary": k}} for k in keys]


@pytest.mark.parametrize("total", ["missing", None, "3", 2.5, True])
def test_without_a_usable_total_paging_runs_to_an_empty_page(tmp_path, total):
    pages = [{"issues": _issues("OPS-1", "OPS-2")}, {"issues": _issues("OPS-3")}, {"issues": []}]
    if total != "missing":
        for page in pages:
            page["total"] = total
    session = _pages(*pages)
    info = tickets.download(session, TOKEN, tmp_path, jira_url=JIRA, page_size=2)
    assert info["record_count"] == 3
    assert [r[2]["params"]["startAt"] for r in session.requests] == [0, 2, 3]


def test_without_a_total_the_cap_still_stops_the_paging(tmp_path):
    # The second answer holds more than the one issue asked for; the cap still holds.
    session = _pages({"issues": _issues("OPS-1", "OPS-2")}, {"issues": _issues("OPS-3", "OPS-4")})
    info = tickets.download(session, TOKEN, tmp_path, jira_url=JIRA, page_size=2, max_issues=3)
    assert info["record_count"] == 3
    assert [r[2]["params"]["maxResults"] for r in session.requests] == [2, 1]


@pytest.mark.parametrize("body", [{"issues": {"OPS-1": {}}, "total": 1}, {"issues": "OPS-1"}, ["OPS-1"]])
def test_a_page_that_is_not_a_list_of_issues_stops_the_run(tmp_path, body):
    with pytest.raises(tickets.DownloadError, match="not a list|not a search result"):
        tickets.download(_pages(body), TOKEN, tmp_path, jira_url=JIRA)
    assert not list(tmp_path.iterdir())


def test_a_200_that_is_not_json_says_so_without_a_traceback(tmp_path, capsys):
    class HtmlResponse(FakeResponse):
        def json(self):
            raise json.JSONDecodeError("Expecting value", "<html>", 0)

    session = FakeSession(gets=[lambda url, kw: HtmlResponse(200, None, "<html>Sign in</html>")])
    code = tickets.main(["--jira-url", JIRA, "--output-dir", str(tmp_path / "t")], session=session, env=TOKEN)
    err = capsys.readouterr().err
    assert code == 1 and "not JSON" in err and "Traceback" not in err
    assert not (tmp_path / "t").exists()


def test_restricted_comments_are_skipped_unless_asked_for(tmp_path):
    tickets.download(jira_session(), TOKEN, tmp_path / "a", jira_url=JIRA)
    by_key = {r["key"]: r for r in _records(tmp_path / "a")}
    assert by_key["OPS-3"]["notes"] == [] and "Admins only" not in (tmp_path / "a" / "records.json").read_text()

    info = tickets.download(jira_session(), TOKEN, tmp_path / "b", jira_url=JIRA, include_restricted_comments=True)
    by_key = {r["key"]: r for r in _records(tmp_path / "b")}
    assert [n["body"] for n in by_key["OPS-3"]["notes"]] == ["Admins only: the theory group is over budget, see [~user]."]
    assert info["sources"][0]["restricted_comments"] is True


def test_the_restricted_comment_option_is_off_on_the_command_line_unless_given(tmp_path):
    tickets.main(["--jira-url", JIRA, "--output-dir", str(tmp_path / "a")], session=jira_session(), env=TOKEN)
    tickets.main(["--jira-url", JIRA, "--output-dir", str(tmp_path / "b"), "--include-restricted-comments"],
                 session=jira_session(), env=TOKEN)
    assert "Admins only" not in (tmp_path / "a" / "records.json").read_text()
    assert "Admins only" in (tmp_path / "b" / "records.json").read_text()


def test_login_mentions_become_a_placeholder(tmp_path):
    assert tickets.jira_record(
        {"key": "OPS-1", "fields": {"description": "ping [~jdoe] and [~accountid:5b10a2844c20165700ede21g]",
                                     "comment": {"comments": [{"body": "thanks [~ada.l]"}]}}}, JIRA,
    )["description"] == "ping [~user] and [~user]"
    tickets.download(jira_session(), TOKEN, tmp_path, jira_url=JIRA)
    by_key = {r["key"]: r for r in _records(tmp_path)}
    assert by_key["OPS-3"]["description"] == "Please raise the default home quota to 50 GB, as [~user] asked."


def test_jira_issues_map_onto_the_flat_ticket_shape(tmp_path):
    tickets.download(jira_session(), TOKEN, tmp_path, jira_url=JIRA, jira_projects=["OPS"])
    by_key = {r["key"]: r for r in _records(tmp_path)}
    assert set(by_key) == {"OPS-12", "OPS-9", "OPS-3"}
    one = by_key["OPS-12"]
    assert one["system"] == "jira" and one["url"] == f"{JIRA}/browse/OPS-12"
    assert (one["subject"], one["project"], one["tracker"], one["status"], one["priority"]) == (
        "GPU node gpu07 drops out of the pool", "Cluster Operations", "Incident", "In Progress", "High")
    assert (one["author"], one["assignee"]) == ("Katherine Johnson", "Dorothy Vaughan")
    assert (one["created"], one["updated"], one["closed"]) == (
        "2026-10-14T07:02:11.000+0000", "2026-10-14T11:20:45.000+0000", "")
    assert one["notes"][0] == {"author": "Dorothy Vaughan", "created": "2026-10-14T11:20:45.000+0000",
                               "body": "Reseated the GPU; watching it overnight."}
    closed = by_key["OPS-9"]
    assert closed["closed"] == "2026-10-06T16:45:00.000+0000"
    assert closed["assignee"] == "" and closed["description"] == "" and closed["notes"] == []


def test_people_are_written_as_display_names_never_logins_or_addresses(tmp_path):
    tickets.download(jira_session(), TOKEN, tmp_path, jira_url=JIRA)
    written = (tmp_path / "records.json").read_text()
    for login in ("kjohnson", "dvaughan", "mhamilton", "dorothy.vaughan@example.org"):
        assert login not in written


def test_a_missing_token_is_refused_by_name_before_any_request(tmp_path):
    session = jira_session()
    with pytest.raises(tickets.DownloadError, match="JIRA_API_TOKEN is not set"):
        tickets.download(session, {}, tmp_path, jira_url=JIRA, jira_projects=["OPS"])
    assert session.requests == [] and not list(tmp_path.iterdir())


def test_a_missing_token_stops_a_two_tracker_run_before_redmine_is_read(tmp_path):
    session = FakeSession()
    with pytest.raises(tickets.DownloadError, match="JIRA_API_TOKEN"):
        tickets.download(session, {}, tmp_path, redmine_url=REDMINE, jira_url=JIRA)
    assert session.requests == []


def test_the_token_is_never_written_or_printed(tmp_path, capsys):
    out = tmp_path / "tickets"
    code = tickets.main(["--jira-url", JIRA, "--jira-project", "OPS", "--output-dir", str(out)],
                        session=jira_session(), env=TOKEN)
    assert code == 0
    printed = capsys.readouterr()
    assert "token-fixture" not in printed.out + printed.err
    assert "token-fixture" not in "".join(p.read_text() for p in out.iterdir())
    assert json.loads(printed.out)["record_count"] == 3


@pytest.mark.parametrize("status, words", [(401, "JIRA_API_TOKEN was rejected"), (400, "JQL was refused"), (500, "HTTP 500")])
def test_a_refused_search_stops_the_run_and_writes_nothing(tmp_path, status, words):
    session = FakeSession(gets=[lambda url, kw: FakeResponse(status, {"errorMessages": ["no"]})])
    with pytest.raises(tickets.DownloadError, match=words):
        tickets.download(session, TOKEN, tmp_path, jira_url=JIRA)
    assert not list(tmp_path.iterdir())


def test_a_jira_url_that_is_not_https_is_refused_before_any_request(tmp_path):
    session = jira_session("http://jira.example.org")
    with pytest.raises(tickets.DownloadError, match="https"):
        tickets.download(session, TOKEN, tmp_path, jira_url="http://jira.example.org")
    assert session.requests == []


def test_jql_is_passed_as_given_and_cannot_be_mixed_with_projects(tmp_path):
    session = jira_session()
    tickets.download(session, TOKEN, tmp_path, jira_url=JIRA, jql="project = OPS AND component = batch")
    assert session.requests[0][2]["params"]["jql"] == "project = OPS AND component = batch"
    with pytest.raises(tickets.DownloadError, match="not both"):
        tickets.download(jira_session(), TOKEN, tmp_path, jira_url=JIRA, jira_projects=["OPS"], jql="x")


@pytest.mark.parametrize("key", ["ops", "OPS) OR (1=1", "OPS,ADMIN", ""])
def test_a_project_that_is_not_a_project_key_is_refused(tmp_path, key):
    session = jira_session()
    with pytest.raises(tickets.DownloadError, match="project key"):
        tickets.download(session, TOKEN, tmp_path, jira_url=JIRA, jira_projects=[key])
    assert session.requests == []


def test_the_cap_limits_the_jira_read(tmp_path):
    session = jira_session()
    info = tickets.download(session, TOKEN, tmp_path, jira_url=JIRA, page_size=2, max_issues=2)
    assert info["record_count"] == 2 and len(session.requests) == 1


def test_no_issues_is_not_written_as_an_empty_cache(tmp_path):
    session = FakeSession(gets=[lambda url, kw: FakeResponse(200, {"startAt": 0, "total": 0, "issues": []})])
    with pytest.raises(tickets.DownloadError, match="no issues"):
        tickets.download(session, TOKEN, tmp_path, jira_url=JIRA)
    assert not list(tmp_path.iterdir())


def _both_session():
    session = jira_session()

    def issues(url, kwargs):
        if url != f"{REDMINE}/issues.json":
            return None
        return FakeResponse(200, load("redmine_issues_page1.json" if kwargs["params"]["offset"] == 0 else "redmine_issues_page2.json"))

    session.gets.append(issues)
    return session


def test_both_trackers_fill_one_cache_and_the_reader_keeps_them_apart(tmp_path):
    out = tmp_path / "tickets"
    info = tickets.download(_both_session(), TOKEN, out, redmine_url=REDMINE, jira_url=JIRA, page_size=2)
    assert info["record_count"] == 6 and [s["system"] for s in info["sources"]] == ["redmine", "jira"]

    run = TicketSource(records_path=str(out / "records.json"), meta_path=str(out / "meta.json")).run(
        "r", mode="scope_complete")
    assert run.completed_scope is True and run.health.record_count == 6
    nodes = {f.node_id: f.attrs for f in run.facts if getattr(f, "subtype", None) == "tracker_ticket"}
    assert {"tracker_ticket:redmine:4812", "tracker_ticket:jira:OPS-12"} <= set(nodes)
    # Each link comes from its own tracker's root, not the one base_url the pack was given.
    assert nodes["tracker_ticket:redmine:4812"]["url"] == f"{REDMINE}/issues/4812"
    assert nodes["tracker_ticket:jira:OPS-12"]["url"] == f"{JIRA}/browse/OPS-12"


def test_redmine_alone_reads_no_jira_and_needs_no_token(tmp_path):
    session = _both_session()
    info = tickets.download(session, {}, tmp_path, redmine_url=REDMINE, page_size=2)
    assert info["record_count"] == 3 and all("/issues.json" in r[1] for r in session.requests)
    assert {r["system"] for r in _records(tmp_path)} == {"redmine"}


def test_no_tracker_is_refused(tmp_path):
    with pytest.raises(tickets.DownloadError, match="--redmine-url, --jira-url"):
        tickets.download(FakeSession(), TOKEN, tmp_path)
    assert tickets.main(["--output-dir", str(tmp_path)], session=FakeSession(), env={}) == 1


def test_redmine_stays_the_default_tracker():
    questions = {q["id"]: q for q in yaml.safe_load((PACKS_DIR / "tickets" / "questions.yaml").read_text())["init_questions"]}
    assert questions["system"]["default"] == "redmine"
    assert TicketSource().system == "redmine"
    assert TicketSource().credential_refs == ("REDMINE_API_KEY",)
