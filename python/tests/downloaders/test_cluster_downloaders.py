"""The ServiceNow and Redmine downloaders over a fake HTTP session. No request leaves the process."""
import json

import pytest

from archi.downloaders import redmine, servicenow
from archi.sources.servicenow import ServiceNowSource
from archi.sources.tickets import TicketSource
from packs.cluster_fixtures import FakeResponse, FakeSession, load

INSTANCE = "https://example.service-now.com"
OAUTH_ENV = {"SERVICENOW_CLIENT_ID": "cid-fixture", "SERVICENOW_CLIENT_SECRET": "secret-fixture"}
BASIC_ENV = {"SERVICENOW_USERNAME": "user-fixture", "SERVICENOW_PASSWORD": "pw-fixture"}


def _snow_session(rows_by_table, token="tok-fixture", page_limit=None):
    session = FakeSession()

    def token_route(url, kwargs):
        if url == f"{INSTANCE}/oauth_token.do":
            return FakeResponse(200, {"access_token": token})

    def table_route(url, kwargs):
        prefix = f"{INSTANCE}/api/now/table/"
        if not url.startswith(prefix):
            return None
        rows = [dict(r) for r in rows_by_table.get(url[len(prefix):], [])]
        for row in rows:
            row.pop("table", None)
        params = kwargs["params"]
        start, size = params["sysparm_offset"], params["sysparm_limit"]
        return FakeResponse(200, {"result": rows[start:start + size]})

    session.posts.append(token_route)
    session.gets.append(table_route)
    return session


def _rows():
    by_table = {}
    for row in load("servicenow_records.json"):
        by_table.setdefault(row["table"], []).append(row)
    return by_table


def test_servicenow_oauth_download_writes_the_cache_the_reader_reads(tmp_path):
    session = _snow_session(_rows())
    info = servicenow.download(session, INSTANCE + "/", "oauth", OAUTH_ENV, tmp_path / "servicenow", page_size=2)
    assert info["record_count"] == 7
    # the token request carried the grant and the pair; every table call carried the bearer token
    post = [r for r in session.requests if r[0] == "POST"][0]
    assert post[2]["data"]["grant_type"] == "client_credentials" and post[2]["data"]["client_id"] == "cid-fixture"
    gets = [r for r in session.requests if r[0] == "GET"]
    assert all(r[2]["headers"]["Authorization"] == "Bearer tok-fixture" for r in gets)
    assert {r[1].rsplit("/", 1)[1] for r in gets} == {"incident", "sc_request", "kb_knowledge"}
    assert all(r[2]["params"]["sysparm_display_value"] == "true" for r in gets)
    assert all(r[2]["params"]["sysparm_query"] == "ORDERBYsys_id" for r in gets), "a stable order: paging cannot skip or repeat"
    assert len([r for r in gets if r[1].endswith("/incident")]) == 2, "three incidents at page size 2 take two pages"

    run = ServiceNowSource(records_path=str(tmp_path / "servicenow" / "records.json"),
                           meta_path=str(tmp_path / "servicenow" / "meta.json")).run("r", mode="scope_complete")
    assert run.completed_scope is True and run.health.record_count == 7


def test_servicenow_basic_auth_sends_the_pair_and_no_token_request(tmp_path):
    session = _snow_session(_rows())
    servicenow.download(session, INSTANCE, "basic", BASIC_ENV, tmp_path, tables=["incident"])
    assert not [r for r in session.requests if r[0] == "POST"]
    assert all(r[2]["auth"] == ("user-fixture", "pw-fixture") for r in session.requests)


def test_a_missing_credential_is_named_by_variable_and_nothing_is_requested(tmp_path):
    session = _snow_session(_rows())
    with pytest.raises(servicenow.DownloadError) as caught:
        servicenow.download(session, INSTANCE, "oauth", {"SERVICENOW_CLIENT_ID": "x"}, tmp_path)
    assert "SERVICENOW_CLIENT_SECRET" in str(caught.value) and "SERVICENOW_CLIENT_ID" not in str(caught.value)
    assert session.requests == [] and not list(tmp_path.iterdir())


def test_no_credential_value_is_written_or_printed(tmp_path, capsys):
    session = _snow_session(_rows())
    code = servicenow.main(["--instance-url", INSTANCE, "--auth", "oauth", "--output-dir", str(tmp_path)],
                           session=session, env=OAUTH_ENV)
    shown = capsys.readouterr()
    assert code == 0
    written = "".join(p.read_text() for p in tmp_path.iterdir())
    for value in (*OAUTH_ENV.values(), "tok-fixture"):
        assert value not in written and value not in shown.out and value not in shown.err


def test_an_unreadable_table_stops_the_run_and_names_the_table(tmp_path, capsys):
    session = FakeSession(gets=[lambda url, kw: FakeResponse(403, {})],
                          posts=[lambda url, kw: FakeResponse(200, {"access_token": "t"})])
    code = servicenow.main(["--instance-url", INSTANCE, "--output-dir", str(tmp_path)], session=session, env=OAUTH_ENV)
    assert code == 1 and "incident" in capsys.readouterr().err and not list(tmp_path.iterdir())


def test_a_refused_token_request_stops_the_run(tmp_path):
    session = FakeSession(posts=[lambda url, kw: FakeResponse(401, {})])
    with pytest.raises(servicenow.DownloadError, match="HTTP 401"):
        servicenow.download(session, INSTANCE, "oauth", OAUTH_ENV, tmp_path)


def test_zero_records_is_not_written_as_an_empty_cache(tmp_path):
    session = _snow_session({})
    with pytest.raises(servicenow.DownloadError, match="no records"):
        servicenow.download(session, INSTANCE, "oauth", OAUTH_ENV, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("url", ["http://example.service-now.com", "ftp://x", "example.service-now.com"])
def test_a_url_that_is_not_https_is_refused_before_any_request(tmp_path, url):
    session = _snow_session(_rows())
    with pytest.raises(servicenow.DownloadError, match="https"):
        servicenow.download(session, url, "basic", BASIC_ENV, tmp_path)
    assert session.requests == []


def test_an_unknown_table_is_refused(tmp_path):
    with pytest.raises(servicenow.DownloadError, match="unknown table"):
        servicenow.download(_snow_session({}), INSTANCE, "oauth", OAUTH_ENV, tmp_path, tables=["problem"])


# --- Redmine ----------------------------------------------------------------------

REDMINE = "https://redmine.example.org"


def _redmine_session():
    session = FakeSession()

    def issues(url, kwargs):
        if url != f"{REDMINE}/issues.json":
            return None
        return FakeResponse(200, load("redmine_issues_page1.json" if kwargs["params"]["offset"] == 0 else "redmine_issues_page2.json"))

    session.gets.append(issues)
    return session


def test_redmine_download_pages_until_the_total_and_the_reader_reads_it(tmp_path):
    session = _redmine_session()
    info = redmine.download(session, REDMINE + "/", {}, tmp_path / "tickets", page_size=2)
    assert info["record_count"] == 3 and info["authenticated"] is False
    assert [r[2]["params"]["offset"] for r in session.requests] == [0, 2]
    assert all("X-Redmine-API-Key" not in r[2]["headers"] for r in session.requests), "anonymous without a key"
    assert session.requests[0][2]["params"]["status_id"] == "*" and session.requests[0][2]["params"]["include"] == "journals"

    run = TicketSource(records_path=str(tmp_path / "tickets" / "records.json"),
                       meta_path=str(tmp_path / "tickets" / "meta.json")).run("r", mode="scope_complete")
    assert run.completed_scope is True and run.health.record_count == 3


def test_redmine_key_is_sent_as_a_header_and_never_written(tmp_path):
    session = _redmine_session()
    redmine.download(session, REDMINE, {"REDMINE_API_KEY": "key-fixture"}, tmp_path, page_size=2)
    assert all(r[2]["headers"]["X-Redmine-API-Key"] == "key-fixture" for r in session.requests)
    assert "key-fixture" not in "".join(p.read_text() for p in tmp_path.iterdir())
    assert json.loads((tmp_path / "meta.json").read_text())["authenticated"] is True


def test_a_private_redmine_project_without_a_key_says_so(tmp_path):
    session = FakeSession(gets=[lambda url, kw: FakeResponse(403, {})])
    with pytest.raises(redmine.DownloadError, match="REDMINE_API_KEY"):
        redmine.download(session, REDMINE, {}, tmp_path, projects=["ops"])
    assert not list(tmp_path.iterdir())


def test_redmine_cap_and_project_filter(tmp_path):
    session = _redmine_session()
    info = redmine.download(session, REDMINE, {}, tmp_path, projects=["ops"], page_size=2, max_issues=2)
    assert info["record_count"] == 2
    assert session.requests[0][2]["params"]["project_id"] == "ops"
