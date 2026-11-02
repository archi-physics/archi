"""req.compops-snapshot-builder -- the moved JIRA downloader, with no network.

A fake ``jira`` module stands in for the CERN client, so the test drives the
real fetch, record shaping and cache writing, then reads the cache back with
the archi JIRA reader.
"""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace as NS

import pytest

from archi.downloaders import jira as download_jira
from archi.sources.jira import JiraIssueSource


def _issue(key, *, summary, assignee, comments=()):
    fields = NS(
        summary=summary,
        description=f"Details for {key}",
        status="Open",
        priority="Major",
        issuetype="Bug",
        assignee=NS(displayName=assignee),
        reporter=NS(displayName="Grace Hopper"),
        created="2026-08-01T10:00:00.000+0000",
        updated="2026-08-02T10:00:00.000+0000",
        labels=["ops"],
        components=[NS(name="transfers")],
        resolution=None,
        fixVersions=[],
        comment=NS(
            total=len(comments),
            comments=[
                NS(author=NS(displayName="Ada Lovelace"), created="2026-08-03", body=b)
                for b in comments
            ],
        ),
        issuelinks=[NS(outwardIssue=NS(key="CMSPROD-1"), type="Relates")],
        subtasks=[],
        parent=None,
        environment="prod",
        duedate=None,
    )
    return NS(key=key, fields=fields)


class FakeJIRA:
    calls: list[tuple] = []

    def __init__(self, server, token_auth=None):
        self.server = server
        self.token = token_auth

    def search_issues(self, jql, startAt, maxResults, fields):
        FakeJIRA.calls.append((jql, startAt, maxResults))
        issues = [
            _issue("CMSPROD-2", summary="Stuck transfers", assignee="Ada Lovelace",
                   comments=["Retrying now"]),
            _issue("CMSPROD-3", summary="Site in downtime", assignee="Enrico Fermi"),
        ]
        return issues[startAt:startAt + maxResults]

    def comments(self, key, startAt, maxResults):  # pragma: no cover - not paged here
        return []


@pytest.fixture
def fake_jira(monkeypatch):
    FakeJIRA.calls = []
    monkeypatch.setitem(sys.modules, "jira", types.SimpleNamespace(JIRA=FakeJIRA))
    monkeypatch.setenv("CERN_JIRA_TOKEN", "test-token-not-a-secret")


def test_download_writes_a_cache_the_reader_accepts(tmp_path, fake_jira, capsys):
    out = tmp_path / "data" / "jira"
    assert download_jira.main(
        ["--output-dir", str(out), "--project", "CMSPROD", "--page-size", "1"]
    ) == 0
    records = json.loads((out / "records.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    assert [r["key"] for r in records] == ["CMSPROD-2", "CMSPROD-3"]
    assert records[0]["recent_comments"][0]["body"] == "Retrying now"
    assert records[0]["issue_links"] == [
        {"key": "CMSPROD-1", "type": "Relates", "direction": "outward"}
    ]
    assert meta["record_count"] == 2 and meta["projects"] == ["CMSPROD"]
    # Paged one issue at a time until a short page.
    assert [c[1] for c in FakeJIRA.calls] == [0, 1, 2]
    assert json.loads(capsys.readouterr().out)["record_count"] == 2

    run = JiraIssueSource(base=str(tmp_path)).run("r", mode="scope_complete")
    assert list(run.facts)
    assert run.completed_scope is True and run.health.status == "ok"
    assert run.health.record_count == 2


def test_default_output_follows_archi_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    assert download_jira.default_output_dir() == tmp_path / "data" / "jira"
    monkeypatch.delenv("ARCHI_DATA_ROOT")
    monkeypatch.chdir(tmp_path)
    assert download_jira.default_output_dir() == tmp_path / "data" / "jira"


def test_probe_only_writes_nothing(tmp_path, fake_jira, capsys):
    out = tmp_path / "jira"
    assert download_jira.main(
        ["--output-dir", str(out), "--project", "CMSPROD", "--max-issues", "1", "--probe-only"]
    ) == 0
    assert not out.exists()
    report = json.loads(capsys.readouterr().out)
    assert report == {"ok": True, "record_count": 1, "first_key": "CMSPROD-2", "projects": ["CMSPROD"]}


def test_no_token_reads_anonymously(tmp_path, monkeypatch, capsys):
    for name in ("CERN_JIRA_TOKEN", "JIRA_CERN_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    seen = {}

    class Recording(FakeJIRA):
        def __init__(self, server, token_auth=None):
            seen["token_auth"] = token_auth
            seen["kwargs_has_token"] = token_auth is not None
            super().__init__(server, token_auth)

    monkeypatch.setitem(sys.modules, "jira", types.SimpleNamespace(JIRA=Recording))
    out = tmp_path / "jira"
    assert download_jira.main(
        ["--output-dir", str(out), "--project", "ADAPTYST"]
    ) == 0
    assert seen["kwargs_has_token"] is False
    assert len(json.loads((out / "records.json").read_text())) == 2
    assert "reading anonymously" in capsys.readouterr().err


class _JIRAError(Exception):
    def __init__(self, status_code, text):
        super().__init__(text)
        self.status_code = status_code
        self.text = text


def _refusing_jira(monkeypatch):
    class Refusing(FakeJIRA):
        def search_issues(self, jql, startAt, maxResults, fields):
            raise _JIRAError(
                400, "The value 'CMSCOMPPR' does not exist for the field 'project'."
            )

    monkeypatch.setitem(sys.modules, "jira", types.SimpleNamespace(JIRA=Refusing))


def test_project_not_visible_anonymously_says_so(tmp_path, monkeypatch):
    for name in ("CERN_JIRA_TOKEN", "JIRA_CERN_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    _refusing_jira(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        download_jira.main(["--output-dir", str(tmp_path), "--project", "CMSCOMPPR"])
    message = str(caught.value)
    assert "CMSCOMPPR is not visible without a token" in message
    assert "CERN_JIRA_TOKEN" in message
    assert not (tmp_path / "records.json").exists()


def test_project_refused_with_a_token_does_not_blame_the_missing_token(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CERN_JIRA_TOKEN", "test-token-not-a-secret")
    _refusing_jira(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        download_jira.main(["--output-dir", str(tmp_path), "--project", "CMSCOMPPR"])
    message = str(caught.value)
    assert "could not be read with the token given" in message
    assert "without a token" not in message


def test_missing_jira_package_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("CERN_JIRA_TOKEN", "test-token-not-a-secret")
    monkeypatch.setitem(sys.modules, "jira", None)  # import raises ImportError
    with pytest.raises(SystemExit, match="'jira' Python package is not installed.*pip install jira"):
        download_jira.main(["--output-dir", str(tmp_path)])


def _failing_jira(monkeypatch, status, text, str_value="HEADERS Set-Cookie: JSESSIONID=secret"):
    class Err(_JIRAError):
        def __str__(self):
            return str_value

    class Failing(FakeJIRA):
        def search_issues(self, jql, startAt, maxResults, fields):
            raise Err(status, text)

    monkeypatch.setitem(sys.modules, "jira", types.SimpleNamespace(JIRA=Failing))


def test_401_with_a_token_says_rejected_or_lacks_access(tmp_path, monkeypatch):
    monkeypatch.setenv("CERN_JIRA_TOKEN", "test-token-not-a-secret")
    _failing_jira(monkeypatch, 401, "")
    with pytest.raises(SystemExit) as caught:
        download_jira.main(["--output-dir", str(tmp_path), "--project", "X"])
    message = str(caught.value)
    assert "token was rejected or lacks access" in message
    assert "HTTP 401" in message


def test_error_message_never_echoes_the_raw_error_string(tmp_path, monkeypatch):
    monkeypatch.setenv("CERN_JIRA_TOKEN", "test-token-not-a-secret")
    for status in (401, 403, 500):
        _failing_jira(monkeypatch, status, "")  # empty body: no fallback to str(exc)
        with pytest.raises(SystemExit) as caught:
            download_jira.main(["--output-dir", str(tmp_path), "--project", "X"])
        message = str(caught.value)
        assert "Set-Cookie" not in message and "JSESSIONID" not in message
        assert f"{status}" in message
