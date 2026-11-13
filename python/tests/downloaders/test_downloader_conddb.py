"""req.catalog-downloaders.conddb -- the CondDB global tag downloader, no network.

The pages are synthetic (global tag names ``FAKE*``) in the shape the CMS
CondDB browser's search page has: an HTML page whose JavaScript builds four
tables with ``new API.DataTables.Table({"data": [...], "headers": [...]},
"<name>")``, of which ``"gts"`` holds the global tags. The fake server
answers ``GET <base>/search/Prod/<term>`` the way the browser's own query
does: ``name ILIKE '%term%' OR release ILIKE '%term%' LIMIT 3000``, no
order, ``%`` and ``_`` left as SQL wildcards, case-insensitive.
"""
from __future__ import annotations

import http.cookiejar
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import pytest
import requests

from support.responses import FakeResponse

from okg.deployment import NodeFact

from archi.downloaders import _catalog, conddb
from archi.sources.conddb import CondDBGlobalTagSource

BASE = "https://cms-conddb.cern.ch/cmsDbBrowser"
SEARCH = f"{BASE}/search/Prod/"
GTS_HEADERS = ["", "Global Tag", "Type", "Release", "Insertion Time", "Snapshot Time", "Description"]
NO_SNAPSHOT = "9999-12-31 23:59:59 (UTC)"
LOGIN = (
    "https://auth.cern.ch/auth/realms/cern/protocol/openid-connect/auth"
    "?client_id=cms-conddb&response_type=code"
)
SSO_HINT = "python -m archi.downloaders.sso_login cms_conddb"


def gt(name, release, *, type_="Global Tag", inserted="2026-10-28 11:48:38 (UTC)",
       snapshot=NO_SNAPSHOT, description="A fake global tag"):
    return ["", name, type_, release, inserted, snapshot, description]


def page(rows, *, headers=GTS_HEADERS, gts_tables=1) -> str:
    """A search page in the browser's shape, its gts table holding ``rows``."""

    def table(name, data, table_headers):
        payload = json.dumps({"data": data, "headers": table_headers})
        return (
            f"        {name}_tbl = new API.DataTables.Table({payload}, \"{name}\");\n"
            f"        {name}_tbl.addCheckboxColumn();\n"
        )

    scripts = "".join(table("gts", rows, headers) for _ in range(gts_tables))
    scripts += table("entries", [], ["Global Tag Name", "Type", "Record", "Label", "Tag"])
    scripts += table("tags", [], ["", "Tag", "Time Type", "Object Type", "Synchronization"])
    scripts += table("payloads", [], ["", "Payload", "Object Type", "Insertion Time"])
    return (
        "<!DOCTYPE html>\n<html><head><title>cmsDbBrowser</title></head><body>\n"
        "<script>\n    $(function () {\n" + scripts + "    });\n</script>\n</body></html>\n"
    )


def _like(term: str) -> re.Pattern:
    """``ILIKE '%term%'`` as a regex: ``_`` any one character, ``%`` any run."""
    body = "".join({"_": ".", "%": ".*"}.get(ch, re.escape(ch)) for ch in term)
    return re.compile(body, re.IGNORECASE | re.DOTALL)


class FakeCondDB:
    """The browser's search, over a list of gts rows; records every call."""

    def __init__(self, rows, *, cap=3000):
        self.rows = list(rows)
        self.cap = cap
        self.calls: list[tuple[str, dict]] = []
        self.sessions: list[requests.Session] = []
        self.overrides: dict[str, object] = {}

    def __call__(self, session, url, **kwargs):
        self.calls.append((url, kwargs))
        self.sessions.append(session)
        if not url.startswith(SEARCH):
            return FakeResponse(404, "not found")
        term = unquote(url[len(SEARCH):])
        if term in self.overrides:
            answer = self.overrides[term]
            if isinstance(answer, BaseException):
                raise answer
            return answer
        pattern = _like(term)
        hits = [row for row in self.rows if pattern.search(row[1]) or pattern.search(row[3])]
        return FakeResponse(200, page(hits[: self.cap]), {"Content-Type": "text/html"})

    def terms(self):
        return [unquote(url[len(SEARCH):]) for url, _ in self.calls]


def write_cookie_file(path: Path, *, domain="cms-conddb.cern.ch", expires_in=12 * 3600):
    path.parent.mkdir(parents=True, exist_ok=True)
    jar = http.cookiejar.MozillaCookieJar(str(path))
    jar.set_cookie(
        http.cookiejar.Cookie(
            version=0, name="fake_session", value="fake-session-value", port=None,
            port_specified=False, domain=domain, domain_specified=True,
            domain_initial_dot=False, path="/", path_specified=True, secure=True,
            expires=int(time.time()) + expires_in, discard=False, comment=None,
            comment_url=None, rest={},
        )
    )
    jar.save(ignore_discard=True, ignore_expires=True)
    return path


# A small catalog. Series 14_0 and 15_1 are ordinary; 3,001 tags in major 13
# (1,600 in 13_0, 1,401 in 13_1) make the CMSSW_13_ search (and the CMSSW_1_
# search, whose wildcards also match majors 10-19) return the 3,000 cap.
SMALL = [
    gt("140X_FAKEdataRun3_v1", "CMSSW_14_0_X", inserted="2024-03-01 09:00:00 (UTC)",
       snapshot="2024-03-01 08:59:00 (UTC)", description="first"),
    gt("140X_FAKEdataRun3_v2", "CMSSW_14_0_X", inserted="2024-04-01 10:30:15 (UTC)"),
    gt("151X_FAKEmcRun4_Candidate_2026_10_28", "CMSSW_15_1_X", type_="Candidate"),
    gt("151X_FAKEmcRun4_Queue", "CMSSW_15_1_X", type_="Queue"),
    gt("1_2_FAKEancient_v1", "CMSSW_1_2_0", inserted="2007-05-01 00:00:00 (UTC)"),
]
BIG_SERIES = [
    gt(f"13{minor}X_FAKEbulk_{i:04d}", f"CMSSW_13_{minor}_X")
    for minor, count in ((0, 1600), (1, 1401))
    for i in range(count)
]


@pytest.fixture(autouse=True)
def _no_grid_ca_dir_and_no_sleep(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path / "no-grid-ca")
    monkeypatch.setattr(_catalog.time, "sleep", lambda seconds: None)


@pytest.fixture
def root(tmp_path, monkeypatch):
    path = tmp_path / "root"
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(path))
    monkeypatch.delenv("CONDDB_COOKIE_FILE", raising=False)
    monkeypatch.delenv("ARCHI_COOKIE_DIR", raising=False)
    return path


@pytest.fixture
def cookie_file(tmp_path, root):
    return write_cookie_file(tmp_path / "cookies" / "cms_conddb.txt")


def serve(monkeypatch, rows, **kwargs) -> FakeCondDB:
    fake = FakeCondDB(rows, **kwargs)

    def get(session, url, **kw):
        return fake(session, url, **kw)

    monkeypatch.setattr(requests.Session, "get", get)
    return fake


def _main(output: Path, cookie_file: Path, *extra: str) -> int:
    return conddb.main(["--output-dir", str(output), "--cookie-file", str(cookie_file), *extra])


def _records(output: Path):
    return json.loads((output / "records.json").read_text(encoding="utf-8"))


# --- the round trip -----------------------------------------------------


def test_every_global_tag_reaches_the_reader_with_capped_majors_refined(
    tmp_path, root, cookie_file, monkeypatch, capsys
):
    upstream = serve(monkeypatch, SMALL + BIG_SERIES)
    output = tmp_path / "conddb-global-tags"
    assert _main(output, cookie_file) == 0

    run = CondDBGlobalTagSource(records_path=str(output / "records.json")).run(
        "run-1", mode="scope_complete"
    )
    nodes = {f.node_id: f.attrs for f in run.facts if isinstance(f, NodeFact)}
    assert run.health.status == "ok" and run.completed_scope is True
    assert run.health.record_count == len(SMALL) + len(BIG_SERIES)
    v2 = nodes["global_tag:140X_FAKEdataRun3_v2"]
    assert v2["release"] == "CMSSW_14_0_X"
    assert v2["created_at"] == "2024-04-01T10:30:15+00:00"

    # Majors 1..99 once each; the two capped ones refined over minors 0..99.
    terms = upstream.terms()
    assert terms[:99] == [f"CMSSW_{m}_" for m in range(1, 100)]
    assert sorted(terms[99:]) == sorted(
        [f"CMSSW_{major}_{minor}_" for major in (1, 13) for minor in range(100)]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "output": str(output),
        "records": len(SMALL) + len(BIG_SERIES),
        "types": {"Candidate": 1, "Global Tag": 3 + 3001, "Queue": 1},
        "searches": 99 + 200,
        "refined_majors": [1, 13],
    }


def test_records_are_sorted_by_name_with_only_the_keys_the_reader_reads(
    tmp_path, root, cookie_file, monkeypatch
):
    serve(monkeypatch, SMALL)
    output = tmp_path / "conddb-global-tags"
    assert _main(output, cookie_file) == 0
    assert _records(output) == [
        {
            "name": "140X_FAKEdataRun3_v1",
            "release": "CMSSW_14_0_X",
            "description": "first",
            "created_at": "2024-03-01T09:00:00+00:00",
        },
        {
            "name": "140X_FAKEdataRun3_v2",
            "release": "CMSSW_14_0_X",
            "description": "A fake global tag",
            "created_at": "2024-04-01T10:30:15+00:00",
        },
        {
            "name": "151X_FAKEmcRun4_Candidate_2026_10_28",
            "release": "CMSSW_15_1_X",
            "description": "A fake global tag",
            "created_at": "2026-10-28T11:48:38+00:00",
        },
        {
            "name": "151X_FAKEmcRun4_Queue",
            "release": "CMSSW_15_1_X",
            "description": "A fake global tag",
            "created_at": "2026-10-28T11:48:38+00:00",
        },
        {
            "name": "1_2_FAKEancient_v1",
            "release": "CMSSW_1_2_0",
            "description": "A fake global tag",
            "created_at": "2007-05-01T00:00:00+00:00",
        },
    ]


def test_only_search_gets_are_made_with_the_session_cookie_over_one_session(
    tmp_path, root, cookie_file, monkeypatch, capsys
):
    upstream = serve(monkeypatch, SMALL)
    assert _main(tmp_path / "conddb-global-tags", cookie_file, "--timeout", "17") == 0
    assert len(upstream.calls) == 99
    for url, kwargs in upstream.calls:
        assert url.startswith(SEARCH)
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"] == {"Accept": "text/html"}
        assert kwargs["timeout"] == 17
    assert all(session is upstream.sessions[0] for session in upstream.sessions)
    cookies = {(c.domain, c.name) for c in upstream.sessions[0].cookies}
    assert cookies == {("cms-conddb.cern.ch", "fake_session")}
    # The cookie value never reaches the output.
    captured = capsys.readouterr()
    assert "fake-session-value" not in captured.out + captured.err


def test_progress_goes_to_stderr_and_only_the_summary_to_stdout(
    tmp_path, root, cookie_file, monkeypatch, capsys
):
    serve(monkeypatch, SMALL)
    assert _main(tmp_path / "conddb-global-tags", cookie_file) == 0
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert [line.split(",")[0] for line in lines] == [
        "conddb: 20 searches",
        "conddb: 40 searches",
        "conddb: 60 searches",
        "conddb: 80 searches",
        "conddb: 99 searches",
    ]
    assert lines[-1].startswith("conddb: 99 searches, 5 global tags, ")
    assert json.loads(captured.out)["records"] == 5


def test_a_description_quoting_the_table_code_is_read_as_text(
    tmp_path, root, cookie_file, monkeypatch
):
    quoting = 'copied from new API.DataTables.Table({"data": [], "headers": []}, "gts"); by hand'
    serve(monkeypatch, SMALL + [gt("140X_FAKEquoting_v1", "CMSSW_14_0_X", description=quoting)])
    output = tmp_path / "conddb-global-tags"
    assert _main(output, cookie_file) == 0
    records = {record["name"]: record for record in _records(output)}
    assert records["140X_FAKEquoting_v1"]["description"] == quoting


def test_the_default_output_is_conddb_global_tags_under_archi_data_root(
    root, cookie_file, monkeypatch
):
    serve(monkeypatch, SMALL)
    assert conddb.main(["--cookie-file", str(cookie_file)]) == 0
    assert (root / "conddb-global-tags" / "records.json").is_file()


# --- fail closed: nothing is written ------------------------------------


def _old_cache(output: Path) -> dict[str, bytes]:
    output.mkdir(parents=True)
    (output / "records.json").write_text(json.dumps([{"name": "OLD_v1"}]) + "\n")
    return _snapshot(output)


def _snapshot(output: Path) -> dict[str, bytes]:
    """The cache files (the write lock file excluded)."""
    return {p.name: p.read_bytes() for p in output.iterdir() if not p.name.startswith(".")}


def _fails(tmp_path, cookie_file, capsys, *extra: str) -> str:
    """Run against an existing cache; assert exit 1 and the cache untouched."""
    output = tmp_path / "conddb-global-tags"
    old = _old_cache(output)
    assert _main(output, cookie_file, *extra) == 1
    assert _snapshot(output) == old
    assert [p.name for p in output.iterdir() if p.name.startswith(".staging.")] == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "conddb download failed, cache not written" in captured.err
    return captured.err


def test_a_refined_search_still_at_the_cap_fails(tmp_path, root, cookie_file, monkeypatch, capsys):
    # 3,000 tags in one series: CMSSW_13_ is capped, and so is CMSSW_13_0_.
    upstream = serve(
        monkeypatch, SMALL + [gt(f"130X_FAKEbulk_{i:04d}", "CMSSW_13_0_X") for i in range(3000)]
    )
    err = _fails(tmp_path, cookie_file, capsys)
    assert f"{SEARCH}CMSSW_13_0_ returned 3000 rows" in err
    assert upstream.terms()[-1] == "CMSSW_13_0_"


def test_a_release_not_named_cmssw_major_minor_fails(tmp_path, root, cookie_file, monkeypatch, capsys):
    # Found through its name (the search also matches names), with a release
    # the listing by release would never reach.
    serve(monkeypatch, SMALL + [gt("FAKE_CMSSW_14_0_patch", "CMSSW_14-0")])
    err = _fails(tmp_path, cookie_file, capsys)
    assert "FAKE_CMSSW_14_0_patch has release 'CMSSW_14-0'" in err


def test_a_release_outside_the_searched_majors_fails(tmp_path, root, cookie_file, monkeypatch, capsys):
    # CMSSW_100_ is matched by the CMSSW_10_ search's wildcards, but no
    # search covers major 100 as a whole.
    serve(monkeypatch, SMALL + [gt("1000X_FAKEfuture_v1", "CMSSW_100_0_X")])
    err = _fails(tmp_path, cookie_file, capsys)
    assert "1000X_FAKEfuture_v1 has release CMSSW_100_0_X, which no complete search covers" in err


def test_a_minor_above_99_of_a_refined_major_fails(tmp_path, root, cookie_file, monkeypatch, capsys):
    # Major 13 is refined; CMSSW_13_100_X comes back through the CMSSW_13_10_
    # search's wildcard, but no search covers minor 100 as a whole.
    upstream = serve(monkeypatch, SMALL + BIG_SERIES + [gt("13100X_FAKEfar_v1", "CMSSW_13_100_X")])
    err = _fails(tmp_path, cookie_file, capsys)
    assert "13100X_FAKEfar_v1 has release CMSSW_13_100_X, which no complete search covers" in err
    assert "CMSSW_13_10_" in upstream.terms()


@pytest.mark.parametrize(
    "column, value",
    [(3, "CMSSW_14_1_X"), (4, "2024-03-02 09:00:00 (UTC)"), (6, "edited between two searches")],
    ids=["release", "inserted", "description"],
)
def test_one_name_with_two_different_written_values_fails(
    tmp_path, root, cookie_file, monkeypatch, capsys, column, value
):
    upstream = serve(monkeypatch, SMALL)
    # CMSSW_1_ and CMSSW_14_ both return the 14_0 tags; let the second differ.
    changed = [list(row) for row in SMALL if row[3] == "CMSSW_14_0_X"]
    changed[0][column] = value
    upstream.overrides["CMSSW_14_"] = FakeResponse(200, page(changed))
    err = _fails(tmp_path, cookie_file, capsys)
    assert "140X_FAKEdataRun3_v1 came back as two different rows" in err


@pytest.mark.parametrize(
    "column, value",
    [(2, "Global Tag"), (5, "2026-11-13 08:00:00 (UTC)")],
    ids=["type", "snapshot"],
)
def test_a_type_or_snapshot_change_between_searches_is_not_a_conflict(
    tmp_path, root, cookie_file, monkeypatch, column, value
):
    # A candidate promoted, or a snapshot set, while the run goes on: neither
    # value is written, so the two rows agree on the record.
    upstream = serve(monkeypatch, SMALL)
    changed = [list(row) for row in SMALL if row[3] == "CMSSW_15_1_X"]
    changed[0][column] = value
    upstream.overrides["CMSSW_15_"] = FakeResponse(200, page(changed))
    output = tmp_path / "conddb-global-tags"
    assert _main(output, cookie_file) == 0
    names = [record["name"] for record in _records(output)]
    assert names.count("151X_FAKEmcRun4_Candidate_2026_10_28") == 1
    assert len(names) == len(SMALL)


def test_an_empty_catalog_fails(tmp_path, root, cookie_file, monkeypatch, capsys):
    serve(monkeypatch, [])
    err = _fails(tmp_path, cookie_file, capsys)
    assert "refusing to write an empty catalog" in err


@pytest.mark.parametrize(
    "answer, message",
    [
        (page(SMALL, headers=GTS_HEADERS[1:]), "the 'gts' table has headers"),
        (page(SMALL, headers=[*GTS_HEADERS[:-1], "Comment"]), "the 'gts' table has headers"),
        (page(SMALL, gts_tables=0), "holds 0 'gts' tables, not 1"),
        (page(SMALL, gts_tables=2), "holds 2 'gts' tables, not 1"),
        (page([row[:6] for row in SMALL]), "a gts row is not 7 strings"),
        (page([gt("140X_FAKE_v3", "CMSSW_14_0_X", type_="Archived")]),
         "global tag 140X_FAKE_v3 has type 'Archived', not one of Global Tag, Candidate, Queue"),
        (page([gt("140X_FAKE_v3", "CMSSW_14_0_X", inserted="2024-04-01T10:30:15Z")]),
         "has Insertion Time '2024-04-01T10:30:15Z'"),
        (page([gt("", "CMSSW_14_0_X")]), "a gts row has no global tag name"),
        ("<html><script>t = new API.DataTables.Table({\"data\": [}, \"gts\");</script></html>",
         "is not inline JSON"),
    ],
    ids=["header-missing", "header-renamed", "no-gts", "two-gts", "short-row",
         "unknown-type", "time-format", "no-name", "broken-json"],
)
def test_a_drifted_page_fails(tmp_path, root, cookie_file, monkeypatch, capsys, answer, message):
    upstream = serve(monkeypatch, SMALL)
    upstream.overrides["CMSSW_14_"] = FakeResponse(200, answer)
    err = _fails(tmp_path, cookie_file, capsys)
    assert f"{SEARCH}CMSSW_14_: " in err
    assert message in err


# --- the SSO session ----------------------------------------------------


def test_a_redirect_to_the_login_page_fails_with_the_login_command(
    tmp_path, root, cookie_file, monkeypatch, capsys
):
    upstream = serve(monkeypatch, SMALL)
    upstream.overrides["CMSSW_1_"] = FakeResponse(302, "", {"Location": LOGIN})
    err = _fails(tmp_path, cookie_file, capsys)
    assert "HTTP 302" in err and LOGIN in err
    assert SSO_HINT in err
    # Not followed, not retried, and nothing after it.
    assert upstream.terms() == ["CMSSW_1_"]


def test_a_login_page_answered_with_200_fails_with_the_login_command(
    tmp_path, root, cookie_file, monkeypatch, capsys
):
    upstream = serve(monkeypatch, SMALL)
    login_page = '<html><head><title>Sign in to CERN</title></head><form action="https://auth.cern.ch/x">'
    upstream.overrides["CMSSW_1_"] = FakeResponse(200, login_page)
    err = _fails(tmp_path, cookie_file, capsys)
    assert "answered the CERN login page" in err and SSO_HINT in err


@pytest.mark.parametrize(
    "make, reason",
    [
        (lambda path: None, "cookie file is missing"),
        (lambda path: path.write_text(""), "cookie file could not be parsed"),
        (lambda path: path.write_text("not a cookie file\n"), "cookie file could not be parsed"),
        (lambda path: write_cookie_file(path, expires_in=-60), "cookies are expired"),
        (lambda path: write_cookie_file(path, domain="other.example"),
         "no cookies for domain cms-conddb.cern.ch"),
    ],
    ids=["missing", "empty", "not-netscape", "expired", "other-domain"],
)
def test_an_unusable_cookie_file_stops_the_run_before_any_request(
    tmp_path, root, monkeypatch, capsys, make, reason
):
    upstream = serve(monkeypatch, SMALL)
    path = tmp_path / "cookies" / "cms_conddb.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    make(path)
    err = _fails(tmp_path, path, capsys)
    assert f"the CondDB cookie file {path} is unusable" in err
    assert reason in err
    assert SSO_HINT in err
    assert upstream.calls == []


def test_the_cookie_file_comes_from_conddb_cookie_file(tmp_path, root, monkeypatch):
    upstream = serve(monkeypatch, SMALL)
    path = write_cookie_file(tmp_path / "elsewhere" / "jar.txt")
    monkeypatch.setenv("CONDDB_COOKIE_FILE", str(path))
    assert conddb.main(["--output-dir", str(tmp_path / "out")]) == 0
    assert upstream.calls


def test_the_default_cookie_file_is_where_sso_login_writes_it(tmp_path, root, monkeypatch, capsys):
    upstream = serve(monkeypatch, SMALL)
    monkeypatch.setenv("ARCHI_COOKIE_DIR", str(tmp_path / "jar-dir"))
    assert conddb.main(["--output-dir", str(tmp_path / "out")]) == 1
    assert f"{tmp_path / 'jar-dir' / 'cms_conddb.txt'} is unusable" in capsys.readouterr().err
    assert upstream.calls == []
    write_cookie_file(tmp_path / "jar-dir" / "cms_conddb.txt")
    assert conddb.main(["--output-dir", str(tmp_path / "out")]) == 0


# --- TLS ----------------------------------------------------------------


def test_a_ca_file_is_used_as_it_is(tmp_path, root, cookie_file, monkeypatch):
    # As in the other catalog downloaders: the file is the whole trust store.
    bundle = tmp_path / "cern-cas.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\nFAKECA\n-----END CERTIFICATE-----\n")
    upstream = serve(monkeypatch, SMALL)
    assert _main(tmp_path / "out", cookie_file, "--ca-path", str(bundle)) == 0
    assert {kwargs["verify"] for _, kwargs in upstream.calls} == {str(bundle)}


def test_a_ca_directory_is_used_as_it_is(tmp_path, root, cookie_file, monkeypatch):
    upstream = serve(monkeypatch, SMALL)
    assert _main(tmp_path / "out", cookie_file, "--ca-path", str(tmp_path)) == 0
    assert {kwargs["verify"] for _, kwargs in upstream.calls} == {str(tmp_path)}


def test_without_ca_path_verification_stays_on(tmp_path, root, cookie_file, monkeypatch):
    upstream = serve(monkeypatch, SMALL)
    assert _main(tmp_path / "out", cookie_file) == 0
    assert {kwargs["verify"] for _, kwargs in upstream.calls} == {True}


def test_a_missing_ca_path_fails_before_any_request(tmp_path, root, cookie_file, monkeypatch, capsys):
    upstream = serve(monkeypatch, SMALL)
    err = _fails(tmp_path, cookie_file, capsys, "--ca-path", str(tmp_path / "nowhere.pem"))
    assert "is neither a file nor a directory" in err
    assert upstream.calls == []
