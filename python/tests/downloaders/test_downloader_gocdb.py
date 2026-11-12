"""req.catalog-downloaders.gocdb -- the GOCDB downtime downloader (issue #903), no network.

``tests/fixtures/gocdb/downtimes.xml`` is synthetic (sites ``XX-ALPHA``,
``YY-OTHER`` and ``ZZ-UNRELATED``, hosts under ``.example``) in the element
layout of the anonymous
``goc.egi.eu/gocdbpi/public/?method=get_downtime&windowstart=...&windowend=...``
answer seen on 2026-11-12: one ``DOWNTIME`` element per affected service, so
downtime 9101 appears twice, once with a filled ``AFFECTED_ENDPOINTS``.
Its hosts match the services in ``tests/fixtures/cric-core/service.json``,
and the CRIC caches the reader also needs are the ``cric`` and ``cric-core``
fixtures.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from support.responses import Attempts, FakeResponse

from okg.deployment import EdgeFact, NodeFact

from archi.downloaders import _catalog, gocdb
from archi.sources.gocdb import GoCDBDowntimeSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
XML = (FIXTURES / "gocdb" / "downtimes.xml").read_text(encoding="utf-8")
BASE = "https://gocdb.example/gocdbpi/public/"


@pytest.fixture(autouse=True)
def _no_grid_ca_dir_and_no_sleep(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path / "no-grid-ca")
    monkeypatch.setattr(_catalog.time, "sleep", lambda seconds: None)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """An ARCHI_DATA_ROOT that already holds the two CRIC caches."""
    root = tmp_path / "root"
    (root / "cric").mkdir(parents=True)
    (root / "cric-core").mkdir(parents=True)
    shutil.copy(FIXTURES / "cric" / "site.json", root / "cric" / "sites.json")
    shutil.copy(FIXTURES / "cric-core" / "service.json", root / "cric-core" / "services.json")
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(root))
    return root


class Upstream:
    """Answers every GET with ``answer`` (its query holds today's window,
    so no URL map); records each call."""

    def __init__(self, monkeypatch):
        self.answer = FakeResponse(200, XML, {"Content-Type": "application/xml"})
        self.calls: list[tuple[str, dict]] = []

        def get(session, url, **kwargs):
            self.calls.append((url, kwargs))
            answer = self.answer.pop(0) if isinstance(self.answer, Attempts) else self.answer
            if isinstance(answer, BaseException):
                raise answer
            return answer

        monkeypatch.setattr(requests.Session, "get", get)

    @property
    def url(self) -> str:
        assert len(self.calls) == 1
        return self.calls[0][0]

    def query(self) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.url).query)


@pytest.fixture
def upstream(monkeypatch):
    return Upstream(monkeypatch)


def _main(*extra: str) -> int:
    return gocdb.main(["--base-url", BASE, *extra])


def _records(root: Path) -> Path:
    return root / "gocdb-downtimes" / "records.json"


def _reader(root: Path) -> GoCDBDowntimeSource:
    return GoCDBDowntimeSource(
        records_path=str(_records(root)),
        sites_path=str(root / "cric" / "sites.json"),
        services_path=str(root / "cric-core" / "services.json"),
    )


def test_download_feeds_the_reader_downtimes_linked_to_cric_services(root, upstream, capsys):
    assert _main() == 0

    run = _reader(root).run("run-1", mode="scope_complete")
    facts = list(run.facts)
    assert run.health.status == "ok"
    assert run.completed_scope is True
    assert run.health.record_count == 3
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    assert set(nodes) == {"downtime:9101", "downtime:9102", "downtime:9103"}
    first = nodes["downtime:9101"].attrs
    assert first["primary_key"] == "900101G0"
    assert first["site"] == "XX-ALPHA"
    assert first["severity"] == "outage"
    assert first["classification"] == "scheduled"
    assert first["description"] == "Cooling maintenance & firmware upgrade"
    assert first["start_time"] == "2026-11-14T04:00:00+00:00"
    assert first["end_time"] == "2026-11-14T16:00:00+00:00"
    assert first["affected_hostnames"] == ["ce01.alpha.example", "se01.alpha.example"]
    assert first["affected_services"] == ["org.opensciencegrid.htcondorce", "webdav"]
    edges = {(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)}
    # GOCDB's HOSTED_BY is a GOCDB site (a CRIC rcsite), not a CMS site
    # name, so only the hostname -> CRIC-core service links form.
    assert edges == {
        ("downtime:9101", "affects", "svc:XX-ALPHA-CE-HTCONDOR-CE-ce01.alpha.example"),
        ("downtime:9102", "affects", "svc:YY-OTHER-CE-ARC-CE-arc01.other.example"),
    }

    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(_records(root))
    assert summary["rows"] == 4
    assert summary["records"] == 3
    assert summary["url"] == upstream.url


def _today() -> datetime:
    return datetime.now(timezone.utc)


def _sent_window(upstream: Upstream) -> tuple[str, str]:
    query = upstream.query()
    assert query["method"] == ["get_downtime"]
    assert set(query) == {"method", "windowstart", "windowend"}
    return query["windowstart"][0], query["windowend"][0]


def _expected(days_back: int, days_ahead: int, *instants: datetime) -> set[tuple[str, str]]:
    # Computed around the call, so a run across UTC midnight still matches.
    return {
        (
            (at - timedelta(days=days_back)).date().isoformat(),
            (at + timedelta(days=days_ahead)).date().isoformat(),
        )
        for at in instants
    }


def test_the_default_window_is_seven_days_back_and_sixty_ahead(root, upstream):
    before = _today()
    assert _main() == 0
    after = _today()
    assert _sent_window(upstream) in _expected(7, 60, before, after)
    assert upstream.url.startswith(BASE + "?")


def test_the_window_flags_set_the_dates_sent(root, upstream, capsys):
    before = _today()
    assert _main("--days-back", "0", "--days-ahead", "3") == 0
    after = _today()
    sent = _sent_window(upstream)
    assert sent in _expected(0, 3, before, after)
    assert json.loads(capsys.readouterr().out)["window"] == list(sent)


def test_a_negative_window_is_a_usage_error(root, upstream):
    with pytest.raises(SystemExit) as exc:
        _main("--days-back", "-1")
    assert exc.value.code == 2
    assert upstream.calls == []


def test_the_request_is_anonymous_and_asks_for_xml(root, upstream):
    assert _main("--timeout", "33") == 0
    (_, kwargs), = upstream.calls
    assert "cert" not in kwargs
    assert kwargs["timeout"] == 33
    assert kwargs["headers"]["Accept"] == "application/xml"
    assert kwargs["allow_redirects"] is False


def test_the_default_output_is_gocdb_downtimes_under_archi_data_root(root, upstream):
    assert _main() == 0
    assert sorted(p.name for p in (root / "gocdb-downtimes").iterdir()) == [".lock", "records.json"]


def _reversed_downtimes(xml: str) -> str:
    head, _, rest = xml.partition("  <DOWNTIME ")
    body, _, tail = rest.rpartition("</DOWNTIME>\n")
    elements = ("  <DOWNTIME " + body + "</DOWNTIME>\n").split("</DOWNTIME>\n")[:-1]
    return head + "".join(e + "</DOWNTIME>\n" for e in reversed(elements)) + tail


def test_rows_are_written_sorted_one_per_service(root, upstream):
    # GOCDB's order is not part of its contract; sorted rows keep the
    # reader's content hash when only the order changes.
    upstream.answer = FakeResponse(200, _reversed_downtimes(XML))
    assert _main() == 0
    rows = json.loads(_records(root).read_text(encoding="utf-8"))
    assert [(r["downtime_id"], r["hostname"]) for r in rows] == [
        (9101, "ce01.alpha.example"),
        (9101, "se01.alpha.example"),
        (9102, "arc01.other.example"),
        (9103, "bdii.unrelated.example"),
    ]
    assert rows[2] == {
        "downtime_id": 9102,
        "primary_key": "900102G0",
        "classification": "UNSCHEDULED",
        "severity": "WARNING",
        "description": "Worker nodes partly drained",
        "hosted_by": "YY-OTHER",
        "hostname": "arc01.other.example",
        "service_type": "ARC-CE",
        "start_date": "2026-11-10T09:00:00+00:00",
        "end_date": "2026-11-20T09:00:00+00:00",
    }


def test_explicit_cric_cache_paths_are_used(tmp_path, root, upstream):
    elsewhere = tmp_path / "elsewhere"
    shutil.move(root / "cric", elsewhere / "cric")
    shutil.move(root / "cric-core", elsewhere / "cric-core")
    output = tmp_path / "out"
    assert _main(
        "--output-dir", str(output),
        "--sites-path", str(elsewhere / "cric" / "sites.json"),
        "--services-path", str(elsewhere / "cric-core" / "services.json"),
    ) == 0
    assert (output / "records.json").is_file()


# --- fail closed ---------------------------------------------------------


ENTITY_DOC = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE results [<!ENTITY e "injected">]>\n'
    + XML.split("\n", 1)[1].replace("Worker nodes partly drained", "&e;")
)


def _raw(content: bytes) -> FakeResponse:
    """A 200 answer whose body is ``content`` as sent (not UTF-8 text)."""
    response = FakeResponse(200)
    response.content = content
    return response


def _old_cache(root: Path) -> bytes:
    _records(root).parent.mkdir(parents=True)
    _records(root).write_text(json.dumps([{"downtime_id": 1}]) + "\n", encoding="utf-8")
    return _records(root).read_bytes()


def _assert_untouched(root: Path, old: bytes) -> None:
    # The write lock file may be there; nothing else, and no staging directory.
    directory = _records(root).parent
    assert sorted(p.name for p in directory.iterdir()) in (["records.json"], [".lock", "records.json"])
    assert _records(root).read_bytes() == old


@pytest.mark.parametrize(
    "answer, message",
    [
        (Attempts([FakeResponse(503, "busy")] * 3), "HTTP 503 after 3 attempts"),
        (
            FakeResponse(302, "", {"Location": "https://login.example/sso"}),
            "redirect to https://login.example/sso",
        ),
        (requests.ConnectionError("refused"), "ConnectionError"),
        (FakeResponse(200, "<results><DOWNTIME ID='1'>"), "not XML"),
        (
            FakeResponse(
                200,
                "Failed to parse time string (bogus) at position 0 (b): "
                "The timezone could not be found in the database",
            ),
            "not XML",
        ),
        (
            FakeResponse(200, "<error>Unsupported parameter: page</error>"),
            "not a GOCDB results document (root <error>): Unsupported parameter: page",
        ),
        (
            FakeResponse(
                200,
                '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "x">]><results>&a;</results>',
            ),
            "declares a DTD",
        ),
        (_raw(ENTITY_DOC.replace("UTF-8", "UTF-16").encode("utf-16")), "declares a DTD"),
        (_raw(ENTITY_DOC.replace("UTF-8", "UTF-16").encode("utf-16-le")), "declares a DTD"),
        (
            FakeResponse(200, "<!DOCTYPE html>\n<html><body>Scheduled maintenance</body></html>"),
            "declares a DTD; refusing to parse it: <!DOCTYPE html>\n<html><body>Scheduled maintenance",
        ),
        (
            FakeResponse(200, '<?xml version="1.0" encoding="UTF-8"?>\n<results/>\n'),
            "no downtimes in this window",
        ),
        (
            FakeResponse(200, XML.replace('ID="9102"', 'ID="9102a"')),
            "a DOWNTIME has ID '9102a', not an integer",
        ),
        (
            FakeResponse(200, XML.replace("<END_DATE>1795165200</END_DATE>", "<END_DATE/>")),
            "DOWNTIME 9102 has END_DATE '', not epoch seconds",
        ),
        (
            FakeResponse(200, XML.replace("1793520000", "9" * 31)),
            f"DOWNTIME 9103 has START_DATE '{'9' * 31}', not epoch seconds",
        ),
        (
            FakeResponse(200, XML.replace('ID="9102"', 'ID="\u00b2"')),
            "a DOWNTIME has ID '\u00b2', not an integer",
        ),
    ],
    ids=[
        "http-error",
        "login-redirect",
        "connection-error",
        "truncated-xml",
        "plain-text-error",
        "error-document",
        "dtd",
        "dtd-utf16-bom",
        "dtd-utf16-no-bom",
        "html-maintenance-page",
        "empty-window",
        "non-integer-id",
        "missing-date",
        "date-out-of-range",
        "non-ascii-digit-id",
    ],
)
def test_a_failed_or_unusable_answer_writes_nothing(root, upstream, capsys, answer, message):
    upstream.answer = answer
    old = _old_cache(root)
    assert _main() == 1
    err = capsys.readouterr().err
    assert err.startswith("gocdb download failed, cache not written: ")
    assert upstream.calls[0][0] in err
    assert message in err
    _assert_untouched(root, old)


def test_a_downtime_the_reader_skips_writes_nothing(root, upstream, capsys):
    # ID 0 parses as an integer, but the reader skips it and withholds the
    # complete scope, so the check refuses the whole answer.
    upstream.answer = FakeResponse(200, XML.replace('ID="9103"', 'ID="0"'))
    old = _old_cache(root)
    assert _main() == 1
    err = capsys.readouterr().err
    assert "the reader refuses the downloaded answers" in err
    assert "skipped" in err
    assert upstream.url in err
    assert ".staging." not in err
    _assert_untouched(root, old)


def test_a_cric_cache_the_reader_refuses_writes_nothing(root, upstream, capsys):
    (root / "cric" / "sites.json").write_text(json.dumps({"error": "bad query"}), encoding="utf-8")
    old = _old_cache(root)
    assert _main() == 1
    err = capsys.readouterr().err
    assert "the reader refuses the downloaded answers (health endpoint_failed" in err
    assert str(root / "cric" / "sites.json") in err
    _assert_untouched(root, old)


@pytest.mark.parametrize("missing", ["cric/sites.json", "cric-core/services.json"])
def test_a_missing_cric_cache_stops_before_any_request(root, upstream, capsys, missing):
    (root / missing).unlink()
    old = _old_cache(root)
    assert _main() == 1
    err = capsys.readouterr().err
    assert str(root / missing) in err
    assert "python -m archi.downloaders.cric" in err
    assert upstream.calls == []
    _assert_untouched(root, old)

