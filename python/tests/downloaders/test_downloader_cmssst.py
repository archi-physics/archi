"""req.cmssst-site-status.download -- the CMS SST downloader, no network.

summary.js is the real file fetched on 2026-10-07 (91 sites); the site JSON
fixtures are real too, minus the per-host ``elements`` list, which these
tests put back (small) to prove the downloader drops it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from archi.downloaders import cmssst
from archi.downloaders.cmssst import DownloadError, collect, parse_site_list, write_records

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cmssst"
BASE = "https://cmssst.example/siteStatus/data/"
SITES = ("T1_US_FNAL", "T2_US_MIT", "T2_CH_CERN")


def _summary() -> str:
    return (FIXTURES / "summary.js").read_text(encoding="utf-8")


def _site_body(site: str) -> str:
    payload = json.loads((FIXTURES / f"{site}.json").read_text(encoding="utf-8"))
    payload["elements"] = [{"host": "se.example", "type": "WEBDAV", "metrics": {}}]
    return json.dumps(payload)


def _fetcher(bodies: dict[str, str]):
    calls: list[tuple[str, int]] = []

    def fetch(url: str, timeout: int) -> str:
        calls.append((url, timeout))
        if url not in bodies:
            raise DownloadError(f"GET {url} returned HTTP 404")
        return bodies[url]

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_parses_the_real_summary_site_list():
    sites = parse_site_list(_summary())
    assert len(sites) == 91
    assert sites[0] == "T0_CH_CERN"
    assert {"T1_US_FNAL", "T2_US_MIT", "T2_CH_CERN"} <= set(sites)
    assert len(set(sites)) == len(sites)


@pytest.mark.parametrize(
    "script, message",
    [
        ("<html>Service unavailable</html>", "no 'var siteStatusData"),
        ("var siteStatusData = [\n];\n", "holds no site entry"),
        ('var siteStatusData = [ { site: "bogus", today: "o" } ];', "not CMS site names"),
    ],
    ids=["error-page", "empty-list", "bad-site-name"],
)
def test_a_bad_summary_is_refused(script, message):
    with pytest.raises(DownloadError, match=message):
        parse_site_list(script)


def test_parse_is_bounded_to_the_array_and_skips_comments():
    # Review finding 3: commented-out entries and a later array in the
    # same file are not sites.
    script = (
        "var siteStatusData = [\n"
        '   { site: "T2_US_MIT", today: "o" },\n'
        '   // { site: "T2_XX_LineComment", today: "o" },\n'
        '   /* { site: "T2_XX_BlockComment",\n      today: "o" }, */\n'
        '   { site: "T1_US_FNAL", msg: "see // not a comment", today: "o" }\n'
        "];\n"
        'var otherData = [ { site: "T2_XX_Trailing" } ];\n'
    )
    assert parse_site_list(script) == ["T2_US_MIT", "T1_US_FNAL"]


def test_an_unclosed_array_is_refused():
    with pytest.raises(DownloadError, match="never closed"):
        parse_site_list('var siteStatusData = [ { site: "T2_US_MIT" },')


def test_collect_keeps_site_fields_and_drops_elements(tmp_path):
    summary = 'var siteStatusData = [\n' + ",\n".join(
        f'   {{ site: "{s}", ggus: [0, 0, 0], today: "oo" }}' for s in SITES
    ) + "\n];\n"
    bodies = {BASE + "summary.js": summary}
    bodies.update({f"{BASE}{s}.json": _site_body(s) for s in SITES})
    fetch = _fetcher(bodies)
    records = collect(fetch, base_url=BASE, timeout=7)
    assert [r["site"] for r in records] == list(SITES)
    assert all(set(r) == {"site", "time", "alert", "msg", "ggus", "metrics"} for r in records)
    fnal = records[0]
    assert fnal["time"] == 1791383941
    assert fnal["ggus"] == [[1003499, 1785852955], [1003889, 1789473399]]
    assert len(fnal["metrics"]) == 18
    assert all(timeout == 7 for _, timeout in fetch.calls)

    out = tmp_path / "data" / "cmssst-site-status" / "records.json"
    write_records(out, records)
    assert json.loads(out.read_text(encoding="utf-8")) == records
    assert not [p for p in out.parent.iterdir() if p.name != "records.json"]


@pytest.mark.parametrize(
    "body, message",
    [
        ('{"error": "not found"}', "names site None"),
        ("[]", "is a list, not an object"),
        ("<html>500</html>", "is not JSON"),
        (json.dumps({"site": "T2_US_MIT", "time": 5, "metrics": {}}), "no 'metrics'"),
        (json.dumps({"site": "T2_US_MIT", "metrics": {"x": {}}}), "positive integer 'time'"),
    ],
    ids=["error-body", "list", "html", "empty-metrics", "no-time"],
)
def test_an_error_shaped_site_body_fails_and_writes_nothing(tmp_path, body, message):
    bodies = {
        f"{BASE}T1_US_FNAL.json": _site_body("T1_US_FNAL"),
        f"{BASE}T2_US_MIT.json": body,
    }
    out = tmp_path / "records.json"
    out.write_text("[\"previous cache\"]", encoding="utf-8")
    with pytest.raises(DownloadError, match=message):
        records = collect(
            _fetcher(bodies), base_url=BASE, sites=["T1_US_FNAL", "T2_US_MIT"]
        )
        write_records(out, records)
    assert out.read_text(encoding="utf-8") == "[\"previous cache\"]"


def test_a_failed_fetch_fails_the_whole_download():
    bodies = {f"{BASE}T1_US_FNAL.json": _site_body("T1_US_FNAL")}
    with pytest.raises(DownloadError, match="HTTP 404"):
        collect(_fetcher(bodies), base_url=BASE, sites=["T1_US_FNAL", "T2_US_MIT"])


def test_an_empty_result_is_never_written(tmp_path):
    with pytest.raises(DownloadError, match="empty cache"):
        collect(_fetcher({}), base_url=BASE, sites=[])
    with pytest.raises(DownloadError, match="empty cache"):
        write_records(tmp_path / "records.json", [])
    assert not (tmp_path / "records.json").exists()


def test_main_reports_failure_and_leaves_no_cache(tmp_path, monkeypatch, capsys):
    def offline(url, timeout):
        raise DownloadError(f"GET {url} returned HTTP 503")

    monkeypatch.setattr(cmssst, "http_fetch", offline)
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    assert cmssst.main([]) == 1
    assert "cache not written" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_main_writes_the_default_path(tmp_path, monkeypatch):
    bodies = {cmssst.DEFAULT_BASE_URL + "summary.js": (
        'var siteStatusData = [ { site: "T2_US_MIT", today: "o" } ];'
    ), f"{cmssst.DEFAULT_BASE_URL}T2_US_MIT.json": _site_body("T2_US_MIT")}
    monkeypatch.setattr(cmssst, "http_fetch", _fetcher(bodies))
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    assert cmssst.main([]) == 0
    written = json.loads(
        (tmp_path / "data" / "cmssst-site-status" / "records.json").read_text()
    )
    assert [r["site"] for r in written] == ["T2_US_MIT"]
