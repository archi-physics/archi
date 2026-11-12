"""req.catalog-downloaders.cric -- the CMS CRIC downloader (issue #903), no network.

The fixtures in ``tests/fixtures/cric/`` are synthetic (sites ``T1_XX_Alpha``
and ``T2_XX_Beta``, users ``aexample`` and ``bexample``, hosts under
``.example``) in the key layout of the real answers:

- ``site.json``, ``storageunit.json``, ``computeunit.json`` follow the
  anonymous ``cms-cric.cern.ch/api/cms/{site,storageunit,computeunit}/query/?json``
  answers seen on 2026-11-12, cut to two sites with the nested ``gWMS`` and
  ``glideinentries`` lists shortened.
- ``facility.json`` and ``site-responsibilities.json``: those endpoints
  answer an anonymous request with a 302 to ``/accounts/login/``, so their
  layout comes from CRIC's API help page (``/api/cms/facility/query/help/``
  lists ``name``, ``fullname``, ``country``, ``timezone``, ``state``,
  ``cmssites``) and from the reader contract (rows of ``username``,
  ``site_name``, ``role``, the columns ``archi.snapshot.groups`` checks; a
  null username or site, which the reader drops, as real exports have).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from support.responses import FakeResponse, patch_session_get
from support.x509 import write_cert

from okg.deployment import EdgeFact, NodeFact

from archi.downloaders import _catalog, cric
from archi.downloaders._catalog import DownloadError
from archi.sources.cric import CRICSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cric"
BASE = "https://cric.example"
URLS = {
    "sites.json": f"{BASE}/api/cms/site/query/?json",
    "storage_units.json": f"{BASE}/api/cms/storageunit/query/?json",
    "compute_units.json": f"{BASE}/api/cms/computeunit/query/?json",
    "facilities.json": f"{BASE}/api/cms/facility/query/?json",
    "responsibilities.json": (
        f"{BASE}/api/accounts/user/query/?json&preset=site-responsibilities"
    ),
}
FIXTURE_FOR = {
    "sites.json": "site.json",
    "storage_units.json": "storageunit.json",
    "compute_units.json": "computeunit.json",
    "facilities.json": "facility.json",
    "responsibilities.json": "site-responsibilities.json",
}


@pytest.fixture(autouse=True)
def _no_grid_ca_dir_and_no_sleep(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path / "no-grid-ca")
    monkeypatch.setattr(_catalog.time, "sleep", lambda seconds: None)


@pytest.fixture
def proxy(tmp_path, monkeypatch):
    path = tmp_path / "x509up_test"
    write_cert(path, not_after=datetime.now(timezone.utc) + timedelta(hours=12))
    monkeypatch.setenv("X509_USER_PROXY", str(path))
    return path


@pytest.fixture
def upstream(monkeypatch):
    return patch_session_get(
        monkeypatch,
        {
            URLS[name]: json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
            for name, fixture in FIXTURE_FOR.items()
        },
    )


def _main(output: Path, *extra: str) -> int:
    return cric.main(["--base-url", BASE, "--output-dir", str(output), *extra])


def _reader(output: Path) -> CRICSource:
    return CRICSource(
        sites_path=str(output / "sites.json"),
        storage_units_path=str(output / "storage_units.json"),
        compute_units_path=str(output / "compute_units.json"),
        facilities_path=str(output / "facilities.json"),
        responsibilities_path=str(output / "responsibilities.json"),
    )


def _old_cache(output: Path) -> dict[str, bytes]:
    output.mkdir(parents=True)
    for name in URLS:
        (output / name).write_text(json.dumps({"old": name}) + "\n", encoding="utf-8")
    return _snapshot(output)


def _snapshot(output: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in output.iterdir()}


def _assert_untouched(output: Path, old: dict[str, bytes]) -> None:
    assert _snapshot(output) == old
    # No staging directory is left next to the cache.
    assert sorted(p.name for p in output.parent.iterdir() if p.name.startswith(".")) == []


# --- the round trip -----------------------------------------------------


def test_download_feeds_the_reader_a_complete_topology(tmp_path, proxy, upstream, capsys):
    output = tmp_path / "cric"
    assert _main(output) == 0

    assert sorted(p.name for p in output.iterdir()) == sorted(URLS)
    run = _reader(output).run("run-1", mode="scope_complete")
    facts = list(run.facts)
    assert run.health.status == "ok"
    assert run.completed_scope is True
    nodes = {f.node_id for f in facts if isinstance(f, NodeFact)}
    assert nodes == {
        "facility:XX_Alpha",
        "site:T1_XX_Alpha",
        "site:T2_XX_Beta",
        "se:T1_XX_Alpha_Disk",
        "se:T2_XX_Beta",
        "ce:CU_T1_XX_Alpha",
        "ce:CU_T2_XX_Beta",
        "op:aexample",
        "op:bexample",
    }
    edges = {(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)}
    assert ("op:aexample", "responsible_for", "site:T2_XX_Beta") in edges
    assert ("facility:XX_Alpha", "contains", "site:T1_XX_Alpha") in edges
    assert ("site:T2_XX_Beta", "contains", "ce:CU_T2_XX_Beta") in edges

    summary = json.loads(capsys.readouterr().out)
    assert summary["files"] == {
        "compute_units.json": 2,
        "facilities.json": 1,
        "responsibilities.json": 5,
        "sites.json": 2,
        "storage_units.json": 2,
    }
    assert summary["records"] == 9
    assert summary["output"] == str(output)


def test_the_cache_keeps_the_upstream_payload_and_its_tier_levels(tmp_path, proxy, upstream):
    output = tmp_path / "cric"
    assert _main(output) == 0
    sites = json.loads((output / "sites.json").read_text(encoding="utf-8"))
    assert sites == upstream.answers[URLS["sites.json"]]
    assert {name: site["tier_level"] for name, site in sites.items()} == {
        "T1_XX_Alpha": 1,
        "T2_XX_Beta": 2,
    }
    responsibilities = json.loads((output / "responsibilities.json").read_text())
    assert responsibilities == upstream.answers[URLS["responsibilities.json"]]


def test_every_request_carries_the_proxy(tmp_path, proxy, upstream):
    assert _main(tmp_path / "cric", "--timeout", "17") == 0
    assert sorted(url for url, _ in upstream.calls) == sorted(URLS.values())
    for _, kwargs in upstream.calls:
        assert kwargs["cert"] == (str(proxy), str(proxy))
        assert kwargs["timeout"] == 17


def test_the_default_output_is_cric_under_archi_data_root(tmp_path, proxy, upstream, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "root"))
    assert cric.main(["--base-url", BASE]) == 0
    assert sorted(p.name for p in (tmp_path / "root" / "cric").iterdir()) == sorted(URLS)


def test_an_explicit_proxy_overrides_the_environment(tmp_path, proxy, upstream, monkeypatch):
    other = tmp_path / "other-proxy"
    write_cert(other, not_after=datetime.now(timezone.utc) + timedelta(hours=1))
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "absent"))
    assert _main(tmp_path / "cric", "--proxy", str(other)) == 0
    assert {kwargs["cert"] for _, kwargs in upstream.calls} == {(str(other), str(other))}


# --- fail closed --------------------------------------------------------


def test_no_proxy_stops_before_any_request(tmp_path, upstream, monkeypatch, capsys):
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "absent"))
    output = tmp_path / "cric"
    old = _old_cache(output)
    assert _main(output) == 1
    assert upstream.calls == []
    assert "voms-proxy-init -voms cms" in capsys.readouterr().err
    _assert_untouched(output, old)


def test_an_expired_proxy_stops_before_any_request(tmp_path, upstream, monkeypatch, capsys):
    expired = tmp_path / "expired"
    write_cert(expired, not_after=datetime.now(timezone.utc) - timedelta(minutes=5))
    monkeypatch.setenv("X509_USER_PROXY", str(expired))
    assert _main(tmp_path / "cric") == 1
    assert upstream.calls == []
    assert "expired" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name, response, message",
    [
        (
            "facilities.json",
            FakeResponse(302, "", {"Location": "/accounts/login/?next=/api/cms/facility/query/"}),
            "HTTP 302",
        ),
        ("sites.json", FakeResponse(503, "Service Unavailable"), "HTTP 503 after 3 attempts"),
        ("compute_units.json", FakeResponse(200, "<html>Sign in</html>"), "not JSON"),
        ("storage_units.json", FakeResponse(200, "{\"truncated\": "), "not JSON"),
    ],
    ids=["login-redirect", "http-error", "html-body", "truncated-json"],
)
def test_a_failed_fetch_writes_nothing(tmp_path, proxy, upstream, capsys, name, response, message):
    upstream.answers[URLS[name]] = response
    output = tmp_path / "cric"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert err.startswith("cric download failed, cache not written: ")
    assert URLS[name] in err
    assert message in err
    assert str(proxy) not in err
    _assert_untouched(output, old)


# Pre-checks: refused before the reader runs, with the answer's URL.


def _sites_without_tier(answers):
    # The reader would read the site as Tier-0; the downloader refuses it.
    del answers[URLS["sites.json"]]["T2_XX_Beta"]["tier_level"]


def _sites_bool_tier(answers):
    answers[URLS["sites.json"]]["T2_XX_Beta"]["tier_level"] = True


def _sites_error_body(answers):
    answers[URLS["sites.json"]] = {"error": "internal error"}


def _sites_empty(answers):
    answers[URLS["sites.json"]] = {}


def _sites_list(answers):
    answers[URLS["sites.json"]] = [{"name": "T2_XX_Beta"}]


def _storage_value_not_object(answers):
    answers[URLS["storage_units.json"]]["T2_XX_Beta"] = "DISK"


def _resp_no_result(answers):
    answers[URLS["responsibilities.json"]] = {"desc": {"columns": ["username", "site_name", "role"]}}


def _resp_empty_result(answers):
    answers[URLS["responsibilities.json"]]["result"] = []


def _resp_sites_compat_rows(answers):
    answers[URLS["responsibilities.json"]] = {
        "desc": {"columns": ["site_name", "tier_level", "tier", "country", "usage"]},
        "result": [["T2_XX_Beta", 2, "T2", "XX", "production"]],
    }


def _resp_short_row(answers):
    answers[URLS["responsibilities.json"]]["result"].append(["dexample", "Beta"])


@pytest.mark.parametrize(
    "name, mutate, message",
    [
        ("sites.json", _sites_without_tier, "site T2_XX_Beta has no integer tier_level"),
        ("sites.json", _sites_bool_tier, "site T2_XX_Beta has no integer tier_level"),
        ("sites.json", _sites_error_body, "API error body"),
        ("sites.json", _sites_empty, "empty"),
        ("sites.json", _sites_list, "a list, not a JSON object"),
        ("storage_units.json", _storage_value_not_object, "entry T2_XX_Beta is a str"),
        ("responsibilities.json", _resp_no_result, "no 'result' list"),
        ("responsibilities.json", _resp_empty_result, "empty 'result'"),
        ("responsibilities.json", _resp_sites_compat_rows, "row 0 is not"),
        ("responsibilities.json", _resp_short_row, "row 5 is not"),
    ],
    ids=lambda value: value.__name__.strip("_") if callable(value) else None,
)
def test_a_pre_check_refusal_writes_nothing(tmp_path, proxy, upstream, capsys, name, mutate, message):
    mutate(upstream.answers)
    output = tmp_path / "cric"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert URLS[name] in err
    assert message in err
    _assert_untouched(output, old)


# Reader refusals: answers that pass the pre-checks but that the reader
# would not accept with health ok and a complete scope.


def _facility_cmssite_is_a_list(answers):
    # Review finding: a list entry is unhashable in the reader's site lookup.
    answers[URLS["facilities.json"]]["XX_Alpha"]["cmssites"] = [["T1_XX_Alpha"]]


def _resp_no_title_matches(answers):
    answers[URLS["responsibilities.json"]]["result"] = [
        ["aexample", "Renamed Alpha", "Site Executive"],
        ["bexample", None, "Site Admin"],
    ]


def _no_cms_site_name(answers):
    sites = answers[URLS["sites.json"]]
    answers[URLS["sites.json"]] = {f"XX-{name}": site for name, site in sites.items()}


@pytest.mark.parametrize(
    "name, mutate, message",
    [
        ("facilities.json", _facility_cmssite_is_a_list, "unexpected shape"),
        ("responsibilities.json", _resp_no_title_matches, "none maps to an operator"),
        ("sites.json", _no_cms_site_name, "shaped like a CMS site name"),
    ],
    ids=lambda value: value.__name__.strip("_") if callable(value) else None,
)
def test_a_reader_refusal_writes_nothing(tmp_path, proxy, upstream, capsys, name, mutate, message):
    mutate(upstream.answers)
    output = tmp_path / "cric"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert "the reader refuses the downloaded answers (health endpoint_failed" in err
    assert message in err
    assert URLS[name] in err
    assert ".staging." not in err
    _assert_untouched(output, old)


def test_collect_raises_download_error_before_any_write(upstream):
    upstream.answers[URLS["sites.json"]] = {}
    with pytest.raises(DownloadError, match="empty"):
        cric.collect(lambda url: upstream.answers[url], base_url=BASE)
