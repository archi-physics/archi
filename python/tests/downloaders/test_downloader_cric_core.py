"""req.catalog-downloaders.cric-core -- the WLCG CRIC core downloader (issue #903), no network.

The fixtures in ``tests/fixtures/cric-core/`` are synthetic (rcsites
``XX-ALPHA`` and ``YY-OTHER``, hosts under ``.example``) in the key layout of
the anonymous ``wlcg-cric.cern.ch/api/core/{service,rcsite,federation}/query/?json``
answers seen on 2026-11-12: a CE service with an ``endpoint``, a virtual SE
service without one, an rcsite whose ``sites`` list mixes CMS and ATLAS
sites, and a federation that serves CMS next to one that serves only ATLAS.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from okg.deployment import EdgeFact, NodeFact

from archi.downloaders import cric_core
from archi.sources.cric import CRICCoreSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cric-core"
BASE = "https://wlcg-cric.example"
URLS = {
    "services.json": f"{BASE}/api/core/service/query/?json",
    "rcsites.json": f"{BASE}/api/core/rcsite/query/?json",
    "federations.json": f"{BASE}/api/core/federation/query/?json",
}
FIXTURE_FOR = {
    "services.json": "service.json",
    "rcsites.json": "rcsite.json",
    "federations.json": "federation.json",
}
CMS_FEDERATION = "XX-ALPHA-FED"


class _Response:
    def __init__(self, status_code=200, body="", headers=None):
        self.status_code = status_code
        self.text = body
        self.headers = headers or {}

    def json(self):
        return json.loads(self.text)


@pytest.fixture
def upstream(monkeypatch):
    answers: dict[str, object] = {
        URLS[name]: json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
        for name, fixture in FIXTURE_FOR.items()
    }
    calls: list[tuple[str, dict]] = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        answer = answers.get(url, _Response(404, "not found"))
        if isinstance(answer, _Response):
            return answer
        return _Response(200, json.dumps(answer))

    monkeypatch.setattr(requests, "get", fake_get)
    fake_get.answers = answers  # type: ignore[attr-defined]
    fake_get.calls = calls  # type: ignore[attr-defined]
    return fake_get


def _main(output: Path, *extra: str) -> int:
    return cric_core.main(["--base-url", BASE, "--output-dir", str(output), *extra])


def _old_cache(output: Path) -> dict[str, str]:
    output.mkdir(parents=True)
    old = {}
    for name in URLS:
        text = json.dumps({"old": name}) + "\n"
        (output / name).write_text(text, encoding="utf-8")
        old[name] = text
    return old


def _assert_untouched(output: Path, old: dict[str, str]) -> None:
    assert {p.name: p.read_text(encoding="utf-8") for p in output.iterdir()} == old


def test_download_feeds_the_reader_services_and_cms_federations(tmp_path, upstream, capsys):
    output = tmp_path / "cric-core"
    assert _main(output) == 0

    assert sorted(p.name for p in output.iterdir()) == sorted(URLS)
    source = CRICCoreSource(
        services_path=str(output / "services.json"),
        rcsites_path=str(output / "rcsites.json"),
        federations_path=str(output / "federations.json"),
    )
    run = source.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    assert run.health.status == "ok"
    assert run.completed_scope is True
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    assert set(nodes) == {
        "svc:XX-ALPHA-CE-HTCONDOR-CE-ce01.alpha.example",
        "svc:XX-ALPHA_SE",
        "svc:YY-OTHER-CE-ARC-CE-arc01.other.example",
        "fed:XX-ALPHA-FED",
    }
    assert nodes["fed:XX-ALPHA-FED"].attrs["pledge_cpu"] == 42000
    edges = {(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)}
    assert edges == {
        ("site:T1_XX_Alpha", "contains", "svc:XX-ALPHA-CE-HTCONDOR-CE-ce01.alpha.example"),
        ("site:T1_XX_Alpha", "contains", "svc:XX-ALPHA_SE"),
        ("site:T1_XX_Alpha", "member_of", "fed:XX-ALPHA-FED"),
    }

    summary = json.loads(capsys.readouterr().out)
    assert summary["files"] == {"federations.json": 2, "rcsites.json": 2, "services.json": 3}
    assert summary["cms_federations"] == 1


def test_no_credential_is_sent(tmp_path, upstream):
    assert _main(tmp_path / "cric-core", "--timeout", "33") == 0
    assert sorted(url for url, _ in upstream.calls) == sorted(URLS.values())
    for _, kwargs in upstream.calls:
        assert "cert" not in kwargs
        assert kwargs["timeout"] == 33


def test_the_default_output_is_cric_core_under_archi_data_root(tmp_path, upstream, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "root"))
    assert cric_core.main(["--base-url", BASE]) == 0
    assert sorted(p.name for p in (tmp_path / "root" / "cric-core").iterdir()) == sorted(URLS)


@pytest.mark.parametrize(
    "name, response, message",
    [
        ("federations.json", _Response(500, "Internal Server Error"), "HTTP 500"),
        ("services.json", _Response(200, "<html>maintenance</html>"), "not JSON"),
    ],
    ids=["http-error", "html-body"],
)
def test_a_failed_fetch_writes_nothing(tmp_path, upstream, capsys, name, response, message):
    upstream.answers[URLS[name]] = response
    output = tmp_path / "cric-core"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert URLS[name] in err
    assert message in err
    _assert_untouched(output, old)


def _no_cms_federation(answers):
    del answers[URLS["federations.json"]][CMS_FEDERATION]


def _cms_only_in_pledges(answers):
    federation = answers[URLS["federations.json"]][CMS_FEDERATION]
    federation["vos"] = ["atlas"]


def _federations_empty(answers):
    answers[URLS["federations.json"]] = {}


def _services_error_body(answers):
    answers[URLS["services.json"]] = {"errors": ["bad query"]}


def _service_without_rcsite(answers):
    del answers[URLS["services.json"]]["XX-ALPHA_SE"]["rcsite"]


def _rcsite_sites_not_list(answers):
    answers[URLS["rcsites.json"]]["XX-ALPHA"]["sites"] = {"T1_XX_Alpha": "cms"}


def _pledges_not_nested(answers):
    answers[URLS["federations.json"]][CMS_FEDERATION]["pledges"] = {"2026": [40000]}


def _federation_vos_not_list(answers):
    answers[URLS["federations.json"]][CMS_FEDERATION]["vos"] = "cms"


@pytest.mark.parametrize(
    "name, mutate, message",
    [
        ("federations.json", _no_cms_federation, "federations serves CMS"),
        ("federations.json", _federations_empty, "empty"),
        ("services.json", _services_error_body, "API error body"),
        ("services.json", _service_without_rcsite, "XX-ALPHA_SE has no 'rcsite'"),
        ("rcsites.json", _rcsite_sites_not_list, "XX-ALPHA 'sites' is not a list"),
        ("federations.json", _pledges_not_nested, "XX-ALPHA-FED 'pledges'"),
        ("federations.json", _federation_vos_not_list, "XX-ALPHA-FED 'vos' is not a list"),
    ],
    ids=lambda value: value.__name__.strip("_") if callable(value) else None,
)
def test_an_unusable_payload_writes_nothing(tmp_path, upstream, capsys, name, mutate, message):
    mutate(upstream.answers)
    output = tmp_path / "cric-core"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert URLS[name] in err
    assert message in err
    _assert_untouched(output, old)


def test_a_federation_serving_cms_only_through_pledges_counts(tmp_path, upstream):
    # The reader's rule: a 'cms' VO, or a 'cms' entry in any pledge quarter.
    _cms_only_in_pledges(upstream.answers)
    assert _main(tmp_path / "cric-core") == 0
