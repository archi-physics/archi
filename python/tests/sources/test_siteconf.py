"""req.w2.sources-catalogs — SITECONFSource emission (fixture mode)."""
import json

from okg.deployment import (
    EdgeFact,
    NodeFact,
    PreflightResult,
)

from archi.sources.siteconf import SiteConfRecord, SITECONFSource

RECORD = SiteConfRecord(
    site_name="T2_US_MIT",
    storage_protocols=("XRootD", "WebDAV"),
    se_endpoints=("xrootd.cmsaf.mit.edu",),
    ce_endpoints=("ce01.cmsaf.mit.edu",),
    local_stage_out="T2_US_MIT",
    fallback_stage_out="T1_US_FNAL",
    frontier_config="http://frontier.cern.ch",
    raw_storage_json='[{"protocol": "XRootD"}]',
)


def _source(tmp_path, records):
    (tmp_path / "data" / "cric").mkdir(parents=True)
    (tmp_path / "data" / "cric" / "sites.json").write_text(
        json.dumps({"T2_US_MIT": {}})
    )
    return SITECONFSource(records=records, base=str(tmp_path))


def test_fixture_records_emit_site_config_and_contains_edge(tmp_path):
    source = _source(tmp_path, [RECORD])
    run = source.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    nodes = [f for f in facts if isinstance(f, NodeFact)]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.node_id == "siteconf:T2_US_MIT"
    assert node.subtype == "site_config"
    assert node.attrs["storage_protocols"] == ["XRootD", "WebDAV"]
    assert node.attrs["se_endpoints"] == ["xrootd.cmsaf.mit.edu"]
    assert node.attrs["local_stage_out"] == "T2_US_MIT"
    assert "SITECONF for T2_US_MIT" in node.attrs["text"]
    edges = [f for f in facts if isinstance(f, EdgeFact)]
    assert [(e.src, e.edge_type, e.dst) for e in edges] == [
        ("site:T2_US_MIT", "contains", "siteconf:T2_US_MIT"),
    ]
    assert edges[0].provenance == "authoritative"
    assert run.health.mode == "fixture"


def test_unknown_site_gets_node_but_no_edge_and_preflight_fixture(tmp_path):
    record = SiteConfRecord(site_name="T2_XX_Nowhere", raw_storage_json="{}")
    source = _source(tmp_path, [record])
    facts = list(source.run("run-1").facts)
    assert [f.node_id for f in facts if isinstance(f, NodeFact)] == [
        "siteconf:T2_XX_Nowhere"
    ]
    assert [f for f in facts if isinstance(f, EdgeFact)] == []
    result = source.preflight()
    assert result.status == "ok"
    assert result.mode == "fixture"
    assert result.record_count == 1


# --- circleback-fixes regressions: live-crawl failure modes ---


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _ok_preflight(self, mode="live"):
    return PreflightResult(
        source_name="siteconf", status="ok", mode="live"
    )


def test_empty_project_list_fails_loud_instead_of_empty_scope(
    tmp_path, monkeypatch
):
    # HTTP-200 empty project list (token without group visibility) used
    # to become ok/completed_scope over 0 records -> retract-all.
    monkeypatch.setattr(SITECONFSource, "preflight", _ok_preflight)
    monkeypatch.setattr(
        SITECONFSource,
        "_list_projects",
        lambda self, session, headers: ([], False),
    )
    source = _source(tmp_path, None)
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "empty project list" in run.health.reason


def test_zero_parsed_records_from_projects_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(SITECONFSource, "preflight", _ok_preflight)
    monkeypatch.setattr(
        SITECONFSource,
        "_list_projects",
        lambda self, session, headers: (
            [{"id": 7, "path": "not-a-site"}], False
        ),
    )
    source = _source(tmp_path, None)
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "zero SITECONF records" in run.health.reason


def test_preflight_group_probe_flags_invisible_group(monkeypatch):
    import requests

    monkeypatch.setenv("CERN_GITLAB_TOKEN", "tok")

    def fake_get(url, **kwargs):
        if url.endswith("/api/v4/user"):
            return _Resp(200, {"username": "svc"})
        return _Resp(200, [])  # group visible check: empty list

    monkeypatch.setattr(requests, "get", fake_get)
    result = SITECONFSource().preflight()
    assert result.status == "endpoint_failed"
    assert "group" in result.reason


def test_preflight_group_probe_includes_subgroups_like_the_crawl(
    monkeypatch,
):
    # A group whose projects live only in subgroups must not get a
    # false-negative preflight: the probe must send include_subgroups
    # exactly like the crawl's listing does.
    import requests

    monkeypatch.setenv("CERN_GITLAB_TOKEN", "tok")

    def fake_get(url, params=None, **kwargs):
        if url.endswith("/api/v4/user"):
            return _Resp(200, {"username": "svc"})
        if (params or {}).get("include_subgroups") == "true":
            return _Resp(200, [{"id": 11, "path": "T2_US_MIT"}])
        return _Resp(200, [])  # subgroup-only project is invisible

    monkeypatch.setattr(requests, "get", fake_get)
    result = SITECONFSource().preflight()
    assert result.status == "ok"


def test_max_projects_truncation_never_claims_scope(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(SITECONFSource, "preflight", _ok_preflight)

    class _FakeSession:
        def get(self, url, params=None, headers=None, timeout=None):
            if params and params.get("page") == 1:
                return _Resp(200, [
                    {"id": 11, "path": "T2_US_MIT"},
                    {"id": 12, "path": "T2_US_XYZ"},
                ])
            return _Resp(200, [])

    monkeypatch.setattr(requests, "Session", _FakeSession)
    monkeypatch.setattr(
        SITECONFSource,
        "_fetch_site_config",
        lambda self, session, **kwargs: SiteConfRecord(
            site_name=kwargs["site_name"], raw_storage_json="{}"
        ),
    )
    source = SITECONFSource(
        records=None, base=str(tmp_path), max_projects=1
    )
    (tmp_path / "data" / "cric").mkdir(parents=True)
    (tmp_path / "data" / "cric" / "sites.json").write_text(
        json.dumps({"T2_US_MIT": {}})
    )
    run = source.run("run-1", mode="scope_complete")
    nodes = [f for f in run.facts if isinstance(f, NodeFact)]
    assert [n.node_id for n in nodes] == ["siteconf:T2_US_MIT"]
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "max_projects" in run.health.reason
