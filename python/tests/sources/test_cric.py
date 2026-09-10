"""req.w2.sources-catalogs — CRICSource / CRICCoreSource emission, offline."""
import json

import pytest

from okg.deployment import EdgeFact, NodeFact

from archi.sources.cric import CRICCoreSource, CRICSource

SITES = {
    "T2_US_MIT": {
        "tier_level": 2,
        "country": "United States",
        "country_code": "US",
        "facility": "MIT",
        "status": "prod",
        "state": "ACTIVE",
        "sitedb_title": "MIT Bates",
        "computeunits": {"MIT-CE1": {}},
    },
}
FACILITIES = {
    "MIT": {
        "country": "United States",
        "timezone": "America/New_York",
        "fullname": "Massachusetts Institute of Technology",
        "state": "ACTIVE",
        "cmssites": [{"name": "T2_US_MIT"}, {"name": "T2_XX_Unknown"}],
    },
}
STORAGE_UNITS = {
    "MIT-SE": {
        "type": "DISK",
        "pledged-CMS": 4200.0,
        "state": "ACTIVE",
        "site": {"name": "T2_US_MIT"},
    },
}
COMPUTE_UNITS = {
    "MIT-CE1": {
        "corepower": 11.0,
        "pledged_cms": 10000.0,
        "potential_max": 20000.0,
        "promised": 10000.0,
        "state": "ACTIVE",
    },
}
RESPONSIBILITIES = {
    "result": [
        ["adalove", "MIT Bates", "Site Executive"],
        ["adalove", "MIT Bates", "Site Executive"],  # dup -> one edge
        ["ghopper", "Unknown Title", "Site Admin"],  # unmapped -> dropped
    ]
}


def _write_cric(tmp_path):
    root = tmp_path / "data" / "cric"
    root.mkdir(parents=True)
    (root / "sites.json").write_text(json.dumps(SITES))
    (root / "facilities.json").write_text(json.dumps(FACILITIES))
    (root / "storage_units.json").write_text(json.dumps(STORAGE_UNITS))
    (root / "compute_units.json").write_text(json.dumps(COMPUTE_UNITS))
    (root / "responsibilities.json").write_text(json.dumps(RESPONSIBILITIES))
    return CRICSource(base=str(tmp_path))


def test_cric_topology_nodes_and_edges(tmp_path):
    source = _write_cric(tmp_path)
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    assert set(nodes) == {
        "facility:MIT",
        "site:T2_US_MIT",
        "se:MIT-SE",
        "ce:MIT-CE1",
        "op:adalove",
    }
    site = nodes["site:T2_US_MIT"]
    assert site.subtype == "site"
    assert site.attrs["tier_level"] == "T2"
    assert site.attrs["label"] == "MIT Bates"  # sitedb_title becomes label
    assert "sitedb_title" not in site.attrs
    assert nodes["facility:MIT"].subtype == "facility"
    assert nodes["se:MIT-SE"].subtype == "storage_endpoint"
    assert nodes["ce:MIT-CE1"].subtype == "compute_endpoint"
    assert nodes["op:adalove"].subtype == "operator"

    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    assert edges == {
        ("facility:MIT", "contains", "site:T2_US_MIT"),
        ("site:T2_US_MIT", "contains", "ce:MIT-CE1"),
        ("site:T2_US_MIT", "contains", "se:MIT-SE"),
        ("op:adalove", "responsible_for", "site:T2_US_MIT"),
    }
    resp = [
        e for e in facts
        if isinstance(e, EdgeFact) and e.edge_type == "responsible_for"
    ]
    assert len(resp) == 1  # dedup + title-mapped only
    assert resp[0].attrs == {"role": "Site Executive"}


def test_cric_preflight_missing_cache(tmp_path):
    source = CRICSource(base=str(tmp_path))
    result = source.preflight()
    assert result.status == "cache_missing"
    assert result.required is True


def test_drifted_responsibilities_payload_fails_loud(tmp_path):
    # circleback-fixes regression: an error-shaped/drifted payload used
    # to be read as "no responsibilities" under ok/completed_scope,
    # retracting every operator record.
    source = _write_cric(tmp_path)
    (tmp_path / "data" / "cric" / "responsibilities.json").write_text(
        json.dumps({"error": "auth required"})
    )
    with pytest.raises(ValueError, match="responsibilit"):
        source.run("run-1", mode="scope_complete")
    result = source.preflight()
    assert result.status == "endpoint_failed"
    assert "result" in result.reason


CORE_SERVICES = {
    "reqmgr2-cmsweb.cern.ch": {
        "type": "webservice",
        "flavour": "REQMGR",
        "endpoint": "cmsweb.cern.ch/reqmgr2",
        "is_monitored": True,
        "rcsite": "CERN-PROD",
    },
}
CORE_RCSITES = {
    "CERN-PROD": {
        "sites": [
            {"name": "T0_CH_CERN", "vo_name": "cms"},
            {"name": "ATLAS-SITE", "vo_name": "atlas"},
        ],
    },
}
CORE_FEDERATIONS = {
    "CH-CERN": {
        "accounting_name": "CH-CERN",
        "tier_level": 0,
        "country": "Switzerland",
        "infrastructure": "WLCG",
        "vos": ["cms", "atlas"],
        "rcsites": ["CERN-PROD"],
        "pledges": {
            "2026": {"Q1": {"cms": {"CPU": 423000, "Disk": 47000}}},
        },
    },
    "XX-ATLASONLY": {
        "vos": ["atlas"],
        "rcsites": [],
        "pledges": {},
    },
}


def test_cric_core_services_and_federations(tmp_path):
    root = tmp_path / "data" / "cric-core"
    root.mkdir(parents=True)
    (root / "services.json").write_text(json.dumps(CORE_SERVICES))
    (root / "rcsites.json").write_text(json.dumps(CORE_RCSITES))
    (root / "federations.json").write_text(json.dumps(CORE_FEDERATIONS))
    source = CRICCoreSource(base=str(tmp_path))
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    # non-cms federation is filtered out
    assert set(nodes) == {"svc:reqmgr2-cmsweb.cern.ch", "fed:CH-CERN"}
    svc = nodes["svc:reqmgr2-cmsweb.cern.ch"]
    assert svc.subtype == "infrastructure_service"
    assert svc.attrs["service_type"] == "webservice"
    assert svc.attrs["endpoint"] == "cmsweb.cern.ch/reqmgr2"
    fed = nodes["fed:CH-CERN"]
    assert fed.subtype == "federation"
    assert fed.attrs["pledge_cpu"] == 423000
    assert fed.attrs["pledge_year"] == "2026"
    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    # edges only reach cms-VO rcsite members
    assert edges == {
        ("site:T0_CH_CERN", "contains", "svc:reqmgr2-cmsweb.cern.ch"),
        ("site:T0_CH_CERN", "member_of", "fed:CH-CERN"),
    }
