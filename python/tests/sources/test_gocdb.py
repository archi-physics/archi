"""req.w2.sources-catalogs — GoCDBDowntimeSource emission, offline."""
import json

from okg.deployment import EdgeFact, NodeFact

from archi.sources.gocdb import GoCDBDowntimeSource

# Two rows with the same downtime_id: services/hostnames aggregate.
RECORDS = [
    {
        "downtime_id": 101,
        "primary_key": "101G0",
        "classification": "SCHEDULED",
        "severity": "OUTAGE",
        "description": "Cooling maintenance",
        "start_date": "2026-08-01T06:00:00",
        "end_date": "2026-08-01T18:00:00",
        "hosted_by": "T2_US_MIT",
        "service_type": "CE",
        "hostname": "ce01.cmsaf.mit.edu",
    },
    {
        "downtime_id": 101,
        "primary_key": "101G0",
        "classification": "SCHEDULED",
        "severity": "OUTAGE",
        "description": "Cooling maintenance",
        "start_date": "2026-08-01T06:00:00",
        "end_date": "2026-08-01T18:00:00",
        "hosted_by": "T2_US_MIT",
        "service_type": "SRM",
        "hostname": "cmsweb.cern.ch",
    },
]
SITES = {"T2_US_MIT": {}}
SERVICES = {
    "reqmgr2": {"endpoint": "cmsweb.cern.ch/reqmgr2"},
}


def _source(tmp_path):
    (tmp_path / "data" / "gocdb-downtimes").mkdir(parents=True)
    (tmp_path / "data" / "gocdb-downtimes" / "records.json").write_text(
        json.dumps(RECORDS)
    )
    (tmp_path / "data" / "cric").mkdir(parents=True)
    (tmp_path / "data" / "cric" / "sites.json").write_text(json.dumps(SITES))
    (tmp_path / "data" / "cric-core").mkdir(parents=True)
    (tmp_path / "data" / "cric-core" / "services.json").write_text(
        json.dumps(SERVICES)
    )
    return GoCDBDowntimeSource(base=str(tmp_path))


def test_downtime_dedup_and_affects_edges(tmp_path):
    source = _source(tmp_path)
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = [f for f in facts if isinstance(f, NodeFact)]
    assert len(nodes) == 1  # deduped by downtime_id
    node = nodes[0]
    assert node.node_id == "downtime:101"
    assert node.subtype == "downtime"
    assert node.attrs["severity"] == "outage"
    assert node.attrs["classification"] == "scheduled"
    assert node.attrs["affected_services"] == ["CE", "SRM"]
    assert node.attrs["affected_hostnames"] == [
        "ce01.cmsaf.mit.edu",
        "cmsweb.cern.ch",
    ]
    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    # site edge (hosted_by in known sites) + service edge via endpoint
    # host lookup; the unknown hostname produces no edge.
    assert edges == {
        ("downtime:101", "affects", "site:T2_US_MIT"),
        ("downtime:101", "affects", "svc:reqmgr2"),
    }


def test_preflight_reports_missing_records_cache(tmp_path):
    source = GoCDBDowntimeSource(base=str(tmp_path))
    result = source.preflight()
    assert result.status == "cache_missing"
    assert result.required is True


# --- circleback-fixes regressions ---


def test_skipped_cache_items_never_claim_scope(tmp_path):
    source = _source(tmp_path)
    (tmp_path / "data" / "gocdb-downtimes" / "records.json").write_text(
        json.dumps(RECORDS + ["junk", {"downtime_id": 0}])
    )
    run = source.run("run-1", mode="scope_complete")
    nodes = {f.node_id for f in run.facts if isinstance(f, NodeFact)}
    assert "downtime:101" in nodes  # survivors still emitted
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 2" in run.health.reason


def test_all_items_unparseable_is_endpoint_failed(tmp_path):
    source = _source(tmp_path)
    (tmp_path / "data" / "gocdb-downtimes" / "records.json").write_text(
        json.dumps(["junk", {"downtime_id": "not-a-number"}])
    )
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_scheme_prefixed_endpoint_maps_hostname_to_service(tmp_path):
    # circleback-fixes regression: "https://h:8443/p".split("/", 1)[0]
    # yielded "https:", silently breaking hostname -> service matching.
    source = _source(tmp_path)
    (tmp_path / "data" / "cric-core" / "services.json").write_text(
        json.dumps(
            {"reqmgr2": {"endpoint": "https://cmsweb.cern.ch:8443/reqmgr2"}}
        )
    )
    facts = list(source.run("run-1", mode="scope_complete").facts)
    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    assert ("downtime:101", "affects", "svc:reqmgr2") in edges
