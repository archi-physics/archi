"""req.w2.sources-catalogs — WMStatsWorkflowSource emission, offline."""
import json

from okg.deployment import EdgeFact, NodeFact

from archi.sources.wmstats import WMStatsWorkflowSource

# ReqMgr CamelCase key spelling on purpose (the parser accepts both).
RECORDS = [
    {
        "RequestName": "pdmvserv_task_TOP-Run3Summer23-00001",
        "RequestType": "TaskChain",
        "RequestStatus": "running-open",
        "Campaign": "Run3Summer23",
        "PrepID": "TOP-Run3Summer23-00001",
        "RequestPriority": 85000,
        "CMSSWVersion": "CMSSW_13_0_13",
        "InputDataset": "/TTto2L2Nu/Run3Summer23-v1/GEN-SIM",
        "OutputDatasets": ["/TTto2L2Nu/Run3Summer23-v1/AODSIM"],
        "RequestDate": "2026-01-15",
    },
]


def _source(tmp_path, records=RECORDS):
    root = tmp_path / "data" / "wmstats-workflows"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(records))
    return WMStatsWorkflowSource(base=str(tmp_path))


def test_workflow_dataset_and_release_edges(tmp_path):
    source = _source(tmp_path)
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    wf_id = "workflow:pdmvserv_task_TOP-Run3Summer23-00001"
    assert set(nodes) == {
        wf_id,
        "dataset:/TTto2L2Nu/Run3Summer23-v1/GEN-SIM",
        "dataset:/TTto2L2Nu/Run3Summer23-v1/AODSIM",
    }
    wf = nodes[wf_id]
    assert wf.subtype == "workflow"
    assert wf.attrs["request_type"] == "TaskChain"
    assert wf.attrs["status"] == "running-open"
    assert wf.attrs["priority"] == 85000
    assert wf.attrs["output_dataset"] == "/TTto2L2Nu/Run3Summer23-v1/AODSIM"
    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    assert edges == {
        (wf_id, "consumes", "dataset:/TTto2L2Nu/Run3Summer23-v1/GEN-SIM"),
        (wf_id, "produces", "dataset:/TTto2L2Nu/Run3Summer23-v1/AODSIM"),
        (wf_id, "depends_on", "cmssw_release:CMSSW_13_0_13"),
    }


def test_preflight_optional_and_empty_cache(tmp_path):
    missing = WMStatsWorkflowSource(base=str(tmp_path))
    assert missing.preflight().status == "cache_missing"
    assert missing.preflight().required is False
    # An empty cache is a failed fetch, not an optional skip: okg treats
    # skipped_optional as healthy enough to retract a complete scope.
    empty = _source(tmp_path, records=[])
    assert empty.preflight().status == "cache_missing"


# --- circleback-fixes regressions ---


def test_skipped_cache_items_never_claim_scope(tmp_path):
    source = _source(
        tmp_path, records=RECORDS + ["junk", {"Campaign": "no name"}]
    )
    run = source.run("run-1", mode="scope_complete")
    nodes = {f.node_id for f in run.facts if isinstance(f, NodeFact)}
    wf_id = "workflow:pdmvserv_task_TOP-Run3Summer23-00001"
    assert wf_id in nodes  # survivors still emitted
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 2" in run.health.reason


def test_all_items_unparseable_is_endpoint_failed(tmp_path):
    source = _source(tmp_path, records=["junk", 42])
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


# --- wmstats-collection-shape: the collector's status_payloads object ---
# Fake data in the real Aug 31 structure: one WMStats API answer per status,
# {"result": [{name: record}]}, several statuses empty, a RequestorDN on
# every record (the reader reads neither it nor Requestor).

DN = "/DC=org/DC=example/OU=Users/CN=fakeuser/CN=000000/CN=Fake User"
SECOND = {
    **RECORDS[0],
    "RequestName": "fakeuser_ReReco_Run2026A_FakePD_000002",
    "RequestType": "ReReco",
    "RequestStatus": "assigned",
    "InputDataset": "/FakePD/Run2026A-v1/RAW",
    "OutputDatasets": ["/FakePD/Run2026A-ReReco-v1/AOD"],
}


def _status_payloads(*records):
    by_status = {
        status: {"result": []}
        for status in ("new", "assignment-approved", "assigned", "running-open")
    }
    for record in records:
        fake = {**record, "Requestor": "fakeuser", "RequestorDN": DN}
        entry = {record["RequestName"]: fake}
        results = by_status[record["RequestStatus"]]["result"]
        if results:
            results[0].update(entry)
        else:
            results.append(entry)
    return {"cutoff_utc": "2026-01-01T00:00:00Z", "status_payloads": by_status}


def _authority(tmp_path, count):
    path = tmp_path / "data" / "wmstats-workflows" / "authority.json"
    path.write_text(json.dumps({"record_count": count, "complete": True}))


def _facts(run):
    return sorted(repr(f) for f in run.facts)


def _content(run):
    """Facts without their run revision (the cache bytes differ)."""
    out = set()
    for fact in run.facts:
        if isinstance(fact, NodeFact):
            out.add(("node", fact.node_id, json.dumps(fact.attrs, sort_keys=True)))
        else:
            out.add(("edge", fact.src, fact.edge_type, fact.dst))
    return out


def test_collection_shape_reads_like_the_flat_list(tmp_path):
    flat = _source(tmp_path / "flat", records=RECORDS + [SECOND])
    shaped = _source(tmp_path / "shaped", records=_status_payloads(*RECORDS, SECOND))
    run = shaped.run("run-1", mode="scope_complete")
    assert run.health.status == "ok"
    assert run.completed_scope is True
    assert run.health.record_count == 2
    facts = _content(run)
    assert facts == _content(flat.run("run-1", mode="scope_complete"))
    assert {f[1] for f in facts if f[0] == "node"} >= {
        f"workflow:{RECORDS[0]['RequestName']}",
        f"workflow:{SECOND['RequestName']}",
    }
    assert not any(DN in str(fact) or "fakeuser" == fact[-1] for fact in facts)
    assert shaped.preflight().status == "ok"


def test_collection_shape_with_no_workflow_is_cache_missing(tmp_path):
    for name, payload in (
        ("none", {"cutoff_utc": "2026-01-01T00:00:00Z", "status_payloads": {}}),
        ("all-empty", _status_payloads()),
    ):
        source = _source(tmp_path / name, records=payload)
        run = source.run("run-1", mode="scope_complete")
        assert list(run.facts) == []
        assert run.completed_scope is False
        assert run.health.status == "cache_missing"
        assert source.preflight().status == "cache_missing"


def test_collection_shape_that_drifted_is_endpoint_failed(tmp_path):
    for name, payload in (
        ("no-key", {"cutoff_utc": "2026-01-01T00:00:00Z"}),
        ("error-body", {"error": "backend unavailable"}),
        ("no-result", {"status_payloads": {"new": {"error": "timeout"}}}),
    ):
        source = _source(tmp_path / name, records=payload)
        run = source.run("run-1", mode="scope_complete")
        assert list(run.facts) == []
        assert run.completed_scope is False
        assert run.health.status == "endpoint_failed"
        assert source.preflight().status == "endpoint_failed"


def test_a_workflow_under_two_statuses_is_emitted_once_without_scope(tmp_path):
    payload = _status_payloads(*RECORDS)
    moved = {**RECORDS[0], "RequestStatus": "assigned"}
    payload["status_payloads"]["assigned"]["result"].append(
        {moved["RequestName"]: moved}
    )
    source = _source(tmp_path, records=payload)
    run = source.run("run-1", mode="scope_complete")
    workflows = [
        f for f in run.facts if isinstance(f, NodeFact) and f.subtype == "workflow"
    ]
    assert len(workflows) == 1
    # "assigned" comes before "running-open" in the file: first one wins.
    assert workflows[0].attrs["status"] == "assigned"
    assert run.completed_scope is False
    assert "skipped 1" in run.health.reason


def test_authority_record_count_must_match(tmp_path):
    good = _source(tmp_path / "good", records=_status_payloads(*RECORDS, SECOND))
    _authority(tmp_path / "good", 2)
    assert good.run("run-1", mode="scope_complete").completed_scope is True
    bad = _source(tmp_path / "bad", records=_status_payloads(*RECORDS, SECOND))
    _authority(tmp_path / "bad", 3)
    run = bad.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "holds 2 workflow records" in run.health.reason
    assert "counts 3" in run.health.reason
    assert bad.preflight().status == "endpoint_failed"


# --- catalog-downloaders: authority.json is one of the reader's cache paths ---
# The reader reads authority.json when it sits beside records.json, so it
# lists it in cache_paths then: the downloader's reader check accepts it, and
# the content hash and change probe see a change to it.


def test_cache_paths_list_authority_json_only_when_it_is_there(tmp_path):
    without = _source(tmp_path / "without", records=_status_payloads(*RECORDS))
    assert without.cache_paths == ("data/wmstats-workflows/records.json",)

    with_it = _source(tmp_path / "with", records=_status_payloads(*RECORDS))
    _authority(tmp_path / "with", 1)
    assert with_it.cache_paths == (
        "data/wmstats-workflows/records.json",
        "data/wmstats-workflows/authority.json",
    )


def test_an_absolute_records_path_lists_the_authority_beside_it(tmp_path):
    records = tmp_path / "cache" / "records.json"
    records.parent.mkdir()
    records.write_text(json.dumps(_status_payloads(*RECORDS)))
    (records.parent / "authority.json").write_text(json.dumps({"record_count": 1}))
    source = WMStatsWorkflowSource(records_path=str(records))
    assert source.cache_paths == (str(records), str(records.parent / "authority.json"))
    assert source.run("run-1", mode="scope_complete").completed_scope is True


def test_a_change_to_authority_json_alone_changes_the_content_hash(tmp_path):
    source = _source(tmp_path, records=_status_payloads(*RECORDS))
    _authority(tmp_path, 1)
    first = WMStatsWorkflowSource(base=str(tmp_path))
    before_run = first.run("run-1", mode="scope_complete")
    before_token = first.change_probe.build_token()

    path = tmp_path / "data" / "wmstats-workflows" / "authority.json"
    path.write_text(json.dumps({"record_count": 1, "complete": False}))
    after_run = source.run("run-2", mode="scope_complete")

    assert after_run.completed_scope is True
    assert after_run.health.content_hash != before_run.health.content_hash
    assert first.change_probe.build_token() != before_token


def test_a_cache_without_authority_json_reads_and_probes_as_before(tmp_path):
    source = _source(tmp_path, records=_status_payloads(*RECORDS, SECOND))
    run = source.run("run-1", mode="scope_complete")
    assert run.health.status == "ok"
    assert run.completed_scope is True
    assert run.health.record_count == 2
    assert source.preflight().status == "ok"
    assert source.change_probe.build_token()
