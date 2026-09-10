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
    empty = _source(tmp_path, records=[])
    assert empty.preflight().status == "skipped_optional"


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
