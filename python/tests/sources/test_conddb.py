"""req.w2.sources-catalogs — CondDBGlobalTagSource emission, offline."""
import json

from okg.deployment import EdgeFact, NodeFact

from archi.sources.conddb import CondDBGlobalTagSource

RECORDS = [
    {
        "name": "140X_dataRun3_v2",
        "release": "CMSSW_14_0_X",
        "scenario": "data",
        "description": "Run3 data GT",
        "snapshot_time": "2024-05-01",
    },
    {"name": "140X_dataRun3_v1", "release": "CMSSW_14_0_X"},
]


def _source(tmp_path, *, with_cmssw):
    root = tmp_path / "data" / "conddb-global-tags"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(RECORDS))
    if with_cmssw:
        cmssw = tmp_path / "data" / "cmssw-releases"
        cmssw.mkdir(parents=True)
        (cmssw / "records.json").write_text(
            json.dumps([{"label": "CMSSW_14_0_1"}])
        )
    return CondDBGlobalTagSource(base=str(tmp_path))


def test_tag_nodes_supersedes_and_release_dependency(tmp_path):
    source = _source(tmp_path, with_cmssw=True)
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    # CMSSW_14_0_X family is a known target (from the cmssw cache), so
    # no conddb-only release-family node is emitted.
    assert set(nodes) == {
        "global_tag:140X_dataRun3_v2",
        "global_tag:140X_dataRun3_v1",
    }
    tag = nodes["global_tag:140X_dataRun3_v2"]
    assert tag.subtype == "global_tag"
    assert tag.attrs["scenario"] == "data"
    assert tag.attrs["created_at"] == "2024-05-01"  # snapshot_time alias
    edges = {
        (e.src, e.edge_type, e.dst)
        for e in facts
        if isinstance(e, EdgeFact)
    }
    assert edges == {
        (
            "global_tag:140X_dataRun3_v2",
            "supersedes",
            "global_tag:140X_dataRun3_v1",
        ),
        (
            "global_tag:140X_dataRun3_v2",
            "depends_on",
            "cmssw_release:CMSSW_14_0_X",
        ),
        (
            "global_tag:140X_dataRun3_v1",
            "depends_on",
            "cmssw_release:CMSSW_14_0_X",
        ),
    }


def test_missing_cmssw_cache_emits_conddb_only_family_node(tmp_path):
    source = _source(tmp_path, with_cmssw=False)
    facts = list(source.run("run-1").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    family = nodes["cmssw_release:CMSSW_14_0_X"]
    assert family.subtype == "cmssw_release"
    assert family.attrs["state"] == "referenced_by_conddb"
    assert family.attrs["family"] is True
    depends = {
        (e.src, e.dst)
        for e in facts
        if isinstance(e, EdgeFact) and e.edge_type == "depends_on"
    }
    assert ("global_tag:140X_dataRun3_v2", "cmssw_release:CMSSW_14_0_X") in depends


# --- circleback-fixes regressions ---


def test_skipped_cache_items_never_claim_scope(tmp_path):
    source = _source(tmp_path, with_cmssw=True)
    (tmp_path / "data" / "conddb-global-tags" / "records.json").write_text(
        json.dumps(RECORDS + ["junk", {"scenario": "no name"}])
    )
    run = source.run("run-1", mode="scope_complete")
    nodes = {f.node_id for f in run.facts if isinstance(f, NodeFact)}
    assert "global_tag:140X_dataRun3_v2" in nodes  # survivors still emitted
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 2" in run.health.reason


def test_all_items_unparseable_is_endpoint_failed(tmp_path):
    source = _source(tmp_path, with_cmssw=True)
    (tmp_path / "data" / "conddb-global-tags" / "records.json").write_text(
        json.dumps(["junk"])
    )
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_change_probe_covers_cmssw_cache(tmp_path):
    # circleback-fixes regression: run() reads the cmssw cache for
    # release-target gating, so a cmssw-cache update must fire the probe.
    source = _source(tmp_path, with_cmssw=True)
    before = source.change_probe.build_token()
    (tmp_path / "data" / "cmssw-releases" / "records.json").write_text(
        json.dumps([{"label": "CMSSW_14_0_2"}])
    )
    after = source.change_probe.build_token()
    assert before != after
