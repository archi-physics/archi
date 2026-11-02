"""req.w2.sources-catalogs — DBSDatasetSource emission, offline."""
import json

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.sources.dbs import DBSDatasetSource

# DBS REST key spelling on purpose (dataset / *_ds_name aliases).
RECORDS = [
    {
        "dataset": "/TTto2L2Nu/Run3Summer23-v1/AODSIM",
        "data_tier_name": "AODSIM",
        "primary_ds_name": "TTto2L2Nu",
        "processed_ds_name": "Run3Summer23-v1",
        "physics_group_name": "TOP",
        "creation_date": "1700000000",
        "dataset_access_type": "VALID",
        "dataset_size": 123456789,
        "nfiles": 42,
        "nevents": 1000000,
    },
    {
        "dataset": "/TTto2L2Nu/Run3Summer23-v1/MINIAODSIM",
        "data_tier_name": "MINIAODSIM",
        "primary_ds_name": "TTto2L2Nu",
        "processed_ds_name": "Run3Summer23-v1",
    },
]


def _source(tmp_path, records):
    root = tmp_path / "data" / "dbs-datasets"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(records))
    return DBSDatasetSource(base=str(tmp_path))


def test_dataset_nodes_and_tier_chain_edges(tmp_path):
    source = _source(tmp_path, RECORDS)
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    assert set(nodes) == {
        "dataset:/TTto2L2Nu/Run3Summer23-v1/AODSIM",
        "dataset:/TTto2L2Nu/Run3Summer23-v1/MINIAODSIM",
    }
    aod = nodes["dataset:/TTto2L2Nu/Run3Summer23-v1/AODSIM"]
    assert aod.subtype == "dataset"
    assert aod.attrs["tier"] == "AODSIM"
    assert aod.attrs["era"] == "Run3Summer23"
    assert aod.attrs["primary_dataset"] == "TTto2L2Nu"
    assert aod.attrs["total_files"] == 42
    edges = [f for f in facts if isinstance(f, EdgeFact)]
    assert len(edges) == 1
    edge = edges[0]
    # MINIAODSIM (later tier) derives_from AODSIM
    assert edge.edge_type == "derives_from"
    assert edge.src == "dataset:/TTto2L2Nu/Run3Summer23-v1/MINIAODSIM"
    assert edge.dst == "dataset:/TTto2L2Nu/Run3Summer23-v1/AODSIM"
    assert edge.provenance == "derived_deterministic"
    assert edge.attrs["relationship"] == "tier_chain"


def test_preflight_optional_cache(tmp_path):
    missing = DBSDatasetSource(base=str(tmp_path))
    assert missing.preflight().status == "cache_missing"
    assert missing.preflight().required is False
    # An empty cache is a failed fetch, not an optional skip: okg treats
    # skipped_optional as healthy enough to retract a complete scope.
    empty = _source(tmp_path, [])
    assert empty.preflight().status == "cache_missing"


# --- circleback-fixes regressions ---


def test_skipped_cache_items_never_claim_scope(tmp_path):
    source = _source(tmp_path, RECORDS + ["junk", {"data_tier_name": "RAW"}])
    run = source.run("run-1", mode="scope_complete")
    nodes = {f.node_id for f in run.facts if isinstance(f, NodeFact)}
    # survivors still emitted, but drift forfeits the scope claim
    assert "dataset:/TTto2L2Nu/Run3Summer23-v1/AODSIM" in nodes
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 2" in run.health.reason


def test_all_items_unparseable_is_endpoint_failed(tmp_path):
    source = _source(tmp_path, ["junk", 42])
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


# --- reference-catalog-complete-scope regressions ---
# okg runs a reference_catalog source in release_new (chosen automatically)
# or, when a mode is set explicitly, release_unchanged. DBS deliberately
# claims no complete scope there (no deletion by absence) and instead
# returns a checkpoint that names its input content, so okg accepts an
# unchanged rerun as a no-op. scope_complete and reconcile keep main's
# reader-level behaviour; okg does not currently run a reference_catalog
# source in either mode (it refuses them at run start).

_CURSOR_KEY = "input_content_sha256"


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_release_modes_claim_no_scope_and_return_a_checkpoint(tmp_path, mode):
    run = _source(tmp_path, RECORDS).run("run-1", mode=mode)
    assert run.completed_scope is False
    assert run.run_mode == mode
    assert run.health.status == "ok"
    assert run.next_checkpoint is not None
    cursor = run.next_checkpoint.as_cursor()
    assert set(cursor) == {_CURSOR_KEY}
    assert len(cursor[_CURSOR_KEY]) == 64


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_release_checkpoint_follows_the_cache_content(tmp_path, mode):
    source = _source(tmp_path, RECORDS)
    cache = tmp_path / "data" / "dbs-datasets" / "records.json"
    first = source.run("run-1", mode=mode).next_checkpoint.as_cursor()
    again = source.run("run-2", mode=mode).next_checkpoint.as_cursor()
    assert again == first
    cache.write_text(json.dumps(RECORDS[:1]))
    changed = source.run("run-3", mode=mode).next_checkpoint.as_cursor()
    assert changed != first


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_reference_catalog_modes_keep_the_guards(tmp_path, mode):
    skipped = _source(tmp_path / "skipped", RECORDS + ["junk"])
    assert skipped.run("run-1", mode=mode).completed_scope is False
    empty = _source(tmp_path / "empty", []).run("run-2", mode=mode)
    assert empty.completed_scope is False
    assert empty.health.status == "cache_missing"


@pytest.mark.parametrize("mode", ["scope_complete", "reconcile"])
def test_whole_scope_modes_still_claim_without_a_checkpoint(tmp_path, mode):
    run = _source(tmp_path, RECORDS).run("run-1", mode=mode)
    assert run.completed_scope is True
    assert run.next_checkpoint is None


def test_cursor_mode_still_claims_no_scope(tmp_path):
    run = _source(tmp_path, RECORDS).run("run-1", mode="cursor")
    assert run.completed_scope is False
    assert run.next_checkpoint is None
