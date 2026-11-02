"""req.w2.sources-catalogs — CondDBGlobalTagSource emission, offline."""
import json

import pytest
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
        # The cache lists the cms-bot integration-series row for 14.0, the
        # record a global tag's `release: CMSSW_14_0_X` names.
        (cmssw / "records.json").write_text(
            json.dumps(
                [
                    {"label": "CMSSW_14_0_1"},
                    {"label": "CMSSW_14_0_X", "type": "Development", "state": "IB"},
                ]
            )
        )
    return CondDBGlobalTagSource(base=str(tmp_path))


def test_tag_nodes_supersedes_and_release_dependency(tmp_path):
    source = _source(tmp_path, with_cmssw=True)
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    # CMSSW_14_0_X is a known target (the cmssw cache lists that row), so no
    # conddb-only release node is emitted. A series value names the record
    # with that label, the same record the GlobalTagReleaseLinker rule
    # (global_tag.release = cmssw_release.release) reaches; never the family.
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


def test_a_concrete_release_no_longer_implies_the_series_id(tmp_path):
    """`cmssw_release:CMSSW_14_0_X` is known only when the cache lists that row.

    The CMSSW source once minted a family node at that id from any concrete
    14.0 release. Families now live at `cmssw_release_family:*`, so a cache
    with only CMSSW_14_0_1 leaves the series id to a conddb placeholder.
    """
    source = _source(tmp_path, with_cmssw=False)
    cmssw = tmp_path / "data" / "cmssw-releases"
    cmssw.mkdir(parents=True)
    (cmssw / "records.json").write_text(json.dumps([{"label": "CMSSW_14_0_1"}]))
    facts = list(source.run("run-1").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    assert "cmssw_release:CMSSW_14_0_X" in nodes
    assert not any(n.startswith("cmssw_release_family:") for n in nodes)


def test_missing_cmssw_cache_emits_conddb_only_release_node(tmp_path):
    source = _source(tmp_path, with_cmssw=False)
    facts = list(source.run("run-1").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    series = nodes["cmssw_release:CMSSW_14_0_X"]
    assert series.subtype == "cmssw_release"
    assert series.attrs["state"] == "referenced_by_conddb"
    assert series.attrs["release"] == "CMSSW_14_0_X"
    assert series.attrs["release_kind"] == "integration-series"
    assert "release_type" not in series.attrs
    assert "family" not in series.attrs
    assert not any(n.startswith("cmssw_release_family:") for n in nodes)
    depends = {
        (e.src, e.dst)
        for e in facts
        if isinstance(e, EdgeFact) and e.edge_type == "depends_on"
    }
    assert ("global_tag:140X_dataRun3_v2", "cmssw_release:CMSSW_14_0_X") in depends


def test_conddb_only_concrete_release_is_not_called_a_family(tmp_path):
    """A concrete release CondDB names but the cache lacks keeps its label id.

    It once got `release_type: release_family`; release_type now holds only the
    CMSSW source's own value, which CondDB does not know.
    """
    source = _source(tmp_path, with_cmssw=False)
    (tmp_path / "data" / "conddb-global-tags" / "records.json").write_text(
        json.dumps([{"name": "141X_mcRun3_v1", "release": "CMSSW_14_1_0_pre2"}])
    )
    facts = list(source.run("run-1").facts)
    nodes = {f.node_id: f for f in facts if isinstance(f, NodeFact)}
    placeholder = nodes["cmssw_release:CMSSW_14_1_0_pre2"]
    assert placeholder.attrs["release_kind"] == "pre-release"
    assert placeholder.attrs["state"] == "referenced_by_conddb"
    assert "release_type" not in placeholder.attrs
    assert "family" not in placeholder.attrs


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


# --- reference-catalog-complete-scope regressions ---
# okg runs a reference_catalog source in release_new (chosen automatically)
# or, when a mode is set explicitly, release_unchanged. CondDB deliberately
# claims no complete scope there (no deletion by absence) and instead
# returns a checkpoint that names its input content -- the global-tag cache
# and the CMSSW cache it reads for release targets -- so okg accepts an
# unchanged rerun as a no-op. scope_complete and reconcile keep main's
# reader-level behaviour; okg does not currently run a reference_catalog
# source in either mode (it refuses them at run start).

_CURSOR_KEY = "input_content_sha256"


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_release_modes_claim_no_scope_and_return_a_checkpoint(tmp_path, mode):
    run = _source(tmp_path, with_cmssw=True).run("run-1", mode=mode)
    assert run.completed_scope is False
    assert run.run_mode == mode
    assert run.health.status == "ok"
    assert run.next_checkpoint is not None
    cursor = run.next_checkpoint.as_cursor()
    assert set(cursor) == {_CURSOR_KEY}
    assert len(cursor[_CURSOR_KEY]) == 64


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_release_checkpoint_follows_both_caches(tmp_path, mode):
    source = _source(tmp_path, with_cmssw=True)
    first = source.run("run-1", mode=mode).next_checkpoint.as_cursor()
    again = source.run("run-2", mode=mode).next_checkpoint.as_cursor()
    assert again == first
    (tmp_path / "data" / "cmssw-releases" / "records.json").write_text(
        json.dumps([{"label": "CMSSW_14_0_2"}])
    )
    cmssw_changed = source.run("run-3", mode=mode).next_checkpoint.as_cursor()
    assert cmssw_changed != first
    (tmp_path / "data" / "conddb-global-tags" / "records.json").write_text(
        json.dumps(RECORDS[:1])
    )
    tags_changed = source.run("run-4", mode=mode).next_checkpoint.as_cursor()
    assert tags_changed != cmssw_changed


@pytest.mark.parametrize("mode", ["release_new", "release_unchanged"])
def test_reference_catalog_modes_keep_the_guards(tmp_path, mode):
    source = _source(tmp_path, with_cmssw=True)
    cache = tmp_path / "data" / "conddb-global-tags" / "records.json"
    cache.write_text(json.dumps(RECORDS + ["junk"]))
    assert source.run("run-1", mode=mode).completed_scope is False
    cache.write_text(json.dumps([]))
    empty = source.run("run-2", mode=mode)
    assert empty.completed_scope is False
    assert empty.health.status == "cache_missing"


@pytest.mark.parametrize("mode", ["scope_complete", "reconcile"])
def test_whole_scope_modes_still_claim_without_a_checkpoint(tmp_path, mode):
    run = _source(tmp_path, with_cmssw=True).run("run-1", mode=mode)
    assert run.completed_scope is True
    assert run.next_checkpoint is None


def test_cursor_mode_still_claims_no_scope(tmp_path):
    run = _source(tmp_path, with_cmssw=True).run("run-1", mode="cursor")
    assert run.completed_scope is False
    assert run.next_checkpoint is None
