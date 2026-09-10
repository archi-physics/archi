"""req.w2.sources-catalogs — CMSSWReleaseSource emission, offline.

Covers both modes: the cms cache-backed records path (canonical) and
the W1 releases.map option, which now flows through the same emission
(family nodes + supersedes edges).
"""
import json

from okg.deployment import EdgeFact, NodeFact

from archi.sources.cmssw import CMSSWReleaseSource, parse_releases_map

RECORDS = [
    {
        "label": "CMSSW_14_0_1",
        "type": "Production",
        "state": "Announced",
        "architecture": ["el8_amd64_gcc12"],
        "release_date": "2024-03-01",
    },
    {
        "label": "CMSSW_14_0_2",
        "type": "Production",
        "state": "Announced",
        "architecture": ["el8_amd64_gcc12", "el9_amd64_gcc12"],
    },
    {
        "label": "CMSSW_14_0_2_patch1",
        "type": "Production",
        "state": "Announced",
        "architecture": "el8_amd64_gcc12",
    },
]

RELEASES_MAP = "\n".join([
    "architecture=el8_amd64_gcc12;label=CMSSW_14_0_1;type=Production;state=Announced;prodarch=1;",
    "architecture=el9_amd64_gcc12;label=CMSSW_14_0_1;type=Production;state=Announced;prodarch=0;",
    "architecture=el8_amd64_gcc12;label=CMSSW_14_0_2;type=Production;state=Announced;prodarch=1;",
    "architecture=el8_amd64_gcc12;label=NOT_A_RELEASE;type=Production;state=Announced;",
    "",
])

# Same map without the NOT_A_RELEASE junk line: a fully parseable map,
# for tests that assert an intact completed-scope claim.
CLEAN_RELEASES_MAP = "\n".join(
    line for line in RELEASES_MAP.splitlines() if "NOT_A_RELEASE" not in line
)


def _nodes(facts):
    return {f.node_id: f for f in facts if isinstance(f, NodeFact)}


def _supersedes(facts):
    return {
        (e.src, e.dst)
        for e in facts
        if isinstance(e, EdgeFact) and e.edge_type == "supersedes"
    }


def test_cache_backed_records_families_and_supersedes(tmp_path):
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(RECORDS))
    source = CMSSWReleaseSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    nodes = _nodes(facts)
    assert set(nodes) == {
        "cmssw_release:CMSSW_14_0_X",
        "cmssw_release:CMSSW_14_0_1",
        "cmssw_release:CMSSW_14_0_2",
        "cmssw_release:CMSSW_14_0_2_patch1",
    }
    assert all(n.subtype == "cmssw_release" for n in nodes.values())
    family = nodes["cmssw_release:CMSSW_14_0_X"]
    assert family.attrs["release_type"] == "release_family"
    assert family.attrs["family"] is True
    rel = nodes["cmssw_release:CMSSW_14_0_2"]
    assert rel.attrs["major"] == 14 and rel.attrs["patch"] == 2
    assert rel.attrs["architecture"] == "el8_amd64_gcc12,el9_amd64_gcc12"
    patch = nodes["cmssw_release:CMSSW_14_0_2_patch1"]
    assert patch.attrs["release_type"] == "patch"
    assert _supersedes(facts) == {
        ("cmssw_release:CMSSW_14_0_2", "cmssw_release:CMSSW_14_0_1"),
        ("cmssw_release:CMSSW_14_0_2_patch1", "cmssw_release:CMSSW_14_0_2"),
    }
    assert run.completed_scope is True
    assert run.health.record_count == 3


def test_releases_map_mode_same_emission_shape(tmp_path):
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "releases.map").write_text(RELEASES_MAP)
    source = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        base=str(tmp_path),
    )
    facts = list(source.run("run-1", mode="scope_complete").facts)
    nodes = _nodes(facts)
    assert set(nodes) == {
        "cmssw_release:CMSSW_14_0_X",
        "cmssw_release:CMSSW_14_0_1",
        "cmssw_release:CMSSW_14_0_2",
    }
    # architectures aggregated across duplicate map lines, sorted
    rel1 = nodes["cmssw_release:CMSSW_14_0_1"]
    assert rel1.attrs["architecture"] == "el8_amd64_gcc12,el9_amd64_gcc12"
    assert _supersedes(facts) == {
        ("cmssw_release:CMSSW_14_0_2", "cmssw_release:CMSSW_14_0_1"),
    }


def test_parse_releases_map_limit():
    records = parse_releases_map(RELEASES_MAP, limit=1)
    assert [r.label for r in records] == ["CMSSW_14_0_2"]


# --- circleback-fixes regressions ---


def test_skipped_cache_items_never_claim_scope(tmp_path):
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "records.json").write_text(
        json.dumps(RECORDS + ["junk", {"type": "no label"}])
    )
    source = CMSSWReleaseSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    nodes = _nodes(run.facts)
    assert "cmssw_release:CMSSW_14_0_1" in nodes  # survivors still emitted
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 2" in run.health.reason


def test_all_items_unparseable_is_endpoint_failed(tmp_path):
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(["junk", 42]))
    source = CMSSWReleaseSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_limit_truncation_never_claims_scope(tmp_path):
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "releases.map").write_text(CLEAN_RELEASES_MAP)
    source = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        limit=1,
        base=str(tmp_path),
    )
    run = source.run("run-1", mode="scope_complete")
    assert run.completed_scope is False
    assert "limit=1" in run.health.reason
    # a cap that does not actually truncate keeps the scope claim
    untruncated = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        limit=10,
        base=str(tmp_path),
    )
    assert untruncated.run(
        "run-2", mode="scope_complete"
    ).completed_scope is True


def test_map_skipped_lines_never_claim_scope(tmp_path):
    # RELEASES_MAP carries one unparseable line (NOT_A_RELEASE): the
    # survivors are emitted, but the scope claim is forfeited.
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "releases.map").write_text(RELEASES_MAP)
    source = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        base=str(tmp_path),
    )
    run = source.run("run-1", mode="scope_complete")
    nodes = _nodes(run.facts)
    assert "cmssw_release:CMSSW_14_0_1" in nodes
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 1" in run.health.reason


def test_map_zero_parse_from_nonempty_map_is_endpoint_failed(tmp_path):
    # Reviewer repro: a key rename (label= -> tag=) filtered every line
    # uncounted, claiming ok + completed_scope=True over zero records.
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "releases.map").write_text(
        CLEAN_RELEASES_MAP.replace("label=", "tag=")
    )
    source = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        base=str(tmp_path),
    )
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
