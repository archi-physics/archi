"""req.w2.sources-catalogs — CMSSWReleaseSource emission, offline.

Covers both modes: the cms cache-backed records path (canonical) and
the W1 releases.map option, which now flows through the same emission
(family nodes + supersedes edges).
"""

import hashlib
import io
import json
from pathlib import Path

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.auth.cache import content_hash
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

RELEASES_MAP = "\n".join(
    [
        "architecture=el8_amd64_gcc12;label=CMSSW_14_0_1;type=Production;state=Announced;prodarch=1;",
        "architecture=el9_amd64_gcc12;label=CMSSW_14_0_1;type=Production;state=Announced;prodarch=0;",
        "architecture=el8_amd64_gcc12;label=CMSSW_14_0_2;type=Production;state=Announced;prodarch=1;",
        "architecture=el8_amd64_gcc12;label=CMSSW_BAD;type=Production;state=Announced;",
        "",
    ]
)

# Same map without the malformed CMSSW label: a fully parseable map,
# for tests that assert an intact completed-scope claim.
CLEAN_RELEASES_MAP = "\n".join(
    line for line in RELEASES_MAP.splitlines() if "CMSSW_BAD" not in line
)


def test_map_ignores_declared_family_rows_and_non_cmssw_products(tmp_path):
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "releases.map").write_text(
        CLEAN_RELEASES_MAP
        + "\narchitecture=el9;label=CMSSW_20_1_X;type=Development;"
        + "\narchitecture=slc6;label=CMSSW_6_2_X_SLHC;type=Development;"
        + "\narchitecture=slc3;label=ECALTBH4_0_2_2;type=Development;"
    )
    source = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        base=str(tmp_path),
    )
    run = source.run("run-1", mode="scope_complete")
    assert run.completed_scope is True


def _run_map(tmp_path, body):
    root = tmp_path / "data" / "cmssw-releases"
    root.mkdir(parents=True)
    (root / "releases.map").write_text(body)
    source = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        base=str(tmp_path),
    )
    return source, source.run("run-1", mode="scope_complete")


def test_family_rows_keep_the_scope_claim_but_malformed_rows_still_break_it(
    tmp_path,
):
    """The live map's family rows must not cost the run its scope claim.

    The real cms-bot map carries 63 labels this parser could not read as
    releases: 60 integration-build family rows (`CMSSW_20_1_X` and friends,
    including `CMSSW_9_4_AN_X` and `CMSSW_6_2_X_SLHC`) and 3 rows for another
    product entirely (`ECALTBH4_*`). Counting them as unparseable made
    `completed_scope` False on every run, which made the second ingest of any
    cern-team deployment a zero-fact `partial` that admission rejects -- and
    because cmssw_releases is required_for_baseline, that blocked the publish
    and left the deployment in repair_required. A deployment could be ingested
    exactly once.

    The claim has to stay honest in the other direction, so the second half
    asserts a genuinely unreadable CMSSW row still degrades the scope.
    """
    families = (
        "\narchitecture=el9;label=CMSSW_20_1_X;type=Development;"
        "\narchitecture=slc6;label=CMSSW_9_4_AN_X;type=Development;"
        "\narchitecture=slc6;label=CMSSW_6_2_X_SLHC;type=Development;"
        "\narchitecture=slc3;label=ECALTBH4_0_2_2;type=Development;"
    )
    _source, run = _run_map(tmp_path, CLEAN_RELEASES_MAP + families)
    facts = list(run.facts)
    assert run.completed_scope is True
    assert "unparseable" not in (run.health.reason or "")
    # Dropped, not turned into releases: the family nodes this connector emits
    # are derived from the concrete releases, so a family ROW would duplicate
    # one. CMSSW_20_1_X has no concrete release here and must not appear.
    labels = {
        n.attrs["label"]
        for n in facts
        if isinstance(n, NodeFact) and n.subtype == "cmssw_release"
    }
    assert labels == {"CMSSW_14_0_1", "CMSSW_14_0_2", "CMSSW_14_0_X"}

    _source2, degraded = _run_map(
        tmp_path / "second",
        CLEAN_RELEASES_MAP
        + families
        + "\narchitecture=el9;label=CMSSW_NOT_A_VERSION;type=Production;",
    )
    list(degraded.facts)
    assert degraded.completed_scope is False
    assert "unparseable" in (degraded.health.reason or "")


def test_a_wholesale_label_rename_degrades_the_scope(tmp_path):
    """Schema drift must never look like "the catalog is empty, and I'm sure".

    `deletion_semantics: missing_from_completed_scope` turns a completed scope
    into a retraction order for everything absent from it. So a map whose only
    change is a renamed label prefix -- every row `CMSSW14_0_1` instead of
    `CMSSW_14_0_1` -- must not parse to an empty record set under a healthy
    claim, or the next publish would retract every release in the graph and
    report success.

    Two guards are checked, because either alone would miss a variant of this:
    an unrecognised label counts as a skip, AND an empty record set never
    claims a complete scope.
    """
    renamed = CLEAN_RELEASES_MAP.replace("label=CMSSW_", "label=CMSSW")
    _source, run = _run_map(tmp_path, renamed)
    assert list(run.facts) == []
    assert run.completed_scope is False

    # The second guard on its own: nothing to skip, still no claim.
    _source2, empty = _run_map(tmp_path / "empty", "")
    assert list(empty.facts) == []
    assert empty.completed_scope is False
    assert "no complete scope claimed" in (empty.health.reason or "")


def test_only_known_other_products_are_waived(tmp_path):
    """The out-of-scope product list is closed, so a new one gets noticed.

    The three `ECALTBH4_*` rows cms-bot ships are waived by name -- they are
    not CMSSW releases and never were. A product nobody has looked at yet is
    an unknown, so it degrades the scope until someone adds it deliberately.
    Waiving "anything that is not CMSSW_" is what let a wholesale rename past.
    """
    known = (
        "\narchitecture=slc3;label=ECALTBH4_0_2_0_pre2;type=Development;"
        "\narchitecture=slc3;label=ECALTBH4_0_2_2;type=Development;"
        "\narchitecture=slc3;label=ECALTBH4_0_4_0;type=Development;"
    )
    _source, run = _run_map(tmp_path, CLEAN_RELEASES_MAP + known)
    list(run.facts)
    assert run.completed_scope is True
    assert "unparseable" not in (run.health.reason or "")

    _source2, unknown = _run_map(
        tmp_path / "unknown",
        CLEAN_RELEASES_MAP
        + "\narchitecture=el9;label=NEWPRODUCT_1_0;type=Production;",
    )
    list(unknown.facts)
    assert unknown.completed_scope is False
    assert "unparseable" in (unknown.health.reason or "")


@pytest.mark.parametrize(
    "label",
    ["CMSSW_14_0_1TYPO", "CMSSW_0_0_0junk", "CMSSW_14_0_1 ", "CMSSW_14_0_1x"],
)
def test_garbage_glued_to_a_valid_triple_is_not_a_release(tmp_path, label):
    """A separator-less suffix is restricted, so a typo cannot become a node.

    Unrestricted, each of these parsed as a release of `CMSSW_14_0_1` (or
    `CMSSW_0_0_0`) and got a `supersedes` edge pointing at the real release --
    a fabricated release in a catalog whose whole job is to be authoritative.
    They have to be skips.
    """
    _source, run = _run_map(
        tmp_path,
        CLEAN_RELEASES_MAP + f"\narchitecture=el9;label={label};type=Production;",
    )
    labels = {
        n.attrs["label"]
        for n in run.facts
        if isinstance(n, NodeFact) and n.subtype == "cmssw_release"
    }
    assert label not in labels
    assert run.completed_scope is False


def test_a_build_suffix_without_a_separator_is_a_release(tmp_path):
    """`CMSSW_1_4_3g483` is a real build in the live map, not junk.

    cms-bot carries it on two architectures beside `CMSSW_1_4_3` -- a Geant4
    8.3 variant. It is the one label of the 63 that is a release rather than a
    family or another product, and it parses only because the separator before
    the build suffix is optional.
    """
    _source, run = _run_map(
        tmp_path,
        CLEAN_RELEASES_MAP
        + "\narchitecture=slc3_ia32_gcc323;label=CMSSW_1_4_3;"
        "type=Production;state=Deprecated;prodarch=1;"
        "\narchitecture=slc3_ia32_gcc323;label=CMSSW_1_4_3g483;"
        "type=Development;state=Deprecated;prodarch=1;",
    )
    facts = list(run.facts)
    assert run.completed_scope is True
    variant = _nodes(facts)["cmssw_release:CMSSW_1_4_3g483"]
    assert variant.attrs["suffix"] == "g483"
    # It supersedes the release it is a variant of, not some other patch.
    assert (
        "cmssw_release:CMSSW_1_4_3g483",
        "cmssw_release:CMSSW_1_4_3",
    ) in _supersedes(facts)


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
    assert untruncated.run("run-2", mode="scope_complete").completed_scope is True


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
    (root / "releases.map").write_text(CLEAN_RELEASES_MAP.replace("label=", "tag="))
    source = CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        base=str(tmp_path),
    )
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


@pytest.fixture
def frozen_map(tmp_path, monkeypatch):
    def refuse_network(*args, **kwargs):
        pytest.fail("a frozen map must never fetch")

    monkeypatch.setattr("urllib.request.urlopen", refuse_network)
    path = tmp_path / "releases.map"
    body = CLEAN_RELEASES_MAP.encode()
    path.write_bytes(body)
    pin = "sha256:" + hashlib.sha256(body).hexdigest()
    return path, body, pin


def _frozen_source(path, pin, **kwargs):
    return CMSSWReleaseSource(
        map_cache_path=str(path), map_cache_digest=pin, fetch=False, **kwargs
    )


def test_frozen_map_verified_emission_and_revision(frozen_map):
    path, body, pin = frozen_map
    source = _frozen_source(path, pin)
    preflight = source.preflight()
    run = source.run("frozen", mode="scope_complete")
    facts = list(run.facts)
    assert set(_nodes(facts)) == {
        "cmssw_release:CMSSW_14_0_X",
        "cmssw_release:CMSSW_14_0_1",
        "cmssw_release:CMSSW_14_0_2",
    }
    assert _supersedes(facts) == {
        ("cmssw_release:CMSSW_14_0_2", "cmssw_release:CMSSW_14_0_1")
    }
    expected = content_hash((str(path),))
    assert preflight.status == "ok"
    assert preflight.content_hash == run.health.content_hash == expected
    assert all(f.source_revision["content_hash"] == expected for f in facts)
    assert all(f.source_revision["map_cache_digest"] == pin for f in facts)
    assert run.completed_scope is True


@pytest.mark.parametrize(
    "pin", ["", "abc", "sha256:" + "A" * 64, "sha256:" + "0" * 63, 1]
)
def test_frozen_map_rejects_malformed_pin(pin):
    with pytest.raises(ValueError, match="map_cache_digest"):
        CMSSWReleaseSource(map_cache_path="releases.map", map_cache_digest=pin)


@pytest.mark.parametrize(
    "kwargs", [{}, {"map_cache_path": "releases.map", "fetch": True}]
)
def test_frozen_map_requires_path_and_disallows_fetch(kwargs):
    with pytest.raises(ValueError, match="requires map_cache_path and fetch=False"):
        CMSSWReleaseSource(map_cache_digest="sha256:" + "0" * 64, **kwargs)


def test_frozen_map_missing_never_fetches(frozen_map):
    path, _, pin = frozen_map
    source = _frozen_source(path, pin)
    path.unlink()
    assert source.preflight().status == "cache_missing"
    with pytest.raises(FileNotFoundError):
        source.run("missing", mode="scope_complete")
    assert not path.exists()


def test_frozen_map_changed_after_preflight_refuses(frozen_map):
    path, _, pin = frozen_map
    source = _frozen_source(path, pin)
    assert source.preflight().status == "ok"
    path.write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        source.preflight()
    with pytest.raises(ValueError, match="digest mismatch"):
        source.run("changed", mode="scope_complete")


def test_frozen_map_revision_uses_verified_buffer(frozen_map, monkeypatch):
    path, body, pin = frozen_map
    expected = content_hash((str(path),))
    source = _frozen_source(path, pin)
    read_bytes = Path.read_bytes
    reads = []

    def replace_after_read(selected):
        data = read_bytes(selected)
        if selected == path:
            reads.append(data)
            path.write_text("changed after reading")
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    run = source.run("race", mode="scope_complete")
    facts = list(run.facts)
    assert reads == [body]
    assert len(_nodes(facts)) == 3
    assert run.health.content_hash == expected
    assert all(f.source_revision["content_hash"] == expected for f in facts)
    assert all(f.source_revision["map_cache_digest"] == pin for f in facts)


@pytest.mark.parametrize(
    "body,limit",
    [(RELEASES_MAP.encode(), 0), (CLEAN_RELEASES_MAP.encode(), 1), (b"tag=unknown", 0)],
)
def test_frozen_map_incomplete_input_never_claims_scope(frozen_map, body, limit):
    path, _, _ = frozen_map
    path.write_bytes(body)
    source = _frozen_source(
        path, "sha256:" + hashlib.sha256(body).hexdigest(), limit=limit
    )
    assert source.run("incomplete", mode="scope_complete").completed_scope is False


def test_frozen_map_invalid_utf8_refuses(frozen_map):
    path, _, _ = frozen_map
    path.write_bytes(b"\xff")
    source = _frozen_source(path, "sha256:" + hashlib.sha256(b"\xff").hexdigest())
    with pytest.raises(UnicodeDecodeError):
        source.run("invalid", mode="scope_complete")


def test_frozen_map_probe_binding_and_file_changes(frozen_map):
    path, _, pin = frozen_map
    source = _frozen_source(path, pin)
    token = source.change_probe.probe(cursor=None, sync_scope=None).token
    cursor = {"change_probe_token": token}
    assert not source.change_probe.probe(cursor=cursor, sync_scope=None).changed
    other = _frozen_source(path, "sha256:" + "0" * 64)
    assert other.change_probe.probe(cursor=cursor, sync_scope=None).changed
    with pytest.raises(ValueError, match="digest mismatch"):
        other.run("wrong-pin")
    path.write_text("changed")
    assert source.change_probe.probe(cursor=cursor, sync_scope=None).changed
    with pytest.raises(ValueError, match="digest mismatch"):
        source.run("changed")
    path.unlink()
    assert source.change_probe.probe(cursor=cursor, sync_scope=None).changed
    with pytest.raises(FileNotFoundError):
        source.run("missing")


def test_frozen_map_accepts_cache_path_alias(frozen_map):
    path, _, pin = frozen_map
    source = CMSSWReleaseSource(cache_path=str(path), map_cache_digest=pin)
    assert source.preflight().status == "ok"


def test_unpinned_live_map_still_fetches(tmp_path, monkeypatch):
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return io.BytesIO(CLEAN_RELEASES_MAP.encode())

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    source = CMSSWReleaseSource(map_cache_path=str(tmp_path / "live.map"), fetch=True)
    assert source.run("live", mode="scope_complete").completed_scope is True
    assert calls == [source.releases_map_url]
