"""Offline verification for the W1 seam-proof source.

Run with an okg-bearing interpreter (okg is a host dependency, not an
archi dependency):

    <okg-venv>/bin/python -m pytest \
        pact/changes/w1-prove-the-seam/spike/test_seam_source.py -v

Deliberately not under tests/ — the v2 CI environment does not carry
okg, and this spike is change-scoped evidence, not product code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from archi.sources.cmssw import CMSSWReleaseSource, _parse_releases

SAMPLE = """\
architecture=slc7_amd64_gcc900;label=CMSSW_12_0_0;type=Production;state=Announced;prodarch=1;
architecture=slc7_amd64_gcc10;label=CMSSW_12_0_0;type=Production;state=Announced;prodarch=0;
architecture=el8_amd64_gcc11;label=CMSSW_13_0_1;type=Development;state=Announced;prodarch=1;
not-a-release-line
architecture=el8_amd64_gcc11;label=NOT_CMSSW;type=Production;state=Announced;
"""


def test_parse_groups_architectures_and_orders():
    records = _parse_releases(SAMPLE, limit=0)
    assert [r["release"] for r in records] == ["CMSSW_12_0_0", "CMSSW_13_0_1"]
    twelve = records[0]
    assert twelve["architecture"] == ["slc7_amd64_gcc10", "slc7_amd64_gcc900"]
    assert (twelve["major"], twelve["minor"], twelve["patch"]) == (12, 0, 0)
    assert records[1]["release_type"] == "Development"


def test_parse_limit_keeps_newest():
    records = _parse_releases(SAMPLE, limit=1)
    assert [r["release"] for r in records] == ["CMSSW_13_0_1"]


def test_adapter_emits_nodefacts_from_cache(tmp_path):
    cache = tmp_path / "releases.map"
    cache.write_text(SAMPLE, encoding="utf-8")
    src = CMSSWReleaseSource(cache_path=str(cache), fetch=False, limit=0)
    run = src.run("run-test-1", mode="scope_complete")
    facts = list(run.facts)
    assert run.completed_scope is True
    assert {f.node_id for f in facts} == {
        "cmssw_release:CMSSW_12_0_0",
        "cmssw_release:CMSSW_13_0_1",
    }
    fact = facts[0]
    assert fact.subtype == "cmssw_release"
    assert fact.op == "I"
    assert fact.source_record_id == {"release": fact.attrs["release"]}
    assert fact.source_revision["content_hash"]
    assert run.health.record_count == 2


def test_probe_declares_content_hash_kind():
    assert CMSSWReleaseSource.change_probe_kind == "content_hash"
