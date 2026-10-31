"""pact reference-targets-hep — dataset and conditions matchers.

The scanner turns text into edges only where the graph already has the
node. These tests hold both halves of that: a named target links, and a
named non-target does not. The false-positive tests matter most --
prose is full of slashes, and a dataset pattern that matches file paths
would attach analysis notes to whatever happened to share a shape.
"""
from __future__ import annotations

import json

import pytest

from archi.sources.docs import (
    _DATASET_RE,
    _GLOBAL_TAG_RE,
    _known_datasets,
    _known_global_tags,
    _reference_edges,
    _reference_targets,
)

DATASET = "/TTToSemiLeptonic_TuneCP5_13TeV/RunIISummer20UL18MiniAODv2/MINIAODSIM"
TAG = "106X_dataRun2_v32"


def _edges(text, targets):
    return list(_reference_edges("chunk:1", text, {"path": "p"}, {"run_id": "r"}, targets))


def _targets(**kwargs):
    base = {"site": set(), "release": set(), "jira": set(),
            "service": {}, "dataset": set(), "global_tag": set()}
    base.update(kwargs)
    return base


# --- the patterns ------------------------------------------------------

@pytest.mark.parametrize("text", [
    DATASET,
    f"Use {DATASET} for the signal sample.",
    "/SingleMuon/Run2018D-UL2018_MiniAODv2-v1/MINIAOD",
    "/DYJetsToLL_M-50/RunIISummer20UL17NanoAODv9-106X/NANOAODSIM",
])
def test_dataset_pattern_matches_real_dataset_paths(text):
    assert _DATASET_RE.findall(text)


@pytest.mark.parametrize("text", [
    "/afs/cern.ch/user/j/jdoe/work",          # a file path
    "https://twiki.cern.ch/twiki/bin/view/CMS",   # a URL
    "see signal/background/ratio for details",    # prose with slashes
    "/store/data/Run2018D/SingleMuon/MINIAOD/file.root",  # a FILE, not a dataset
    "the efficiency is 1/2/3 percent",
])
def test_dataset_pattern_ignores_slash_bearing_prose(text):
    """The tier alternation is the defence. Without it every path,
    URL and fraction becomes a candidate on every chunk of every
    document -- discarded later by the intersection, but only after
    the scan has paid for them."""
    assert _DATASET_RE.findall(text) == []


@pytest.mark.parametrize("text,expected", [
    ("106X_dataRun2_v32", ["106X_dataRun2_v32"]),
    ("124X_mcRun3_2022_realistic_v12", ["124X_mcRun3_2022_realistic_v12"]),
    ("Use 106X_dataRun2_v32 with the UL18 campaign.", ["106X_dataRun2_v32"]),
    ("CMSSW_10_6_30 is a release, not a tag", []),
    ("run 306X or X_marks", []),
])
def test_global_tag_pattern(text, expected):
    assert _GLOBAL_TAG_RE.findall(text) == expected


# --- edges emitted only for known targets ------------------------------

def test_known_dataset_links_and_unknown_does_not():
    text = f"Signal: {DATASET} and background: /Other/Campaign/MINIAODSIM"
    edges = _edges(text, _targets(dataset={DATASET}))
    assert [e.dst for e in edges] == [f"dataset:{DATASET}"]
    assert edges[0].edge_type == "references"
    assert edges[0].attrs["match_type"] == "dbs_dataset"


def test_known_global_tag_links_and_unknown_does_not():
    text = f"Conditions: {TAG}, previously 106X_dataRun2_v28."
    edges = _edges(text, _targets(global_tag={TAG}))
    assert [e.dst for e in edges] == [f"global_tag:{TAG}"]
    assert edges[0].attrs["match_type"] == "conditions_global_tag"


def test_nothing_is_emitted_without_targets():
    """A deployment that installs neither DBS nor CondDB sees no
    behaviour change at all."""
    assert _edges(f"{DATASET} and {TAG}", _targets()) == []


def test_both_matchers_can_fire_on_one_chunk():
    text = f"Run {DATASET} with conditions {TAG}."
    edges = _edges(text, _targets(dataset={DATASET}, global_tag={TAG}))
    assert {e.attrs["match_type"] for e in edges} == {
        "dbs_dataset", "conditions_global_tag"
    }


# --- target loading ----------------------------------------------------

def test_dataset_cache_accepts_both_shapes(tmp_path):
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([{"dataset": DATASET}, {"name": "/A/B/AOD"}]))
    assert _known_datasets(str(as_list)) == {DATASET, "/A/B/AOD"}

    as_map = tmp_path / "map.json"
    as_map.write_text(json.dumps({DATASET: {"files": 10}}))
    assert _known_datasets(str(as_map)) == {DATASET}


def test_global_tag_cache_accepts_both_shapes(tmp_path):
    as_list = tmp_path / "tags.json"
    as_list.write_text(json.dumps([{"name": TAG}, {"tag_name": "124X_mc_v1"}]))
    assert _known_global_tags(str(as_list)) == {TAG, "124X_mc_v1"}


def test_unconfigured_is_silent_but_configured_and_missing_raises(tmp_path):
    """The difference between "this deployment has no datasets" and
    "this deployment lost its dataset cache" has to stay visible."""
    assert _known_datasets(None) == set()
    assert _known_global_tags(None) == set()
    with pytest.raises(FileNotFoundError):
        _known_datasets(str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        _known_global_tags(str(tmp_path / "absent.json"))


def test_reference_targets_exposes_both_keys():
    targets = _reference_targets()
    assert targets["dataset"] == set()
    assert targets["global_tag"] == set()


def test_sources_accept_the_new_paths():
    """The knobs have to reach the sources that scan text, or the
    matchers are unreachable from a deployment manifest."""
    from archi.sources.docs import DocumentationSource, SSOCookieDocsSource
    from archi.sources.twiki import TwikiEOSSource

    twiki = TwikiEOSSource(eos_root="/tmp", datasets_path="d.json",
                           global_tags_path="t.json")
    assert twiki.datasets_path == "d.json"
    docs = DocumentationSource(datasets_path="d.json", global_tags_path="t.json")
    assert docs.global_tags_path == "t.json"
    sso = SSOCookieDocsSource(
        sitemap_url="https://example.org/sitemap.xml",
        cookie_file_env="X", datasets_path="d.json",
    )
    assert sso.datasets_path == "d.json"


def test_twiki_sources_accept_releases_map_path(tmp_path):
    """`releases_map_path` has to reach the TWiki sources too.

    `_reference_targets` and the docs sources have accepted it since the
    live CMSSW connector started writing cms-bot's `releases.map`
    instead of a `records.json` nobody produces. The TWiki sources did
    not, so a manifest that set it got `TypeError: got an unexpected
    keyword argument` — but only once the ingest reached construct,
    after the connector had already read the whole snapshot.
    """
    from archi.sources.twiki import TwikiCrawlSource, TwikiEOSSource

    eos = TwikiEOSSource(eos_root="/tmp", releases_map_path="r.map")
    assert eos.releases_map_path == "r.map"
    crawl = TwikiCrawlSource(
        base_url="https://twiki.example.org",
        seed_topics=["CMS/SeedTopic"],
        max_depth=1,
        releases_map_path="r.map",
    )
    assert crawl.releases_map_path == "r.map"


def test_releases_map_path_is_in_the_twiki_change_probe(tmp_path):
    """Swapping the release spine has to invalidate cached state, the
    same as swapping any other reference cache."""
    from archi.sources.twiki import TwikiEOSSource

    a = TwikiEOSSource(eos_root="/tmp", releases_map_path="a.map")
    b = TwikiEOSSource(eos_root="/tmp", releases_map_path="b.map")
    assert a.change_probe._config_hash != b.change_probe._config_hash


def test_releases_map_reaches_the_release_target_set(tmp_path):
    """End to end: a map on disk becomes the `release` target set the
    chunk scanner intersects against."""
    from archi.sources.docs import _reference_targets

    path = tmp_path / "releases.map"
    path.write_text(
        "label=CMSSW_10_6_30;state=Announced;type=Production;prodarch=1;\n",
        encoding="utf-8",
    )
    targets = _reference_targets(releases_map_path=str(path))
    assert "CMSSW_10_6_30" in targets["release"]

    with pytest.raises(FileNotFoundError):
        _reference_targets(releases_map_path=str(tmp_path / "absent.map"))
