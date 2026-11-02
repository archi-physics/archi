"""req.cmssst-site-status.reader -- CMSSSTSiteStatusSource, offline.

The fixtures are real site JSON fetched from cmssst.web.cern.ch on
2026-10-07 (snapshot 14:39 UTC), with only the per-host ``elements`` list
removed, as the downloader removes it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from okg.deployment import EdgeFact, NodeFact

from archi.sources.cmssst import (
    CMSSSTSiteStatusAdapter,
    CMSSSTSiteStatusSource,
    STATUS_WORDS,
    current_status,
    decode_letter,
    records_from_payload,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cmssst"
SITES = ("T1_US_FNAL", "T2_US_MIT", "T2_CH_CERN")
RECORDS = "data/cmssst-site-status/records.json"


def _sample(site: str) -> dict:
    return json.loads((FIXTURES / f"{site}.json").read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload, rel: str = RECORDS) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(data)


def _source(tmp_path: Path, payload=None, **kwargs) -> CMSSSTSiteStatusSource:
    _write(tmp_path, [_sample(s) for s in SITES] if payload is None else payload)
    return CMSSSTSiteStatusSource(base=str(tmp_path), **kwargs)


def _facts(source, mode="scope_complete"):
    run = source.run("run-1", mode=mode)
    return run, list(run.facts)


def _pages(facts):
    return {
        f.node_id: f
        for f in facts
        if isinstance(f, NodeFact) and f.subtype == "documentation_page"
    }


# --- decoding ----------------------------------------------------------------


def test_legend_letters_decode_to_the_page_words():
    assert decode_letter("o") == "ok (site state or evaluation good)"
    assert decode_letter("w") == "warning (site or evaluation has issues)"
    assert decode_letter("e") == "error (site or evaluation failed)"
    assert decode_letter("d") == "site or service in full downtime"
    assert decode_letter("p") == "site in partial downtime"
    assert decode_letter("a") == "site or service in unscheduled downtime"
    assert decode_letter("r") == "site at risk of downtime"
    assert decode_letter("W") == "site not in service (Waiting Room)"
    assert decode_letter("M") == "site not in service (Morgue)"
    assert decode_letter("T").startswith("error") and "Waiting Room" in decode_letter("T")
    assert decode_letter("H").startswith("ok") and "Morgue" in decode_letter("H")
    assert "unrecognised status code 'Z'" == decode_letter("Z")
    # Every letter the real summary.js uses has a meaning (site names hold
    # underscores, so the pattern matches only the status strings).
    summary = (FIXTURES / "summary.js").read_text(encoding="utf-8")
    body = summary[summary.index("siteStatusData"):]
    used = set("".join(re.findall(r'"([A-Za-z]+)"', body)))
    assert len(used) > 10
    assert used <= set(STATUS_WORDS)


def _record(today: dict[str, str], snapshot: int = 1791383941):
    item = {
        "site": "T2_XX_Test",
        "time": snapshot,
        "ggus": [],
        "metrics": {name: {"today": [letters]} for name, letters in today.items()},
    }
    records, skipped = records_from_payload([item])
    assert skipped == 0
    return records[0]


def test_current_status_is_the_latest_known_bin_at_or_before_the_snapshot():
    # 14:39 UTC is bin 58. A later bin (scheduled change) is not "now".
    letters = "o" * 50 + "e" * 5 + "u" * 4 + "W" * 37
    record = _record({"LifeStatus": letters})
    assert current_status(record, "LifeStatus") == (
        "error (site or evaluation failed), latest evaluation"
    )


def test_downtime_is_read_at_the_snapshot_bin_only():
    earlier = _record({"Downtime": "d" * 40 + "u" * 56})
    assert current_status(earlier, "Downtime") == "no downtime at the snapshot time"
    now = _record({"Downtime": "u" * 58 + "a" * 38})
    assert current_status(now, "Downtime") == (
        "site or service in unscheduled downtime, at the snapshot time"
    )


def test_an_override_that_ended_earlier_today_is_not_reported():
    # Review finding 1: 20 bins of an error override, then none (u) from
    # 05:00; at 14:39 no override is in effect.
    record = _record({"manProdStatus": "e" * 20 + "u" * 76})
    assert current_status(record, "manProdStatus") == (
        "no manual override set at the snapshot time"
    )
    active = _record({"manProdStatus": "u" * 50 + "W" * 46})
    assert current_status(active, "manProdStatus") == (
        "site not in service (Waiting Room), at the snapshot time"
    )


def test_a_refresh_with_unchanged_statuses_keeps_text_and_chunk_ids(tmp_path):
    # Review finding 2: a 15-minute refresh (later time, evaluations one
    # bin further on) with the same statuses must not churn the chunks.
    first = [_sample(s) for s in SITES]
    later = [_sample(s) for s in SITES]
    for item in later:
        item["time"] += 900
        for bins in item["metrics"].values():
            joined = "".join(bins["today"])
            last = max((i for i, c in enumerate(joined) if c != "u"), default=None)
            if last is not None and last < 95:
                joined = joined[: last + 1] + joined[last] + joined[last + 2 :]
            bins["today"] = [joined]
    _, a = _facts(_source(tmp_path / "a", first))
    _, b = _facts(_source(tmp_path / "b", later))

    def stable(facts):
        return sorted(
            (f.node_id, f.attrs["text"])
            for f in facts
            if isinstance(f, NodeFact)
        )

    assert stable(a) == stable(b)
    pa = _pages(a)["documentation_page:cmssst:T1_US_FNAL"]
    pb = _pages(b)["documentation_page:cmssst:T1_US_FNAL"]
    assert pa.attrs["observed_at"] != pb.attrs["observed_at"]


def test_unknown_overrides_and_missing_metrics_say_so():
    record = _record({"manLifeStatus": "u" * 96, "ProdStatus": "u" * 96})
    assert current_status(record, "manLifeStatus") == "no manual override set at the snapshot time"
    assert current_status(record, "ProdStatus") == "unknown (no evaluation yet today)"
    assert current_status(record, "CrabStatus") == "not reported on the page"


# --- emitted pages -----------------------------------------------------------


def test_one_citable_page_per_site_with_url_and_title(tmp_path):
    run, facts = _facts(_source(tmp_path))
    pages = _pages(facts)
    assert sorted(pages) == sorted(f"documentation_page:cmssst:{s}" for s in SITES)
    fnal = pages["documentation_page:cmssst:T1_US_FNAL"]
    assert fnal.attrs["url"] == (
        "https://cmssst.web.cern.ch/siteStatus/detail.html?site=T1_US_FNAL"
    )
    assert fnal.attrs["title"] == "T1_US_FNAL site status"
    assert fnal.source_record_id == {"site": "T1_US_FNAL"}
    assert fnal.attrs["observed_at"] == "2026-10-07T14:39:01+00:00"
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.record_count == 3


def test_fnal_page_text_states_the_current_status_in_words(tmp_path):
    _, facts = _facts(_source(tmp_path))
    body = _pages(facts)["documentation_page:cmssst:T1_US_FNAL"].attrs["body"]
    assert "14:39" not in body and "14:30" not in body
    assert "Life Status: ok (site state or evaluation good), latest evaluation." in body
    assert (
        "Production (Prod) Status: ok (site state or evaluation good), latest evaluation."
        in body
    )
    assert "CRAB analysis (Crab) Status manual override: no manual override set at the snapshot time." in body
    assert "Downtime: no downtime at the snapshot time." in body
    assert "GGUS tickets listed for T1_US_FNAL: 1003499 (opened 2026-08-04" in body
    assert "1003889 (opened 2026-09-15" in body


def test_mit_warning_is_decoded(tmp_path):
    _, facts = _facts(_source(tmp_path))
    body = _pages(facts)["documentation_page:cmssst:T2_US_MIT"].attrs["body"]
    assert "Site Readiness: warning (site or evaluation has issues)" in body
    assert "SAM Status: warning (site or evaluation has issues)" in body
    assert "GGUS tickets listed for T2_US_MIT: 1002767" in body


def test_page_is_chunked_and_chunks_carry_the_status_text(tmp_path):
    _, facts = _facts(_source(tmp_path))
    chunks = [f for f in facts if isinstance(f, NodeFact) and f.subtype == "document_chunk"]
    contains = [
        f for f in facts if isinstance(f, EdgeFact) and f.edge_type == "contains"
    ]
    assert len(chunks) == len(contains) >= 3
    srcs = {e.src for e in contains}
    assert srcs == set(_pages(facts))
    assert any("T2_CH_CERN current site status" in c.attrs["text"] for c in chunks)
    # No site edge without a sites cache.
    assert not [f for f in facts if isinstance(f, EdgeFact) and f.edge_type == "references"]


def test_identity_is_stable_and_revision_follows_content(tmp_path):
    first_source = _source(tmp_path / "a")
    _, first = _facts(first_source)
    changed = [_sample(s) for s in SITES]
    changed[0]["time"] += 900
    changed[0]["metrics"]["LifeStatus"]["today"] = ["W" * 48, "W" * 48]
    _, second = _facts(_source(tmp_path / "b", changed))
    a = _pages(first)["documentation_page:cmssst:T1_US_FNAL"]
    b = _pages(second)["documentation_page:cmssst:T1_US_FNAL"]
    assert a.node_id == b.node_id and a.source_record_id == b.source_record_id
    assert a.source_revision["content_hash"] != b.source_revision["content_hash"]
    assert "Life Status: site not in service (Waiting Room)" in b.attrs["body"]


def test_site_edge_only_to_sites_the_cric_cache_lists(tmp_path):
    _write(tmp_path, {"T1_US_FNAL": {}, "T2_US_MIT": {}}, "data/cric/sites.json")
    source = _source(tmp_path, sites_path="data/cric/sites.json")
    _, facts = _facts(source)
    edges = {
        (f.src, f.edge_type, f.dst)
        for f in facts
        if isinstance(f, EdgeFact) and f.edge_type == "references"
    }
    assert edges == {
        ("documentation_page:cmssst:T1_US_FNAL", "references", "site:T1_US_FNAL"),
        ("documentation_page:cmssst:T2_US_MIT", "references", "site:T2_US_MIT"),
    }


def test_a_configured_empty_sites_cache_is_refused(tmp_path):
    _write(tmp_path, {}, "data/cric/sites.json")
    source = _source(tmp_path, sites_path="data/cric/sites.json")
    run, facts = _facts(source)
    assert facts == []
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert source.preflight().status == "cache_missing"


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, status",
    [
        (b"", "cache_missing"),
        (b"[]", "cache_missing"),
        (b'[{"site": "T1_US_FNAL"', "cache_missing"),
        (b'{"error": "upstream down"}', "endpoint_failed"),
    ],
    ids=["zero-bytes", "empty-list", "truncated", "error-body"],
)
@pytest.mark.parametrize("mode", ["scope_complete", "reconcile", "cursor"])
def test_unusable_cache_gives_no_facts_and_no_scope(tmp_path, payload, status, mode):
    source = _source(tmp_path, payload)
    run, facts = _facts(source, mode)
    assert facts == []
    assert run.completed_scope is False
    assert run.health.status == status
    assert source.preflight().status == status


def test_missing_cache_file_is_cache_missing(tmp_path):
    source = CMSSSTSiteStatusSource(base=str(tmp_path))
    run, facts = _facts(source)
    assert facts == [] and run.completed_scope is False
    assert run.health.status == "cache_missing"
    assert source.preflight().status == "cache_missing"


@pytest.mark.parametrize(
    "bad",
    [
        "not an object",
        {"site": "not-a-site", "time": 1, "metrics": {"x": {"today": []}}},
        {"site": "T2_XX_A", "time": 0, "metrics": {"x": {"today": []}}},
        {"site": "T2_XX_A", "time": 5, "metrics": {}},
        {"site": "T2_XX_A", "time": 5, "metrics": {"x": {"today": [1]}}},
        {"site": "T2_XX_A", "time": 5, "ggus": [[1]], "metrics": {"x": {"today": []}}},
    ],
    ids=["not-object", "bad-site", "no-time", "no-metrics", "bad-bins", "bad-ggus"],
)
def test_a_malformed_record_is_skipped_and_stops_the_scope_claim(tmp_path, bad):
    source = _source(tmp_path, [_sample("T1_US_FNAL"), bad])
    run, facts = _facts(source)
    assert sorted(_pages(facts)) == ["documentation_page:cmssst:T1_US_FNAL"]
    assert run.completed_scope is False
    assert "skipped 1 unparseable" in run.health.reason
    assert "skipped 1 unparseable" in source.preflight().reason


def test_all_records_malformed_is_endpoint_failed(tmp_path):
    source = _source(tmp_path, [{"site": "nope"}])
    run, facts = _facts(source)
    assert _pages(facts) == {}
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_adapter_declares_the_reader_profile():
    assert CMSSSTSiteStatusAdapter.reader_class is CMSSSTSiteStatusSource
    assert CMSSSTSiteStatusAdapter.profile == CMSSSTSiteStatusSource.profile
    assert (
        CMSSSTSiteStatusAdapter.change_probe_kind
        == CMSSSTSiteStatusSource.change_probe_kind
    )
