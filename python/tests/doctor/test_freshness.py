"""req.r3-context-quality.freshness: `archi doctor` reports each source's age
against its pack's max_staleness (acc.r3-context-quality.report).

The ages come from okg's watermarks (`okg doctor --check sources --json`,
`sources[].coverage.watermark.last_reconcile_at`). These tests feed that
shape from fixtures; the real okg is not run.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from archi.bundles.duration import format_age, parse_duration
from archi.bundles.lock import LOCK_VERSION
from archi.stack.checks import FAIL, PASS, SKIP, Context, Probes, source_freshness
from archi.stack.freshness import evaluate

REPO = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
SLOS = {
    "docsite.docsite": {"max_staleness": "7d"},
    "monit-live.monit_sam": {"max_staleness": "36h"},
}


def okg_source(name, age_hours):
    stamp = None if age_hours is None else (NOW - timedelta(hours=age_hours)).isoformat()
    return {"name": name, "ownership_id": f"x.{name}",
            "coverage": {"watermark": {"last_reconcile_at": stamp} if stamp else {}}}


def by_source(rows):
    return {r["source"]: r for r in rows}


def test_a_source_older_than_its_slo_fails_and_one_within_it_passes():
    rows = by_source(evaluate(SLOS, [okg_source("docsite.docsite", 24 * 6),
                                     okg_source("monit-live.monit_sam", 40)], NOW))
    assert rows["docsite.docsite"]["ok"] is True
    assert rows["monit-live.monit_sam"]["ok"] is False
    assert rows["monit-live.monit_sam"]["age"] == "40h"
    assert rows["monit-live.monit_sam"]["max_staleness"] == "36h"
    assert "over the 36h limit" in rows["monit-live.monit_sam"]["reason"]


def test_the_limit_itself_is_within_the_slo():
    [row] = evaluate({"a.b": {"max_staleness": "36h"}}, [okg_source("a.b", 36)], NOW)
    assert row["ok"] is True


def test_a_source_okg_never_ran_or_does_not_list_is_not_ok():
    rows = by_source(evaluate(SLOS, [okg_source("docsite.docsite", None)], NOW))
    assert rows["docsite.docsite"]["ok"] is False
    assert rows["docsite.docsite"]["age"] is None
    assert "no completed run" in rows["docsite.docsite"]["reason"]
    assert rows["monit-live.monit_sam"]["ok"] is False
    assert "does not list" in rows["monit-live.monit_sam"]["reason"]


def test_another_packs_source_with_the_same_short_name_does_not_match():
    """git-code.repo must not pass on the strength of some other pack's `repo`."""
    [row] = evaluate({"git-code.repo": {"max_staleness": "7d"}}, [okg_source("repo", 1)], NOW)
    assert row["ok"] is False
    assert "does not list" in row["reason"]


@pytest.mark.parametrize("text,seconds", [("90s", 90), ("5m", 300), ("36h", 129600), ("7d", 604800), ("2w", 1209600)])
def test_durations(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "0h", "36", "h", "1.5h", "PT36H", "-1d", "7 d"])
def test_bad_durations_are_refused(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_format_age():
    assert [format_age(s) for s in (30, 300, 3600, 5400, 40 * 3600, 3 * 86400 + 4 * 3600)] == [
        "30s", "5m", "1h", "1h 30m", "40h", "3d 4h"]


# ---------------------------------------------------------------------------
# The doctor check
# ---------------------------------------------------------------------------


def context(tmp_path, report, slos=SLOS, lock=True):
    if lock:
        (tmp_path / "archi.lock").write_text(
            yaml.safe_dump({"lock_version": LOCK_VERSION, "composed": {"freshness": slos}}))
    calls = []

    def okg_json(*args):
        calls.append(args)
        return report

    probes = Probes(okg_json=okg_json, now=lambda: NOW)
    ctx = Context(SimpleNamespace(dir=tmp_path, name="myteam"), {}, probes)
    return ctx, calls


def good_report(*rows, runtime=True):
    return {"returncode": 0, "data": {"ok": True, "runtime_available": runtime,
                                      "runtime_error": None if runtime else "dsn_unavailable",
                                      "sources": list(rows)}}


def test_doctor_flags_a_source_older_than_its_slo(tmp_path):
    ctx, calls = context(tmp_path, good_report(okg_source("docsite.docsite", 24 * 9),
                                               okg_source("monit-live.monit_sam", 2)))
    result = source_freshness(ctx)
    assert result.status == FAIL
    assert "docsite.docsite" in result.message
    assert result.remediation.strip() and "\n" not in result.remediation
    assert calls == [("doctor", "--check", "sources", "--deployment", "myteam")]


def test_doctor_passes_sources_within_their_slo_and_exposes_them_as_json(tmp_path):
    ctx, _ = context(tmp_path, good_report(okg_source("docsite.docsite", 5),
                                           okg_source("monit-live.monit_sam", 30)))
    result = source_freshness(ctx)
    assert result.status == PASS
    payload = json.loads(json.dumps(result.as_dict()))
    rows = payload["data"]["sources"]
    assert {r["source"] for r in rows} == set(SLOS)
    assert all({"source", "age", "max_staleness", "ok"} <= set(r) for r in rows)
    assert {r["source"]: r["age"] for r in rows}["monit-live.monit_sam"] == "30h"


def test_doctor_fails_when_okg_cannot_read_watermarks(tmp_path):
    ctx, _ = context(tmp_path, good_report(runtime=False))
    result = source_freshness(ctx)
    assert result.status == FAIL
    assert "dsn_unavailable" in result.message


def test_doctor_skips_openly_without_a_lock_or_a_runner(tmp_path):
    ctx, _ = context(tmp_path, good_report(), lock=False)
    skipped = source_freshness(ctx)
    assert skipped.status == SKIP and "archi.lock" in skipped.message
    ctx, _ = context(tmp_path, good_report(), slos={})
    assert source_freshness(ctx).status == SKIP
    ctx.probes.okg_json = None
    assert source_freshness(ctx).status == SKIP


# ---------------------------------------------------------------------------
# The shipped packs
# ---------------------------------------------------------------------------


def test_every_shipped_context_pack_source_declares_max_staleness():
    from archi.bundles.manifest import load_pack

    checked = 0
    for pack_dir in sorted((REPO / "bundles" / "packs").iterdir()):
        pack = load_pack(pack_dir)  # raises when a source lacks max_staleness
        if pack.kind != "context" or not pack.sources():
            continue
        assert set(pack.freshness) == {f"{pack.name}.{s}" for s in pack.sources()}
        assert all(seconds > 0 for seconds in pack.max_staleness_seconds().values())
        checked += 1
    assert checked >= 10
