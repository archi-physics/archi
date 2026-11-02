"""empty-cache-fails-loudly: an empty or broken cache never claims scope.

okg retracts a source's rows when a run reports ``completed_scope=True``
with health ``ok``, ``not_applicable`` or ``skipped_optional``. A cache
that is empty, missing, unreadable, zero bytes or truncated is a failed
fetch, so every cache-backed source must report it as a failing health
status with no facts and no complete scope — and ``preflight()`` must
reach the same verdict as ``run()``. All data here is made up.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from archi.sources.conddb import CondDBGlobalTagSource
from archi.sources.cric import CRICCoreSource, CRICSource
from archi.sources.dbs import DBSDatasetSource
from archi.sources.docs import DocumentationSource
from archi.sources.dqm import DQMSource
from archi.sources.gocdb import GoCDBDowntimeSource
from archi.sources.indico import IndicoSource
from archi.sources.jira import JiraIssueSource
from archi.sources.wmstats import WMStatsWorkflowSource

MODES = ("scope_complete", "reconcile")

# name -> (factory(base), records path, one valid fake record)
LIST_SOURCES: dict[str, tuple[Callable[[str], Any], str, dict[str, Any]]] = {
    "dbs": (
        lambda base: DBSDatasetSource(base=base),
        "data/dbs-datasets/records.json",
        {"dataset": "/FakePrimary/FakeEra-v1/AOD", "data_tier_name": "AOD"},
    ),
    "conddb": (
        lambda base: CondDBGlobalTagSource(base=base),
        "data/conddb-global-tags/records.json",
        {"name": "FAKE_GT_v1", "release": "CMSSW_99_0_0"},
    ),
    "wmstats": (
        lambda base: WMStatsWorkflowSource(base=base),
        "data/wmstats-workflows/records.json",
        {"RequestName": "fake_workflow_0001", "Campaign": "FakeCampaign"},
    ),
    "dqm": (
        lambda base: DQMSource(base=base),
        "data/dqm/records.json",
        {"cert_name": "Cert_Fake_Golden", "filename": "fake.json"},
    ),
    "indico": (
        lambda base: IndicoSource(base=base),
        "data/indico/records.json",
        {"id": "900001", "title": "Fake meeting"},
    ),
    "jira": (
        lambda base: JiraIssueSource(base=base),
        "data/jira/records.json",
        {"key": "FAKE-1", "summary": "A fake issue"},
    ),
    "docs": (
        lambda base: DocumentationSource(base=base),
        "data/docsite/records.json",
        {"url": "https://docs.example.org/a/", "title": "A", "body": "Body"},
    ),
}

# Raw file contents that are not a usable cache. Each is cache_missing.
BROKEN_BYTES = {
    "empty-list": b"[]",
    "empty-list-whitespace": b"  [ ]\n",
    "zero-bytes": b"",
    "whitespace-only": b"\n  \n",
    "truncated-json": b'[{"key": "FAKE-1", "summary": "cut',
}


def _write(base: Path, rel: str, data: bytes) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _assert_refused(source: Any, path: Path, status: str, mode: str) -> None:
    run = source.run("run-1", mode=mode)
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == status
    assert run.health.cache_path == str(path)
    assert str(path) in run.health.reason
    assert "no complete scope claimed" in run.health.reason
    preflight = source.preflight()
    assert preflight.status == status
    assert preflight.cache_path == str(path)


# --- JSON-list caches ---------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", sorted(BROKEN_BYTES))
@pytest.mark.parametrize("name", sorted(LIST_SOURCES))
def test_broken_list_cache_is_cache_missing(tmp_path, name, case, mode):
    factory, rel, _record = LIST_SOURCES[name]
    path = _write(tmp_path, rel, BROKEN_BYTES[case])
    _assert_refused(factory(str(tmp_path)), path, "cache_missing", mode)


@pytest.mark.parametrize("name", sorted(LIST_SOURCES))
def test_missing_list_cache_reports_instead_of_raising(tmp_path, name):
    factory, rel, _record = LIST_SOURCES[name]
    _assert_refused(
        factory(str(tmp_path)), tmp_path / rel, "cache_missing",
        "scope_complete",
    )


@pytest.mark.parametrize("name", sorted(LIST_SOURCES))
def test_error_shaped_list_cache_is_endpoint_failed(tmp_path, name):
    factory, rel, _record = LIST_SOURCES[name]
    path = _write(tmp_path, rel, b'{"error": "fake upstream error"}')
    _assert_refused(
        factory(str(tmp_path)), path, "endpoint_failed", "scope_complete"
    )


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a mode-000 file",
)
@pytest.mark.parametrize("name", sorted(LIST_SOURCES))
def test_unreadable_list_cache_is_cache_missing(tmp_path, name):
    factory, rel, record = LIST_SOURCES[name]
    path = _write(tmp_path, rel, json.dumps([record]).encode())
    path.chmod(0)
    try:
        _assert_refused(
            factory(str(tmp_path)), path, "cache_missing", "scope_complete"
        )
        assert "could not be read" in factory(str(tmp_path)).preflight().reason
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("name", sorted(LIST_SOURCES))
def test_one_record_still_claims_scope(tmp_path, name):
    # Control: the guard is about zero records, not few.
    factory, rel, record = LIST_SOURCES[name]
    _write(tmp_path, rel, json.dumps([record]).encode())
    source = factory(str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts)
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.preflight().status == "ok"


@pytest.mark.parametrize("name", sorted(LIST_SOURCES))
def test_all_items_unparseable_agree_in_preflight(tmp_path, name):
    # run() already said endpoint_failed here; preflight said ok (or,
    # for dbs/conddb/wmstats, skipped_optional).
    factory, rel, _record = LIST_SOURCES[name]
    _write(tmp_path, rel, b'["junk", 7]')
    source = factory(str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert source.preflight().status == "endpoint_failed"


def test_jira_empty_cache_fails_even_when_meta_agrees(tmp_path):
    # meta.json saying record_count=0 must not bless an empty cache.
    factory, rel, _record = LIST_SOURCES["jira"]
    path = _write(tmp_path, rel, b"[]")
    _write(tmp_path, "data/jira/meta.json", b'{"record_count": 0}')
    _assert_refused(factory(str(tmp_path)), path, "cache_missing", "reconcile")


def test_jira_keyless_items_stop_the_scope_claim(tmp_path):
    factory, rel, record = LIST_SOURCES["jira"]
    _write(tmp_path, rel, json.dumps([record, {"summary": "no key"}]).encode())
    source = factory(str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert len([f for f in run.facts if getattr(f, "subtype", "") == "jira_issue"]) == 1
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 1 unparseable" in run.health.reason


def test_jira_preflight_reports_meta_mismatch_like_run(tmp_path):
    factory, rel, record = LIST_SOURCES["jira"]
    _write(tmp_path, rel, json.dumps([record]).encode())
    _write(tmp_path, "data/jira/meta.json", b'{"record_count": 5}')
    source = factory(str(tmp_path))
    assert source.run("run-1", mode="scope_complete").health.status == (
        "endpoint_failed"
    )
    preflight = source.preflight()
    assert preflight.status == "endpoint_failed"
    assert "record_count=5" in preflight.reason


def test_docs_urlless_items_stop_the_scope_claim(tmp_path):
    factory, rel, record = LIST_SOURCES["docs"]
    payload = [record, dict(record, title="duplicate"), {"title": "no url"}]
    _write(tmp_path, rel, json.dumps(payload).encode())
    source = factory(str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert run.completed_scope is False
    assert "skipped 1 unparseable" in run.health.reason
    # A duplicate URL alone is deduplication and keeps the claim.
    _write(tmp_path, rel, json.dumps(payload[:2]).encode())
    assert source.run("run-1", mode="scope_complete").completed_scope is True


# --- GOCDB --------------------------------------------------------------------

GOCDB_RECORDS = "data/gocdb-downtimes/records.json"
GOCDB_SITES = "data/cric/sites.json"
GOCDB_SERVICES = "data/cric-core/services.json"
FAKE_DOWNTIME = {
    "downtime_id": 4242,
    "hosted_by": "T2_XX_Fake",
    "hostname": "se.fake.example.org",
    "service_type": "SRM",
}


def _gocdb(tmp_path, records: bytes, *, sites: bytes = b'{"T2_XX_Fake": {}}',
           services: bytes = b'{"svc-se.fake.example.org": {}}'):
    _write(tmp_path, GOCDB_RECORDS, records)
    _write(tmp_path, GOCDB_SITES, sites)
    _write(tmp_path, GOCDB_SERVICES, services)
    return GoCDBDowntimeSource(base=str(tmp_path))


@pytest.mark.parametrize("mode", MODES)
def test_gocdb_empty_downtime_list_is_cache_missing(tmp_path, mode):
    # Review of #21: an empty downtime list follows the operator rule
    # (an empty cache is a loud failure), not skipped_optional.
    source = _gocdb(tmp_path, b"[]")
    _assert_refused(source, tmp_path / GOCDB_RECORDS, "cache_missing", mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("which", [GOCDB_SITES, GOCDB_SERVICES])
def test_gocdb_empty_topology_is_cache_missing(tmp_path, which, mode):
    # An empty topology dropped every affects edge while the run still
    # claimed a complete scope, retracting the edges.
    source = _gocdb(tmp_path, json.dumps([FAKE_DOWNTIME]).encode())
    _write(tmp_path, which, b"{}")
    _assert_refused(source, tmp_path / which, "cache_missing", mode)


@pytest.mark.parametrize("which", [GOCDB_SITES, GOCDB_SERVICES])
def test_gocdb_topology_error_body_is_endpoint_failed(tmp_path, which):
    source = _gocdb(tmp_path, json.dumps([FAKE_DOWNTIME]).encode())
    _write(tmp_path, which, b'{"error": {"code": 403}}')
    _assert_refused(source, tmp_path / which, "endpoint_failed", "scope_complete")


@pytest.mark.parametrize("case", ["zero-bytes", "truncated-json"])
def test_gocdb_broken_downtime_cache_is_cache_missing(tmp_path, case):
    source = _gocdb(tmp_path, BROKEN_BYTES[case])
    _assert_refused(
        source, tmp_path / GOCDB_RECORDS, "cache_missing", "scope_complete"
    )


@pytest.mark.parametrize("which", [GOCDB_RECORDS, GOCDB_SITES, GOCDB_SERVICES])
def test_gocdb_missing_cache_file_agrees_in_preflight(tmp_path, which):
    # main: run() raised FileNotFoundError, and a missing topology cache
    # passed preflight as ok.
    source = _gocdb(tmp_path, json.dumps([FAKE_DOWNTIME]).encode())
    (tmp_path / which).unlink()
    _assert_refused(source, tmp_path / which, "cache_missing", "scope_complete")


def test_gocdb_one_downtime_still_claims_scope(tmp_path):
    source = _gocdb(tmp_path, json.dumps([FAKE_DOWNTIME]).encode())
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts)
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.preflight().status == "ok"


# --- CRIC and CRIC core (JSON-object caches) ----------------------------------

CRIC_FILES = {
    "data/cric/sites.json": {"T2_XX_Fake": {"tier_level": 2,
                                            "sitedb_title": "Fake Site"}},
    "data/cric/facilities.json": {"FakeFacility": {"cmssites": ["T2_XX_Fake"]}},
    "data/cric/storage_units.json": {"FAKE-SE": {"site": {"name": "T2_XX_Fake"}}},
    "data/cric/compute_units.json": {"FAKE-CE": {"corepower": 10.0}},
    "data/cric/responsibilities.json": {
        "result": [["fakeuser", "Fake Site", "Site Admin"]]
    },
}
CORE_FILES = {
    "data/cric-core/services.json": {"fake-svc": {"type": "webservice",
                                                  "rcsite": "FAKE-RC"}},
    "data/cric-core/rcsites.json": {"FAKE-RC": {"sites": [
        {"name": "T2_XX_Fake", "vo_name": "cms"}]}},
    "data/cric-core/federations.json": {"XX-FAKE": {"vos": ["cms"],
                                                    "rcsites": ["FAKE-RC"]}},
}


def _write_all(tmp_path, files, overrides=None):
    for rel, payload in files.items():
        data = (overrides or {}).get(rel)
        _write(tmp_path, rel, json.dumps(payload).encode() if data is None
               else data)


def _cric(kind):
    return CRICSource if kind == "cric" else CRICCoreSource


def _files(kind):
    return CRIC_FILES if kind == "cric" else CORE_FILES


ALL_CRIC = [("cric", rel) for rel in CRIC_FILES] + [
    ("cric_core", rel) for rel in CORE_FILES
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind,rel", ALL_CRIC)
def test_cric_empty_object_is_cache_missing(tmp_path, kind, rel, mode):
    # main: {} in any file claimed a complete scope over what was left
    # (all five {} -> zero records, health ok).
    _write_all(tmp_path, _files(kind), {rel: b"{}"})
    source = _cric(kind)(base=str(tmp_path))
    _assert_refused(source, tmp_path / rel, "cache_missing", mode)


@pytest.mark.parametrize("kind,rel", ALL_CRIC)
def test_cric_list_instead_of_object_is_endpoint_failed(tmp_path, kind, rel):
    # main: run() crashed with AttributeError on a [] cache.
    _write_all(tmp_path, _files(kind), {rel: b"[]"})
    source = _cric(kind)(base=str(tmp_path))
    _assert_refused(source, tmp_path / rel, "endpoint_failed", "scope_complete")


@pytest.mark.parametrize("case", ["zero-bytes", "truncated-json"])
@pytest.mark.parametrize("kind,rel", ALL_CRIC)
def test_cric_broken_file_is_cache_missing(tmp_path, kind, rel, case):
    _write_all(tmp_path, _files(kind), {rel: BROKEN_BYTES[case]})
    source = _cric(kind)(base=str(tmp_path))
    _assert_refused(source, tmp_path / rel, "cache_missing", "scope_complete")


@pytest.mark.parametrize("kind", ["cric", "cric_core"])
def test_cric_missing_files_named_in_run_and_preflight(tmp_path, kind):
    source = _cric(kind)(base=str(tmp_path))
    first = tmp_path / next(iter(_files(kind)))
    _assert_refused(source, first, "cache_missing", "scope_complete")
    assert f"{len(_files(kind))} of {len(_files(kind))}" in (
        source.run("r", mode="scope_complete").health.reason
    )


def test_cric_empty_responsibility_rows_are_cache_missing(tmp_path):
    rel = "data/cric/responsibilities.json"
    _write_all(tmp_path, CRIC_FILES, {rel: b'{"result": []}'})
    _assert_refused(
        CRICSource(base=str(tmp_path)), tmp_path / rel, "cache_missing",
        "reconcile",
    )


def test_cric_core_without_a_cms_federation_is_endpoint_failed(tmp_path):
    rel = "data/cric-core/federations.json"
    _write_all(tmp_path, CORE_FILES,
               {rel: b'{"XX-OTHER": {"vos": ["othervo"], "rcsites": []}}'})
    _assert_refused(
        CRICCoreSource(base=str(tmp_path)), tmp_path / rel, "endpoint_failed",
        "scope_complete",
    )


@pytest.mark.parametrize("kind", ["cric", "cric_core"])
def test_cric_full_caches_still_claim_scope(tmp_path, kind):
    _write_all(tmp_path, _files(kind))
    source = _cric(kind)(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts)
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.preflight().status == "ok"


# --- CRIC: error bodies, non-object entries, malformed rows (review of #21) ---

ERROR_BODY = b'{"error": {"code": 403, "message": "Forbidden"}}'
# Every name -> object cache (responsibilities is a row list; above).
OBJECT_CRIC = [
    (kind, rel) for kind, rel in ALL_CRIC
    if not rel.endswith("responsibilities.json")
]


@pytest.mark.parametrize("kind,rel", OBJECT_CRIC)
def test_cric_error_body_object_is_endpoint_failed(tmp_path, kind, rel):
    # 64ed7e74: the "error" key was read as one entry named "error", and
    # the run claimed a complete scope, retracting every real record.
    _write_all(tmp_path, _files(kind), {rel: ERROR_BODY})
    source = _cric(kind)(base=str(tmp_path))
    _assert_refused(source, tmp_path / rel, "endpoint_failed", "scope_complete")
    assert "error body" in source.preflight().reason


def test_cric_sites_without_a_cms_site_name_is_endpoint_failed(tmp_path):
    rel = "data/cric/sites.json"
    _write_all(tmp_path, CRIC_FILES, {rel: b'{"status": {"ok": false}}'})
    _assert_refused(
        CRICSource(base=str(tmp_path)), tmp_path / rel, "endpoint_failed",
        "scope_complete",
    )


@pytest.mark.parametrize("value", ['"Forbidden"', "[1, 2]", "null", "7"])
@pytest.mark.parametrize("kind,rel", OBJECT_CRIC)
def test_cric_only_non_object_values_is_endpoint_failed(
    tmp_path, kind, rel, value
):
    # 64ed7e74: AttributeError out of run() and preflight().
    key = "T2_XX_Broken" if rel.endswith("cric/sites.json") else "broken"
    _write_all(tmp_path, _files(kind), {rel: f'{{"{key}": {value}}}'.encode()})
    source = _cric(kind)(base=str(tmp_path))
    _assert_refused(source, tmp_path / rel, "endpoint_failed", "scope_complete")


@pytest.mark.parametrize("kind,rel", OBJECT_CRIC)
def test_cric_one_non_object_entry_is_skipped_and_stops_the_claim(
    tmp_path, kind, rel
):
    payload = dict(_files(kind)[rel])
    payload["T2_XX_Broken" if rel.endswith("cric/sites.json") else "broken"] = (
        "not an object"
    )
    _write_all(tmp_path, _files(kind), {rel: json.dumps(payload).encode()})
    source = _cric(kind)(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts)
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert f"1 in {tmp_path / rel}" in run.health.reason
    preflight = source.preflight()
    assert preflight.status == "ok"
    assert "no complete scope claimed" in preflight.reason


@pytest.mark.parametrize(
    "rows", [[[]], [["fakeuser", "Fake Site"]], ["not a row"], [None]]
)
def test_cric_no_three_field_row_is_endpoint_failed(tmp_path, rows):
    # 64ed7e74: {"result": [[]]} was skipped silently under a complete
    # scope, retracting every operator.
    rel = "data/cric/responsibilities.json"
    _write_all(tmp_path, CRIC_FILES, {rel: json.dumps({"result": rows}).encode()})
    _assert_refused(
        CRICSource(base=str(tmp_path)), tmp_path / rel, "endpoint_failed",
        "reconcile",
    )


def test_cric_malformed_row_among_good_ones_stops_the_claim(tmp_path):
    rel = "data/cric/responsibilities.json"
    rows = CRIC_FILES[rel]["result"] + [["fakeuser", "Fake Site"],
                                        ["x", "y", "z", "extra"]]
    _write_all(tmp_path, CRIC_FILES, {rel: json.dumps({"result": rows}).encode()})
    source = CRICSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts)
    assert run.completed_scope is False
    assert f"2 in {tmp_path / rel}" in run.health.reason
    assert "no complete scope claimed" in source.preflight().reason


def test_cric_null_username_or_site_stays_a_plain_drop(tmp_path):
    # Real exports carry rows with a null site; those are not drift.
    rel = "data/cric/responsibilities.json"
    rows = CRIC_FILES[rel]["result"] + [["someone", None, "Site Admin"]]
    _write_all(tmp_path, CRIC_FILES, {rel: json.dumps({"result": rows}).encode()})
    run = CRICSource(base=str(tmp_path)).run("run-1", mode="scope_complete")
    assert run.completed_scope is True
    assert run.health.status == "ok"


@pytest.mark.parametrize("kind,rel,payload", [
    ("cric", "data/cric/facilities.json", {"FakeFacility": {"cmssites": 5}}),
    ("cric_core", "data/cric-core/rcsites.json", {"FAKE-RC": {"sites": 5}}),
    ("cric_core", "data/cric-core/federations.json",
     {"XX-FAKE": {"vos": ["cms"], "pledges": {"2026": "not an object"}}}),
])
def test_cric_unexpected_nested_shape_never_crashes(tmp_path, kind, rel, payload):
    _write_all(tmp_path, _files(kind), {rel: json.dumps(payload).encode()})
    source = _cric(kind)(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "unexpected shape" in run.health.reason
    assert source.preflight().status == "endpoint_failed"


@pytest.mark.parametrize("rows", [
    [["fakeuser", None, "Site Admin"], ["other", None, "Site Executive"]],
    [["fakeuser", "Renamed Fake Site", "Site Admin"]],
], ids=["all-null-site", "no-title-matches"])
def test_cric_rows_that_yield_no_operator_are_endpoint_failed(tmp_path, rows):
    # Re-review of #21: rows present but zero operators claimed a
    # complete scope and would retract every operator.
    rel = "data/cric/responsibilities.json"
    _write_all(tmp_path, CRIC_FILES, {rel: json.dumps({"result": rows}).encode()})
    source = CRICSource(base=str(tmp_path))
    _assert_refused(source, tmp_path / rel, "endpoint_failed", "scope_complete")
    assert "none maps to an operator" in source.preflight().reason


def test_cric_rows_with_some_null_sites_still_claim_scope(tmp_path):
    # Control: real-like rows, some with a null site, one that maps.
    rel = "data/cric/responsibilities.json"
    rows = [["fakeuser", "Fake Site", "Site Admin"],
            ["other", None, "Site Admin"],
            ["third", "Unknown Title", "Site Executive"]]
    _write_all(tmp_path, CRIC_FILES, {rel: json.dumps({"result": rows}).encode()})
    source = CRICSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert source.preflight().status == "ok"
