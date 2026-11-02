"""task.w2.sources-monit — the four MONIT sources, offline.

Fixtures mirror the two cache shapes the cms originals consume: a JSON
list of plain record mappings, or a raw MONIT ``_msearch`` response
(``{"responses": [...]}``). Live paths run against a monkeypatched
``requests.post``.
"""
import copy
import json
import time

import pytest

from okg.deployment import (
    EdgeFact,
    NodeFact,
    ProgressMarker,
)
from okg.deployment import MutableApiProbe

import archi.sources.monit as monit_mod
from archi.sources.monit import (
    MONITCondorSource,
    MONITRucioDatasetSource,
    MONITRucioTransferSource,
    MONITSAMSource,
    _today,
)

TOKEN_ENV = "ARCHI_T_MONIT_TOKEN"


def _nodes(facts, subtype):
    return [f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype]


def _edges(facts, edge_type):
    return [f for f in facts if isinstance(f, EdgeFact) and f.edge_type == edge_type]


class _FakeHTTPResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _fake_post(monkeypatch, respond):
    """Patch requests.post; `respond(url, headers, data)` -> payload."""
    calls = []

    def _post(url, headers=None, data=None, timeout=None):
        calls.append({"url": url, "headers": headers, "data": data})
        payload = respond(url, headers, data)
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr(monit_mod.requests, "post", _post)
    return calls


# --- fixtures: raw MONIT _msearch response shapes -----------------------------

SITEMON_RESPONSE = {
    "responses": [{
        "aggregations": {"groups": {"buckets": [
            {
                "key": "T1_DE_KIT",
                "status_breakdown": {"buckets": [
                    {"key": "OK", "doc_count": 18},
                    {"key": "WARNING", "doc_count": 1},
                    {"key": "CRITICAL", "doc_count": 1},
                ]},
            },
            {
                "key": "T3_US_Baylor",
                "status_breakdown": {"buckets": [
                    {"key": "OK", "doc_count": 1},
                    {"key": "CRITICAL", "doc_count": 3},
                ]},
            },
            {"key": "", "status_breakdown": {"buckets": []}},
        ]}}
    }]
}

CONDOR_RESPONSE = {
    "responses": [{
        "aggregations": {"sites": {"buckets": [{
            "key": "T2_US_MIT",
            "doc_count": 100,
            "total_jobs": {"value": 120},
            "total_core_hrs": {"value": 400.0},
            "total_cpu_time_hrs": {"value": 300.0},
            "avg_queue_hrs": {"value": 2.5},
            "by_status": {"buckets": [
                {"key": "Running", "doc_count": 60},
                {"key": "Idle", "doc_count": 30},
                {"key": "Held", "doc_count": 10},
            ]},
            "by_job_type": {"buckets": [
                {"key": "analysis", "doc_count": 80},
            ]},
            "by_error": {"buckets": [
                {"key": "SystemError", "doc_count": 4},
            ]},
        }]}}
    }]
}

TRANSFER_RESPONSE = {
    "responses": [{
        "aggregations": {"by_src_rse": {"buckets": [
            {
                "key": "T1_DE_KIT_Disk",
                "by_dst_rse": {"buckets": [{
                    "key": "T2_US_MIT",
                    "bytes_total": {"value": 1000},
                    "avg_duration": {"value": 100.0},
                    "by_event_type": {"buckets": [
                        {"key": "transfer-done", "doc_count": 8},
                        {"key": "transfer-failed", "doc_count": 2},
                    ]},
                    "top_activity": {"buckets": [
                        {"key": "Production Output", "doc_count": 9},
                    ]},
                    "failure_reasons": {"buckets": [
                        {"key": "CHECKSUM MISMATCH", "doc_count": 2},
                    ]},
                }]},
            },
            {
                # Same site pair through the _Tape RSE: must merge.
                "key": "T1_DE_KIT_Tape",
                "by_dst_rse": {"buckets": [{
                    "key": "T2_US_MIT_Disk",
                    "bytes_total": {"value": 500},
                    "avg_duration": {"value": 200.0},
                    "by_event_type": {"buckets": [
                        {"key": "transfer-done", "doc_count": 5},
                    ]},
                    "top_activity": {"buckets": []},
                    "failure_reasons": {"buckets": [
                        {"key": "CHECKSUM MISMATCH", "doc_count": 1},
                    ]},
                }]},
            },
        ]}}
    }]
}

DATASET_BUCKET = {
    "key": {"dataset": "/Prim/Era-v1/RECO"},
    "replicas": {"buckets": [
        {"key": "T1_DE_KIT_Disk"},
        {"key": "T2_US_MIT"},
    ]},
    "sample": {"hits": {"hits": [{
        "_source": {"data": {
            "data_tier_name": "RECO",
            "acquisition_era_name": "Era",
            "dbs_event_count": 5,
            "dbs_n_files": 2,
            "dbs_size": 100,
            "physics_group_name": "Tracker",
        }}
    }]}},
}

DATASET_RESPONSE = {
    "responses": [{
        "aggregations": {"datasets": {"buckets": [
            DATASET_BUCKET,
            {
                "key": {"dataset": "/GenericTTbar/x/GEN-SIM"},
                "replicas": {"buckets": [{"key": "T2_CH_CERN"}]},
                "sample": {"hits": {"hits": []}},
            },
        ]}}
    }]
}


def _sam_source(tmp_path, **overrides):
    kwargs = {
        "records_path": "data/monit-sam/records.json",
        "token_env": TOKEN_ENV,
        "base": str(tmp_path),
    }
    kwargs.update(overrides)
    return MONITSAMSource(**kwargs)


def _write(tmp_path, rel, payload):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# --- SAM ---------------------------------------------------------------------

def test_sam_cache_raw_response_parse_and_emission(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-sam/records.json", SITEMON_RESPONSE)
    source = _sam_source(tmp_path)
    run = source.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    snaps = {n.node_id: n for n in _nodes(facts, "monitoring_snapshot")}
    today = _today()
    assert set(snaps) == {
        f"monitoring_snapshot:sam:T1_DE_KIT:{today}",
        f"monitoring_snapshot:sam:T3_US_Baylor:{today}",
    }
    kit = snaps[f"monitoring_snapshot:sam:T1_DE_KIT:{today}"]
    assert kit.attrs["value"] == 90.0
    assert kit.attrs["status"] == "OK"  # >= 90 -> OK
    assert kit.attrs["total_tests"] == 20
    assert kit.attrs["ok_count"] == 18
    assert kit.attrs["metric"] == "sam_site_availability"
    assert kit.source_revision["run_id"] == "run-1"
    assert kit.source_revision["mode"] == "cache"
    baylor = snaps[f"monitoring_snapshot:sam:T3_US_Baylor:{today}"]
    assert baylor.attrs["status"] == "CRITICAL"  # 25% < 70
    hosts = _edges(facts, "hosts")
    assert {(e.src, e.dst) for e in hosts} == {
        ("site:T1_DE_KIT", kit.node_id),
        ("site:T3_US_Baylor", baylor.node_id),
    }
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.mode == "cache"
    assert run.health.record_count == 2
    # Cache runs never carry credential refs or a live endpoint.
    assert run.health.credential_refs == ()
    assert run.health.endpoint is None


def test_sam_cache_list_of_records(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-sam/records.json", [{
        "site": "T2_US_MIT",
        "snapshot_date": "2026-08-01",
        "availability_pct": 75.0,
        "status": "WARNING",
        "total_tests": 4,
        "ok_count": 3,
    }])
    facts = list(_sam_source(tmp_path).run("r").facts)
    snap = _nodes(facts, "monitoring_snapshot")[0]
    assert snap.node_id == "monitoring_snapshot:sam:T2_US_MIT:2026-08-01"
    assert snap.attrs["value"] == 75.0
    assert snap.attrs["observed_at"] == "2026-08-01"


def test_sam_completed_scope_by_mode(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-sam/records.json", SITEMON_RESPONSE)
    source = _sam_source(tmp_path)
    assert source.run("r", mode="cursor").completed_scope is False
    assert source.run("r", mode="scope_complete").completed_scope is True
    assert source.run("r", mode="reconcile").completed_scope is True


def test_sam_missing_credential_never_completes_scope(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    source = _sam_source(tmp_path)  # no cache file, no token
    run = source.run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "missing_credential"
    assert run.health.credential_refs == (TOKEN_ENV,)
    assert TOKEN_ENV in run.health.reason


def test_sam_corrupt_cache_is_endpoint_failed_not_complete(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    path = tmp_path / "data" / "monit-sam" / "records.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    run = _sam_source(tmp_path).run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_sam_empty_result_is_skipped_optional(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-sam/records.json", [])
    run = _sam_source(tmp_path).run("r")
    assert list(run.facts) == []
    assert run.health.status == "skipped_optional"
    assert "no site buckets" in run.health.reason


def test_sam_preflight_states(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    source = _sam_source(tmp_path)
    result = source.preflight()
    assert result.status == "missing_credential"
    assert result.credential_refs == (TOKEN_ENV,)
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    result = source.preflight()
    assert result.status == "ok"
    assert result.mode == "live"
    assert result.endpoint == "https://monit-grafana.cern.ch"
    _write(tmp_path, "data/monit-sam/records.json", SITEMON_RESPONSE)
    result = source.preflight()
    assert result.status == "ok"
    assert result.mode == "cache"
    assert result.record_count == 2
    assert result.content_hash


def test_sam_live_query_and_auth_header(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: SITEMON_RESPONSE)
    source = _sam_source(
        tmp_path,
        grafana_base_url="https://grafana.example.org/",
        datasource_id=4242,
        index="sitemon-test-*",
        query_filter="data.vo:other",
        max_sites=7,
    )
    run = source.run("r", mode="scope_complete")
    facts = list(run.facts)
    assert len(_nodes(facts, "monitoring_snapshot")) == 2
    assert run.health.status == "ok"
    assert run.health.mode == "live"
    assert run.health.credential_refs == (TOKEN_ENV,)
    assert run.health.endpoint == "https://grafana.example.org"
    (call,) = calls
    assert call["url"] == (
        "https://grafana.example.org/api/datasources/proxy/4242/_msearch"
    )
    assert call["headers"]["Authorization"] == "Bearer test-token"
    meta_line, query_line, _ = call["data"].split("\n")
    assert json.loads(meta_line)["index"] == ["sitemon-test-*"]
    query = json.loads(query_line)
    assert query["aggs"]["groups"]["terms"]["size"] == 7
    assert (
        query["query"]["bool"]["must"][0]["query_string"]["query"]
        == "data.vo:other"
    )


def test_sam_live_opensearch_error_is_endpoint_failed(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    _fake_post(monkeypatch, lambda *a: {
        "responses": [{"error": {"type": "x_content_parse_exception",
                                 "reason": "bad query"}}]
    })
    run = _sam_source(tmp_path).run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "bad query" in run.health.reason


def test_monit_probe_declarations(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    source = _sam_source(tmp_path)
    for cls in (MONITSAMSource, MONITCondorSource,
                MONITRucioTransferSource, MONITRucioDatasetSource):
        assert cls.profile == "mutable_api"
        assert cls.change_probe_kind == "mutable_api"
    assert isinstance(source.change_probe, MutableApiProbe)
    # No cache: each probe token is fresh (forced live read).
    assert source.change_probe.build_token() != source.change_probe.build_token()
    # Cache present: token is stable until the cache bytes change.
    _write(tmp_path, "data/monit-sam/records.json", SITEMON_RESPONSE)
    source = _sam_source(tmp_path)
    first = source.change_probe.build_token()
    assert source.change_probe.build_token() == first
    _write(tmp_path, "data/monit-sam/records.json", [])
    assert source.change_probe.build_token() != first


def test_source_name_param(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    source = _sam_source(tmp_path, source_name="monit_sam_dev")
    assert source.preflight().source_name == "monit_sam_dev"


# --- Condor ------------------------------------------------------------------

def test_condor_cache_raw_response(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-condor/records.json", CONDOR_RESPONSE)
    source = MONITCondorSource(
        records_path="data/monit-condor/records.json",
        token_env=TOKEN_ENV,
        base=str(tmp_path),
    )
    run = source.run("r", mode="scope_complete")
    facts = list(run.facts)
    snap = _nodes(facts, "monitoring_snapshot")[0]
    today = _today()
    assert snap.node_id == f"monitoring_snapshot:condor:T2_US_MIT:{today}"
    assert snap.attrs["metric"] == "condor_compute_summary"
    assert snap.attrs["total_jobs"] == 120
    assert snap.attrs["core_hours"] == 400.0
    assert snap.attrs["cpu_efficiency"] == 0.75
    assert snap.attrs["avg_queue_hours"] == 2.5
    assert snap.attrs["jobs_running"] == 60
    assert snap.attrs["jobs_idle"] == 30
    assert snap.attrs["jobs_held"] == 10
    assert snap.attrs["job_type_breakdown"] == {"analysis": 80}
    assert snap.attrs["error_breakdown"] == {"SystemError": 4}
    (edge,) = _edges(facts, "hosts")
    assert edge.src == "site:T2_US_MIT"
    assert edge.dst == snap.node_id
    assert run.health.status == "ok"


# --- Rucio transfer ------------------------------------------------------------

def test_rucio_transfer_rse_merge_and_edges(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(
        tmp_path, "data/monit-rucio-transfer/records.json", TRANSFER_RESPONSE
    )
    source = MONITRucioTransferSource(
        records_path="data/monit-rucio-transfer/records.json",
        token_env=TOKEN_ENV,
        base=str(tmp_path),
    )
    facts = list(source.run("r", mode="scope_complete").facts)
    (node,) = _nodes(facts, "transfer_job")
    today = _today()
    # _Disk and _Tape RSEs collapse onto one site pair and merge.
    assert node.node_id == f"transfer_job:rucio:T1_DE_KIT:T2_US_MIT:{today}"
    assert node.attrs["done_count"] == 13
    assert node.attrs["failed_count"] == 2
    assert node.attrs["total_bytes"] == 1500
    assert node.attrs["num_files"] == 15
    assert node.attrs["state"] == "mixed"
    assert node.attrs["success_rate"] == round(13 / 15, 4)
    # Weighted average: (100*10 + 200*5) / 15
    assert node.attrs["avg_duration_s"] == round(2000 / 15, 2)
    assert node.attrs["failure_reasons"] == {"CHECKSUM MISMATCH": 3}
    assert node.attrs["top_activity"] == "Production Output"
    hosts = _edges(facts, "hosts")
    assert {(e.src, e.attrs["role"]) for e in hosts} == {
        ("site:T1_DE_KIT", "transfer_source"),
        ("site:T2_US_MIT", "transfer_destination"),
    }
    (t_edge,) = _edges(facts, "transfers_to")
    assert (t_edge.src, t_edge.dst) == ("site:T1_DE_KIT", "site:T2_US_MIT")


def test_rucio_transfer_list_cache_dedupes(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-rucio-transfer/records.json", [
        {"src_rse": "T1_DE_KIT_Disk", "dst_rse": "T2_US_MIT",
         "snapshot_date": "2026-08-01", "done_count": 3, "bytes_total": 10},
        {"src_rse": "T1_DE_KIT_Tape", "dst_rse": "T2_US_MIT",
         "snapshot_date": "2026-08-01", "failed_count": 1, "bytes_total": 5},
    ])
    source = MONITRucioTransferSource(
        records_path="data/monit-rucio-transfer/records.json",
        token_env=TOKEN_ENV,
        base=str(tmp_path),
    )
    facts = list(source.run("r").facts)
    (node,) = _nodes(facts, "transfer_job")
    assert node.attrs["done_count"] == 3
    assert node.attrs["failed_count"] == 1
    assert node.attrs["total_bytes"] == 15


# --- Rucio datasets -------------------------------------------------------------

def _dataset_source(tmp_path, **overrides):
    kwargs = {
        "records_path": "data/monit-rucio-datasets/records.json",
        "page_cache_dir": "data/monit-rucio-datasets/pages",
        "token_env": TOKEN_ENV,
        "base": str(tmp_path),
    }
    kwargs.update(overrides)
    return MONITRucioDatasetSource(**kwargs)


def test_rucio_dataset_cache_raw_response(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(
        tmp_path, "data/monit-rucio-datasets/records.json", DATASET_RESPONSE
    )
    run = _dataset_source(tmp_path).run("r", mode="scope_complete")
    facts = list(run.facts)
    # The GenericTTbar functional-test dataset is excluded.
    (node,) = _nodes(facts, "dataset")
    assert node.node_id == "dataset:/Prim/Era-v1/RECO"
    assert node.attrs["tier"] == "RECO"
    assert node.attrs["era"] == "Era"
    assert node.attrs["event_count"] == 5
    assert node.attrs["file_count"] == 2
    assert node.attrs["total_size_bytes"] == 100
    assert node.attrs["physics_group"] == "Tracker"
    assert node.attrs["replica_sites"] == ["T1_DE_KIT", "T2_US_MIT"]
    assert node.attrs["replica_rses"] == ["T1_DE_KIT_Disk", "T2_US_MIT"]
    assert node.attrs["overlay_source"] == "monit_rucio_daily_stats"
    hosts = _edges(facts, "hosts")
    assert {(e.src, e.attrs["relationship"]) for e in hosts} == {
        ("site:T1_DE_KIT", "dataset_replica"),
        ("site:T2_US_MIT", "dataset_replica"),
        ("se:T1_DE_KIT_Disk", "dataset_replica_rse"),
        ("se:T2_US_MIT", "dataset_replica_rse"),
    }
    assert run.completed_scope is True


def test_rucio_dataset_exclude_patterns_param(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(
        tmp_path, "data/monit-rucio-datasets/records.json", DATASET_RESPONSE
    )
    source = _dataset_source(tmp_path, exclude_dataset_patterns=("/Prim/",))
    facts = list(source.run("r").facts)
    (node,) = _nodes(facts, "dataset")
    assert node.node_id == "dataset:/GenericTTbar/x/GEN-SIM"


def test_rucio_dataset_missing_credential(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    run = _dataset_source(tmp_path).run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "missing_credential"


def test_rucio_dataset_live_pagination_progress_and_page_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    page2_bucket = {
        "key": {"dataset": "/Second/Era-v1/AOD"},
        "replicas": {"buckets": [{"key": "T2_CH_CERN"}]},
        "sample": {"hits": {"hits": []}},
    }
    page1 = {"responses": [{"aggregations": {"datasets": {
        "buckets": [DATASET_BUCKET],
        "after_key": {"dataset": "/Prim/Era-v1/RECO"},
    }}}]}
    page2 = {"responses": [{"aggregations": {"datasets": {
        "buckets": [page2_bucket],
    }}}]}

    def _respond(url, headers, data):
        query = json.loads(data.split("\n")[1])
        composite = query["aggs"]["datasets"]["composite"]
        return page2 if "after" in composite else page1

    calls = _fake_post(monkeypatch, _respond)
    source = _dataset_source(tmp_path)  # no records cache -> live stream
    run = source.run("r", mode="scope_complete")
    facts = list(run.facts)
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.mode == "live"
    nodes = _nodes(facts, "dataset")
    assert {n.node_id for n in nodes} == {
        "dataset:/Prim/Era-v1/RECO",
        "dataset:/Second/Era-v1/AOD",
    }
    markers = [f for f in facts if isinstance(f, ProgressMarker)]
    assert {m.record_id for m in markers} == {
        "/Prim/Era-v1/RECO",
        "/Second/Era-v1/AOD",
    }
    assert len(calls) == 2
    # Live pages were cached under the query fingerprint...
    pages = sorted(
        (tmp_path / "data" / "monit-rucio-datasets" / "pages").rglob("*.json")
    )
    assert [p.name for p in pages] == ["page-000000.json", "page-000001.json"]
    # ...and a rerun with matching progress skips re-emission but still
    # yields the per-record markers (and reads from the page cache).
    progress = {f"record:{m.record_id}": m.fingerprint for m in markers}
    rerun = source.run("r2", mode="scope_complete", since_progress=progress)
    rerun_facts = list(rerun.facts)
    assert _nodes(rerun_facts, "dataset") == []
    assert len(
        [f for f in rerun_facts if isinstance(f, ProgressMarker)]
    ) == 2
    assert len(calls) == 2  # no new HTTP reads


def test_rucio_dataset_list_cache_records(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-rucio-datasets/records.json", [{
        "dataset": "/Prim/Era-v1/RECO",
        "snapshot_date": "2026-08-01",
        "data_tier": "RECO",
        "replica_sites": ["T1_DE_KIT_Disk"],
        "replica_rses": ["T1_DE_KIT_Disk"],
    }])
    facts = list(_dataset_source(tmp_path).run("r").facts)
    (node,) = _nodes(facts, "dataset")
    assert node.attrs["replica_sites"] == ["T1_DE_KIT"]
    assert node.attrs["replica_rses"] == ["T1_DE_KIT_Disk"]


def test_required_kwonly_and_no_positional_params():
    with pytest.raises(TypeError):
        MONITSAMSource("positional")  # keyword-only constructor


# --- adversarial-review regressions (pact/changes/circleback-fixes) ----------

def test_sam_live_timed_out_response_degrades_scope(tmp_path, monkeypatch):
    """Finding 2: an HTTP-200 response with timed_out=true must emit its
    data but report endpoint_failed and never claim a completed scope."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    partial = copy.deepcopy(SITEMON_RESPONSE)
    partial["responses"][0]["timed_out"] = True
    _fake_post(monkeypatch, lambda *a: partial)
    run = _sam_source(tmp_path).run("r", mode="scope_complete")
    facts = list(run.facts)
    assert len(_nodes(facts, "monitoring_snapshot")) == 2  # data emitted
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "timed_out" in run.health.reason


def test_condor_live_shard_failures_degrade_scope(tmp_path, monkeypatch):
    """Finding 2: _shards.failed > 0 means an unknown slice of the window
    is missing; the run must not claim a completed scope."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    partial = copy.deepcopy(CONDOR_RESPONSE)
    partial["responses"][0]["_shards"] = {
        "total": 40, "successful": 12, "skipped": 0, "failed": 28,
    }
    _fake_post(monkeypatch, lambda *a: partial)
    source = MONITCondorSource(
        records_path="data/monit-condor/records.json",
        token_env=TOKEN_ENV,
        base=str(tmp_path),
    )
    run = source.run("r", mode="reconcile")
    facts = list(run.facts)
    assert len(_nodes(facts, "monitoring_snapshot")) == 1  # data emitted
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "shards failed 28/40" in run.health.reason


def test_transfer_live_terms_truncation_degrades_scope(tmp_path, monkeypatch):
    """Finding 3: sum_other_doc_count > 0 on a bucket-limited terms agg
    means sites were silently dropped; never claim the scope."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    partial = copy.deepcopy(TRANSFER_RESPONSE)
    partial["responses"][0]["aggregations"]["by_src_rse"][
        "sum_other_doc_count"
    ] = 123456
    _fake_post(monkeypatch, lambda *a: partial)
    source = MONITRucioTransferSource(
        records_path="data/monit-rucio-transfer/records.json",
        token_env=TOKEN_ENV,
        base=str(tmp_path),
    )
    run = source.run("r", mode="scope_complete")
    facts = list(run.facts)
    assert _nodes(facts, "transfer_job")  # data still emitted
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "by_src_rse" in run.health.reason
    assert "sum_other_doc_count=123456" in run.health.reason


def test_nested_terms_truncation_is_detected(tmp_path, monkeypatch):
    """Finding 3: truncation markers on nested (per-bucket) aggregations
    are found by the recursive walk, not just top-level aggs."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    partial = copy.deepcopy(SITEMON_RESPONSE)
    groups = partial["responses"][0]["aggregations"]["groups"]
    groups["buckets"][0]["status_breakdown"][
        "doc_count_error_upper_bound"
    ] = 7
    _fake_post(monkeypatch, lambda *a: partial)
    run = _sam_source(tmp_path).run("r", mode="scope_complete")
    list(run.facts)
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "groups.status_breakdown" in run.health.reason


def test_cached_raw_response_with_partial_markers_degrades_scope(
    tmp_path, monkeypatch
):
    """Finding 2 (cache variant): a records cache holding a raw _msearch
    response that was itself partial must not claim a completed scope."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    partial = copy.deepcopy(SITEMON_RESPONSE)
    partial["responses"][0]["timed_out"] = True
    _write(tmp_path, "data/monit-sam/records.json", partial)
    run = _sam_source(tmp_path).run("r", mode="scope_complete")
    facts = list(run.facts)
    assert len(_nodes(facts, "monitoring_snapshot")) == 2
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_sam_live_zero_buckets_never_claims_empty_scope(
    tmp_path, monkeypatch
):
    """Finding 4: with ignore_unavailable wildcard queries a zero-bucket
    result is indistinguishable from a renamed index or stalled
    pipeline; claiming a complete empty scope would retract everything."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    _fake_post(monkeypatch, lambda *a: {"responses": [{
        "timed_out": False,
        "_shards": {"total": 40, "successful": 40, "failed": 0},
        "aggregations": {"groups": {"buckets": []}},
    }]})
    run = _sam_source(tmp_path).run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "skipped_optional"
    assert "no complete scope" in run.health.reason


def test_sam_empty_cache_never_claims_scope(tmp_path, monkeypatch):
    """Finding 4 (cache variant): an empty records cache also never
    claims a complete empty scope."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _write(tmp_path, "data/monit-sam/records.json", [])
    run = _sam_source(tmp_path).run("r", mode="scope_complete")
    assert run.completed_scope is False
    assert run.health.status == "skipped_optional"


def test_rucio_dataset_live_zero_buckets_never_claims_empty_scope(
    tmp_path, monkeypatch
):
    """Finding 4 (streaming path): a zero-bucket first page must return a
    skipped_optional run with completed_scope=False, not a completed
    empty stream."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    empty = {"responses": [{"aggregations": {"datasets": {"buckets": []}}}]}
    calls = _fake_post(monkeypatch, lambda *a: empty)
    run = _dataset_source(tmp_path).run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "skipped_optional"
    assert "no complete scope" in run.health.reason
    assert len(calls) == 1


def test_rucio_dataset_live_partial_first_page_fails_run(
    tmp_path, monkeypatch
):
    """Finding 2 (streaming path): a degraded first page fails the run
    before any facts or scope claims are produced, and the partial page
    is never cached for later replay."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    partial = copy.deepcopy(DATASET_RESPONSE)
    partial["responses"][0]["timed_out"] = True
    _fake_post(monkeypatch, lambda *a: partial)
    run = _dataset_source(tmp_path).run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "partial MONIT OpenSearch response" in run.health.reason
    pages_dir = tmp_path / "data" / "monit-rucio-datasets" / "pages"
    assert not pages_dir.exists() or not list(pages_dir.rglob("*.json"))


def test_rucio_dataset_live_partial_later_page_raises_midstream(
    tmp_path, monkeypatch
):
    """Finding 2 (streaming path): a degraded later page raises so the
    runner fails the run instead of committing a partial scope."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    page1 = {"responses": [{"aggregations": {"datasets": {
        "buckets": [DATASET_BUCKET],
        "after_key": {"dataset": "/Prim/Era-v1/RECO"},
    }}}]}
    page2 = {"responses": [{
        "timed_out": True,
        "aggregations": {"datasets": {"buckets": []}},
    }]}

    def _respond(url, headers, data):
        query = json.loads(data.split("\n")[1])
        composite = query["aggs"]["datasets"]["composite"]
        return page2 if "after" in composite else page1

    _fake_post(monkeypatch, _respond)
    run = _dataset_source(tmp_path).run("r", mode="scope_complete")
    with pytest.raises(RuntimeError, match="partial MONIT OpenSearch"):
        list(run.facts)


def test_rucio_dataset_page_cache_is_day_scoped(tmp_path, monkeypatch):
    """Finding 1 (BLOCKER): after one live run, a later day's run must
    re-query MONIT instead of replaying the previous day's cached pages
    (which would emit day-1's datasets stamped with day-2's date under a
    completed-scope claim)."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    state = {"dataset": "/Day1/Era-v1/RECO"}

    def _respond(url, headers, data):
        return {"responses": [{"aggregations": {"datasets": {"buckets": [{
            "key": {"dataset": state["dataset"]},
            "replicas": {"buckets": [{"key": "T2_CH_CERN"}]},
            "sample": {"hits": {"hits": []}},
        }]}}}]}

    calls = _fake_post(monkeypatch, _respond)
    # Freeze day 1 at the real current UTC date so the entries written
    # by the run (cached_at = real now) belong to "day 1".
    from datetime import datetime, timedelta, timezone

    day1 = datetime.now(timezone.utc)
    day2 = day1 + timedelta(days=1)
    monkeypatch.setattr(
        monit_mod, "_today", lambda: day1.strftime("%Y-%m-%d")
    )
    run1 = _dataset_source(tmp_path).run("d1", mode="scope_complete")
    names1 = {n.attrs["name"] for n in _nodes(list(run1.facts), "dataset")}
    assert names1 == {"/Day1/Era-v1/RECO"}
    assert len(calls) == 1

    # Same day, same query: the cache legitimately replays (crash/resume).
    rerun = _dataset_source(tmp_path).run("d1b", mode="scope_complete")
    assert {
        n.attrs["name"] for n in _nodes(list(rerun.facts), "dataset")
    } == names1
    assert len(calls) == 1  # no new HTTP

    # Next day the upstream changed completely: the run must NOT replay
    # day-1's pages and must emit day-2's datasets.
    state["dataset"] = "/Day2/Era-v1/RECO"
    monkeypatch.setattr(
        monit_mod, "_today", lambda: day2.strftime("%Y-%m-%d")
    )
    run2 = _dataset_source(tmp_path).run("d2", mode="scope_complete")
    names2 = {n.attrs["name"] for n in _nodes(list(run2.facts), "dataset")}
    assert names2 == {"/Day2/Era-v1/RECO"}
    assert len(calls) == 2  # fresh HTTP on the new day
    assert run2.completed_scope is True


def test_rucio_dataset_page_cache_honors_ttl(tmp_path, monkeypatch):
    """Finding 1: even a same-day cache entry only satisfies a run while
    it is younger than page_cache_ttl_hours."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: DATASET_RESPONSE)
    list(_dataset_source(tmp_path).run("r1", mode="scope_complete").facts)
    assert len(calls) == 1
    # Same day, default TTL: replayed.
    list(_dataset_source(tmp_path).run("r2", mode="scope_complete").facts)
    assert len(calls) == 1
    # TTL exceeded: the same-day entry no longer satisfies the run.
    time.sleep(0.01)
    strict = _dataset_source(tmp_path, page_cache_ttl_hours=0.0)
    list(strict.run("r3", mode="scope_complete").facts)
    assert len(calls) == 2


def test_rucio_dataset_page_cache_entry_without_cached_at_is_stale(
    tmp_path, monkeypatch
):
    """Finding 1: an entry of unknown age (no/corrupt cached_at, e.g. a
    legacy cache) can never satisfy a run."""
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    calls = _fake_post(monkeypatch, lambda *a: DATASET_RESPONSE)
    list(_dataset_source(tmp_path).run("r1", mode="scope_complete").facts)
    assert len(calls) == 1
    pages = sorted(
        (tmp_path / "data" / "monit-rucio-datasets" / "pages").rglob("*.json")
    )
    assert pages
    for page in pages:
        entry = json.loads(page.read_text())
        entry.pop("cached_at", None)
        page.write_text(json.dumps(entry))
    list(_dataset_source(tmp_path).run("r2", mode="scope_complete").facts)
    assert len(calls) == 2
