"""req.catalog-downloaders.dbs -- the DBS dataset metadata downloader (issue #903), no network.

The answers are synthetic (primary datasets ``FakePD*``, an invented DN in
``create_by``) in the layout the DBS client documents (``dbs.apis.dbsClient``
``listDatasets`` with ``detail``, ``listFileSummaries``): ``datasets`` answers
a JSON list of dataset rows, ``[]`` for a name DBS does not know;
``filesummaries`` answers a list holding one row of ``num_file``,
``num_lumi``, ``num_block``, ``num_event`` and ``file_size``. The WMStats
cache the default scope comes from is written by the wmstats downloader's
own code from ReqMgr2-shaped answers.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from support.responses import FakeResponse, patch_session_get
from support.x509 import write_cert

from okg.deployment import EdgeFact, NodeFact

from archi.downloaders import _catalog, dbs, wmstats
from archi.downloaders._catalog import write_group_atomic
from archi.sources.dbs import DBSDatasetSource

BASE = "https://dbs.example/dbs/prod/global/DBSReader"
DN = "/DC=org/DC=example/OU=Users/CN=fakeuser/CN=000000/CN=Fake User"

RAW = "/FakePD/Run2026A-v1/RAW"
AOD = "/FakePD/Run2026A-ReReco-v1/AOD"
MINIAOD = "/FakePD/Run2026A-ReReco-v1/MINIAOD"
UNKNOWN = "/FakePD/Run2026A-NotYetRegistered-v1/NANOAOD"
PATTERN = "/*/Run2026B*/AOD"
PATTERN_HITS = ("/FakePD2/Run2026B-PromptReco-v1/AOD", "/FakePD3/Run2026B-PromptReco-v1/AOD")


def _row(name, *, access="VALID", created=1761990600):
    _, primary, processed, tier = name.split("/")
    return {
        "dataset": name,
        "dataset_id": abs(hash(name)) % 10_000_000,
        "primary_ds_name": primary,
        "primary_ds_type": "data",
        "processed_ds_name": processed,
        "data_tier_name": tier,
        "dataset_access_type": access,
        "physics_group_name": None,
        "acquisition_era_name": processed.split("-")[0],
        "processing_version": 1,
        "prep_id": None,
        "xtcrosssection": None,
        "creation_date": created,
        "create_by": DN,
        "last_modification_date": created + 3600,
        "last_modified_by": DN,
    }


def _summary(files, events, size):
    return [
        {"num_file": files, "num_lumi": files * 10, "num_block": 1, "num_event": events, "file_size": size}
    ]


SUMMARIES = {
    RAW: _summary(120, 2_400_000, 3_000_000_000_000),
    AOD: _summary(40, 2_399_000, 900_000_000_000),
    MINIAOD: _summary(10, 2_399_000, 120_000_000_000),
    PATTERN_HITS[0]: _summary(5, 50_000, 10_000_000_000),
    PATTERN_HITS[1]: _summary(7, 70_000, 14_000_000_000),
}


def _datasets_url(name):
    return f"{BASE}/datasets?dataset={name}&dataset_access_type=*&detail=1"


def _pattern_url(pattern):
    return f"{BASE}/datasets?dataset={pattern}&detail=1"


def _summary_url(name):
    return f"{BASE}/filesummaries?dataset={name}&validFileOnly=1"


def _answers():
    answers = {_datasets_url(name): [_row(name)] for name in (RAW, AOD, MINIAOD)}
    answers[_datasets_url(MINIAOD)] = [_row(MINIAOD, access="PRODUCTION")]
    answers[_datasets_url(UNKNOWN)] = []
    answers[_pattern_url(PATTERN)] = [_row(name) for name in PATTERN_HITS]
    answers.update({_summary_url(name): summary for name, summary in SUMMARIES.items()})
    return answers


def _wmstats_cache(root: Path, *requests) -> Path:
    """A WMStats cache written by the wmstats downloader's own collect()."""
    statuses = ("running-open",)
    urls = wmstats.status_urls("https://cmsweb.example", statuses)
    answer = {"result": [{r["RequestName"]: r for r in requests}]}
    records, authority = wmstats.collect(lambda url: answer, urls)
    directory = root / "wmstats-workflows"
    write_group_atomic(directory, {"records.json": records, "authority.json": authority})
    return directory / "records.json"


def _wf(name, input_dataset, outputs):
    doc = {
        "RequestName": name,
        "RequestType": "ReReco",
        "RequestStatus": "running-open",
        "OutputDatasets": list(outputs),
        "Requestor": "fakeuser",
        "RequestorDN": DN,
    }
    if input_dataset:
        doc["InputDataset"] = input_dataset
    return doc


WORKFLOWS = (
    _wf("fakeuser_ReReco_Run2026A_FakePD_261101_000001", RAW, [AOD, MINIAOD]),
    # Reads the same RAW; its output is not registered in DBS yet.
    _wf("fakeuser_ReReco_Run2026A_FakePD_261101_000002", RAW, [UNKNOWN]),
)


@pytest.fixture(autouse=True)
def _no_grid_ca_dir_and_no_sleep(tmp_path, monkeypatch):
    monkeypatch.setattr(_catalog, "GRID_CA_DIR", tmp_path / "no-grid-ca")
    monkeypatch.setattr(_catalog.time, "sleep", lambda seconds: None)


@pytest.fixture
def proxy(tmp_path, monkeypatch):
    path = tmp_path / "x509up_test"
    write_cert(path, not_after=datetime.now(timezone.utc) + timedelta(hours=12))
    monkeypatch.setenv("X509_USER_PROXY", str(path))
    return path


@pytest.fixture
def wmstats_records(tmp_path, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(root))
    return _wmstats_cache(root, *WORKFLOWS)


@pytest.fixture
def upstream(monkeypatch):
    return patch_session_get(monkeypatch, _answers())


def _main(output: Path, *extra: str) -> int:
    return dbs.main(["--base-url", BASE, "--output-dir", str(output), *extra])


def _records(output: Path):
    return json.loads((output / "records.json").read_text(encoding="utf-8"))


def _old_cache(output: Path) -> dict[str, bytes]:
    output.mkdir(parents=True)
    (output / "records.json").write_text(json.dumps([{"dataset": "/Old/Old-v1/AOD"}]) + "\n")
    return _snapshot(output)


def _snapshot(output: Path) -> dict[str, bytes]:
    """The cache files (the write lock file excluded)."""
    return {p.name: p.read_bytes() for p in output.iterdir() if not p.name.startswith(".")}


def _assert_untouched(output: Path, old: dict[str, bytes]) -> None:
    assert _snapshot(output) == old
    assert [p.name for p in output.iterdir() if p.name.startswith(".staging.")] == []


# --- the round trip -----------------------------------------------------


def test_download_feeds_the_reader_the_workflow_datasets(
    tmp_path, proxy, wmstats_records, upstream, capsys
):
    output = tmp_path / "dbs-datasets"
    assert _main(output) == 0

    assert sorted(p.name for p in output.iterdir()) == [".lock", "records.json"]
    run = DBSDatasetSource(records_path=str(output / "records.json")).run(
        "run-1", mode="scope_complete"
    )
    facts = list(run.facts)
    assert run.health.status == "ok"
    assert run.completed_scope is True
    assert run.health.record_count == 3
    nodes = {f.node_id: f.attrs for f in facts if isinstance(f, NodeFact)}
    assert set(nodes) == {f"dataset:{RAW}", f"dataset:{AOD}", f"dataset:{MINIAOD}"}
    assert nodes[f"dataset:{AOD}"]["tier"] == "AOD"
    assert nodes[f"dataset:{AOD}"]["total_events"] == 2_399_000
    assert nodes[f"dataset:{AOD}"]["total_files"] == 40
    assert nodes[f"dataset:{AOD}"]["total_size_bytes"] == 900_000_000_000
    assert nodes[f"dataset:{MINIAOD}"]["dataset_access_type"] == "PRODUCTION"
    edges = {(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)}
    assert (f"dataset:{AOD}", "derives_from", f"dataset:{RAW}") in edges

    summary = json.loads(capsys.readouterr().out)
    assert summary["records"] == 3
    assert summary["workflow_datasets"] == 4
    assert summary["unresolved"] == 1
    assert summary["unresolved_sample"] == [UNKNOWN]
    assert summary["patterns"] == {}
    assert summary["requests"] == 4 + 3
    assert summary["output"] == str(output)


def test_a_record_holds_only_the_metadata_the_reader_reads(tmp_path, proxy, wmstats_records, upstream):
    output = tmp_path / "dbs-datasets"
    assert _main(output) == 0
    records = {record["dataset"]: record for record in _records(output)}
    assert records[AOD] == {
        "dataset": AOD,
        "primary_ds_name": "FakePD",
        "processed_ds_name": "Run2026A-ReReco-v1",
        "data_tier_name": "AOD",
        "dataset_access_type": "VALID",
        "physics_group_name": None,
        "creation_date": 1761990600,
        "nevents": 2_399_000,
        "nfiles": 40,
        "dataset_size": 900_000_000_000,
    }
    # No DN or user name of whoever registered the dataset.
    assert DN not in (output / "records.json").read_text(encoding="utf-8")


def test_the_scope_is_the_workflow_datasets_plus_patterns_each_requested_once(
    tmp_path, proxy, wmstats_records, upstream, capsys
):
    # One pattern hit is also a workflow dataset: it is asked for once.
    upstream.answers[_pattern_url(PATTERN)].append(_row(AOD))
    output = tmp_path / "dbs-datasets"
    assert _main(output, "--pattern", PATTERN, "--pattern", PATTERN) == 0

    urls = [url for url, _ in upstream.calls]
    assert len(urls) == len(set(urls))
    assert sorted(urls) == sorted(
        [
            _pattern_url(PATTERN),
            _datasets_url(RAW),
            _datasets_url(MINIAOD),
            _datasets_url(UNKNOWN),
            _summary_url(RAW),
            _summary_url(AOD),
            _summary_url(MINIAOD),
            _summary_url(PATTERN_HITS[0]),
            _summary_url(PATTERN_HITS[1]),
        ]
    )
    # Metadata only: no file, lumi, block or run listing is ever called.
    endpoints = {url[len(BASE) + 1 :].split("?")[0] for url in urls}
    assert endpoints == {"datasets", "filesummaries"}
    assert {record["dataset"] for record in _records(output)} == {
        RAW,
        AOD,
        MINIAOD,
        *PATTERN_HITS,
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["patterns"] == {PATTERN: 3}
    assert summary["records"] == 5


def test_patterns_alone_need_no_wmstats_cache(tmp_path, proxy, upstream, monkeypatch, capsys):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "empty-root"))
    output = tmp_path / "dbs-datasets"
    assert _main(output, "--pattern", PATTERN) == 0
    assert sorted(url for url, _ in upstream.calls) == sorted(
        [_pattern_url(PATTERN), *(_summary_url(name) for name in PATTERN_HITS)]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["workflow_datasets"] == 0
    missing = tmp_path / "empty-root" / "wmstats-workflows" / "records.json"
    assert summary["workflow_datasets_skipped"] == f"no WMStats cache at {missing}"


def test_an_explicit_wmstats_cache_that_is_missing_fails_even_with_patterns(
    tmp_path, proxy, upstream, capsys
):
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    missing = tmp_path / "nowhere" / "records.json"
    assert _main(output, "--wmstats-records", str(missing), "--pattern", PATTERN) == 1
    err = capsys.readouterr().err
    assert f"no WMStats cache at {missing}" in err
    assert upstream.calls == []
    _assert_untouched(output, old)


def test_a_relative_archi_data_root_finds_the_wmstats_cache(
    tmp_path, proxy, wmstats_records, upstream, monkeypatch, capsys
):
    # Review finding: the existence check and the reader resolved the
    # relative path against different bases (cwd vs data/data/...).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARCHI_DATA_ROOT", "root")
    assert _main(tmp_path / "dbs-datasets") == 0
    assert {record["dataset"] for record in _records(tmp_path / "dbs-datasets")} == {
        RAW,
        AOD,
        MINIAOD,
    }
    assert json.loads(capsys.readouterr().out)["workflow_datasets"] == 4


def test_a_relative_wmstats_records_path_is_taken_under_the_data_root(
    tmp_path, proxy, upstream, monkeypatch
):
    # Review finding: it only existed under the data root, so the old check
    # missed it and wrote a pattern-only catalog.
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "root"))
    _wmstats_cache(tmp_path / "root" / "elsewhere", *WORKFLOWS)
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "dbs-datasets"
    relative = "elsewhere/wmstats-workflows/records.json"
    assert _main(output, "--wmstats-records", relative, "--pattern", PATTERN) == 0
    assert {record["dataset"] for record in _records(output)} == {RAW, AOD, MINIAOD, *PATTERN_HITS}


def test_every_request_carries_the_proxy(tmp_path, proxy, wmstats_records, upstream):
    assert _main(tmp_path / "dbs-datasets", "--timeout", "17") == 0
    assert upstream.calls
    for _, kwargs in upstream.calls:
        assert kwargs["cert"] == (str(proxy), str(proxy))
        assert kwargs["timeout"] == 17


def test_the_default_output_is_dbs_datasets_under_archi_data_root(
    tmp_path, proxy, wmstats_records, upstream
):
    assert dbs.main(["--base-url", BASE]) == 0
    assert (tmp_path / "root" / "dbs-datasets" / "records.json").is_file()


def test_an_explicit_wmstats_cache_path_is_read(tmp_path, proxy, upstream, monkeypatch):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "empty-root"))
    records = _wmstats_cache(tmp_path / "elsewhere", WORKFLOWS[0])
    output = tmp_path / "dbs-datasets"
    assert _main(output, "--wmstats-records", str(records)) == 0
    assert {record["dataset"] for record in _records(output)} == {RAW, AOD, MINIAOD}


def test_limit_caps_the_datasets_looked_up_and_needs_an_explicit_output(
    tmp_path, proxy, wmstats_records, upstream, capsys
):
    output = tmp_path / "trial"
    assert _main(output, "--limit", "3") == 0
    # The first three names in sorted order: the unknown NANOAOD, the AOD and
    # the MINIAOD; RAW is never asked for.
    assert {record["dataset"] for record in _records(output)} == {AOD, MINIAOD}
    assert sorted(url for url, _ in upstream.calls) == sorted(
        [
            _datasets_url(UNKNOWN),
            _datasets_url(AOD),
            _datasets_url(MINIAOD),
            _summary_url(AOD),
            _summary_url(MINIAOD),
        ]
    )
    assert json.loads(capsys.readouterr().out)["limit"] == 3


def test_limit_refuses_a_relative_output_dir_named_dbs_datasets(
    tmp_path, proxy, wmstats_records, upstream, monkeypatch, capsys
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(SystemExit) as exc:
        dbs.main(["--base-url", BASE, "--limit", "2", "--output-dir", "dbs-datasets"])
    assert exc.value.code == 2
    assert upstream.calls == []
    assert "--limit" in capsys.readouterr().err


def test_limit_refuses_a_symlink_to_a_dbs_datasets_dir(tmp_path, proxy, wmstats_records, upstream, capsys):
    real = tmp_path / "other-root" / "dbs-datasets"
    real.mkdir(parents=True)
    link = tmp_path / "trial"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SystemExit) as exc:
        _main(link, "--limit", "2")
    assert exc.value.code == 2
    assert upstream.calls == []
    assert "--limit" in capsys.readouterr().err


@pytest.mark.parametrize("explicit_default", [False, True])
def test_limit_refuses_the_packs_cache(tmp_path, proxy, wmstats_records, upstream, capsys, explicit_default):
    argv = ["--base-url", BASE, "--limit", "2"]
    if explicit_default:
        argv += ["--output-dir", str(tmp_path / "root" / "dbs-datasets")]
    with pytest.raises(SystemExit) as exc:
        dbs.main(argv)
    assert exc.value.code == 2
    assert upstream.calls == []
    assert "--limit" in capsys.readouterr().err


# --- fail closed --------------------------------------------------------


def test_no_wmstats_cache_and_no_pattern_names_the_wmstats_downloader(
    tmp_path, proxy, upstream, monkeypatch, capsys
):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "empty-root"))
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert "python -m archi.downloaders.wmstats" in err
    assert "--pattern" in err
    assert upstream.calls == []
    _assert_untouched(output, old)


def test_an_unusable_wmstats_cache_stops_the_run(tmp_path, proxy, wmstats_records, upstream, capsys):
    (wmstats_records.parent / "authority.json").write_text(json.dumps({"record_count": 99}))
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output, "--pattern", PATTERN) == 1
    err = capsys.readouterr().err
    assert str(wmstats_records) in err
    assert "python -m archi.downloaders.wmstats" in err
    assert upstream.calls == []
    _assert_untouched(output, old)


def test_no_proxy_stops_before_any_request(tmp_path, wmstats_records, upstream, monkeypatch, capsys):
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "absent"))
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output) == 1
    assert upstream.calls == []
    assert "voms-proxy-init -voms cms" in capsys.readouterr().err
    _assert_untouched(output, old)


def test_an_expired_proxy_stops_before_any_request(tmp_path, wmstats_records, upstream, monkeypatch, capsys):
    expired = tmp_path / "expired"
    write_cert(expired, not_after=datetime.now(timezone.utc) - timedelta(minutes=5))
    monkeypatch.setenv("X509_USER_PROXY", str(expired))
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output) == 1
    assert upstream.calls == []
    assert "expired" in capsys.readouterr().err
    _assert_untouched(output, old)


@pytest.mark.parametrize(
    "url, response, message",
    [
        (
            _datasets_url(RAW),
            FakeResponse(302, "", {"Location": "https://auth.example/login"}),
            "HTTP 302",
        ),
        (_summary_url(AOD), FakeResponse(503, "Service Unavailable"), "HTTP 503"),
        (_datasets_url(AOD), FakeResponse(400, '[{"error": {"reason": "bad"}}]'), "HTTP 400"),
        (_summary_url(RAW), FakeResponse(200, "<html>Sign in</html>"), "not JSON"),
    ],
    ids=["login-redirect", "http-error", "dbs-error", "html-body"],
)
def test_a_failed_fetch_writes_nothing(
    tmp_path, proxy, wmstats_records, upstream, capsys, url, response, message
):
    upstream.answers[url] = response
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert err.splitlines()[-1].startswith("dbs download failed, cache not written: ")
    assert url in err
    assert message in err
    assert str(proxy) not in err
    _assert_untouched(output, old)


@pytest.mark.parametrize(
    "url, answer, message",
    [
        (_datasets_url(RAW), {"error": "bad"}, "not a list of dataset rows"),
        (_datasets_url(RAW), ["/FakePD/Run2026A-v1/RAW"], "not a list of dataset rows"),
        (_datasets_url(RAW), [_row(AOD)], f"asked for {RAW}"),
        (_pattern_url(PATTERN), [{"dataset_id": 1}], "not a list of dataset rows"),
        (_summary_url(AOD), {"num_file": 1}, "not a list of at most one summary row"),
        (_summary_url(AOD), _summary(1, 1, 1) * 2, "not a list of at most one summary row"),
    ],
    ids=["error-body", "names-only", "other-dataset", "pattern-row-no-name", "summary-object", "two-summaries"],
)
def test_a_malformed_answer_writes_nothing(
    tmp_path, proxy, wmstats_records, upstream, capsys, url, answer, message
):
    upstream.answers[url] = answer
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output, "--pattern", PATTERN) == 1
    err = capsys.readouterr().err
    assert url in err
    assert message in err
    _assert_untouched(output, old)


def test_no_dataset_resolving_writes_nothing(tmp_path, proxy, wmstats_records, monkeypatch, capsys):
    upstream = patch_session_get(
        monkeypatch, {_datasets_url(name): [] for name in (RAW, AOD, MINIAOD, UNKNOWN)}
    )
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert "none of the 4 datasets is known to DBS" in err
    assert BASE in err
    assert len(upstream.calls) == 4
    _assert_untouched(output, old)


def test_an_empty_summary_reads_as_zero_counts(tmp_path, proxy, wmstats_records, upstream):
    # A PRODUCTION dataset with no valid file yet.
    upstream.answers[_summary_url(MINIAOD)] = []
    output = tmp_path / "dbs-datasets"
    assert _main(output) == 0
    records = {record["dataset"]: record for record in _records(output)}
    assert (records[MINIAOD]["nfiles"], records[MINIAOD]["nevents"], records[MINIAOD]["dataset_size"]) == (0, 0, 0)


def test_a_payload_the_reader_refuses_writes_nothing(tmp_path, proxy, wmstats_records, upstream, capsys):
    # The reader casts the event count to int and fails on this one.
    upstream.answers[_summary_url(AOD)] = _summary(40, "unknown", 1)
    output = tmp_path / "dbs-datasets"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert "the reader failed on the downloaded answers" in err
    assert "unknown" in err
    assert BASE in err
    assert ".staging." not in err
    _assert_untouched(output, old)


def test_a_summary_row_with_null_counts_is_kept_and_reads_as_zero(
    tmp_path, proxy, wmstats_records, upstream
):
    # DBS answers null events and size when no file has an event count.
    upstream.answers[_summary_url(MINIAOD)] = [
        {"num_file": 3, "num_lumi": None, "num_block": 1, "num_event": None, "file_size": None}
    ]
    output = tmp_path / "dbs-datasets"
    assert _main(output) == 0
    record = {r["dataset"]: r for r in _records(output)}[MINIAOD]
    assert (record["nfiles"], record["nevents"], record["dataset_size"]) == (3, None, None)
    run = DBSDatasetSource(records_path=str(output / "records.json")).run("r", mode="scope_complete")
    node = {f.node_id: f.attrs for f in run.facts if isinstance(f, NodeFact)}[f"dataset:{MINIAOD}"]
    assert (node["total_files"], node["total_events"], node["total_size_bytes"]) == (3, 0, 0)


def test_names_and_patterns_are_url_encoded(tmp_path, proxy, monkeypatch):
    odd = "/Fake+PD/Run2026A-v1#2/RAW"
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "root"))
    _wmstats_cache(tmp_path / "root", _wf("fakeuser_ReReco_Odd_000009", odd, []))
    pattern = "/Fake+PD*/Run2026A*/RAW"
    encoded = "/Fake%2BPD/Run2026A-v1%232/RAW"
    upstream = patch_session_get(
        monkeypatch,
        {
            f"{BASE}/datasets?dataset=/Fake%2BPD*/Run2026A*/RAW&detail=1": [],
            f"{BASE}/datasets?dataset={encoded}&dataset_access_type=*&detail=1": [_row(odd)],
            f"{BASE}/filesummaries?dataset={encoded}&validFileOnly=1": _summary(1, 10, 100),
        },
    )
    output = tmp_path / "dbs-datasets"
    assert _main(output, "--pattern", pattern) == 0
    assert len(upstream.calls) == 3
    assert [r["dataset"] for r in _records(output)] == [odd]


def test_progress_goes_to_stderr_and_stdout_stays_the_summary(tmp_path, proxy, monkeypatch, capsys):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "empty-root"))
    names = [f"/FakeMany/Run2026C-v{i}/AOD" for i in range(450)]
    answers = {_pattern_url("/FakeMany/*/AOD"): [_row(name) for name in names]}
    answers.update({_summary_url(name): _summary(1, 1, 1) for name in names})
    patch_session_get(monkeypatch, answers)
    assert _main(tmp_path / "dbs-datasets", "--pattern", "/FakeMany/*/AOD") == 0
    out, err = capsys.readouterr()
    summary = json.loads(out)
    assert summary["records"] == 450
    lines = [line for line in err.splitlines() if line.startswith("dbs: ")]
    assert [line.split(" datasets")[0] for line in lines] == [
        "dbs: 200/450",
        "dbs: 400/450",
        "dbs: 450/450",
    ]
    assert "201 requests" in lines[0]
