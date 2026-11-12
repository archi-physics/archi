"""req.catalog-downloaders.wmstats -- the WMStats workflow downloader (issue #903), no network.

The answers are synthetic (requestor ``fakeuser``, primary datasets
``FakePD*``, an invented DN) in the layout of the ReqMgr2 request API,
``/reqmgr2/data/request?status=<status>`` (WMCore
``ReqMgr/Service/Request.py``, default ``common_dict=0``): ``{"result":
[{<RequestName>: <request document>}]}``, or ``{"result": []}`` when no
request is in that status. That is the per-status answer the WMStats
collection shape holds (pact ``wmstats-collection-shape``). A TaskChain
document keeps its per-task fields under ``Task1``, as ReqMgr2 stores it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from support.responses import FakeResponse, patch_session_get
from support.x509 import write_cert

from okg.deployment import EdgeFact, NodeFact

from archi.downloaders import _catalog, wmstats
from archi.sources.wmstats import WMStatsWorkflowSource

BASE = "https://cmsweb.example"
DEFAULT_STATUSES = (
    "assigned",
    "staging",
    "staged",
    "acquired",
    "running-open",
    "running-closed",
    "force-complete",
    "completed",
    "closed-out",
)
DN = "/DC=org/DC=example/OU=Users/CN=fakeuser/CN=000000/CN=Fake User"


def _url(status: str) -> str:
    return f"{BASE}/reqmgr2/data/request?status={status}"


def _request(name, status, *, request_type="ReReco", input_dataset="", outputs=(), **extra):
    doc = {
        "_id": name,
        "RequestName": name,
        "RequestType": request_type,
        "RequestStatus": status,
        "RequestPriority": 110000,
        "Campaign": "Run2026FakeCampaign",
        "PrepID": f"FAKE-{name[-6:]}",
        "CMSSWVersion": "CMSSW_15_0_1",
        "OutputDatasets": list(outputs),
        "RequestDate": [2026, 11, 1, 9, 30, 0],
        "Requestor": "fakeuser",
        "RequestorDN": DN,
        "RequestTransition": [
            {"Status": "new", "UpdateTime": 1761990600, "DN": DN},
            {"Status": status, "UpdateTime": 1762000000, "DN": DN},
        ],
        **extra,
    }
    if input_dataset:
        doc["InputDataset"] = input_dataset
    return doc


RERECO = _request(
    "fakeuser_ReReco_Run2026A_FakePD_261101_000001",
    "running-open",
    input_dataset="/FakePD/Run2026A-v1/RAW",
    outputs=["/FakePD/Run2026A-ReReco-v1/AOD", "/FakePD/Run2026A-ReReco-v1/MINIAOD"],
)
STEPCHAIN = _request(
    "fakeuser_SC_FakePD2_Run2026B_261102_000002",
    "assigned",
    request_type="StepChain",
    outputs=["/FakePD2/Run2026B-Fake-v1/NANOAOD"],
    StepChain=1,
    Step1={"StepName": "FakeStep", "InputDataset": "/FakePD2/Run2026B-v1/MINIAOD"},
)
COMPLETED = _request(
    "fakeuser_MC_FakeGen_261103_000003",
    "completed",
    request_type="MonteCarlo",
    outputs=["/FakeGen_13p6TeV/Run2026Fake-v1/GEN-SIM"],
)


def _answer(*docs):
    return {"result": [{doc["RequestName"]: doc for doc in docs}] if docs else []}


def _answers(*docs):
    by_status = {status: [] for status in DEFAULT_STATUSES}
    for doc in docs:
        by_status[doc["RequestStatus"]].append(doc)
    return {_url(status): _answer(*found) for status, found in by_status.items()}


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
def upstream(monkeypatch):
    return patch_session_get(monkeypatch, _answers(RERECO, STEPCHAIN, COMPLETED))


def _main(output: Path, *extra: str) -> int:
    return wmstats.main(["--base-url", BASE, "--output-dir", str(output), *extra])


def _reader(output: Path) -> WMStatsWorkflowSource:
    return WMStatsWorkflowSource(records_path=str(output / "records.json"))


def _read(output: Path, name: str):
    return json.loads((output / name).read_text(encoding="utf-8"))


def _old_cache(output: Path) -> dict[str, bytes]:
    output.mkdir(parents=True)
    (output / "records.json").write_text(json.dumps({"old": True}) + "\n", encoding="utf-8")
    (output / "authority.json").write_text(json.dumps({"record_count": 7}) + "\n", encoding="utf-8")
    return _snapshot(output)


def _snapshot(output: Path) -> dict[str, bytes]:
    """The cache files (the write lock file excluded)."""
    return {p.name: p.read_bytes() for p in output.iterdir() if not p.name.startswith(".")}


def _assert_untouched(output: Path, old: dict[str, bytes]) -> None:
    assert _snapshot(output) == old
    assert [p.name for p in output.iterdir() if p.name.startswith(".staging.")] == []


# --- the round trip -----------------------------------------------------


def test_download_feeds_the_reader_every_workflow_with_a_complete_scope(
    tmp_path, proxy, upstream, capsys
):
    output = tmp_path / "wmstats-workflows"
    assert _main(output) == 0

    assert sorted(p.name for p in output.iterdir()) == [".lock", "authority.json", "records.json"]
    run = _reader(output).run("run-1", mode="scope_complete")
    facts = list(run.facts)
    assert run.health.status == "ok"
    assert run.completed_scope is True
    assert run.health.record_count == 3
    workflows = {
        f.node_id: f.attrs for f in facts if isinstance(f, NodeFact) and f.subtype == "workflow"
    }
    assert set(workflows) == {
        "workflow:fakeuser_ReReco_Run2026A_FakePD_261101_000001",
        "workflow:fakeuser_SC_FakePD2_Run2026B_261102_000002",
        "workflow:fakeuser_MC_FakeGen_261103_000003",
    }
    rereco = workflows["workflow:fakeuser_ReReco_Run2026A_FakePD_261101_000001"]
    assert rereco["status"] == "running-open"
    assert rereco["priority"] == 110000
    edges = {(e.src, e.edge_type, e.dst) for e in facts if isinstance(e, EdgeFact)}
    assert (
        "workflow:fakeuser_ReReco_Run2026A_FakePD_261101_000001",
        "consumes",
        "dataset:/FakePD/Run2026A-v1/RAW",
    ) in edges
    assert (
        "workflow:fakeuser_MC_FakeGen_261103_000003",
        "depends_on",
        "cmssw_release:CMSSW_15_0_1",
    ) in edges

    summary = json.loads(capsys.readouterr().out)
    assert summary["records"] == 3
    assert summary["deliberate_merge_count"] == 0
    assert summary["statuses"] == {
        "assigned": 1,
        "staging": 0,
        "staged": 0,
        "acquired": 0,
        "running-open": 1,
        "running-closed": 0,
        "force-complete": 0,
        "completed": 1,
        "closed-out": 0,
    }
    assert summary["output"] == str(output)


def test_the_cache_is_the_collection_shape_with_its_authority_count(tmp_path, proxy, upstream):
    output = tmp_path / "wmstats-workflows"
    assert _main(output) == 0

    records = _read(output, "records.json")
    assert sorted(records) == ["cutoff_utc", "status_payloads"]
    datetime.strptime(records["cutoff_utc"], "%Y-%m-%dT%H:%M:%SZ")
    # Each status keeps the ReqMgr2 answer as sent, the empty ones too (the
    # file's keys are sorted; the reader holds no name twice, so order is moot).
    assert sorted(records["status_payloads"]) == sorted(DEFAULT_STATUSES)
    for status in DEFAULT_STATUSES:
        assert records["status_payloads"][status] == upstream.answers[_url(status)]
    assert _read(output, "authority.json") == {
        "record_count": 3,
        "deliberate_merge_count": 0,
        "statuses": list(DEFAULT_STATUSES),
    }


def test_one_request_per_status_in_transition_order_each_with_the_proxy(
    tmp_path, proxy, upstream
):
    assert _main(tmp_path / "wmstats-workflows", "--timeout", "17") == 0
    assert [url for url, _ in upstream.calls] == [_url(status) for status in DEFAULT_STATUSES]
    for _, kwargs in upstream.calls:
        assert kwargs["cert"] == (str(proxy), str(proxy))
        assert kwargs["timeout"] == 17


def test_status_flags_choose_the_statuses_queried_in_transition_order(
    tmp_path, proxy, upstream, capsys
):
    output = tmp_path / "wmstats-workflows"
    assert _main(output, "--status", "completed", "--status", "assigned") == 0
    assert [url for url, _ in upstream.calls] == [_url("assigned"), _url("completed")]
    assert sorted(_read(output, "records.json")["status_payloads"]) == ["assigned", "completed"]
    assert _read(output, "authority.json")["statuses"] == ["assigned", "completed"]
    assert json.loads(capsys.readouterr().out)["records"] == 2


def test_an_unknown_status_is_refused_before_any_request(tmp_path, proxy, upstream, capsys):
    with pytest.raises(SystemExit) as exc:
        _main(tmp_path / "wmstats-workflows", "--status", "runing-open")
    assert exc.value.code == 2
    assert upstream.calls == []
    assert "runing-open" in capsys.readouterr().err


def test_a_workflow_seen_under_two_statuses_keeps_the_later_answer(
    tmp_path, proxy, monkeypatch, capsys
):
    # It moved from running-open to running-closed between the two requests.
    moved = {**RERECO, "RequestStatus": "running-closed"}
    answers = _answers(RERECO, STEPCHAIN)
    answers[_url("running-closed")] = _answer(moved)
    patch_session_get(monkeypatch, answers)
    output = tmp_path / "wmstats-workflows"
    assert _main(output) == 0

    records = _read(output, "records.json")["status_payloads"]
    assert records["running-open"] == {"result": []}
    assert records["running-closed"] == _answer(moved)
    assert _read(output, "authority.json")["record_count"] == 2
    assert _read(output, "authority.json")["deliberate_merge_count"] == 1
    run = _reader(output).run("run-1", mode="scope_complete")
    nodes = {f.node_id: f for f in run.facts if isinstance(f, NodeFact)}
    assert run.completed_scope is True
    assert nodes["workflow:fakeuser_ReReco_Run2026A_FakePD_261101_000001"].attrs[
        "status"
    ] == "running-closed"
    summary = json.loads(capsys.readouterr().out)
    assert summary["deliberate_merge_count"] == 1
    assert summary["records"] == 2


def test_the_default_output_is_wmstats_workflows_under_archi_data_root(
    tmp_path, proxy, upstream, monkeypatch
):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path / "root"))
    assert wmstats.main(["--base-url", BASE]) == 0
    assert sorted(p.name for p in (tmp_path / "root" / "wmstats-workflows").iterdir()) == [
        ".lock",
        "authority.json",
        "records.json",
    ]


def test_an_explicit_proxy_overrides_the_environment(tmp_path, proxy, upstream, monkeypatch):
    other = tmp_path / "other-proxy"
    write_cert(other, not_after=datetime.now(timezone.utc) + timedelta(hours=1))
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "absent"))
    assert _main(tmp_path / "wmstats-workflows", "--proxy", str(other)) == 0
    assert {kwargs["cert"] for _, kwargs in upstream.calls} == {(str(other), str(other))}


# --- fail closed --------------------------------------------------------


def test_no_proxy_stops_before_any_request(tmp_path, upstream, monkeypatch, capsys):
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "absent"))
    output = tmp_path / "wmstats-workflows"
    old = _old_cache(output)
    assert _main(output) == 1
    assert upstream.calls == []
    assert "voms-proxy-init -voms cms" in capsys.readouterr().err
    _assert_untouched(output, old)


def test_an_expired_proxy_stops_before_any_request(tmp_path, upstream, monkeypatch, capsys):
    expired = tmp_path / "expired"
    write_cert(expired, not_after=datetime.now(timezone.utc) - timedelta(minutes=5))
    monkeypatch.setenv("X509_USER_PROXY", str(expired))
    output = tmp_path / "wmstats-workflows"
    old = _old_cache(output)
    assert _main(output) == 1
    assert upstream.calls == []
    assert "expired" in capsys.readouterr().err
    _assert_untouched(output, old)


@pytest.mark.parametrize(
    "status, response, message",
    [
        (
            "completed",
            FakeResponse(302, "", {"Location": "https://auth.example/login?next=/reqmgr2"}),
            "HTTP 302",
        ),
        ("running-open", FakeResponse(503, "Service Unavailable"), "HTTP 503"),
        ("staged", FakeResponse(200, "<html>Sign in</html>"), "not JSON"),
        ("assigned", FakeResponse(200, '{"result": [{"trunc'), "not JSON"),
    ],
    ids=["login-redirect", "http-error", "html-body", "truncated-json"],
)
def test_a_failed_fetch_writes_nothing(tmp_path, proxy, upstream, capsys, status, response, message):
    upstream.answers[_url(status)] = response
    output = tmp_path / "wmstats-workflows"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert err.splitlines()[-1].startswith("wmstats download failed, cache not written: ")
    assert _url(status) in err
    assert message in err
    assert str(proxy) not in err
    _assert_untouched(output, old)


@pytest.mark.parametrize(
    "status, answer, message",
    [
        ("completed", {"error": "couch view timeout"}, "no 'result' list"),
        ("completed", [{"x": 1}], "no 'result' list"),
        ("completed", {"result": {"wf": {}}}, "no 'result' list"),
        ("completed", {"result": ["fakeuser_MC_FakeGen_261103_000003"]}, "entry 0 is not"),
        (
            "completed",
            {"result": [{"fakeuser_MC_FakeGen_261103_000003": "completed"}]},
            "request fakeuser_MC_FakeGen_261103_000003 is not an object",
        ),
    ],
    ids=["error-body", "list-body", "result-not-list", "entry-not-object", "record-not-object"],
)
def test_a_malformed_answer_writes_nothing(tmp_path, proxy, upstream, capsys, status, answer, message):
    upstream.answers[_url(status)] = answer
    output = tmp_path / "wmstats-workflows"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert _url(status) in err
    assert message in err
    _assert_untouched(output, old)


def test_no_workflow_in_any_status_writes_nothing(tmp_path, proxy, monkeypatch, capsys):
    upstream = patch_session_get(monkeypatch, _answers())
    output = tmp_path / "wmstats-workflows"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert "no workflow in any of the 9 statuses" in err
    assert _url("assigned") in err
    assert len(upstream.calls) == len(DEFAULT_STATUSES)
    _assert_untouched(output, old)


def test_a_payload_the_reader_refuses_writes_nothing(tmp_path, proxy, upstream, capsys):
    # A request without a RequestName: the reader skips it and claims no scope.
    nameless = {k: v for k, v in COMPLETED.items() if k != "RequestName"}
    upstream.answers[_url("completed")] = {"result": [{COMPLETED["RequestName"]: nameless}]}
    output = tmp_path / "wmstats-workflows"
    old = _old_cache(output)
    assert _main(output) == 1
    err = capsys.readouterr().err
    assert "the reader refuses the downloaded answers" in err
    assert "completed_scope=False" in err
    assert _url("completed") in err
    assert ".staging." not in err
    _assert_untouched(output, old)


def test_a_workflow_seen_under_three_statuses_is_merged_twice(tmp_path, proxy, monkeypatch, capsys):
    answers = _answers(STEPCHAIN)
    for status in ("running-open", "running-closed", "force-complete"):
        answers[_url(status)] = _answer({**RERECO, "RequestStatus": status})
    patch_session_get(monkeypatch, answers)
    output = tmp_path / "wmstats-workflows"
    assert _main(output) == 0

    payloads = _read(output, "records.json")["status_payloads"]
    assert payloads["running-open"] == {"result": []}
    assert payloads["running-closed"] == {"result": []}
    assert payloads["force-complete"] == answers[_url("force-complete")]
    assert _read(output, "authority.json")["record_count"] == 2
    assert _read(output, "authority.json")["deliberate_merge_count"] == 2
    assert json.loads(capsys.readouterr().out)["deliberate_merge_count"] == 2


def test_progress_goes_to_stderr_once_per_status_and_stdout_stays_the_summary(
    tmp_path, proxy, upstream, capsys
):
    assert _main(tmp_path / "wmstats-workflows") == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["records"] == 3
    lines = [line for line in err.splitlines() if line.startswith("wmstats: ")]
    assert [line.split(",")[0] for line in lines] == [
        f"wmstats: {status} {index}/9" for index, status in enumerate(DEFAULT_STATUSES, start=1)
    ]
    assert "running-open 5/9, 1 workflows" in lines[4]


def test_a_refused_first_run_leaves_at_most_the_lock(tmp_path, proxy, upstream):
    nameless = {k: v for k, v in COMPLETED.items() if k != "RequestName"}
    upstream.answers[_url("completed")] = {"result": [{COMPLETED["RequestName"]: nameless}]}
    output = tmp_path / "wmstats-workflows"
    assert _main(output) == 1
    assert {p.name for p in output.iterdir()} <= {".lock"}
