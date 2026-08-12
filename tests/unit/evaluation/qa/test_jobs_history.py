import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.evaluation.qa.console as console_module
import src.evaluation.qa.history as history_module
import src.evaluation.qa.jobs as jobs_module
from src.evaluation.qa.console import EvaluationConsoleService
from src.evaluation.qa.history import EvaluationHistory
from src.evaluation.qa.jobs import EvaluationJobManager, JobConflictError
from src.evaluation.qa.schema import ConsoleMetadata, RunManifest

from src.evaluation.qa.artifacts import (  # isort: skip
    artifact_hashes,
    read_json,
    write_json,
    write_jsonl,
)


class _RetryWorkflow:
    def retry_plan(self, _parent_path):
        return {"retry_attempt_ids": ["item-1::attempt-1"]}

    def retry(self, parent_path, output_dir):
        for name in ("input.snapshot.json", "preparation.jsonl"):
            (output_dir / name).write_bytes((parent_path / name).read_bytes())
        write_json(output_dir / "summary.json", {"overall_attempt_pass_rate": 1.0})
        (output_dir / "report.md").write_text("# Retry report\n")
        return {
            "schema_version": "qa-v1",
            "run_id": "retry-run",
            "status": "scored",
            "input": {"snapshot": "input.snapshot.json"},
            "attempts": 1,
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {"status": "completed"},
            },
            "artifacts": artifact_hashes(
                output_dir,
                {
                    "input.snapshot.json",
                    "preparation.jsonl",
                    "summary.json",
                    "report.md",
                },
            ),
        }


class _NoRetryWorkflow:
    def retry_plan(self, _parent_path):
        raise ValueError("evaluation run has no failed attempts to retry")


def _use_process_workflow(monkeypatch, workflow_name):
    production_popen = jobs_module.subprocess.Popen

    def test_popen(command, *args, **kwargs):
        command = list(command)
        assert command[1:3] == ["-m", "src.evaluation.qa.worker"]
        command[2] = "tests.unit.evaluation.qa.worker_process"
        return production_popen(command, *args, **kwargs)

    monkeypatch.setattr(jobs_module.subprocess, "Popen", test_popen)
    monkeypatch.setenv("ARCHI_TEST_WORKFLOW", workflow_name)


def _write_preparation_artifacts(run_dir):
    write_json(
        run_dir / "input.snapshot.json",
        [
            {
                "id": "item",
                "question": "Question",
                "answer": "Answer",
                "time_sensitive": True,
            }
        ],
    )
    write_jsonl(
        run_dir / "preparation.jsonl",
        [
            {
                "item_id": "item",
                "status": "skipped_time_sensitive",
                "category": None,
                "answer_mode": None,
                "answer_source": None,
            }
        ],
    )
    return {"input.snapshot.json", "preparation.jsonl"}


def _write_legacy_prepared_workspace(run_dir):
    write_json(
        run_dir / "input.snapshot.json",
        [
            {
                "id": "item",
                "question": "Question",
                "answer": "Answer",
                "time_sensitive": False,
            }
        ],
    )
    prepared = {
        "item_id": "item",
        "question": "Question",
        "answer": "Answer",
        "time_sensitive": False,
        "category": None,
        "answer_mode": None,
        "answer_source": None,
        "atom_source": "supplied",
        "gold_atoms": [{"id": "A1", "text": "Answer", "required": True}],
    }
    write_jsonl(run_dir / "prepared_items.jsonl", [prepared])
    write_jsonl(
        run_dir / "preparation_results.jsonl",
        [
            {
                "item_id": "item",
                "status": "prepared",
                "category": None,
                "answer_mode": None,
                "answer_source": None,
            }
        ],
    )
    artifacts = {
        "input.snapshot.json",
        "prepared_items.jsonl",
        "preparation_results.jsonl",
    }
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "qa-v0",
            "run_id": "legacy-run",
            "status": "prepared",
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(run_dir, artifacts),
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
            },
        },
    )
    return prepared


def _write_scored_history_run(
    run_dir,
    *,
    source_path="/private/evaluations/golden.json",
    metadata=None,
    durations=(100, 200, 500, None),
):
    run_dir.mkdir()
    preparation_artifacts = _write_preparation_artifacts(run_dir)
    answers = []
    statuses = ("answer_ready", "answer_ready", "execution_failed", "answer_ready")
    for ordinal, (status, duration_ms) in enumerate(zip(statuses, durations), 1):
        answer = {
            "item_id": "item",
            "attempt_id": f"item-attempt-{ordinal}",
            "ordinal": ordinal,
            "status": status,
        }
        if duration_ms is not None:
            answer["duration_ms"] = duration_ms
        answers.append(answer)
    write_jsonl(run_dir / "answers.jsonl", answers)
    write_json(
        run_dir / "summary.json",
        {
            "overall_attempt_pass_rate": 1 / 3,
            "passed_attempts": 1,
            "quality_accounted_attempts": 3,
            "attempt_lifecycle_counts": {
                "scored": 2,
                "execution_failed": 1,
                "evaluation_failed": 1,
            },
        },
    )
    (run_dir / "report.md").write_text("# Report\n")
    artifact_names = preparation_artifacts | {
        "answers.jsonl",
        "summary.json",
        "report.md",
    }
    if metadata is not None:
        write_json(run_dir / "console_metadata.json", metadata)
        artifact_names.add("console_metadata.json")
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": run_dir.name,
            "status": "scored",
            "attempts": 4,
            "input": {
                "source_path": source_path,
                "snapshot": "input.snapshot.json",
            },
            "artifacts": artifact_hashes(run_dir, artifact_names),
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {
                    "status": "completed",
                    "completed_at": "2026-07-24T10:00:00+00:00",
                },
            },
        },
    )


def test_console_job_exposes_current_atom_draft_status(tmp_path):
    service = EvaluationConsoleService(
        tmp_path,
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=tmp_path,
    )
    draft_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    draft_path = service.catalog.drafts_dir / draft_id / "draft.json"
    write_json(draft_path, {"id": draft_id, "status": "open"})
    write_json(
        service.catalog.jobs_dir / f"{job_id}.json",
        {
            "id": job_id,
            "kind": "generate_atoms",
            "status": "completed",
            "context": {"dataset_id": str(uuid.uuid4())},
            "result": {"draft_id": draft_id},
        },
    )

    assert service.get_job(job_id)["result"]["draft_status"] == "open"
    write_json(draft_path, {"id": draft_id, "status": "saved"})
    assert service.get_job(job_id)["result"]["draft_status"] == "saved"
    service.jobs.close()


def test_console_atom_generation_constructs_evaluator_directly(monkeypatch, tmp_path):
    profiles = []

    class Evaluator:
        def extract_gold(self, question, answer):
            return {
                "atoms": [
                    {"id": "A1", "text": answer, "required": True},
                ]
            }

    def evaluator_runtime(profile):
        profiles.append(profile)
        return Evaluator()

    monkeypatch.setattr(console_module, "LangChainEvaluatorRuntime", evaluator_runtime)
    service = EvaluationConsoleService(
        tmp_path,
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=tmp_path,
    )
    dataset, _created = service.catalog.import_dataset(
        "Dataset",
        "dataset.json",
        json.dumps(
            [
                {
                    "id": "item",
                    "question": "Question",
                    "answer": "Answer",
                    "time_sensitive": False,
                }
            ]
        ).encode(),
    )

    job = service.start_atom_generation(dataset["id"], "builtin")
    completed = service.jobs.wait(job["id"], timeout=2)
    draft = service.catalog.get_atom_draft(completed["result"]["draft_id"])

    assert completed["status"] == "completed"
    assert len(profiles) == 1
    assert draft["items"][0]["atoms"] == [
        {"id": "A1", "text": "Answer", "required": True},
    ]
    service.jobs.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_workers", 0),
        ("run_workers", 17),
        ("run_workers", True),
        ("score_workers", 0),
        ("score_workers", 17),
        ("score_workers", "2"),
    ],
)
def test_console_rejects_invalid_phase_workers_before_catalog_or_job(
    field, value, tmp_path
):
    service = EvaluationConsoleService(
        tmp_path,
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=tmp_path,
    )
    values = {"run_workers": 1, "score_workers": 1, field: value}

    with pytest.raises(ValueError, match=f"{field} must be an integer from 1 to 16"):
        service.start_evaluation(
            name="Run",
            dataset_id="missing",
            profile_id="builtin",
            agent_spec="agent.md",
            attempts=1,
            **values,
        )

    assert service.jobs.list() == []
    service.jobs.close()


def test_console_persists_and_passes_phase_workers(monkeypatch, tmp_path):
    _use_process_workflow(
        monkeypatch,
        "recording",
    )
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "agent.md").write_text(
        "---\nname: Agent\ntools: [search]\n---\nPrompt\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("services: {}\n")
    service = EvaluationConsoleService(
        tmp_path / "console",
        agent_config_path=config_path,
        agents_dir=agents_dir,
    )
    dataset, _created = service.catalog.import_dataset(
        "Dataset",
        "dataset.json",
        b'[{"id":"item","question":"Q","answer":"A","time_sensitive":false}]',
    )

    job = service.start_evaluation(
        name="Parallel run",
        dataset_id=dataset["id"],
        profile_id="builtin",
        agent_spec="agent.md",
        attempts=2,
        run_workers=4,
        score_workers=3,
    )
    completed = service.jobs.wait(job["id"], timeout=2)

    assert completed["status"] == "completed"
    assert job["context"]["run_workers"] == 4
    assert job["context"]["score_workers"] == 3
    worker_arguments = read_json(
        service.catalog.runs_dir
        / job["context"]["workspace_id"]
        / "worker_arguments.json"
    )
    assert worker_arguments == {
        "evaluator_profile_path": None,
        "run_workers": 4,
        "score_workers": 3,
    }
    metadata = read_json(
        service.catalog.runs_dir
        / job["context"]["workspace_id"]
        / "console_metadata.json"
    )
    assert metadata["run_workers"] == 4
    assert metadata["score_workers"] == 3
    service.jobs.close()


def test_job_manager_enforces_single_flight_and_persists_result(tmp_path):
    manager = EvaluationJobManager(tmp_path)
    release = threading.Event()

    job = manager.start(
        "generate_atoms",
        lambda: (release.wait(2), {"draft_id": "draft"})[1],
    )

    with pytest.raises(JobConflictError, match="already"):
        manager.start("generate_atoms", lambda: {})
    release.set()
    completed = manager.wait(job["id"], timeout=2)

    assert completed["status"] == "completed"
    assert completed["result"] == {"draft_id": "draft"}
    manager.close()


def test_job_manager_marks_stale_work_interrupted(tmp_path):
    job_id = "040bb55f-739c-46a8-a297-f49f54d1e759"
    write_json(
        tmp_path / f"{job_id}.json",
        {"id": job_id, "kind": "evaluation", "status": "running"},
    )

    manager = EvaluationJobManager(tmp_path)

    assert manager.get(job_id)["status"] == "interrupted"
    manager.close()


def test_job_manager_terminates_running_evaluation_process(monkeypatch, tmp_path):
    _use_process_workflow(
        monkeypatch,
        "slow",
    )
    manager = EvaluationJobManager(tmp_path)
    terminal_callback_statuses = []
    request = {
        "operation": "composite",
        "output_dir": str(tmp_path / "run"),
        "dataset": str(tmp_path / "dataset.json"),
        "agent_config": str(tmp_path / "config.yaml"),
        "agent_spec": str(tmp_path / "agent.md"),
        "evaluator_profile_path": str(tmp_path / "profile.yaml"),
        "attempts": 1,
        "run_workers": 1,
        "score_workers": 1,
    }
    job = manager.start_process(
        request,
        context={"workspace_id": "run", "attempts": 1},
    )
    deadline = time.monotonic() + 5
    while manager.get(job["id"])["status"] != "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    process_id = manager._processes[job["id"]].pid

    def on_terminated(terminated_job):
        terminal_callback_statuses.append(terminated_job["status"])
        with pytest.raises(JobConflictError, match="already cancel_requested"):
            manager.start("generate_atoms", lambda: {})

    canceled = manager.cancel(job["id"], on_terminated=on_terminated)

    assert canceled["status"] == "canceled"
    assert terminal_callback_statuses == ["cancel_requested"]
    assert manager.wait(job["id"], timeout=2)["status"] == "canceled"
    with pytest.raises(ProcessLookupError):
        os.kill(process_id, 0)
    manager.close()


def test_job_manager_kills_signal_resistant_evaluation_descendants(
    monkeypatch, tmp_path
):
    _use_process_workflow(
        monkeypatch,
        "descendant",
    )
    monkeypatch.setattr(jobs_module, "PROCESS_TERMINATION_GRACE_SECONDS", 0.2)
    manager = EvaluationJobManager(tmp_path / "jobs")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    request = {
        "operation": "composite",
        "output_dir": str(output_dir),
        "dataset": str(tmp_path / "dataset.json"),
        "agent_config": str(tmp_path / "config.yaml"),
        "agent_spec": str(tmp_path / "agent.md"),
        "evaluator_profile_path": str(tmp_path / "profile.yaml"),
        "attempts": 1,
        "run_workers": 1,
        "score_workers": 1,
    }
    job = manager.start_process(
        request,
        context={"workspace_id": "run", "attempts": 1},
    )
    child_pid_path = output_dir / "child.pid"
    deadline = time.monotonic() + 5
    while not child_pid_path.is_file():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    child_pid = int(child_pid_path.read_text())

    canceled = manager.cancel(job["id"])

    assert canceled["status"] == "canceled"
    status_path = Path(f"/proc/{child_pid}/stat")
    try:
        child_state = status_path.read_text().split()[2]
    except FileNotFoundError:
        child_state = None
    assert child_state in {None, "Z"}
    manager.close()


def test_job_manager_preserves_completed_state_when_cancel_loses_race(
    monkeypatch, tmp_path
):
    _use_process_workflow(
        monkeypatch,
        "recording",
    )
    manager = EvaluationJobManager(tmp_path / "jobs")
    output_dir = tmp_path / "runs" / "run"
    output_dir.mkdir(parents=True)
    write_json(output_dir / "console_metadata.json", {"name": "Completed run"})
    request = {
        "operation": "composite",
        "output_dir": str(output_dir),
        "dataset": str(tmp_path / "dataset.json"),
        "agent_config": str(tmp_path / "config.yaml"),
        "agent_spec": str(tmp_path / "agent.md"),
        "evaluator_profile_path": str(tmp_path / "profile.yaml"),
        "attempts": 1,
        "run_workers": 1,
        "score_workers": 1,
    }
    completed_job = manager.start_process(
        request, context={"workspace_id": "run", "attempts": 1}
    )
    completed = manager.wait(completed_job["id"], timeout=2)

    with pytest.raises(JobConflictError, match="already completed"):
        manager.cancel(completed_job["id"])

    assert manager.get(completed_job["id"])["status"] == "completed"
    manager.close()


def test_console_conflicting_launch_leaves_no_history_workspace(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "agent.md").write_text("---\nname: Agent\ntools: []\n---\nPrompt\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("services: {}\n")
    service = EvaluationConsoleService(
        tmp_path / "console",
        agent_config_path=config_path,
        agents_dir=agents_dir,
    )
    dataset, _created = service.catalog.import_dataset(
        "Dataset",
        "dataset.json",
        b'[{"id":"item","question":"Q","answer":"A","time_sensitive":false}]',
    )
    release = threading.Event()
    active = service.jobs.start("generate_atoms", lambda: release.wait(2))

    with pytest.raises(JobConflictError, match="already"):
        service.start_evaluation(
            name="Conflicting run",
            dataset_id=dataset["id"],
            profile_id="builtin",
            agent_spec="agent.md",
            attempts=1,
        )

    assert list(service.catalog.runs_dir.iterdir()) == []
    release.set()
    service.jobs.wait(active["id"], timeout=2)
    service.jobs.close()


def test_console_cancellation_persists_valid_unscored_history(monkeypatch, tmp_path):
    _use_process_workflow(
        monkeypatch,
        "slow",
    )
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "agent.md").write_text("---\nname: Agent\ntools: []\n---\nPrompt\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("services: {}\n")
    service = EvaluationConsoleService(
        tmp_path / "console",
        agent_config_path=config_path,
        agents_dir=agents_dir,
    )
    dataset, _created = service.catalog.import_dataset(
        "Dataset",
        "dataset.json",
        b'[{"id":"item","question":"Q","answer":"A","time_sensitive":false}]',
    )
    job = service.start_evaluation(
        name="Canceled run",
        dataset_id=dataset["id"],
        profile_id="builtin",
        agent_spec="agent.md",
        attempts=2,
    )
    deadline = time.monotonic() + 5
    while service.get_job(job["id"])["status"] != "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)

    payload = service.cancel_evaluation(job["id"])
    rows = service.history.list_runs()
    detail = service.history.get_run(payload["history_id"])
    metadata = read_json(
        service.catalog.runs_dir
        / job["context"]["workspace_id"]
        / "console_metadata.json"
    )

    assert payload["job"]["status"] == "canceled"
    assert service.cancel_evaluation(job["id"]) == payload
    assert rows == [
        {
            "id": payload["history_id"],
            "run_id": job["context"]["workspace_id"],
            "name": "Canceled run",
            "status": "canceled",
            "created_at": metadata["created_at"],
            "dataset_id": dataset["id"],
            "dataset_key": dataset["id"],
            "dataset_name": "Dataset",
            "profile_id": "builtin",
            "profile_name": "Built-in QA profile",
            "agent_spec": "agent.md",
            "attempts": 2,
            "retry_of_history_id": None,
            "retry_number": None,
            "overall_attempt_pass_rate": None,
            "passed_attempts": None,
            "quality_accounted_attempts": None,
            "attempt_lifecycle_counts": None,
            "technical_failure_rate": None,
            "latency": None,
            "schema_version": "qa-v1",
            "capabilities": {"retry_failed": False},
            "valid": True,
        }
    ]
    assert detail["manifest"]["status"] == "canceled"
    assert detail["report_available"] is False
    assert detail["capabilities"] == {"retry_failed": False}
    with pytest.raises(LookupError, match="report not found"):
        service.history.get_report(payload["history_id"])

    next_job = service.jobs.start("generate_atoms", lambda: {"draft_id": "next"})
    assert service.jobs.wait(next_job["id"], timeout=2)["status"] == "completed"
    service.jobs.close()


def test_history_lists_valid_runs_and_isolates_invalid_ones(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    preparation_artifacts = _write_preparation_artifacts(valid)
    write_json(
        valid / "summary.json",
        {"overall_attempt_pass_rate": 0.75, "items": []},
    )
    (valid / "report.md").write_text("# Report\n")
    write_json(
        valid / "console_metadata.json",
        {
            "name": "Retry run",
            "retry_of_history_id": "a" * 24,
            "retry_number": 2,
        },
    )
    write_json(
        valid / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "run-1",
            "status": "scored",
            "attempts": 2,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(
                valid,
                preparation_artifacts
                | {"summary.json", "report.md", "console_metadata.json"},
            ),
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {
                    "status": "completed",
                    "completed_at": "2026-07-24T10:00:00+00:00",
                },
            },
        },
    )
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "manifest.json").write_text(json.dumps({"schema_version": "qa-v99"}))
    history = EvaluationHistory(tmp_path)

    rows = history.list_runs()

    assert len(rows) == 2
    valid_row = next(row for row in rows if row["valid"])
    invalid_row = next(row for row in rows if not row["valid"])
    assert valid_row["overall_attempt_pass_rate"] == 0.75
    assert valid_row["retry_of_history_id"] == "a" * 24
    assert valid_row["retry_number"] == 2
    assert history.get_report(valid_row["id"]) == "# Report\n"
    assert "unsupported run schema" in invalid_row["error"]


def test_history_loads_manifest_and_console_metadata_as_typed_models(tmp_path):
    metadata_payload = {
        "name": "Typed run",
        "dataset_id": "dataset-1",
        "run_workers": 2,
        "extension": {"kept": True},
    }
    run = tmp_path / "typed-run"
    _write_scored_history_run(run, metadata=metadata_payload)
    history = EvaluationHistory(tmp_path)

    manifest = history._load_manifest(run)
    metadata = history._load_console_metadata(run, manifest)

    assert isinstance(manifest, RunManifest)
    assert isinstance(metadata, ConsoleMetadata)
    assert metadata.to_dict() == metadata_payload
    assert history.get_run(history.id_for_path(run))["metadata"] == metadata_payload


def test_history_projects_compact_latency_quality_and_failure_trends(tmp_path):
    dataset_id = str(uuid.uuid4())
    metadata = {
        "name": "Trend run",
        "dataset_id": dataset_id,
        "dataset_name": "Reviewed golden set",
        "created_at": "2026-07-24T09:00:00+00:00",
        "retry_of_history_id": "a" * 24,
        "retry_number": 2,
    }
    run = tmp_path / "trend-run"
    _write_scored_history_run(run, metadata=metadata)
    history = EvaluationHistory(tmp_path)

    row = history.list_runs()[0]

    assert row == {
        "id": history.id_for_path(run),
        "run_id": "trend-run",
        "name": "Trend run",
        "status": "scored",
        "created_at": "2026-07-24T09:00:00+00:00",
        "dataset_id": dataset_id,
        "dataset_key": dataset_id,
        "dataset_name": "Reviewed golden set",
        "profile_id": None,
        "profile_name": None,
        "agent_spec": None,
        "attempts": 4,
        "retry_of_history_id": "a" * 24,
        "retry_number": 2,
        "overall_attempt_pass_rate": 1 / 3,
        "passed_attempts": 1,
        "quality_accounted_attempts": 3,
        "attempt_lifecycle_counts": {
            "scored": 2,
            "execution_failed": 1,
            "evaluation_failed": 1,
        },
        "technical_failure_rate": 0.5,
        "latency": {
            "total_attempts": 4,
            "timed_attempts": 3,
            "average_ms": 800 / 3,
            "best_ms": 100,
            "worst_ms": 500,
        },
        "schema_version": "qa-v1",
        "capabilities": {"retry_failed": True},
        "valid": True,
    }


def test_history_uses_snapshot_identity_and_basename_for_cli_runs(tmp_path):
    run = tmp_path / "cli-run"
    _write_scored_history_run(run, source_path=r"C:\private\sets\golden.json")
    history = EvaluationHistory(tmp_path)
    manifest = read_json(run / "manifest.json")

    row = history.list_runs()[0]

    assert row["dataset_key"] == (
        f"snapshot:{manifest['artifacts']['input.snapshot.json']}"
    )
    assert row["dataset_name"] == "golden.json"
    assert "private" not in json.dumps(row)


def test_history_streams_latency_rows_without_loading_complete_answers(
    tmp_path, monkeypatch
):
    run = tmp_path / "streamed-run"
    _write_scored_history_run(run)
    monkeypatch.setattr(
        history_module,
        "read_jsonl",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("compact history must stream answers")
        ),
    )

    row = EvaluationHistory(tmp_path).list_runs()[0]

    assert row["latency"]["timed_attempts"] == 3


def test_history_skips_expensive_artifacts_before_utc_cutoff(tmp_path, monkeypatch):
    old = tmp_path / "old-run"
    _write_scored_history_run(
        old,
        metadata={"created_at": "2026-07-31T23:59:59+00:00"},
    )
    at_cutoff = tmp_path / "cutoff-run"
    _write_scored_history_run(
        at_cutoff,
        metadata={"created_at": "2026-08-01T00:00:00+00:00"},
    )
    undated = tmp_path / "undated-run"
    _write_scored_history_run(undated)
    undated_manifest = read_json(undated / "manifest.json")
    del undated_manifest["phases"]["score"]["completed_at"]
    write_json(undated / "manifest.json", undated_manifest)

    summary_reads = []
    latency_reads = []
    verified_artifacts = []
    original_read_json = history_module.read_json
    original_latency_trend = EvaluationHistory._latency_trend
    original_verify_hashes = history_module.verify_hashes

    def recording_read_json(path):
        if path.name == "summary.json":
            summary_reads.append(path.parent.name)
        return original_read_json(path)

    def recording_latency_trend(path, manifest):
        latency_reads.append(path.name)
        return original_latency_trend(path, manifest)

    def recording_verify_hashes(path, artifacts, filenames):
        verified_artifacts.extend((path.name, filename) for filename in filenames)
        return original_verify_hashes(path, artifacts, filenames)

    monkeypatch.setattr(history_module, "read_json", recording_read_json)
    monkeypatch.setattr(
        EvaluationHistory,
        "_latency_trend",
        staticmethod(recording_latency_trend),
    )
    monkeypatch.setattr(history_module, "verify_hashes", recording_verify_hashes)

    rows = EvaluationHistory(tmp_path).list_runs(
        cutoff=datetime(2026, 8, 1, tzinfo=timezone.utc)
    )

    assert {row["name"] for row in rows} == {"cutoff-run", "undated-run"}
    assert set(summary_reads) == {"cutoff-run", "undated-run"}
    assert set(latency_reads) == {"cutoff-run", "undated-run"}
    assert not {
        filename
        for run_name, filename in verified_artifacts
        if run_name == "old-run" and filename in {"summary.json", "answers.jsonl"}
    }


def test_history_rejects_naive_cutoff(tmp_path):
    history = EvaluationHistory(tmp_path)

    with pytest.raises(ValueError, match="cutoff must include a timezone"):
        history.list_runs(cutoff=datetime(2026, 8, 1))


def test_history_isolates_invalid_attempt_duration_from_other_runs(tmp_path):
    valid = tmp_path / "valid-trend"
    _write_scored_history_run(valid)
    invalid = tmp_path / "invalid-trend"
    _write_scored_history_run(invalid, durations=(100, True, 500, None))
    history = EvaluationHistory(tmp_path)

    rows = history.list_runs()

    assert sum(row["valid"] for row in rows) == 1
    invalid_row = next(row for row in rows if not row["valid"])
    assert invalid_row["name"] == "invalid-trend"
    assert invalid_row["error"] == (
        "answer row 2 duration_ms must be a non-negative integer"
    )


def test_history_isolates_invalid_timestamp_and_inconsistent_pass_counts(tmp_path):
    valid = tmp_path / "valid-trend"
    _write_scored_history_run(valid)
    invalid_timestamp = tmp_path / "invalid-timestamp"
    _write_scored_history_run(
        invalid_timestamp,
        metadata={
            "created_at": "not-a-timestamp",
            "dataset_name": "Invalid timestamp",
        },
    )
    invalid_counts = tmp_path / "invalid-counts"
    _write_scored_history_run(invalid_counts)
    summary = read_json(invalid_counts / "summary.json")
    summary["overall_attempt_pass_rate"] = 1.0
    write_json(invalid_counts / "summary.json", summary)
    manifest = read_json(invalid_counts / "manifest.json")
    manifest["artifacts"] = artifact_hashes(
        invalid_counts,
        set(manifest["artifacts"]),
    )
    write_json(invalid_counts / "manifest.json", manifest)

    rows = EvaluationHistory(tmp_path).list_runs()

    assert sum(row["valid"] for row in rows) == 1
    errors = {row["name"]: row["error"] for row in rows if not row["valid"]}
    assert errors == {
        "invalid-counts": "summary pass rate does not match its attempt counts",
        "invalid-timestamp": (
            "run creation timestamp must be an ISO-8601 timestamp with a timezone"
        ),
    }


def test_history_reads_legacy_v0_runs_without_rewriting_artifacts(tmp_path):
    run = tmp_path / "legacy"
    run.mkdir()
    write_json(
        run / "input.snapshot.json",
        [
            {
                "id": "prepared",
                "question": "Question",
                "answer": "Answer",
                "time_sensitive": False,
            },
            {
                "id": "skipped",
                "question": "Current question",
                "answer": "Current answer",
                "time_sensitive": True,
            },
            {
                "id": "failed",
                "question": "Failed question",
                "answer": "Failed answer",
                "time_sensitive": False,
            },
        ],
    )
    write_jsonl(
        run / "prepared_items.jsonl",
        [
            {
                "item_id": "prepared",
                "question": "Question",
                "answer": "Answer",
                "time_sensitive": False,
                "category": "general",
                "answer_mode": None,
                "answer_source": None,
                "atom_source": "supplied",
                "gold_atoms": [{"id": "A1", "text": "Answer", "required": True}],
            }
        ],
    )
    write_jsonl(
        run / "preparation_results.jsonl",
        [
            {
                "item_id": "prepared",
                "status": "prepared",
                "category": "general",
                "answer_mode": None,
                "answer_source": None,
            },
            {
                "item_id": "skipped",
                "status": "skipped_time_sensitive",
                "category": "live",
                "answer_mode": None,
                "answer_source": None,
            },
            {
                "item_id": "failed",
                "status": "preparation_failed",
                "category": None,
                "answer_mode": None,
                "answer_source": None,
                "error": "provider unavailable",
            },
        ],
    )
    write_json(run / "summary.json", {"overall_attempt_pass_rate": 1.0})
    (run / "report.md").write_text("# Legacy report\n")
    artifacts = {
        "input.snapshot.json",
        "prepared_items.jsonl",
        "preparation_results.jsonl",
        "summary.json",
        "report.md",
    }
    write_json(
        run / "manifest.json",
        {
            "schema_version": "qa-v0",
            "run_id": "legacy-run",
            "status": "scored",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(run, artifacts),
            "phases": {
                "prepare": {"status": "completed", "input_items": 3},
                "run": {"status": "completed"},
                "score": {"status": "completed"},
            },
        },
    )
    original_manifest = (run / "manifest.json").read_bytes()
    legacy_manifest = read_json(run / "manifest.json")
    history = EvaluationHistory(tmp_path)

    rows = history.list_runs()
    payload = history.get_run(history.id_for_path(run))

    assert rows == [
        {
            "id": history.id_for_path(run),
            "run_id": "legacy-run",
            "name": "legacy-run",
            "status": "scored",
            "created_at": "",
            "dataset_id": None,
            "dataset_key": (
                "snapshot:" + legacy_manifest["artifacts"]["input.snapshot.json"]
            ),
            "dataset_name": "CLI snapshot",
            "profile_id": None,
            "profile_name": None,
            "agent_spec": None,
            "attempts": 1,
            "retry_of_history_id": None,
            "retry_number": None,
            "overall_attempt_pass_rate": 1.0,
            "passed_attempts": None,
            "quality_accounted_attempts": None,
            "attempt_lifecycle_counts": None,
            "technical_failure_rate": None,
            "latency": None,
            "schema_version": "qa-v0",
            "capabilities": {"retry_failed": False},
            "valid": True,
        }
    ]
    assert payload["preparation"] == [
        {
            "item_id": "prepared",
            "status": "prepared",
            "category": "general",
            "answer_mode": None,
            "answer_source": None,
            "question": "Question",
            "answer": "Answer",
            "time_sensitive": False,
            "atom_source": "supplied",
            "gold_atoms": [{"id": "A1", "text": "Answer", "required": True}],
        },
        {
            "item_id": "skipped",
            "status": "skipped_time_sensitive",
            "category": "live",
            "answer_mode": None,
            "answer_source": None,
        },
        {
            "item_id": "failed",
            "status": "preparation_failed",
            "category": None,
            "answer_mode": None,
            "answer_source": None,
            "error": "provider unavailable",
        },
    ]
    assert payload["prepared_items"] == [payload["preparation"][0]]
    assert payload["capabilities"] == {"retry_failed": False}
    assert history.get_report(history.id_for_path(run)) == "# Legacy report\n"
    assert (run / "manifest.json").read_bytes() == original_manifest
    assert not (run / "preparation.jsonl").exists()


def test_history_rejects_tampered_legacy_v0_preparation(tmp_path):
    run = tmp_path / "legacy"
    run.mkdir()
    prepared = _write_legacy_prepared_workspace(run)
    prepared["answer"] = "Tampered"
    write_jsonl(run / "prepared_items.jsonl", [prepared])
    history = EvaluationHistory(tmp_path)

    with pytest.raises(
        ValueError,
        match="workspace artifact hash mismatch: prepared_items.jsonl",
    ):
        history.get_run(history.id_for_path(run))


def test_console_rejects_legacy_v0_retry_before_workflow_or_job(monkeypatch, tmp_path):
    workflow_constructed = False

    def workflow_factory():
        nonlocal workflow_constructed
        workflow_constructed = True
        return _RetryWorkflow()

    monkeypatch.setattr(console_module, "QAWorkflow", workflow_factory)

    service = EvaluationConsoleService(
        tmp_path,
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=tmp_path,
    )
    run = service.catalog.runs_dir / "legacy"
    run.mkdir()
    _write_legacy_prepared_workspace(run)

    with pytest.raises(ValueError, match="legacy evaluation runs cannot be retried"):
        service.start_evaluation_retry(service.history.id_for_path(run))

    assert workflow_constructed is False
    assert service.jobs.list() == []
    service.jobs.close()


def test_history_derives_prepared_items_from_canonical_preparation(
    monkeypatch, tmp_path
):
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "input.snapshot.json",
        [
            {
                "id": "item",
                "question": "Question",
                "answer": "Answer",
                "time_sensitive": False,
                "expected_atoms": [{"id": "A1", "text": "Answer", "required": True}],
            }
        ],
    )
    preparation = [
        {
            "item_id": "item",
            "status": "prepared",
            "category": None,
            "answer_mode": None,
            "answer_source": None,
            "question": "Question",
            "answer": "Answer",
            "time_sensitive": False,
            "atom_source": "supplied",
            "gold_atoms": [{"id": "A1", "text": "Answer", "required": True}],
        }
    ]
    write_jsonl(run / "preparation.jsonl", preparation)
    write_json(
        run / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "run-1",
            "status": "prepared",
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(
                run, {"input.snapshot.json", "preparation.jsonl"}
            ),
            "phases": {"prepare": {"status": "completed", "input_items": 1}},
        },
    )
    history = EvaluationHistory(tmp_path)
    history_id = history.id_for_path(run)
    monkeypatch.setattr(
        history_module,
        "load_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("history must consume preparation.jsonl directly")
        ),
        raising=False,
    )

    payload = history.get_run(history_id)

    assert payload["preparation"] == preparation
    assert payload["prepared_items"] == preparation


def test_history_exposes_complete_tool_call_trace_unchanged(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "input.snapshot.json",
        [
            {
                "id": "item",
                "question": "Question",
                "answer": "Answer",
                "time_sensitive": False,
                "expected_atoms": [{"id": "A1", "text": "Answer", "required": True}],
            }
        ],
    )
    write_jsonl(
        run / "preparation.jsonl",
        [
            {
                "item_id": "item",
                "status": "prepared",
                "category": None,
                "answer_mode": None,
                "answer_source": None,
                "question": "Question",
                "answer": "Answer",
                "time_sensitive": False,
                "atom_source": "supplied",
                "gold_atoms": [{"id": "A1", "text": "Answer", "required": True}],
            }
        ],
    )
    tool_calls = [
        {
            "ordinal": 1,
            "name": "search",
            "status": "success",
            "query": '{"query": "complete"}',
            "response": "complete response",
            "duration_ms": 13,
        },
        {
            "ordinal": 2,
            "name": "unfinished",
            "status": "incomplete",
            "query": "complete unfinished query",
        },
    ]
    write_jsonl(
        run / "answers.jsonl",
        [
            {
                "item_id": "item",
                "attempt_id": "item-attempt-1",
                "ordinal": 1,
                "status": "answer_ready",
                "duration_ms": 25,
                "tool_calls": tool_calls,
                "answer": "Answer",
            }
        ],
    )
    artifacts = {"input.snapshot.json", "preparation.jsonl", "answers.jsonl"}
    write_json(
        run / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "run-1",
            "status": "run_completed",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(run, artifacts),
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
            },
        },
    )
    history = EvaluationHistory(tmp_path)

    payload = history.get_run(history.id_for_path(run))

    assert payload["answers"][0]["tool_calls"] == tool_calls


def test_history_rejects_invalid_tool_trace_at_artifact_boundary(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    artifacts = _write_preparation_artifacts(run)
    write_jsonl(
        run / "answers.jsonl",
        [
            {
                "item_id": "item",
                "attempt_id": "item-attempt-1",
                "ordinal": 1,
                "status": "answer_ready",
                "duration_ms": 25,
                "tool_calls": [
                    {
                        "ordinal": 1,
                        "name": "search",
                        "status": "success",
                        "query": "missing terminal response",
                    }
                ],
                "answer": "Answer",
            }
        ],
    )
    artifacts.add("answers.jsonl")
    write_json(
        run / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "run-1",
            "status": "run_completed",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(run, artifacts),
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
            },
        },
    )
    history = EvaluationHistory(tmp_path)

    with pytest.raises(ValueError, match="successful tool-call records require"):
        history.get_run(history.id_for_path(run))


def test_history_rejects_tampered_declared_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    preparation_artifacts = _write_preparation_artifacts(run)
    write_json(run / "summary.json", {"overall_attempt_pass_rate": 1.0})
    (run / "report.md").write_text("# Report\n")
    write_json(
        run / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "run-1",
            "status": "scored",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {"status": "completed"},
            },
            "artifacts": artifact_hashes(
                run, preparation_artifacts | {"summary.json", "report.md"}
            ),
        },
    )
    (run / "summary.json").write_text('{"overall_attempt_pass_rate": 0.0}\n')
    history = EvaluationHistory(tmp_path)

    row = history.list_runs()[0]

    assert row["valid"] is False
    assert "hash mismatch" in row["error"]


def test_console_retry_keeps_root_name_across_retry_generations(monkeypatch, tmp_path):
    _use_process_workflow(
        monkeypatch,
        "retry",
    )
    monkeypatch.setattr(console_module, "QAWorkflow", _RetryWorkflow)
    service = EvaluationConsoleService(
        tmp_path,
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=tmp_path,
    )
    parent = service.catalog.runs_dir / "parent"
    parent.mkdir()
    preparation_artifacts = _write_preparation_artifacts(parent)
    write_json(parent / "summary.json", {"overall_attempt_pass_rate": 0.5})
    (parent / "report.md").write_text("# Parent report\n")
    write_json(
        parent / "console_metadata.json",
        {
            "name": "Original run · retry 1",
            "retry_number": 1,
            "retry_root_name": "Original run",
        },
    )
    write_json(
        parent / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "parent-run",
            "status": "scored",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(
                parent,
                preparation_artifacts
                | {"summary.json", "report.md", "console_metadata.json"},
            ),
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {"status": "completed"},
            },
        },
    )
    history_id = service.history.id_for_path(parent)

    job = service.start_evaluation_retry(history_id)
    completed = service.jobs.wait(job["id"], timeout=2)

    assert completed["status"] == "completed"
    successor = service.history.run_path(completed["result"]["history_id"])
    metadata = read_json(successor / "console_metadata.json")
    assert metadata["name"] == "Original run · retry 2"
    assert metadata["retry_root_name"] == "Original run"
    assert metadata["retry_number"] == 2
    service.jobs.close()


def test_console_retry_without_failures_creates_no_job_or_successor(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(console_module, "QAWorkflow", _NoRetryWorkflow)
    service = EvaluationConsoleService(
        tmp_path,
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=tmp_path,
    )
    parent = service.catalog.runs_dir / "parent"
    parent.mkdir()
    preparation_artifacts = _write_preparation_artifacts(parent)
    write_json(parent / "summary.json", {"overall_attempt_pass_rate": 1.0})
    (parent / "report.md").write_text("# Complete report\n")
    write_json(
        parent / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "complete-run",
            "status": "scored",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(
                parent, preparation_artifacts | {"summary.json", "report.md"}
            ),
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {"status": "completed"},
            },
        },
    )
    history_id = service.history.id_for_path(parent)

    with pytest.raises(ValueError, match="no failed attempts"):
        service.start_evaluation_retry(history_id)

    assert service.jobs.list() == []
    assert list(service.catalog.runs_dir.iterdir()) == [parent]
    service.jobs.close()


def test_history_rejects_missing_declared_artifact(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    preparation_artifacts = _write_preparation_artifacts(run)
    write_json(run / "summary.json", {"overall_attempt_pass_rate": 1.0})
    (run / "report.md").write_text("# Report\n")
    write_json(
        run / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "run-1",
            "status": "scored",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {"status": "completed"},
            },
            "artifacts": artifact_hashes(
                run, preparation_artifacts | {"summary.json", "report.md"}
            ),
        },
    )
    (run / "summary.json").unlink()

    row = EvaluationHistory(tmp_path).list_runs()[0]

    assert row["valid"] is False
    assert "required workspace artifact is missing" in row["error"]


def test_history_does_not_follow_run_directory_symlinks(tmp_path):
    external = tmp_path.parent / "external-evaluation-run"
    external.mkdir()
    (tmp_path / "linked").symlink_to(external, target_is_directory=True)

    assert EvaluationHistory(tmp_path).list_runs() == []
