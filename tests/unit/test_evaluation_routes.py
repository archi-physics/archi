import io
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from flask import Flask

from src.evaluation.qa import artifacts as artifact_utils
from src.evaluation.qa import console as console_module
from src.evaluation.qa.catalog import EvaluationCatalog
from src.evaluation.qa.console import EvaluationConsoleService
from src.evaluation.qa.history import EvaluationHistory
from src.interfaces.chat_app import evaluation_routes
from src.interfaces.chat_app.evaluation_routes import register_evaluations
from src.utils.rbac.permission_enum import Permission


class _Jobs:
    def list(self):
        return []

    def get(self, job_id):
        return {"id": job_id, "status": "completed"}


class _Service:
    def __init__(self, tmp_path):
        self.catalog = EvaluationCatalog(tmp_path)
        self.history = EvaluationHistory(self.catalog.runs_dir)
        self.jobs = _Jobs()
        self.started = []

    def list_agents(self):
        return [{"id": "agent.md", "name": "Agent"}]

    def get_agent_snapshot(self, agent_filename):
        if agent_filename != "agent.md":
            raise ValueError("selected agent spec does not exist")
        return {
            "id": "agent.md",
            "name": "Agent",
            "tools": ["search"],
            "content": "---\nname: Agent\ntools: [search]\n---\nAnswer carefully.",
        }

    def list_jobs(self):
        return self.jobs.list()

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def start_atom_generation(self, dataset_id, profile_id):
        self.started.append(("atoms", dataset_id, profile_id))
        return {"id": "job-atoms", "status": "queued"}

    def start_atom_retry(self, draft_id):
        self.started.append(("atom-retry", draft_id))
        return {"id": "job-atom-retry", "status": "queued"}

    def start_evaluation(self, **kwargs):
        self.started.append(("evaluation", kwargs))
        return {"id": "job-run", "status": "queued"}

    def start_evaluation_retry(self, history_id):
        self.started.append(("evaluation-retry", history_id))
        return {"id": "job-evaluation-retry", "status": "queued"}

    def cancel_evaluation(self, job_id):
        self.started.append(("cancel", job_id))
        return {
            "job": {"id": job_id, "kind": "evaluation", "status": "canceled"},
            "history_id": "canceled-history-id",
        }


def _app(tmp_path, denied_permissions=None):
    app = Flask(
        __name__,
        template_folder="../../src/interfaces/chat_app/templates",
        static_folder="../../src/interfaces/chat_app/static",
    )
    permissions = []

    def require_perm(permission):
        def decorator(view):
            def wrapped(*args, **kwargs):
                permissions.append(permission)
                if permission in (denied_permissions or set()):
                    return "Forbidden", 403
                return view(*args, **kwargs)

            return wrapped

        return decorator

    service = _Service(tmp_path)
    register_evaluations(app, require_perm=require_perm, service=service)
    return app, service, permissions


def _dataset():
    return b'[{"id":"one","question":"Q","answer":"A",' b'"time_sensitive":false}]'


def _dataset_with_atom():
    return (
        b'[{"id":"one","question":"Q","answer":"A","time_sensitive":false,'
        b'"expected_atoms":[{"id":"A1","text":"A","required":true}]}]'
    )


def _partial_dataset():
    return (
        b'[{"id":"with-atom","question":"Q1","answer":"A1",'
        b'"time_sensitive":false,"expected_atoms":[{"id":"A1","text":"A1",'
        b'"required":true}]},{"id":"without-atom","question":"Q2","answer":"A2",'
        b'"time_sensitive":false}]'
    )


def _write_trend_run(service):
    run = service.catalog.runs_dir / "trend-run"
    run.mkdir()
    artifact_utils.write_json(run / "input.snapshot.json", [])
    artifact_utils.write_jsonl(run / "preparation.jsonl", [])
    artifact_utils.write_jsonl(
        run / "answers.jsonl",
        [
            {
                "item_id": "one",
                "attempt_id": "one-attempt-1",
                "ordinal": 1,
                "status": "answer_ready",
                "duration_ms": 240,
                "answer": "private response that must not enter compact history",
            }
        ],
    )
    artifact_utils.write_json(
        run / "summary.json",
        {
            "overall_attempt_pass_rate": 1.0,
            "passed_attempts": 1,
            "quality_accounted_attempts": 1,
            "attempt_lifecycle_counts": {
                "scored": 1,
                "execution_failed": 0,
                "evaluation_failed": 0,
            },
        },
    )
    (run / "report.md").write_text("# Trend report\n")
    artifact_utils.write_json(
        run / "console_metadata.json",
        {
            "name": "Trend run",
            "dataset_id": "dataset-id",
            "dataset_name": "Dataset",
            "created_at": "2026-08-05T10:00:00+00:00",
        },
    )
    artifacts = {
        "input.snapshot.json",
        "preparation.jsonl",
        "answers.jsonl",
        "summary.json",
        "report.md",
        "console_metadata.json",
    }
    artifact_utils.write_json(
        run / "manifest.json",
        {
            "schema_version": "qa-v1",
            "run_id": "trend-run",
            "status": "scored",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_utils.artifact_hashes(run, artifacts),
            "phases": {
                "prepare": {"status": "completed", "input_items": 0},
                "run": {"status": "completed"},
                "score": {
                    "status": "completed",
                    "completed_at": "2026-08-05T10:00:00+00:00",
                },
            },
        },
    )


def test_catalog_import_and_launch_routes_use_separate_permissions(tmp_path):
    app, service, permissions = _app(tmp_path)
    client = app.test_client()

    imported = client.post(
        "/api/evaluations/datasets",
        data={"name": "Set", "file": (io.BytesIO(_dataset()), "set.json")},
        content_type="multipart/form-data",
    )
    dataset_id = imported.get_json()["dataset"]["id"]
    listed = client.get("/api/evaluations/catalog")
    agent_snapshot = client.get("/api/evaluations/agents/agent.md")
    generated = client.post(
        f"/api/evaluations/datasets/{dataset_id}/generate-atoms",
        json={"profile_id": "builtin"},
    )
    review_dataset, _created = service.catalog.import_dataset(
        "Review set", "review.json", _dataset_with_atom()
    )
    reviewed = client.post(
        f"/api/evaluations/datasets/{review_dataset['id']}/review-atoms",
    )
    atom_retry = client.post(
        "/api/evaluations/atom-drafts/draft-id/retry-failed",
    )
    launched = client.post(
        "/api/evaluations/runs",
        json={
            "name": "Run",
            "dataset_id": dataset_id,
            "profile_id": "builtin",
            "agent_spec": "agent.md",
            "attempts": 2,
            "run_workers": 4,
            "score_workers": 3,
        },
    )
    evaluation_retry = client.post(
        "/api/evaluations/runs/history-id/retry-failed",
    )

    assert imported.status_code == 201
    assert listed.status_code == 200
    assert agent_snapshot.status_code == 200
    assert agent_snapshot.get_json()["agent"]["tools"] == ["search"]
    assert generated.status_code == 202
    assert reviewed.status_code == 201
    assert atom_retry.status_code == 202
    assert reviewed.get_json()["draft"]["items"][0]["atoms"] == [
        {"id": "A1", "text": "A", "required": True}
    ]
    assert launched.status_code == 202
    assert evaluation_retry.status_code == 202
    assert permissions == [
        Permission.Evaluations.MANAGE,
        Permission.Evaluations.VIEW,
        Permission.Evaluations.VIEW,
        Permission.Evaluations.MANAGE,
        Permission.Evaluations.MANAGE,
        Permission.Evaluations.MANAGE,
        Permission.Evaluations.RUN,
        Permission.Evaluations.RUN,
    ]
    launched_call = next(entry for entry in service.started if entry[0] == "evaluation")
    assert launched_call[1]["attempts"] == 2
    assert launched_call[1]["run_workers"] == 4
    assert launched_call[1]["score_workers"] == 3


def test_agent_snapshot_is_validated_bounded_and_kept_inside_catalog(
    tmp_path, monkeypatch
):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    agent_path = agents_dir / "reviewer.md"
    content = "---\nname: Reviewer\ntools:\n  - search\n---\nAnswer from evidence.\n"
    agent_path.write_text(content, encoding="utf-8")
    service = EvaluationConsoleService(
        tmp_path / "console",
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=agents_dir,
    )

    assert service.get_agent_snapshot("reviewer.md") == {
        "id": "reviewer.md",
        "name": "Reviewer",
        "tools": ["search"],
        "content": content,
    }

    with pytest.raises(ValueError, match="does not exist"):
        service.get_agent_snapshot("missing.md")
    with pytest.raises(ValueError, match="catalog agent filename"):
        service.get_agent_snapshot("../reviewer.md")

    invalid_path = agents_dir / "invalid.md"
    invalid_path.write_text("not an agent spec", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.md missing YAML frontmatter"):
        service.get_agent_snapshot("invalid.md")

    outside_path = tmp_path / "outside.md"
    outside_path.write_text(content, encoding="utf-8")
    (agents_dir / "outside.md").symlink_to(outside_path)
    with pytest.raises(ValueError, match="outside the configured agents directory"):
        service.get_agent_snapshot("outside.md")

    monkeypatch.setattr("src.evaluation.qa.console.MAX_AGENT_SNAPSHOT_BYTES", 8)
    with pytest.raises(ValueError, match="256 KiB limit"):
        service.get_agent_snapshot("reviewer.md")


def test_agent_snapshot_route_reports_missing_selection_with_view_permission(tmp_path):
    app, _service, permissions = _app(tmp_path)

    response = app.test_client().get("/api/evaluations/agents/missing.md")

    assert response.status_code == 400
    assert response.get_json() == {"error": "selected agent spec does not exist"}
    assert permissions == [Permission.Evaluations.VIEW]


def test_run_history_route_exposes_compact_trend_projection(tmp_path, monkeypatch):
    app, service, permissions = _app(tmp_path)
    _write_trend_run(service)
    monkeypatch.setattr(
        evaluation_routes,
        "_utc_now",
        lambda: datetime(2026, 8, 12, 12, tzinfo=timezone.utc),
        raising=False,
    )

    response = app.test_client().get("/api/evaluations/runs")

    assert response.status_code == 200
    row = response.get_json()["runs"][0]
    assert row["dataset_key"] == "dataset-id"
    assert row["latency"] == {
        "total_attempts": 1,
        "timed_attempts": 1,
        "average_ms": 240.0,
        "best_ms": 240,
        "worst_ms": 240,
    }
    assert row["overall_attempt_pass_rate"] == 1.0
    assert row["technical_failure_rate"] == 0.0
    assert "private response" not in response.get_data(as_text=True)
    assert permissions == [Permission.Evaluations.VIEW]


@pytest.mark.parametrize(
    ("history_range", "days"),
    [("7d", 7), ("30d", 30), ("90d", 90), ("365d", 365)],
)
def test_run_history_route_passes_explicit_utc_cutoff(
    tmp_path, monkeypatch, history_range, days
):
    app, service, permissions = _app(tmp_path)
    now = datetime(2026, 8, 12, 12, 30, tzinfo=timezone.utc)
    cutoffs = []
    monkeypatch.setattr(evaluation_routes, "_utc_now", lambda: now, raising=False)
    monkeypatch.setattr(
        service.history,
        "list_runs",
        lambda *, cutoff: cutoffs.append(cutoff) or [],
    )

    response = app.test_client().get(
        "/api/evaluations/runs", query_string={"range": history_range}
    )

    assert response.status_code == 200
    assert cutoffs == [now - timedelta(days=days)]
    assert cutoffs[0].tzinfo is timezone.utc
    assert permissions == [Permission.Evaluations.VIEW]


def test_run_history_route_defaults_to_90_days(tmp_path, monkeypatch):
    app, service, _permissions = _app(tmp_path)
    now = datetime(2026, 8, 12, 12, 30, tzinfo=timezone.utc)
    cutoffs = []
    monkeypatch.setattr(evaluation_routes, "_utc_now", lambda: now, raising=False)
    monkeypatch.setattr(
        service.history,
        "list_runs",
        lambda *, cutoff: cutoffs.append(cutoff) or [],
    )

    response = app.test_client().get("/api/evaluations/runs")

    assert response.status_code == 200
    assert cutoffs == [now - timedelta(days=90)]


@pytest.mark.parametrize("history_range", ["all", "30", "", "90D"])
def test_run_history_route_rejects_unsupported_range(
    tmp_path, monkeypatch, history_range
):
    app, service, _permissions = _app(tmp_path)
    listed = []
    monkeypatch.setattr(
        service.history,
        "list_runs",
        lambda *, cutoff: listed.append(cutoff) or [],
    )

    response = app.test_client().get(
        "/api/evaluations/runs", query_string={"range": history_range}
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "range must be one of: 7d, 30d, 90d, 365d"}
    assert listed == []


def test_launch_route_defaults_phase_workers_for_older_clients(tmp_path):
    app, service, _permissions = _app(tmp_path)

    response = app.test_client().post(
        "/api/evaluations/runs",
        json={
            "name": "Run",
            "dataset_id": "dataset-id",
            "profile_id": "builtin",
            "agent_spec": "agent.md",
            "attempts": 1,
        },
    )

    assert response.status_code == 202
    assert service.started == [
        (
            "evaluation",
            {
                "name": "Run",
                "dataset_id": "dataset-id",
                "profile_id": "builtin",
                "agent_spec": "agent.md",
                "attempts": 1,
                "run_workers": 1,
                "score_workers": 1,
            },
        )
    ]


def test_partial_dataset_review_never_starts_generation_or_provider(
    monkeypatch, tmp_path
):
    workflow_calls = []

    def workflow_factory():
        workflow_calls.append("created")
        raise AssertionError("the provider workflow must not be created")

    monkeypatch.setattr(console_module, "QAWorkflow", workflow_factory)

    service = EvaluationConsoleService(
        tmp_path,
        agent_config_path=tmp_path / "config.yaml",
        agents_dir=tmp_path,
    )
    dataset, _created = service.catalog.import_dataset(
        "Partial", "partial.json", _partial_dataset()
    )
    app = Flask(
        __name__,
        template_folder="../../src/interfaces/chat_app/templates",
        static_folder="../../src/interfaces/chat_app/static",
    )

    def allow(_permission):
        return lambda view: view

    register_evaluations(app, require_perm=allow, service=service)
    client = app.test_client()

    generated = client.post(
        f"/api/evaluations/datasets/{dataset['id']}/generate-atoms",
        json={"profile_id": "builtin"},
    )
    reviewed = client.post(
        f"/api/evaluations/datasets/{dataset['id']}/review-atoms",
    )

    assert generated.status_code == 400
    assert generated.get_json() == {
        "error": (
            "atom generation requires a dataset with zero atoms; review its "
            "existing atoms instead"
        )
    }
    assert reviewed.status_code == 201
    assert reviewed.get_json()["draft"]["items"][1]["atoms"] == []
    assert workflow_calls == []
    assert service.list_jobs() == []


def test_dataset_summary_does_not_expose_answers_or_atoms(tmp_path):
    app, _service, _permissions = _app(tmp_path)
    client = app.test_client()
    client.post(
        "/api/evaluations/datasets",
        data={"name": "Set", "file": (io.BytesIO(_dataset()), "set.json")},
        content_type="multipart/form-data",
    )

    payload = client.get("/api/evaluations/datasets").get_json()
    serialized = str(payload)

    assert "question" not in serialized
    assert "'answer'" not in serialized
    assert "expected_atoms" not in serialized


def test_path_like_catalog_identifier_is_rejected(tmp_path):
    app, _service, _permissions = _app(tmp_path)
    response = app.test_client().get("/api/evaluations/datasets/..%2F..%2Fetc%2Fpasswd")

    assert response.status_code in {404, 400}


def test_dataset_upload_is_bounded_before_catalog_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_routes, "MAX_IMPORT_BYTES", 8)
    app, _service, _permissions = _app(tmp_path)

    response = app.test_client().post(
        "/api/evaluations/datasets",
        data={"name": "Set", "file": (io.BytesIO(b"x" * 32), "set.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "dataset upload exceeds the 25 MB limit"}


def test_view_only_user_cannot_import_a_dataset(tmp_path):
    app, service, permissions = _app(
        tmp_path, denied_permissions={Permission.Evaluations.MANAGE}
    )

    response = app.test_client().post(
        "/api/evaluations/datasets",
        data={"name": "Set", "file": (io.BytesIO(_dataset()), "set.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 403
    assert service.catalog.list_datasets() == []
    assert permissions == [Permission.Evaluations.MANAGE]


def test_retry_routes_enforce_manage_and_run_permissions(tmp_path):
    atom_app, atom_service, atom_permissions = _app(
        tmp_path / "atoms",
        denied_permissions={Permission.Evaluations.MANAGE},
    )
    atom_response = atom_app.test_client().post(
        "/api/evaluations/atom-drafts/draft-id/retry-failed"
    )

    run_app, run_service, run_permissions = _app(
        tmp_path / "runs",
        denied_permissions={Permission.Evaluations.RUN},
    )
    run_response = run_app.test_client().post(
        "/api/evaluations/runs/history-id/retry-failed"
    )

    assert atom_response.status_code == 403
    assert run_response.status_code == 403
    assert atom_service.started == []
    assert run_service.started == []
    assert atom_permissions == [Permission.Evaluations.MANAGE]
    assert run_permissions == [Permission.Evaluations.RUN]


def test_cancel_route_uses_run_permission_and_returns_canceled_history(tmp_path):
    app, service, permissions = _app(tmp_path)

    response = app.test_client().post("/api/evaluations/jobs/job-run/cancel")

    assert response.status_code == 200
    assert response.get_json() == {
        "job": {"id": "job-run", "kind": "evaluation", "status": "canceled"},
        "history_id": "canceled-history-id",
    }
    assert service.started == [("cancel", "job-run")]
    assert permissions == [Permission.Evaluations.RUN]


def test_cancel_route_denies_user_without_run_permission(tmp_path):
    app, service, permissions = _app(
        tmp_path, denied_permissions={Permission.Evaluations.RUN}
    )

    response = app.test_client().post("/api/evaluations/jobs/job-run/cancel")

    assert response.status_code == 403
    assert service.started == []
    assert permissions == [Permission.Evaluations.RUN]
