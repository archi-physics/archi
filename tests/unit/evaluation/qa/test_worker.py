import pytest

from src.evaluation.qa import worker
from src.evaluation.qa.artifacts import write_json


class _Workflow:
    def composite(self, **kwargs):
        write_json(kwargs["output_dir"] / "worker_arguments.json", {})
        return {"run_id": "run-1", "artifacts": {}}


def test_worker_runs_composite_and_publishes_history_identity(tmp_path, monkeypatch):
    output_dir = tmp_path / "runs" / "workspace"
    output_dir.mkdir(parents=True)
    write_json(output_dir / "console_metadata.json", {"name": "Worker run"})

    monkeypatch.setattr(worker, "QAWorkflow", _Workflow)

    result = worker.execute(
        {
            "operation": "composite",
            "output_dir": str(output_dir),
            "dataset": str(tmp_path / "dataset.json"),
            "agent_config": str(tmp_path / "config.yaml"),
            "agent_spec": str(tmp_path / "agent.md"),
            "evaluator_profile_path": str(tmp_path / "profile.yaml"),
            "attempts": 1,
            "run_workers": 2,
            "score_workers": 3,
        }
    )

    assert result["run_id"] == "run-1"
    assert result["history_id"]


def test_worker_rejects_unknown_request_fields(tmp_path):
    with pytest.raises(ValueError, match="invalid fields"):
        worker.execute(
            {
                "operation": "composite",
                "output_dir": str(tmp_path),
                "dataset": "dataset.json",
                "agent_config": "config.yaml",
                "agent_spec": "agent.md",
                "evaluator_profile_path": "profile.yaml",
                "attempts": 1,
                "run_workers": 1,
                "score_workers": 1,
                "unexpected": True,
            }
        )
