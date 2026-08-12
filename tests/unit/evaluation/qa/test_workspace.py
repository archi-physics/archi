import pytest

from src.evaluation.qa.artifacts import write_jsonl
from src.evaluation.qa.schema import RunManifest
from src.evaluation.qa.workspace import EvaluationWorkspace

SHA = "a" * 64


def _manifest():
    return RunManifest.from_dict(
        {
            "schema_version": "qa-v1",
            "run_id": "run-1",
            "status": "run_completed",
            "attempts": 1,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": {
                "input.snapshot.json": SHA,
                "preparation.jsonl": SHA,
            },
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
            },
        }
    )


def _write_preparation(run_dir):
    write_jsonl(
        run_dir / "preparation.jsonl",
        [
            {
                "item_id": "item",
                "status": "prepared",
                "category": None,
                "answer_mode": None,
                "answer_source": None,
                "question": "Question",
                "answer": "Gold",
                "time_sensitive": False,
                "atom_source": "supplied",
                "gold_atoms": [{"id": "A1", "text": "Gold", "required": True}],
            }
        ],
    )


def test_workspace_pairs_preparation_with_validated_attempts(tmp_path):
    _write_preparation(tmp_path)
    write_jsonl(
        tmp_path / "answers.jsonl",
        [
            {
                "item_id": "item",
                "attempt_id": "item-attempt-1",
                "ordinal": 1,
                "agent_config_sha256": SHA,
                "agent_spec_sha256": SHA,
                "status": "answer_ready",
                "duration_ms": 10,
                "tool_calls": [],
                "answer": "Answer",
            }
        ],
    )

    pairs = list(EvaluationWorkspace.iter_answer_pairs(tmp_path, _manifest()))

    assert pairs[0][0].item_id == "item"
    assert pairs[0][1]["answer"] == "Answer"
    assert EvaluationWorkspace.validate_answer_pairs(tmp_path, _manifest()) is None


def test_workspace_rejects_attempt_identity_drift(tmp_path):
    _write_preparation(tmp_path)
    write_jsonl(
        tmp_path / "answers.jsonl",
        [
            {
                "item_id": "other",
                "attempt_id": "other-attempt-1",
                "ordinal": 1,
                "agent_config_sha256": SHA,
                "agent_spec_sha256": SHA,
                "status": "answer_ready",
                "duration_ms": 10,
                "tool_calls": [],
                "answer": "Answer",
            }
        ],
    )

    with pytest.raises(ValueError, match="attempt slot identities"):
        list(EvaluationWorkspace.iter_answer_pairs(tmp_path, _manifest()))
