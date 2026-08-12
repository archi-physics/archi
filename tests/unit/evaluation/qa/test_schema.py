import pytest

from src.evaluation.qa.schema import (  # isort: skip
    AnswerAttempt,
    AnswerStatus,
    ConsoleMetadata,
    EvaluationResult,
    EvaluationStatus,
    RunManifest,
    RunStatus,
)

SHA = "a" * 64


def _manifest(**overrides):
    raw = {
        "schema_version": "qa-v1",
        "run_id": "run-1",
        "status": "prepared",
        "input": {"snapshot": "input.snapshot.json"},
        "artifacts": {
            "input.snapshot.json": SHA,
            "preparation.jsonl": SHA,
        },
        "phases": {"prepare": {"status": "completed", "input_items": 1}},
    }
    raw.update(overrides)
    return raw


def _identity():
    return {
        "item_id": "item-1",
        "attempt_id": "item-1-attempt-1",
        "ordinal": 1,
        "agent_config_sha256": SHA,
        "agent_spec_sha256": SHA,
    }


def test_manifest_roundtrip_preserves_typed_state_and_extensions():
    raw = _manifest(extension={"kept": True})

    manifest = RunManifest.from_dict(raw)

    assert manifest.status is RunStatus.PREPARED
    assert manifest.snapshot == "input.snapshot.json"
    assert manifest.preparation_input_items == 1
    assert manifest.to_dict() == raw


def test_console_metadata_roundtrip_preserves_fields_and_extensions():
    raw = {
        "name": "Console run",
        "dataset_id": "dataset-1",
        "run_workers": 2,
        "profile_name": None,
        "extension": {"kept": True},
    }

    metadata = ConsoleMetadata.from_dict(raw)

    assert metadata.name == "Console run"
    assert metadata.dataset_id == "dataset-1"
    assert metadata.run_workers == 2
    assert metadata.to_dict() == raw


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "console metadata must be an object"),
        ({"name": 12}, "console metadata name must be a string"),
        ({"dataset_id": ""}, "console dataset ID must be a non-empty string"),
        (
            {"score_workers": True},
            "console metadata score_workers must be a positive integer",
        ),
    ],
)
def test_console_metadata_rejects_invalid_field_types(raw, message):
    with pytest.raises(ValueError, match=message):
        ConsoleMetadata.from_dict(raw)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": []}, "unsupported run schema"),
        ({"status": "unknown"}, "unsupported run status"),
        ({"phases": {}}, "manifest phase state is incomplete"),
        ({"artifacts": {}}, "manifest is missing preparation artifacts"),
        (
            {"input": {"snapshot": "input.snapshot.json", "source_path": 42}},
            "manifest input source path must be a string",
        ),
        (
            {
                "status": "run_completed",
                "attempts": 0,
                "phases": {
                    "prepare": {"status": "completed", "input_items": 1},
                    "run": {"status": "completed"},
                },
            },
            "manifest attempts must be a positive integer",
        ),
    ],
)
def test_manifest_rejects_invalid_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        RunManifest.from_dict(_manifest(**overrides))


def test_answer_and_result_models_validate_closed_statuses_and_roundtrip():
    answer_row = {
        **_identity(),
        "status": "answer_ready",
        "duration_ms": 15,
        "tool_calls": [],
        "answer": "Answer",
        "extension": {"kept": True},
    }
    result_row = {
        **_identity(),
        "status": "evaluation_failed",
        "error": "provider unavailable",
    }

    answer = AnswerAttempt.from_dict(answer_row, context="answer")
    result = EvaluationResult.from_dict(result_row, context="result")

    assert answer.status is AnswerStatus.ANSWER_READY
    assert answer.to_dict() == answer_row
    assert result.status is EvaluationStatus.EVALUATION_FAILED
    assert result.to_dict() == result_row


def test_failed_answer_requires_structured_error():
    with pytest.raises(ValueError, match="error must contain type and message"):
        AnswerAttempt.from_dict(
            {
                **_identity(),
                "status": "execution_failed",
                "duration_ms": 15,
                "tool_calls": [],
                "error": "failure",
            },
            context="answer",
        )
