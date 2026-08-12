from collections import Counter

from src.evaluation.qa.phases import execute_attempts, score_attempts
from src.evaluation.qa.preparation import PreparationRecord
from src.evaluation.qa.validation import Atom

SHA = "a" * 64


def _prepared(item_id):
    return PreparationRecord(
        item_id=item_id,
        status="prepared",
        question=f"Question {item_id}",
        answer="Gold",
        time_sensitive=False,
        gold_atoms=(Atom(id="A1", text="Gold", required=True),),
        atom_source="supplied",
    )


def _identity(item_id, ordinal=1):
    return {
        "item_id": item_id,
        "attempt_id": f"{item_id}-attempt-{ordinal}",
        "ordinal": ordinal,
        "agent_config_sha256": SHA,
        "agent_spec_sha256": SHA,
    }


def test_execute_attempts_preserves_task_order_with_worker_local_runtimes():
    constructed = Counter()

    class Runtime:
        tool_calls = []

        def __init__(self):
            constructed["runtimes"] += 1

        def run(self, question):
            return f"answer:{question}"

    tasks = [(_prepared(item_id), _identity(item_id)) for item_id in ("a", "b", "c")]

    answers = list(
        execute_attempts(
            tasks,
            Runtime,
            2,
            thread_name_prefix="test-execution",
        )
    )

    assert [row["item_id"] for row in answers] == ["a", "b", "c"]
    assert [row["answer"] for row in answers] == [
        "answer:Question a",
        "answer:Question b",
        "answer:Question c",
    ]
    assert 1 <= constructed["runtimes"] <= 2


def test_execute_attempts_does_not_construct_runtime_for_empty_retry_batch():
    def unexpected_runtime():
        raise AssertionError("runtime must not be constructed")

    assert (
        list(
            execute_attempts(
                [],
                unexpected_runtime,
                1,
                thread_name_prefix="test-empty-execution",
            )
        )
        == []
    )


def test_score_attempts_does_not_construct_evaluator_for_execution_failures():
    prepared = _prepared("item")
    failed_answer = {
        **_identity("item"),
        "status": "execution_failed",
        "duration_ms": 10,
        "tool_calls": [],
        "error": {"type": "RuntimeError", "message": "failure"},
    }

    def unexpected_evaluator():
        raise AssertionError("evaluator must not be constructed")

    assert list(
        score_attempts(
            [(prepared, failed_answer)],
            unexpected_evaluator,
            2,
            thread_name_prefix="test-scoring",
        )
    ) == [
        {
            **_identity("item"),
            "status": "execution_failed",
            "error": {"type": "RuntimeError", "message": "failure"},
        }
    ]
