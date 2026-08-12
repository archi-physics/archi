# isort: skip_file
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import local
from time import perf_counter
from typing import (  # isort: skip
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from .preparation import AnswerComparator, PreparationRecord
from .schema import AttemptIdentity
from .scoring import score_attempt
from .tool_traces import serialize_tool_call_records
from .validation import validate_judgments

ExecutionTask = Tuple[PreparationRecord, Dict[str, Any]]
ScoringTask = Tuple[PreparationRecord, Dict[str, Any]]


def _batches(items: Iterable[Any], size: int) -> Iterator[List[Any]]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def _duration_ms(started_at: float) -> int:
    return max(0, int(round((perf_counter() - started_at) * 1000)))


def run_attempt(
    runtime: Any,
    prepared: PreparationRecord,
    identity: Dict[str, Any],
) -> Dict[str, Any]:
    started_at = perf_counter()
    try:
        answer = runtime.run(prepared.prepared_question)
    except Exception as exc:
        duration_ms = _duration_ms(started_at)
        return {
            **identity,
            "status": "execution_failed",
            "duration_ms": duration_ms,
            "tool_calls": serialize_tool_call_records(
                runtime.tool_calls,
                context="tested-agent tool_calls",
            ),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        **identity,
        "status": "answer_ready",
        "duration_ms": _duration_ms(started_at),
        "tool_calls": serialize_tool_call_records(
            runtime.tool_calls,
            context="tested-agent tool_calls",
        ),
        "answer": answer,
    }


def execute_attempts(
    tasks: Iterable[ExecutionTask],
    runtime_factory: Callable[[], Any],
    workers: int,
    *,
    thread_name_prefix: str,
) -> Iterator[Dict[str, Any]]:
    if workers == 1:
        runtime = None
        for prepared, identity in tasks:
            if runtime is None:
                runtime = runtime_factory()
            yield run_attempt(runtime, prepared, identity)
        return

    worker_state = local()

    def execute(task: ExecutionTask) -> Dict[str, Any]:
        runtime = worker_state.__dict__.get("runtime")
        if runtime is None:
            runtime = runtime_factory()
            worker_state.runtime = runtime
        return run_attempt(runtime, *task)

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    ) as executor:
        for batch in _batches(tasks, workers * 2):
            yield from executor.map(execute, batch)


def score_answer(
    prepared: PreparationRecord,
    answer: Dict[str, Any],
    evaluator: AnswerComparator,
) -> Dict[str, Any]:
    identity = AttemptIdentity.from_dict(answer, context="attempt").to_dict()
    gold_atoms = prepared.prepared_gold_atoms
    try:
        judgments = validate_judgments(
            evaluator.compare(
                prepared.prepared_question,
                gold_atoms,
                answer["answer"],
            ),
            gold_atoms=gold_atoms,
            context=f"comparison for attempt {answer['attempt_id']}",
        )
        if any(judgment.outcome == "unjudgeable" for judgment in judgments):
            raise ValueError("comparator returned an unjudgeable outcome")
        return {
            **identity,
            "status": "scored",
            "answer": answer["answer"],
            "judgments": [judgment.to_dict() for judgment in judgments],
            **score_attempt(gold_atoms, judgments),
        }
    except Exception as exc:
        return {
            **identity,
            "status": "evaluation_failed",
            "error": str(exc),
        }


def score_attempts(
    tasks: Iterable[ScoringTask],
    evaluator_factory: Callable[[], AnswerComparator],
    workers: int,
    *,
    thread_name_prefix: str,
) -> Iterator[Dict[str, Any]]:
    def score(
        task: ScoringTask, evaluator: Optional[AnswerComparator]
    ) -> Dict[str, Any]:
        prepared, answer = task
        if answer["status"] == "execution_failed":
            return {
                **AttemptIdentity.from_dict(answer, context="attempt").to_dict(),
                "status": "execution_failed",
                "error": answer["error"],
            }
        assert evaluator is not None
        return score_answer(prepared, answer, evaluator)

    if workers == 1:
        evaluator = None
        for task in tasks:
            if task[1]["status"] != "execution_failed" and evaluator is None:
                evaluator = evaluator_factory()
            yield score(task, evaluator)
        return

    worker_state = local()

    def score_parallel(task: ScoringTask) -> Dict[str, Any]:
        if task[1]["status"] == "execution_failed":
            return score(task, None)
        evaluator = worker_state.__dict__.get("evaluator")
        if evaluator is None:
            evaluator = evaluator_factory()
            worker_state.evaluator = evaluator
        return score(task, evaluator)

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    ) as executor:
        for batch in _batches(tasks, workers * 2):
            yield from executor.map(score_parallel, batch)
