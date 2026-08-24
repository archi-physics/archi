import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.evaluation.qa.phases as phases_module
import src.evaluation.qa.workflow as workflow_module
from src.evaluation.qa.artifacts import read_json, read_jsonl
from src.evaluation.qa.workflow import QAWorkflow
from src.evaluation.qa.workspace import EvaluationWorkspace


class _EvaluatorFactory:
    def __init__(self, not_mentioned_questions=None):
        self.calls = Counter()
        self.not_mentioned_questions = set(not_mentioned_questions or [])

    def __call__(self, profile):
        factory = self

        class Evaluator:
            def extract_gold(self, question, answer):
                factory.calls["gold"] += 1
                return {
                    "atoms": [
                        {"id": "required", "text": answer, "required": True},
                        {
                            "id": "optional",
                            "text": "Optional detail",
                            "required": False,
                        },
                    ]
                }

            def compare(self, question, gold_atoms, answer):
                factory.calls["compare"] += 1
                if answer == "malformed":
                    return {"wrong": []}
                return {
                    "judgments": [
                        {
                            "atom_id": atom.id,
                            "outcome": (
                                "not_mentioned"
                                if question in factory.not_mentioned_questions
                                else ("entailed" if atom.required else "not_mentioned")
                            ),
                            "rationale": "deterministic fake",
                        }
                        for atom in gold_atoms
                    ]
                }

        return Evaluator()


class _AgentFactory:
    def __init__(self, failures=None, malformed_questions=None, tool_calls=None):
        self.calls = Counter()
        self.failures = failures or set()
        self.malformed_questions = malformed_questions or set()
        self.tool_calls = tool_calls or []

    def __call__(self, config, spec, pipeline_class, vectorstore=None):
        factory = self

        class Agent:
            def __init__(self):
                self.tool_calls = []

            def run(self, question):
                self.tool_calls = [dict(call) for call in factory.tool_calls]
                factory.calls[question] += 1
                ordinal = factory.calls[question]
                if (question, ordinal) in factory.failures:
                    raise RuntimeError("agent failed")
                return (
                    "malformed" if question in factory.malformed_questions else "answer"
                )

        return Agent()


@pytest.fixture
def agent_inputs(monkeypatch):
    config = {
        "services": {
            "chat_app": {
                "agent_class": "FakeAgent",
                "default_provider": "fake",
                "default_model": "fake-model",
            }
        }
    }
    spec = SimpleNamespace(tools=["fake"])
    monkeypatch.setattr(
        workflow_module,
        "load_agent_inputs",
        lambda config_path, spec_path: (
            config,
            spec,
            "---\nname: Fake\ntools: [fake]\n---\nPrompt\n",
            object,
        ),
    )
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", _AgentFactory())
    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", _EvaluatorFactory()
    )
    return config


def _dataset(path):
    path.write_text(
        json.dumps(
            [
                {
                    "id": "inferred",
                    "question": "inferred question",
                    "answer": "expected",
                    "time_sensitive": False,
                },
                {
                    "id": "supplied",
                    "question": "supplied question",
                    "answer": "supplied expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {
                            "id": "required",
                            "text": "supplied expected",
                            "required": True,
                        },
                        {
                            "id": "optional",
                            "text": "Optional detail",
                            "required": False,
                        },
                    ],
                },
                {
                    "id": "live",
                    "question": "live question",
                    "answer": "current",
                    "time_sensitive": True,
                },
            ]
        )
    )


def _semantic_artifacts(run_dir):
    artifacts = {
        name: read_jsonl(run_dir / name)
        for name in (
            "preparation.jsonl",
            "answers.jsonl",
            "evaluation_results.jsonl",
        )
    }
    artifacts["answers.jsonl"] = [
        {key: value for key, value in row.items() if key != "duration_ms"}
        for row in artifacts["answers.jsonl"]
    ]
    return artifacts | {"summary.json": read_json(run_dir / "summary.json")}


def test_composite_and_staged_workflows_are_equivalent_at_four_attempts(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    staged = tmp_path / "staged"
    composite = tmp_path / "composite"

    staged_evaluator = _EvaluatorFactory()
    staged_agent = _AgentFactory()
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", staged_agent)
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", staged_evaluator)
    staged_workflow = QAWorkflow()
    staged_workflow.prepare(dataset, staged)
    staged_workflow.run(staged, tmp_path / "agent.yaml", tmp_path / "agent.md", 4)
    staged_workflow.score(staged)

    composite_evaluator = _EvaluatorFactory()
    composite_agent = _AgentFactory()
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", composite_agent)
    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", composite_evaluator
    )
    QAWorkflow().composite(
        dataset,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        composite,
        attempts=4,
    )

    assert _semantic_artifacts(staged) == _semantic_artifacts(composite)
    assert staged_evaluator.calls == Counter({"compare": 8, "gold": 1})
    assert staged_agent.calls == Counter(
        {"inferred question": 4, "supplied question": 4}
    )
    summary = read_json(staged / "summary.json")
    assert summary["item_lifecycle_counts"] == {
        "prepared": 2,
        "preparation_failed": 0,
        "skipped_time_sensitive": 1,
        "skipped_live": 0,
    }
    assert summary["attempt_lifecycle_counts"] == {
        "execution_failed": 0,
        "evaluation_failed": 0,
        "scored": 8,
        "live_validation_failed": 0,
    }
    assert summary["overall_attempt_pass_rate"] == 1.0
    assert all(
        isinstance(row["duration_ms"], int) and row["duration_ms"] >= 0
        for row in read_jsonl(staged / "answers.jsonl")
    )
    assert set(summary["provenance"]) == {
        "agent_config_sha256",
        "agent_spec_sha256",
        "evaluator_profile_sha256",
    }
    manifest = read_json(staged / "manifest.json")
    assert manifest["versions"] == {
        "scoring": "1",
        "prompts": {
            "gold": "qa-gold-atoms-v1",
            "comparator": "qa-answer-comparator-v1",
        },
    }
    assert "code_revision" not in manifest
    assert "sha256" not in manifest["input"]
    assert "sha256" not in manifest["evaluator_profile"]
    assert set(manifest["agent"]) == {
        "agent_class",
        "provider",
        "model",
        "config_artifact",
        "spec_artifact",
    }
    report = (staged / "report.md").read_text()
    assert "- Agent class: `FakeAgent`" in report
    assert "- Agent:" not in report
    assert "fake-model" not in report


def test_run_and_score_workers_overlap_with_isolated_runtimes_and_ordered_artifacts(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "parallel.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": item_id,
                    "question": question,
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                }
                for item_id, question in (
                    ("slow", "slow question"),
                    ("fast", "fast question"),
                )
            ]
        )
    )
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)

    shared_vectorstore = object()
    vectorstore_initializations = []
    monkeypatch.setattr(
        workflow_module,
        "LazyVectorstore",
        lambda config: vectorstore_initializations.append(config) or shared_vectorstore,
    )
    monkeypatch.setattr(
        workflow_module,
        "load_agent_inputs",
        lambda config_path, spec_path: (
            agent_inputs,
            SimpleNamespace(tools=["search_vectorstore_hybrid"]),
            "---\nname: Fake\ntools: [search_vectorstore_hybrid]\n---\nPrompt\n",
            object,
        ),
    )

    class AgentFactory:
        def __init__(self):
            self.instances = 0
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()
            self.started = threading.Barrier(2)

        def __call__(self, config, spec, pipeline_class, vectorstore=None):
            factory = self
            factory.instances += 1
            assert vectorstore is shared_vectorstore

            class Agent:
                def __init__(self):
                    self.tool_calls = []
                    self.active = False

                def run(self, question):
                    with factory.lock:
                        assert self.active is False
                        self.active = True
                        factory.active += 1
                        factory.maximum_active = max(
                            factory.maximum_active, factory.active
                        )
                    try:
                        factory.started.wait(timeout=2)
                        if question == "slow question":
                            time.sleep(0.04)
                        return f"answer for {question}"
                    finally:
                        with factory.lock:
                            self.active = False
                            factory.active -= 1

            return Agent()

    agent_factory = AgentFactory()
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent_factory)
    workflow.run(
        run_dir,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        run_workers=2,
    )

    answers = read_jsonl(run_dir / "answers.jsonl")
    assert [row["item_id"] for row in answers] == ["slow", "fast"]
    assert agent_factory.maximum_active == 2
    assert agent_factory.instances == 2
    assert vectorstore_initializations == [agent_inputs]

    class EvaluatorFactory:
        def __init__(self):
            self.instances = 0
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()
            self.started = threading.Barrier(2)

        def __call__(self, profile):
            factory = self
            factory.instances += 1

            class Evaluator:
                def __init__(self):
                    self.active = False

                def compare(self, question, gold_atoms, answer):
                    with factory.lock:
                        assert self.active is False
                        self.active = True
                        factory.active += 1
                        factory.maximum_active = max(
                            factory.maximum_active, factory.active
                        )
                    try:
                        factory.started.wait(timeout=2)
                        if question == "slow question":
                            time.sleep(0.04)
                        return {
                            "judgments": [
                                {
                                    "atom_id": atom.id,
                                    "outcome": "entailed",
                                    "rationale": "deterministic fake",
                                }
                                for atom in gold_atoms
                            ]
                        }
                    finally:
                        with factory.lock:
                            self.active = False
                            factory.active -= 1

            return Evaluator()

    evaluator_factory = EvaluatorFactory()
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", evaluator_factory)
    manifest = workflow.score(run_dir, score_workers=2)

    results = read_jsonl(run_dir / "evaluation_results.jsonl")
    assert [row["item_id"] for row in results] == ["slow", "fast"]
    assert evaluator_factory.maximum_active == 2
    assert evaluator_factory.instances == 2
    assert manifest["phases"]["run"]["workers"] == 2
    assert manifest["phases"]["score"]["workers"] == 2


def test_composite_finishes_run_phase_before_parallel_scoring(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "barrier.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": f"item-{index}",
                    "question": f"question {index}",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                }
                for index in range(4)
            ]
        )
    )
    state = {"completed_runs": 0}
    lock = threading.Lock()

    class Agent:
        tool_calls = []

        def run(self, question):
            with lock:
                state["completed_runs"] += 1
            return "answer"

    class Evaluator:
        def extract_gold(self, question, answer):
            raise AssertionError("supplied atoms must not be regenerated")

        def compare(self, question, gold_atoms, answer):
            with lock:
                assert state["completed_runs"] == 4
            return {
                "judgments": [
                    {
                        "atom_id": atom.id,
                        "outcome": "entailed",
                        "rationale": "deterministic fake",
                    }
                    for atom in gold_atoms
                ]
            }

    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", lambda *args: Agent())
    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", lambda profile: Evaluator()
    )

    QAWorkflow().composite(
        dataset,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        tmp_path / "run",
        run_workers=2,
        score_workers=2,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_workers", 0),
        ("run_workers", 17),
        ("run_workers", True),
        ("score_workers", 0),
        ("score_workers", 17),
        ("score_workers", 1.5),
    ],
)
def test_composite_rejects_invalid_worker_counts_before_inputs_or_providers(
    field, value, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        workflow_module,
        "load_agent_inputs",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("agent inputs must not be loaded")
        ),
    )

    with pytest.raises(ValueError, match=f"{field} must be an integer from 1 to 16"):
        QAWorkflow().composite(
            tmp_path / "dataset.json",
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            tmp_path / "run",
            **{field: value},
        )

    assert not (tmp_path / "run").exists()


def test_failure_accounting_preserves_slots_and_denominators(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "item",
                    "question": "question",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                },
                {
                    "id": "bad-eval",
                    "question": "bad eval",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                },
            ]
        )
    )
    evaluator = _EvaluatorFactory()
    agent = _AgentFactory(failures={("question", 2)}, malformed_questions={"bad eval"})
    run_dir = tmp_path / "run"

    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent)
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", evaluator)
    QAWorkflow().composite(
        dataset, tmp_path / "agent.yaml", tmp_path / "agent.md", run_dir, attempts=2
    )

    results = read_jsonl(run_dir / "evaluation_results.jsonl")
    assert Counter(row["status"] for row in results) == Counter(
        {"scored": 1, "execution_failed": 1, "evaluation_failed": 2}
    )
    summary = read_json(run_dir / "summary.json")
    assert summary["quality_accounted_attempts"] == 2
    assert summary["overall_attempt_pass_rate"] == 0.5
    assert summary["item_macro_exclusion_count"] == 1


def test_run_persists_agent_duration_for_success_and_failure(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "latency-dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "success",
                    "question": "successful question",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                },
                {
                    "id": "failure",
                    "question": "failing question",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                },
            ]
        )
    )
    ticks = iter((10.0, 10.125, 20.0, 20.5))
    monkeypatch.setattr(phases_module, "perf_counter", lambda: next(ticks))
    run_dir = tmp_path / "run"
    agent = _AgentFactory(
        failures={("failing question", 1)},
        tool_calls=[
            {
                "ordinal": 1,
                "name": "search",
                "status": "success",
                "query": '{"query": "complete"}',
                "response": '{"matches": ["answer"]}',
                "duration_ms": 75,
            }
        ],
    )
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent)
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)

    workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")

    answers = {row["item_id"]: row for row in read_jsonl(run_dir / "answers.jsonl")}
    assert answers["success"]["status"] == "answer_ready"
    assert answers["success"]["duration_ms"] == 125
    assert answers["success"]["tool_calls"] == [
        {
            "ordinal": 1,
            "name": "search",
            "status": "success",
            "query": '{"query": "complete"}',
            "response": '{"matches": ["answer"]}',
            "duration_ms": 75,
        }
    ]
    assert answers["failure"]["status"] == "execution_failed"
    assert answers["failure"]["duration_ms"] == 500
    assert answers["failure"]["tool_calls"] == answers["success"]["tool_calls"]


def test_run_stops_failure_timer_before_formatting_the_exception(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "latency-dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "failure",
                    "question": "failing question",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                }
            ]
        )
    )
    clock = {"now": 20.0}
    monkeypatch.setattr(phases_module, "perf_counter", lambda: clock["now"])

    class SlowStringError(RuntimeError):
        def __str__(self):
            clock["now"] = 21.5
            return "agent failed"

    class FailingAgentFactory:
        def __call__(self, config, spec, pipeline_class, vectorstore=None):
            class Agent:
                def __init__(self):
                    self.tool_calls = []

                def run(self, question):
                    clock["now"] = 20.5
                    raise SlowStringError()

            return Agent()

    run_dir = tmp_path / "run"
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", FailingAgentFactory())
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)

    workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")

    [answer] = read_jsonl(run_dir / "answers.jsonl")
    assert answer["duration_ms"] == 500
    assert answer["error"] == {
        "type": "SlowStringError",
        "message": "agent failed",
    }


def test_score_does_not_initialize_evaluator_when_all_executions_failed(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "item",
                    "question": "question",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                }
            ]
        )
    )
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        workflow_module,
        "ArchiAgentRuntime",
        _AgentFactory(failures={("question", 1)}),
    )
    setup_workflow = QAWorkflow()
    setup_workflow.prepare(dataset, run_dir)
    setup_workflow.run(
        run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md", attempts=1
    )

    def unexpected_evaluator(profile):
        raise AssertionError("evaluator must not be initialized")

    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", unexpected_evaluator
    )
    QAWorkflow().score(run_dir)

    assert read_jsonl(run_dir / "evaluation_results.jsonl")[0]["status"] == (
        "execution_failed"
    )


def test_score_validates_then_evaluates_answer_pairs_one_at_a_time(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    workflow.run(
        run_dir,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        attempts=2,
    )
    evaluator = _EvaluatorFactory()
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", evaluator)
    real_pairs = workflow._iter_answer_pairs
    real_validate = EvaluationWorkspace.validate_answer_pairs
    real_load_preparation = workflow._load_preparation
    real_build_summary = workflow_module.build_summary
    state = {"validated": False, "scoring_yields": 0}

    def guarded_validate(validate_run_dir, manifest):
        assert evaluator.calls["compare"] == 0
        real_validate(validate_run_dir, manifest)
        state["validated"] = True

    def guarded_pairs(pair_run_dir, manifest):
        assert state["validated"]
        for prepared, answer in real_pairs(pair_run_dir, manifest):
            assert evaluator.calls["compare"] == state["scoring_yields"]
            state["scoring_yields"] += 1
            yield prepared, answer

    def guarded_load_preparation(load_run_dir, manifest):
        assert evaluator.calls["compare"] == 4
        return real_load_preparation(load_run_dir, manifest)

    def guarded_build_summary(*args, **kwargs):
        assert kwargs.get("item_sink") is not None
        return real_build_summary(*args, **kwargs)

    monkeypatch.setattr(
        EvaluationWorkspace,
        "validate_answer_pairs",
        guarded_validate,
    )
    monkeypatch.setattr(workflow, "_iter_answer_pairs", guarded_pairs)
    monkeypatch.setattr(workflow, "_load_preparation", guarded_load_preparation)
    monkeypatch.setattr(workflow_module, "build_summary", guarded_build_summary)

    workflow.score(run_dir)

    assert state == {"validated": True, "scoring_yields": 4}
    assert evaluator.calls == Counter({"compare": 4})
    assert len(read_jsonl(run_dir / "evaluation_results.jsonl")) == 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "fewer terminal slots"),
        ("extra", "more terminal slots"),
        ("reordered", "attempt slot identities"),
    ],
)
def test_score_rejects_invalid_answer_sequence_before_evaluator_calls(
    mutation, message, agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")
    answers_path = run_dir / "answers.jsonl"
    answers = read_jsonl(answers_path)
    if mutation == "missing":
        answers.pop()
    elif mutation == "extra":
        answers.append(dict(answers[-1]))
    else:
        answers.reverse()
    answers_path.write_text(
        "".join(json.dumps(answer) + "\n" for answer in answers),
        encoding="utf-8",
    )
    manifest = read_json(run_dir / "manifest.json")
    manifest["artifacts"]["answers.jsonl"] = workflow_module.sha256_file(answers_path)
    workflow_module.write_json(run_dir / "manifest.json", manifest)

    def unexpected_evaluator(profile):
        raise AssertionError("invalid answer sequences must fail before evaluation")

    monkeypatch.setattr(
        workflow_module,
        "LangChainEvaluatorRuntime",
        unexpected_evaluator,
    )

    with pytest.raises(ValueError, match=message):
        workflow.score(run_dir)

    assert not (run_dir / "evaluation_results.jsonl").exists()


def test_score_model_initialization_failure_aborts_phase(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    setup_workflow = QAWorkflow()
    setup_workflow.prepare(dataset, run_dir)
    setup_workflow.run(
        run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md", attempts=1
    )

    def failing_evaluator(profile):
        raise TypeError("temperature is not accepted")

    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", failing_evaluator)
    with pytest.raises(TypeError, match="temperature is not accepted"):
        QAWorkflow().score(run_dir)

    for name in (
        "evaluation_results.jsonl",
        "summary.json",
        "report.md",
    ):
        assert not (run_dir / name).exists()


def test_run_runtime_construction_failure_aborts_without_attempt_artifacts(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        workflow_module,
        "ArchiAgentRuntime",
        lambda config, spec, pipeline_class, vectorstore=None: (_ for _ in ()).throw(
            RuntimeError("agent runtime failed to initialize")
        ),
    )
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)

    with pytest.raises(RuntimeError, match="runtime failed to initialize"):
        workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")

    assert not (run_dir / "answers.jsonl").exists()


def test_hash_tamper_fails_before_agent_call(agent_inputs, monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    agent = _AgentFactory()
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent)
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    with (run_dir / "preparation.jsonl").open("a") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")

    assert agent.calls == Counter()


def test_run_validates_then_streams_preparation_records(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    agent = _AgentFactory()
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent)
    real_iterator = workflow_module.iter_preparation_records
    state = {"iterations": 0, "previous_question": None}

    def guarded_iterator(path, *, expected_count=None):
        state["iterations"] += 1
        iteration = state["iterations"]
        for record in real_iterator(path, expected_count=expected_count):
            if iteration == 1:
                assert agent.calls == Counter()
            else:
                previous_question = state["previous_question"]
                if previous_question is not None:
                    assert agent.calls[previous_question] == 2
                if record.status == "prepared":
                    state["previous_question"] = record.prepared_question
            yield record

    monkeypatch.setattr(
        workflow_module,
        "iter_preparation_records",
        guarded_iterator,
    )
    monkeypatch.setattr(
        workflow,
        "_load_preparation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("run must not materialize preparation records")
        ),
    )

    workflow.run(
        run_dir,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        attempts=2,
    )

    assert state["iterations"] == 2
    assert agent.calls == Counter({"inferred question": 2, "supplied question": 2})
    assert len(read_jsonl(run_dir / "answers.jsonl")) == 4


def test_run_preflight_rejects_late_invalid_preparation_before_agent_calls(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    preparation_path = run_dir / "preparation.jsonl"
    rows = read_jsonl(preparation_path)
    rows[-1]["unexpected"] = "invalid"
    preparation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = read_json(run_dir / "manifest.json")
    manifest["artifacts"]["preparation.jsonl"] = workflow_module.sha256_file(
        preparation_path
    )
    workflow_module.write_json(run_dir / "manifest.json", manifest)
    agent = _AgentFactory()
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent)

    with pytest.raises(ValueError, match="invalid fields.*unexpected"):
        workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")

    assert agent.calls == Counter()
    assert not (run_dir / "answers.jsonl").exists()


def test_prepare_validates_all_rows_before_evaluator_calls(monkeypatch, tmp_path):
    dataset = tmp_path / "invalid.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "valid",
                    "answer": "valid",
                    "time_sensitive": False,
                },
                {
                    "question": "invalid",
                    "answer": "invalid",
                    "time_sensitive": False,
                    "unexpected": "not allowed",
                },
            ]
        )
    )

    def unexpected_evaluator(profile):
        raise AssertionError(
            "invalid snapshots must fail before evaluator construction"
        )

    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", unexpected_evaluator
    )

    with pytest.raises(ValueError, match="unknown field.*unexpected"):
        QAWorkflow().prepare(dataset, tmp_path / "run")

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("dataset_format", ["json", "jsonl"])
def test_prepare_streams_snapshot_and_preparation(
    dataset_format, monkeypatch, tmp_path
):
    rows = [
        {
            "id": "inferred",
            "question": "inferred question",
            "answer": "expected",
            "time_sensitive": False,
        },
        {
            "id": "supplied",
            "question": "supplied question",
            "answer": "supplied expected",
            "time_sensitive": False,
            "expected_atoms": [
                {
                    "id": "required",
                    "text": "supplied expected",
                    "required": True,
                }
            ],
        },
    ]
    dataset = tmp_path / f"dataset.{dataset_format}"
    dataset_bytes = (
        json.dumps(rows, indent=2).encode("utf-8")
        if dataset_format == "json"
        else ("\n".join(json.dumps(row) for row in rows) + "\n\n").encode("utf-8")
    )
    dataset.write_bytes(dataset_bytes)
    run_dir = tmp_path / "run"
    evaluator = _EvaluatorFactory()
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", evaluator)
    real_iter_dataset_items = workflow_module.iter_dataset_items
    real_prepare_dataset_item = workflow_module.prepare_dataset_item
    iteration = {"count": 0, "awaiting_preparation": None}

    def guarded_items(path):
        iteration["count"] += 1
        for item in real_iter_dataset_items(path):
            if iteration["count"] == 2:
                if iteration["awaiting_preparation"] is not None:
                    raise AssertionError(
                        "preparation must consume each item before loading the next"
                    )
                iteration["awaiting_preparation"] = item.id
            yield item

    def tracked_preparation(item, extractor):
        assert iteration["awaiting_preparation"] == item.id
        iteration["awaiting_preparation"] = None
        return real_prepare_dataset_item(item, extractor)

    monkeypatch.setattr(workflow_module, "iter_dataset_items", guarded_items)
    monkeypatch.setattr(workflow_module, "prepare_dataset_item", tracked_preparation)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (_ for _ in ()).throw(
            AssertionError("preparation must not read complete file bytes")
        ),
    )

    manifest = QAWorkflow().prepare(dataset, run_dir)

    with (run_dir / f"input.snapshot.{dataset_format}").open("rb") as snapshot:
        assert snapshot.read() == dataset_bytes
    assert [row["item_id"] for row in read_jsonl(run_dir / "preparation.jsonl")] == [
        "inferred",
        "supplied",
    ]
    assert manifest["phases"]["prepare"]["input_items"] == 2
    assert manifest["phases"]["prepare"]["prepared_items"] == 2
    assert evaluator.calls == Counter({"gold": 1})
    assert iteration == {"count": 2, "awaiting_preparation": None}


@pytest.mark.parametrize("dataset_format", ["json", "jsonl"])
def test_prepare_validates_complete_source_before_evaluator_calls(
    dataset_format, monkeypatch, tmp_path
):
    dataset = tmp_path / f"invalid.{dataset_format}"
    rows = [
        {
            "id": "valid",
            "question": "valid",
            "answer": "valid",
            "time_sensitive": False,
        },
        {
            "id": "invalid",
            "question": "invalid",
            "answer": "invalid",
            "time_sensitive": False,
            "unexpected": "not allowed",
        },
    ]
    dataset.write_text(
        (
            json.dumps(rows)
            if dataset_format == "json"
            else "\n".join(json.dumps(row) for row in rows) + "\n"
        ),
        encoding="utf-8",
    )

    def unexpected_evaluator(profile):
        raise AssertionError(
            "invalid snapshots must fail before evaluator construction"
        )

    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", unexpected_evaluator
    )

    with pytest.raises(ValueError, match="unknown field.*unexpected"):
        QAWorkflow().prepare(dataset, tmp_path / "run")

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("dataset_format", ["json", "jsonl"])
def test_prepare_validates_complete_snapshot_before_provider_calls(
    dataset_format, monkeypatch, tmp_path
):
    dataset = tmp_path / f"dataset.{dataset_format}"
    row = {
        "id": "valid",
        "question": "valid",
        "answer": "valid",
        "time_sensitive": False,
    }
    dataset.write_text(
        json.dumps([row]) if dataset_format == "json" else json.dumps(row) + "\n",
        encoding="utf-8",
    )

    def unexpected_evaluator(profile):
        raise AssertionError(
            "invalid snapshots must fail before evaluator construction"
        )

    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", unexpected_evaluator
    )

    def copy_invalid_snapshot(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text() + " invalid", encoding="utf-8")

    monkeypatch.setattr(
        workflow_module,
        "copy_file_atomic",
        copy_invalid_snapshot,
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        QAWorkflow().prepare(dataset, tmp_path / "run")

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("dataset_format", ["json", "jsonl"])
def test_prepare_invalid_source_preserves_overwritten_workspace(
    dataset_format, tmp_path
):
    dataset = tmp_path / f"invalid.{dataset_format}"
    dataset.write_text(
        (
            '[{"id":"valid","question":"Q","answer":"A",'
            '"time_sensitive":false},{"invalid":true}]'
            if dataset_format == "json"
            else '{"id":"valid","question":"Q","answer":"A",'
            '"time_sensitive":false}\n{"invalid":true}\n'
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = run_dir / "report.md"
    report.write_text("previous report", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown field.*invalid"):
        QAWorkflow().prepare(dataset, run_dir, overwrite=True)

    assert report.read_text(encoding="utf-8") == "previous report"
    assert sorted(path.name for path in run_dir.iterdir()) == ["report.md"]


def test_prepare_overwrite_preserves_unknown_files_and_removes_downstream(
    agent_inputs, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.composite(dataset, tmp_path / "agent.yaml", tmp_path / "agent.md", run_dir)
    unknown = run_dir / "operator-notes.txt"
    unknown.write_text("keep me")

    workflow.prepare(dataset, run_dir, overwrite=True)

    assert unknown.read_text() == "keep me"
    assert not (run_dir / "answers.jsonl").exists()
    assert not (run_dir / "summary.json").exists()


def test_prepare_rejects_any_existing_owned_artifact_without_overwrite(tmp_path):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.md").write_text("stale score output")

    with pytest.raises(ValueError, match="report.md.*--overwrite"):
        QAWorkflow().prepare(dataset, run_dir)


def test_prepare_does_not_recursively_delete_directory_at_artifact_path(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    nested = run_dir / "answers.jsonl"
    nested.mkdir(parents=True)
    marker = nested / "operator-data"
    marker.write_text("keep me")
    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", _EvaluatorFactory()
    )

    with pytest.raises(ValueError, match="artifact path.*directories"):
        QAWorkflow().prepare(dataset, run_dir, overwrite=True)

    assert marker.read_text() == "keep me"


def test_prepare_preflight_failure_preserves_overwritten_workspace(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = run_dir / "report.md"
    report.write_text("previous report")

    def failing_evaluator(profile):
        raise RuntimeError("provider configuration failed")

    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", failing_evaluator)
    with pytest.raises(RuntimeError, match="provider configuration failed"):
        QAWorkflow().prepare(dataset, run_dir, overwrite=True)

    assert report.read_text() == "previous report"


def test_run_rejects_downstream_score_artifact_without_overwrite(
    agent_inputs, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    (run_dir / "report.md").write_text("stale score output")

    with pytest.raises(ValueError, match="report.md.*--overwrite"):
        workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")


def test_score_rejects_any_existing_score_artifact_without_overwrite(
    agent_inputs, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")
    (run_dir / "report.md").write_text("stale score output")

    with pytest.raises(ValueError, match="report.md.*--overwrite"):
        workflow.score(run_dir)


def test_score_preflight_failure_preserves_overwritten_report(agent_inputs, tmp_path):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")
    answers_path = run_dir / "answers.jsonl"
    answers = read_jsonl(answers_path)
    answers[0]["status"] = "invalid"
    answers_path.write_text(
        "".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8"
    )
    manifest = read_json(run_dir / "manifest.json")
    manifest["artifacts"]["answers.jsonl"] = workflow_module.sha256_file(answers_path)
    workflow_module.write_json(run_dir / "manifest.json", manifest)
    report = run_dir / "report.md"
    report.write_text("previous report")

    with pytest.raises(ValueError, match="non-terminal or unsupported"):
        workflow.score(run_dir, overwrite=True)

    assert report.read_text() == "previous report"


def test_run_persists_admin_visible_answer_without_redaction(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"

    class SecretAgentFactory:
        def __call__(self, config, spec, pipeline_class, vectorstore=None):
            class Agent:
                tool_calls = []

                def run(self, question):
                    return "answer configured-secret-value"

            return Agent()

    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", SecretAgentFactory())
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)
    workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")

    answers = read_jsonl(run_dir / "answers.jsonl")
    assert {row["answer"] for row in answers} == {"answer configured-secret-value"}


def test_retry_creates_complete_successor_and_invokes_only_failed_phases(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "retry-dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": item_id,
                    "question": question,
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                }
                for item_id, question in (
                    ("scored", "scored question"),
                    ("execution", "execution question"),
                    ("evaluation", "evaluation question"),
                )
            ]
        )
    )
    parent = tmp_path / "parent"
    parent_agent = _AgentFactory(
        failures={("execution question", 1)},
        malformed_questions={"evaluation question"},
        tool_calls=[
            {
                "ordinal": 1,
                "name": "parent-tool",
                "status": "success",
                "query": "parent query",
                "response": "parent response",
                "duration_ms": 25,
            }
        ],
    )
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", parent_agent)
    monkeypatch.setattr(
        workflow_module,
        "LangChainEvaluatorRuntime",
        _EvaluatorFactory(not_mentioned_questions={"scored question"}),
    )
    QAWorkflow().composite(
        dataset,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        parent,
        run_workers=2,
        score_workers=2,
    )
    parent_bytes = {
        path.name: path.read_bytes() for path in parent.iterdir() if path.is_file()
    }
    parent_answers = {
        row["attempt_id"]: row for row in read_jsonl(parent / "answers.jsonl")
    }
    parent_results = {
        row["attempt_id"]: row
        for row in read_jsonl(parent / "evaluation_results.jsonl")
    }
    retry_agent = _AgentFactory(
        tool_calls=[
            {
                "ordinal": 1,
                "name": "retry-tool",
                "status": "success",
                "query": "retry query",
                "response": "retry response",
                "duration_ms": 40,
            }
        ]
    )

    class RecoveringEvaluatorFactory:
        def __init__(self):
            self.calls = Counter()

        def __call__(self, profile):
            factory = self

            class Evaluator:
                def compare(self, question, gold_atoms, answer):
                    factory.calls[question] += 1
                    return {
                        "judgments": [
                            {
                                "atom_id": atom.id,
                                "outcome": "entailed",
                                "rationale": "retry succeeded",
                            }
                            for atom in gold_atoms
                        ]
                    }

            return Evaluator()

    retry_evaluator = RecoveringEvaluatorFactory()
    successor = tmp_path / "successor"
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", retry_agent)
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", retry_evaluator)

    def unexpected_dataset_load(*_args, **_kwargs):
        raise AssertionError("retry must consume preparation.jsonl directly")

    monkeypatch.setattr(
        workflow_module,
        "load_dataset",
        unexpected_dataset_load,
        raising=False,
    )
    monkeypatch.setattr(
        workflow_module,
        "iter_dataset_items",
        unexpected_dataset_load,
    )
    retry_workflow = QAWorkflow()
    ticks = iter((30.0, 30.4))
    monkeypatch.setattr(phases_module, "perf_counter", lambda: next(ticks))

    manifest = retry_workflow.retry(parent, successor)

    successor_answers = {
        row["attempt_id"]: row for row in read_jsonl(successor / "answers.jsonl")
    }
    successor_results = {
        row["attempt_id"]: row
        for row in read_jsonl(successor / "evaluation_results.jsonl")
    }
    assert retry_agent.calls == Counter({"execution question": 1})
    assert retry_evaluator.calls == Counter(
        {"execution question": 1, "evaluation question": 1}
    )
    assert parent_results["scored-attempt-1"]["passed"] is False
    assert successor_answers["scored-attempt-1"] == parent_answers["scored-attempt-1"]
    assert successor_results["scored-attempt-1"] == parent_results["scored-attempt-1"]
    assert (
        successor_answers["evaluation-attempt-1"]
        == parent_answers["evaluation-attempt-1"]
    )
    assert successor_answers["execution-attempt-1"]["duration_ms"] == 400
    assert parent_answers["execution-attempt-1"]["tool_calls"][0]["name"] == (
        "parent-tool"
    )
    assert successor_answers["execution-attempt-1"]["tool_calls"] == [
        {
            "ordinal": 1,
            "name": "retry-tool",
            "status": "success",
            "query": "retry query",
            "response": "retry response",
            "duration_ms": 40,
        }
    ]
    assert {row["status"] for row in successor_results.values()} == {"scored"}
    assert manifest["retry"] == {
        "parent_run_id": read_json(parent / "manifest.json")["run_id"],
        "retry_attempt_count": 2,
        "execution_attempt_count": 1,
        "evaluation_attempt_count": 1,
        "live_validation_attempt_count": 0,
        "carried_forward_attempt_count": 1,
    }
    assert manifest["phases"]["run"]["actual_agent_executions"] == 1
    assert manifest["phases"]["run"]["workers"] == 2
    assert manifest["phases"]["score"]["workers"] == 2
    assert read_json(successor / "summary.json")["attempt_lifecycle_counts"] == {
        "execution_failed": 0,
        "evaluation_failed": 0,
        "scored": 3,
        "live_validation_failed": 0,
    }
    for artifact in (
        "input.snapshot.json",
        "preparation.jsonl",
        "evaluator_profile.resolved.yaml",
        "agent_config.resolved.yaml",
        "agent_spec.resolved.md",
    ):
        assert (successor / artifact).read_bytes() == (parent / artifact).read_bytes()
    assert {
        path.name: path.read_bytes() for path in parent.iterdir() if path.is_file()
    } == parent_bytes
    with pytest.raises(ValueError, match="no failed attempts"):
        retry_workflow.retry_plan(successor)


def test_retry_rejects_tampered_parent_before_provider_or_output(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "item",
                    "question": "question",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                }
            ]
        )
    )
    parent = tmp_path / "parent"
    monkeypatch.setattr(
        workflow_module,
        "ArchiAgentRuntime",
        _AgentFactory(failures={("question", 1)}),
    )
    QAWorkflow().composite(
        dataset,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        parent,
    )
    with (parent / "evaluation_results.jsonl").open("a") as handle:
        handle.write("{}\n")
    evaluator = _EvaluatorFactory()
    agent = _AgentFactory()
    successor = tmp_path / "successor"
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent)
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", evaluator)

    with pytest.raises(ValueError, match="hash mismatch"):
        QAWorkflow().retry(parent, successor)

    assert evaluator.calls == Counter()
    assert agent.calls == Counter()
    assert not successor.exists()


def test_retry_defaults_legacy_manifests_to_one_worker_per_phase(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "item",
                    "question": "question",
                    "answer": "expected",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "required", "text": "expected", "required": True}
                    ],
                }
            ]
        )
    )
    parent = tmp_path / "parent"
    monkeypatch.setattr(
        workflow_module,
        "ArchiAgentRuntime",
        _AgentFactory(failures={("question", 1)}),
    )
    QAWorkflow().composite(
        dataset,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        parent,
    )
    parent_manifest = read_json(parent / "manifest.json")
    parent_manifest["phases"]["run"].pop("workers")
    parent_manifest["phases"]["score"].pop("workers")
    workflow_module.write_json(parent / "manifest.json", parent_manifest)
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", _AgentFactory())
    monkeypatch.setattr(
        workflow_module, "LangChainEvaluatorRuntime", _EvaluatorFactory()
    )

    manifest = QAWorkflow().retry(parent, tmp_path / "successor")

    assert manifest["phases"]["run"]["workers"] == 1
    assert manifest["phases"]["score"]["workers"] == 1


def test_run_and_score_do_not_decode_the_input_snapshot(
    agent_inputs, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    run_dir = tmp_path / "run"
    workflow = QAWorkflow()
    workflow.prepare(dataset, run_dir)

    def unexpected_dataset_load(*_args, **_kwargs):
        raise AssertionError(
            "downstream phases must consume preparation.jsonl directly"
        )

    monkeypatch.setattr(
        workflow_module,
        "load_dataset",
        unexpected_dataset_load,
        raising=False,
    )
    monkeypatch.setattr(
        workflow_module,
        "iter_dataset_items",
        unexpected_dataset_load,
    )

    workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")
    workflow.score(run_dir)

    assert read_json(run_dir / "manifest.json")["status"] == "scored"


def test_composite_validates_selected_agent_inputs_before_gold_provider_call(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    evaluator = _EvaluatorFactory()
    monkeypatch.setattr(
        workflow_module,
        "load_agent_inputs",
        lambda config_path, spec_path: (_ for _ in ()).throw(
            ValueError("invalid agent")
        ),
    )

    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", evaluator)
    with pytest.raises(ValueError, match="invalid agent"):
        QAWorkflow().composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            tmp_path / "run",
        )

    assert evaluator.calls == Counter()
    assert not (tmp_path / "run").exists()
