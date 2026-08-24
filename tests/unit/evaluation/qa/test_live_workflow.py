import json
import sys
from collections import Counter, deque
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult

import src.evaluation.qa.workflow as workflow_module
from src.evaluation.qa.artifacts import read_json, read_jsonl
from src.evaluation.qa.oracle import OracleCallEvidence
from src.evaluation.qa.workflow import QAWorkflow


class SequenceInvoker:
    def __init__(self, values):
        self.values = deque(values)
        self.calls = []

    def invoke(self, call):
        self.calls.append((call.id, call.tool))
        value = self.values.popleft()
        if isinstance(value, Exception):
            raise value
        return (
            CallToolResult(content=[], structuredContent=value),
            OracleCallEvidence(call.id, 2, True),
        )


class EvaluatorFactory:
    def __init__(self):
        self.calls = Counter()

    def __call__(self, _profile):
        owner = self

        class Evaluator:
            def extract_gold(self, question, answer):
                owner.calls["extract"] += 1
                return {"atoms": [{"id": "required", "text": answer, "required": True}]}

            def compare(self, question, atoms, answer):
                owner.calls["compare"] += 1
                return {
                    "judgments": [
                        {
                            "atom_id": atom.id,
                            "outcome": "entailed",
                            "rationale": "deterministic",
                        }
                        for atom in atoms
                    ]
                }

        return Evaluator()


class AgentFactory:
    def __init__(self):
        self.calls = Counter()

    def __call__(self, *_args, **_kwargs):
        owner = self

        class Agent:
            tool_calls = []

            def run(self, question):
                owner.calls[question] += 1
                return "agent answer"

        return Agent()


def _dataset(path, *, include_static=False):
    items = []
    if include_static:
        items.append(
            {
                "id": "static",
                "question": "Fixed?",
                "answer": "fixed",
                "time_sensitive": False,
                "expected_atoms": [{"id": "fixed", "text": "fixed", "required": True}],
            }
        )
    items.append(
        {
            "id": "live",
            "question": "Current value?",
            "time_sensitive": True,
            "oracle": {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "lookup",
                        "server": "read-model",
                        "tool": "current",
                        "arguments": {},
                        "answer_fields": {"value": "/value"},
                        "metadata_fields": {"revision": "/revision"},
                    }
                ],
            },
        }
    )
    path.write_text(
        json.dumps({"schema_version": "qa-dataset-v2", "items": items}),
        encoding="utf-8",
    )


@pytest.fixture
def runtimes(monkeypatch):
    evaluator = EvaluatorFactory()
    agent = AgentFactory()
    config = {
        "services": {
            "chat_app": {
                "agent_class": "FakeAgent",
                "default_provider": "fake",
                "default_model": "fake-model",
            }
        }
    }
    spec = SimpleNamespace(tools=[])
    monkeypatch.setattr(workflow_module, "LangChainEvaluatorRuntime", evaluator)
    monkeypatch.setattr(workflow_module, "ArchiAgentRuntime", agent)
    monkeypatch.setattr(
        workflow_module,
        "load_agent_inputs",
        lambda *_args: (config, spec, "---\nname: fake\ntools: []\n---\n", object),
    )
    return evaluator, agent


def _run(monkeypatch, tmp_path, values, runtimes, *, include_static=False):
    dataset = tmp_path / "dataset.json"
    run_dir = tmp_path / "run"
    _dataset(dataset, include_static=include_static)
    invoker = SequenceInvoker(values)
    monkeypatch.setattr(
        workflow_module.EvaluatorMCPRegistry,
        "load",
        classmethod(lambda cls, path=None: invoker),
    )
    QAWorkflow().composite(
        dataset,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        run_dir,
    )
    return run_dir, invoker, runtimes


class TestLiveWorkflow:
    def test_persists_truthful_runtime_phase_at_each_active_boundary(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        run_dir = tmp_path / "run"
        _dataset(dataset)
        baseline = {"value": 7, "revision": "r1"}
        preparation_invoker = SequenceInvoker([baseline])
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: preparation_invoker),
        )
        workflow = QAWorkflow()
        workflow.prepare(dataset, run_dir)

        observed_phases = []

        class PhaseCheckingInvoker(SequenceInvoker):
            def invoke(self, call):
                observed_phases.append(
                    read_json(run_dir / "manifest.json")["runtime_phase"]
                )
                return super().invoke(call)

        class PhaseCheckingAgent:
            tool_calls = []

            def run(self, _question):
                observed_phases.append(
                    read_json(run_dir / "manifest.json")["runtime_phase"]
                )
                return "agent answer"

        class PhaseCheckingEvaluator:
            def compare(self, _question, atoms, _answer):
                observed_phases.append(
                    read_json(run_dir / "manifest.json")["runtime_phase"]
                )
                return {
                    "judgments": [
                        {
                            "atom_id": atom.id,
                            "outcome": "entailed",
                            "rationale": "deterministic",
                        }
                        for atom in atoms
                    ]
                }

        run_invoker = PhaseCheckingInvoker([baseline, baseline])
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: run_invoker),
        )
        monkeypatch.setattr(
            workflow_module,
            "ArchiAgentRuntime",
            lambda *_args, **_kwargs: PhaseCheckingAgent(),
        )
        workflow.run(run_dir, tmp_path / "agent.yaml", tmp_path / "agent.md")
        monkeypatch.setattr(
            workflow_module,
            "LangChainEvaluatorRuntime",
            lambda _profile: PhaseCheckingEvaluator(),
        )
        workflow.score(run_dir)

        assert observed_phases == [
            "checking_live_answers",
            "running_attempts",
            "running_attempts",
            "scoring",
        ]
        assert "runtime_phase" not in read_json(run_dir / "manifest.json")

    def test_live_extractor_failure_never_persists_selected_truth(
        self, monkeypatch, tmp_path
    ):
        dataset = tmp_path / "dataset.json"
        run_dir = tmp_path / "run"
        _dataset(dataset)
        sentinel = "RAW_SELECTED_TRUTH_SENTINEL"
        invoker = SequenceInvoker([{"value": sentinel, "revision": "r1"}])

        class EchoingExtractor:
            def extract_gold(self, question, answer):
                raise RuntimeError(f"extractor echoed {answer}")

        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )
        monkeypatch.setattr(
            workflow_module,
            "LangChainEvaluatorRuntime",
            lambda _profile: EchoingExtractor(),
        )

        QAWorkflow().prepare(dataset, run_dir)

        [record] = read_jsonl(run_dir / "preparation.jsonl")
        assert record["error"] == "Gold extraction failed (RuntimeError)."
        assert record["oracle_calls"] == [
            {"call_id": "lookup", "duration_ms": 2, "success": True}
        ]
        assert sentinel not in json.dumps(record)

    def test_failed_preparation_call_contributes_sanitized_summary_telemetry(
        self, monkeypatch, tmp_path, runtimes
    ):
        run_dir, _invoker, _runtimes = _run(
            monkeypatch,
            tmp_path,
            [RuntimeError("provider secret sentinel")],
            runtimes,
            include_static=True,
        )

        summary = read_json(run_dir / "summary.json")
        failed = read_jsonl(run_dir / "preparation.jsonl")[1]
        assert summary["oracle_calls_failed"] == 1
        assert failed["oracle_calls"][0]["success"] is False
        assert "provider secret sentinel" not in json.dumps(failed)

    def test_complete_workflow_uses_real_stdio_mcp_transport(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        run_dir = tmp_path / "run"
        registry = tmp_path / "mcp.yaml"
        _dataset(dataset)
        document = json.loads(dataset.read_text(encoding="utf-8"))
        call = document["items"][0]["oracle"]["calls"][0]
        call["tool"] = "current_capacity"
        call["arguments"] = {"service": "primary"}
        call["answer_fields"] = {"value": "/available"}
        dataset.write_text(json.dumps(document), encoding="utf-8")
        registry.write_text(
            "schema_version: qa-evaluation-mcp-v1\n"
            "servers:\n"
            "  read-model:\n"
            "    transport: stdio\n"
            f"    command: {sys.executable}\n"
            "    args: [-m, tests.unit.evaluation.qa.fake_mcp_server]\n"
            "    authentication: {mode: inherited_environment}\n",
            encoding="utf-8",
        )

        manifest = QAWorkflow().composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            run_dir,
            mcp_config_path=registry,
        )

        assert manifest["status"] == "scored"
        assert read_jsonl(run_dir / "evaluation_results.jsonl")[0]["status"] == (
            "scored"
        )
        assert [row["phase"] for row in read_jsonl(run_dir / "live_checks.jsonl")] == [
            "pre_run",
            "post_run",
        ]
        assert read_json(run_dir / "summary.json")["oracle_calls_succeeded"] == 3

    def test_console_gate_rechecks_before_authorized_partial_continue(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        run_dir = tmp_path / "run"
        _dataset(dataset, include_static=True)
        invoker = SequenceInvoker(
            [
                {"value": 7, "revision": "r1"},
                {"value": 8, "revision": "r2"},
                {"value": 8, "revision": "r3"},
            ]
        )
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )
        workflow = QAWorkflow()

        gated = workflow.composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            run_dir,
            pause_on_live_mismatch=True,
        )

        assert gated["status"] == "attention_required"
        assert gated["attention_required"]["no_agent_attempts_started"] is True
        assert not (run_dir / "answers.jsonl").exists()
        assert not (run_dir / "evaluation_results.jsonl").exists()

        workflow.run(
            run_dir,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            overwrite=True,
            pause_on_live_mismatch=True,
            authorize_staged_invalid=True,
        )
        workflow.score(run_dir)

        assert len(invoker.calls) == 3
        assert [row["item_id"] for row in read_jsonl(run_dir / "answers.jsonl")] == [
            "static"
        ]
        assert [
            row["status"] for row in read_jsonl(run_dir / "evaluation_results.jsonl")
        ] == [
            "scored",
            "live_validation_failed",
        ]

    def test_high_cardinality_gate_persists_only_compact_attention_counts(
        self, monkeypatch, tmp_path, runtimes
    ):
        item_count = 64
        dataset = tmp_path / "many-live.json"
        dataset.write_text(
            json.dumps(
                {
                    "schema_version": "qa-dataset-v2",
                    "items": [
                        {
                            "id": f"live-{index}",
                            "question": f"Current value {index}?",
                            "time_sensitive": True,
                            "oracle": {
                                "kind": "mcp",
                                "calls": [
                                    {
                                        "id": "lookup",
                                        "server": "read-model",
                                        "tool": "current",
                                        "arguments": {"index": index},
                                        "answer_fields": {"value": "/value"},
                                    }
                                ],
                            },
                        }
                        for index in range(item_count)
                    ],
                }
            ),
            encoding="utf-8",
        )
        invoker = SequenceInvoker(
            [{"value": 1}] * item_count + [{"value": 2}] * item_count
        )
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )

        manifest = QAWorkflow().composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            tmp_path / "run",
            pause_on_live_mismatch=True,
        )

        attention = manifest["attention_required"]
        assert attention["live_items"] == item_count
        assert attention["affected_item_count"] == item_count
        assert attention["reason_counts"] == {
            "answer_changed": item_count,
            "oracle_failed": 0,
        }
        assert "affected_items" not in attention
        assert len(json.dumps(attention)) < 512

    def test_all_matching_observations_are_scored(
        self, monkeypatch, tmp_path, runtimes
    ):
        value = {"value": 7, "revision": "r1"}
        run_dir, invoker, (evaluator, agent) = _run(
            monkeypatch, tmp_path, [value, value, value], runtimes
        )

        preparation = read_jsonl(run_dir / "preparation.jsonl")
        checks = read_jsonl(run_dir / "live_checks.jsonl")
        results = read_jsonl(run_dir / "evaluation_results.jsonl")

        assert len(invoker.calls) == 3
        assert preparation[0]["answer"] == {"lookup": {"value": 7}}
        assert preparation[0]["oracle_metadata"] == {"lookup": {"revision": "r1"}}
        assert [check["phase"] for check in checks] == ["pre_run", "post_run"]
        assert len(read_jsonl(run_dir / "answers.jsonl")) == 1
        assert results[0]["status"] == "scored"
        assert evaluator.calls == Counter({"extract": 1, "compare": 1})
        assert agent.calls == Counter({"Current value?": 1})
        assert read_json(run_dir / "manifest.json")["schema_version"] == "qa-v2"

    def test_pre_run_change_creates_slots_without_agent_answers(
        self, monkeypatch, tmp_path, runtimes
    ):
        baseline = {"value": 7, "revision": "r1"}
        changed = {"value": 8, "revision": "r2"}
        run_dir, invoker, (evaluator, agent) = _run(
            monkeypatch,
            tmp_path,
            [baseline, changed],
            runtimes,
            include_static=True,
        )

        answers = read_jsonl(run_dir / "answers.jsonl")
        results = read_jsonl(run_dir / "evaluation_results.jsonl")

        assert len(invoker.calls) == 2
        assert [answer["item_id"] for answer in answers] == ["static"]
        assert [result["status"] for result in results] == [
            "scored",
            "live_validation_failed",
        ]
        assert results[1]["live_validation"] == {
            "phase": "pre_run",
            "reason": "answer_changed",
            "detail": "The resolved answer no longer matches the approved baseline.",
        }
        assert agent.calls == Counter({"Fixed?": 1})
        assert evaluator.calls == Counter({"extract": 1, "compare": 1})

    def test_post_run_change_preserves_answer_but_excludes_score(
        self, monkeypatch, tmp_path, runtimes
    ):
        baseline = {"value": 7, "revision": "r1"}
        changed = {"value": 8, "revision": "r2"}
        run_dir, _invoker, (evaluator, agent) = _run(
            monkeypatch, tmp_path, [baseline, baseline, changed], runtimes
        )

        answers = read_jsonl(run_dir / "answers.jsonl")
        result = read_jsonl(run_dir / "evaluation_results.jsonl")[0]
        summary = read_json(run_dir / "summary.json")

        assert len(answers) == 1
        assert result["status"] == "live_validation_failed"
        assert result["live_validation"]["phase"] == "post_run"
        assert summary["quality_accounted_attempts"] == 0
        assert summary["attempt_lifecycle_counts"]["live_validation_failed"] == 1
        assert evaluator.calls == Counter({"extract": 1})
        assert agent.calls == Counter({"Current value?": 1})

    def test_skip_live_omits_calls_and_scoring_membership(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        run_dir = tmp_path / "run"
        _dataset(dataset, include_static=True)
        invoker = SequenceInvoker([])
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )

        QAWorkflow().composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            run_dir,
            skip_live=True,
        )

        assert invoker.calls == []
        assert [row["status"] for row in read_jsonl(run_dir / "preparation.jsonl")] == [
            "prepared",
            "skipped_live",
        ]
        assert [row["item_id"] for row in read_jsonl(run_dir / "answers.jsonl")] == [
            "static"
        ]

    def test_retry_reresolves_live_failure_and_never_reuses_an_answer(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        parent = tmp_path / "parent"
        successor = tmp_path / "successor"
        _dataset(dataset)
        baseline = {"value": 7, "revision": "r1"}
        changed = {"value": 8, "revision": "r2"}
        invoker = SequenceInvoker([baseline, changed, baseline, baseline])
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )
        workflow = QAWorkflow()
        workflow.composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            parent,
        )

        assert read_jsonl(parent / "answers.jsonl") == []
        manifest = workflow.retry(parent, successor)

        assert manifest["status"] == "scored"
        assert manifest["phases"]["run"]["attempt_slots"] == 1
        assert manifest["phases"]["run"]["actual_agent_executions"] == 1
        assert len(read_jsonl(successor / "answers.jsonl")) == 1
        assert [
            row["status"] for row in read_jsonl(successor / "evaluation_results.jsonl")
        ] == ["scored"]
        assert [
            row["phase"] for row in read_jsonl(successor / "live_checks.jsonl")
        ] == [
            "pre_run",
            "post_run",
        ]
        assert read_json(successor / "summary.json")["oracle_calls_succeeded"] == 3

    def test_retry_persistent_pre_run_failure_keeps_slot_without_an_answer(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        parent = tmp_path / "parent"
        successor = tmp_path / "successor"
        _dataset(dataset)
        baseline = {"value": 7, "revision": "r1"}
        changed = {"value": 8, "revision": "r2"}
        invoker = SequenceInvoker([baseline, changed, changed])
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )
        workflow = QAWorkflow()
        workflow.composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            parent,
        )

        manifest = workflow.retry(parent, successor)

        assert manifest["phases"]["run"]["attempt_slots"] == 1
        assert manifest["phases"]["run"]["actual_agent_executions"] == 0
        assert read_jsonl(successor / "answers.jsonl") == []
        assert read_jsonl(successor / "evaluation_results.jsonl")[0]["status"] == (
            "live_validation_failed"
        )
        assert read_json(successor / "summary.json")["oracle_calls_succeeded"] == 2

    def test_retry_post_run_failure_executes_the_complete_question_again(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        parent = tmp_path / "parent"
        successor = tmp_path / "successor"
        _dataset(dataset)
        baseline = {"value": 7, "revision": "r1"}
        changed = {"value": 8, "revision": "r2"}
        invoker = SequenceInvoker([baseline, baseline, changed, baseline, baseline])
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )
        workflow = QAWorkflow()
        workflow.composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            parent,
        )

        assert (
            read_jsonl(parent / "evaluation_results.jsonl")[0]["live_validation"][
                "phase"
            ]
            == "post_run"
        )
        assert len(read_jsonl(parent / "answers.jsonl")) == 1

        manifest = workflow.retry(parent, successor)

        assert manifest["status"] == "scored"
        assert manifest["phases"]["run"]["actual_agent_executions"] == 1
        assert len(read_jsonl(successor / "answers.jsonl")) == 1
        assert read_jsonl(successor / "evaluation_results.jsonl")[0]["status"] == (
            "scored"
        )
        _evaluator, agent = runtimes
        assert agent.calls == Counter({"Current value?": 2})

    def test_retry_writes_multiple_live_checks_in_phase_major_order(
        self, monkeypatch, tmp_path, runtimes
    ):
        dataset = tmp_path / "dataset.json"
        parent = tmp_path / "parent"
        successor = tmp_path / "successor"
        items = []
        for index in range(2):
            items.append(
                {
                    "id": f"live-{index}",
                    "question": f"Current {index}?",
                    "time_sensitive": True,
                    "oracle": {
                        "kind": "mcp",
                        "calls": [
                            {
                                "id": "lookup",
                                "server": "read-model",
                                "tool": "current",
                                "arguments": {"index": index},
                                "answer_fields": {"value": "/value"},
                            }
                        ],
                    },
                }
            )
        dataset.write_text(
            json.dumps({"schema_version": "qa-dataset-v2", "items": items}),
            encoding="utf-8",
        )
        baseline = {"value": 7}
        changed = {"value": 8}
        invoker = SequenceInvoker(
            [
                baseline,
                baseline,
                changed,
                changed,
                baseline,
                baseline,
                baseline,
                baseline,
            ]
        )
        monkeypatch.setattr(
            workflow_module.EvaluatorMCPRegistry,
            "load",
            classmethod(lambda cls, path=None: invoker),
        )
        workflow = QAWorkflow()
        workflow.composite(
            dataset,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            parent,
        )

        workflow.retry(parent, successor)

        assert [
            (row["item_id"], row["phase"])
            for row in read_jsonl(successor / "live_checks.jsonl")
        ] == [
            ("live-0", "pre_run"),
            ("live-1", "pre_run"),
            ("live-0", "post_run"),
            ("live-1", "post_run"),
        ]


def _paused_at_live_gate(monkeypatch, tmp_path):
    """Drive one run to the live-mismatch gate and return its workflow and dir."""
    dataset = tmp_path / "dataset.json"
    run_dir = tmp_path / "run"
    _dataset(dataset, include_static=True)
    invoker = SequenceInvoker(
        [
            {"value": 7, "revision": "r1"},
            {"value": 8, "revision": "r2"},
            {"value": 8, "revision": "r3"},
        ]
    )
    monkeypatch.setattr(
        workflow_module.EvaluatorMCPRegistry,
        "load",
        classmethod(lambda cls, path=None: invoker),
    )
    workflow = QAWorkflow()
    gated = workflow.composite(
        dataset,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        run_dir,
        pause_on_live_mismatch=True,
    )
    assert gated["status"] == "attention_required"
    return workflow, run_dir


@pytest.mark.parametrize(
    "artifact", ["agent_config.resolved.yaml", "agent_spec.resolved.md"]
)
def test_continue_refuses_tampered_frozen_agent_inputs(
    monkeypatch, tmp_path, runtimes, artifact
):
    """A resumed run must re-verify the agent inputs its paused run froze.

    The continuation reads those two artifacts, rewrites them, and records
    fresh digests. Without a check against the paused manifest, an edit made
    while the run waited is re-sealed as if the run had always used it: the
    system under test changes with no new run id and no evidence.
    """
    workflow, run_dir = _paused_at_live_gate(monkeypatch, tmp_path)
    target = run_dir / artifact
    target.write_text(
        target.read_text(encoding="utf-8") + "\ntampered: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as caught:
        workflow.run(
            run_dir,
            tmp_path / "agent.yaml",
            tmp_path / "agent.md",
            overwrite=True,
            pause_on_live_mismatch=True,
            authorize_staged_invalid=True,
        )

    assert str(caught.value) == f"workspace artifact hash mismatch: {artifact}"


def test_continue_accepts_frozen_agent_inputs_rewritten_byte_for_byte(
    monkeypatch, tmp_path, runtimes
):
    """The resume check reads content, so an identical rewrite still runs."""
    workflow, run_dir = _paused_at_live_gate(monkeypatch, tmp_path)
    for artifact in ("agent_config.resolved.yaml", "agent_spec.resolved.md"):
        path = run_dir / artifact
        path.write_bytes(path.read_bytes())

    manifest = workflow.run(
        run_dir,
        tmp_path / "agent.yaml",
        tmp_path / "agent.md",
        overwrite=True,
        pause_on_live_mismatch=True,
        authorize_staged_invalid=True,
    )

    assert manifest["phases"]["run"]["status"] == "completed"
    assert [row["item_id"] for row in read_jsonl(run_dir / "answers.jsonl")] == [
        "static"
    ]
