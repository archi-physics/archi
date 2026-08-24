# isort: skip_file
"""Deterministic local runtime for evaluation-console browser tests."""

import os
import sys
import time
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, request

import src.evaluation.qa.console as console_module
import src.evaluation.qa.jobs as jobs_module
from src.evaluation.qa.artifacts import (
    artifact_hashes,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_text,
)
from src.evaluation.qa.console import EvaluationConsoleService
from src.evaluation.qa.workflow import QAWorkflow as ProductionQAWorkflow
from src.interfaces.chat_app.evaluation_routes import register_evaluations


class FakeEvaluator:
    atom_calls = Counter()

    def extract_gold(self, question, answer):
        self.atom_calls[question] += 1
        if question == "Recover this atom?" and self.atom_calls[question] == 1:
            raise RuntimeError("Deterministic first atom attempt failed.")
        if question == "Recover this atom?":
            time.sleep(0.4)
        return {
            "atoms": [
                {
                    "id": "A1",
                    "text": answer.strip().splitlines()[0][:240],
                    "required": True,
                }
            ]
        }

    def compare(self, question, gold_atoms, answer):
        return {
            "judgments": [
                {
                    "atom_id": atom.id,
                    "outcome": "entailed",
                    "rationale": "Deterministic browser fixture.",
                }
                for atom in gold_atoms
            ]
        }


class FakeAgentRuntime:
    def __init__(self, *_args, **_kwargs):
        self.tool_calls = []

    def run(self, question):
        self.tool_calls = []
        return f"Deterministic answer for: {question}"


class FakeWorkflow:
    require_worker_count = staticmethod(ProductionQAWorkflow.require_worker_count)

    def run(self, *args, **kwargs):
        return ProductionQAWorkflow().run(*args, **kwargs)

    def score(self, *args, **kwargs):
        return ProductionQAWorkflow().score(*args, **kwargs)

    def composite(
        self,
        dataset,
        agent_config,
        agent_spec,
        output_dir,
        evaluator_profile_path=None,
        attempts=1,
        overwrite=False,
        run_workers=1,
        score_workers=1,
        mcp_config_path=None,
        trusted_dataset=False,
        pause_on_live_mismatch=False,
    ):
        from src.evaluation.qa.dataset import DatasetSchemaVersion, iter_dataset_items

        first_item = next(iter_dataset_items(dataset, allow_materialized_live=True))
        if first_item.schema_version is DatasetSchemaVersion.V2:
            return ProductionQAWorkflow().composite(
                dataset,
                agent_config,
                agent_spec,
                output_dir,
                evaluator_profile_path=evaluator_profile_path,
                attempts=attempts,
                overwrite=overwrite,
                run_workers=run_workers,
                score_workers=score_workers,
                mcp_config_path=mcp_config_path,
                trusted_dataset=trusted_dataset,
                pause_on_live_mismatch=pause_on_live_mismatch,
            )
        from src.evaluation.qa.validation import load_dataset

        all_items = load_dataset(dataset)[1]
        items = [item for item in all_items if not item.time_sensitive]
        run_id = "browser-e2e-run"
        preparation = []
        for item in all_items:
            row = {
                "item_id": item.id,
                "category": item.category,
                "answer_mode": item.answer_mode,
                "answer_source": item.answer_source,
            }
            if item.time_sensitive:
                preparation.append(
                    {
                        **row,
                        "status": "skipped_time_sensitive",
                    }
                )
                continue
            atoms = item.expected_atoms
            atom_source = "supplied"
            if atoms is None:
                gold_atoms = [
                    {
                        "id": "A1",
                        "text": item.answer.strip().splitlines()[0][:240],
                        "required": True,
                    }
                ]
                atom_source = "inferred"
            else:
                gold_atoms = [atom.to_dict() for atom in atoms]
            preparation.append(
                {
                    **row,
                    "status": "prepared",
                    "question": item.question,
                    "answer": item.answer,
                    "time_sensitive": item.time_sensitive,
                    "atom_source": atom_source,
                    "gold_atoms": gold_atoms,
                }
            )
        answers = []
        results = []
        for item in items:
            for ordinal in range(1, attempts + 1):
                attempt_id = f"{item.id}-attempt-{ordinal}"
                base = {
                    "item_id": item.id,
                    "attempt_id": attempt_id,
                    "ordinal": ordinal,
                }
                if item.id == "retry-execution":
                    answers.append(
                        {
                            **base,
                            "status": "execution_failed",
                            "duration_ms": 450 + ordinal,
                            "tool_calls": [
                                {
                                    "ordinal": 1,
                                    "name": "mock_lookup",
                                    "status": "error",
                                    "query": '{"record_id": "missing"}',
                                    "error": "Deterministic tool lookup failed.",
                                    "duration_ms": 125 + ordinal,
                                }
                            ],
                            "error": {
                                "type": "RuntimeError",
                                "message": "Deterministic execution failure.",
                            },
                        }
                    )
                    results.append(
                        {
                            **base,
                            "status": "execution_failed",
                            "error": {
                                "type": "RuntimeError",
                                "message": "Deterministic execution failure.",
                            },
                        }
                    )
                    continue
                answers.append(
                    {
                        **base,
                        "status": "answer_ready",
                        "duration_ms": 450 + ordinal,
                        "tool_calls": [
                            {
                                "ordinal": 1,
                                "name": "mock_lookup",
                                "status": "success",
                                "query": '{"record_id": "mock-entry"}',
                                "response": (
                                    '{"record_id": "mock-entry", '
                                    '"value": "<complete>42</complete>"}'
                                ),
                                "duration_ms": 125 + ordinal,
                            }
                        ],
                        "answer": item.answer,
                    }
                )
                if item.id == "retry-evaluation":
                    results.append(
                        {
                            **base,
                            "status": "evaluation_failed",
                            "error": "Deterministic comparison failure.",
                        }
                    )
                    continue
                results.append(
                    {
                        **base,
                        "status": "scored",
                        "answer": item.answer,
                        "passed": True,
                        "atom_score": 1.0,
                        "required_atom_recall": 1.0,
                        "judgments": [
                            {
                                "atom_id": atom.id,
                                "outcome": "entailed",
                                "rationale": "Deterministic browser fixture.",
                            }
                            for atom in item.expected_atoms
                        ],
                    }
                )
        (output_dir / "input.snapshot.json").write_bytes(dataset.read_bytes())
        write_jsonl(output_dir / "preparation.jsonl", preparation)
        write_jsonl(output_dir / "answers.jsonl", answers)
        write_jsonl(output_dir / "evaluation_results.jsonl", results)
        scored_count = sum(result["status"] == "scored" for result in results)
        execution_failed = sum(
            result["status"] == "execution_failed" for result in results
        )
        evaluation_failed = sum(
            result["status"] == "evaluation_failed" for result in results
        )
        quality_accounted = scored_count + execution_failed
        summary = {
            "overall_attempt_pass_rate": (
                scored_count / quality_accounted if quality_accounted else None
            ),
            "passed_attempts": scored_count,
            "quality_accounted_attempts": quality_accounted,
            "macro_mean_scored_attempt_required_atom_recall": 1.0,
            "attempt_lifecycle_counts": {
                "scored": scored_count,
                "execution_failed": execution_failed,
                "evaluation_failed": evaluation_failed,
            },
            "items": [],
        }
        write_json(output_dir / "summary.json", summary)
        write_text(output_dir / "report.md", "# Browser E2E report\n")
        artifact_names = {
            "input.snapshot.json",
            "preparation.jsonl",
            "answers.jsonl",
            "evaluation_results.jsonl",
            "summary.json",
            "report.md",
        }
        manifest = {
            "schema_version": "qa-v1",
            "run_id": run_id,
            "status": "scored",
            "attempts": attempts,
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(output_dir, artifact_names),
            "phases": {
                "prepare": {"status": "completed", "input_items": len(all_items)},
                "run": {"status": "completed", "workers": run_workers},
                "score": {
                    "status": "completed",
                    "workers": score_workers,
                    "completed_at": "2026-07-24T12:00:00+00:00",
                },
            },
        }
        write_json(output_dir / "manifest.json", manifest)
        return manifest

    def retry_plan(self, parent_run_dir):
        manifest = read_json(parent_run_dir / "manifest.json")
        if manifest.get("schema_version") == "qa-v2":
            return ProductionQAWorkflow().retry_plan(parent_run_dir)
        results = read_jsonl(parent_run_dir / "evaluation_results.jsonl")
        retryable = [
            result
            for result in results
            if result["status"] in {"execution_failed", "evaluation_failed"}
        ]
        if not retryable:
            raise ValueError("evaluation run has no failed attempts to retry")
        return {
            "parent_run_id": manifest["run_id"],
            "retry_attempt_count": len(retryable),
            "execution_attempt_count": sum(
                result["status"] == "execution_failed" for result in retryable
            ),
            "evaluation_attempt_count": sum(
                result["status"] == "evaluation_failed" for result in retryable
            ),
            "live_validation_attempt_count": 0,
            "carried_forward_attempt_count": sum(
                result["status"] == "scored" for result in results
            ),
        }

    def retry(self, parent_run_dir, output_dir, mcp_config_path=None):
        parent_manifest = read_json(parent_run_dir / "manifest.json")
        if parent_manifest.get("schema_version") == "qa-v2":
            return ProductionQAWorkflow().retry(
                parent_run_dir,
                output_dir,
                mcp_config_path=mcp_config_path,
            )
        plan = self.retry_plan(parent_run_dir)
        preparation = read_jsonl(parent_run_dir / "preparation.jsonl")
        prepared = [row for row in preparation if row["status"] == "prepared"]
        answers = {
            row["attempt_id"]: row
            for row in read_jsonl(parent_run_dir / "answers.jsonl")
        }
        parent_results = read_jsonl(parent_run_dir / "evaluation_results.jsonl")
        prepared_by_id = {row["item_id"]: row for row in prepared}
        successor_answers = []
        successor_results = []
        retry_ids = {
            result["attempt_id"]
            for result in parent_results
            if result["status"] in {"execution_failed", "evaluation_failed"}
        }
        for result in parent_results:
            attempt_id = result["attempt_id"]
            if attempt_id not in retry_ids:
                successor_answers.append(answers[attempt_id])
                successor_results.append(result)
                continue
            prepared_item = prepared_by_id[result["item_id"]]
            answer = {
                "item_id": result["item_id"],
                "attempt_id": attempt_id,
                "ordinal": result["ordinal"],
                "status": "answer_ready",
                "duration_ms": 325 + result["ordinal"],
                "tool_calls": [
                    {
                        "ordinal": 1,
                        "name": "mock_retry",
                        "status": "success",
                        "query": '{"retry": true}',
                        "response": '{"recovered": true}',
                        "duration_ms": 100 + result["ordinal"],
                    }
                ],
                "answer": answers[attempt_id].get(
                    "answer", f"Recovered answer for {result['item_id']}"
                ),
            }
            successor_answers.append(answer)
            successor_results.append(
                {
                    "item_id": result["item_id"],
                    "attempt_id": attempt_id,
                    "ordinal": result["ordinal"],
                    "status": "scored",
                    "answer": answer["answer"],
                    "passed": True,
                    "atom_score": 1.0,
                    "required_atom_recall": 1.0,
                    "judgments": [
                        {
                            "atom_id": atom["id"],
                            "outcome": "entailed",
                            "rationale": "Deterministic retry succeeded.",
                        }
                        for atom in prepared_item["gold_atoms"]
                    ],
                }
            )
        (output_dir / "input.snapshot.json").write_bytes(
            (parent_run_dir / "input.snapshot.json").read_bytes()
        )
        write_jsonl(output_dir / "preparation.jsonl", preparation)
        write_jsonl(output_dir / "answers.jsonl", successor_answers)
        write_jsonl(output_dir / "evaluation_results.jsonl", successor_results)
        summary = {
            "overall_attempt_pass_rate": 1.0,
            "passed_attempts": len(successor_results),
            "quality_accounted_attempts": len(successor_results),
            "macro_mean_scored_attempt_required_atom_recall": 1.0,
            "attempt_lifecycle_counts": {
                "scored": len(successor_results),
                "execution_failed": 0,
                "evaluation_failed": 0,
            },
            "items": [],
        }
        write_json(output_dir / "summary.json", summary)
        write_text(output_dir / "report.md", "# Browser E2E retry report\n")
        artifact_names = {
            "input.snapshot.json",
            "preparation.jsonl",
            "answers.jsonl",
            "evaluation_results.jsonl",
            "summary.json",
            "report.md",
        }
        manifest = {
            "schema_version": "qa-v1",
            "run_id": "browser-e2e-retry-run",
            "status": "scored",
            "attempts": parent_manifest["attempts"],
            "input": {"snapshot": "input.snapshot.json"},
            "artifacts": artifact_hashes(output_dir, artifact_names),
            "phases": {
                "prepare": {
                    "status": "completed",
                    "input_items": parent_manifest["phases"]["prepare"]["input_items"],
                },
                "run": {
                    "status": "completed",
                    "workers": parent_manifest["phases"]["run"].get("workers", 1),
                },
                "score": {
                    "status": "completed",
                    "workers": parent_manifest["phases"]["score"].get("workers", 1),
                    "completed_at": "2026-07-24T12:01:00+00:00",
                },
            },
            "retry": plan,
        }
        write_json(output_dir / "manifest.json", manifest)
        return manifest


def create_app():
    repository = Path(__file__).resolve().parents[2]
    app = Flask(
        __name__,
        template_folder=str(repository / "src/interfaces/chat_app/templates"),
        static_folder=str(repository / "src/interfaces/chat_app/static"),
    )
    root = Path(os.environ["EVALUATION_TEST_ROOT"])
    config_path = root / "config.yaml"
    write_text(
        config_path,
        "services:\n  chat_app:\n    agent_class: CMSCompOpsAgent\n"
        "    default_provider: fake\n    default_model: fake-model\n",
    )
    live_value_path = root / "live-value.txt"
    write_text(live_value_path, "7\n")
    os.environ["QA_FAKE_MCP_VALUE_FILE"] = str(live_value_path)
    mcp_config_path = root / "qa_evaluation_mcp.yaml"
    write_text(
        mcp_config_path,
        "schema_version: qa-evaluation-mcp-v1\n"
        "servers:\n"
        "  fixture:\n"
        "    transport: stdio\n"
        f"    command: {sys.executable}\n"
        "    args: [-m, tests.unit.evaluation.qa.fake_mcp_server]\n"
        "    authentication: {mode: inherited_environment}\n",
    )
    console_module.LangChainEvaluatorRuntime = lambda profile: FakeEvaluator()
    console_module.QAWorkflow = FakeWorkflow
    production_popen = jobs_module.subprocess.Popen

    def evaluation_test_popen(command, *args, **kwargs):
        command = list(command)
        if command[1:3] == ["-m", "src.evaluation.qa.worker"]:
            command[2] = "tests.ui.evaluation_test_worker"
        return production_popen(command, *args, **kwargs)

    jobs_module.subprocess.Popen = evaluation_test_popen
    service = EvaluationConsoleService(
        root,
        agent_config_path=config_path,
        agents_dir=repository / "examples/agents",
        mcp_config_path=mcp_config_path,
    )

    @app.post("/api/evaluation-test/live-value")
    def set_live_value():
        payload = request.get_json()
        value = payload.get("value") if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            return jsonify({"error": "value must be an integer"}), 400
        write_text(live_value_path, f"{value}\n")
        return jsonify({"value": value})

    def allow(_permission):
        return None

    register_evaluations(app, authorize_request=allow, service=service)
    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=2787, debug=False)
