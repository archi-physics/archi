# isort: skip_file
from __future__ import annotations

import uuid
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .artifacts import (  # isort: skip
    AtomicJsonlWriter,
    artifact_hashes,
    copy_file_atomic,
    iter_jsonl,
    sha256_file,
    utc_now,
    verify_hashes,
    write_json,
    write_jsonl,
    write_text,
    write_yaml,
)
from .constants import (  # isort: skip
    OWNED_FILES,
    PROMPT_VERSIONS,
    RUN_FILES,
    SCHEMA_VERSION,
    SCORE_FILES,
    SCORING_VERSION,
)
from .profile import load_profile
from .phases import execute_attempts, score_attempts
from .preparation import (
    PreparationRecord,
    iter_preparation_records,
    prepare_dataset_item,
)
from .runtime import (
    ArchiAgentRuntime,
    LangChainEvaluatorRuntime,
    LazyVectorstore,
    load_agent_inputs,
)
from .schema import RunManifest
from .scoring import build_summary, render_report
from .validation import dataset_source_format, iter_dataset_items
from .workspace import EvaluationWorkspace


class QAWorkflow:
    MAX_PHASE_WORKERS = 16

    @staticmethod
    def _require_positive_attempts(attempts: int) -> None:
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
            raise ValueError("attempts must be a positive integer")

    @classmethod
    def require_worker_count(cls, workers: int, field: str) -> None:
        if (
            isinstance(workers, bool)
            or not isinstance(workers, int)
            or workers < 1
            or workers > cls.MAX_PHASE_WORKERS
        ):
            raise ValueError(
                f"{field} must be an integer from 1 to {cls.MAX_PHASE_WORKERS}"
            )

    @staticmethod
    def _remove_owned(run_dir: Path, names: set) -> None:
        directories = sorted(
            name
            for name in names
            if (run_dir / name).is_dir() and not (run_dir / name).is_symlink()
        )
        if directories:
            raise ValueError(
                "QA artifact path(s) are directories: " + ", ".join(directories)
            )
        for name in names:
            path = run_dir / name
            if path.is_file() or path.is_symlink():
                path.unlink()

    @staticmethod
    def _existing_owned(run_dir: Path, names: set) -> List[str]:
        if not run_dir.exists():
            return []
        return sorted(name for name in names if (run_dir / name).exists())

    @staticmethod
    def _load_manifest(run_dir: Path) -> Dict[str, Any]:
        return EvaluationWorkspace.load_manifest(run_dir).to_dict()

    @staticmethod
    def _phase_complete(manifest: Dict[str, Any], phase: str) -> None:
        RunManifest.from_dict(manifest).require_phase_complete(phase)

    @staticmethod
    def _load_preparation(
        run_dir: Path, manifest: Dict[str, Any]
    ) -> List[PreparationRecord]:
        return EvaluationWorkspace.load_preparation(
            run_dir,
            RunManifest.from_dict(manifest),
        )

    @staticmethod
    def _iter_answer_pairs(
        run_dir: Path, manifest: Dict[str, Any]
    ) -> Iterator[Tuple[PreparationRecord, Dict[str, Any]]]:
        yield from EvaluationWorkspace.iter_answer_pairs(
            run_dir,
            RunManifest.from_dict(manifest),
        )

    def _load_retry_parent(self, run_dir: Path) -> Tuple[
        Dict[str, Any],
        List[PreparationRecord],
        Dict[str, Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        manifest, preparation, answers_by_id, results = (
            EvaluationWorkspace.load_retry_parent(run_dir)
        )
        return (
            manifest.to_dict(),
            preparation,
            answers_by_id,
            results,
        )

    @staticmethod
    def _retry_plan(
        manifest: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        retryable = [
            row
            for row in results
            if row["status"] in {"execution_failed", "evaluation_failed"}
        ]
        if not retryable:
            raise ValueError("evaluation run has no failed attempts to retry")
        return {
            "parent_run_id": manifest["run_id"],
            "retry_attempt_ids": [row["attempt_id"] for row in retryable],
            "execution_attempt_ids": [
                row["attempt_id"]
                for row in retryable
                if row["status"] == "execution_failed"
            ],
            "evaluation_attempt_ids": [
                row["attempt_id"]
                for row in retryable
                if row["status"] == "evaluation_failed"
            ],
            "carried_forward_attempt_ids": [
                row["attempt_id"] for row in results if row["status"] == "scored"
            ],
        }

    def retry_plan(self, run_dir: Path) -> Dict[str, Any]:
        manifest, _preparation, _answers, results = self._load_retry_parent(run_dir)
        return self._retry_plan(manifest, results)

    def prepare(
        self,
        dataset: Path,
        output_dir: Path,
        evaluator_profile_path: Optional[Path] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        dataset_format = dataset_source_format(dataset)
        profile = load_profile(evaluator_profile_path)
        existing = self._existing_owned(output_dir, OWNED_FILES)
        if existing and not overwrite:
            raise ValueError(
                "output directory already contains QA artifacts: "
                + ", ".join(existing)
                + "; use --overwrite"
            )
        started_at = utc_now()
        snapshot_name = f"input.snapshot.{dataset_format}"
        snapshot_path = output_dir / snapshot_name
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f".{output_dir.name}-prepare-", dir=str(output_dir.parent)
        ) as staging_dir:
            staged_snapshot = Path(staging_dir) / snapshot_name
            copy_file_atomic(dataset, staged_snapshot)
            snapshot_item_count = sum(
                1 for _item in iter_dataset_items(staged_snapshot)
            )
            evaluator = LangChainEvaluatorRuntime(profile)
            output_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                self._remove_owned(output_dir, OWNED_FILES)
            staged_snapshot.replace(snapshot_path)
        write_yaml(output_dir / "evaluator_profile.resolved.yaml", profile.to_dict())

        prepared_item_count = 0
        with AtomicJsonlWriter(output_dir / "preparation.jsonl") as preparation_writer:
            for item in iter_dataset_items(snapshot_path):
                record = prepare_dataset_item(item, evaluator)
                preparation_writer.write(record.to_dict())
                prepared_item_count += record.status == "prepared"
        prep_names = {
            snapshot_name,
            "preparation.jsonl",
            "evaluator_profile.resolved.yaml",
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": str(uuid.uuid4()),
            "status": "prepared",
            "versions": {
                "scoring": SCORING_VERSION,
                "prompts": PROMPT_VERSIONS,
            },
            "input": {
                "source_path": str(dataset.resolve()),
                "snapshot": snapshot_name,
            },
            "evaluator_profile": {
                "artifact": "evaluator_profile.resolved.yaml",
            },
            "artifacts": artifact_hashes(output_dir, prep_names),
            "phases": {
                "prepare": {
                    "status": "completed",
                    "started_at": started_at,
                    "completed_at": utc_now(),
                    "input_items": snapshot_item_count,
                    "prepared_items": prepared_item_count,
                }
            },
        }
        write_json(output_dir / "manifest.json", manifest)
        return manifest

    def run(
        self,
        run_dir: Path,
        agent_config: Path,
        agent_spec: Path,
        attempts: int = 1,
        overwrite: bool = False,
        run_workers: int = 1,
        _resolved_agent_inputs: Optional[Tuple[Dict[str, Any], Any, str, type]] = None,
    ) -> Dict[str, Any]:
        self._require_positive_attempts(attempts)
        self.require_worker_count(run_workers, "run_workers")
        manifest = self._load_manifest(run_dir)
        self._phase_complete(manifest, "prepare")
        snapshot = manifest["input"]["snapshot"]
        verify_hashes(
            run_dir,
            manifest["artifacts"],
            {
                snapshot,
                "preparation.jsonl",
                "evaluator_profile.resolved.yaml",
            },
        )
        existing = self._existing_owned(run_dir, RUN_FILES | SCORE_FILES)
        if existing and not overwrite:
            raise ValueError(
                "run or downstream artifacts already exist: "
                + ", ".join(existing)
                + "; use --overwrite"
            )
        preparation_path = run_dir / "preparation.jsonl"
        preparation_count = manifest["phases"]["prepare"]["input_items"]
        prepared_item_count = sum(
            record.status == "prepared"
            for record in iter_preparation_records(
                preparation_path,
                expected_count=preparation_count,
            )
        )
        if prepared_item_count == 0:
            raise ValueError("run requires at least one prepared item")
        config, spec, spec_text, pipeline_class = (
            _resolved_agent_inputs
            if _resolved_agent_inputs is not None
            else load_agent_inputs(agent_config, agent_spec)
        )
        if overwrite:
            self._remove_owned(run_dir, RUN_FILES | SCORE_FILES)
            manifest["phases"].pop("run", None)
            manifest["phases"].pop("score", None)
            manifest.pop("attempts", None)
            manifest.pop("agent", None)
            for name in RUN_FILES | SCORE_FILES:
                manifest["artifacts"].pop(name, None)
        write_yaml(run_dir / "agent_config.resolved.yaml", config)
        write_text(run_dir / "agent_spec.resolved.md", spec_text)
        config_hash = sha256_file(run_dir / "agent_config.resolved.yaml")
        spec_hash = sha256_file(run_dir / "agent_spec.resolved.md")
        chat = config["services"]["chat_app"]
        vectorstore = (
            LazyVectorstore(config)
            if "search_vectorstore_hybrid" in spec.tools
            else None
        )
        started_at = utc_now()
        attempt_slots = 0

        def tasks() -> Iterator[Tuple[PreparationRecord, Dict[str, Any]]]:
            for prepared in iter_preparation_records(
                preparation_path,
                expected_count=preparation_count,
            ):
                if prepared.status != "prepared":
                    continue
                for ordinal in range(1, attempts + 1):
                    yield prepared, {
                        "item_id": prepared.item_id,
                        "attempt_id": f"{prepared.item_id}-attempt-{ordinal}",
                        "ordinal": ordinal,
                        "agent_config_sha256": config_hash,
                        "agent_spec_sha256": spec_hash,
                    }

        with AtomicJsonlWriter(run_dir / "answers.jsonl") as answer_writer:
            for answer_row in execute_attempts(
                tasks(),
                lambda: ArchiAgentRuntime(
                    config,
                    spec,
                    pipeline_class,
                    vectorstore,
                ),
                run_workers,
                thread_name_prefix="archi-qa-run",
            ):
                attempt_slots += 1
                answer_writer.write(answer_row)
        manifest["attempts"] = attempts
        manifest["agent"] = {
            "agent_class": chat["agent_class"],
            "provider": chat["default_provider"],
            "model": chat["default_model"],
            "config_artifact": "agent_config.resolved.yaml",
            "spec_artifact": "agent_spec.resolved.md",
        }
        manifest["artifacts"].update(artifact_hashes(run_dir, RUN_FILES))
        manifest["phases"]["run"] = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "attempt_slots": attempt_slots,
            "workers": run_workers,
        }
        manifest["status"] = "run_completed"
        write_json(run_dir / "manifest.json", manifest)
        return manifest

    def score(
        self,
        run_dir: Path,
        evaluator_profile_path: Optional[Path] = None,
        overwrite: bool = False,
        score_workers: int = 1,
    ) -> Dict[str, Any]:
        self.require_worker_count(score_workers, "score_workers")
        manifest = self._load_manifest(run_dir)
        self._phase_complete(manifest, "prepare")
        self._phase_complete(manifest, "run")
        verify_hashes(
            run_dir,
            manifest["artifacts"],
            {
                manifest["input"]["snapshot"],
                "preparation.jsonl",
                "evaluator_profile.resolved.yaml",
                "agent_config.resolved.yaml",
                "agent_spec.resolved.md",
                "answers.jsonl",
            },
        )
        existing = self._existing_owned(run_dir, SCORE_FILES)
        if existing and not overwrite:
            raise ValueError(
                "score phase artifacts already exist: "
                + ", ".join(existing)
                + "; use --overwrite"
            )
        stored_profile = load_profile(run_dir / "evaluator_profile.resolved.yaml")
        if evaluator_profile_path is not None:
            supplied = load_profile(evaluator_profile_path)
            if supplied != stored_profile:
                raise ValueError(
                    "supplied evaluator profile does not match the prepared profile"
                )
        profile = stored_profile
        EvaluationWorkspace.validate_answer_pairs(
            run_dir,
            RunManifest.from_dict(manifest),
        )
        if overwrite:
            self._remove_owned(run_dir, SCORE_FILES)
            manifest["phases"].pop("score", None)
            for name in SCORE_FILES:
                manifest["artifacts"].pop(name, None)
        started_at = utc_now()

        with AtomicJsonlWriter(run_dir / "evaluation_results.jsonl") as result_writer:
            for result in score_attempts(
                self._iter_answer_pairs(run_dir, manifest),
                lambda: LangChainEvaluatorRuntime(profile),
                score_workers,
                thread_name_prefix="archi-qa-score",
            ):
                result_writer.write(result)

        preparation = self._load_preparation(run_dir, manifest)
        summary = build_summary(
            preparation,
            iter_jsonl(run_dir / "evaluation_results.jsonl"),
        )
        summary["provenance"] = {
            "agent_config_sha256": manifest["artifacts"]["agent_config.resolved.yaml"],
            "agent_spec_sha256": manifest["artifacts"]["agent_spec.resolved.md"],
            "evaluator_profile_sha256": manifest["artifacts"][
                "evaluator_profile.resolved.yaml"
            ],
        }
        write_json(run_dir / "summary.json", summary)
        manifest["status"] = "scored"
        manifest["phases"]["score"] = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "scored_attempts": summary["attempt_lifecycle_counts"]["scored"],
            "workers": score_workers,
        }
        manifest["artifacts"].update(
            artifact_hashes(run_dir, SCORE_FILES - {"report.md"})
        )
        write_text(run_dir / "report.md", render_report(summary, manifest))
        manifest["artifacts"]["report.md"] = sha256_file(run_dir / "report.md")
        write_json(run_dir / "manifest.json", manifest)
        return manifest

    def retry(self, parent_run_dir: Path, output_dir: Path) -> Dict[str, Any]:
        (
            parent_manifest,
            preparation,
            parent_answers,
            parent_results,
        ) = self._load_retry_parent(parent_run_dir)
        plan = self._retry_plan(parent_manifest, parent_results)
        run_workers = parent_manifest["phases"]["run"].get("workers", 1)
        score_workers = parent_manifest["phases"]["score"].get("workers", 1)
        self.require_worker_count(run_workers, "run_workers")
        self.require_worker_count(score_workers, "score_workers")
        existing = self._existing_owned(output_dir, OWNED_FILES)
        if existing:
            raise ValueError(
                "retry output directory already contains QA artifacts: "
                + ", ".join(existing)
            )
        retry_ids = set(plan["retry_attempt_ids"])
        execution_ids = set(plan["execution_attempt_ids"])
        prepared_by_id = {
            record.item_id: record
            for record in preparation
            if record.status == "prepared"
        }
        if execution_ids:
            config, spec, _spec_text, pipeline_class = load_agent_inputs(
                parent_run_dir / "agent_config.resolved.yaml",
                parent_run_dir / "agent_spec.resolved.md",
            )
            vectorstore = (
                LazyVectorstore(config)
                if "search_vectorstore_hybrid" in spec.tools
                else None
            )
        else:
            vectorstore = None

        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = parent_manifest["input"]["snapshot"]
        copied_artifacts = {
            snapshot,
            "preparation.jsonl",
            "evaluator_profile.resolved.yaml",
            "agent_config.resolved.yaml",
            "agent_spec.resolved.md",
        }
        for name in copied_artifacts:
            copy_file_atomic(parent_run_dir / name, output_dir / name)
        manifest = {
            "schema_version": parent_manifest["schema_version"],
            "run_id": str(uuid.uuid4()),
            "status": "prepared",
            "versions": deepcopy(parent_manifest["versions"]),
            "input": deepcopy(parent_manifest["input"]),
            "evaluator_profile": deepcopy(parent_manifest["evaluator_profile"]),
            "attempts": parent_manifest["attempts"],
            "agent": deepcopy(parent_manifest["agent"]),
            "artifacts": artifact_hashes(
                output_dir,
                {
                    snapshot,
                    "preparation.jsonl",
                    "evaluator_profile.resolved.yaml",
                },
            ),
            "phases": {
                "prepare": {
                    **deepcopy(parent_manifest["phases"]["prepare"]),
                    "source": "carried_forward",
                }
            },
            "retry": plan,
        }
        write_json(output_dir / "manifest.json", manifest)

        started_at = utc_now()
        execution_tasks = (
            (
                prepared_by_id[result["item_id"]],
                {
                    key: parent_answers[result["attempt_id"]][key]
                    for key in (
                        "item_id",
                        "attempt_id",
                        "ordinal",
                        "agent_config_sha256",
                        "agent_spec_sha256",
                    )
                },
            )
            for result in parent_results
            if result["attempt_id"] in execution_ids
        )
        retried_answers = {
            answer["attempt_id"]: answer
            for answer in execute_attempts(
                execution_tasks,
                lambda: ArchiAgentRuntime(
                    config,
                    spec,
                    pipeline_class,
                    vectorstore,
                ),
                run_workers,
                thread_name_prefix="archi-qa-retry-run",
            )
        }

        successor_answers = [
            retried_answers.get(
                result["attempt_id"], parent_answers[result["attempt_id"]]
            )
            for result in parent_results
        ]
        write_jsonl(output_dir / "answers.jsonl", successor_answers)
        manifest["artifacts"].update(artifact_hashes(output_dir, RUN_FILES))
        manifest["phases"]["run"] = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "attempt_slots": len(successor_answers),
            "retried_attempts": len(execution_ids),
            "workers": run_workers,
        }
        manifest["status"] = "run_completed"
        write_json(output_dir / "manifest.json", manifest)

        profile = load_profile(output_dir / "evaluator_profile.resolved.yaml")
        successor_answers_by_id = {row["attempt_id"]: row for row in successor_answers}
        started_at = utc_now()
        score_tasks = (
            (
                prepared_by_id[result["item_id"]],
                successor_answers_by_id[result["attempt_id"]],
            )
            for result in parent_results
            if result["attempt_id"] in retry_ids
        )
        retried_results = {
            result["attempt_id"]: result
            for result in score_attempts(
                score_tasks,
                lambda: LangChainEvaluatorRuntime(profile),
                score_workers,
                thread_name_prefix="archi-qa-retry-score",
            )
        }

        successor_results = [
            retried_results.get(result["attempt_id"], result)
            for result in parent_results
        ]
        write_jsonl(output_dir / "evaluation_results.jsonl", successor_results)
        summary = build_summary(
            preparation,
            successor_results,
        )
        summary["provenance"] = {
            "agent_config_sha256": manifest["artifacts"]["agent_config.resolved.yaml"],
            "agent_spec_sha256": manifest["artifacts"]["agent_spec.resolved.md"],
            "evaluator_profile_sha256": manifest["artifacts"][
                "evaluator_profile.resolved.yaml"
            ],
        }
        write_json(output_dir / "summary.json", summary)
        manifest["status"] = "scored"
        manifest["phases"]["score"] = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "scored_attempts": summary["attempt_lifecycle_counts"]["scored"],
            "retried_attempts": len(retry_ids),
            "workers": score_workers,
        }
        manifest["artifacts"].update(
            artifact_hashes(output_dir, SCORE_FILES - {"report.md"})
        )
        write_text(output_dir / "report.md", render_report(summary, manifest))
        manifest["artifacts"]["report.md"] = sha256_file(output_dir / "report.md")
        write_json(output_dir / "manifest.json", manifest)
        return manifest

    def composite(
        self,
        dataset: Path,
        agent_config: Path,
        agent_spec: Path,
        output_dir: Path,
        evaluator_profile_path: Optional[Path] = None,
        attempts: int = 1,
        overwrite: bool = False,
        run_workers: int = 1,
        score_workers: int = 1,
    ) -> Dict[str, Any]:
        self._require_positive_attempts(attempts)
        self.require_worker_count(run_workers, "run_workers")
        self.require_worker_count(score_workers, "score_workers")
        resolved_agent_inputs = load_agent_inputs(agent_config, agent_spec)
        self.prepare(dataset, output_dir, evaluator_profile_path, overwrite)
        self.run(
            output_dir,
            agent_config,
            agent_spec,
            attempts,
            overwrite=False,
            run_workers=run_workers,
            _resolved_agent_inputs=resolved_agent_inputs,
        )
        return self.score(
            output_dir,
            overwrite=False,
            score_workers=score_workers,
        )
