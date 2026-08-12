# isort: skip_file
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from .artifacts import iter_jsonl, read_json, read_jsonl, verify_hashes
from .constants import PROMPT_VERSIONS, SCORING_VERSION
from .preparation import (  # isort: skip
    PreparationRecord,
    iter_preparation_records,
    load_preparation_records,
)
from .schema import (  # isort: skip
    AnswerAttempt,
    AnswerStatus,
    EvaluationResult,
    EvaluationStatus,
    RunManifest,
    RunStatus,
    UnsupportedSchemaError,
)


class EvaluationWorkspace:
    """Validated access to one current-schema evaluation workspace."""

    @staticmethod
    def load_manifest(run_dir: Path) -> RunManifest:
        path = run_dir / "manifest.json"
        if not path.exists():
            raise ValueError(f"run workspace has no manifest: {run_dir}")
        try:
            return RunManifest.from_dict(read_json(path))
        except UnsupportedSchemaError as exc:
            raise ValueError(
                "run workspace manifest has an unsupported schema"
            ) from exc

    @staticmethod
    def load_preparation(
        run_dir: Path, manifest: RunManifest
    ) -> List[PreparationRecord]:
        return load_preparation_records(
            run_dir / "preparation.jsonl",
            expected_count=manifest.preparation_input_items,
        )

    @staticmethod
    def iter_answer_pairs(
        run_dir: Path, manifest: RunManifest
    ) -> Iterator[Tuple[PreparationRecord, Dict[str, Any]]]:
        answers = iter_jsonl(run_dir / "answers.jsonl")
        for prepared in iter_preparation_records(
            run_dir / "preparation.jsonl",
            expected_count=manifest.preparation_input_items,
        ):
            if prepared.status != "prepared":
                continue
            for ordinal in range(1, manifest.required_attempts + 1):
                try:
                    raw_answer = next(answers)
                except StopIteration:
                    raise ValueError(
                        "run contains fewer terminal slots than the prepared workspace"
                    ) from None
                answer = AnswerAttempt.from_dict(
                    raw_answer,
                    context=f"run answer {raw_answer.get('attempt_id', ordinal)}",
                )
                expected_identity = (
                    prepared.item_id,
                    ordinal,
                    f"{prepared.item_id}-attempt-{ordinal}",
                )
                actual_identity = (
                    answer.identity.item_id,
                    answer.identity.ordinal,
                    answer.identity.attempt_id,
                )
                if actual_identity != expected_identity:
                    raise ValueError(
                        "run attempt slot identities do not match the prepared workspace"
                    )
                yield prepared, answer.to_dict()

        try:
            next(answers)
        except StopIteration:
            return
        raise ValueError("run contains more terminal slots than the prepared workspace")

    @staticmethod
    def validate_answer_pairs(run_dir: Path, manifest: RunManifest) -> None:
        """Validate every answer slot without retaining artifact rows in memory."""
        for _pair in EvaluationWorkspace.iter_answer_pairs(run_dir, manifest):
            pass

    @classmethod
    def load_retry_parent(cls, run_dir: Path) -> Tuple[
        RunManifest,
        List[PreparationRecord],
        Dict[str, Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        manifest = cls.load_manifest(run_dir)
        if manifest.status is not RunStatus.SCORED:
            raise ValueError("evaluation retry requires a complete scored run")
        for phase in ("prepare", "run", "score"):
            manifest.require_phase_complete(phase)
        if manifest.versions != {
            "scoring": SCORING_VERSION,
            "prompts": PROMPT_VERSIONS,
        }:
            raise ValueError("evaluation retry requires current QA artifact versions")
        required_artifacts = {
            manifest.snapshot,
            "preparation.jsonl",
            "evaluator_profile.resolved.yaml",
            "agent_config.resolved.yaml",
            "agent_spec.resolved.md",
            "answers.jsonl",
            "evaluation_results.jsonl",
            "summary.json",
            "report.md",
        }
        verify_hashes(run_dir, manifest.artifacts, required_artifacts)
        preparation = cls.load_preparation(run_dir, manifest)
        prepared_records = [
            record for record in preparation if record.status == "prepared"
        ]
        raw_answers = read_jsonl(run_dir / "answers.jsonl")
        answers = [
            AnswerAttempt.from_dict(row, context=f"parent answer row {index}")
            for index, row in enumerate(raw_answers, 1)
        ]
        results = [
            EvaluationResult.from_dict(row, context=f"parent result row {index}")
            for index, row in enumerate(
                read_jsonl(run_dir / "evaluation_results.jsonl"), 1
            )
        ]
        expected_identities = {
            (
                prepared.item_id,
                ordinal,
                f"{prepared.item_id}-attempt-{ordinal}",
            )
            for prepared in prepared_records
            for ordinal in range(1, manifest.required_attempts + 1)
        }
        answer_identities = {
            (
                answer.identity.item_id,
                answer.identity.ordinal,
                answer.identity.attempt_id,
            )
            for answer in answers
        }
        result_identities = {
            (
                result.identity.item_id,
                result.identity.ordinal,
                result.identity.attempt_id,
            )
            for result in results
        }
        if (
            len(answers) != len(expected_identities)
            or len(results) != len(expected_identities)
            or answer_identities != expected_identities
            or result_identities != expected_identities
        ):
            raise ValueError(
                "parent run attempt identities do not match the prepared workspace"
            )
        answers_by_id = {
            answer.identity.attempt_id: answer.to_dict() for answer in answers
        }
        if len(answers_by_id) != len(answers):
            raise ValueError("parent run contains duplicate attempt answers")

        config_hash = manifest.artifact_digest("agent_config.resolved.yaml")
        spec_hash = manifest.artifact_digest("agent_spec.resolved.md")
        result_rows = []
        for result in results:
            answer = answers_by_id[result.identity.attempt_id]
            expected_answer_status = (
                AnswerStatus.EXECUTION_FAILED.value
                if result.status is EvaluationStatus.EXECUTION_FAILED
                else AnswerStatus.ANSWER_READY.value
            )
            if answer["status"] != expected_answer_status:
                raise ValueError(
                    "parent run answer and evaluation result statuses disagree"
                )
            for identity in (
                AnswerAttempt.from_dict(
                    answer, context=f"parent answer {result.identity.attempt_id}"
                ).identity,
                result.identity,
            ):
                if (
                    identity.agent_config_sha256 != config_hash
                    or identity.agent_spec_sha256 != spec_hash
                ):
                    raise ValueError("parent run attempt provenance is inconsistent")
            result_rows.append(result.to_dict())
        return manifest, preparation, answers_by_id, result_rows
