# isort: skip_file
from __future__ import annotations

import sqlite3
import uuid
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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
    EvaluationRuntimePhase,
    OWNED_FILES,
    PROMPT_VERSIONS,
    RUN_FILES,
    SCHEMA_VERSION,
    SCORE_FILES,
    SCORING_VERSION,
)
from .live_checks import (
    LiveCheckPhase,
    LiveValidation,
    iter_live_checks,
    iter_precheck_decisions,
    observe_live_item,
    validation_against_baseline,
)
from .oracle import OracleResolver
from .oracle_config import EvaluatorMCPRegistry
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
from .schema import AnswerAttempt
from .scoring import build_summary, write_report, write_summary
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
        Iterable[PreparationRecord],
        Iterable[Dict[str, Any]],
        Any,
    ]:
        store = EvaluationWorkspace.open_retry_parent(run_dir)
        return (
            store.manifest.to_dict(),
            store.preparation_rows,
            store.result_rows,
            store,
        )

    @staticmethod
    def _retry_plan(
        manifest: Dict[str, Any], results: Iterable[Dict[str, Any]]
    ) -> Dict[str, Any]:
        counts = {
            "retry_attempt_count": 0,
            "execution_attempt_count": 0,
            "evaluation_attempt_count": 0,
            "live_validation_attempt_count": 0,
            "carried_forward_attempt_count": 0,
        }
        for row in results:
            status = row["status"]
            if status == "scored":
                counts["carried_forward_attempt_count"] += 1
            elif status in {
                "execution_failed",
                "evaluation_failed",
                "live_validation_failed",
            }:
                counts["retry_attempt_count"] += 1
                counts[f"{status.removesuffix('_failed')}_attempt_count"] += 1
        if not counts["retry_attempt_count"]:
            raise ValueError("evaluation run has no failed attempts to retry")
        return {"parent_run_id": manifest["run_id"], **counts}

    def retry_plan(self, run_dir: Path) -> Dict[str, Any]:
        manifest, _preparation, results, store = self._load_retry_parent(run_dir)
        try:
            return self._retry_plan(manifest, results)
        finally:
            store.close()

    def prepare(
        self,
        dataset: Path,
        output_dir: Path,
        evaluator_profile_path: Optional[Path] = None,
        overwrite: bool = False,
        mcp_config_path: Optional[Path] = None,
        skip_live: bool = False,
        trusted_dataset: bool = False,
    ) -> Dict[str, Any]:
        dataset_format = dataset_source_format(dataset)
        profile = load_profile(evaluator_profile_path)
        registry = EvaluatorMCPRegistry.load(mcp_config_path)
        resolver = OracleResolver(registry)
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
            validation_items = (
                iter_dataset_items(staged_snapshot, allow_materialized_live=True)
                if trusted_dataset
                else iter_dataset_items(staged_snapshot)
            )
            snapshot_item_count = sum(1 for _item in validation_items)
            evaluator = LangChainEvaluatorRuntime(profile)
            output_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                self._remove_owned(output_dir, OWNED_FILES)
            staged_snapshot.replace(snapshot_path)
        write_yaml(output_dir / "evaluator_profile.resolved.yaml", profile.to_dict())

        prepared_item_count = 0
        contains_live_answers = False
        dataset_schema_version = None
        with AtomicJsonlWriter(output_dir / "preparation.jsonl") as preparation_writer:
            preparation_items = (
                iter_dataset_items(snapshot_path, allow_materialized_live=True)
                if trusted_dataset
                else iter_dataset_items(snapshot_path)
            )
            for item in preparation_items:
                dataset_schema_version = item.schema_version.value
                record = (
                    prepare_dataset_item(item, evaluator)
                    if not item.is_live and not skip_live
                    else prepare_dataset_item(
                        item,
                        evaluator,
                        resolver,
                        skip_live=skip_live,
                    )
                )
                preparation_writer.write(record.to_dict())
                prepared_item_count += record.status == "prepared"
                contains_live_answers = contains_live_answers or (
                    record.status == "prepared" and bool(record.time_sensitive)
                )
        prep_names = {
            snapshot_name,
            "preparation.jsonl",
            "evaluator_profile.resolved.yaml",
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": str(uuid.uuid4()),
            "status": "prepared",
            "contains_live_answers": contains_live_answers,
            "versions": {
                "scoring": SCORING_VERSION,
                "prompts": PROMPT_VERSIONS,
            },
            "input": {
                "source_path": str(dataset.resolve()),
                "snapshot": snapshot_name,
                "dataset_schema_version": dataset_schema_version,
                "trusted_catalog_source": trusted_dataset,
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
                    "skip_live": skip_live,
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
        mcp_config_path: Optional[Path] = None,
        pause_on_live_mismatch: bool = False,
        authorize_staged_invalid: bool = False,
        _resolved_agent_inputs: Optional[Tuple[Dict[str, Any], Any, str, type]] = None,
    ) -> Dict[str, Any]:
        self._require_positive_attempts(attempts)
        self.require_worker_count(run_workers, "run_workers")
        manifest = self._load_manifest(run_dir)
        self._phase_complete(manifest, "prepare")
        snapshot = manifest["input"]["snapshot"]
        required_inputs = {
            snapshot,
            "preparation.jsonl",
            "evaluator_profile.resolved.yaml",
        }
        if authorize_staged_invalid:
            required_inputs.add("live_checks.jsonl")
        # A resumed run re-reads the agent inputs its paused run froze, then
        # rewrites them and records fresh digests. Verify them with the rest of
        # the inputs first: the workspace sits on a host mount, so without this a
        # rewrite made while the run waited is re-sealed as if the run had always
        # used it. A first run has no recorded digest, so nothing is added here.
        required_inputs.update(
            name
            for name in ("agent_config.resolved.yaml", "agent_spec.resolved.md")
            if name in manifest["artifacts"]
        )
        verify_hashes(
            run_dir,
            manifest["artifacts"],
            required_inputs,
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
        prepared_item_count = 0
        has_prepared_live = False
        for record in iter_preparation_records(
            preparation_path,
            expected_count=preparation_count,
        ):
            if record.status == "prepared":
                prepared_item_count += 1
                has_prepared_live = has_prepared_live or bool(record.time_sensitive)
        if prepared_item_count == 0:
            raise ValueError("run requires at least one prepared item")
        registry = EvaluatorMCPRegistry.load(mcp_config_path)
        resolver = OracleResolver(registry)
        config, spec, spec_text, pipeline_class = (
            _resolved_agent_inputs
            if _resolved_agent_inputs is not None
            else load_agent_inputs(agent_config, agent_spec)
        )
        if overwrite:
            owned = RUN_FILES | SCORE_FILES
            if authorize_staged_invalid:
                owned = owned - {"live_checks.jsonl"}
            self._remove_owned(run_dir, owned)
            manifest["phases"].pop("run", None)
            manifest["phases"].pop("score", None)
            manifest.pop("attempts", None)
            manifest.pop("agent", None)
            manifest.pop("attention_required", None)
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
        manifest["runtime_phase"] = (
            EvaluationRuntimePhase.CHECKING_LIVE_ANSWERS.value
            if has_prepared_live
            else EvaluationRuntimePhase.RUNNING_ATTEMPTS.value
        )
        write_json(run_dir / "manifest.json", manifest)
        actual_answers = 0
        live_item_count = 0
        invalid_pre_check_count = 0
        newly_invalid_count = 0
        precheck_reason_counts = {
            "answer_changed": 0,
            "oracle_failed": 0,
        }

        with TemporaryDirectory(
            prefix=f".{run_dir.name}-live-checks-", dir=str(run_dir.parent)
        ) as live_staging_dir:
            authorized_connection = None
            staged_checks_path = run_dir / "live_checks.jsonl"
            if authorize_staged_invalid:
                if not staged_checks_path.is_file():
                    raise ValueError("continued run is missing staged live checks")
                authorized_connection = sqlite3.connect(
                    str(Path(live_staging_dir) / "authorized.sqlite3")
                )
                authorized_connection.execute(
                    "CREATE TABLE authorized (item_id TEXT PRIMARY KEY)"
                )
                for prepared, _check, validation in iter_precheck_decisions(
                    preparation_path,
                    staged_checks_path,
                    expected_preparation_count=preparation_count,
                ):
                    if validation is not None:
                        authorized_connection.execute(
                            "INSERT INTO authorized VALUES (?)", (prepared.item_id,)
                        )
                authorized_connection.commit()
                staged_checks_path.unlink()
            pre_checks_path = Path(live_staging_dir) / "pre-run.jsonl"
            try:
                with AtomicJsonlWriter(pre_checks_path) as check_writer:
                    if has_prepared_live:
                        for prepared in iter_preparation_records(
                            preparation_path,
                            expected_count=preparation_count,
                        ):
                            if (
                                prepared.status == "prepared"
                                and prepared.time_sensitive
                            ):
                                live_item_count += 1
                                check = observe_live_item(
                                    prepared,
                                    resolver,
                                    LiveCheckPhase.PRE_RUN,
                                )
                                check_writer.write(check.to_dict())
                                validation = validation_against_baseline(
                                    prepared, check
                                )
                                if validation is not None:
                                    invalid_pre_check_count += 1
                                    precheck_reason_counts[validation.reason.value] += 1
                                    authorized = (
                                        authorized_connection is not None
                                        and authorized_connection.execute(
                                            "SELECT 1 FROM authorized WHERE item_id = ?",
                                            (prepared.item_id,),
                                        ).fetchone()
                                        is not None
                                    )
                                    if not authorized:
                                        newly_invalid_count += 1
            finally:
                if authorized_connection is not None:
                    authorized_connection.close()
            precheck_completed_at = utc_now()

            precheck_status = (
                "last_check_unavailable"
                if precheck_reason_counts["oracle_failed"]
                else (
                    "change_observed" if invalid_pre_check_count else "matched_baseline"
                )
            )
            if pause_on_live_mismatch and newly_invalid_count:
                with AtomicJsonlWriter(run_dir / "live_checks.jsonl") as check_writer:
                    for check in iter_live_checks(pre_checks_path):
                        check_writer.write(check.to_dict())
                manifest["attempts"] = attempts
                manifest["agent"] = {
                    "agent_class": chat["agent_class"],
                    "provider": chat["default_provider"],
                    "model": chat["default_model"],
                    "config_artifact": "agent_config.resolved.yaml",
                    "spec_artifact": "agent_spec.resolved.md",
                }
                manifest["artifacts"].update(
                    artifact_hashes(
                        run_dir,
                        {
                            "agent_config.resolved.yaml",
                            "agent_spec.resolved.md",
                            "live_checks.jsonl",
                        },
                    )
                )
                manifest["phases"]["run"] = {
                    "status": "attention_required",
                    "started_at": started_at,
                    "checked_at": precheck_completed_at,
                    "live_check_status": precheck_status,
                    "workers": run_workers,
                }
                manifest["attention_required"] = {
                    "live_items": live_item_count,
                    "affected_item_count": invalid_pre_check_count,
                    "reason_counts": precheck_reason_counts,
                    "no_agent_attempts_started": True,
                    "checked_at": precheck_completed_at,
                    "can_continue": manifest["phases"]["prepare"]["prepared_items"]
                    > invalid_pre_check_count,
                }
                manifest["status"] = "attention_required"
                manifest.pop("runtime_phase", None)
                write_json(run_dir / "manifest.json", manifest)
                return manifest

            manifest["runtime_phase"] = EvaluationRuntimePhase.RUNNING_ATTEMPTS.value
            write_json(run_dir / "manifest.json", manifest)

            def admitted_records() -> Iterator[PreparationRecord]:
                pre_checks = iter_live_checks(
                    pre_checks_path,
                    phase=LiveCheckPhase.PRE_RUN,
                )
                for prepared in iter_preparation_records(
                    preparation_path,
                    expected_count=preparation_count,
                ):
                    if prepared.status != "prepared":
                        continue
                    if prepared.time_sensitive:
                        try:
                            check = next(pre_checks)
                        except StopIteration:
                            raise ValueError(
                                "pre-run live checks are missing an item"
                            ) from None
                        if check.item_id != prepared.item_id:
                            raise ValueError(
                                "pre-run live check order does not match preparation"
                            )
                        if validation_against_baseline(prepared, check) is not None:
                            continue
                    yield prepared
                try:
                    next(pre_checks)
                except StopIteration:
                    return
                raise ValueError("pre-run live checks contain an extra item")

            def tasks() -> Iterator[Tuple[PreparationRecord, Dict[str, Any]]]:
                for prepared in admitted_records():
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
                    actual_answers += 1
                    answer_writer.write(answer_row)

            with AtomicJsonlWriter(run_dir / "live_checks.jsonl") as check_writer:
                for check in iter_live_checks(pre_checks_path):
                    check_writer.write(check.to_dict())
                if has_prepared_live:
                    for prepared in admitted_records():
                        if prepared.time_sensitive:
                            check_writer.write(
                                observe_live_item(
                                    prepared,
                                    resolver,
                                    LiveCheckPhase.POST_RUN,
                                ).to_dict()
                            )
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
            "checked_at": precheck_completed_at,
            "live_check_status": precheck_status,
            "completed_at": utc_now(),
            "attempt_slots": prepared_item_count * attempts,
            "actual_agent_executions": actual_answers,
            "workers": run_workers,
        }
        manifest["status"] = "run_completed"
        manifest.pop("attention_required", None)
        manifest.pop("runtime_phase", None)
        write_json(run_dir / "manifest.json", manifest)
        return manifest

    @staticmethod
    def _iter_live_decisions(
        run_dir: Path,
        manifest: Dict[str, Any],
    ) -> Iterator[Tuple[PreparationRecord, Optional[LiveValidation]]]:
        preparation_count = manifest["phases"]["prepare"]["input_items"]
        pre_checks = iter_live_checks(
            run_dir / "live_checks.jsonl", phase=LiveCheckPhase.PRE_RUN
        )
        post_checks = iter_live_checks(
            run_dir / "live_checks.jsonl", phase=LiveCheckPhase.POST_RUN
        )
        for prepared in iter_preparation_records(
            run_dir / "preparation.jsonl", expected_count=preparation_count
        ):
            if prepared.status != "prepared":
                continue
            validation = None
            if prepared.time_sensitive:
                try:
                    pre_check = next(pre_checks)
                except StopIteration:
                    raise ValueError("live checks are missing a pre-run item") from None
                if pre_check.item_id != prepared.item_id:
                    raise ValueError(
                        "pre-run live check order does not match preparation"
                    )
                validation = validation_against_baseline(prepared, pre_check)
                if validation is None:
                    try:
                        post_check = next(post_checks)
                    except StopIteration:
                        raise ValueError(
                            "live checks are missing a post-run item"
                        ) from None
                    if post_check.item_id != prepared.item_id:
                        raise ValueError(
                            "post-run live check order does not match preparation"
                        )
                    validation = validation_against_baseline(prepared, post_check)
            yield prepared, validation
        for checks, label in ((pre_checks, "pre-run"), (post_checks, "post-run")):
            try:
                next(checks)
            except StopIteration:
                continue
            raise ValueError(f"live checks contain an extra {label} item")

    @classmethod
    def _iter_scoring_pairs(
        cls, run_dir: Path, manifest: Dict[str, Any]
    ) -> Iterator[Tuple[PreparationRecord, Dict[str, Any]]]:
        answers = iter_jsonl(run_dir / "answers.jsonl")
        attempts = manifest["attempts"]
        for prepared, validation in cls._iter_live_decisions(run_dir, manifest):
            pre_run_failure = (
                validation is not None and validation.phase is LiveCheckPhase.PRE_RUN
            )
            if pre_run_failure:
                continue
            for ordinal in range(1, attempts + 1):
                try:
                    raw_answer = next(answers)
                except StopIteration:
                    raise ValueError(
                        "run contains fewer terminal slots than admitted membership"
                    ) from None
                answer = AnswerAttempt.from_dict(
                    raw_answer,
                    context=f"run answer {raw_answer.get('attempt_id', ordinal)}",
                )
                expected = (
                    prepared.item_id,
                    ordinal,
                    f"{prepared.item_id}-attempt-{ordinal}",
                )
                actual = (
                    answer.identity.item_id,
                    answer.identity.ordinal,
                    answer.identity.attempt_id,
                )
                if expected != actual:
                    raise ValueError(
                        "run attempt slot identities do not match admitted membership"
                    )
                if validation is None:
                    yield prepared, answer.to_dict()
        try:
            next(answers)
        except StopIteration:
            return
        raise ValueError("run contains more terminal slots than admitted membership")

    @classmethod
    def _iter_terminal_plan(
        cls, run_dir: Path, manifest: Dict[str, Any]
    ) -> Iterator[Optional[Dict[str, Any]]]:
        answers = iter_jsonl(run_dir / "answers.jsonl")
        attempts = manifest["attempts"]
        config_hash = manifest["artifacts"]["agent_config.resolved.yaml"]
        spec_hash = manifest["artifacts"]["agent_spec.resolved.md"]
        for prepared, validation in cls._iter_live_decisions(run_dir, manifest):
            pre_run_failure = (
                validation is not None and validation.phase is LiveCheckPhase.PRE_RUN
            )
            for ordinal in range(1, attempts + 1):
                if pre_run_failure:
                    identity = {
                        "item_id": prepared.item_id,
                        "attempt_id": f"{prepared.item_id}-attempt-{ordinal}",
                        "ordinal": ordinal,
                        "agent_config_sha256": config_hash,
                        "agent_spec_sha256": spec_hash,
                    }
                else:
                    try:
                        raw_answer = next(answers)
                    except StopIteration:
                        raise ValueError(
                            "run contains fewer terminal slots than admitted membership"
                        ) from None
                    identity = AnswerAttempt.from_dict(
                        raw_answer,
                        context=f"run answer {raw_answer.get('attempt_id', ordinal)}",
                    ).identity.to_dict()
                if validation is None:
                    yield None
                else:
                    yield {
                        **identity,
                        "status": "live_validation_failed",
                        "live_validation": validation.to_dict(),
                    }
        try:
            next(answers)
        except StopIteration:
            return
        raise ValueError("run contains more terminal slots than admitted membership")

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
                "live_checks.jsonl",
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
        has_prepared_live = any(
            record.status == "prepared" and bool(record.time_sensitive)
            for record in iter_preparation_records(
                run_dir / "preparation.jsonl",
                expected_count=manifest["phases"]["prepare"]["input_items"],
            )
        )
        if has_prepared_live:
            for _plan_entry in self._iter_terminal_plan(run_dir, manifest):
                pass
        else:
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
        manifest["runtime_phase"] = EvaluationRuntimePhase.SCORING.value
        write_json(run_dir / "manifest.json", manifest)

        scoring_pairs = (
            self._iter_scoring_pairs(run_dir, manifest)
            if has_prepared_live
            else self._iter_answer_pairs(run_dir, manifest)
        )
        scored_results = iter(
            score_attempts(
                scoring_pairs,
                lambda: LangChainEvaluatorRuntime(profile),
                score_workers,
                thread_name_prefix="archi-qa-score",
            )
        )
        with AtomicJsonlWriter(run_dir / "evaluation_results.jsonl") as result_writer:
            if not has_prepared_live:
                for result in scored_results:
                    result_writer.write(result)
            else:
                for terminal_failure in self._iter_terminal_plan(run_dir, manifest):
                    if terminal_failure is not None:
                        result_writer.write(terminal_failure)
                    else:
                        try:
                            result_writer.write(next(scored_results))
                        except StopIteration:
                            raise ValueError(
                                "score phase produced fewer results than eligible slots"
                            ) from None
                try:
                    next(scored_results)
                except StopIteration:
                    pass
                else:
                    raise ValueError("score phase produced extra eligible results")

        with TemporaryDirectory(prefix=".qa-summary-items-") as temporary:
            item_rows_path = Path(temporary) / "items.jsonl"
            with AtomicJsonlWriter(item_rows_path) as item_writer:
                summary = build_summary(
                    iter_preparation_records(
                        run_dir / "preparation.jsonl",
                        expected_count=manifest["phases"]["prepare"]["input_items"],
                    ),
                    iter_jsonl(run_dir / "evaluation_results.jsonl"),
                    iter_jsonl(run_dir / "live_checks.jsonl"),
                    item_sink=item_writer.write,
                )
            summary["provenance"] = {
                "agent_config_sha256": manifest["artifacts"][
                    "agent_config.resolved.yaml"
                ],
                "agent_spec_sha256": manifest["artifacts"]["agent_spec.resolved.md"],
                "evaluator_profile_sha256": manifest["artifacts"][
                    "evaluator_profile.resolved.yaml"
                ],
            }
            write_summary(run_dir / "summary.json", summary, iter_jsonl(item_rows_path))
            manifest["status"] = "scored"
            manifest.pop("runtime_phase", None)
            write_report(
                run_dir / "report.md",
                summary,
                manifest,
                iter_jsonl(item_rows_path),
            )
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
        manifest["artifacts"]["report.md"] = sha256_file(run_dir / "report.md")
        write_json(run_dir / "manifest.json", manifest)
        return manifest

    def retry(
        self,
        parent_run_dir: Path,
        output_dir: Path,
        mcp_config_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        opened_stores = []
        try:
            return self._retry_with_open_parent(
                parent_run_dir,
                output_dir,
                mcp_config_path,
                opened_stores,
            )
        finally:
            for store in opened_stores:
                store.close()

    def _retry_with_open_parent(
        self,
        parent_run_dir: Path,
        output_dir: Path,
        mcp_config_path: Optional[Path],
        opened_stores: List[Any],
    ) -> Dict[str, Any]:
        (
            parent_manifest,
            preparation,
            parent_results,
            parent_store,
        ) = self._load_retry_parent(parent_run_dir)
        opened_stores.append(parent_store)
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
        parent_store.initialize_successor_state()
        for result in parent_results:
            if result["status"] in {
                "execution_failed",
                "evaluation_failed",
                "live_validation_failed",
            }:
                parent_store.index_retry_attempt(result)

        parent_live_checks_path = parent_run_dir / "live_checks.jsonl"
        if parent_live_checks_path.is_file():
            for check in iter_live_checks(parent_live_checks_path):
                parent_store.store_live_check(fresh=False, check=check)
        if plan["execution_attempt_count"] or plan["live_validation_attempt_count"]:
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

        if plan["live_validation_attempt_count"]:
            resolver = OracleResolver(EvaluatorMCPRegistry.load(mcp_config_path))
            for prepared in preparation:
                if not parent_store.item_has_live_retry(prepared.item_id):
                    continue
                check = observe_live_item(prepared, resolver, LiveCheckPhase.PRE_RUN)
                parent_store.store_live_check(fresh=True, check=check)
                validation = validation_against_baseline(prepared, check)
                if validation is not None:
                    parent_store.store_live_validation(prepared.item_id, validation)
                else:
                    parent_store.promote_live_retry_to_execution(prepared.item_id)

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
            "contains_live_answers": bool(parent_manifest.get("contains_live_answers")),
        }
        write_json(output_dir / "manifest.json", manifest)

        started_at = utc_now()
        execution_tasks = (
            (
                parent_store.get_prepared(result["item_id"]),
                (
                    {
                        key: parent_store.get_answer(result["attempt_id"])[key]
                        for key in (
                            "item_id",
                            "attempt_id",
                            "ordinal",
                            "agent_config_sha256",
                            "agent_spec_sha256",
                        )
                    }
                    if parent_store.get_answer(result["attempt_id"]) is not None
                    else {
                        key: result[key]
                        for key in (
                            "item_id",
                            "attempt_id",
                            "ordinal",
                            "agent_config_sha256",
                            "agent_spec_sha256",
                        )
                    }
                ),
            )
            for result in parent_results
            if parent_store.retry_kind(result["attempt_id"]) == "execution"
        )
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
        ):
            parent_store.store_retried_answer(answer)

        if plan["live_validation_attempt_count"]:
            for prepared in preparation:
                if (
                    not parent_store.item_has_live_retry(prepared.item_id)
                    or parent_store.live_validation(prepared.item_id) is not None
                ):
                    continue
                check = observe_live_item(prepared, resolver, LiveCheckPhase.POST_RUN)
                validation = validation_against_baseline(prepared, check)
                parent_store.store_live_check(fresh=True, check=check)
                if validation is not None:
                    parent_store.store_live_validation(prepared.item_id, validation)

        with AtomicJsonlWriter(output_dir / "live_checks.jsonl") as check_writer:
            for prepared in preparation:
                if prepared.status != "prepared" or not prepared.time_sensitive:
                    continue
                pre = parent_store.live_check(prepared.item_id, LiveCheckPhase.PRE_RUN)
                if pre is None:
                    raise ValueError("parent run is missing pre-run live evidence")
                check_writer.write(pre.to_dict())
            for prepared in preparation:
                if prepared.status != "prepared" or not prepared.time_sensitive:
                    continue
                pre = parent_store.live_check(prepared.item_id, LiveCheckPhase.PRE_RUN)
                if pre is None:
                    raise ValueError("parent run is missing pre-run live evidence")
                if validation_against_baseline(prepared, pre) is None:
                    post = parent_store.live_check(
                        prepared.item_id, LiveCheckPhase.POST_RUN
                    )
                    if post is None:
                        raise ValueError("parent run is missing post-run live evidence")
                    check_writer.write(post.to_dict())

        with AtomicJsonlWriter(output_dir / "answers.jsonl") as answer_writer:
            for result in parent_results:
                attempt_id = result["attempt_id"]
                answer = parent_store.retried_answer(attempt_id)
                if (
                    parent_store.retry_kind(attempt_id)
                    in {"live_validation", "execution"}
                    and parent_store.item_has_live_retry(result["item_id"])
                    and parent_store.live_validation(result["item_id"]) is not None
                    and answer is None
                ):
                    continue
                answer = answer or parent_store.get_answer(attempt_id)
                if answer is None:
                    raise ValueError("retry could not produce a required agent answer")
                answer_writer.write(answer)
                parent_store.store_successor_answer(answer)
        manifest["artifacts"].update(artifact_hashes(output_dir, RUN_FILES))
        manifest["phases"]["run"] = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "attempt_slots": len(parent_results),
            "actual_agent_executions": parent_store.execution_retry_count(),
            "retried_attempts": parent_store.execution_retry_count(),
            "workers": run_workers,
        }
        manifest["status"] = "run_completed"
        write_json(output_dir / "manifest.json", manifest)

        profile = load_profile(output_dir / "evaluator_profile.resolved.yaml")
        started_at = utc_now()
        score_tasks = (
            (
                parent_store.get_prepared(result["item_id"]),
                parent_store.successor_answer(result["attempt_id"]),
            )
            for result in parent_results
            if parent_store.retry_kind(result["attempt_id"])
            in {"execution", "evaluation"}
            and not (
                parent_store.item_has_live_retry(result["item_id"])
                and parent_store.live_validation(result["item_id"]) is not None
            )
        )
        for result in score_attempts(
            score_tasks,
            lambda: LangChainEvaluatorRuntime(profile),
            score_workers,
            thread_name_prefix="archi-qa-retry-score",
        ):
            parent_store.store_retried_result(result)

        with AtomicJsonlWriter(
            output_dir / "evaluation_results.jsonl"
        ) as result_writer:
            for result in parent_results:
                validation = (
                    parent_store.live_validation(result["item_id"])
                    if parent_store.item_has_live_retry(result["item_id"])
                    else None
                )
                if validation is not None:
                    successor = {
                        **{
                            key: result[key]
                            for key in (
                                "item_id",
                                "attempt_id",
                                "ordinal",
                                "agent_config_sha256",
                                "agent_spec_sha256",
                            )
                        },
                        "status": "live_validation_failed",
                        "live_validation": validation,
                    }
                else:
                    successor = (
                        parent_store.retried_result(result["attempt_id"]) or result
                    )
                result_writer.write(successor)
        with TemporaryDirectory(prefix=".qa-summary-items-") as temporary:
            item_rows_path = Path(temporary) / "items.jsonl"
            with AtomicJsonlWriter(item_rows_path) as item_writer:
                summary = build_summary(
                    preparation,
                    iter_jsonl(output_dir / "evaluation_results.jsonl"),
                    iter_jsonl(output_dir / "live_checks.jsonl"),
                    item_sink=item_writer.write,
                )
            summary["provenance"] = {
                "agent_config_sha256": manifest["artifacts"][
                    "agent_config.resolved.yaml"
                ],
                "agent_spec_sha256": manifest["artifacts"]["agent_spec.resolved.md"],
                "evaluator_profile_sha256": manifest["artifacts"][
                    "evaluator_profile.resolved.yaml"
                ],
            }
            write_summary(
                output_dir / "summary.json", summary, iter_jsonl(item_rows_path)
            )
            manifest["status"] = "scored"
            write_report(
                output_dir / "report.md",
                summary,
                manifest,
                iter_jsonl(item_rows_path),
            )
        manifest["phases"]["score"] = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "scored_attempts": summary["attempt_lifecycle_counts"]["scored"],
            "retried_attempts": plan["retry_attempt_count"],
            "workers": score_workers,
        }
        manifest["artifacts"].update(
            artifact_hashes(output_dir, SCORE_FILES - {"report.md"})
        )
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
        mcp_config_path: Optional[Path] = None,
        skip_live: bool = False,
        trusted_dataset: bool = False,
        pause_on_live_mismatch: bool = False,
        authorize_staged_invalid: bool = False,
    ) -> Dict[str, Any]:
        self._require_positive_attempts(attempts)
        self.require_worker_count(run_workers, "run_workers")
        self.require_worker_count(score_workers, "score_workers")
        resolved_agent_inputs = load_agent_inputs(agent_config, agent_spec)
        self.prepare(
            dataset,
            output_dir,
            evaluator_profile_path,
            overwrite,
            mcp_config_path=mcp_config_path,
            skip_live=skip_live,
            trusted_dataset=trusted_dataset,
        )
        run_manifest = self.run(
            output_dir,
            agent_config,
            agent_spec,
            attempts,
            overwrite=False,
            run_workers=run_workers,
            mcp_config_path=mcp_config_path,
            pause_on_live_mismatch=pause_on_live_mismatch,
            authorize_staged_invalid=authorize_staged_invalid,
            _resolved_agent_inputs=resolved_agent_inputs,
        )
        if run_manifest["status"] == "attention_required":
            return run_manifest
        return self.score(
            output_dir,
            overwrite=False,
            score_workers=score_workers,
        )
