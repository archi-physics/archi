# isort: skip_file
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .artifacts import iter_jsonl, read_json, verify_hashes
from .constants import PROMPT_VERSIONS, SCORING_VERSION
from .live_checks import LiveCheck, LiveCheckPhase, LiveValidation, live_check_from_dict
from .preparation import (  # isort: skip
    PreparationRecord,
    iter_preparation_records,
    load_preparation_records,
    preparation_record_from_dict,
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
            return RunManifest.from_dict(
                read_json(path), supported_schema_versions=("qa-v1", "qa-v2")
            )
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
    def open_retry_parent(cls, run_dir: Path) -> "RetryParentStore":
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
        if manifest.schema_version == "qa-v2":
            required_artifacts.add("live_checks.jsonl")
        verify_hashes(run_dir, manifest.artifacts, required_artifacts)
        return RetryParentStore(run_dir, manifest)


class RetryParentStore:
    """Disk-backed validated retry join state."""

    def __init__(self, run_dir: Path, manifest: RunManifest):
        self.run_dir = run_dir
        self.manifest = manifest
        self._temporary = tempfile.TemporaryDirectory(prefix=".qa-retry-parent-")
        self.connection = sqlite3.connect(
            str(Path(self._temporary.name) / "parent.sqlite3")
        )
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "RetryParentStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def __del__(self) -> None:
        self.close()

    def _load(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE preparation (ordinal INTEGER PRIMARY KEY, item_id TEXT UNIQUE NOT NULL, status TEXT NOT NULL, row_json TEXT NOT NULL);
            CREATE TABLE expected (attempt_id TEXT PRIMARY KEY, item_id TEXT NOT NULL, attempt_ordinal INTEGER NOT NULL);
            CREATE TABLE answers (attempt_id TEXT PRIMARY KEY, status TEXT NOT NULL, row_json TEXT NOT NULL);
            CREATE TABLE results (ordinal INTEGER PRIMARY KEY, attempt_id TEXT UNIQUE NOT NULL, item_id TEXT NOT NULL, status TEXT NOT NULL, row_json TEXT NOT NULL);
            """
        )
        for index, record in enumerate(
            iter_preparation_records(
                self.run_dir / "preparation.jsonl",
                expected_count=self.manifest.preparation_input_items,
            ),
            1,
        ):
            self.connection.execute(
                "INSERT INTO preparation VALUES (?, ?, ?, ?)",
                (index, record.item_id, record.status, json.dumps(record.to_dict())),
            )
            if record.status == "prepared":
                for ordinal in range(1, self.manifest.required_attempts + 1):
                    self.connection.execute(
                        "INSERT INTO expected VALUES (?, ?, ?)",
                        (
                            f"{record.item_id}-attempt-{ordinal}",
                            record.item_id,
                            ordinal,
                        ),
                    )
        config_hash = self.manifest.artifact_digest("agent_config.resolved.yaml")
        spec_hash = self.manifest.artifact_digest("agent_spec.resolved.md")
        for index, row in enumerate(iter_jsonl(self.run_dir / "answers.jsonl"), 1):
            answer = AnswerAttempt.from_dict(row, context=f"parent answer row {index}")
            identity = answer.identity
            if (
                identity.agent_config_sha256 != config_hash
                or identity.agent_spec_sha256 != spec_hash
            ):
                raise ValueError("parent run attempt provenance is inconsistent")
            try:
                self.connection.execute(
                    "INSERT INTO answers VALUES (?, ?, ?)",
                    (
                        identity.attempt_id,
                        answer.status.value,
                        json.dumps(answer.to_dict()),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "parent run contains duplicate attempt answers"
                ) from exc
        for index, row in enumerate(
            iter_jsonl(self.run_dir / "evaluation_results.jsonl"), 1
        ):
            result = EvaluationResult.from_dict(
                row, context=f"parent result row {index}"
            )
            identity = result.identity
            if (
                identity.agent_config_sha256 != config_hash
                or identity.agent_spec_sha256 != spec_hash
            ):
                raise ValueError("parent run attempt provenance is inconsistent")
            expected = self.connection.execute(
                "SELECT item_id, attempt_ordinal FROM expected WHERE attempt_id = ?",
                (identity.attempt_id,),
            ).fetchone()
            if expected != (identity.item_id, identity.ordinal):
                raise ValueError(
                    "parent run attempt identities do not match the prepared workspace"
                )
            answer = self.connection.execute(
                "SELECT status FROM answers WHERE attempt_id = ?",
                (identity.attempt_id,),
            ).fetchone()
            if result.status is EvaluationStatus.LIVE_VALIDATION_FAILED:
                validation = result.payload.get("live_validation")
                phase = (
                    validation.get("phase") if isinstance(validation, dict) else None
                )
                expected_answer = (
                    None if phase == "pre_run" else (AnswerStatus.ANSWER_READY.value,)
                )
                if phase not in {"pre_run", "post_run"} or answer != expected_answer:
                    raise ValueError(
                        "parent run answer and live-validation phase disagree"
                    )
            else:
                expected_status = (
                    AnswerStatus.EXECUTION_FAILED.value
                    if result.status is EvaluationStatus.EXECUTION_FAILED
                    else AnswerStatus.ANSWER_READY.value
                )
                if answer != (expected_status,):
                    raise ValueError(
                        "parent run answer and evaluation result statuses disagree"
                    )
            try:
                self.connection.execute(
                    "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
                    (
                        index,
                        identity.attempt_id,
                        identity.item_id,
                        result.status.value,
                        json.dumps(result.to_dict()),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "parent run attempt identities do not match the prepared workspace"
                ) from exc
        missing_result = self.connection.execute(
            "SELECT 1 FROM expected e LEFT JOIN results r USING(attempt_id) WHERE r.attempt_id IS NULL LIMIT 1"
        ).fetchone()
        extra_answer = self.connection.execute(
            "SELECT 1 FROM answers a LEFT JOIN results r USING(attempt_id) WHERE r.attempt_id IS NULL LIMIT 1"
        ).fetchone()
        if missing_result is not None or extra_answer is not None:
            raise ValueError(
                "parent run attempt identities do not match the prepared workspace"
            )
        self.connection.commit()

    def iter_preparation(self) -> Iterator[PreparationRecord]:
        for (row_json,) in self.connection.execute(
            "SELECT row_json FROM preparation ORDER BY ordinal"
        ):
            raw = json.loads(row_json)
            yield preparation_record_from_dict(raw)

    def iter_results(self) -> Iterator[Dict[str, Any]]:
        for (row_json,) in self.connection.execute(
            "SELECT row_json FROM results ORDER BY ordinal"
        ):
            yield json.loads(row_json)

    def get_prepared(self, item_id: str) -> PreparationRecord:
        row = self.connection.execute(
            "SELECT row_json FROM preparation WHERE item_id = ? AND status = 'prepared'",
            (item_id,),
        ).fetchone()
        if row is None:
            raise ValueError("retry references an unknown prepared item")
        return preparation_record_from_dict(json.loads(row[0]))

    def get_answer(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT row_json FROM answers WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def initialize_successor_state(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE retry_attempts (attempt_id TEXT PRIMARY KEY, item_id TEXT NOT NULL, kind TEXT NOT NULL, origin TEXT NOT NULL);
            CREATE TABLE fresh_checks (item_id TEXT NOT NULL, phase TEXT NOT NULL, row_json TEXT NOT NULL, PRIMARY KEY (item_id, phase));
            CREATE TABLE parent_checks (item_id TEXT NOT NULL, phase TEXT NOT NULL, row_json TEXT NOT NULL, PRIMARY KEY (item_id, phase));
            CREATE TABLE live_validations (item_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE retried_answers (attempt_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE successor_answers (attempt_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE retried_results (attempt_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            """
        )

    def index_retry_attempt(self, result: Dict[str, Any]) -> None:
        kind = result["status"].removesuffix("_failed")
        self.connection.execute(
            "INSERT INTO retry_attempts VALUES (?, ?, ?, ?)",
            (result["attempt_id"], result["item_id"], kind, kind),
        )

    def retry_kind(self, attempt_id: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT kind FROM retry_attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        return row[0] if row is not None else None

    def item_has_live_retry(self, item_id: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM retry_attempts WHERE item_id = ? AND origin = 'live_validation' LIMIT 1",
                (item_id,),
            ).fetchone()
            is not None
        )

    def promote_live_retry_to_execution(self, item_id: str) -> None:
        self.connection.execute(
            "UPDATE retry_attempts SET kind = 'execution' WHERE item_id = ? AND kind = 'live_validation'",
            (item_id,),
        )

    def execution_retry_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM retry_attempts WHERE kind = 'execution'"
        ).fetchone()
        return int(row[0])

    def _store_attempt_row(
        self, table: str, attempt_id: str, row: Dict[str, Any]
    ) -> None:
        self.connection.execute(
            f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
            (attempt_id, json.dumps(row, ensure_ascii=False)),
        )

    def _load_attempt_row(
        self, table: str, attempt_id: str
    ) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            f"SELECT row_json FROM {table} WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def store_retried_answer(self, row: Dict[str, Any]) -> None:
        self._store_attempt_row("retried_answers", row["attempt_id"], row)

    def retried_answer(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        return self._load_attempt_row("retried_answers", attempt_id)

    def store_successor_answer(self, row: Dict[str, Any]) -> None:
        self._store_attempt_row("successor_answers", row["attempt_id"], row)

    def successor_answer(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        return self._load_attempt_row("successor_answers", attempt_id)

    def store_retried_result(self, row: Dict[str, Any]) -> None:
        self._store_attempt_row("retried_results", row["attempt_id"], row)

    def retried_result(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        return self._load_attempt_row("retried_results", attempt_id)

    def store_live_check(self, *, fresh: bool, check: LiveCheck) -> None:
        table = "fresh_checks" if fresh else "parent_checks"
        self.connection.execute(
            f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?)",
            (
                check.item_id,
                check.phase.value,
                json.dumps(check.to_dict(), ensure_ascii=False),
            ),
        )

    def live_check(self, item_id: str, phase: LiveCheckPhase) -> Optional[LiveCheck]:
        row = self.connection.execute(
            "SELECT row_json FROM fresh_checks WHERE item_id = ? AND phase = ?",
            (item_id, phase.value),
        ).fetchone()
        if row is None:
            row = self.connection.execute(
                "SELECT row_json FROM parent_checks WHERE item_id = ? AND phase = ?",
                (item_id, phase.value),
            ).fetchone()
        return (
            live_check_from_dict(json.loads(row[0]), "retry live check")
            if row is not None
            else None
        )

    def store_live_validation(self, item_id: str, validation: LiveValidation) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO live_validations VALUES (?, ?)",
            (item_id, json.dumps(validation.to_dict(), ensure_ascii=False)),
        )

    def live_validation(self, item_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT row_json FROM live_validations WHERE item_id = ?", (item_id,)
        ).fetchone()
        return json.loads(row[0]) if row is not None else None

    @property
    def preparation_rows(self) -> "PreparationRows":
        return PreparationRows(self)

    @property
    def result_rows(self) -> "ResultRows":
        return ResultRows(self)

    @property
    def answer_rows(self) -> "AnswerRows":
        return AnswerRows(self)


class PreparationRows:
    def __init__(self, store: RetryParentStore):
        self.store = store

    def __iter__(self) -> Iterator[PreparationRecord]:
        return self.store.iter_preparation()


class ResultRows:
    def __init__(self, store: RetryParentStore):
        self.store = store

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.store.iter_results()

    def __len__(self) -> int:
        row = self.store.connection.execute("SELECT COUNT(*) FROM results").fetchone()
        return int(row[0])


class AnswerRows(Mapping):
    def __init__(self, store: RetryParentStore):
        self.store = store

    def __getitem__(self, attempt_id: str) -> Dict[str, Any]:
        answer = self.store.get_answer(attempt_id)
        if answer is None:
            raise KeyError(attempt_id)
        return answer

    def __iter__(self) -> Iterator[str]:
        for (attempt_id,) in self.store.connection.execute(
            "SELECT attempt_id FROM answers"
        ):
            yield attempt_id

    def __len__(self) -> int:
        row = self.store.connection.execute("SELECT COUNT(*) FROM answers").fetchone()
        return int(row[0])
