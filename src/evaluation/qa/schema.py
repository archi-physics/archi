from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from .constants import SCHEMA_VERSION
from .tool_traces import serialize_tool_call_records


class RunStatus(str, Enum):
    PREPARED = "prepared"
    ATTENTION_REQUIRED = "attention_required"
    RUN_COMPLETED = "run_completed"
    SCORED = "scored"


class AnswerStatus(str, Enum):
    ANSWER_READY = "answer_ready"
    EXECUTION_FAILED = "execution_failed"


class EvaluationStatus(str, Enum):
    SCORED = "scored"
    EXECUTION_FAILED = "execution_failed"
    EVALUATION_FAILED = "evaluation_failed"
    LIVE_VALIDATION_FAILED = "live_validation_failed"


class UnsupportedSchemaError(ValueError):
    pass


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _optional_string(value: Any, context: str) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _optional_nonempty_string(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    return _nonempty_string(value, context)


def _optional_positive_integer(value: Any, context: str) -> Optional[int]:
    if value is None:
        return None
    return _positive_integer(value, context)


@dataclass(frozen=True)
class ConsoleMetadata:
    name: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    profile_id: Optional[str] = None
    profile_name: Optional[str] = None
    agent_spec: Optional[str] = None
    attempts: Optional[int] = None
    run_workers: Optional[int] = None
    score_workers: Optional[int] = None
    created_at: Optional[str] = None
    retry_of_history_id: Optional[str] = None
    retry_number: Optional[int] = None
    retry_root_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    _present_fields: FrozenSet[str] = field(default_factory=frozenset, repr=False)

    @classmethod
    def from_dict(cls, raw: Any) -> "ConsoleMetadata":
        if not isinstance(raw, dict):
            raise ValueError("console metadata must be an object")
        string_fields = {
            "name",
            "dataset_id",
            "dataset_name",
            "profile_id",
            "profile_name",
            "agent_spec",
            "created_at",
            "retry_of_history_id",
            "retry_root_name",
        }
        integer_fields = {"attempts", "run_workers", "score_workers", "retry_number"}
        known_fields = string_fields | integer_fields
        return cls(
            name=_optional_string(raw.get("name"), "console metadata name"),
            dataset_id=_optional_nonempty_string(
                raw.get("dataset_id"), "console dataset ID"
            ),
            dataset_name=_optional_nonempty_string(
                raw.get("dataset_name"), "console dataset name"
            ),
            profile_id=_optional_string(
                raw.get("profile_id"), "console metadata profile_id"
            ),
            profile_name=_optional_string(
                raw.get("profile_name"), "console metadata profile_name"
            ),
            agent_spec=_optional_string(
                raw.get("agent_spec"), "console metadata agent_spec"
            ),
            attempts=_optional_positive_integer(
                raw.get("attempts"), "console metadata attempts"
            ),
            run_workers=_optional_positive_integer(
                raw.get("run_workers"), "console metadata run_workers"
            ),
            score_workers=_optional_positive_integer(
                raw.get("score_workers"), "console metadata score_workers"
            ),
            created_at=_optional_string(
                raw.get("created_at"), "console metadata created_at"
            ),
            retry_of_history_id=_optional_string(
                raw.get("retry_of_history_id"),
                "console metadata retry_of_history_id",
            ),
            retry_number=_optional_positive_integer(
                raw.get("retry_number"), "console metadata retry_number"
            ),
            retry_root_name=_optional_string(
                raw.get("retry_root_name"), "console metadata retry_root_name"
            ),
            extra={
                key: deepcopy(value)
                for key, value in raw.items()
                if key not in known_fields
            },
            _present_fields=frozenset(raw) & known_fields,
        )

    def to_dict(self) -> Dict[str, Any]:
        raw = deepcopy(self.extra)
        values = {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "agent_spec": self.agent_spec,
            "attempts": self.attempts,
            "run_workers": self.run_workers,
            "score_workers": self.score_workers,
            "created_at": self.created_at,
            "retry_of_history_id": self.retry_of_history_id,
            "retry_number": self.retry_number,
            "retry_root_name": self.retry_root_name,
        }
        raw.update(
            {
                name: value
                for name, value in values.items()
                if name in self._present_fields
            }
        )
        return raw


@dataclass(frozen=True)
class CanceledRunRecord:
    run_id: str
    job_id: str
    canceled_at: str
    attempts: int
    metadata: ConsoleMetadata

    @classmethod
    def from_dict(cls, raw: Any) -> "CanceledRunRecord":
        expected = {
            "schema_version",
            "status",
            "run_id",
            "job_id",
            "canceled_at",
            "attempts",
            "metadata",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("canceled run record has invalid fields")
        if raw["schema_version"] != SCHEMA_VERSION or raw["status"] != "canceled":
            raise ValueError("canceled run record has an unsupported schema or status")
        return cls(
            run_id=_nonempty_string(raw["run_id"], "canceled run_id"),
            job_id=_nonempty_string(raw["job_id"], "canceled job_id"),
            canceled_at=_nonempty_string(raw["canceled_at"], "canceled timestamp"),
            attempts=_positive_integer(raw["attempts"], "canceled attempts"),
            metadata=ConsoleMetadata.from_dict(raw["metadata"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "canceled",
            "run_id": self.run_id,
            "job_id": self.job_id,
            "canceled_at": self.canceled_at,
            "attempts": self.attempts,
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class RunManifest:
    schema_version: str
    run_id: str
    status: RunStatus
    input: Dict[str, Any]
    artifacts: Dict[str, str]
    phases: Dict[str, Dict[str, Any]]
    versions: Optional[Dict[str, Any]] = None
    evaluator_profile: Optional[Dict[str, Any]] = None
    attempts: Optional[int] = None
    agent: Optional[Dict[str, Any]] = None
    retry: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        *,
        supported_schema_versions: Iterable[str] = ("qa-v1", SCHEMA_VERSION),
        preparation_files: Tuple[str, ...] = ("preparation.jsonl",),
    ) -> "RunManifest":
        if not isinstance(raw, dict):
            raise ValueError("manifest must be an object")
        schema_version = raw.get("schema_version")
        if not isinstance(schema_version, str) or schema_version not in set(
            supported_schema_versions
        ):
            raise UnsupportedSchemaError("unsupported run schema")
        run_id = _nonempty_string(raw.get("run_id"), "manifest run_id")
        try:
            status = RunStatus(raw.get("status"))
        except ValueError as exc:
            raise ValueError("unsupported run status") from exc

        phases = raw.get("phases")
        if not isinstance(phases, dict):
            raise ValueError("manifest phases must be an object")
        required_phases = {
            RunStatus.PREPARED: ("prepare",),
            RunStatus.ATTENTION_REQUIRED: ("prepare",),
            RunStatus.RUN_COMPLETED: ("prepare", "run"),
            RunStatus.SCORED: ("prepare", "run", "score"),
        }[status]
        if any(
            not isinstance(phases.get(phase), dict)
            or phases[phase].get("status") != "completed"
            for phase in required_phases
        ):
            raise ValueError("manifest phase state is incomplete")

        attempts = raw.get("attempts")
        if status is not RunStatus.PREPARED:
            attempts = _positive_integer(attempts, "manifest attempts")
        elif attempts is not None:
            attempts = _positive_integer(attempts, "manifest attempts")

        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("manifest artifacts must be an object")
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in artifacts.items()
        ):
            raise ValueError("manifest contains an invalid artifact entry")

        input_details = raw.get("input")
        if not isinstance(input_details, dict):
            raise ValueError("manifest input must be an object")
        snapshot = input_details.get("snapshot")
        if not isinstance(snapshot, str) or Path(snapshot).name != snapshot:
            raise ValueError("manifest input snapshot is invalid")
        source_path = input_details.get("source_path")
        if source_path is not None and not isinstance(source_path, str):
            raise ValueError("manifest input source path must be a string")
        if not {snapshot, *preparation_files}.issubset(artifacts):
            raise ValueError("manifest is missing preparation artifacts")
        if (
            schema_version == SCHEMA_VERSION
            and status
            in {
                RunStatus.ATTENTION_REQUIRED,
                RunStatus.RUN_COMPLETED,
                RunStatus.SCORED,
            }
            and "live_checks.jsonl" not in artifacts
        ):
            raise ValueError("qa-v2 manifest is missing live-check artifacts")
        if status is RunStatus.SCORED and not {"summary.json", "report.md"}.issubset(
            artifacts
        ):
            raise ValueError("scored manifest is missing result artifacts")

        known_fields = {
            "schema_version",
            "run_id",
            "status",
            "versions",
            "input",
            "evaluator_profile",
            "attempts",
            "agent",
            "artifacts",
            "phases",
            "retry",
        }
        return cls(
            schema_version=schema_version,
            run_id=run_id,
            status=status,
            versions=deepcopy(raw.get("versions")),
            input=deepcopy(input_details),
            evaluator_profile=deepcopy(raw.get("evaluator_profile")),
            attempts=attempts,
            agent=deepcopy(raw.get("agent")),
            artifacts=deepcopy(artifacts),
            phases=deepcopy(phases),
            retry=deepcopy(raw.get("retry")),
            extra={
                key: deepcopy(value)
                for key, value in raw.items()
                if key not in known_fields
            },
        )

    @property
    def snapshot(self) -> str:
        return self.input["snapshot"]

    @property
    def source_path(self) -> Optional[str]:
        return self.input.get("source_path")

    @property
    def required_attempts(self) -> int:
        if self.attempts is None:
            raise ValueError("manifest attempts must be a positive integer")
        return self.attempts

    @property
    def preparation_input_items(self) -> int:
        value = self.phases["prepare"].get("input_items")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "manifest prepare phase input_items must be a non-negative integer"
            )
        return value

    def require_phase_complete(self, phase: str) -> None:
        phase_state = self.phases.get(phase)
        if (
            not isinstance(phase_state, dict)
            or phase_state.get("status") != "completed"
        ):
            raise ValueError(f"run workspace {phase} phase is not complete")

    def artifact_digest(self, name: str) -> str:
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise ValueError(f"manifest is missing artifact: {name}") from exc

    def to_dict(self) -> Dict[str, Any]:
        raw = deepcopy(self.extra)
        raw.update(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "status": self.status.value,
                "input": deepcopy(self.input),
                "artifacts": deepcopy(self.artifacts),
                "phases": deepcopy(self.phases),
            }
        )
        for name, value in (
            ("versions", self.versions),
            ("evaluator_profile", self.evaluator_profile),
            ("attempts", self.attempts),
            ("agent", self.agent),
            ("retry", self.retry),
        ):
            if value is not None:
                raw[name] = deepcopy(value)
        return raw


@dataclass(frozen=True)
class AttemptIdentity:
    item_id: str
    attempt_id: str
    ordinal: int
    agent_config_sha256: str
    agent_spec_sha256: str

    @classmethod
    def from_dict(cls, row: Dict[str, Any], *, context: str) -> "AttemptIdentity":
        ordinal = _positive_integer(row.get("ordinal"), f"{context}.ordinal")
        config_hash = _nonempty_string(
            row.get("agent_config_sha256"), f"{context}.agent_config_sha256"
        )
        spec_hash = _nonempty_string(
            row.get("agent_spec_sha256"), f"{context}.agent_spec_sha256"
        )
        if re.fullmatch(r"[0-9a-f]{64}", config_hash) is None:
            raise ValueError(f"{context}.agent_config_sha256 must be a SHA-256 digest")
        if re.fullmatch(r"[0-9a-f]{64}", spec_hash) is None:
            raise ValueError(f"{context}.agent_spec_sha256 must be a SHA-256 digest")
        return cls(
            item_id=_nonempty_string(row.get("item_id"), f"{context}.item_id"),
            attempt_id=_nonempty_string(row.get("attempt_id"), f"{context}.attempt_id"),
            ordinal=ordinal,
            agent_config_sha256=config_hash,
            agent_spec_sha256=spec_hash,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "attempt_id": self.attempt_id,
            "ordinal": self.ordinal,
            "agent_config_sha256": self.agent_config_sha256,
            "agent_spec_sha256": self.agent_spec_sha256,
        }


@dataclass(frozen=True)
class AnswerAttempt:
    identity: AttemptIdentity
    status: AnswerStatus
    duration_ms: int
    tool_calls: Tuple[Dict[str, Any], ...]
    answer: Optional[str] = None
    error: Optional[Dict[str, str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any, *, context: str) -> "AnswerAttempt":
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        try:
            status = AnswerStatus(raw.get("status"))
        except ValueError as exc:
            raise ValueError(
                "run contains a non-terminal or unsupported attempt status"
            ) from exc
        duration_ms = raw.get("duration_ms")
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or duration_ms < 0
        ):
            raise ValueError(f"{context}.duration_ms must be a non-negative integer")
        tool_calls = serialize_tool_call_records(
            raw.get("tool_calls"),
            context=f"{context}.tool_calls",
        )
        answer = raw.get("answer")
        error = raw.get("error")
        if status is AnswerStatus.ANSWER_READY:
            if not isinstance(answer, str):
                raise ValueError(f"{context}.answer must be a string")
            if error is not None:
                raise ValueError(f"{context} cannot contain both answer and error")
        else:
            if answer is not None:
                raise ValueError(f"{context} cannot contain both answer and error")
            if (
                not isinstance(error, dict)
                or not isinstance(error.get("type"), str)
                or not isinstance(error.get("message"), str)
            ):
                raise ValueError(f"{context}.error must contain type and message")
            error = {"type": error["type"], "message": error["message"]}
        return cls(
            identity=AttemptIdentity.from_dict(raw, context=context),
            status=status,
            duration_ms=duration_ms,
            tool_calls=tuple(tool_calls),
            answer=answer,
            error=error,
            extra={
                key: deepcopy(value)
                for key, value in raw.items()
                if key
                not in {
                    "item_id",
                    "attempt_id",
                    "ordinal",
                    "agent_config_sha256",
                    "agent_spec_sha256",
                    "status",
                    "duration_ms",
                    "tool_calls",
                    "answer",
                    "error",
                }
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        row = {
            **deepcopy(self.extra),
            **self.identity.to_dict(),
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "tool_calls": [deepcopy(call) for call in self.tool_calls],
        }
        if self.status is AnswerStatus.ANSWER_READY:
            row["answer"] = self.answer
        else:
            row["error"] = deepcopy(self.error)
        return row


@dataclass(frozen=True)
class EvaluationResult:
    identity: AttemptIdentity
    status: EvaluationStatus
    payload: Dict[str, Any]

    @classmethod
    def from_dict(cls, raw: Any, *, context: str) -> "EvaluationResult":
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        try:
            status = EvaluationStatus(raw.get("status"))
        except ValueError as exc:
            raise ValueError(
                "parent run contains an unsupported result status"
            ) from exc
        identity = AttemptIdentity.from_dict(raw, context=context)
        identity_fields = set(identity.to_dict()) | {"status"}
        return cls(
            identity=identity,
            status=status,
            payload={
                key: deepcopy(value)
                for key, value in raw.items()
                if key not in identity_fields
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "status": self.status.value,
            **deepcopy(self.payload),
        }
