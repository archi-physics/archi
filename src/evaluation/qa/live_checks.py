# isort: skip_file
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from .artifacts import iter_jsonl
from .oracle import (
    DIAGNOSTIC_LIMIT,
    OracleCallEvidence,
    OracleResolutionError,
    OracleResolver,
    answer_sha256,
    bounded_diagnostic,
    oracle_call_evidence_from_dict,
    validate_json_value,
)
from .preparation import PreparationRecord, iter_preparation_records


class LiveCheckPhase(str, Enum):
    PRE_RUN = "pre_run"
    POST_RUN = "post_run"


class LiveCheckStatus(str, Enum):
    RESOLVED = "resolved"
    FAILED = "failed"


class LiveFailureReason(str, Enum):
    ORACLE_FAILED = "oracle_failed"
    ANSWER_CHANGED = "answer_changed"


@dataclass(frozen=True)
class LiveValidation:
    phase: LiveCheckPhase
    reason: LiveFailureReason
    detail: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.detail, str)
            or not self.detail
            or len(self.detail) > DIAGNOSTIC_LIMIT
            or "\x00" in self.detail
            or any("\ud800" <= char <= "\udfff" for char in self.detail)
        ):
            raise ValueError("live validation detail must be bounded non-empty text")

    def to_dict(self) -> Dict[str, str]:
        return {
            "phase": self.phase.value,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LiveCheck:
    item_id: str
    phase: LiveCheckPhase
    status: LiveCheckStatus
    answer: Optional[Dict[str, Any]] = None
    answer_sha256: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    calls: Tuple[OracleCallEvidence, ...] = ()
    live_validation: Optional[LiveValidation] = None

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("live check item_id must be a non-empty string")
        if self.status is LiveCheckStatus.RESOLVED:
            if (
                not isinstance(self.answer, dict)
                or not self.answer
                or not isinstance(self.answer_sha256, str)
                or self.answer_sha256 != answer_sha256(self.answer)
                or not isinstance(self.metadata, dict)
                or self.live_validation is not None
            ):
                raise ValueError("resolved live check has invalid fields")
            validate_json_value(self.metadata, "resolved live check metadata")
        elif (
            self.answer is not None
            or self.answer_sha256 is not None
            or self.metadata is not None
            or self.live_validation is None
            or self.live_validation.reason is not LiveFailureReason.ORACLE_FAILED
        ):
            raise ValueError("failed live check has invalid fields")

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "item_id": self.item_id,
            "phase": self.phase.value,
            "status": self.status.value,
            "calls": [call.to_dict() for call in self.calls],
        }
        if self.status is LiveCheckStatus.RESOLVED:
            row.update(
                {
                    "answer": self.answer,
                    "answer_sha256": self.answer_sha256,
                    "metadata": self.metadata,
                }
            )
        else:
            assert self.live_validation is not None
            row["live_validation"] = self.live_validation.to_dict()
        return row


def observe_live_item(
    prepared: PreparationRecord,
    resolver: OracleResolver,
    phase: LiveCheckPhase,
) -> LiveCheck:
    if prepared.status != "prepared" or not prepared.time_sensitive:
        raise ValueError("live checks require a prepared live item")
    assert prepared.oracle is not None
    try:
        resolved = resolver.resolve(prepared.oracle)
    except OracleResolutionError as exc:
        return LiveCheck(
            item_id=prepared.item_id,
            phase=phase,
            status=LiveCheckStatus.FAILED,
            calls=exc.calls,
            live_validation=LiveValidation(
                phase=phase,
                reason=LiveFailureReason.ORACLE_FAILED,
                detail=bounded_diagnostic(exc.detail),
            ),
        )
    return LiveCheck(
        item_id=prepared.item_id,
        phase=phase,
        status=LiveCheckStatus.RESOLVED,
        answer=resolved.answer,
        answer_sha256=resolved.answer_sha256,
        metadata=resolved.metadata,
        calls=resolved.calls,
    )


def validation_against_baseline(
    prepared: PreparationRecord, check: LiveCheck
) -> Optional[LiveValidation]:
    if not prepared.time_sensitive:
        return None
    if check.status is LiveCheckStatus.FAILED:
        assert check.live_validation is not None
        return check.live_validation
    if check.answer_sha256 != prepared.answer_sha256:
        return LiveValidation(
            phase=check.phase,
            reason=LiveFailureReason.ANSWER_CHANGED,
            detail="The resolved answer no longer matches the approved baseline.",
        )
    return None


def _call_evidence(raw: Any, context: str) -> Tuple[OracleCallEvidence, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a list")
    values = []
    for index, row in enumerate(raw):
        item_context = f"{context}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{item_context} must be an object")
        required = {"call_id", "duration_ms", "success"}
        allowed = required | {"error"}
        if not required.issubset(row) or set(row) - allowed:
            raise ValueError(f"{item_context} has invalid fields")
        values.append(oracle_call_evidence_from_dict(row, item_context))
    return tuple(values)


def live_check_from_dict(raw: Any, context: str) -> LiveCheck:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    try:
        phase = LiveCheckPhase(raw.get("phase"))
        status = LiveCheckStatus(raw.get("status"))
    except ValueError as exc:
        raise ValueError(f"{context} has unsupported phase or status") from exc
    base = {"item_id", "phase", "status", "calls"}
    expected = (
        base | {"answer", "answer_sha256", "metadata"}
        if status is LiveCheckStatus.RESOLVED
        else base | {"live_validation"}
    )
    if set(raw) != expected:
        raise ValueError(f"{context} has invalid fields")
    validation = None
    if status is LiveCheckStatus.FAILED:
        value = raw["live_validation"]
        if not isinstance(value, dict) or set(value) != {"phase", "reason", "detail"}:
            raise ValueError(f"{context}.live_validation has invalid fields")
        try:
            validation = LiveValidation(
                phase=LiveCheckPhase(value["phase"]),
                reason=LiveFailureReason(value["reason"]),
                detail=value["detail"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.live_validation is invalid") from exc
        if validation.phase is not phase:
            raise ValueError(f"{context}.live_validation phase does not match")
    return LiveCheck(
        item_id=raw["item_id"],
        phase=phase,
        status=status,
        answer=raw.get("answer"),
        answer_sha256=raw.get("answer_sha256"),
        metadata=raw.get("metadata"),
        calls=_call_evidence(raw["calls"], f"{context}.calls"),
        live_validation=validation,
    )


def iter_live_checks(
    path: Path, *, phase: Optional[LiveCheckPhase] = None
) -> Iterator[LiveCheck]:
    for index, row in enumerate(iter_jsonl(path), 1):
        check = live_check_from_dict(row, f"live check row {index}")
        if phase is None or check.phase is phase:
            yield check


def iter_precheck_decisions(
    preparation_path: Path,
    checks_path: Path,
    *,
    expected_preparation_count: int,
) -> Iterator[Tuple[PreparationRecord, LiveCheck, Optional[LiveValidation]]]:
    """Lockstep prepared live rows with ordered pre-run observations."""
    checks = iter_live_checks(checks_path, phase=LiveCheckPhase.PRE_RUN)
    for prepared in iter_preparation_records(
        preparation_path,
        expected_count=expected_preparation_count,
    ):
        if prepared.status != "prepared" or not prepared.time_sensitive:
            continue
        try:
            check = next(checks)
        except StopIteration:
            raise ValueError("pre-run live checks are missing an item") from None
        if check.item_id != prepared.item_id:
            raise ValueError("pre-run live check order does not match preparation")
        yield prepared, check, validation_against_baseline(prepared, check)
    try:
        next(checks)
    except StopIteration:
        return
    raise ValueError("pre-run live checks contain an extra item")
