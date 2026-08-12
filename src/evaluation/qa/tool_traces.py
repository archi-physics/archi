from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class ToolCallRecord:
    """Validated persisted evidence for one observed tested-agent tool call."""

    ordinal: int
    name: str
    status: ToolCallStatus
    query: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal <= 0
        ):
            raise ValueError("tool-call ordinal must be a positive integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool-call name must be a non-empty string")
        if not isinstance(self.status, ToolCallStatus):
            raise ValueError("tool-call status must be a ToolCallStatus")
        if self.duration_ms is not None and (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ValueError("tool-call duration_ms must be a non-negative integer")

        text_fields = {
            "query": self.query,
            "response": self.response,
            "error": self.error,
        }
        if any(
            value is not None and not isinstance(value, str)
            for value in text_fields.values()
        ):
            raise ValueError("tool-call query, response, and error must be strings")
        if self.status == ToolCallStatus.INCOMPLETE and self.duration_ms is not None:
            raise ValueError("incomplete tool-call records cannot contain duration")

        # A missing query identifies a legacy timing-only record. It may not carry
        # terminal detail that the historical artifact did not capture.
        if self.query is None:
            if self.status == ToolCallStatus.INCOMPLETE:
                raise ValueError("incomplete tool-call records require a query")
            if self.response is not None or self.error is not None:
                raise ValueError(
                    "legacy tool-call records cannot contain response detail"
                )
            return
        if self.status == ToolCallStatus.SUCCESS:
            if self.response is None or self.error is not None:
                raise ValueError("successful tool-call records require only a response")
        elif self.status == ToolCallStatus.ERROR:
            if self.error is None or self.response is not None:
                raise ValueError("failed tool-call records require only an error")
        elif self.response is not None or self.error is not None:
            raise ValueError(
                "incomplete tool-call records cannot contain terminal detail"
            )

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "ordinal": self.ordinal,
            "name": self.name,
            "status": self.status.value,
        }
        for field in ("query", "response", "error", "duration_ms"):
            value = getattr(self, field)
            if value is not None:
                row[field] = value
        return row


def _require_exact_keys(
    row: Dict[str, Any], allowed: Set[str], *, context: str
) -> None:
    unknown = sorted(set(row) - allowed)
    missing = sorted({"ordinal", "name", "status"} - set(row))
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unknown:
        details.append("unknown: " + ", ".join(unknown))
    if details:
        raise ValueError(f"{context} has invalid fields ({'; '.join(details)})")


def tool_call_record_from_dict(row: Dict[str, Any], *, context: str) -> ToolCallRecord:
    if not isinstance(row, dict):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        row,
        {"ordinal", "name", "status", "query", "response", "error", "duration_ms"},
        context=context,
    )
    null_fields = sorted(
        field
        for field in ("query", "response", "error", "duration_ms")
        if field in row and row[field] is None
    )
    if null_fields:
        raise ValueError(
            f"{context} has null fields that must be omitted: {', '.join(null_fields)}"
        )
    try:
        status = ToolCallStatus(row["status"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has an unsupported status") from exc
    try:
        return ToolCallRecord(
            ordinal=row["ordinal"],
            name=row["name"],
            status=status,
            query=row.get("query"),
            response=row.get("response"),
            error=row.get("error"),
            duration_ms=row.get("duration_ms"),
        )
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}") from exc


def load_tool_call_records(value: Any, *, context: str) -> List[ToolCallRecord]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    records = [
        tool_call_record_from_dict(row, context=f"{context}[{index}]")
        for index, row in enumerate(value)
    ]
    ordinals = [record.ordinal for record in records]
    if ordinals != sorted(set(ordinals)):
        raise ValueError(f"{context} ordinals must be unique and ordered")
    return records


def serialize_tool_call_records(
    value: Sequence[Dict[str, Any]], *, context: str
) -> List[Dict[str, Any]]:
    return [
        record.to_dict()
        for record in load_tool_call_records(list(value), context=context)
    ]
