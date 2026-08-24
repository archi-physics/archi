from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mcp.types import CallToolResult, TextContent

DIAGNOSTIC_LIMIT = 4096
MAX_SELECTED_ORACLE_BYTES = 25 * 1024 * 1024
ORACLE_KIND = "mcp"


def _strict_keys(value: Mapping[str, Any], allowed: set, context: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unknown:
        details.append("unknown: " + ", ".join(unknown))
    if details:
        raise ValueError(f"{context} has invalid fields ({'; '.join(details)})")


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    if "\x00" in value or any("\ud800" <= char <= "\udfff" for char in value):
        raise ValueError(f"{context} must contain valid Unicode scalar values")
    return value


def validate_json_value(value: Any, context: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            _nonempty_string(value, context) if value else value
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} must not contain non-finite numbers")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, f"{context}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} object keys must be strings")
            if "\x00" in key or any("\ud800" <= char <= "\udfff" for char in key):
                raise ValueError(
                    f"{context} object keys must contain valid Unicode scalar values"
                )
            validate_json_value(item, f"{context}.{key}")
        return value
    raise ValueError(f"{context} must contain only JSON values")


def canonical_json(value: Any) -> str:
    validate_json_value(value, "canonical JSON")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def answer_sha256(value: Dict[str, Any]) -> str:
    if not value:
        raise ValueError("answer data must be a non-empty object")
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key '{key}'")
        value[key] = item
    return value


def parse_strict_json_object(text: str, context: str) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains non-finite number {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{context} must be one strict JSON object: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{context} must be a non-empty JSON object")
    validate_json_value(value, context)
    return value


def _validate_pointer(pointer: Any, context: str) -> str:
    if not isinstance(pointer, str):
        raise ValueError(f"{context} must be an RFC 6901 JSON Pointer")
    if pointer and not pointer.startswith("/"):
        raise ValueError(f"{context} must be an RFC 6901 JSON Pointer")
    index = 0
    while index < len(pointer):
        if pointer[index] == "~":
            if index + 1 >= len(pointer) or pointer[index + 1] not in {"0", "1"}:
                raise ValueError(f"{context} must be an RFC 6901 JSON Pointer")
            index += 2
            continue
        index += 1
    return pointer


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    _validate_pointer(pointer, "JSON pointer")
    current = value
    if pointer == "":
        return deepcopy(current)
    for encoded_token in pointer[1:].split("/"):
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"JSON pointer '{pointer}' does not exist")
            current = current[token]
        elif isinstance(current, list):
            if token == "-" or not token.isdigit():
                raise ValueError(f"JSON pointer '{pointer}' does not exist")
            if len(token) > 1 and token.startswith("0"):
                raise ValueError(f"JSON pointer '{pointer}' has an invalid array index")
            index = int(token)
            if index >= len(current):
                raise ValueError(f"JSON pointer '{pointer}' does not exist")
            current = current[index]
        else:
            raise ValueError(f"JSON pointer '{pointer}' does not exist")
    validate_json_value(current, f"selected value at {pointer}")
    return deepcopy(current)


def _parse_field_mapping(
    raw: Any, context: str, *, required_when_present: bool
) -> Optional[Tuple[Tuple[str, str], ...]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    if required_when_present and not raw:
        raise ValueError(f"{context} must be a non-empty object")
    fields: List[Tuple[str, str]] = []
    for output_name, pointer in raw.items():
        name = _nonempty_string(output_name, f"{context} output name")
        fields.append((name, _validate_pointer(pointer, f"{context}.{name}")))
    return tuple(fields)


@dataclass(frozen=True)
class OracleCall:
    id: str
    server: str
    tool: str
    arguments: Dict[str, Any]
    answer_fields: Optional[Tuple[Tuple[str, str], ...]] = None
    metadata_fields: Optional[Tuple[Tuple[str, str], ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "id": self.id,
            "server": self.server,
            "tool": self.tool,
            "arguments": deepcopy(self.arguments),
        }
        if self.answer_fields is not None:
            value["answer_fields"] = dict(self.answer_fields)
        if self.metadata_fields is not None:
            value["metadata_fields"] = dict(self.metadata_fields)
        return value


@dataclass(frozen=True)
class OracleRecipe:
    kind: str
    calls: Tuple[OracleCall, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "calls": [call.to_dict() for call in self.calls]}


def parse_oracle_recipe(raw: Any, context: str = "oracle") -> OracleRecipe:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    _strict_keys(raw, {"kind", "calls"}, context)
    if raw["kind"] != ORACLE_KIND:
        raise ValueError(f"{context}.kind must be '{ORACLE_KIND}'")
    raw_calls = raw["calls"]
    if not isinstance(raw_calls, list) or not raw_calls:
        raise ValueError(f"{context}.calls must be a non-empty list")
    calls: List[OracleCall] = []
    seen_ids = set()
    allowed = {
        "id",
        "server",
        "tool",
        "arguments",
        "answer_fields",
        "metadata_fields",
    }
    for index, value in enumerate(raw_calls):
        call_context = f"{context}.calls[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{call_context} must be an object")
        required = {"id", "server", "tool", "arguments"}
        unknown = sorted(set(value) - allowed)
        missing = sorted(required - set(value))
        if missing or unknown:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                f"{call_context} has invalid fields ({'; '.join(details)})"
            )
        call_id = _nonempty_string(value["id"], f"{call_context}.id")
        if call_id in seen_ids:
            raise ValueError(f"{context} contains duplicate call id '{call_id}'")
        seen_ids.add(call_id)
        arguments = value["arguments"]
        if not isinstance(arguments, dict):
            raise ValueError(f"{call_context}.arguments must be an object")
        validate_json_value(arguments, f"{call_context}.arguments")
        calls.append(
            OracleCall(
                id=call_id,
                server=_nonempty_string(value["server"], f"{call_context}.server"),
                tool=_nonempty_string(value["tool"], f"{call_context}.tool"),
                arguments=deepcopy(arguments),
                answer_fields=_parse_field_mapping(
                    value.get("answer_fields"),
                    f"{call_context}.answer_fields",
                    required_when_present=True,
                ),
                metadata_fields=_parse_field_mapping(
                    value.get("metadata_fields"),
                    f"{call_context}.metadata_fields",
                    required_when_present=False,
                ),
            )
        )
    return OracleRecipe(kind=ORACLE_KIND, calls=tuple(calls))


def normalize_call_tool_result(result: CallToolResult) -> Dict[str, Any]:
    if result.isError:
        raise ValueError("MCP tool returned isError: true")
    if result.structuredContent is not None:
        payload = result.structuredContent
        if not isinstance(payload, dict) or not payload:
            raise ValueError("MCP structuredContent must be a non-empty JSON object")
        validate_json_value(payload, "MCP structuredContent")
        return deepcopy(payload)
    if len(result.content) != 1 or not isinstance(result.content[0], TextContent):
        raise ValueError(
            "MCP content fallback must contain exactly one TextContent block"
        )
    return parse_strict_json_object(result.content[0].text, "MCP TextContent")


class OraclePayloadError(ValueError):
    """A provider-output failure whose message is safe to persist."""


def _normalize_provider_payload(result: CallToolResult) -> Dict[str, Any]:
    try:
        return normalize_call_tool_result(result)
    except Exception as exc:
        raise OraclePayloadError("Evaluator MCP result was invalid.") from exc


def _select_provider_data(
    payload: Dict[str, Any],
    fields: Optional[Tuple[Tuple[str, str], ...]],
) -> Dict[str, Any]:
    try:
        return select_call_data(payload, fields)
    except Exception as exc:
        raise OraclePayloadError(
            "Evaluator MCP result did not match the configured selection."
        ) from exc


def select_call_data(
    payload: Dict[str, Any], fields: Optional[Tuple[Tuple[str, str], ...]]
) -> Dict[str, Any]:
    if fields is None:
        return deepcopy(payload)
    selected = {
        output_name: resolve_json_pointer(payload, pointer)
        for output_name, pointer in fields
    }
    if not selected:
        raise ValueError("selected answer data must be a non-empty object")
    return selected


def bounded_diagnostic(message: Any, secrets: Sequence[str] = ()) -> str:
    text = str(message)
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = text or "Oracle operation failed."
    if len(text) > DIAGNOSTIC_LIMIT:
        text = text[: DIAGNOSTIC_LIMIT - 1] + "…"
    return text


@dataclass(frozen=True)
class OracleCallEvidence:
    call_id: str
    duration_ms: int
    success: bool
    error: Optional[str] = None

    def __post_init__(self) -> None:
        _nonempty_string(self.call_id, "oracle call evidence call_id")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ValueError(
                "oracle call evidence duration_ms must be a non-negative integer"
            )
        if not isinstance(self.success, bool):
            raise ValueError("oracle call evidence success must be a boolean")
        if self.success:
            if self.error is not None:
                raise ValueError("successful oracle call evidence cannot contain error")
        elif (
            not isinstance(self.error, str)
            or not self.error
            or len(self.error) > DIAGNOSTIC_LIMIT
        ):
            raise ValueError(
                "failed oracle call evidence requires a bounded non-empty error"
            )
        if self.error is not None:
            _nonempty_string(self.error, "oracle call evidence error")

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "call_id": self.call_id,
            "duration_ms": self.duration_ms,
            "success": self.success,
        }
        if self.error is not None:
            value["error"] = self.error
        return value


def oracle_call_evidence_from_dict(raw: Any, context: str) -> OracleCallEvidence:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    required = {"call_id", "duration_ms", "success"}
    allowed = required | {"error"}
    if not required.issubset(raw) or set(raw) - allowed:
        raise ValueError(f"{context} has invalid fields")
    try:
        return OracleCallEvidence(
            call_id=raw["call_id"],
            duration_ms=raw["duration_ms"],
            success=raw["success"],
            error=raw.get("error"),
        )
    except ValueError as exc:
        raise ValueError(f"{context} has invalid evidence values") from exc


@dataclass(frozen=True)
class ResolvedOracle:
    answer: Dict[str, Any]
    answer_sha256: str
    metadata: Dict[str, Any]
    calls: Tuple[OracleCallEvidence, ...]


class OracleResolutionError(RuntimeError):
    def __init__(self, detail: str, calls: Sequence[OracleCallEvidence]):
        super().__init__(detail)
        self.detail = bounded_diagnostic(detail)
        self.calls = tuple(calls)


class OracleInvoker:
    def invoke(self, call: OracleCall) -> Tuple[CallToolResult, OracleCallEvidence]:
        raise NotImplementedError


class OracleResolver:
    def __init__(self, invoker: OracleInvoker):
        self._invoker = invoker

    def resolve(self, recipe: OracleRecipe) -> ResolvedOracle:
        answer: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}
        evidence: List[OracleCallEvidence] = []
        for call in recipe.calls:
            call_evidence: Optional[OracleCallEvidence] = None
            try:
                result, call_evidence = self._invoker.invoke(call)
                payload = _normalize_provider_payload(result)
                selected_answer = _select_provider_data(payload, call.answer_fields)
                selected_metadata = (
                    {}
                    if not call.metadata_fields
                    else _select_provider_data(payload, call.metadata_fields)
                )
                selected_size = len(
                    canonical_json(
                        {"answer": selected_answer, "metadata": selected_metadata}
                    ).encode("utf-8")
                )
                if selected_size > MAX_SELECTED_ORACLE_BYTES:
                    raise OraclePayloadError(
                        "Evaluator MCP selected result exceeds the 25 MB limit."
                    )
                answer[call.id] = selected_answer
                metadata[call.id] = selected_metadata
                evidence.append(call_evidence)
            except Exception as exc:
                if isinstance(exc, OracleResolutionError):
                    evidence.extend(exc.calls)
                    detail = exc.detail
                elif isinstance(exc, OraclePayloadError):
                    detail = bounded_diagnostic(exc)
                else:
                    detail = f"Evaluator MCP call failed ({type(exc).__name__})."
                if not evidence or evidence[-1].call_id != call.id:
                    evidence.append(
                        OracleCallEvidence(
                            call_id=call.id,
                            duration_ms=(
                                call_evidence.duration_ms
                                if call_evidence is not None
                                else 0
                            ),
                            success=False,
                            error=detail,
                        )
                    )
                raise OracleResolutionError(detail, evidence) from exc
        return ResolvedOracle(
            answer=answer,
            answer_sha256=answer_sha256(answer),
            metadata=metadata,
            calls=tuple(evidence),
        )
