from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from src.archi.utils.output_dataclass import PipelineOutput

JIRA_TRACE_SECTION_MAX_CHARS = 4000


def trigger_error_message(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if not detail:
        return prefix
    return f"{prefix}: {detail}"


def extract_answer(result: object) -> Optional[str]:
    answer = getattr(result, "answer", None)
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    if not answer:
        return None
    return answer


def build_jira_comment_body(answer: str, result: object) -> str:
    sections = []
    reasoning_trace = extract_reasoning_trace(result)
    if reasoning_trace:
        sections.append(
            format_jira_panel("Reasoning trace", format_jira_noformat(reasoning_trace))
        )

    tool_calls = extract_tool_calls_trace(result)
    if tool_calls:
        sections.append(format_tool_calls_panel(tool_calls))

    if not sections:
        return answer.strip()
    return "\n\n".join([answer.strip(), *sections])


def extract_reasoning_trace(result: object) -> str:
    if not isinstance(result, PipelineOutput):
        return ""

    reasoning_blocks = []
    for message in result.messages:
        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
        reasoning_content = additional_kwargs.get("reasoning_content")
        if reasoning_content:
            reasoning_blocks.append(str(reasoning_content).strip())
    return "\n\n".join(block for block in reasoning_blocks if block)


def extract_tool_calls_trace(result: object) -> list[dict[str, Any]]:
    if not isinstance(result, PipelineOutput):
        return []
    return result.extract_tool_calls()


def format_tool_calls_panel(tool_calls: list[dict[str, Any]]) -> str:
    return format_jira_panel(
        "Tool calls", format_jira_noformat(format_tool_calls_trace(tool_calls))
    )


def format_tool_calls_trace(tool_calls: list[dict[str, Any]]) -> str:
    parts = []
    for index, tool_call in enumerate(tool_calls, start=1):
        tool_name = str(tool_call.get("name") or "unknown")
        tool_args = serialize_jira_trace_value(tool_call.get("args"))
        tool_result = serialize_jira_trace_value(tool_call.get("result"))
        parts.append(
            "\n".join(
                [
                    f"Tool call {index}: {tool_name}",
                    "Input:",
                    tool_args or "No input captured.",
                    "",
                    "Output:",
                    tool_result or "No output captured.",
                ]
            )
        )
    return "\n\n".join(parts)


def format_jira_panel(title: str, body: str) -> str:
    return f"{{panel:title={title}}}\n{body}\n{{panel}}"


def format_jira_noformat(value: str) -> str:
    text = sanitize_jira_noformat(truncate_jira_trace_text(value))
    return f"{{noformat}}\n{text}\n{{noformat}}"


def truncate_jira_trace_text(value: str) -> str:
    text = value.strip()
    if len(text) <= JIRA_TRACE_SECTION_MAX_CHARS:
        return text
    omitted = len(text) - JIRA_TRACE_SECTION_MAX_CHARS
    return (
        f"{text[:JIRA_TRACE_SECTION_MAX_CHARS].rstrip()}"
        f"\n\n[truncated {omitted} characters]"
    )


def sanitize_jira_noformat(value: str) -> str:
    return re.sub(r"\{noformat\}", "{ noformat }", value, flags=re.IGNORECASE)


def serialize_jira_trace_value(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def format_source_context(source_documents: Iterable[Any]) -> tuple[str, str]:
    link = ""
    context_parts = []
    for index, document in enumerate(source_documents, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        document_link = str(metadata.get("url") or "")
        if not link and document_link:
            link = document_link
        title = str(metadata.get("title") or metadata.get("display_name") or "No Title")
        content = str(getattr(document, "page_content", "") or "")
        context_parts.append(f"Source {index}: {title} ({document_link})\n\n{content}")
    return link, "\n\n\n\n".join(context_parts)
