"""Utilities for aggregating run-level documents, notes, and tool calls."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document


class RunMemory:
    """Track documents, notes, and tool call inputs produced in one run."""
    # TODO for now we return langchain's Document objects. We could think about returning the same Resource classes we use when collecting these (or vice versa) to reduce the amount of dataclasses to worry about.
    # TODO we don't collect retriever scores

    def __init__(self) -> None:
        self._document_events: List[Tuple[str, List[Document]]] = []
        self._notes: List[str] = []
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        self._pending_tool_inputs_by_name: Dict[str, List[Any]] = {}

    def record(self, stage: str, documents: Iterable[Document]) -> None:
        """Store the documents captured for a specific stage or tool call."""
        docs_list: List[Document] = [doc for doc in documents if doc]
        if not docs_list:
            return
        self._document_events.append((stage, docs_list))
    
    def record_documents(self, stage: str, documents: Iterable[Document]) -> None:
        """Convenience wrapper that records documents and appends a note.

        This is a small helper used by agent callbacks so they can record a
        batch of documents and also create a short collector note in one call.
        """
        docs_list = [doc for doc in documents if doc]
        if not docs_list:
            return
        self.record(stage, docs_list)
        self.note(f"{stage} returned {len(docs_list)} document(s).")

    def note(self, message: str) -> None:
        """Append a textual note describing an intermediate step."""
        if not message:
            return
        self._notes.append(message)

    def record_tool_call(self, tool_call_id: str, tool_name: str, tool_input: Any) -> None:
        """Store/refresh a tool call entry keyed by call id."""
        if not tool_call_id:
            return
        existing = self._tool_runs.get(tool_call_id, {})
        self._tool_runs[tool_call_id] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name or existing.get("tool_name", "unknown"),
            "tool_input": tool_input if tool_input not in (None, "") else existing.get("tool_input", {}),
            "documents": existing.get("documents", []),
        }

    def record_tool_input(self, tool_name: str, tool_input: Any) -> None:
        """Record a runtime tool input before the tool_call_id is known."""
        if not tool_name:
            return

        # Prefer attaching to an already-seen call id with missing args.
        # This happens when providers stream tool_call ids first, then execute.
        for tool_call_id, run in self._tool_runs.items():
            if run.get("tool_name") != tool_name:
                continue
            existing_input = run.get("tool_input", {})
            if existing_input in (None, "", {}, []):
                self._tool_runs[tool_call_id] = {
                    **run,
                    "tool_input": tool_input,
                }
                return

        queue = self._pending_tool_inputs_by_name.setdefault(tool_name, [])
        queue.append(tool_input)

    def resolve_tool_input(self, tool_call_id: str, tool_name: str, tool_args: Any) -> Any:
        """Resolve empty tool args from pending runtime inputs and bind to call id."""
        if tool_args not in (None, "", {}, []):
            return tool_args
        if not tool_call_id or not tool_name:
            return tool_args
        queue = self._pending_tool_inputs_by_name.get(tool_name) or []
        if not queue:
            return tool_args
        resolved = queue.pop(0)
        self.record_tool_call(tool_call_id, tool_name, resolved)
        return resolved

    def record_tool_calls_from_message(self, message: Any) -> None:
        """Extract tool calls from an LLM message and persist inputs by id."""
        tool_calls = getattr(message, "tool_calls", None) or []
        raw_args_by_id: Dict[str, Any] = {}
        raw_name_by_id: Dict[str, str] = {}

        additional = getattr(message, "additional_kwargs", {}) or {}
        raw_tool_calls = additional.get("tool_calls") or []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            raw_id = raw_call.get("id")
            function_obj = raw_call.get("function") or {}
            raw_name = function_obj.get("name")
            raw_arguments = function_obj.get("arguments")
            parsed = self._parse_tool_arguments(raw_arguments)
            if raw_id and parsed is not None:
                raw_args_by_id[raw_id] = parsed
            if raw_id and isinstance(raw_name, str) and raw_name.strip():
                raw_name_by_id[raw_id] = raw_name.strip()

        # Newer streaming payloads may expose incremental tool chunks separately.
        tool_call_chunks = getattr(message, "tool_call_chunks", None) or []
        for chunk in tool_call_chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("id")
            chunk_name = chunk.get("name")
            chunk_args = self._parse_tool_arguments(chunk.get("args"))
            if chunk_id and chunk_args is not None:
                raw_args_by_id[chunk_id] = chunk_args
            if chunk_id and isinstance(chunk_name, str) and chunk_name.strip():
                raw_name_by_id[chunk_id] = chunk_name.strip()

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool_call_id = call.get("id", "")
            if not tool_call_id:
                continue
            tool_args = call.get("args", {})
            if tool_args in (None, "", {}, []):
                tool_args = raw_args_by_id.get(tool_call_id, tool_args)
            tool_name = call.get("name", "unknown")
            if (not tool_name or str(tool_name).strip().lower() == "unknown") and tool_call_id in raw_name_by_id:
                tool_name = raw_name_by_id[tool_call_id]
            tool_args = self.resolve_tool_input(tool_call_id, tool_name, tool_args)
            self.record_tool_call(tool_call_id, tool_name, tool_args)

    def record_tool_documents(self, tool_call_id: str, documents: Iterable[Document]) -> None:
        """Attach retrieved documents to a previously seen tool call."""
        if not tool_call_id:
            return
        docs_list = [doc for doc in documents if doc]
        if not docs_list:
            return
        current = self._tool_runs.get(tool_call_id)
        if not current:
            current = {
                "tool_call_id": tool_call_id,
                "tool_name": "unknown",
                "tool_input": {},
                "documents": [],
            }
        current_docs = current.get("documents", [])
        current_docs.extend(docs_list)
        current["documents"] = current_docs
        self._tool_runs[tool_call_id] = current

    @property
    def notes(self) -> Sequence[str]:
        return tuple(self._notes)

    @property
    def events(self) -> Sequence[Tuple[str, List[Document]]]:
        return tuple(self._document_events)

    @property
    def tool_runs(self) -> Sequence[Dict[str, Any]]:
        return tuple(self._tool_runs.values())

    def unique_documents(self) -> List[Document]:
        """Return documents with simple deduplication by source metadata."""
        seen: set[Tuple[str, str, str]] = set()
        collected: List[Document] = []
        for _, docs in self._document_events:
            for doc in docs:
                key = self._document_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(doc)
        return collected

    def intermediate_steps(self) -> List[str]:
        """Combine stored notes with document-event breadcrumbs."""
        steps = list(self._notes)
        for stage, docs in self._document_events:
            steps.append(f"{stage}: {len(docs)} document(s)")
        return steps

    def tool_inputs_by_id(self) -> Dict[str, Dict[str, Any]]:
        """Return a stable mapping of tool call ids to serialized tool input."""
        payload: Dict[str, Dict[str, Any]] = {}
        for tool_call_id, run in self._tool_runs.items():
            payload[tool_call_id] = {
                "tool_call_id": tool_call_id,
                "tool_name": run.get("tool_name", "unknown"),
                "tool_input": run.get("tool_input", {}),
            }
        return payload

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any) -> Optional[Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if not isinstance(raw_arguments, str):
            return None
        if not raw_arguments.strip():
            return None
        try:
            return json.loads(raw_arguments)
        except Exception:
            return {"_raw_arguments": raw_arguments}

    @staticmethod
    def _document_key(doc: Document) -> Tuple[str, str, str]:
        metadata = doc.metadata or {}
        return (
            str(metadata.get("document_id") or metadata.get("id") or metadata.get("source", "")),
            str(metadata.get("path") or metadata.get("file_path") or ""),
            doc.page_content[:200] if doc.page_content else "",
        )
