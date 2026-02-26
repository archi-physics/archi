"""Shared event formatter for converting PipelineOutput into structured JSON events.

Both Chat.stream() (regular chat) and _stream_arm() (A/B testing) use this
formatter so event structure is defined in exactly one place.

Usage::

    formatter = PipelineEventFormatter(message_content_fn=self._message_content)
    for output in pipeline.stream(...):
        for event in formatter.process(output):
            # caller adds context fields (arm, conversation_id, timestamp …)
            yield event
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Iterator, Optional

from src.archi.utils.output_dataclass import PipelineOutput

logger = logging.getLogger(__name__)


class PipelineEventFormatter:
    """Stateful converter: PipelineOutput → structured streaming event dicts.

    Key behaviours:
    * **Deferred tool_start** – on ``tool_start`` events the formatter parses
      and *remembers* tool calls but does NOT yield events.  When the
      corresponding ``tool_output`` arrives it yields ``tool_start`` then
      ``tool_output``, ensuring every tool-start has a matching output.
    * **Progressive merging** – tool info is aggregated from ``tool_calls``,
      ``additional_kwargs.tool_calls``, ``tool_call_chunks``, and
      ``metadata.tool_inputs_by_id`` so the emitted ``tool_start`` carries the
      best-available name and args.
    * **Caller decorates** – yielded events contain only the canonical fields
      (``type`` + type-specific data).  Callers add ``conversation_id``,
      ``timestamp``, ``arm``, etc.
    """

    def __init__(
        self,
        *,
        message_content_fn: Callable,
        max_step_chars: int = 800,
    ) -> None:
        self._message_content = message_content_fn
        self._max_chars = max_step_chars

        # Tool-call tracking
        self._emitted_ids: set[str] = set()        # all tool_call_ids we've seen
        self._emitted_start_ids: set[str] = set()   # ids we've yielded tool_start for
        self._pending_ids: list[str] = []            # ids awaiting their output (ordered)
        self._calls: Dict[str, Dict[str, Any]] = {}  # id → {tool_name, tool_args}
        self._synthetic_counter: int = 0

        # Public counters for callers
        self.tool_call_count: int = 0
        self.last_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, output: PipelineOutput) -> Iterator[Dict[str, Any]]:
        """Yield zero or more event dicts from a single *PipelineOutput*."""
        if not isinstance(output, PipelineOutput):
            return

        meta = output.metadata or {}
        event_type = meta.get("event_type", "text")

        handler = {
            "tool_start": self._on_tool_start,
            "tool_output": self._on_tool_output,
            "tool_end": self._on_tool_end,
            "thinking_start": self._on_thinking_start,
            "thinking_end": self._on_thinking_end,
            "text": self._on_text,
            "final": self._on_final,
        }.get(event_type)

        if handler:
            yield from handler(output, meta)
        else:
            yield from self._on_unknown(output, meta, event_type)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_id(self, name: str) -> str:
        self._synthetic_counter += 1
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "unknown")).strip("_") or "unknown"
        return f"synthetic_tool_{self._synthetic_counter}_{safe}"

    @staticmethod
    def _empty_args(args: Any) -> bool:
        return args in (None, "", {}, [])

    @staticmethod
    def _meaningful(name: Any, args: Any) -> bool:
        if isinstance(name, str) and name.strip() and name.strip().lower() != "unknown":
            return True
        return args not in (None, "", {}, [])

    def _remember(self, tc_id: str, name: Any, args: Any) -> None:
        if not tc_id:
            return
        cur = self._calls.get(tc_id, {})
        merged_name = (
            name
            if isinstance(name, str) and name.strip() and name.strip().lower() != "unknown"
            else cur.get("tool_name", "unknown")
        )
        merged_args = args if not self._empty_args(args) else cur.get("tool_args", {})
        self._calls[tc_id] = {
            "tool_name": merged_name or "unknown",
            "tool_args": merged_args,
        }

    def _truncate(self, text: str) -> tuple:
        """Return (display_text, truncated_bool, full_length_or_None)."""
        if self._max_chars and len(text) > self._max_chars:
            return text[: self._max_chars - 3].rstrip() + "...", True, len(text)
        return text, False, None

    # ------------------------------------------------------------------
    # Raw arg extraction from message additional_kwargs / tool_call_chunks
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_raw_tool_info(msg) -> tuple:
        """Parse additional_kwargs.tool_calls and tool_call_chunks.

        Returns (raw_args_by_id, raw_names_by_id) dicts.
        """
        raw_args: Dict[str, Any] = {}
        raw_names: Dict[str, str] = {}
        if msg is None:
            return raw_args, raw_names

        try:
            additional = getattr(msg, "additional_kwargs", {}) or {}
            for raw_call in additional.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                rid = raw_call.get("id")
                fn = raw_call.get("function") or {}
                rname = fn.get("name")
                rargs = fn.get("arguments")
                parsed = _try_parse_args(rargs)
                if rid and parsed is not None:
                    raw_args[rid] = parsed
                if rid and isinstance(rname, str) and rname.strip():
                    raw_names[rid] = rname.strip()

            for chunk in getattr(msg, "tool_call_chunks", []) or []:
                if not isinstance(chunk, dict):
                    continue
                cid = chunk.get("id")
                cname = chunk.get("name")
                cargs = chunk.get("args")
                parsed = _try_parse_args(cargs)
                if cid and parsed is not None:
                    raw_args[cid] = parsed
                if cid and isinstance(cname, str) and cname.strip():
                    raw_names[cid] = cname.strip()
        except Exception:
            pass

        return raw_args, raw_names

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_tool_start(self, output: PipelineOutput, meta: dict) -> Iterator[Dict[str, Any]]:
        """Parse and remember tool calls; actual emission is deferred."""
        msg = (output.messages or [None])[0]
        tool_calls = getattr(msg, "tool_calls", None) if msg else None
        memory = meta.get("tool_inputs_by_id", {}) or {}
        raw_args, raw_names = self._extract_raw_tool_info(msg)

        if tool_calls:
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                args = tc.get("args", {})
                if self._empty_args(args):
                    args = raw_args.get(tc_id, args)
                if self._empty_args(args):
                    fb = memory.get(tc_id, {})
                    if isinstance(fb, dict):
                        args = fb.get("tool_input", args)
                name = tc.get("name", "unknown")
                if (not name or str(name).strip().lower() == "unknown") and tc_id in raw_names:
                    name = raw_names[tc_id]
                if not name and isinstance(memory.get(tc_id), dict):
                    name = memory[tc_id].get("tool_name", "unknown")
                if not tc_id and not self._meaningful(name, args):
                    continue
                if not tc_id:
                    tc_id = self._next_id(name)
                self._remember(tc_id, name, args)
                if tc_id in self._emitted_ids:
                    continue
                self._emitted_ids.add(tc_id)
                self._pending_ids.append(tc_id)
                self.tool_call_count += 1
        elif memory:
            for mid, mc in memory.items():
                if not isinstance(mc, dict):
                    continue
                name = mc.get("tool_name", "unknown")
                args = mc.get("tool_input", {})
                if not self._meaningful(name, args):
                    continue
                tc_id = mid or self._next_id(name)
                if tc_id in self._emitted_ids:
                    continue
                self._emitted_ids.add(tc_id)
                self._pending_ids.append(tc_id)
                self._remember(tc_id, name, args)
                self.tool_call_count += 1

        # Deferred – don't yield anything here
        return
        yield  # type: ignore[misc]  # makes this a generator function

    def _on_tool_output(self, output: PipelineOutput, meta: dict) -> Iterator[Dict[str, Any]]:
        """Emit deferred tool_start (if needed) then tool_output."""
        msg = (output.messages or [None])[0]
        tool_output = self._message_content(msg) if msg else ""
        tc_id = getattr(msg, "tool_call_id", "") if msg else ""

        if not tc_id and self._pending_ids:
            tc_id = self._pending_ids.pop(0)
        elif tc_id in self._pending_ids:
            self._pending_ids.remove(tc_id)

        # Emit deferred tool_start if not yet sent
        if tc_id and tc_id not in self._emitted_start_ids:
            memory = meta.get("tool_inputs_by_id", {}) or {}
            fb = memory.get(tc_id, {})
            fb_name: str = "unknown"
            fb_args: Any = {}
            if isinstance(fb, dict):
                fb_name = fb.get("tool_name", "unknown")
                fb_args = fb.get("tool_input", {})
            self._remember(tc_id, fb_name, fb_args)
            info = self._calls.get(tc_id, {})
            self._emitted_start_ids.add(tc_id)
            yield {
                "type": "tool_start",
                "tool_call_id": tc_id,
                "tool_name": info.get("tool_name", "unknown"),
                "tool_args": info.get("tool_args", {}),
            }

        display, truncated, full_length = self._truncate(tool_output)
        evt: Dict[str, Any] = {
            "type": "tool_output",
            "tool_call_id": tc_id,
            "output": display,
            "truncated": truncated,
        }
        if full_length is not None:
            evt["full_length"] = full_length
        yield evt

    def _on_tool_end(self, _output: PipelineOutput, meta: dict) -> Iterator[Dict[str, Any]]:
        yield {
            "type": "tool_end",
            "tool_call_id": meta.get("tool_call_id", ""),
            "status": meta.get("status", "success"),
            "duration_ms": meta.get("duration_ms"),
        }

    def _on_thinking_start(self, _output: PipelineOutput, meta: dict) -> Iterator[Dict[str, Any]]:
        yield {
            "type": "thinking_start",
            "step_id": meta.get("step_id", ""),
        }

    def _on_thinking_end(self, _output: PipelineOutput, meta: dict) -> Iterator[Dict[str, Any]]:
        yield {
            "type": "thinking_end",
            "step_id": meta.get("step_id", ""),
            "duration_ms": meta.get("duration_ms"),
            "thinking_content": meta.get("thinking_content", ""),
        }

    def _on_text(self, output: PipelineOutput, _meta: dict) -> Iterator[Dict[str, Any]]:
        content = output.answer or ""
        if content:
            self.last_text = content
            yield {
                "type": "text",
                "content": content,
            }

    def _on_final(self, _output: PipelineOutput, _meta: dict) -> Iterator[Dict[str, Any]]:
        # Callers handle finalization themselves
        return
        yield  # type: ignore[misc]

    def _on_unknown(self, output: PipelineOutput, _meta: dict, event_type: str) -> Iterator[Dict[str, Any]]:
        """Fallback for unrecognised event types."""
        if getattr(output, "final", False):
            return
        content = output.answer or ""
        if content:
            yield {
                "type": event_type,
                "content": content,
            }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _try_parse_args(raw: Any) -> Any:
    """Attempt to parse raw tool arguments into a dict."""
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw_arguments": raw}
    elif isinstance(raw, dict):
        return raw
    return None
