# isort: skip_file
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage

from .constants import (  # isort: skip
    COMPARATOR_SYSTEM_PROMPT,
    GOLD_SYSTEM_PROMPT,
)
from .profile import EvaluatorProfile
from .tool_traces import ToolCallRecord, ToolCallStatus
from .validation import Atom

GOLD_ATOM_SCHEMA = {
    "title": "atoms",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "atoms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "required": {"type": "boolean"},
                },
                "required": ["id", "text", "required"],
            },
        }
    },
    "required": ["atoms"],
}

JUDGMENT_SCHEMA = {
    "title": "judgments",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "atom_id": {"type": "string"},
                    "outcome": {
                        "type": "string",
                        "enum": [
                            "entailed",
                            "not_mentioned",
                            "contradicted",
                            "unjudgeable",
                        ],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["atom_id", "outcome", "rationale"],
            },
        }
    },
    "required": ["judgments"],
}


@dataclass(frozen=True)
class _ActiveToolCall:
    ordinal: int
    name: str
    query: str
    started_at: float


def _trace_text(value: Any) -> str:
    if isinstance(value, BaseMessage):
        value = value.content
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


class ToolTimingCallback(BaseCallbackHandler):
    """Collect the complete observed tool trace for one agent attempt."""

    run_inline = True

    def __init__(self) -> None:
        self._active: Dict[UUID, _ActiveToolCall] = {}
        self._next_ordinal = 1
        self._completed: List[ToolCallRecord] = []
        self._lock = Lock()

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        started_at = perf_counter()
        with self._lock:
            ordinal = self._next_ordinal
            self._next_ordinal += 1
        active = _ActiveToolCall(
            ordinal=ordinal,
            name=serialized["name"],
            query=_trace_text(inputs if inputs is not None else input_str),
            started_at=started_at,
        )
        with self._lock:
            self._active[run_id] = active

    def _finish(
        self,
        run_id: UUID,
        status: ToolCallStatus,
        output: Any,
    ) -> None:
        ended_at = perf_counter()
        text = _trace_text(output)
        with self._lock:
            active = self._active.pop(run_id)
            self._completed.append(
                ToolCallRecord(
                    ordinal=active.ordinal,
                    name=active.name,
                    status=status,
                    query=active.query,
                    response=text if status == ToolCallStatus.SUCCESS else None,
                    error=text if status == ToolCallStatus.ERROR else None,
                    duration_ms=max(
                        0,
                        int(round((ended_at - active.started_at) * 1000)),
                    ),
                )
            )

    @property
    def traces(self) -> List[Dict[str, Any]]:
        with self._lock:
            traces = list(self._completed)
            traces.extend(
                ToolCallRecord(
                    ordinal=active.ordinal,
                    name=active.name,
                    status=ToolCallStatus.INCOMPLETE,
                    query=active.query,
                )
                for active in self._active.values()
            )
        return [
            trace.to_dict() for trace in sorted(traces, key=lambda item: item.ordinal)
        ]

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        status = (
            ToolCallStatus.ERROR
            if isinstance(output, ToolMessage) and output.status == "error"
            else ToolCallStatus.SUCCESS
        )
        self._finish(run_id, status, output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, ToolCallStatus.ERROR, error)


class LangChainEvaluatorRuntime:
    def __init__(
        self,
        profile: EvaluatorProfile,
        model_factory: Optional[Callable[..., Any]] = None,
    ):
        if model_factory is None:
            from src.archi.providers import get_model

            model_factory = get_model
        self._models = {
            component: model_factory(
                descriptor.provider,
                descriptor.model,
                {},
                **descriptor.provider_kwargs(),
            )
            for component, descriptor in profile.components()
        }

    @staticmethod
    def _structured(
        model: Any, schema: Dict[str, Any], prompt: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        structured = model.with_structured_output(schema)
        result = structured.invoke(
            [
                ("system", prompt),
                ("human", json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            ]
        )
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        if not isinstance(result, dict):
            raise ValueError("structured evaluator returned a non-object")
        return result

    def extract_gold(self, question: str, answer: str) -> Dict[str, Any]:
        return self._structured(
            self._models["atoms_extractor"],
            GOLD_ATOM_SCHEMA,
            GOLD_SYSTEM_PROMPT,
            {"question": question, "answer": answer},
        )

    def compare(
        self,
        question: str,
        gold_atoms: Sequence[Atom],
        answer: str,
    ) -> Dict[str, Any]:
        return self._structured(
            self._models["evaluator"],
            JUDGMENT_SCHEMA,
            COMPARATOR_SYSTEM_PROMPT,
            {
                "question": question,
                "gold_atoms": [atom.to_dict() for atom in gold_atoms],
                "answer": answer,
            },
        )


def _validate_local_file(path: Path, suffixes: set, label: str) -> Path:
    raw = str(path)
    if raw == "-" or "://" in raw:
        raise ValueError(f"{label} must be a local file path")
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{label} must be an existing file: {path}")
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"{label} must use one of: {', '.join(sorted(suffixes))}")
    return resolved


def load_agent_inputs(
    config_path: Path, spec_path: Path
) -> Tuple[Dict[str, Any], Any, str, type]:
    from src.archi.pipelines.agents.agent_spec import (
        AgentSpecError,
        load_agent_spec_from_text,
    )

    resolved_config_path = _validate_local_file(
        config_path, {".yaml", ".yml"}, "agent config"
    )
    resolved_spec_path = _validate_local_file(spec_path, {".md"}, "agent spec")
    try:
        config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid agent config YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("agent config must be an object")
    services = config.get("services")
    chat = services.get("chat_app") if isinstance(services, dict) else None
    if not isinstance(chat, dict):
        raise ValueError("agent config requires services.chat_app")
    required = ("agent_class", "default_provider", "default_model")
    missing = [
        name
        for name in required
        if not isinstance(chat.get(name), str) or not chat[name].strip()
    ]
    if missing:
        raise ValueError(
            "agent config services.chat_app is missing non-empty field(s): "
            + ", ".join(missing)
        )
    spec_text = resolved_spec_path.read_text(encoding="utf-8")
    try:
        spec = replace(
            load_agent_spec_from_text(spec_text), source_path=resolved_spec_path
        )
    except (AgentSpecError, OSError) as exc:
        raise ValueError(f"invalid agent spec: {exc}") from exc
    pipelines = import_module("src.archi.pipelines")
    try:
        pipeline_class = getattr(pipelines, chat["agent_class"])
    except AttributeError as exc:
        raise ValueError(f"unknown agent class '{chat['agent_class']}'") from exc
    return config, spec, spec_text, pipeline_class


class LazyVectorstore:
    """Thread-safe, lazy vector-store cache owned by one workflow phase."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._vectorstore: Optional[Any] = None
        self._lock = Lock()

    def _load(self) -> Any:
        from src.archi.utils.vectorstore_connector import VectorstoreConnector

        return VectorstoreConnector(self._config).get_vectorstore()

    def get(self) -> Any:
        with self._lock:
            if self._vectorstore is None:
                self._vectorstore = self._load()
            return self._vectorstore


# TODO: Remove this evaluation-specific runtime once the generic `archi`
# runtime is refactored to initialize vector-store connections and other tool
# dependencies only when they are selected by the resolved agent config/spec.
# Until then, this adapter avoids creating dependencies that QA attempts do not
# need.
class ArchiAgentRuntime:
    def __init__(
        self,
        config: Dict[str, Any],
        spec: Any,
        pipeline_class: type,
        vectorstore: Optional[LazyVectorstore] = None,
    ):
        self.config = config
        self.spec = spec
        self.pipeline_class = pipeline_class
        self.tool_calls: List[Dict[str, Any]] = []
        self._pipeline: Optional[Any] = None
        self._vectorstore: Optional[Any] = None
        self._selected_tool_names = set(getattr(self.spec, "tools", []) or [])
        self._shared_vectorstore = vectorstore
        if (
            "search_vectorstore_hybrid" in self._selected_tool_names
            and self._shared_vectorstore is None
        ):
            raise ValueError("vector-search runtime requires a shared vector store")

    def _runtime_for_attempt(self) -> Tuple[Any, Optional[Any]]:
        if self._pipeline is not None:
            return self._pipeline, self._vectorstore

        chat = self.config["services"]["chat_app"]
        if "search_vectorstore_hybrid" in self._selected_tool_names:
            assert self._shared_vectorstore is not None
            vectorstore = self._shared_vectorstore.get()
        else:
            vectorstore = None
        pipeline = self.pipeline_class(
            config=deepcopy(self.config),
            agent_spec=deepcopy(self.spec),
            default_provider=chat["default_provider"],
            default_model=chat["default_model"],
        )
        if "mcp" in self._selected_tool_names and not pipeline.loaded_mcp_tools:
            raise RuntimeError(
                "agent spec selected 'mcp', but no MCP tools were loaded"
            )

        # Cache only a completely initialized runtime. A failed initialization
        # remains retryable by the next independently accounted attempt.
        self._pipeline = pipeline
        self._vectorstore = vectorstore
        return pipeline, vectorstore

    def run(self, question: str) -> str:
        self.tool_calls = []
        timing_callback = ToolTimingCallback()
        pipeline, vectorstore = self._runtime_for_attempt()
        try:
            output = pipeline.invoke(
                history=[("User", question)],
                vectorstore=vectorstore,
                callbacks=[timing_callback],
            )
        finally:
            self.tool_calls = sorted(
                timing_callback.traces,
                key=lambda timing: timing["ordinal"],
            )
        answer = output.answer
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Archi produced no usable terminal answer")
        return answer
