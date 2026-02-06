from typing import Any, Callable, Dict, List, Optional, Sequence, Iterator, AsyncIterator, Set, Tuple
import time

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage
try:
    from langchain_core.messages import BaseMessageChunk
except ImportError:
    BaseMessageChunk = None
from langgraph.graph.state import CompiledStateGraph

from src.archi.pipelines.agents.utils.prompt_utils import read_prompt
from src.archi.providers import get_model
from src.archi.utils.output_dataclass import PipelineOutput
from src.archi.pipelines.agents.utils.document_memory import DocumentMemory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseReActAgent:
    """
    BaseReActAgent provides a foundational structure for building pipeline classes that
    process user queries using configurable language models and prompts.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *args,
        agent_spec: Optional[Any] = None,
        default_provider: Optional[str] = None,
        default_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.config = config
        self.archi_config = self.config.get("archi") or {}
        self.dm_config = self.config.get("data_manager", {})
        pipeline_map = self.archi_config.get("pipeline_map", {}) if isinstance(self.archi_config, dict) else {}
        self.pipeline_config = pipeline_map.get(self.__class__.__name__, {}) if isinstance(pipeline_map, dict) else {}
        self.agent_spec = agent_spec
        self.default_provider = default_provider
        self.default_model = default_model
        self.selected_tool_names: List[str] = []
        if agent_spec is not None:
            self.selected_tool_names = list(getattr(agent_spec, "tools", []) or [])
        self._active_memory: Optional[DocumentMemory] = None
        self._static_tools: Optional[List[Callable]] = None
        self._active_tools: List[Callable] = []
        self._static_middleware: Optional[List[Callable]] = None
        self._active_middleware: List[Callable] = []
        self.agent: Optional[CompiledStateGraph] = None
        self.agent_llm: Optional[Any] = None
        self.agent_prompt: Optional[str] = None

        self._init_llms()
        self._init_prompts()

        if self.agent_llm is None:
            if not self.llms:
                raise ValueError(f"No LLMs configured for agent {self.__class__.__name__}")
            self.agent_llm = self.llms.get("chat_model") or next(iter(self.llms.values()))
        if self.agent_prompt is None:
            self.agent_prompt = self.prompts.get("agent_prompt")

    def create_document_memory(self) -> DocumentMemory:
        """Instantiate a fresh document memory for an agent run."""
        return DocumentMemory()

    def start_run_memory(self) -> DocumentMemory:
        """Create and store the active memory for the current run."""
        memory = self.create_document_memory()
        self._active_memory = memory
        return memory

    @property
    def active_memory(self) -> Optional[DocumentMemory]:
        """Return the memory currently associated with the run, if any."""
        return self._active_memory

    def finalize_output(
        self,
        *,
        answer: str,
        memory: Optional[DocumentMemory] = None,
        messages: Optional[Sequence[BaseMessage]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        final: bool = True,
    ) -> PipelineOutput:
        """
        Compose a PipelineOutput from the provided components.
        
        If not final, drop documents and only keep the latest message.
        """
        documents = memory.unique_documents() if (memory and final) else []
        resolved_messages: List[BaseMessage] = []
        if messages:
            if isinstance(messages, (list, tuple)):
                resolved_messages = list(messages) if final else [messages[-1]]
            else:
                resolved_messages = [messages]
        return PipelineOutput(
            answer=answer,
            source_documents=documents,
            messages=resolved_messages,
            metadata=metadata or {},
            final=final,
        )

    def invoke(self, **kwargs) -> PipelineOutput:
        """Synchronously invoke the agent graph and return the final output."""
        logger.debug("Invoking %s", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)
        logger.debug("Agent refreshed, invoking now")
        answer_output = self.agent.invoke(agent_inputs, {"recursion_limit": 50})
        logger.debug("Agent invocation completed")
        logger.debug(answer_output)
        messages = self._extract_messages(answer_output)
        metadata = self._metadata_from_agent_output(answer_output)
        output = self._build_output_from_messages(messages, metadata=metadata)
        return output

    def stream(self, **kwargs) -> Iterator[PipelineOutput]:
        """Stream agent updates synchronously with structured trace events."""
        logger.debug("Streaming %s", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)

        all_messages: List[BaseMessage] = []  # Accumulated full messages
        accumulated_content = ""  # Accumulated content from streaming
        emitted_tool_starts: Set[str] = set()
        
        for event in self.agent.stream(agent_inputs, stream_mode="messages"):
            
            messages = self._extract_messages(event)
            if not messages:
                continue
            
            message = messages[-1]
            msg_type = str(getattr(message, "type", "")).lower()
            msg_class = type(message).__name__.lower()
            
            # Track all non-chunk messages
            if "chunk" not in msg_class:
                all_messages.extend(messages)
            
            # Detect tool call start (AIMessage with tool_calls)
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.debug("Received stream event type=%s: %s", type(event).__name__, str(event)[:1000])
                new_tool_call = False
                for tc in message.tool_calls:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in emitted_tool_starts:
                        emitted_tool_starts.add(tc_id)
                        new_tool_call = True
                if new_tool_call:
                    yield self.finalize_output(
                        answer="",
                        memory=self.active_memory,
                        messages=[message],
                        metadata={"event_type": "tool_start"},
                        final=False,
                    )
            
            # Detect tool result (ToolMessage with tool_call_id)
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id:
                logger.debug("Received stream event type=%s: %s", type(event).__name__, str(event)[:1000])
                yield self.finalize_output(
                    answer="",
                    memory=self.active_memory,
                    messages=[message],
                    metadata={
                        "event_type": "tool_output",
                    },
                    final=False,
                )
            
            # AI content streaming - accumulate content from chunks
            if msg_type in {"ai", "assistant"} or "ai" in msg_class:
                if not getattr(message, "tool_calls", None):
                    content = self._message_content(message)
                    if content:
                        # For chunks, content is delta; for full messages, content is cumulative
                        if "chunk" in msg_class:
                            accumulated_content += content
                        else:
                            # Full message - use its content directly
                            accumulated_content = content
                        
                        yield self.finalize_output(
                            answer=accumulated_content,
                            memory=self.active_memory,
                            messages=[message],
                            metadata={"event_type": "text"},
                            final=False,
                        )
        
        # Final output
        logger.debug("Stream finished. accumulated_content='%s', all_messages count=%d", 
                     accumulated_content[:100] if accumulated_content else "", len(all_messages))
        
        final_answer = ""
        if all_messages:
            # Find the last AI message with content
            for msg in reversed(all_messages):
                msg_type = str(getattr(msg, "type", "")).lower()
                if msg_type in {"ai", "assistant"} or "ai" in type(msg).__name__.lower():
                    content = self._message_content(msg)
                    if content:
                        final_answer = content
                        logger.debug("Found final answer from AI message: %s", content[:100])
                        break
        if not final_answer:
            final_answer = accumulated_content
        
        if final_answer:
            yield self.finalize_output(
                answer=final_answer,
                memory=self.active_memory,
                messages=all_messages,
                metadata={"event_type": "final"},
                final=True,
            )
        else:
            logger.warning("No final answer found from stream. Messages: %s", 
                          [self._format_message(m) for m in all_messages[:5]])
            output = self._build_output_from_messages(all_messages)
            output.metadata["event_type"] = "final"
            yield output

    async def astream(self, **kwargs) -> AsyncIterator[PipelineOutput]:
        """Stream agent updates asynchronously with structured trace events."""
        logger.debug("Streaming %s asynchronously", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)

        all_messages: List[BaseMessage] = []
        accumulated_content = ""
        emitted_tool_starts: Set[str] = set()
        
        async for event in self.agent.astream(agent_inputs, stream_mode="messages"):
            messages = self._extract_messages(event)
            if not messages:
                continue
            
            message = messages[-1]
            msg_type = str(getattr(message, "type", "")).lower()
            msg_class = type(message).__name__.lower()
            
            # Track all non-chunk messages
            if "chunk" not in msg_class:
                all_messages.extend(messages)
            
            # Detect tool call start
            if hasattr(message, "tool_calls") and message.tool_calls:
                new_tool_call = False
                for tc in message.tool_calls:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in emitted_tool_starts:
                        emitted_tool_starts.add(tc_id)
                        new_tool_call = True
                if new_tool_call:
                    yield self.finalize_output(
                        answer="",
                        messages=[message],
                        metadata={"event_type": "tool_start"},
                        final=False,
                    )
            
            # Detect tool result
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id:
                yield self.finalize_output(
                    answer="",
                    messages=[message],
                    metadata={
                        "event_type": "tool_output",
                    },
                    final=False,
                )
            
            # AI content streaming - accumulate content from chunks
            if msg_type in {"ai", "assistant"} or "ai" in msg_class:
                if not getattr(message, "tool_calls", None):
                    content = self._message_content(message)
                    if content:
                        if "chunk" in msg_class:
                            accumulated_content += content
                        else:
                            accumulated_content = content
                        
                        yield self.finalize_output(
                            answer=accumulated_content,
                            messages=[message],
                            metadata={"event_type": "text"},
                            final=False,
                        )
        
        # Final output
        logger.debug("Async stream finished. accumulated_content='%s', all_messages count=%d", 
                     accumulated_content[:100] if accumulated_content else "", len(all_messages))
        
        final_answer = ""
        if all_messages:
            for msg in reversed(all_messages):
                msg_type = str(getattr(msg, "type", "")).lower()
                if msg_type in {"ai", "assistant"} or "ai" in type(msg).__name__.lower():
                    content = self._message_content(msg)
                    if content:
                        final_answer = content
                        logger.debug("Found final answer from AI message: %s", content[:100])
                        break
        if not final_answer:
            final_answer = accumulated_content
        
        if final_answer:
            yield self.finalize_output(
                answer=final_answer,
                memory=self.active_memory,
                messages=all_messages,
                metadata={"event_type": "final"},
                final=True,
            )
        else:
            logger.warning("No final answer found from async stream. Messages: %s", 
                          [self._format_message(m) for m in all_messages[:5]])
            output = self._build_output_from_messages(all_messages)
            output.metadata["event_type"] = "final"
            yield output

    def _init_llms(self) -> None:
        """Initialise language models for the agent."""

        self.llms: Dict[str, Any] = {}

        if self.default_provider and self.default_model:
            instance = get_model(self.default_provider, self.default_model)
            self.llms["chat_model"] = instance
            self.agent_llm = instance
            return

        models_config = self.pipeline_config.get("models", {}) if isinstance(self.pipeline_config, dict) else {}
        all_models = dict(models_config.get("required", {}), **models_config.get("optional", {}))
        initialised_models: Dict[str, Any] = {}

        for model_name, model_class_name in all_models.items():
            if model_class_name in initialised_models:
                self.llms[model_name] = initialised_models[model_class_name]
                logger.debug(
                    "Reusing initialised model '%s' of class '%s'",
                    model_name,
                    model_class_name,
                )
                continue

            provider, model_id = self._parse_provider_model(model_class_name)
            instance = get_model(provider, model_id)
            self.llms[model_name] = instance
            initialised_models[model_class_name] = instance

    @staticmethod
    def _parse_provider_model(model_ref: str) -> Tuple[str, str]:
        """Expect model_ref as 'provider/model'. Raise if malformed."""
        if not isinstance(model_ref, str) or "/" not in model_ref:
            raise ValueError(f"Model reference must be 'provider/model', got '{model_ref}'")
        provider, model_id = model_ref.split("/", 1)
        if not provider or not model_id:
            raise ValueError(f"Invalid model reference '{model_ref}'")
        return provider, model_id

    def _init_prompts(self) -> None:
        """Initialise prompts defined in pipeline configuration or agent spec."""

        if self.agent_spec is not None:
            self.prompts = {}
            self.agent_prompt = getattr(self.agent_spec, "prompt", None)
            return

        prompts_config = self.pipeline_config.get("prompts", {}) if isinstance(self.pipeline_config, dict) else {}
        required = prompts_config.get("required", {})
        optional = prompts_config.get("optional", {})
        all_prompts = {**optional, **required}

        self.prompts: Dict[str, SystemMessage] = {}
        for name, path in all_prompts.items():
            if not path:
                continue
            try:
                prompt_template = read_prompt(path)
            except FileNotFoundError as exc:
                if name in required:
                    raise FileNotFoundError(
                        f"Required prompt file '{path}' for '{name}' not found: {exc}"
                    ) from exc
                logger.warning(
                    "Optional prompt file '%s' for '%s' not found or unreadable: %s",
                    path,
                    name,
                    exc,
                )
                continue
            self.prompts[name] = str(prompt_template) # TODO at some point, make a validated prompt class to check these?

    def get_tool_registry(self) -> Dict[str, Callable[[], Any]]:
        """Return a mapping of tool names to callables that build tools."""
        return {}

    def _select_tools_from_registry(self, tool_names: Sequence[str]) -> List[Callable]:
        registry = self.get_tool_registry() or {}
        if not tool_names:
            return []
        tools: List[Callable] = []
        for name in tool_names:
            builder = registry.get(name)
            if not builder:
                logger.warning("Tool '%s' not found in registry for %s", name, self.__class__.__name__)
                continue
            built = builder()
            if isinstance(built, (list, tuple)):
                tools.extend(list(built))
            else:
                tools.append(built)
        return tools

    def rebuild_static_tools(self) -> List[Callable]:
        """Recompute and cache the static tool list."""
        self._static_tools = list(self._build_static_tools())
        return self._static_tools

    @property
    def tools(self) -> List[Callable]:
        """Return the cached static tools, rebuilding if necessary."""
        if self._static_tools is None:
            return self.rebuild_static_tools()
        return list(self._static_tools)
    
    def rebuild_static_middleware(self) -> List[Callable]:
        """Recompute and cache the static middleware list."""
        self._static_middleware = list(self._build_static_middleware())
        return self._static_middleware
    
    @property
    def middleware(self) -> List[Callable]:
        """Return the cached static middleware, rebuilding if necessary."""
        if self._static_middleware is None:
            return self.rebuild_static_middleware()
        return list(self._static_middleware)

    @tools.setter
    def tools(self, value: Sequence[Callable]) -> None:
        """Explicitly set the static tools cache."""
        self._static_tools = list(value)

    def refresh_agent(
        self,
        *,
        static_tools: Optional[Sequence[Callable]] = None,
        extra_tools: Optional[Sequence[Callable]] = None,
        middleware: Optional[Sequence[Callable]] = None,
        force: bool = False,
    ) -> CompiledStateGraph:
        """Ensure the LangGraph agent reflects the latest tool set."""
        base_tools = list(static_tools) if static_tools is not None else self.tools
        toolset: List[Callable] = list(base_tools)
        if extra_tools:
            toolset.extend(extra_tools)
       
        middleware = list(middleware) if middleware is not None else self.middleware

        requires_refresh = (
            force
            or self.agent is None
            or len(toolset) != len(self._active_tools)
            or any(a is not b for a, b in zip(toolset, self._active_tools))
        )
        if requires_refresh:
            logger.debug("Refreshing agent %s", self.__class__.__name__)
            self.agent = self._create_agent(toolset, middleware)
            self._active_tools = list(toolset)
            self._active_middleware = list(middleware)
        return self.agent

    def _create_agent(self, tools: Sequence[Callable], middleware: Sequence[Callable]) -> CompiledStateGraph:
        """Create the LangGraph agent with the specified LLM, tools, and system prompt."""
        logger.debug("Creating agent %s with:", self.__class__.__name__)
        logger.debug("%d tools", len(tools))
        logger.debug("%d middleware components", len(middleware))
        return create_agent(
            model=self.agent_llm,
            tools=tools,
            middleware=middleware,
            system_prompt=self.agent_prompt,
        )

    def _build_static_tools(self) -> List[Callable]:
        """Build static tools from the agent's tool registry."""
        return self._select_tools_from_registry(self.selected_tool_names)
    
    def _build_static_middleware(self) -> List[Callable]:
        """Build and returns static middleware defined in the config."""
        return []

    def _prepare_agent_inputs(self, **kwargs) -> Dict[str, Any]:
        """Subclasses must implement to provide agent input payloads."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _prepare_agent_inputs")

    def _metadata_from_agent_output(self, answer_output: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses to enrich metadata returned to callers."""
        return {}

    def _extract_messages(self, payload: Any) -> List[BaseMessage]:
        """Pull LangChain messages from a stream/update payload."""
        message_types = (BaseMessage,)
        if BaseMessageChunk is not None:
            message_types = (BaseMessage, BaseMessageChunk)

        if isinstance(payload, message_types):
            return [payload]
        if isinstance(payload, list) and all(isinstance(msg, message_types) for msg in payload):
            return list(payload)
        if isinstance(payload, tuple) and payload and isinstance(payload[0], message_types):
            return [payload[0]]
        def _messages_from_container(container: Any) -> List[BaseMessage]:
            if isinstance(container, dict):
                messages = container.get("messages")
                if isinstance(messages, list) and all(isinstance(msg, BaseMessage) for msg in messages):
                    return messages
            return []

        direct = _messages_from_container(payload)
        if direct:
            return direct
        if isinstance(payload, dict):
            for value in payload.values():
                nested = _messages_from_container(value)
                if nested:
                    return nested
        return []

    def _message_content(self, message: BaseMessage) -> str:
        """Normalise message content to a printable string."""
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        return str(content)

    def _format_message(self, message: BaseMessage) -> str:
        """Condense a message for logging/metadata storage."""
        role = getattr(message, "type", message.__class__.__name__)
        content = self._message_content(message)
        if len(content) > 400:
            content = f"{content[:397]}..."
        return f"{role}: {content}"

    def _build_output_from_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        final: bool = True,
    ) -> PipelineOutput:
        """Create a PipelineOutput from the agent's message history."""
        if messages:
            answer_text = self._message_content(messages[-1]) or "No answer generated by the agent."
        else:
            answer_text = "No answer generated by the agent."
        safe_metadata = dict(metadata or {})
        return self.finalize_output(
            answer=answer_text,
            memory=self.active_memory,
            messages=messages,
            metadata=safe_metadata,
            final=final,
        )
