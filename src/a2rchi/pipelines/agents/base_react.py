from typing import Any, Callable, Dict, List, Optional, Sequence, Iterator, AsyncIterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.a2rchi.pipelines.agents.utils.prompt_utils import read_prompt
from src.a2rchi.utils.output_dataclass import PipelineOutput
from src.a2rchi.pipelines.agents.utils.document_memory import DocumentMemory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseReActAgent:
    """
    BaseReActAgent provides a foundational structure for building pipeline classes that
    process user queries using configurable language models and prompts.
    """

    DEFAULT_RECURSION_LIMIT = 100

    def __init__(
        self,
        config: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        self.config = config
        self.a2rchi_config = self.config["a2rchi"]
        self.dm_config = self.config["data_manager"]
        self.pipeline_config = self.a2rchi_config["pipeline_map"][self.__class__.__name__]
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
        tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
        final: bool = True,
    ) -> PipelineOutput:
        """Compose a PipelineOutput from the provided components."""
        documents = memory.unique_documents() if memory else []
        resolved_messages = list(messages or [])
        resolved_tool_calls = (
            list(tool_calls) if tool_calls is not None else self._extract_tool_calls(resolved_messages)
        )
        return PipelineOutput(
            answer=answer,
            source_documents=documents,
            messages=resolved_messages,
            metadata=metadata or {},
            final=final,
            tool_calls=resolved_tool_calls,
        )

    def invoke(self, **kwargs) -> PipelineOutput:
        """Synchronously invoke the agent graph and return the final output."""
        logger.debug("Invoking %s", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)
        logger.debug("Agent refreshed, invoking now")
        recursion_limit = self._recursion_limit()
        try:
            answer_output = self.agent.invoke(agent_inputs, {"recursion_limit": recursion_limit})
            logger.debug("Agent invocation completed")
            logger.debug(answer_output)
            messages = self._extract_messages(answer_output)
            metadata = self._metadata_from_agent_output(answer_output)
            output = self._build_output_from_messages(messages, metadata=metadata)
            return output
        except GraphRecursionError as exc:
            logger.warning(
                "Recursion limit hit for %s (limit=%s): %s",
                self.__class__.__name__,
                recursion_limit,
                exc,
            )
            return self._handle_recursion_limit_error(
                error=exc,
                recursion_limit=recursion_limit,
                latest_messages=[],
                agent_inputs=agent_inputs,
            )

    def stream(self, **kwargs) -> Iterator[PipelineOutput]:
        """Stream agent updates synchronously."""
        logger.debug("Streaming %s", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)

        latest_messages: List[BaseMessage] = []
        recursion_limit = self._recursion_limit()
        try:
            for event in self.agent.stream(agent_inputs, stream_mode="updates", config={"recursion_limit": recursion_limit}):
                logger.debug("Received stream event: %s", event)
                messages = self._extract_messages(event)
                if messages:
                    latest_messages = messages
                    content = self._message_content(messages[-1])
                    tool_calls = self._extract_tool_calls(messages)
                    yield self.finalize_output(
                        answer=content,
                        memory=self.active_memory,
                        messages=messages,
                        metadata={},
                        tool_calls=tool_calls,
                        final=False,
                    )
        except GraphRecursionError as exc:
            logger.warning(
                "Recursion limit hit during stream for %s (limit=%s): %s",
                self.__class__.__name__,
                recursion_limit,
                exc,
            )
            yield self._handle_recursion_limit_error(
                error=exc,
                recursion_limit=recursion_limit,
                latest_messages=latest_messages,
                agent_inputs=agent_inputs,
            )
            return
        yield self._build_output_from_messages(latest_messages)

    async def astream(self, **kwargs) -> AsyncIterator[PipelineOutput]:
        """Stream agent updates asynchronously."""
        logger.debug("Streaming %s asynchronously", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)

        latest_messages: List[BaseMessage] = []
        recursion_limit = self._recursion_limit()
        try:
            async for event in self.agent.astream(agent_inputs, stream_mode="updates", config={"recursion_limit": recursion_limit}):
                messages = self._extract_messages(event)
                if messages:
                    latest_messages = messages
                    content = self._message_content(messages[-1])
                    if content:
                        yield self.finalize_output(
                            answer=content,
                            memory=self.active_memory,
                            messages=messages,
                            metadata={},
                            final=False,
                        )
        except GraphRecursionError as exc:
            logger.warning(
                "Recursion limit hit during async stream for %s (limit=%s): %s",
                self.__class__.__name__,
                recursion_limit,
                exc,
            )
            yield await self._handle_recursion_limit_error_async(
                error=exc,
                recursion_limit=recursion_limit,
                latest_messages=latest_messages,
                agent_inputs=agent_inputs,
            )
            return
        yield self._build_output_from_messages(latest_messages)

    def _init_llms(self) -> None:
        """Initialise language models declared for the pipeline."""

        model_class_map = self.a2rchi_config["model_class_map"]
        models_config = self.pipeline_config.get("models", {})
        self.llms: Dict[str, Any] = {}

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

            model_entry = model_class_map[model_class_name]
            model_class = model_entry["class"]
            model_kwargs = model_entry["kwargs"]
            instance = model_class(**model_kwargs)
            self.llms[model_name] = instance
            initialised_models[model_class_name] = instance

    def _init_prompts(self) -> None:
        """Initialise prompts defined in pipeline configuration."""

        prompts_config = self.pipeline_config.get("prompts", {})
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

    def rebuild_static_tools(self) -> List[Callable]:
        """Recompute and cache the static tool list."""
        self._static_tools = self._filter_tools(self._build_static_tools())
        return self._static_tools

    @property
    def tools(self) -> List[Callable]:
        """Return the cached static tools, rebuilding if necessary."""
        if self._static_tools is None:
            return self.rebuild_static_tools()
        return list(self._static_tools)
    
    def rebuild_static_middleware(self) -> List[Callable]:
        """Recompute and cache the static middleware list."""
        self._static_middleware = self._filter_middleware(self._build_static_middleware())
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
            toolset.extend(self._filter_tools(extra_tools))
       
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

    def _filter_tools(self, tools: Sequence[Callable]) -> List[Callable]:
        enabled = self.pipeline_config.get("tools", {}).get("enabled")
        if not enabled:
            return list(tools)
        enabled_set = set(enabled)
        include_all = "*" in enabled_set
        filtered: List[Callable] = []
        for tool in tools:
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
            if include_all or tool_name in enabled_set:
                filtered.append(tool)
        return filtered

    def _filter_middleware(self, middleware: Sequence[Callable]) -> List[Callable]:
        enabled = self.pipeline_config.get("middleware", {}).get("enabled")
        if not enabled:
            return list(middleware)
        enabled_set = set(enabled)
        filtered: List[Callable] = []
        for item in middleware:
            name = item.__class__.__name__
            if name in enabled_set:
                filtered.append(item)
        return filtered

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
        """Build and returns static tools defined in the config."""
        return []
    
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

    def _extract_tool_calls(self, messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        tool_results: Dict[str, Any] = {}
        for msg in messages:
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                tool_results[tool_call_id] = getattr(msg, "content", "")
        tool_calls: List[Dict[str, Any]] = []
        for msg in messages:
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                continue
            for call in calls:
                if isinstance(call, dict):
                    entry = dict(call)
                    tool_call_id = entry.get("id")
                else:
                    tool_call_id = getattr(call, "id", None)
                    entry = {
                        "name": getattr(call, "name", None),
                        "args": getattr(call, "args", None),
                        "id": tool_call_id,
                        "type": getattr(call, "type", None),
                    }
                if tool_call_id and tool_call_id in tool_results:
                    entry["result"] = tool_results[tool_call_id]
                tool_calls.append(entry)
        return tool_calls

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

    def _recursion_limit(self) -> int:
        """Read and validate recursion limit from pipeline config."""
        value = self.pipeline_config.get("recursion_limit", self.DEFAULT_RECURSION_LIMIT)
        try:
            limit = int(value)
            if limit <= 0:
                raise ValueError("recursion_limit must be positive")
            logger.info("Using recursion_limit=%s for %s", limit, self.__class__.__name__)
            return limit
        except Exception:
            logger.warning("Invalid recursion_limit '%s' for %s; using default %s", value, self.__class__.__name__, self.DEFAULT_RECURSION_LIMIT)
            return self.DEFAULT_RECURSION_LIMIT

    def _last_user_message_content(self, messages: Sequence[BaseMessage]) -> Optional[str]:
        """Extract content of the most recent user/human message."""
        for msg in reversed(list(messages or [])):
            role = getattr(msg, "type", "").lower()
            if role in ("human", "user"):
                return self._message_content(msg)
        return None

    def _recursion_metadata(self, recursion_limit: int, error: Exception) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "recursion_exhausted": True,
            "recursion_limit": recursion_limit,
            "error": str(error),
        }
        last_node = getattr(error, "node", None) or getattr(error, "step", None)
        if last_node:
            metadata["last_node"] = last_node
        return metadata

    def _handle_recursion_limit_error(
        self,
        *,
        error: Exception,
        recursion_limit: int,
        latest_messages: Sequence[BaseMessage],
        agent_inputs: Optional[Dict[str, Any]] = None,
    ) -> PipelineOutput:
        """Build a best-effort response after recursion exhaustion."""
        metadata = self._recursion_metadata(recursion_limit, error)
        wrap_message = self._generate_wrap_up_message(
            recursion_limit=recursion_limit,
            error=error,
            latest_messages=latest_messages,
            agent_inputs=agent_inputs,
        )
        messages: List[BaseMessage] = list(latest_messages) if latest_messages else []
        if wrap_message:
            messages.append(wrap_message)
        else:
            messages.append(
                AIMessage(
                    content=(
                        f"Recursion limit {recursion_limit} reached. "
                        "No additional summary could be generated."
                    )
                )
            )
        tool_calls = self._extract_tool_calls(messages)
        return self.finalize_output(
            answer=self._message_content(messages[-1]),
            memory=self.active_memory,
            messages=messages,
            metadata=metadata,
            tool_calls=tool_calls,
            final=True,
        )

    async def _handle_recursion_limit_error_async(
        self,
        *,
        error: Exception,
        recursion_limit: int,
        latest_messages: Sequence[BaseMessage],
        agent_inputs: Optional[Dict[str, Any]] = None,
    ) -> PipelineOutput:
        """Async wrapper to build a best-effort response after recursion exhaustion."""
        metadata = self._recursion_metadata(recursion_limit, error)
        wrap_message = await self._generate_wrap_up_message_async(
            recursion_limit=recursion_limit,
            error=error,
            latest_messages=latest_messages,
            agent_inputs=agent_inputs,
        )
        messages: List[BaseMessage] = list(latest_messages) if latest_messages else []
        if wrap_message:
            messages.append(wrap_message)
        else:
            messages.append(
                AIMessage(
                    content=(
                        f"Recursion limit {recursion_limit} reached. "
                        "No additional summary could be generated."
                    )
                )
            )
        tool_calls = self._extract_tool_calls(messages)
        return self.finalize_output(
            answer=self._message_content(messages[-1]),
            memory=self.active_memory,
            messages=messages,
            metadata=metadata,
            tool_calls=tool_calls,
            final=True,
        )

    def _generate_wrap_up_message(
        self,
        *,
        recursion_limit: int,
        error: Exception,
        latest_messages: Sequence[BaseMessage],
        agent_inputs: Optional[Dict[str, Any]],
    ) -> Optional[BaseMessage]:
        """Perform a single LLM-only wrap-up to summarize steps and answer."""
        prompt = self._build_wrap_up_prompt(recursion_limit, error, latest_messages, agent_inputs)
        try:
            response = self.agent_llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Provide the final response now.")])
            if isinstance(response, BaseMessage):
                return response
            return AIMessage(content=str(response))
        except Exception as exc:
            logger.error("Failed to generate wrap-up message after recursion limit: %s", exc)
            return AIMessage(
                content=(
                    f"Recursion limit {recursion_limit} reached and wrap-up generation failed: {exc}"
                )
            )

    async def _generate_wrap_up_message_async(
        self,
        *,
        recursion_limit: int,
        error: Exception,
        latest_messages: Sequence[BaseMessage],
        agent_inputs: Optional[Dict[str, Any]],
    ) -> Optional[BaseMessage]:
        """Async LLM-only wrap-up to summarize steps and answer."""
        prompt = self._build_wrap_up_prompt(recursion_limit, error, latest_messages, agent_inputs)
        try:
            if hasattr(self.agent_llm, "ainvoke"):
                response = await self.agent_llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content="Provide the final response now.")])
            else:
                response = self.agent_llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Provide the final response now.")])
            if isinstance(response, BaseMessage):
                return response
            return AIMessage(content=str(response))
        except Exception as exc:
            logger.error("Failed to generate async wrap-up message after recursion limit: %s", exc)
            return AIMessage(
                content=(
                    f"Recursion limit {recursion_limit} reached and wrap-up generation failed: {exc}"
                )
            )

    def _build_wrap_up_prompt(
        self,
        recursion_limit: int,
        error: Exception,
        latest_messages: Sequence[BaseMessage],
        agent_inputs: Optional[Dict[str, Any]],
    ) -> str:
        """Construct a concise wrap-up prompt using gathered context."""
        messages = list(latest_messages or [])
        input_messages = []
        if agent_inputs and isinstance(agent_inputs, dict):
            input_messages = agent_inputs.get("messages") or []
        user_question = self._last_user_message_content(messages or input_messages) or "Unavailable"

        conversation_snippets = []
        for msg in messages[-6:]:
            conversation_snippets.append(f"- {self._format_message(msg)}")

        memory = self.active_memory
        notes = memory.intermediate_steps() if memory else []
        document_summaries: List[str] = []
        if memory:
            for doc in memory.unique_documents()[:5]:
                metadata = doc.metadata or {}
                location = metadata.get("path") or metadata.get("source") or metadata.get("document_id") or "document"
                snippet = (doc.page_content or "")[:400]
                document_summaries.append(f"- {location}: {snippet}")

        prompt_sections: List[str] = [
            (
                "You are finalizing an interrupted ReAct agent run. The graph hit its recursion limit "
                f"({recursion_limit}) and can no longer call tools. Provide one concise wrap-up response: "
                "summarize what was attempted, cite retrieved evidence briefly, and answer the user's request "
                "as best as possible. Do NOT call tools."
            ),
            f"User request or latest message:\n{user_question}",
        ]
        if conversation_snippets:
            prompt_sections.append("Recent conversation (latest last):\n" + "\n".join(conversation_snippets))
        if notes:
            prompt_sections.append("Notes / steps recorded:\n" + "\n".join(f"- {n}" for n in notes))
        if document_summaries:
            prompt_sections.append("Retrieved documents (truncated):\n" + "\n".join(document_summaries))
        error_text = str(error) if error else ""
        if error_text:
            prompt_sections.append(f"Error detail: {error_text}")
        prompt_sections.append(
            "Respond with:\n"
            "1) Brief summary of what was attempted.\n"
            "2) Best possible answer using the above context.\n"
            f"3) Explicitly note that the run stopped after hitting the recursion limit {recursion_limit}."
        )
        return "\n\n".join(prompt_sections)
