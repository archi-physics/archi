from typing import Any, Callable, Dict, List, Optional, Sequence, Iterator, AsyncIterator, Set, Tuple
import re
import time
import uuid

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
try:
    from langchain_core.messages import BaseMessageChunk
except ImportError:
    BaseMessageChunk = None
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.archi.pipelines.agents.utils.prompt_utils import read_prompt
from src.archi.pipelines.agents.utils.history_utils import infer_speaker
from src.archi.providers import get_model
from src.archi.providers.base import ProviderType
from src.archi.utils.output_dataclass import PipelineOutput
from src.archi.pipelines.agents.utils.run_memory import RunMemory
from src.archi.pipelines.agents.utils.mcp_utils import AsyncLoopThread
from src.archi.pipelines.agents.tools import initialize_mcp_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseReActAgent:
    """
    BaseReActAgent provides a foundational structure for building pipeline classes that
    process user queries using configurable language models and prompts.
    """
    DEFAULT_RECURSION_LIMIT = 50

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
        self._active_memory: Optional[RunMemory] = None
        self._static_tools: Optional[List[Callable]] = None
        self._mcp_tools: Optional[List[Callable]] = None
        self._active_tools: List[Callable] = []
        self._static_middleware: Optional[List[Callable]] = None
        self._active_middleware: List[Callable] = []
        self.agent: Optional[CompiledStateGraph] = None
        self.agent_llm: Optional[Any] = None
        self.agent_prompt: Optional[str] = None

        self.mcp_client = None


        self._init_llms()
        self._init_prompts()

        if self.agent_llm is None:
            if not self.llms:
                raise ValueError(f"No LLMs configured for agent {self.__class__.__name__}")
            self.agent_llm = self.llms.get("chat_model") or next(iter(self.llms.values()))
        if self.agent_prompt is None:
            self.agent_prompt = self.prompts.get("agent_prompt")

    def create_run_memory(self) -> RunMemory:
        """Instantiate a fresh run memory for an agent run."""
        return RunMemory()

    def start_run_memory(self) -> RunMemory:
        """Create and store the active memory for the current run."""
        memory = self.create_run_memory()
        self._active_memory = memory
        return memory

    @property
    def active_memory(self) -> Optional[RunMemory]:
        """Return the memory currently associated with the run, if any."""
        return self._active_memory

    def finalize_output(
        self,
        *,
        answer: str,
        memory: Optional[RunMemory] = None,
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
        resolved_metadata = dict(metadata or {})
        if memory:
            try:
                tool_inputs_by_id = memory.tool_inputs_by_id()
                if tool_inputs_by_id:
                    resolved_metadata.setdefault("tool_inputs_by_id", tool_inputs_by_id)
            except Exception as exc:
                logger.debug("Failed to attach tool_inputs_by_id to metadata: %s", exc)
        return PipelineOutput(
            answer=answer,
            source_documents=documents,
            messages=resolved_messages,
            metadata=resolved_metadata,
            final=final,
        )

    def _extract_usage_from_metadata(self, response_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """Normalize token usage from response_metadata when available."""
        if not response_metadata:
            return None
        # Different providers use different keys
        usage = response_metadata.get("usage") or response_metadata.get("token_usage")
        if usage:
            return {
                "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens", 0),
                "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        # Ollama format
        if "prompt_eval_count" in response_metadata or "eval_count" in response_metadata:
            prompt_tokens = response_metadata.get("prompt_eval_count", 0)
            completion_tokens = response_metadata.get("eval_count", 0)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return None

    def _extract_model_from_metadata(self, response_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract model name from response_metadata when available."""
        if not response_metadata:
            return None
        return response_metadata.get("model") or response_metadata.get("model_name")

    def _parse_thinking_content(self, text: str) -> Tuple[str, str]:
        """
        Parse text to separate thinking content from visible content.
        
        Handles <think>...</think> tags used by models like Qwen3.
        Returns (visible_content, thinking_content).
        """
        if not text:
            return "", ""
        
        # Extract all thinking blocks
        thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        thinking_matches = thinking_pattern.findall(text)
        thinking_content = "\n".join(thinking_matches)
        
        # Remove thinking blocks from visible content
        visible_content = thinking_pattern.sub('', text).strip()
        
        return visible_content, thinking_content

    def _extract_usage_from_messages(self, messages: List[BaseMessage]) -> Optional[Dict[str, int]]:
        """
        Sum token usage across ALL AI messages in the turn.
        
        In a multi-step agent loop, the LLM is called multiple times
        (thinking, tool decisions, final answer). Each call reports its
        own prompt_tokens and completion_tokens. We sum them to show
        total token throughput for the entire turn.
        """
        total_prompt = 0
        total_completion = 0
        found_any = False
        
        for msg in messages:
            msg_type = str(getattr(msg, "type", "")).lower()
            if msg_type not in {"ai", "assistant"} and "ai" not in type(msg).__name__.lower():
                continue
            usage = self._extract_usage_from_message(msg)
            if usage:
                total_prompt += usage.get("prompt_tokens", 0)
                total_completion += usage.get("completion_tokens", 0)
                found_any = True
        
        if not found_any:
            return None
        
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        }

    def _extract_usage_from_message(self, message: BaseMessage) -> Optional[Dict[str, int]]:
        """Extract normalized usage from a single message or chunk."""
        usage_metadata = getattr(message, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", prompt_tokens + completion_tokens)
            if prompt_tokens or completion_tokens or total_tokens:
                return {
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "total_tokens": int(total_tokens or 0),
                }

        response_metadata = getattr(message, "response_metadata", None)
        return self._extract_usage_from_metadata(response_metadata)

    def _extract_model_from_messages(self, messages: List[BaseMessage]) -> Optional[str]:
        """Extract model name from the last AI message with response_metadata."""
        for msg in reversed(messages):
            msg_type = str(getattr(msg, "type", "")).lower()
            if msg_type not in {"ai", "assistant"} and "ai" not in type(msg).__name__.lower():
                continue
            response_metadata = getattr(msg, "response_metadata", None)
            model = self._extract_model_from_metadata(response_metadata)
            if model:
                return model
        return None

    def _extract_reasoning_from_messages(self, messages: List[BaseMessage]) -> str:
        """Extract reasoning content from the last AI message, if present."""
        for msg in reversed(messages):
            msg_type = str(getattr(msg, "type", "")).lower()
            if msg_type not in {"ai", "assistant"} and "ai" not in type(msg).__name__.lower():
                continue
            additional_kwargs = getattr(msg, "additional_kwargs", None) or {}
            reasoning_content = additional_kwargs.get("reasoning_content", "")
            if reasoning_content:
                return str(reasoning_content)
        return ""

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
        """Stream agent updates synchronously with structured trace events."""
        logger.debug("Streaming %s", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)
        recursion_limit = self._recursion_limit()

        all_messages: List[BaseMessage] = []  # Accumulated full messages
        usage_messages: List[BaseMessage] = []  # Includes chunks with usage metadata
        latest_messages: List[BaseMessage] = []
        accumulated_content = ""  # Accumulated raw content from streaming
        emitted_tool_starts: Set[str] = set()
        
        # Thinking state tracking
        thinking_step_id: Optional[str] = None
        thinking_start_time: Optional[float] = None
        accumulated_thinking = ""  # Captured thinking content from <think> tags
        last_visible_content = ""  # Last visible content emitted (without thinking)
        last_response_metadata: Optional[Dict[str, Any]] = None
        
        try:
            for event in self.agent.stream(
                agent_inputs,
                stream_mode="messages",
                config={"recursion_limit": recursion_limit},
            ):

                messages = self._extract_messages(event)
                if not messages:
                    continue

                latest_messages = list(messages)
                message = messages[-1]
                msg_type = str(getattr(message, "type", "")).lower()
                msg_class = type(message).__name__.lower()

                if msg_type in {"ai", "assistant"} or "ai" in msg_class:
                    usage_messages.append(message)

                response_metadata = getattr(message, "response_metadata", None)
                if response_metadata:
                    last_response_metadata = response_metadata

                if self.active_memory:
                    try:
                        self.active_memory.record_tool_calls_from_message(message)
                    except Exception as exc:
                        logger.debug("Failed to record tool calls from stream message: %s", exc)
                
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
                        # End thinking phase if active before tool execution
                        if thinking_step_id is not None:
                            duration_ms = int((time.time() - thinking_start_time) * 1000) if thinking_start_time else 0
                            yield self.finalize_output(
                                answer="",
                                memory=self.active_memory,
                                messages=[],
                                metadata={
                                    "event_type": "thinking_end",
                                    "step_id": thinking_step_id,
                                    "duration_ms": duration_ms,
                                    "thinking_content": accumulated_thinking,
                                },
                                final=False,
                            )
                            thinking_step_id = None
                            thinking_start_time = None
                            accumulated_thinking = ""
                        
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
                        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
                        reasoning_content = additional_kwargs.get("reasoning_content", "")
                        if content or reasoning_content:
                            # Start thinking phase if not already active
                            if thinking_step_id is None:
                                thinking_step_id = str(uuid.uuid4())
                                thinking_start_time = time.time()
                                yield self.finalize_output(
                                    answer="",
                                    memory=self.active_memory,
                                    messages=[],
                                    metadata={
                                        "event_type": "thinking_start",
                                        "step_id": thinking_step_id,
                                    },
                                    final=False,
                                )
                            
                            if content:
                                # For chunks, content is delta; for full messages, content is cumulative
                                if "chunk" in msg_class:
                                    accumulated_content += content
                                else:
                                    # Full message - use its content directly
                                    accumulated_content = content

                            if reasoning_content:
                                # Ollama sends thinking as deltas, so accumulate
                                accumulated_thinking += reasoning_content
                                visible_content = accumulated_content
                            else:
                                # Parse thinking vs visible content
                                visible_content, thinking_content = self._parse_thinking_content(accumulated_content)
                                if not accumulated_thinking:
                                    accumulated_thinking = thinking_content
                            
                            # Only emit if visible content changed
                            if visible_content != last_visible_content:
                                last_visible_content = visible_content
                                yield self.finalize_output(
                                    answer=visible_content,
                                    memory=self.active_memory,
                                    messages=[message],
                                    metadata={"event_type": "text"},
                                    final=False,
                                )
        except GraphRecursionError as exc:
            logger.warning(
                "Recursion limit hit during stream for %s (limit=%s): %s",
                self.__class__.__name__,
                recursion_limit,
                exc,
            )
            if thinking_step_id is not None:
                duration_ms = int((time.time() - thinking_start_time) * 1000) if thinking_start_time else 0
                yield self.finalize_output(
                    answer="",
                    memory=self.active_memory,
                    messages=[],
                    metadata={
                        "event_type": "thinking_end",
                        "step_id": thinking_step_id,
                        "duration_ms": duration_ms,
                        "thinking_content": accumulated_thinking,
                    },
                    final=False,
                )
            recursion_output = self._handle_recursion_limit_error(
                error=exc,
                recursion_limit=recursion_limit,
                latest_messages=all_messages or latest_messages,
                agent_inputs=agent_inputs,
            )
            yield recursion_output
            return
        
        # Final output
        logger.debug("Stream finished. accumulated_content='%s', all_messages count=%d", 
                 accumulated_content[:100] if accumulated_content else "", len(all_messages))
        
        # End thinking phase if still active
        if thinking_step_id is not None:
            if not accumulated_thinking and all_messages:
                accumulated_thinking = self._extract_reasoning_from_messages(all_messages)
            duration_ms = int((time.time() - thinking_start_time) * 1000) if thinking_start_time else 0
            yield self.finalize_output(
                answer="",
                memory=self.active_memory,
                messages=[],
                metadata={
                    "event_type": "thinking_end",
                    "step_id": thinking_step_id,
                    "duration_ms": duration_ms,
                    "thinking_content": accumulated_thinking,
                },
                final=False,
            )
        
        final_answer = ""
        if all_messages:
            # Find the last AI message with content
            for msg in reversed(all_messages):
                msg_type = str(getattr(msg, "type", "")).lower()
                if msg_type in {"ai", "assistant"} or "ai" in type(msg).__name__.lower():
                    content = self._message_content(msg)
                    if content:
                        # Strip thinking from final answer
                        final_answer, _ = self._parse_thinking_content(content)
                        logger.debug("Found final answer from AI message: %s", final_answer[:100] if final_answer else "")
                        break
        if not final_answer:
            # Strip thinking from accumulated content
            final_answer, _ = self._parse_thinking_content(accumulated_content)
        
        # Extract usage and model info for final event
        usage = self._extract_usage_from_messages(usage_messages or all_messages)
        model = self._extract_model_from_messages(all_messages)
        if usage is None:
            usage = self._extract_usage_from_metadata(last_response_metadata)
        if model is None:
            model = self._extract_model_from_metadata(last_response_metadata)
        final_metadata = {
            "event_type": "final",
            "usage": usage,
            "model": model,
        }
        
        if final_answer:
            yield self.finalize_output(
                answer=final_answer,
                memory=self.active_memory,
                messages=all_messages,
                metadata=final_metadata,
                final=True,
            )
        else:
            logger.warning("No final answer found from stream. Messages: %s",
                          [self._format_message(m) for m in all_messages[:5]])
            output = self._build_output_from_messages(all_messages)
            output.metadata.update(final_metadata)
            yield output

    async def astream(self, **kwargs) -> AsyncIterator[PipelineOutput]:
        """Stream agent updates asynchronously with structured trace events."""
        logger.debug("Streaming %s asynchronously", self.__class__.__name__)
        agent_inputs = self._prepare_agent_inputs(**kwargs)
        if self.agent is None:
            self.refresh_agent(force=True)
        recursion_limit = self._recursion_limit()

        all_messages: List[BaseMessage] = []
        usage_messages: List[BaseMessage] = []
        latest_messages: List[BaseMessage] = []
        accumulated_content = ""
        emitted_tool_starts: Set[str] = set()
        
        # Thinking state tracking
        thinking_step_id: Optional[str] = None
        thinking_start_time: Optional[float] = None
        accumulated_thinking = ""  # Captured thinking content from <think> tags
        last_visible_content = ""  # Last visible content emitted (without thinking)
        last_response_metadata: Optional[Dict[str, Any]] = None
        
        try:
            async for event in self.agent.astream(
                agent_inputs,
                stream_mode="messages",
                config={"recursion_limit": recursion_limit},
            ):
                messages = self._extract_messages(event)
                if not messages:
                    continue

                latest_messages = list(messages)
                message = messages[-1]
                msg_type = str(getattr(message, "type", "")).lower()
                msg_class = type(message).__name__.lower()

                if msg_type in {"ai", "assistant"} or "ai" in msg_class:
                    usage_messages.append(message)
                
                response_metadata = getattr(message, "response_metadata", None)
                if response_metadata:
                    last_response_metadata = response_metadata

                if self.active_memory:
                    try:
                        self.active_memory.record_tool_calls_from_message(message)
                    except Exception as exc:
                        logger.debug("Failed to record tool calls from async stream message: %s", exc)
                
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
                        # End thinking phase if active before tool execution
                        if thinking_step_id is not None:
                            duration_ms = int((time.time() - thinking_start_time) * 1000) if thinking_start_time else 0
                            yield self.finalize_output(
                                answer="",
                                memory=self.active_memory,
                                messages=[],
                                metadata={
                                    "event_type": "thinking_end",
                                    "step_id": thinking_step_id,
                                    "duration_ms": duration_ms,
                                    "thinking_content": accumulated_thinking,
                                },
                                final=False,
                            )
                            thinking_step_id = None
                            thinking_start_time = None
                            accumulated_thinking = ""
                        
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
                        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
                        reasoning_content = additional_kwargs.get("reasoning_content", "")
                        if content or reasoning_content:
                            # Start thinking phase if not already active
                            if thinking_step_id is None:
                                thinking_step_id = str(uuid.uuid4())
                                thinking_start_time = time.time()
                                yield self.finalize_output(
                                    answer="",
                                    memory=self.active_memory,
                                    messages=[],
                                    metadata={
                                        "event_type": "thinking_start",
                                        "step_id": thinking_step_id,
                                    },
                                    final=False,
                                )
                            
                            if content:
                                if "chunk" in msg_class:
                                    accumulated_content += content
                                else:
                                    accumulated_content = content

                            if reasoning_content:
                                # Ollama sends thinking as deltas, so accumulate
                                accumulated_thinking += reasoning_content
                                visible_content = accumulated_content
                            else:
                                # Parse thinking vs visible content
                                visible_content, thinking_content = self._parse_thinking_content(accumulated_content)
                                if not accumulated_thinking:
                                    accumulated_thinking = thinking_content
                            
                            # Only emit if visible content changed
                            if visible_content != last_visible_content:
                                last_visible_content = visible_content
                                yield self.finalize_output(
                                    answer=visible_content,
                                    messages=[message],
                                    metadata={"event_type": "text"},
                                    final=False,
                                )
        except GraphRecursionError as exc:
            logger.warning(
                "Recursion limit hit during async stream for %s (limit=%s): %s",
                self.__class__.__name__,
                recursion_limit,
                exc,
            )
            if thinking_step_id is not None:
                duration_ms = int((time.time() - thinking_start_time) * 1000) if thinking_start_time else 0
                yield self.finalize_output(
                    answer="",
                    memory=self.active_memory,
                    messages=[],
                    metadata={
                        "event_type": "thinking_end",
                        "step_id": thinking_step_id,
                        "duration_ms": duration_ms,
                        "thinking_content": accumulated_thinking,
                    },
                    final=False,
                )
            recursion_output = await self._handle_recursion_limit_error_async(
                error=exc,
                recursion_limit=recursion_limit,
                latest_messages=all_messages or latest_messages,
                agent_inputs=agent_inputs,
            )
            yield recursion_output
            return
        
        # Final output
        logger.debug("Async stream finished. accumulated_content='%s', all_messages count=%d", 
                 accumulated_content[:100] if accumulated_content else "", len(all_messages))
        
        # End thinking phase if still active
        if thinking_step_id is not None:
            if not accumulated_thinking and all_messages:
                accumulated_thinking = self._extract_reasoning_from_messages(all_messages)
            duration_ms = int((time.time() - thinking_start_time) * 1000) if thinking_start_time else 0
            yield self.finalize_output(
                answer="",
                memory=self.active_memory,
                messages=[],
                metadata={
                    "event_type": "thinking_end",
                    "step_id": thinking_step_id,
                    "duration_ms": duration_ms,
                    "thinking_content": accumulated_thinking,
                },
                final=False,
            )
        
        final_answer = ""
        if all_messages:
            for msg in reversed(all_messages):
                msg_type = str(getattr(msg, "type", "")).lower()
                if msg_type in {"ai", "assistant"} or "ai" in type(msg).__name__.lower():
                    content = self._message_content(msg)
                    if content:
                        # Strip thinking from final answer
                        final_answer, _ = self._parse_thinking_content(content)
                        logger.debug("Found final answer from AI message: %s", final_answer[:100] if final_answer else "")
                        break
        if not final_answer:
            # Strip thinking from accumulated content
            final_answer, _ = self._parse_thinking_content(accumulated_content)
        
        # Extract usage and model info for final event
        usage = self._extract_usage_from_messages(usage_messages or all_messages)
        model = self._extract_model_from_messages(all_messages)
        if usage is None:
            usage = self._extract_usage_from_metadata(last_response_metadata)
        if model is None:
            model = self._extract_model_from_metadata(last_response_metadata)
        final_metadata = {
            "event_type": "final",
            "usage": usage,
            "model": model,
        }
        
        if final_answer:
            yield self.finalize_output(
                answer=final_answer,
                memory=self.active_memory,
                messages=all_messages,
                metadata=final_metadata,
                final=True,
            )
        else:
            logger.warning("No final answer found from async stream. Messages: %s",
                          [self._format_message(m) for m in all_messages[:5]])
            output = self._build_output_from_messages(all_messages)
            output.metadata.update(final_metadata)
            yield output

    def _init_llms(self) -> None:
        """Initialise language models for the agent."""

        self.llms: Dict[str, Any] = {}
        providers_config = {}
        if isinstance(self.config, dict):
            services_cfg = self.config.get("services", {}) if isinstance(self.config.get("services", {}), dict) else {}
            chat_cfg = services_cfg.get("chat_app", {}) if isinstance(services_cfg, dict) else {}
            providers_config = chat_cfg.get("providers", {}) if isinstance(chat_cfg, dict) else {}

        if self.default_provider and not self.default_model:
            raise ValueError("default_model is required when default_provider is set for agent pipelines.")
        if self.default_model and not self.default_provider:
            raise ValueError("default_provider is required when default_model is set for agent pipelines.")

        if self.default_provider and self.default_model:
            provider_config = self._build_provider_config(self.default_provider, providers_config)
            instance = get_model(self.default_provider, self.default_model, provider_config)
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
            provider_config = self._build_provider_config(provider, providers_config)
            instance = get_model(provider, model_id, provider_config)
            self.llms[model_name] = instance
            initialised_models[model_class_name] = instance

    @staticmethod
    def _build_provider_config(provider: str, providers_config: Dict[str, Any]) -> dict:
        provider_key = provider.lower() if isinstance(provider, str) else str(provider)
        cfg = providers_config.get(provider_key, {}) if isinstance(providers_config, dict) else {}
        if not cfg:
            return {}

        extra = {}
        try:
            provider_type = ProviderType(provider_key)
            if provider_type == ProviderType.LOCAL and cfg.get("mode"):
                extra["local_mode"] = cfg.get("mode")
        except Exception:
            pass

        return {
            "base_url": cfg.get("base_url"),
            "models": cfg.get("models", []),
            "default_model": cfg.get("default_model"),
            "extra_kwargs": extra,
        }

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

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return a mapping of tool names to descriptions for UI display."""
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

        if "mcp" in self.selected_tool_names:
            if self._mcp_tools is None:
                built = self._build_mcp_tools()
                self._mcp_tools = list(built or [])
            toolset.extend(self._mcp_tools)

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
        """Build and returns static tools defined in the config."""
        selected = list(self.selected_tool_names or [])
        static_names = [name for name in selected if name != "mcp"]
        return self._select_tools_from_registry(static_names)

    def _build_mcp_tools(self) -> List[Callable]:
        """Retrieve MCP tools from servers defined in the config and keep those server connections alive"""
        try:
            self._async_runner = AsyncLoopThread.get_instance()

            # Initialize MCP client on the background loop
            # The client and sessions will live on this loop
            client, mcp_tools = self._async_runner.run(initialize_mcp_client())
            if client is None:
                logger.info("No MCP servers configured.")
                return None
            self.mcp_client = client

            # Create synchronous wrappers that use the SAME loop
            def make_synchronous(async_tool):
                """
                Wrap an async tool for synchronous execution.

                Key difference from broken version:
                - Uses self._async_runner.run() instead of asyncio.run()
                - Runs on the SAME loop where the client was initialized
                - Session streams remain valid
                """
                # Capture the runner in closure
                runner = self._async_runner

                def sync_wrapper(*args, **kwargs):
                    if runner.in_loop_thread():
                        raise RuntimeError("sync_wrapper called from MCP loop thread; would deadlock")
                    # Run on the background loop - NOT a new loop!
                    return runner.run(async_tool.coroutine(*args, **kwargs))

                # Assign the wrapper to the tool's 'func' attribute
                async_tool.func = sync_wrapper
                return async_tool

            # Apply the patch to all fetched tools
            if mcp_tools:
                synchronous_mcp_tools = [make_synchronous(t) for t in mcp_tools]
                logger.info(f"Loaded and patched {len(synchronous_mcp_tools)} MCP tools for sync execution.")
                return synchronous_mcp_tools

        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}", exc_info=True)

    def _build_static_middleware(self) -> List[Callable]:
        """Build and returns static middleware defined in the config."""
        return []

    def _store_documents(self, stage: str, docs: Sequence[Document]) -> None:
        """Centralised helper used by tools to record documents into the active memory."""
        memory = self.active_memory
        if not memory:
            return
        # Prefer memory convenience method if available
        try:
            logger.debug("Recording %d documents from stage '%s' via record_documents", len(docs), stage)
            memory.record_documents(stage, docs)
        except Exception:
            # fallback to explicit record + note
            memory.record(stage, docs)
            memory.note(f"{stage} returned {len(list(docs))} document(s).")

    def _store_tool_input(self, tool_name: str, tool_input: Any) -> None:
        """Store runtime tool input so streamed tool ids can be backfilled with arguments."""
        memory = self.active_memory
        if not memory:
            return
        try:
            memory.record_tool_input(tool_name, tool_input)
        except Exception as exc:
            logger.debug("Failed to record tool input for %s: %s", tool_name, exc)

    def _prepare_inputs(self, history: Any, **kwargs) -> Dict[str, Any]:
        """Create list of messages using LangChain's formatting."""
        history = history or []
        history_messages = [infer_speaker(msg[0])(msg[1]) for msg in history]
        return {"history": history_messages}

    def _prepare_agent_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare agent state and formatted inputs shared by invoke/stream."""
        memory = self.start_run_memory()

        vectorstore = kwargs.get("vectorstore")
        if vectorstore and hasattr(self, "_update_vector_retrievers"):
            self._update_vector_retrievers(vectorstore)  # type: ignore[call-arg]
        elif vectorstore is None:
            if hasattr(self, "_vector_retrievers"):
                self._vector_retrievers = None  # type: ignore[attr-defined]
            if hasattr(self, "_vector_tools"):
                self._vector_tools = None  # type: ignore[attr-defined]

        extra_tools = None
        if hasattr(self, "_vector_tools"):
            extra_tools = self._vector_tools if self._vector_tools else None  # type: ignore[attr-defined]

        self.refresh_agent(extra_tools=extra_tools)

        inputs = self._prepare_inputs(history=kwargs.get("history"))
        history_messages = inputs["history"]
        if history_messages:
            memory.note(f"History contains {len(history_messages)} message(s).")
            last_message = history_messages[-1]
            content = self._message_content(last_message)
            if content:
                snippet = content if len(content) <= 200 else f"{content[:197]}..."
                memory.note(f"Latest user message: {snippet}")

        # --- Token trimming based on model context window ---
        try:
            if hasattr(self.agent_llm, "get_num_tokens_from_messages"):

                context_window = self._get_model_context_window()
                # Guard against None or invalid values
                if not isinstance(context_window, int) or context_window <= 0:
                    logger.debug(
                    "Invalid context window (%s), skipping trimming.",
                    context_window,
                    )
                    return {"messages": history_messages}

                safety_margin = int(context_window * 0.15)
                max_prompt_tokens = context_window - safety_margin

                logger.debug("Model: %s", getattr(self.agent_llm, "model", "unknown"))
                logger.debug("Context window: %d", context_window)
                logger.debug("Prompt token budget: %d", max_prompt_tokens)

                token_count = self.agent_llm.get_num_tokens_from_messages(history_messages)

                # Soft compression phase
                compression_round = 0
                while token_count >= max_prompt_tokens and len(history_messages) > 1:
                    compression_round += 1
                    logger.debug("Compression round %d triggered.", compression_round)

                    history_messages = self._compress_history(history_messages)
                    token_count = self.agent_llm.get_num_tokens_from_messages(
                        history_messages
                    )

                    # Prevent infinite compression loop
                    if compression_round > 3:
                        logger.warning("Exceeded max compression rounds.")
                        break

                   # Hard safeguard: crop if still too large
                if token_count >= max_prompt_tokens:
                    logger.warning("History still exceeds token limit (%d >= %d). Forcibly cropping.",token_count,max_prompt_tokens,)
                    keep_last_n = 4
                    history_messages = history_messages[-keep_last_n:]
                    token_count = self.agent_llm.get_num_tokens_from_messages(history_messages)

                    # --- Brutal safeguard: truncate content ---
                    while (token_count >= max_prompt_tokens and len(history_messages) > 1):
                        history_messages.pop(0)
                        token_count = self.agent_llm.get_num_tokens_from_messages(history_messages)

                logger.debug("Final trimmed token count: %d", token_count)

        except Exception as e:
            logger.debug("Token trimming skipped: %s", e)

        return {"messages": history_messages}

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
        if isinstance(payload, tuple) and len(payload) > 1 and isinstance(payload[1], message_types):
            return [payload[1]]
        if (
            isinstance(payload, tuple)
            and len(payload) > 1
            and isinstance(payload[1], list)
            and all(isinstance(msg, message_types) for msg in payload[1])
        ):
            return list(payload[1])
        def _messages_from_container(container: Any) -> List[BaseMessage]:
            if isinstance(container, dict):
                messages = container.get("messages")
                if isinstance(messages, list) and all(isinstance(msg, message_types) for msg in messages):
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


    def _get_model_context_window(self) -> Optional[int]:
        """
        Retrieve context_window from the configured provider + model
        using the provider abstraction layer.
        """
        try:
            if not self.default_provider or not self.default_model:
                return None

            from src.archi.providers import get_provider

            # Get provider instance (no reconstruction hacks)
            provider = get_provider(self.default_provider)

            if not provider:
                return None

            model_info = provider.get_model_info(self.default_model)
            if model_info:
                return model_info.context_window

        except Exception as e:
            logger.debug("Could not determine context window: %s", e)

        return None

    def _compress_history(self, history_messages):
        """
        Compress older conversation messages into a summary.
        Keeps the last few messages intact.
        """

        keep_last_n = 4

        if len(history_messages) <= keep_last_n:
            return history_messages

        recent = history_messages[-keep_last_n:]
        older = history_messages[:-keep_last_n]

        if not older:
            return history_messages

        # Only summarize half of older messages to avoid overflow
        chunk_size = max(1, len(older) // 2)
        chunk = older[:chunk_size]

        summary = self._summarize_messages(chunk)
        summary_message = AIMessage(content="Summary of earlier conversation:\n" + summary)
        return [summary_message] + older[chunk_size:] + recent


    def _summarize_messages(self, messages):

        texts = []
        for m in messages:
            content = self._message_content(m)
            if content:
                texts.append(content)

        combined_text = "\n".join(texts)

        if not combined_text.strip():
            return "Previous conversation summarized."

        try:
            response = self.agent_llm.invoke([
                SystemMessage(
                    content="Summarize the following conversation concisely, "
                            "preserving important facts and decisions."
                ),
                HumanMessage(content=combined_text),
            ])

            if isinstance(response, BaseMessage):
                return self._message_content(response)

            return str(response)

        except Exception as e:
            logger.warning("Summarization failed: %s", e)
            return "Earlier conversation summarized due to length constraints."



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
        """Read and validate recursion limit from config."""
        value = None
        if isinstance(self.pipeline_config, dict):
            value = self.pipeline_config.get("recursion_limit")
        if value is None and isinstance(self.config, dict):
            services_cfg = self.config.get("services", {})
            if isinstance(services_cfg, dict):
                chat_cfg = services_cfg.get("chat_app", {})
                if isinstance(chat_cfg, dict):
                    value = chat_cfg.get("recursion_limit")
        if value is None:
            value = self.DEFAULT_RECURSION_LIMIT
        try:
            limit = int(value)
            if limit <= 0:
                raise ValueError("recursion_limit must be positive")
            logger.info("Using recursion_limit=%s for %s", limit, self.__class__.__name__)
            return limit
        except Exception:
            logger.warning(
                "Invalid recursion_limit '%s' for %s; using default %s",
                value,
                self.__class__.__name__,
                self.DEFAULT_RECURSION_LIMIT,
            )
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
            "event_type": "final",
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
        return self.finalize_output(
            answer=self._message_content(messages[-1]),
            memory=self.active_memory,
            messages=messages,
            metadata=metadata,
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
        return self.finalize_output(
            answer=self._message_content(messages[-1]),
            memory=self.active_memory,
            messages=messages,
            metadata=metadata,
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
            response = self.agent_llm.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content="Provide the final response now."),
                ]
            )
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
                response = await self.agent_llm.ainvoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content="Provide the final response now."),
                    ]
                )
            else:
                response = self.agent_llm.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content="Provide the final response now."),
                    ]
                )
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
                location = (
                    metadata.get("path")
                    or metadata.get("source")
                    or metadata.get("document_id")
                    or "document"
                )
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
