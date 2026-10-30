"""A small tool-calling agent over an instance's MCP endpoint.

The loop is the whole agent: send the question and the MCP tools to the model,
run the tool calls it asks for, feed the results back, stop when it answers.
The model comes from an OpenAI-compatible chat endpoint (Ollama's `/v1`), or
from OpenAI's Responses API for a model that takes tools with a reasoning
effort only there; the tools come from the instance's streamable-HTTP MCP server with its bearer
token, the supported surface (ADR 0002 D5). Both sit behind two small
interfaces so the tests run with a fake model and a fake MCP and no network.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol
from urllib.parse import urlparse

#: The harness's own part of the system message: only the answer format. How to search and
#: read comes from the MCP server's instructions and the bundle's chat prompt, the same
#: guidance an operator's chat agent gets. It comes last and overrides the chat prompt's
#: citation style, since the checks read node ids. The example ids are placeholders that
#: no smoke set uses, so copying them never passes a check.
ANSWER_FORMAT = """\
# Answer format for this run (overrides any other citation style above)

Answer only from what the tools return; if they return nothing relevant, say so.
End your answer with a line "Sources:" listing the node ids you used. Copy each \
id in full, with the part before the colon, exactly as the tool results show it \
(ids look like some_type:12345 or other_type:abc-1; those two are only examples of \
the shape). Do not invent ids."""

DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_TOOL_CHARS = 6000
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class ModelError(RuntimeError):
    """The chat endpoint could not answer.

    `status` is the HTTP status when the endpoint answered with one; `sent` says whether the
    request reached the endpoint (so it may be billed); `retry_after` is the server's
    Retry-After in seconds, when it gave one."""

    def __init__(self, message: str, *, status: Optional[int] = None, sent: bool = False,
                 retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status = status
        self.sent = sent
        self.retry_after = retry_after


class ToolError(RuntimeError):
    """The MCP server could not be reached or listed its tools."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ChatReply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: input tokens the endpoint served from its prompt cache (OpenAI `prompt_tokens_details`)
    cached_tokens: int = 0
    #: completion tokens spent on reasoning (OpenAI `completion_tokens_details`); part of completion_tokens
    reasoning_tokens: int = 0
    #: the choice's finish_reason ("stop", "tool_calls", "length", ...), when the endpoint gave one
    finish_reason: str = ""
    #: whether the reply carried token usage; a paid run charges a reply without it at its bound
    usage_reported: bool = True
    #: the Responses API output items (reasoning, function calls), sent back as they came on the next
    #: turn; empty from chat/completions
    output_items: tuple[Mapping[str, Any], ...] = ()


def assistant_message(reply: ChatReply) -> dict:
    """The chat-format assistant message for a reply with tool calls, as the next request sends it back.
    A Responses reply also carries its output items, which `OpenAIResponses` sends back verbatim."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(dict(c.arguments))}}
            for c in reply.tool_calls
        ],
    }
    if reply.output_items:
        message["output_items"] = [dict(item) for item in reply.output_items]
    return message


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply: ...


class ToolBox(Protocol):
    #: the `instructions` of the server's initialize result, or None
    instructions: Optional[str]

    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(self, name: str, arguments: Mapping[str, Any]) -> tuple[str, bool]:
        """The tool's text and whether it reported an error."""
        ...


def sha(text: Optional[str]) -> Optional[str]:
    return None if text is None else "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_system_prompt(instructions: Optional[str], chat_prompt: Optional[str]) -> str:
    """The system message: what the MCP server serves about itself (`instructions` in its
    initialize result), then the bundle's chat system prompt, then the answer format last,
    so a small model keeps the citation rule the checks depend on."""
    parts = []
    if instructions:
        parts.append("# Instructions served by the MCP server\n\n" + instructions.strip())
    if chat_prompt:
        parts.append("# The chat agent's system prompt\n\n" + chat_prompt.strip())
    parts.append(ANSWER_FORMAT)
    return "\n\n".join(parts)


def prompt_hash(tools: list[ToolSpec], system: str = ANSWER_FORMAT) -> str:
    """One digest over the system message and the tool schemas the model saw."""
    body = json.dumps(
        {"system": system, "tools": [[t.name, t.description, t.input_schema] for t in tools]},
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def openai_tools(tools: list[ToolSpec]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": dict(t.input_schema) or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def is_local_url(url: str) -> bool:
    """Whether `url` is on this host or a private network."""
    host = (urlparse(url).hostname or "").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A bare name ("ollama", a compose service) or a .local / .localhost name.
        return host in LOCAL_HOSTS or "." not in host or host.endswith((".local", ".localhost"))
    return address.is_loopback or address.is_private


def require_local(url: str) -> None:
    """A smoke run costs nothing: refuse a model endpoint off this host or private network."""
    if is_local_url(url):
        return
    raise ModelError(
        f"model endpoint {url} is not on this host or a private network; the smoke harness "
        "uses local models only (no paid calls). Pass --allow-remote-model to override."
    )


class OpenAIChat:
    """`POST {base_url}/chat/completions`, as Ollama and vLLM serve it."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "ollama", temperature: Optional[float] = 0.0,
                 timeout: float = 600.0, no_think: bool = False, reasoning_effort: Optional[str] = None,
                 max_completion_tokens: Optional[int] = None) -> None:
        """`temperature=None` sends neither a temperature nor a seed (reasoning models refuse both),
        and neither is ever sent to an endpoint off this host or private network;
        `reasoning_effort` and `max_completion_tokens` are sent only when set."""
        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.no_think = no_think
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens

    def __repr__(self) -> str:  # never show the key
        return f"OpenAIChat({self.base_url!r}, {self.name!r})"

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "stream": False,
        }
        if self.temperature is not None and is_local_url(self.base_url):
            body["temperature"] = self.temperature
            body["seed"] = 0
        if tools:
            body["tools"] = tools
        if self.no_think:
            body["reasoning_effort"] = "none"
        elif self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens:
            body["max_completion_tokens"] = self.max_completion_tokens
        response = _post(self.base_url, "chat/completions", body, self.api_key, self.timeout)
        if response.status_code != 200:
            raise ModelError(f"{self.base_url}: HTTP {response.status_code}: {response.text[:300]}",
                             status=response.status_code, sent=True, retry_after=_retry_after(response))
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"{self.base_url}: unreadable reply: {response.text[:300]}", sent=True) from exc
        if not isinstance(message, dict):
            raise ModelError(f"{self.base_url}: unreadable reply: {response.text[:300]}", sent=True)
        calls = []
        for i, raw in enumerate(message.get("tool_calls") or []):
            fn = raw.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {"_unparsed": args}
            calls.append(ToolCall(str(raw.get("id") or f"call_{i}"), str(fn.get("name") or ""), args))
        raw_usage = payload.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return ChatReply(
            content=message.get("content") or "",
            tool_calls=tuple(calls),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=int(prompt_details.get("cached_tokens") or 0) if isinstance(prompt_details, dict) else 0,
            reasoning_tokens=(int(completion_details.get("reasoning_tokens") or 0)
                              if isinstance(completion_details, dict) else 0),
            finish_reason=str((choice.get("finish_reason") if isinstance(choice, dict) else "") or ""),
            usage_reported="prompt_tokens" in usage or "completion_tokens" in usage,
        )


def _post(base_url: str, path: str, body: dict, api_key: str, timeout: float):
    """POST `body` to `{base_url}/{path}`; a transport failure becomes a ModelError."""
    import requests

    try:
        return requests.post(
            f"{base_url}/{path}",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        # A connection that was never made cannot be billed; anything later (a read that timed
        # out, a connection reset mid-reply) may have been, so it counts as sent.
        sent = not (isinstance(exc, requests.ConnectTimeout)
                    or (isinstance(exc, requests.ConnectionError)
                        and ("NewConnectionError" in str(exc) or "Failed to establish" in str(exc))))
        raise ModelError(f"{base_url}: {exc}", sent=sent) from exc


def _retry_after(response) -> Optional[float]:
    try:
        return float(response.headers.get("Retry-After") or "")
    except (TypeError, ValueError):
        return None


#: Hosts whose models take function tools with a reasoning effort only on `/v1/responses`
#: (gpt-6-luna answers HTTP 400 to both on chat/completions).
RESPONSES_HOSTS = ("api.openai.com",)
MODEL_APIS = ("auto", "chat", "responses")


def model_api(url: str, choice: str = "auto") -> str:
    """Which API a model call uses: `auto` is `responses` for OpenAI's own endpoint and `chat`
    (chat/completions) for everything else, such as Ollama or vLLM."""
    if choice not in MODEL_APIS:
        raise ModelError(f"the model API is one of {', '.join(MODEL_APIS)} (got {choice!r})")
    if choice != "auto":
        return choice
    return "responses" if (urlparse(url).hostname or "").lower() in RESPONSES_HOSTS else "chat"


def make_model(url: str, model: str, *, api: str = "auto", **kwargs: Any) -> "OpenAIChat":
    """The client for `model_api(url, api)`; `kwargs` are OpenAIChat's."""
    cls = OpenAIResponses if model_api(url, api) == "responses" else OpenAIChat
    return cls(url, model, **kwargs)


def responses_tools(tools: list[dict]) -> list[dict]:
    """Chat-format function tools as the Responses API takes them: flat, and not strict, since the
    MCP schemas are not written for strict mode and chat/completions never applied it."""
    out = []
    for tool in tools:
        fn = tool.get("function") or {}
        out.append({
            "type": "function",
            "name": fn.get("name") or "",
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            "strict": False,
        })
    return out


def responses_input(messages: list[dict]) -> list[dict]:
    """Chat-format messages as Responses input items. An assistant turn that came from the Responses
    API is sent back as its own output items (reasoning included, as OpenAI asks for reasoning models
    with tools); a tool result becomes a `function_call_output` for its call id."""
    items: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": str(message.get("tool_call_id") or ""),
                          "output": message.get("content") or ""})
        elif role == "assistant" and message.get("output_items"):
            items.extend(dict(item) for item in message["output_items"])
        elif role == "assistant":
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append({"type": "function_call", "call_id": str(call.get("id") or ""),
                              "name": fn.get("name") or "", "arguments": fn.get("arguments") or "{}"})
        else:
            items.append({"role": role, "content": message.get("content") or ""})
    return items


def _error_text(response) -> str:
    """An HTTP error reply, readably: OpenAI's `error` object (type, code, param, message) when the body
    has one, else the body's start."""
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        error = None
    if not isinstance(error, dict) or not error.get("message"):
        return response.text[:300]
    tags = [f"{k} {error[k]}" for k in ("type", "code", "param") if error.get(k)]
    return f"{str(error['message'])[:500]}" + (f" ({', '.join(tags)})" if tags else "")


class OpenAIResponses(OpenAIChat):
    """`POST {base_url}/responses`, OpenAI's Responses API, for a model that takes function tools with a
    reasoning effort only there. It takes the same chat-format messages and tools as OpenAIChat and
    answers the same ChatReply, so the loops, the ledger and the cap are the same for both. Nothing is
    stored at OpenAI (`store: false`): each request carries the whole conversation, with the reasoning
    items of earlier turns, encrypted by OpenAI, sent back as they came."""

    def __repr__(self) -> str:  # never show the key
        return f"OpenAIResponses({self.base_url!r}, {self.name!r})"

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        body: dict[str, Any] = {"model": self.name, "input": responses_input(messages), "store": False}
        if self.temperature is not None and is_local_url(self.base_url):
            body["temperature"] = self.temperature
        if tools:
            body["tools"] = responses_tools(tools)
        effort = "none" if self.no_think else self.reasoning_effort
        if effort:
            body["reasoning"] = {"effort": effort}
        if self.max_completion_tokens:
            body["max_output_tokens"] = self.max_completion_tokens
        response = _post(self.base_url, "responses", body, self.api_key, self.timeout)
        if response.status_code != 200:
            raise ModelError(f"{self.base_url}/responses: HTTP {response.status_code}: {_error_text(response)}",
                             status=response.status_code, sent=True, retry_after=_retry_after(response))
        try:
            payload = response.json()
            output = payload["output"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelError(f"{self.base_url}/responses: unreadable reply: {response.text[:300]}",
                             sent=True) from exc
        if not isinstance(payload, dict) or not isinstance(output, list):
            raise ModelError(f"{self.base_url}/responses: unreadable reply: {response.text[:300]}", sent=True)
        status = str(payload.get("status") or "")
        if status == "failed":
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            raise ModelError(f"{self.base_url}/responses: the response failed: "
                             f"{error.get('message') or response.text[:300]} ({error.get('code') or 'no code'})",
                             status=500 if error.get("code") == "server_error" else None, sent=True)
        text, calls = [], []
        for i, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text.append(str(part.get("text") or ""))
                    elif isinstance(part, dict) and part.get("type") == "refusal":
                        text.append(str(part.get("refusal") or ""))
            elif item.get("type") == "function_call":
                args = item.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {"_unparsed": args}
                calls.append(ToolCall(str(item.get("call_id") or f"call_{i}"), str(item.get("name") or ""), args))
        incomplete = payload.get("incomplete_details") if isinstance(payload.get("incomplete_details"), dict) else {}
        if calls:
            finish = "tool_calls"
        elif status == "incomplete":
            reason = str(incomplete.get("reason") or "incomplete")
            finish = "length" if reason == "max_output_tokens" else reason
        else:
            finish = "stop"
        raw_usage = payload.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return ChatReply(
            content="".join(text),
            tool_calls=tuple(calls),
            prompt_tokens=int(usage.get("input_tokens") or 0),
            # reasoning tokens are billed as output and counted in output_tokens, as in chat/completions
            completion_tokens=int(usage.get("output_tokens") or 0),
            cached_tokens=int(input_details.get("cached_tokens") or 0) if isinstance(input_details, dict) else 0,
            reasoning_tokens=(int(output_details.get("reasoning_tokens") or 0)
                              if isinstance(output_details, dict) else 0),
            finish_reason=finish,
            usage_reported="input_tokens" in usage or "output_tokens" in usage,
            output_items=tuple(dict(item) for item in output if isinstance(item, dict)) if calls else (),
        )


class McpToolBox:
    """The instance's MCP endpoint (streamable HTTP, bearer token), held open for a run."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token
        self._stack = None
        self._session = None
        self.instructions: Optional[str] = None

    async def __aenter__(self) -> "McpToolBox":
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        stack = AsyncExitStack()
        try:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(self.url, headers={"Authorization": f"Bearer {self.token}"})
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            initialized = await session.initialize()
        except BaseException as exc:
            await stack.aclose()
            raise ToolError(f"{self.url}: {_describe(exc)}") from exc
        self._stack, self._session = stack, session
        self.instructions = getattr(initialized, "instructions", None)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._session.list_tools()
        return [ToolSpec(t.name, t.description or "", t.inputSchema or {}) for t in result.tools]

    async def call(self, name: str, arguments: Mapping[str, Any], *,
                   meta: Optional[Mapping[str, Any]] = None) -> tuple[str, bool]:
        """`meta` goes in the request's `_meta` (okg reads `okg/result_format` there)."""
        result = await self._session.call_tool(name, dict(arguments), meta=dict(meta) if meta else None)
        text = "".join(getattr(c, "text", "") for c in result.content)
        return text, bool(result.isError)


def _describe(exc: BaseException) -> str:
    inner = getattr(exc, "exceptions", None)
    if inner:
        return "; ".join(_describe(e) for e in inner)
    return f"{type(exc).__name__}: {exc}"


@dataclass
class ToolRecord:
    name: str
    arguments: Mapping[str, Any]
    chars: int
    is_error: bool


@dataclass
class Trace:
    """What one question's run did."""

    answer: str = ""
    tool_calls: list[ToolRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    steps: int = 0
    seconds: float = 0.0
    #: the text of every tool result as the model saw it, for reading node ids out of it
    results: list[str] = field(default_factory=list)
    error: Optional[str] = None
    finished: bool = False


async def ask(model: ChatModel, tools: ToolBox, specs: list[ToolSpec], question: str, *,
              system: str = ANSWER_FORMAT, max_steps: int = DEFAULT_MAX_STEPS, max_tool_chars: int = DEFAULT_MAX_TOOL_CHARS) -> Trace:
    """Run the loop for one question. A model failure ends it with `error` set."""
    trace = Trace()
    started = time.monotonic()
    schema = openai_tools(specs)
    known = {s.name for s in specs}
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    try:
        for _ in range(max_steps):
            trace.steps += 1
            reply = model.complete(messages, schema)
            trace.prompt_tokens += reply.prompt_tokens
            trace.completion_tokens += reply.completion_tokens
            if not reply.tool_calls:
                trace.answer = reply.content
                trace.finished = True
                break
            messages.append(assistant_message(reply))
            for call in reply.tool_calls:
                if call.name not in known:
                    text, is_error = f"error: no tool named {call.name!r}; tools are {sorted(known)}", True
                else:
                    try:
                        text, is_error = await tools.call(call.name, call.arguments)
                    except Exception as exc:  # noqa: BLE001 - a failing tool is data for the model
                        text, is_error = f"error: {_describe(exc)}", True
                trace.tool_calls.append(ToolRecord(call.name, dict(call.arguments), len(text), is_error))
                shown = text if len(text) <= max_tool_chars else text[:max_tool_chars] + "\n[cut]"
                trace.results.append(shown)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": shown})
    except ModelError as exc:
        trace.error = str(exc)
    trace.seconds = time.monotonic() - started
    return trace
