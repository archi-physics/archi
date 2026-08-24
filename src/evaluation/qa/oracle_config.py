# isort: skip_file
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import anyio
import httpx
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from .oracle import (
    OracleCall,
    OracleCallEvidence,
    OracleInvoker,
    OracleResolutionError,
    bounded_diagnostic,
)

MCP_CONFIG_SCHEMA_VERSION = "qa-evaluation-mcp-v1"
ORACLE_TIMEOUT_SECONDS = 120
ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


class _SafeInvocationError(ValueError):
    """An evaluator-owned diagnostic that contains no provider payload."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> Dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class MCPTransport(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class AuthenticationMode(str, Enum):
    INHERITED_ENVIRONMENT = "inherited_environment"
    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH_CLIENT_CREDENTIALS = "oauth_client_credentials"


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _exact_fields(
    raw: Mapping[str, Any], required: set, optional: set, context: str
) -> None:
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required - optional)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValueError(f"{context} has invalid fields ({'; '.join(details)})")


def _absolute_http_url(value: Any, context: str) -> str:
    url = _nonempty(value, context)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{context} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{context} must not contain credentials")
    return url


def _env_name(value: Any, context: str) -> str:
    name = _nonempty(value, context)
    if ENV_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"{context} must be an environment variable name")
    return name


def _string_list(value: Any, context: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of strings")
    return tuple(value)


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


@dataclass(frozen=True)
class AuthenticationConfig:
    mode: AuthenticationMode
    token_env: Optional[str] = None
    username_env: Optional[str] = None
    password_env: Optional[str] = None
    token_url: Optional[str] = None
    client_id_env: Optional[str] = None
    client_secret_env: Optional[str] = None
    scopes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPServerConfig:
    alias: str
    transport: MCPTransport
    authentication: AuthenticationConfig
    timeout_seconds: int = ORACLE_TIMEOUT_SECONDS
    command: Optional[str] = None
    args: Tuple[str, ...] = ()
    url: Optional[str] = None


def _parse_authentication(
    raw: Any, *, context: str, transport: MCPTransport
) -> AuthenticationConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    mode_value = raw.get("mode")
    try:
        mode = AuthenticationMode(mode_value)
    except ValueError as exc:
        raise ValueError(f"{context}.mode is unsupported") from exc
    if transport is MCPTransport.STDIO:
        _exact_fields(raw, {"mode"}, set(), context)
        if mode is not AuthenticationMode.INHERITED_ENVIRONMENT:
            raise ValueError(f"{context}.mode must be inherited_environment for stdio")
        return AuthenticationConfig(mode=mode)
    if mode is AuthenticationMode.INHERITED_ENVIRONMENT:
        raise ValueError(f"{context}.mode is unsupported for streamable_http")
    if mode is AuthenticationMode.NONE:
        _exact_fields(raw, {"mode"}, set(), context)
        return AuthenticationConfig(mode=mode)
    if mode is AuthenticationMode.BEARER:
        _exact_fields(raw, {"mode", "token_env"}, set(), context)
        return AuthenticationConfig(
            mode=mode,
            token_env=_env_name(raw["token_env"], f"{context}.token_env"),
        )
    if mode is AuthenticationMode.BASIC:
        _exact_fields(raw, {"mode", "username_env", "password_env"}, set(), context)
        return AuthenticationConfig(
            mode=mode,
            username_env=_env_name(raw["username_env"], f"{context}.username_env"),
            password_env=_env_name(raw["password_env"], f"{context}.password_env"),
        )
    _exact_fields(
        raw,
        {"mode", "token_url", "client_id_env", "client_secret_env"},
        {"scopes"},
        context,
    )
    return AuthenticationConfig(
        mode=mode,
        token_url=_absolute_http_url(raw["token_url"], f"{context}.token_url"),
        client_id_env=_env_name(raw["client_id_env"], f"{context}.client_id_env"),
        client_secret_env=_env_name(
            raw["client_secret_env"], f"{context}.client_secret_env"
        ),
        scopes=_string_list(raw.get("scopes", []), f"{context}.scopes"),
    )


def _parse_server(alias: str, raw: Any) -> MCPServerConfig:
    context = f"MCP server '{alias}'"
    _nonempty(alias, "MCP server alias")
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    try:
        transport = MCPTransport(raw.get("transport"))
    except ValueError as exc:
        raise ValueError(f"{context}.transport is unsupported") from exc
    timeout_seconds = _positive_integer(
        raw.get("timeout_seconds", ORACLE_TIMEOUT_SECONDS),
        f"{context}.timeout_seconds",
    )
    if transport is MCPTransport.STDIO:
        _exact_fields(
            raw,
            {"transport", "command", "authentication"},
            {"args", "timeout_seconds"},
            context,
        )
        return MCPServerConfig(
            alias=alias,
            transport=transport,
            timeout_seconds=timeout_seconds,
            command=_nonempty(raw["command"], f"{context}.command"),
            args=_string_list(raw.get("args", []), f"{context}.args"),
            authentication=_parse_authentication(
                raw["authentication"],
                context=f"{context}.authentication",
                transport=transport,
            ),
        )
    _exact_fields(
        raw,
        {"transport", "url", "authentication"},
        {"timeout_seconds"},
        context,
    )
    return MCPServerConfig(
        alias=alias,
        transport=transport,
        timeout_seconds=timeout_seconds,
        url=_absolute_http_url(raw["url"], f"{context}.url"),
        authentication=_parse_authentication(
            raw["authentication"],
            context=f"{context}.authentication",
            transport=transport,
        ),
    )


def _load_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.load(handle, Loader=_UniqueKeyLoader)
    except PermissionError:
        raise ValueError(
            f"evaluator MCP configuration is not readable: {path}"
        ) from None
    except UnicodeDecodeError as exc:
        raise ValueError("evaluator MCP configuration must be UTF-8") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid evaluator MCP YAML: {exc}") from exc


class EvaluatorMCPRegistry(OracleInvoker):
    def __init__(
        self,
        servers: Mapping[str, MCPServerConfig],
        *,
        configured: bool = True,
    ):
        self._servers = dict(servers)
        self._configured = configured

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EvaluatorMCPRegistry":
        if path is None:
            return cls({}, configured=False)
        resolved_path = path
        try:
            exists = resolved_path.exists()
        except PermissionError:
            raise ValueError(
                f"evaluator MCP configuration is not readable: {resolved_path}"
            ) from None
        if not exists:
            raise ValueError(
                f"evaluator MCP configuration does not exist: {resolved_path}"
            )
        if not resolved_path.is_file():
            raise ValueError(
                f"evaluator MCP configuration must be a file: {resolved_path}"
            )
        raw = _load_yaml(resolved_path)
        if not isinstance(raw, dict):
            raise ValueError("evaluator MCP configuration must be an object")
        _exact_fields(raw, {"schema_version", "servers"}, set(), "MCP configuration")
        if raw["schema_version"] != MCP_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported evaluator MCP schema_version '{raw['schema_version']}'"
            )
        if not isinstance(raw["servers"], dict):
            raise ValueError("MCP configuration servers must be an object")
        return cls(
            {
                _nonempty(alias, "MCP server alias"): _parse_server(alias, value)
                for alias, value in raw["servers"].items()
            }
        )

    @property
    def aliases(self) -> Tuple[str, ...]:
        return tuple(self._servers)

    @staticmethod
    def _secret(name: Optional[str]) -> str:
        if name is None:
            raise _SafeInvocationError(
                "invalid evaluator MCP authentication configuration"
            )
        value = os.environ.get(name)
        if value is None or not value:
            raise _SafeInvocationError(
                f"required environment variable '{name}' is not set"
            )
        return value

    async def _oauth_token(
        self, authentication: AuthenticationConfig
    ) -> Tuple[str, Tuple[str, ...]]:
        client_id = self._secret(authentication.client_id_env)
        client_secret = self._secret(authentication.client_secret_env)
        assert authentication.token_url is not None
        data = {"grant_type": "client_credentials"}
        if authentication.scopes:
            data["scope"] = " ".join(authentication.scopes)
        async with httpx.AsyncClient() as token_client:
            response = await token_client.post(
                authentication.token_url,
                data=data,
                auth=httpx.BasicAuth(client_id, client_secret),
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise _SafeInvocationError("OAuth token response must be an object")
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise _SafeInvocationError("OAuth token response is missing access_token")
        return token, (client_id, client_secret, token)

    async def _http_client(
        self, authentication: AuthenticationConfig
    ) -> Tuple[httpx.AsyncClient, Tuple[str, ...]]:
        if authentication.mode is AuthenticationMode.NONE:
            return httpx.AsyncClient(), ()
        if authentication.mode is AuthenticationMode.BEARER:
            token = self._secret(authentication.token_env)
            return httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}), (
                token,
            )
        if authentication.mode is AuthenticationMode.BASIC:
            username = self._secret(authentication.username_env)
            password = self._secret(authentication.password_env)
            return httpx.AsyncClient(auth=httpx.BasicAuth(username, password)), (
                username,
                password,
            )
        token, secrets = await self._oauth_token(authentication)
        return httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}), secrets

    @staticmethod
    async def _discover_and_call(
        read_stream: Any,
        write_stream: Any,
        call: OracleCall,
        timeout_seconds: int,
    ) -> CallToolResult:
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=timeout_seconds),
        ) as session:
            await session.initialize()
            cursor: Optional[str] = None
            discovered = []
            while True:
                tools = (
                    await session.list_tools()
                    if cursor is None
                    else await session.list_tools(cursor=cursor)
                )
                discovered.extend(tool.name for tool in tools.tools)
                cursor = tools.nextCursor
                if cursor is None:
                    break
            if call.tool not in discovered:
                raise _SafeInvocationError(
                    f"MCP server '{call.server}' does not expose tool '{call.tool}'"
                )
            return await session.call_tool(
                call.tool,
                arguments=call.arguments,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            )

    async def _invoke_async(
        self, call: OracleCall, server: MCPServerConfig
    ) -> Tuple[CallToolResult, Tuple[str, ...]]:
        with anyio.fail_after(server.timeout_seconds):
            if server.transport is MCPTransport.STDIO:
                assert server.command is not None
                parameters = StdioServerParameters(
                    command=server.command,
                    args=list(server.args),
                    env=dict(os.environ),
                )
                async with stdio_client(parameters) as streams:
                    result = await self._discover_and_call(
                        streams[0], streams[1], call, server.timeout_seconds
                    )
                return result, ()
            assert server.url is not None
            client, secrets = await self._http_client(server.authentication)
            try:
                async with client:
                    async with streamable_http_client(
                        server.url, http_client=client
                    ) as streams:
                        result = await self._discover_and_call(
                            streams[0], streams[1], call, server.timeout_seconds
                        )
            except _SafeInvocationError:
                raise
            except Exception as exc:
                raise RuntimeError("streamable HTTP MCP invocation failed") from exc
            return result, secrets

    @staticmethod
    def _configured_secrets(server: MCPServerConfig) -> Tuple[str, ...]:
        authentication = server.authentication
        names = (
            authentication.token_env,
            authentication.username_env,
            authentication.password_env,
            authentication.client_id_env,
            authentication.client_secret_env,
        )
        return tuple(
            value
            for name in names
            if name is not None
            for value in (os.environ.get(name),)
            if value
        )

    def invoke(self, call: OracleCall) -> Tuple[CallToolResult, OracleCallEvidence]:
        started = time.monotonic()
        if not self._configured:
            detail = "Evaluator MCP registry is not configured."
            evidence = OracleCallEvidence(
                call_id=call.id,
                duration_ms=int((time.monotonic() - started) * 1000),
                success=False,
                error=detail,
            )
            raise OracleResolutionError(detail, (evidence,))
        server = self._servers.get(call.server)
        if server is None:
            detail = f"evaluator MCP server alias '{call.server}' is not configured"
            evidence = OracleCallEvidence(
                call_id=call.id,
                duration_ms=int((time.monotonic() - started) * 1000),
                success=False,
                error=detail,
            )
            raise OracleResolutionError(detail, (evidence,))
        secrets: Sequence[str] = self._configured_secrets(server)
        try:
            result, runtime_secrets = anyio.run(self._invoke_async, call, server)
            secrets = (*secrets, *runtime_secrets)
        except Exception as exc:
            if isinstance(exc, _SafeInvocationError):
                detail = bounded_diagnostic(exc, secrets)
            elif isinstance(exc, TimeoutError):
                detail = (
                    f"Evaluator MCP call timed out after {server.timeout_seconds} "
                    "seconds."
                )
            else:
                detail = f"Evaluator MCP call failed ({type(exc).__name__})."
            evidence = OracleCallEvidence(
                call_id=call.id,
                duration_ms=int((time.monotonic() - started) * 1000),
                success=False,
                error=detail,
            )
            raise OracleResolutionError(detail, (evidence,)) from exc
        return result, OracleCallEvidence(
            call_id=call.id,
            duration_ms=int((time.monotonic() - started) * 1000),
            success=True,
        )
