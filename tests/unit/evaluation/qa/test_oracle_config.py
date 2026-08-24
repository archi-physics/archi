# isort: skip_file
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.evaluation.qa.oracle import OracleResolver, parse_oracle_recipe
from src.evaluation.qa.oracle import OracleResolutionError
from src.evaluation.qa.oracle_config import (
    AuthenticationMode,
    EvaluatorMCPRegistry,
    MCPTransport,
)


class TestEvaluatorMCPRegistry:
    def test_transport_exception_does_not_persist_provider_payload(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            "schema_version: qa-evaluation-mcp-v1\n"
            "servers:\n"
            "  fixture:\n"
            "    transport: stdio\n"
            "    command: fixture\n"
            "    authentication: {mode: inherited_environment}\n",
            encoding="utf-8",
        )
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "capacity",
                        "server": "fixture",
                        "tool": "current_capacity",
                        "arguments": {},
                    }
                ],
            }
        )
        monkeypatch.setattr(
            "src.evaluation.qa.oracle_config.anyio.run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(
                    "raw provider body SUPERSECRET Authorization: Bearer token"
                )
            ),
        )

        with pytest.raises(OracleResolutionError) as caught:
            OracleResolver(EvaluatorMCPRegistry.load(path)).resolve(recipe)

        assert caught.value.detail == "Evaluator MCP call failed (RuntimeError)."
        assert "SUPERSECRET" not in caught.value.calls[0].error

    def test_invokes_a_real_stdio_mcp_server(self, tmp_path):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            "schema_version: qa-evaluation-mcp-v1\n"
            "servers:\n"
            "  fixture:\n"
            "    transport: stdio\n"
            f"    command: {sys.executable}\n"
            "    args: [-m, tests.unit.evaluation.qa.fake_mcp_server]\n"
            "    timeout_seconds: 5\n"
            "    authentication: {mode: inherited_environment}\n",
            encoding="utf-8",
        )
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "capacity",
                        "server": "fixture",
                        "tool": "current_capacity",
                        "arguments": {"service": "primary"},
                        "answer_fields": {"available": "/available"},
                        "metadata_fields": {"revision": "/revision"},
                    }
                ],
            },
        )

        resolved = OracleResolver(EvaluatorMCPRegistry.load(path)).resolve(recipe)

        assert resolved.answer == {"capacity": {"available": 7}}
        assert resolved.metadata == {"capacity": {"revision": "fixture-r1"}}
        assert resolved.calls[0].success is True

    def test_invokes_a_real_streamable_http_mcp_server(self, tmp_path):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        environment = dict(os.environ)
        environment.update(
            {
                "QA_FAKE_MCP_TRANSPORT": "streamable-http",
                "QA_FAKE_MCP_PORT": str(port),
            }
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "tests.unit.evaluation.qa.fake_mcp_server"],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5
            while True:
                with socket.socket() as client:
                    if client.connect_ex(("127.0.0.1", port)) == 0:
                        break
                if time.monotonic() >= deadline:
                    raise AssertionError("fake HTTP MCP server did not start")
                time.sleep(0.02)
            path = tmp_path / "mcp.yaml"
            path.write_text(
                "schema_version: qa-evaluation-mcp-v1\n"
                "servers:\n"
                "  fixture:\n"
                "    transport: streamable_http\n"
                f"    url: http://127.0.0.1:{port}/mcp\n"
                "    authentication: {mode: none}\n",
                encoding="utf-8",
            )
            recipe = parse_oracle_recipe(
                {
                    "kind": "mcp",
                    "calls": [
                        {
                            "id": "capacity",
                            "server": "fixture",
                            "tool": "current_capacity",
                            "arguments": {"service": "primary"},
                            "answer_fields": {"available": "/available"},
                        }
                    ],
                }
            )

            resolved = OracleResolver(EvaluatorMCPRegistry.load(path)).resolve(recipe)

            assert resolved.answer == {"capacity": {"available": 7}}
            assert resolved.calls[0].success is True
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def test_omitted_file_is_an_empty_registry(self):
        registry = EvaluatorMCPRegistry.load()

        assert registry.aliases == ()

    def test_explicit_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(
            ValueError,
            match="evaluator MCP configuration does not exist",
        ):
            EvaluatorMCPRegistry.load(tmp_path / "missing.yaml")

    def test_explicit_unreadable_file_is_rejected(self, tmp_path, monkeypatch):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            "schema_version: qa-evaluation-mcp-v1\nservers: {}\n",
            encoding="utf-8",
        )
        original_open = Path.open

        def deny_registry_open(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError
            return original_open(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "open", deny_registry_open)

        with pytest.raises(
            ValueError,
            match="evaluator MCP configuration is not readable",
        ):
            EvaluatorMCPRegistry.load(path)

    def test_empty_registry_reports_missing_registry_before_alias(self):
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "capacity",
                        "server": "dbs",
                        "tool": "current_capacity",
                        "arguments": {},
                    }
                ],
            }
        )

        with pytest.raises(OracleResolutionError) as caught:
            OracleResolver(EvaluatorMCPRegistry.load()).resolve(recipe)

        assert caught.value.detail == "Evaluator MCP registry is not configured."

    def test_configured_empty_registry_reports_missing_alias(self, tmp_path):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            "schema_version: qa-evaluation-mcp-v1\nservers: {}\n",
            encoding="utf-8",
        )
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "capacity",
                        "server": "dbs",
                        "tool": "current_capacity",
                        "arguments": {},
                    }
                ],
            }
        )

        with pytest.raises(OracleResolutionError) as caught:
            OracleResolver(EvaluatorMCPRegistry.load(path)).resolve(recipe)

        assert caught.value.detail == (
            "evaluator MCP server alias 'dbs' is not configured"
        )

    def test_rejects_duplicate_server_aliases(self, tmp_path):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            """schema_version: qa-evaluation-mcp-v1
servers:
  duplicate:
    transport: stdio
    command: first
    authentication: {mode: inherited_environment}
  duplicate:
    transport: stdio
    command: second
    authentication: {mode: inherited_environment}
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="duplicate key 'duplicate'"):
            EvaluatorMCPRegistry.load(path)

    def test_loads_strict_descriptors_without_resolving_secrets(self, tmp_path):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            """
schema_version: qa-evaluation-mcp-v1
servers:
  local:
    transport: stdio
    command: /bin/tool
    args: [--read-only]
    timeout_seconds: 45
    authentication:
      mode: inherited_environment
  remote:
    transport: streamable_http
    url: https://example.test/mcp
    authentication:
      mode: bearer
      token_env: MISSING_UNTIL_INVOCATION
""".strip(),
            encoding="utf-8",
        )

        registry = EvaluatorMCPRegistry.load(path)

        assert registry.aliases == ("local", "remote")
        assert registry._servers["local"].transport is MCPTransport.STDIO
        assert registry._servers["local"].timeout_seconds == 45
        assert registry._servers["remote"].timeout_seconds == 120
        assert (
            registry._servers["remote"].authentication.mode is AuthenticationMode.BEARER
        )

    def test_reports_configured_server_timeout(self, tmp_path, monkeypatch):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            "schema_version: qa-evaluation-mcp-v1\n"
            "servers:\n"
            "  fixture:\n"
            "    transport: stdio\n"
            "    command: fixture\n"
            "    timeout_seconds: 7\n"
            "    authentication: {mode: inherited_environment}\n",
            encoding="utf-8",
        )
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "capacity",
                        "server": "fixture",
                        "tool": "current_capacity",
                        "arguments": {},
                    }
                ],
            }
        )
        monkeypatch.setattr(
            "src.evaluation.qa.oracle_config.anyio.run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
        )

        with pytest.raises(OracleResolutionError) as caught:
            OracleResolver(EvaluatorMCPRegistry.load(path)).resolve(recipe)

        assert caught.value.detail == "Evaluator MCP call timed out after 7 seconds."

    @pytest.mark.parametrize("timeout", [True, 0, -1, 1.5, "30"])
    def test_rejects_invalid_server_timeout(self, tmp_path, timeout):
        path = tmp_path / "mcp.yaml"
        path.write_text(
            "schema_version: qa-evaluation-mcp-v1\n"
            "servers:\n"
            "  fixture:\n"
            "    transport: stdio\n"
            "    command: fixture\n"
            f"    timeout_seconds: {timeout!r}\n"
            "    authentication: {mode: inherited_environment}\n",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="timeout_seconds must be a positive integer"
        ):
            EvaluatorMCPRegistry.load(path)

    @pytest.mark.parametrize(
        "server_yaml, message",
        [
            (
                "transport: stdio\n"
                "command: /bin/tool\n"
                "authentication: {mode: none}\n",
                "inherited_environment",
            ),
            (
                "transport: streamable_http\n"
                "url: relative/mcp\n"
                "authentication: {mode: none}\n",
                "absolute HTTP",
            ),
            (
                "transport: streamable_http\n"
                "url: https://example.test/mcp\n"
                "authentication: {mode: bearer, token_env: not-valid-name}\n",
                "environment variable name",
            ),
            (
                "transport: streamable_http\n"
                "url: https://example.test/mcp\n"
                "authentication: {mode: none}\n"
                "headers: {X-Unsafe: value}\n",
                "unknown: headers",
            ),
        ],
    )
    def test_rejects_invalid_transport_authentication_combinations(
        self, tmp_path, server_yaml, message
    ):
        path = tmp_path / "mcp.yaml"
        indented = "\n".join(f"    {line}" for line in server_yaml.strip().splitlines())
        path.write_text(
            "schema_version: qa-evaluation-mcp-v1\n"
            "servers:\n"
            "  target:\n"
            f"{indented}\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=message):
            EvaluatorMCPRegistry.load(path)
