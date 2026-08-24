# isort: skip_file
import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

import src.evaluation.qa.oracle as oracle_module
from src.evaluation.qa.oracle import (
    OracleCallEvidence,
    OracleResolutionError,
    OracleResolver,
    answer_sha256,
    canonical_json,
    normalize_call_tool_result,
    parse_oracle_recipe,
    resolve_json_pointer,
)


class FakeInvoker:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def invoke(self, call):
        self.calls.append(call.id)
        return (
            CallToolResult(
                content=[], structuredContent=self.values[call.id], isError=False
            ),
            OracleCallEvidence(call.id, 3, True),
        )


class TestOracleResults:
    def test_rejects_oversized_selected_result_before_persistence(self, monkeypatch):
        monkeypatch.setattr(oracle_module, "MAX_SELECTED_ORACLE_BYTES", 32)
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "lookup",
                        "server": "server",
                        "tool": "lookup",
                        "arguments": {},
                    }
                ],
            }
        )

        with pytest.raises(OracleResolutionError) as caught:
            OracleResolver(FakeInvoker({"lookup": {"value": "x" * 64}})).resolve(recipe)

        assert caught.value.detail == (
            "Evaluator MCP selected result exceeds the 25 MB limit."
        )
        assert caught.value.calls[0].success is False

    def test_structured_content_precedes_unparsed_fallback(self):
        result = CallToolResult(
            structuredContent={"value": 7},
            content=[TextContent(type="text", text="not JSON")],
        )

        assert normalize_call_tool_result(result) == {"value": 7}

    @pytest.mark.parametrize(
        "result",
        [
            CallToolResult(content=[], isError=False),
            CallToolResult(
                content=[
                    TextContent(type="text", text="{}"),
                    TextContent(type="text", text="{}"),
                ]
            ),
            CallToolResult(
                content=[ImageContent(type="image", data="AA==", mimeType="image/png")]
            ),
            CallToolResult(content=[TextContent(type="text", text="```json\n{}\n```")]),
            CallToolResult(content=[TextContent(type="text", text='{"x":1,"x":2}')]),
            CallToolResult(content=[TextContent(type="text", text='{"x":NaN}')]),
            CallToolResult(
                content=[TextContent(type="text", text='{"x":1}')], isError=True
            ),
        ],
    )
    def test_rejects_ambiguous_or_invalid_fallback(self, result):
        with pytest.raises(ValueError):
            normalize_call_tool_result(result)

    def test_resolves_rfc6901_tokens_and_array_indices(self):
        value = {"a/b": {"~key": ["zero", {"ok": None}]}}

        assert resolve_json_pointer(value, "/a~1b/~0key/1/ok") is None

    def test_resolves_calls_in_recipe_order_and_separates_metadata(self):
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "first",
                        "server": "server",
                        "tool": "one",
                        "arguments": {},
                        "answer_fields": {"name": "/site/name"},
                        "metadata_fields": {"revision": "/revision"},
                    },
                    {
                        "id": "second",
                        "server": "server",
                        "tool": "two",
                        "arguments": {},
                    },
                ],
            }
        )
        invoker = FakeInvoker(
            {
                "first": {"site": {"name": "A"}, "revision": 3},
                "second": {"count": 2},
            }
        )

        resolved = OracleResolver(invoker).resolve(recipe)

        assert invoker.calls == ["first", "second"]
        assert resolved.answer == {
            "first": {"name": "A"},
            "second": {"count": 2},
        }
        assert resolved.metadata == {
            "first": {"revision": 3},
            "second": {},
        }
        assert resolved.answer_sha256 == answer_sha256(resolved.answer)

    def test_explicit_empty_metadata_selection_resolves_to_empty_object(self):
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "lookup",
                        "server": "server",
                        "tool": "lookup",
                        "arguments": {},
                        "answer_fields": {"value": "/value"},
                        "metadata_fields": {},
                    }
                ],
            }
        )

        resolved = OracleResolver(FakeInvoker({"lookup": {"value": 7}})).resolve(recipe)

        assert resolved.metadata == {"lookup": {}}

    def test_canonical_hash_sorts_objects_and_preserves_array_order(self):
        left = {"b": [2, 1], "a": "é"}
        same = {"a": "é", "b": [2, 1]}
        different = {"a": "é", "b": [1, 2]}

        assert canonical_json(left) == '{"a":"é","b":[2,1]}'
        assert answer_sha256(left) == answer_sha256(same)
        assert answer_sha256(left) != answer_sha256(different)

    def test_invalid_tool_payload_preserves_measured_call_duration(self):
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "lookup",
                        "server": "server",
                        "tool": "lookup",
                        "arguments": {},
                    }
                ],
            }
        )
        invoker = FakeInvoker({"lookup": {}})

        with pytest.raises(OracleResolutionError) as caught:
            OracleResolver(invoker).resolve(recipe)

        assert caught.value.calls == (
            OracleCallEvidence(
                call_id="lookup",
                duration_ms=3,
                success=False,
                error="Evaluator MCP result was invalid.",
            ),
        )

    @pytest.mark.parametrize(
        "result",
        [
            CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text='{"RAW_SECRET_SENTINEL":1,"RAW_SECRET_SENTINEL":2}',
                    )
                ]
            ),
            CallToolResult(
                content=[],
                structuredContent={"RAW_SECRET_SENTINEL": object()},
            ),
        ],
    )
    def test_provider_payload_failures_never_persist_raw_output(self, result):
        recipe = parse_oracle_recipe(
            {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "lookup",
                        "server": "server",
                        "tool": "lookup",
                        "arguments": {},
                    }
                ],
            }
        )

        class InvalidInvoker:
            def invoke(self, call):
                return result, OracleCallEvidence(call.id, 7, True)

        with pytest.raises(OracleResolutionError) as caught:
            OracleResolver(InvalidInvoker()).resolve(recipe)

        assert caught.value.detail == "Evaluator MCP result was invalid."
        assert "RAW_SECRET_SENTINEL" not in str(caught.value)
        assert "RAW_SECRET_SENTINEL" not in caught.value.calls[0].error
