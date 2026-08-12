import pytest

from src.evaluation.qa.tool_traces import serialize_tool_call_records


def test_tool_call_records_accept_detailed_and_legacy_variants():
    rows = [
        {
            "ordinal": 1,
            "name": "search",
            "status": "success",
            "query": '{"query": "complete"}',
            "response": '{"matches": ["answer"]}',
            "duration_ms": 12,
        },
        {
            "ordinal": 2,
            "name": "lookup",
            "status": "error",
            "query": "missing",
            "error": "not found",
            "duration_ms": 3,
        },
        {
            "ordinal": 3,
            "name": "unfinished",
            "status": "incomplete",
            "query": "next",
        },
        {
            "ordinal": 4,
            "name": "historical",
            "status": "success",
            "duration_ms": 2,
        },
    ]

    assert serialize_tool_call_records(rows, context="test trace") == rows


@pytest.mark.parametrize(
    "row, message",
    [
        (
            {"ordinal": 1, "name": "search", "status": "success", "query": "q"},
            "require only a response",
        ),
        (
            {
                "ordinal": 1,
                "name": "search",
                "status": "incomplete",
                "query": "q",
                "response": "invented",
            },
            "cannot contain terminal detail",
        ),
        (
            {
                "ordinal": 1,
                "name": "search",
                "status": "incomplete",
                "query": "q",
                "duration_ms": 1,
            },
            "cannot contain duration",
        ),
        (
            {
                "ordinal": 1,
                "name": "historical-incomplete",
                "status": "incomplete",
                "duration_ms": 1,
            },
            "cannot contain duration",
        ),
        (
            {
                "ordinal": 1,
                "name": "missing-query",
                "status": "incomplete",
            },
            "require a query",
        ),
        (
            {
                "ordinal": 1,
                "name": "null-response",
                "status": "success",
                "query": "q",
                "response": None,
            },
            "null fields that must be omitted: response",
        ),
        (
            {
                "ordinal": 1,
                "name": "search",
                "status": "success",
                "duration_ms": 1.5,
            },
            "non-negative integer",
        ),
        (
            {
                "ordinal": 1,
                "name": "search",
                "status": "success",
                "duration_ms": 1,
                "synthetic": True,
            },
            "invalid fields",
        ),
    ],
)
def test_tool_call_records_reject_invalid_artifact_rows(row, message):
    with pytest.raises(ValueError, match=message):
        serialize_tool_call_records([row], context="test trace")


def test_tool_call_records_require_unique_ordered_ordinals():
    with pytest.raises(ValueError, match="unique and ordered"):
        serialize_tool_call_records(
            [
                {
                    "ordinal": 2,
                    "name": "second",
                    "status": "success",
                    "duration_ms": 1,
                },
                {
                    "ordinal": 1,
                    "name": "first",
                    "status": "success",
                    "duration_ms": 1,
                },
            ],
            context="test trace",
        )
