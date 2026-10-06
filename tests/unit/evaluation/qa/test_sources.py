import json
from dataclasses import replace

import pytest

from src.evaluation.qa import dataset, preparation
from src.evaluation.qa.artifacts import write_jsonl
from src.evaluation.qa.catalog import EvaluationCatalog
from src.evaluation.qa.phases import score_answer, score_attempts
from src.evaluation.qa.scoring import build_summary
from src.evaluation.qa.sources import SourceEvaluation, evaluate_sources

ATOM = {"id": "A1", "text": "The answer is yes.", "required": True}
ROW = {"id": "item", "question": "Yes?", "answer": "Yes", "time_sensitive": False}


class Evaluator:
    def extract_gold(self, question, answer):
        return {"atoms": [ATOM]}

    def compare(self, question, gold_atoms, answer):
        return {
            "judgments": [{"atom_id": "A1", "outcome": "entailed", "rationale": "yes"}]
        }


def call(ordinal, response, **overrides):
    return {
        "ordinal": ordinal,
        "name": f"tool-{ordinal}",
        "status": "success",
        "query": "query",
        "response": response,
        **overrides,
    }


def test_literal_casefold_matches_all_successful_responses_only():
    sources = ["https://example.org/a?x=1", "Straße.pdf", "Never.pdf"]
    calls = [
        call(1, "Found HTTPS://EXAMPLE.ORG/A?X=1 and STRASSE.PDF"),
        call(2, "https://example.org/a?x=1 twice https://example.org/a?x=1"),
        call(3, "https://exampleXorg/a?x=1", query="Never.pdf"),
        {
            "ordinal": 4,
            "name": "failed",
            "status": "error",
            "query": "q",
            "error": "Never.pdf",
        },
        {"ordinal": 5, "name": "pending", "status": "incomplete", "query": "Never.pdf"},
    ]
    result = evaluate_sources(sources, calls).to_dict()
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["matches"][0]["matching_calls"] == [
        {"ordinal": 1, "name": "tool-1"},
        {"ordinal": 2, "name": "tool-2"},
    ]
    assert result["matches"][1]["outcome"] == "matched"
    assert result["matches"][2]["outcome"] == "missing"
    assert SourceEvaluation.from_dict(result).to_dict() == result


@pytest.mark.parametrize(
    "calls", [None, [{"ordinal": 1, "name": "old", "status": "success"}]]
)
def test_missing_historical_evidence_is_unavailable(calls):
    result = evaluate_sources(["Doc.pdf"], calls).to_dict()
    assert result["status"] == "unavailable"
    assert result["recall"] is None
    assert result["matches"][0]["outcome"] == "unavailable"


def test_proven_matches_survive_missing_historical_evidence():
    calls = [call(1, "Doc.pdf"), {"ordinal": 2, "name": "old", "status": "success"}]
    result = evaluate_sources(["Doc.pdf", "Other.pdf"], calls).to_dict()
    assert [match["outcome"] for match in result["matches"]] == [
        "matched",
        "unavailable",
    ]
    assert result["recall"] is None
    assert evaluate_sources(["Doc.pdf"], calls).recall == 1.0


def test_recorded_empty_calls_mean_missing_sources():
    assert evaluate_sources(["Doc.pdf"], []).recall == 0


@pytest.mark.parametrize(
    "sources",
    [
        None,
        "doc",
        [""],
        [" "],
        [4],
        ["a\nb"],
        ["a\rb"],
        ["Doc", "doc"],
        ["Straße", "STRASSE"],
    ],
)
def test_dataset_rejects_invalid_sources(sources):
    with pytest.raises(ValueError, match="expected_sources"):
        dataset.validate_dataset_rows([{**ROW, "expected_sources": sources}])


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("atoms_supplied", [False, True])
def test_upload_prepare_review_publish_preserves_and_edits_sources(
    tmp_path, v2, atoms_supplied
):
    row = {**ROW, "expected_sources": ["Uploaded.pdf"]}
    if atoms_supplied:
        row["expected_atoms"] = [ATOM]
    raw = {"schema_version": "qa-dataset-v2", "items": [row]} if v2 else [row]
    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset(
        "Sources", "sources.json", json.dumps(raw).encode()
    )
    item = dataset.load_dataset(catalog.dataset_path(parent["id"]))[1][0]
    assert dataset.dataset_item_to_dict(item)["expected_sources"] == ["Uploaded.pdf"]
    prepared = preparation.prepare_dataset_item(item, Evaluator())
    write_jsonl(tmp_path / "preparation.jsonl", [prepared.to_dict()])
    assert preparation.load_preparation_records(tmp_path / "preparation.jsonl")[
        0
    ].expected_sources == ("Uploaded.pdf",)
    draft = (
        catalog.create_atom_review_draft(parent["id"])
        if atoms_supplied
        else catalog.create_atom_draft(parent["id"], "builtin", Evaluator())
    )
    assert draft["items"][0]["expected_sources"] == ["Uploaded.pdf"]
    child = catalog.save_reviewed_dataset(
        draft["id"],
        "Edited",
        [
            {
                "item_id": "item",
                "atoms": [ATOM],
                "expected_sources": ["Edited.pdf", "https://example.org/doc"],
            }
        ],
    )
    final = dataset.load_dataset(catalog.dataset_path(child["id"]))[1][0]
    assert final.expected_sources == ("Edited.pdf", "https://example.org/doc")
    assert dataset.load_dataset(catalog.dataset_path(parent["id"]))[1][
        0
    ].expected_sources == ("Uploaded.pdf",)


@pytest.mark.parametrize(
    "review_sources, expected", [(None, ("Uploaded.pdf",)), ([], ())]
)
def test_review_preserves_omitted_sources_and_allows_removal(
    tmp_path, review_sources, expected
):
    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset(
        "Sources",
        "sources.json",
        json.dumps(
            [{**ROW, "expected_atoms": [ATOM], "expected_sources": ["Uploaded.pdf"]}]
        ).encode(),
    )
    draft = catalog.create_atom_review_draft(parent["id"])
    reviewed = {"item_id": "item", "atoms": [ATOM]}
    if review_sources is not None:
        reviewed["expected_sources"] = review_sources
    child = catalog.save_reviewed_dataset(draft["id"], "Reviewed", [reviewed])
    assert (
        dataset.load_dataset(catalog.dataset_path(child["id"]))[1][0].expected_sources
        == expected
    )


def test_source_scoring_is_independent_of_answer_judgment_and_summary(tmp_path):
    prepared = preparation.prepare_dataset_item(
        dataset.validate_dataset_rows([{**ROW, "expected_sources": ["Doc.pdf"]}])[0],
        Evaluator(),
    )
    answer = {
        "item_id": "item",
        "attempt_id": "attempt-1",
        "ordinal": 1,
        "agent_config_sha256": "a" * 64,
        "agent_spec_sha256": "b" * 64,
        "status": "answer_ready",
        "answer": "Yes Doc.pdf",
        "tool_calls": [],
    }
    result = score_answer(prepared, answer, Evaluator())
    assert result["passed"] is True
    assert result["source_evaluation"]["recall"] == 0

    class FailingEvaluator:
        def compare(self, *_args):
            raise RuntimeError("comparator down")

    failed = score_answer(
        prepared, {**answer, "tool_calls": [call(1, "DOC.PDF")]}, FailingEvaluator()
    )
    assert failed["status"] == "evaluation_failed"
    assert failed["source_evaluation"]["recall"] == 1
    summary = build_summary([prepared], [result, failed])
    assert summary["source_evaluation"] == {
        "scored_attempts": 2,
        "unavailable_attempts": 0,
        "all_matched_attempts": 1,
        "mean_recall": 0.5,
        "all_matched_rate": 0.5,
    }
    assert "source_evaluation" not in score_answer(
        replace(prepared, expected_sources=()), answer, Evaluator()
    )


def test_source_scoring_survives_agent_execution_failure():
    prepared = preparation.prepare_dataset_item(
        dataset.validate_dataset_rows([{**ROW, "expected_sources": ["Doc.pdf"]}])[0],
        Evaluator(),
    )
    answer = {
        "item_id": "item",
        "attempt_id": "attempt-1",
        "ordinal": 1,
        "agent_config_sha256": "a" * 64,
        "agent_spec_sha256": "b" * 64,
        "status": "execution_failed",
        "error": {"type": "RuntimeError", "message": "down"},
        "tool_calls": [call(1, "Doc.pdf")],
    }
    results = list(
        score_attempts(
            [(prepared, answer)],
            lambda: pytest.fail("must not create comparator"),
            1,
            thread_name_prefix="test",
        )
    )
    assert results[0]["source_evaluation"]["recall"] == 1


def test_source_result_rejects_inconsistent_artifact():
    raw = evaluate_sources(["Doc.pdf"], []).to_dict()
    raw["recall"] = 1
    with pytest.raises(ValueError, match="disagree"):
        SourceEvaluation.from_dict(raw)


def test_generated_retry_preserves_uploaded_sources(tmp_path):
    class FailingExtractor:
        def extract_gold(self, *_args):
            raise RuntimeError("temporarily down")

    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset(
        "Retry sources",
        "retry.json",
        json.dumps([{**ROW, "expected_sources": ["Uploaded.pdf"]}]).encode(),
    )
    draft = catalog.create_atom_draft(parent["id"], "builtin", FailingExtractor())
    assert draft["items"][0]["expected_sources"] == ["Uploaded.pdf"]
    assert draft["items"][0]["status"] == "preparation_failed"
    retried = catalog.retry_failed_atom_items(draft["id"], Evaluator())
    assert retried["items"][0]["expected_sources"] == ["Uploaded.pdf"]
    assert retried["items"][0]["status"] == "prepared"


def test_live_preparation_preserves_sources_and_does_not_use_oracle_evidence():
    from mcp.types import CallToolResult

    from src.evaluation.qa.dataset import V2DatasetReader
    from src.evaluation.qa.oracle import OracleCallEvidence, OracleResolver

    class Invoker:
        def invoke(self, request):
            return CallToolResult(
                content=[], structuredContent={"value": "Doc.pdf"}
            ), OracleCallEvidence(request.id, 1, True)

    item = V2DatasetReader(allow_materialized_live=False).read_row(
        {
            "id": "item",
            "question": "What is current?",
            "time_sensitive": True,
            "expected_sources": ["Doc.pdf"],
            "oracle": {
                "kind": "mcp",
                "calls": [
                    {
                        "id": "lookup",
                        "server": "fixture",
                        "tool": "current",
                        "arguments": {},
                        "answer_fields": {"value": "/value"},
                    }
                ],
            },
        },
        index=1,
    )
    prepared = preparation.prepare_dataset_item(
        item, Evaluator(), OracleResolver(Invoker())
    )
    assert prepared.expected_sources == ("Doc.pdf",)
    answer = {
        "item_id": "item",
        "attempt_id": "attempt-1",
        "ordinal": 1,
        "agent_config_sha256": "a" * 64,
        "agent_spec_sha256": "b" * 64,
        "status": "answer_ready",
        "answer": "Doc.pdf",
        "tool_calls": [],
    }
    assert (
        score_answer(prepared, answer, Evaluator())["source_evaluation"]["recall"] == 0
    )
