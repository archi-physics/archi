import json
from collections import deque

import pytest
from mcp.types import CallToolResult

from src.evaluation.qa.artifacts import write_json
from src.evaluation.qa.catalog import EvaluationCatalog
from src.evaluation.qa.oracle import OracleCallEvidence, OracleResolver


class Extractor:
    def extract_gold(self, question, answer):
        return {"atoms": [{"id": "required", "text": answer, "required": True}]}


class Invoker:
    def __init__(self, values):
        self.values = deque(values)
        self.calls = 0

    def invoke(self, call):
        self.calls += 1
        return (
            CallToolResult(content=[], structuredContent=self.values.popleft()),
            OracleCallEvidence(call.id, 1, True),
        )


def _parent_blob():
    return (
        json.dumps(
            {
                "schema_version": "qa-dataset-v2",
                "items": [
                    {
                        "id": "static",
                        "question": "Fixed?",
                        "answer": "fixed",
                        "time_sensitive": False,
                    },
                    {
                        "id": "live",
                        "question": "Current?",
                        "time_sensitive": True,
                        "oracle": {
                            "kind": "mcp",
                            "calls": [
                                {
                                    "id": "lookup",
                                    "server": "read-model",
                                    "tool": "current",
                                    "arguments": {},
                                    "answer_fields": {"value": "/value"},
                                }
                            ],
                        },
                    },
                ],
            }
        )
        + "\n"
    ).encode()


class TestLiveCatalog:
    def test_materializes_approved_complete_child_without_mutating_parent(
        self, tmp_path
    ):
        catalog = EvaluationCatalog(tmp_path / "catalog")
        parent, _ = catalog.import_dataset("Parent", "parent.json", _parent_blob())
        assert parent["dataset_role"] == "definition_parent"
        parent_bytes = catalog.dataset_path(parent["id"]).read_bytes()
        invoker = Invoker([{"value": 7}])

        draft = catalog.create_atom_draft(
            parent["id"],
            "builtin",
            Extractor(),
            OracleResolver(invoker),
        )
        child = catalog.save_reviewed_dataset(
            draft["id"],
            "Approved",
            [
                {"item_id": row["item_id"], "atoms": row["atoms"]}
                for row in draft["items"]
                if row["status"] == "prepared"
            ],
            approval_actor="manager@example.test",
        )

        assert catalog.dataset_path(parent["id"]).read_bytes() == parent_bytes
        assert child["parent_dataset_id"] == parent["id"]
        assert child["generation_scope"] == "complete"
        assert child["dataset_role"] == "approved_child"
        assert child["contains_live_answers"] is True
        assert child["approval_actor"] == "manager@example.test"
        child_items = catalog.dataset_items(child["id"])
        assert child_items[1].answer == {"lookup": {"value": 7}}
        assert child_items[1].expected_atoms[0].required is True

    def test_static_only_child_omits_live_membership_and_calls(self, tmp_path):
        catalog = EvaluationCatalog(tmp_path / "catalog")
        parent, _ = catalog.import_dataset("Parent", "parent.json", _parent_blob())
        invoker = Invoker([])

        draft = catalog.create_atom_draft(
            parent["id"],
            "builtin",
            Extractor(),
            OracleResolver(invoker),
            static_only=True,
        )
        child = catalog.save_reviewed_dataset(
            draft["id"],
            "Static",
            [{"item_id": "static", "atoms": draft["items"][0]["atoms"]}],
        )

        assert invoker.calls == 0
        assert child["generation_scope"] == "static_only"
        assert child["dataset_role"] == "approved_child"
        assert child["contains_live_answers"] is False
        assert [item.id for item in catalog.dataset_items(child["id"])] == ["static"]

    def test_field_stripped_v2_entry_cannot_masquerade_as_legacy(self, tmp_path):
        catalog = EvaluationCatalog(tmp_path / "catalog")
        parent, _ = catalog.import_dataset("Parent", "parent.json", _parent_blob())
        directory = catalog.datasets_dir / parent["id"]
        legacy_fields = {
            "id",
            "name",
            "source_filename",
            "format",
            "sha256",
            "item_count",
            "eligible_item_count",
            "time_sensitive_item_count",
            "supplied_atom_item_count",
            "atom_count",
            "categories",
            "answer_sources",
            "parent_dataset_id",
            "created_at",
        }
        write_json(
            directory / "metadata.json",
            {key: value for key, value in parent.items() if key in legacy_fields},
        )
        (directory / "integrity.json").unlink()

        with pytest.raises(ValueError, match="integrity manifest is missing"):
            catalog.get_dataset(parent["id"])

    def test_approval_rejects_a_tampered_definition_parent(self, tmp_path):
        catalog = EvaluationCatalog(tmp_path / "catalog")
        parent, _ = catalog.import_dataset("Parent", "parent.json", _parent_blob())
        draft = catalog.create_atom_draft(
            parent["id"],
            "builtin",
            Extractor(),
            OracleResolver(Invoker([{"value": 7}])),
        )
        catalog.dataset_path(parent["id"]).write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError, match="integrity verification failed"):
            catalog.save_reviewed_dataset(
                draft["id"],
                "Untrusted",
                [
                    {"item_id": row["item_id"], "atoms": row["atoms"]}
                    for row in draft["items"]
                    if row["status"] == "prepared"
                ],
            )

    def test_failed_complete_draft_can_switch_to_static_only(self, tmp_path):
        catalog = EvaluationCatalog(tmp_path / "catalog")
        parent, _ = catalog.import_dataset("Parent", "parent.json", _parent_blob())
        draft = catalog.create_atom_draft(
            parent["id"],
            "builtin",
            Extractor(),
            OracleResolver(Invoker([])),
        )

        changed = catalog.make_atom_draft_static_only(draft["id"])
        live = next(row for row in changed["items"] if row["item_id"] == "live")
        child = catalog.save_reviewed_dataset(
            changed["id"],
            "Static after failure",
            [
                {"item_id": row["item_id"], "atoms": row["atoms"]}
                for row in changed["items"]
                if row["status"] == "prepared"
            ],
        )

        assert changed["generation_scope"] == "static_only"
        assert live == {
            "item_id": "live",
            "question": "Current?",
            "answer": None,
            "time_sensitive": True,
            "status": "skipped_live",
            "live_state": "omitted",
        }
        assert [item.id for item in catalog.dataset_items(child["id"])] == ["static"]

    def test_refresh_re_resolves_live_and_publishes_sibling(self, tmp_path):
        catalog = EvaluationCatalog(tmp_path / "catalog")
        parent, _ = catalog.import_dataset("Parent", "parent.json", _parent_blob())
        first = Invoker([{"value": 7}])
        draft = catalog.create_atom_draft(
            parent["id"],
            "builtin",
            Extractor(),
            OracleResolver(first),
        )
        child = catalog.save_reviewed_dataset(
            draft["id"],
            "First",
            [
                {"item_id": row["item_id"], "atoms": row["atoms"]}
                for row in draft["items"]
                if row["status"] == "prepared"
            ],
        )
        child_bytes = catalog.dataset_path(child["id"]).read_bytes()

        refresh = catalog.create_refresh_draft(
            child["id"],
            "builtin",
            Extractor(),
            OracleResolver(Invoker([{"value": 8}])),
        )
        live_row = next(row for row in refresh["items"] if row["item_id"] == "live")
        sibling = catalog.save_reviewed_dataset(
            refresh["id"],
            "Second",
            [
                {"item_id": row["item_id"], "atoms": row["atoms"]}
                for row in refresh["items"]
                if row["status"] == "prepared"
            ],
        )

        assert live_row["live_state"] == "changed"
        assert live_row["previous_answer"] == {"lookup": {"value": 7}}
        assert sibling["parent_dataset_id"] == parent["id"]
        assert sibling["based_on_child_id"] == child["id"]
        assert sibling["generation_scope"] == "refresh_live"
        assert catalog.dataset_path(child["id"]).read_bytes() == child_bytes
        assert catalog.dataset_items(sibling["id"])[1].answer == {
            "lookup": {"value": 8}
        }


def test_review_draft_save_error_points_live_items_at_the_generating_draft(tmp_path):
    """Including a live row from a review-atoms draft must name the way out.

    A review-atoms draft skips live rows at creation, so saving one can never
    resolve. The generic "must resolve" error gives the operator no pointer to
    the draft that does work — the generate-atoms job's own draft, whose id the
    job result carries as ``draft_id``.
    """
    catalog = EvaluationCatalog(tmp_path / "catalog")
    blob = (
        json.dumps(
            {
                "schema_version": "qa-dataset-v2",
                "items": [
                    {
                        "id": "static",
                        "question": "Fixed?",
                        "answer": "fixed",
                        "time_sensitive": False,
                        "expected_atoms": [
                            {"id": "a1", "text": "The answer is fixed.", "required": True}
                        ],
                    },
                    {
                        "id": "live",
                        "question": "Current?",
                        "time_sensitive": True,
                        "oracle": {
                            "kind": "mcp",
                            "calls": [
                                {
                                    "id": "lookup",
                                    "server": "read-model",
                                    "tool": "current",
                                    "arguments": {},
                                    "answer_fields": {"value": "/value"},
                                }
                            ],
                        },
                    },
                ],
            }
        )
        + "\n"
    ).encode()
    parent, _ = catalog.import_dataset("Parent", "parent.json", blob)
    draft = catalog.create_atom_review_draft(parent["id"])
    statuses = {row["item_id"]: row["status"] for row in draft["items"]}
    assert statuses == {"static": "prepared", "live": "skipped_time_sensitive"}

    # The operator answers the "at least one atom" rejection by hand-writing an
    # atom for the skipped live row — and then hits the opaque wall this fix
    # replaces.
    reviewed = [
        {
            "item_id": row["item_id"],
            "atoms": row["atoms"]
            or [{"id": "hand", "text": "Hand-written for a live row.", "required": True}],
        }
        for row in draft["items"]
    ]
    with pytest.raises(ValueError, match="generate-atoms") as caught:
        catalog.save_reviewed_dataset(
            draft["id"],
            "Reviewed",
            reviewed,
            approval_actor="manager@example.test",
        )

    assert "live" in str(caught.value)
    assert "draft_id" in str(caught.value)
