import json

import pytest

from src.evaluation.qa.catalog import EvaluationCatalog
from src.evaluation.qa.validation import load_dataset


class _Evaluator:
    def extract_gold(self, question, answer):
        return {
            "atoms": [
                {
                    "id": "A1",
                    "text": f"Reviewed: {answer}",
                    "required": True,
                }
            ]
        }


class _SelectiveEvaluator:
    def __init__(self, failing_answers=None):
        self.failing_answers = set(failing_answers or [])
        self.calls = []

    def extract_gold(self, question, answer):
        self.calls.append((question, answer))
        if answer in self.failing_answers:
            raise RuntimeError(f"cannot extract {answer}")
        return {
            "atoms": [
                {
                    "id": "A1",
                    "text": f"Generated: {answer}",
                    "required": True,
                }
            ]
        }


def _dataset_blob():
    return json.dumps(
        [
            {
                "id": "eligible",
                "question": "Q",
                "answer": "A",
                "time_sensitive": False,
                "category": "operations",
                "answer_source": "static_docs_tickets",
            },
            {
                "id": "changing",
                "question": "Current?",
                "answer": "Changes",
                "time_sensitive": True,
                "answer_source": "rucio",
            },
        ]
    ).encode()


def _partially_atomized_dataset_blob():
    return json.dumps(
        [
            {
                "id": "with-atom",
                "question": "Answered?",
                "answer": "Yes",
                "time_sensitive": False,
                "answer_source": "existing_source",
                "expected_atoms": [
                    {"id": "A1", "text": "The answer is yes.", "required": True}
                ],
            },
            {
                "id": "without-atom",
                "question": "Missing?",
                "answer": "Not yet",
                "time_sensitive": False,
                "answer_source": "missing_source",
            },
            {
                "id": "changing",
                "question": "Current?",
                "answer": "Changes",
                "time_sensitive": True,
                "answer_source": "changing_source",
            },
        ]
    ).encode()


def _skipped_atom_dataset_blob():
    return json.dumps(
        [
            {
                "id": "eligible",
                "question": "Needs review?",
                "answer": "Yes",
                "time_sensitive": False,
            },
            {
                "id": "changing",
                "question": "Current?",
                "answer": "Changes",
                "time_sensitive": True,
                "expected_atoms": [
                    {
                        "id": "current-A1",
                        "text": "The current answer changes.",
                        "required": True,
                    }
                ],
            },
        ]
    ).encode()


def _two_item_dataset_blob():
    return json.dumps(
        [
            {
                "id": "first",
                "question": "First?",
                "answer": "First answer",
                "time_sensitive": False,
            },
            {
                "id": "second",
                "question": "Second?",
                "answer": "Second answer",
                "time_sensitive": False,
            },
        ]
    ).encode()


def test_dataset_import_preserves_mocked_source_and_reports_metadata(tmp_path):
    source = json.dumps(
        [
            {
                "id": "mock-entry",
                "question": "What value does the mock entry contain?",
                "answer": "The mock entry contains 42.",
                "time_sensitive": False,
                "category": "test-fixture",
                "answer_source": "mocked-entry",
            }
        ]
    ).encode()
    catalog = EvaluationCatalog(tmp_path)

    metadata, created = catalog.import_dataset(
        "Mock dataset", "mock-dataset.json", source
    )

    assert created is True
    assert metadata["item_count"] == 1
    assert metadata["eligible_item_count"] == 1
    assert metadata["time_sensitive_item_count"] == 0
    assert metadata["categories"] == ["test-fixture"]
    assert metadata["answer_sources"] == ["mocked-entry"]
    assert catalog.dataset_path(metadata["id"]).read_bytes() == source
    assert "answer" not in metadata


def test_legacy_catalog_entry_is_verified_and_backfilled_on_upgrade(tmp_path):
    from src.evaluation.qa.artifacts import write_json

    catalog = EvaluationCatalog(tmp_path)
    metadata, _ = catalog.import_dataset("Legacy", "legacy.json", _dataset_blob())
    directory = catalog.datasets_dir / metadata["id"]
    legacy_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "schema_version",
            "dataset_role",
            "publication_schema",
            "based_on_child_id",
            "generation_scope",
            "approval_actor",
            "approval_time",
            "contains_live_answers",
        }
    }
    write_json(directory / "metadata.json", legacy_metadata)
    (directory / "integrity.json").unlink()

    upgraded = catalog.get_dataset(metadata["id"])

    assert upgraded == {
        **legacy_metadata,
        "schema_version": "qa-dataset-v1",
        "dataset_role": "legacy",
        "publication_schema": "qa-dataset-v1",
        "based_on_child_id": None,
        "generation_scope": None,
        "approval_actor": None,
        "approval_time": None,
        "contains_live_answers": False,
    }
    assert (directory / "integrity.json").is_file()


def test_catalog_rejects_integrity_bound_metadata_with_invalid_semantic_role(tmp_path):
    from src.evaluation.qa.artifacts import read_json, write_json

    catalog = EvaluationCatalog(tmp_path)
    metadata, _ = catalog.import_dataset("Dataset", "dataset.json", _dataset_blob())
    directory = catalog.datasets_dir / metadata["id"]
    invalid = read_json(directory / "metadata.json")
    invalid["dataset_role"] = "definition"
    write_json(directory / "metadata.json", invalid)
    catalog._write_dataset_integrity(directory, f"source.{metadata['format']}")

    with pytest.raises(ValueError, match="invalid role"):
        catalog.get_dataset(metadata["id"])


def test_dataset_import_is_immutable_and_content_deduplicated(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    first, created = catalog.import_dataset("One", "set.json", _dataset_blob())
    second, duplicated = catalog.import_dataset("Two", "another.json", _dataset_blob())

    assert created is True
    assert duplicated is False
    assert second["id"] == first["id"]
    assert catalog.dataset_path(first["id"]).read_bytes() == _dataset_blob()


def test_atom_review_creates_new_dataset_and_preserves_parent_bytes(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset("Parent", "set.json", _dataset_blob())
    parent_path = catalog.dataset_path(parent["id"])
    before = parent_path.read_bytes()
    draft = catalog.create_atom_draft(parent["id"], "builtin", _Evaluator())

    assert draft["items"][0]["status"] == "prepared"
    assert draft["items"][0]["answer_source"] == "static_docs_tickets"
    assert draft["items"][1]["status"] == "skipped_time_sensitive"

    child = catalog.save_reviewed_dataset(
        draft["id"],
        "Reviewed",
        [
            {
                "item_id": "eligible",
                "atoms": [{"id": "gold-1", "text": "Double checked", "required": True}],
            }
        ],
    )

    assert child["id"] != parent["id"]
    assert child["parent_dataset_id"] == parent["id"]
    assert child["supplied_atom_item_count"] == 1
    assert parent_path.read_bytes() == before
    child_items = load_dataset(catalog.dataset_path(child["id"]))[1]
    assert child_items[0].expected_atoms[0].text == "Double checked"
    assert child_items[1].expected_atoms is None


def test_pre_upgrade_inline_atom_draft_can_still_be_saved(tmp_path):
    from src.evaluation.qa.artifacts import write_json

    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset("Parent", "set.json", _dataset_blob())
    draft = catalog.create_atom_draft(parent["id"], "builtin", _Evaluator())
    draft_path = catalog.drafts_dir / draft["id"] / "draft.json"
    legacy = {
        key: value
        for key, value in draft.items()
        if key
        in {
            "id",
            "dataset_id",
            "dataset_name",
            "profile_id",
            "status",
            "created_at",
            "items",
        }
    }
    write_json(draft_path, legacy)
    (draft_path.parent / "items.jsonl").unlink()

    child = catalog.save_reviewed_dataset(
        draft["id"],
        "Legacy reviewed",
        [{"item_id": "eligible", "atoms": draft["items"][0]["atoms"]}],
    )

    assert child["parent_dataset_id"] == parent["id"]
    assert child["dataset_role"] == "approved_child"


def test_review_draft_preserves_existing_atoms_and_leaves_missing_atoms_empty(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    dataset, _ = catalog.import_dataset(
        "Partial", "partial.json", _partially_atomized_dataset_blob()
    )

    draft = catalog.create_atom_review_draft(dataset["id"])

    assert dataset["atom_count"] == 1
    assert draft["profile_id"] is None
    assert draft["items"] == [
        {
            "item_id": "with-atom",
            "question": "Answered?",
            "answer": "Yes",
            "time_sensitive": False,
            "status": "prepared",
            "answer_source": "existing_source",
            "atom_source": "supplied",
            "atoms": [{"id": "A1", "text": "The answer is yes.", "required": True}],
        },
        {
            "item_id": "without-atom",
            "question": "Missing?",
            "answer": "Not yet",
            "time_sensitive": False,
            "status": "missing_atoms",
            "answer_source": "missing_source",
            "atoms": [],
        },
        {
            "item_id": "changing",
            "question": "Current?",
            "answer": "Changes",
            "time_sensitive": True,
            "status": "skipped_time_sensitive",
            "answer_source": "changing_source",
            "atoms": [],
        },
    ]


def test_review_save_preserves_existing_atoms_on_skipped_items(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset(
        "Skipped atom", "skipped.json", _skipped_atom_dataset_blob()
    )
    draft = catalog.create_atom_review_draft(parent["id"])

    child = catalog.save_reviewed_dataset(
        draft["id"],
        "Reviewed",
        [
            {
                "item_id": "eligible",
                "atoms": [
                    {
                        "id": "reviewed-A1",
                        "text": "The eligible answer was reviewed.",
                        "required": True,
                    }
                ],
            }
        ],
    )

    child_items = load_dataset(catalog.dataset_path(child["id"]))[1]
    assert child["atom_count"] == 2
    assert child_items[0].expected_atoms[0].id == "reviewed-A1"
    assert child_items[1].expected_atoms[0].to_dict() == {
        "id": "current-A1",
        "text": "The current answer changes.",
        "required": True,
    }


def test_generation_is_rejected_when_dataset_contains_any_atom(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    dataset, _ = catalog.import_dataset(
        "Partial", "partial.json", _partially_atomized_dataset_blob()
    )

    with pytest.raises(
        ValueError, match="atom generation requires a dataset with zero atoms"
    ):
        catalog.create_atom_draft(dataset["id"], "builtin", _Evaluator())


def test_review_is_rejected_when_dataset_contains_zero_atoms(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    dataset, _ = catalog.import_dataset("Empty", "empty.json", _dataset_blob())

    with pytest.raises(
        ValueError, match="atom review requires a dataset with at least one atom"
    ):
        catalog.create_atom_review_draft(dataset["id"])


def test_atom_retry_updates_only_failed_rows_in_the_same_open_draft(
    monkeypatch, tmp_path
):
    catalog = EvaluationCatalog(tmp_path)
    dataset, _ = catalog.import_dataset(
        "Two items", "two.json", _two_item_dataset_blob()
    )
    parent_bytes = catalog.dataset_path(dataset["id"]).read_bytes()
    initial = _SelectiveEvaluator(failing_answers={"Second answer"})
    draft = catalog.create_atom_draft(dataset["id"], "builtin", initial)
    first_before = draft["items"][0]
    retry = _SelectiveEvaluator()
    monkeypatch.setattr(
        catalog,
        "dataset_items",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("atom retry must stream the dataset")
        ),
    )

    retried = catalog.retry_failed_atom_items(draft["id"], retry)

    assert retried["id"] == draft["id"]
    assert retried["items"][0] == first_before
    assert retried["items"][1] == {
        "item_id": "second",
        "question": "Second?",
        "answer": "Second answer",
        "time_sensitive": False,
        "status": "prepared",
        "atom_source": "inferred",
        "atoms": [
            {
                "id": "A1",
                "text": "Generated: Second answer",
                "required": True,
            }
        ],
    }
    assert retry.calls == [("Second?", "Second answer")]
    assert catalog.dataset_path(dataset["id"]).read_bytes() == parent_bytes


def test_atom_retry_rejects_drafts_without_retryable_generation_failures(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    dataset, _ = catalog.import_dataset(
        "Two items", "two.json", _two_item_dataset_blob()
    )
    generated = catalog.create_atom_draft(dataset["id"], "builtin", _Evaluator())

    with pytest.raises(ValueError, match="no failed items"):
        catalog.atom_retry_details(generated["id"])

    reviewed_parent, _ = catalog.import_dataset(
        "Reviewed", "reviewed.json", _partially_atomized_dataset_blob()
    )
    reviewed = catalog.create_atom_review_draft(reviewed_parent["id"])
    with pytest.raises(ValueError, match="only generated atom drafts"):
        catalog.atom_retry_details(reviewed["id"])


def test_each_saved_review_has_its_own_parent_lineage(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset("Parent", "set.json", _dataset_blob())
    reviewed_items = [
        {
            "item_id": "eligible",
            "atoms": [{"id": "gold-1", "text": "Double checked", "required": True}],
        }
    ]

    first_draft = catalog.create_atom_draft(parent["id"], "builtin", _Evaluator())
    first = catalog.save_reviewed_dataset(
        first_draft["id"], "Reviewed one", reviewed_items
    )
    second_draft = catalog.create_atom_draft(parent["id"], "builtin", _Evaluator())
    second = catalog.save_reviewed_dataset(
        second_draft["id"], "Reviewed two", reviewed_items
    )

    assert second["id"] != first["id"]
    assert first["parent_dataset_id"] == parent["id"]
    assert second["parent_dataset_id"] == parent["id"]


def test_invalid_review_leaves_draft_open_and_creates_no_child(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    parent, _ = catalog.import_dataset("Parent", "set.json", _dataset_blob())
    draft = catalog.create_atom_draft(parent["id"], "builtin", _Evaluator())

    with pytest.raises(ValueError, match="at least one required atom"):
        catalog.save_reviewed_dataset(
            draft["id"],
            "Invalid",
            [
                {
                    "item_id": "eligible",
                    "atoms": [
                        {"id": "optional", "text": "Only optional", "required": False}
                    ],
                }
            ],
        )

    assert catalog.get_atom_draft(draft["id"])["status"] == "open"
    assert len(catalog.list_datasets()) == 1


def test_removed_dataset_fields_are_rejected(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    legacy = json.dumps(
        [{"question": "Q", "expected_answer": "A", "freshness": "static"}]
    ).encode()

    with pytest.raises(ValueError, match="expected_answer.*freshness"):
        catalog.import_dataset("Legacy", "legacy.json", legacy)


def test_profile_catalog_contains_builtin_and_validates_imports(tmp_path):
    catalog = EvaluationCatalog(tmp_path)
    blob = b"""
version: 1
qa:
  atoms_extractor:
    provider: local
    model: atom-model
  evaluator:
    provider: local
    model: judge-model
"""
    profile, created = catalog.import_profile("Local", "local.yaml", blob)

    assert created is True
    assert profile["components"]["evaluator"]["model"] == "judge-model"
    assert {item["id"] for item in catalog.list_profiles()} == {
        "builtin",
        profile["id"],
    }
