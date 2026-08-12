# isort: off
import pytest

from src.evaluation.qa.artifacts import write_jsonl
from src.evaluation.qa.preparation import (
    PreparationRecord,
    iter_preparation_records,
    load_preparation_records,
    prepare_dataset_item,
    prepare_dataset_items,
)
from src.evaluation.qa.validation import Atom, DatasetItem

# isort: on


def _item(
    item_id,
    *,
    time_sensitive=False,
    expected_atoms=None,
):
    return DatasetItem(
        id=item_id,
        question=f"Question {item_id}",
        answer=f"Answer {item_id}",
        time_sensitive=time_sensitive,
        category="category",
        answer_mode="direct_answer",
        answer_source="source",
        expected_atoms=expected_atoms,
    )


class _Extractor:
    def extract_gold(self, question, answer):
        if answer.endswith("failed"):
            raise RuntimeError("extraction failed")
        return {
            "atoms": [
                {
                    "id": "A1",
                    "text": answer,
                    "required": True,
                }
            ]
        }


def _record_fields(item, *, prepared=False):
    fields = {
        "item_id": item.id,
        "category": item.category,
        "answer_mode": item.answer_mode,
        "answer_source": item.answer_source,
    }
    if prepared:
        fields.update(
            {
                "question": item.question,
                "answer": item.answer,
                "time_sensitive": False,
            }
        )
    return fields


class TestPreparationRecord:
    @pytest.mark.parametrize(
        "record, message",
        [
            (
                lambda: PreparationRecord(
                    **_record_fields(_item("prepared"), prepared=True),
                    status="prepared",
                    gold_atoms=None,
                    atom_source="inferred",
                ),
                "requires gold atoms",
            ),
            (
                lambda: PreparationRecord(
                    **_record_fields(_item("failed")),
                    status="preparation_failed",
                ),
                "requires an error",
            ),
            (
                lambda: PreparationRecord(
                    **_record_fields(_item("skipped", time_sensitive=True)),
                    status="skipped_time_sensitive",
                    error="unexpected",
                ),
                "cannot contain output",
            ),
            (
                lambda: PreparationRecord(
                    **_record_fields(_item("unknown", time_sensitive=True)),
                    status="unknown",
                ),
                "unsupported preparation status",
            ),
            (
                lambda: PreparationRecord(
                    **_record_fields(_item("prepared"), prepared=True),
                    status="prepared",
                    gold_atoms=(Atom(id="A1", text="answer", required=True),),
                    atom_source="unknown",
                ),
                "requires an atom source",
            ),
            (
                lambda: PreparationRecord(
                    **_record_fields(_item("prepared"), prepared=True),
                    status="prepared",
                    gold_atoms=(Atom(id="A1", text="answer", required=True),),
                    atom_source="inferred",
                    error="unexpected",
                ),
                "cannot contain an error",
            ),
            (
                lambda: PreparationRecord(
                    **_record_fields(_item("failed")),
                    status="preparation_failed",
                    question="unexpected",
                    error="unexpected",
                ),
                "cannot contain prepared output",
            ),
        ],
    )
    def test_rejects_status_field_mismatches(self, record, message):
        with pytest.raises(ValueError, match=message):
            record()

    def test_prepares_one_terminal_record_per_input(self):
        supplied_atom = Atom(id="S1", text="supplied", required=True)
        items = [
            _item("supplied", expected_atoms=[supplied_atom]),
            _item("inferred"),
            _item("skipped", time_sensitive=True),
            _item("failed"),
        ]

        records = prepare_dataset_items(items, _Extractor())

        assert [record.status for record in records] == [
            "prepared",
            "prepared",
            "skipped_time_sensitive",
            "preparation_failed",
        ]
        assert records[0].atom_source == "supplied"
        assert records[0].prepared_gold_atoms == (supplied_atom,)
        assert records[1].atom_source == "inferred"
        assert records[2].to_dict() == {
            "item_id": "skipped",
            "status": "skipped_time_sensitive",
            "category": "category",
            "answer_mode": "direct_answer",
            "answer_source": "source",
        }
        assert records[3].error == "extraction failed"

    def test_prepares_one_item_without_accumulating_a_collection(self):
        record = prepare_dataset_item(_item("inferred"), _Extractor())

        assert record.status == "prepared"
        assert record.item_id == "inferred"
        assert record.atom_source == "inferred"


class TestPreparationArtifact:
    def test_round_trips_records_without_loading_the_input_snapshot(self, tmp_path):
        supplied_atom = Atom(id="S1", text="supplied", required=True)
        items = [
            _item("supplied", expected_atoms=[supplied_atom]),
            _item("skipped", time_sensitive=True),
        ]
        records = prepare_dataset_items(items, _Extractor())
        path = tmp_path / "preparation.jsonl"
        write_jsonl(path, (record.to_dict() for record in records))

        loaded = load_preparation_records(path, expected_count=2)

        assert loaded == records

    def test_rejects_duplicate_item_records(self, tmp_path):
        item = _item("item", time_sensitive=True)
        row = PreparationRecord(
            **_record_fields(item),
            status="skipped_time_sensitive",
        ).to_dict()
        path = tmp_path / "preparation.jsonl"
        write_jsonl(path, [row, row])

        with pytest.raises(ValueError, match="duplicate item IDs"):
            load_preparation_records(path, expected_count=2)

    def test_rejects_fields_inconsistent_with_status(self, tmp_path):
        item = _item("item", time_sensitive=True)
        row = PreparationRecord(
            **_record_fields(item),
            status="skipped_time_sensitive",
        ).to_dict()
        row["error"] = "not allowed"
        path = tmp_path / "preparation.jsonl"
        write_jsonl(path, [row])

        with pytest.raises(ValueError, match="invalid fields"):
            load_preparation_records(path, expected_count=1)

    def test_rejects_non_normalized_prepared_text(self, tmp_path):
        record = prepare_dataset_item(_item("item"), _Extractor())
        row = record.to_dict()
        row["question"] = "Question\r\nitem"
        path = tmp_path / "preparation.jsonl"
        write_jsonl(path, [row])

        with pytest.raises(ValueError, match="normalized newlines"):
            load_preparation_records(path, expected_count=1)

    def test_rejects_record_count_that_disagrees_with_manifest(self, tmp_path):
        item = _item("item", time_sensitive=True)
        path = tmp_path / "preparation.jsonl"
        write_jsonl(
            path,
            [
                PreparationRecord(
                    **_record_fields(item),
                    status="skipped_time_sensitive",
                ).to_dict()
            ],
        )

        with pytest.raises(ValueError, match="exactly one row per input item"):
            load_preparation_records(path, expected_count=2)

        with pytest.raises(ValueError, match="exactly one row per input item"):
            list(iter_preparation_records(path, expected_count=2))

    def test_preserves_authoritative_artifact_order(self, tmp_path):
        first = _item("first", time_sensitive=True)
        second = _item("second", time_sensitive=True)
        path = tmp_path / "preparation.jsonl"
        write_jsonl(
            path,
            [
                PreparationRecord(
                    **_record_fields(second),
                    status="skipped_time_sensitive",
                ).to_dict(),
                PreparationRecord(
                    **_record_fields(first),
                    status="skipped_time_sensitive",
                ).to_dict(),
            ],
        )

        records = load_preparation_records(path, expected_count=2)

        assert [record.item_id for record in records] == ["second", "first"]

    def test_iterates_records_lazily(self, tmp_path):
        first = prepare_dataset_item(_item("first"), _Extractor())
        second = prepare_dataset_item(_item("second"), _Extractor())
        path = tmp_path / "preparation.jsonl"
        write_jsonl(path, [first.to_dict(), second.to_dict()])

        records = iter_preparation_records(path)

        assert next(records) == first
        assert next(records) == second
        with pytest.raises(StopIteration):
            next(records)
