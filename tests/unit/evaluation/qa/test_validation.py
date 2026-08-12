# isort: skip_file
import io
import json
from pathlib import Path

import pytest

from src.evaluation.qa.validation import (
    Atom,
    iter_dataset_items,
    validate_dataset_rows,
    validate_judgments,
)  # isort: skip


class TestDatasetValidation:
    def test_rejects_empty_dataset(self):
        with pytest.raises(ValueError, match="at least one row"):
            validate_dataset_rows([])

    def test_accepts_strict_row_and_derives_stable_id(self):
        rows = [
            {
                "question": "What is the quota?\r\n",
                "answer": "2.8 TB\r",
                "time_sensitive": False,
                "category": "storage",
                "answer_source": "static_docs_tickets",
                "expected_atoms": [
                    {"id": "quota", "text": "The quota is 2.8 TB", "required": True}
                ],
            }
        ]

        first = validate_dataset_rows(rows)[0]
        second = validate_dataset_rows(rows)[0]

        assert first.id == second.id
        assert first.question == "What is the quota?\n"
        assert first.answer == "2.8 TB\n"
        assert first.category == "storage"
        assert first.answer_source == "static_docs_tickets"
        assert first.expected_atoms == [
            Atom(id="quota", text="The quota is 2.8 TB", required=True)
        ]

    def test_supplied_atom_text_is_not_rewritten(self):
        item = validate_dataset_rows(
            [
                {
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "g1", "text": "line one\r\nline two", "required": True}
                    ],
                }
            ]
        )[0]

        assert item.expected_atoms[0].text == "line one\r\nline two"

    @pytest.mark.parametrize(
        "row,error",
        [
            (
                {
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": "false",
                },
                "time_sensitive must be a boolean",
            ),
            (
                {
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": False,
                    "expected_answer": "legacy",
                },
                r"unknown field\(s\): expected_answer",
            ),
            (
                {
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": False,
                    "expected_atoms": [],
                },
                "at least one atom",
            ),
            (
                {
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": False,
                    "expected_atoms": [
                        {"id": "optional", "text": "A", "required": False}
                    ],
                },
                "at least one required atom",
            ),
            (
                {
                    "id": "invalid-unicode",
                    "question": "\ud800",
                    "answer": "A",
                    "time_sensitive": False,
                },
                "valid Unicode scalar values",
            ),
        ],
    )
    def test_rejects_invalid_rows(self, row, error):
        with pytest.raises(ValueError, match=error):
            validate_dataset_rows([row])

    def test_rejects_duplicate_derived_ids(self):
        row = {"question": "Q", "answer": "A", "time_sensitive": False}
        with pytest.raises(ValueError, match="duplicate or colliding id"):
            validate_dataset_rows([row, row])

    @pytest.mark.parametrize("suffix", [".json", ".jsonl"])
    def test_streams_rows_lazily(self, suffix, tmp_path):
        path = tmp_path / f"dataset{suffix}"
        valid = {
            "id": "first",
            "question": "Q",
            "answer": "A",
            "time_sensitive": False,
        }
        invalid = {
            "id": "second",
            "question": "Q2",
            "answer": "A2",
            "time_sensitive": False,
            "unknown": "field",
        }
        if suffix == ".json":
            path.write_text(json.dumps([valid, invalid]), encoding="utf-8")
        else:
            path.write_text(
                json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
                encoding="utf-8",
            )

        items = iter_dataset_items(path)

        assert next(items).id == "first"
        with pytest.raises(ValueError, match="unknown field.*unknown"):
            next(items)

    @pytest.mark.parametrize("suffix", [".json", ".jsonl"])
    def test_streaming_rejects_duplicate_ids_across_rows(self, suffix, tmp_path):
        path = tmp_path / f"dataset{suffix}"
        row = {
            "id": "duplicate",
            "question": "Q",
            "answer": "A",
            "time_sensitive": False,
        }
        if suffix == ".json":
            path.write_text(json.dumps([row, row]), encoding="utf-8")
        else:
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )

        with pytest.raises(ValueError, match="duplicate or colliding id"):
            list(iter_dataset_items(path))

    def test_streams_multiline_json_array_items(self, tmp_path):
        path = tmp_path / "dataset.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "first",
                        "question": "Q",
                        "answer": "A",
                        "time_sensitive": False,
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        assert [item.id for item in iter_dataset_items(path)] == ["first"]

    def test_json_streaming_never_requests_the_complete_file(
        self, monkeypatch, tmp_path
    ):
        path = tmp_path / "dataset.json"
        payload = json.dumps(
            [
                {
                    "id": "first",
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": False,
                }
            ]
        ).encode("utf-8")
        path.write_bytes(payload)
        requested_sizes = []
        real_open = Path.open

        class BoundedReader(io.BytesIO):
            def read(self, size=-1):
                if size is None or size < 0:
                    raise AssertionError("JSON parsing must use bounded reads")
                requested_sizes.append(size)
                return super().read(size)

        def tracked_open(opened_path, *args, **kwargs):
            if opened_path == path:
                return BoundedReader(payload)
            return real_open(opened_path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", tracked_open)

        assert [item.id for item in iter_dataset_items(path)] == ["first"]
        assert requested_sizes
        assert all(size <= 64 * 1024 for size in requested_sizes)

    def test_json_streaming_yields_before_a_malformed_later_item(self, tmp_path):
        path = tmp_path / "dataset.json"
        first = json.dumps(
            {
                "id": "first",
                "question": "Q",
                "answer": "A",
                "time_sensitive": False,
            }
        )
        path.write_text(f'[{first},{{"broken":]', encoding="utf-8")
        items = iter_dataset_items(path)

        assert next(items).id == "first"
        with pytest.raises(ValueError, match="invalid JSON dataset"):
            next(items)

    def test_jsonl_streaming_consumes_one_line_per_yield(self, monkeypatch, tmp_path):
        path = tmp_path / "dataset.jsonl"
        rows = [
            json.dumps(
                {
                    "id": "first",
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": False,
                }
            )
            + "\n",
            json.dumps(
                {
                    "id": "second",
                    "question": "Q2",
                    "answer": "A2",
                    "time_sensitive": False,
                    "unknown": "field",
                }
            )
            + "\n",
        ]
        path.write_text("".join(rows), encoding="utf-8")
        consumed_lines = []
        real_open = Path.open

        class LineReader:
            def __init__(self):
                self._lines = iter(rows)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                line = next(self._lines)
                consumed_lines.append(line)
                return line

            def read(self, *args, **kwargs):
                raise AssertionError("JSONL parsing must iterate over lines")

            def readlines(self, *args, **kwargs):
                raise AssertionError("JSONL parsing must not collect all lines")

        def tracked_open(opened_path, *args, **kwargs):
            if opened_path == path:
                return LineReader()
            return real_open(opened_path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", tracked_open)
        items = iter_dataset_items(path)

        assert next(items).id == "first"
        assert consumed_lines == rows[:1]
        with pytest.raises(ValueError, match="unknown field.*unknown"):
            next(items)
        assert consumed_lines == rows

    def test_streaming_json_requires_an_array(self, tmp_path):
        path = tmp_path / "dataset.json"
        path.write_text(
            json.dumps(
                {
                    "id": "first",
                    "question": "Q",
                    "answer": "A",
                    "time_sensitive": False,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="JSON must contain an array"):
            list(iter_dataset_items(path))

    @pytest.mark.parametrize("suffix", [".json", ".jsonl"])
    def test_streaming_rejects_unmatched_unicode_surrogates(self, suffix, tmp_path):
        path = tmp_path / f"dataset{suffix}"
        row = (
            '{"id":"first","question":"\\ud800","answer":"A",' '"time_sensitive":false}'
        )
        path.write_text(f"[{row}]" if suffix == ".json" else f"{row}\n")

        with pytest.raises(ValueError, match="valid Unicode scalar values"):
            list(iter_dataset_items(path))

    @pytest.mark.parametrize(
        ("suffix", "contents", "error"),
        [
            (".json", "[]", "at least one row"),
            (".jsonl", "\n\n", "at least one row"),
            (
                ".json",
                '[{"id":"first","question":"Q","answer":"A",'
                '"time_sensitive":false}] trailing',
                "invalid JSON dataset",
            ),
            (".jsonl", "{invalid}\n", "invalid JSONL at line 1"),
        ],
    )
    def test_streaming_rejects_empty_or_malformed_input(
        self, suffix, contents, error, tmp_path
    ):
        path = tmp_path / f"dataset{suffix}"
        path.write_text(contents, encoding="utf-8")

        with pytest.raises(ValueError, match=error):
            list(iter_dataset_items(path))


class TestJudgmentValidation:
    def test_rejects_response_atom_evidence_fields(self):
        gold = [Atom(id="g1", text="gold", required=True)]
        raw = {
            "judgments": [
                {
                    "atom_id": "g1",
                    "outcome": "entailed",
                    "supporting_response_atom_ids": ["r1"],
                    "rationale": "match",
                }
            ]
        }

        with pytest.raises(
            ValueError, match=r"unknown field\(s\): supporting_response_atom_ids"
        ):
            validate_judgments(raw, gold_atoms=gold, context="comparison")


class TestJudgmentValidationCompleteness:
    def test_rejects_missing_gold_judgment(self):
        gold = [
            Atom(id="g1", text="one", required=True),
            Atom(id="g2", text="two", required=False),
        ]
        with pytest.raises(ValueError, match="missing judgment.*g2"):
            validate_judgments(
                {
                    "judgments": [
                        {
                            "atom_id": "g1",
                            "outcome": "not_mentioned",
                            "rationale": "missing",
                        }
                    ]
                },
                gold_atoms=gold,
                context="comparison",
            )

    def test_rejects_empty_rationale_for_any_outcome(self):
        with pytest.raises(ValueError, match="rationale must be a non-empty string"):
            validate_judgments(
                {
                    "judgments": [
                        {
                            "atom_id": "g1",
                            "outcome": "not_mentioned",
                            "rationale": "",
                        }
                    ]
                },
                gold_atoms=[Atom(id="g1", text="gold", required=True)],
                context="comparison",
            )
