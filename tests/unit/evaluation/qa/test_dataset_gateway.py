# isort: skip_file
import json

import pytest

from src.evaluation.qa.dataset import (
    DATASET_V2_SCHEMA_VERSION,
    DatasetGateway,
    DatasetItemState,
    DatasetSchemaVersion,
)


class TestDatasetGateway:
    @pytest.mark.parametrize("suffix", [".json", ".jsonl"])
    def test_reads_v2_static_and_unresolved_live(self, tmp_path, suffix):
        rows = [
            {
                "id": "static",
                "question": "Fixed?",
                "answer": "Yes",
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
                        }
                    ],
                },
            },
        ]
        path = tmp_path / f"dataset{suffix}"
        if suffix == ".json":
            path.write_text(
                json.dumps(
                    {"schema_version": DATASET_V2_SCHEMA_VERSION, "items": rows}
                ),
                encoding="utf-8",
            )
        else:
            path.write_text(
                "\n".join(
                    [json.dumps({"schema_version": DATASET_V2_SCHEMA_VERSION})]
                    + [json.dumps(row) for row in rows]
                ),
                encoding="utf-8",
            )

        dataset = DatasetGateway().read(path)
        items = list(dataset)

        assert dataset.schema_version is DatasetSchemaVersion.V2
        assert [item.state for item in items] == [
            DatasetItemState.STATIC,
            DatasetItemState.UNRESOLVED_LIVE,
        ]
        assert items[1].oracle.calls[0].server == "read-model"

    def test_rejects_unknown_explicit_version(self, tmp_path):
        path = tmp_path / "dataset.json"
        path.write_text(
            json.dumps({"schema_version": "qa-dataset-v3", "items": [{}]}),
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="unsupported dataset schema_version 'qa-dataset-v3'"
        ):
            DatasetGateway().read(path)

    def test_external_materialized_live_requires_trusted_catalog(self, tmp_path):
        row = {
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
                    }
                ],
            },
            "answer": {"lookup": {"value": 7}},
            "expected_atoms": [
                {"id": "value", "text": "The value is 7", "required": True}
            ],
        }
        path = tmp_path / "dataset.json"
        path.write_text(
            json.dumps({"schema_version": DATASET_V2_SCHEMA_VERSION, "items": [row]}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="not portable"):
            list(DatasetGateway().read(path))

        trusted = list(DatasetGateway().read(path, allow_materialized_live=True))
        assert trusted[0].state is DatasetItemState.MATERIALIZED_LIVE

    def test_rejects_duplicate_keys_in_json_items(self, tmp_path):
        path = tmp_path / "duplicate.json"
        path.write_text(
            '[{"id":"one","question":"Q","question":"hidden",'
            '"answer":"A","time_sensitive":false}]',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="duplicate key 'question'"):
            list(DatasetGateway().read(path))

    def test_v1_time_sensitive_row_retains_legacy_state(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            json.dumps(
                {"question": "Old live?", "answer": "Old", "time_sensitive": True}
            ),
            encoding="utf-8",
        )

        dataset = DatasetGateway().read(path)
        item = next(iter(dataset))

        assert dataset.schema_version is DatasetSchemaVersion.V1
        assert item.state is DatasetItemState.LEGACY_TIME_SENSITIVE
