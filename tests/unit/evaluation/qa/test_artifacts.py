# isort: skip_file
import io
import json
from pathlib import Path

import pytest

from src.evaluation.qa.artifacts import AtomicJsonlWriter, copy_file_atomic


class TestAtomicJsonlWriter:
    def test_replaces_complete_artifact(self, tmp_path):
        path = tmp_path / "rows.jsonl"

        with AtomicJsonlWriter(path) as writer:
            writer.write({"id": 1})
            writer.write({"id": 2})

        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"id": 1},
            {"id": 2},
        ]

    def test_failure_preserves_previous_artifact(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        path.write_text('{"old": true}\n')

        with pytest.raises(RuntimeError, match="stop"):
            with AtomicJsonlWriter(path) as writer:
                writer.write({"new": True})
                raise RuntimeError("stop")

        assert path.read_text() == '{"old": true}\n'


def test_copy_file_atomic_copies_multiple_chunks_exactly(tmp_path):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    payload = (b"a" * (1024 * 1024)) + (b"b" * 17)
    source.write_bytes(payload)

    copy_file_atomic(source, target)

    assert target.read_bytes() == payload


def test_copy_file_atomic_never_requests_the_complete_source(monkeypatch, tmp_path):
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    payload = b"bounded copy"
    source.write_bytes(payload)
    requested_sizes = []
    real_open = Path.open

    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            if size is None or size < 0:
                raise AssertionError("snapshot copying must use bounded reads")
            requested_sizes.append(size)
            return super().read(size)

    def tracked_open(path, *args, **kwargs):
        if path == source:
            return BoundedReader(payload)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    copy_file_atomic(source, target)

    assert target.read_bytes() == payload
    assert requested_sizes
    assert set(requested_sizes) == {1024 * 1024}


def test_copy_file_atomic_preserves_target_when_source_read_fails(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"previous")

    with pytest.raises(FileNotFoundError):
        copy_file_atomic(tmp_path / "missing.jsonl", target)

    assert target.read_bytes() == b"previous"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["target.jsonl"]
