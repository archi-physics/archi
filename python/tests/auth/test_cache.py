"""req.w2.auth — cache resolve/hash/probe behavior, offline."""
import json
import time
from pathlib import Path

from archi.auth import cache


def test_resolve_absolute_path_passes_through(tmp_path):
    target = tmp_path / "x.json"
    assert cache.resolve_repo_path(target) == target


def test_resolve_with_explicit_base(tmp_path):
    resolved = cache.resolve_repo_path("data/a.json", base=tmp_path)
    assert resolved == tmp_path / "data" / "a.json"


def test_resolve_env_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv(cache.DATA_ROOT_ENV, str(tmp_path))
    assert cache.resolve_repo_path("a.json") == tmp_path / "a.json"


def test_explicit_base_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv(cache.DATA_ROOT_ENV, str(tmp_path / "env"))
    resolved = cache.resolve_repo_path("a.json", base=tmp_path / "param")
    assert resolved == tmp_path / "param" / "a.json"


def test_resolve_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv(cache.DATA_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert cache.resolve_repo_path("a.json") == tmp_path / "a.json"


def test_load_json(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"k": [1, 2]}))
    assert cache.load_json("a.json", base=tmp_path) == {"k": [1, 2]}


def test_content_hash_stable_and_order_insensitive(tmp_path):
    (tmp_path / "a.json").write_text("[1]")
    (tmp_path / "b.json").write_text("[2]")
    h1 = cache.content_hash(["a.json", "b.json"], base=tmp_path)
    h2 = cache.content_hash(["b.json", "a.json"], base=tmp_path)
    assert h1 == h2
    assert len(h1) == 64


def test_content_hash_changes_with_content(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("[1]")
    before = cache.content_hash(["a.json"], base=tmp_path)
    path.write_text("[1, 2]")
    assert cache.content_hash(["a.json"], base=tmp_path) != before


def test_content_hash_is_label_sensitive(tmp_path):
    (tmp_path / "a.json").write_text("[1]")
    (tmp_path / "b.json").write_text("[1]")
    assert cache.content_hash(["a.json"], base=tmp_path) != cache.content_hash(
        ["b.json"], base=tmp_path
    )


def test_content_hash_change_probe_tracks_file_content(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("[1]")
    probe = cache.content_hash_change_probe(
        cache_paths=["a.json", "missing.json"],
        config={"cache_path": "a.json"},
        emit_targets=test_content_hash_change_probe_tracks_file_content,
        base=tmp_path,
    )
    before = probe.build_token()
    assert probe.build_token() == before
    path.write_text("[1, 2]")
    assert probe.build_token() != before


def test_forced_live_probe_without_cache_always_advances(tmp_path):
    probe = cache.cache_or_forced_live_change_probe(
        cache_paths=["missing.json"],
        config={},
        emit_targets=test_forced_live_probe_without_cache_always_advances,
        base=tmp_path,
    )
    first = probe.build_token()
    time.sleep(0.002)
    assert probe.build_token() != first


def test_forced_live_probe_with_cache_uses_content_hash(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("[1]")
    probe = cache.cache_or_forced_live_change_probe(
        cache_paths=["a.json"],
        config={},
        emit_targets=test_forced_live_probe_with_cache_uses_content_hash,
        base=tmp_path,
    )
    first = probe.build_token()
    time.sleep(0.002)
    assert probe.build_token() == first
    path.write_text("[2]")
    assert probe.build_token() != first


# --- stream-content-hash regressions ---


def _whole_file_content_hash(paths, *, base):
    # The algorithm content_hash used before it streamed: the reference the
    # streamed digest must match byte for byte, so stored values hold.
    import hashlib

    digest = hashlib.sha256()
    resolved = sorted((str(p), cache.resolve_repo_path(p, base=base)) for p in paths)
    for label, path in resolved:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_test_files(tmp_path):
    chunk = 1024 * 1024  # the streaming chunk size
    files = {
        "empty.json": b"",
        "small.json": b'[{"name": "x"}]',
        # Exactly two chunks, and one that ends mid-chunk.
        "aligned.bin": bytes(range(256)) * (2 * chunk // 256),
        "large.bin": (b"0123456789abcdef" * (chunk // 16)) * 3 + b"tail",
    }
    for name, body in files.items():
        (tmp_path / name).write_bytes(body)
    return list(files)


def test_streamed_content_hash_equals_the_whole_file_hash(tmp_path):
    names = _write_test_files(tmp_path)
    for subset in (names, ["empty.json"], ["large.bin"], ["small.json", "aligned.bin"]):
        assert cache.content_hash(subset, base=tmp_path) == _whole_file_content_hash(
            subset, base=tmp_path
        )


def test_content_hash_never_reads_a_whole_file(tmp_path, monkeypatch):
    names = _write_test_files(tmp_path)
    expected = _whole_file_content_hash(names, base=tmp_path)

    def refuse(self):
        raise AssertionError(f"content_hash read {self} whole")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert cache.content_hash(names, base=tmp_path) == expected


def test_content_hash_peak_memory_stays_far_below_the_file_size(tmp_path):
    import tracemalloc

    size = 50 * 1024 * 1024
    block = bytes(range(256)) * 4096  # 1 MiB
    with (tmp_path / "big.bin").open("wb") as handle:
        for _ in range(size // len(block)):
            handle.write(block)
    tracemalloc.start()
    try:
        cache.content_hash(["big.bin"], base=tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024, peak
