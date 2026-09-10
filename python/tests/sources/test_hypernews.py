"""req.w2.sources-catalogs — HyperNewsSource emission from cache, offline."""
import hashlib
import json

from okg.deployment import (
    EdgeFact,
    NodeFact,
    PreflightResult,
)

from archi.sources.hypernews import HyperNewsSource

RECORDS = [
    {
        "thread_id": "comp-ops/123",
        "title": "Transfers stuck at T2_US_MIT",
        "url": "https://hypernews.cern.ch/HyperNews/CMS/get/comp-ops/123.html",
        "forum_name": "comp-ops",
        "body": "Transfers to T2_US_MIT are stuck since yesterday.",
        "author": "Ada Lovelace",
        "date": "2026-08-01",
        "reply_count": 3,
    },
]


def _source(tmp_path, **kwargs):
    root = tmp_path / "data" / "hypernews"
    root.mkdir(parents=True)
    (root / "records.json").write_text(json.dumps(RECORDS))
    return HyperNewsSource(base=str(tmp_path), **kwargs)


def test_thread_node_and_chunk_emission_from_cache(tmp_path):
    source = _source(tmp_path)
    run = source.run("run-1", mode="scope_complete")
    facts = list(run.facts)
    threads = [
        f for f in facts
        if isinstance(f, NodeFact) and f.subtype == "forum_thread"
    ]
    assert len(threads) == 1
    thread = threads[0]
    assert thread.node_id == "hn:comp-ops/123"
    assert thread.attrs["title"] == "Transfers stuck at T2_US_MIT"
    assert thread.attrs["forum_name"] == "comp-ops"
    assert thread.attrs["reply_count"] == 3
    chunks = [
        f for f in facts
        if isinstance(f, NodeFact) and f.subtype == "document_chunk"
    ]
    assert len(chunks) == 1
    chunk = chunks[0]
    text = chunk.attrs["text"]
    # Parity: the cms original hashes with a literal backslash-zero
    # separator (escaped f-string), like jira.py.
    expected = "chunk:" + hashlib.sha256(
        f"hn:comp-ops/123\\0{0}\\0{text}".encode("utf-8")
    ).hexdigest()[:16]
    assert chunk.node_id == expected
    assert chunk.attrs["chunker_name"] == "cms_hypernews_window_v1"
    assert chunk.attrs["heading_path"] == "Transfers stuck at T2_US_MIT"
    contains = [
        e for e in facts
        if isinstance(e, EdgeFact) and e.edge_type == "contains"
    ]
    assert [(e.src, e.dst) for e in contains] == [
        ("hn:comp-ops/123", chunk.node_id)
    ]
    assert run.health.mode == "cache"
    assert run.completed_scope is True


def test_reference_edges_with_configured_targets(tmp_path):
    (tmp_path / "data" / "cric").mkdir(parents=True)
    (tmp_path / "data" / "cric" / "sites.json").write_text(
        json.dumps({"T2_US_MIT": {}})
    )
    source = _source(tmp_path, sites_path="data/cric/sites.json")
    facts = list(source.run("run-1").facts)
    refs = [
        e for e in facts
        if isinstance(e, EdgeFact) and e.edge_type == "references"
    ]
    assert {(e.dst, e.attrs["match_type"]) for e in refs} == {
        ("site:T2_US_MIT", "cms_site"),
    }


def test_missing_cache_and_cookie_reports_not_ok_and_no_facts(tmp_path):
    source = HyperNewsSource(base=str(tmp_path), required=True)
    run = source.run("run-1", mode="reconcile")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status != "ok"


# --- circleback-fixes regressions: live-crawl and cache failure modes ---

LISTING = "\n".join([
    '<li value="1"><a href="/HyperNews/CMS/get/comp-ops/1.html">'
    "First thread</a></li>",
    '<li value="2"><a href="/HyperNews/CMS/get/comp-ops/2.html">'
    "Second thread</a></li>",
])
THREAD_PAGE = "<html><body>Thread body text</body></html>"


class _FakeResp:
    def __init__(self, text, status_code=200, url=""):
        self.text = text
        self.status_code = status_code
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _cookie_env(tmp_path, monkeypatch):
    cookie = tmp_path / "cookies.txt"
    cookie.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("HYPERNEWS_COOKIE_FILE", str(cookie))


def _patch_http(monkeypatch, handler):
    import requests

    def _get(self, url, timeout=None, **kwargs):
        return handler(url)

    monkeypatch.setattr(requests.Session, "get", _get)


def _ok_preflight(self, mode="live"):
    return PreflightResult(
        source_name="hypernews", status="ok", mode="live"
    )


def test_forum_listing_failure_degrades_and_never_claims_scope(
    tmp_path, monkeypatch
):
    _cookie_env(tmp_path, monkeypatch)

    def handler(url):
        if url.endswith("/get/comp-ops.html"):
            return _FakeResp(LISTING)
        if url.endswith("/get/mcOps.html"):
            raise ConnectionError("forum listing down")
        return _FakeResp(THREAD_PAGE)

    _patch_http(monkeypatch, handler)
    source = HyperNewsSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    threads = [
        f for f in run.facts
        if isinstance(f, NodeFact) and f.subtype == "forum_thread"
    ]
    # the reachable forum's threads are still emitted...
    assert {t.node_id for t in threads} == {"hn:comp-ops/1", "hn:comp-ops/2"}
    # ...but a partial crawl never claims a complete scope
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "mcOps" in run.health.reason
    # and is never persisted as a replayable "complete" cache
    assert not (tmp_path / "data" / "hypernews" / "records.json").exists()


def test_total_fetch_failure_writes_no_cache_and_fails_loud(
    tmp_path, monkeypatch
):
    _cookie_env(tmp_path, monkeypatch)
    monkeypatch.setattr(HyperNewsSource, "preflight", _ok_preflight)

    def handler(url):
        raise ConnectionError("everything down")

    _patch_http(monkeypatch, handler)
    source = HyperNewsSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "forum listings failed" in run.health.reason
    # the old code persisted [] here, poisoning every later run
    assert not (tmp_path / "data" / "hypernews" / "records.json").exists()


def test_zero_threads_across_forums_is_failure_not_empty_success(
    tmp_path, monkeypatch
):
    _cookie_env(tmp_path, monkeypatch)

    def handler(url):
        return _FakeResp("<html>markup with no thread list</html>")

    _patch_http(monkeypatch, handler)
    source = HyperNewsSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "zero threads" in run.health.reason
    assert not (tmp_path / "data" / "hypernews" / "records.json").exists()


def test_failed_hydration_drops_record_instead_of_blanking_it(
    tmp_path, monkeypatch
):
    _cookie_env(tmp_path, monkeypatch)

    def handler(url):
        if url.endswith("/get/comp-ops.html"):
            return _FakeResp(LISTING)
        if url.endswith("/comp-ops/1.html"):
            return _FakeResp(THREAD_PAGE)
        raise ConnectionError("thread fetch failed")

    _patch_http(monkeypatch, handler)
    source = HyperNewsSource(base=str(tmp_path), forums=["comp-ops"])
    run = source.run("run-1", mode="scope_complete")
    threads = [
        f for f in run.facts
        if isinstance(f, NodeFact) and f.subtype == "forum_thread"
    ]
    # the un-hydrated thread is dropped, not emitted with a blank body
    assert {t.node_id for t in threads} == {"hn:comp-ops/1"}
    assert threads[0].attrs["body"] == "Thread body text"
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "dropped" in run.health.reason
    assert not (tmp_path / "data" / "hypernews" / "records.json").exists()


def test_empty_cache_refuses_complete_scope(tmp_path):
    root = tmp_path / "data" / "hypernews"
    root.mkdir(parents=True)
    (root / "records.json").write_text("[]")
    source = HyperNewsSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"


def test_preflight_empty_cache_matches_run_refusal(tmp_path):
    # Preflight must not report ok/record_count=0 for a cache that
    # run() then refuses with endpoint_failed.
    root = tmp_path / "data" / "hypernews"
    root.mkdir(parents=True)
    (root / "records.json").write_text("[]")
    source = HyperNewsSource(base=str(tmp_path))
    result = source.preflight()
    assert result.status == "endpoint_failed"
    assert result.record_count == 0
    run = source.run("run-1", mode="scope_complete")
    assert run.health.status == "endpoint_failed"
    assert result.reason == run.health.reason


def test_clean_live_crawl_claims_scope_writes_cache_and_replays(
    tmp_path, monkeypatch
):
    # Happy path: a fully successful multi-forum crawl keeps its
    # completed-scope claim, persists the cache, and a second run reads
    # that cache and still claims scope.
    _cookie_env(tmp_path, monkeypatch)
    mcops_listing = (
        '<li value="3"><a href="/HyperNews/CMS/get/mcOps/3.html">'
        "Third thread</a></li>"
    )

    def handler(url):
        if url.endswith("/get/comp-ops.html"):
            return _FakeResp(LISTING)
        if url.endswith("/get/mcOps.html"):
            return _FakeResp(mcops_listing)
        return _FakeResp(THREAD_PAGE)

    _patch_http(monkeypatch, handler)
    expected_ids = {"hn:comp-ops/1", "hn:comp-ops/2", "hn:mcOps/3"}
    source = HyperNewsSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    threads = [
        f for f in run.facts
        if isinstance(f, NodeFact) and f.subtype == "forum_thread"
    ]
    assert {t.node_id for t in threads} == expected_ids
    assert all(t.attrs["body"] == "Thread body text" for t in threads)
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert run.health.mode == "live"
    assert (tmp_path / "data" / "hypernews" / "records.json").is_file()

    replay = HyperNewsSource(base=str(tmp_path)).run(
        "run-2", mode="scope_complete"
    )
    replay_threads = [
        f for f in replay.facts
        if isinstance(f, NodeFact) and f.subtype == "forum_thread"
    ]
    assert {t.node_id for t in replay_threads} == expected_ids
    assert replay.completed_scope is True
    assert replay.health.status == "ok"
    assert replay.health.mode == "cache"


def test_cache_skips_unparseable_items_without_scope_claim(tmp_path):
    root = tmp_path / "data" / "hypernews"
    root.mkdir(parents=True)
    (root / "records.json").write_text(
        json.dumps(RECORDS + ["junk", {"title": "no thread id"}])
    )
    source = HyperNewsSource(base=str(tmp_path))
    run = source.run("run-1", mode="scope_complete")
    threads = [
        f for f in run.facts
        if isinstance(f, NodeFact) and f.subtype == "forum_thread"
    ]
    assert [t.node_id for t in threads] == ["hn:comp-ops/123"]
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "skipped 2" in run.health.reason


def test_max_threads_truncation_never_claims_scope(tmp_path, monkeypatch):
    _cookie_env(tmp_path, monkeypatch)

    def handler(url):
        if url.endswith("/get/comp-ops.html"):
            return _FakeResp(LISTING)
        return _FakeResp(THREAD_PAGE)

    _patch_http(monkeypatch, handler)
    source = HyperNewsSource(
        base=str(tmp_path), forums=["comp-ops"], max_threads=1
    )
    run = source.run("run-1", mode="scope_complete")
    threads = [
        f for f in run.facts
        if isinstance(f, NodeFact) and f.subtype == "forum_thread"
    ]
    assert len(threads) == 1
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "max_threads" in run.health.reason
    # a truncated window must not be persisted as a complete cache
    assert not (tmp_path / "data" / "hypernews" / "records.json").exists()
