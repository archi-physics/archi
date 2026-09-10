"""req.w2.sources-parity — DocumentationSource + SSOCookieDocsSource, offline."""
import hashlib
import json
import time
from datetime import datetime, timezone

import pytest

from okg.deployment import EdgeFact, NodeFact
from okg.deployment import ContentHashProbe
from okg.deployment import MutableApiProbe

import archi.sources.docs as docs_mod
from archi.sources.docs import DocumentationSource, SSOCookieDocsSource

_FAR_FUTURE = int(datetime(2035, 1, 1, tzinfo=timezone.utc).timestamp())
_PAST = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())

PAGE_PLAIN = {
    "url": "https://docs.example.cern.ch/ops/",
    "title": "Ops Guide",
    "body": (
        "Transfers at T2_US_MIT need CMSSW_14_0_2; see CMSPROD-100. "
        "Submit via cmsweb.cern.ch when reqmgr2 is up."
    ),
    "site_name": "docs.example.cern.ch",
}
PAGE_REPO = {
    "url": "https://gitlab.cern.ch/cms/ops-repo/-/blob/master/docs/guide.md",
    "title": "Repo Guide",
    "body": "How the ops repo is laid out.",
    "source_repo": "cms/ops-repo",
    "path": "docs/guide.md",
}


def _write_docsite(tmp_path, records, with_targets=False):
    (tmp_path / "data" / "docsite").mkdir(parents=True, exist_ok=True)
    if records is not None:
        (tmp_path / "data" / "docsite" / "records.json").write_text(
            json.dumps(records)
        )
    kwargs = {}
    if with_targets:
        (tmp_path / "data" / "cric").mkdir(parents=True)
        (tmp_path / "data" / "cric" / "sites.json").write_text(
            json.dumps({"T2_US_MIT": {}})
        )
        (tmp_path / "data" / "releases").mkdir(parents=True)
        (tmp_path / "data" / "releases" / "records.json").write_text(
            json.dumps([{"label": "CMSSW_14_0_2"}])
        )
        (tmp_path / "data" / "jira").mkdir(parents=True)
        (tmp_path / "data" / "jira" / "records.json").write_text(
            json.dumps([{"key": "CMSPROD-100"}])
        )
        (tmp_path / "data" / "services").mkdir(parents=True)
        (tmp_path / "data" / "services" / "services.json").write_text(
            json.dumps(
                {"reqmgr2": {"endpoint": "https://cmsweb.cern.ch/reqmgr2"}}
            )
        )
        kwargs = {
            "sites_path": "data/cric/sites.json",
            "releases_path": "data/releases/records.json",
            "jira_records_path": "data/jira/records.json",
            "services_path": "data/services/services.json",
        }
    return DocumentationSource(
        records_path="data/docsite/records.json",
        base=str(tmp_path),
        **kwargs,
    )


def _nodes(facts, subtype):
    return [f for f in facts if isinstance(f, NodeFact) and f.subtype == subtype]


def _edges(facts, edge_type):
    return [f for f in facts if isinstance(f, EdgeFact) and f.edge_type == edge_type]


# --- DocumentationSource: parse + emission ----------------------------------

def test_page_node_ids_and_attrs(tmp_path):
    source = _write_docsite(tmp_path, [PAGE_PLAIN, PAGE_REPO])
    facts = list(source.run("run-1").facts)
    pages = {n.node_id: n for n in _nodes(facts, "documentation_page")}
    url_hash_id = (
        "documentation_page:"
        + hashlib.sha256(PAGE_PLAIN["url"].encode("utf-8")).hexdigest()[:16]
    )
    assert set(pages) == {
        url_hash_id,
        "documentation_page:cms_ops-repo:docs_guide.md",
    }
    plain = pages[url_hash_id]
    assert plain.attrs["title"] == "Ops Guide"
    assert plain.attrs["label"] == "Ops Guide"
    assert plain.attrs["text"].startswith("Ops Guide ")
    assert plain.source_record_id == {"url": PAGE_PLAIN["url"]}
    assert plain.source_revision["run_id"] == "run-1"
    # Parity attrs the original emitted on every documentation_page.
    for page in pages.values():
        assert page.attrs["record_kind"] == "documentation_page"
        assert page.attrs["service_aliases"] == []


def test_payload_dedup_and_junk_tolerance(tmp_path):
    source = _write_docsite(
        tmp_path,
        [PAGE_PLAIN, dict(PAGE_PLAIN, title="dup"), "junk", {"title": "no url"}],
    )
    facts = list(source.run("r").facts)
    pages = _nodes(facts, "documentation_page")
    assert len(pages) == 1
    assert pages[0].attrs["title"] == "Ops Guide"


def test_repo_node_and_contains_edge(tmp_path):
    source = _write_docsite(tmp_path, [PAGE_REPO])
    facts = list(source.run("r").facts)
    repos = _nodes(facts, "software_repository")
    assert len(repos) == 1
    repo = repos[0]
    assert repo.node_id == "software_repository:cms/ops-repo"
    assert repo.attrs["url"] == "https://gitlab.cern.ch/cms/ops-repo"
    edges = _edges(facts, "contains")
    repo_edges = [e for e in edges if e.src == repo.node_id]
    assert len(repo_edges) == 1
    assert repo_edges[0].dst == "documentation_page:cms_ops-repo:docs_guide.md"
    assert repo_edges[0].attrs == {"relationship": "repo_documentation"}


def test_repo_base_url_param(tmp_path):
    _write_docsite(tmp_path, [PAGE_REPO])
    source = DocumentationSource(
        records_path="data/docsite/records.json",
        repo_base_url="https://gitlab.example.org/",
        base=str(tmp_path),
    )
    facts = list(source.run("r").facts)
    repo = _nodes(facts, "software_repository")[0]
    assert repo.attrs["url"] == "https://gitlab.example.org/cms/ops-repo"


def test_chunk_emission_ids_and_windowing(tmp_path):
    long_body = "x" * 4200  # forces two overlapping windows
    source = _write_docsite(
        tmp_path,
        [dict(PAGE_PLAIN, body=long_body)],
    )
    facts = list(source.run("r").facts)
    page = _nodes(facts, "documentation_page")[0]
    chunks = _nodes(facts, "document_chunk")
    assert [c.attrs["char_offset"] for c in chunks] == [0, 3800]
    first = chunks[0]
    expected = "chunk:" + hashlib.sha256(
        f"{page.node_id}\x000\x00{first.attrs['text']}".encode("utf-8")
    ).hexdigest()[:16]
    assert first.node_id == expected
    assert first.attrs["chunker_name"] == "cms_document_window_v1"
    assert first.attrs["heading_path"] == "Ops Guide"
    page_chunk_edges = [
        e for e in _edges(facts, "contains") if e.src == page.node_id
    ]
    assert [e.attrs["chunk_index"] for e in page_chunk_edges] == [0, 1]
    assert first.source_record_id == {"url": PAGE_PLAIN["url"], "chunk_index": 0}


def test_reference_edges_all_four_kinds(tmp_path):
    source = _write_docsite(tmp_path, [PAGE_PLAIN], with_targets=True)
    facts = list(source.run("r").facts)
    chunk = _nodes(facts, "document_chunk")[0]
    refs = _edges(facts, "references")
    matches = {(e.attrs["match_type"], e.dst) for e in refs}
    assert matches == {
        ("cms_site", "site:T2_US_MIT"),
        ("cmssw_release", "cmssw_release:CMSSW_14_0_2"),
        ("jira_issue", "jira:CMSPROD-100"),
        ("infrastructure_service", "svc:reqmgr2"),
    }
    assert all(e.src == chunk.node_id for e in refs)
    assert all(e.provenance == "derived_deterministic" for e in refs)
    assert all(e.confidence == 0.95 for e in refs)


def test_missing_reference_caches_mean_no_reference_edges(tmp_path):
    # Unconfigured (None) reference caches are skipped silently.
    source = _write_docsite(tmp_path, [PAGE_PLAIN], with_targets=False)
    facts = list(source.run("r").facts)
    assert _edges(facts, "references") == []


@pytest.mark.parametrize(
    "param", ["sites_path", "releases_path", "services_path"]
)
def test_configured_but_missing_reference_cache_raises(tmp_path, param):
    # A configured path whose file is absent must raise (as the original
    # did), not silently emit zero reference edges of that kind.
    _write_docsite(tmp_path, [PAGE_PLAIN])
    source = DocumentationSource(
        records_path="data/docsite/records.json",
        base=str(tmp_path),
        **{param: "data/absent/cache.json"},
    )
    with pytest.raises(FileNotFoundError, match="data/absent/cache.json"):
        source.run("r")


# --- DocumentationSource: record-set/deletion + preflight + probe -----------

def test_completed_scope_by_mode(tmp_path):
    source = _write_docsite(tmp_path, [PAGE_PLAIN])
    assert source.run("r", mode="cursor").completed_scope is False
    assert source.run("r", mode="scope_complete").completed_scope is True
    assert source.run("r", mode="reconcile").completed_scope is True


def test_missing_cache_run_is_safe(tmp_path):
    source = _write_docsite(tmp_path, records=None)
    run = source.run("r", mode="reconcile")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "cache_missing"


def test_preflight(tmp_path):
    missing = _write_docsite(tmp_path, records=None)
    result = missing.preflight()
    assert result.status == "cache_missing"
    assert result.required is True
    present = _write_docsite(tmp_path, [PAGE_PLAIN, PAGE_REPO])
    result = present.preflight()
    assert result.status == "ok"
    assert result.record_count == 2
    assert result.content_hash


def test_docsite_probe_declarations(tmp_path):
    source = _write_docsite(tmp_path, [PAGE_PLAIN])
    assert DocumentationSource.profile == "discovery_crawl"
    assert DocumentationSource.change_probe_kind == "content_hash"
    assert isinstance(source.change_probe, ContentHashProbe)
    first = source.change_probe.build_token()
    assert source.change_probe.build_token() == first
    (tmp_path / "data" / "docsite" / "records.json").write_text(
        json.dumps([PAGE_PLAIN, PAGE_REPO])
    )
    assert source.change_probe.build_token() != first


def test_source_name_param_renames_health(tmp_path):
    (tmp_path / "data" / "gitlab-docs").mkdir(parents=True)
    (tmp_path / "data" / "gitlab-docs" / "records.json").write_text(
        json.dumps([PAGE_REPO])
    )
    source = DocumentationSource(
        source_name="gitlab_docs",
        records_path="data/gitlab-docs/records.json",
        base=str(tmp_path),
    )
    assert source.preflight().source_name == "gitlab_docs"
    assert "gitlab_docs" in source.run("r").health.reason


# --- SSOCookieDocsSource -----------------------------------------------------

SITEMAP = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.example.cern.ch/a/</loc></url>
  <url><loc>https://docs.example.cern.ch/bounced/</loc></url>
  <url><loc>https://docs.example.cern.ch/empty/</loc></url>
  <url><loc>https://docs.example.cern.ch/broken/</loc></url>
  <url><loc>https://docs.example.cern.ch/b/</loc></url>
</urlset>
"""
SITEMAP_URL = "https://docs.example.cern.ch/sitemap.xml"
COOKIE_ENV = "ARCHI_T_DOCS_COOKIE_FILE"


class _FakeResponse:
    def __init__(
        self, status_code=200, text="", url="", content=b"", headers=None
    ):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.content = content or text.encode("utf-8")
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _default_responses():
    return {
        SITEMAP_URL: _FakeResponse(200, "", SITEMAP_URL, SITEMAP),
        "https://docs.example.cern.ch/a/": _FakeResponse(
            200,
            "<html><title>Page &amp; A</title><body><p>Alpha body"
            " CMSSW_14_0_2</p></body></html>",
            "https://docs.example.cern.ch/a/",
        ),
        "https://docs.example.cern.ch/bounced/": _FakeResponse(
            200,
            "<html><title>Sign in to CERN</title></html>",
            "https://auth.cern.ch/auth/realms/cern",
        ),
        "https://docs.example.cern.ch/empty/": _FakeResponse(
            200,
            "<html><body><script>void(0)</script></body></html>",
            "https://docs.example.cern.ch/empty/",
        ),
        "https://docs.example.cern.ch/broken/": _FakeResponse(
            404, "gone", "https://docs.example.cern.ch/broken/"
        ),
        "https://docs.example.cern.ch/b/": _FakeResponse(
            200,
            "<html><body>Beta body</body></html>",
            "https://docs.example.cern.ch/b/",
        ),
    }


def _fake_sessions(monkeypatch, responses):
    sessions = []

    class _FakeSession:
        def __init__(self):
            self.cookies = None
            sessions.append(self)

        def get(self, url, **kwargs):
            return responses[url]

    monkeypatch.setattr(docs_mod.requests, "Session", _FakeSession)
    return sessions


def _write_cookie_file(path, *, expires=_FAR_FUTURE):
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".cern.ch\tTRUE\t/\tTRUE\t{expires}\tsessionid\ttest-value\n"
    )


def _sso_source(**overrides):
    kwargs = {
        "sitemap_url": SITEMAP_URL,
        "cookie_file_env": COOKIE_ENV,
        "source_name": "cmsweb_docs",
    }
    kwargs.update(overrides)
    return SSOCookieDocsSource(**kwargs)


def test_sso_preflight_missing_cookie_env(monkeypatch):
    monkeypatch.delenv(COOKIE_ENV, raising=False)
    result = _sso_source().preflight()
    assert result.status == "missing_credential"
    assert result.credential_refs == (COOKIE_ENV,)
    assert result.required is False


def test_sso_preflight_fresh_and_expired_cookie(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    _write_cookie_file(cookie_path, expires=_FAR_FUTURE)
    result = _sso_source().preflight()
    assert result.status == "ok"
    assert "live cookie" in result.reason
    _write_cookie_file(cookie_path, expires=_PAST)
    result = _sso_source().preflight()
    assert result.status == "auth_failed"
    assert "expired" in result.reason


def test_sso_run_wires_cookie_jar_and_crawls_sitemap(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    sessions = _fake_sessions(monkeypatch, _default_responses())
    run = _sso_source().run("run-9")
    facts = list(run.facts)
    # The Netscape cookie file was loaded into the crawl session.
    assert sessions and {c.name for c in sessions[0].cookies} == {"sessionid"}
    pages = {n.attrs["url"]: n for n in _nodes(facts, "documentation_page")}
    # login-bounce, empty-body, and HTTP-error pages are skipped.
    assert set(pages) == {
        "https://docs.example.cern.ch/a/",
        "https://docs.example.cern.ch/b/",
    }
    page_a = pages["https://docs.example.cern.ch/a/"]
    assert page_a.attrs["title"] == "Page & A"
    assert page_a.attrs["site_name"] == "docs.example.cern.ch"
    assert "Alpha body" in page_a.attrs["body"]
    assert page_a.source_revision["sitemap_url"] == SITEMAP_URL
    # bounce + HTTP error are per-page failures -> partial health.
    assert run.health.status == "endpoint_failed"
    assert "partial crawl: 2/5" in run.health.reason
    assert run.health.mode == "live"
    assert run.health.record_count == 2
    assert run.health.credential_refs == (COOKIE_ENV,)


def test_sso_partial_crawl_never_claims_complete_scope(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    _fake_sessions(monkeypatch, _default_responses())
    run = _sso_source().run("r", mode="scope_complete")
    facts = list(run.facts)
    # The successful pages are still emitted...
    assert {n.attrs["url"] for n in _nodes(facts, "documentation_page")} == {
        "https://docs.example.cern.ch/a/",
        "https://docs.example.cern.ch/b/",
    }
    # ...but a partially failed crawl must never claim a complete scope
    # (missing_from_completed_scope would retract the failed pages).
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "2/5" in run.health.reason
    assert "https://docs.example.cern.ch/bounced/" in run.health.reason
    assert "https://docs.example.cern.ch/broken/" in run.health.reason


def test_sso_clean_crawl_claims_complete_scope(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    sitemap = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        b"  <url><loc>https://docs.example.cern.ch/a/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/empty/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/b/</loc></url>\n"
        b"</urlset>\n"
    )
    responses = _default_responses()
    responses[SITEMAP_URL] = _FakeResponse(200, "", SITEMAP_URL, sitemap)
    _fake_sessions(monkeypatch, responses)
    run = _sso_source().run("r", mode="scope_complete")
    pages = _nodes(list(run.facts), "documentation_page")
    # Zero failures: empty-body pages are skipped without counting as
    # failures, so the run may claim the complete scope.
    assert {p.attrs["url"] for p in pages} == {
        "https://docs.example.cern.ch/a/",
        "https://docs.example.cern.ch/b/",
    }
    assert run.completed_scope is True
    assert run.health.status == "ok"


def test_sso_login_lookalike_pages_are_ingested_not_bounced(
    monkeypatch, tmp_path
):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    sitemap = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        b"  <url><loc>https://docs.example.cern.ch/login-guide/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/sso/usage/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/auth-mention/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/real-bounce/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/proxy-bounce/</loc></url>\n"
        b"</urlset>\n"
    )
    responses = {
        SITEMAP_URL: _FakeResponse(200, "", SITEMAP_URL, sitemap),
        # Non-redirected 200s whose path contains /login or /sso/, or
        # whose body merely mentions auth.cern.ch: legit docs pages.
        "https://docs.example.cern.ch/login-guide/": _FakeResponse(
            200,
            "<html><title>Login Guide</title><body>How to log in"
            " to the cluster.</body></html>",
            "https://docs.example.cern.ch/login-guide/",
        ),
        "https://docs.example.cern.ch/sso/usage/": _FakeResponse(
            200,
            "<html><title>SSO Usage</title><body>Configuring service"
            " accounts.</body></html>",
            "https://docs.example.cern.ch/sso/usage/",
        ),
        "https://docs.example.cern.ch/auth-mention/": _FakeResponse(
            200,
            "<html><title>Tokens</title><body>Point your browser at"
            " auth.cern.ch to fetch a token.</body></html>",
            "https://docs.example.cern.ch/auth-mention/",
        ),
        # Redirected to an SSO host: a real bounce.
        "https://docs.example.cern.ch/real-bounce/": _FakeResponse(
            200,
            "<html><title>Sign in to CERN</title></html>",
            "https://auth.cern.ch/auth/realms/cern",
        ),
        # Redirected away + login-looking body: also a bounce.
        "https://docs.example.cern.ch/proxy-bounce/": _FakeResponse(
            200,
            "<html><title>Sign in to CERN</title><body>Sign in to CERN"
            "</body></html>",
            "https://sso-proxy.example.org/gate?next=proxy-bounce",
        ),
    }
    _fake_sessions(monkeypatch, responses)
    run = _sso_source().run("r", mode="scope_complete")
    pages = {
        n.attrs["url"] for n in _nodes(list(run.facts), "documentation_page")
    }
    assert pages == {
        "https://docs.example.cern.ch/login-guide/",
        "https://docs.example.cern.ch/sso/usage/",
        "https://docs.example.cern.ch/auth-mention/",
    }
    # The two real bounces count as failures: partial, incomplete scope.
    assert run.completed_scope is False
    assert run.health.status == "endpoint_failed"
    assert "2/5" in run.health.reason


def test_sso_run_honors_max_pages(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    _fake_sessions(monkeypatch, _default_responses())
    run = _sso_source(max_pages=1).run("r")
    pages = _nodes(list(run.facts), "documentation_page")
    assert [p.attrs["url"] for p in pages] == [
        "https://docs.example.cern.ch/a/"
    ]


def test_sso_max_pages_truncation_never_claims_scope(monkeypatch, tmp_path):
    """Finding 5 (adversarial review): when max_pages actually truncates
    the sitemap frontier, a clean crawl must surface the truncation and
    never claim a complete scope — the un-crawled pages would otherwise
    be retracted under missing_from_completed_scope."""
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    sitemap = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        b"  <url><loc>https://docs.example.cern.ch/a/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/empty/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/b/</loc></url>\n"
        b"</urlset>\n"
    )
    responses = _default_responses()
    responses[SITEMAP_URL] = _FakeResponse(200, "", SITEMAP_URL, sitemap)
    _fake_sessions(monkeypatch, responses)
    run = _sso_source(max_pages=2).run("r", mode="scope_complete")
    pages = _nodes(list(run.facts), "documentation_page")
    # Only the truncated frontier was crawled and emitted...
    assert {p.attrs["url"] for p in pages} == {
        "https://docs.example.cern.ch/a/"
    }
    # ...and the run must not claim the scope, with truncation surfaced.
    assert run.completed_scope is False
    assert run.health.status == "ok"
    assert "max_pages=2" in run.health.reason
    assert "2/3" in run.health.reason
    assert "no complete scope" in run.health.reason
    # A cap that does not actually truncate changes nothing.
    _fake_sessions(monkeypatch, responses)
    run = _sso_source(max_pages=3).run("r", mode="scope_complete")
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert "max_pages" not in run.health.reason


def test_sso_non_html_sitemap_entries_skipped_not_ingested(
    monkeypatch, tmp_path
):
    """Finding 10 (adversarial review): PDFs/binaries in the sitemap must
    be skipped by Content-Type (counted, excluded from scope by design)
    instead of being regex-stripped into mojibake documentation pages."""
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    sitemap = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        b"  <url><loc>https://docs.example.cern.ch/a/</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/report.pdf</loc></url>\n"
        b"  <url><loc>https://docs.example.cern.ch/b/</loc></url>\n"
        b"</urlset>\n"
    )
    responses = _default_responses()
    responses[SITEMAP_URL] = _FakeResponse(200, "", SITEMAP_URL, sitemap)
    # An explicit HTML content type (with charset) still ingests.
    responses["https://docs.example.cern.ch/a/"] = _FakeResponse(
        200,
        "<html><title>Page A</title><body>Alpha body</body></html>",
        "https://docs.example.cern.ch/a/",
        headers={"Content-Type": "text/html; charset=utf-8"},
    )
    responses["https://docs.example.cern.ch/report.pdf"] = _FakeResponse(
        200,
        "%PDF-1.4 \x93\xff binary stream",
        "https://docs.example.cern.ch/report.pdf",
        content=b"%PDF-1.4 binary",
        headers={"Content-Type": "application/pdf"},
    )
    _fake_sessions(monkeypatch, responses)
    run = _sso_source().run("r", mode="scope_complete")
    pages = {
        n.attrs["url"] for n in _nodes(list(run.facts), "documentation_page")
    }
    # The PDF is not a documentation page; HTML pages (with or without a
    # Content-Type header) are.
    assert pages == {
        "https://docs.example.cern.ch/a/",
        "https://docs.example.cern.ch/b/",
    }
    # Skipped non-HTML entries are excluded from scope by design: they
    # are surfaced in health but do not block the scope claim.
    assert run.completed_scope is True
    assert run.health.status == "ok"
    assert "skipped 1 non-HTML" in run.health.reason


def test_sso_sitemap_login_bounce_raises(monkeypatch, tmp_path):
    cookie_path = tmp_path / "sso.txt"
    _write_cookie_file(cookie_path)
    monkeypatch.setenv(COOKIE_ENV, str(cookie_path))
    responses = _default_responses()
    responses[SITEMAP_URL] = _FakeResponse(
        200,
        "<title>Sign in to CERN</title>",
        "https://auth.cern.ch/auth/realms/cern",
    )
    _fake_sessions(monkeypatch, responses)
    with pytest.raises(RuntimeError, match="login page"):
        _sso_source().run("r")


def test_sso_run_without_cookie_yields_no_records(monkeypatch):
    monkeypatch.delenv(COOKIE_ENV, raising=False)
    run = _sso_source().run("r")
    assert list(run.facts) == []
    # A missing cookie is an auth failure, never a healthy empty crawl
    # that could retract every page via missing_from_completed_scope.
    assert run.completed_scope is False
    assert run.health.status == "auth_failed"
    assert run.health.record_count == 0
    assert run.health.mode == "live"
    assert run.health.credential_refs == (COOKIE_ENV,)
    assert "no complete scope" in run.health.reason


def test_sso_run_missing_or_unparseable_cookie_never_completes_scope(
    monkeypatch, tmp_path
):
    # Env var set but the file does not exist.
    monkeypatch.setenv(COOKIE_ENV, str(tmp_path / "absent.txt"))
    run = _sso_source().run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "auth_failed"
    # File exists but is not a Netscape cookie jar.
    bad = tmp_path / "bad.txt"
    bad.write_text("definitely not a cookie jar")
    monkeypatch.setenv(COOKIE_ENV, str(bad))
    run = _sso_source().run("r", mode="scope_complete")
    assert list(run.facts) == []
    assert run.completed_scope is False
    assert run.health.status == "auth_failed"


def test_sso_probe_declarations():
    source = _sso_source()
    assert SSOCookieDocsSource.profile == "discovery_crawl"
    assert SSOCookieDocsSource.change_probe_kind == "mutable_api"
    assert isinstance(source.change_probe, MutableApiProbe)
    assert source.cache_paths == ()
    first = source.change_probe.build_token()
    time.sleep(0.002)
    assert source.change_probe.build_token() != first


def test_sso_required_params():
    with pytest.raises(TypeError):
        SSOCookieDocsSource()  # sitemap_url and cookie_file_env are required
