"""req.compops-snapshot-builder -- the moved static-docs downloader, no network.

``requests.Session`` is replaced by a fake that answers the GitLab API and
public-page URLs the manifest names; the written caches are then read back
by the archi documentation reader.
"""
from __future__ import annotations

import json
from urllib.parse import quote

import pytest
import yaml

from archi.downloaders import static_docs
from archi.sources.docs import DocumentationSource


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"", headers=None, url=""):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.text = content.decode("utf-8", "replace")
        self.headers = headers or {}
        self.url = url

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


PROJECT = "cmsdmops/Documentation"
ENC = quote(PROJECT, safe="")
API = f"{static_docs.GITLAB_BASE}/api/v4/projects/{ENC}"
FILES = {
    "README.md": b"# DM operations\nHow to recover a stuck rule.\n",
    "scripts/check.py": b"print('ok')\n",
    "images/logo.png": b"\x89PNG",
    "node_modules/x/readme.md": b"# vendored\n",
}
PUBLIC_URL = "https://fts3-docs.web.cern.ch/fts3-docs/"
PUBLIC_HTML = (
    b"<html><head><title>FTS3 docs</title><script>var x=1;</script></head>"
    b"<body><h1>FTS3</h1><p>Transfer   service\tdocs.</p></body></html>"
)


class FakeSession:
    requests: list[tuple[str, dict]] = []

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        FakeSession.requests.append((url, dict(headers or {})))
        if url == API:
            return FakeResponse(payload={"default_branch": "master", "path_with_namespace": PROJECT})
        if url == f"{API}/repository/tree":
            tree = [{"type": "blob", "path": p} for p in FILES] + [{"type": "tree", "path": "scripts"}]
            return FakeResponse(payload=tree, headers={"X-Total-Pages": "1"})
        prefix = f"{API}/repository/files/"
        if url.startswith(prefix) and url.endswith("/raw"):
            path = url[len(prefix):-len("/raw")].replace("%2F", "/")
            return FakeResponse(content=FILES[path])
        if url == PUBLIC_URL:
            return FakeResponse(content=PUBLIC_HTML, headers={"content-type": "text/html; charset=utf-8"}, url=url)
        return FakeResponse(status=404, url=url)


def _manifest(tmp_path):
    packaged = yaml.safe_load(static_docs.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    packaged["targets"]["gitlab_docs"]["repos"] = [{"project": PROJECT, "status": "ready"}]
    packaged["targets"]["public_docs"]["urls"] = [
        {"name": "fts3_docs", "status": "ready", "url": PUBLIC_URL},
        {"name": "skipped", "status": "deferred-auth", "url": "https://example.invalid/"},
    ]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(packaged), encoding="utf-8")
    return path


@pytest.fixture
def fake_session(monkeypatch):
    FakeSession.requests = []
    monkeypatch.setattr(static_docs.requests, "Session", FakeSession)
    for name in ("CERN_GITLAB_TOKEN", "GITLAB_CERN_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_download_writes_caches_the_docs_reader_accepts(tmp_path, monkeypatch, fake_session, capsys):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    assert static_docs.main(["--manifest", str(_manifest(tmp_path))]) == 0
    summary = json.loads(capsys.readouterr().out)
    gitlab = json.loads((tmp_path / "data/gitlab-docs/records.json").read_text())
    assert [r["file_path"] for r in gitlab] == ["README.md", "scripts/check.py"]
    assert gitlab[0]["title"] == "DM operations"
    assert gitlab[0]["url"] == f"https://gitlab.cern.ch/{PROJECT}/-/blob/master/README.md"
    assert summary["gitlab"]["skipped_files"] == 2  # .png suffix, node_modules path
    docsite = json.loads((tmp_path / "data/docsite/records.json").read_text())
    assert docsite == [
        {
            "title": "FTS3 docs",
            "url": PUBLIC_URL,
            # The page title is part of the text, as in the okg-deployments copy.
            "body": "FTS3 docsFTS3Transfer service docs.",
            "site_name": "fts3-docs.web.cern.ch",
        }
    ]
    for group, name in (("gitlab-docs", "gitlab_docs"), ("docsite", "docsite")):
        meta = json.loads((tmp_path / f"data/{group}/meta.json").read_text())
        assert meta["source_name"] == name
        run = DocumentationSource(
            source_name=name, records_path=f"data/{group}/records.json", base=str(tmp_path)
        ).run("r", mode="scope_complete")
        assert list(run.facts) and run.completed_scope and run.health.status == "ok"
    # No token set: every GitLab call is anonymous.
    assert all(headers == {} for _, headers in FakeSession.requests)


def test_merge_keeps_existing_records_outside_the_manifest(tmp_path, fake_session, capsys):
    out = tmp_path / "docs.json"
    out.write_text(json.dumps([{"url": "https://old.example/page", "title": "Old", "body": "kept"}]))
    assert static_docs.main(
        ["--manifest", str(_manifest(tmp_path)), "--skip-gitlab", "--docsite-output", str(out)]
    ) == 0
    urls = [r["url"] for r in json.loads(out.read_text())]
    assert sorted(urls) == sorted(["https://old.example/page", PUBLIC_URL])


def test_token_is_tried_first_and_401_falls_back_to_anonymous(monkeypatch):
    monkeypatch.setenv("CERN_GITLAB_TOKEN", "glpat-test-not-a-secret")
    candidates = static_docs._gitlab_header_candidates()
    assert candidates == [{"PRIVATE-TOKEN": "glpat-test-not-a-secret"}, {}]

    seen = []

    class Session:
        def get(self, url, params=None, headers=None, timeout=None):
            seen.append(headers)
            return FakeResponse(status=401 if headers else 200)

    resp = static_docs._gitlab_get(Session(), "https://x", timeout=1, header_candidates=candidates)
    assert resp.status_code == 200 and seen == [candidates[0], {}]


def test_packaged_manifest_names_archi_cache_paths():
    text = static_docs.DEFAULT_MANIFEST.read_text(encoding="utf-8")
    manifest = yaml.safe_load(text)
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert "data/cms/" not in body
    assert manifest["targets"]["gitlab_docs"]["cache_path"] == "data/gitlab-docs/records.json"
    assert manifest["targets"]["public_docs"]["cache_path"] == "data/docsite/records.json"
    ready = [r for r in manifest["targets"]["gitlab_docs"]["repos"] if r["status"] == "ready"]
    assert len(ready) == 6


def test_urls_replace_the_manifest_targets_and_skip_gitlab(tmp_path, monkeypatch, fake_session, capsys):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    assert static_docs.main(["--skip-gitlab", "--urls", f"{PUBLIC_URL}, {PUBLIC_URL}"]) == 0
    docsite = json.loads((tmp_path / "data/docsite/records.json").read_text())
    assert [r["url"] for r in docsite] == [PUBLIC_URL]  # repeated URLs fetch once
    assert [url for url, _ in FakeSession.requests] == [PUBLIC_URL]  # no manifest target, no GitLab
    assert not (tmp_path / "data/gitlab-docs").exists()


def test_urls_is_strict_a_page_that_cannot_be_cached_fails_and_writes_nothing(
    tmp_path, monkeypatch, fake_session, capsys
):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    gone = "https://example.invalid/missing/"
    assert static_docs.main(["--skip-gitlab", "--urls", f"{PUBLIC_URL},{gone}"]) == 1
    captured = capsys.readouterr()
    assert "not cached" in captured.err and "example_invalid_missing" in captured.err
    assert not (tmp_path / "data/docsite").exists()


def test_urls_naming_nothing_is_refused(tmp_path, monkeypatch, fake_session, capsys):
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    assert static_docs.main(["--skip-gitlab", "--urls", " , "]) == 2


def test_urls_connection_error_is_a_clean_failure(tmp_path, monkeypatch, capsys):
    class Down:
        def get(self, url, **kwargs):
            raise static_docs.requests.ConnectionError("name resolution failed")

    monkeypatch.setattr(static_docs.requests, "Session", Down)
    monkeypatch.setenv("ARCHI_DATA_ROOT", str(tmp_path))
    assert static_docs.main(["--skip-gitlab", "--urls", PUBLIC_URL]) == 1
    assert "could not be fetched: name resolution failed" in capsys.readouterr().err
    assert not (tmp_path / "data/docsite").exists()
