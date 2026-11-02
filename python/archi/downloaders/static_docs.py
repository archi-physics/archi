#!/usr/bin/env python3
"""Download static documentation caches from the source download manifest.

Moved from okg-deployments ``cms/scripts/download_static_docs.py`` (main
``32f2b4e3c2``). It understands the manifest shipped next to this module
(``source-download-manifest.yaml``) and emits records in the shape
``archi.sources.docs.DocumentationSource`` reads. It merges new records into
existing cache files by URL, so an expansion run does not remove older cache
entries that are outside the current manifest.

    python -m archi.downloaders.static_docs [--manifest <file>]
    python -m archi.downloaders.static_docs --skip-gitlab --urls URL[,URL...]

Outputs default to ``data/gitlab-docs/records.json`` and
``data/docsite/records.json`` under ``ARCHI_DATA_ROOT`` (or the current
directory when that variable is unset). GitLab reads use ``CERN_GITLAB_TOKEN``
or ``GITLAB_CERN_TOKEN`` when set, and retry anonymously on 401/403.

Changes from the okg-deployments copy: the default manifest is the packaged
copy, the default outputs follow ``ARCHI_DATA_ROOT`` instead of the old
``data/cms/...`` layout, ``main`` takes an ``argv`` list, and ``lxml`` is
imported only when a public page is parsed (it is an okg dependency, not an
archi one). Record shapes are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
import yaml


DATA_ROOT_ENV = "ARCHI_DATA_ROOT"
DEFAULT_MANIFEST = Path(__file__).with_name("source-download-manifest.yaml")
GITLAB_OUTPUT = Path("data/gitlab-docs/records.json")
DOCSITE_OUTPUT = Path("data/docsite/records.json")
GITLAB_BASE = "https://gitlab.cern.ch"


def _data_root() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV)
    return Path(raw).expanduser() if raw else Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--gitlab-output", default=None)
    parser.add_argument("--docsite-output", default=None)
    parser.add_argument("--max-bytes", type=int, default=512_000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--skip-public", action="store_true")
    parser.add_argument("--skip-gitlab", action="store_true")
    parser.add_argument(
        "--urls", default=None,
        help="comma- or space-separated public page URLs to fetch instead of the "
             "manifest's public_docs targets. Strict: a page that cannot be "
             "fetched fails the run and nothing is written.",
    )
    args = parser.parse_args(argv)
    if args.gitlab_output is None:
        args.gitlab_output = str(_data_root() / GITLAB_OUTPUT)
    if args.docsite_output is None:
        args.docsite_output = str(_data_root() / DOCSITE_OUTPUT)

    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "gitlab": {},
        "public_docs": {},
    }

    if not args.skip_gitlab:
        gitlab_records, gitlab_summary = _download_gitlab_docs(
            manifest,
            max_bytes=args.max_bytes,
            timeout=args.timeout,
        )
        summary["gitlab"] = _merge_records(
            Path(args.gitlab_output),
            gitlab_records,
            source_name="gitlab_docs",
        ) | gitlab_summary

    if not args.skip_public:
        if args.urls is not None:
            urls = _split_urls(args.urls)
            if not urls:
                print("error: --urls names no URL", file=sys.stderr)
                return 2
            manifest = {"targets": {"public_docs": {"urls": [
                {"name": _url_name(url), "status": "ready", "url": url} for url in urls
            ]}}}
        try:
            public_records, public_summary = _download_public_docs(
                manifest,
                timeout=args.timeout,
            )
        except requests.RequestException as exc:
            if args.urls is None:
                raise
            print(f"error: a requested page could not be fetched: {exc}", file=sys.stderr)
            return 1
        if args.urls is not None and public_summary["skipped_urls"]:
            skipped = sorted(
                name for name, info in public_summary["urls"].items() if not info.get("emitted")
            ) or ["a page with no text"]
            print(json.dumps(summary | {"public_docs": public_summary}, indent=2, sort_keys=True))
            print(
                f"error: {public_summary['skipped_urls']} requested page(s) not cached "
                f"({', '.join(skipped)}); nothing was written",
                file=sys.stderr,
            )
            return 1
        summary["public_docs"] = _merge_records(
            Path(args.docsite_output),
            public_records,
            source_name="docsite",
        ) | public_summary

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _split_urls(raw: str) -> list[str]:
    seen: list[str] = []
    for item in re.split(r"[\s,]+", raw.strip()):
        if item and item not in seen:
            seen.append(item)
    return seen


def _url_name(url: str) -> str:
    parsed = urlparse(url)
    return re.sub(r"[^a-z0-9]+", "_", (parsed.netloc + parsed.path).lower()).strip("_") or url


def _download_gitlab_docs(
    manifest: dict[str, Any],
    *,
    max_bytes: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    header_candidates = _gitlab_header_candidates()
    targets = manifest["targets"]["gitlab_docs"]["repos"]
    suffixes = _allowed_suffixes(manifest)
    skip_fragments = tuple(
        str(item) for item in manifest.get("policy", {}).get(
            "default_skip_path_fragments", ()
        )
    )

    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "repos": {},
        "downloaded_records": 0,
        "skipped_files": 0,
    }
    session = requests.Session()

    for target in targets:
        if target.get("status") != "ready":
            continue
        project = str(target["project"])
        enc_project = quote(project, safe="")
        project_resp = _gitlab_get(
            session,
            f"{GITLAB_BASE}/api/v4/projects/{enc_project}",
            timeout=timeout,
            header_candidates=header_candidates,
        )
        project_resp.raise_for_status()
        project_doc = project_resp.json()
        ref = project_doc.get("default_branch") or "main"
        path_with_namespace = project_doc.get("path_with_namespace") or project
        tree = _gitlab_tree(
            session,
            enc_project,
            ref=ref,
            timeout=timeout,
            header_candidates=header_candidates,
        )

        fetched = 0
        skipped = 0
        for item in tree:
            if item.get("type") != "blob":
                continue
            path = str(item.get("path") or "")
            reason = _skip_reason(
                path,
                suffixes=suffixes,
                skip_fragments=skip_fragments,
            )
            if reason:
                skipped += 1
                continue
            raw_resp = _gitlab_get(
                session,
                f"{GITLAB_BASE}/api/v4/projects/{enc_project}/"
                f"repository/files/{quote(path, safe='')}/raw",
                params={"ref": ref},
                timeout=timeout,
                header_candidates=header_candidates,
            )
            if raw_resp.status_code != 200:
                skipped += 1
                continue
            if len(raw_resp.content) > max_bytes:
                skipped += 1
                continue
            body = _decode_text(raw_resp.content)
            if not body.strip():
                skipped += 1
                continue
            records.append({
                "title": _title_for_path(path, body),
                "url": (
                    f"{GITLAB_BASE}/{path_with_namespace}/-/blob/"
                    f"{ref}/{path}"
                ),
                "body": _clean_text(body),
                "site_name": f"gitlab.cern.ch/{path_with_namespace}",
                "file_path": path,
                "repo": path_with_namespace,
            })
            fetched += 1

        summary["repos"][path_with_namespace] = {
            "tree_entries": len(tree),
            "fetched_records": fetched,
            "skipped_files": skipped,
            "default_branch": ref,
        }
        summary["downloaded_records"] += fetched
        summary["skipped_files"] += skipped

    return records, summary


def _gitlab_tree(
    session: requests.Session,
    enc_project: str,
    *,
    ref: str,
    timeout: int,
    header_candidates: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = _gitlab_get(
            session,
            f"{GITLAB_BASE}/api/v4/projects/{enc_project}/repository/tree",
            params={
                "recursive": "true",
                "per_page": 100,
                "page": page,
                "ref": ref,
            },
            timeout=timeout,
            header_candidates=header_candidates,
        )
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        total_pages = int(resp.headers.get("X-Total-Pages") or page)
        if page >= total_pages or not batch:
            return rows
        page += 1


def _gitlab_header_candidates() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for name in ("CERN_GITLAB_TOKEN", "GITLAB_CERN_TOKEN"):
        token = os.environ.get(name)
        if token and token not in seen:
            candidates.append({"PRIVATE-TOKEN": token})
            seen.add(token)
    candidates.append({})
    return candidates


def _gitlab_get(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    header_candidates: list[dict[str, str]],
    params: dict[str, Any] | None = None,
) -> requests.Response:
    last_response: requests.Response | None = None
    for headers in header_candidates:
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        last_response = resp
        if resp.status_code not in {401, 403}:
            return resp
    if last_response is not None:
        return last_response
    raise RuntimeError(f"{url}: no GitLab request attempted")


def _download_public_docs(
    manifest: dict[str, Any],
    *,
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    urls = manifest["targets"]["public_docs"]["urls"]
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "urls": {},
        "downloaded_records": 0,
        "skipped_urls": 0,
    }
    session = requests.Session()
    for item in urls:
        if item.get("status") != "ready":
            continue
        url = str(item["url"])
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        content_type = resp.headers.get("content-type", "").split(";")[0]
        if resp.status_code != 200 or "html" not in content_type:
            summary["skipped_urls"] += 1
            summary["urls"][item["name"]] = {
                "http_status": resp.status_code,
                "content_type": content_type,
                "emitted": False,
            }
            continue
        title, body = _html_title_body(resp.text)
        if not body:
            summary["skipped_urls"] += 1
            continue
        records.append({
            "title": title or str(item["name"]),
            "url": resp.url,
            "body": body,
            "site_name": urlparse(resp.url).netloc,
        })
        summary["urls"][item["name"]] = {
            "http_status": resp.status_code,
            "content_type": content_type,
            "final_url": resp.url,
            "emitted": True,
        }
        summary["downloaded_records"] += 1
    return records, summary


def _merge_records(
    path: Path,
    records: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, Any]:
    existing = _load_records(path)
    by_url = {
        str(record.get("url") or ""): record
        for record in existing
        if isinstance(record, dict) and record.get("url")
    }
    before = len(by_url)
    for record in records:
        by_url[str(record["url"])] = record
    merged = list(by_url.values())
    merged.sort(key=lambda item: (
        str(item.get("repo") or item.get("source_repo") or ""),
        str(item.get("file_path") or item.get("path") or ""),
        str(item.get("url") or ""),
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(merged, ensure_ascii=False, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")
    meta_path = path.with_name("meta.json")
    meta = {
        "fetched_at": time.time(),
        "record_count": len(merged),
        "content_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_name": source_name,
        "ttl_seconds": 3600,
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_path": str(path),
        "meta_path": str(meta_path),
        "records_before": before,
        "records_after": len(merged),
        "records_added_or_replaced": len(records),
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected JSON list")
    return [item for item in payload if isinstance(item, dict)]


def _allowed_suffixes(manifest: dict[str, Any]) -> set[str]:
    policy = manifest.get("policy", {})
    suffixes = set()
    for key in (
        "default_text_suffixes",
        "default_config_suffixes",
        "default_operator_code_suffixes",
    ):
        suffixes.update(str(item).lower() for item in policy.get(key, ()))
    return suffixes


def _skip_reason(
    path: str,
    *,
    suffixes: set[str],
    skip_fragments: tuple[str, ...],
) -> str:
    normalized = "/" + path.replace("\\", "/").strip("/")
    if any(fragment in normalized for fragment in skip_fragments):
        return "skip_path_fragment"
    suffix = Path(path).suffix.lower()
    if suffix not in suffixes:
        return "unsupported_suffix"
    return ""


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _title_for_path(path: str, body: str) -> str:
    if Path(path).suffix.lower() in {".md", ".rst"}:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or Path(path).stem
            if stripped and not set(stripped) <= {"=", "-", "~", "`"}:
                return stripped[:120]
    return re.sub(r"[_-]+", " ", Path(path).stem).strip().title() or path


def _html_title_body(markup: str) -> tuple[str, str]:
    from lxml import html as lxml_html

    doc = lxml_html.fromstring(markup)
    for bad in doc.xpath("//script|//style|//noscript"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)
    title = " ".join(doc.xpath("string(//title)").split())
    body = _clean_text(doc.text_content())
    return title, body


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


if __name__ == "__main__":
    raise SystemExit(main())
