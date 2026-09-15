"""Validate bounded input bytes and derive the caches Archi readers consume."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

FILES = frozenset(
    {
        "records.json",
        "fetch-receipt.json",
        "documentation.html",
        "documentation-fetch-receipt.json",
        "releases.map",
    }
)
MAX_BYTES = 8 * 1024 * 1024
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def digest(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def strict_json(body: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate snapshot JSON key")
            result[key] = value
        return result

    return json.loads(
        body,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("nonfinite snapshot JSON")
        ),
    )


def read_regular(path: Path, expected: str, limit: int = MAX_BYTES) -> bytes:
    if not isinstance(expected, str) or not DIGEST.fullmatch(expected):
        raise ValueError("invalid expected SHA256")
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("snapshot path must be absolute and normalized")
    # Anchor every directory component so a concurrent rename cannot redirect
    # this read through a replacement symlink. NONBLOCK prevents FIFO hangs
    # before fstat can refuse a non-regular final member. This protects only
    # the bytes returned here: the Jira and documentation readers reopen their
    # files by path afterwards, so the configuration tree must stay private to
    # the installation owner between verification and the run.
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            os.close(parent)
            parent = child
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
    except OSError as exc:
        raise ValueError("snapshot member unavailable or symlink/invalid path") from exc
    finally:
        os.close(parent)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("snapshot input must be a regular file")
        body = stream.read(limit + 1)
    if len(body) > limit or digest(body) != expected:
        raise ValueError("snapshot bytes differ from declared digest or size limit")
    return body


def read_snapshot(path: Path, expected_digest: str) -> dict[str, bytes]:
    manifest = strict_json(read_regular(path, expected_digest, 1024 * 1024))
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "files"}:
        raise ValueError("unsupported snapshot manifest")
    # Only the complete five-file shape can prepare a frozen installation.
    if (
        manifest["schema"] != "archi.frozen-snapshot/v1"
        or not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != FILES
    ):
        raise ValueError("snapshot inventory does not match schema")
    return {
        name: read_regular(path.parent / name, pin)
        for name, pin in manifest["files"].items()
    }


def prepare_caches(snapshot: dict[str, bytes]) -> dict[str, bytes]:
    if set(snapshot) != FILES:
        raise ValueError(
            "new frozen installation requires a complete five-file snapshot, including authentic releases.map"
        )
    records = strict_json(snapshot["records.json"])
    receipt = strict_json(snapshot["fetch-receipt.json"])
    page = strict_json(snapshot["documentation-fetch-receipt.json"])
    if (
        not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], dict)
    ):
        raise ValueError("bounded snapshot requires exactly one issue")
    key = records[0].get("key")
    if not isinstance(key, str) or not re.fullmatch(r"CMSPROD-[0-9]+", key):
        raise ValueError("unsupported bounded issue identity")
    if (
        not isinstance(receipt, dict)
        or receipt.get("complete_project_scope") is not False
        or receipt.get("record_count") != 1
    ):
        raise ValueError("issue receipt scope differs")
    fetched_at = receipt.get("fetched_at")
    if not isinstance(fetched_at, str) or not fetched_at:
        raise ValueError("issue capture time absent")
    url = "https://fts3-docs.web.cern.ch/fts3-docs/"
    if (
        not isinstance(page, dict)
        or page.get("url") != url
        or page.get("final_url") != url
        or page.get("http_status") != 200
    ):
        raise ValueError("documentation receipt differs")
    from archi.sources.docs import _extract_body, _extract_title

    html = snapshot["documentation.html"].decode("utf-8")
    title, body = _extract_title(html), _extract_body(html)
    if not title or not body:
        raise ValueError("documentation extraction is empty")
    cache = {
        "snapshots/jira/records.json": snapshot["records.json"],
        "snapshots/jira/meta.json": json.dumps(
            {"record_count": 1, "fetched_at": fetched_at, "projects": ["CMSPROD"]},
            sort_keys=True,
        ).encode(),
        "snapshots/docsite/records.json": json.dumps(
            [
                {
                    "url": url,
                    "title": title,
                    "body": body,
                    "site_name": urlparse(url).netloc,
                }
            ],
            sort_keys=True,
        ).encode(),
        "snapshots/cmssw/releases.map": snapshot["releases.map"],
    }
    cache.update(
        {"snapshots/receipts/" + name: value for name, value in snapshot.items()}
    )
    # verify_cache reads every member under the same limit; refuse here rather
    # than package a cache that every later run would reject.
    if any(len(value) > MAX_BYTES for value in cache.values()):
        raise ValueError("derived snapshot cache member exceeds the size limit")
    return cache


def verify_records(
    root: str, files: dict[str, str], member: str, key: str, allowlist: list[str]
) -> None:
    """Refuse a verified cache whose record identifiers differ from the allowlist."""
    if (
        not isinstance(allowlist, list)
        or not allowlist
        or not all(isinstance(item, str) and item for item in allowlist)
    ):
        raise ValueError("record allowlist must be a non-empty list of identifiers")
    records = strict_json(read_regular(Path(root) / member, files[member]))
    if (
        not isinstance(records, list)
        or [item.get(key) if isinstance(item, dict) else None for item in records]
        != allowlist
    ):
        raise ValueError("snapshot record identifiers differ from the allowlist")


def verify_cache(root: str, files: dict[str, str]) -> None:
    path = Path(root)
    expected = {
        "snapshots/jira/records.json",
        "snapshots/jira/meta.json",
        "snapshots/docsite/records.json",
        "snapshots/cmssw/releases.map",
    } | {"snapshots/receipts/" + name for name in FILES}
    if not path.is_absolute() or not isinstance(files, dict) or set(files) != expected:
        raise ValueError("frozen cache authority has the wrong root or inventory")
    for name, pin in files.items():
        if str(PurePosixPath(name)) != name:
            raise ValueError("noncanonical cache member")
        read_regular(path / name, pin)
