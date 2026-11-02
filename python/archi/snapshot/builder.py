"""Build and verify a checksummed cache snapshot for the archi readers.

``build`` reads one source directory per group (a config maps each group to
its directory and collection date, so one snapshot can mix collection dates),
and for every group:

1. checks each file the reader reads: present, valid JSON of the expected
   shape, every record well-formed by the reader's own skip rule. Any failure
   refuses the group with :class:`GroupRefused`, which names the group, the
   file and the record;
2. removes every email address from every string (JSON keys and values, and
   whole text files) with :func:`archi.enrichment.anonymizer.redact_email_addresses`;
3. drops every field the reader does not read (``groups.GroupSpec.schema``),
   plus the group's configured ``drop_fields``, keeping its ``keep_fields``;
4. runs the group's archi reader over the result: it must report health
   ``ok`` and a completed scope, and its facts must equal the facts it emits
   from the step-2 copy (so step 3 dropped nothing the reader reads).

If any group is refused, nothing is written. Otherwise each group becomes one
deterministic ``<group>.tar.zst`` (sorted members, fixed owner, mode and
mtime, PAX format, ``zstd -T1`` at a fixed level) and ``snapshot.lock.yaml``
records, per group, its collection date, archive SHA-256 and size, file
count, record count (as the reader counts records) and a digest of the
unpacked contents. For TWiki it also records the empty topic files it left
out (zero bytes or only whitespace/NUL bytes): ``empty_topic_files`` (a
count) and ``empty_topic_names``. More than the reader's limit
(max(10, 0.1% of the topic files)) refuses the group.

``verify`` checks archives against a lock file on a machine that has only the
archives: size and SHA-256 of each archive, then its members (count, paths
inside the group's directory, contents digest).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import bisect
import errno
import functools
import fnmatch
import hashlib
import html
import io
import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Optional

import yaml

from archi.enrichment.anonymizer import (
    email_address_spans,
    obfuscated_email_address_spans,
    redact_email_addresses,
    redact_email_addresses_with_count,
    redact_obfuscated_email_addresses,
    redact_obfuscated_email_addresses_with_count,
)
from archi.snapshot.groups import (
    GROUPS,
    SIGNATURE_SHARE,
    GroupSpec,
    JsonFile,
    Schema,
)

LOCK_NAME = "snapshot.lock.yaml"
LOCK_VERSION = 1
#: Every member's mtime (2000-01-01T00:00:00Z). Any fixed value works; zero
#: makes some tar tools warn about implausibly old files.
FIXED_MTIME = 946684800
DEFAULT_ZSTD_LEVEL = 19
VALIDATION_RUN_ID = "snapshot-validate"
_DATE = r"\d{4}-\d{2}-\d{2}"
_COLLECTED_RE = re.compile(rf"^({_DATE})(?:\.\.({_DATE}))?$")
_GROUP_KEYS = {"path", "collected", "file_dates", "note", "drop_fields", "keep_fields"}
#: A group note is a line or two for the lock, not a document.
NOTE_MAX_CHARS = 300
_CONFIG_KEYS = {"snapshot", "zstd_level", "groups"}


class SnapshotError(Exception):
    """The config, an archive or a lock file is unusable."""


class GroupRefused(SnapshotError):
    """One group's cache cannot go into a snapshot."""

    def __init__(self, group: str, reason: str) -> None:
        super().__init__(f"group {group!r} refused: {reason}")
        self.group = group
        self.reason = reason


class BuildRefused(SnapshotError):
    """One or more groups were refused; nothing was written."""

    def __init__(self, refusals: list[GroupRefused]) -> None:
        lines = "\n".join(f"  - {r}" for r in refusals)
        super().__init__(
            f"{len(refusals)} group(s) refused; no snapshot written:\n{lines}"
        )
        self.refusals = refusals


# --- config ------------------------------------------------------------------


@dataclass(frozen=True)
class GroupConfig:
    name: str
    path: Path
    collected: str
    drop_fields: tuple[str, ...] = ()
    keep_fields: tuple[str, ...] = ()
    #: Dates of single files that differ from ``collected`` (for example a
    #: file fetched earlier than the rest), by file name in the group
    #: directory. Carried into the lock.
    file_dates: tuple[tuple[str, str], ...] = ()
    #: Free text carried into the lock (for example how the dates were read).
    note: Optional[str] = None


@dataclass(frozen=True)
class SnapshotConfig:
    snapshot: str
    groups: tuple[GroupConfig, ...]
    zstd_level: int = DEFAULT_ZSTD_LEVEL


def load_config(path: str | Path) -> SnapshotConfig:
    """Read a build config. Relative group paths resolve against its directory."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SnapshotError(f"{config_path}: cannot read config: {exc}") from exc
    return parse_config(raw, base=config_path.parent)


def parse_config(raw: Any, *, base: Path) -> SnapshotConfig:
    if not isinstance(raw, dict):
        raise SnapshotError("config must be a mapping")
    unknown = set(raw) - _CONFIG_KEYS
    if unknown:
        raise SnapshotError(f"unknown config keys: {sorted(unknown)}")
    snapshot = raw.get("snapshot")
    if not isinstance(snapshot, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", snapshot):
        raise SnapshotError(
            "config 'snapshot' must be a name of letters, digits, '.', '_' or '-'"
        )
    level = raw.get("zstd_level", DEFAULT_ZSTD_LEVEL)
    if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 19:
        raise SnapshotError("config 'zstd_level' must be an integer from 1 to 19")
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, dict) or not groups_raw:
        raise SnapshotError("config 'groups' must map at least one group name")
    groups = []
    for name in sorted(groups_raw):
        if name not in GROUPS:
            raise SnapshotError(
                f"unknown group {name!r}; known groups: {sorted(GROUPS)}"
            )
        groups.append(_parse_group(name, groups_raw[name], base))
    return SnapshotConfig(snapshot=snapshot, groups=tuple(groups), zstd_level=level)


def _parse_group(name: str, raw: Any, base: Path) -> GroupConfig:
    if not isinstance(raw, dict):
        raise SnapshotError(f"group {name!r}: expected a mapping")
    unknown = set(raw) - _GROUP_KEYS
    if unknown:
        raise SnapshotError(f"group {name!r}: unknown keys {sorted(unknown)}")
    if not raw.get("path"):
        raise SnapshotError(f"group {name!r}: 'path' is required")
    if "collected" not in raw or raw["collected"] in (None, ""):
        raise SnapshotError(
            f"group {name!r}: 'collected' (the collection date) is required"
        )
    collected = _parse_collected(name, raw["collected"])
    path = Path(str(raw["path"])).expanduser()
    if not path.is_absolute():
        path = base / path
    return GroupConfig(
        name=name,
        path=path,
        collected=collected,
        drop_fields=_field_list(name, raw, "drop_fields"),
        keep_fields=_field_list(name, raw, "keep_fields"),
        file_dates=_file_dates(name, raw.get("file_dates")),
        note=_note(name, raw.get("note")),
    )


def _note(name: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError(f"group {name!r}: 'note' must be non-empty text")
    if len(value) > NOTE_MAX_CHARS:
        raise SnapshotError(
            f"group {name!r}: 'note' is {len(value)} characters; "
            f"the limit is {NOTE_MAX_CHARS}"
        )
    return value


def _file_dates(name: str, value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise SnapshotError(
            f"group {name!r}: 'file_dates' must map file names to dates"
        )
    return tuple(
        sorted(
            (str(file), _parse_collected(name, date, field=f"file_dates[{file!r}]"))
            for file, date in value.items()
        )
    )


def _parse_collected(name: str, value: Any, *, field: str = "collected") -> str:
    """``YYYY-MM-DD`` or a range ``YYYY-MM-DD..YYYY-MM-DD``, as a string.

    ``field`` names the config entry in errors (``collected`` or a
    ``file_dates`` entry).
    """
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value.isoformat()
    match = _COLLECTED_RE.match(str(value))
    if not match:
        raise SnapshotError(
            f"group {name!r}: {field!r} must be YYYY-MM-DD or "
            f"YYYY-MM-DD..YYYY-MM-DD, got {value!r}"
        )
    try:
        start = dt.date.fromisoformat(match.group(1))
        end = dt.date.fromisoformat(match.group(2)) if match.group(2) else start
    except ValueError as exc:
        raise SnapshotError(f"group {name!r}: {field!r} {value!r}: {exc}") from exc
    if end < start:
        raise SnapshotError(f"group {name!r}: {field!r} range ends before it starts")
    return str(value)


def _field_list(name: str, raw: dict, key: str) -> tuple[str, ...]:
    value = raw.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise SnapshotError(f"group {name!r}: {key!r} must be a list of field names")
    return tuple(sorted(set(value)))


# --- per-group preparation ---------------------------------------------------


@dataclass
class PreparedGroup:
    """A validated group: final file bytes by archive path, plus its lock row."""

    name: str
    collected: str
    files: dict[str, bytes]
    record_count: int
    addresses_removed: int
    obfuscated_removed: int
    ansi_stripped: int
    #: Addresses removed whole because only NFKC or an invisible character
    #: (soft hyphen, zero-width space or joiners, word joiner) hid them.
    normalized_removed: int
    dropped_fields: tuple[str, ...]
    kept_fields: tuple[str, ...]
    deep_dropped_keys: tuple[str, ...] = ()
    input_file: Optional[str] = None
    #: For groups whose text may fall back on invalid bytes (TWiki):
    #: pages, pages with fallback runs, and fallback bytes per decoding.
    text_stats: Optional[dict[str, Any]] = None
    file_dates: tuple[tuple[str, str], ...] = ()
    note: Optional[str] = None
    #: For groups that skip empty text files (TWiki): the ones left out,
    #: by path in the source directory. None for other groups.
    empty_files: Optional[tuple[str, ...]] = None


def prepare_group(
    group: GroupConfig, *, tmp_dir: str | Path | None = None
) -> PreparedGroup:
    """Validate, redact, prune and reader-check one group, or raise GroupRefused.

    An unreadable file (permission denied, an I/O error) or JSON nested too
    deeply to check refuses the group like any other defect, instead of
    ending the build with a traceback.
    """
    try:
        return _prepare_group(group, tmp_dir)
    except GroupRefused:
        raise
    except OSError as exc:
        raise GroupRefused(
            group.name, f"cannot read its cache: {type(exc).__name__}: {exc}"
        ) from exc
    except RecursionError as exc:
        raise GroupRefused(
            group.name, "a JSON value is nested too deeply to check"
        ) from exc


#: Rounds of (addresses, then spelled-out addresses) before a string must
#: be a fixed point of both; removing one token can join text into another.
_REDACTION_ROUNDS = 4


class _Counter:
    def __init__(self) -> None:
        self.removed = 0
        self.obfuscated = 0
        self.ansi = 0
        self.normalized = 0

    def strip_ansi(self, text: str) -> str:
        clean, count = _STRIP_RE.subn("", text)
        self.ansi += count
        return clean

    def redact(self, text: str) -> str:
        """Remove addresses, then spelled-out addresses (``jdoe[at]cern.ch``),
        repeating until neither changes the text (the post-check refuses a
        string that is still not clean)."""
        for _ in range(_REDACTION_ROUNDS):
            clean, removed = redact_email_addresses_with_count(text)
            clean, obfuscated = redact_obfuscated_email_addresses_with_count(clean)
            self.removed += removed
            self.obfuscated += obfuscated
            if clean == text:
                break
            text = clean
        return text


def _select_variant(spec: GroupSpec, source: Path) -> GroupSpec:
    if not spec.variants:
        return spec
    for variant in spec.variants:
        assert variant.primary_input is not None
        candidate = source / variant.primary_input
        if candidate.exists() or candidate.is_symlink():
            return variant
    names = " nor ".join(v.primary_input or "?" for v in spec.variants)
    raise GroupRefused(spec.name, f"neither {names} is in {source}")


def _prepare_group(group: GroupConfig, tmp_dir: str | Path | None) -> PreparedGroup:
    if not group.path.is_dir():
        raise GroupRefused(group.name, f"source directory {group.path} does not exist")
    # The configured directory may itself be a link; nothing inside it may be.
    source = group.path.resolve(strict=True)
    spec = _select_variant(GROUPS[group.name], source)
    deep = set(spec.deep_drop_keys)
    drop = set(spec.default_drop_fields) | set(group.drop_fields) | deep
    keep = (set(spec.default_keep_fields) | set(group.keep_fields)) - drop
    if (group.drop_fields or group.keep_fields) and spec.record_file is None:
        raise GroupRefused(
            group.name, "drop_fields/keep_fields need a JSON records file"
        )

    counter = _Counter()
    redacted: dict[str, bytes] = {}
    final: dict[str, bytes] = {}
    for json_file in spec.json_files:
        path = source / json_file.name
        if not (path.exists() or path.is_symlink()):
            if json_file.required:
                raise GroupRefused(group.name, f"{json_file.name} is missing in {source}")
            continue
        payload = _load_json(group.name, _safe_file(group.name, source, json_file.name))
        _check_shape(group.name, json_file, payload)
        clean = _redact_json(group.name, json_file.name, payload, counter)
        # Again after redaction: a record whose identity was only an address
        # is one the reader would now skip or drop.
        _check_shape(group.name, json_file, clean)
        archive_path = f"{spec.archive_dir}/{json_file.name}"
        redacted[archive_path] = _dump_json(clean)
        record_level = json_file is spec.record_file
        pruned = _prune_file(
            json_file,
            clean,
            drop=drop if record_level else set(),
            keep=keep if record_level else set(),
        )
        final[archive_path] = _dump_json(_drop_keys_deep(pruned, deep))
    text_stats: Optional[dict[str, Any]] = None
    empty_files: Optional[list[str]] = None
    if spec.text is not None:
        fallback = spec.text.fallback_invalid_bytes
        text_files = _text_files(spec, source)
        if spec.text.skip_empty:
            # Imported here, as groups.py imports the readers: verify runs
            # without the readers' dependencies.
            from archi.sources.twiki import is_blank_topic

            empty_files = []
        if fallback:
            text_stats = {
                "text_pages": 0,
                "pages_valid_utf8": 0,
                "pages_with_fallback_runs": 0,
                "pages_bytes_dropped": 0,
                "fallback_bytes": {"cp1252": 0, "latin-1": 0},
                "renamed_files": [],
            }
        for rel, path, shown in text_files:
            data = path.read_bytes()
            blank = empty_files is not None and is_blank_topic(data)
            if shown is not None and text_stats is not None:
                # An empty topic is listed too, though it is not archived.
                text_stats["renamed_files"].append(
                    {"source_name": shown, "archive_name": rel, "archived": not blank}
                )
            if blank:
                assert empty_files is not None
                empty_files.append(rel)
                continue
            page = _read_text(group.name, rel, path, fallback=fallback, data=data)
            text = page.text
            if text_stats is not None:
                text_stats["text_pages"] += 1
                if page.fallback_bytes:
                    text_stats["pages_with_fallback_runs"] += 1
                    for kind, count in page.fallback_bytes.items():
                        text_stats["fallback_bytes"][kind] += count
                    if page.bytes_dropped:
                        text_stats["pages_bytes_dropped"] += 1
                else:
                    text_stats["pages_valid_utf8"] += 1
            data = _clean(group.name, rel, text, counter).encode("utf-8")
            archive_path = f"{spec.archive_dir}/{rel}"
            redacted[archive_path] = data
            final[archive_path] = data
        if empty_files:
            _check_empty_files(group.name, empty_files, len(text_files))
    if not final:
        raise GroupRefused(group.name, f"no cache files found in {source}")

    for archive_path in final:
        try:
            spans = _hidden_address_spans(group.name, "path", archive_path)
        except GroupRefused as exc:
            raise GroupRefused(
                group.name, f"file path {archive_path}: path rule: {exc.reason}"
            ) from exc
        if spans:
            raise GroupRefused(
                group.name,
                f"file path {archive_path}: path rule: an address in it is "
                "written with full-width or invisible characters",
            )
        if redact_email_addresses(archive_path) != archive_path:
            raise GroupRefused(
                group.name, f"a file path contains an email address: {archive_path}"
            )
    for file_name, _date in group.file_dates:
        if f"{spec.archive_dir}/{file_name}" not in final:
            raise GroupRefused(
                group.name,
                f"file_dates names {file_name}, which the group does not archive",
            )
    _check_no_addresses(group.name, final)
    if group.note is not None:
        _check_string(group.name, "note", group.note)
    record_count = _reader_check(spec, redacted, final, tmp_dir)
    return PreparedGroup(
        name=group.name,
        collected=group.collected,
        files=final,
        record_count=record_count,
        addresses_removed=counter.removed,
        obfuscated_removed=counter.obfuscated,
        ansi_stripped=counter.ansi,
        normalized_removed=counter.normalized,
        dropped_fields=tuple(sorted(drop - deep)),
        kept_fields=tuple(sorted(keep)),
        deep_dropped_keys=tuple(sorted(deep)),
        input_file=spec.primary_input,
        text_stats=text_stats,
        file_dates=group.file_dates,
        note=group.note,
        empty_files=None if empty_files is None else tuple(empty_files),
    )


def _check_empty_files(group: str, empty: list[str], total: int) -> None:
    """Refuse the group when its empty text files look like an interrupted
    sync rather than topics empty upstream (the reader's rule)."""
    from archi.sources.twiki import empty_topic_limit, empty_topics_tolerated

    if len(empty) == total:
        raise GroupRefused(
            group,
            f"all {total} text files are zero bytes or only whitespace/NUL "
            f"bytes, e.g. {', '.join(empty[:3])}",
        )
    if not empty_topics_tolerated(len(empty), total):
        raise GroupRefused(
            group,
            f"{len(empty)} of {total} text files are zero bytes or only "
            "whitespace/NUL bytes, more than the "
            f"{empty_topic_limit(total)} allowed (an interrupted sync), "
            f"e.g. {', '.join(empty[:3])}",
        )


def _safe_file(group: str, root: Path, rel: str) -> Path:
    """``root/rel`` if it is a regular file reached without links or dot names."""
    parts = PurePosixPath(rel).parts
    if any(part.startswith(".") for part in parts):
        raise GroupRefused(
            group, f"{_shown_name(rel)}: hidden path (a name starts with '.')"
        )
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise GroupRefused(group, f"{_shown_name(rel)}: symbolic link")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise GroupRefused(group, f"{_shown_name(rel)}: resolves outside {root}")
    if not resolved.is_file():
        raise GroupRefused(group, f"{_shown_name(rel)}: not a regular file")
    return resolved


def _check_tree(group: str, root: Path) -> list[str]:
    """Every file below ``root``; refuses on any link or dot-named entry."""

    def fail(exc: OSError) -> None:
        raise exc

    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=fail):
        for name in (*dirnames, *filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            if name.startswith("."):
                raise GroupRefused(
                    group, f"{_shown_name(rel)}: hidden path (a name starts with '.')"
                )
            if full.is_symlink():
                raise GroupRefused(group, f"{_shown_name(rel)}: symbolic link")
        files.extend(
            (Path(dirpath) / name).relative_to(root).as_posix() for name in filenames
        )
    return files


def _drop_keys_deep(value: Any, keys: set[str]) -> Any:
    if not keys:
        return value
    if isinstance(value, dict):
        return {k: _drop_keys_deep(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_drop_keys_deep(item, keys) for item in value]
    return value


def _load_json(group: str, path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise GroupRefused(group, f"{path.name} is not UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GroupRefused(group, f"{path.name} is not valid JSON: {exc}") from exc


def _status_payload_records(
    group: str, spec: JsonFile, payload: Any
) -> list[tuple[str, Any]]:
    """The records of a WMStats collection object (see ``JsonFile.shape``)."""
    by_status = payload.get("status_payloads") if isinstance(payload, dict) else None
    if not isinstance(by_status, dict):
        raise GroupRefused(
            group, f"{spec.name}: expected a JSON list or a status_payloads object"
        )
    records: list[tuple[str, Any]] = []
    for status, body in by_status.items():
        result = body.get("result") if isinstance(body, dict) else None
        if not isinstance(result, list):
            raise GroupRefused(
                group, f"{spec.name}: status {status!r} has no result list"
            )
        for index, entry in enumerate(result):
            where = f"status {status!r} result {index}"
            if not isinstance(entry, dict):
                records.append((where, entry))
                continue
            records.extend(
                (f"{where} entry {key!r}", item) for key, item in entry.items()
            )
    return records


def _check_shape(group: str, spec: JsonFile, payload: Any) -> None:
    if spec.shape == "status_payloads" and not isinstance(payload, list):
        items: Iterable[tuple[str, Any]] = _status_payload_records(
            group, spec, payload
        )
    elif spec.shape in {"list", "status_payloads"}:
        if not isinstance(payload, list):
            raise GroupRefused(group, f"{spec.name}: expected a JSON list")
        items = (
            (f"record {index}", item) for index, item in enumerate(payload)
        )
    elif spec.shape == "mapping":
        if not isinstance(payload, dict):
            raise GroupRefused(group, f"{spec.name}: expected a JSON object")
        items = ((f"entry {key!r}", value) for key, value in payload.items())
    else:
        if not isinstance(payload, dict):
            raise GroupRefused(group, f"{spec.name}: expected a JSON object")
        reason = spec.identity(payload) if spec.identity is not None else None
        if reason:
            raise GroupRefused(group, f"{spec.name}: {reason}")
        return
    records = list(items)
    if spec.identity is not None:
        for where, item in records:
            reason = spec.identity(item)
            if reason:
                raise GroupRefused(
                    group, f"{spec.name} {where} is malformed: {reason}"
                )
    if spec.signature is not None:
        if not records:
            raise GroupRefused(group, f"{spec.name} holds no records")
        carrying = sum(1 for _, item in records if spec.signature.matches(item))
        if carrying < SIGNATURE_SHARE * len(records):
            raise GroupRefused(
                group,
                f"{spec.name}: only {carrying} of {len(records)} records carry "
                f"{spec.signature.description} ({SIGNATURE_SHARE:.0%} needed); "
                "is it another export?",
            )


def _clean(group: str, where: str, text: str, counter: _Counter) -> str:
    """Check ``text`` for hidden addresses, strip its colour codes, redact it.

    The check runs on the raw text and again on the stripped text, since
    stripping can join what it separated. Addresses hidden only by NFKC or
    an invisible character are removed whole first (see
    :func:`_hidden_address_spans`).
    """
    surrogate = _SURROGATE_RE.search(text)
    if surrogate:
        raise GroupRefused(
            group,
            f"{where}, line {_line_number(text, surrogate.start())}: surrogate "
            "rule: an unpaired UTF-16 surrogate (not valid Unicode)",
        )
    text = _remove_normalized(group, where, text, counter)
    stripped = counter.strip_ansi(text)
    if stripped != text:
        stripped = _remove_normalized(group, where, stripped, counter)
    return counter.redact(stripped)


def _remove_normalized(group: str, where: str, text: str, counter: _Counter) -> str:
    for _ in range(_REDACTION_ROUNDS):
        spans = _hidden_address_spans(group, where, text)
        if not spans:
            return text
        counter.normalized += len(spans)
        pieces = []
        last = 0
        for start, end in spans:
            pieces.append(text[last:start])
            last = end
        pieces.append(text[last:])
        text = "".join(pieces)
    if _hidden_address_spans(group, where, text):
        raise GroupRefused(
            group,
            f"{where}: span rule: an address is still hidden after "
            f"{_REDACTION_ROUNDS} rounds of removing others",
        )
    return text


def _redact_json(
    group: str, file_name: str, value: Any, counter: _Counter, path: str = ""
) -> Any:
    """Redact every string and key of ``value``. ``path`` locates it for a
    refusal message (``records.json [3].description``); a key that is not
    short plain text shows as ``?``, so a message never repeats one."""
    where = f"{file_name} {path or '(top level)'}"
    if isinstance(value, str):
        return _clean(group, where, value, counter)
    if isinstance(value, list):
        return [
            _redact_json(group, file_name, item, counter, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            shown = key if _PATH_KEY_RE.fullmatch(key) else "?"
            clean_key = _clean(group, f"{where} (key {shown})", key, counter)
            if clean_key in out:
                raise GroupRefused(
                    group,
                    f"{file_name}: removing addresses makes two keys equal "
                    f"({clean_key!r})",
                )
            out[clean_key] = _redact_json(
                group, file_name, item, counter, f"{path}.{shown}"
            )
        return out
    return value


#: A JSON key short and plain enough to show in a refusal message.
_PATH_KEY_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")


def _prune(value: Any, schema: Schema) -> Any:
    if schema is None:
        return value
    if isinstance(value, dict):
        return {k: _prune(v, schema[k]) for k, v in value.items() if k in schema}
    if isinstance(value, list):
        return [_prune(item, schema) for item in value]
    return value


def _prune_file(spec: JsonFile, payload: Any, *, drop: set[str], keep: set[str]) -> Any:
    """Keep the fields the schema names (plus ``keep``), then remove ``drop``.

    ``drop`` and ``keep`` name top-level fields of each record.
    """
    schema: Schema = spec.schema
    if schema is not None and keep:
        schema = {**{name: None for name in keep}, **schema}

    def record(item: Any) -> Any:
        pruned = _prune(item, schema)
        if drop and isinstance(pruned, dict):
            pruned = {k: v for k, v in pruned.items() if k not in drop}
        return pruned

    if spec.shape in {"list", "status_payloads"} and isinstance(payload, list):
        return [record(item) for item in payload]
    if spec.shape == "status_payloads":
        # Only the containers the reader walks survive (cutoff_utc goes).
        return {
            "status_payloads": {
                status: {
                    "result": [
                        {key: record(item) for key, item in entry.items()}
                        if isinstance(entry, dict)
                        else entry
                        for entry in body["result"]
                    ]
                }
                for status, body in payload["status_payloads"].items()
            }
        }
    if spec.shape == "mapping":
        return {key: record(item) for key, item in payload.items()}
    return record(payload)


def _dump_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _text_files(spec: GroupSpec, source: Path) -> list[tuple[str, Path, Optional[str]]]:
    """``(archive name, path, source name if renamed)`` for each text file."""
    assert spec.text is not None
    if spec.text.recursive:
        names = [
            rel
            for rel in _check_tree(spec.name, source)
            if fnmatch.fnmatchcase(PurePosixPath(rel).name, spec.text.pattern)
        ]
    else:
        candidate = source / spec.text.pattern
        if not (candidate.exists() or candidate.is_symlink()):
            raise GroupRefused(spec.name, f"{spec.text.pattern} is missing in {source}")
        names = [spec.text.pattern]
    files: list[tuple[str, Path, Optional[str]]] = []
    seen: dict[str, str] = {}
    folders: dict[str, str] = {}
    for rel in sorted(names):
        path = _safe_file(spec.name, source, rel)
        archive_rel, renamed = _archive_name(
            spec.name, rel, spec.text.fallback_invalid_bytes
        )
        parents = [p.as_posix() for p in PurePosixPath(archive_rel).parents][:-1]
        clash = (
            seen.get(archive_rel)
            or folders.get(archive_rel)
            or next((seen[p] for p in parents if p in seen), None)
        )
        if clash is not None:
            raise GroupRefused(
                spec.name,
                f"file names {_shown_name(clash)} and {_shown_name(rel)} "
                f"collide as {archive_rel} once their invalid bytes are decoded",
            )
        seen[archive_rel] = rel
        for parent in parents:
            folders.setdefault(parent, rel)
        files.append((archive_rel, path, _shown_name(rel) if renamed else None))
    return files


def _archive_name(group: str, rel: str, fallback: bool) -> tuple[str, bool]:
    """``rel`` as valid Unicode, and whether its bytes had to be decoded.

    Python reads a file name that is not UTF-8 with ``surrogateescape``: each
    invalid byte becomes one lone surrogate (U+DC80 to U+DCFF), which UTF-8
    cannot store, so the reader (and the archive) would fail on it. With
    ``fallback`` each such byte becomes its cp1252 (else Latin-1) character,
    as in page text (a TWiki ``DsTotauTo`` + ``0xB5`` x3 reads as ``µµµ``);
    without it the group is refused. It is refused too when a byte would
    become a control or invisible character (0x81, 0xA0, 0xAD ...), or when
    the name with its invalid bytes left out holds an address (``jdoe`` +
    0x93 + ``@cern.ch`` would otherwise be stored as ``jdoe“@cern.ch``).
    """
    if not _SURROGATE_RE.search(rel):
        return rel, False
    shown = _shown_name(rel)
    if not fallback:
        raise GroupRefused(group, f"file name {shown} is not valid UTF-8")
    chars = []
    for char in rel:
        if "\udc80" <= char <= "\udcff":
            char = _fallback_char(ord(char) - 0xDC00)[0]
            if not char.isprintable() or _HIDDEN_RE.match(char):
                raise GroupRefused(
                    group,
                    f"file name {shown}: an invalid byte would become the "
                    f"hidden character U+{ord(char):04X}",
                )
        chars.append(char)
    decoded = "".join(chars)
    dropped = "".join(char for char in rel if not _SURROGATE_RE.match(char))
    for label, reading in (
        ("decoded", decoded),
        ("with its invalid bytes left out", dropped),
        ("as listed in the lock", shown),
    ):
        _check_string(group, f"file name {shown} ({label})", reading)
    return decoded, True


def _shown_name(rel: str) -> str:
    """A file name as printable ASCII-safe text: invalid bytes as ``\\xNN``."""
    return os.fsencode(rel).decode("utf-8", errors="backslashreplace")


#: C0 controls other than tab, newline, vertical tab, form feed and carriage
#: return; DEL; and C1. Counted by the binary guard.
_CONTROL_RE = re.compile("[\x00-\x08\x0e-\x1f\x7f-\x9f]")
#: Invisible characters the redactor itself reads inside an address (soft
#: hyphen, zero-width space, non-joiner and joiner, word joiner). Hidden, but
#: an address that only they (or NFKC) hide is removed whole, not refused.
_SOFT_HIDDEN = frozenset("\u00ad\u200b\u200c\u200d\u2060")
#: Hidden characters that change what a screen shows: the control
#: characters above (with backspace, ESC and C1) and every invisible format
#: character (Unicode category Cf: bidi controls such as U+200E, U+202E,
#: U+2068, and U+FEFF ...) except :data:`_SOFT_HIDDEN`.
_HARD_HIDDEN_RE = re.compile(
    "[\x00-\x08\x0e-\x1f\x7f-\x9f"
    + "".join(
        re.escape(chr(code))
        for code in range(0x110000)
        if unicodedata.category(chr(code)) == "Cf"
        and chr(code) not in _SOFT_HIDDEN
    )
    + "]"
)
#: All hidden characters: :data:`_HARD_HIDDEN_RE` and :data:`_SOFT_HIDDEN`.
_HIDDEN_RE = re.compile(
    _HARD_HIDDEN_RE.pattern[:-1]
    + "".join(sorted(_SOFT_HIDDEN))
    + "]"
)
#: HTML character references a browser would decode: numeric, hex and
#: named (with or without the ``;`` where browsers accept that).
_ENTITY_RE = re.compile(
    r"&(?:#[0-9]{1,7};?|#[xX][0-9a-fA-F]{1,6};?|[A-Za-z][A-Za-z0-9]{1,31};?)"
)
#: An unpaired UTF-16 surrogate: JSON can carry one, UTF-8 cannot store it.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")
#: A page is binary when more than this share of its characters, and at
#: least ``CONTROL_MIN_COUNT`` of them, are control characters.
CONTROL_SHARE_LIMIT = 0.01
CONTROL_MIN_COUNT = 16
#: What is stripped from stored text after the hidden-address check and
#: before redaction: ANSI CSI sequences (terminal colours pasted from logs,
#: ``ESC [`` + parameters + a final byte) and two-character escapes
#: (``ESC`` + one byte from ``0`` to ``~``, such as ``ESC 7`` and ``ESC 8``,
#: save and restore cursor).
_STRIP_RE = re.compile(
    r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]|\x1b[\x30-\x7e]"
)
#: Longest body of an OSC, DCS, SOS, PM or APC string the hidden-address
#: check reads as one sequence. Past it, or at a control character inside
#: the body, the opening character is a lone hidden character; this keeps
#: the scan linear on unterminated strings.
STRING_SEQUENCE_CAP = 4096
#: Terminal escape sequences, for the hidden-address check only (they are
#: not stripped), tried in this order: CSI in its 7-bit and 8-bit (U+009B)
#: forms; OSC, DCS, SOS, PM and APC strings (``ESC ] 0;title BEL``,
#: ``ESC P`` ... ``ESC \``, 7-bit or 8-bit); and any other ``ESC`` +
#: optional intermediates + a final byte (``ESC ( B`` from ``tput sgr0``,
#: ``ESC >``, ``ESC \``).
_SEQUENCE_RE = re.compile(
    r"(?:\x1b\[|\x9b)[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"
    r"|(?:\x1b[\]PX^_]|[\x90\x98\x9d\x9e\x9f])"
    rf"[^\x00-\x07\x0e-\x1f\x7f-\x9f]{{0,{STRING_SEQUENCE_CAP}}}"
    r"(?:\x07|\x1b\\|\x9c)"
    r"|\x1b[\x20-\x2f]*[\x30-\x7e]"
)
#: Text the hidden-address check can pass over unread: printable ASCII and
#: tab, newline, vertical tab, form feed and carriage return.
_PLAIN_RE = re.compile(r"[\x09-\x0d\x20-\x7e]*")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")


@functools.lru_cache(maxsize=65536)
def _fold(char: str) -> str:
    """NFKC of one character; the ideographic full stops (U+3002, and
    U+FF61, which NFKC makes U+3002) read as ``.``."""
    folded = unicodedata.normalize("NFKC", char)
    return "." if folded == "\u3002" else folded


def _reads_as_itself(text: str) -> bool:
    """Whether both readings of :func:`_canonical` are ``text`` itself: no
    hidden character, no terminal sequence, nothing NFKC changes."""
    if _decode_entities(text) is not None:
        return False
    if _PLAIN_RE.fullmatch(text):
        return True
    if _HIDDEN_RE.search(text) or _SEQUENCE_RE.search(text):
        return False
    return all(_fold(char) == char for char in set(_NON_ASCII_RE.findall(text)))


@dataclass
class _Decoded:
    """``raw`` with its HTML character references decoded, mapped back.

    Segment ``k`` starts at ``starts[k]`` in ``text`` and comes from
    ``raw[sources[k]:ends[k]]``: a run copied one for one, or one reference
    (listed by its decoded positions in ``entities``).
    """

    text: str
    starts: list[int]
    sources: list[int]
    ends: list[int]
    copied: list[bool]
    entities: list[int]

    def raw_start(self, index: int) -> int:
        k = bisect.bisect_right(self.starts, index) - 1
        if self.copied[k]:
            return self.sources[k] + index - self.starts[k]
        return self.sources[k]

    def raw_end(self, index: int) -> int:
        k = bisect.bisect_right(self.starts, index) - 1
        if self.copied[k]:
            return self.sources[k] + index - self.starts[k] + 1
        return self.ends[k]


def _decode_entities(raw: str) -> Optional[_Decoded]:
    """``raw`` as a browser shows its character references, or None when
    it has none that decode."""
    if "&" not in raw:
        return None
    parts: list[str] = []
    starts: list[int] = []
    sources: list[int] = []
    ends: list[int] = []
    copied: list[bool] = []
    entities: list[int] = []
    length = 0
    last = 0
    for match in _ENTITY_RE.finditer(raw):
        decoded = html.unescape(match.group())
        if decoded == match.group():
            continue
        if match.start() > last:
            parts.append(raw[last : match.start()])
            starts.append(length)
            sources.append(last)
            ends.append(match.start())
            copied.append(True)
            length += match.start() - last
        parts.append(decoded)
        starts.append(length)
        sources.append(match.start())
        ends.append(match.end())
        copied.append(False)
        entities.extend(range(length, length + len(decoded)))
        length += len(decoded)
        last = match.end()
    if not entities:
        return None
    if last < len(raw):
        parts.append(raw[last:])
        starts.append(length)
        sources.append(last)
        ends.append(len(raw))
        copied.append(True)
    return _Decoded("".join(parts), starts, sources, ends, copied, entities)


def _bases(text: str) -> list[tuple[str, Optional[_Decoded]]]:
    """The texts the hidden-address check reads: ``text`` itself and, when
    it has character references that decode, ``text`` decoded."""
    decoded = _decode_entities(text)
    if decoded is None:
        return [(text, None)]
    return [(text, None), (decoded.text, decoded)]


def _readings(base: str) -> list[str]:
    """The canonical readings of ``base`` (see :func:`_canonical`)."""
    texts = [_canonical(base).text]
    if _SEQUENCE_RE.search(base):
        texts.append(_canonical(base, sequences=False).text)
    return texts


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


_CONTROL_NAMES = {
    "\x08": "a backspace",
    "\x1b": "an ESC",
    "\x7f": "a DEL",
}


def _describe(char: str) -> str:
    """``char`` named for a refusal message."""
    if char in _CONTROL_NAMES:
        return _CONTROL_NAMES[char]
    name = unicodedata.name(char, "")
    kind = "control" if not name else name
    return f"U+{ord(char):04X} ({kind})"


@dataclass
class _Page:
    text: str
    fallback_bytes: dict[str, int]
    bytes_dropped: bool = False


def _fallback_char(byte: int) -> tuple[str, str]:
    """One invalid UTF-8 byte as cp1252, or Latin-1 where cp1252 has none."""
    try:
        return bytes([byte]).decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return chr(byte), "latin-1"


def _redaction_mask(text: str) -> bytearray:
    """Which characters of ``text`` the builder's redaction removes.

    The same steps as :meth:`_Counter.redact` (addresses, then spelled-out
    addresses, repeated until neither changes the text), tracked back to
    positions in ``text``.
    """
    mask = bytearray(len(text))
    current = text
    positions = list(range(len(text)))
    for _ in range(_REDACTION_ROUNDS):
        changed = False
        for spans_of in (email_address_spans, obfuscated_email_address_spans):
            spans = spans_of(current)
            if not spans:
                continue
            changed = True
            drop = bytearray(len(current))
            for start, end in spans:
                drop[start:end] = b"\x01" * (end - start)
                for index in range(start, end):
                    mask[positions[index]] = 1
            current = "".join(c for c, d in zip(current, drop, strict=True) if not d)
            positions = [p for p, d in zip(positions, drop, strict=True) if not d]
        if not changed:
            break
    return mask


_HARD = 1
_SOFT = 2


@dataclass
class _Canonical:
    """``text`` as the hidden-address check reads it, mapped back to ``text``.

    ``text`` minus its removed characters (hidden characters and, with
    ``sequences``, every terminal sequence), with NFKC applied to each
    remaining character. ``removed[i]`` is ``_HARD`` for a control
    character, a terminal-sequence character or a Cf character, ``_SOFT``
    for one of :data:`_SOFT_HIDDEN`, else 0. ``starts[k]`` is where segment
    ``k`` begins in the canonical text and ``sources[k]`` where it comes
    from in the raw text; a segment is a run copied one for one
    (``widths[k] == 1``), or one raw character that NFKC changed (listed in
    ``changed``).
    """

    text: str
    removed: bytearray
    starts: list[int]
    sources: list[int]
    widths: list[int]
    changed: list[int]

    def raw(self, index: int) -> int:
        k = bisect.bisect_right(self.starts, index) - 1
        if self.widths[k] == 1:
            return self.sources[k] + index - self.starts[k]
        return self.sources[k]


def _canonical(text: str, *, sequences: bool = True) -> _Canonical:
    removed = bytearray(len(text))
    for match in _HIDDEN_RE.finditer(text):
        removed[match.start()] = _SOFT if match.group() in _SOFT_HIDDEN else _HARD
    if sequences:
        for match in _SEQUENCE_RE.finditer(text):
            start, end = match.span()
            removed[start:end] = bytes([_HARD]) * (end - start)
    parts: list[str] = []
    starts: list[int] = []
    sources: list[int] = []
    widths: list[int] = []
    changed: list[int] = []
    length = 0
    index = 0
    while index < len(text):
        end = _PLAIN_RE.match(text, index).end()
        while index < end:
            # A run of plain characters, cut where a removed one sits.
            if removed[index]:
                index += 1
                continue
            stops = (removed.find(_HARD, index, end), removed.find(_SOFT, index, end))
            stop = min((stop for stop in stops if stop >= 0), default=end)
            parts.append(text[index:stop])
            starts.append(length)
            sources.append(index)
            widths.append(1)
            length += stop - index
            index = stop
        if index >= len(text):
            break
        if not removed[index]:
            char = text[index]
            folded = _fold(char)
            parts.append(folded)
            starts.append(length)
            sources.append(index)
            if folded == char:
                widths.append(1)
            else:
                widths.append(0)
                changed.append(index)
            length += len(folded)
        index += 1
    return _Canonical("".join(parts), removed, starts, sources, widths, changed)


def _hidden_address_spans(group: str, where: str, text: str) -> list[tuple[int, int]]:
    """Refuse ``text`` if a control character hides an address in it; return
    the raw spans of addresses that only NFKC or an invisible character hide.

    Runs on the raw value, before terminal sequences are stripped. The value
    is read as :func:`_canonical` does: terminal sequences (:data:`_SEQUENCE_RE`)
    and hidden characters (:data:`_HIDDEN_RE`) removed and NFKC applied per
    character, as a terminal shows it; and, when it has a terminal sequence,
    with only the hidden characters removed, as a program that ignores
    control characters sees it. Both redactors (addresses, then spelled-out
    addresses, repeated) match each reading, and each match is mapped back
    to the raw value:

    - it refuses the group when it contains or directly touches a control
      character, ``ESC``, a terminal-sequence character or a Cf character
      other than :data:`_SOFT_HIDDEN`;
    - else, when it contains a :data:`_SOFT_HIDDEN` character or a character
      NFKC changed (``jdoe<U+FF20>example.org``, ``jdoe [at]
      example<U+FF0E>org``), its raw span, first character to last, is
      returned for removal;
    - else it is left to the normal redaction. A character NFKC changed or
      an invisible one that only touches a match (``<U+FF1A>jdoe@example.org``,
      ``jdoe@example.org<U+2026>``) has no effect.

    A value that reads as itself (:func:`_reads_as_itself`) returns no
    spans at once: the normal redaction sees exactly what the readings see.
    """
    if _reads_as_itself(text):
        return []
    spans: list[tuple[int, int]] = []
    for base, decoded in _bases(text):
        readings = [True]
        if _SEQUENCE_RE.search(base):
            readings.append(False)
        for sequences in readings:
            view = _canonical(base, sequences=sequences)
            # A decoded character reference counts as a changed character.
            changed = (
                view.changed
                if decoded is None
                else sorted(set(view.changed) | set(decoded.entities))
            )
            if not changed and not any(view.removed):
                continue
            mask = bytes(_redaction_mask(view.text))
            for run in re.finditer(b"\x01+", mask):
                low = view.raw(run.start())
                high = view.raw(run.end() - 1)
                window = view.removed[max(low - 1, 0) : high + 2]
                if _HARD in window:
                    at = max(low - 1, 0) + window.index(_HARD)
                    raw = at if decoded is None else decoded.raw_start(at)
                    what = (
                        _describe(base[at])
                        if _HIDDEN_RE.match(base[at])
                        else "part of a terminal escape sequence"
                    )
                    raise GroupRefused(
                        group,
                        f"{where}, line {_line_number(text, raw)}: span rule: "
                        f"an address contains or touches {what}",
                    )
                first = bisect.bisect_left(changed, low)
                if _SOFT in view.removed[low : high + 1] or (
                    first < len(changed) and changed[first] <= high
                ):
                    if decoded is None:
                        spans.append((low, high + 1))
                    else:
                        spans.append((decoded.raw_start(low), decoded.raw_end(high)))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _decode_with_fallback(data: bytes) -> _Page:
    """Decode UTF-8, falling back per invalid byte; drop them if that hides
    an address.

    ``surrogateescape`` keeps each invalid byte as one placeholder character,
    so two views of the page line up character for character:

    - the fallback view, where each invalid byte becomes its cp1252 (else
      Latin-1) character, which keeps a stray Windows quote readable;
    - the dropped view, where invalid bytes are simply left out.

    A printable fallback character next to or inside an address (``jean.dupont``,
    a stray 0x93, ``@cern.ch``, or ``jdoe``, 0x93, ``[at]cern.ch``) can hide it
    from the redactor. So when any character that redaction removes from the
    dropped view (addresses and spelled-out addresses) would survive redaction
    of the fallback view, the page uses the dropped view instead. It does
    the same when redaction of the fallback view removes a control
    character that an invalid byte became (Latin-1 C1), since the span rule
    would refuse that address.
    """
    raw = data.decode("utf-8", errors="surrogateescape")
    counts = {"cp1252": 0, "latin-1": 0}
    fallback_chars: list[str] = []
    kept_positions: list[int] = []
    fallback_positions: list[int] = []
    for index, char in enumerate(raw):
        if "\udc80" <= char <= "\udcff":
            decoded, kind = _fallback_char(ord(char) - 0xDC00)
            counts[kind] += 1
            fallback_chars.append(decoded)
            fallback_positions.append(index)
        else:
            fallback_chars.append(char)
            kept_positions.append(index)
    if not any(counts.values()):
        return _Page(raw, {})
    fallback_view = "".join(fallback_chars)
    dropped_view = "".join(raw[i] for i in kept_positions)
    removed = _redaction_mask(fallback_view)
    found = _redaction_mask(dropped_view)
    survives = any(
        flag and not removed[kept_positions[i]] for i, flag in enumerate(found)
    )
    # The redactor reads a C1 control from the Latin-1 fallback as part of
    # an address (``a.b@c<U+0081>d.ch`` is removed whole), but the span rule
    # refuses an address that holds a control character. Dropping the byte
    # removes the same address without refusing the group.
    inside = any(
        removed[i] and _HARD_HIDDEN_RE.match(fallback_view[i])
        for i in fallback_positions
    )
    if survives or inside:
        return _Page(dropped_view, counts, bytes_dropped=True)
    return _Page(fallback_view, counts)


def _read_text(
    group: str,
    rel: str,
    path: Path,
    *,
    fallback: bool = False,
    data: Optional[bytes] = None,
) -> _Page:
    """The file's text, decoded and checked.

    A NUL byte always refuses the group (UTF-16 puts a NUL between the letters
    of an address, hiding it from the redactor). Invalid UTF-8 refuses it
    unless ``fallback`` (see :func:`_decode_with_fallback`). Decoded text
    that is more than 1% control characters, and at least 16 of them, not
    counting ANSI colour sequences, is refused as binary. The text is
    returned with its colour sequences; they are stripped after the
    hidden-address check.
    """
    if data is None:
        data = path.read_bytes()
    if b"\0" in data:
        raise GroupRefused(group, f"{rel} contains NUL bytes (not UTF-8 text)")
    if fallback:
        page = _decode_with_fallback(data)
    else:
        try:
            page = _Page(data.decode("utf-8"), {})
        except UnicodeDecodeError as exc:
            raise GroupRefused(group, f"{rel} is not UTF-8: {exc}") from exc
    text = _STRIP_RE.sub("", page.text)
    if text:
        controls = len(_CONTROL_RE.findall(text))
        if controls >= CONTROL_MIN_COUNT and controls / len(text) > CONTROL_SHARE_LIMIT:
            raise GroupRefused(
                group,
                f"{rel} looks binary: {controls} of {len(text)} characters "
                f"({100 * controls / len(text):.1f}%) are control characters",
            )
    return page


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)


def _check_no_addresses(group: str, files: Mapping[str, bytes]) -> None:
    """Every string must be a fixed point of both redactors, also in both
    readings of :func:`_canonical` (terminal sequences and hidden characters
    removed, or only hidden characters removed, NFKC applied): a control
    character inside an address (for example a C1 control from the Latin-1
    fallback in ``a.b@c<U+0081>d.ch``) hides it from the redactor, so such a
    string refuses the group."""
    for path, data in files.items():
        text = data.decode("utf-8")
        strings = _strings(json.loads(text)) if path.endswith(".json") else [text]
        for value in strings:
            _check_string(group, path, value)


def _check_string(group: str, where: str, value: str) -> None:
    if redact_email_addresses(value) != value:
        raise GroupRefused(group, f"{where}: an address survived redaction")
    if redact_obfuscated_email_addresses(value) != value:
        raise GroupRefused(
            group, f"{where}: a spelled-out address survived redaction"
        )
    if _reads_as_itself(value):
        return
    if _address_in_a_reading(value):
        lines = value.split("\n")
        number = next(
            (n for n, line in enumerate(lines, 1) if _address_in_a_reading(line)),
            None,
        )
        at = "" if number is None else f", line {number}"
        raise GroupRefused(
            group,
            f"{where}{at}: stored-text rule: an address shows in the stored "
            "text once hidden characters are removed and NFKC applied",
        )


def _address_in_a_reading(value: str) -> bool:
    return any(
        redact_email_addresses(reading) != reading
        or redact_obfuscated_email_addresses(reading) != reading
        for base, _ in _bases(value)
        for reading in _readings(base)
    )


def _reader_check(
    spec: GroupSpec,
    redacted: Mapping[str, bytes],
    final: Mapping[str, bytes],
    tmp_dir: str | Path | None = None,
) -> int:
    """Run the reader over two staged copies in one temporary directory.

    The directory is made under ``tmp_dir`` (default: the system temp dir)
    and removed when this returns or raises. It holds two copies of the group,
    so it needs about twice the group's size.
    """
    with tempfile.TemporaryDirectory(
        prefix=f"snapshot-{spec.name}-", dir=tmp_dir
    ) as tmp:
        full_root = Path(tmp) / "redacted"
        final_root = Path(tmp) / "final"
        try:
            for root, files in ((full_root, redacted), (final_root, final)):
                _stage(root, spec.validation_stubs)
                _stage(root, files)
        except OSError as exc:
            full = exc.errno in (errno.ENOSPC, errno.EDQUOT)
            raise GroupRefused(
                spec.name,
                "cannot write the reader-check copy in "
                f"{tmp}{' (disk full?)' if full else ''}: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        full_facts, _ = _run_reader(spec, full_root)
        final_facts, record_count = _run_reader(spec, final_root)
    if full_facts != final_facts:
        detail = _first_difference(full_facts, final_facts)
        raise GroupRefused(
            spec.name,
            "dropping unread fields changed what the reader emits, so the "
            f"group's schema misses a field the reader reads ({detail})",
        )
    return record_count


def _stage(root: Path, files: Mapping[str, bytes]) -> None:
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _run_reader(spec: GroupSpec, root: Path) -> tuple[list[str], int]:
    try:
        reader = spec.make_reader(root)
        run = reader.run(VALIDATION_RUN_ID, mode="scope_complete")
        facts = [normalize_fact(fact) for fact in run.facts]
    except Exception as exc:  # the reader's own refusal, reported verbatim
        raise GroupRefused(
            spec.name, f"the archi reader failed: {type(exc).__name__}: {exc}"
        ) from exc
    health = run.health
    status = getattr(health, "status", None)
    if status != "ok" or not run.completed_scope:
        reason = getattr(health, "reason", "") or "no reason given"
        raise GroupRefused(
            spec.name,
            "the archi reader did not accept the whole cache "
            f"(health {status!r}, completed_scope={run.completed_scope}): {reason}",
        )
    count = getattr(health, "record_count", None)
    if not isinstance(count, int) or count <= 0:
        raise GroupRefused(
            spec.name, f"the archi reader reported {count!r} records"
        )
    return facts, count


def normalize_fact(fact: Any) -> str:
    """A fact as canonical JSON, without its run-specific revision."""
    if dataclasses.is_dataclass(fact):
        body = dataclasses.asdict(fact)
    else:
        body = dict(vars(fact))
    body.pop("source_revision", None)
    body["fact_type"] = type(fact).__name__
    return json.dumps(body, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _first_difference(left: list[str], right: list[str]) -> str:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return f"fact {index}: {a[:200]} != {b[:200]}"
    return f"{len(left)} facts before pruning, {len(right)} after"


# --- archives ----------------------------------------------------------------


def zstd_version() -> str:
    try:
        out = subprocess.run(
            ["zstd", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SnapshotError(
            "the zstd command is required to build or verify a snapshot"
        ) from exc
    match = re.search(r"v\d+\.\d+\.\d+", out)
    return match.group(0) if match else out.strip()


def write_archive(files: Mapping[str, bytes], target: Path, *, level: int) -> None:
    """Write ``files`` as a deterministic tar compressed by ``zstd -T1``."""
    with target.open("wb") as out:
        proc = subprocess.Popen(
            ["zstd", f"-{level}", "-T1", "-q", "-c", "--no-progress", "-"],
            stdin=subprocess.PIPE,
            stdout=out,
        )
        assert proc.stdin is not None
        try:
            with tarfile.open(
                fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT
            ) as tar:
                for name in sorted(files):
                    data = files[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = FIXED_MTIME
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.type = tarfile.REGTYPE
                    tar.addfile(info, io.BytesIO(data))
        finally:
            proc.stdin.close()
            code = proc.wait()
    if code != 0:
        raise SnapshotError(f"zstd exited {code} while writing {target.name}")


def read_archive(path: Path) -> dict[str, bytes]:
    """Decompress and read every member; refuses non-file or escaping members."""
    proc = subprocess.run(
        ["zstd", "-d", "-q", "-c", str(path)], capture_output=True
    )
    if proc.returncode != 0:
        raise SnapshotError(
            f"{path.name}: zstd cannot decompress it: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tar:
            for info in tar:
                name = info.name
                pure = PurePosixPath(name)
                if not info.isreg() or pure.is_absolute() or ".." in pure.parts:
                    raise SnapshotError(f"{path.name}: unsafe member {name!r}")
                if name in members:
                    raise SnapshotError(f"{path.name}: duplicate member {name!r}")
                handle = tar.extractfile(info)
                assert handle is not None
                members[name] = handle.read()
    except tarfile.TarError as exc:
        raise SnapshotError(f"{path.name}: not a readable tar: {exc}") from exc
    return members


def contents_digest(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(files[name]).hexdigest().encode("ascii") + b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- build and verify --------------------------------------------------------


def build(
    config: SnapshotConfig,
    out_dir: str | Path,
    *,
    built_by: Optional[str] = None,
    tmp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build every group, or raise BuildRefused having written nothing.

    ``tmp_dir`` holds the reader-check copies (default: the system temp dir).
    """
    if tmp_dir is not None and not Path(tmp_dir).is_dir():
        raise SnapshotError(f"--tmp-dir {tmp_dir} is not a directory")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise SnapshotError(f"{out} exists and is not empty; refusing to overwrite")
    version = zstd_version()
    prepared: list[PreparedGroup] = []
    refusals: list[GroupRefused] = []
    for group in config.groups:
        try:
            prepared.append(prepare_group(group, tmp_dir=tmp_dir))
        except GroupRefused as exc:
            refusals.append(exc)
    if refusals:
        raise BuildRefused(refusals)

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        groups_lock: dict[str, Any] = {}
        for group in prepared:
            archive = staging / f"{group.name}.tar.zst"
            write_archive(group.files, archive, level=config.zstd_level)
            row: dict[str, Any] = {
                "collected": group.collected,
                **({"file_dates": dict(group.file_dates)} if group.file_dates else {}),
                **({"note": group.note} if group.note is not None else {}),
                "archive": archive.name,
                "sha256": _sha256_file(archive),
                "bytes": archive.stat().st_size,
                "file_count": len(group.files),
                "record_count": group.record_count,
                "addresses_removed": group.addresses_removed,
                "obfuscated_addresses_removed": group.obfuscated_removed,
                "ansi_sequences_stripped": group.ansi_stripped,
                "normalized_addresses_removed": group.normalized_removed,
                "contents_sha256": contents_digest(group.files),
                "archive_dir": GROUPS[group.name].archive_dir,
                "dropped_fields": list(group.dropped_fields),
                "kept_extra_fields": list(group.kept_fields),
            }
            if group.deep_dropped_keys:
                row["dropped_keys_at_any_depth"] = list(group.deep_dropped_keys)
            if group.input_file:
                row["input_file"] = group.input_file
            if group.text_stats is not None:
                row.update(group.text_stats)
            if group.empty_files is not None:
                row["empty_topic_files"] = len(group.empty_files)
                row["empty_topic_names"] = list(group.empty_files)
            groups_lock[group.name] = row
        lock = {
            "lock_version": LOCK_VERSION,
            "snapshot": config.snapshot,
            "built_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "built_by": built_by or os.environ.get("USER") or "unknown",
            "built_on": socket.gethostname(),
            "archi": _archi_provenance(),
            "archive_format": {
                "tar": "pax",
                "mtime": FIXED_MTIME,
                "compressor": "zstd",
                "zstd_version": version,
                "zstd_level": config.zstd_level,
            },
            "groups": groups_lock,
        }
        (staging / LOCK_NAME).write_text(
            yaml.safe_dump(lock, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        if out.exists():
            out.rmdir()
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return lock


def _archi_provenance() -> dict[str, Any]:
    from archi import __version__ as archi_version

    package_dir = Path(__file__).resolve().parents[1]
    info: dict[str, Any] = {"version": archi_version}
    try:
        head = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(package_dir), "status", "--porcelain", "--", "."],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        info["commit"] = head
        info["tree_dirty"] = bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        info["commit"] = None
        info["commit_note"] = "archi is not running from a git checkout"
    return info


@dataclass
class VerifyResult:
    ok: bool
    lines: list[str]


def verify(lock_path: str | Path, archive_dir: str | Path) -> VerifyResult:
    """Check every archive a lock names; report every problem, not just the first."""
    lock_file = Path(lock_path)
    directory = Path(archive_dir)
    try:
        lock = yaml.safe_load(lock_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SnapshotError(f"{lock_file}: cannot read lock: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("lock_version") != LOCK_VERSION:
        raise SnapshotError(f"{lock_file}: not a version-{LOCK_VERSION} snapshot lock")
    groups = lock.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise SnapshotError(f"{lock_file}: lock names no groups")
    lines: list[str] = []
    ok = True
    for name in sorted(groups):
        problem = _verify_group(name, groups[name], directory)
        if problem:
            ok = False
            lines.append(f"FAIL {name}: {problem}")
        else:
            lines.append(f"ok   {name}")
    return VerifyResult(ok=ok, lines=lines)


def _verify_group(name: str, row: Any, directory: Path) -> Optional[str]:
    if not isinstance(row, dict):
        return "lock row is not a mapping"
    archive_name = row.get("archive")
    if not isinstance(archive_name, str) or "/" in archive_name:
        return f"lock names no valid archive ({archive_name!r})"
    archive = directory / archive_name
    if not archive.is_file():
        return f"{archive_name} is missing"
    size = archive.stat().st_size
    if size != row.get("bytes"):
        return f"{archive_name} is {size} bytes, lock says {row.get('bytes')}"
    sha = _sha256_file(archive)
    if sha != row.get("sha256"):
        return f"{archive_name} SHA-256 {sha} does not match lock {row.get('sha256')}"
    try:
        members = read_archive(archive)
    except SnapshotError as exc:
        return str(exc)
    if len(members) != row.get("file_count"):
        return f"{len(members)} files, lock says {row.get('file_count')}"
    prefix = f"{row.get('archive_dir')}/"
    outside = sorted(m for m in members if not m.startswith(prefix))
    if outside:
        return f"member outside {prefix}: {outside[0]}"
    if contents_digest(members) != row.get("contents_sha256"):
        return "unpacked contents do not match the lock's contents_sha256"
    if "empty_topic_files" in row or "empty_topic_names" in row:
        names = row.get("empty_topic_names")
        if not isinstance(names, list) or row.get("empty_topic_files") != len(names):
            return (
                f"lock says {row.get('empty_topic_files')!r} empty topic files "
                f"but names {names!r}"
            )
        archived = sorted(n for n in names if f"{prefix}{n}" in members)
        if archived:
            return f"empty topic file {archived[0]} is in the archive"
    return None
