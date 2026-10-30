"""The raw-document corpus the agent-bash arm reads (PACT r3-context-quality, task raw-corpus).

`archi eval --raw-corpus` turns an instance's own connector caches, the files its graph
was built from, into a directory of plain files, one per record:

1. Each enabled source of the instance's `source_registry.yaml` whose cache an
   `archi.snapshot` group reads is packed with `archi.snapshot.builder.build`: every
   file checked against the source's own reader, email addresses removed, fields the
   reader does not read dropped. A group the snapshot refuses refuses the corpus.
2. The archives are unpacked to `<data root>/eval/raw-corpus/<bundle>/<lock sha>/`,
   `<lock sha>` being the first 16 hex digits of the snapshot lock's SHA-256:

       README.md            what this is, for a person
       corpus.json          the same, for a program: sources, left-out sources, digest
       MANIFEST.txt         `<path>\\t<sha256>` per record file, sorted
       keys.json            per source: the names a node id may give -> record file, and
                            each file's natural key
       snapshot.lock.yaml   the lock of the snapshot the files came from
       records/             one directory per snapshot group, one file per record

   Only `records/` is mounted into the agent's sandbox (at /corpus), so the agent never
   reads the key map or the manifest, which name every record.
3. The corpus digest is `sha256:` of MANIFEST.txt. A run records it, and refuses a
   corpus whose files no longer match their manifest.

TWiki keeps only the pages the instance's TWiki reader reads with its own params (the
physics filter, seed topics, skip patterns), so the corpus matches what the graph read. A
git-code checkout goes in as its tracked files at HEAD, under `records/git-repo/<repo>/`:
no snapshot group checks a source tree, so email addresses are removed here, and binary and
oversized files are left out and counted. The instance's published generation (read through
MCP when the corpus is built) is recorded beside the digest.

Sources with no checked cache (live fetches, derived sources) are left out, each with its
reason, in corpus.json and README.md, and the coverage line says so in one sentence. So is ops-record: its
records exist only in the graph (okg admission), it has no cache, and an export through
the MCP tools could not be shown complete, so it is left out of both agent arms: gold
records from it count as unavailable in both, never as misses.

Derived files and okg state are never read: only the cache files a snapshot group names.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Optional, Sequence
from urllib.parse import quote

import yaml

CORPUS_SUBDIR = ("eval", "raw-corpus")
RECORDS_DIR = "records"
MANIFEST = "MANIFEST.txt"
KEYS = "keys.json"
INFO = "corpus.json"
README = "README.md"
CORPUS_SCHEMA = "archi.raw-corpus/1"

#: ops-record's producer (archi.ops.record.PRODUCER_ID) and its subtypes' prefix.
OPS_RECORD = "ops-record"
OPS_RECORD_PRODUCER = "workflow:archi-ops"
OPS_RECORD_REASON = (
    "ops-record has no cache: its records exist only in the graph (okg admission). An export through the "
    "MCP tools could not be shown complete, so ops-record is left out of both agent arms; its gold "
    "records count as unavailable in both"
)


@dataclass(frozen=True)
class Group:
    """How a registry source maps onto an `archi.snapshot` group."""

    group: str
    module: str
    #: the registry param that locates the cache: a `*_path` file (its directory is the
    #: group's) or a `*_root` directory (the group's directory itself)
    param: str
    #: record-key fields of the group's list records, first non-blank wins
    key_fields: tuple[str, ...] = ()


#: The TWiki reader params that decide which pages it reads; the corpus keeps exactly those pages.
#: (`page_index_path` only names where the reader writes the ids it emitted: it changes no page.)
TWIKI_SCOPE_PARAMS = ("web_root", "physics_filter", "seed_topics", "max_depth", "skip_patterns", "max_files")

#: Registry source name -> its snapshot group. A source whose registry module differs is
#: left out (indico_live is not the cache reader `indico` is).
SOURCE_GROUPS: dict[str, Group] = {
    "docsite": Group("docsite", "archi.sources.docs", "records_path", ("url",)),
    "gitlab_docs": Group("gitlab-docs", "archi.sources.docs", "records_path", ("url",)),
    "jira": Group("jira", "archi.sources.jira", "records_path", ("key", "issue_key")),
    "indico": Group("indico", "archi.sources.indico", "records_path", ("id", "event_id")),
    "dqm": Group("dqm", "archi.sources.dqm", "records_path", ("cert_name",)),
    "gocdb_downtimes": Group("gocdb-downtimes", "archi.sources.gocdb", "records_path", ("downtime_id",)),
    "cmssst_site_status": Group("cmssst-site-status", "archi.sources.cmssst", "records_path", ("site",)),
    "cric": Group("cric", "archi.sources.cric", "sites_path"),
    "cric_core": Group("cric-core", "archi.sources.cric", "services_path"),
    "cmssw_releases": Group("cmssw-releases", "archi.sources.cmssw", "map_cache_path", ("label",)),
    "conddb_global_tags": Group("conddb-global-tags", "archi.sources.conddb", "records_path", ("name", "tag_name")),
    "wmstats_workflows": Group("wmstats", "archi.sources.wmstats", "records_path",
                               ("workflow_name", "request_name", "RequestName")),
    "dbs_datasets": Group("dbs", "archi.sources.dbs", "records_path", ("dataset_name", "dataset")),
    "twiki_eos": Group("twiki-eos", "archi.sources.twiki", "eos_root"),
    "tickets": Group("tickets", "archi.sources.tickets", "records_path", ("key", "id")),
    "servicenow": Group("servicenow", "archi.sources.servicenow", "records_path", ("number",)),
    "node_health": Group("node-health", "archi.sources.node_health", "records_path", ("node",)),
    "slurm": Group("slurm", "archi.sources.slurm", "records_path"),
    "htcondor": Group("htcondor", "archi.sources.htcondor", "records_path"),
}

#: Why a known source has no files in the corpus.
LEFT_OUT_REASONS = {
    "indico_live": "fetched live from Indico; there is no cache for archi.snapshot to check",
    "cds": ("harvested live from the CDS OAI-PMH endpoint at ingest; no copy is kept on disk, so its records "
            "exist only in the graph, and an export through the MCP tools could not be shown complete"),
    "siteconf": "read live from GitLab; there is no cache for archi.snapshot to check",
    "hypernews": "crawled live through a login; there is no cache for archi.snapshot to check",
    "analysis_notes": ("a mirror of LaTeX trees (sources and figures) that no archi.snapshot group checks or "
                       "removes email addresses from; including it needs such a group first"),
    "physics_entities": "derived from other sources' records, not a cache",
    "physics_vocabulary": "a vocabulary shipped with archi, not a cache",
    "git_history": "commit history, not files; the corpus holds the checkout's tracked files (git_repo)",
    "code_structure": "derived from the git checkout, whose tracked files the corpus holds (git_repo)",
    "code_text": "derived from the git checkout, whose tracked files the corpus holds (git_repo)",
}
#: The git-code pack's file reader, and the corpus source its files go under: okg names its
#: producer `<deployment>.git-repo`, so a gold id's provenance finds them there.
GIT_FILES_MODULE = "okg.substrate.library.sources.git_graph_files"
GIT_SOURCE = "git_repo"
#: Repository files larger than this are left out of the corpus (and counted).
MAX_REPO_FILE_BYTES = 1024 * 1024
_MONIT = "queried live from MONIT; there is no cache for archi.snapshot to check"
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
#: Characters kept as they are in a record file name; anything else is %-escaped.
_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
_MAX_NAME = 150


class CorpusError(RuntimeError):
    """The corpus cannot be built, found or trusted; the message says why."""


# --- which sources go in -----------------------------------------------------------------


@dataclass(frozen=True)
class PlannedSource:
    source: str
    group: str
    directory: Path
    #: the registry entry's params with the deployment's values filled in
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    included: list[PlannedSource] = field(default_factory=list)
    left_out: dict[str, str] = field(default_factory=dict)
    #: git-code checkouts: (registry source, repository name, checkout directory, exclude globs)
    repos: list[tuple[str, str, Path, tuple[str, ...]]] = field(default_factory=list)


def plan_sources(registry: Mapping[str, Any], *, data_root: Path, answers: Optional[Mapping[str, Any]] = None,
                 deployment_dir: Optional[Path] = None) -> Plan:
    """Which enabled registry sources the snapshot can pack, and why each other one is left out."""
    values = {str(k): str(v) for k, v in (answers or {}).items() if v is not None}
    values["archi_data_root"] = str(data_root)
    plan = Plan()
    groups_seen: dict[str, str] = {}
    for key in sorted(registry):
        entry = registry[key]
        # A rendered instance names a source `<pack>.<source>` (twiki.twiki_eos); the groups, the
        # reasons and the gold lookup (raw_arm.record_for) all use the source name alone.
        name = key.rsplit(".", 1)[-1]
        if not isinstance(entry, Mapping):
            plan.left_out[name] = "its registry entry is not a mapping"
            continue
        if entry.get("enabled") is False:
            continue  # not enabled on the instance: neither in nor left out
        if entry.get("module") == GIT_FILES_MODULE:
            params = _fill(dict(entry.get("params") or {}), values)
            root = params.get("repo_root")
            if not isinstance(root, str) or not root or _PLACEHOLDER.search(root):
                plan.left_out[name] = f"its repo_root {root!r} names a value the deployment does not record"
                continue
            directory = _locate(Path(root), data_root, deployment_dir)
            repo = str(params.get("repo") or directory.name)
            if _PLACEHOLDER.search(repo) or "/" in repo or repo in (".", ".."):
                repo = directory.name
            globs = tuple(str(g) for g in params.get("exclude_globs") or ())
            plan.repos.append((name, repo, directory, globs))
            continue
        spec = SOURCE_GROUPS.get(name)
        if spec is None:
            plan.left_out[name] = (_MONIT if name.startswith("monit_")
                                   else LEFT_OUT_REASONS.get(name, "no archi.snapshot group reads this source's cache"))
            continue
        if entry.get("module") != spec.module:
            plan.left_out[name] = (f"its reader is {entry.get('module')}, not the cache reader {spec.module} "
                                   f"that the snapshot group {spec.group} checks")
            continue
        params = _fill(dict(entry.get("params") or {}), values)
        raw = (entry.get("params") or {}).get(spec.param)
        if not raw:
            plan.left_out[name] = f"its registry entry sets no {spec.param}"
            continue
        text = _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), str(raw))
        if _PLACEHOLDER.search(text):
            plan.left_out[name] = f"{spec.param} {raw} names a value the deployment does not record"
            continue
        path = _locate(Path(text), data_root, deployment_dir)
        directory = path if spec.param.endswith("_root") else path.parent
        if spec.group in groups_seen:
            raise CorpusError(f"sources {groups_seen[spec.group]} and {name} both map to snapshot group {spec.group}")
        groups_seen[spec.group] = name
        if spec.group == "twiki-eos":
            unresolved = [k for k in TWIKI_SCOPE_PARAMS if isinstance(params.get(k), str) and _PLACEHOLDER.search(params[k])]
            if unresolved:
                plan.left_out[name] = (f"{', '.join(unresolved)} name a value the deployment does not record, so the "
                                       "pages the reader keeps are unknown")
                continue
        plan.included.append(PlannedSource(name, spec.group, directory, params))
    plan.left_out.setdefault(OPS_RECORD, OPS_RECORD_REASON)
    return plan


def _fill(value: Any, values: Mapping[str, str]) -> Any:
    """`${name}` placeholders filled from the deployment's values; "true" and "false" read as booleans
    (a registry rendered from an answer holds the string)."""
    if isinstance(value, dict):
        return {k: _fill(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, values) for v in value]
    if isinstance(value, str):
        text = _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), value)
        return {"true": True, "false": False}.get(text.strip().lower(), text)
    return value


def _locate(path: Path, data_root: Path, deployment_dir: Optional[Path]) -> Path:
    if path.is_absolute():
        return path
    for base in (deployment_dir, data_root):
        if base is not None and (Path(base) / path).exists():
            return Path(base) / path
    return Path(deployment_dir or data_root) / path


def read_registry(deployment_dir: Path) -> dict[str, Any]:
    path = Path(deployment_dir) / "source_registry.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CorpusError(f"cannot read the instance's source registry {path}: {exc}") from exc
    sources = data.get("sources") if isinstance(data, Mapping) else None
    if not isinstance(sources, Mapping) or not sources:
        raise CorpusError(f"{path} lists no sources")
    return dict(sources)


# --- one file per record -----------------------------------------------------------------


@dataclass(frozen=True)
class RecordFile:
    key: str
    path: str  # relative to records/
    data: bytes
    #: other names a node id's tail may give this record (twiki's web:topic, a node id)
    aliases: tuple[str, ...] = ()


def safe_name(key: str) -> str:
    name = quote(key, safe=_SAFE)
    if name.startswith("."):
        name = "%2E" + name[1:]
    if len(name) > _MAX_NAME:
        name = name[:100] + "~" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return name or "%00empty"


def _dump(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _key(record: Any, fields: tuple[str, ...]) -> str:
    if isinstance(record, Mapping):
        for name in fields:
            value = record.get(name)
            if value not in (None, "") and str(value).strip():
                return str(value).strip()
    return ""


def _by_key(group: str, prefix: str, items: list[tuple[str, Any]], *, ext: str = ".json",
            aliases=lambda key: ()) -> Iterator[RecordFile]:
    """One file per key; records sharing a key go into one file as a list."""
    merged: dict[str, list[Any]] = {}
    for key, record in items:
        if not key:
            raise CorpusError(f"group {group}: a record has no key, so it cannot become a file")
        merged.setdefault(key, []).append(record)
    for key, records in merged.items():
        body = records[0] if len(records) == 1 else records
        yield RecordFile(key, f"{prefix}/{safe_name(key)}{ext}", _dump(body), tuple(aliases(key)))


def explode(group: str, files: Mapping[str, bytes], archive_dir: str) -> list[RecordFile]:
    """The record files of one unpacked snapshot group (`files` by archive path)."""
    local = {PurePosixPath(p).relative_to(archive_dir).as_posix(): data for p, data in files.items()}
    spec = next((g for g in SOURCE_GROUPS.values() if g.group == group), None)
    if spec is None:
        raise CorpusError(f"no corpus layout for snapshot group {group}")
    out: list[RecordFile] = []
    if group == "twiki-eos":
        for rel, data in sorted(local.items()):
            stem = rel[:-4] if rel.endswith(".txt") else rel
            names = {stem.replace("/", ":"), PurePosixPath(stem).name}
            out.append(RecordFile(stem, f"{group}/{rel}", data, tuple(sorted(names - {stem}))))
        return out
    if group == "cmssw-releases" and "releases.map" in local:
        items = []
        for line in local["releases.map"].decode("utf-8").splitlines():
            fields = dict(part.split("=", 1) for part in line.strip().strip(";").split(";") if "=" in part)
            if fields.get("label"):
                items.append((fields["label"], line))
        merged: dict[str, list[str]] = {}
        for key, line in items:
            merged.setdefault(key, []).append(line)
        return [RecordFile(key, f"{group}/{safe_name(key)}.txt", ("\n".join(lines) + "\n").encode("utf-8"))
                for key, lines in merged.items()]
    for name, data in sorted(local.items()):
        if name in ("meta.json", "authority.json"):
            continue  # counts the collector wrote, not records
        payload = json.loads(data.decode("utf-8"))
        stem = PurePosixPath(name).stem
        if group in ("cric", "cric-core") and isinstance(payload, Mapping) and name != "responsibilities.json":
            prefix = _CRIC_PREFIX.get(stem, f"{stem}:")
            out += _by_key(group, f"{group}/{stem}", [(str(k), v) for k, v in payload.items()],
                           aliases=lambda key, prefix=prefix: (prefix + key,))
        elif name == "responsibilities.json":
            rows = [(str(row[1] or "no-site"), row) for row in payload.get("result") or [] if isinstance(row, list)]
            out += _by_key(group, f"{group}/{stem}", rows)
        elif group in ("slurm", "htcondor"):
            out += _snapshot_object(group, payload)
        elif group == "wmstats" and isinstance(payload, Mapping):
            items = []
            for body in (payload.get("status_payloads") or {}).values():
                for entry in body.get("result") or []:
                    for wf_name, record in (entry.items() if isinstance(entry, Mapping) else ()):
                        items.append((_key(record, spec.key_fields) or str(wf_name), record))
            out += _by_key(group, group, items)
        elif isinstance(payload, list):
            out += _by_key(group, group, [(_key(r, spec.key_fields), r) for r in payload])
        else:
            raise CorpusError(f"group {group}: {name} is not a list of records")
    return out


#: CRIC file -> the node-id prefix its reader mints (archi.sources.cric), so a node id finds its
#: record even when a site and a storage unit share a name.
_CRIC_PREFIX = {"sites": "site:", "storage_units": "se:", "compute_units": "ce:", "facilities": "facility:",
                "services": "svc:", "federations": "fed:"}

#: slurm and htcondor caches are one snapshot object; each listed thing is a record whose
#: node id is `<subtype>:<cluster or pool>:<name>`.
_SNAPSHOT_LISTS = {
    "slurm": (("partitions", "name", "slurm_partition"), ("nodes", "name", "slurm_node"),
              ("reservations", "name", "slurm_reservation"), ("jobs", "job_id", "slurm_job")),
    "htcondor": (("machines", "name", "condor_machine"), ("schedds", "name", "condor_schedd")),
}


def _snapshot_object(group: str, payload: Any) -> list[RecordFile]:
    if not isinstance(payload, Mapping):
        raise CorpusError(f"group {group}: records.json is not an object")
    scope = str(payload.get("cluster") or payload.get("pool") or "").strip()
    out = []
    summary = {k: v for k, v in payload.items() if not isinstance(v, list)}
    key = f"{'cluster' if group == 'slurm' else 'pool'}:{scope}"
    alias = f"condor_pool:{scope}" if group == "htcondor" else ""
    out.append(RecordFile(key, f"{group}/{safe_name(scope or group)}.json", _dump(summary),
                          (alias,) if alias else ()))
    for list_name, field_name, subtype in _SNAPSHOT_LISTS[group]:
        items = []
        for item in payload.get(list_name) or []:
            name = str(item.get(field_name) or "").strip() if isinstance(item, Mapping) else ""
            items.append((name, item))
        out += _by_key(group, f"{group}/{list_name}", items,
                       aliases=lambda name, subtype=subtype: (f"{subtype}:{scope}:{name}",))
    return out


# --- building ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """A corpus on disk, read from its corpus.json."""

    path: Path
    digest: str
    lock_sha256: str
    bundle: str
    sources: Mapping[str, Any]
    left_out: Mapping[str, str]
    built_at: str
    #: the instance's published generation when the corpus was built, or None if unknown
    generation: Optional[str] = None
    #: one line: what the corpus covers and what it leaves out
    coverage: str = ""

    @property
    def records(self) -> Path:
        return self.path / RECORDS_DIR

    def keys(self) -> dict[str, Any]:
        return json.loads((self.path / KEYS).read_text(encoding="utf-8"))

    def manifest_paths(self) -> frozenset[str]:
        lines = (self.path / MANIFEST).read_text(encoding="utf-8").splitlines()
        return frozenset(line.split("\t", 1)[0] for line in lines if line)


def corpus_root(data_root: Path, bundle: str) -> Path:
    from archi.ops.context_gate import bundle_path_name

    return Path(data_root).joinpath(*CORPUS_SUBDIR, bundle_path_name(bundle))


def _collected(directory: Path) -> str:
    try:
        stamps = [p.stat().st_mtime for p in directory.iterdir() if p.is_file()] or [directory.stat().st_mtime]
    except OSError:
        stamps = [datetime.now(timezone.utc).timestamp()]
    return datetime.fromtimestamp(max(stamps), timezone.utc).strftime("%Y-%m-%d")


def repo_records(repo: str, root: Path, exclude_globs: Sequence[str] = ()) -> tuple[list[RecordFile], str, dict]:
    """A git-code checkout's tracked files at HEAD, one record each, under `git-repo/<repo>/`.

    Not packed through archi.snapshot (no group checks a source tree): email addresses are
    removed here, as the snapshot does, and binary files (a NUL byte) and files over
    MAX_REPO_FILE_BYTES are left out and counted. Returns (records, HEAD sha, left-out counts)."""
    import fnmatch
    import subprocess

    from archi.enrichment.anonymizer import redact_email_addresses

    def git(*args: str) -> bytes:
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CorpusError(f"git-code: cannot read the checkout {root} ({type(exc).__name__})") from exc

    head = git("rev-parse", "HEAD").decode().strip()
    out: list[RecordFile] = []
    left = {"binary": 0, "too_large": 0, "excluded": 0, "not_a_file": 0}
    for raw in git("ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "replace")
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts:
            left["not_a_file"] += 1
            continue
        if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch("/" + rel, g) for g in exclude_globs):
            left["excluded"] += 1
            continue
        path = root / rel
        if path.is_symlink() or not path.is_file():
            left["not_a_file"] += 1
            continue
        if path.stat().st_size > MAX_REPO_FILE_BYTES:
            left["too_large"] += 1
            continue
        data = path.read_bytes()
        if b"\0" in data:
            left["binary"] += 1
            continue
        text = redact_email_addresses(data.decode("utf-8", "replace"))
        key = f"{repo}/{rel}"
        out.append(RecordFile(key, f"git-repo/{repo}/{rel}", text.encode("utf-8"),
                              (rel, f"repos/{repo}/{rel}")))
    return out, head, left


def twiki_scope(files: Mapping[str, bytes], params: Mapping[str, Any]) -> tuple[set[str], str]:
    """The topic files (relative to the TWiki root) the instance's TWiki reader keeps, with its own
    params (physics filter, seed topics, skip patterns), and a note on how it was scoped."""
    from archi.sources.twiki import TwikiEOSSource

    kwargs = {k: params[k] for k in TWIKI_SCOPE_PARAMS if k in params and params[k] is not None}
    with tempfile.TemporaryDirectory(prefix="raw-corpus-twiki-") as tmp:
        root = Path(tmp)
        for rel, data in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        try:
            reader = TwikiEOSSource(eos_root=str(root), **kwargs)
            read = reader._records_from_root(root)
            kept = reader._physics_scoped(list(read.records))
        except (TypeError, ValueError) as exc:
            raise CorpusError(f"the TWiki reader refused the instance's params: {exc}") from exc
    notes = []
    if kwargs.get("physics_filter"):
        notes.append("physics filter")
    if kwargs.get("seed_topics"):
        notes.append("seed topics")
    if kwargs.get("max_files"):
        notes.append(f"at most {kwargs['max_files']} pages")
    return {r.source_path for r in kept}, ", ".join(notes)


def coverage_line(bundle: str, sources: Mapping[str, Any], left_out: Mapping[str, str]) -> str:
    """`<bundle>: twiki-eos (physics filter), cmssw_releases; cds not included: <why>; ...`"""
    parts = [name + (f" ({row['scope']})" if row.get("scope") else "") for name, row in sources.items()]
    line = f"{bundle}: {', '.join(parts) or 'nothing'}"
    if left_out:
        line += "; " + "; ".join(f"{name} not included: {why}" for name, why in left_out.items())
    return line


def build_corpus(plan: Plan, *, bundle: str, out_root: Path, built_by: Optional[str] = None,
                 tmp_dir: Optional[Path] = None, generation: Optional[str] = None) -> Corpus:
    """Pack `plan.included` with archi.snapshot and unpack it to `<out_root>/<lock sha>/`.

    `generation` is the instance's published generation as the corpus is built; it is recorded
    beside the digest so a run can tell when the graph has moved on. Nothing is written when a
    group is refused; the error names each refusal."""
    from archi.snapshot.builder import (
        LOCK_NAME,
        GroupConfig,
        SnapshotConfig,
        SnapshotError,
        build,
        read_archive,
    )
    from archi.snapshot.groups import GROUPS

    if not plan.included and not plan.repos:
        raise CorpusError("no enabled source of this instance has a cache archi.snapshot can check; "
                          f"left out: {', '.join(sorted(plan.left_out))}")
    config = SnapshotConfig(
        snapshot=f"raw-corpus-{bundle}",
        groups=tuple(GroupConfig(name=p.group, path=p.directory, collected=_collected(p.directory))
                     for p in sorted(plan.included, key=lambda p: p.group)),
    )
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".raw-corpus-", dir=out_root) as scratch:
        snap = Path(scratch) / "snapshot"
        lock: dict[str, Any] = {"groups": {}}
        lock_bytes = b""
        if plan.included:
            try:
                lock = build(config, snap, built_by=built_by, tmp_dir=tmp_dir)
            except SnapshotError as exc:
                raise CorpusError(f"archi.snapshot refused the caches: {exc}") from exc
            lock_bytes = (snap / LOCK_NAME).read_bytes()
        repos = [(name, repo, *repo_records(repo, directory, globs)) for name, repo, directory, globs in plan.repos]
        heads = {repo: head for _, repo, _, head, _ in repos}
        # the snapshot lock's sha; with git-code checkouts, of the lock and their HEADs
        lock_sha = hashlib.sha256(lock_bytes + (json.dumps(heads, sort_keys=True).encode() if heads else b"")
                                  ).hexdigest()
        target = out_root / lock_sha[:16]
        if target.exists():
            raise CorpusError(f"{target} exists already")
        staging = Path(scratch) / "corpus"
        records = staging / RECORDS_DIR
        records.mkdir(parents=True)
        manifest: list[tuple[str, str]] = []
        keys: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        for planned in sorted(plan.included, key=lambda p: p.source):
            group_lock = lock["groups"][planned.group]
            files = read_archive(snap / group_lock["archive"])
            exploded = explode(planned.group, files, GROUPS[planned.group].archive_dir)
            scope = ""
            if planned.group == "twiki-eos":
                prefix = GROUPS[planned.group].archive_dir + "/"
                kept, scope = twiki_scope({k[len(prefix):]: v for k, v in files.items() if k.startswith(prefix)},
                                          planned.params)
                exploded = [r for r in exploded if r.path[len("twiki-eos/"):] in kept]
            table: dict[str, str] = {}
            ambiguous: set[str] = set()
            natural: dict[str, str] = {}
            for record in exploded:
                target_file = records / record.path
                if target_file.exists():
                    raise CorpusError(f"group {planned.group}: two records map to {record.path}")
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_bytes(record.data)
                manifest.append((record.path, hashlib.sha256(record.data).hexdigest()))
                natural[record.path] = record.key
                for name in (record.key, *record.aliases):
                    if name in table and table[name] != record.path:
                        ambiguous.add(name)
                    table.setdefault(name, record.path)
            keys[planned.source] = {
                "group": planned.group,
                # every name a node id may give (natural key, node id, twiki web:topic) -> file
                "lookup": {k: v for k, v in sorted(table.items()) if k not in ambiguous},
                # names two records share: never used to resolve a gold id
                "ambiguous": sorted(ambiguous),
                # file -> the record's natural key, the one the scorer looks for in tool output
                "natural_key": dict(sorted(natural.items())),
            }
            sources[planned.source] = {
                "group": planned.group,
                "cache": str(planned.directory),
                "collected": group_lock["collected"],
                "record_count": group_lock["record_count"],
                "files": len(exploded),
                "addresses_removed": group_lock["addresses_removed"],
                **({"scope": scope} if scope else {}),
            }
        if repos:
            table, ambiguous, natural, skipped = {}, set(), {}, {}
            for name, repo, exploded, head, left in repos:
                for record in exploded:
                    target_file = records / record.path
                    if target_file.exists():
                        raise CorpusError(f"git-code: two files map to {record.path}")
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_bytes(record.data)
                    manifest.append((record.path, hashlib.sha256(record.data).hexdigest()))
                    natural[record.path] = record.key
                    for alias in (record.key, *record.aliases):
                        if alias in table and table[alias] != record.path:
                            ambiguous.add(alias)
                        table.setdefault(alias, record.path)
                skipped[repo] = left
            keys[GIT_SOURCE] = {"group": "git-code",
                                "lookup": {k: v for k, v in sorted(table.items()) if k not in ambiguous},
                                "ambiguous": sorted(ambiguous), "natural_key": dict(sorted(natural.items()))}
            sources[GIT_SOURCE] = {
                "group": "git-code",
                "registry_sources": sorted({name for name, *_ in repos}),
                "repos": {repo: {"head": head, "files": len(exploded), "left_out_files": left}
                          for _, repo, exploded, head, left in repos},
                "files": sum(len(exploded) for _, _, exploded, _, _ in repos),
                "record_count": sum(len(exploded) for _, _, exploded, _, _ in repos),
                "collected": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "scope": "tracked files at " + ", ".join(f"{repo} @{head[:8]}" for repo, head in heads.items()),
            }
        manifest.sort()
        manifest_text = "".join(f"{path}\t{sha}\n" for path, sha in manifest)
        (staging / MANIFEST).write_text(manifest_text, encoding="utf-8")
        digest = "sha256:" + hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        info = {
            "schema": CORPUS_SCHEMA,
            "bundle": bundle,
            "digest": digest,
            "lock_sha256": lock_sha,
            "built_at": built_at,
            "generation": generation,
            "coverage": coverage_line(bundle, sources, dict(sorted(plan.left_out.items()))),
            "files": len(manifest),
            "sources": sources,
            "left_out": dict(sorted(plan.left_out.items())),
        }
        (staging / KEYS).write_text(json.dumps({"schema": CORPUS_SCHEMA, "sources": keys,
                                                "left_out": info["left_out"]}, indent=1) + "\n", encoding="utf-8")
        (staging / INFO).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        (staging / README).write_text(_readme(info), encoding="utf-8")
        if lock_bytes:
            shutil.copyfile(snap / LOCK_NAME, staging / LOCK_NAME)
        os.rename(staging, target)
    return load_corpus(target)


def _readme(info: Mapping[str, Any]) -> str:
    lines = [
        f"# Raw-document corpus: {info['bundle']}",
        "",
        "The files this instance's graph was built from, one file per record, for the agent-bash arm of "
        "`archi eval --context` (PACT r3-context-quality). Built by `archi eval --raw-corpus` from the "
        "instance's connector caches: each cache was packed with `archi.snapshot` (checked with its "
        "source's own reader, email addresses removed, fields the reader does not read dropped) and "
        "unpacked here. Derived files and okg state are not included.",
        "",
        f"- digest: `{info['digest']}` (sha256 of MANIFEST.txt)",
        f"- snapshot lock sha256: `{info['lock_sha256']}`",
        f"- built {info['built_at']} at published generation {info['generation'] or 'unknown'}; "
        f"{info['files']} record files",
        f"- coverage: {info['coverage']}",
        "",
        "Only `records/` is mounted into the agent's sandbox, at /corpus.",
        "",
        "| source | group | records (reader count) | files | collected |",
        "|---|---|---|---|---|",
    ]
    for name, row in info["sources"].items():
        lines.append(f"| {name} | {row['group']} | {row['record_count']} | {row['files']} | {row['collected']} |")
    lines += ["", "Left out:", ""]
    lines += [f"- **{name}**: {reason}" for name, reason in info["left_out"].items()]
    return "\n".join(lines) + "\n"


def load_corpus(path: Path) -> Corpus:
    path = Path(path)
    try:
        info = json.loads((path / INFO).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorpusError(f"{path} is not a raw corpus ({INFO}: {exc})") from exc
    if not isinstance(info, Mapping) or info.get("schema") != CORPUS_SCHEMA:
        raise CorpusError(f"{path / INFO} is not a {CORPUS_SCHEMA} corpus")
    generation = info.get("generation")
    return Corpus(path, str(info.get("digest")), str(info.get("lock_sha256")), str(info.get("bundle")),
                  dict(info.get("sources") or {}), dict(info.get("left_out") or {}), str(info.get("built_at")),
                  None if generation is None else str(generation), str(info.get("coverage") or ""))


def verify_corpus(path: Path, *, bundle: Optional[str] = None) -> Corpus:
    """The corpus at `path`, after checking every record file against MANIFEST.txt and the
    manifest against the digest corpus.json records. Any difference refuses it."""
    corpus = load_corpus(path)
    if bundle is not None and corpus.bundle != bundle:
        raise CorpusError(f"{path} is the corpus of bundle {corpus.bundle!r}, not {bundle!r}")
    try:
        text = (corpus.path / MANIFEST).read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"{path}: cannot read {MANIFEST}: {exc}") from exc
    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != corpus.digest:
        raise CorpusError(f"{path}: {MANIFEST} has digest {digest}, but {INFO} records {corpus.digest}")
    listed = {}
    for line in text.splitlines():
        rel, _, sha = line.partition("\t")
        listed[rel] = sha
    on_disk = set()
    if corpus.records.is_symlink():
        raise CorpusError(f"{path}: records/ is a symbolic link; the sandbox would mount what it points at")
    for root, dirs, names in os.walk(corpus.records):
        for name in dirs + names:
            if Path(root, name).is_symlink():
                rel = Path(root, name).relative_to(corpus.records).as_posix()
                raise CorpusError(f"{path}: records/{rel} is a symbolic link; a corpus holds plain files only")
        for name in names:
            on_disk.add(Path(root, name).relative_to(corpus.records).as_posix())
    extra, missing = sorted(on_disk - set(listed)), sorted(set(listed) - on_disk)
    if extra or missing:
        raise CorpusError(f"{path}: records differ from {MANIFEST} (not listed: {extra[:3]}; missing: {missing[:3]})")
    for rel, sha in listed.items():
        if hashlib.sha256((corpus.records / rel).read_bytes()).hexdigest() != sha:
            raise CorpusError(f"{path}: {rel} changed since the corpus was built")
    return corpus


def newest_corpus(data_root: Path, bundle: str) -> Path:
    root = corpus_root(data_root, bundle)
    found = []
    for child in root.iterdir() if root.is_dir() else []:
        if child.name.startswith(".") or not (child / INFO).is_file():
            continue
        try:
            found.append((load_corpus(child).built_at, child))
        except CorpusError:
            continue
    if not found:
        raise CorpusError(f"no raw corpus of bundle {bundle} under {root}; build one with `archi eval --raw-corpus`")
    return max(found)[1]
