"""What each snapshot group holds, and how its archi reader reads it.

One ``GroupSpec`` per cache-backed source group. It names:

- the files the reader reads, where they sit in ``ARCHI_DATA_ROOT``
  (``archive_dir``), and their JSON shape;
- ``identity``: the per-record check that mirrors the reader's own skip
  rule. A record that fails it is one the reader would skip or silently
  drop, so the builder refuses the whole group instead;
- ``schema``: the fields the reader reads. Every other field is dropped from
  the snapshot. A schema maps a key to ``None`` (keep the value whole) or to a
  nested schema (applied to a dict, or to each dict in a list). The builder
  proves the schema complete on every build: the reader's facts from the
  pruned cache must equal its facts from the unpruned one, or the group is
  refused;
- ``make_reader``: builds the reader over a staged data root.

The schemas were read off the reader code in ``archi/sources/`` at archi-okg
``b4dac901f5``. When a reader starts reading a new field, the build refuses
the group (the facts differ) until the schema here names that field.

Paths follow the reader defaults and ``docs/docs/reference/connector-caches.md``
(``data/<group>/...``); the TWiki EOS tree goes to ``data/twiki-eos``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

Schema = Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class JsonFile:
    """One JSON cache file of a group."""

    name: str
    #: ``list`` (a list of records), ``mapping`` (an object whose values are
    #: records, keyed by name), ``object`` (one object, schema applied to it)
    #: or ``status_payloads`` (the WMStats collection object
    #: ``{"status_payloads": {status: {"result": [{name: record}]}}}``, or a
    #: plain list of records; schema applied to each record).
    shape: str
    schema: Schema = None
    #: Returns a reason when a record (for ``object`` files, the whole object)
    #: is malformed, else ``None``.
    identity: Optional[Callable[[Any], Optional[str]]] = None
    required: bool = True
    #: What marks this export among files of the same container type. At
    #: least ``SIGNATURE_SHARE`` of the file's records must carry it, and the
    #: file must hold at least one record.
    signature: Optional["Signature"] = None


@dataclass(frozen=True)
class Signature:
    description: str
    matches: Callable[[Any], bool]


#: The readers treat signature keys as optional, so one legitimate record
#: without the key must not refuse its group; a swapped export carries the key
#: in (almost) none of its records.
SIGNATURE_SHARE = 0.95


@dataclass(frozen=True)
class TextFiles:
    """Plain-text cache files: one exact name, or every match under the dir.

    A file with a NUL byte refuses the group (UTF-16 would hide an address
    from the redactor), and so does one whose decoded text is mostly control
    characters (binary). Text that is not UTF-8 refuses it too, unless
    ``fallback_invalid_bytes``: then the file is decoded as UTF-8 and only
    each invalid byte run is decoded as cp1252 (Latin-1 for the bytes cp1252
    leaves undefined). Valid UTF-8 around it stays intact, so one stray
    Windows quote cannot garble the addresses on the rest of the page.
    """

    pattern: str
    recursive: bool = False
    fallback_invalid_bytes: bool = False
    #: Leave out files that are zero bytes or only whitespace/NUL bytes
    #: (TWiki topics empty upstream; the reader skips them too) and list
    #: them in the lock. More than the reader's ``empty_topic_limit`` of
    #: them refuses the group, as the reader refuses its scope.
    skip_empty: bool = False


@dataclass(frozen=True)
class GroupSpec:
    name: str
    archive_dir: str
    make_reader: Callable[[Path], Any]
    json_files: tuple[JsonFile, ...] = ()
    text: Optional[TextFiles] = None
    #: Top-level record fields always dropped / always kept for this group,
    #: on top of the schema and of the build config.
    default_drop_fields: frozenset[str] = field(default_factory=frozenset)
    default_keep_fields: frozenset[str] = field(default_factory=frozenset)
    #: Keys removed at every depth of every JSON file, whatever the schema
    #: and the config's keep_fields say.
    deep_drop_keys: frozenset[str] = field(default_factory=frozenset)
    #: Alternative specs for the same group, tried in order: the first whose
    #: ``primary_input`` exists in the source directory is used.
    variants: tuple["GroupSpec", ...] = ()
    #: The file whose presence selects this spec among a group's variants.
    primary_input: Optional[str] = None
    #: Files another group owns that this reader refuses to run without.
    #: Staged (empty) for the validation run only, never archived: the other
    #: group's own archive carries the real ones.
    validation_stubs: Mapping[str, bytes] = field(default_factory=dict)

    @property
    def record_file(self) -> Optional[JsonFile]:
        """The list or mapping file that per-record config drops apply to."""
        for spec in self.json_files:
            if spec.shape in {"list", "mapping", "status_payloads"}:
                return spec
        return None


def _keys(*names: str) -> dict[str, None]:
    return {name: None for name in names}


def _require_dict(item: Any) -> Optional[str]:
    return None if isinstance(item, dict) else "record is not a JSON object"


def _require_text(*keys: str) -> Callable[[Any], Optional[str]]:
    """Malformed unless the record is an object with one of ``keys`` non-blank."""

    def check(item: Any) -> Optional[str]:
        if not isinstance(item, dict):
            return "record is not a JSON object"
        if any(str(item.get(key) or "").strip() for key in keys):
            return None
        return f"record has no {' / '.join(keys)} value"

    return check


# Signatures: a key the right export carries and its siblings of the same
# container type do not, so a file swapped for another export (same JSON
# shape, wrong content) is refused instead of misread. Counted on the real
# Aug 31 and June 12-16 caches (2026-09-28, counts only): every CRIC and
# CRIC-core key is in 100% of its own export's records and 0% of each of the
# six sibling exports', and the JIRA, Indico, CondDB and CMSSW signatures hold
# for every record. The DBS one (a /primary/processed/tier path) was not
# checked against real data.


def _has_key(*keys: str) -> Signature:
    return Signature(
        f"a {' / '.join(keys)} key",
        lambda item: isinstance(item, dict) and any(key in item for key in keys),
    )


def _text_matches(pattern: str, *keys: str) -> Signature:
    compiled = re.compile(pattern)

    def matches(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        value = next((item.get(k) for k in keys if item.get(k)), "")
        return compiled.fullmatch(str(value)) is not None

    return Signature(f"a {' / '.join(keys)} matching {pattern}", matches)


RESPONSIBILITY_COLUMNS = ["username", "site_name", "role"]


def _responsibilities_check(payload: Any) -> Optional[str]:
    """CRIC's responsibilities export, not another ``{desc, result}`` export.

    The reader takes fields 1-3 of each row as username, site title and role.
    The ``sites-compat`` export has the same container but five other columns
    per row; read as responsibilities it would mint wrong operator nodes.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
        return "expected a 'result' list"
    desc = payload.get("desc")
    columns = desc.get("columns") if isinstance(desc, dict) else None
    if columns is not None and columns != RESPONSIBILITY_COLUMNS:
        return (
            f"columns are {columns}, expected {RESPONSIBILITY_COLUMNS} "
            "(is it another CRIC export, such as sites-compat?)"
        )
    for index, row in enumerate(payload["result"]):
        if not isinstance(row, list) or len(row) != len(RESPONSIBILITY_COLUMNS):
            size = len(row) if isinstance(row, list) else type(row).__name__
            return (
                f"row {index} has {size} fields, expected 3 "
                "(username, site title, role)"
            )
        username, site, role = row
        if not isinstance(username, str) or not isinstance(role, str):
            return f"row {index}: username and role must be strings"
        if site is not None and not isinstance(site, str):
            return f"row {index}: site title must be a string or null"
    return None


def _require_record_count(item: Any) -> Optional[str]:
    count = item.get("record_count") if isinstance(item, dict) else None
    if isinstance(count, int) and not isinstance(count, bool):
        return None
    return "record_count is missing or not an integer"


_CMS_SITE = re.compile(r"T[0-9]_[A-Z]{2}_\w+")


def _cmssst_identity(item: Any) -> Optional[str]:
    """The CMS SST reader skips a record without a CMS site name."""
    if not isinstance(item, dict):
        return "record is not a JSON object"
    if not _CMS_SITE.fullmatch(str(item.get("site") or "")):
        return "site is missing or not a CMS site name"
    return None


def _gocdb_identity(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return "record is not a JSON object"
    try:
        downtime_id = int(item.get("downtime_id") or 0)
    except (TypeError, ValueError):
        return "downtime_id is not numeric"
    if downtime_id <= 0:
        return "downtime_id is missing or not positive"
    return None


# --- readers -----------------------------------------------------------------
# Imported lazily: the readers import okg.deployment, which the downloaders and
# the lock-file verifier do not need.


def _jira(root: Path) -> Any:
    from archi.sources.jira import JiraIssueSource

    return JiraIssueSource(
        records_path="data/jira/records.json",
        meta_path="data/jira/meta.json",
        base=str(root),
    )


def _cric(root: Path) -> Any:
    from archi.sources.cric import CRICSource

    return CRICSource(base=str(root))


def _cric_core(root: Path) -> Any:
    from archi.sources.cric import CRICCoreSource

    return CRICCoreSource(base=str(root))


def _cmssw_map(root: Path) -> Any:
    from archi.sources.cmssw import CMSSWReleaseSource

    return CMSSWReleaseSource(
        map_cache_path="data/cmssw-releases/releases.map",
        fetch=False,
        base=str(root),
    )


def _cmssw_records(root: Path) -> Any:
    from archi.sources.cmssw import CMSSWReleaseSource

    return CMSSWReleaseSource(
        records_path="data/cmssw-releases/records.json", base=str(root)
    )


def _indico(root: Path) -> Any:
    from archi.sources.indico import IndicoSource

    return IndicoSource(records_path="data/indico/records.json", base=str(root))


def _dqm(root: Path) -> Any:
    from archi.sources.dqm import DQMSource

    return DQMSource(records_path="data/dqm/records.json", base=str(root))


def _gocdb(root: Path) -> Any:
    from archi.sources.gocdb import GoCDBDowntimeSource

    return GoCDBDowntimeSource(
        records_path="data/gocdb-downtimes/records.json", base=str(root)
    )


def _docs(source_name: str, directory: str) -> Callable[[Path], Any]:
    def make(root: Path) -> Any:
        from archi.sources.docs import DocumentationSource

        return DocumentationSource(
            source_name=source_name,
            records_path=f"data/{directory}/records.json",
            base=str(root),
        )

    return make


def _twiki(root: Path) -> Any:
    from archi.sources.twiki import TwikiEOSSource

    return TwikiEOSSource(eos_root=str(root / "data" / "twiki-eos"))


def _conddb(root: Path) -> Any:
    from archi.sources.conddb import CondDBGlobalTagSource

    return CondDBGlobalTagSource(
        records_path="data/conddb-global-tags/records.json", base=str(root)
    )


def _wmstats(root: Path) -> Any:
    from archi.sources.wmstats import WMStatsWorkflowSource

    return WMStatsWorkflowSource(
        records_path="data/wmstats-workflows/records.json", base=str(root)
    )


def _cmssst(root: Path) -> Any:
    from archi.sources.cmssst import CMSSSTSiteStatusSource

    return CMSSSTSiteStatusSource(
        records_path="data/cmssst-site-status/records.json", base=str(root)
    )


def _dbs(root: Path) -> Any:
    from archi.sources.dbs import DBSDatasetSource

    return DBSDatasetSource(
        records_path="data/dbs-datasets/records.json", base=str(root)
    )


# The cluster-manager caches (``${archi_data_root}/<name>/records.json``, see
# connector-caches.md). Their readers skip a malformed item and then withhold
# the completed scope, which the reader check refuses, so the identity rule is
# only "an object". No schema: the records are kept whole (their downloaders
# write only what the readers read, and the readers remove email addresses).


def _cluster(module: str, cls: str, name: str, *, meta: bool = True) -> Callable[[Path], Any]:
    def make(root: Path) -> Any:
        import importlib

        source = getattr(importlib.import_module(f"archi.sources.{module}"), cls)
        kwargs: dict[str, Any] = {"records_path": f"data/{name}/records.json", "base": str(root)}
        if meta:
            kwargs["meta_path"] = f"data/{name}/meta.json"
        return source(**kwargs)

    return make


# --- schemas -----------------------------------------------------------------

_JIRA_PERSON = _keys("displayName", "name", "key")
_JIRA_COMMENT = {
    "author": _JIRA_PERSON,
    **_keys("created", "body", "text", "renderedBody"),
}
_JIRA_FIELDS = {
    **_keys(
        "summary",
        "description",
        "project",
        "status",
        "priority",
        "issuetype",
        "created",
        "updated",
        "environment",
        "resolution",
        "parent",
        "components",
        "labels",
        "fixVersions",
        "fix_versions",
        "issuelinks",
        "subtasks",
    ),
    "assignee": _JIRA_PERSON,
    "reporter": _JIRA_PERSON,
    "comment": {"comments": _JIRA_COMMENT},
}
JIRA_SCHEMA = {
    **_keys(
        "key",
        "issue_key",
        "summary",
        "description",
        "project",
        "status",
        "priority",
        "issue_type",
        "created",
        "updated",
        "environment",
        "resolution",
        "parent_key",
        "parent",
        "comment_count",
        "components",
        "labels",
        "fix_versions",
        "issue_links",
        "subtasks",
    ),
    "fields": _JIRA_FIELDS,
    "assignee": _JIRA_PERSON,
    "reporter": _JIRA_PERSON,
    "recent_comments": _JIRA_COMMENT,
    "comments": _JIRA_COMMENT,
}

_INDICO_PERSON = _keys("fullName", "first_name", "last_name")
INDICO_SCHEMA = {
    **_keys(
        "id",
        "event_id",
        "title",
        "url",
        "description",
        "type",
        "event_type",
        "category",
        "categoryId",
        "_contributions_text",
    ),
    "startDate": _keys("date"),
    "endDate": _keys("date"),
    "speakers": _INDICO_PERSON,
    "creators": _INDICO_PERSON,
    "chairs": _INDICO_PERSON,
    "folders": {"attachments": _keys("download_url", "url")},
    "_pdf_texts": _keys("text", "title", "filename", "url", "download_url"),
}

# CRIC files are objects keyed by name; the schema applies to each value.
# Nested values (pledges, vos, rcsites) are kept whole.
CRIC_SCHEMA = _keys(
    "facility",
    "tier_level",
    "country",
    "country_code",
    "timezone",
    "state",
    "status",
    "is_monitored",
    "sitedb_title",
    "endpoint",
    "flavour",
    "type",
    "site",
    "corepower",
    "pledged_cms",
    "pledged-CMS",
    "potential_max",
    "promised",
    "name",
    "fullname",
    "rcsite",
    "rcsites",
    "cmssites",
    "computeunits",
    "sites",
    "infrastructure",
    "accounting_name",
    "pledges",
    "vos",
    "vo_name",
)

DOCS_SCHEMA = _keys(
    "title", "url", "body", "site_name", "source_repo", "repo", "path", "file_path"
)

WMSTATS_SCHEMA = _keys(
    "RequestName",
    "request_name",
    "workflow_name",
    "RequestType",
    "request_type",
    "RequestStatus",
    "status",
    "RequestPriority",
    "priority",
    "Campaign",
    "campaign",
    "CMSSWVersion",
    "cmssw_version",
    "PrepID",
    "prep_id",
    "InputDataset",
    "input_dataset",
    "OutputDatasets",
    "output_datasets",
    "RequestDate",
    "created_at",
    "updated_at",
)

DBS_SCHEMA = _keys(
    "dataset",
    "dataset_name",
    "primary_ds_name",
    "primary_dataset",
    "processed_ds_name",
    "processed_dataset",
    "data_tier_name",
    "data_tier",
    "dataset_access_type",
    "physics_group_name",
    "physics_group",
    "creation_date",
    "nevents",
    "total_events",
    "nfiles",
    "total_files",
    "dataset_size",
    "total_size_bytes",
)


def _cric_mapping(name: str, signature: str) -> JsonFile:
    return JsonFile(
        name, "mapping", CRIC_SCHEMA, _require_dict, signature=_has_key(signature)
    )


GROUPS: dict[str, GroupSpec] = {
    spec.name: spec
    for spec in (
        GroupSpec(
            "cric",
            "data/cric",
            _cric,
            json_files=(
                _cric_mapping("sites.json", "sitedb_title"),
                _cric_mapping("storage_units.json", "pledged-CMS"),
                _cric_mapping("compute_units.json", "potential_max"),
                _cric_mapping("facilities.json", "cmssites"),
                # The reader refuses a payload without a ``result`` list.
                JsonFile(
                    "responsibilities.json",
                    "object",
                    _keys("result"),
                    _responsibilities_check,
                ),
            ),
        ),
        GroupSpec(
            "cric-core",
            "data/cric-core",
            _cric_core,
            json_files=(
                _cric_mapping("services.json", "rcsite"),
                _cric_mapping("rcsites.json", "sites"),
                _cric_mapping("federations.json", "accounting_name"),
            ),
        ),
        GroupSpec(
            "cmssw-releases",
            "data/cmssw-releases",
            _cmssw_map,
            # The cms-bot map when the cache has one (it fixes the release
            # set's vintage); else the JSON release list the reader also reads.
            variants=(
                GroupSpec(
                    "cmssw-releases",
                    "data/cmssw-releases",
                    _cmssw_map,
                    text=TextFiles("releases.map"),
                    primary_input="releases.map",
                ),
                GroupSpec(
                    "cmssw-releases",
                    "data/cmssw-releases",
                    _cmssw_records,
                    json_files=(
                        JsonFile(
                            "records.json",
                            "list",
                            _keys(
                                "label",
                                "type",
                                "state",
                                "architecture",
                                "release_notes",
                                "release_date",
                                "production_architecture",
                            ),
                            _require_text("label"),
                            signature=_text_matches(r"CMSSW_\S+", "label"),
                        ),
                    ),
                    primary_input="records.json",
                ),
            ),
        ),
        GroupSpec(
            "jira",
            "data/jira",
            _jira,
            json_files=(
                JsonFile(
                    "records.json",
                    "list",
                    JIRA_SCHEMA,
                    _require_text("key", "issue_key"),
                    signature=_text_matches(
                        r"[A-Z][A-Z0-9_]*-[0-9]+", "key", "issue_key"
                    ),
                ),
                # Optional for the reader; when present it must match.
                JsonFile("meta.json", "object", _keys("record_count"), required=False),
            ),
        ),
        GroupSpec(
            "indico",
            "data/indico",
            _indico,
            json_files=(
                JsonFile(
                    "records.json",
                    "list",
                    INDICO_SCHEMA,
                    _require_text("id", "event_id"),
                    signature=_has_key("_contributions_text", "_pdf_texts"),
                ),
            ),
        ),
        GroupSpec(
            "dqm",
            "data/dqm",
            _dqm,
            json_files=(
                JsonFile(
                    "records.json",
                    "list",
                    _keys(
                        "cert_name",
                        "filename",
                        "run_range",
                        "datasets",
                        "num_lumi_sections",
                    ),
                    _require_text("cert_name"),
                ),
            ),
        ),
        GroupSpec(
            "gocdb-downtimes",
            "data/gocdb-downtimes",
            _gocdb,
            json_files=(
                JsonFile(
                    "records.json",
                    "list",
                    _keys(
                        "downtime_id",
                        "primary_key",
                        "hostname",
                        "hosted_by",
                        "service_type",
                        "severity",
                        "classification",
                        "description",
                        "start_date",
                        "end_date",
                        "endpoint",
                    ),
                    _gocdb_identity,
                ),
            ),
            # The reader matches hostnames against CRIC; it requires both,
            # and refuses an empty one (it would drop every affects edge),
            # so each stub holds one entry that matches no real host.
            validation_stubs={
                "data/cric/sites.json": b'{"T0_ZZ_SnapshotStub": {}}\n',
                "data/cric-core/services.json": (
                    b'{"snapshot-stub.invalid": '
                    b'{"endpoint": "https://snapshot-stub.invalid"}}\n'
                ),
            },
        ),
        GroupSpec(
            "cmssst-site-status",
            "data/cmssst-site-status",
            _cmssst,
            json_files=(
                JsonFile(
                    "records.json",
                    "list",
                    # metrics kept whole: the reader reads each metric's
                    # "today" range, and the page's other ranges are small.
                    _keys("site", "time", "alert", "msg", "ggus", "metrics"),
                    _cmssst_identity,
                    signature=_has_key("metrics"),
                ),
            ),
        ),
        GroupSpec(
            "gitlab-docs",
            "data/gitlab-docs",
            _docs("gitlab_docs", "gitlab-docs"),
            json_files=(
                JsonFile("records.json", "list", DOCS_SCHEMA, _require_text("url")),
            ),
        ),
        GroupSpec(
            "docsite",
            "data/docsite",
            _docs("docsite", "docsite"),
            json_files=(
                JsonFile("records.json", "list", DOCS_SCHEMA, _require_text("url")),
            ),
        ),
        GroupSpec(
            "twiki-eos",
            "data/twiki-eos",
            _twiki,
            # TWiki pages predate UTF-8 (lead decision, 2026-09-28): invalid
            # byte runs decode as cp1252, the Windows superset of Latin-1.
            text=TextFiles(
                "*.txt",
                recursive=True,
                fallback_invalid_bytes=True,
                skip_empty=True,
            ),
        ),
        GroupSpec(
            "conddb-global-tags",
            "data/conddb-global-tags",
            _conddb,
            json_files=(
                JsonFile(
                    "records.json",
                    "list",
                    _keys(
                        "name",
                        "tag_name",
                        "description",
                        "release",
                        "scenario",
                        "snapshot_time",
                        "created_at",
                    ),
                    _require_text("name", "tag_name"),
                    signature=_has_key("release", "scenario"),
                ),
            ),
        ),
        GroupSpec(
            "wmstats",
            "data/wmstats-workflows",
            _wmstats,
            json_files=(
                JsonFile(
                    "records.json",
                    "status_payloads",
                    WMSTATS_SCHEMA,
                    _require_text("workflow_name", "request_name", "RequestName"),
                ),
                # The collector's own record count; the reader checks the
                # records file against it, so the build does too.
                JsonFile(
                    "authority.json",
                    "object",
                    _keys("record_count"),
                    _require_record_count,
                    required=False,
                ),
            ),
            # Jason, 2026-09-28: drop the requestor's certificate DN, keep the
            # requestor's user name. The reader reads neither field. A DN
            # also sits in each RequestTransition entry, so every DN key goes,
            # at any depth, even if a config keeps the field that holds it.
            deep_drop_keys=frozenset({"RequestorDN", "DN"}),
            default_keep_fields=frozenset({"Requestor"}),
        ),
        GroupSpec(
            "dbs",
            "data/dbs-datasets",
            _dbs,
            json_files=(
                JsonFile(
                    "records.json",
                    "list",
                    DBS_SCHEMA,
                    _require_text("dataset_name", "dataset"),
                    signature=_text_matches(
                        r"/[^/]+/[^/]+/[^/]+", "dataset_name", "dataset"
                    ),
                ),
            ),
        ),
        GroupSpec(
            "tickets",
            "data/tickets",
            _cluster("tickets", "TicketSource", "tickets"),
            json_files=(
                JsonFile("records.json", "list", None, _require_text("key", "id")),
                JsonFile("meta.json", "object", _keys("record_count"), required=False),
            ),
        ),
        GroupSpec(
            "servicenow",
            "data/servicenow",
            _cluster("servicenow", "ServiceNowSource", "servicenow"),
            json_files=(
                JsonFile("records.json", "list", None, _require_text("number")),
                JsonFile("meta.json", "object", _keys("record_count"), required=False),
            ),
        ),
        GroupSpec(
            "node-health",
            "data/node-health",
            _cluster("node_health", "NodeHealthSource", "node-health", meta=False),
            json_files=(JsonFile("records.json", "list", None, _require_text("node")),),
        ),
        GroupSpec(
            "slurm",
            "data/slurm",
            _cluster("slurm", "SlurmSource", "slurm"),
            json_files=(
                JsonFile("records.json", "object", None, _require_dict),
                JsonFile("meta.json", "object", _keys("record_count"), required=False),
            ),
        ),
        GroupSpec(
            "htcondor",
            "data/htcondor",
            _cluster("htcondor", "HTCondorSource", "htcondor"),
            json_files=(
                JsonFile("records.json", "object", None, _require_dict),
                JsonFile("meta.json", "object", _keys("record_count"), required=False),
            ),
        ),
    )
}
