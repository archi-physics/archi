"""Cache-backed ServiceNow source: incidents, requests and knowledge-base articles.

It reads the records cache that ``python -m archi.downloaders.servicenow``
writes under ``${archi_data_root}/servicenow/``. The reader never talks to
ServiceNow. The downloader authenticates with OAuth or basic auth, each named
by environment variable (``SERVICENOW_CLIENT_ID`` and ``SERVICENOW_CLIENT_SECRET``,
or ``SERVICENOW_USERNAME`` and ``SERVICENOW_PASSWORD``); this module only names
them in the registry entry (``credential_refs``) so ``archi doctor`` can say
which are missing.

Cache shape: a JSON list of records, each with a ``table``:

- ``incident``: ``number`` (INC...), ``short_description``, ``description``,
  ``state``, ``priority``, ``urgency``, ``impact``, ``category``,
  ``assignment_group``, ``assigned_to``, ``opened_by`` (or ``caller_id``),
  ``cmdb_ci``, ``opened_at``, ``sys_updated_on``, ``resolved_at``,
  ``close_notes``
- ``sc_request``: ``number`` (REQ...), the same descriptive fields and
  ``requested_for``
- ``kb_knowledge``: ``number`` (KB...), ``short_description`` (the title),
  ``text`` (the body), ``author``, ``kb_knowledge_base``, ``workflow_state``,
  ``sys_updated_on``

A reference field may be a plain string (ServiceNow's display value) or a
``{"display_value": ...}`` object; both read the same.

Registry entry: ``bundles/packs/servicenow/sources/servicenow.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from okg.deployment import EdgeFact, NodeFact

from archi.sources._cluster_cache import CacheSource, clean, person_fact, person_id
from archi.sources._sdk_adapter import ReaderAdapter

SYSTEM = "servicenow"

#: table -> (subtype, id prefix, number prefix)
TABLES = {
    "incident": "servicenow_incident",
    "sc_request": "servicenow_request",
    "kb_knowledge": "servicenow_kb_article",
}

#: Credential environment variables by auth method; the downloader reads them.
CREDENTIALS = {
    "oauth": ("SERVICENOW_CLIENT_ID", "SERVICENOW_CLIENT_SECRET"),
    "basic": ("SERVICENOW_USERNAME", "SERVICENOW_PASSWORD"),
}


@dataclass(frozen=True)
class ServiceNowRecord:
    table: str
    number: str
    title: str
    body: str
    state: str = ""
    priority: str = ""
    category: str = ""
    group: str = ""
    config_item: str = ""
    created: str = ""
    updated: str = ""
    resolved: str = ""
    resolution: str = ""
    knowledge_base: str = ""
    sys_id: str = ""
    #: role -> display name; roles are reporter, assignee (tickets) or author (articles)
    people: dict[str, str] = field(default_factory=dict)

    @property
    def subtype(self) -> str:
        return TABLES[self.table]

    @property
    def node_id(self) -> str:
        return f"{self.subtype}:{self.number}"


class ServiceNowSource(CacheSource):
    """Load ServiceNow records from the downloader's cache."""

    name = "servicenow"
    profile = "mutable_api"
    change_probe_kind = "mutable_api"
    description = "ServiceNow records"
    mutable = True

    def __init__(
        self,
        *,
        records_path: str = "servicenow/records.json",
        meta_path: str = "servicenow/meta.json",
        instance_url: str = "",
        auth_method: str = "oauth",
        base: str | None = None,
    ) -> None:
        if auth_method not in CREDENTIALS:
            raise ValueError(f"auth_method must be one of {sorted(CREDENTIALS)}, got {auth_method!r}")
        self.instance_url = instance_url.rstrip("/")
        self.auth_method = auth_method
        super().__init__(
            records_path=records_path,
            meta_path=meta_path,
            credential_refs=CREDENTIALS[auth_method],
            base=base,
            config={"instance_url": self.instance_url, "auth_method": auth_method},
        )

    def record_url(self, record: ServiceNowRecord) -> str:
        if not self.instance_url:
            return ""
        if record.table == "kb_knowledge":
            return f"{self.instance_url}/kb_view.do?sysparm_article={record.number}"
        if record.sys_id:
            return f"{self.instance_url}/{record.table}.do?sys_id={record.sys_id}"
        return f"{self.instance_url}/{record.table}_list.do?sysparm_query=number%3D{record.number}"

    def parse(self, payload: Any) -> tuple[list[ServiceNowRecord], int]:
        records: list[ServiceNowRecord] = []
        seen: set[str] = set()
        skipped = 0
        for item in payload:
            record = _record(item) if isinstance(item, dict) else None
            if record is None or record.node_id in seen:
                skipped += 1
                continue
            seen.add(record.node_id)
            records.append(record)
        return records, skipped

    def facts(self, records: list[ServiceNowRecord], revision: dict[str, Any]) -> Iterator[Any]:
        people: set[str] = set()
        for record in records:
            yield self._node(record, revision)
            for role, name in sorted(record.people.items()):
                pid = person_id(SYSTEM, name)
                if pid not in people:
                    people.add(pid)
                    yield person_fact(SYSTEM, name, revision)
                yield EdgeFact(
                    src=record.node_id,
                    dst=pid,
                    edge_type={"reporter": "reported_by", "assignee": "assigned_to", "author": "authored_by"}[role],
                    source_record_id={"number": record.number, "role": role},
                    source_revision=revision,
                )

    def _node(self, record: ServiceNowRecord, revision: dict[str, Any]) -> NodeFact:
        text = " ".join(filter(None, [
            record.number, record.title, record.body, record.state, record.priority,
            record.category, record.group, record.config_item, record.resolution,
        ]))
        attrs: dict[str, Any] = {
            "label": record.number,
            "number": record.number,
            "title": record.title,
            "description": record.body,
            "state": record.state,
            "created": record.created,
            "updated": record.updated,
            "url": self.record_url(record),
            "text": text,
        }
        if record.table == "kb_knowledge":
            attrs["knowledge_base"] = record.knowledge_base
            attrs["author"] = record.people.get("author", "")
        else:
            attrs.update({
                "priority": record.priority,
                "category": record.category,
                "assignment_group": record.group,
                "assignee": record.people.get("assignee", ""),
                "reporter": record.people.get("reporter", ""),
                "config_item": record.config_item,
                "resolved": record.resolved,
                "resolution": record.resolution,
            })
        return NodeFact(
            node_id=record.node_id,
            subtype=record.subtype,
            attrs=attrs,
            source_record_id={"number": record.number},
            source_revision=revision,
        )


def _record(item: dict[str, Any]) -> ServiceNowRecord | None:
    table = clean(item.get("table") or item.get("sys_class_name"))
    number = clean(item.get("number"))
    if table not in TABLES or not number:
        return None
    people: dict[str, str] = {}
    if table == "kb_knowledge":
        author = clean(item.get("author"))
        if author:
            people["author"] = author
        title = clean(item.get("short_description"))
        body = clean(item.get("text") or item.get("description"))
    else:
        reporter = clean(item.get("opened_by") or item.get("caller_id") or item.get("requested_for"))
        assignee = clean(item.get("assigned_to"))
        if reporter:
            people["reporter"] = reporter
        if assignee:
            people["assignee"] = assignee
        title = clean(item.get("short_description"))
        body = clean(item.get("description"))
    return ServiceNowRecord(
        table=table,
        number=number,
        title=title,
        body=body,
        state=clean(item.get("state") or item.get("request_state") or item.get("workflow_state")),
        priority=clean(item.get("priority")),
        category=clean(item.get("category")),
        group=clean(item.get("assignment_group")),
        config_item=clean(item.get("cmdb_ci")),
        created=clean(item.get("opened_at") or item.get("sys_created_on")),
        updated=clean(item.get("sys_updated_on") or item.get("updated_at")),
        resolved=clean(item.get("resolved_at")),
        resolution=clean(item.get("close_notes")),
        knowledge_base=clean(item.get("kb_knowledge_base")),
        sys_id=clean(item.get("sys_id")),
        people=people,
    )


class ServiceNowAdapter(ReaderAdapter):
    """Registry adapter for :class:`ServiceNowSource` (see ``ReaderAdapter``)."""

    reader_class = ServiceNowSource
    profile = "mutable_api"
    change_probe_kind = "mutable_api"
