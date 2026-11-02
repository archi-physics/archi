"""Cache-backed generic ticket source (Redmine, any JIRA, and any tracker that exports the same shape).

The CERN JIRA pack has its own reader (``archi.sources.jira``). This one is
for the trackers any cluster operator may run. It reads the cache that
``python -m archi.downloaders.tickets`` (Redmine, JIRA or both) or
``python -m archi.downloaders.redmine`` writes under
``${archi_data_root}/tickets/``; any other tracker can fill the same cache
with the flat shape below and use the same pack.

Cache shape: a JSON list. Each item is either the flat shape

    {"system": "redmine", "key": "1234", "subject": "...", "description": "...",
     "project": "ops", "tracker": "Support", "status": "New", "priority": "High",
     "author": "Grace Hopper", "assignee": "Ada Lovelace",
     "created": "2026-01-05T10:00:00Z", "updated": "...", "closed": "",
     "notes": [{"author": "...", "created": "...", "body": "..."}],
     "url": "https://..."}

or a Redmine REST issue (``id``, ``subject``, ``project: {name}``,
``author: {name}``, ``assigned_to: {name}``, ``created_on``, ``journals``).
Both read the same. ``system`` defaults to the ``system`` parameter; the node
id is ``tracker_ticket:<system>:<key>``, so a JIRA ``OPS-7`` and a Redmine
``7`` never meet. A JIRA issue moved to another project gets a new key, so
its node id changes: the next complete download retracts the old node and
publishes a new one. ``url``, when present, is the ticket's link; otherwise the
link is built from ``base_url`` (``/issues/<id>`` for Redmine,
``/browse/<key>`` for JIRA).

The credentials, when a tracker needs one, are named by environment variable
(``REDMINE_API_KEY``, ``JIRA_API_TOKEN``) and used only by the downloader. A
failed fetch names the one for the ``system`` parameter (``JIRA_API_TOKEN``
for ``jira``, else ``REDMINE_API_KEY``).

Email addresses in any text field are removed here, as each record is read
(``clean``), for Redmine and JIRA alike.

Registry entry: ``bundles/packs/tickets/sources/tickets.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from okg.deployment import EdgeFact, NodeFact

from archi.sources._cluster_cache import CacheSource, clean, person_fact, person_id
from archi.sources._sdk_adapter import ReaderAdapter


@dataclass(frozen=True)
class TicketRecord:
    system: str
    key: str
    subject: str
    description: str = ""
    project: str = ""
    tracker: str = ""
    status: str = ""
    priority: str = ""
    author: str = ""
    assignee: str = ""
    created: str = ""
    updated: str = ""
    closed: str = ""
    notes: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    url: str = ""

    @property
    def node_id(self) -> str:
        return f"tracker_ticket:{self.system}:{self.key}"


CREDENTIAL_BY_SYSTEM = {"redmine": "REDMINE_API_KEY", "jira": "JIRA_API_TOKEN"}


class TicketSource(CacheSource):
    """Load tracker tickets from the downloader's cache."""

    name = "tickets"
    profile = "mutable_api"
    change_probe_kind = "mutable_api"
    description = "tracker tickets"
    mutable = True

    def __init__(
        self,
        *,
        records_path: str = "tickets/records.json",
        meta_path: str = "tickets/meta.json",
        system: str = "redmine",
        base_url: str = "",
        credential_ref: str | None = None,
        base: str | None = None,
    ) -> None:
        if not system or not system.replace("-", "").replace("_", "").isalnum() or system != system.lower():
            raise ValueError(f"system must be a lowercase word, got {system!r}")
        if credential_ref is None:
            # The variable the downloader for this system reads, so a failed fetch names the right one.
            credential_ref = CREDENTIAL_BY_SYSTEM.get(system, "REDMINE_API_KEY")
        self.system = system
        self.base_url = base_url.rstrip("/")
        super().__init__(
            records_path=records_path,
            meta_path=meta_path,
            credential_refs=(credential_ref,) if credential_ref else (),
            base=base,
            config={"system": system, "base_url": self.base_url},
        )

    def ticket_url(self, record: TicketRecord) -> str:
        if record.url:
            return record.url
        if not self.base_url:
            return ""
        if record.system == "redmine":
            return f"{self.base_url}/issues/{record.key}"
        if record.system == "jira":
            return f"{self.base_url}/browse/{record.key}"
        return f"{self.base_url}/{record.key}"

    def parse(self, payload: Any) -> tuple[list[TicketRecord], int]:
        records: list[TicketRecord] = []
        seen: set[str] = set()
        skipped = 0
        for item in payload:
            record = _record(item, self.system) if isinstance(item, dict) else None
            if record is None or record.node_id in seen:
                skipped += 1
                continue
            seen.add(record.node_id)
            records.append(record)
        return records, skipped

    def facts(self, records: list[TicketRecord], revision: dict[str, Any]) -> Iterator[Any]:
        people: set[str] = set()
        for record in records:
            yield self._node(record, revision)
            for role, name, edge in (
                ("author", record.author, "reported_by"),
                ("assignee", record.assignee, "assigned_to"),
            ):
                if not name:
                    continue
                pid = person_id(record.system, name)
                if pid not in people:
                    people.add(pid)
                    yield person_fact(record.system, name, revision)
                yield EdgeFact(
                    src=record.node_id,
                    dst=pid,
                    edge_type=edge,
                    source_record_id={"system": record.system, "key": record.key, "role": role},
                    source_revision=revision,
                )

    def _node(self, record: TicketRecord, revision: dict[str, Any]) -> NodeFact:
        notes = " ".join(f"{author}: {body}" for author, _, body in record.notes if body)
        text = " ".join(filter(None, [
            record.key, record.subject, record.description, record.status, record.priority,
            record.tracker, record.project, notes,
        ]))
        return NodeFact(
            node_id=record.node_id,
            subtype="tracker_ticket",
            attrs={
                "label": f"{record.system}#{record.key}",
                "ticket_id": record.key,
                "system": record.system,
                "subject": record.subject,
                "description": record.description,
                "project": record.project,
                "tracker": record.tracker,
                "status": record.status,
                "priority": record.priority,
                "author": record.author,
                "assignee": record.assignee,
                "created": record.created,
                "updated": record.updated,
                "closed": record.closed,
                "note_count": len(record.notes),
                "url": self.ticket_url(record),
                "text": text,
            },
            source_record_id={"system": record.system, "ticket_id": record.key},
            source_revision=revision,
        )


def _notes(item: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    raw = item.get("notes")
    if raw is None:
        # Redmine REST: journals carry the notes; a journal with no note is a field change.
        raw = [
            {"author": j.get("user"), "created": j.get("created_on"), "body": j.get("notes")}
            for j in item.get("journals") or [] if isinstance(j, dict)
        ]
    out = []
    for note in raw or []:
        if isinstance(note, dict) and clean(note.get("body")):
            out.append((clean(note.get("author")), clean(note.get("created")), clean(note.get("body"))))
    return tuple(out)


def _url(value: Any) -> str:
    # A link the downloader wrote (it knows each tracker's root); http(s) only.
    url = clean(value)
    return url if url.lower().startswith(("https://", "http://")) else ""


def _record(item: dict[str, Any], default_system: str) -> TicketRecord | None:
    key = clean(item.get("key") if item.get("key") not in (None, "") else item.get("id"))
    if not key:
        return None
    return TicketRecord(
        system=clean(item.get("system")) or default_system,
        key=key,
        subject=clean(item.get("subject") or item.get("summary")),
        description=clean(item.get("description")),
        project=clean(item.get("project")),
        tracker=clean(item.get("tracker")),
        status=clean(item.get("status")),
        priority=clean(item.get("priority")),
        author=clean(item.get("author") or item.get("reporter")),
        assignee=clean(item.get("assignee") or item.get("assigned_to")),
        created=clean(item.get("created") or item.get("created_on")),
        updated=clean(item.get("updated") or item.get("updated_on")),
        closed=clean(item.get("closed") or item.get("closed_on")),
        notes=_notes(item),
        url=_url(item.get("url")),
    )


class TicketAdapter(ReaderAdapter):
    """Registry adapter for :class:`TicketSource` (see ``ReaderAdapter``)."""

    reader_class = TicketSource
    profile = "mutable_api"
    change_probe_kind = "mutable_api"
