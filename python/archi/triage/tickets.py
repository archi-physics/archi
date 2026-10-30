"""Tickets as the ticket-triage automation reads them.

Each ticket is read through the node subtype a ticket pack already publishes; this module adds no reader.
`Ticket.from_row` turns an `okg filter --jsonl` row of any of them into one shape:

    subtype                                   pack        system                 key        category    component
    tracker_ticket                            tickets     its `system` attribute ticket_id  tracker     project
                                                          (redmine or jira)
    jira_issue                                jira        jira                   issue_key  issue_type  components
    servicenow_incident, servicenow_request   servicenow  servicenow             number     category    config_item

So a JIRA ticket comes from the `jira` pack (`jira:<KEY>`) or from the generic `tickets` pack
(`tracker_ticket:jira:<KEY>`); both are system `jira`, read and written the same way. A `tracker_ticket`
with no `system` is a Redmine ticket; one of another tracker is not triaged (no tool can write to it).
The queue is the project (ServiceNow: the assignment group).

A ticket's entity, the `entity` of its signal and of the incident a run records, is `<system>:<key>`
(`jira:OPS-12`, `redmine:4812`, `servicenow:INC0010001`). The key alone does not say which server holds the
ticket: two JIRA servers can both have a project OPS. Each ticket therefore also carries its link (`url`, as its
reader published it), and `site` is that link's scheme and host. An L2 change goes only to the server the link
names (archi.triage.proposer, archi.tools.tickets), and a key found on two servers is never changed.

Text read from a ticket has email addresses removed here too (the readers already remove them; this is the
second guard, so no address reaches a proposal, a run's stored step output or a ticket note).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

from archi.enrichment.anonymizer import redact_email_addresses


@dataclass(frozen=True)
class Reading:
    """How one node subtype holds tickets."""

    subtype: str
    pack: str
    key: str
    subject: str
    status: str
    queue: str
    category: str
    component: str
    #: the system every row is; None: the row's own `system` attribute says (empty means redmine)
    system: Optional[str] = None
    #: attributes that, when set, mean the ticket is closed
    closed_attrs: tuple[str, ...] = ()
    #: the attribute holding a closed ticket's resolution text
    resolution: Optional[str] = None


READINGS: dict[str, Reading] = {r.subtype: r for r in (
    Reading("tracker_ticket", "tickets", "ticket_id", "subject", "status", "project", "tracker", "project",
            closed_attrs=("closed",)),
    Reading("jira_issue", "jira", "issue_key", "summary", "status", "project", "issue_type", "components",
            system="jira", closed_attrs=("resolution",), resolution="resolution"),
    Reading("servicenow_incident", "servicenow", "number", "title", "state", "assignment_group", "category",
            "config_item", system="servicenow", closed_attrs=("resolved",), resolution="resolution"),
    Reading("servicenow_request", "servicenow", "number", "title", "state", "assignment_group", "category",
            "config_item", system="servicenow", closed_attrs=("resolved",), resolution="resolution"),
)}

#: system -> the subtypes that hold its tickets
SYSTEMS: dict[str, tuple[str, ...]] = {
    "redmine": ("tracker_ticket",),
    "jira": ("jira_issue", "tracker_ticket"),
    "servicenow": ("servicenow_incident", "servicenow_request"),
}

#: the field an assignment proposal sets: ServiceNow routes to a group, the others to a person
ASSIGN_FIELDS = {"redmine": "assignee", "jira": "assignee", "servicenow": "assignment_group"}

#: ServiceNow subtype -> its table, which the live tools write to.
SERVICENOW_TABLES = {"servicenow_incident": "incident", "servicenow_request": "sc_request"}

#: Status names that mean a ticket is finished, whatever the system (compared without case). An instance
#: whose workflow uses others lists them in the proposer's `closed_statuses`.
CLOSED_STATUSES = frozenset({
    "closed", "resolved", "done", "rejected", "cancelled", "canceled", "won't fix", "wont fix", "duplicate",
    "closed complete", "closed incomplete", "closed skipped", "fixed",
})


def closed_statuses(config: Optional[Mapping[str, Any]] = None) -> frozenset[str]:
    """CLOSED_STATUSES plus the proposer config's own `closed_statuses` (lower case)."""
    extra = (config or {}).get("closed_statuses") or ()
    return CLOSED_STATUSES | frozenset(str(s).strip().lower() for s in extra)


def subtypes_for(system: str, packs: Optional[Iterable[str]] = None) -> tuple[str, ...]:
    """The subtypes holding `system`'s tickets, limited to those of `packs` when given (a subtype of a pack the
    instance did not select is not in its catalog, and okg refuses to read it)."""
    wanted = None if packs is None else set(packs)
    return tuple(s for s in SYSTEMS[system] if wanted is None or READINGS[s].pack in wanted)


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_OFFSET = re.compile(r"([+-]\d{2})(\d{2})$")
_STAMP = "%Y%m%dT%H%M%SZ"


class TicketError(ValueError):
    """A ticket or entity that cannot be read."""


def parse_time(value: Any) -> Optional[datetime]:
    """A ticket time as an aware UTC datetime, or None. Reads ISO 8601 with `Z`, `+00:00` or `+0000`, and
    ServiceNow's `2026-10-12 09:40:11` (no zone: UTC)."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = _OFFSET.sub(r"\1:\2", str(value).strip().replace("Z", "+00:00"))
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def site_of(url: str) -> str:
    """`scheme://host[:port]` of a URL, lower case; empty when it has no scheme or host."""
    parts = urlsplit(str(url or ""))
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}" if parts.scheme and parts.netloc else ""


def system_of(row: Mapping[str, Any]) -> str:
    """The ticket system of a row (see the module docstring). Raises TicketError for a row no tool can write to."""
    reading = READINGS.get(str(row.get("subtype") or ""))
    if reading is None:
        raise TicketError(f"{row.get('subtype')!r} is not a ticket subtype ({sorted(READINGS)})")
    system = reading.system or (str((row.get("attrs") or {}).get("system") or "").strip().lower() or "redmine")
    if system not in SYSTEMS:
        raise TicketError(f"{row.get('node_id')!r} is a {system!r} ticket; ticket-triage reads {sorted(SYSTEMS)}")
    return system


def entity_of(system: str, key: str) -> str:
    return f"{system}:{key}"


def parse_entity(entity: str) -> tuple[str, str]:
    system, sep, key = str(entity).partition(":")
    if not sep or system not in SYSTEMS or not key:
        raise TicketError(f"{entity!r} is not a ticket entity (<system>:<key>, system one of {sorted(SYSTEMS)})")
    return system, key


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value if v not in (None, ""))
    return redact_email_addresses(str(value)).strip()


def _list(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return (str(value).strip(),)


@dataclass(frozen=True)
class Ticket:
    system: str
    subtype: str
    node_id: str
    key: str
    subject: str
    description: str = ""
    status: str = ""
    assignee: str = ""
    group: str = ""
    queue: str = ""
    category: str = ""
    components: tuple[str, ...] = field(default_factory=tuple)
    updated: str = ""
    closed: bool = False
    resolution: str = ""
    url: str = ""

    @property
    def entity(self) -> str:
        return entity_of(self.system, self.key)

    @property
    def updated_at(self) -> Optional[datetime]:
        return parse_time(self.updated)

    @property
    def site(self) -> str:
        """`scheme://host[:port]` of the ticket's link, lower case; empty when it has none."""
        return site_of(self.url)

    @property
    def source(self) -> str:
        """Where the ticket was read from: its subtype and site (`jira_issue@https://jira.example.org`). Each
        source keeps its own cursor (archi.triage.signals), since two sources publish on their own schedules."""
        return f"{self.subtype}@{self.site or '-'}"

    @property
    def table(self) -> Optional[str]:
        """The ServiceNow table of the ticket; None for the other systems."""
        return SERVICENOW_TABLES.get(self.subtype)

    @property
    def assigned_to(self) -> str:
        """What the system's assignment field holds now (person, or ServiceNow group)."""
        return self.group if ASSIGN_FIELDS[self.system] == "assignment_group" else self.assignee

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, closed_statuses: frozenset[str] = CLOSED_STATUSES) -> "Ticket":
        system = system_of(row)
        spec = READINGS[str(row["subtype"])]
        a = row.get("attrs") or {}
        key = _text(a.get(spec.key))
        if not key:
            raise TicketError(f"{row.get('node_id')!r}: no {spec.key}")
        status = _text(a.get(spec.status))
        closed = any(_text(a.get(attr)) for attr in spec.closed_attrs) or status.lower() in closed_statuses
        return cls(
            system=system, subtype=spec.subtype, node_id=str(row.get("node_id") or ""), key=key,
            subject=_text(a.get(spec.subject)), description=_text(a.get("description")), status=status,
            assignee=_text(a.get("assignee")), group=_text(a.get("assignment_group")),
            queue=_text(a.get(spec.queue)), category=_text(a.get(spec.category)),
            components=_list(a.get(spec.component)), updated=_text(a.get("updated")), closed=closed,
            resolution=_text(a.get(spec.resolution)) if spec.resolution else "", url=_text(a.get("url")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "components": list(self.components), "entity": self.entity, "table": self.table,
                "site": self.site}

    def as_context(self) -> dict[str, Any]:
        """as_dict without the description: what a run's read step returns, and DBOS stores that output, so the
        ticket's full text stays in the graph and out of the run history."""
        d = self.as_dict()
        d.pop("description")
        return d


def tickets_of(system: str, rows: Iterable[Mapping[str, Any]], *, closed_statuses: frozenset[str] = CLOSED_STATUSES,
               skipped: Optional[list] = None) -> list[Ticket]:
    """The rows that are `system`'s tickets (a tracker_ticket row may be another system's), as Tickets."""
    out = []
    for row in rows:
        try:
            if system_of(row) == system:
                out.append(Ticket.from_row(row, closed_statuses=closed_statuses))
        except TicketError as exc:
            if skipped is not None:
                skipped.append({"what": str(row.get("node_id")), "reason": str(exc)})
    return out


def signal_id(ticket: Ticket) -> str:
    """`<system>-<key>-<updated, UTC>`: one signal per version of a ticket, and a legal run-id part (no `~`)."""
    moment = ticket.updated_at
    stamp = moment.astimezone(timezone.utc).strftime(_STAMP) if moment else "undated"
    return f"{ticket.system}-{_UNSAFE.sub('_', ticket.key)}-{stamp}"


def signal_time(sid: str) -> Optional[datetime]:
    """The update time a signal id holds (signal_id), or None (undated, or not a ticket signal)."""
    try:
        return datetime.strptime(sid.rsplit("-", 1)[-1], _STAMP).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
