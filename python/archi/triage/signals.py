"""New and updated tickets become signals (the scanner's half of a `ticket_queue` trigger).

`archi.ops.workflow.scan_once` calls `add_ticket_signals` on every scan, before it starts runs. For each
automation whose trigger is `{type: ticket_queue, system: [...], queue: ...}` it reads each system's
tickets from the latest published generation and adds one signal per new or updated open ticket:

    signal id   <system>-<key>-<updated, UTC>   (archi.triage.tickets.signal_id)
    entity      <system>:<key>
    source      ticket

A system's tickets are read from every subtype that holds them and whose pack the instance selected (the
automation's gate sources name those packs): JIRA from the `jira` pack's `jira_issue`, from the `tickets`
pack's `tracker_ticket` rows with system `jira`, or both. The same JIRA issue in both makes one signal.

What "new or updated" means. Each automation keeps a cursor per source (`archi_ops.ticket_cursors`): a
source is a subtype at one site (`jira_issue@https://jira.example.org`, archi.triage.tickets.Ticket.source),
and its cursor is the newest `updated` time seen from it (cursors move only after the read's signals are
added, so a failure part way loses nothing). Sources publish on their own schedules, so one
cursor per system would skip, for good, the tickets of a source that publishes late with older times. A
system is read again only when a newer generation is published (a `<system>/*` row records the generation
read). A ticket becomes a signal when its `updated` is at or after its source's cursor and was
not already read at that time (the cursor keeps the tickets seen at its time, so a ticket updated in the same
second as the cursor is not lost and none is read twice), it is not closed, and its
queue is among the instance's `queues` (ops.yaml; unset means every queue). The first read of a source only
sets its cursor: the tickets already in the queue when the automation is installed, or when a new source
appears, are not triaged, so installing it does not start a backlog of runs.

One ticket, one pending signal. When two sources publish the same ticket (the JIRA case above) with
different update times, only the newest becomes a signal, and a version older than one already signalled is
dropped. A new signal for a ticket supersedes that ticket's signals still waiting to run (status `new` ->
`superseded`), so a ticket commented on five times while a run waits makes one more run, not five.

Known limit: a ticket update made by this automation's own L2 action (a status or assignment change) is a
new update, so the ticket is triaged once more; that run finds nothing left to propose and drafts.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from archi.bundles.automation import trigger_systems
from archi.triage.tickets import (CLOSED_STATUSES, SYSTEMS, Ticket, closed_statuses, signal_id, signal_time,
                                  subtypes_for, tickets_of)

log = logging.getLogger(__name__)


def new_or_updated(tickets: Iterable[Ticket], since: Optional[datetime], queues: Iterable[str] = (),
                   seen: Iterable[str] = ()) -> tuple[list[Ticket], Optional[datetime], list[str]]:
    """(the tickets of ONE source to triage, oldest update first; the source's new cursor; the tickets seen at
    that cursor time). A ticket updated after the cursor is new; one updated at the cursor time itself is new
    unless it is in `seen`, the tickets already read at that time (so two tickets updated in the same second
    are both triaged, across two reads). With no cursor (`since` None) nothing is triaged and the cursor
    becomes the newest update seen."""
    tickets = [t for t in tickets if t.updated_at is not None]
    newest = max((t.updated_at for t in tickets), default=None)
    mark = max((m for m in (since, newest) if m is not None), default=None)
    at_mark = sorted({t.entity for t in tickets if t.updated_at == mark} | (set(seen) if mark == since else set()))
    if since is None:
        return [], mark, at_mark
    wanted, done = set(queues), set(seen)
    fresh = [t for t in tickets if (t.updated_at > since or (t.updated_at == since and t.entity not in done))
             and not t.closed and (not wanted or t.queue in wanted)]
    return sorted(fresh, key=lambda t: (t.updated_at, t.entity)), mark, at_mark


def newest_per_ticket(tickets: Iterable[Ticket]) -> list[Ticket]:
    """One ticket per entity: the newest update (two sources may publish the same ticket), oldest first."""
    best: dict[str, Ticket] = {}
    for t in tickets:
        if t.entity not in best or t.updated_at > best[t.entity].updated_at:
            best[t.entity] = t
    return sorted(best.values(), key=lambda t: (t.updated_at, t.entity))


def packs_of(automation: Any) -> Optional[set[str]]:
    """The packs an automation's gate sources name (`<pack>.<source>`); None when it has no gate."""
    gate = getattr(automation, "gate", None)
    return {s.split(".", 1)[0] for s in gate.sources} if gate is not None and gate.sources else None


def _system_marker(system: str) -> str:
    return f"{system}/*"


def scan_queue(io: Any, automation: str, systems: Iterable[str], *, queues: Iterable[str] = (),
               packs: Optional[Iterable[str]] = None, closed_statuses: frozenset[str] = CLOSED_STATUSES,
               say: Callable[[str], None] = log.info) -> list[str]:
    """Read each system's tickets at the latest published generation and add their signals. Returns the
    signal ids it added, as `<automation>/<id>`."""
    generation = io.pin_generation()
    added: list[str] = []
    rows: dict[str, list] = {}  # one read per subtype per scan (tracker_ticket holds two systems)
    for system in systems:
        marker = io.ticket_cursor(automation, _system_marker(system))
        if marker is not None and marker["generation"] == generation:
            continue
        tickets: list[Ticket] = []
        skipped: list[dict] = []
        for subtype in subtypes_for(system, packs):
            if subtype not in rows:
                rows[subtype] = io.read_nodes(subtype, generation)
            tickets += tickets_of(system, rows[subtype], closed_statuses=closed_statuses, skipped=skipped)
        for s in skipped:
            say(f"archi_ops_scan: {automation}: skipped {s['what']}: {s['reason']}")
        by_source: dict[str, list[Ticket]] = {}
        for t in tickets:
            by_source.setdefault(t.source, []).append(t)
        fresh: list[Ticket] = []
        moves: list[tuple] = []  # cursor writes, made only once every signal of this read exists
        for source, group in sorted(by_source.items()):
            cursor = io.ticket_cursor(automation, source)
            found, mark, at_mark = new_or_updated(group, cursor["updated"] if cursor else None, queues,
                                                  (cursor or {}).get("seen") or ())
            fresh += found
            moves.append((source, mark, at_mark))
            if cursor is None:
                say(f"archi_ops_scan: {automation}: first read of {source} ({len(group)} tickets); "
                    "triage starts with the next new or updated ticket")
        for ticket in newest_per_ticket(fresh):
            seen = io.signal_ids_for(automation, ticket.entity)
            latest = max((m for m in (signal_time(s) for s in seen) if m is not None), default=None)
            if latest is not None and ticket.updated_at.replace(microsecond=0) < latest:
                continue  # an older version than one already signalled (a source that publishes late)
            sid = signal_id(ticket)
            if not io.add_signal(automation, sid, ticket.entity, f"{ticket.system} {ticket.key}: {ticket.subject}"[:300],
                                 summary=f"status {ticket.status or '-'}; queue {ticket.queue or '-'}; "
                                         f"updated {ticket.updated}", source="ticket"):
                continue  # signalled before (a ticket updated at its source's cursor)
            superseded = io.supersede_signals(automation, ticket.entity, sid)
            if superseded:
                say(f"archi_ops_scan: {automation}: {sid} supersedes {', '.join(superseded)}")
            added.append(f"{automation}/{sid}")
        # after the signals: a failure above leaves the cursors where they were, so the next scan reads again
        for source, mark, at_mark in moves:
            io.set_ticket_cursor(automation, source, mark, generation, at_mark)
        io.set_ticket_cursor(automation, _system_marker(system), None, generation, [])
    return added


def add_ticket_signals(rt: Any, io: Any, say: Callable[[str], None] = log.info) -> list[str]:
    """`scan_queue` for every automation with a `ticket_queue` trigger. A failure stops that automation's
    signals for this scan (it is logged), not the scan."""
    load = getattr(rt, "automation", None)
    if load is None:
        return []
    added: list[str] = []
    for name in sorted(rt.settings):
        try:
            auto = load(name)
            trigger = auto.trigger
            if trigger.get("type") != "ticket_queue":
                continue
            systems = [s for s in trigger_systems(trigger) if s in SYSTEMS]
            settings = rt.settings[name]
            added += scan_queue(io, name, systems, queues=getattr(settings, "queues", ()), packs=packs_of(auto),
                                closed_statuses=closed_statuses(settings.proposer), say=say)
        except Exception as exc:  # a bad automation file or an unreadable graph stops this automation, not the scan
            say(f"archi_ops_scan: {name}: cannot read its ticket queue ({type(exc).__name__}: {exc})")
    return added
