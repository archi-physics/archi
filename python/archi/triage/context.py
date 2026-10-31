"""The ticket-triage run's pinned read: the ticket, related tickets, documentation and precedents.

`archi.ops.workflow` calls `read_ticket_context` for an automation whose trigger is a `ticket_queue`, at
the generation the run pinned, and nowhere else. Everything comes from okg's read surface at that
generation (`okg filter`, `okg search`), so a proposal can cite only what this read returned:

    ticket      the ticket the signal names (its entity `<system>:<key>`), looked up by its key
    related     other tickets of the same system, ranked by okg's lexical search on the ticket's subject
                and the start of its description; open and closed alike (a closed one carries how it ended)
    docs        documentation ranked the same way, from the subtypes the instance names (`doc_subtypes`)
    precedents  ops-record incidents on the ticket or on a related ticket: what an automation proposed for
                them before (the ops-record pack's operational memory)
    rejected    the changes already put to a person for this ticket and rejected or left to time out (an
                ops_decision `rejected` or `timed_out` on a proposal of one of its incidents), so the proposer
                does not ask again on every comment. Known limit: a decision is seen once a generation holding
                it is published; a run pinned to an older generation cannot see it.
    sites       the link roots (archi.triage.tickets.root_of) of every copy of a ticket with this key, "" for a
                copy with no link; more than one means the key is ambiguous and the proposer changes nothing

The read is a run step, and DBOS stores step output: tickets are returned without their description
(archi.triage.tickets.Ticket.as_context), and the search text is not returned.

A doc subtype the instance does not have is skipped with its reason in `context["skipped"]`; a ticket that is
not in the generation is an error (the run records it and proposes nothing).

Configuration, from the automation's `proposer:` block in ops.yaml: `related_limit` (5), `doc_subtypes`
(servicenow_kb_article, documentation_page), `doc_limit` (3), `precedent_limit` (5).
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional

from archi.tools.tickets import under
from archi.triage.tickets import READINGS, Ticket, TicketError, closed_statuses, parse_entity, subtypes_for, tickets_of

DEFAULT_DOC_SUBTYPES = ("servicenow_kb_article", "documentation_page")
QUERY_CHARS = 300


class ContextError(LookupError):
    """The ticket the signal names cannot be read at the pinned generation."""


def query_text(ticket: Ticket) -> str:
    """What related tickets and documents are searched with: the subject and the start of the description."""
    return " ".join(f"{ticket.subject} {ticket.description[:QUERY_CHARS]}".split()) or ticket.key


def _title(row: Mapping[str, Any]) -> str:
    a = row.get("attrs") or {}
    for key in ("title", "subject", "summary", "name", "number"):
        if a.get(key):
            return str(a[key])
    return str(row.get("node_id"))


def _excerpt(row: Mapping[str, Any], chars: int = 240) -> str:
    a = row.get("attrs") or {}
    text = " ".join(str(a.get("description") or a.get("text") or "").split())
    return text[:chars]


def _ranked(io: Any, ids: list[str], generation: str) -> list[dict]:
    """The rows of `ids`, in the order of `ids` (okg filter --stdin returns them in its own order)."""
    rows = {r["node_id"]: r for r in io.read_ids(ids, generation)}
    return [rows[i] for i in ids if i in rows]


def _endpoint_root(config: Mapping[str, Any], system: str) -> str:
    """The root of the ticket links of the server the instance writes `system` to (proposer config
    `systems.<system>.site`, else `.url`); empty when it names none."""
    sys_config = ((config.get("systems") or {}).get(system) or {})
    return str(sys_config.get("site") or sys_config.get("url") or "")


def _rejected(io: Any, generation: str, incident_ids: set[str],
              verdicts: tuple[str, ...] = ("rejected", "timed_out")) -> list[dict]:
    """`{action_class, payload, proposal, verdict}` of each proposal on one of `incident_ids` that a person
    decided with one of `verdicts` (by default: rejected or let time out). Reads the instance's decisions with
    those verdicts (few: a person writes each), then only their proposals."""
    if not incident_ids:
        return []
    decisions = [d for verdict in verdicts
                 for d in io.read_by_attr("ops_decision", generation, "verdict", verdict)]
    verdict_of = {str((d.get("attrs") or {}).get("proposal_id")): (d.get("attrs") or {}).get("verdict") for d in decisions}
    proposal_ids = sorted({f"ops_proposal:{(d.get('attrs') or {}).get('proposal_id')}" for d in decisions
                           if (d.get("attrs") or {}).get("proposal_id")})
    out = []
    for row in io.read_ids(proposal_ids, generation) if proposal_ids else ():
        a = row.get("attrs") or {}
        if str(a.get("incident_id") or "") not in incident_ids:
            continue
        payload = a.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        out.append({"action_class": a.get("action_class") or "", "payload": payload if isinstance(payload, dict) else {},
                    "proposal": row["node_id"], "verdict": verdict_of.get(str(a.get("record_id")), "rejected")})
    return sorted(out, key=lambda r: r["proposal"])


def read_ticket_context(io: Any, entity: str, generation: str, config: Mapping[str, Any] | None = None,
                        packs: Optional[Iterable[str]] = None) -> dict:
    """`{"rows", "context", "node_ids"}`: the rows a proposal may carry (precedent incidents), the context
    the proposer reads, and every node id a proposal may cite. `packs` limits the ticket subtypes read to
    those of the packs the instance selected (archi.triage.signals.packs_of)."""
    config = config or {}
    closed = closed_statuses(config)
    try:
        system, key = parse_entity(entity)
    except TicketError as exc:
        raise ContextError(str(exc)) from None
    subtypes = subtypes_for(system, packs)

    found: list[Ticket] = []
    for subtype in subtypes:
        found += tickets_of(system, io.read_by_attr(subtype, generation, READINGS[subtype].key, key), closed_statuses=closed)
    if not found:
        raise ContextError(f"ticket {entity} is not in generation {generation}")
    sites = sorted({t.root for t in found})
    # the copy on the server the instance writes to, when the key is on two (the proposer then changes nothing)
    home = _endpoint_root(config, system)
    ticket = next((t for t in found if home and t.url and under(t.url, home)), found[0])
    query = query_text(ticket)
    skipped: list[dict] = []

    related_limit = int(config.get("related_limit", 5))
    related: list[Ticket] = []
    seen = {ticket.entity}
    for subtype in subtypes:
        # a tracker_ticket search also finds the other system's tickets: ask for more, keep this system's
        ids = [i for i in io.search_ids(query, generation, subtype, 2 * related_limit + 1) if i != ticket.node_id]
        for other in tickets_of(system, _ranked(io, ids, generation), closed_statuses=closed, skipped=skipped):
            if other.entity not in seen:  # the same JIRA issue from both packs counts once
                seen.add(other.entity)
                related.append(other)
    related = related[:related_limit]

    docs: list[dict] = []
    doc_limit = int(config.get("doc_limit", 3))
    for subtype in config.get("doc_subtypes") or DEFAULT_DOC_SUBTYPES:
        try:
            ids = io.search_ids(query, generation, subtype, doc_limit)
        except Exception as exc:  # most often: this instance has no such subtype
            skipped.append({"what": f"docs:{subtype}", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"})
            continue
        for row in _ranked(io, ids, generation):
            docs.append({"node_id": row["node_id"], "subtype": row.get("subtype"), "title": _title(row),
                         "url": (row.get("attrs") or {}).get("url") or "", "excerpt": _excerpt(row)})
    docs = docs[:doc_limit]

    precedent_rows: list[dict] = []
    precedent_limit = int(config.get("precedent_limit", 5))
    own_incidents: set[str] = set()
    for subject in [ticket, *related]:
        for row in io.read_by_attr("ops_incident", generation, "entity", subject.entity):
            if subject is ticket:
                own_incidents.add(str((row.get("attrs") or {}).get("record_id") or ""))
            if row["node_id"] not in {r["node_id"] for r in precedent_rows}:
                precedent_rows.append(row)
    rejected = _rejected(io, generation, own_incidents - {""})
    # a reply already approved, rejected or let time out for this ticket: the proposer proposes no second one
    replied = [r for r in _rejected(io, generation, own_incidents - {""}, ("approved", "rejected", "timed_out"))
               if r["action_class"] == "post_reply"]
    precedent_rows.sort(key=lambda r: str((r.get("attrs") or {}).get("detected_at") or ""), reverse=True)
    precedent_rows = precedent_rows[:precedent_limit]
    precedents = [{"node_id": r["node_id"], **{k: (r.get("attrs") or {}).get(k) for k in
                   ("entity", "title", "summary", "status", "detected_at", "automation")}} for r in precedent_rows]

    context = {
        "ticket": ticket.as_context(),
        "related": [t.as_context() for t in related],
        "docs": docs,
        "precedents": precedents,
        "rejected": rejected, "replied": replied,
        "sites": sites,
        "skipped": skipped,
    }
    node_ids = [ticket.node_id, *(t.node_id for t in related), *(d["node_id"] for d in docs),
                *(p["node_id"] for p in precedents)]
    return {"rows": precedent_rows, "context": context, "node_ids": sorted(set(node_ids))}
