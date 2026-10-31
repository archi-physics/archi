"""The ticket-triage proposer: algorithmic, no model call (ADR 0002: algorithmic first).

It reads the run's pinned context (`archi.triage.context`) and writes one proposal, citing by node id
everything it used. Every proposal carries the triage note:

    category           the ticket's own (Redmine tracker, JIRA issue type, ServiceNow category), else the
                       most common among the related tickets
    likely component   the ticket's own (JIRA components, ServiceNow configuration item, Redmine project),
                       else the most common among the related tickets
    related tickets    what okg's search ranked closest, with status and resolution
    precedents         ops-record incidents on the ticket or a related ticket
    documentation      the documents okg's search ranked closest

What it proposes depends on the run's level:

    L0, L1  action class draft_reply: the note and a suggested reply, cited, for a person to send. Nothing
            is sent (draft_reply is a step with no implementation; the loop records the draft).
    L2      assign_ticket when the ticket is unassigned and at least `min_agreement` (2) related tickets,
            more than half of those assigned, went to the same person (ServiceNow: the same group);
            else set_ticket_status when the instance names a `triaged_status` for the system and the
            ticket is in one of its `triage_from` statuses; else draft_reply, which the loop records as
            capped (a draft never runs at L2). Either action runs only after a person approves it.

An L2 action needs the system's API (`systems.<system>.url` and the names of its credential variables);
without it the run drafts and the summary says why. The proposal's payload carries the endpoint and the
ticket's link, so the person approving sees exactly where the change goes. It proposes no change, and drafts
instead, when:

    the ticket's link is not under the endpoint (`systems.<system>.site`, else `.url`): the ticket was read
        from another server than the one the instance writes to (two JIRA servers can share a key);
    the same key is on more than one server: which ticket the signal meant cannot be told;
    a person already rejected the same change for this ticket (the context's `rejected`).

The note written on the ticket names related tickets by their keys, never by graph node ids. The confidence is a heuristic (how much of the note
the related tickets agree on), not a calibrated probability.

ops.yaml:

    automations:
      ticket-triage:
        proposer:
          kind: ticket-triage
          related_limit: 5
          doc_subtypes: [servicenow_kb_article, documentation_page]
          min_agreement: 2
          closed_statuses: [Abandoned]           # added to the built-in list
          systems:
            redmine:
              url: https://redmine.example.org
              # site: https://redmine.example.org   # ticket links root, only when url is a loopback proxy
              token_env: REDMINE_API_KEY
              triaged_status: In Progress
              triage_from: [New]
            servicenow:
              url: https://example.service-now.com
              user_env: SERVICENOW_USERNAME
              password_env: SERVICENOW_PASSWORD
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from archi.enrichment.anonymizer import redact_email_addresses
from archi.ops.proposers import Proposed, ProposeRequest, ProposerError
from archi.tools.tickets import ENDPOINT_KEYS, LOOPBACK, under
from archi.triage.tickets import ASSIGN_FIELDS

DEFAULT_TRIAGE_FROM = ("new", "open", "to do", "backlog")

class _Cites:
    """Numbered citations: `[n]` in the text, `[n] node id` under Sources."""

    def __init__(self) -> None:
        self.ids: list[str] = []

    def __call__(self, node_id: str) -> str:
        if node_id not in self.ids:
            self.ids.append(node_id)
        return f"[{self.ids.index(node_id) + 1}]"

    def many(self, node_ids) -> str:
        return "".join(self(n) for n in node_ids)


def _majority(values: list[tuple[str, str]]) -> tuple[str, list[str], int, int]:
    """(value, node ids that carry it, its count, how many carried any value) of `(value, node id)` pairs."""
    present = [(v, n) for v, n in values if v]
    if not present:
        return "", [], 0, 0
    value, count = Counter(v for v, _ in present).most_common(1)[0]
    return value, [n for v, n in present if v == value], count, len(present)


def _endpoint(config: Mapping[str, Any]) -> Optional[dict]:
    if not config.get("url"):
        return None
    return {k: str(config[k]) for k in ENDPOINT_KEYS if config.get(k)}


class TriageProposer:
    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})
        self.min_agreement = int(self.config.get("min_agreement", 2))

    def system_config(self, system: str) -> Mapping[str, Any]:
        return (self.config.get("systems") or {}).get(system) or {}

    def propose(self, request: ProposeRequest) -> Proposed:
        ctx = request.context
        ticket = ctx.get("ticket")
        if not ticket:
            raise ProposerError("the context has no ticket (the run's read is not a ticket-triage read)")
        related = list(ctx.get("related") or ())
        docs = list(ctx.get("docs") or ())
        precedents = list(ctx.get("precedents") or ())
        cite = _Cites()
        cite(ticket["node_id"])

        category, cat_ids, cat_n, cat_of = (ticket["category"], [ticket["node_id"]], 1, 1) if ticket["category"] \
            else _majority([(r["category"], r["node_id"]) for r in related])
        own_components = ", ".join(ticket["components"])
        component, comp_ids, comp_n, comp_of = (own_components, [ticket["node_id"]], 1, 1) if own_components \
            else _majority([(c, r["node_id"]) for r in related for c in r["components"]])
        assign_field = ASSIGN_FIELDS[ticket["system"]]
        assignee, as_ids, as_n, as_of = _majority(
            [((r["group"] if assign_field == "assignment_group" else r["assignee"]), r["node_id"]) for r in related])
        agreed = as_n >= self.min_agreement and as_n * 2 > as_of

        label = f"{ticket['system']} {ticket['key']}"
        lines = [f"Triage draft for {label}: {ticket['subject']}",
                 "(Drafted by the ticket-triage automation for a person to review. Nothing was sent.)", ""]

        def basis(n: int, of: int, ids: list[str]) -> str:
            if ids == [ticket["node_id"]]:
                return f"from the ticket {cite(ticket['node_id'])}"
            return f"{n} of {of} related tickets {cite.many(ids)}"

        lines.append(f"- Category: {category} ({basis(cat_n, cat_of, cat_ids)})" if category
                     else "- Category: unknown (neither the ticket nor a related ticket names one)")
        lines.append(f"- Likely component: {component} ({basis(comp_n, comp_of, comp_ids)})" if component
                     else "- Likely component: unknown")
        if agreed:
            lines.append(f"- Usually handled by: {redact_email_addresses(assignee)} ({as_n} of {as_of} related tickets {cite.many(as_ids)})")
        lines.append("- Related tickets:" if related else "- Related tickets: none found")
        for r in related:
            end = f"; resolution: {r['resolution']}" if r["resolution"] else ""
            lines.append(f"  {cite(r['node_id'])} {r['system']} {r['key']}: {r['subject']} ({r['status'] or 'no status'}{end})")
        lines.append("- Precedents in the operational record:" if precedents else "- Precedents: none recorded")
        for p in precedents:
            lines.append(f"  {cite(p['node_id'])} {p.get('title') or p['node_id']} ({p.get('status') or '-'}, "
                         f"{p.get('detected_at') or '-'})")
        lines.append("- Documentation:" if docs else "- Documentation: none found")
        for d in docs:
            lines.append(f"  {cite(d['node_id'])} {d['title']}" + (f" <{d['url']}>" if d.get("url") else ""))

        reply = ["", "Suggested reply:", "Thank you for the report."]
        solved = next((r for r in related if r["closed"]), None)
        if solved is not None:
            how = f" It ended as: {solved['resolution']}." if solved["resolution"] else ""
            reply.append(f"A similar ticket, {solved['system']} {solved['key']} {cite(solved['node_id'])}, "
                         f"was closed before.{how}")
        if docs:
            reply.append(f"This documentation may help: {docs[0]['title']} {cite(docs[0]['node_id'])}.")
        reply.append("We will follow up on this ticket.")
        sources = ["", "Sources:", *(f"[{i + 1}] {node}" for i, node in enumerate(cite.ids))]
        body = "\n".join(lines + reply + sources)

        agreement = max(comp_n / comp_of if comp_of else 0.0, cat_n / cat_of if cat_of else 0.0)
        confidence = round(min(0.9, 0.3 + (0.2 if related else 0.0) + (0.1 if docs else 0.0) + 0.3 * agreement), 2)
        summary = (f"Triage {label}: category {category or 'unknown'}; component {component or 'unknown'}; "
                   f"{len(related)} related ticket(s), {len(precedents)} precedent(s), {len(docs)} document(s)")
        draft = {
            "summary": summary,
            "evidence_ids": list(cite.ids),
            "action_class": "draft_reply",
            "payload": {"system": ticket["system"], "ticket": ticket["key"], "body": body,
                        "category": category, "component": component},
            "confidence": confidence,
            "rollback": "Nothing to undo: the draft was not sent.",
        }
        if request.level < 2:
            return Proposed(draft, cost=0.0)
        keys = {r["node_id"]: f"{r['system']} {r['key']}" for r in related}
        return Proposed(self.l2(ticket, draft, assignee if agreed else "", [keys.get(n, n) for n in as_ids], ctx),
                        cost=0.0)

    def l2(self, ticket: Mapping[str, Any], draft: dict, assignee: str, by: list[str], ctx: Mapping[str, Any]) -> dict:
        """The L2 proposal: an assignment, a status change, or the draft (recorded as capped). `by` names the
        related tickets that agree on the assignee, as `<system> <key>`."""
        system, key = ticket["system"], ticket["key"]
        config = self.system_config(system)
        endpoint = _endpoint(config)

        def no_action(why: str) -> dict:
            return {**draft, "summary": f"{draft['summary']}; no L2 action: {why}"}

        if endpoint is None:
            return no_action(f"proposer.systems.{system}.url is not set")
        sites = set(ctx.get("sites") or ())
        if len(sites) > 1:  # "" is a copy with no link: it could be any server
            shown = ", ".join(sorted(s or "(no link)" for s in sites))
            return no_action(f"the key {key} is on more than one {system} server ({shown})")
        if endpoint.get("site") and urlsplit(endpoint["url"]).hostname not in LOOPBACK and \
                endpoint["site"].rstrip("/") != endpoint["url"].rstrip("/"):
            return no_action(f"proposer.systems.{system}.site differs from its url; a site is only for a loopback proxy")
        root = endpoint.get("site") or endpoint["url"]
        if not ticket.get("url") or not under(ticket["url"], root):
            return no_action(f"the ticket's link ({ticket.get('url') or 'none'}) is not under {root}, the server "
                             f"proposer.systems.{system} writes to")
        base = {"system": system, "ticket": key, "ticket_url": ticket["url"], "endpoint": endpoint}
        if ticket.get("table"):
            base["table"] = ticket["table"]
        rejected = list(ctx.get("rejected") or ())

        def was_rejected(action_class: str, field: str, value: str) -> Optional[str]:
            for r in rejected:
                if r["action_class"] == action_class and str((r["payload"] or {}).get(field) or "").casefold() == \
                        value.casefold():
                    how = "let time out" if r.get("verdict") == "timed_out" else "rejected"
                    return f"{how} ({r['proposal']})"
            return None

        field = ASSIGN_FIELDS[system]
        current = ticket["group"] if field == "assignment_group" else ticket["assignee"]
        skipped = ""
        if assignee and not current:
            before = was_rejected("assign_ticket", "value", assignee)
            shown = redact_email_addresses(assignee)  # the payload keeps the value; text people read does not
            if before is None:
                note = (f"Assigned by the ticket-triage automation: {len(by)} related tickets went to {shown} "
                        f"({', '.join(by)}).")
                return {**draft,
                        "summary": f"{draft['summary']}; proposes: set {field} of {system} {key} to {shown}",
                        "action_class": "assign_ticket",
                        "payload": {**base, "field": field, "value": assignee, "expected": current, "note": note},
                        "rollback": f"Set the {field} of {system} {key} back to unassigned."}
            skipped = f"assigning it to {shown} was {before} before; "
        target = str(config.get("triaged_status") or "")
        from_statuses = {s.lower() for s in (config.get("triage_from") or DEFAULT_TRIAGE_FROM)}
        if target and ticket["status"] != target and ticket["status"].lower() in from_statuses:
            before = was_rejected("set_ticket_status", "status", target)
            if before is not None:
                return no_action(f"{skipped}moving it to {target} was {before} before")
            note = (f"Triaged by the ticket-triage automation: category {draft['payload']['category'] or 'unknown'}, "
                    f"likely component {draft['payload']['component'] or 'unknown'}.")
            return {**draft,
                    "summary": f"{draft['summary']}; proposes: set the status of {system} {key} to {target}",
                    "action_class": "set_ticket_status",
                    "payload": {**base, "status": target, "expected_status": ticket["status"], "note": note},
                    "rollback": f"Set the status of {system} {key} back to {ticket['status']}."}
        return no_action(f"{skipped}nothing to assign or move" if skipped else "nothing to assign or move")


def build(config: Mapping[str, Any], runtime: Any) -> TriageProposer:
    """`archi.ops.proposers.PROPOSER_KINDS["ticket-triage"]`: from the `proposer:` block of ops.yaml."""
    return TriageProposer(config)
