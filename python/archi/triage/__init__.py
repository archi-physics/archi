"""The ticket-triage automation (ADR 0002 section 7.4, PACT r3-auto-ticket-triage).

    tickets    one shape for a Redmine, JIRA or ServiceNow ticket node
    signals    the scanner's half: a new or updated ticket in the published graph becomes a signal
    context    the run's pinned read: the ticket, related tickets, documentation and precedents
    proposer   the triage draft (L1) and the status or assignment proposal (L2); algorithmic, no model call

The live tools that change a ticket after a person approves are `archi.tools.tickets`. The pack is
`bundles/packs/ticket-triage/`.
"""
