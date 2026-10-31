# Ticket triage

The `ticket-triage` automation reads new and updated tickets from Redmine, JIRA or ServiceNow,
whichever this instance ingests (JIRA through the `jira` pack or the `tickets` pack's generic JIRA
reader). For each one it writes a cited triage draft. At L2 it can also propose
one change to the ticket, which happens only after a person approves it. Every proposal is recorded in
the operational record (`ops_proposal`), whatever it proposes.

## What starts a run

A ticket whose `updated` time is newer than the last one the automation saw from that ticket's source (a
source is one ticket subtype on one server; each keeps its own place), once a published generation holds
it. Closed tickets are skipped. `queues:` in `ops.yaml` limits it to some projects (ServiceNow: assignment
groups). The tickets already in the queue on the day the automation is installed, or the day a new source
appears, are not triaged.

One ticket has at most one run at a time and one signal waiting: a newer update replaces the waiting one.
A run waiting for a person to approve a change to one ticket does not hold up the other tickets (at most
five runs at once).

## What a run reads, at its pinned generation only

1. The ticket, by its key.
2. Related tickets: okg's search over the same system, ranked by the ticket's subject and the start of
   its description. Closed ones show how they ended.
3. Precedents: ops-record incidents on the ticket or a related ticket, and what was proposed and decided
   for them (see `ops_precedents.md`).
4. Documentation: the document subtypes `proposer.doc_subtypes` names (by default ServiceNow knowledge
   articles and documentation pages).

A proposal may cite only what this read returned. Each claim in a draft carries a `[n]` that points at a
node id under **Sources**.

## What it proposes

| Level | Proposal | What happens |
|---|---|---|
| L0 | the triage draft | recorded only |
| L1 (default) | the triage draft: category, likely component, related tickets, precedents, documentation, and a suggested reply | recorded on the proposal for a person to send; nothing is sent |
| L2 | `post_reply` when `systems.<system>.post_replies` is set and the model wrote the reply (see below); else `assign_ticket` if the ticket is unassigned and most related tickets went to the same person (ServiceNow: group); else `set_ticket_status` to `triaged_status` for a ticket in one of `triage_from`; else the draft. A change a person already rejected for this ticket is not proposed again | an escalation waits for a person; only an approval runs the live tool, which notes the approval on the ticket (naming related tickets by key). A rejected or timed-out proposal never runs |

Every automation starts in **shadow**: it proposes and records, opens no escalation and changes nothing,
until an operator promotes it (`archi ops promote ticket-triage`, then an approved PACT change).
With no shadow trial, `archi ops promote ticket-triage --no-trial "<decision>"` names the decision that waived
it; an operator still approves the change.

## Going back to shadow

Set `mode: shadow` for `ticket-triage` in `ops.yaml`. From the next run on nothing is sent, and a run already
waiting for an approval sends nothing either: the step that acts reads `ops.yaml` again. No restart is needed.

## Reading what it did

- `archi ops status`: level, mode, budget and shadow scores.
- `archi ops pending`, then `archi ops approve <escalation> --as <you>` or `archi ops reject`.
- The draft of a run is the `draft` of its `ops_proposal` (`okg filter --type ops_proposal --where
  attrs.automation=ticket-triage`).
- `archi ops halt` stops every automation on the instance; `archi ops resume` starts them again.

## Raising it to L2

Set `automations.ticket-triage.level: L2` in `archi.yaml`, and in `ops.yaml` give each system its API and
the names of its credential variables:

```yaml
automations:
  ticket-triage:
    proposer:
      kind: ticket-triage
      systems:
        redmine: {url: https://redmine.example.org, token_env: REDMINE_API_KEY, triaged_status: In Progress, triage_from: [New]}
        jira: {url: https://jira.example.org, token_env: JIRA_TOKEN, triaged_status: In Progress}
        servicenow: {url: https://example.service-now.com, user_env: SERVICENOW_USERNAME, password_env: SERVICENOW_PASSWORD}
```

The worker's environment holds the variables; archi never writes their values anywhere. Without a
system's `url` the run drafts and its summary says why.

A change goes only to the server the ticket was read from. The proposal carries the ticket's link, and
both the proposer and the live tool refuse unless that link is under the system's `url` (or under `site`,
when `url` is a local proxy: `site` is the root of the ticket links the real server publishes). A ticket key
found on two servers of one system (two JIRA servers with the same project key) is drafted, never changed.
An endpoint URL may not carry a user name or password.

## A model-written reply

With `proposer.drafter` set, the suggested reply is written by a model from the records the run read, citing
them as `[node id]`. A reply that cites anything else, or a model that fails, is not used: the template reply
stays and the summary says why. Each call is charged to the pack budget, so `price` is required.

```yaml
    proposer:
      kind: ticket-triage
      drafter:
        model: gpt-6-luna
        price: {input_per_million: 0.10, output_per_million: 0.50, cached_input_per_million: 0.01}
      systems:
        redmine: {url: https://redmine.example.org, token_env: REDMINE_API_KEY, post_replies: true}
```

The model's key is read from `OPENAI_API_KEY` (`api_key_env` names another variable). With `post_replies`, the
L2 proposal is `post_reply`: after a person approves, the reply is posted on the ticket (a Redmine note, a JIRA
comment, a ServiceNow comment) with a line naming the approval. A posted note cannot be withdrawn by archi.
