# Operational Precedents

Use this skill before proposing an action on an incident, and whenever a
question asks what happened last time, how a problem was fixed before, or
whether a fix worked. The operational memory holds past incidents with what
was proposed, who decided, what ran, and how it turned out.

## The Records

Five subtypes, one per step of the operations loop. Every record has a
`record_id`, and its node id is `<subtype>:<record_id>`.

- `ops_incident`: what broke. `entity` is the thing that broke (a site such
  as T2_US_MIT, a node, a service), `error_signature` the normalised error,
  `status` open, mitigated, resolved or false_alarm (a person judged it not
  a real problem), `ticket_keys` the tickets tracking it.
- `ops_proposal`: what an automation proposed for an incident
  (`incident_id`): `action_class`, `payload`, `confidence`, `rollback`, and
  `evidence_ids`, the okg node ids it cited.
- `ops_decision`: the verdict on a proposal (`proposal_id`): approved,
  rejected or timed_out; `actor_kind` person or automation; `escalation_id`
  when a person decided through an escalation.
- `ops_action`: what the mediator ran for a proposal (`proposal_id`):
  `status` succeeded or failed, and its `result`.
- `ops_outcome`: whether the incident (`incident_id`) was fixed: `result`
  resolved, not_resolved, partial, unknown or no_action_needed (a person
  checked and nothing needed doing), and `resolution` in words.
  `check_method` ticket_resolution means a person resolved the ticket named
  in `ticket_key`.

Edges: proposal `addresses` incident; decision `decides` proposal; action
`executes` proposal and is `authorized_by` its decision; outcome
`outcome_of` incident and `evaluates` action; a proposal `cites` the
precedent incidents it relied on.

## Retrieval Strategy

1. Name the entity. Take it from the incident being triaged (`entity`), or
   from the question (a site, node, service or dataset name, exactly as
   operators write it).
2. List past incidents on it:
   `filter(where={"type": "ops_incident", "attrs.entity": "<entity>"})`.
   Leave out the incident you are triaging. Rank those with the same
   `error_signature` first, then the most recent.
3. If nothing matches, search the symptom instead, in the operational memory
   only: `search(query="<error text> <entity>", profile="precedents")`. For a
   ticket key, `filter(where={"type": "ops_outcome", "attrs.ticket_key":
   "<key>"})` finds the outcome its resolution produced, and its incident.
4. For each precedent, read what happened to it:
   `expand(input={node_id: "ops_incident:<record_id>"}, direction="in")`
   returns its proposals (`addresses`) and outcomes (`outcome_of`). Expand
   each proposal the same way for its decision (`decides`) and action
   (`executes`). The same records can be read by attribute:
   `filter(where={"type": "ops_outcome", "attrs.incident_id": "<record_id>"})`.
5. Open the evidence a precedent cited (`evidence_ids`, and the ticket in
   `ticket_keys` or `ticket_key`) with `inspect` when the reasoning matters,
   not just the verdict.

## Using a Precedent

- Report each precedent as: when, what broke, what was proposed, who
  decided, what ran, and the outcome. Cite the incident and outcome node ids.
- Only an outcome says whether a fix worked. An approved decision or a
  succeeded action is not an outcome: say "no outcome recorded" when there
  is none.
- `result: unknown` from a ticket resolution means the ticket closed without
  a fix (Won't Fix, Duplicate). Do not present it as a fix.
- `archi.ops.precedents` lists an incident's outcomes oldest first by
  `checked_at`; the last one is the latest. When outcomes disagree, report
  both with their dates.
- A precedent whose proposal was rejected or timed out is still useful: say
  why it was refused (`reason`).
- When you propose an action because of a precedent, put the precedent's
  incident node id in the proposal's evidence.
- Precedents are context, not authority. The current incident's own evidence
  decides; a precedent on another entity, or with another error signature,
  is weaker and must be labelled as such.
