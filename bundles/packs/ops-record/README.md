# ops-record

The operational memory (ADR 0002 §6.2): what automations and people did
about incidents, written into the graph next to the docs and tickets it
cites, so the next triage can see what happened last time. PACT
`r3-ops-record`.

The pack is a context pack with no sources. Records arrive through okg
admission from automations, and ticket resolutions arrive through the ticket
packs.

## Records

| Subtype | What it holds | Links |
|---|---|---|
| `ops_incident` | what broke (`entity`, `entity_kind`), the symptom (`error_signature`), `status`, `ticket_keys` | |
| `ops_proposal` | what an automation proposed: `action_class`, `payload`, `confidence`, `rollback`, `evidence_ids` | `addresses` incident; `cites` precedent incidents |
| `ops_decision` | `verdict` (approved, rejected, timed_out), `escalation_id`, `reason` | `decides` proposal |
| `ops_action` | what the mediator ran: `mediator`, `status`, `result` | `executes` proposal; `authorized_by` decision |
| `ops_outcome` | whether it worked: `result`, `resolution`, `check_method`, `ticket_key` | `outcome_of` incident; `evaluates` action |

Every record has `record_id` (the node id is `<subtype>:<record_id>`),
`actor_kind` (automation or person), `actor`, `automation` and
`generation_read`. The schema is `schemas/ops_record.yaml`, and the edges are
in `schemas/bridges/ops_record.yaml`.

## Writing records

`archi.ops.record` builds the payload and submits it with
`okg admission submit` as the `workflow` producer `workflow:archi-ops`. The
producer policy's output signature names exactly what the payload writes, so
admission checks every node against the catalog and every edge against the
narrowings. Nothing is written by direct SQL.

```bash
python -m archi.ops.record submit --deployment myteam --records run.jsonl
```

`run.jsonl` holds one record per line, `{"kind": "incident" | "proposal" |
"decision" | "action" | "outcome", <attributes>}`. A proposal's cited node
ids can be given as `"evidence": [...]`. From Python:

```python
from archi.ops.record import OpsRecords, submit

records = OpsRecords()
records.add("incident", record_id="inc-1", title="...", entity="T2_US_MIT", ...)
records.add("proposal", record_id="prop-1", incident_id="inc-1", summary="...",
            evidence=["jira:OPS-11"], generation_read="okg:...:gen:...")
submit(records, deployment="myteam")
```

The records are in the graph once okg publishes the next generation.

A record linked to one admitted in an earlier payload, such as an outcome
checked hours after its incident, must carry that record with it:
`records.carry(row)`, where `row` is the earlier record as `okg filter
--jsonl` prints it. `submit` refuses a batch with an edge to a record that is
not in it.

### Ticket resolutions

An incident names the tickets that track it in `ticket_keys`. Once a ticket
connector reports one of them resolved, the person's resolution becomes the
incident's outcome:

```bash
python -m archi.ops.record link-tickets --deployment myteam
```

The command reads incidents, tickets (`jira_issue`) and outcomes through
`okg filter`. For each resolved tracking ticket without an outcome, it admits
an `ops_outcome` with `check_method: ticket_resolution`, the `ticket_key`,
`actor_kind: person` and the assignee. If the resolution is a fix, it also
writes the incident again as `resolved`. A resolution that is not a fix (Won't
Fix, Duplicate) gives `result: unknown`. Running the command again adds
nothing.

## Recalling precedents

Before an automation proposes anything, it reads what happened last time.
The pack ships three things for that:

- the playbook `skills/ops_precedents.md`, which the `ops_precedents` skill
  trigger offers for any ops record. It teaches an agent to filter incidents
  by `entity`, expand each one to its proposals, decisions, actions and
  outcomes, and to treat only an outcome as proof that a fix worked;
- the `precedents` search profile, which searches the five ops subtypes and
  nothing else: `search(query=..., profile="precedents")` over MCP, or
  `okg search --scope precedents`;
- `archi.ops.precedents`, the same walk as a function and a command:

```bash
python -m archi.ops.precedents --deployment myteam --incident inc-20261003-t2usmit-srm
python -m archi.ops.precedents --deployment myteam --entity T2_US_MIT
```

It returns past incidents on the entity, excluding the one being triaged.
Those with the same `error_signature` come first, then the most recent. Each
carries its proposals (with their decisions and actions) and its outcomes.

## Proof

- `python/tests/packs/test_ops_record.py`: the slice composes with okg's
  catalog on cern-team and cluster-manager.
- `python/tests/packs/test_ops_record_write.py`: the payload, the policy and
  ticket linking.
- `python/tests/packs/test_ops_record_recall.py` and `python/tests/test_skills.py`:
  a fixture incident retrieves its precedent with the outcome; the playbook.
- `python/tests/packs/scratch_ops_record_run.sh`: the write and recall paths on
  a real okg instance.

## What okg does not do yet

Each gap below was found on the scratch instance. The drafted okg asks are in
[`pact/changes/r3-ops-record/okg-asks.md`](../../../pact/changes/r3-ops-record/okg-asks.md).

1. **Edges to nodes outside the payload.** Generic admission refuses an edge
   whose other end is already in the graph but not in the submitted payload.
   For that reason, references to tickets, cited evidence and the entity that
   broke are attributes and not edges, and earlier ops records travel with
   the records that link to them.
2. **No admission-routed write for deployment workflows.** The deployment
   SDK's `emit_node_fact` and `emit_edge_fact` bypass admission. So
   automations write through `okg admission submit` (CLI), which
   `archi.ops.record` wraps.
3. **List attributes are left out of search text.** `ticket_keys` is
   therefore not searchable. A ticket's outcome is found by its scalar
   `ticket_key`.
