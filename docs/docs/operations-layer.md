# The operations layer

The operations layer runs automations over the context layer. It is the
purpose of Archi 3.0. Agents propose. A separate component, the **mediator**,
acts. No agent holds a credential that can change an external system.

## The loop

Every automation follows the same loop.

```
signal -> context read -> proposal -> decision -> action -> outcome
```

1. **Signal.** A schedule fires, a ticket arrives or a watched value crosses
   a threshold.
2. **Context read.** The runtime reads the graph at one pinned generation and
   checks the [context gate](context-layer.md#the-context-gate).
3. **Proposal.** The agent writes a typed proposal: a summary, the ids of the
   evidence it used, an action class and payload, a confidence and a
   rollback.
4. **Decision.** The autonomy policy, and at L2 a person, decide.
5. **Action.** The mediator carries out an approved action class.
6. **Outcome.** The result is checked later and written to the
   [ops-record](context-layer.md#ops-record-the-operational-memory).

The loop runs as an OKG deployment workflow, scheduled by DBOS (a durable
workflow library). Durable state lives in the database, never in a local
file.

## Automation packs

An [automation pack](reference/pack-format.md#automation-packs) declares one
automation:

- a **trigger**: a schedule, a new ticket in a queue, or a watched signal
- a **context gate**
- a **playbook**: the instructions the agent follows
- **action classes**: each with a typed payload, its mediator, whether it is
  reversible, a blast-radius note and the highest level it may have
- a **proposal schema** and a **budget** (cost per run and per day, runs per
  hour)

Tool packs ship the live tools that act as mediators. An automation never
calls an external system any other way. See
[Writing an automation](writing-an-automation.md).

## Autonomy levels

| Level | Meaning | In 3.0 |
|---|---|---|
| L0 observe | Writes findings to the ops-record only. | Yes |
| L1 draft | Drafts a reply or a ticket for a person to send. | Yes |
| L2 propose | Proposes an action. It runs only after a person approves an escalation. A timed-out proposal never runs. | Yes |
| L3 act within limits | Runs a reversible action without review, inside rate and budget limits. | No |

**Why not L3.** There is no live track record of L2 automations yet. Without
one, nobody can say that an automation is safe to run unreviewed. So the
3.0 runtime refuses an L3 setting wherever it appears: in a pack, a bundle,
`archi.yaml`, or at run start. L3 is revisited once L2 automations have a
live history.

Three rules set the level:

- The instance sets each automation's level in its own `archi.yaml`. A
  bundle only sets an upper bound.
- The effective level is the lowest of the instance level, the bundle bound
  and the pack maximum.
- An action class has its own highest level too.

```yaml
# archi.yaml
automations:
  instance-health:
    level: L1
```

## Shadow and promotion

Every automation starts in **shadow** at its configured level. It proposes
and records, and nothing runs. Its proposals are scored against what people
did on the same incidents: how often it agreed, how often it was wrong, how
quickly it would have responded.

Going live needs recorded shadow results and an operator's approval:

```bash
archi ops promote <automation>
```

The command refuses unless the shadow report passes the thresholds (by
default at least 30 proposals, 20 answered, 50% coverage, 80% agreement and
at most a 10% false-proposal rate; each can be changed with a flag). It then
opens a change in the instance repository that holds the report. An operator
approves that change with `okg pact decide`. Archi never takes that step.
Until it is approved, the automation stays in shadow. Editing `ops.yaml` to
say `live` changes nothing.

A live automation whose context gate fails runs in shadow for that run. It
stays in shadow until a promotion is approved after the failure.

## Halt

```bash
archi ops halt
```

stops every automation on the instance. It pauses the scanner, rejects each
open escalation, cancels runs that have not finished and records them as
`halted`. `archi ops resume` undoes the pause. Halt wraps OKG's pause and
drain until OKG ships a first-class kill switch.

## archi ops

| Command | What it does |
|---|---|
| `archi ops status` | Levels, schedules, pending escalations, budgets and shadow scores. |
| `archi ops pending` | Escalations waiting for a person. |
| `archi ops approve <id>` / `reject <id>` | Decide an escalation. |
| `archi ops halt` / `resume` | Stop or restart all automations. |
| `archi ops promote <automation>` | Ask for an automation to go live. |

The full list is in the [command reference](reference/cli.md). The details
of the runtime (the worker, `ops.yaml`, the mediator contract, recovery after
a crash) are in
[Running automations](reference/pack-format.md#running-automations-archi-ops).

## What ships

Two automation packs ship. [`ticket-triage`](library/packs/ticket-triage.md), the reference
automation in the design, triages new and updated Redmine, JIRA and ServiceNow tickets: a cited
draft at L1, an approved assignment or status change at L2. At SubMIT it replaces the v2 Redmine
bot at the 3.0 cutover, with no shadow trial first (board d717). [`instance-health`](library/packs/instance-health.md) is
an automation that watches the instance itself (doctor results, source
freshness, failed ingests, automation error rates) and keeps one issue per
problem in the instance repository. It starts at L1 and runs in shadow
until promoted.
