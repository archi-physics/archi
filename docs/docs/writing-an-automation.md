# Writing an automation

An automation is an automation pack. It declares what starts a run, what
context it needs, what the agent reads, and which actions it may propose.
You write the declaration. The runtime runs it.

Read [the operations layer](operations-layer.md) first. The full key list is
in [Automation packs](reference/pack-format.md#automation-packs). The
shipped example is [`instance-health`](library/packs/instance-health.md).

## Steps

1. **Make the pack.** `kind: automation`, with `requires.packs` naming the
   packs whose live tools will carry out its actions, and `ops-record` so
   it can record what it does.
2. **Choose the trigger.** One of:
    - `schedule` with a `cron` of five fields
    - `ticket_queue` with a `system` (`redmine`, `jira` or `servicenow`) and
      a `queue`
    - `signal` with a `source` and a `condition`

   Archi refuses a cron that can fire more often than `budget.runs_per_hour`.
3. **Set the context gate.** Name the sources, the maximum staleness, the
   minimum context-eval score and the invariants that must hold. A gate must
   check something. If it fails, the run drops to shadow and says why.
4. **Write the playbook.** A Markdown file under the pack's `playbooks/`
   directory. It is the instruction the agent follows.
5. **Define the action classes.** For each: a `payload_schema` (JSON Schema),
   a `mediator`, `reversible`, a `blast_radius` sentence and a `max_level`.
   The mediator is either a live tool (`archi.tools.x:fn`) or a workflow
   step. An action that changes an outside system must be a live tool.
   A live tool that can run at L2 must also be listed under
   `idempotent_tools`, so a retry has the effect once.
6. **Fix the proposal schema.** It must require `summary`, `evidence_ids`,
   `action_class`, `payload`, `confidence` and `rollback`. A proposal
   without evidence ids is invalid.
7. **Set the budget and levels.** `cost_per_run`, `cost_per_day` and
   `runs_per_hour` must be positive. `default_level` and `max_level` are
   `L0`, `L1` or `L2`. L3 is refused.
8. **Check it.**

   ```bash
   archi bundles validate path/to/your-pack
   ```

   The command lists every problem in one run: a missing playbook, an action
   class that is not defined, a mediator that resolves to nothing, an invalid
   schema.

## Turn it on

An instance selects the pack (`archi init --with` or `archi add`), sets the
level in `archi.yaml` and names the automation in `ops.yaml`. See
[Running automations](reference/pack-format.md#running-automations-archi-ops).
Every automation starts in shadow. [Promotion](operations-layer.md#shadow-and-promotion)
to live needs scored shadow results and an operator's approval.

## Rules to keep in mind

- Agents propose. Mediators act. Never give an agent a credential.
- Prefer a fixed check to an agent where one will do.
- Name the highest level honestly. L2 means a person approves every action.
