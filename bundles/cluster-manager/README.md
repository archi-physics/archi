# cluster-manager bundle

A knowledge instance for the people who run a batch cluster (ADR 0002 section 5).
It works for any cluster operator: no pack in it, default or optional, depends
on CERN.

| Pack | Default | What it brings | Credential |
|---|---|---|---|
| `docsite` | yes | Documentation sites. The bundle points it at the Slurm and HTCondor manuals; replace `--docsite-urls` with your own site's pages. | none |
| `tickets` | yes | Tracker tickets from Redmine, any JIRA, or both, from a cache. | optional `REDMINE_API_KEY`; `JIRA_API_TOKEN` for a JIRA, read only by the downloader |
| `ops-record` | yes | Operational memory: incidents, proposals, decisions, actions, outcomes. | none |
| `servicenow` | no | ServiceNow incidents, requests and KB articles, from a cache. | OAuth pair or basic-auth pair, read only by the downloader |
| `slurm` | no | Slurm partitions, nodes, reservations and job summaries from an exported snapshot, plus live `squeue`/`sinfo` tools and the gated actions below. | none in archi |
| `htcondor` | no | An HTCondor pool snapshot, plus live `condor_status`/`condor_q` tools. | none in archi |
| `git-code` | no | A repository's files and history (site configs, scripts). | none for a public repository |
| `node-health` | no | Per-node health checks from your own health exporter. | none |
| `ticket-triage` | no | The ticket-triage automation: a cited triage draft for each new or updated Redmine or JIRA ticket of the tickets pack (and, when selected, ServiceNow); at L2 a status or assignment change that runs only after a person approves it. Starts in shadow at L1. | none at L1; at L2 the tracker's API credential, named in `ops.yaml` |

## Install

```bash
archi render --bundle cluster-manager --with slurm --with servicenow --instance .
okg install --profile cluster-manager --deployment-name <slug> \
  --postgres-dsn "$OKG_DSN" --archi-data-root "$PWD/archi-data/data" \
  --servicenow-instance-url https://example.service-now.com
```

Each context pack reads a cache under `archi-data/data/<name>/` and the cache must exist
before the first ingest: a missing cache blocks the publish rather than publishing less. `tickets`
is a default but reads a tracker you must point it at, so its URL (`--tickets-base-url`) is a required
question with no placeholder default; for a JIRA, also answer `--tickets-system jira`. With no tracker,
leave it out with `--without tickets`. `docsite`
is the credential-free default that downloads public pages at init, so the bundle always has something to publish. The commands that write the caches, each run as the account the site gives it:

```bash
python -m archi.downloaders.tickets --redmine-url https://redmine.example.org       # tickets, Redmine
JIRA_API_TOKEN=... python -m archi.downloaders.tickets \
  --jira-url https://jira.example.org --jira-project OPS                            # tickets, JIRA (or --jql)
python -m archi.downloaders.servicenow --instance-url https://example.service-now.com --auth oauth
python -m archi.downloaders.slurm --cluster <name>                                  # on a login node
python -m archi.downloaders.htcondor --pool <name>                                  # on a submit node
```

The Slurm and HTCondor exporters run read-only client commands (`sinfo`, `scontrol show reservation`,
`sacct`, `condor_status`). `node-health` reads a JSON list your own health exporter writes; the shape
is in `python/archi/sources/node_health.py`.

## Live tools and actions

The `slurm` and `htcondor` packs ship live tools with the boundary `external_live`: what they return
is a reading from the scheduler, not graph evidence. The reads are `archi_slurm_squeue`, `archi_slurm_sinfo`,
`archi_condor_status` and `archi_condor_q`.

Slurm also ships three action classes, `drain_node`, `resume_node` and `requeue_job`. In 3.0 every
one runs only at L2: an automation proposes it, a person approves the escalation, and only then does
the ops runtime call the tool. The tool checks the approval itself before it builds a command and
refuses a call without one. Each is idempotent: it reads the node or job first and does nothing when it
is already in the requested state. archi ships the tools and each instance owns the access they run with.
Nothing in this bundle changes an HTCondor pool.

## Not in this bundle yet

- The bundle has no eval set, so it has no install smoke of its own yet.
- The SubMIT instance rebuild (`task.r3-bundle-cluster-manager.instance`) needs SubMIT access and has
  not been run.
