# What Archi is

Archi automates operations for scientific computing: experiment computing
operations, analysis user support, data quality and cluster management.
Agents watch the system and propose actions. Once a person approves a
proposal, a separate component carries it out; in 3.0 no automation acts on
its own. Every step is grounded in your facility's own knowledge, and its
result is written back into that knowledge.

Today Archi ships the context layer, the automation runtime and one
automation, `instance-health`, which watches the Archi instance itself. The
first automation for operations work, `ticket-triage`, is in progress.

Archi is built on OKG, a knowledge graph with a publish step, a set of read
tools and a review ledger. You do not need to know OKG to start. This site
explains each OKG idea where Archi uses it.

!!! note
    This is the 3.0 documentation. The earlier chatbot is documented under
    [v2 (legacy)](v2/index.md). See [Moving from v2 to 3.0](migration-from-v2.md).

## Two layers

Automation is only as good as what the agent knows. So Archi has two layers.

1. **The [context layer](context-layer.md)** holds what your facility knows:
   documentation, tickets, catalogs, monitoring, code and the record of past
   operations. It has to be solid. That means current, complete for the
   questions automations depend on, linked across sources, cited on every
   read, and measured.
2. **The [operations layer](operations-layer.md)** runs automations over that
   context. Each one follows the same loop: signal, context, proposal,
   decision, action, outcome. Actions run under explicit autonomy levels, up
   to L2 in 3.0. Outcomes go back into the context layer, so the next triage
   can see what happened last time.

The working rule is: **algorithmic first, agents for judgment.** Fixed checks
(invariants, freshness, thresholds, evals) decide everything they can. Agents
handle what needs reading and judgment.

## What you install

You install a [bundle](bundles-and-packs.md): a named selection of packs. A
pack is the smallest installable piece, such as one source family or one
automation. Four bundles ship today:

| Bundle | For |
|---|---|
| `cern-team` | Any CERN team: documentation and meetings, with no credentials needed to start. |
| `cms-physics` | CMS physics: analysis notes, published summaries, physics TWiki pages and the analyses that link them. |
| `cms-comp-ops` | CMS computing operations: sites, storage, datasets, workflows, site status and past incidents. |
| `cluster-manager` | Any cluster operator, at any site: documentation, tickets, operational memory, and optionally Slurm, HTCondor, ServiceNow and node health. |

The [library](library/index.md) lists every bundle and pack with what it
reads. Your own sources and automations can ship as packs from your own
package. Start with the [quickstart](quickstart.md): it takes an empty machine
to a cited answer with the `cern-team` bundle.

## Parts you can replace

Every component can be swapped. Archi is the glue, and its defaults work out
of the box.

| Part | Default in 3.0 | How you replace it |
|---|---|---|
| Knowledge graph | OKG (AKMON) | Not replaceable in 3.0. The MCP tool surface (`search`, `expand`, `inspect`, `filter`, `map`, `aggregate`) is the contract another store would have to meet. |
| Ingestion | Archi connectors | Any pip package that registers packs through the `archi.packs` entry point. See [Contributing a pack](contributing-a-pack.md). |
| Tools | The instance's OKG MCP server plus Archi live tools | Any MCP server declared in the instance. See [Connecting an agent](connecting-an-agent.md). |
| Automation runtime | OKG deployment workflows | An external runner using the proposal API. This waits on an upstream OKG change. |
| Agent and chat | Open WebUI, built in | Anything that speaks MCP: Claude Code, codex, openclaw, your own. |
| Language model | Whatever the agent is configured with | Agent-side configuration. Archi never stores it. |

## Who it is for

First, operations teams: CMS computing operations, CRAB support, WisDQM and
cluster managers at any site. Second, a team that only wants the context
layer and an agent over it in an afternoon. That team installs the same
bundles with no automation selected.

The design is in [ADR 0002](reference/adr.md).
