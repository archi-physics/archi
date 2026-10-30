<p align="center">
  <img src="https://raw.githubusercontent.com/archi-physics/archi/main/docs/docs/_static/archi-logo.png" width="200" />
</p>

# Archi

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://archi-physics.github.io/archi/)

**Archi automates operations for scientific computing.** It serves the teams
that keep research infrastructure running: experiment computing operations,
analysis user support, data quality and cluster management. Agents watch the
system and propose actions. Once a person approves a proposal, a separate
component carries it out; in 3.0 no automation acts on its own. Every step
cites the facility's own knowledge, and its outcome is written back into that
knowledge, so the next proposal can see what happened last time.

Today Archi ships the context layer, the automation runtime and one
automation, `instance-health`, which watches the Archi instance itself. The
first automation for operations work, `ticket-triage`, is in progress.

## How it works

Automation is only as good as what the agent knows, so Archi has two layers.

1. **The context layer** holds what a facility knows: documentation, tickets,
   catalogs, monitoring, code and, for operations bundles, the record of past
   operations. Archi keeps it *solid* and checks that it is: current, linked
   across sources, cited on every read, and measured by freshness checks and
   evals. An automation can refuse to run when the knowledge it depends on is
   stale.
2. **The operations layer** runs automations over that context. Each follows
   the same loop: signal, context, proposal, decision, action, outcome. Each
   automation runs at a set level, from "only record what it would do" up to
   "act after a person approves", and anything that changes a system goes
   through review first.

The working rule is **algorithmic first, agents for judgment**: fixed checks
decide everything they can, and agents handle what needs reading and judgment.

Any agent that speaks the Model Context Protocol (MCP), the standard way AI
agents call tools, can use an Archi instance: the built-in chat, Claude Code,
Codex or your own agent. You can bring your own agent, language model,
connectors and tools; the defaults work out of the box.

## What you install

You install a **bundle**: a named selection of packs. A pack is one source
family (TWiki, Indico, Redmine, Slurm, the CMS catalogs), one set of tools, or
one automation.

| Bundle | For |
|---|---|
| `cern-team` | Any CERN team: documentation and meetings, with no credentials needed to start. |
| `cms-physics` | CMS physics: analysis notes, published summaries, physics TWiki pages and the analyses that link them. |
| `cms-comp-ops` | CMS computing operations: sites, storage, datasets, workflows, site status and past incidents. |
| `cluster-manager` | Any cluster operator, at any site: documentation, tickets, operational memory, and optionally Slurm, HTCondor, ServiceNow and node health. |

You can also install a bundle with no automations, and get a knowledge base
your team can query, with citations, from any MCP agent. Your own sources and
automations can ship as packs from your own package.

## Start here

Archi is not on PyPI yet; the [quickstart](docs/docs/quickstart.md) gives the
install command. Then:

```bash
archi init myteam && cd myteam && archi up && archi doctor
```

The quickstart takes an empty machine to a cited answer. The full documentation is published at
<https://archi-physics.github.io/archi/> and built from [`docs/docs/`](docs/docs/).

## How it is built

Archi runs on [OKG](https://github.com/mitdbg/akmon), a knowledge graph that
publishes checked versions of its data, serves them to agents through a set of
read tools, and records every review decision. (AKMON is the name of the OKG
project and repository.) Archi supplies everything above it: the connectors and schemas for each source,
the packs and bundles, the automations and their runtime, and the `archi`
command (`init`, `up`, `doctor`, `connect`, `eval`, `ops`). It is distributed
as the Python package `archi-kb` (import name `archi`), built from
[`python/`](python/).

The package holds no credentials, no site configuration and no running
services. Each instance lives in its own repository and carries only its
configuration and secrets.

## Repository layout

- [`python/`](python/): the `archi` package, with its tests.
- [`bundles/`](bundles/): the bundles and the [packs](bundles/packs/) they
  select. See [bundles and packs](docs/docs/bundles-and-packs.md).
- [`skills/`](skills/): playbooks for agents operating over the graph.
- [`docs/`](docs/): the design records ([`docs/adr/`](docs/adr/)) and the
  documentation site (`docs/mkdocs.yml`, pages in `docs/docs/`). The library
  and command reference pages are generated:
  `cd python && python -m archi.docs_gen`.
- [`pact/`](pact/): the change record (requirements, tasks, evidence).

## Key documents

- [ADR 0002](docs/adr/0002-archi-3.0-release.md): what Archi 3.0 is, and its
  milestones.
- [ADR 0001](docs/adr/0001-archi-v3-program-spec.md): the v3 program spec.
- [OKG alignment](docs/okg-alignment.md): which parts of OKG Archi depends
  on, and the OKG version it is tested against.

## v2

The v2 application (the RAG chatbot) lives on the `main` branch and serves
existing deployments until the 3.0 cutover. This branch carries 3.0 only.
See [moving from v2 to 3.0](docs/docs/migration-from-v2.md).

## License

Archi is released under the [MIT License](LICENSE). For project inquiries,
contact paus@mit.edu.
