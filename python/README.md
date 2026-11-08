# archi-kb

**Archi automates operations for scientific computing.** It builds a
knowledge graph from what a facility already writes down, such as
documentation, meetings, tickets, wikis, catalogs, monitoring and code. Agents
answer from that graph and cite the records they used. Automations read the
same graph and propose actions. A person approves each action, and only then
does the automation runtime, a separate process, carry it out. No agent holds
a credential that can change an external system.

This package is Archi. You import it as `archi`. Do not install `archi` from
PyPI: that name belongs to an unrelated package.

## Install

You need Python 3.12, `docker` or `podman` (with `compose`), and `git`.
Archi runs on the `okg` graph engine from AKMON (PyPI package `akmon`), which pip
installs with it. In a new virtual environment (an environment that still has
the old `okg` package needs `pip uninstall okg` first):

```bash
pip install archi-kb==3.0.2
```

## First instance

```bash
archi init myteam --bundle cern-team   # creates ./myteam and publishes a first graph
cd myteam
archi up                               # starts the graph, the MCP server and the chat
archi doctor                           # checks the instance; each failure names its fix
```

`archi up` prints the chat address, the admin login and the MCP endpoint.
MCP (Model Context Protocol) is how agents call the graph's tools.

- **Chat** needs a language model. `archi up` uses an ollama running on the
  same host, or the one you name with `--ollama-url`. You can also add a model
  under Settings > Admin > Connections in the chat.
- **Agents** such as Claude Code bring their own model. In the instance
  directory, run `archi connect claude-code --write`, then export
  `OKG_CHAT_MCP_TOKEN` from the instance's `.env` in the shell that starts
  the agent.

## Bundles

| Bundle | For |
|---|---|
| `cern-team` | Any CERN team: documentation and meetings, with no credentials needed to start. |
| `cms-physics` | CMS physics: `cern-team` plus analysis notes, CDS records and physics TWiki pages. Needs CERN credentials. |
| `cms-comp-ops` | CMS computing operations: `cern-team` plus sites, storage, datasets, workflows and past incidents. Needs CERN credentials. |
| `cluster-manager` | Any cluster operator: documentation, tickets, and optionally Slurm, HTCondor and ServiceNow. |

Automations are optional packs. For example, `--with ticket-triage` adds
cited ticket triage to `cms-comp-ops` or `cluster-manager`. They run in an
`okg runtime worker` that you start yourself, as the documentation describes.

## Documentation

- [Quickstart](https://github.com/archi-physics/archi/blob/v3.0.2/docs/docs/quickstart.md)
- [Documentation](https://github.com/archi-physics/archi/tree/v3.0.2/docs/docs)
- [Release notes for 3.0.2](https://github.com/archi-physics/archi/blob/v3.0.2/docs/release-notes-3.0.2.md)
- [Moving from Archi v2](https://github.com/archi-physics/archi/blob/v3.0.2/docs/docs/migration-from-v2.md)

Archi is released under the MIT License.
