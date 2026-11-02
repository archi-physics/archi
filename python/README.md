# archi-kb

**Archi automates operations for scientific computing.** It builds a
knowledge graph from what a facility already writes down, such as
documentation, meetings, tickets, wikis, catalogs, monitoring and code. Agents
answer from that graph and cite the records they used. Automations read the
same graph and propose actions. A person approves each action, and a separate
component carries it out. No agent holds a credential that can change an
external system.

This package is Archi. You import it as `archi`. Do not install `archi` from
PyPI: that name belongs to an unrelated package.

## Install

Archi runs on the `okg` graph engine, which is not on PyPI. Install it from
its release file, in a Python 3.12 virtual environment, with `docker` or
`podman` (with `compose`) and `git`:

```bash
pip install "okg @ https://github.com/mitdbg/akmon/releases/download/v0.2.0/okg-0.2.0-py3-none-any.whl#sha256=72bac3b67cf29cccd27e77e58ad309104f19fcce255a04a4a6f23b8318b88e12" archi-kb==3.0.0
```

## First instance

```bash
archi init myteam --bundle cern-team   # creates ./myteam and publishes a first graph
cd myteam
archi up                               # starts the graph, the MCP server and the chat
archi doctor                           # checks the instance; each failure names its fix
```

`archi up` prints the chat address and the MCP endpoint. The chat needs a
language model: point it at a local ollama, or add one in the chat's admin
settings. Agents such as Claude Code connect over MCP with
`archi connect claude-code --write`, and need no model from Archi.

## Bundles

| Bundle | For |
|---|---|
| `cern-team` | Any CERN team: documentation and meetings, with no credentials needed to start. |
| `cms-physics` | CMS physics: analysis notes, published summaries, physics TWiki pages. |
| `cms-comp-ops` | CMS computing operations: sites, storage, datasets, workflows and past incidents. |
| `cluster-manager` | Any cluster operator: documentation, tickets, and optionally Slurm, HTCondor and ServiceNow. |

## Documentation

- [Quickstart](https://github.com/archi-physics/archi/blob/v3.0.0/docs/docs/quickstart.md)
- [Documentation](https://github.com/archi-physics/archi/tree/v3.0.0/docs/docs)
- [Release notes for 3.0.0](https://github.com/archi-physics/archi/blob/v3.0.0/docs/release-notes-3.0.0.md)
- [Moving from Archi v2](https://github.com/archi-physics/archi/blob/v3.0.0/docs/docs/migration-from-v2.md)

Archi is released under the MIT License.
