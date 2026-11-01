# Quickstart

From an empty machine to a cited answer from your team's knowledge graph,
with the `cern-team` bundle: five user commands, the same ones the nightly
build runs (the `cd` is not counted, and the fifth is a check).

**You need:** `docker` or `podman` (with `compose`), Python 3.12 (CI tests 3.12; okg 0.2.0 does not install on 3.14),
and `git`. No account, key or root access.

With rootless podman, container health checks run from systemd user timers.
Where there is no systemd user session (inside a container, or on some
shared hosts), the checks may never run, and then `archi up` waits with no time limit.
Use docker there, or log in to a session with a systemd user manager.

```bash
pip install "okg @ https://github.com/mitdbg/akmon/releases/download/v0.2.0/okg-0.2.0-py3-none-any.whl" "archi-kb @ git+https://github.com/archi-physics/archi@archi_v3#subdirectory=python"
archi init myteam --bundle cern-team
cd myteam
archi up
archi doctor --json
python "$ARCHI_CHECKOUT/python/tests/cli/mcp_cited_answer.py" "What is the Knowledge Bases and Agents for CERN workshop?"
```

What each command does:

1. **Install.** `okg` is the knowledge-graph engine. `archi-kb` is Archi. The
   import name is `archi`. `archi-kb` is not on PyPI yet, so it comes from
   git. Do not `pip install archi`: that name on PyPI is an unrelated
   package. Use a virtual environment.
2. **`archi init myteam --bundle cern-team`** creates `./myteam` as a git
   repository. It writes the settings, generates every secret into `.env`,
   starts the graph database and publishes a first graph. The default packs
   need no credential. (`cd myteam` then moves into it.)
3. **`archi up`** starts the stack: graph database, chat database, MCP server
   and Open WebUI. It prints the chat address and the MCP endpoint.
4. **`archi doctor --json`** runs every check and prints one JSON object.
   Each failure carries a fix line. Leave out `--json` for plain lines.
5. **The last command asks a question over MCP** and succeeds only when the
   answer cites graph records. The script is in the Archi repository, so set
   `ARCHI_CHECKOUT` to a clone of it. It needs the `mcp` Python package. In
   daily use you ask in the chat instead, or
   [connect an agent](connecting-an-agent.md).

## Check the result

- `archi doctor` exits 0 when no check failed.
- The last command prints the question, the record ids it cited and the
  source that wrote them.
- The nightly build runs these same commands on an empty runner and reports
  how long the whole path took (the target is 15 minutes). A test keeps this page and
  that build in step.

## Where next

- [What each command does](operating/commands.md), including a local model
  and a remote host.
- [Connect an agent](connecting-an-agent.md) such as Claude Code.
- [Add a source](adding-a-source.md).
- [Operate the instance](operating/index.md).
