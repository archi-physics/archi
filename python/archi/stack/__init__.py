"""One instance, one compose project (ADR 0002 section 8).

`archi init` writes an instance directory; `archi up` / `archi down` run its
compose project; `archi doctor` checks it. This package holds what those
commands share:

- `instance`: the instance directory, `archi.yaml` and `.env`;
- `compose`: the compose file, the MCP image, and the container runtime;
- `okg`: running okg commands with the instance's environment;
- `webui`: the two Open WebUI calls okg does not make for a compose-run site;
- `checks`: one check per absorbed quickstart trap.

archi never reimplements an okg step (ADR 0002 D7). Each place where it
covers an okg gap names the AKMON ask that would remove it.
"""
