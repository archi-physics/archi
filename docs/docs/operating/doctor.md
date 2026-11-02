# archi doctor

```bash
archi doctor          # one line per check, a fix line for each failure
archi doctor --json   # one object, for scripts and agents
```

`archi doctor` runs these in order:

1. `okg deployment lint`: the deployment's files are valid.
2. `okg doctor`: OKG's own health checks.
3. `source-freshness`: each source's age against the `max_staleness` its
   pack declares. See [the context layer](../context-layer.md#freshness).
4. `automation-names`: every name under `automations:` in `archi.yaml` is an
   automation pack the instance selected.
5. One check for each trap Archi absorbs from the old long quickstart, such
   as an instance that is not a git repository, a secret committed by
   mistake, a relative path, a port that is not on loopback, or an MCP
   server that does not answer. Each has an id (`instance-git`,
   `secrets-not-committed`, `absolute-paths`, `ports-loopback`,
   `mcp-answers`, `published`, and others). The
   [footgun ledger](footguns.md) maps each id to the trap it covers.

The exit code is 0 when no check failed. A skipped check says why, for
example `source-freshness` is skipped for an instance with no `archi.lock`.

The JSON object has `ok`, `instance` and a `checks` list. Each check has an
`id`, a `status` (`pass`, `fail`, `warn` or `skip`), a `message` and a
`remediation`. The `published` check reports the node count of the live
generation, which is how the nightly build proves the graph survives a
restart.
