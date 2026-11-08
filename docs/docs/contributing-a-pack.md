# Contributing a pack from your own package

An external group can ship packs, bundles and connectors as an ordinary Python
package. Nothing lands in the archi repository. ADR 0002 section 4.4 is the
design; [Pack and bundle format](reference/pack-format.md) is the reference for the
manifest files. This page says what your package must contain and how to test it.

## What the package declares

1. **Entry points.** One per directory you ship:

    ```toml
    [project.entry-points."archi.packs"]
    mywg = "mywg_packs:packs_dir"        # returns a directory of <pack>/pack.yaml
    [project.entry-points."archi.bundles"]
    mywg = "mywg_packs:bundles_dir"      # optional: a directory of <bundle>/bundle.yaml
    ```

    The value is a callable returning the directory (or the directory itself).
    The entry-point name (`mywg`) is the provider label users see.

2. **Pack manifests**, one `packs/<name>/pack.yaml` each, valid against the
   published schema (`archi bundles validate <dir>`). Package the directories as
   data files in your wheel. Rules that bite external packs:
    - pack names are global: two providers with the same name are both reported
      as a problem, so prefix yours (`mywg-...`);
    - a pack that needs a credential sets `needs_credentials: true` and names
      environment variables in `credentials`; never a value;
    - every context-pack source has a `freshness` entry;
    - automation packs cannot ask for a level above L2.

3. **Connectors.** A connector is the code that pulls one external system into the
   graph. It is an okg `SourceAdapter` (okg's mechanism name), referenced from a
   source template in the pack by `module:` and `class:`:

    ```yaml
    mywg-tickets:
      module: mywg_connectors.tickets
      class: TicketSource
      ownership_id: ${deployment_name}.mywg-tickets
      admission_policy: { ... }        # producer authority, output_signature, as okg requires
    ```

    Your package must be importable in the environment where `okg` runs, so list
    okg as an install-time requirement of the deployment, not of your wheel (okg is
    the host environment; archi does not declare it either). The adapter contract
    (change probes, producer authority, admission mode, record identity) is okg's,
    and [okg alignment](https://github.com/archi-physics/archi/blob/archi_v3/docs/okg-alignment.md) lists the interface archi relies on.

## Versioning

- Each pack and bundle has its own `version` (`MAJOR.MINOR.PATCH`). Raise MAJOR
  when a source's output signature, record identity or a question id changes: an
  instance's `archi.lock` records the selection and `archi diff` reports drift.
- Each bundle states the okg range it supports in `okg:` (PEP 440, for example
  `">=0.6.0,<0.7"`). A pack inherits the check through the bundles that use it.
- The wheel's version is yours. Pin the range of `archi-kb` you were tested
  against in your own `dependencies`.

## What users see

With your wheel installed in the same environment as archi:

```console
$ archi bundles list
...
  mywg-tickets 0.1.0 (context, needs credentials)  [mywg (mywg-packs 0.1.0)]
```

The bracket is the provider: the entry-point name and the distribution and
version that registered it. `archi bundles list --json` has it as `provider`.
Use `archi render --bundle B --with mywg-tickets --instance DIR` or
`archi add mywg-tickets --instance DIR` to install it.

## How to test it

1. `archi bundles validate path/to/packs` must report no problem.
2. `archi bundles list --json` must show your packs with your provider label and
   an empty `problems` list. A broken entry point is listed, not fatal.
3. `archi render --bundle <your bundle> --instance <tmp>` then
   `okg install --profile <name> --dry-run --json --non-interactive ...`
   must report `dry_run_ok` with no blockers.
4. Publish against a real graph database in your own CI, the way
   `archi init` does.

archi's own copy of steps 2 and 3 runs against a fixture package,
`python/tests/bundles/fixtures/thirdparty/archi_fixture_packs/`, built into a wheel
in `python/tests/bundles/test_packaging.py`. Copy that package as a starting point.

## Bringing an MCP server instead

If what you have is a live tool and not ingestion, you do not need a pack:
declare it under `mcp_servers:` in the instance's `archi.yaml` and
`archi connect` registers it next to the instance server
([Connecting other agents](connecting-an-agent.md)).
