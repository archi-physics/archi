# Data persistence

What survives which command.

| You run | The graph | Chat data | Secrets and settings |
|---|---|---|---|
| `archi down` then `archi up` | Kept. The published generation is the same. | Kept | Kept |
| `archi down --purge --yes` | Deleted | Deleted | Kept |
| Delete the instance folder | Kept in the database volume. OKG keeps its ownership claim there, so this is not a reset. | Kept | Gone |

`archi down` stops the stack and keeps every volume. `archi down --purge
--yes` also deletes the graph, chat and web volumes. That is the reset.

The nightly build checks the first row. After the five quickstart commands
it runs `archi down` and `archi up`, then compares the published node count
reported by `archi doctor --json` before and after. The two must match.

Three places hold data. Back up the ones you care about.

- The **volumes** of the compose project `archi-<name>`: the graph and chat
  databases.
- The **instance directory**: configuration, in git. Secrets in `.env` are
  not in git. Keep a copy somewhere safe.
- The **data root**: connector caches and repository checkouts. These can
  be rebuilt by running the connectors again, which may take time.

An instance created before 2026-10-26 kept its graph only inside the
container. After you upgrade Archi, re-run `archi init` for it rather than
`archi up`. See [Upgrade and migration](upgrade-and-migration.md).
