# Operating an instance

An instance is a directory made by `archi init`. It is a git repository that
holds configuration, never data or secrets in git. The day-to-day work is
small:

| Task | Command or page |
|---|---|
| Is everything healthy? | `archi doctor`. See [Doctor](doctor.md). |
| Start or stop the stack | `archi up`, `archi down`. See [Data persistence](persistence.md). |
| Add a source | `archi add <pack>`. See [Adding a source](../adding-a-source.md). |
| See what drifted from the install | `archi diff` |
| Check that the instance answers | `archi eval`. See [archi eval](evals.md). |
| Move to a newer Archi | [Upgrade and migration](upgrade-and-migration.md) |
| Run automations | `archi ops`. See [the operations layer](../operations-layer.md). |
| Something looks wrong | [Footgun ledger](footguns.md) |

More detail on each command is in [What each command does](commands.md).
The settings files are described in [archi.yaml](../reference/archi-yaml.md).

## What lives where

- `archi.yaml`: the instance's settings.
- `archi.lock`: which bundle and pack versions were installed.
- `.env`: the generated secrets and your source credentials, readable only by you, ignored by git.
- `profiles/`, `deployments/`: the rendered profile and the OKG deployment.
- The **data root** (`~/.local/share/archi/<name>` by default): connector
  caches and repository checkouts. It is outside the instance on purpose,
  because `okg install --force` deletes everything inside a deployment.

OKG publishes only from a committed repository. Commit changes to
`archi.yaml` and `ops.yaml` before the next publish.
