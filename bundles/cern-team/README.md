# cern-team bundle

A generic CERN team knowledge instance (ADR 0002 section 5). The bundle is
one manifest, [`bundle.yaml`](bundle.yaml): a selection of packs from
[`bundles/packs/`](../packs/). Each pack carries its own source templates,
schema slices, coverage invariant, playbooks and chat-prompt fragment, so a
pack you leave out leaves nothing behind in the instance.

| Packs | |
|---|---|
| default | `docsite` (documentation sites, from the static-docs cache), `indico` (Indico events, read live), `core-playbooks` (the generic retrieval playbooks) |
| optional | `twiki` (an EOS dump), `cds` (CERN Document Server over OAI), `jira` (the JIRA cache), `git-code` (one Git repository's files and history), `code-text` (function bodies), `code-symbols` (functions, classes, calls) |

`cmssw-releases` is no longer part of cern-team: it went to the CMS bundles.
Any pack, this one included, can still be added with `--with`.

## Install

```bash
archi render --bundle cern-team --with git-code --instance .   # ./profiles/cern-team
okg install --profile cern-team --deployment-name <slug> \
  --postgres-dsn "$OKG_DSN" --archi-data-root "$PWD/archi-data/data" \
  --indico-event-ids '["<event id>"]' \
  --git-code-repo-name <name> --git-code-repo-url <git url>
```

Each pack's questions become `--<pack>-<question>` flags. Then commit the
deployment directory, `okg ingest` and `okg run --once --apply`. The
[quickstart](../../docs/docs/quickstart.md) is the supported install path; the older
[cern-team quickstart](../../docs/cern-team-quickstart.md) (superseded, kept as
reference) shows these manual steps in full.
`archi add <pack>` adds a pack to an installed instance.

## A bare install publishes

`archi init <name> --bundle cern-team` needs no answer and no credential.
The two default sources carry their own input:

- **docsite** declares a `prepare` step (see
  [packs and bundles](../../docs/docs/reference/pack-format.md#preparing-a-packs-input)).
  `archi init` and `archi add` run it before the first ingest: it downloads
  the pages of the `docsite_urls` answer into the records cache. The default is
  two small public CERN documentation sites, the FTS3 and Rucio docs, one
  record each (links are not followed).
- **indico** reads one public event, 1716863 ("Knowledge Bases and Agents for
  CERN"), live, when no event or category is given. It is a single named event, not a bounded walk of a
  category, so the run claims a complete scope of exactly that event, and an
  event that cannot be read fails the run rather than retracting anything.

Answer `docsite_urls`, `indico_event_ids` or `indico_category_ids` to read
your own. Any Indico event or category id you give replaces the default
event; it is read only when you give neither. `cmssw-releases` is not part of this bundle; add it with
`--with cmssw-releases` (existing instances: the
[migration note](../../docs/docs/operating/upgrade-and-migration.md)).

When you install by hand with `okg install` instead of `archi init`, no
`prepare` step runs: select only what you can feed, because every selected
source must be able to read its input or the publish stops for all of them.
docsite then needs its records cache (`python -m archi.downloaders.static_docs`),
indico an event or category, twiki a dump, jira its cache, git-code a
repository URL.

No file here or in the packs carries an operator path or a credential value:
paths are `${archi_data_root}`-relative or install answers, and credentials
are environment-variable names.

## Composition

`archi render` merges the selected packs into one okg profile (the rules are
in [packs and bundles](../../docs/docs/reference/pack-format.md)) and writes
`archi.lock` beside it. The shared HEP vocabulary
(`python/archi/schemas/`) ships in every pack that needs it as identical
files, so composing packs never conflicts on a schema.
