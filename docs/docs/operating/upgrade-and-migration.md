# Migrating an existing cern-team instance

For anyone who already runs an instance made from the `cern-team` bundle. A
new install needs none of this: `archi init myteam` now publishes with no
answers (see the [quickstart](../quickstart.md)).

## What changed

| Before | Now |
|---|---|
| The only default source was `cmssw_releases` (the public CMSSW release catalog). | The default packs are `docsite`, `indico` and `core-playbooks`. `cmssw-releases` is not in the bundle. |
| A bare `archi init` of the pack-based bundle stopped: docsite had no records cache and indico had no event. The quickstart said `--without docsite,indico --with cmssw-releases`. | docsite downloads two small public documentation sites (FTS3 and Rucio) before the first ingest. indico reads one public event (1716863). Both are credential-free, and the workaround is gone. |
| An indico source with no event or category was refused when built. | With no event or category answered, indico reads event 1716863 (`default_event_ids`); before, that was an error. A new pack question, `docsite_urls`, names the pages docsite downloads. |
| Packs `docsite` and `indico` at 0.1.0, bundle at 0.2.0. | docsite 0.2.0, indico 0.2.0, bundle 0.3.0. |

Nothing about an instance that is already published changes by itself. The
bundle only matters when you render, add a pack, or install again.

## Which instance do you have?

Look for `archi.lock` in the instance directory.

**It has an `archi.lock`** (made by `archi init` or `archi render`). Its
registry source ids look like `docsite.docsite`, `indico.indico_live`,
`cmssw-releases.cmssw_releases`, and `archi diff` works.

**It has none** (installed with `okg install --profile cern-team` from the
older single-file profile). Its registry ids are bare names (`cmssw_releases`,
`docsite`, `indico_live`, `twiki_eos`, ...). See
[Instances without an archi.lock](#instances-without-an-archilock).

## Instances created before the graph-volume fix

Before this change the graph database volume was mounted at a path the AKMON
image does not use (the image keeps its data in `/home/postgres/pgdata/data`).
The graph lived only in the container, so `archi down` followed by `archi up`,
or any container recreation (podman-compose recreates on every `up`), lost it.
After you upgrade, the next `archi up` recreates the container on the corrected
volume, which starts empty. Re-init the instance (`archi down --purge --yes`,
remove the directory, `archi init` again) or re-run `okg install` and
`okg provision`. From then on the graph survives `down` and `up`.

## Instances with an archi.lock

### What changes in the registry

Nothing, until you act. The registry and the lock record what you selected,
so an instance that selected `cmssw-releases` keeps it, and one that dropped
docsite or indico keeps them dropped. What you will see:

- `archi diff` reports `pack docsite: version 0.1.0 locked, 0.2.0 installed`
  (and the same for `indico` and the `cern-team` bundle). That is the new
  pack text, not a change to your deployment. Your registry entries do not
  differ, so the `deployment` section stays clean.
- A source id is `<pack>.<source>`. The ownership ids okg records are the
  same as before (`<deployment>.cmssw-releases`, `<deployment>.docsite`,
  `<deployment>.indico-live`), so the records stay owned by the same source.

### Keeping cmssw-releases

If the instance was made with `--with cmssw-releases` (the old quickstart
command), it is already in `archi.lock`, in the registry, and in the graph.
Do nothing.

If it was not, and you want the CMSSW releases:

```bash
archi add cmssw-releases --instance ./myteam
```

This adds the `cmssw-releases.cmssw_releases` source to the deployment,
commits the change, claims ownership, loads the catalog, ingests and
publishes. Answer its question with
`--answer cmssw_releases_releases_map_url=<url>` only if you read the release
map from somewhere other than the public cms-bot one.

### Taking up the new defaults

If the instance has no docsite or indico (it was made with
`--without docsite,indico`), add them when you want them:

```bash
archi add docsite --instance ./myteam
archi add indico  --instance ./myteam
```

`archi add docsite` downloads the pages of `docsite_urls` into
`<data root>/docsite/records.json` first. If a page cannot be fetched it stops
before it edits the deployment. Give your own pages with
`--answer docsite_urls=<url>[,<url>...]`. `archi add <pack> --no-okg` skips the
prepare step (it runs no okg command either), so the cache is not written;
run `python -m archi.downloaders.static_docs` yourself before the first ingest. `archi add indico` reads
event 1716863 unless you answer `--answer indico_event_ids='["<id>"]'` or
`--answer indico_category_ids='["<id>"]'`; either replaces the default.

An instance that already has docsite or indico is not touched: `archi add`
refuses a pack that is already installed, and the new default event applies
only to new installs. To read more events, edit `source_registry.yaml` as
before.

### Check

```bash
archi diff      # `upstream` lines for the new pack versions are expected; deployment: clean
archi doctor
```

To accept the new pack versions in the lock, run `archi render --bundle
cern-team ...` again with the same `--with` and `--without` choices (they are
in `archi.lock` under `selection`) and commit.

## Instances without an archi.lock

These were installed by `okg install --profile cern-team`, before the bundle
was split into packs. `archi add` and `archi diff` need the lock, so they do
not work on these instances, and there is no automatic in-place conversion.
Two routes:

1. **Leave it.** The deployment is a self-contained copy of the old profile.
   It keeps its `cmssw_releases` source and everything else it had, and keeps
   publishing. Nothing in this change reaches it.
2. **Rebuild on the new layout.** Make a new instance and a new database, and
   retire the old one when you are satisfied:

   ```bash
   archi init myteam2 --with cmssw-releases            # keeps the CMSSW releases
   archi init myteam2 --without docsite,indico --with cmssw-releases   # exactly the old default
   ```

   Records in the old database stay in the old database. Removing a source
   from a deployment does not remove its records from a database that already
   holds them, so do not reuse the old database for the new instance. The
   registry ids differ (`cmssw_releases` becomes `cmssw-releases.cmssw_releases`;
   the other bare ids gain their pack name the same way), and so do some
   question names: pack questions are `<pack>_<id>`, for example
   `git_code_repo_url`, where the old profile asked `github_repo_url` and
   `gitlab_repo_url`.

Renaming the registry keys of a pre-lock deployment in place is not
supported and untested: okg's ownership claim and published generations were
made under the old source ids.
