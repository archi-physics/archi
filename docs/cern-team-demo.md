# cern-team bundle — end-to-end install demo (executed runbook)

> **Superseded. Do not install from this page.** It records one run, on
> 2026-08-21, of the cern-team layout before PACT r3-packs-split, when the
> bundle was an okg profile with `source-defaults/` and `.yaml.example` files.
> cern-team is now a bundle of packs (`bundles/cern-team/bundle.yaml`,
> `bundles/packs/`). To install it today, follow [the quickstart](docs/quickstart.md)
> (`archi init` / `archi up` / `archi doctor`); the format is in
> [packs and bundles](docs/reference/pack-format.md). The steps and results
> below are kept only as the record of that run.

The Archi v3 assembly milestone: install the `cern-team` bundle
(`bundles/cern-team/`) into a fresh instance, ingest all four
connectors, publish, and read back. Every command below was executed
on 2026-08-21 on a dev host (okg dev editable checkout, archi wheel
3.0.0a1 built from the working tree); this document is the
reproducible sequence the external-distribution conformance harness consumes. The
actual run results live in
`pact/changes/w2-consolidate-hep-sources/cern-team-demo-evidence.md`.

Steps tagged **[manual]** are manual because the okg profile-installer
contract has no channel for them yet — each is input to the okg
bundle-contract work.

Terminology: the *bundle* (`bundles/cern-team/`) installs an
*instance* (deployment dir + database) whose *connectors* ingest and
whose *playbooks* guide agents.

## 0. Prerequisites

- okg venv with the `okg` CLI (this demo:
  a uv venv, okg dev editable from a sibling
  checkout).
- The archi wheel built from this repo and installed into that venv:

  ```bash
  cd python
  <venv>/bin/python -m hatchling build -t wheel -d dist
  VIRTUAL_ENV=<venv> uv pip install --reinstall dist/archi_kb-3.0.0a1-py3-none-any.whl
  ```

- A Postgres reachable by DSN and an **empty database owned by this
  instance** (never share a database between instances):

  ```bash
  # this demo: a rootless podman container okg-w1-pg on a loopback port
  podman exec okg-w1-pg psql -U postgres -c "CREATE DATABASE okg_cern_team;"
  ```

- Environment for every command that follows:

  ```bash
  archi render --bundle cern-team --instance .          # copies the wheel's bundle to ./profiles/
  export OKG_DEPLOYMENTS_DIR=<deployments root>
  export OKG_DSN='postgresql://<user>:<pw>@127.0.0.1:<port>/okg_cern_team'
  ```

  Run okg from this directory: its profile search includes
  `./profiles/<name>/profile.yaml`, so no `OKG_PROFILES_DIR` is needed.

  **No authoring checkout is needed.** The wheel carries `bundles/` and the
  materialised playbooks, and `archi render` copies them out of it. okg's
  profile resolver has no installed-distribution discovery (its cascade is
  explicit override, `OKG_PROFILES_DIR`, `./profiles/`, the substrate's own
  library, then the fetch cache), so archi writes the profile where the
  third step looks. Discovery by entry point is an AKMON ask (ADR 0002 §9).

## 1. Scaffold the instance

```bash
okg install --profile cern-team \
  --deployment-name cern-team-demo \
  --postgres-dsn '${OKG_DSN}' \
  --non-interactive --no-publish --json
```

Notes:

- The bundle's `init_questions` become CLI flags automatically
  (`--postgres-dsn`, `--archi-data-root`); `--deployment-name` is the
  native flag. `okg install --profile` and `okg init --profile` take
  the identical flags — `install` routes profiles without their own
  installer through the same generic scaffold, so either verb works.
- Passing the literal `'${OKG_DSN}'` keeps the credential out of
  `deployment.yaml`; the manifest DSN resolver interpolates `${VAR}`
  at command time.
- `--no-publish` is required: steps 2–3 must land before the first
  catalog load, so the installer's bundled migrate+first-publish
  would fail on a fresh instance.
- Add `--dry-run` first to see the exact file plan without writing.

Expected: `files_written` lists `deployment.yaml`,
`source_registry.yaml`, `invariants.yaml`,
`schemas/cern_team_demo_base.yaml`, and 21 `skills/` files (the
installer follows the bundle's symlinks and copies contents
verbatim). The generated `deployment.yaml` already carries the
`nomos.rollout` deferred block (owner `archi-cern-team`, review_by
2026-12-01), `runtime.enabled: false`, and the `skills_dir` /
`skill_triggers` wiring — the bundle's `deployment-defaults.yaml`
injects all of it through the installer's deployment-defaults merge
(**no** manual step; the task brief listed this as potentially
manual, it is not).

## 2. [manual] Copy the archi schema slices from the wheel

The profile installer has no schema-asset channel, and the connector
subtypes (`jira_issue`, `documentation_page`, `software_repository`,
`cmssw_release`, ...) ship in the wheel, not in any substrate library
module.

```bash
SITE=$(<venv>/bin/python -c 'import archi, pathlib; print(pathlib.Path(archi.__file__).parent)')
DEP=$OKG_DEPLOYMENTS_DIR/cern-team-demo
cp "$SITE"/schemas/operations.yaml "$SITE"/schemas/sources.yaml "$DEP"/schemas/
mkdir -p "$DEP"/schemas/bridges
cp "$SITE"/schemas/bridges/sources.yaml "$DEP"/schemas/bridges/sources.yaml
```

**Copy `schemas/bridges/operations.yaml` verbatim too** — as of the
portable-bridge fix it composes on this bundle unmodified:

```bash
cp "$SITE"/schemas/bridges/operations.yaml "$DEP"/schemas/bridges/operations.yaml
```

This step used to be a hand-written prune. The wheel's operations bridge
references subtypes owned by substrate modules this bundle does not
compose (`dataset`, `ticket`, `service`, `repo`, `source_file`, ...) and
its narrowings lacked `optional_when_subtypes_missing`, so
`okg catalog load` failed with `bridge_subtype_unknown` (first failure:
narrowing `documentation_page_references_dataset`, child subtype
`dataset`). Both this bundle and the comp-ops instance therefore carried
their own, *differently* pruned copies. The shipped file now flags
exactly the narrowings whose endpoints vary by consumer, so the composer
skips those and keeps the rest strict — an accidentally dropped module
still fails loudly. Regression-tested for both consumers in
`python/tests/test_bridge_portability.py`.

(Also inherited from W1: narrowings placed outside `schemas/bridges/`
are silently ignored and only fail at ingest as
`ProducerPolicyViolation` — keep the bridges/ split.)

## 3. Materialize connector data

Everything lives under the instance's `archi_data_root` (init answer;
default `data`, deployment-dir-relative):

```text
$DEP/data/
  jira/records.json        # issue records (flat-cache or REST shapes)
  jira/meta.json           # {"record_count": N, "fetched_at": ..., "projects": [...]}
  docsite/records.json     # [{url, title, body, site_name|source_repo+path}, ...]
  twiki-snapshot/*.txt     # raw topic markup incl. %META lines
  cmssw-releases/          # created by the connector; releases.map lands here
```

The demo corpus: 4 JIRA issues (mixed flat-cache and REST shapes,
people, comments, cross-links), 3 doc pages (one repo-backed, one
citing issue CERNTEAM-201), 3 TWiki topics with `%META:TOPICINFO` /
`%META:TOPICPARENT` and a `[[CompOpsDemoTransfers][...]]` wiki link.
`cmssw_releases` needs no local data — it fetches the public cms-bot
`releases.map` live on first ingest (no auth).

TWiki snapshot layout (friction found live): `web_root` names the web
of the snapshot **root** — topic files of that web sit directly in
`twiki-snapshot/`, and subdirectories are subwebs. Putting topics
under a `CMS/` subdir while `web_root: CMS` double-prefixes every
page id (`twiki:CMS:CMS:<Topic>`).

## 4. Bring the instance up

```bash
# Anchor relative connector paths (belt and braces: also run the
# ingest from the deployment dir — twiki's eos_root resolves against
# the CWD, not ARCHI_DATA_ROOT).
export ARCHI_DATA_ROOT=$DEP
cd $DEP
```

Fresh database on the bundled okg Postgres stack? Install the
required extensions first — the stack only installs them into its
initial database, and a bare `okg migrate` fails with
`extension_missing` (friction; the installer's own first-publish path
has provision-parity but the standalone migrate does not):

```bash
psql "$OKG_DSN" \
  -c "CREATE EXTENSION IF NOT EXISTS btree_gist" \
  -c "CREATE EXTENSION IF NOT EXISTS fuzzystrmatch" \
  -c "CREATE EXTENSION IF NOT EXISTS pg_textsearch" \
  -c "CREATE EXTENSION IF NOT EXISTS pg_trgm" \
  -c "CREATE EXTENSION IF NOT EXISTS timescaledb" \
  -c "CREATE EXTENSION IF NOT EXISTS vector"
```

```bash
okg migrate --deployment cern-team-demo --apply --json
# expected: statements_executed ~729, multiphase 0020 applied,
# idempotent_second_run true
```

**[manual] Role passwords** (known friction): migrate creates the
loopback roles without passwords on password-authenticated servers;
`okg catalog load` fails on them until set (values are the fixed
loopback-only passwords from okg `local_database_roles.py`):

```bash
psql "$OKG_DSN" -c "ALTER ROLE okg_mcp PASSWORD 'okg_mcp'" \
                -c "ALTER ROLE app_rw  PASSWORD 'okg_rw'" \
                -c "ALTER ROLE app_ro  PASSWORD 'okg_ro'"
```

```bash
# Claim AFTER schemas/ + source_registry.yaml are final: the claim
# records the manifest identity, and a later edit makes the next
# publish refuse with catalog_ownership_mismatch (recovery: re-run
# this claim).
okg catalog ownership claim --deployment "$DEP" --json
# expected: claimed true (status latest_incompatible on a fresh DB is
# fine — it names the migration's empty catalog v1)

okg catalog load --deployment "$DEP" --apply --json
# expected: applied true, 35 subtypes, 100 narrowings, bridge
# narrowings from both bridges/operations.yaml (1) and
# bridges/sources.yaml (11)

okg deployment lint cern-team-demo --json
# expected: ok true; only two "skipped" informational findings
# (record-template routes not applicable; live checks skipped) —
# zero blocker, zero warning

okg ingest --deployment cern-team-demo --progress --json
# expected: completeness complete; all four connectors OK
#   cmssw_releases +314 nodes +270 edges   (live releases.map fetch)
#   docsite        +7 nodes   +5 edges
#   jira           +11 nodes  +15 edges
#   twiki_eos      +6 nodes   +6 edges
# and a published generation in publishes[label=commit]
```

Idempotence check: run the same `okg ingest` again — identical
write set, 0 retractions, a new published generation.

## 5. Read-back proof

```bash
okg status --deployment cern-team-demo --json
# live_subtypes: cmssw_release 314, document_chunk 10,
# documentation_page 6, jira_issue 4, person 3, software_repository 1
# publishing.latest_published_status: published

okg search --deployment cern-team-demo --query "kronos transfer queue stalls"
# 7 hits spanning all three text connectors at the pinned generation:
# jira:CERNTEAM-201 (jira), the CompOpsDemoTransfers/Home chunks
# (twiki_eos), the Team Ops Guide page + chunk (docsite)

okg trace node jira:CERNTEAM-201 --deployment cern-team-demo
# shows the cross-connector edge: incoming `references` from
# chunk:<docsite ops chunk> with source cern-team-demo.docsite

okg trace node "twiki:CMS:CompOpsDemoTransfers" --deployment cern-team-demo
# shows the topic-to-topic edge: `references` -> twiki:CMS:CompOpsDemoHome

okg search --deployment cern-team-demo --query "CMSSW_15_0_15"
# live-fetched catalog hits; trace a release to see supersedes edges
```

Note on bare SQL: `psql` reads of `okg.v_nodes`/`v_edges` are refused
by design (`okg graph read requires a Chronos checkout`); use
`okg search` / `okg trace` / `okg status` as above.

## Friction log (chronological, each recorded for the okg bundle contract / substrate)

1. **`extension_missing` on standalone migrate** against a fresh
   database on the bundled Postgres stack (extensions exist only in
   the stack's initial DB). Fixed by CREATE EXTENSION (step 4). The
   installer's own bundled first-publish path ensures extensions;
   the standalone `okg migrate` does not.
2. **`bridge_subtype_unknown` on catalog load** when the wheel's
   `schemas/bridges/operations.yaml` is copied verbatim into the
   minimal module trio (first: `documentation_page_references_dataset`
   → `dataset`). Resolved with the pruned bridge (step 2). Ask:
   packaged bridges should flag cross-module narrowings
   `optional_when_subtypes_missing` (as `bridges/sources.yaml`
   already does for the MONIT narrowings) or ship split per family.
3. **`deployment.source_registry.profile_invalid` lint blocker** for
   `cmssw_releases`: the `reference_catalog` source profile admits
   only `record_identity_kind: domain_key`, but the
   `archi/sources/cmssw.py` docstring template says `remote_id` (the
   W1 registry already used `domain_key`). Bundle ships `domain_key`;
   the docstring template needs the one-word fix.
4. **`catalog_ownership_mismatch` publish refusal** after editing
   `source_registry.yaml` between `okg catalog ownership claim` and
   `okg ingest` (drifted_blocks=[source_registry]). By design;
   recovery is re-running the claim. Runbook now orders the claim
   after all manifest edits.
5. **TWiki snapshot layout**: `web_root` names the snapshot root's
   web; a `<web_root>/` subdir inside the root double-prefixes page
   ids (`twiki:CMS:CMS:...`). Documented in the bundle
   source-default; a loud warning from the connector when the first
   path segment equals `web_root` would catch it earlier.
6. **Stale records are not retracted** after their identity leaves a
   completed scope: with
   `deletion_semantics: missing_from_completed_scope`, renaming the
   twiki snapshot files left the old `twiki:CMS:CMS:*` facts live
   through plain re-ingest, `--mode reconcile`, and
   `--mode reconcile --reset-cursor` (retract counters 0 each time).
   The demo instance was rebuilt on a fresh database instead.
   Substrate ask: scope-diff retraction, or a loud "N records left
   the completed scope but deletion is unimplemented" health warning.
7. **Cosmetic**: every ingest prints `deployment declares no
   sync_root` — noise for cache-backed connectors resolved via
   ARCHI_DATA_ROOT.

Resolved contract questions worth keeping: the profile installer
follows symlinked `skills/` entries (the bundle symlinks the repo's
`skills/` — no duplication); `deployment-defaults.yaml` can inject
arbitrary non-reserved manifest keys, which covers the
`nomos.rollout` block, `runtime`, and the skills wiring; `${VAR}`
DSN references survive scaffold and resolve at command time.
