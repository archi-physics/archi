# cern-team bundle install demo — run evidence (assembly milestone)

**When/where:** 2026-08-21, submit82. okg dev editable checkout at
`/home/submit/lavezzo/archi/archi_v3/okg` (venv
`/work/submit/lavezzo/okg-venv`); archi wheel `3.0.0a1` rebuilt from
branch `w2-twiki` (HEAD `35e5a621`) and reinstalled into the venv.
Postgres: rootless podman `okg-w1-pg` on 127.0.0.1:5455, **new**
database `okg_cern_team` (the w1-scratch deployment and its
`okg_w1_scratch` database were not touched). Deployments root
`/work/submit/lavezzo/okg-scratch`; instance dir
`/work/submit/lavezzo/okg-scratch/cern-team-demo`.

**What ran:** the complete runbook in `docs/cern-team-demo.md` —
`okg install --profile cern-team` (bundle at `bundles/cern-team/`,
resolved via `OKG_PROFILES_DIR=<repo>/bundles`) → wheel schema-slice
copy with pruned operations bridge → demo corpus (4 JIRA issues, 3
doc pages with one issue citation, 3 TWiki topics with a wiki link) →
extensions + `okg migrate` → role passwords → ownership claim →
`okg catalog load --apply` (catalog v2, 35 subtypes, 100 narrowings)
→ `okg deployment lint` → `okg ingest` twice → read-back. The final
sequence was executed end-to-end on a clean database after the
iteration frictions below were resolved.

## Lint

`okg deployment lint cern-team-demo --json`: **ok: true** — zero
blocker, zero warning; the only findings are the two informational
"skipped" notes every static lint emits (record-template routes not
applicable; live checks skipped).

## Ingest (first publish on the clean database)

| connector      | status | nodes | edges | retract | health | notes |
|----------------|--------|------:|------:|--------:|--------|-------|
| cmssw_releases | OK     |   314 |   270 |       0 | ok     | live cms-bot releases.map fetch (498 KB cached to `data/cmssw-releases/releases.map`), limit 300 releases + family nodes, supersedes chains |
| docsite        | OK     |     7 |     5 |       0 | ok     | 3 pages + 3 chunks + 1 software_repository; includes the cross-connector `references` edge to jira:CERNTEAM-201 |
| jira           | OK     |    11 |    15 |       0 | ok     | 4 issues + 3 persons + 4 chunks; mixed flat-cache and REST record shapes |
| twiki_eos      | OK     |     6 |     6 |       0 | ok     | 3 topics + 3 chunks; topic_parent + wiki_link references |

- completeness: `complete`, failed sources: none.
- First published generation:
  `okg:cern-team-demo:branch:default:gen:20260821T155941201299Z:2e82253200b6`.
- **Idempotence:** an immediately repeated `okg ingest` produced an
  identical write set (same per-connector counts, 0 retractions) and
  published
  `okg:cern-team-demo:branch:default:gen:20260821T155948019759Z:56ad3d3dafec`
  — the generation pinned by every read-back below.
- `okg status`: latest_published_status **published**, 338 nodes /
  296 edges; live subtypes: cmssw_release 314, document_chunk 10,
  documentation_page 6, jira_issue 4, person 3,
  software_repository 1. The bundle's core-pages-floor invariant
  (documentation_page / jira_issue / cmssw_release ≥ 1) is satisfied;
  publish gates green.

## Read-back

- `okg search --deployment cern-team-demo --query "kronos transfer
  queue stalls"` → **7 hits spanning all three text connectors** at
  pinned generation `...56ad3d3dafec`: `jira:CERNTEAM-201`
  (jira_issue), `chunk:cfb4b3cee38b459c` + `chunk:756f6ce72b42df99` +
  `chunk:6f118c271c320ac0` (twiki_eos chunks),
  `documentation_page:451ee47f08d8f0f5` + `chunk:ef4af04e022cf896`
  (docsite Team Ops Guide), `chunk:ba1316c6cbef53ea` (jira chunk).
- **Cross-connector edge**, `okg trace node jira:CERNTEAM-201`:
  incoming `references` edge (edge_id 1486363125616942354) from
  `chunk:ef4af04e022cf896`, source `cern-team-demo.docsite`,
  provenance derived_deterministic, with full nomos source lineage
  (docsite record url https://docs.cern-team-demo.example.org/ops/,
  chunk_index 0).
- **Topic-to-topic edge**, `okg trace node
  "twiki:CMS:CompOpsDemoTransfers"`: `references` →
  `twiki:CMS:CompOpsDemoHome`; `contains` → its chunk.
- **Live catalog**, `okg search ... --query "CMSSW_15_0_15"`: release
  + patch hits from the live fetch; `okg trace node
  "cmssw_release:CMSSW_15_0_15_patch1"` shows the supersedes chain
  (patch1 supersedes CMSSW_15_0_15; superseded by patch2).

## What the installer supports vs. manual steps

Supported by the profile installer (no manual step needed):

- Bundle discovery via `OKG_PROFILES_DIR` (parent dir of bundle
  dirs); init questions auto-injected as CLI flags
  (`--postgres-dsn`, `--archi-data-root`).
- `${var}` interpolation of init answers into source-defaults →
  `source_registry.yaml`; `${OKG_DSN}` accepted verbatim for the DSN
  (resolved at command time, credential never written).
- invariants.yaml copied verbatim; skills/ copied verbatim **through
  symlinks** (the bundle symlinks the repo's `skills/` — 19 playbooks
  + README + skill-triggers.yaml, no duplication; note: the task
  brief said 18 playbooks — `indico.md` landed with the Indico
  connector port after the brief was written).
- `deployment-defaults.yaml` merge injects arbitrary non-reserved
  manifest keys: the **nomos.rollout deferred block (owner
  archi-cern-team, review_by 2026-12-01) needed no manual step**,
  ditto `runtime.enabled: false` and the `skills_dir` /
  `skill_triggers` wiring.

Manual steps (each flagged [#1179] in `docs/cern-team-demo.md` and
the bundle README):

1. Schema-slice copy from the installed wheel
   (operations.yaml + sources.yaml + bridges/sources.yaml), **with a
   pruned bridges/operations.yaml** (see friction 2).
2. `ALTER ROLE okg_mcp/app_rw/app_ro PASSWORD ...` after
   `okg migrate` (known role-provisioning friction).
3. (Environment-conditional) `CREATE EXTENSION ...` on a fresh
   database of the bundled Postgres stack before `okg migrate`.

## Frictions (full detail in docs/cern-team-demo.md §Friction log)

1. `extension_missing` from standalone `okg migrate` on a fresh DB —
   the bundled stack installs extensions only into its initial
   database; the install path has provision parity, bare migrate
   does not.
2. `bridge_subtype_unknown` on catalog load when the wheel's full
   `bridges/operations.yaml` is copied verbatim under the minimal
   module trio (first: `documentation_page_references_dataset` →
   `dataset`; needs modules dataset/ticket/service/repo_starter/
   git_graph to compose verbatim). Resolved with a pruned bridge
   carrying only `cmssw_release_supersedes_release`. Ask: flag
   cross-module narrowings `optional_when_subtypes_missing` (as
   bridges/sources.yaml already does for MONIT) or split per family.
3. Lint blocker `deployment.source_registry.profile_invalid`: the
   `reference_catalog` profile admits only
   `record_identity_kind: domain_key`; the archi/sources/cmssw.py
   docstring template says `remote_id`. Bundle ships domain_key
   (matching the proven W1 registry); the docstring template needs
   correcting.
4. `catalog_ownership_mismatch` publish refusal after editing
   source_registry.yaml post-claim (drifted_blocks=[source_registry]).
   By design; the runbook now claims after all manifest edits, with
   re-claim as the documented recovery.
5. TWiki snapshot layout: `web_root` names the snapshot ROOT's web; a
   `CMS/` subdir under the root double-prefixed page ids
   (`twiki:CMS:CMS:<Topic>`). Fixed by flattening; documented in the
   bundle source-default.
6. **Deletion semantics not enforced**: after the twiki layout fix
   changed every record identity, the stale `twiki:CMS:CMS:*` facts
   stayed live through plain re-ingest, `--mode reconcile`, and
   `--mode reconcile --reset-cursor` (retract_n/retract_e = 0 every
   time) despite `missing_from_completed_scope`. The demo instance
   was rebuilt on a fresh database (drop/create `okg_cern_team`) and
   the full sequence re-executed cleanly. Substrate ask: scope-diff
   retraction or a loud unimplemented-deletion health warning. (The
   W1 evidence only ever tested idempotence, never deletion — this
   is the first time the gap surfaced.)
7. Cosmetic: `deployment declares no sync_root` warning on every
   ingest.

## Suite

`python -m pytest python/tests/ -q` after all bundle work:
**250 passed in 5.80s** — the 250 baseline maintained (the bundle
adds no python code; `bundles/` carries no operator paths or
credential values, `${...}` placeholders throughout).
