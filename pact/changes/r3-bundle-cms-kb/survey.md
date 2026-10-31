# cms-kb survey (task.r3-bundle-cms-kb.survey)

Surveyed `archi-physics/cms-kb` at `a3bd550` (main, 2026-10-21). Pins at that
commit: archi-okg `faa731ff`, okg `aa2fda75`.

## Decision

Option A, chosen by Jason on 2026-10-25. 3.0 ships one physics bundle,
`cms-physics`. The cms-kb deployment becomes its validation instance. ADR
0002 D4 and §5 record this, and PACT `r3-bundle-cms-physics` carries the
work.

## Finding

cms-kb is already a CMS **physics** knowledge base. Its README title is "CMS
Physics Knowledge Base", and its profile describes itself as "the CMS
physics-analysis knowledge instance". The `cms-physics` bundle that ADR 0002
§5 plans (cds, twiki with the physics filter, a `cms-analyses` pack of
analysis ids, notes and publications) covers the same ground as cms-kb, and
covers less of it. Nothing in cms-kb is operations-specific. So there is no
real boundary to draw between the two: they are one bundle under two names.

## Inventory

| Item | What it is | Goes to |
|---|---|---|
| `twiki_eos` source, `physics_filter: true` | archi `TwikiEOSAdapter`; 43.5k topics in, 15,762 pages out | `twiki` pack (exists), physics filter set by the bundle |
| `cds` source | archi `CDSAdapter` (in archi-okg main) | `cds` pack |
| `cmssw_releases` source | archi `CMSSWReleaseAdapter` | `cmssw-releases` pack |
| `github_repo`, `indico` (`.example`, off) | archi adapters | optional `git-code`, `indico` |
| `analysis_notes` source | `cms_kb.analysis_notes` + `_an_parse` (1,523 lines). Reads LaTeX trees from `gitlab.cern.ch/tdr/notes`: notes, sections, figures, bibliography, PAG hubs, bib→TWiki bridge | new `analysis-notes` pack (moves into archi) |
| `physics_vocabulary` source | `cms_kb.physics_vocabulary` (250 lines): PAG/POG spine as nodes | `cms-analyses` pack |
| `physics_entities` source | `cms_kb.physics_entities*` (1,453 lines): 180 alias-matched entity nodes, `references` edges, no model involved | `cms-analyses` pack |
| `PhysicsCrossLinker` + `cross_links/*.yaml` | declarative cross-source rules (bibliography, CDS↔notes, PAG/POG, TWiki groups) | `cms-analyses` pack |
| `schemas/` archi slices (`sources`, `operations`, their bridges) | generated copies of archi-okg schemas | the packs that own them |
| `schemas/` `cms_kb_*` slices (analysis notes, physics entities, connective, tier A) | cms-kb's own, by design: the sync script leaves them alone, and the drift test checks only that they don't duplicate an archi slice | move to `analysis-notes` and `cms-analyses` |
| `modules.yaml` | `document_starter`, `person`, `extraction`, same as cern-team | bundle (inherited from cern-team) |
| `invariants.yaml` | physics-page floor (10,000) and analysis-note floor (2,500), both calibrated on real builds | the floors go to their packs; the thresholds are instance values |
| `deployment-defaults.yaml` | Nomos enforce + CERN SSO role-only admission, `org_operational_private` posture, per-source policies | posture to the bundle; OIDC audience, ports and paths to the instance |
| `deployment-defaults.yaml` publisher block | statement timeout raised to 5,400 s for the cold first publish | instance; file upstream as a cold-rebuild default |
| `skills/` | symlink to archi-okg skills (whole set, including rucio/condor ops playbooks) plus its own `chat-system-prompt.md` | bundle selects the physics-relevant playbooks; prompt goes to the bundle |
| `benchmarks/golden-set` | 100 reviewed questions (82 mined from ARC review exchanges), 100 reserve, 20 graph regression | bundle smoke eval set (`r3-bundle-evals`) |
| `benchmarks/arc`, `benchmarks/entity-candidates` | tools that mine the golden set and entity aliases | stay in cms-kb (research tooling) |
| init answers | TWiki/AN mirror roots on the site's shared filesystem, cache root, ports, model `qwen3.8:27B`, OIDC issuer/audience | instance |
| `scripts/setup-postgres.sh`, `ops/pg` | own Postgres container on a free port | replaced by `archi up` (`r3-golden-path`) |
| `scripts/chat-up.sh` | works around `okg chat-instance up` failing on hosts where `host.docker.internal` can't reach the host | footgun ledger; file upstream |
| `scripts/sync-archi-schemas.sh` | copies schemas because okg refuses a `schemas:` path outside the profile | gone once bundles ship without symlinks (`r3-bundle-format.packaging`) |
| `setup.sh` | venv, env vars, idle-in-transaction timeout of two hours | `archi init`; the timeout is a footgun-ledger entry |
| "run graph writes from inside the okg submodule" (`foreign_checkout`) | operator rule | footgun ledger; file upstream |
| `scripts/fetch-ans.sh`, `export-twiki-pages.py`, `watch-build.sh`, `backup-chat-db.sh`, `operator-up.sh` | operator utilities | `fetch-ans` goes with the `analysis-notes` pack; the rest stay with the instance |
| `src/deployment.yaml`, `pipeline.yaml`, `source_registry.yaml`, `enricher_registry.yaml` | pre-overhaul manifests (last touched 2026-08-10); the README says the repo has no deployment manifest | dead; delete in cms-kb |

Two things in the cms-kb README are out of date. It says CDS has no connector
anywhere, but `cds.yaml` installs archi's `CDSAdapter`. It also says the repo
has no `deployment.yaml`, but the four old files under `src/` are still there.

## Options for the decision

**A. Merge (recommended).** Ship one bundle, `cms-physics`, built from what
cms-kb proved. cms-kb's sources become archi packs (`analysis-notes`, and
`cms-analyses` holding vocabulary, entities and cross-links). The cms-kb
deployment becomes the validation instance of `cms-physics`, and the cms-kb
repo shrinks to an instance repo plus its research tooling. 3.0 ships four
bundles, not five. This changes D4, the §5 table, and merges
`r3-bundle-cms-kb` into `r3-bundle-cms-physics`.

**B. Layer.** `cms-physics` holds the shared base (cds, physics-filtered
twiki, cmssw-releases, PAG/POG vocabulary). `cms-kb` extends it with the
analysis-note corpus, entities and cross-links. The split would follow
access: analysis notes need a CERN GitLab mirror, and the base does not. This
keeps five bundles, but the line is thin, and no user has asked for the base
without the notes.

**C. Keep the names, define cms-kb as an instance.** `cms-kb` is not a
bundle; it is the deployment name of the `cms-physics` validation instance.
This is option A in practice, and it keeps the name people already use for
the service.

## Not checked

The survey is read-only. I did not install or rebuild anything. I also
didn't check how much of the archi-okg skills set the cms-kb chat really uses.
