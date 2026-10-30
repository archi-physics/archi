# Archi ↔ OKG alignment page

**Audience:** anyone (human or agent) working on the OKG side of the Archi program.
This is the one page to read to stay in sync with the Archi side. It is versioned in
this repo; the copy on the `archi_v3` branch is canonical once merged, the `w2-twiki`
branch carries the freshest state between merges.

**Program documents:** the Archi program spec is
[`docs/adr/0001-archi-v3-program-spec.md`](adr/0001-archi-v3-program-spec.md) (this
repo). OKG has a matching program; the two describe the same three-layer
architecture. Terminology follows that program: *distribution / bundle / instance /
connector / enricher / playbook / live tool / automation*.

## What Archi is, in one line

Archi is the HEP **distribution** installed on OKG: a pip package (`python/` in this
repo, wheel name `archi`) shipping connectors, enrichers, live tools, schemas,
playbooks, and bundles — no credentials, no site config, no running services, no
reimplementation of OKG services.

## Current state (update this section when it changes)

**The cms-analyses pack adds one private okg import (2026-10-26).**
`archi.enrichment.physics_cross_links.PhysicsCrossLinker` subclasses okg's
`DeclarativeLinker` (already imported by `GlobalTagReleaseLinker`) and calls
`okg.substrate.publisher.enricher_loader.derive_write_envelope` to derive the
write envelope it labels `untrusted`. That function is private substrate, so it
joins the enricher-side imports below. The analysis-notes and cms-analyses
sources use only the public `okg.deployment` SDK. Tested against the v0.2.0
release (no pin change). PACT `r3-bundle-cms-physics`.

**archi composes bundles itself and reads okg's profile planner (2026-10-25).**
okg has no profile composition (ADR 0002 §9 ask 3), so `archi render` writes one
ordinary profile into the instance's `./profiles/`, which okg's discovery
cascade already searches; no step sets `OKG_PROFILES_DIR`. `archi diff` and
`archi add` import okg's planner (`build_plan`, `discover_profile`,
`planned_writes`, `source_default_entries`) to see what a profile produces for a
deployment's answers, and `archi add` runs the claim → load → ingest → publish
commands. Tested against the v0.2.0 release (no pin change). PACT
`r3-bundle-format`.

**The distribution is now `archi-kb` (2026-10-25).** ADR 0002 D2: the name
`archi` on PyPI belongs to an unrelated package. The import name is still
`archi`, and nothing on the okg side changes. The wheel is now
`archi_kb-<version>-py3-none-any.whl`. It is not on PyPI yet; trusted
publishing is PACT `r3-release-home` task `publish`. No pin change.

**First use of okg's image SDK (2026-10-21).** `archi.sources.indico_live` imports
`okg.deployment.ImageRef` and, with the off-by-default `attachment_images`
option, puts `ImageRef.url` values on talk records so an agent can open a
talk's PDF, PNG or JPEG attachment with `inspect(kind="image")`. okg fetches
the URL at read time, only from hosts in the deployment's
`agent_surface.images.url.prefixes`; archi downloads and stores nothing. Tested
against the v0.2.0 release (no pin change).

**CI and the quickstart now use the public AKMON v0.2.0 release instead of
private `mitdbg/okg` (2026-10-20).** AKMON (`github.com/mitdbg/akmon`) is the
public release of okg. The Python package and the `okg` command keep their
names, so no `import okg` line changed. CI installs the v0.2.0 wheel by URL and
sha256, with no deploy key, so fork PRs now run the full suite instead of being
skipped. The release was cut from okg `54e6cf44ee`, which contains the old pin
`ac078aabd`, so this is a move forward, not a downgrade. The full archi suite
gives 1795 passed and 1 failed against the wheel with no source change; the one failure is a 1-second timing assertion in `test_anonymizer.py` that also fails on unmodified `main` on a loaded host. Files okg left out of the public
tree (agent-memory and enterprise-RAG benchmarks, two persona prompts, part of
the operator console) are not used by archi. v0.2.0 also exports `COMPLETED_SCOPE_RUN_MODES` from `okg.deployment`, so the "known limit under an okg that does not count prior admissions as history" below no longer applies to this release; switching `archi.sources._run_modes` to import the set is a follow-up, not done here.

**Reference catalogs are no longer refused under okg's release modes
(2026-10-05).** Since okg `d84f8fd275` (in `ac078aabd`, the pin at the time), okg runs a
`reference_catalog` source in run mode `release_new`, which it picks
automatically; `release_unchanged` comes only from an explicit mode setting.
Both complete the scope in okg's table. CMSSW releases, CondDB global tags, DBS
datasets and GitHub repositories only recognised `scope_complete` and
`reconcile`, so they never claimed a complete scope and returned no cursor;
okg admission then refused every later run that emitted nothing new as
`partial`, which blocked publishing. A downstream CMS instance turned its CMSSW
releases source off on 2026-10-01 for this reason. Now:

- **CMSSW releases and GitHub repositories** claim a complete scope in both
  release modes. Deletion by absence is live for them: each declares
  `deletion_semantics: missing_from_completed_scope`, so a valid run retracts
  records no longer in its input. Before, it could not happen: until
  `d84f8fd275` such a retraction failed the run, and since then the readers
  never claimed a complete scope. GitHub repositories withholds the claim when
  no configured slug is valid, and refuses an explicit empty `repos: []` at
  construction (a missing `repos` keeps the defaults). In `release_unchanged`
  a catalog that shrank fails the run loudly, because okg refuses retractions
  in that mode.
- **DBS datasets and CondDB global tags** deliberately claim no complete scope
  in the release modes, so they never delete by absence: a narrower cache
  export would retract an unbounded share of the catalog, and okg has no guard
  on retraction size yet (mitdbg/akmon#19). They return a cursor
  (`input_content_sha256`, the SHA-256 of the input caches; for CondDB also the
  CMSSW cache it reads). The cursor only gives the source stored state, which
  lets okg accept an unchanged rerun as a no-op; okg still re-reads the cache
  on every run. Costs: only a completed scope advances okg's last reconcile
  time, so these two read as stale in freshness checks; and okg refuses
  `scope_complete` and `reconcile` for a `reference_catalog` source at run
  start, so under current okg no run removes datasets or tags that leave the
  export. They stay until mitdbg/akmon#19 lands and these sources switch to deleting.
  Known limit under an okg that does not count prior admissions as history: a deployment where DBS or
  CondDB already published facts without a cursor (the state the old bug
  leaves) is still refused `partial` on its first run after this fix, because
  every fact dedupes and no cursor is stored yet. A newer okg fixes this:
  it counts a prior accepted and published admission of the source as
  history, so that first unchanged rerun is accepted as `incomplete_scope_noop`,
  with no deletion and no blocked publish. It applies once the instance's okg
  includes that change. A fresh deployment, or one where the source never published,
  is unaffected either way.

The scope-completing modes live in one archi module, `archi.sources._run_modes`,
which a test holds equal to okg's `COMPLETED_SCOPE_RUN_MODES` and to the
`reference_catalog` profile's accepted modes. Connectors may not import
`okg.substrate` (`okg lint deployments`, DEPLOYMENT-001), so the set is copied,
not imported. AKMON v0.2.0 now exports the set from `okg.deployment`,
so the readers can import it from there; that switch is a follow-up, not done
here. The cursor uses the public `okg.deployment.Checkpoint`, added to the
import block below.

**Cache downloaders and a snapshot builder moved in (2026-09-28, comp-ops A4).**
The JIRA, static-docs and SSO-login downloaders and the download manifest moved
from okg-deployments `cms/scripts` into `archi.downloaders`; they import nothing
from okg. `python -m archi.snapshot build` packs existing caches into one
checksummed, email-free `tar.zst` per group plus `snapshot.lock.yaml` (collection
date, SHA-256, file and record counts per group); `verify` checks archives
against the lock. Each group is validated by running its own archi reader. No
real snapshot is built yet. See [cache snapshots](cache-snapshot.md). No okg
import was added, so the import block below is unchanged.

**Comp-ops readers can now be run by okg (2026-09-27).** CRIC, CRIC core, DQM,
GOCDB downtimes, CondDB global tags, DBS datasets, WMStats workflows and GitHub
repositories each gain a registry adapter (`CRICAdapter`, `CRICCoreAdapter`,
`DQMAdapter`, `GoCDBDowntimeAdapter`, `CondDBGlobalTagAdapter`,
`DBSDatasetAdapter`, `WMStatsWorkflowAdapter`, `GitHubRepoAdapter`), and the
registry template in each reader's docstring now names it. Before, the template
named the bare reader, and okg's runner failed every run because the reader's
result has no `next_cursor`. Tested against okg `5b2fd076c`.

**Their templates now pass okg's install checks (2026-09-27).** The eight
templates had declared profile combinations okg refuses; they now declare
`scoped_locator` (CRIC, CRIC core, DQM, GOCDB), `domain_key` (CondDB, DBS,
GitHub repositories) and `updated_at` revisions (WMStats), which is what the
okg-deployments cms registry declared for the same reader code. No reader
code changed, so no node is re-keyed. The GitHub template's empty `params:`
became `{}`, which strict admission requires. Tests run each template through
okg's profile check and strict registry admission, on both okg `5b2fd076c`
and okg `dev` @ `fdf5638bf`. HyperNews, SITECONF, the four MONIT readers and
`CERNPreflightSource` still have no adapter.

**CI reads `mitdbg/okg` directly; pin bumped `5b2fd076c` → `ac078aabd`
(2026-09-27). Superseded 2026-10-20 by the AKMON wheel above.** Every CI run since 2026-09-24 failed at the okg install
with `Authentication failed`: the fork token no longer worked. CI now installs okg
over SSH from `mitdbg/okg` with a read-only deploy key held in the repository
secret `OKG_DEPLOY_KEY`; CI no longer uses the fork or `OKG_REPO_TOKEN`. The pin moved
to okg `dev` in the same change; the full archi suite passes against it with no
source change.

**Archi installs as-is on current okg (2026-09-15).** okg `dev` @
`5b2fd076c` removed the framework's Archi-specific install code. Archi needed no
change: the full suite passes against that commit, and `okg install --profile
cern-team` still plans cleanly. A test deployment that needs repeatable data uses
this normal install with the cache-backed Jira and documentation sources and the
hash-pinned CMSSW option below, and records the data hashes in its own evidence.
A consumer-side frozen install package (#643) was closed unmerged.

**Frozen CMSSW input contract (2026-09-14).** This change adds an explicit
path and SHA256 binding that never fetches in frozen mode. It preserves the live
profile. The historical CMSSW cache is still unavailable; unit fixtures are not
real-deployment baseline evidence. See [frozen CMSSW inputs](frozen-cmssw-input.md).

*Last updated 2026-10-20, tested against AKMON v0.2.0 @ `e2c618010b` (cut from okg `54e6cf44ee`); archi branch
`main` @ `e1c28860`.*

**Pin bumped `34efbad1b` → `5b2fd076c` (2026-09-15).** The fork was
synced to okg `dev` first; it had no commits of its own. Full archi suite
**357 passed** with none skipped, `test_alignment_page.py` included. No source
change needed.

**Pin bumped `1475c87d5` → `34efbad1b` (2026-09-10), Sprint 11's 45 merges.** Full
archi suite **339 passed**, `test_alignment_page.py` included, so all ten guarded
`okg.substrate.*` symbols still resolve across 663 commits. No source change needed.

**CI was testing a different okg than this page claimed.** Until 2026-09-10 the
workflow installed `lucalavezzo/okg@f5ec3b58d` — a *fork*, whose `dev` silently
stopped tracking upstream on 2026-08-25. So while this page recorded two pin bumps
and a passing suite, CI re-checked a three-week-old okg; every "passed against `dev`"
claim here was true of local runs only. Nothing failed, which is why nobody noticed.
Found only because the connector SDK migration failed CI on `ImportError: cannot
import name 'EdgeFact'` — the SDK did not exist in the pinned commit.

**CI stopped using the fork on 2026-09-27 (deploy key since superseded by the AKMON wheel, 2026-10-20).** It could not do so while CI
authenticated with `OKG_REPO_TOKEN`, a fine-grained token scoped to the fork's
account (reading `mitdbg/okg` returned 403, run `34494575759`). The read-only
deploy key on `mitdbg/okg` replaces both, so a pin bump no longer needs a fork sync.
The workflow release must still equal the release in the *Last updated* line above.

**Our migration debt, now measured rather than estimated.** 44 imports of
`okg.substrate.*` across eleven private modules:

| Side | Sites | Status |
|---|---|---|
| Connector — `library.sources.base` (17), `sources.preflight` (4), `content_hash_probe` (3), `mutable_api_probe` (3), `sources.redaction` (1) | **28** | **Migratable now.** `okg.deployment` exports 82 names; 17 of the 20 we need are present. `SourceRun` → `ConnectorRun` and `SourceHealth` → `ConnectorHealth` are renames touching ~133 references. We do not use `content_hash_items`. |
| Enricher — `enrichers.base` (4), `enrichers.derived_edges` (4), `library.linkers` (5), `linkers.declarative` (1), `alias.protocol` (1), `publisher.enricher_loader` (1) | **16** | **Blocked** on the enricher read surface, deferred at our own recommendation. |

One of the sixteen is `okg.substrate.library.linkers._chronos` — underscore-private by
Python convention, and the single item we flagged upstream as most worth absorbing.

**Sprint 11 closed 2026-09-10 with our three reports absorbed.** Its
closing summary retains the **connectors-first** example and records the enricher
read surface as **deferred** — both our recommendation. Sprint 14 owns
**provider/admin bootstrap** (our chat findings 2 and 3) and **operator
acceptance** including install and chat (our install ask). **The
external-distribution proof did not pass** and moved to Sprint 14; our wheel
remains its required real arm.

**Owed by us, per that summary:** trusted admission policy, a real legacy inventory,
and **acknowledgement of the current v2 contract — the v1 one is now insufficient**.
The v1 digest we acknowledged (`sha256:db38e9bb…`) is still valid for v1; v2 is
`sha256:5cba96ce160313402809651027e0d0c708610542a485c8ac0a88a578b2c69f6a`. Held
deliberately until the real arm can run, since the external-distribution proof still lacks the consumer
environment, sealed packages and lifecycle receipts that would make an
acknowledgement outlive the day it is issued.

**The bundle now ships the assistant's system prompt (2026-09-08, PR #632).**
`skills/chat-system-prompt.md` (since PACT r3-packs-split the renderer writes it
into the rendered profile from `python/archi/bundles/base/` and each pack's
`prompt_fragment`), declared as
`chat.preset.system_prompt_ref`. This is required, not decorative: `okg chat sync`
refuses a deployment that declares neither `mcp_instructions` nor a
`system_prompt_ref`, because Open WebUI never shows the model the MCP server's own
instructions — without it the assistant has graph tools registered and no idea it
has them. Tests assert the prompt is declared, shipped, non-empty, and still names
the six operators. **For OKG's chat work: we think that refusal is right, and it argues the
prompt belongs to what a bundle supplies rather than being optional decoration.**

The rest of the chat surface (`chat:` and `search:` blocks, the preset, the MCP tool
list, the `schemas/` slices, the `skills/` slot) was already on `archi_v3` and is
unchanged. The full chain is verified end to end on a fresh database: install →
publish 2,586 nodes → `chat-instance up` → `mcp-serve` on the port the bundle
declares → `chat sync` exiting 0 with the graph tools bound and the prompt applied.

**Three chat findings for OKG's chat work, drafted (outside this repository) and verified against
`dev` @ `1475c87d5`. Worst first: a synced instance opens on a model with **no graph
access**, because `chat.models.default` is both the preset's `base_model_id`
(`sync.py:4937`) and the site default (`sync.py:4432`) while the vendor selects
`default_models.split(',')[0]` — there is no deployment-side workaround, and the
failure is a confident ungrounded answer with no warning. Second, the model provider
cannot be declared at all (`chat-instance up` injects two hardcoded variables,
`instance.py:1253`) and sits in none of the six reconciled surfaces, so `chat sync`
reports green on an instance that cannot answer. Third, the first admin account is
not bootstrapped. Filed separately; recorded here so the chat design has them.

**Pin bumped 2d528e824 → 1475c87d5, drift closed (2026-09-08).** A 1153-commit
fast-forward, and **the import surface survived it unchanged**: full archi suite
**339 passed**, including `test_alignment_page.py`, which mechanically checks every
one of the ten private `okg.substrate.*` symbols below still resolves. No source
change was needed on our side.

The external-distribution conformance contract landed inside that gap —
`conformance.py` did not exist at the old pin and is 2,132 lines at `dev`, arriving
in `46c42c7e0` (merged 2026-08-28), with a follow-up on 08-30. That
had made our acknowledgement stale twice over: the earlier acknowledgement named branch
`f4081329` and schema digest `sha256:301ec97f…`, while the catalog digest is now
`sha256:db38e9bbbe571dbb9efec29192f786d851095ca412f5a5fda17d1af81f62e662`. **Re-issued
2026-09-08** upstream,
naming `dev` rather than a branch and vouching for `archi_v3` @ `0aec65f9`, so the
real conformance arm no longer returns
`conformance_external_acknowledgement_pending`. Note for whoever re-runs it: that
acknowledgement binds one Archi revision to one OKG schema digest, so a commit to
either side stales it again — an open question on that comment asks whether it is
meant to be re-issued per run.

**PACT corpus migrated to v5 (2026-09-08).** okg cut over on 2026-09-07 and we followed the same day, using okg's own `pact migrate`: all nine
manifests under `pact/changes/` and `pact/archive/` are now `pact_version: 5`, zero
refused, zero destructive notes. Two things for anyone reading our ledger:

- **Our evidence carries but does not bind.** `check` reports the migrated evidence
  rows as `evidence_unbound` and W2's five requirements as unproven. v5 binds
  receipts to a commit and ours predate that model. This is the same conclusion v4's
  staleness guard had already reached after PR #617 touched `python/archi` — not new
  breakage — but it does mean our ledger currently asserts less than its prose does,
  and re-attaching receipts is outstanding work.
- **`pact/project.yaml` stays at `pact_version: 4` deliberately**, matching okg's own
  choice for theirs: the v4 compatibility reader consults it and the v5 CLI never
  does. Note also that `okg pact view` still parses a v5 manifest but reports
  `normative_v4: true` on it — degraded output, not authority.

**Sprint 11 is the Archi chain.** Its exit criteria end in the
external-distribution proof, run on a real deployment — which OKG's own note says
only the `cern-team` wheel on SubMIT satisfies. Design notes for the chain's first
items are posted and adversarially reviewed; our answers to the seven open
questions went upstream.
In flight and directly relevant to us: owner-keyed module providers (the
D11 gap this page has flagged since 2026-08-19), package v2 with an OKG
compatibility window, and the SDK connector fact surface with a `partial`
source-run status — which closes our own `partial` status friction. None had
merged as of 2026-09-08.

**Pin bumped, drift closed (2026-08-27).** The 210-commit exposure this page
flagged is resolved: we fast-forwarded to okg `dev` @ `2d528e824` and re-ran the
full archi suite — **337 passed**, including `test_alignment_page.py`, which
mechanically checks every one of the ten private `okg.substrate.*` symbols below
still resolves. The import surface survived the bump unchanged.

**The profile `schemas:` slot landed, and it removes most of our install
friction.** The slot now materializes distribution-owned schema files before the
first catalog load. Verified end to end on an empty database: `okg install
--profile cern-team` alone — no `--no-publish`, no manual schema copy, no manual
migrate/claim/load — reaches `first publish complete`, generation
`gen:20260827T142617739260Z:5e3c0fc5ada0`, 2,586 nodes / 2,279 edges. The
quickstart went from ten steps to one command. Two notes for you: (1) the
contract refuses symlinked schema assets, so our bundle carries real copies of
`python/archi/schemas/` with a byte-identity test guarding the duplication —
worth knowing if other distributions try to link rather than copy; (2) passing
the manifest's deferred `'${OKG_DSN}'` literal now fails the install's own
migrate step (`missing "=" after "${OKG_DSN}"`), so a one-command install must
pass a real DSN and the password lands in `deployment.yaml`. That is a small
regression in the credential-indirection story and the only reason we cannot
recommend the literal form any more.

**Sync channel** (established 2026-08-19; surface-change heads-ups go to OKG's
Archi program). Re-pin validation 2026-08-19 against the strict registry-admission schema
(`8b65a330a`): all four scratch registry entries (bootstrap fixture,
cmssw-releases, jira, docsite) lint clean, ingest OK, and publish — **zero
refusals**; full test suite 199/199 on the new pin. OKG's chat and identity branch is merged into `dev`
(2026-08-13): chat, MCP HTTP auth, and principal mapping are now landed premises —
Archi's W9 targets `dev`. **Upstream state re-verified 2026-08-24:** the
chat/console slices are **merged** — the remaining W9 dependency is the parent
per-bundle chat work, still open. Of our three filed frictions, **the
`output_scope_summary` hand-duplication is closed upstream**; narrowings outside
`schemas/bridges/` silently ignored (the real bug of the three) and no
`partial` SourceHealth status remain open. Our import surface is CI-gated on the
okg side
(`tests/substrate/contracts/test_external_import_surface.py`). D11 (shared ontology
modules in the wheel) is planned against the deployment-product packaging wave
(digest-pinned distribution assets), not an env-var override.

**All eight W2 tasks are done and evidence-gated** (PACT change
`w2-consolidate-hep-sources`; suite 250/250): packaging, `archi.auth`, all 12
connector families (JIRA + docs ingest-proven live; TWiki EOS reader + crawler over
one parser core; MONIT; and the catalog/feed batch — CRIC, CRIC-core, CMSSW
releases, CondDB, DBS, DQM, GitHub repos, HyperNews, Indico, SITECONF, GOCDB,
WMStats — fixture-tested, registry templates in the ingest-proven strict shape),
4 live tools, 5 enrichers + packaged defaults + the anonymizer (anonymize_data
cutover gate closed), 18 playbooks, and the comp-ops lint re-check (zero
regressions vs baseline; sole delta a version-introduced skipped notice). Schema
slices: `operations.yaml` + `sources.yaml` (+ bridges), every class/narrowing
single-defined. **The cern-team bundle + end-to-end install demo are DONE and gated
(PACT w6-cern-team-bundle)** — `okg install --profile cern-team` on a fresh DB: lint
zero blocker/warning, four connectors ingest OK (live cmssw releases.map fetch, 314
nodes), generation `gen:20260821T155948019759Z:56ad3d3dafec` published with
cross-connector reference edges, idempotent re-run. **The reproducible runbook was
`docs/cern-team-demo.md` (now marked superseded) — the external-distribution release-claim artifact at the time**; run results
+ 7-item friction log in `pact/changes/w2-consolidate-hep-sources/`. For the install contract: two
manual steps remain outside the profile contract (wheel schema-slice copy incl. a
pruned operations bridge; role passwords post-migrate) — both flagged in the runbook.
**New substrate finding for this channel: deletion semantics not enforced** — stale
records survived reconcile + `--reset-cursor` under `missing_from_completed_scope`
(demo friction 6; repro in the runbook) — being triaged upstream as a separate
substrate correctness issue (upstream response, 2026-08-21). The consolidated PR
(#610) is **merged**, and OKG has pinned `archi_v3` @ `728739e6` as the first
real external-consumer baseline for the deployment-product work. Both manual
install steps now have upstream owners: the schema-slice copy + hand-pruned
operations bridge land in `package-and-migrate-archi-products` (digest-bound
schema/ontology/bridge assets), and role/credential provisioning lands in
`apply-and-operate-deployment-instances` (secret refs from the instance,
lifecycle-provisioned roles). The conformance harness will use `cern-team` as its
first real consumer; the final external-distribution claim adds artifact-only checks (no
`OKG_PROFILES_DIR` or authoring checkout, no manual schema/password steps,
lock/release readback, defined no-change reapply). One question is open to us on
the channel: whether sealing the v3 wheel together with a materialized bundle +
playbook payload conflicts with our distribution boundary (maintainer's answer
pending).
Comp-ops instance model (maintainer, 2026-08-21): the instance repo is
`gitlab.cern.ch/archi/cms-compops` (branch `archi_v3`); `okg-deployments/cms` is the
frozen parity reference and **retires once the cms-compops v3 instance reaches
parity** (retirement is mitdbg's call — relevant to your packaging PACT's proof
targets). **W7's local half is done and evidence-gated** (PACT change
`w7-compops-instance`): the full v3 instance definition (deployment manifest,
21-connector registry, materialized schemas incl. a pruned operations bridge,
20 playbooks, relocated parity auditor, credentialed bring-up runbook,
`versions.lock` pinning okg `f5ec3b58d` + archi `728739e6`) lives in
`cms-compops@archi_v3`; local validation on a fresh DB: lint zero
blocker/warning, 13 connectors ingest OK (one live fetch), 8 credential-gated
connectors fail clean with no scope claims, published generation idempotent on
re-run (evidence: `docs/w7-local-validation.md` there). The credentialed live
bring-up + strict parity run on a CERN host is the remaining half
(`docs/bring-up.md`). New friction for this channel: PACT gates resolve the
graph-projection deployment from `OKG_PACT_GRAPH_DEPLOYMENT` (default
`okg-workspace`) and ignore `pact/project.yaml`'s
`graph_projection.deployment` — external repos need the env var exported or
gates refuse with an okg-workspace database-identity mismatch. New PACT v4 note
from the re-pin: `change.tier` is now mandatory (`missing_lifecycle_tier`
otherwise). The superseded v2 application has been removed from the archi_v3
line (W10 teardown, PR #611 merged 2026-08-21); v2 lives on `main` until
instance cutover.

**Circle-back adversarial review + fixes (2026-08-24, PACT change
`circleback-fixes`):** a four-domain adversarial review of `archi_v3` @
`728739e6` (your pinned external-consumer baseline) found 3 blockers / 15
majors, dominated by connectors claiming `completed_scope` after silently
losing input — a mass-retraction hazard under `missing_from_completed_scope`.
All confirmed blocker/major findings are fixed with per-finding regression
tests (suite 254 → 320); ~85% of the bugs exist identically in the frozen
canonical `okg-deployments/cms` copies (deviations recorded in the change
dir's notes files). Two items are routed to this channel instead of patched
locally: (1) **likely root cause for the deletion-semantics finding you are
triaging** — no adapter ever provides `SourceRun.record_set`, and the runner
then silently skips retraction synthesis rather than failing loudly; (2) the
enricher derived-edge lifecycle (attrs-independent dedupe keys block
re-derivation after any transient retraction; no evidence-based retraction) —
both substrate-contract questions. Also relevant to the sealed-artifact
design: bundle playbooks are symlinks escaping `bundles/`, so packaging must
dereference (your "materialized payload" wording already covers this).

## The exact substrate surface Archi consumes today

**The connector half is done (2026-09-10).** Everything Archi's connectors need now
comes from the public `okg.deployment` SDK, so OKG's boundary lint has nothing left
to find on that side. What remains below is enricher-only, and is blocked on the
enricher read surface, which OKG deferred at our own recommendation.

**Python imports (all of them).** The first entry is the public SDK; everything
below it is still private substrate, and all of it is enricher-side.

```
okg.deployment:
    NodeFact, EdgeFact, ImageRef, ProgressMarker,
    ConnectorRun, ConnectorHealth, PreflightResult, Checkpoint,
    ConnectorAdapter,
    ContentHashProbe, MutableApiProbe,
    file_preflight, credential_preflight, http_preflight, redact,
    deployment_workflow, deployment_scheduled, deployment_step,
    request_escalation, cost_budget
okg.deployment._internal.context:     bind_sdk_context
okg.deployment._internal.cost_budget: BudgetExhaustedError
okg.deployment._internal.escalation:  EscalationTimeoutError
okg.substrate.db.connection:          connect
okg.substrate.generation_resolver:    resolve_okg_generation
okg.substrate.runtime.escalation:     EscalationStateError
okg.substrate.runtime.escalation_response: respond_to_escalation
okg.substrate.enrichers.base:     EnrichResult, IncrementalContext
okg.substrate.enrichers.derived_edges:
    DerivedEdgeCandidate, insert_deterministic_edges, mint_edge_id
okg.substrate.library.linkers:    _chronos
okg.substrate.library.linkers.declarative: DeclarativeLinker
okg.substrate.publisher.enricher_loader: derive_write_envelope
okg.substrate.alias.protocol:     AliasMatch
okg.substrate.deployment_bootstrap.profile_init:
    build_plan, discover_profile, planned_writes, source_default_entries
```

**The ops runtime (2026-10-25).** `python/archi/ops/workflow.py` is the one module that
imports `okg.deployment`: the workflow, scheduled and step decorators, `request_escalation`
and `cost_budget`, all public. It also imports three private names from
`okg.deployment._internal` (`bind_sdk_context`, `EscalationTimeoutError`,
`BudgetExhaustedError`) and one substrate error (`EscalationStateError`), in one function
(`_private_okg`), until okg offers a public context binding (ask O9 in
`docs/ops-runtime-akmon-asks.md`). `python/archi/ops/okg_io.py` and `archi ops` use three more
substrate names to answer escalations and pin a generation (`respond_to_escalation`, `connect`,
`resolve_okg_generation`: asks O3 and O7). Each module's docstring tells what else it reads
(SQL tables, CLI output) and the ask that removes it.

**The bundle tooling reads okg's profile planner (2026-10-25).** `archi diff`
and `archi add` (`python/archi/bundles/lock.py`, PACT `r3-bundle-format`) ask
okg what a rendered profile produces for a deployment's answers, so they compare
and extend a registry with okg's own interpolation rather than a copy of it.
These four names are the profile installer's planner; profile composition in
okg (ADR 0002 §9 ask 3) would replace both the renderer and this use.

**The dropped fields are read by the substrate, not by us (2026-09-16).** Archi
never referenced `next_cursor`, but the runner does, on the result of whatever
class a source registry names. So a reader that returns `ConnectorRun` cannot be
registered directly: every run raised `AttributeError: 'ConnectorRun' object has
no attribute 'next_cursor'` from the connector migration until this was fixed.
`ConnectorAdapter` is the framework's bridge, and the shipped source templates
(now `bundles/packs/*/sources/`) name one `<Reader>Adapter` per reader. The readers themselves are unchanged.

**An adapter's authority must be a string literal (2026-09-20).** The substrate
reads a source's `profile` and `change_probe_kind` off the class *before* it
imports or constructs anything, so neither may be computed. `change_probe_kind`
it reads by parsing the module's AST — `_class_level_str_attr` in okg
`substrate/deployment_lint.py` accepts only an `ast.Constant` string — and
`profile` it reads with `inspect.getattr_static` in
`substrate/ingest/adapter_factory.py`. Mirroring the reader with
`profile = Reader.profile` parses as an `ast.Attribute`, reads as absent, and
fails every source with `deployment.source_registry.probe_missing`. Each adapter
therefore declares both as literals, and `test_bundle_source_adapters.py` parses
the source files to hold them equal to the reader's own values.

The adapters forward nothing else. `cache_paths` is an Archi reader detail — the
readers that have it pass it to their own preflight and content hash — and the
name appears nowhere in okg, so there is nothing on the substrate side to bridge.

Every SDK name above was verified to be the *same object* as the substrate name it
replaced, except `ConnectorRun`, which is a genuinely narrower type: it drops
`next_cursor`, `effective_scope_complete`, `bootstrap_identity` and three
`applied_token_*` fields — which Archi does not reference directly — and retains
`record_authority_stream`, `record_set` and `record_set_replacement_keys`, the
fields OKG's connector-SDK design note warned would cost incremental sources their changed-slice
retractions.

If you change any of the remaining `okg.substrate.*` entries on `dev`, Archi breaks.
`_chronos` is underscore-private by Python convention and remains the single item
most worth absorbing into the SDK.

**Contracts consumed as data/CLI (not imports):** the source-registry entry schema
(`source_class`, `record_identity_*`, `change_probe_kind` soundness, admission
policy with `output_signature` + `output_scope_summary`, `sync:` block); deployment
manifest keys (`modules:`, `schema_dir`, `nomos.rollout` deferral); the
edit → `catalog ownership claim` → `catalog load --apply` → `ingest` sequence;
`okg deployment lint` codes; LinkML deployment schemas + `schemas/bridges/`
narrowings; install profiles (the `./profiles/` step of the discovery cascade,
which `archi render` writes into) for bundles; the closed
`SourceHealth`/preflight status vocabulary; MCP read surface for validation.

## Mapping: OKG work items ↔ Archi needs and evidence

| OKG work item | Archi's stake | Evidence we already have for you |
|---|---|---|
| External schemas/bundles | **The wheel now carries the bundle and its materialised playbooks** (force-include; the repo's playbook symlinks are dereferenced at build), so an install needs no authoring checkout — `archi-profiles-dir` prints the installed payload path. What remains on your side: the profile resolver has **no installed-distribution discovery** (cascade = explicit override → env var → `./profiles/` → substrate library → fetch cache, `profile_init.py`). archi works around it without the env var: `archi render` writes the profile into the instance's `./profiles/` and okg is run from there (PACT `r3-bundle-format`); entry-point discovery is ADR 0002 §9 ask 2. Also still open: shared ontology *modules* (`load_modules_root()` is package-locked) and a versioned distribution contract | ADR §7 ask 2; deployment-scoped schema slices shipping in the wheel (`python/archi/schemas/`) |
| Per-user MCP | Blocks multi-user instances (cms-knowledge-base bundle) | ADR D12 |
| Connector/enricher SDK | Replaces the import surface above; please treat that list as the minimum viable SDK | W1 friction log `pact/changes/w1-prove-the-seam/FRICTION.md` (6 items: scaffold nomos gap, ownership-claim discoverability, role passwords, --json purity, path-vs-name resolution); wheel + scratch-deployment reproduction in the same dir |
| External review | Unblocks the queue-bot (Redmine/JIRA reply) — our W8, same first consumer you named. Approve-with-edits + expiry confirmed in your scope: exactly what we needed | ADR §5.3, §7 ask 3 |
| Chat | Archi ships per-bundle chat config only | ADR W9 |
| Automation accounting | Matches our kill criterion (defer until >1 real automation) | ADR §7 ask 4 |
| End-to-end proof | **Our `cern-team` bundle install demo is this proof** — an external, wheel-installed distribution driven end to end on a SubMIT host. Coordinate before building a synthetic consumer | W1 seam proof (done); demo lands with the bundle work |

**Packaged bridges now compose as shipped — fixed on our side, and the fix is
worth copying.** (Previously reported here as an external-schemas blocker.) Copying the wheel's
`schemas/bridges/operations.yaml` verbatim fails `okg catalog load` with
`bridge_subtype_unknown`, because its narrowings lack
`optional_when_subtypes_missing` and therefore require every referenced subtype
to be in the composed module set. First failure, verbatim from the executed
runbook (`docs/cern-team-demo.md`, step 2): narrowing
`documentation_page_references_dataset`, child subtype `dataset`. Both
consumers hit it independently and both had to hand-prune: the cern-team demo
(friction 2 in `docs/cern-team-demo.md`) and the comp-ops instance, whose
committed copy "prunes the five narrowings owned by the ticket/service/code
modules" (`cms-compops docs/bring-up.md:63-67`). **The fix already has
precedent in our own tree** — `schemas/bridges/sources.yaml` flags exactly this
case for the MONIT narrowings. Ask, unchanged from the demo's friction log:
packaged bridges should flag cross-module narrowings
`optional_when_subtypes_missing`, or ship split per family. **We have now done
the former in our own bridge and it works**: flagging only the narrowings whose
endpoints vary by consumer (18 of 88, covering `Dataset`, `Ticket`, `Service`,
`Endpoint`, `Repo`, `SourceFile`, `Comment`) makes one shipped file compose
verbatim for both the cern-team trio and the comp-ops module set — 168 and 201
narrowings respectively, the latter identical to what its hand-pruned copy
produced, with lint still clean. Everything both consumers guarantee stays
strict, so a module dropped by accident still fails loudly rather than silently
composing fewer narrowings (a hazard we reported upstream). **Suggested for the packaged
substrate bridges too** — this removes a manual install step and strengthens the
artifact-only story, since a materialised payload that still needs hand-pruning
at install is not artifact-only.

Additional substrate friction found while porting, not yet in any issue:
narrowings outside `schemas/bridges/` silently ignored (lint green, ingest-time
ProducerPolicyViolation); `output_scope_summary` must hand-duplicate
`output_signature`; no `partial` status in the closed SourceHealth vocabulary.
Details + repros: `okg-asks-drafts.md` items 6–7 in the archi_v3 working notes and
`pact/changes/w2-consolidate-hep-sources/parity-ingest-demonstration.md`.

## How to sync with us

- Read this page + the PACT change dirs under `pact/changes/` (porting matrix,
  ingest demonstrations, reconciliation notes — every deviation from the canonical
  cms code is recorded there).
- Interface drift: we pin the okg `dev` commit we tested against at the top of this
  page and re-audit on bump; if you change something in the consumed surface above,
  a heads-up to OKG referencing this page is enough.
- Bugs we found in shared/canonical code: two TWiki parser bugs in
  `okg-deployments/cms/cms_sources/twiki_eos.py` (the `=code=` regex eats `=` from
  assignments across lines; title-less heading markers absorb the next line) —
  fixed on the Archi side with documented deviation
  (`pact/changes/w2-consolidate-hep-sources/v2-reconciliation-notes.md`) and
  **upstreamed into the canonical copy**.
