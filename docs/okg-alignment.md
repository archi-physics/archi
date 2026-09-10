# Archi ↔ OKG alignment page

**Audience:** anyone (human or agent) working on mitdbg/okg#1178 and its sub-issues.
This is the one page to read to stay in sync with the Archi side. It is versioned in
this repo; the copy on the `archi_v3` branch is canonical once merged, the `w2-twiki`
branch carries the freshest state between merges.

**Program documents:** the Archi program spec is
[`docs/adr/0001-archi-v3-program-spec.md`](adr/0001-archi-v3-program-spec.md) (this
repo). okg#1178 is the OKG-side program; the two describe the same three-layer
architecture. Terminology follows #1178: *distribution / bundle / instance /
connector / enricher / playbook / live tool / automation*.

## What Archi is, in one line

Archi is the HEP **distribution** installed on OKG: a pip package (`python/` in this
repo, wheel name `archi`) shipping connectors, enrichers, live tools, schemas,
playbooks, and bundles — no credentials, no site config, no running services, no
reimplementation of OKG services.

## Current state (update this section when it changes)

*Last updated 2026-09-10, tested against okg `dev` @ `34efbad1b`; archi branch
`archi_v3` with PRs #610–#639 merged.*

**Pin bumped `1475c87d5` → `34efbad1b` (2026-09-10), Sprint 11's 45 merges.** Full
archi suite **339 passed**, `test_alignment_page.py` included, so all ten guarded
`okg.substrate.*` symbols still resolve across 663 commits. No source change needed.

**Our migration debt, now measured rather than estimated.** 43 imports of
`okg.substrate.*` across ten private modules:

| Side | Sites | Status |
|---|---|---|
| Connector — `library.sources.base` (17), `sources.preflight` (4), `content_hash_probe` (3), `mutable_api_probe` (3), `sources.redaction` (1) | **28** | **Migratable now.** `okg.deployment` exports 82 names; 17 of the 20 we need are present. `SourceRun` → `ConnectorRun` and `SourceHealth` → `ConnectorHealth` are renames touching ~133 references. We do not use `content_hash_items`. |
| Enricher — `enrichers.base` (4), `enrichers.derived_edges` (4), `library.linkers` (5), `linkers.declarative` (1), `alias.protocol` (1) | **15** | **Blocked** on #1181 slice 5, the enricher read surface, deferred at our own recommendation. |

One of the fifteen is `okg.substrate.library.linkers._chronos` — underscore-private by
Python convention, and the single item we flagged upstream as most worth absorbing.

**Sprint 11 closed 2026-09-10 with our three reports absorbed.** Its
[closing summary](https://github.com/mitdbg/okg/issues/1698#issuecomment-5616923942)
retains the **connectors-first** example and records the enricher read surface as
**deferred** — both our recommendation on
[#1181](https://github.com/mitdbg/okg/issues/1181#issuecomment-5591973861). Sprint 14
**#1792 owns provider/admin bootstrap** (our
[#1183 findings](https://github.com/mitdbg/okg/issues/1183#issuecomment-5591286914) 2
and 3) and **#1795 owns operator acceptance** including install and chat (our
[#1179 ask](https://github.com/mitdbg/okg/issues/1179#issuecomment-5594656468)).
**#1185 did not pass** and moved to Sprint 14 (#1789); our wheel remains its required
real arm.

**Owed by us, per that summary:** trusted admission policy, a real legacy inventory,
and **acknowledgement of the current v2 contract — the v1 one is now insufficient**.
The v1 digest we acknowledged (`sha256:db38e9bb…`) is still valid for v1; v2 is
`sha256:5cba96ce160313402809651027e0d0c708610542a485c8ac0a88a578b2c69f6a`. Held
deliberately until the real arm can run, since #1185 still lacks the consumer
environment, sealed packages and lifecycle receipts that would make an
acknowledgement outlive the day it is issued.

**The bundle now ships the assistant's system prompt (2026-09-08, PR #632).**
`bundles/cern-team/skills/chat-system-prompt.md`, declared as
`chat.preset.system_prompt_ref`. This is required, not decorative: `okg chat sync`
refuses a deployment that declares neither `mcp_instructions` nor a
`system_prompt_ref`, because Open WebUI never shows the model the MCP server's own
instructions — without it the assistant has graph tools registered and no idea it
has them. Tests assert the prompt is declared, shipped, non-empty, and still names
the six operators. **For #1183: we think that refusal is right, and it argues the
prompt belongs to what a bundle supplies rather than being optional decoration.**

The rest of the chat surface (`chat:` and `search:` blocks, the preset, the MCP tool
list, the `schemas/` slices, the `skills/` slot) was already on `archi_v3` and is
unchanged. The full chain is verified end to end on a fresh database: install →
publish 2,586 nodes → `chat-instance up` → `mcp-serve` on the port the bundle
declares → `chat sync` exiting 0 with the graph tools bound and the prompt applied.

**Three chat findings for #1183, drafted in `docs/outbound/`** and verified against
`dev` @ `1475c87d5`. Worst first: a synced instance opens on a model with **no graph
access**, because `chat.models.default` is both the preset's `base_model_id`
(`sync.py:4937`) and the site default (`sync.py:4432`) while the vendor selects
`default_models.split(',')[0]` — there is no deployment-side workaround, and the
failure is a confident ungrounded answer with no warning. Second, the model provider
cannot be declared at all (`chat-instance up` injects two hardcoded variables,
`instance.py:1253`) and sits in none of the six reconciled surfaces, so `chat sync`
reports green on an instance that cannot answer. Third, the first admin account is
not bootstrapped. Filed separately; recorded here so #1183's design has them.

**Pin bumped 2d528e824 → 1475c87d5, drift closed (2026-09-08).** A 1153-commit
fast-forward, and **the import surface survived it unchanged**: full archi suite
**339 passed**, including `test_alignment_page.py`, which mechanically checks every
one of the ten private `okg.substrate.*` symbols below still resolves. No source
change was needed on our side.

The external-distribution conformance contract landed inside that gap —
`conformance.py` did not exist at the old pin and is 2,132 lines at `dev`, arriving
in `46c42c7e0` via PR #1377 (merged 2026-08-28), with #1406 following on 08-30. That
had made our acknowledgement stale twice over: comment `5440260363` named branch
`f4081329` and schema digest `sha256:301ec97f…`, while the catalog digest is now
`sha256:db38e9bbbe571dbb9efec29192f786d851095ca412f5a5fda17d1af81f62e662`. **Re-issued
2026-09-08** at
[#1185 comment `5591114065`](https://github.com/mitdbg/okg/issues/1185#issuecomment-5591114065),
naming `dev` rather than a branch and vouching for `archi_v3` @ `0aec65f9`, so the
real conformance arm no longer returns
`conformance_external_acknowledgement_pending`. Note for whoever re-runs it: that
acknowledgement binds one Archi revision to one OKG schema digest, so a commit to
either side stales it again — an open question on that comment asks whether it is
meant to be re-issued per run.

**PACT corpus migrated to v5 (2026-09-08).** okg cut over on 2026-09-07 (#1691,
#1687) and we followed the same day, using okg's own `pact migrate`: all nine
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

**Sprint 11 (okg#1698) is the Archi chain.** Its exit criteria are
#1178 → #1179 → #1181 → #1180 → #1183 → #1184 → #1185, with #1185 run on a real
deployment — which their own #1185 note says only the `cern-team` wheel on SubMIT
satisfies. Design notes for #1178, #1179, #1181 and #1185 are posted and
adversarially reviewed; our answers to #1178's seven open questions are
[comment `5590581856`](https://github.com/mitdbg/okg/issues/1178#issuecomment-5590581856).
In flight and directly relevant to us: **#1737** owner-keyed module providers (the
D11 gap this page has flagged since 2026-08-19), **#1740** package v2 with an OKG
compatibility window, **#1742** the SDK connector fact surface and a `partial`
source-run status — which closes our own **#1284**. None had merged as of
2026-09-08.

**Pin bumped, drift closed (2026-08-27).** The 210-commit exposure this page
flagged is resolved: we fast-forwarded to okg `dev` @ `2d528e824` and re-ran the
full archi suite — **337 passed**, including `test_alignment_page.py`, which
mechanically checks every one of the ten private `okg.substrate.*` symbols below
still resolves. The import surface survived the bump unchanged.

**okg#1367 landed, and it removes most of our install friction.** The profile
`schemas:` slot now materializes distribution-owned schema files before the
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

**Sync channel: okg#1178** (established 2026-08-19; surface-change heads-ups land
there). Re-pin validation 2026-08-19 against the strict registry-admission schema
(`8b65a330a`, #1273): all four scratch registry entries (bootstrap fixture,
cmssw-releases, jira, docsite) lint clean, ingest OK, and publish — **zero
refusals**; full test suite 199/199 on the new pin. okg#1006 is merged into `dev`
(2026-08-13): chat, MCP HTTP auth, and principal mapping are now landed premises —
Archi's W9 targets `dev`. **Upstream state re-verified 2026-08-24:** #1275 (the
chat/console slices) is **merged** — the remaining W9 dependency is its parent
#1183, still open. Of our three filed frictions, **#1283 (`output_scope_summary`
hand-duplication) is closed upstream**; #1282 (narrowings outside
`schemas/bridges/` silently ignored — the real bug of the three) and #1284 (no
`partial` SourceHealth status) remain open. Our import surface is CI-gated on the
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
cross-connector reference edges, idempotent re-run. **The reproducible runbook is
`docs/cern-team-demo.md` — this is the okg#1185 release-claim artifact**; run results
+ 7-item friction log in `pact/changes/w2-consolidate-hep-sources/`. For #1179: two
manual steps remain outside the profile contract (wheel schema-slice copy incl. a
pruned operations bridge; role passwords post-migrate) — both flagged in the runbook.
**New substrate finding for this channel: deletion semantics not enforced** — stale
records survived reconcile + `--reset-cursor` under `missing_from_completed_scope`
(demo friction 6; repro in the runbook) — being triaged upstream as a separate
substrate correctness issue (#1178 response, 2026-08-21). The consolidated PR
(#610) is **merged**, and okg#1178 has pinned `archi_v3` @ `728739e6` as the first
real external-consumer baseline for the deployment-product work. Both manual
install steps now have upstream owners: the schema-slice copy + hand-pruned
operations bridge land in `package-and-migrate-archi-products` (digest-bound
schema/ontology/bridge assets), and role/credential provisioning lands in
`apply-and-operate-deployment-instances` (secret refs from the instance,
lifecycle-provisioned roles). The conformance harness will use `cern-team` as its
first real consumer; the final #1185 claim adds artifact-only checks (no
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
comes from the public `okg.deployment` SDK, so #1181's boundary lint has nothing left
to find on that side. What remains below is enricher-only, and is blocked on #1181
slice 5 — the enricher read surface, deferred at
[our own recommendation](https://github.com/mitdbg/okg/issues/1181#issuecomment-5591973861).

**Python imports (all of them).** The first entry is the public SDK; everything
below it is still private substrate, and all of it is enricher-side.

```
okg.deployment:
    NodeFact, EdgeFact, ProgressMarker,
    ConnectorRun, ConnectorHealth, PreflightResult,
    ContentHashProbe, MutableApiProbe,
    file_preflight, credential_preflight, http_preflight, redact
okg.substrate.enrichers.base:     EnrichResult, IncrementalContext
okg.substrate.enrichers.derived_edges:
    DerivedEdgeCandidate, insert_deterministic_edges, mint_edge_id
okg.substrate.library.linkers:    _chronos
okg.substrate.library.linkers.declarative: DeclarativeLinker
okg.substrate.alias.protocol:     AliasMatch
```

Every SDK name above was verified to be the *same object* as the substrate name it
replaced, except `ConnectorRun`, which is a genuinely narrower type: it drops
`next_cursor`, `effective_scope_complete`, `bootstrap_identity` and three
`applied_token_*` fields — none of which Archi ever referenced — and retains
`record_authority_stream`, `record_set` and `record_set_replacement_keys`, the
fields #1181's design note warned would cost incremental sources their changed-slice
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
narrowings; install profiles (`OKG_PROFILES_DIR` cascade) for bundles; the closed
`SourceHealth`/preflight status vocabulary; MCP read surface for validation.

## Mapping: #1178 sub-issues ↔ Archi needs and evidence

| OKG issue | Archi's stake | Evidence we already have for you |
|---|---|---|
| #1179 external schemas/bundles | **The wheel now carries the bundle and its materialised playbooks** (force-include; the repo's playbook symlinks are dereferenced at build), so an install needs no authoring checkout — `archi-profiles-dir` prints the installed payload path. What remains on your side: the profile resolver has **no installed-distribution discovery**, so `OKG_PROFILES_DIR` is still mandatory (cascade = explicit override → env var → `./profiles/` → substrate library → fetch cache, `profile_init.py`). Also still open: shared ontology *modules* (`load_modules_root()` is package-locked) and a versioned distribution contract | ADR §7 ask 2; deployment-scoped schema slices shipping in the wheel (`python/archi/schemas/`) |
| #1180 per-user MCP | Blocks multi-user instances (cms-knowledge-base bundle) | ADR D12 |
| #1181 connector/enricher SDK | Replaces the import surface above; please treat that list as the minimum viable SDK | W1 friction log `pact/changes/w1-prove-the-seam/FRICTION.md` (6 items: scaffold nomos gap, ownership-claim discoverability, role passwords, --json purity, path-vs-name resolution); wheel + scratch-deployment reproduction in the same dir |
| #1182 external review | Unblocks the queue-bot (Redmine/JIRA reply) — our W8, same first consumer you named. Approve-with-edits + expiry confirmed in your scope: exactly what we needed | ADR §5.3, §7 ask 3 |
| #1183 chat | Archi ships per-bundle chat config only | ADR W9 |
| #1184 automation accounting | Matches our kill criterion (defer until >1 real automation) | ADR §7 ask 4 |
| #1185 end-to-end proof | **Our `cern-team` bundle install demo is this proof** — an external, wheel-installed distribution driven end to end on a SubMIT host. Coordinate before building a synthetic consumer | W1 seam proof (done); demo lands with the bundle work |

**Packaged bridges now compose as shipped — fixed on our side, and the fix is
worth copying.** (Previously reported here as a #1179 blocker.) Copying the wheel's
`schemas/bridges/operations.yaml` verbatim fails `okg catalog load` with
`bridge_subtype_unknown`, because its narrowings lack
`optional_when_subtypes_missing` and therefore require every referenced subtype
to be in the composed module set. First failure, verbatim from the executed
runbook (`docs/cern-team-demo.md:106-110`): narrowing
`documentation_page_references_dataset`, child subtype `dataset`. Both
consumers hit it independently and both had to hand-prune: the cern-team demo
(friction 2, `docs/cern-team-demo.md:269-275`) and the comp-ops instance, whose
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
composing fewer narrowings (the okg#1282 hazard). **Suggested for the packaged
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
  a heads-up on okg#1178 referencing this page is enough.
- Bugs we found in shared/canonical code: two TWiki parser bugs in
  `okg-deployments/cms/cms_sources/twiki_eos.py` (the `=code=` regex eats `=` from
  assignments across lines; title-less heading markers absorb the next line) —
  fixed on the Archi side with documented deviation
  (`pact/changes/w2-consolidate-hep-sources/v2-reconciliation-notes.md`) and
  **upstreamed into the canonical copy via mitdbg/okg-deployments#88**.
