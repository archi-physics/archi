# circleback-fixes — catalog/feed connector deviations (notes-catalogs)

Fixes for the confirmed adversarial-review findings in the catalog/feed
connectors (hypernews, siteconf, cric, cmssw, conddb, dbs, dqm, gocdb,
wmstats, indico). Each entry documents the finding, the fix, the
behavior change, and whether the bug is shared with the frozen canonical
okg-deployments/cms copy (`main@f33a9c4`) — where it is, the fix is a
deliberate deviation from canonical behavior.

Guiding rule for every fix: the substrate retracts records missing from
a completed scope (`missing_from_completed_scope`), so a false
`completed_scope` claim mass-retracts good records. Partial data with
`completed_scope=False` is fine; claiming completeness is not.

Vocabulary note: the review prescribed a "degraded" health status, but
`okg.substrate.library.sources.base.PREFLIGHT_STATUSES` has no such
status (closed set: ok / skipped_optional / missing_credential /
auth_failed / tls_failed / endpoint_failed / cache_missing /
not_applicable / acquire_stale / acquire_failed). Partial failures
therefore follow the repo's existing docs.py/twiki.py partial-crawl
idiom: emit the surviving records, `status="endpoint_failed"`,
`completed_scope=False`, details in `reason`. Benign degradations
(item skips with survivors, cap truncation) keep `status="ok"` but
forfeit the scope claim.

## 1. BLOCKER — hypernews: per-forum listing failure swallowed

- **Finding:** `_fetch` caught per-forum listing errors with
  `except Exception: continue` while `run()` reported ok +
  `completed_scope=True`; the failed forum's threads were then
  retracted.
- **Fix:** `_fetch` returns a `_FetchOutcome` recording
  `failed_forums`; any forum failure → `completed_scope=False`,
  `status="endpoint_failed"`, failed forums named in `health.reason`
  (endpoint_failed for all-fail too — see vocabulary note). Surviving
  forums' threads are still emitted.
- **Behavior change:** partial crawls no longer look healthy/complete
  and are not persisted to the records cache (see #2).
- **Shared with canonical:** yes — deliberate deviation.
- **Test:** `test_hypernews.py::test_forum_listing_failure_degrades_and_never_claims_scope`

## 2. BLOCKER — hypernews: total fetch failure persisted an empty cache

- **Finding:** `_fetch_and_cache` wrote `records.json` even when the
  fetch produced zero records, claiming ok/complete over zero records;
  later runs replayed the poisoned empty cache forever.
- **Fix:** cache writing split out (`_write_cache`) and performed only
  after a fully successful, untruncated crawl. Total failure (all
  forums failed, cookie unreadable, or zero threads parsed across all
  configured forums) → `endpoint_failed`, no facts, no scope claim, no
  cache write. A pre-existing empty/unusable cache is also refused at
  read time (`endpoint_failed`, reason says to delete the cache to
  force a re-fetch) instead of replaying as an empty complete scope.
- **Behavior change:** zero-record crawls are failures, not empty
  successes; empty caches no longer replay. A dict-shaped cache without
  a `records` key now raises `ValueError` (was: silently read as zero
  threads). Configuring `max_threads` now means the cache is never
  written (a truncated snapshot must not replay as complete), so every
  run re-fetches.
- **Shared with canonical:** yes — deliberate deviation.
- **Tests:** `test_hypernews.py::test_total_fetch_failure_writes_no_cache_and_fails_loud`,
  `::test_zero_threads_across_forums_is_failure_not_empty_success`,
  `::test_empty_cache_refuses_complete_scope`

## 3. MAJOR — hypernews: failed hydration emitted a blanked record

- **Finding:** `_hydrate` returned the un-hydrated listing stub on
  fetch failure, blanking body/author and dropping chunks under a
  complete scope.
- **Fix:** failed hydrations return `None`, are dropped from the
  emission, counted in `_FetchOutcome.failed_hydrations`, and force
  `completed_scope=False` + `endpoint_failed` with the drop count in
  `health.reason`; the partial result is not cached.
- **Behavior change:** a transient thread-fetch failure leaves the
  previous good record in place (no scope claim → no retraction)
  instead of overwriting it with a blank.
- **Shared with canonical:** yes — deliberate deviation.
- **Test:** `test_hypernews.py::test_failed_hydration_drops_record_instead_of_blanking_it`

## 4. MAJOR — siteconf: HTTP-200 empty project list → retract-all

- **Finding:** preflight validated only `/api/v4/user`; a token
  without group visibility yields an HTTP-200 *empty* project list,
  which became ok/complete over 0 records.
- **Fix:** (a) in `run()`, a live fetch that produces no records
  (empty project list, or projects listed but zero SITECONF records
  parsed) → `endpoint_failed`, no facts, no scope claim; (b) preflight
  now also probes `/api/v4/groups/<id>/projects?per_page=1` —
  401/403 → `auth_failed`, empty/non-list/HTTP>=400 →
  `endpoint_failed`.
- **Behavior change:** a token that can log in but cannot see the
  SITECONF group now fails preflight and can never produce an empty
  completed scope.
- **Shared with canonical:** yes — deliberate deviation.
- **Tests:** `test_siteconf.py::test_empty_project_list_fails_loud_instead_of_empty_scope`,
  `::test_zero_parsed_records_from_projects_fails_loud`,
  `::test_preflight_group_probe_flags_invisible_group`

## 5. MAJOR — cric: error-shaped responsibilities cache read as empty

- **Finding:** the responsibilities payload was read as
  `.get("result", [])`, so an error-shaped/drifted cache silently meant
  "no responsibilities" under ok/complete (retracting all operators).
- **Fix:** `_responsibilities_result` requires a dict with a `result`
  list and raises `ValueError` otherwise (run fails loudly; the
  matching top-level-shape checks in the sibling sources also raise).
  `CRICSource.preflight` catches the drift and reports
  `endpoint_failed` instead of raising.
- **Behavior change:** a drifted responsibilities cache aborts the run
  instead of emptying the operator set.
- **Shared with canonical:** yes — deliberate deviation.
- **Test:** `test_cric.py::test_drifted_responsibilities_payload_fails_loud`

## 6. MAJOR systemic — item-level parse loops silently skipped records

- **Finding:** cmssw, conddb, dbs, dqm, gocdb, wmstats, indico, and
  hypernews (cache path) skipped non-dict/missing-key items with
  `continue`; schema drift could turn a full cache into zero records
  under a healthy completed-scope run.
- **Fix:** each `_records()` loop now counts skips
  (`_records_with_skips` / `_records_with_details`); a new shared
  policy helper `archi/sources/_cache_report.py::skipped_items_status`
  (plus a `skipped_count` parameter on `cache_source_health`)
  implements: any skipped item → `completed_scope=False` with the skip
  count in `health.reason` (status stays ok when survivors exist);
  zero parsed records from a non-empty payload →
  `status="endpoint_failed"`. Per-item tolerance is kept — one bad
  record still never fails the run. Indico's duplicate-event dedup is
  deliberately *not* counted as a skip (the record is still
  represented). GoCDB additionally counts (rather than crashes on)
  non-numeric `downtime_id` values.
- **Behavior change:** drifted caches degrade loudly instead of
  silently retracting; `skipped_items_status` is a new helper absent
  from the canonical `_cache.py`.
- **Shared with canonical:** yes — deliberate deviation.
- **Tests:** `test_{cmssw,conddb,dbs,dqm,gocdb,wmstats,indico,hypernews}.py::test_skipped_cache_items_never_claim_scope`
  (hypernews: `test_cache_skips_unparseable_items_without_scope_claim`)
  and `::test_all_items_unparseable_is_endpoint_failed` per family.

## 7. MINOR — caps truncate while claiming complete scope

- **Finding:** cmssw `limit`, hypernews `max_threads`, siteconf
  `max_projects` truncate to the newest/first N while claiming
  complete scope — a sliding retention window where each upstream
  append retracts the oldest in-window record.
- **Fix:** each source now detects when its cap actually truncated
  (`_records_from_map` compares against the full parse;
  `_parse_threads`/`_list_projects` report a truncation flag) and
  forfeits the scope claim with the cap named in `health.reason`.
  A cap that does not truncate keeps the claim.
- **Behavior change:** capped runs are cursor-style updates, never
  scope authorities; hypernews additionally stops caching truncated
  crawls (see #2).
- **Shared with canonical:** yes — deliberate deviation
  (cmssw `limit` is archi-W1-only; the truncation-forfeits-scope rule
  itself is the deviation for hypernews/siteconf).
- **Tests:** `test_cmssw.py::test_limit_truncation_never_claims_scope`,
  `test_hypernews.py::test_max_threads_truncation_never_claims_scope`,
  `test_siteconf.py::test_max_projects_truncation_never_claims_scope`

## 8. MINOR — conddb: change probe missed cmssw_records_path

- **Finding:** the change probe covered only `records_path`, but
  `run()` also reads `cmssw_records_path` (release-target gating); a
  cmssw-cache update changed emitted facts without firing the probe.
- **Fix:** the probe's path set and config now include
  `cmssw_records_path` (the probe tolerates a missing file). The
  `cache_paths` property is deliberately unchanged: it defines this
  source's own record authority for preflight and `content_hash`,
  and the cmssw cache is optional (a missing file would make
  `content_hash` raise and preflight report `cache_missing`).
- **Behavior change:** editing the cmssw cache now busts the conddb
  probe token.
- **Shared with canonical:** yes — deliberate deviation.
- **Test:** `test_conddb.py::test_change_probe_covers_cmssw_cache`

## 9. MINOR — gocdb: scheme-prefixed endpoint host parse

- **Finding:** `endpoint.split("/", 1)[0].split(":", 1)[0]` turned
  `https://host.cern.ch:8443/path` into `https:`, silently breaking
  hostname → service `affects` matching.
- **Fix:** new `_endpoint_host` uses `urllib.parse.urlparse(...)
  .hostname` when a scheme is present; bare `host[:port][/path]`
  endpoints keep the original split (canonical behavior preserved for
  the schemeless case).
- **Behavior change:** scheme-prefixed endpoints now produce
  `affects` edges; a `https:` key no longer pollutes the lookup.
- **Shared with canonical:** yes — deliberate deviation.
- **Test:** `test_gocdb.py::test_scheme_prefixed_endpoint_maps_hostname_to_service`

## 10. MINOR — indico: chunk node_id collisions across events

- **Finding:** chunk `node_id` was a content hash of the chunk text
  alone, so identical boilerplate in different events collided onto
  one node with contradictory parents.
- **Fix:** chunk ids are salted with the parent document node id +
  chunk index (`chunk:sha256(f"{doc_id}\0{index}\0{text}")[:16]`,
  hypernews-pattern literal backslash-zero separator).
  `content_sha256` still hashes the text alone.
- **Behavior change:** **all indico chunk node ids change — expect a
  one-time churn (retract old ids / insert new ids) on the next
  re-ingest of any deployment carrying indico chunks.** Cross-event
  dedup of identical boilerplate is intentionally given up in favor of
  parent-correct provenance.
- **Shared with canonical:** yes — deliberate deviation.
- **Test:** `test_indico.py::test_identical_pdf_text_in_different_events_gets_distinct_chunks`

## Round-2 review deltas (same discipline, same branch)

An independent adversarial review of the merged wave confirmed the
fixes hold and surfaced these deltas, fixed here:

- **MAJOR — cmssw releases.map zero-parse (extends finding 6):**
  `parse_releases_map` filtered unparseable lines uncounted, so a
  non-empty map whose every line failed the label pattern (e.g. a
  `label=` key rename) still claimed ok + `completed_scope=True` over
  zero records — in the exact mode the shipped cern-team bundle uses.
  The map parser now counts filtered lines as skips
  (`_parse_map_with_skips`): survivors forfeit the scope claim, and
  zero parsed from a non-empty map is `endpoint_failed`.
  Shared with canonical: no (the releases.map mode is archi-W1-only).
  Tests: `test_cmssw.py::test_map_skipped_lines_never_claim_scope`,
  `::test_map_zero_parse_from_nonempty_map_is_endpoint_failed`.
- **MINOR — cern-team bundle default `limit: 300` removed
  (consequence of finding 7):** the real releases.map has far more
  than 300 releases, so the default install would truncate every run,
  leaving `completed_scope` permanently False and deletion semantics
  permanently disabled. The bundle now ships without a limit (the live
  demo ingested the full map fine); the yaml comments why.
  File: `bundles/cern-team/source-defaults/cmssw_releases.yaml`.
- **MINOR — siteconf group probe missed `include_subgroups=true`
  (extends finding 4):** the new preflight group probe listed only
  direct group projects while the crawl includes subgroups, so a group
  whose projects live only in subgroups got a false-negative preflight
  that permanently blocked `run()`. The probe now mirrors the crawl's
  listing parameters. Test:
  `test_siteconf.py::test_preflight_group_probe_includes_subgroups_like_the_crawl`.
- **LOW — hypernews preflight vs run() on an empty cache (extends
  finding 2):** preflight reported ok/record_count=0 for an empty
  well-formed cache that run() then refused with `endpoint_failed`.
  Preflight now returns the same `endpoint_failed` refusal (shared
  `_empty_cache_reason`). Test:
  `test_hypernews.py::test_preflight_empty_cache_matches_run_refusal`.
- **Coverage — hypernews live happy path:** added the missing test
  that a clean multi-forum crawl reports ok + `completed_scope=True`,
  persists the records cache, and a second run replays that cache and
  still claims scope.
  Test: `test_hypernews.py::test_clean_live_crawl_claims_scope_writes_cache_and_replays`.

### Empty-cache asymmetry (recorded rationale, no code change)

The reviewer flagged that dbs/conddb/wmstats/dqm/gocdb/indico still
claim a complete *empty* scope when their cache is an explicitly-empty
(`[]`) file, unlike hypernews's new refusal. This asymmetry is
intentional: those catalog caches are operator/fetcher-authored inputs
— the authority over "this catalog is currently empty" lies outside
the connector, so an explicitly-empty file is a legitimate statement
of scope (and preflight already surfaces it as `skipped_optional`
where the cache is optional). HyperNews, by contrast, writes its own
cache from its own crawl, so an empty file there is by construction a
crawl failure artifact (self-poisoning risk), never an operator
statement — hence the stricter refusal rule applies only there.

## Explicitly not attempted

- `SourceRun.record_set` plumbing (review finding "C3") — substrate-side
  coordination, out of scope for this change.
- No files outside the catalog/feed domain were touched (no docs.py /
  monit.py / jira.py / twiki.py, no enrichment/, no
  docs/okg-alignment.md — the okg import surface is unchanged). The
  round-2 deltas additionally touched
  `bundles/cern-team/source-defaults/cmssw_releases.yaml` (the cmssw
  connector's shipped default), per the reviewer's direction.
