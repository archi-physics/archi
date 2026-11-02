# circleback-fixes — live connectors & live tools (cb-fix-live)

Fixes for the confirmed adversarial-review findings in the live-proven
connectors (`python/archi/sources/monit.py`, `docs.py`, `jira.py`) and
live tools (`python/archi/tools/monit.py`). Branch `cb-fix-live` off
`728739e6`.

Core discipline applied throughout: the substrate retracts records
missing from a completed scope (`missing_from_completed_scope`), so a
false completed-scope claim causes mass retraction. Every fix moves in
the safe direction — **never claim `completed_scope` when any input
failed, was truncated, capped, or is indistinguishable from an upstream
outage**. Partial data with `completed_scope=False` is fine.
`SourceHealth.status` values come from okg's closed vocabulary
(`okg.substrate.library.sources.base.PREFLIGHT_STATUSES`); there is no
`degraded`/`partial`, so degraded-but-emitting runs report
`endpoint_failed` (the same mapping docs.py's partial-crawl path already
used).

Most fixes are deliberate deviations from the frozen canonical
`okg-deployments/cms` copies (`cms/cms_sources/monit.py`,
`cms/cms_sources/docs.py`, `cms/cms_sources/jira.py`,
`cms/cms_tools/monit_live.py` at `main@f33a9c4`) — noted per finding as
`shared_with_canonical`.

---

## 1. BLOCKER — rucio-dataset live page cache replays stale days forever

- **Finding**: `monit.py` `_load_live_page`/`_query_fingerprint` keyed
  the live page cache on a date-free fingerprint (hashing the literal
  `"now-24h"`/`"now"` strings) and never invalidated entries. After one
  successful live run, every later run replayed the cached pages with
  zero HTTP and still claimed a completed scope: day 2 emitted day 1's
  datasets stamped with day 2's date.
- **Fix**: the fingerprint now includes the resolved snapshot date
  (`_today()`), with `query_version` bumped to
  `rucio_dataset_composite_v3` so all date-free v2 entries are
  invalidated. Additionally each entry's stored `cached_at` is honored:
  it must be parseable, on the same UTC day as the run's snapshot date,
  and younger than the new `page_cache_ttl_hours` constructor/registry
  param (default 24.0). Entries without/with corrupt `cached_at`
  (legacy caches) never satisfy a run. Partial pages are never written
  to the cache (see finding 2).
- **`cache_live_pages: true` default — decision**: kept `true`,
  documented in the module docstring and registry template. With
  date-scoped keys + same-day TTL, the cache can only replay the
  current snapshot day's pages, which is exactly its intended role
  (crash/resume within a day without re-querying MONIT). Flipping the
  default to `false` would drop that resume capability without any
  additional safety, since cross-day replay is now impossible.
- **Behavior change**: a new day's run always issues fresh HTTP;
  same-day reruns still replay (resume) within the TTL; existing v2
  cache directories become dead weight (never read) and are refetched
  once.
- **Tests**: `test_rucio_dataset_page_cache_is_day_scoped`,
  `test_rucio_dataset_page_cache_honors_ttl`,
  `test_rucio_dataset_page_cache_entry_without_cached_at_is_stale`.
- **shared_with_canonical**: yes (`cms_sources/monit.py` has the same
  date-free fingerprint and no `cached_at` check).

## 2. MAJOR — partial OpenSearch responses accepted as complete

- **Finding**: `_monit_msearch` consumers never checked
  `timed_out: true` or `_shards.failed > 0`; an HTTP-200 partial
  response fed a run that claimed a completed scope (all four sources:
  SAM/Condor/Transfer/Dataset).
- **Fix**: new `_ResponseQuality` assessment
  (`timed_out`/`_shards.failed`/terms truncation, see finding 3) is
  computed for every live response and for cached raw `_msearch`
  payloads (a cached partial response is equally partial). A degraded
  response still emits the records it carries but reports
  `endpoint_failed` (closed vocabulary has no `degraded`) with details
  in `reason`, and `completed_scope=False`. The dataset overlay's
  *streaming* path cannot retroactively downgrade its already-returned
  health/scope claim, so it raises on any partial page — degraded first
  page becomes an `endpoint_failed` run before any facts are produced;
  a degraded later page raises mid-stream so the runner fails the run —
  and never caches partial pages.
- **Behavior change**: timed-out/shard-degraded snapshots no longer
  become retraction baselines; the emitted data is still available for
  cursor-style consumption.
- **Tests**: `test_sam_live_timed_out_response_degrades_scope`,
  `test_condor_live_shard_failures_degrade_scope`,
  `test_cached_raw_response_with_partial_markers_degrades_scope`,
  `test_rucio_dataset_live_partial_first_page_fails_run`,
  `test_rucio_dataset_live_partial_later_page_raises_midstream`.
- **shared_with_canonical**: yes.

## 3. MAJOR — terms-aggregation caps silently truncate

- **Finding**: `sum_other_doc_count` / `doc_count_error_upper_bound`
  were never read; a bucket-limited terms aggregation (`max_sites`,
  `max_src_sites`, `max_replica_rses`, ...) that dropped buckets still
  claimed a completed scope.
- **Fix**: `_ResponseQuality` recursively walks the aggregation tree
  (nested aggregations live inside each bucket) and flags any
  `buckets` aggregation with positive `sum_other_doc_count` or
  `doc_count_error_upper_bound`, naming the aggregation path and both
  counters in the health reason. Truncation ⇒ `completed_scope=False`
  (same degraded handling as finding 2; in the dataset streaming path
  it raises).
- **Behavior change**: capped snapshots are marked partial instead of
  silently retracting everything beyond the cap.
- **Tests**: `test_transfer_live_terms_truncation_degrades_scope`,
  `test_nested_terms_truncation_is_detected`.
- **shared_with_canonical**: yes.

## 4. MAJOR — zero-bucket result claimed a complete empty scope

- **Finding**: with `ignore_unavailable: true` wildcard-index queries,
  a renamed index or stalled ingestion pipeline returns zero buckets —
  indistinguishable from a genuinely empty window — and `_source_run`
  still claimed the mode's scope: a retract-everything hazard.
- **Fix**: zero records ⇒ `skipped_optional` **with**
  `completed_scope=False` in `_source_run` (live and cache paths). The
  dataset streaming path now prefetches page 0 and returns the same
  `skipped_optional`/`completed_scope=False` run on a zero-bucket first
  page instead of claiming a completed empty stream. The module
  docstring bullet that allowed "may still claim the mode's scope (the
  scope is genuinely empty)" is rewritten accordingly.
- **Behavior change**: a genuinely empty window no longer triggers
  retraction of previously ingested records; it reports
  `skipped_optional` and leaves the prior generation in place.
- **Tests**: `test_sam_live_zero_buckets_never_claims_empty_scope`,
  `test_sam_empty_cache_never_claims_scope`,
  `test_rucio_dataset_live_zero_buckets_never_claims_empty_scope`.
- **shared_with_canonical**: yes.

## 5. MAJOR — docs SSO crawl: max_pages truncation claimed complete scope

- **Finding**: `SSOCookieDocsSource._crawl` truncated the sitemap
  frontier *before* crawling and computed `total_urls` post-truncation,
  so a clean truncated crawl looked complete and claimed the scope —
  retracting every un-crawled page.
- **Fix**: `_CrawlOutcome` now carries `sitemap_total` (pre-truncation)
  and `truncated`; when `max_pages` actually truncates, the run emits
  what it crawled, surfaces `sitemap frontier truncated by
  max_pages=N (crawled/total)` in health, and never claims
  `completed_scope` (status stays `ok` — the crawl itself succeeded;
  the scope claim is what changes). The partial-crawl
  (`endpoint_failed`) branch also carries the truncation note.
- **Behavior change**: capped crawls become partial by configuration;
  a cap that does not truncate changes nothing.
- **Tests**: `test_sso_max_pages_truncation_never_claims_scope`.
- **shared_with_canonical**: yes (same pre-crawl slice in
  `cms_sources/docs.py`, which claimed `ok`/complete).

## 6. MAJOR — jira run() never cross-checked meta.json's record_count

- **Finding**: `meta.json` was read only in the cache-missing preflight
  branch; a truncated-but-valid records cache (meta reports 72000, file
  parses 2) claimed a complete scope in `run()`.
- **Fix**: `run()` now compares the parsed record count against
  `meta.json`'s `record_count` (via the existing `_expected_count`,
  which tolerates absent/corrupt meta and a missing key by returning
  `None` — no cross-check, clean run). On a mismatch: facts still
  emitted, `endpoint_failed` with both counts in the reason,
  `completed_scope=False`.
- **Test-fixture note**: `python/tests/sources/test_jira.py`'s
  `_write_caches` used to write `record_count: 72000` beside 1–2
  records; fixtures that want a clean run now carry a matching count
  (the cache-missing preflight fixtures keep 72000, which their
  assertions report on).
- **Behavior change**: truncated caches degrade instead of silently
  retracting the dropped issues.
- **Tests**: `test_meta_record_count_mismatch_never_claims_scope`,
  `test_meta_without_record_count_stays_clean`,
  `test_meta_record_count_match_claims_scope`.
- **shared_with_canonical**: yes.

## 7. MINOR — live tools: ConnectionError / non-JSON 200 escape uncaught

- **Finding**: `monit_search`/`monit_aggregate` caught only
  `Timeout` and `HTTPError`; `requests.ConnectionError` and a non-JSON
  200 body (Grafana HTML error page) escaped as raw exceptions.
- **Fix**: both cores also catch
  `(requests.exceptions.RequestException, json.JSONDecodeError)` and
  return the same structured error payload as the timeout path
  (`MONIT OpenSearch request failed: <ExceptionType>` /
  `MONIT OpenSearch returned a non-JSON response body`). Messages are
  built from the exception type only, never its text (which can embed
  URLs). A missing token still raises `RuntimeError` (harness
  `requires_env` gating, unchanged).
- **Tests**: `test_search_connection_error_is_structured`,
  `test_aggregate_connection_error_is_structured`,
  `test_search_non_json_200_body_is_structured`,
  `test_aggregate_non_json_200_body_is_structured`.
- **shared_with_canonical**: yes.

## 8. MINOR — keyword-then-raw terms retry conflated zero matches with wrong field

- **Finding**: a legitimate zero-bucket `.keyword` result (zero
  matching documents) triggered the raw-field retry, whose fielddata
  error was surfaced as the result.
- **Fix**: zero matches ⇒ valid empty result, returned as-is, no retry.
  The raw-field retry fires only when (a) the `.keyword` attempt failed
  with a *field-related* error (`_looks_like_field_error`: mentions the
  field, or markers like `no mapping found`/`fielddata`/`not
  aggregatable`), or (b) documents matched (`hits.total > 0`) yet the
  `.keyword` aggregation produced zero buckets — the unmapped-`.keyword`
  case (numeric fields such as `data.ExitCode`). A failing raw retry
  never replaces a valid empty `.keyword` result.
- **Deliberate refinement of the finding's letter** (which said "only
  retry on a field-related error, not on empty buckets"): an unmapped
  `.keyword` sub-field on a numeric group_by returns empty buckets
  *without any error*, so an error-only rule would permanently break
  terms aggregation over numeric fields — a real regression for e.g.
  `group_by=data.ExitCode`. The matched-but-bucketless condition keeps
  that path working while fully fixing the reported conflation: a
  zero-match window can never trigger the retry, and a retry error can
  never clobber a valid empty result.
- **Tests**: `test_aggregate_zero_match_empty_buckets_is_valid_result`,
  `test_aggregate_field_error_triggers_raw_retry`,
  `test_aggregate_failing_raw_retry_keeps_valid_empty_result`, updated
  `test_aggregate_keyword_retry_falls_back_to_raw_field`.
- **shared_with_canonical**: yes.

## 9. MINOR — echoed time_window drifted from the actual query body

- **Finding**: `_time_window` applied defaulting/stripping the query
  body did not: blank inputs queried raw `""`/`"  "` while echoing
  `now-24h`/`now` (or an empty string), so the payload claimed a window
  the query never used.
- **Fix**: `_sanitize_time` (strip, then default) runs once at the top
  of both cores and the exact same values feed the query body and the
  echoed `time_window`; `_time_window` no longer defaults or strips.
- **Behavior change**: blank/padded window inputs now query the
  defaults they always echoed; well-formed inputs are unchanged.
- **Tests**: `test_time_window_echo_matches_query_body`.
- **shared_with_canonical**: yes.

## 10. MINOR — docs SSO crawl regex-stripped PDFs/binaries into mojibake pages

- **Finding**: the crawl never checked `Content-Type`; binary sitemap
  entries were tag-stripped into mojibake `documentation_page` records.
- **Fix**: responses whose `Content-Type` media type does not contain
  `html` are skipped and counted (`skipped_non_html` in
  `_CrawlOutcome`, surfaced in health). A missing header is treated as
  HTML (conservative; real `requests` responses always carry one).
- **Scope decision, documented**: skipped non-HTML entries are
  **excluded from the source's scope by design** — they are not crawl
  failures and do not block the `completed_scope` claim. Consequence:
  a previously mis-ingested binary page is retracted on the next clean
  completed-scope run, which is the correct outcome (it never was a
  documentation page).
- **Tests**: `test_sso_non_html_sitemap_entries_skipped_not_ingested`.
- **shared_with_canonical**: yes.

## 11. MAJOR (template hygiene) — docs.py docstring registry templates violated their own prerequisites

- **Finding**: the file's prerequisites (lines 52–53) require
  `output_scope_summary` alongside `output_signature` and a standard
  `sync:` block, yet the `gitlab_docs` and `cmsweb_docs` templates
  omitted `output_signature` + `output_scope_summary` (`gitlab_docs`
  also omitted `sync`), and the `docsite` template omitted `sync` —
  copies of these templates fail strict admission
  (`admission_policy_block_drift`) after a green-looking start.
- **Fix**: all three docstring templates now carry the full
  ingest-proven strict shape, mirroring
  `bundles/cern-team/source-defaults/docsite.yaml`:
  - `docsite`: added the missing `sync:` block, and the
    `jira_records_path` param its (already-uncommented)
    `document_chunk references jira_issue` edge requires — the bundle
    carries both.
  - `gitlab_docs`: added `output_signature` + `output_scope_summary`
    (documentation_page / software_repository / document_chunk with the
    two `contains` edges; reference edges kept in the commented
    optional pattern to match its params) and the `sync:` block.
  - `cmsweb_docs`: added `output_signature` + `output_scope_summary`
    (documentation_page / document_chunk and `documentation_page
    contains document_chunk` only — the SSO crawl emits no
    `software_repository` nodes; reference edges commented, matching
    its params).
  `output_scope_summary` duplicates `output_signature` per the okg#1283
  requirement in every template.
- **Behavior change**: docstring-only (templates operators copy); no
  runtime change.
- **Tests**: none (docstring templates are not executed); covered by
  review.
- **shared_with_canonical**: no — the templates are archi-v3 docstring
  additions; the canonical cms module has no such templates.

---

## Status mapping note (applies to findings 2, 3, 6)

The review asked for "status degraded". `PREFLIGHT_STATUSES` is a
closed vocabulary without `degraded`/`partial`; the chosen mapping is
`endpoint_failed` with the degradation detailed in `reason` and
`record_count` carrying the emitted count — the same mapping the
already-reviewed `SSOCookieDocsSource` partial-crawl path uses for
"data emitted, scope not claimed". What the substrate keys retraction
on — `completed_scope=False` — is exact in every degraded path.
