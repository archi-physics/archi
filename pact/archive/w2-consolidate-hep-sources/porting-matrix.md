# W2 porting matrix — every source, where it comes from, where it lands

Provenance refs: okg-deployments `main@f33a9c4` (`cms/` tree), archi `dev@28b977d1`
(Scrapy layer) and `main@9c9e1cb0`, okg `dev@21c5b8c3e`. Per ADR 0001 invariant 7:
everything is *rewritten* against the current `SourceAdapter` + registry contract
(sound change probes, producer authority, admission mode), never transplanted; moves
and rewrites are separate commits. W2 does **not** switch the comp-ops instance to
the package — that is W7. The comp-ops instance must stay green throughout (D10).

## Packaging (decided here, ratified by this change's approval)

The v3 package lives under **`python/`** (`python/pyproject.toml`,
`python/archi/`), building `archi-3.x` wheels without touching v2's root
`setup.py`/`pyproject.toml` — v2 stays deployable from the same branch
(invariant 2). W10 flattens `python/` to the repo root when v2 dies. The W1
spike (`pact/changes/w1-prove-the-seam/spike/`) is the seed: its
`archi/sources/cmssw.py` merges into the real `cmssw.py` port below.

## Sources — from `okg-deployments/cms/cms_sources/` (22 classes, 9,328 LOC)

| From (LOC) | To | Probe | Credentials (instance-side) | Notes |
|---|---|---|---|---|
| `jira.py` JiraIssueSource (505) | `archi/sources/jira.py` | mutable_api | `CERN_JIRA_TOKEN` | **merge** with v2 collector (`src/data_manager/collectors/tickets/integrations/jira.py`, 236) + `src/utils/jira.py` JQL helpers (26). Parity family. |
| `docs.py` DocumentationSource + CMSWebDocsSource (1,396) | `archi/sources/docs.py` | discovery_crawl | SSO cookie for cmsweb | **drop dead `GitHubFileContentSource`** and fix the stale `cms.github-file-content` invariant reference while porting. Parity families (docsite, gitlab_docs). |
| `twiki_eos.py` (515) | `archi/sources/twiki.py` | discovery_crawl / content_hash | `TWIKI_EOS_ROOT` or crawl auth | **three-way merge**: cms `twiki_eos.py` ⊕ `cern-twiki/cern_twiki/source.py` (1,242) ⊕ `wisdqm/wisdqm_sources/docs.py` parser. Two acquisition modes: `eos_snapshot` and live crawl (archi-dev TWiki spider informs the crawl mode). One parser. |
| `monit.py` 4 classes (1,931) | `archi/sources/monit.py` | mutable_api | `MONIT_GRAFANA_TOKEN` | SAM, Condor, Rucio transfers, Rucio datasets. |
| `indico.py` (385) | `archi/sources/indico.py` | discovery_crawl | Indico token (optional) | merge v2 on-demand tool `indico_ingest.py` (162) as a fetch mode. |
| `cric.py` (349) + `cric_core.py` (279) | `archi/sources/cric.py` | discovery_crawl | none (cache/public) | two classes, one module. |
| `cmssw.py` (324) | `archi/sources/cmssw.py` | reference_catalog | none | merge W1 spike's releases.map fetch + cms version's `supersedes` edges. W1 friction 6: needs a mutable_api-style probe for the remote fetch path. |
| `conddb.py` (269) | `archi/sources/conddb.py` | reference_catalog | `CONDDB_COOKIE_FILE` | global tags. |
| `dbs.py` (258) | `archi/sources/dbs.py` | reference_catalog | X509 proxy | dataset catalog. |
| `dqm.py` (281) | `archi/sources/dqm.py` | discovery_crawl | none/cert | data certification + runs. |
| `gocdb.py` (280) | `archi/sources/gocdb.py` | discovery_crawl | none | downtimes joined to CRIC. |
| `siteconf.py` (442) | `archi/sources/siteconf.py` | discovery_crawl | `CERN_GITLAB_TOKEN` | per-site SITECONF. |
| `wmstats.py` (257) | `archi/sources/wmstats.py` | mutable_api | X509 | workflows. |
| `hypernews.py` (586) | `archi/sources/hypernews.py` | discovery_crawl | cookie | forum threads. |
| `github_repos.py` (193) | `archi/sources/github_repos.py` | reference_catalog | `GITHUB_TOKEN` (optional) | repo identity nodes. |
| `preflight.py` CMSPreflightSource (674) | `archi/auth/preflight.py` | live_overlay | the whole credential surface | emits no graph facts; generic CERN auth probes (SSO cookie, X509, TLS, tokens). |
| `_cache.py` (207) | `archi/auth/cache.py` (landed there with task.w2.auth; matrix originally said `archi/sources/_cache.py`) | — | — | unified with the wisdqm 55-LOC fork; `resolve_repo_path` base is explicit param → `ARCHI_DATA_ROOT` → cwd (no repo-layout assumptions, no hardcoded operator paths). `json_record_count`/`cache_preflight_result`/`cache_source_health` deliberately dropped — they move with the individual source ports. |
| `alias.py` CMSProjectionAliasBackend (192) | `archi/enrichment/alias.py` | — | — | type-aware alias resolution. |

## From archi v2 (take `origin/dev` for anything scraper-shaped)

| From | To | Notes |
|---|---|---|
| dev Scrapy `AuthProvider` + middlewares (CERN SSO) | `archi/auth/sso.py` | replaces main's `sso_scraper.py` (superseded). |
| dev anonymizer (markitdown-aware) + `metadata.py` + `slide_converter.py` | `archi/enrichment/` | dev anonymizer, not main's 83-LOC version. |
| dev Discourse spider | `archi/sources/discourse.py` | **gated**: cms legacy inventory marks Discourse `defer`; port only when an instance asks (rule of demand, not speculation). |
| `monit_opensearch.py` (667, agent tool) | `archi/tools/monit.py` | merge with `cms/cms_tools/monit_live.py` (4 MCP-exposed live tools, `boundary: external_live`). |
| redmine collector (192) + mailer-adjacent ingest | `archi/sources/redmine.py` | **post-first-draft** (SubMIT's ticket system; parity contract has redmine disabled for submit76). |

## Enrichment — from `cms/enrichers/` (~957 LOC) + config blocks

| From | To |
|---|---|
| `chunk_reference_rollup.py` (320), `jira_affects.py` (230), `dqm_run_range.py` (139), `meeting_document_reference_rollup.py` (241), `declarative.py` global-tag↔release (26) | `archi/enrichment/` |
| 8 extractor regexes + `identifier_patterns.yaml` + `linker_config.yaml` + `cross_links/` from `cms/deployment.yaml` | `archi/enrichment/defaults/` as importable data an instance's config selects/overrides (plus wisdqm's `hlt_path`, `l1t_seed`, `dqm_workspace` patterns — second call site exists) |

## Not ported (with reasons)

- v2 `main` scraper layer — superseded by `dev`'s Scrapy rewrite.
- `sso_scraper.py` (main) — superseded by dev AuthProvider.
- `piazza.py` — retired (ADR 0001 W8).
- vectorstore / embeddings — OKG owns retrieval.
- cern-twiki's consistency-checker stack — OKG-side dogfooding, only its TWiki *parser* is taken.
- `external_live_authority` registry entry — already a framework class, nothing to port.

## Order of implementation (tasks in pact.yaml)

1. Packaging skeleton (`python/`) + CI wheel-build lane (needs `OKG_REPO_TOKEN` secret for okg-dependent tests — ask Luca).
2. `archi/auth/` (preflight, SSO, cache util) — everything else depends on it.
3. Parity families: `jira`, `docs` — these are Archi's current corpus (per `a2rchi-strict-source-parity.yaml`: allowed families are jira, docsite, gitlab_docs).
4. `twiki` three-way merge.
5. `monit` sources + `archi/tools/monit.py` live tools.
6. Catalogs: `cric`, `cmssw` (merge W1 spike), `dbs`, `conddb`, `dqm`, `gocdb`, `siteconf`, `wmstats`, `github_repos`, `hypernews`, `indico`.
7. `archi/enrichment/` (enrichers + defaults + alias backend).

Each source lands with: unit tests (fixture-driven, runnable offline), a registry-entry
template (documented in the module docstring), `okg deployment lint` green on a scratch
registry entry, and a provenance line (source repo + ref + what changed) in the commit.
