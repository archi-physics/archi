# TWiki: wisdqm vs cms — comparison for the shared-source decision

(Explore-agent comparison, 2026-08-12, okg-deployments main@f33a9c4. Decision
context: Luca 2026-08-12 — two ingestors, one EOS reader + one crawler;
cern-twiki hardening not taken yet.)

## Deltas in one table

| Concern | cms twiki_eos.py | wisdqm |
|---|---|---|
| Acquisition | adapter IS the acquirer; whole EOS tree rglob | out-of-band downloader; seeded shifter topics + depth-1 BFS; live-HTML fallback with SSO cookies when no snapshot |
| Markup | strips headings; collapses ALL whitespace (line structure destroyed) | headings→markdown; newlines preserved; HTML tables→pipe tables; img→markdown — generic improvements, not DQM-specific |
| %META (author/version/parent) | parsed | dropped — pure regression on their side |
| Chunking | flat page-level window | section-scoped windows with heading_path + line ranges |
| Extra emissions | chunk→entity reference edges (sites/releases/jira/services) | section/media/table/list nodes; NO entity-mention edges |
| Node ids | twiki:CMS:<Topic> | same — **compatible** |
| Internal state | — | fork is duplicated inside wisdqm itself (adapter + downloader copies) |

## Bottom line

wisdqm cannot adopt cms's class unchanged without real regressions (loses
sections, media, tables, seeded crawl, SSO fallback). The tractable split:

1. **Shared parser core** (kills the 3-way + wisdqm-internal duplication):
   `strip_twiki(heading_style=markdown|drop, whitespace=preserve|collapse)`,
   `parse_meta` (from cms — wisdqm gets author/version back for free),
   wiki-link/bare-wikiword extraction, `twiki_page_id`/node-id minting,
   viewauth→view URL canonicalization, skip patterns.
2. **Two adapters over the core** (per decision): `TwikiEOSSource`
   (cms lift, parameterized: web_root un-hardcoded, seeds+depth optional,
   reference targets injectable) and `TwikiCrawlSource` (live crawl,
   SSO cookies via archi.auth).
3. **Stays wisdqm-local** until a second consumer: section/media/table/list
   node emission, shift-list partitioning, GitLab record co-tenancy.
4. Latent wisdqm bug to fix at their migration: `_topic_snapshot_path`
   basename fallback can bind Web/Topic to a root-level Topic.txt.

OPEN (asked of Luca): whether the shared parser core includes wisdqm's
text-fidelity options (markdown headings / preserved newlines / table+image
preservation as flags) from day one, or ships cms-exact and adds them later.
