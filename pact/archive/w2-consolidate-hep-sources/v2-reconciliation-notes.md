# v2 ↔ okg-deployments reconciliation notes — parked for later

Running list of deltas noticed while porting (doctrine: okg-deployments/cms is the
canonical base; v2 blends case-by-case; maintainer directive 2026-08-12: *note*
conflicts here and move on — architecture first, circle back later). Nothing in
this file blocks assembly.

## JIRA (decided during port)
- **Taken from v2**: JQL helpers (`parse_jira_project_keys`, `quote_jql_string`, 26 LOC), the `<base_url>/browse/<key>` issue-URL attr, JQL query builder for future live fetches.
- **Deferred**: the v2 opt-in **Anonymizer** pass over issue text — moves to the enrichment port. **Gates cutover for `anonymize_data` instances** (raw PII otherwise).
- Cosmetic: v3's JQL quotes project keys (`project = "KEY"`); v2 used unquoted. Semantically identical.

## MONIT live tools — v2 capabilities the cms tools lack (none ported)
From `src/archi/pipelines/agents/tools/monit_opensearch.py` vs `cms_tools/monit_live.py`:
1. Runtime tool factories minting a named tool for any index (cms: four fixed functions; the v3 port's generic cores recover most of this).
2. Skill-markdown injection into the LLM-facing tool description.
3. Full-document retrieval (`_source: True`, recursive flattening) vs fixed summary-field whitelist.
4. LLM-formatted text output (aligned bucket tables) vs structured JSON.
5. 50,000-char output-size guard for the context window (cms: none).
6. Higher result ceilings (50/10 vs 10/5).
7. Per-call `time_field` / `search_type` (cms hard-codes both).
8. Per-client URL+timeout (partially recovered by v3 param cores).
9. Catch-all exception → error string (cms lets ConnectionError propagate).
10. Logging + remediation hints in error text.
Inverse: v2 lacks cms's pagination, token redaction, query-length cap.

## Skills
- cms-compops repo (`configs/comp_ops/skills/`) has `condor_raw_metric.md` (212 l) and
  `rucio_events.md` (217 l) — denser OpenSearch field guides than okg-deployments'
  counterparts. Delta list from the skills port agent to be appended here; merge
  decision deferred.
- `rucio_mcp.md` / `dbs.md` describe MCP sidecars (external images) not yet in v3 —
  not ported; revisit when those sidecars enter the picture.

## Chat-era features (recorded in ADR/spec, listed here for one-stop reference)
- Personal playbooks → Open WebUI per-user prompts (verify at W9); per-user enablement
  of shared packs + invocation analytics + agent-drafted playbooks: dropped, accepted
  losses pending W9 sign-off.
- v2 indico on-demand ingest tool (`indico_ingest.py`) — noted in the indico port
  docstring, code not taken.

## Substrate friction already filed as asks (okg-asks-drafts.md)
- No `partial` in SourceHealth's closed status vocabulary (asks file, candidate 8).
- Narrowings outside schemas/bridges silently ignored (ask 6); output_scope_summary
  duplication (ask 7).

## Skills — cms-compops richness delta (from the skills port, 2026-08-12)
Different genre, strictly additive to the 18 ported okg skills (graph-retrieval
discipline); the compops files are index-level field guides for the live tools:
- `condor_raw_metric.md` (212 l vs ported 38 l): monit_prod_condor_raw_metric*
  envelope + ~90 fields (time-field guidance, CMS_JobType/Workflow/Campaign,
  CpuEff formula, GLIDEIN_*, Chirp metrics), 6 query patterns, 8 agg recipes.
  Relevant the moment archi_monit_* live tools are wired. Needs light
  de-site-ification (T2_US_MIT, fnal hostnames) if merged.
- `rucio_events.md` (217 l vs 52 l combined): full event-type taxonomy incl. the
  gotcha that deletion events use data.rse not src/dst_rse; RSE conventions;
  aggregate parameter contract. Same de-site-ification note.
- `rucio_mcp.md` / `dbs.md`: describe MCP sidecars (registry.cern.ch images)
  absent from v3 — revisit with the sidecar decision; dbs.md carries a vocms0014
  drafting marker to strip if ported.

## TWiki parser bugs shared by the canonical cms original (fixed in v3, upstream candidates)
- =code= regex strips = from assignments across lines (cms twiki_eos.py:442) and title-less heading markers absorb the next line — both fixed in archi with documented deviation; cms/okg-deployments may want the same fixes.

## Enrichment port oddities (2026-08-21)
- Historical `cms_*` names kept in enricher `name` attrs / alias backend ("cms_projection")
  / linker default: uuid5 dedupe keys + progress rows embed them; renaming re-keys the
  comp-ops parity corpus. Pinned by test.
- Original warts preserved verbatim: duplicate "jira_issue" in JiraChunkReferenceRollup
  reads_subtypes; jira_affects couples to the `jira:` node-id prefix; misindented attrs
  dict in dqm_run_range.
- v2 anonymizer signoff heuristic deletes any <=3-word <p> (IGNORECASE) — original
  behavior kept; watch at cutover.
- okg venv has spaCy but NOT en_core_web_sm: default-config NER would attempt a network
  download mid-ingest (download_missing_model=True kept from v2; instances should
  pre-install the model or disable NER).
