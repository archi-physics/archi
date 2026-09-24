# Parity-family scratch-ingest demonstration (acc.w2.sources)

**When/where:** 2026-08-11/12, submit82, okg `dev@21c5b8c3e`, scratch deployment
`w1-scratch` (rootless podman postgres :5455), archi wheel `3.0.0a1` built from the
working tree and installed into the okg venv.

**Wired:** registry entries `jira` (JiraIssueSource, `mutable_api`, remote_id/issue_key)
and `docsite` (DocumentationSource, `discovery_crawl`, scoped_locator/url), both
fast-track with full `output_signature` + `output_scope_summary`; fixtures of 4 JIRA
issues (mixed flat-cache and REST shapes, people, comments, cross-links) and 3 doc
pages (one repo-backed, one citing a fixture issue); ontology = `person` + `extraction`
modules composed in the manifest, plus the packaged deployment slice now shipped as
`archi/schemas/sources.yaml` + `archi/schemas/bridges/sources.yaml` (subtypes
`jira_issue`, `documentation_page`, `software_repository` — no library module declares
them).

**Results:**
- `okg deployment lint`: **zero findings for both new sources** (pre-existing info
  findings on the scaffold fixture/cmssw entries unchanged).
- `okg ingest`: all sources OK; **published**
  `okg:w1-scratch:branch:default:gen:20260812T005035657166Z:669af8640cb2`.
  jira: 11 nodes / 14 edges; docsite: 7 nodes / 5 edges including the cross-source
  `document_chunk → references → jira:ARCHI-101` edge.
- Idempotence: second ingest run produced an identical write set, 0 retractions.
- Read-back: `okg search --query "tachyon buffer drain stalled"` returned 4 hits
  spanning both sources; `okg trace node jira:ARCHI-101` shows full attrs.

**Bug found and fixed in-branch:** JIRA REST `fields.parent` dicts leaked their Python
repr into chunk text (`jira.py` parent_key built with bare `str()`); now routed through
`_name_or_key`, regression test `test_api_parent_dict_yields_key_not_repr`.

**Friction recorded** (feeding okg-asks-drafts.md asks 6–7): deployment-schema
narrowings outside `schemas/bridges/` are silently ignored — lint green, ingest-time
`ProducerPolicyViolation`; `output_scope_summary` must hand-duplicate
`output_signature`; the original docstring templates lacked the sync block and scope
summary — templates now carry the ingest-proven versions and the three prerequisites.
