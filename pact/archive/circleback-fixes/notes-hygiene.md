# Circle-back fixes — hygiene notes (schemas/bundle/playbooks reviewer)

Fixed directly (no behavioral risk):

1. `skills/indico.md` rewritten for the v3 runtime. The old file was v2-only:
   it directed agents to `INDICO_*` MCP sidecar tools, `ingest_indico_event`,
   `search_metadata_index`, the "data-manager", the "vectorstore", and a
   hardcoded `/shared/indico-downloads/<event_id>/` volume — none of which
   exist on a v3 OKG instance (Archi ships no MCP servers). The rewrite
   describes what the `archi.sources.indico` connector projects into the
   graph and how to answer Indico questions from OKG reads, including the
   snapshot-freshness caveat. shared_with_canonical: true (the canonical
   deployment shipped the same v2 playbook).

2. `skills/source_document_exploration.md` — de-site-ification artifact
   "`inspect`, `inspect`" corrected to "`inspect`, `aggregate`" (the
   schema/source-shape pairing; `expand` is covered separately below it).

3. `bundles/cern-team/source-defaults/cmssw_releases.yaml` — stale NOTE
   claimed the cmssw.py docstring template says `remote_id`; the template
   says `domain_key`. Comment corrected, substrate-constraint explanation
   kept.

4. `bundles/cern-team/source-defaults/indico.yaml.example` — the header
   claimed its required narrowings were "already copied by the install
   runbook"; the runbook's pruned bridge copy excludes
   `meeting_minutes_contains_document`. The comment now states exactly
   which narrowing must be copied additionally when enabling the
   connector, and names the ingest-time failure otherwise (okg#1282
   trap). The structural fix (prune list vs per-connector bridge
   fragments) belongs to the sealed-artifact packaging wave.

5. `python/archi/schemas/bridges/sources.yaml` — added the LinkML header
   block (id/name/prefixes/imports) its sibling `bridges/operations.yaml`
   carries; the catalog composer reads only `classes.EdgeNarrowings`, so this
   is consistency-only (verified: parses, guard tests pass).

Deliberately NOT changed (recorded for the ledger):

- `bundles/cern-team/deployment-defaults.yaml` placeholder contact — a
  documented post-install edit; making it an init question is #1179-adjacent
  bundle work, not a bug fix.
- Five `dataset`-subtype skill triggers with no dataset module in the
  cern-team composition — dead-but-harmless routing; the dataset module is
  expected with later bundles.
- Playbook symlinks escaping the bundle dir (S3) — works in-repo; the fix
  belongs in the sealed-artifact materialization contract (okg#1178 told:
  materialization must dereference).
