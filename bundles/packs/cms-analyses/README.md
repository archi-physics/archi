# cms-analyses pack

Links CMS analyses across CDS records, TWiki pages and analysis notes. It
adds no documents of its own: it adds the hubs those documents point at and
the edges between them. Every edge comes from an identifier or a declared
alias already in the records; nothing is inferred by a model.

| Part | Where | What it does |
|---|---|---|
| `physics_vocabulary` source | `sources/physics_vocabulary.yaml` | The 17 Physics Analysis Groups (`pag:HIG`) and 8 Physics Object Groups (`pog:MUON`) as hub nodes. Declared in code, since nothing publishes the list as data. |
| `physics_entities` source | `sources/physics_entities.yaml` | 180 entity nodes (81 techniques, 58 concepts, 41 topics), and a `references` edge from every TWiki page and analysis note whose text matches a declared alias. |
| cross-link rules | `cross_links/*.yaml` | Declarative rules run by okg's `DeclarativeLinker`, listed below. |
| schema slice | `schemas/` | The entity classes, `PhysicsObjectGroup`, and the edge narrowings the rules and the entity source need. |
| playbook | `playbooks/analysis_lookup.md` | How an agent answers "what do we have on HIG-99-001". |

The pack requires `twiki`, `cds` and `analysis-notes`, and reads their
answers (`--twiki-eos-root`, `--twiki-physics-filter`,
`--analysis-notes-mirror-root`). It fetches nothing itself, but the TWiki dump
and the note mirror it reads need a CERN account, so it says
`needs_credentials: true`.

## The rules

| Rule | Joins | By |
|---|---|---|
| `cms_analyses_cds_record_documents_analysis_note` | CDS record and note (both ways) | `cds_record.analysis_code` equals `analysis_note.an_id` |
| `cms_analyses_reference_names_cds_record` | a note's bibliography entry and the CDS record it cites | `reference.analysis_code` equals `cds_record.analysis_code` |
| `cms_analyses_cds_record_belongs_to_pag` | CDS record and its PAG hub | `pag_code` equals the hub `code` |
| `cms_analyses_documentation_page_in_pag` | TWiki page and PAG hub | the PAG code in the page title |
| `cms_analyses_documentation_page_in_pog`, `..._alt` | TWiki page and POG hub | the POG root in the page title (`Muon`, `BTag`, `BTV`, ...) |
| `cms_analyses_pag_pog_same_group` | the PAG and POG hub of one group (both ways) | the `pag_code` the POG hub carries |

A report number with no PAG token (`CMS-NOTE-2099-001`, `CMS-CR-2099-213`)
has an empty `analysis_code` and finds no partner, rather than a wrong one.

TWiki page names drop the hyphens of an analysis id (`HIG99001`), and the
twiki source records the hyphenated id in the page's `analysis_id`. No rule
joins a page to a note by that id yet; today a page and a note meet at their
PAG hub, through the note's citations, and through shared entities.

## Wiring the linker

The packs have no enricher key yet, so the instance adds the linker to its
deployment manifest by hand:

```yaml
enrichers:
  - class: archi.enrichment.physics_cross_links.PhysicsCrossLinker
```

If the block is missing the publish still succeeds and no derived edge
exists. The warn invariant `cms_analyses_enricher_edges` flags that case: it
fires when there is no CDS-to-note edge and no PAG-to-POG edge.

`PhysicsCrossLinker` loads the four files in `cross_links/` from the installed
pack, labels its edges `untrusted` (they derive from user-written text) and
declares the write envelope okg checks at admission. The schema bridge
`schemas/bridges/cms_analyses.yaml` declares every edge triple the rules can
write; `tests/packs/test_cms_analyses.py` holds the two in step.

## Entities

`python/archi/sources/_physics_entities_vocabulary.py` is the list: each entry
is a code, a label and case-insensitive alias regexes. A document names an
entity when an alias matches its text; the edge is one per (document, entity).
Each entity node carries how many notes and pages reached it in the run
(`documents_reached`) and a `selectivity` band: `broad` (1000 or more),
`useful`, or `rare` (150 or fewer). A `broad` entity such as `jet` names most
documents, so it cannot narrow a search alone.

Matching is exact and deterministic, so the source remembers its answers
between runs in `match_cache_path`, keyed on the document text and the
vocabulary. Change an alias and the cache is discarded whole.

The matcher is literal. `jet energy (correction|scale)` matches `jet energy
correction` and `jet energy scale`, not the plural `corrections`. Fix a
miss by editing the alias, not by loosening the matcher.

To add a group or an entity, edit the list in code and the tests; the change
is a commit, not a silent change in what a corpus happened to contain.

## Coverage invariant

`cms_analyses_coverage` (error) requires live hubs and entities of all five
subtypes. Both sources declare their nodes in code, so a missing subtype means
a source did not run.

## Not covered

- Analysis code. The playbook tells an agent to search an analysis id as text
  in the code pack's chunks; no edge joins an analysis to a repository.
- A TWiki page to a note by analysis id (see above).
