# CMS analysis lookup in the OKG

Use this when a question names a CMS analysis by its id (`HIG-99-001`,
`CMS-PAS-TOP-99-003`, `AN-99-120`) or asks what exists for an analysis,
group or technique. The cms-analyses pack links the same analysis across
CDS records, analysis notes and TWiki pages; what it cannot link, say so.

## One analysis, several spellings

An analysis id is `<PAG>-<YY>-<NNN>`. The same analysis appears as:

- `HIG-99-001`: the analysis note directory (`analysis_note`, id
  `an:HIG-99-001`).
- `CMS-PAS-HIG-99-001`, `CMS-HIG-99-001`: a CDS report number
  (`cds_record`, attrs `report_number` and `analysis_code`).
- `HIG99001`, `HIG99001Review`: a TWiki topic name, which drops the
  hyphens because a topic name is a WikiWord (`documentation_page`, ids
  `twiki:<Web>:<Topic>`). Search the joined form as well as the hyphenated
  one; a topic's `analysis_id` attr, when set, holds the hyphenated id.
- `AN-99-120`: a legacy note id. It shares nothing with a CDS report
  number, so it reaches the published record only through citations.

## How the links run

- **Summary and note.** A `cds_record` whose `analysis_code` equals a
  note's `an_id` is linked to it, in both directions. Expand either node
  with the `references` edge.
- **Citation to summary.** A `reference` (a bibliography entry of a note)
  that carries an `analysis_code` links to the `cds_record` with that code.
  This is how a legacy note reaches the published analyses it cites:
  note, then reference, then CDS record. A conference record or a bare CERN
  preprint number has an empty code and links to nothing.
- **Note to TWiki.** A note references a `documentation_page` only when one
  of its bibliography URLs names that topic and the topic was ingested.
- **Group hubs.** Notes, CDS records and TWiki pages link to
  `physics_analysis_group` hubs (`pag:HIG`); TWiki pages whose titles carry
  a POG root also link to `physics_object_group` hubs (`pog:MUON`). The PAG
  and POG hub of the same group (BTV and the b tagging POG, for example)
  are linked to each other, so expand both when asked about a group.
- **Techniques, concepts and topics.** A note or page that names one by a
  declared alias references the `physics_technique`, `physics_concept` or
  `physics_topic` node. A match is a verbatim alias in the document text,
  not an inference. Check the entity's `selectivity`: `broad` entities
  (`jet`, `muon`) name most documents and cannot narrow a search alone.

## How to answer

1. Normalise the id to `<PAG>-<YY>-<NNN>` and search all the spellings
   above. A hit on one spelling does not prove the others are absent.
2. `inspect` the `analysis_note` and the `cds_record`; expand
   `references` from each to find the other, the TWiki pages and the hubs.
3. Cite each claim by node id: the note id and section for what the note
   says, the report number for what was published, the topic name for what
   the wiki says.
4. Say what is missing. No `cds_record` for a note means no summary was
   found in the harvested OAI sets, not that none exists. No TWiki edge means
   the bibliography names no ingested topic, not that no wiki page exists.
5. Code is not linked. To find code for an analysis, search the id as text
   in the code pack's chunks and report what the search returns.
6. Report only the attrs the nodes list; do not infer a group or year
   beyond what the id shape says.

## When NOT to lean on this data

- Analyses that exist only as private notes or unpublished work are present
  only if they are in the note mirror the instance was built from. Absence
  is evidence about the mirror's scope.
- The graph holds the mirror's snapshot. For "latest version" questions,
  check the freshness of the note and CDS sources first.
