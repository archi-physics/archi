"""Archi enrichment layer (task.w2.enrichment).

Deterministic enrichers, the declarative cross-link wrapper, the
projection alias backend, packaged extraction/linker defaults, and the
text anonymizer for connector emission hooks.

Provenance: ported from ``okg-deployments`` ``main@f33a9c4``
(``cms/enrichers/``, ``cms/cms_sources/alias.py``, the ``cms`` and
``wisdqm`` deployment manifests and catalog YAML), plus the archi v2
anonymizer from ``dev@28b977d1``. okg substrate imports are kept
as-is; deployment-local paths became parameters backed by packaged
defaults (:mod:`archi.enrichment.defaults`).

Deployment wiring template (adapted from the cms deployment manifest,
``deployment.yaml`` enrichers block, lines 236-242)::

    enrichers:
      - class: archi.enrichment.chunk_reference_rollup.JiraChunkReferenceRollup
      - class: archi.enrichment.jira_affects.JiraAffectsFromReferences
      - class: archi.enrichment.dqm_run_range.DQMRunRangeLinker
      - class: archi.enrichment.meeting_document_reference_rollup.MeetingDocumentReferenceRollup
      - class: archi.enrichment.declarative.GlobalTagReleaseLinker
      - class: okg.substrate.library.linkers.code_reference.CodeReferenceResolver

    alias_resolver:
      on_empty: drop
      backends:
        - class: archi.enrichment.alias.ProjectionAliasBackend

Enricher ``name`` attributes keep their historical ``cms_*`` values on
purpose: prior CMS generations key progress rows and dedupe keys on
them, and the comp-ops instance is the parity target for cutover.
"""
