"""Roll Indico attachment document references up to meeting records.

Provenance: verbatim port of
``cms/enrichers/meeting_document_reference_rollup.py`` (241 LOC) from
``okg-deployments`` ``main@f33a9c4``; only the module docstring
changed. okg substrate imports are unchanged, and the SQL, subtype
tables, dedupe-key recipe, and enricher ``name`` are kept byte-stable
so cutover does not re-key existing derived edges.

Deployment wiring (enrichers block)::

    enrichers:
      - class: archi.enrichment.meeting_document_reference_rollup.MeetingDocumentReferenceRollup
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import psycopg

from okg.substrate.enrichers.base import (
    EnrichResult,
    IncrementalContext,
)
from okg.substrate.enrichers.derived_edges import (
    DerivedEdgeCandidate,
    insert_deterministic_edges,
)
from okg.substrate.library.linkers import _chronos


_TARGET_SUBTYPES = (
    "site",
    "jira_issue",
    "infrastructure_service",
    "cmssw_release",
    "global_tag",
    "dataset",
    "workflow",
    "run",
)


class MeetingDocumentReferenceRollup:
    """Inherit references from Indico attachment documents.

    Indico PDF text is represented as document/document_chunk rows owned by
    the meeting. The generic chunk rollup handles direct meeting -> chunk
    containment, but Indico's path is meeting -> document -> chunk. This
    linker fills that two-hop gap without changing connector output.
    """

    name = "cms_meeting_document_reference_rollup"
    publisher_skip_when_clean = True
    publisher_skip_when_catalog_changed_but_inputs_clean = True
    reads_edge_types = ("contains", "references")
    reads_subtypes = (
        "meeting_minutes",
        "document",
        "document_chunk",
        *_TARGET_SUBTYPES,
    )
    emits_edge_types = ("references",)
    target_subtypes = _TARGET_SUBTYPES
    requires_narrowings = tuple(
        ("meeting_minutes", "references", target)
        for target in _TARGET_SUBTYPES
    )

    def enrich(
        self,
        conn: psycopg.Connection,
        *,
        generation_id: int,
        incremental: IncrementalContext | None = None,
    ) -> EnrichResult:
        if _can_skip_rollup(conn, incremental):
            return EnrichResult(self.name, n_edges_emitted=0)

        run_id = str(uuid.uuid4())
        observed_at = datetime.now(timezone.utc)

        with conn.cursor() as cur:
            candidates = _chronos.query(
                cur,
                """
                WITH direct_doc_refs AS (
                    SELECT meeting.node_id AS meeting_id,
                           target.node_id AS target_id,
                           target.subtype AS target_subtype,
                           'document.references' AS support_kind,
                           doc.node_id AS support_id
                      FROM okg.graph_edges meeting_doc
                      JOIN okg.graph_nodes meeting
                        ON meeting.node_id = meeting_doc.src
                       AND meeting.subtype = 'meeting_minutes'
                      JOIN okg.graph_nodes doc
                        ON doc.node_id = meeting_doc.dst
                       AND doc.subtype = 'document'
                      JOIN okg.graph_edges doc_ref
                        ON doc_ref.src = doc.node_id
                       AND doc_ref.edge_type = 'references'
                      JOIN okg.graph_nodes target
                        ON target.node_id = doc_ref.dst
                       AND target.subtype = ANY(:target_subtypes)
                     WHERE meeting_doc.edge_type = 'contains'
                ),
                chunk_refs AS (
                    SELECT meeting.node_id AS meeting_id,
                           target.node_id AS target_id,
                           target.subtype AS target_subtype,
                           'document_chunk.references' AS support_kind,
                           chunk.node_id AS support_id
                      FROM okg.graph_edges meeting_doc
                      JOIN okg.graph_nodes meeting
                        ON meeting.node_id = meeting_doc.src
                       AND meeting.subtype = 'meeting_minutes'
                      JOIN okg.graph_nodes doc
                        ON doc.node_id = meeting_doc.dst
                       AND doc.subtype = 'document'
                      JOIN okg.graph_edges doc_chunk
                        ON doc_chunk.src = doc.node_id
                       AND doc_chunk.edge_type = 'contains'
                      JOIN okg.graph_nodes chunk
                        ON chunk.node_id = doc_chunk.dst
                       AND chunk.subtype = 'document_chunk'
                      JOIN okg.graph_edges chunk_ref
                        ON chunk_ref.src = chunk.node_id
                       AND chunk_ref.edge_type = 'references'
                      JOIN okg.graph_nodes target
                        ON target.node_id = chunk_ref.dst
                       AND target.subtype = ANY(:target_subtypes)
                     WHERE meeting_doc.edge_type = 'contains'
                ),
                combined AS (
                    SELECT * FROM direct_doc_refs
                    UNION ALL
                    SELECT * FROM chunk_refs
                )
                SELECT combined.meeting_id,
                       combined.target_id,
                       combined.target_subtype,
                       count(DISTINCT combined.support_id) AS support_count,
                       array_agg(
                           DISTINCT combined.support_kind
                           ORDER BY combined.support_kind
                       ) AS support_kinds
                  FROM combined
                  LEFT JOIN okg.graph_edges existing
                    ON existing.src = combined.meeting_id
                   AND existing.dst = combined.target_id
                   AND existing.edge_type = 'references'
                 WHERE existing.edge_id IS NULL
                 GROUP BY combined.meeting_id,
                          combined.target_id,
                          combined.target_subtype
                 ORDER BY combined.meeting_id, combined.target_id
                """,
                {"target_subtypes": list(self.target_subtypes)},
            )
            if not candidates:
                return EnrichResult(self.name, n_edges_emitted=0)

            derived_candidates = []
            for row in candidates:
                meeting_id = str(row["meeting_id"])
                target_id = str(row["target_id"])
                target_subtype = str(row["target_subtype"])
                support_count = row["support_count"]
                support_kinds = row["support_kinds"]
                edge_type = "references"
                dedupe_key = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.name}|{meeting_id}|{edge_type}|{target_id}",
                ).hex
                derived_candidates.append(
                    DerivedEdgeCandidate(
                        src=meeting_id,
                        edge_type=edge_type,
                        dst=target_id,
                        attrs={
                            "match_type": "inherited_meeting_document_reference",
                            "inherited_from": list(support_kinds),
                            "support_count": int(support_count),
                            "target_subtype": target_subtype,
                        },
                        source_record_id={
                            "src_node_id": meeting_id,
                            "dst_node_id": target_id,
                            "enricher": self.name,
                        },
                        source_revision={"generation_id": generation_id},
                        dedupe_key=dedupe_key,
                    )
                )

            result = insert_deterministic_edges(
                cur,
                source=f"_enricher:{self.name}",
                candidates=derived_candidates,
                observed_at=observed_at,
                run_id=run_id,
            )

        return EnrichResult(
            self.name,
            n_edges_emitted=result.inserted,
            n_edges_skipped=result.skipped,
        )


def _can_skip_rollup(
    conn: psycopg.Connection,
    incremental: IncrementalContext | None,
) -> bool:
    if incremental is None or incremental.is_first_run:
        return False

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS(
              SELECT 1
                FROM okg.node_facts
               WHERE fact_id > %s
                 AND subtype = ANY(%s)
            )
            """,
            (
                incremental.prev_fact_watermark,
                ["meeting_minutes", "document", "document_chunk"],
            ),
        )
        if bool(cur.fetchone()[0]):
            return False

        cur.execute(
            """
            SELECT EXISTS(
              SELECT 1
                FROM okg.edge_facts
               WHERE fact_id > %s
                 AND edge_type = ANY(%s)
            )
            """,
            (
                incremental.prev_edge_watermark,
                ["contains", "references"],
            ),
        )
        has_relevant_edges = bool(cur.fetchone()[0])

    return not has_relevant_edges
