"""Roll document-chunk references up to parent records.

Provenance: verbatim port of ``cms/enrichers/chunk_reference_rollup.py``
(320 LOC) from ``okg-deployments`` ``main@f33a9c4``; only the module
docstring changed. okg substrate imports are unchanged, and the SQL,
subtype tables, dedupe-key recipe, and enricher ``name`` are kept
byte-stable so cutover does not re-key existing derived edges.

Deployment wiring (enrichers block)::

    enrichers:
      - class: archi.enrichment.chunk_reference_rollup.JiraChunkReferenceRollup
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
    mint_edge_id,
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
_PARENT_SUBTYPES = (
    "jira_issue",
    "documentation_page",
    "forum_thread",
    "meeting_minutes",
    "document",
)
_MAX_INCREMENTAL_SCOPE = 10_000


class JiraChunkReferenceRollup:
    """Inherit references discovered in text chunks.

    Long text sources are chunked so projection can extract references
    without using a truncated parent blob. This enricher copies each
    chunk-level `references` edge to the owning source record, preserving
    the edge type while recording that the support came from one or more
    chunks. The historical class/name are kept stable because prior CMS
    generations already use them in progress rows and dedupe keys.
    """

    name = "cms_jira_chunk_reference_rollup"
    publisher_skip_when_clean = True
    publisher_skip_when_catalog_changed_but_inputs_clean = True
    reads_edge_types = ("contains", "references")
    reads_subtypes = (
        "jira_issue",
        "documentation_page",
        "forum_thread",
        "meeting_minutes",
        "document",
        "document_chunk",
        "site",
        "jira_issue",
        "infrastructure_service",
        "cmssw_release",
        "global_tag",
        "dataset",
        "workflow",
        "run",
    )
    emits_edge_types = ("references",)
    target_subtypes = _TARGET_SUBTYPES
    parent_subtypes = _PARENT_SUBTYPES
    requires_narrowings = tuple(
        (parent, "references", target)
        for parent in _PARENT_SUBTYPES
        for target in _TARGET_SUBTYPES
    )

    def enrich(
        self,
        conn: psycopg.Connection,
        *,
        generation_id: int,
        incremental: IncrementalContext | None = None,
    ) -> EnrichResult:
        if _can_skip_rollup(conn, incremental, self.parent_subtypes):
            return EnrichResult(self.name, n_edges_emitted=0)

        run_id = str(uuid.uuid4())
        observed_at = datetime.now(timezone.utc)
        scope_ids = _incremental_scope_node_ids(
            conn,
            incremental,
            edge_types=("contains", "references"),
        )
        if scope_ids == []:
            return EnrichResult(self.name, n_edges_emitted=0)
        scope_filter = ""
        params: dict[str, object] = {
            "parent_subtypes": list(self.parent_subtypes),
            "target_subtypes": list(self.target_subtypes),
        }
        if scope_ids is not None:
            scope_filter = (
                "AND (parent.src = ANY(:scope_ids) "
                "OR parent.dst = ANY(:scope_ids))"
            )
            params["scope_ids"] = scope_ids

        with conn.cursor() as cur:
            candidates = _chronos.query(
                cur,
                f"""
                SELECT parent.src AS parent_id,
                       parent_node.subtype AS parent_subtype,
                       chunk_ref.dst AS target_id,
                       target.subtype AS target_subtype,
                       count(DISTINCT parent.dst) AS chunk_count
                  FROM okg.graph_edges parent
                  JOIN okg.graph_nodes parent_node
                    ON parent_node.node_id = parent.src
                   AND parent_node.subtype = ANY(:parent_subtypes)
                  JOIN okg.graph_nodes chunk_node
                    ON chunk_node.node_id = parent.dst
                   AND chunk_node.subtype = 'document_chunk'
                  JOIN okg.graph_edges chunk_ref
                    ON chunk_ref.src = parent.dst
                   AND chunk_ref.edge_type = 'references'
                  JOIN okg.graph_nodes target
                    ON target.node_id = chunk_ref.dst
                   AND target.subtype = ANY(:target_subtypes)
                 WHERE parent.edge_type = 'contains'
                   AND parent.src <> chunk_ref.dst
                   {scope_filter}
                GROUP BY parent.src, parent_node.subtype,
                         chunk_ref.dst, target.subtype
                """,
                params,
            )
            existing = _existing_live_pairs(
                cur,
                edge_type="references",
                pairs=[
                    (str(row["parent_id"]), str(row["target_id"]))
                    for row in candidates
                ],
            )
            candidates = [
                row for row in candidates
                if (str(row["parent_id"]), str(row["target_id"])) not in existing
            ]
            if not candidates:
                return EnrichResult(self.name, n_edges_emitted=0)

            derived_candidates = []
            for row in candidates:
                src = str(row["parent_id"])
                parent_subtype = str(row["parent_subtype"])
                dst = str(row["target_id"])
                target_subtype = str(row["target_subtype"])
                chunk_count = row["chunk_count"]
                edge_type = "references"
                dedupe_key = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.name}|{src}|{edge_type}|{dst}",
                ).hex
                derived_candidates.append(
                    DerivedEdgeCandidate(
                        src=src,
                        edge_type=edge_type,
                        dst=dst,
                        attrs={
                            "match_type": "inherited_chunk_reference",
                            "inherited_from": "document_chunk.references",
                            "chunk_count": int(chunk_count),
                            "parent_subtype": parent_subtype,
                            "target_subtype": target_subtype,
                        },
                        source_record_id={
                            "src_node_id": src,
                            "dst_node_id": dst,
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
    parent_subtypes: tuple[str, ...],
) -> bool:
    if incremental is None or incremental.is_first_run:
        return False

    relevant_node_subtypes = (*parent_subtypes, "document_chunk")
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
                list(relevant_node_subtypes),
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


def _incremental_scope_node_ids(
    conn: psycopg.Connection,
    incremental: IncrementalContext | None,
    *,
    edge_types: tuple[str, ...],
) -> list[str] | None:
    if incremental is None or incremental.is_first_run:
        return None

    ids = set(incremental.dirty_node_ids)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT src, dst
              FROM okg.edge_facts
             WHERE fact_id > %s
               AND edge_type = ANY(%s)
            """,
            (
                incremental.prev_edge_watermark,
                list(edge_types),
            ),
        )
        for src, dst in cur.fetchall():
            ids.add(src)
            ids.add(dst)

    if not ids:
        return []
    if len(ids) > _MAX_INCREMENTAL_SCOPE:
        return None
    return sorted(ids)


def _existing_live_pairs(
    cur: psycopg.Cursor,
    *,
    edge_type: str,
    pairs: list[tuple[str, str]],
) -> set[tuple[str, str]]:
    if not pairs:
        return set()
    unique_pairs = sorted(set(pairs))
    pairs_by_edge_id = {
        mint_edge_id(src, edge_type, dst): (src, dst)
        for src, dst in unique_pairs
    }
    existing: set[tuple[str, str]] = set()
    edge_ids = list(pairs_by_edge_id)
    for start in range(0, len(edge_ids), 5_000):
        rows = _chronos.query(
            cur,
            """
            SELECT edge_id
              FROM okg.graph_edges
             WHERE edge_id = ANY(:edge_ids)
               AND edge_type = :edge_type
            """,
            {
                "edge_ids": edge_ids[start:start + 5_000],
                "edge_type": edge_type,
            },
        )
        existing.update(
            pairs_by_edge_id[int(row["edge_id"])]
            for row in rows
            if int(row["edge_id"]) in pairs_by_edge_id
        )
    return existing
