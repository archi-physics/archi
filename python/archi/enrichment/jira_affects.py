"""Promote operational JIRA references into affects edges.

Provenance: verbatim port of ``cms/enrichers/jira_affects.py`` (230
LOC) from ``okg-deployments`` ``main@f33a9c4``; only the module
docstring changed. okg substrate imports are unchanged, and the SQL,
dedupe-key recipe, and enricher ``name`` are kept byte-stable so
cutover does not re-key existing derived edges.

Deployment wiring (enrichers block)::

    enrichers:
      - class: archi.enrichment.jira_affects.JiraAffectsFromReferences
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


_MAX_INCREMENTAL_SCOPE = 10_000


class JiraAffectsFromReferences:
    """Infer impact edges from JIRA references to operational targets.

    JIRA text extraction and chunk rollup already identify explicit
    site/service references. For operational traversal, those references
    are also evidence that the issue affects the referenced site or
    service. This enricher keeps the semantic promotion deterministic and
    auditable by preserving the source reference in edge attrs.
    """

    name = "cms_jira_affects_from_references"
    publisher_skip_when_clean = True
    publisher_skip_when_catalog_changed_but_inputs_clean = True
    reads_edge_types = ("references",)
    reads_subtypes = (
        "jira_issue",
        "site",
        "infrastructure_service",
    )
    emits_edge_types = ("affects",)
    target_subtypes = ("site", "infrastructure_service")
    requires_narrowings = tuple(
        ("jira_issue", "affects", target)
        for target in target_subtypes
    )

    def enrich(
        self,
        conn: psycopg.Connection,
        *,
        generation_id: int,
        incremental: IncrementalContext | None = None,
    ) -> EnrichResult:
        run_id = str(uuid.uuid4())
        observed_at = datetime.now(timezone.utc)
        scope_ids = _incremental_scope_node_ids(
            conn,
            incremental,
            edge_types=("references",),
        )
        if scope_ids == []:
            return EnrichResult(self.name, n_edges_emitted=0)
        scope_filter = ""
        params: dict[str, object] = {
            "target_subtypes": list(self.target_subtypes),
        }
        if scope_ids is not None:
            scope_filter = (
                "AND (ref.src = ANY(:scope_ids) "
                "OR ref.dst = ANY(:scope_ids))"
            )
            params["scope_ids"] = scope_ids

        with conn.cursor() as cur:
            candidates = _chronos.query(
                cur,
                f"""
                SELECT ref.src AS jira_id,
                       ref.dst AS target_id,
                       target.subtype AS target_subtype,
                       count(*) AS reference_count
                  FROM okg.graph_edges ref
                  JOIN okg.graph_nodes jira
                    ON jira.node_id = ref.src
                   AND jira.subtype = 'jira_issue'
                  JOIN okg.graph_nodes target
                    ON target.node_id = ref.dst
                   AND target.subtype = ANY(:target_subtypes)
                 WHERE ref.edge_type = 'references'
                   {scope_filter}
                 GROUP BY ref.src, ref.dst, target.subtype
                """,
                params,
            )
            existing = _existing_live_pairs(
                cur,
                edge_type="affects",
                pairs=[
                    (str(row["jira_id"]), str(row["target_id"]))
                    for row in candidates
                ],
            )
            candidates = [
                row for row in candidates
                if (str(row["jira_id"]), str(row["target_id"])) not in existing
            ]
            if not candidates:
                return EnrichResult(self.name, n_edges_emitted=0)

            derived_candidates = []
            for row in candidates:
                src = str(row["jira_id"])
                dst = str(row["target_id"])
                target_subtype = str(row["target_subtype"])
                reference_count = row["reference_count"]
                edge_type = "affects"
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
                            "match_type": "jira_reference_impact",
                            "derived_from": "jira_issue.references",
                            "reference_count": int(reference_count),
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


def _incremental_scope_node_ids(
    conn: psycopg.Connection,
    incremental: IncrementalContext | None,
    *,
    edge_types: tuple[str, ...],
) -> list[str] | None:
    if incremental is None or incremental.is_first_run:
        return None

    ids = {
        node_id for node_id in incremental.dirty_node_ids
        if node_id.startswith("jira:")
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT src
              FROM okg.edge_facts
             WHERE fact_id > %s
               AND edge_type = ANY(%s)
            """,
            (
                incremental.prev_edge_watermark,
                list(edge_types),
            ),
        )
        for (src,) in cur.fetchall():
            ids.add(src)

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
