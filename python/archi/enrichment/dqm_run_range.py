"""Link DQM certification ranges to live run nodes.

Provenance: verbatim port of ``cms/enrichers/dqm_run_range.py`` (139
LOC) from ``okg-deployments`` ``main@f33a9c4``; only the module
docstring changed. okg substrate imports are unchanged, and the SQL,
dedupe-key recipe, and enricher ``name`` are kept byte-stable so
cutover does not re-key existing derived edges.

Deployment wiring (enrichers block)::

    enrichers:
      - class: archi.enrichment.dqm_run_range.DQMRunRangeLinker
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


class DQMRunRangeLinker:
    """Infer DQM certification coverage for already-known runs.

    The DQM connector emits explicit run edges when a certification JSON
    lists individual run IDs. Many certification records only carry a run
    range, while run nodes arrive from text extraction or other
    connectors. This deterministic linker fills that gap without
    inventing run nodes.
    """

    name = "cms_dqm_run_range_linker"
    publisher_skip_when_clean = True
    publisher_skip_when_catalog_changed_but_inputs_clean = True
    reads_edge_types = ("recorded_during",)
    reads_subtypes = ("data_certification", "run")
    emits_edge_types = ("recorded_during",)
    requires_narrowings = (
        ("data_certification", "recorded_during", "run"),
    )

    def enrich(
        self,
        conn: psycopg.Connection,
        *,
        generation_id: int,
        incremental: IncrementalContext | None = None,
    ) -> EnrichResult:
        del incremental
        run_id = str(uuid.uuid4())
        observed_at = datetime.now(timezone.utc)

        with conn.cursor() as cur:
            candidates = _chronos.query(
                cur,
                """
                WITH certs AS MATERIALIZED (
                    SELECT node_id,
                           attrs,
                           (attrs->>'run_min')::bigint AS run_min,
                           (attrs->>'run_max')::bigint AS run_max
                      FROM okg.graph_nodes
                     WHERE subtype = 'data_certification'
                       AND (attrs->>'run_min') ~ '^[0-9]+$'
                       AND (attrs->>'run_max') ~ '^[0-9]+$'
                ),
                runs AS MATERIALIZED (
                    SELECT node_id,
                           attrs,
                           (attrs->>'run_number')::bigint AS run_number
                      FROM okg.graph_nodes
                     WHERE subtype = 'run'
                       AND (attrs->>'run_number') ~ '^[0-9]+$'
                )
                SELECT cert.node_id AS cert_id,
                       run.node_id AS run_id,
                       cert.attrs->>'certification_id' AS certification_id,
                       cert.run_min,
                       cert.run_max,
                       run.run_number
                  FROM certs cert
                  JOIN runs run
                    ON run.run_number BETWEEN cert.run_min AND cert.run_max
                  LEFT JOIN okg.graph_edges existing
                    ON existing.src = cert.node_id
                   AND existing.dst = run.node_id
                   AND existing.edge_type = 'recorded_during'
                 WHERE existing.edge_id IS NULL
                 ORDER BY cert.node_id, run.node_id
                """
            )
            if not candidates:
                return EnrichResult(self.name, n_edges_emitted=0)

            derived_candidates = []
            for row in candidates:
                cert_id = str(row["cert_id"])
                run_node_id = str(row["run_id"])
                certification_id = row["certification_id"]
                run_min = row["run_min"]
                run_max = row["run_max"]
                run_number = row["run_number"]
                edge_type = "recorded_during"
                dedupe_key = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.name}|{cert_id}|{edge_type}|{run_node_id}",
                ).hex
                derived_candidates.append(
                    DerivedEdgeCandidate(
                        src=cert_id,
                        edge_type=edge_type,
                        dst=run_node_id,
                        attrs={
                        "match_type": "dqm_certification_run_range",
                        "certification_id": certification_id,
                        "run_min": int(run_min),
                        "run_max": int(run_max),
                        "run_number": int(run_number),
                        },
                        source_record_id={
                            "src_node_id": cert_id,
                            "dst_node_id": run_node_id,
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
