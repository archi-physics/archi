"""Link-coverage targets as okg invariants (PACT r3-context-quality, ADR 0002 6.1).

A pack declares, in `pack.yaml`, how well the identifiers it carries must be
linked:

    link_coverage:
      - name: storage_endpoint_site
        from: storage_endpoint     # the nodes that carry the identifier
        edge: contains
        to: site                   # the subtype they must link to
        direction: in              # `out`: from -[edge]-> to; `in`: to -[edge]-> from
        min_fraction: 0.9          # share of `from` nodes that need such an edge
        severity: warn             # `warn` (default) or `error` (blocks publish)

Composition turns each target into one invariant in the format packs already
use (`invariants.yaml`: name, severity, scope, timeout_ms, description, sql).
The invariant returns one row, with the counts, when the share of linked
nodes is below target, and no row when it holds or when the pack has no node of
that subtype (an empty subtype is the coverage invariant's finding).

The SQL sticks to syntax Postgres and SQLite share (no casts, no `ceil`), so a
test can run the generated text over fixture rows.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

DEFAULT_SEVERITY = "warn"
INVARIANT_TIMEOUT_MS = 10000


def invariant_name(pack_name: str, target_name: str) -> str:
    """`cms_catalogs_link_storage_endpoint_site`: matches okg's `^[a-z_][a-z0-9_]*$`."""
    return f"{pack_name.replace('-', '_')}_link_{target_name}"


def coverage_sql(target: Mapping[str, Any]) -> str:
    """The invariant SQL for one target (rows returned mean the target is missed)."""
    node, other = target["from"], target["to"]
    edge = target["edge"]
    if target.get("direction", "out") == "in":
        join = "e.dst = f.node_id"
        peer = "e.src"
    else:
        join = "e.src = f.node_id"
        peer = "e.dst"
    return (
        "WITH carriers AS (\n"
        "  SELECT node_id\n"
        "    FROM okg.v_nodes\n"
        f"   WHERE subtype = '{node}'\n"
        "),\n"
        "linked AS (\n"
        "  SELECT DISTINCT f.node_id\n"
        "    FROM carriers f\n"
        "    JOIN okg.v_edges e\n"
        f"      ON {join} AND e.edge_type = '{edge}'\n"
        "    JOIN okg.v_nodes t\n"
        f"      ON t.node_id = {peer} AND t.subtype = '{other}'\n"
        "),\n"
        "totals AS (\n"
        "  SELECT (SELECT count(*) FROM carriers) AS carrier_count,\n"
        "         (SELECT count(*) FROM linked) AS linked_count\n"
        ")\n"
        f"SELECT '{node}' AS subtype, carrier_count, linked_count,\n"
        f"       {float(target['min_fraction'])!r} AS min_fraction\n"
        "  FROM totals\n"
        " WHERE carrier_count > 0\n"
        f"   AND linked_count * 1.0 < {float(target['min_fraction'])!r} * carrier_count\n"
    )


def link_coverage_invariant(pack_name: str, target: Mapping[str, Any]) -> dict[str, Any]:
    direction = target.get("direction", "out")
    arrow = (
        f"{target['from']} -[{target['edge']}]-> {target['to']}"
        if direction == "out"
        else f"{target['to']} -[{target['edge']}]-> {target['from']}"
    )
    return {
        "name": invariant_name(pack_name, target["name"]),
        "severity": target.get("severity", DEFAULT_SEVERITY),
        "scope": "cross",
        "timeout_ms": INVARIANT_TIMEOUT_MS,
        "description": (
            f"Link coverage ({pack_name} pack): at least "
            f"{target['min_fraction']:.0%} of {target['from']} nodes need a "
            f"{arrow} edge. "
            + (target.get("description") or "").strip()
        ).strip(),
        "sql": coverage_sql(target),
    }


def link_coverage_invariants(pack_name: str, targets: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [link_coverage_invariant(pack_name, t) for t in targets]


def coverage(carriers: Iterable[str], linked: Iterable[str]) -> tuple[int, int, float]:
    """(carriers, linked, fraction) for two id collections; fraction is 1.0 when none carry."""
    carrier_ids = set(carriers)
    linked_ids = carrier_ids & set(linked)
    total = len(carrier_ids)
    return total, len(linked_ids), (len(linked_ids) / total if total else 1.0)
