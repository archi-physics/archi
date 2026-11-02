"""The cms-analyses pack's declarative cross-source links.

The substrate's ``DeclarativeLinker`` owns the behaviour: it walks the
graph, matches nodes by the fields a rule names, and emits derived edges
with ``provenance='derived_deterministic'``. This module only resolves
the rule files the cms-analyses pack ships
(``bundles/packs/cms-analyses/cross_links/``), so the enricher works from
an installed deployment rather than depending on the operator's working
directory.

Wiring, in the deployment's enrichers block::

    enrichers:
      - class: archi.enrichment.physics_cross_links.PhysicsCrossLinker

Packs have no enrichers key yet, so this block is added by hand; the
pack README says so. ``archi.enrichment.declarative.GlobalTagReleaseLinker``
is the same pattern for the global-tag rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from okg.substrate.library.linkers.declarative import DeclarativeLinker
from okg.substrate.publisher.enricher_loader import derive_write_envelope


def _rules_dir() -> Path:
    """The pack's ``cross_links/`` directory, wherever pip put the payload."""
    from archi.paths import packs_dir

    return packs_dir() / "cms-analyses" / "cross_links"


#: The trust label of every edge this linker writes. okg's publisher
#: rejects a derived edge whose producer is labelled more trusted than the
#: least-trusted source it was derived from
#: (``derived_output_trust_exceeds_input``). Every rule here reads TWiki,
#: analysis-note or CDS records, which the packs label ``untrusted`` so
#: an MCP reader gets their text fenced; the linker's output can be no
#: more trusted than that. ``untrusted`` is the only label okg ranks at
#: the bottom of its lattice (``okg.substrate.producer_authority``).
TRUST_LABEL = "untrusted"


def cross_link_rules_files() -> list[str]:
    """Absolute paths to the cms-analyses pack's cross-link rules."""
    directory = _rules_dir()
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"no cross-link rule files in {directory}")
    return [str(path) for path in paths]


class PhysicsCrossLinker(DeclarativeLinker):
    """Link CDS records to the analysis notes and PAGs they name."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "cms_analyses_cross_links")
        # None or an empty list means "not provided". A bare
        # key-presence check would treat rules_files=None as a caller
        # override and let DeclarativeLinker load its entire substrate
        # rule set under this linker's name -- the bug
        # GlobalTagReleaseLinker documents having hit.
        if not any(kwargs.get(key) for key in ("rules", "rules_files", "rules_dir")):
            for key in ("rules", "rules_files", "rules_dir"):
                kwargs.pop(key, None)
            kwargs["rules_files"] = cross_link_rules_files()
        super().__init__(**kwargs)

    @property
    def producer_policy(self) -> dict[str, Any]:
        """The policy okg would derive for this linker, labelled untrusted.

        Without a declared policy okg derives one from the linker's own
        declarations and labels it ``derived-deterministic-trusted``. This
        keeps that derivation -- the same write envelope, record kinds and
        clauses, from okg's ``derive_write_envelope`` -- and changes only
        the label, so the signature cannot drift from the rules.
        """
        producer_id = f"_enricher:{self.name}"
        envelope = derive_write_envelope(self, producer_id=producer_id)
        signature = dict(envelope.signature)
        nodes = list(signature.get("nodes") or [])
        edges = list(signature.get("edges") or [])
        unbounded = set(getattr(envelope, "unbounded", ()) or ())
        if unbounded:
            # okg's own derivation handles an open surface; this one does
            # not, and must not quietly narrow it.
            raise ValueError(
                f"{self.name}: write envelope is unbounded in "
                f"{sorted(unbounded)}; declare its policy by hand"
            )
        kinds = [kind for kind, rows in (("node", nodes), ("edge", edges)) if rows]
        clauses = [
            {
                "id": str(node["clause_id"]),
                "effect": "allow",
                "record_kind": "node",
                "subtype": str(node["subtype"]),
            }
            for node in nodes
        ] + [
            {
                "id": str(edge["clause_id"]),
                "effect": "allow",
                "record_kind": "edge",
                "triple": [
                    str(edge["src_subtype"]),
                    str(edge["edge_type"]),
                    str(edge["dst_subtype"]),
                ],
            }
            for edge in edges
        ]
        return {
            "producer_id": producer_id,
            "producer_kind": "enricher",
            "trust_label": TRUST_LABEL,
            "admission_mode": "fast_track",
            "authority_scope": {
                "source_family": producer_id,
                "enricher_name": self.name,
            },
            "allowed_record_kinds": kinds,
            "output_signature": signature,
            "policy_clauses": clauses,
            "output_scope_summary": {
                "summary": "deterministic enricher output; untrusted inputs",
                "edges": [
                    [
                        str(edge["src_subtype"]),
                        str(edge["edge_type"]),
                        str(edge["dst_subtype"]),
                    ]
                    for edge in edges
                ],
                "write_envelope": envelope.summary(),
            },
        }
