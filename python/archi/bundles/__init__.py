"""Packs and bundles: manifests, discovery, composition (ADR 0002 section 3).

A *pack* is the smallest installable capability (`pack.yaml`); a *bundle* is
a named selection of packs (`bundle.yaml`). The published JSON schemas live
in `schema/`; `archi bundles validate` applies them plus the checks a schema
cannot express.
"""
from archi.bundles.manifest import (
    Bundle,
    ManifestError,
    Pack,
    Problem,
    load_pack,
    validate_bundle,
    validate_pack,
)

__all__ = [
    "Bundle",
    "ManifestError",
    "Pack",
    "Problem",
    "load_pack",
    "validate_bundle",
    "validate_pack",
]
