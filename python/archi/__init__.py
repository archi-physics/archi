"""Archi automates operations for scientific computing.

It has two layers on top of OKG (ADR 0002): a context layer that holds what a
facility knows, kept current, linked, cited and measured; and an operations
layer whose automations turn signals into reviewed proposals and actions,
recording each outcome. This package ships the source adapters, schemas,
enrichment, live tools, playbooks, packs, bundles, automation runtime and
evaluation that make up both layers, and the `archi` command that installs and
runs them. It holds no credentials, no site configuration and no running
services.

okg is a host dependency, deliberately not declared here: adapters are
imported *by* an OKG deployment (`module: archi.sources.<name>` in its
source registry), so the substrate is always already present.
"""

__version__ = "3.0.0"
