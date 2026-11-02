"""Loader for the packaged enrichment defaults (importlib.resources).

The ``defaults/`` data directory next to this module packages the
declarative/config data the cms and wisdqm instances carried in their
deployment trees (okg-deployments ``main@f33a9c4``):

- ``cross_links/*.yaml`` — declarative cross-link rules for the
  substrate ``DeclarativeLinker`` (cms/cross_links).
- ``extraction_rules.yaml`` — the extraction-block regex rules from
  the cms deployment manifest (``extractor.config.rules``).
- ``extraction_rules_dqm.yaml`` — the extra wisdqm rules (hlt_path,
  l1t_seed, dqm_workspace) kept as a separate defaults file.
- ``identifier_patterns.yaml`` — cms/identifier_patterns.yaml, the
  catalog-loaded identifier pattern set.
- ``linker_config.yaml`` — cms/linker_config.yaml, the projection
  linker routes.

These are consumed as data (paste/load into a deployment manifest or
catalog), not imported by okg; PyYAML arrives via the okg host
environment, same posture as the okg imports themselves.

Note: path-returning helpers assume a filesystem install (the normal
pip wheel case); zipped installs are not supported.
"""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def defaults_dir() -> Path:
    """Absolute path of the packaged ``defaults/`` data directory."""
    path = Path(str(resources.files(__package__) / "defaults"))
    if not path.is_dir():
        raise FileNotFoundError(
            f"packaged enrichment defaults directory missing: {path}"
        )
    return path


def cross_links_dir() -> Path:
    """Directory of packaged declarative cross-link rule files."""
    path = defaults_dir() / "cross_links"
    if not path.is_dir():
        raise FileNotFoundError(
            f"packaged cross_links directory missing: {path}"
        )
    return path


def cross_link_rules_file(name: str = "global_tag_release.yaml") -> Path:
    """Path of one packaged cross-link rule file (for ``rules_files``)."""
    path = cross_links_dir() / name
    if not path.is_file():
        available = sorted(p.name for p in cross_links_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"unknown cross-link rules file {name!r}; packaged: {available}"
        )
    return path


def _load_yaml(name: str) -> Any:
    return yaml.safe_load((defaults_dir() / name).read_text(encoding="utf-8"))


def _extraction_rules(name: str) -> list[tuple[str, str, float]]:
    data = _load_yaml(name)
    rules = [
        (str(id_type), str(regex), float(confidence))
        for id_type, regex, confidence in data["rules"]
    ]
    for _, regex, _ in rules:
        re.compile(regex)
    return rules


def load_extraction_rules() -> list[tuple[str, str, float]]:
    """The cms extraction-block rules as (id_type, regex, confidence)."""
    return _extraction_rules("extraction_rules.yaml")


def load_dqm_extraction_rules() -> list[tuple[str, str, float]]:
    """The extra wisdqm rules (hlt_path, l1t_seed, dqm_workspace)."""
    return _extraction_rules("extraction_rules_dqm.yaml")


def load_identifier_patterns() -> list[dict[str, str]]:
    """The catalog identifier-pattern entries (id_type/regex/description)."""
    patterns = _load_yaml("identifier_patterns.yaml")["patterns"]
    for entry in patterns:
        re.compile(entry["regex"])
    return patterns


def load_linker_config() -> dict[str, Any]:
    """The projection linker routes mapping."""
    return _load_yaml("linker_config.yaml")


def load_cross_link_rules(
    name: str = "global_tag_release.yaml",
) -> list[dict[str, Any]]:
    """Parsed rule entries of one packaged cross-link rule file."""
    data = yaml.safe_load(
        cross_link_rules_file(name).read_text(encoding="utf-8")
    )
    return data["rules"]


def load_all() -> dict[str, Any]:
    """Load every packaged default (smoke surface for tests/tooling)."""
    return {
        "extraction_rules": load_extraction_rules(),
        "extraction_rules_dqm": load_dqm_extraction_rules(),
        "identifier_patterns": load_identifier_patterns(),
        "linker_config": load_linker_config(),
        "cross_links": {
            path.name: load_cross_link_rules(path.name)
            for path in sorted(cross_links_dir().glob("*.yaml"))
        },
    }
