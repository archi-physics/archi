"""Render a resolution into an okg profile directory.

okg has no profile composition (ADR 0002 D7, section 9 ask 3), so archi
writes one ordinary profile that `okg install --profile <name>` consumes
unchanged:

    <instance>/profiles/<name>/
        profile.yaml  modules.yaml  deployment-defaults.yaml
        source-defaults/<pack>.<source>.yaml
        schemas/...  skills/...  invariants.yaml  <files>

`profiles/` is the directory okg searches relative to where it runs, so
running okg from the instance directory finds the profile with no
`OKG_PROFILES_DIR`. The tree is rebuilt from scratch on every render, so a
pack dropped from the selection leaves nothing behind.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from archi.bundles.compose import (
    PROMPT_FILE,
    TRIGGERS_FILE,
    Composition,
    compose,
    deployment_defaults,
)
from archi.bundles.resolve import Resolution

PROFILES_DIR = "profiles"
HEADER = (
    "# Rendered by `archi` from the bundles and packs recorded in archi.lock.\n"
    "# Re-render instead of editing; `archi diff` reports hand edits.\n"
)


def profile_name(resolution: Resolution) -> str:
    return "-".join(resolution.given)


def _dump(data: Any) -> str:
    return HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88)


@dataclass(frozen=True)
class Rendered:
    """Where the profile went and what was in it."""

    profile_dir: Path
    name: str
    composition: Composition


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


AUTOMATIONS_DIR = "automations"


def write_automations(instance_dir: Path, comp: Composition) -> None:
    """Write `<instance>/automations/<name>.yaml`, one per selected automation pack.

    The runtime reads these (and `composed.automations` in archi.lock). The
    directory is rewritten on every render, so a dropped pack leaves no file.
    """
    out = instance_dir / AUTOMATIONS_DIR
    if out.exists():
        shutil.rmtree(out)
    for name, body in comp.automations.items():
        _write(out / f"{name}.yaml", _dump(body))


def render(
    resolution: Resolution,
    instance_dir: Path,
    *,
    name: Optional[str] = None,
) -> Rendered:
    """Write the profile for `resolution` under `instance_dir/profiles/<name>/`."""
    name = name or profile_name(resolution)
    comp = compose(resolution)
    defaults = deployment_defaults(resolution, comp, name)

    out = Path(instance_dir) / PROFILES_DIR / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    versions = [b.version for b in resolution.bundles if b.name in resolution.given]
    profile: dict[str, Any] = {
        "name": name,
        "version": versions[0] if len(versions) == 1 else "+".join(versions),
        "description": (
            "Composed by archi from bundle(s) "
            + ", ".join(resolution.given)
            + " with packs "
            + ", ".join(resolution.pack_names)
            + "."
        ),
        "init_questions": comp.questions,
        "modules": "modules.yaml",
        "source_defaults": "source-defaults/",
        "deployment_defaults": "deployment-defaults.yaml",
        "skills": "skills/",
    }
    if comp.invariants:
        profile["invariants"] = "invariants.yaml"
        _write(out / "invariants.yaml", _dump({"invariants": comp.invariants}))
    if comp.schemas:
        profile["schemas"] = "schemas/"
        for rel, data in comp.schemas.items():
            _write(out / "schemas" / rel, data)
    if comp.files:
        profile["files"] = sorted(comp.files)
        for rel, data in comp.files.items():
            _write(out / rel, data)

    _write(out / "profile.yaml", _dump(profile))
    _write(out / "modules.yaml", _dump({"modules": comp.modules}))
    _write(out / "deployment-defaults.yaml", _dump(defaults))
    (out / "source-defaults").mkdir()
    for sid, text in comp.sources.items():
        _write(out / "source-defaults" / f"{sid}.yaml", text)
    for rel, data in comp.playbooks.items():
        _write(out / "skills" / rel, data)
    _write(out / "skills" / PROMPT_FILE, comp.prompt)
    if comp.triggers is not None:
        _write(out / "skills" / TRIGGERS_FILE, _dump(comp.triggers))
    write_automations(Path(instance_dir), comp)
    return Rendered(out, name, comp)


def render_legacy(profile_dir: Path, instance_dir: Path) -> Path:
    """Copy a legacy okg profile (profile.yaml, no bundle.yaml) into the instance.

    Symlinks are followed, so the copy holds real files. okg then finds it in
    `<instance>/profiles/` with no OKG_PROFILES_DIR. Nothing is composed: a
    legacy profile takes no --with/--without.
    """
    out = Path(instance_dir) / PROFILES_DIR / Path(profile_dir).name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(profile_dir, out, symlinks=False)
    return out
