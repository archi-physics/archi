"""Find packs and bundles.

A *pack root* is a directory holding `<name>/pack.yaml` entries; a *bundle
root* holds `<name>/bundle.yaml` entries. Roots come from two entry-point
groups (ADR 0002 section 3.4), so any installed package can contribute:

    [project.entry-points."archi.packs"]
    mywg = "mywg_packs:packs_dir"        # returns a pack root
    [project.entry-points."archi.bundles"]
    mywg = "mywg_packs:bundles_dir"      # returns a bundle root

The entry point's value is a callable returning the directory (or the
directory itself, as a str or Path); its name labels the provider. archi
registers its own roots the same way (`archi.paths:packs_dir`,
`archi.paths:profiles_dir`). A directory under a bundle root that still holds
an okg `profile.yaml` and no `bundle.yaml` is listed as a legacy profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Iterable, Optional

PACK_GROUP = "archi.packs"
BUNDLE_GROUP = "archi.bundles"
LEGACY_PROFILE = "profile.yaml"

from archi.bundles.manifest import (
    BUNDLE_FILE,
    PACK_FILE,
    Bundle,
    Pack,
    Problem,
    load_pack,
    ManifestError,
    read_bundle,
    validate_bundle,
    validate_pack,
)


@dataclass
class Catalog:
    """Every pack and bundle found, plus the problems found on the way.

    A pack or bundle that fails to load is left out of `packs`/`bundles` and
    reported in `problems`, so one broken manifest does not hide the rest.
    """

    packs: dict[str, Pack] = field(default_factory=dict)
    bundles: dict[str, Bundle] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    #: name -> provider label (entry-point name, or the root's path)
    providers: dict[str, str] = field(default_factory=dict)
    #: okg profile directories without a bundle.yaml, by name
    legacy_profiles: dict[str, Path] = field(default_factory=dict)

    def add_pack_dir(self, directory: Path, provider: str = "") -> None:
        try:
            pack = load_pack(directory, defer_requires=True)
        except ManifestError as exc:
            self.problems.extend(exc.problems)
            return
        other = self.packs.get(pack.name)
        if other is not None and other.dir.resolve() != pack.dir.resolve():
            self.problems.append(
                Problem(pack.dir / PACK_FILE, f"pack {pack.name!r} is also provided by {other.dir}")
            )
            return
        self.packs[pack.name] = pack
        self.providers[f"pack:{pack.name}"] = provider or str(directory.parent)

    def add_bundle_dir(self, directory: Path, provider: str = "") -> None:
        bundle, problems = read_bundle(directory)
        self.problems.extend(problems)
        if bundle is None or problems:
            return
        other = self.bundles.get(bundle.name)
        if other is not None and other.dir.resolve() != bundle.dir.resolve():
            self.problems.append(
                Problem(bundle.dir / BUNDLE_FILE, f"bundle {bundle.name!r} is also provided by {other.dir}")
            )
            return
        self.bundles[bundle.name] = bundle
        self.providers[f"bundle:{bundle.name}"] = provider or str(directory.parent)

    def check_packs(self) -> None:
        """Check what needs the other packs: an automation's mediators resolve
        against the live tools of the packs it requires, in any root."""
        for name in sorted(self.packs):
            pack = self.packs[name]
            if pack.kind != "automation":
                continue
            problems = validate_pack(pack.dir, packs=self.packs)
            if problems:
                self.problems.extend(problems)
                del self.packs[name]

    def check_bundles(self) -> None:
        """Check each bundle's references now that every pack is known."""
        failed = []
        for name in sorted(self.bundles):
            problems = validate_bundle(self.bundles[name].dir, self.packs, self.bundles)
            if problems:
                self.problems.extend(problems)
                failed.append(name)
        for name in failed:
            del self.bundles[name]


def _children(root: Path, manifest: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / manifest).is_file())


Root = tuple[str, Path]  # (provider label, directory)


def _as_roots(roots: Iterable) -> list[Root]:
    return [r if isinstance(r, tuple) else ("", Path(r)) for r in roots]


def from_roots(pack_roots: Iterable = (), bundle_roots: Iterable = ()) -> Catalog:
    """Load every pack and bundle under the given roots.

    A root is a directory, or a (provider label, directory) pair.
    """
    catalog = Catalog()
    for provider, root in _as_roots(pack_roots):
        for directory in _children(root, PACK_FILE):
            catalog.add_pack_dir(directory, provider)
    for provider, root in _as_roots(bundle_roots):
        for directory in _children(root, BUNDLE_FILE):
            catalog.add_bundle_dir(directory, provider)
        if root.is_dir():
            for directory in sorted(root.iterdir()):
                if (directory / LEGACY_PROFILE).is_file() and not (directory / BUNDLE_FILE).is_file():
                    catalog.legacy_profiles.setdefault(directory.name, directory)
    catalog.check_packs()
    catalog.check_bundles()
    return catalog


def entry_point_roots(group: str) -> tuple[list[Root], list[Problem]]:
    """(provider, directory) for every installed `group` entry point."""
    roots: list[Root] = []
    problems: list[Problem] = []
    for ep in sorted(entry_points(group=group), key=lambda e: e.name):
        where = Path(f"<entry point {group}:{ep.name} = {ep.value}>")
        try:
            target = ep.load()
            directory = Path(target() if callable(target) else target)
        except Exception as exc:  # a broken third-party package must not hide the rest
            problems.append(Problem(where, f"failed to load: {exc!r}"))
            continue
        provider = ep.name if ep.dist is None else f"{ep.name} ({ep.dist.name} {ep.dist.version})"
        roots.append((provider, directory))
    return roots, problems


def discover(extra_pack_roots: Iterable = (), extra_bundle_roots: Iterable = ()) -> Catalog:
    """Everything installed packages register, plus any extra roots."""
    pack_roots, pack_problems = entry_point_roots(PACK_GROUP)
    bundle_roots, bundle_problems = entry_point_roots(BUNDLE_GROUP)
    catalog = from_roots(
        [*pack_roots, *_as_roots(extra_pack_roots)],
        [*bundle_roots, *_as_roots(extra_bundle_roots)],
    )
    catalog.problems[:0] = pack_problems + bundle_problems
    return catalog
