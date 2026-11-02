"""Locate the payload the wheel carries.

The wheel carries the bundles, packs and playbooks (see
`[tool.hatch.build.targets.wheel.force-include]` in pyproject.toml), and these
helpers point at them wherever pip put them. Before that the files only
existed in an authoring checkout of this repo, so an install required cloning
archi even though the code arrived as a wheel.

`archi render --bundle cern-team --instance DIR` composes a bundle's packs
into `DIR/profiles/cern-team/`, where okg finds it with no OKG_PROFILES_DIR.
`archi-profiles-dir` still prints the payload's `bundles/` directory for a
bundle shipped as a plain okg profile (`<name>/profile.yaml`), which okg
can read through OKG_PROFILES_DIR.
"""
from __future__ import annotations

import argparse
from pathlib import Path


#: Where the wheel puts the repo-root `bundles/` and `skills/` trees. Not
#: `archi/bundles/`, which is the `archi.bundles` code package.
PAYLOAD = "_payload"


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve(name: str) -> Path:
    """The packaged copy, or the checkout's copy when running from source.

    An editable install has no `archi/_payload/` — the payload only lands there
    when the wheel is built. Falling back to the repo root keeps the helper
    honest in a development tree instead of returning a path that does not
    exist; the packaged copy always wins when present.
    """
    packaged = _package_root() / PAYLOAD / name
    if packaged.is_dir():
        return packaged
    checkout = _package_root().parents[1] / name
    if checkout.is_dir():
        return checkout
    return packaged


def profiles_dir() -> Path:
    """The payload's `bundles/`: `<bundle>/bundle.yaml`, `packs/<pack>/pack.yaml`."""
    return _resolve("bundles")


def packs_dir() -> Path:
    """Directory of archi's own packs (`<name>/pack.yaml`)."""
    return profiles_dir() / "packs"


def playbooks_dir() -> Path:
    """The distribution's playbooks, materialised in the wheel (never symlinks)."""
    return _resolve("skills")


def bundle_dir(name: str) -> Path:
    """One bundle by name, e.g. `bundle_dir("cern-team")`.

    A bundle directory holds a `bundle.yaml`, or an okg `profile.yaml` for a
    bundle not yet expressed as packs.
    """

    def is_bundle(path: Path) -> bool:
        return (path / "bundle.yaml").is_file() or (path / "profile.yaml").is_file()

    candidate = profiles_dir() / name
    if not is_bundle(candidate):
        available = sorted(
            p.name for p in profiles_dir().iterdir() if is_bundle(p)
        ) if profiles_dir().is_dir() else []
        raise FileNotFoundError(
            f"no bundle {name!r} in {profiles_dir()}; available: {available or 'none'}"
        )
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="archi-profiles-dir",
        description="Print the installed bundle directory for OKG_PROFILES_DIR.",
    )
    parser.add_argument(
        "--playbooks",
        action="store_true",
        help="print the materialised playbooks directory instead",
    )
    parser.add_argument("--bundle", help="print one bundle's directory by name")
    args = parser.parse_args(argv)
    if args.bundle:
        print(bundle_dir(args.bundle))
    elif args.playbooks:
        print(playbooks_dir())
    else:
        print(profiles_dir())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
