"""Helpers shared by the pack/bundle tests (imported by module name)."""
from pathlib import Path

import yaml

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CATALOG = FIXTURES / "library"


def edit_yaml(path: Path, change) -> None:
    """Load `path`, let `change(data)` mutate it, write it back."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# cern-team, the shipped bundle of packs
# ---------------------------------------------------------------------------
#
# The renderer's real input: bundles/cern-team/bundle.yaml and the packs under
# bundles/packs/ (PACT r3-packs-split). build_cern_team_catalog copies them,
# symlinks followed, so a test can change a pack without touching the repo.

import shutil

REPO = Path(__file__).resolve().parents[3]
CERN_TEAM = REPO / "bundles" / "cern-team"
SHIPPED_PACKS = REPO / "bundles" / "packs"


def _cern_team_manifest():
    return yaml.safe_load((CERN_TEAM / "bundle.yaml").read_text(encoding="utf-8"))


CERN_TEAM_DEFAULT = list(_cern_team_manifest()["packs"]["default"])
CERN_TEAM_OPTIONAL = list(_cern_team_manifest()["packs"]["optional"])


def build_cern_team_catalog(dest: Path) -> Path:
    """Copy the shipped packs and the cern-team bundle under `dest`; return `dest`."""
    shutil.copytree(SHIPPED_PACKS, dest / "packs", symlinks=False)
    shutil.copytree(CERN_TEAM, dest / "cern-team", symlinks=False)
    return dest


def clone_pack(root: Path, source: str, name: str, change=None) -> Path:
    """Copy pack `source` under `root/packs/` as `name`, renaming everything
    that carries the pack name, then let `change(manifest)` edit it."""
    src = root / "packs" / source
    dst = root / "packs" / name
    shutil.copytree(src, dst)
    data = yaml.safe_load((dst / "pack.yaml").read_text())
    data["name"] = name
    if "freshness" in data:
        data["freshness"] = {
            f"{name}.{k.split('.', 1)[1]}": v for k, v in data["freshness"].items()
        }
    if change is not None:
        change(data)
    (dst / "pack.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return dst


def write_bundle(root: Path, name: str, **fields) -> Path:
    data = {"name": name, "version": "0.1.0", "description": f"{name} test bundle."}
    data.update(fields)
    data.setdefault("okg", ">=0.4.0,<0.5")
    bdir = root / name
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "bundle.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return bdir


def load_catalog(root: Path):
    from archi.bundles import discovery

    catalog = discovery.from_roots([root / "packs"], [root])
    assert catalog.problems == [], [str(p) for p in catalog.problems]
    return catalog
