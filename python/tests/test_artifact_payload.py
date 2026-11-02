"""The wheel must be installable as an artifact, not as a checkout.

`archi render --bundle cern-team` needs the bundle manifest and every pack it
names, with their templates, schema slices and playbooks. Before the
force-include those existed only in an authoring checkout of this repo. These
tests assert the *built wheel* carries them — `test_bundle_cern_team.py` checks
the working tree, which is exactly the thing that stayed green while the
artifact was incomplete.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PREFIX = "archi/_payload/bundles/cern-team/"
PACKS_PREFIX = "archi/_payload/bundles/packs/"


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory) -> list[str]:
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(out)],
        cwd=REPO_ROOT / "python",
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-800:]
    wheels = sorted(out.glob("archi_kb-3.*.whl"))
    assert wheels, "no wheel produced"
    with zipfile.ZipFile(wheels[-1]) as zf:
        return zf.namelist(), str(wheels[-1])


def test_wheel_ships_the_bundle_and_its_packs(wheel_names):
    import yaml

    names, _ = wheel_names
    assert f"{BUNDLE_PREFIX}bundle.yaml" in names
    bundle = yaml.safe_load((REPO_ROOT / "bundles" / "cern-team" / "bundle.yaml").read_text())
    for pack in bundle["packs"]["default"] + bundle["packs"]["optional"]:
        assert f"{PACKS_PREFIX}{pack}/pack.yaml" in names, pack


def test_wheel_ships_playbooks_dereferenced(wheel_names):
    """The packs' playbooks are symlinks into skills/ in the repo; a wheel
    cannot carry a link usefully, so they must arrive as real content."""
    names, wheel = wheel_names
    playbooks = [
        n
        for n in names
        if n.startswith(f"{PACKS_PREFIX}core-playbooks/playbooks/") and n.endswith(".md")
    ]
    assert len(playbooks) >= 13, playbooks
    with zipfile.ZipFile(wheel) as zf:
        for name in playbooks:
            body = zf.read(name).decode("utf-8")
            assert len(body) > 200, f"{name} looks like an unresolved link: {body!r}"
            assert body.lstrip().startswith("#"), f"{name} is not a playbook body"


def test_wheel_ships_the_distribution_playbooks(wheel_names):
    names, _ = wheel_names
    assert sum(1 for n in names if n.startswith("archi/_payload/skills/") and n.endswith(".md")) >= 15


def test_paths_helpers_resolve_to_a_real_directory():
    """Packaged copy when installed, checkout copy when running from source --
    either way the caller gets a directory that exists, never a dangling path."""
    from archi import paths

    assert paths.profiles_dir().name == "bundles"
    assert paths.playbooks_dir().name == "skills"
    assert paths.profiles_dir().is_dir()
    assert (paths.profiles_dir() / "cern-team" / "bundle.yaml").is_file()
    assert paths.bundle_dir("cern-team") == paths.profiles_dir() / "cern-team"
    assert paths.playbooks_dir().is_dir()


def test_bundle_dir_reports_available_bundles_when_missing():
    from archi import paths

    with pytest.raises(FileNotFoundError) as excinfo:
        paths.bundle_dir("no-such-bundle")
    assert "no-such-bundle" in str(excinfo.value)
    assert "available" in str(excinfo.value)


def test_wheel_carries_every_pack_schema_slice(wheel_names):
    """A pack's `schemas:` slot is only real if the ARTIFACT carries the files.

    The slices are symlinks into python/archi/schemas/ in the repo. force-include
    maps all of ../bundles, so this passes today; it exists so a narrowed
    include cannot silently drop the schemas and leave the artifact unable to
    complete an install.
    """
    names = set(wheel_names[0])
    missing = []
    for pack_yaml in sorted((REPO_ROOT / "bundles" / "packs").glob("*/pack.yaml")):
        schemas = pack_yaml.parent / "schemas"
        if not schemas.is_dir():
            continue
        for path in sorted(schemas.rglob("*.yaml")):
            name = f"archi/_payload/{path.relative_to(REPO_ROOT).as_posix()}"
            if name not in names:
                missing.append(name)
    assert not missing, f"wheel is missing: {missing}"
