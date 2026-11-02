"""req.r3-bundle-format.packaging: what the built wheel carries and finds.

Every test here works on built artifacts, not the working tree: the wheel is
built with hatchling and unpacked into a directory that goes first on
PYTHONPATH of a child process, so `archi` and its entry points come from the
wheel (importlib.metadata keeps the first distribution of a name on the path).

- acc.r3-bundle-format.wheel: no symlink in the wheel; each bundle in the
  repo is listed by `archi bundles list` run from the wheel.
- acc.r3-bundle-format.third-party: a fixture package registering a pack and
  a bundle through entry points is listed, renders, and okg accepts the result.
"""
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from bundles.bundle_support import FIXTURES, REPO

THIRD_PARTY = FIXTURES / "thirdparty"
OKG = str(Path(sys.executable).with_name("okg"))


def _build_wheel(project: Path, out: Path) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(out)],
        cwd=project, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    (wheel,) = out.glob("*.whl")
    return wheel


@pytest.fixture(scope="module")
def archi_wheel(tmp_path_factory) -> Path:
    return _build_wheel(REPO / "python", tmp_path_factory.mktemp("archi-wheel"))


@pytest.fixture(scope="module")
def site(tmp_path_factory, archi_wheel) -> Path:
    """archi's wheel and the third-party fixture's wheel, unpacked side by side."""
    site = tmp_path_factory.mktemp("site")
    zipfile.ZipFile(archi_wheel).extractall(site)
    tp = _build_wheel(THIRD_PARTY, tmp_path_factory.mktemp("tp-wheel"))
    zipfile.ZipFile(tp).extractall(site)
    return site


def _run(site: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "OKG_PROFILES_DIR"}
    env["PYTHONPATH"] = str(site)
    return subprocess.run(
        [sys.executable, "-c",
         "import sys, archi; from archi.bundles.cli import main; "
         "print('ARCHI_FROM', archi.__file__, file=sys.stderr); sys.exit(main(sys.argv[1:]))",
         *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=120,
    )


def _listing(site, cwd):
    result = _run(site, cwd, "bundles", "list", "--json")
    assert f"ARCHI_FROM {site}" in result.stderr, result.stderr  # the wheel, not the checkout
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _okg(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "OKG_PROFILES_DIR"}
    env["HOME"] = str(cwd)
    return subprocess.run([OKG, *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=120)


# ---------------------------------------------------------------------------
# acc.r3-bundle-format.wheel
# ---------------------------------------------------------------------------


def test_wheel_contains_no_symlinks(archi_wheel):
    with zipfile.ZipFile(archi_wheel) as zf:
        links = [i.filename for i in zf.infolist() if stat.S_ISLNK(i.external_attr >> 16)]
    assert links == []


def test_wheel_ships_every_payload_file_as_real_content(archi_wheel):
    """Each file under bundles/ and skills/, symlinks followed, is in the wheel
    byte for byte under archi/_payload/."""
    with zipfile.ZipFile(archi_wheel) as zf:
        names = set(zf.namelist())
        for tree in ("bundles", "skills"):
            for path in sorted((REPO / tree).rglob("*")):
                if not path.is_file():
                    continue
                name = f"archi/_payload/{path.relative_to(REPO).as_posix()}"
                assert name in names, name
                assert zf.read(name) == path.read_bytes(), name


def test_wheel_ships_the_manifest_schemas(archi_wheel):
    with zipfile.ZipFile(archi_wheel) as zf:
        names = zf.namelist()
    for name in ("pack.schema.json", "bundle.schema.json"):
        assert f"archi/bundles/schema/{name}" in names
    for name in ("questions.yaml", "deployment-defaults.yaml", "chat-system-prompt.md"):
        assert f"archi/bundles/base/{name}" in names


def test_wheel_registers_the_entry_points(archi_wheel):
    with zipfile.ZipFile(archi_wheel) as zf:
        (ep_file,) = [n for n in zf.namelist() if n.endswith(".dist-info/entry_points.txt")]
        text = zf.read(ep_file).decode()
    assert '[archi.packs]\narchi = archi.paths:packs_dir' in text
    assert '[archi.bundles]\narchi = archi.paths:profiles_dir' in text
    assert "archi = archi.bundles.cli:main" in text


def test_bundles_list_from_the_wheel_lists_every_bundle(site, tmp_path):
    listing = _listing(site, tmp_path)
    found = {b["name"]: b["dir"] for b in listing["bundles"]}
    found.update({p["name"]: p["dir"] for p in listing["legacy_profiles"]})
    expected = sorted(
        d.name for d in (REPO / "bundles").iterdir()
        if (d / "bundle.yaml").is_file() or (d / "profile.yaml").is_file()
    )
    assert expected, "the repo ships no bundle"
    for name in expected:
        assert name in found, name
        assert Path(found[name]).is_relative_to(site), found[name]
    expected_packs = sorted(
        d.name for d in (REPO / "bundles" / "packs").glob("*") if (d / "pack.yaml").is_file()
    )
    archi_packs = sorted(p["name"] for p in listing["packs"] if p["provider"].startswith("archi "))
    assert archi_packs == expected_packs


def test_the_shipped_cern_team_bundle_installs_without_okg_profiles_dir(site, tmp_path):
    """`archi render --bundle cern-team` composes the wheel's packs into the
    instance; okg, run from there, finds the profile with OKG_PROFILES_DIR
    unset and accepts it."""
    instance = tmp_path / "instance"
    result = _run(site, tmp_path, "render", "--bundle", "cern-team", "--instance", str(instance))
    assert result.returncode == 0, result.stderr
    assert "with packs docsite, indico, core-playbooks" in result.stdout
    rendered = instance / "profiles" / "cern-team"
    assert not [p for p in rendered.rglob("*") if p.is_symlink()]
    report = _okg(instance, "install", "--profile", "cern-team", "--deployment-name", "qt",
                  "--postgres-dsn", "postgresql://localhost:1/qt",
                  "--deployments-root", str(tmp_path / "deployments"),
                  "--dry-run", "--json", "--non-interactive")
    assert report.returncode == 0, report.stderr[-1500:] + report.stdout[-1500:]
    data = json.loads(report.stdout)
    assert data["status"] == "dry_run_ok" and data["blockers"] == []


# ---------------------------------------------------------------------------
# acc.r3-bundle-format.third-party
# ---------------------------------------------------------------------------


def test_a_third_party_pack_and_bundle_are_listed(site, tmp_path):
    listing = _listing(site, tmp_path)
    packs = {p["name"]: p for p in listing["packs"]}
    bundles = {b["name"]: b for b in listing["bundles"]}
    assert packs["repo-mirror"]["provider"] == "fixture (archi-fixture-packs 0.0.1)"
    assert bundles["mirror-team"]["provider"] == "fixture (archi-fixture-packs 0.0.1)"
    assert Path(packs["repo-mirror"]["dir"]).is_relative_to(site / "archi_fixture_packs")
    assert listing["problems"] == []

    show = _run(site, tmp_path, "bundles", "show", "repo-mirror")
    assert show.returncode == 0 and json.loads(show.stdout)["type"] == "pack"


def test_a_third_party_bundle_installs(site, tmp_path):
    instance = tmp_path / "instance"
    result = _run(site, tmp_path, "render", "--bundle", "mirror-team", "--instance", str(instance))
    assert result.returncode == 0, result.stderr
    assert "with packs repo-mirror" in result.stdout
    report = _okg(instance, "install", "--profile", "mirror-team", "--deployment-name", "qt",
                  "--postgres-dsn", "postgresql://localhost:1/qt",
                  "--repo-mirror-url", "https://example.org/demo.git",
                  "--deployments-root", str(tmp_path / "deployments"),
                  "--dry-run", "--json", "--non-interactive")
    assert report.returncode == 0, report.stderr[-1500:] + report.stdout[-1500:]
    data = json.loads(report.stdout)
    assert data["status"] == "dry_run_ok" and data["blockers"] == []
    assert data["answers"]["repo_mirror_url"] == "https://example.org/demo.git"


def test_a_broken_entry_point_is_reported_not_fatal(site, tmp_path):
    """A third-party package whose entry point fails to load is a listed
    problem; everything else is still found."""
    broken = tmp_path / "broken"
    dist = broken / "archi_broken-0.0.1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: archi-broken\nVersion: 0.0.1\n")
    (dist / "entry_points.txt").write_text("[archi.packs]\nbroken = no_such_module:packs_dir\n")
    env_site = f"{site}{os.pathsep}{broken}"
    result = _run(Path(env_site), tmp_path, "bundles", "list", "--json")
    listing = json.loads(result.stdout)
    assert result.returncode == 1
    assert any("archi.packs:broken" in p and "failed to load" in p for p in listing["problems"])
    assert "repo-mirror" in {p["name"] for p in listing["packs"]}
