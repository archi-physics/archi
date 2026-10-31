"""req.w2.packaging — the v3 package is the only build on this line.

The superseded v2 application was removed from the archi_v3 line
(W10 teardown, pulled forward); v2 lives on the `main` branch only.

Run with an okg-bearing interpreter (adapter modules import okg at
module scope):

    <okg-venv>/bin/python -m pytest python/tests/ -v
"""
import re
import subprocess
import sys
import zipfile
from importlib import resources
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_and_version():
    import archi

    assert re.fullmatch(r"3\.\d+\.\d+(a|b|rc)?\d*", archi.__version__)


def test_source_adapter_importable():
    from archi.sources.cmssw import CMSSWReleaseSource

    assert CMSSWReleaseSource.profile == "reference_catalog"
    assert CMSSWReleaseSource.change_probe_kind == "content_hash"


def test_schema_ships_in_package():
    schema_files = [
        p.name
        for p in resources.files("archi.schemas").iterdir()
        if p.name.endswith(".yaml")
    ]
    # operations.yaml absorbed the W1 operations_w1.yaml CMSSWRelease
    # slice with the catalogs port (req.w2.sources-catalogs).
    assert "operations.yaml" in schema_files
    assert "sources.yaml" in schema_files


def test_v3_does_not_import_v2():
    """python/archi must never import from the v2 tree (src/)."""
    v3_files = list((REPO_ROOT / "python" / "archi").rglob("*.py"))
    assert v3_files
    for path in v3_files:
        text = path.read_text(encoding="utf-8")
        assert "from src." not in text and "import src." not in text, path


def test_v2_tree_removed():
    """The superseded v2 application is gone from this line (W10 teardown);
    v2 lives on `main` only."""
    for v2_path in ("src", "setup.py", "configs"):
        assert not (REPO_ROOT / v2_path).exists(), v2_path


def test_no_operator_paths_in_package():
    """req.w2.auth lint floor, applied from day one: no hardcoded
    operator/host paths in package code. A path inside a container image
    (not on any operator's machine) is allowed on a line that says so with
    `# container-internal path`."""
    pattern = re.compile(r"[\"'](?:/Users/|/root/|/home/|/work/)")
    for path in (REPO_ROOT / "python" / "archi").rglob("*.py"):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.rstrip().endswith("# container-internal path"):
                continue
            assert not pattern.search(line), f"{path}:{lineno}: {line.strip()}"


def test_wheel_builds():
    """The distribution builds from python/ without touching the root."""
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel",
         "-d", str(REPO_ROOT / "python" / "dist")],
        cwd=REPO_ROOT / "python",
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-800:]
    wheels = list((REPO_ROOT / "python" / "dist").glob("archi_kb-3.*.whl"))
    assert wheels


def test_distribution_is_archi_kb_importing_archi(tmp_path):
    """req.r3-release-home.dist: the wheel is named archi-kb (the PyPI name
    `archi` is someone else's) and still installs the `archi` package."""
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=REPO_ROOT / "python",
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-800:]
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        (metadata,) = [n for n in names if n.endswith(".dist-info/METADATA")]
        headers = zf.read(metadata).decode().split("\n\n", 1)[0].splitlines()
    assert "Name: archi-kb" in headers
    assert "archi/__init__.py" in names
    top_level = {n.split("/", 1)[0] for n in names}
    assert top_level - {metadata.split("/", 1)[0]} == {"archi"}
