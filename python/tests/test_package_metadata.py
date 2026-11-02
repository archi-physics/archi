"""The PyPI project page: what `archi-kb` shows there, and that it names this version.

A published version's description, links and license file cannot be changed
afterwards, so these are checked before the tag:

- the wheel declares a readme, a license with its text, and project links;
- every readme link is absolute, because PyPI cannot resolve a relative one;
- every versioned link, release-notes file and install line names the version
  being released;
- every install line uses the exact okg wheel (URL and sha256) that CI tests.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((REPO / "python" / "pyproject.toml").read_text(encoding="utf-8"))["project"]
VERSION = PROJECT["version"]
LINK = re.compile(r"\]\(([^)]+)\)")
CI = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

# The pages a user installs from: the PyPI page, the repository README and the docs.
INSTALL_PAGES = ["python/README.md", "README.md", "docs/docs/quickstart.md",
                 "docs/docs/migration-from-v2.md", f"docs/release-notes-{VERSION}.md"]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def pypi_readme() -> str:
    assert PROJECT.get("readme") == "README.md", "pyproject must name python/README.md as the readme"
    return text("python/README.md")


def test_the_package_declares_a_readme_license_and_links():
    pypi_readme()
    assert PROJECT.get("license") == "MIT"
    assert PROJECT.get("license-files") == ["LICENSE"]
    assert {"Homepage", "Documentation", "Release notes", "Issues"} <= set(PROJECT.get("urls", {}))


def test_the_license_text_in_the_package_is_the_repository_license():
    assert text("python/LICENSE") == text("LICENSE")


def test_every_readme_link_is_absolute():
    links = LINK.findall(pypi_readme())
    assert links, "the PyPI readme links to the docs"
    assert all(link.startswith("https://") for link in links), links


def test_versioned_links_name_this_version():
    for url in [*PROJECT["urls"].values(), *LINK.findall(pypi_readme())]:
        tags = re.findall(r"/(?:tree|blob)/([^/]+)/", url)
        assert all(tag == f"v{VERSION}" for tag in tags), url
        notes = re.findall(r"release-notes-([0-9][^/]*?)\.md", url)
        assert all(n == VERSION for n in notes), url


def test_this_versions_release_notes_exist_and_are_linked():
    assert (REPO / "docs" / f"release-notes-{VERSION}.md").is_file()
    assert f"docs/release-notes-{VERSION}.md" in PROJECT["urls"]["Release notes"]
    assert f"docs/release-notes-{VERSION}.md" in pypi_readme()


def test_release_notes_link_to_their_own_tag():
    # 3.0.0's notes linked to main, which held v2, and the links 404ed on the
    # GitHub release page. Each notes file must name its own tag instead.
    repo_link = re.compile(r"https://github\.com/archi-physics/archi/(?:blob|tree)/([^/]+)/")
    notes = sorted((REPO / "docs").glob("release-notes-*.md"))
    assert notes
    for path in notes:
        version = path.stem.removeprefix("release-notes-")
        refs = repo_link.findall(path.read_text(encoding="utf-8"))
        assert refs and set(refs) == {f"v{version}"}, (path.name, sorted(set(refs)))


def test_install_lines_pin_this_version():
    for rel in INSTALL_PAGES:
        pins = re.findall(r"archi-kb==([0-9][^\s\"`]*)", text(rel))
        assert pins and set(pins) == {VERSION}, (rel, pins)


def test_install_lines_use_the_okg_wheel_ci_tests():
    pinned = re.search(r'okg @ (https://\S+?#sha256=[0-9a-f]{64})', CI)
    assert pinned, "ci.yml pins the okg wheel by URL and sha256"
    for rel in INSTALL_PAGES:
        wheels = re.findall(r"okg @ (https://[^\s\"]+)", text(rel))
        assert wheels and set(wheels) == {pinned.group(1)}, (rel, wheels)


def test_no_install_page_says_archi_is_unpublished():
    for rel in INSTALL_PAGES:
        assert "not on PyPI yet" not in text(rel), rel
