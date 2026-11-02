"""The PyPI project page: what `archi-kb` shows there, and that it names this version.

A published version's description and links cannot be changed afterwards, so
these are checked before the tag: the wheel has a readme and project links,
every link is absolute (PyPI cannot resolve a relative one), every versioned
link and install line names the version being released, and the repository
README installs that same version.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((REPO / "python" / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
VERSION = PROJECT["version"]
README = (REPO / "python" / PROJECT["readme"]).read_text(encoding="utf-8")
LINK = re.compile(r"\]\(([^)]+)\)")


def test_the_package_declares_a_readme_license_and_links():
    assert PROJECT["readme"] == "README.md"
    assert PROJECT["license"] == "MIT"
    assert {"Homepage", "Documentation", "Release notes", "Issues"} <= set(PROJECT["urls"])


def test_every_readme_link_is_absolute():
    links = LINK.findall(README)
    assert links, "the PyPI readme links to the docs"
    assert all(link.startswith("https://") for link in links), links


def test_versioned_links_name_this_version():
    for url in [*PROJECT["urls"].values(), *LINK.findall(README)]:
        tags = re.findall(r"/(?:tree|blob)/([^/]+)/", url)
        assert all(tag == f"v{VERSION}" for tag in tags), url


def test_install_lines_pin_this_version():
    for name, text in [("python/README.md", README),
                       ("README.md", (REPO / "README.md").read_text(encoding="utf-8"))]:
        pins = re.findall(r"archi-kb==([0-9][^\s\"`]*)", text)
        assert pins and set(pins) == {VERSION}, (name, pins)


def test_the_readme_does_not_say_archi_is_unpublished():
    for text in [README, (REPO / "README.md").read_text(encoding="utf-8")]:
        assert "not on PyPI yet" not in text
