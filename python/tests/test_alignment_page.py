"""Policy guard: docs/okg-alignment.md must not drift from reality.

The alignment page is the OKG-side program's window into Archi
(cross-repo coordination). Its documented substrate-import surface must
exactly match what python/archi actually imports — if you add, remove,
or move an okg import, update the page's import block in the same
change. The page must also carry a parseable "Last updated" line.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "python" / "archi"
PAGE = REPO / "docs" / "okg-alignment.md"

_IMPORT_RE = re.compile(
    r"from (okg[\w.]*) import (\([^)]*\)|[^\n]+)", re.S
)


def _code_imports():
    found = set()
    for path in PKG.rglob("*.py"):
        for mod, block in _IMPORT_RE.findall(
            path.read_text(encoding="utf-8")
        ):
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", block):
                if name != "as":
                    found.add((mod, name))
    return found


def _page_imports():
    text = PAGE.read_text(encoding="utf-8")
    match = re.search(r"```\n(okg\..*?)```", text, re.S)
    assert match, "alignment page lost its substrate-import code block"
    found = set()
    current = None
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        header = re.match(r"(okg[\w.]+):\s*(.*)", line)
        if header:
            current = header.group(1)
            rest = header.group(2)
        else:
            rest = line
        assert current, f"symbols before any module header: {line!r}"
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rest):
            found.add((current, name))
    return found


def test_documented_import_surface_matches_code():
    code = _code_imports()
    page = _page_imports()
    missing_from_page = sorted(code - page)
    stale_on_page = sorted(page - code)
    assert not missing_from_page and not stale_on_page, (
        "docs/okg-alignment.md import block drifted from python/archi.\n"
        f"In code but not on the page: {missing_from_page}\n"
        f"On the page but not in code: {stale_on_page}\n"
        "Update the page's 'exact substrate surface' block in this "
        "same change (cross-repo coordination policy)."
    )


def test_page_carries_last_updated_pins():
    text = PAGE.read_text(encoding="utf-8")
    assert re.search(
        r"\*Last updated \d{4}-\d{2}-\d{2}.*@ `[0-9a-f]{7,}`", text
    ), (
        "alignment page needs a parseable '*Last updated YYYY-MM-DD ... "
        "@ `commit`' line with current branch/okg pins"
    )
