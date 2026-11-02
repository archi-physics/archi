"""acc.r3-docs.gen: the generated docs pages equal what the generator produces now.

`archi.docs_gen` renders one page per bundle and per pack from their
manifests, and the CLI reference from the argparse parser. The pages are
committed under docs/docs/. A manifest or parser change without a regenerated
page fails here. Fix it with `cd python && python -m archi.docs_gen`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from archi import docs_gen
from archi.bundles.discovery import from_roots


def test_committed_pages_equal_the_generator_output():
    problems = docs_gen.stale()
    assert not problems, (
        "generated docs pages are out of date; run `cd python && python -m archi.docs_gen`:\n"
        + "\n".join(problems)
    )


def test_every_bundle_and_pack_has_a_page():
    catalog = from_roots([docs_gen.BUNDLES_DIR / "packs"], [docs_gen.BUNDLES_DIR])
    expected = {f"library/bundles/{n}.md" for n in catalog.bundles} | {
        f"library/packs/{n}.md" for n in catalog.packs
    }
    assert expected and expected <= set(docs_gen.generate())
    for rel in expected:
        assert (docs_gen.DOCS_DIR / rel).is_file(), rel


def test_pages_carry_the_manifest_facts():
    pages = docs_gen.generate()
    indico = pages["library/packs/indico.md"]
    assert "# Pack: indico" in indico and "`indico.indico_live`" in indico and "`24h`" in indico
    health = pages["library/packs/instance-health.md"]
    assert "`L2`" in health and "draft_issue" in health
    cern = pages["library/bundles/cern-team.md"]
    assert "[`docsite`](../packs/docsite.md)" in cern


def test_generation_is_deterministic():
    assert docs_gen.generate() == docs_gen.generate()


def test_a_stale_or_orphan_page_is_reported(tmp_path: Path):
    for rel, text in docs_gen.generate().items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert docs_gen.stale(docs_dir=tmp_path) == []
    (tmp_path / "library/packs/indico.md").write_text("edited\n", encoding="utf-8")
    (tmp_path / "library/packs/gone.md").write_text("orphan\n", encoding="utf-8")
    (tmp_path / "library/bundles/cern-team.md").unlink()
    assert sorted(docs_gen.stale(docs_dir=tmp_path)) == [
        "missing: library/bundles/cern-team.md",
        "orphan: library/packs/gone.md",
        "stale: library/packs/indico.md",
    ]


def test_pack_manifest_change_changes_the_page(tmp_path: Path):
    import shutil

    shutil.copytree(docs_gen.BUNDLES_DIR, tmp_path / "bundles")
    manifest = tmp_path / "bundles" / "packs" / "indico" / "pack.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("max_staleness: 24h", "max_staleness: 12h"),
        encoding="utf-8",
    )
    changed = docs_gen.generate(tmp_path / "bundles")["library/packs/indico.md"]
    assert "`12h`" in changed and "`12h`" not in docs_gen.generate()["library/packs/indico.md"]


@pytest.mark.parametrize("name", ["cern-team", "cms-physics", "cms-comp-ops"])
def test_bundle_pages_link_to_existing_pack_pages(name: str):
    page = docs_gen.generate()[f"library/bundles/{name}.md"]
    import re

    for target in re.findall(r"\]\(\.\./packs/([a-z0-9-]+)\.md\)", page):
        assert f"library/packs/{target}.md" in docs_gen.generate()
