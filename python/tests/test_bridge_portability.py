"""The shipped bridges must compose verbatim on every consumer.

W3's done-clause is "the cern-team bundle and the comp-ops instance compose the
same files without divergence". Before this test both consumers hand-pruned
`bridges/operations.yaml` -- and pruned *differently* -- because narrowings
whose endpoint subtypes come from modules a consumer does not load raised
`bridge_subtype_unknown` at catalog load.

`optional_when_subtypes_missing: 'true'` makes the composer skip such a
narrowing instead of failing (modules_compose.py: the row is never appended, so
it never reaches `_validate_narrowing_subtypes`). It is applied ONLY to
narrowings whose endpoints genuinely vary between consumers; everything whose
subtypes both consumers guarantee stays strict, so a module dropped by accident
still fails loudly rather than silently composing fewer narrowings.
"""
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("okg")
from okg.substrate.catalog.modules_compose import (  # noqa: E402
    _pascal_to_snake,
    compose_catalog,
)

SCHEMAS = Path(__file__).resolve().parents[1] / "archi" / "schemas"

# The two real consumers of the wheel's schema slices.
CERN_TEAM_MODULES = ["document_starter", "person", "extraction"]
COMPOPS_MODULES = ["document_starter", "person", "extraction", "dataset", "repo_starter"]


def _compose(modules: list[str]):
    tmp = Path(tempfile.mkdtemp())
    try:
        dep = tmp / "probe"
        (dep / "schemas" / "bridges").mkdir(parents=True)
        for name in ("operations.yaml", "sources.yaml"):
            shutil.copy(SCHEMAS / name, dep / "schemas" / name)
            shutil.copy(SCHEMAS / "bridges" / name, dep / "schemas" / "bridges" / name)
        (dep / "deployment.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "probe",
                    "postgres": {"dsn": "postgresql://unused/probe"},
                    "schema_dir": "schemas",
                    "modules": modules,
                }
            )
        )
        return compose_catalog(dep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize(
    "modules,label",
    [(CERN_TEAM_MODULES, "cern-team"), (COMPOPS_MODULES, "cms-compops")],
)
def test_shipped_bridges_compose_unmodified(modules, label):
    """No hand-pruning: the wheel's files as shipped, for both module sets."""
    composed = _compose(modules)
    assert composed.narrowings, f"{label} composed no narrowings"


def test_compops_superset_gains_narrowings():
    """The consumer that loads more modules gets strictly more narrowings --
    proving the flag skips only what is genuinely absent rather than dropping
    narrowings wholesale."""
    cern = _compose(CERN_TEAM_MODULES)
    compops = _compose(COMPOPS_MODULES)
    assert len(compops.narrowings) > len(cern.narrowings)
    assert set(cern.subtypes) <= set(compops.subtypes)


def test_flag_is_scoped_to_varying_endpoints_only():
    """Everything both consumers guarantee stays strict, so an accidentally
    dropped module still fails loudly (a silently skipped
    narrowing surfaces later as an ingest-time ProducerPolicyViolation)."""
    guaranteed = set(_compose(CERN_TEAM_MODULES).subtypes) & set(
        _compose(COMPOPS_MODULES).subtypes
    )
    doc = yaml.safe_load((SCHEMAS / "bridges" / "operations.yaml").read_text())
    over_flagged = []
    for name, attr in doc["classes"]["EdgeNarrowings"]["attributes"].items():
        ann = attr.get("annotations", {})
        if not ann.get("optional_when_subtypes_missing"):
            continue
        endpoints = [
            e.strip()
            for e in f"{ann.get('src_subtypes','')},{ann.get('dst_subtypes','')}".split(",")
            if e.strip()
        ]
        if all(_pascal_to_snake(e) in guaranteed for e in endpoints):
            over_flagged.append(name)
    assert not over_flagged, f"flag applied where both consumers guarantee the subtypes: {over_flagged}"
