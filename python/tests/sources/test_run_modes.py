"""archi's copy of the scope-completing run modes equals okg's.

Connectors may not import ``okg.substrate`` (``okg lint deployments``,
DEPLOYMENT-001), so ``archi.sources._run_modes`` keeps its own copy.
Tests are not linted, so this test reads okg's set directly and fails
at the first okg pin whose set differs.
"""

import importlib

from archi.sources._run_modes import COMPLETED_SCOPE_RUN_MODES, RELEASE_RUN_MODES


def _okg_completed_scope_run_modes():
    # Test-only lookup: prefer the public SDK export when the pinned okg
    # has it (okg branch claude/connector-complete-scope-modes adds it),
    # otherwise read the substrate module that defines the table.
    deployment = importlib.import_module("okg.deployment")
    if hasattr(deployment, "COMPLETED_SCOPE_RUN_MODES"):
        return deployment.COMPLETED_SCOPE_RUN_MODES
    module = importlib.import_module("okg.substrate.source_run_modes")
    return module.COMPLETED_SCOPE_RUN_MODES


def test_archi_completed_scope_run_modes_match_okg():
    assert COMPLETED_SCOPE_RUN_MODES == frozenset(_okg_completed_scope_run_modes())


def test_archi_release_run_modes_match_okg_reference_catalog():
    # Test-only lookup of okg's per-profile table; okg.deployment does not
    # export it.
    module = importlib.import_module("okg.substrate.source_run_modes")
    accepted = module.PROFILE_RUN_MODES["reference_catalog"].accepted
    assert RELEASE_RUN_MODES == frozenset(accepted)
