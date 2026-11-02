"""Shared fixtures for the pack/bundle tests.

`catalog_copy` hands each test its own copy of `fixtures/library/`, so a test
can break a manifest without touching the others.
"""
import shutil

import pytest

from bundles.bundle_support import CATALOG


@pytest.fixture
def catalog_copy(tmp_path):
    root = tmp_path / "catalog"
    shutil.copytree(CATALOG, root)
    return root
