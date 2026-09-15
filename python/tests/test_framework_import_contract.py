"""The consumer owns verification of its actual framework import surface.

Keep this separate from test_alignment_page: that static alignment check also
runs in substrate-free CI. This contract requires the selected framework.
"""

import importlib

import pytest
from test_alignment_page import _code_imports


@pytest.mark.parametrize("module_name,symbol", sorted(_code_imports()))
def test_actual_consumer_framework_import_resolves(module_name, symbol):
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"selected framework lacks {module_name}.{symbol}"
