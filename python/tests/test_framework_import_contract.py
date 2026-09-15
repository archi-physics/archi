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
    if not hasattr(module, symbol):
        # `from package import submodule` binds a submodule the package need not
        # have imported yet, so try it as a module before refusing.
        try:
            importlib.import_module(f"{module_name}.{symbol}")
        except ModuleNotFoundError:
            pytest.fail(f"selected framework lacks {module_name}.{symbol}")
