"""Import the working-tree package ahead of any installed archi."""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


import subprocess

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_prepare: let archi init/add run real pack prepare steps (network)"
    )


@pytest.fixture(autouse=True)
def _no_real_prepare_steps(request, monkeypatch):
    """Unit tests never run a pack's real prepare step.

    A prepare step downloads (docsite fetches public pages). `archi init` and
    `archi add` call `archi.bundles.prepare.run_steps` with the default
    runner unless a test passes its own (`prepare_runner=` for init); with the
    default runner this refuses, so a test that forgot to inject one fails
    instead of using the network. A test that really needs one marks itself
    `real_prepare`.
    """
    if request.node.get_closest_marker("real_prepare"):
        return
    from archi.bundles import prepare

    real = prepare.run_steps

    def guarded(steps, data_root, *, echo=print, runner=subprocess.run):
        if steps and runner is subprocess.run:
            raise AssertionError(
                "a unit test ran a real pack prepare step "
                f"({', '.join(s.label for s in steps)}); inject a fake runner "
                "(init: prepare_runner=) or mark the test real_prepare"
            )
        return real(steps, data_root, echo=echo, runner=runner)

    monkeypatch.setattr(prepare, "run_steps", guarded)
