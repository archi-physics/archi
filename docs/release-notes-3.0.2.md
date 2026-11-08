# archi 3.0.2

One command installs Archi. The graph engine, `okg` from AKMON, is now on PyPI
as the package `akmon`, and `archi-kb` declares it, so pip installs both.
Archi now runs on AKMON 0.6.0 instead of 0.4.0. Everything else is the same as
[3.0.1](https://github.com/archi-physics/archi/blob/v3.0.2/docs/release-notes-3.0.1.md).

## Install

You need Python 3.12 (akmon does not install on 3.14), `docker` or `podman`
(with `compose`), and `git`. In a new virtual environment:

```bash
pip install archi-kb==3.0.2
```

This installs `akmon` 0.6.x with it, the version Archi is tested on. The
command is still `okg` and the Python import is still `okg`. Do not
`pip install archi`: that name on PyPI belongs to an unrelated package.

## What changed

- **`archi-kb` depends on `akmon>=0.6.0,<0.7`.** Before, the engine came only
  as a wheel file on AKMON's GitHub releases. PyPI does not let a package
  depend on a URL, so `archi-kb` declared nothing, `pip install archi-kb`
  succeeded without a graph engine, and every install line had to name the
  wheel and its sha256.
- **AKMON renamed its package from `okg` to `akmon` in 0.6.0.** Only the
  package name changed; `okg` the command and `import okg` did not. Archi now
  reads the engine's version from `akmon` (and from `okg` for an older
  engine), and the MCP container installs the wheel named for the instance's
  engine version (`akmon-0.6.0-…` from 0.6.0, `okg-…` before).
- **The bundles support AKMON 0.6.** `cern-team` 0.5.0, `cms-comp-ops`,
  `cluster-manager` and `cms-physics` 0.3.0 declare `okg: ">=0.6.0,<0.7"`.
- **`archi doctor` warns when both `okg` and `akmon` are installed.** They
  install the same files, so `pip uninstall okg` alone would break `akmon`.
- **The install pages and the nightly test use the one line.** The nightly
  installs the checkout with no separate engine step, so it proves that pip
  finds `akmon` on PyPI.

## If you already installed 3.0.1

3.0.1 ran on the `okg` 0.4.0 wheel. Install 3.0.2 in a new virtual
environment. In the old one, `pip install archi-kb==3.0.2` would put `akmon`
over `okg`'s files and leave both packages installed; `archi doctor` warns
about that. To reuse an environment anyway, first run `pip uninstall -y okg`.

An instance made with 3.0.1 records okg 0.4.0. Under 3.0.2, `archi doctor`
reports that the installed engine (0.6.0) differs from the instance's. Moving
an existing instance from 0.4.0 to 0.6.0 is not tested. Keep such an
instance on 3.0.1 in its own environment, or make a new instance with 3.0.2.

Start with the [quickstart](https://github.com/archi-physics/archi/blob/v3.0.2/docs/docs/quickstart.md).
The full history is in the [changelog](https://github.com/archi-physics/archi/blob/v3.0.2/CHANGELOG.md).
