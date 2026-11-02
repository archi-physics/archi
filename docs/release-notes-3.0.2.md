# archi 3.0.2

One command installs Archi. `okg`, the graph engine from AKMON, is now on PyPI,
and `archi-kb` declares it, so pip installs both. Everything else is the same
as [3.0.1](https://github.com/archi-physics/archi/blob/v3.0.2/docs/release-notes-3.0.1.md).

## Install

You need Python 3.12 (okg does not install on 3.14), `docker` or `podman`
(with `compose`), and `git`. In a virtual environment:

```bash
pip install archi-kb==3.0.2
```

This installs `okg` 0.4.x with it, the version Archi is tested on. Do not
`pip install archi`: that name on PyPI belongs to an unrelated package.

## What changed

- **`archi-kb` depends on `okg>=0.4.0,<0.5`.** Before, okg came only as a wheel
  file on AKMON's GitHub releases. PyPI does not let a package depend on a
  URL, so `archi-kb` declared nothing, and `pip install archi-kb` succeeded
  without a graph engine. Every install line had to name the wheel and its
  sha256.
- **The install pages and the nightly test use the one line.** The nightly
  installs the checkout with no separate okg step, so it proves that pip
  finds okg on PyPI.

## If you already installed 3.0.1

In the same environment, `pip install archi-kb==3.0.2` upgrades Archi. The okg
already there (0.4.0) satisfies the new requirement, so pip leaves it as it
is. An instance made with 3.0.1 keeps working: it records okg 0.4.0, and that
does not change.

Start with the [quickstart](https://github.com/archi-physics/archi/blob/v3.0.2/docs/docs/quickstart.md).
The full history is in the [changelog](https://github.com/archi-physics/archi/blob/v3.0.2/CHANGELOG.md).
