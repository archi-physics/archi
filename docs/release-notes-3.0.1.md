# archi 3.0.1

Archi 3.0.1 runs on AKMON 0.4.0, the current release of the graph engine.
3.0.0 ran on AKMON 0.2.0. The package is `archi-kb` on PyPI. You import it as
`archi`. Everything in [3.0.0](https://github.com/archi-physics/archi/blob/v3.0.1/docs/release-notes-3.0.0.md)
is still here, and so are its gaps.

## What changed

- **AKMON 0.4.0.** CI, the nightly test and every install line now use the
  0.4.0 wheel, pinned by sha256. The four bundles declare
  `okg >=0.4.0,<0.5`, and their versions go up by one minor step for that
  reason. From 0.3.0 and 0.4.0, agents get exact-id matches ranked first, a plain
  answer when no record has a name, and a smaller database. See the
  [AKMON 0.4.0 release](https://github.com/mitdbg/akmon/releases/tag/v0.4.0).
- **Agents are told about file texts again.** AKMON 0.4.0 tells agents about
  `doc:` text records only when it recognises a document source, and it does not
  recognise the `code-text` pack's source. An instance with `code-text` now
  declares those records itself, so its agents are told how to read them.
- **Instance names cannot contain `_`.** AKMON 0.4.0's chat refuses such a
  deployment name, so `archi init my_team` used to succeed and `archi up` then
  failed. `archi init` now refuses the name at once. Use `my-team`.
- **Release notes link to their own tag.** 3.0.0's links pointed at `main`,
  which still holds Archi v2, so three of them were broken.

## Install

You need `docker` or `podman` (with `compose`), Python 3.12 (AKMON 0.4.0 does
not install on 3.14), and `git`. Use a virtual environment.

```bash
pip install "okg @ https://github.com/mitdbg/akmon/releases/download/v0.4.0/okg-0.4.0-py3-none-any.whl#sha256=18ccec94040e23bccae9b51274b6aafb5bfbb9d512dc6c37506d27fac31c1abc" archi-kb==3.0.1
```

`okg` is AKMON's graph engine. It is not on PyPI, so it comes from the AKMON
release. Do not `pip install archi`: that name belongs to an unrelated package.

## If you made an instance with 3.0.0

An instance records the okg version it was made with, and `archi doctor` fails
when the installed okg differs. Keep a 3.0.0 instance's environment on
archi-kb 3.0.0 with okg 0.2.0, or make the instance again with 3.0.1.
Moving an existing instance's database from AKMON 0.2.0 to 0.4.0 is not tested
in 3.0.1. AKMON's upgrade deletes data it no longer uses, so back up first.

Start with the [quickstart](https://github.com/archi-physics/archi/blob/v3.0.1/docs/docs/quickstart.md).
The full history is in the [changelog](https://github.com/archi-physics/archi/blob/v3.0.1/CHANGELOG.md).
