"""The instance lifecycle commands: `archi init`, `up`, `down`, `doctor`, `ingest`, and `archi ops`.

`archi.bundles.cli` owns the parser; `register` adds these commands to it.
"""
from __future__ import annotations

import argparse


def register(sub, roots, load_ops: bool = True) -> None:
    from archi.cli import connect, doctor, evaluate, ingest, init, up

    init.add_parser(sub, roots)
    connect.add_parser(sub)
    up.add_parsers(sub)
    doctor.add_parser(sub)
    ingest.add_parser(sub)
    evaluate.add_parser(sub, roots)
    if load_ops:
        from archi.cli import ops

        ops.add_parser(sub, roots)
    else:
        # `archi ops` is the one command whose module is not imported for every other `archi` command:
        # this stub lists it in `archi --help`, and main() loads the real parser when ops is the command.
        stub = sub.add_parser("ops", help=OPS_HELP, add_help=False)
        stub.add_argument("rest", nargs=argparse.REMAINDER)
        stub.set_defaults(func=_ops_not_loaded)


OPS_HELP = "status, halt, resume, promote and answer the operations automations"


def _ops_not_loaded(args: argparse.Namespace) -> int:  # pragma: no cover - main() loads the real parser first
    raise SystemExit("archi ops: the ops parser was not loaded (build the parser with load_ops=True)")


def finalize(args: argparse.Namespace) -> None:
    from archi.cli.init import finalize_args

    finalize_args(args)
