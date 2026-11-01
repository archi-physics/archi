"""`archi ingest [--instance DIR] [--include SOURCE,...]`: run okg's ingest for an instance.

It runs `okg ingest --deployment <name>` from the instance directory, in the
instance's environment: the shell's, plus every value in the instance's `.env`
(a value exported in the shell wins). A source reads its credential from that
environment, so a token written into `.env` reaches it. `okg ingest` run by
hand does not read `.env`.

Exit code 0 when okg's ingest succeeded, okg's own code when it failed, 2 when
the directory is not an instance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archi.stack.instance import Instance, InstanceError, read_env
from archi.stack.okg import Okg, OkgError


def ingest_args(instance: Instance, include: str | None = None) -> list[str]:
    args = ["ingest", "--deployment", instance.name]
    if include:
        args += ["--include", include]
    return args


def cmd_ingest(args: argparse.Namespace, runner=None) -> int:
    try:
        instance = Instance.load(args.instance)
        secrets = read_env(instance.env_path)
    except InstanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    okg = Okg(instance, secrets) if runner is None else Okg(instance, secrets, runner=runner)
    try:
        result = okg.run(*ingest_args(instance, args.include))
    except OkgError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.result.returncode if exc.result is not None and exc.result.returncode else 1
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser(
        "ingest",
        help="run okg ingest for the instance, with the credentials in its .env",
    )
    p.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    p.add_argument("--include", default=None,
                   help="comma-separated source ids to ingest (default: every source)")
    p.set_defaults(func=cmd_ingest)
