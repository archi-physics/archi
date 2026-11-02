"""`archi ingest [--instance DIR] [--include SOURCE,...] [--publish]`: run okg's ingest for an instance.

It runs `okg ingest --deployment <name>` from the instance directory, in the
instance's environment: the shell's, plus every value in the instance's `.env`
(a value exported in the shell wins). A source reads its credential from that
environment, so a token written into `.env` reaches it. `okg ingest` run by
hand does not read `.env`.

With an automation pack selected (the ops runtime is on), okg would queue the
ingest for a runtime worker, which does not have this environment: the ingest
runs in-process (`--inline`), as `archi init` does. `--publish` then publishes a
new generation (`okg run --once --apply`), as `archi add` does.

Exit code 0 when every okg step succeeded, okg's own code when one failed, 2
when the directory is not an instance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archi.stack.instance import Instance, InstanceError, read_env
from archi.stack.okg import Okg, OkgError


def ingest_steps(instance: Instance, include: str | None = None, *, publish: bool = False,
                 inline: bool = False) -> list[list[str]]:
    flag = ["--inline"] if inline else []
    ingest = ["ingest", "--deployment", instance.name]
    if include:
        ingest += ["--include", include]
    steps = [ingest + flag]
    if publish:
        steps.append(["run", "--once", "--deployment", instance.name, "--apply", *flag])
    return steps


def cmd_ingest(args: argparse.Namespace, runner=None) -> int:
    from archi.cli.init import has_automations

    try:
        instance = Instance.load(args.instance)
        secrets = read_env(instance.env_path)
    except InstanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    okg = Okg(instance, secrets) if runner is None else Okg(instance, secrets, runner=runner)
    steps = ingest_steps(instance, args.include, publish=getattr(args, "publish", False),
                         inline=has_automations(instance.dir))
    for step in steps:
        try:
            result = okg.run(*step, ok_codes=(0, 8) if step[0] == "run" else (0,))
        except OkgError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return exc.result.returncode if exc.result is not None and exc.result.returncode else 1
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser(
        "ingest",
        help="run okg ingest for the instance, with the credentials in its .env",
    )
    p.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    p.add_argument("--include", default=None,
                   help="comma-separated source ids to ingest (default: every source)")
    p.add_argument("--publish", action="store_true",
                   help="then publish a new generation (okg run --once --apply)")
    p.set_defaults(func=cmd_ingest)
