"""`archi doctor [--json]`: okg deployment lint + okg doctor + archi's footgun checks.

Text output prints one line per check and, under each failure, one
remediation line. `--json` prints the same as one object for agents:

    {"ok": bool, "instance": "...", "checks": [{"id", "status", "message",
     "remediation", "footgun", "data"}, ...]}

Exit code 0 when no check failed (warnings and skips do not fail), 1
otherwise, 2 when the directory is not an instance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archi.stack import compose as stack_compose
from archi.stack.checks import FAIL, Context, Probes, run_checks
from archi.stack.instance import Instance, InstanceError, read_env
from archi.stack.okg import Okg, OkgError

MARK = {"pass": "ok  ", "fail": "FAIL", "warn": "warn", "skip": "skip"}


def default_probes(instance: Instance, secrets) -> Probes:
    okg = Okg(instance, secrets, echo=lambda _line: None)
    compose = stack_compose.Compose(instance)
    return Probes(services=compose.services, okg_json=okg.json)


def doctor(instance: Instance, probes: Probes | None = None) -> dict:
    try:
        secrets = read_env(instance.env_path)
    except InstanceError:
        secrets = {}
    ctx = Context(instance, secrets, probes or default_probes(instance, secrets))
    results = run_checks(ctx)
    return {
        "ok": not any(r.status == FAIL for r in results),
        "instance": str(instance.dir),
        "checks": [r.as_dict() for r in results],
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    try:
        instance = Instance.load(args.instance)
    except InstanceError as exc:
        if args.json:
            print(json.dumps({"ok": False, "instance": str(args.instance), "error": str(exc), "checks": []}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        report = doctor(instance)
    except OkgError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f"{MARK.get(check['status'], check['status'])}  {check['id']}: {check['message']}")
            if check["status"] in ("fail", "warn") and check["remediation"]:
                print(f"      fix: {check['remediation']}")
        failed = sum(c["status"] == "fail" for c in report["checks"])
        print("all checks passed" if report["ok"] else f"{failed} check(s) failed")
    return 0 if report["ok"] else 1


def add_parser(sub) -> None:
    p = sub.add_parser("doctor", help="okg lint + doctor + archi's footgun checks, with remediations")
    p.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    p.add_argument("--json", action="store_true", help="one JSON object, for agents")
    p.set_defaults(func=cmd_doctor)
