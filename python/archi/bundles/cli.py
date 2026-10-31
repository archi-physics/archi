"""The `archi` command: pack and bundle tooling.

    archi bundles list [--json]         every pack and bundle installed packages register
    archi bundles show NAME             one pack or bundle in full
    archi bundles validate [PATH ...]   check pack.yaml / bundle.yaml
    archi render --bundle B [--with P] [--without P] [--instance DIR]
                                        compose packs into an okg profile + archi.lock
    archi add PACK [--answer ID=VALUE]  add a pack to an installed instance
    archi diff [--json]                 drift from archi.lock
    archi init NAME [--bundle B]        a published instance (archi.cli.init)
    archi up / archi down               the instance's compose project (archi.cli.up)
    archi doctor [--json]               okg lint + doctor + footgun checks (archi.cli.doctor)
    archi eval [--bundle B]             a bundle's smoke questions over the instance MCP (archi.cli.evaluate)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from archi.bundles import discovery
from archi.bundles.manifest import BUNDLE_FILE, PACK_FILE, Problem, validate_bundle, validate_pack


def _manifest_dir(path: Path) -> Path:
    return path.parent if path.name in (PACK_FILE, BUNDLE_FILE) else path


def _catalog(args: argparse.Namespace) -> discovery.Catalog:
    return discovery.discover(args.pack_root, args.bundle_root)


def cmd_validate(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    problems: list[Problem] = []
    checked = 0
    if not args.paths:
        problems = list(catalog.problems)
        checked = len(catalog.packs) + len(catalog.bundles) + len(
            {p.path.parent for p in catalog.problems}
        )
    for raw in args.paths:
        directory = _manifest_dir(Path(raw))
        checked += 1
        if (directory / PACK_FILE).is_file():
            problems.extend(validate_pack(directory, packs=catalog.packs))
        elif (directory / BUNDLE_FILE).is_file():
            problems.extend(validate_bundle(directory, catalog.packs, catalog.bundles))
        else:
            problems.append(Problem(directory, f"holds neither {PACK_FILE} nor {BUNDLE_FILE}"))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"ok: {checked} manifest(s) valid")
    return 0


def _listing(catalog: discovery.Catalog) -> dict:
    return {
        "bundles": [
            {
                "name": b.name,
                "version": b.version,
                "description": b.manifest["description"],
                "extends": list(b.extends),
                "default": list(b.default_packs),
                "optional": list(b.optional_packs),
                "companions": {k: list(v) for k, v in b.companions.items()},
                "okg": b.okg,
                "provider": catalog.providers[f"bundle:{name}"],
                "dir": str(b.dir),
            }
            for name, b in sorted(catalog.bundles.items())
        ],
        "packs": [
            {
                "name": p.name,
                "kind": p.kind,
                "version": p.version,
                "description": p.manifest["description"],
                "needs_credentials": p.needs_credentials,
                "credentials": list(p.credentials),
                "provider": catalog.providers[f"pack:{name}"],
                "dir": str(p.dir),
            }
            for name, p in sorted(catalog.packs.items())
        ],
        "legacy_profiles": [
            {"name": name, "dir": str(path)} for name, path in sorted(catalog.legacy_profiles.items())
        ],
        "problems": [str(p) for p in catalog.problems],
    }


def cmd_list(args: argparse.Namespace) -> int:
    listing = _listing(_catalog(args))
    if args.json:
        print(json.dumps(listing, indent=2))
        return 1 if listing["problems"] else 0
    print("bundles:")
    for b in listing["bundles"]:
        print(f"  {b['name']} {b['version']}  default: {', '.join(b['default'])}"
              f"  optional: {', '.join(b['optional']) or '-'}"
              + "".join(f"  with {k}: {', '.join(v)}" for k, v in b["companions"].items())
              + f"  [{b['provider']}]")
    print("packs:")
    for p in listing["packs"]:
        creds = "needs credentials" if p["needs_credentials"] else "no credentials needed"
        print(f"  {p['name']} {p['version']} ({p['kind']}, {creds})  [{p['provider']}]")
    if listing["legacy_profiles"]:
        print("legacy okg profiles (not composable; `archi render --bundle <name>` copies one):")
        for lp in listing["legacy_profiles"]:
            print(f"  {lp['name']}  {lp['dir']}")
    for problem in listing["problems"]:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if listing["problems"] else 0


def cmd_show(args: argparse.Namespace) -> int:
    listing = _listing(_catalog(args))
    matches = [
        {"type": kind[:-1], **item}
        for kind in ("bundles", "packs")
        for item in listing[kind]
        if item["name"] == args.name
    ]
    if not matches:
        print(f"no pack or bundle named {args.name!r}", file=sys.stderr)
        return 1
    print(json.dumps(matches if len(matches) > 1 else matches[0], indent=2))
    return 0


def _packs_arg(values: list[str]) -> list[str]:
    """--with a,b --with c -> [a, b, c]"""
    return [name for value in values for name in value.split(",") if name]


def cmd_render(args: argparse.Namespace) -> int:
    from archi.bundles.resolve import CompositionError

    catalog = _catalog(args)
    for problem in catalog.problems:
        print(f"warning: {problem}", file=sys.stderr)
    legacy = [b for b in args.bundle if b not in catalog.bundles and b in catalog.legacy_profiles]
    if legacy:
        from archi.bundles.render import render_legacy

        if len(args.bundle) > 1 or args.with_packs or args.without_packs or args.name:
            print(f"error: {legacy[0]!r} is a legacy okg profile; it cannot be combined "
                  "with other bundles, --with, --without or --name", file=sys.stderr)
            return 1
        out = render_legacy(catalog.legacy_profiles[legacy[0]], args.instance)
        print(f"copied legacy profile {legacy[0]!r} to {out}")
        print(f"run okg from {args.instance}:")
        print(f"  okg install --profile {legacy[0]} --deployment-name <slug> --postgres-dsn ...")
        return 0
    from archi.bundles.lock import render_and_lock

    try:
        resolution, rendered, _ = render_and_lock(
            catalog, args.instance, args.bundle,
            _packs_arg(args.with_packs), _packs_arg(args.without_packs),
            name=args.name, deployment=args.deployment.resolve() if args.deployment else None,
        )
    except CompositionError as exc:
        print(f"error ({exc.code}): {exc}", file=sys.stderr)
        return 1
    for note in resolution.notes:
        print(f"note: {note}", file=sys.stderr)
    print(f"rendered profile {rendered.name!r} with packs {', '.join(resolution.pack_names)}")
    print(f"  {rendered.profile_dir}")
    print(f"  {Path(args.instance) / 'archi.lock'}")
    print(f"run okg from {args.instance} so it finds profiles/{rendered.name}/:")
    print(f"  okg install --profile {rendered.name} --deployment-name <slug> --postgres-dsn ...")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    from archi.bundles.lock import LockError, add
    from archi.bundles.resolve import CompositionError

    answers = {}
    for item in args.answer:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"error: --answer {item!r} is not ID=VALUE", file=sys.stderr)
            return 1
        answers[key.replace("-", "_")] = value
    try:
        result = add(
            args.instance, _catalog(args), args.pack, answers,
            deployment_dir=args.deployment, run_okg=not args.no_okg,
        )
    except (LockError, CompositionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"added pack {result.pack!r}: sources {', '.join(result.added_sources) or 'none'}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from archi.bundles.lock import LockError, diff

    try:
        drift = diff(args.instance, _catalog(args), args.deployment)
    except LockError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(drift.as_dict(), indent=2))
    else:
        for area, items in drift.as_dict().items():
            for item in items:
                print(f"{area}: {item}")
        if drift.clean:
            print("no drift from archi.lock")
    return 0 if drift.clean else 1


def build_parser(load_ops: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="archi", description="Archi pack and bundle tooling.")
    sub = parser.add_subparsers(dest="command", required=True)

    def roots(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pack-root", action="append", type=Path, default=[],
            help="extra directory of <name>/pack.yaml entries (repeatable)",
        )
        p.add_argument(
            "--bundle-root", action="append", type=Path, default=[],
            help="extra directory of <name>/bundle.yaml entries (repeatable)",
        )

    bundles = sub.add_parser("bundles", help="inspect and check packs and bundles")
    bsub = bundles.add_subparsers(dest="bundles_command", required=True)
    listing = bsub.add_parser("list", help="every pack and bundle installed packages register")
    listing.add_argument("--json", action="store_true", help="machine-readable output")
    roots(listing)
    listing.set_defaults(func=cmd_list)
    show = bsub.add_parser("show", help="one pack or bundle in full (JSON)")
    show.add_argument("name")
    roots(show)
    show.set_defaults(func=cmd_show)

    validate = bsub.add_parser(
        "validate",
        help="check manifests against the published schemas and the composition rules",
    )
    validate.add_argument(
        "paths", nargs="*",
        help="pack or bundle directories (or their manifest files); default: every one found",
    )
    roots(validate)
    validate.set_defaults(func=cmd_validate)

    render = sub.add_parser("render", help="compose bundles and packs into one okg profile")
    render.add_argument("--bundle", action="append", required=True,
                        help="bundle to install (repeatable; chat scalars come from the last)")
    render.add_argument("--with", dest="with_packs", action="append", default=[],
                        help="add packs (comma-separated or repeated)")
    render.add_argument("--without", dest="without_packs", action="append", default=[],
                        help="remove packs (comma-separated or repeated)")
    render.add_argument("--instance", type=Path, default=Path("."),
                        help="instance directory; the profile goes to <instance>/profiles/<name>/")
    render.add_argument("--name", help="profile name (default: the bundle names joined by -)")
    render.add_argument("--deployment", type=Path,
                        help="installed okg deployment directory to record in archi.lock")
    roots(render)
    render.set_defaults(func=cmd_render)

    add = sub.add_parser("add", help="add a pack to an installed instance and publish again")
    add.add_argument("pack")
    add.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    add.add_argument("--deployment", type=Path,
                     help="okg deployment directory (default: the one archi.lock records)")
    add.add_argument("--answer", action="append", default=[], metavar="ID=VALUE",
                     help="answer one of the new pack's questions, by rendered id (repeatable)")
    add.add_argument("--no-okg", action="store_true",
                     help="edit and commit the deployment, but run no okg command")
    roots(add)
    add.set_defaults(func=cmd_add)

    drift = sub.add_parser("diff", help="report drift from archi.lock")
    drift.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    drift.add_argument("--deployment", type=Path,
                       help="okg deployment directory (default: the one archi.lock records)")
    drift.add_argument("--json", action="store_true")
    roots(drift)
    drift.set_defaults(func=cmd_diff)

    from archi.cli import register

    register(sub, roots, load_ops)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    from archi.cli import finalize

    # Progress lines interleave with okg and compose output in order.
    sys.stdout.reconfigure(line_buffering=True)
    # Only the ops command needs archi.cli.ops (and what it pulls in); every other command skips it.
    wanted = sys.argv[1:] if argv is None else argv
    args = build_parser(load_ops="ops" in wanted).parse_args(argv)
    finalize(args)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
