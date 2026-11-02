"""Command line: ``python -m archi.snapshot {build,verify}``."""
from __future__ import annotations

import argparse
import signal
import sys
import threading

from archi.snapshot.builder import (
    BuildRefused,
    SnapshotError,
    build,
    load_config,
    verify,
)


def _terminate(signum: int, frame: object) -> None:
    """Turn SIGTERM into an exception, so every ``with`` block and the
    build's own cleanup run: the reader-check copies in --tmp-dir and the
    output staging directory are removed before the process exits. A second
    SIGTERM during that cleanup is ignored, so it cannot cut it short."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    if threading.current_thread() is not threading.main_thread():
        return _main(argv)
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        return _main(argv)
    finally:
        # getsignal/signal return None for a handler not set from Python;
        # the default action is the honest thing to put back then.
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m archi.snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser(
        "build",
        help="validate, redact and pack each configured group, plus a lock file",
    )
    b.add_argument("--config", required=True, help="YAML: snapshot name and groups")
    b.add_argument("--out", required=True, help="new or empty output directory")
    b.add_argument("--built-by", default=None, help="recorded in the lock file")
    b.add_argument(
        "--tmp-dir",
        default=None,
        help=(
            "where the reader-check copies go (about twice the largest group); "
            "default: the system temp dir. Removed on exit, on refusal, on "
            "Ctrl-C and on SIGTERM; a SIGKILL leaves it behind."
        ),
    )
    v = sub.add_parser("verify", help="check archives against snapshot.lock.yaml")
    v.add_argument("--lock", required=True)
    v.add_argument("--archives", required=True, help="directory holding the archives")
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            lock = build(
                load_config(args.config),
                args.out,
                built_by=args.built_by,
                tmp_dir=args.tmp_dir,
            )
            for name, row in lock["groups"].items():
                print(
                    f"{name}: {row['record_count']} records, {row['file_count']} "
                    f"files, collected {row['collected']}, sha256 {row['sha256']}"
                )
            print(f"wrote {args.out}")
            return 0
        result = verify(args.lock, args.archives)
        print("\n".join(result.lines))
        return 0 if result.ok else 1
    except BuildRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
