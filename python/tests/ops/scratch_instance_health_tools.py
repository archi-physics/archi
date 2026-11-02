"""Helpers for scratch_instance_health_run.sh (PACT r3-ops-self-monitoring, acc.issue). Scratch use only.

    python scratch_instance_health_tools.py <command> ...

Environment: ARCHI_INSTANCE, OKG_DSN, ARCHI_OPS_DEPLOYMENT. Most commands are those of
scratch_ops_runtime_tools.py with the automation set to instance-health.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scratch_ops_runtime_tools as base  # noqa: E402

AUTOMATION = "instance-health"
base.AUTOMATION = AUTOMATION


def cmd_init_instance(a: argparse.Namespace) -> None:
    """`archi.cli.init.init` itself, from an empty directory, for cern-team + ops-record + jira + instance-health.
    Only the container start is left out: the lab database already runs, and its port and password replace the
    generated ones (as scratch_ops_runtime_tools does)."""
    from archi.bundles import discovery
    from archi.cli import init as init_mod

    inst_dir, dsn, deployment = base.env()
    run_dir = inst_dir.parent
    parsed = urlparse(dsn)
    real_secrets = init_mod.generate_secrets
    init_mod.free_ports = lambda n: [parsed.port, 18765, 18080][:n]
    init_mod.generate_secrets = lambda: {**real_secrets(), "ARCHI_GRAPH_DB_PASSWORD": parsed.password}
    catalog = discovery.from_roots([run_dir / "catalog" / "packs"], [run_dir / "catalog"])
    init_mod.init(
        deployment, bundles=["cern-team"], catalog=catalog, directory=inst_dir,
        with_packs=["ops-record", "jira", AUTOMATION], without_packs=["docsite", "indico"],
        data_root=run_dir / "data", runtime="docker",
        compose_runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
        wait_db=lambda _dsn: None, echo=print,
    )


def cmd_configure(a: argparse.Namespace) -> None:
    """Level in archi.yaml; mode and proposer in ops.yaml; then commit, as an operator would."""
    from archi.cli.init import commit_all

    inst, _, deployment = base.env()
    archi = base.load(inst / "archi.yaml")
    archi.setdefault("name", deployment)
    if a.level:
        archi.setdefault("automations", {}).setdefault(AUTOMATION, {})["level"] = a.level
    (inst / "archi.yaml").write_text(yaml.safe_dump(archi, sort_keys=False), encoding="utf-8")
    ops = base.load(inst / "ops.yaml")
    entry = ops.setdefault("automations", {}).setdefault(AUTOMATION, {})
    if a.mode:
        entry["mode"] = a.mode
    proposer = entry.setdefault("proposer", {"kind": "instance-health"})
    if a.ignore:
        proposer["ignore_checks"] = a.ignore.split(",")
    (inst / "ops.yaml").write_text(yaml.safe_dump(ops, sort_keys=False), encoding="utf-8")
    commit_all(inst, f"configure: {' '.join(sys.argv[2:])}")


def cmd_signal(a: argparse.Namespace) -> None:
    """Add a schedule signal (a stand-in for the next cron time) and print its run id."""
    base.io().add_signal(AUTOMATION, a.id, os.environ["ARCHI_OPS_DEPLOYMENT"], f"scheduled run ({a.id})", source="schedule")
    print(f"{AUTOMATION}~{a.id}~1")


def cmd_tick(a: argparse.Namespace) -> None:
    """The run id the scanner makes for the current cron time (archi.ops.schedule)."""
    from archi.ops.loop import make_run_id
    from archi.ops.schedule import latest_fire, tick_id

    block = base.load(Path(os.environ["ARCHI_INSTANCE"]) / "automations" / f"{AUTOMATION}.yaml")
    print(make_run_id(AUTOMATION, tick_id(latest_fire(block["trigger"]["cron"], datetime.now(timezone.utc))), 1))


def cmd_signals(a: argparse.Namespace) -> None:
    import psycopg

    with psycopg.connect(os.environ["OKG_DSN"], autocommit=True) as conn:
        rows = conn.execute("SELECT signal_id, source, status FROM archi_ops.signals WHERE automation = %s ORDER BY created_at",
                            (AUTOMATION,)).fetchall()
    print(json.dumps([{"signal_id": r[0], "source": r[1], "status": r[2]} for r in rows]))


def cmd_latest(a: argparse.Namespace) -> None:
    """Wait for the newest signal not among the ids already seen (`--seen a,b`) and print its run id."""
    import time

    import psycopg

    seen = set(filter(None, a.seen.split(",")))
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        with psycopg.connect(os.environ["OKG_DSN"], autocommit=True) as conn:
            rows = conn.execute("SELECT signal_id FROM archi_ops.signals WHERE automation = %s ORDER BY created_at DESC",
                                (AUTOMATION,)).fetchall()
        fresh = [r[0] for r in rows if r[0] not in seen]
        if fresh:
            print(f"{AUTOMATION}~{fresh[0]}~1")
            return
        time.sleep(1)
    sys.exit(3)


def cmd_runs(a: argparse.Namespace) -> None:
    """Every instance-health run, oldest first: id, DBOS state, and the loop's status, mode and mediated flag."""
    import psycopg

    with psycopg.connect(os.environ["OKG_DSN"], autocommit=True) as conn:
        rows = conn.execute("SELECT workflow_uuid, status, output FROM dbos.workflow_status WHERE workflow_uuid LIKE %s "
                            "ORDER BY created_at", (f"{AUTOMATION}~%",)).fetchall()
    out = []
    for wid, state, output in rows:
        body = base.decode(output) if output else {}
        body = body if isinstance(body, dict) else {}
        out.append({"id": wid, "state": state, "status": body.get("status"), "mode": body.get("mode"),
                    "mediated": bool(body.get("mediated"))})
    print(json.dumps(out))


def cmd_issues(a: argparse.Namespace) -> None:
    """The issue files in the instance repository: key, state, events, content hash; and the commits that touched them."""
    inst = Path(os.environ["ARCHI_INSTANCE"])
    out = {"files": [], "commits": [], "dirty": False}
    for path in sorted((inst / ".archi" / "issues").glob("*.md")) if (inst / ".archi" / "issues").is_dir() else []:
        text = path.read_text(encoding="utf-8")
        meta = yaml.safe_load(text.split("---\n", 2)[1])
        out["files"].append({"file": path.name, "key": meta["key"], "state": meta["state"],
                             "events": [e["event"] for e in meta["events"]], "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]})
    log = subprocess.run(["git", "-C", str(inst), "log", "--format=%h %s", "--", ".archi/issues"], capture_output=True, text=True)
    out["commits"] = log.stdout.splitlines()
    status = subprocess.run(["git", "-C", str(inst), "status", "--porcelain"], capture_output=True, text=True)
    out["dirty"] = bool(status.stdout.strip())
    out["status"] = status.stdout.strip().splitlines()
    print(json.dumps(out))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-instance").set_defaults(f=cmd_init_instance)
    c = sub.add_parser("configure")
    c.add_argument("--level")
    c.add_argument("--mode", choices=["shadow", "live"])
    c.add_argument("--ignore", help="comma-separated doctor check ids")
    c.set_defaults(f=cmd_configure)
    s = sub.add_parser("signal")
    s.add_argument("id")
    s.set_defaults(f=cmd_signal)
    sub.add_parser("tick").set_defaults(f=cmd_tick)
    sub.add_parser("runs").set_defaults(f=cmd_runs)
    sub.add_parser("signals").set_defaults(f=cmd_signals)
    lt = sub.add_parser("latest")
    lt.add_argument("--seen", default="")
    lt.add_argument("--timeout", type=int, default=90)
    lt.set_defaults(f=cmd_latest)
    sub.add_parser("issues").set_defaults(f=cmd_issues)
    w = sub.add_parser("wait")
    w.add_argument("wid")
    w.add_argument("--status")
    w.add_argument("--timeout", type=int, default=300)
    w.set_defaults(f=base.cmd_wait)
    h = sub.add_parser("human-action")
    h.add_argument("wid")
    h.set_defaults(f=base.cmd_human_action)
    ap = sub.add_parser("approve-promotion")
    ap.add_argument("change")
    ap.set_defaults(f=base.cmd_approve_promotion)
    st = sub.add_parser("source")
    st.add_argument("state", choices=["stale", "fresh"])
    st.set_defaults(f=base.cmd_stale)
    pa = sub.add_parser("pact")
    pa.add_argument("prefix")
    pa.set_defaults(f=base.cmd_pact)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
