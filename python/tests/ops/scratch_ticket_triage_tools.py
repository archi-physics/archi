"""Helpers for scratch_ticket_triage_run.sh (PACT r3-auto-ticket-triage, acc.fixture). Scratch use only.

    python scratch_ticket_triage_tools.py <command> ...

Environment: ARCHI_INSTANCE, OKG_DSN, ARCHI_OPS_DEPLOYMENT, and TT_STATE (the fake ticket system's state
file, served by `ticket_fakes.py serve`). The waits, escalations, promotions and records come from
scratch_ops_runtime_tools.py with the automation set to ticket-triage.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import psycopg
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scratch_ops_runtime_tools as base  # noqa: E402
from ticket_fakes import CACHE_DIRS, FakeTicketSystem, arrival, load_cache  # noqa: E402

AUTOMATION = "ticket-triage"
base.AUTOMATION = AUTOMATION
REPO = HERE.parents[2]
TOKEN_ENV = "TT_REDMINE_TOKEN"


def data_dir() -> Path:
    return Path(os.environ["ARCHI_INSTANCE"]).parent / "data"


def write_cache(system: str, records: list) -> None:
    out = data_dir() / CACHE_DIRS[system]
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(records, indent=1))
    (out / "meta.json").write_text(json.dumps({"record_count": len(records)}))


def read_cache(system: str) -> list:
    return json.loads((data_dir() / CACHE_DIRS[system] / "records.json").read_text())


def cmd_inputs(a: argparse.Namespace) -> None:
    """The shipped packs and the cluster-manager bundle (copied), the fixture caches, the fake ticket system."""
    run = Path(os.environ["ARCHI_INSTANCE"]).parent
    shutil.copytree(REPO / "bundles" / "packs", run / "catalog" / "packs")
    shutil.copytree(REPO / "bundles" / "cluster-manager", run / "catalog" / "cluster-manager")
    for system in CACHE_DIRS:
        write_cache(system, load_cache(system))
    # docsite is a cluster-manager default; its cache stands in for the public pages its prepare step downloads
    (data_dir() / "docsite").mkdir(parents=True, exist_ok=True)
    (data_dir() / "docsite" / "records.json").write_text(json.dumps([{
        "title": "Worker node scratch space", "url": "https://docs.example.org/scratch.html",
        "body": "Each worker node has a local /scratch filesystem. When it is full, jobs fail with ENOSPC; "
                "drain the node and remove stale job directories. " * 5,
        "site_name": "example-docs", "source_repo": "example/docs", "path": "scratch.md"}]))
    FakeTicketSystem(path=Path(os.environ["TT_STATE"])).save()
    print(json.dumps({s: len(load_cache(s)) for s in CACHE_DIRS}))


def cmd_init_instance(a: argparse.Namespace) -> None:
    """`archi.cli.init.init` itself for cluster-manager + servicenow + jira + ticket-triage. Skipped: the container
    start (the lab database runs) and docsite's download step (its cache is written by `inputs`)."""
    from archi.bundles import discovery
    from archi.cli import init as init_mod

    inst_dir, dsn, deployment = base.env()
    run_dir = inst_dir.parent
    parsed = urlparse(dsn)
    real_secrets = init_mod.generate_secrets
    init_mod.free_ports = lambda n: [parsed.port, 18766, 18081][:n]
    init_mod.generate_secrets = lambda: {**real_secrets(), "ARCHI_GRAPH_DB_PASSWORD": parsed.password}
    catalog = discovery.from_roots([run_dir / "catalog" / "packs"], [run_dir / "catalog"])
    init_mod.init(
        deployment, bundles=["cluster-manager"], catalog=catalog, directory=inst_dir,
        with_packs=["servicenow", "jira", AUTOMATION],
        answers={"tickets_base_url": "https://redmine.example.org",
                 "servicenow_instance_url": "https://example.service-now.com"},
        data_root=data_dir(), runtime="docker",
        compose_runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
        wait_db=lambda _dsn: None, echo=print,
        prepare_runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )


def cmd_configure(a: argparse.Namespace) -> None:
    """Level in archi.yaml; mode, timeout and the proposer (Redmine API: the fake ticket system) in ops.yaml."""
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
    if a.timeout is not None:
        entry["timeout_seconds"] = a.timeout
    entry["proposer"] = {"kind": AUTOMATION, "doc_subtypes": ["servicenow_kb_article", "documentation_page"],
                         "systems": {"redmine": {"url": f"http://127.0.0.1:{a.port}", "site": "https://redmine.example.org",
                                                 "token_env": TOKEN_ENV,
                                                 "triaged_status": "In Progress", "triage_from": ["New"]}}}
    (inst / "ops.yaml").write_text(yaml.safe_dump(ops, sort_keys=False), encoding="utf-8")
    commit_all(inst, f"configure: {' '.join(sys.argv[2:])}")


def cmd_arrive(a: argparse.Namespace) -> None:
    """A new ticket: into the source's cache (published by the next `okg run`) and into the fake ticket system.
    Prints the signal id the scanner will make for it."""
    from archi.triage.tickets import Ticket, signal_id
    from ticket_fakes import to_row

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ") if a.system != "jira" else now.strftime("%Y-%m-%dT%H:%M:%S.000+0000")
    item = arrival(a.system, a.key, a.subject, a.description, stamp, tracker=a.tracker)
    write_cache(a.system, read_cache(a.system) + [item])
    system = FakeTicketSystem(path=Path(os.environ["TT_STATE"]))
    if a.system == "servicenow":
        system.add("servicenow", a.key, table="incident", state="New", assignment_group="")
    else:
        system.add(a.tracker if a.system == "redmine" else a.system, a.key, status=item["status"], assignee="")
    print(signal_id(Ticket.from_row(to_row(a.system, item))))


def cmd_ticket(a: argparse.Namespace) -> None:
    """The fake ticket system's view of one ticket, and how many writes reached it."""
    system = FakeTicketSystem(path=Path(os.environ["TT_STATE"]))
    print(json.dumps(system.ticket(a.system, a.key)))


def cmd_signals(a: argparse.Namespace) -> None:
    with psycopg.connect(os.environ["OKG_DSN"], autocommit=True) as conn:
        rows = conn.execute("SELECT signal_id, entity, source, status FROM archi_ops.signals WHERE automation = %s "
                            "ORDER BY created_at", (AUTOMATION,)).fetchall()
    print(json.dumps([dict(zip(("signal_id", "entity", "source", "status"), r)) for r in rows]))


def cmd_cursors(a: argparse.Namespace) -> None:
    with psycopg.connect(os.environ["OKG_DSN"], autocommit=True) as conn:
        rows = conn.execute("SELECT source, updated, generation, seen FROM archi_ops.ticket_cursors WHERE automation = %s "
                            "ORDER BY source", (AUTOMATION,)).fetchall()
    print(json.dumps({r[0]: {"updated": r[1].isoformat() if r[1] else None, "generation": r[2], "seen": list(r[3] or ())}
                      for r in rows}))


def cmd_wait_signal(a: argparse.Namespace) -> None:
    """Wait until the scanner has added the signal."""
    import time

    deadline = time.time() + a.timeout
    while time.time() < deadline:
        with psycopg.connect(os.environ["OKG_DSN"], autocommit=True) as conn:
            if conn.execute("SELECT 1 FROM archi_ops.signals WHERE automation = %s AND signal_id = %s",
                            (AUTOMATION, a.id)).fetchone():
                print(f"{AUTOMATION}~{a.id}~1")
                return
        subprocess.run([str(Path(sys.executable).with_name("okg")), "runtime", "trigger-schedule", "--deployment",
                        os.environ["ARCHI_OPS_DEPLOYMENT"], "archi_ops_scan"], capture_output=True)
        time.sleep(5)
    sys.exit(3)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inputs").set_defaults(f=cmd_inputs)
    sub.add_parser("init-instance").set_defaults(f=cmd_init_instance)
    c = sub.add_parser("configure")
    c.add_argument("--level")
    c.add_argument("--mode", choices=["shadow", "live"])
    c.add_argument("--timeout", type=int)
    c.add_argument("--port", type=int, required=True)
    c.set_defaults(f=cmd_configure)
    ar = sub.add_parser("arrive")
    ar.add_argument("system", choices=sorted(CACHE_DIRS))
    ar.add_argument("key")
    ar.add_argument("subject")
    ar.add_argument("--description", default="Jobs fail with ENOSPC on /scratch.")
    ar.add_argument("--tracker", default="redmine", help="the tickets cache's own system: redmine, or jira (generic JIRA)")
    ar.set_defaults(f=cmd_arrive)
    t = sub.add_parser("ticket")
    t.add_argument("system")
    t.add_argument("key")
    t.set_defaults(f=cmd_ticket)
    sub.add_parser("signals").set_defaults(f=cmd_signals)
    sub.add_parser("cursors").set_defaults(f=cmd_cursors)
    ws = sub.add_parser("wait-signal")
    ws.add_argument("id")
    ws.add_argument("--timeout", type=int, default=240)
    ws.set_defaults(f=cmd_wait_signal)
    w = sub.add_parser("wait")
    w.add_argument("wid")
    w.add_argument("--status")
    w.add_argument("--timeout", type=int, default=240)
    w.set_defaults(f=base.cmd_wait)
    pe = sub.add_parser("pending")
    pe.add_argument("wid")
    pe.add_argument("--timeout", type=int, default=240)
    pe.set_defaults(f=base.cmd_pending)
    for name, func in (("escalation", base.cmd_escalation), ("human-action", base.cmd_human_action)):
        q = sub.add_parser(name)
        q.add_argument("wid")
        q.set_defaults(f=func)
    apr = sub.add_parser("approve-promotion")
    apr.add_argument("change")
    apr.set_defaults(f=base.cmd_approve_promotion)
    pa = sub.add_parser("pact")
    pa.add_argument("prefix")
    pa.set_defaults(f=base.cmd_pact)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
