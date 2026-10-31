"""`archi ops status | halt | resume | pending | approve | reject` (PACT r3-ops-runtime req.control).

    archi ops status [--json]      levels, schedules, pending escalations, budgets, shadow summary
    archi ops halt [--no-cancel]   stop every automation: pause the scanner, close open
                                   escalations, cancel runs that have not finished, record them.
                                   --no-cancel leaves runs in flight; their mediator step refuses
    archi ops resume               un-pause what halt paused; a cancelled run's signal runs again
                                   as a new attempt (the cancelled run itself is not resumed)
    archi ops pending              escalations waiting for a person
    archi ops approve ESC [--note] answer an escalation about an ops proposal of this instance
    archi ops reject  ESC [--note]
    archi ops promote AUTOMATION   ask for a shadow automation to go live: needs a passing shadow report;
                                   opens a PACT change in the instance repo holding the report and the
                                   thresholds, names it in ops.yaml and stops. An operator approves the
                                   change (`okg pact decide`, a step archi never takes); the runtime
                                   counts `mode: live` only while it is approved (archi.ops.promotion)

decided_by is the OS user. `--as X` is recorded as "<os user> for X": nothing authenticates the
caller until okg offers an authenticated proposal API (docs/ops-runtime-akmon-asks.md, O3), so a
claimed name must not read as a verified one.

The logic is in functions over an `OpsIO` (archi.ops.okg_io), so tests pass a
fake. The DSN comes from --dsn, else OKG_DSN, else the instance's own secrets.
Halt order matters (spike-results.md): cancel stops a run only at a step
boundary and does not wake a run waiting on an escalation, and leaves that
escalation open. So halt pauses the scanner first, then answers each open
escalation of a run it is about to cancel with `reject` (decided_by
"archi ops halt"), then cancels the run.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Optional

import yaml

from archi.cli import OPS_HELP
from archi.ops.control import BudgetUsage, build_status, halt_plan, resume_plan
from archi.ops.escalation import OPEN_STATES, SUBJECT_PREFIX
from archi.ops.loop import AutomationConfig, load_automation, parse_run_id
from archi.ops.record import OpsRecords, RecordError, submit
from archi.ops.runtime_config import OPS_FILE, SCANNER, raw_levels, read_settings

HALT_ACTOR = "archi ops halt"
#: halt must be quick: it waits this long for okg to confirm the halted records
SUBMIT_WAIT_SECONDS = 20.0


class OpsCliError(RuntimeError):
    """A message for the operator, printed as `error: ...`."""


# --- logic (tested with a fake OpsIO) -----------------------------------------------------


def automations(instance_dir: Path) -> list[AutomationConfig]:
    """Every rendered automation, configured as the runtime would configure it."""
    settings = read_settings(instance_dir)
    levels = raw_levels(instance_dir)
    out = []
    for path in sorted((Path(instance_dir) / "automations").glob("*.yaml")):
        name = path.stem
        s = settings.get(name)
        out.append(load_automation(
            instance_dir, name, instance_level=levels.get(name), mode=s.mode if s else "shadow",
            timeout_seconds=s.timeout_seconds if s else 4 * 3600, estimate_cost=s.estimate_cost if s else 0.0))
    return out


def status_report(instance_dir: Path, io, *, proposals: Optional[list] = None,
                  shadow: Optional[Callable[[str], Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Levels, schedules, pending escalations, budgets and the shadow summary, as data."""
    configs = automations(instance_dir)
    paused = {name for name, is_paused in io.schedules().items() if is_paused}
    budgets = {}
    for c in configs:
        spent, cap = io.day_spend(c.name)
        budgets[c.name] = BudgetUsage(c.cost_per_run, c.cost_per_day if cap is None else cap, spent)
    rows = build_status(configs, paused_schedules=paused, budgets=budgets,
                        proposals=[p.get("attrs", p) for p in (proposals or [])])
    out = []
    for row in rows:
        d = row.as_dict()
        if d.get("mode") == "live":  # ops.yaml asks for live; it counts only once the promotion is approved
            cfg = next((c for c in configs if c.name == row.name), None)
            d["promotion"] = io.promotion_status(instance_dir, getattr(cfg, "promotion_change", None), row.name)
        if shadow is not None:
            try:
                d["shadow_report"] = shadow(row.name)
            except Exception as exc:  # the report needs published records; say so, do not hide it
                d["shadow_report"] = {"error": f"{type(exc).__name__}: {exc}"}
        out.append(d)
    return {
        "scanner": {name: ("paused" if is_paused else "active") for name, is_paused in io.schedules().items()},
        "automations": out,
        "pending": io.escalations(open_only=True),
    }


def halted_records(run, signal: Mapping[str, Any], io, who: str) -> OpsRecords:
    """The records for a run halt cancelled: its incident (kept as published when it is),
    a placeholder proposal and a `halted` decision. A cancelled run writes none itself."""
    wid, generation = run.workflow_id, io.pin_generation()
    records = OpsRecords()
    incident_id = f"inc-{wid}"
    published = io.read_nodes("ops_incident", generation, {"record_id": incident_id})
    if published:
        records.carry(published[0])
    else:
        records.add("incident", record_id=incident_id, title=signal.get("title") or run.automation,
                    entity=signal.get("entity") or "unknown", status="open",
                    signal=f"{signal.get('source') or 'signal'}:{signal.get('id') or wid}",
                    summary="the run was cancelled by archi ops halt before it recorded anything",
                    automation=run.automation, actor_kind="automation", actor=run.automation)
    proposal_id = f"prop-{wid}-halt"
    records.add("proposal", evidence=[f"ops_incident:{incident_id}"], record_id=proposal_id,
                incident_id=incident_id, summary="no proposal: the run was halted",
                mode="placeholder", mode_reason="placeholder: archi ops halt cancelled the run",
                proposed_at=_now(), automation=run.automation, actor_kind="person", actor=who,
                generation_read=generation)
    records.add("decision", record_id=f"dec-{wid}-halt", proposal_id=proposal_id, verdict="halted",
                reason=f"cancelled by archi ops halt ({run.status} at the time)", decided_at=_now(),
                automation=run.automation, actor_kind="person", actor=who, generation_read=generation)
    return records


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def close_escalations(io, workflow_id: str, log: Callable[[str], None]) -> tuple[list[str], list[dict[str, str]]]:
    """Answer each open escalation of a run with `reject`. Returns (closed, stayed_open).

    An escalation that cannot be answered because it was decided or timed out meanwhile is not open any
    more and counts as closed; one that is still open after a failed answer is reported with the error.
    """
    closed, stayed_open = [], []
    for esc in io.escalations(open_only=True, workflow_ids=[workflow_id]):
        esc_id = esc["escalation_id"]
        try:
            io.respond(esc_id, "reject", note="closed by archi ops halt", decided_by=HALT_ACTOR)
            closed.append(esc_id)
            log(f"closed escalation {esc_id} (reject)")
        except Exception as exc:
            found = io.escalation(esc_id)
            if found is not None and (found[1].get("status") not in OPEN_STATES):
                closed.append(esc_id)
                log(f"escalation {esc_id} was already {found[1].get('status')}")
            else:
                stayed_open.append({"escalation_id": esc_id, "error": f"{type(exc).__name__}: {exc}"})
                log(f"escalation {esc_id} STAYED OPEN: {type(exc).__name__}: {exc}")
    return closed, stayed_open


def do_halt(io, *, who: str = HALT_ACTOR, cancel: bool = True, submit_records: Callable[..., Any] = submit,
            log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> dict[str, Any]:
    """Stop every automation; return what was paused, closed, cancelled and recorded.

    With `cancel=False` (`--no-cancel`) runs in flight are left alone: they keep their escalations, and
    their mediator step refuses to act because the scanner schedule is paused. They finish and record
    `halted`. The default cancels them, which stops each at its next step boundary.
    """
    io.ensure_tables()
    plan = halt_plan(io.schedules(), io.runs())
    for name in plan.pause:
        io.pause(name)
        log(f"paused schedule {name}")
    io.record_halt(plan.pause, who)
    closed, stayed_open, cancelled, recorded, unrecorded = [], [], [], [], []
    for run in plan.cancel if cancel else ():
        run_closed, run_open = close_escalations(io, run.workflow_id, log)
        closed += run_closed
        stayed_open += run_open
        signal = io.run_input(run.workflow_id)
        io.cancel(run.workflow_id)
        cancelled.append(run.workflow_id)
        log(f"cancelled run {run.workflow_id} (was {run.status})")
        try:
            submit_records(halted_records(run, signal, io, who), deployment=io.deployment, dsn=io.dsn,
                           idempotency_key=f"{run.workflow_id}|halt", max_wait_seconds=SUBMIT_WAIT_SECONDS)
            recorded.append(run.workflow_id)
        except Exception as exc:
            # Most likely the worker is down. Its queued admission would keep okg from starting a worker
            # again (RuntimeCutoverBlocked), so cancel it: the halted record is not written.
            cancelled_admissions = io.cancel_unconfirmed_admission(f"{run.workflow_id}|halt")
            log(f"halted record for {run.workflow_id} not confirmed within {SUBMIT_WAIT_SECONDS:g}s; "
                f"cancelled {len(cancelled_admissions)} queued admission(s), so the record is not written "
                f"({type(exc).__name__}: {exc})")
            unrecorded.append(run.workflow_id)
    return {"paused": list(plan.pause), "already_paused": list(plan.already_paused), "closed": closed,
            "stayed_open": stayed_open, "cancelled": cancelled, "recorded": recorded, "unrecorded": unrecorded,
            "left_running": [] if cancel else [r.workflow_id for r in plan.cancel],
            "left_final": [r.workflow_id for r in plan.final]}


def do_resume(io, log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> dict[str, Any]:
    """Un-pause the schedules halt paused, and give each signal whose run halt cancelled a new attempt.

    A cancelled run itself is not resumed (DBOS does not rerun a cancelled workflow id); its signal gets
    attempt + 1, so the scanner starts a new run with a new id.
    """
    io.ensure_tables()
    names = resume_plan(io.schedules(), io.paused_by_halt())
    for name in names:
        io.resume(name)
        io.clear_halt(name)
        log(f"resumed schedule {name}")
    requeued = io.requeue_cancelled_signals()
    for automation, signal_id, attempt in requeued:
        log(f"signal {signal_id} of {automation} will run again as attempt {attempt}")
    return {"resumed": list(names), "requeued": [list(r) for r in requeued],
            "note": "cancelled runs are not resumed; their signals run again as a new attempt"}


def os_user() -> str:
    return getpass.getuser()


def decider_name(as_user: Optional[str]) -> str:
    """`decided_by` for an operator answer: always the OS user, and `--as X` is recorded as
    "<os user> for X". Nothing authenticates the caller yet (O3), so a claimed name must not
    read as a verified one."""
    user = os_user()
    return f"{user} for {as_user}" if as_user and as_user != user else user


def do_respond(io, escalation_id: str, verdict: str, note: Optional[str], as_user: Optional[str],
               automations: Optional[Collection[str]] = None) -> dict[str, Any]:
    """Approve or reject an escalation about an ops proposal of this deployment.

    Refuses (OpsCliError, exit 2) an escalation that does not exist, is not about an `ops_proposal:`, was
    not opened by one of this instance's automation runs, or is no longer open.
    """
    found = io.escalation(escalation_id)
    if found is None:
        raise OpsCliError(f"no escalation {escalation_id!r} in this deployment")
    subject, attrs = found
    if not subject.startswith(SUBJECT_PREFIX):
        raise OpsCliError(f"escalation {escalation_id} is about {subject!r}, not an ops proposal; "
                          "archi ops answers only ops proposals")
    try:
        automation = parse_run_id(attrs.get("requested_by_workflow") or "")[0]
    except ValueError:
        raise OpsCliError(f"escalation {escalation_id} was not opened by an archi ops run "
                          f"({attrs.get('requested_by_workflow')!r})") from None
    if automations is not None and automation not in automations:
        raise OpsCliError(f"escalation {escalation_id} belongs to automation {automation!r}, "
                          "which this instance does not have")
    status = attrs.get("status")
    if status not in OPEN_STATES:
        by = f", by {attrs['decided_by']}" if attrs.get("decided_by") else ""
        raise OpsCliError(f"escalation {escalation_id} is {status}{by}; it can no longer be answered")
    decided_by = decider_name(as_user)
    io.respond(escalation_id, verdict, note=note, decided_by=decided_by)
    return {"escalation_id": escalation_id, "verdict": verdict, "decided_by": decided_by}


# --- the command line ----------------------------------------------------------------------


def resolve_dsn(args: argparse.Namespace) -> str:
    if args.dsn:
        return args.dsn
    if os.environ.get("OKG_DSN"):
        return os.environ["OKG_DSN"]
    from archi.stack.instance import Instance, InstanceError, graph_dsn, read_env

    try:
        instance = Instance.load(args.instance)
        return graph_dsn(instance, read_env(instance.env_path))
    except (InstanceError, KeyError, OSError) as exc:
        raise OpsCliError(f"no database: pass --dsn or set OKG_DSN ({exc})") from exc


def deployment_name(args: argparse.Namespace) -> str:
    if args.deployment:
        return args.deployment
    if os.environ.get("ARCHI_OPS_DEPLOYMENT"):
        return os.environ["ARCHI_OPS_DEPLOYMENT"]
    data = yaml.safe_load((Path(args.instance) / "archi.yaml").read_text(encoding="utf-8")) or {}
    if not data.get("name"):
        raise OpsCliError("no deployment: pass --deployment or set ARCHI_OPS_DEPLOYMENT")
    return data["name"]


def make_io(args: argparse.Namespace):
    from archi.ops.okg_io import OpsIO

    return OpsIO(resolve_dsn(args), deployment_name(args))


def _line(row: Mapping[str, Any]) -> str:
    b, s = row["budget"], row["shadow"]
    level = row["level"] or f"REFUSED ({row['level_problem']})"
    cap = "-" if b["per_day_cap"] is None else f"{b['per_day_cap']:g}"
    spent = "-" if b["day_spent"] is None else f"{b['day_spent']:g}"
    mode = row["mode"]
    p = row.get("promotion")
    if mode == "live" and p is not None:
        mode = ("live (promotion approved" + (f"; no shadow evidence: {p['no_trial']}" if p.get("no_trial") else "") + ")"
                if p.get("approved") else f"live requested, runs in shadow until approved ({p.get('reason')})")
    return (f"{row['name']}: level {level}, mode {mode}, budget today {spent} of {cap}"
            f"{' (exhausted)' if b['exhausted'] else ''}, proposals shadow {s['shadow_proposals']} "
            f"live {s['live_proposals']}, demotions {s['demotions']}")


def cmd_status(args: argparse.Namespace) -> int:
    from archi.ops import shadow as shadow_mod
    from archi.ops.record import read_nodes

    io = make_io(args)
    try:
        proposals = read_nodes(io.deployment, "ops_proposal", dsn=io.dsn)
    except RecordError:
        proposals = []
    source = shadow_mod.okg_source(io.deployment, dsn=io.dsn)
    report = status_report(Path(args.instance), io, proposals=proposals,
                           shadow=lambda name: shadow_mod.score_automation(source, name))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print("scanner: " + (", ".join(f"{n} {s}" for n, s in report["scanner"].items()) or "no schedule found"))
    for row in report["automations"]:
        print(_line(row))
    print(f"pending escalations: {len(report['pending'])}")
    for esc in report["pending"]:
        print(f"  {esc['escalation_id']}  {esc['proposal_id']}  until {esc['timeout_at']}")
    return 0


def cmd_halt(args: argparse.Namespace) -> int:
    result = do_halt(make_io(args), who=args.who or getpass.getuser(), cancel=not args.no_cancel)
    print(json.dumps(result, indent=2) if args.json else
          f"halted: paused {len(result['paused'])}, closed {len(result['closed'])} escalation(s), "
          f"cancelled {len(result['cancelled'])} run(s), left running {len(result['left_running'])}")
    if result["stayed_open"]:
        ids = ", ".join(e["escalation_id"] for e in result["stayed_open"])
        print(f"error: escalation(s) stayed open: {ids}", file=sys.stderr)
        return 1
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    result = do_resume(make_io(args))
    print(json.dumps(result, indent=2) if args.json else f"resumed {len(result['resumed'])} schedule(s)")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    rows = make_io(args).escalations(open_only=True)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    if not rows:
        print("no escalations are waiting")
    for esc in rows:
        first = (esc.get("reason") or "").splitlines()[0] if esc.get("reason") else ""
        print(f"{esc['escalation_id']}  proposal {esc['proposal_id']}  until {esc['timeout_at']}\n    {first}")
    return 0


def cmd_respond(args: argparse.Namespace) -> int:
    verdict = "approve" if args.command_name == "approve" else "reject"
    io = make_io(args)
    result = do_respond(io, args.escalation_id, verdict, args.note, args.as_user,
                        automations=[c.name for c in automations(Path(args.instance))])
    past = {"approve": "approved", "reject": "rejected"}[verdict]
    print(json.dumps(result) if args.json else f"{past} {result['escalation_id']} as {result['decided_by']}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from archi.ops import shadow as shadow_mod
    from archi.ops.promotion import PromotionRefused, promote

    if args.no_trial and not (args.note or "").strip():
        raise OpsCliError("--no-trial needs --note naming the decision that waived the shadow trial "
                          "(for example a board item); nothing was changed")
    io = make_io(args)
    thresholds = {"min_proposals": args.min_proposals, "min_answered": args.min_answered,
                  "min_response_coverage": args.min_coverage, "min_agreement": args.min_agreement,
                  "max_false_rate": args.max_false_rate}
    source = shadow_mod.okg_source(io.deployment, dsn=io.dsn)
    report = shadow_mod.score_automation(source, args.automation)
    readiness = shadow_mod.promotion_readiness(report, **thresholds)
    try:
        result = promote(Path(args.instance), args.automation, report=report, readiness=readiness,
                         thresholds=thresholds, principal=decider_name(args.as_user), note=args.note or "",
                         dsn=io.dsn, deployment=io.deployment,
                         no_trial=(args.note or "").strip() if args.no_trial else "")
    except PromotionRefused as exc:
        raise OpsCliError(str(exc)) from exc
    print(json.dumps(result, indent=2) if args.json else
          f"opened PACT change {result['change']} (awaiting review) and committed it; {result['next']}")
    return 0


def _guarded(func: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        try:
            return func(args)
        except OpsCliError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # okg's own errors (a closed escalation, a refused payload) read plainly
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    return run


def add_parser(sub, roots=None) -> None:
    ops = sub.add_parser("ops", help=OPS_HELP)
    ops.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    ops.add_argument("--deployment", help="okg deployment name (default: the instance's name)")
    ops.add_argument("--dsn", help="graph database (default: $OKG_DSN, else the instance's secrets)")
    osub = ops.add_subparsers(dest="ops_command", required=True)

    def command(name: str, help_: str, func, *, extra=None):
        p = osub.add_parser(name, help=help_)
        p.add_argument("--json", action="store_true", help="machine-readable output")
        if extra:
            extra(p)
        p.set_defaults(func=_guarded(func), command_name=name)
        return p

    command("status", "levels, schedules, pending escalations, budgets and shadow scores", cmd_status)
    def halt_args(p):
        p.add_argument("--as", dest="who", help="who is halting (default: the OS user)")
        p.add_argument("--no-cancel", action="store_true",
                       help="pause the scanner but leave runs in flight; their mediator step refuses to act")

    command("halt", "stop every automation on this instance", cmd_halt, extra=halt_args)

    def promote_args(p):
        p.add_argument("automation")
        p.add_argument("--note", help="a note for the PACT decision")
        p.add_argument("--as", dest="as_user", help="who is deciding (recorded as \"<os user> for X\")")
        p.add_argument("--min-proposals", type=int, default=30)
        p.add_argument("--min-answered", type=int, default=20)
        p.add_argument("--min-coverage", type=float, default=0.5)
        p.add_argument("--min-agreement", type=float, default=0.8)
        p.add_argument("--max-false-rate", type=float, default=0.1)
        p.add_argument("--no-trial", action="store_true",
                       help="go live with no shadow results, on the decision --note names (required); the change "
                            "records that there is no shadow evidence, and an operator still approves it")

    command("promote", "ask for an automation to go live: opens a PACT change for an operator to approve", cmd_promote,
            extra=promote_args)
    command("resume", "resume the schedules halt paused", cmd_resume)
    command("pending", "escalations waiting for a person", cmd_pending)
    for verb in ("approve", "reject"):
        def extra(p, verb=verb):
            p.add_argument("escalation_id")
            p.add_argument("--note", help="a note recorded with the decision")
            p.add_argument("--as", dest="as_user", help="who is deciding (default: the OS user)")
        command(verb, f"{verb} an escalation", cmd_respond, extra=extra)
