"""The database, okg and DBOS calls the ops runtime and `archi ops` make.

One class, `OpsIO`, so the worker and the CLI read the same things the same
way and tests can swap it for a fake. It imports okg's substrate (never
`okg.deployment`; that import lives only in `archi.ops.workflow`), `psycopg`
and `dbos`, each when first used.

Escalations are read from `okg.escalation_live` (current version:
`valid_to_gen IS NULL`). Operators answer through okg's own
`respond_to_escalation`; the `okg runtime escalation respond` command is
broken in okg 0.2.0 (ImportError in cli/runtime.py:1154).

Everything here that reaches into okg beyond its public CLI and `okg.deployment`,
with the upstream ask (docs/ops-runtime-akmon-asks.md) that removes it:

| What | Where | Ask |
|---|---|---|
| raw SQL on `okg.escalation_live` (list, read one, decision and status) | `escalation`, `escalations` | O3: an API to list, show and answer escalations |
| `okg.substrate.runtime.escalation_response.respond_to_escalation` | `respond` | O7: the CLI command that wraps it is broken; O3 |
| `okg.substrate.db.connection.connect` | `respond` | O3 |
| `okg.substrate.generation_resolver.resolve_okg_generation` | `pin_generation` | O3: a public "latest published generation" |
| raw SQL on `okg.deployment_cost_caps` and `okg.deployment_cost_ledger` | `day_spend` | O4: cost attribution per actor |
| `dbos.DBOSClient` (schedules, workflow lists, cancel) | `schedules`, `pause`, `resume`, `runs`, `run_input`, `cancel`, `cancel_unconfirmed_admission`, `requeue_cancelled_signals` | O5: a first-class kill switch |
| the `okg_admission_submit` workflow name and its input | `cancel_unconfirmed_admission` | O13: a queued admission blocks the next worker start |
| `okg pact view <change> --format status --json` (`derived_state`) | `promotion_status` | none: documented CLI; its JSON shape is not versioned |
| raw SQL on `okg.pact_decisions` (when a change was approved) | `promotion_status` | O15: say when and by whom a PACT change was approved |
| `okg doctor --check sources` JSON (`sources[].coverage.watermark`) | `doctor_sources` | none: documented CLI output, but its shape is not versioned |

The tables archi creates itself are in the `archi_ops` schema of the deployment
database: `signals`, `halts`, `mediations`, `demotions`, `ticket_cursors`.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from archi.ops.control import LIVE_RUN_STATES, RunInfo
from archi.ops.escalation import OPEN_STATES, SUBJECT_PREFIX, Decision, decision_from_row
from archi.ops.record import RecordError, _okg
from archi.ops.runtime_config import RUN_WORKFLOW, SCANNER

SIGNALS_DDL = """
CREATE SCHEMA IF NOT EXISTS archi_ops;
CREATE TABLE IF NOT EXISTS archi_ops.signals (
    automation text NOT NULL,
    signal_id  text NOT NULL,
    entity     text NOT NULL,
    title      text NOT NULL,
    source     text NOT NULL DEFAULT 'table',
    summary    text,
    status     text NOT NULL DEFAULT 'new',
    attempt    integer NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (automation, signal_id)
);
-- one row per mediator call, written before the call: a replay can tell "never started" from
-- "started, outcome unknown", so a mediator runs at most once per idempotency key
CREATE TABLE IF NOT EXISTS archi_ops.mediations (
    idempotency_key text PRIMARY KEY,
    proposal_id     text NOT NULL,
    status          text NOT NULL DEFAULT 'attempted',
    result          text,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz
);
-- a demotion to shadow, written when it happens (a run reads it at once; the published ops records
-- carry the same history later)
CREATE TABLE IF NOT EXISTS archi_ops.demotions (
    automation text NOT NULL,
    run_id     text NOT NULL,
    demoted_at timestamptz NOT NULL,
    reason     text NOT NULL,
    PRIMARY KEY (automation, run_id)
);
-- how far a ticket_queue automation has read each ticket source (a subtype at one site): the newest
-- `updated` time seen and the generation it was read from; `<system>/*` rows hold only the generation
-- (archi.triage.signals)
CREATE TABLE IF NOT EXISTS archi_ops.ticket_cursors (
    automation text NOT NULL,
    source     text NOT NULL,
    updated    timestamptz,
    generation text NOT NULL,
    read_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (automation, source)
);
-- schedules `archi ops halt` paused, so `resume` leaves alone any an operator paused earlier
CREATE TABLE IF NOT EXISTS archi_ops.halts (
    schedule  text PRIMARY KEY,
    paused_at timestamptz NOT NULL DEFAULT now(),
    paused_by text NOT NULL
);
"""


#: pg_advisory_xact_lock key that serialises the archi_ops DDL ("archiops" as a number).
DDL_LOCK_KEY = 0x6172636869_6f70 % (2 ** 63)


def _val(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


@dataclass
class OpsIO:
    dsn: str
    deployment: str
    okg: Optional[str] = None

    # --- connections ---------------------------------------------------------

    def connect(self):
        import psycopg

        return psycopg.connect(self.dsn, autocommit=True)

    def _client(self):
        """A DBOSClient on the deployment's database (as okg's admin code builds it)."""
        from dbos import DBOSClient

        return DBOSClient(database_url=self.dsn, system_database_url=self.dsn, application_database_url=self.dsn)

    def _okg(self, *args: str, stdin: Optional[str] = None) -> subprocess.CompletedProcess:
        cmd = [self.okg or _okg(), *args, "--dsn", self.dsn]
        return subprocess.run(cmd, capture_output=True, text=True, input=stdin)

    # --- the graph -----------------------------------------------------------

    def pin_generation(self) -> str:
        """The latest published generation (the checkpoint id okg's readers take)."""
        from okg.substrate.generation_resolver import resolve_okg_generation

        with self.connect() as conn:
            resolved = resolve_okg_generation(conn, deployment_name=self.deployment)
        if resolved is None:
            raise RecordError(f"deployment {self.deployment!r} has no published generation")
        return resolved.generation_id

    def read_nodes(self, subtype: str, generation: str, where: Optional[Mapping[str, str]] = None) -> list[dict]:
        """Every node of `subtype` at `generation` matching `where` ({attr: value})."""
        args = ["filter", "--deployment", self.deployment, "--type", subtype, "--generation-id", str(generation),
                "--all", "--jsonl"]
        for key, value in (where or {}).items():
            args += ["--where", f"attrs.{key}={value}"]
        result = self._okg(*args)
        if result.returncode != 0:
            raise RecordError(f"okg filter --type {subtype} failed: {(result.stderr or result.stdout)[-2000:]}")
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    def read_by_attr(self, subtype: str, generation: str, attr: str, value: str) -> list[dict]:
        """Nodes of `subtype` at `generation` whose attribute equals `value`, compared as text (a JSON
        condition, so a key such as `4812` is not read as a number)."""
        where = json.dumps({"type": subtype, f"attrs.{attr}": str(value)})
        result = self._okg("filter", "--deployment", self.deployment, "--type", subtype, "--generation-id",
                           str(generation), "--where", where, "--all", "--jsonl")
        if result.returncode != 0:
            raise RecordError(f"okg filter --type {subtype} {attr}={value!r} failed: "
                              f"{(result.stderr or result.stdout)[-2000:]}")
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    def read_ids(self, node_ids: Sequence[str], generation: str) -> list[dict]:
        """The nodes with these ids at `generation`, in no particular order."""
        if not node_ids:
            return []
        result = self._okg("filter", "--deployment", self.deployment, "--stdin", "--generation-id", str(generation),
                           "--all", "--jsonl", stdin="\n".join(node_ids) + "\n")
        if result.returncode != 0:
            raise RecordError(f"okg filter --stdin failed: {(result.stderr or result.stdout)[-2000:]}")
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    def search_ids(self, query: str, generation: str, subtype: str, limit: int) -> list[str]:
        """Node ids of `subtype` ranked by okg's lexical search for `query` at `generation`, best first."""
        result = self._okg("search", "--deployment", self.deployment, f"--query={query}", "--generation-id",
                           str(generation), "--subtype", subtype, "--limit", str(limit), "--json")
        try:
            reply = json.loads(result.stdout)
        except ValueError:
            reply = None
        if result.returncode != 0 or not isinstance(reply, Mapping):
            raise RecordError(f"okg search --subtype {subtype} failed: {(result.stderr or result.stdout)[-2000:]}")
        return [str(row["node_id"]) for row in reply.get("results") or () if isinstance(row, Mapping)
                and row.get("node_id")]

    def doctor_sources(self) -> list[dict]:
        """okg's per-source rows (`okg doctor --check sources`), with watermarks."""
        result = self._okg("doctor", "--check", "sources", "--deployment", self.deployment, "--json")
        try:
            data = json.loads(result.stdout)
        except ValueError:
            raise RecordError(f"okg doctor did not answer: {(result.stderr or result.stdout)[-1000:]}") from None
        rows = data.get("sources") if isinstance(data, Mapping) else None
        if rows is None:
            raise RecordError("okg doctor --check sources returned no `sources` list")
        return list(rows)

    # --- escalations ---------------------------------------------------------

    def escalation(self, escalation_id: str) -> Optional[tuple[str, dict]]:
        """(subject node id, attrs) of the current version of an escalation, or None."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT subject_node_id, attrs FROM okg.escalation_live "
                "WHERE escalation_id = %s AND valid_to_gen IS NULL", (escalation_id,)).fetchone()
        return (row[0], dict(row[1])) if row else None

    def decision_for(self, escalation_id: str) -> Decision:
        found = self.escalation(escalation_id)
        return decision_from_row(escalation_id, found[0] if found else None, found[1] if found else None)

    def escalations(self, *, open_only: bool = True, workflow_ids: Optional[Sequence[str]] = None) -> list[dict]:
        """Escalations about ops proposals, newest first, with the fields an operator needs."""
        sql = ("SELECT escalation_id, subject_node_id, attrs FROM okg.escalation_live "
               "WHERE valid_to_gen IS NULL AND subject_node_id LIKE %s")
        params: list[Any] = [SUBJECT_PREFIX + "%"]
        if open_only:
            sql += " AND attrs->>'status' = ANY(%s)"
            params.append(list(OPEN_STATES))
        if workflow_ids is not None:
            sql += " AND attrs->>'requested_by_workflow' = ANY(%s)"
            params.append(list(workflow_ids))
        sql += " ORDER BY attrs->>'timeout_at'"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{"escalation_id": r[0], "proposal_id": r[1][len(SUBJECT_PREFIX):], **{
            k: r[2].get(k) for k in ("status", "reason", "timeout_at", "decided_by", "decided_at",
                                     "requested_by_workflow")}} for r in rows]

    def respond(self, escalation_id: str, verdict: str, *, note: Optional[str], decided_by: str) -> Any:
        """Answer an escalation (approve | reject). Raises okg's EscalationStateError if it is closed."""
        from okg.substrate.db.connection import connect
        from okg.substrate.runtime.escalation_response import respond_to_escalation

        decision = {"verdict": verdict, **({"note": note} if note else {})}
        with connect(self.dsn) as conn:
            return respond_to_escalation(conn, deployment=self.deployment, escalation_id=escalation_id,
                                         decision=decision, decided_by=decided_by, send_wake=True)

    # --- schedules and runs --------------------------------------------------

    def schedules(self) -> dict[str, bool]:
        """The ops scanner schedule and whether it is paused."""
        out = {}
        for entry in self._client().list_schedules():
            name = _val(entry, "schedule_name")
            if name == SCANNER:
                out[name] = str(_val(entry, "status")).upper() == "PAUSED"
        return out

    def pause(self, name: str) -> None:
        client = self._client()
        client.pause_schedule(name)
        if str(_val(client.get_schedule(name), "status")).upper() != "PAUSED":
            raise RecordError(f"schedule {name!r} did not persist PAUSED")

    def resume(self, name: str) -> None:
        client = self._client()
        client.resume_schedule(name)
        if str(_val(client.get_schedule(name), "status")).upper() != "ACTIVE":
            raise RecordError(f"schedule {name!r} did not persist ACTIVE")

    def runs(self, *, live_only: bool = True) -> list[RunInfo]:
        statuses = sorted(LIVE_RUN_STATES) if live_only else None
        out = []
        for wf in self._client().list_workflows(name=RUN_WORKFLOW, status=statuses, load_input=True):
            wid, status = _val(wf, "workflow_id"), str(_val(wf, "status"))
            args = (_val(wf, "input") or {}).get("args") or []
            signal = args[1] if len(args) > 1 and isinstance(args[1], Mapping) else {}
            out.append(RunInfo(wid, args[0] if args else wid, status, str(signal.get("entity") or "")))
        return out

    def run_input(self, workflow_id: str) -> Mapping[str, Any]:
        """The `signal` argument a run was started with."""
        for wf in self._client().list_workflows(workflow_ids=[workflow_id], load_input=True):
            args = (_val(wf, "input") or {}).get("args") or []
            if len(args) > 1:
                return args[1]
        return {}

    def cancel(self, workflow_id: str) -> None:
        self._client().cancel_workflow(workflow_id)

    def cancel_unconfirmed_admission(self, idempotency_key: str) -> list[str]:
        """Cancel the queued okg admission workflow(s) submitted with `idempotency_key`.

        An `okg admission submit` made while no worker runs leaves an ENQUEUED
        `okg_admission_submit` with no application version, and okg 0.2.0 then refuses to
        start a worker (RuntimeCutoverBlocked) until it is cancelled. Only workflows whose
        input carries this exact key are touched. Returns the cancelled workflow ids.
        """
        client = self._client()
        cancelled = []
        for wf in client.list_workflows(name="okg_admission_submit", status=["ENQUEUED"], load_input=True):
            if idempotency_key in json.dumps(_val(wf, "input"), default=str):
                client.cancel_workflow(_val(wf, "workflow_id"))
                cancelled.append(_val(wf, "workflow_id"))
        return cancelled

    # --- budgets -------------------------------------------------------------

    def day_spend(self, automation: str, today: Optional[datetime] = None) -> tuple[Optional[float], Optional[float]]:
        """(spent, cap) of the automation's budget for the UTC day, None when never used."""
        day = (today or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
        scope = day_scope(automation, day)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT caps.cap_usd, COALESCE(sum(l.usd_consumed), 0) FROM okg.deployment_cost_caps caps "
                "LEFT JOIN okg.deployment_cost_ledger l ON l.scope = caps.scope "
                "WHERE caps.scope = %s GROUP BY caps.cap_usd", (scope,)).fetchone()
        return (float(row[1]), float(row[0])) if row else (None, None)

    # --- signals (the table trigger the scratch automations use) and halt state ---

    def ensure_tables(self) -> None:
        """Create the archi_ops tables. Idempotent, and safe to run from `archi up` and a starting worker at once:
        `CREATE ... IF NOT EXISTS` is not atomic in Postgres, so the DDL runs in one transaction behind a
        transaction-level advisory lock with a fixed key."""
        with self.connect() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (DDL_LOCK_KEY,))
            conn.execute(SIGNALS_DDL)

    def record_halt(self, schedules: Sequence[str], by: str) -> None:
        with self.connect() as conn:
            for name in schedules:
                conn.execute("INSERT INTO archi_ops.halts (schedule, paused_by) VALUES (%s, %s) "
                             "ON CONFLICT DO NOTHING", (name, by))

    def paused_by_halt(self) -> list[str]:
        with self.connect() as conn:
            return [r[0] for r in conn.execute("SELECT schedule FROM archi_ops.halts").fetchall()]

    def clear_halt(self, schedule: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM archi_ops.halts WHERE schedule = %s", (schedule,))

    def add_signal(self, automation: str, signal_id: str, entity: str, title: str, *, summary: Optional[str] = None,
                   source: str = "table") -> bool:
        """Add a signal; one that exists already (same automation and id) is left as it is. True when added."""
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO archi_ops.signals (automation, signal_id, entity, title, summary, source) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (automation, signal_id, entity, title, summary, source))
            return bool(cur.rowcount)

    def signal_ids_for(self, automation: str, entity: str) -> list[str]:
        """The ids of every signal an automation has for `entity`, whatever their status."""
        with self.connect() as conn:
            rows = conn.execute("SELECT signal_id FROM archi_ops.signals WHERE automation = %s AND entity = %s",
                                (automation, entity)).fetchall()
        return [r[0] for r in rows]

    def supersede_signals(self, automation: str, entity: str, keep: str) -> list[str]:
        """Mark `entity`'s signals still waiting to run (status `new`), other than `keep`, `superseded`."""
        with self.connect() as conn:
            rows = conn.execute(
                "UPDATE archi_ops.signals SET status = 'superseded' WHERE automation = %s AND entity = %s "
                "AND status = 'new' AND signal_id <> %s RETURNING signal_id", (automation, entity, keep)).fetchall()
        return sorted(r[0] for r in rows)

    def new_signals(self, automation: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT signal_id, entity, title, source, summary, attempt FROM archi_ops.signals "
                "WHERE automation = %s AND status = 'new' ORDER BY created_at, signal_id", (automation,)).fetchall()
        return [dict(zip(("id", "entity", "title", "source", "summary", "attempt"), r)) for r in rows]

    def requeue_cancelled_signals(self) -> list[tuple[str, str, int]]:
        """Give each signal whose run was cancelled a new attempt, so it can run again.

        Returns (automation, signal id, new attempt). The attempt is part of the run id, and DBOS
        does not rerun a workflow id that is already CANCELLED.
        """
        from archi.ops.loop import make_run_id

        with self.connect() as conn:
            rows = conn.execute("SELECT automation, signal_id, attempt FROM archi_ops.signals "
                                "WHERE status = 'started'").fetchall()
        if not rows:
            return []
        ids = {make_run_id(a, sid, att): (a, sid, att) for a, sid, att in rows}
        requeued = []
        for wf in self._client().list_workflows(workflow_ids=list(ids), load_input=False):
            if str(_val(wf, "status")).upper() == "CANCELLED":
                automation, signal_id, attempt = ids[_val(wf, "workflow_id")]
                with self.connect() as conn:
                    conn.execute("UPDATE archi_ops.signals SET status = 'new', attempt = %s "
                                 "WHERE automation = %s AND signal_id = %s AND attempt = %s",
                                 (attempt + 1, automation, signal_id, attempt))
                requeued.append((automation, signal_id, attempt + 1))
        return requeued

    # --- mediator attempts ---------------------------------------------------

    def begin_mediation(self, key: str, proposal_id: str) -> Optional[dict]:
        """Record that the mediator is about to be called for `key`.

        Returns None when this call made the record (go ahead), or the earlier record
        `{status, result}`: `attempted` means a call began and its end was never recorded.
        """
        with self.connect() as conn:
            made = conn.execute(
                "INSERT INTO archi_ops.mediations (idempotency_key, proposal_id) VALUES (%s, %s) "
                "ON CONFLICT (idempotency_key) DO NOTHING RETURNING 1", (key, proposal_id)).fetchone()
        return None if made else self.mediation(key)

    def mediation(self, key: str) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute("SELECT status, result FROM archi_ops.mediations WHERE idempotency_key = %s",
                               (key,)).fetchone()
        return {"status": row[0], "result": row[1]} if row else None

    def finish_mediation(self, key: str, status: str, result: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE archi_ops.mediations SET status = %s, result = %s, finished_at = now() "
                         "WHERE idempotency_key = %s", (status, result, key))

    # --- demotions and promotions --------------------------------------------

    def record_demotion(self, automation: str, run_id: str, at: datetime, reason: str) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO archi_ops.demotions (automation, run_id, demoted_at, reason) "
                         "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING", (automation, run_id, at, reason))

    def latest_demotion(self, automation: str) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute("SELECT demoted_at, reason FROM archi_ops.demotions WHERE automation = %s "
                               "ORDER BY demoted_at DESC LIMIT 1", (automation,)).fetchone()
        return {"at": row[0].isoformat(), "reason": row[1]} if row else None

    def promotion_status(self, instance_dir: Path, change: Optional[str], automation: Optional[str] = None) -> dict:
        """Is `change` an approved promotion of `automation`? `{approved, at, reason}`.

        Bound first, then asked of the ledger. The change must carry this automation's name in its own
        `promotion.json` (an id in ops.yaml proves nothing), and must be the newest promotion change for the
        automation. Approved then means okg's decision ledger derives it as `retired` (an approve verdict).
        Whether the approval came after the latest demotion is the loop's check (it has the demotion log).
        Anything unreadable, including a missing okg binary, is not approved.
        """
        from archi.ops.promotion import latest_promotion_change, promotion_record

        def no(reason: str) -> dict:
            return {"approved": False, "at": None, "reason": reason}

        if not change:
            return no("ops.yaml names no promotion change; run `archi ops promote <automation>` and have an "
                      "operator approve the change it opens")
        record = promotion_record(instance_dir, change)
        if record is None:
            return no(f"PACT change {change} is not an archi promotion (it has no {'promotion.json'}); "
                      "`archi ops promote` opens the change this automation needs")
        if automation is not None and record.get("automation") != automation:
            return no(f"PACT change {change} is a promotion of {record.get('automation')!r}, not of {automation!r}")
        newest = latest_promotion_change(instance_dir, record.get("automation"))
        if newest != change:
            return no(f"PACT change {change} is not the newest promotion of {record.get('automation')!r}; "
                      f"{newest} is, and it needs its own approval")
        env = {**os.environ, "OKG_PACT_QUIET_NUDGE": "1", "OKG_PACT_GRAPH_DEPLOYMENT": self.deployment,
               "OKG_DSN": self.dsn}
        env.setdefault("OKG_DEPLOYMENTS_DIR", str(Path(instance_dir) / "deployments"))
        try:
            done = subprocess.run([self.okg or _okg(), "pact", "--root", str(instance_dir), "view", change,
                                   "--format", "status", "--json"], capture_output=True, text=True, env=env,
                                  cwd=str(instance_dir))
        except OSError as exc:  # no okg binary, or it cannot run: shadow, with the reason
            return no(f"could not run okg to read PACT change {change}: {type(exc).__name__}: {exc}")
        try:
            derived = json.loads(done.stdout)["derived_state"]
        except (ValueError, KeyError, TypeError):
            return {"approved": False, "at": None, "reason": f"could not read the state of PACT change {change}: "
                    f"{(done.stderr or done.stdout)[-300:].strip()}"}
        state = derived.get("state")
        if state != "retired" or not derived.get("authoritative"):
            return {"approved": False, "at": None, "reason": (
                f"PACT change {change} is {state}"
                + ("" if derived.get("authoritative") else " (the ledger was not consulted)")
                + f"; an operator approves it with `okg pact decide {change} --verdict approve`")}
        with self.connect() as conn:
            row = conn.execute("SELECT decided_at FROM okg.pact_decisions WHERE change_id = %s AND verdict = 'approve' "
                               "ORDER BY id DESC LIMIT 1", (change,)).fetchone()
        if row is None:
            return {"approved": False, "at": None, "reason": f"PACT change {change} has no approve verdict in the ledger"}
        return {"approved": True, "at": row[0].isoformat(), "reason": ""}

    def ticket_cursor(self, automation: str, source: str) -> Optional[dict]:
        """`{updated, generation}` for a ticket_queue automation's reading of one source, or None (never read)."""
        with self.connect() as conn:
            row = conn.execute("SELECT updated, generation FROM archi_ops.ticket_cursors WHERE automation = %s "
                               "AND source = %s", (automation, source)).fetchone()
        return {"updated": row[0], "generation": row[1]} if row else None

    def set_ticket_cursor(self, automation: str, source: str, updated: Optional[datetime], generation: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO archi_ops.ticket_cursors (automation, source, updated, generation) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (automation, source) DO UPDATE SET updated = EXCLUDED.updated, "
                "generation = EXCLUDED.generation, read_at = now()", (automation, source, updated, generation))

    def mark_signal(self, automation: str, signal_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE archi_ops.signals SET status = %s WHERE automation = %s AND signal_id = %s",
                         (status, automation, signal_id))

    def mark_signal_started(self, automation: str, signal_id: str) -> None:
        self.mark_signal(automation, signal_id, "started")


def day_scope(automation: str, day: str) -> str:
    """The cost_budget scope of one automation's UTC day."""
    return f"day:{automation}:{day}"
