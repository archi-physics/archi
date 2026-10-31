"""OpsIO's promotion lookup: `mode: live` counts only for a PACT change the ledger says a person approved."""
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pytest

from archi.ops import okg_io
from archi.ops.okg_io import OpsIO

APPROVED_AT = datetime(2026, 10, 25, 15, tzinfo=timezone.utc)


def promo(root, change, automation="triage", at="2026-10-25T10:00:00Z", where="changes"):
    """A promotion change as `archi ops promote` leaves it: a directory holding promotion.json."""
    directory = Path(root) / "pact" / where / change
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "promotion.json").write_text(json.dumps(
        {"automation": automation, "change": change, "requested_at": at}))


class Conn:
    def __init__(self, row):
        self.row, self.sql = row, []

    def transaction(self):
        self.sql.append(("BEGIN", ()))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.sql.append((sql, params))
        return self

    def fetchone(self):
        return self.row


def io_with(monkeypatch, tmp_path, view=None, stdout=None, row=(APPROVED_AT,), returncode=0):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"], seen["cwd"] = cmd, kw["env"], kw["cwd"]
        out = stdout if stdout is not None else json.dumps({"derived_state": view})
        return subprocess.CompletedProcess(cmd, returncode, out, "")

    monkeypatch.setattr(okg_io.subprocess, "run", fake_run)
    promo(tmp_path, "promote-x")  # the change every default case asks about: a promotion of "triage"
    io = OpsIO("postgresql://x", "dep", okg="okg")
    conn = Conn(row)
    monkeypatch.setattr(io, "connect", lambda: conn)
    return io, seen, conn


def test_an_approved_change_is_read_with_its_approval_time(monkeypatch, tmp_path):
    io, seen, conn = io_with(monkeypatch, tmp_path, {"state": "retired", "authoritative": True})
    status = io.promotion_status(tmp_path, "promote-x", "triage")
    assert status == {"approved": True, "at": APPROVED_AT.isoformat(), "reason": ""}
    assert seen["cmd"] == ["okg", "pact", "--root", str(tmp_path), "view", "promote-x", "--format", "status", "--json"]
    assert seen["env"]["OKG_PACT_GRAPH_DEPLOYMENT"] == "dep" and seen["env"]["OKG_DSN"] == "postgresql://x"
    assert seen["cwd"] == str(tmp_path)
    assert conn.sql[0][1] == ("promote-x",) and "verdict = 'approve'" in conn.sql[0][0]


@pytest.mark.parametrize("view, fragment", [
    ({"state": "awaiting_review", "authoritative": True}, "is awaiting_review; an operator approves it with `okg pact decide promote-x"),
    ({"state": "in_progress", "authoritative": True}, "is in_progress"),
    ({"state": "retired", "authoritative": False}, "the ledger was not consulted"),
])
def test_a_change_that_is_not_approved_says_how_to_approve_it(monkeypatch, tmp_path, view, fragment):
    io, _, _ = io_with(monkeypatch, tmp_path, view)
    status = io.promotion_status(tmp_path, "promote-x", "triage")
    assert status["approved"] is False and status["at"] is None and fragment in status["reason"]


def test_a_retired_change_with_no_approve_verdict_is_not_approved(monkeypatch, tmp_path):
    io, _, _ = io_with(monkeypatch, tmp_path, {"state": "retired", "authoritative": True}, row=None)
    status = io.promotion_status(tmp_path, "promote-x", "triage")
    assert status["approved"] is False and "no approve verdict" in status["reason"]


def test_an_unreadable_answer_is_not_approved(monkeypatch, tmp_path):
    io, _, _ = io_with(monkeypatch, tmp_path, stdout="not json", returncode=1)
    status = io.promotion_status(tmp_path, "promote-x", "triage")
    assert status["approved"] is False and "could not read the state of PACT change promote-x" in status["reason"]


def test_no_change_named_is_not_approved_and_asks_for_a_promotion(monkeypatch, tmp_path):
    io, seen, _ = io_with(monkeypatch, tmp_path, {})
    status = io.promotion_status(tmp_path, None, "triage")
    assert status["approved"] is False and "names no promotion change" in status["reason"] and "cmd" not in seen


def test_demotions_are_written_and_read_in_the_archi_ops_table(monkeypatch):
    io = OpsIO("postgresql://x", "dep")
    conn = Conn((APPROVED_AT, "demoted to shadow: x"))
    monkeypatch.setattr(io, "connect", lambda: conn)
    io.record_demotion("triage", "triage~s~1", APPROVED_AT, "demoted to shadow: x")
    assert "INSERT INTO archi_ops.demotions" in conn.sql[0][0] and "ON CONFLICT DO NOTHING" in conn.sql[0][0]
    assert io.latest_demotion("triage") == {"at": APPROVED_AT.isoformat(), "reason": "demoted to shadow: x"}
    conn.row = None
    assert io.latest_demotion("triage") is None


def test_the_tables_include_everything_the_runtime_writes():
    for table in ("signals", "halts", "mediations", "demotions"):
        assert f"archi_ops.{table}" in okg_io.SIGNALS_DDL


APPROVED_VIEW = {"state": "retired", "authoritative": True}


def test_an_approved_change_that_is_not_an_archi_promotion_does_not_make_an_automation_live(monkeypatch, tmp_path):
    """Hand-editing ops.yaml to name some other approved PACT change must not work."""
    io, seen, _ = io_with(monkeypatch, tmp_path, APPROVED_VIEW)
    (tmp_path / "pact" / "archive" / "fix-the-docs").mkdir(parents=True)  # approved, but not a promotion
    status = io.promotion_status(tmp_path, "fix-the-docs", "triage")
    assert status["approved"] is False and "is not an archi promotion" in status["reason"]
    status = io.promotion_status(tmp_path, "not-even-a-directory", "triage")
    assert status["approved"] is False and "is not an archi promotion" in status["reason"]


def test_another_automations_approved_promotion_does_not_count(monkeypatch, tmp_path):
    io, _, _ = io_with(monkeypatch, tmp_path, APPROVED_VIEW)
    promo(tmp_path, "promote-other", automation="other", at="2026-10-25T11:00:00Z", where="archive")
    status = io.promotion_status(tmp_path, "promote-other", "triage")
    assert status["approved"] is False
    assert "is a promotion of 'other', not of 'triage'" in status["reason"]


def test_an_old_approved_promotion_is_superseded_by_a_newer_one(monkeypatch, tmp_path):
    io, _, _ = io_with(monkeypatch, tmp_path, APPROVED_VIEW)
    promo(tmp_path, "promote-new", automation="triage", at="2026-10-26T09:00:00Z")  # opened later, not approved yet
    status = io.promotion_status(tmp_path, "promote-x", "triage")
    assert status["approved"] is False
    assert "not the newest promotion of 'triage'" in status["reason"] and "promote-new is" in status["reason"]
    assert io.promotion_status(tmp_path, "promote-new", "triage")["approved"] is True  # the newest one is the one asked about


def test_the_newest_promotion_is_found_across_changes_and_archive(tmp_path):
    from archi.ops.promotion import latest_promotion_change

    assert latest_promotion_change(tmp_path, "triage") is None
    promo(tmp_path, "a", at="2026-10-25T10:00:00Z", where="archive")
    promo(tmp_path, "b", at="2026-10-25T12:00:00Z", where="changes")
    promo(tmp_path, "c", automation="other", at="2026-10-25T13:00:00Z")
    assert latest_promotion_change(tmp_path, "triage") == "b"
    assert latest_promotion_change(tmp_path, "other") == "c"


def test_a_missing_okg_binary_lands_in_shadow_with_a_reason_not_an_exception(monkeypatch, tmp_path):
    promo(tmp_path, "promote-x")

    def no_binary(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'okg'")

    monkeypatch.setattr(okg_io.subprocess, "run", no_binary)
    status = OpsIO("postgresql://x", "dep", okg="okg").promotion_status(tmp_path, "promote-x", "triage")
    assert status["approved"] is False and "could not run okg" in status["reason"] and "FileNotFoundError" in status["reason"]


def test_ensure_tables_takes_the_advisory_lock_inside_one_transaction_before_any_ddl(monkeypatch):
    io = OpsIO("postgresql://x", "dep")
    conn = Conn(None)
    monkeypatch.setattr(io, "connect", lambda: conn)
    io.ensure_tables()
    kinds = [sql.split()[0] for sql, _ in conn.sql]
    assert kinds == ["BEGIN", "SELECT", "CREATE"][: len(kinds)] and kinds[0] == "BEGIN"
    assert "pg_advisory_xact_lock" in conn.sql[1][0] and conn.sql[1][1] == (okg_io.DDL_LOCK_KEY,)
    assert conn.sql[2][0] is okg_io.SIGNALS_DDL
    assert 0 < okg_io.DDL_LOCK_KEY < 2 ** 63


def test_two_concurrent_ensure_tables_never_interleave_their_ddl(monkeypatch):
    """A fake database whose advisory lock serialises transactions, as Postgres's does. Without the lock the two
    DDL batches interleave; with it they run one after the other."""
    import threading
    import time

    lock, log = threading.Lock(), []

    class Db(Conn):
        def __enter__(self):
            return self

        def transaction(self):
            return self

        def execute(self, sql, params=()):
            if "pg_advisory_xact_lock" in sql:
                lock.acquire()
                self.holding = True
            elif sql is okg_io.SIGNALS_DDL:
                log.append(("start", threading.get_ident()))
                time.sleep(0.05)
                log.append(("end", threading.get_ident()))
            return self

        def __exit__(self, *exc):
            if getattr(self, "holding", False):
                lock.release()
            return False

    ios = [OpsIO("postgresql://x", "dep") for _ in range(2)]
    for handle in ios:
        monkeypatch.setattr(handle, "connect", lambda: Db(None))
    threads = [threading.Thread(target=h.ensure_tables) for h in ios]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert [k for k, _ in log] == ["start", "end", "start", "end"]


class Workflows:
    def __init__(self, rows):
        self.rows = rows

    def list_workflows(self, workflow_ids, load_input=False):
        return [{"workflow_id": w, "status": "CANCELLED"} for w in workflow_ids]


class RequeueConn(Conn):
    """Started signals for the first SELECT; then the status each UPDATE ... RETURNING gives back."""

    def __init__(self, started, after):
        super().__init__(None)
        self.started, self.after = started, list(after)

    def fetchall(self):
        return self.started

    def fetchone(self):
        return (self.after.pop(0),)


def test_a_cancelled_ticket_signal_with_a_newer_one_is_superseded_not_requeued(monkeypatch):
    """Second review: version 1's run was cancelled after version 2's signal had started. Requeueing version 1
    would triage the ticket again, so the UPDATE supersedes a ticket signal that has a newer one, and only a
    signal that went back to `new` is reported as requeued."""
    io = OpsIO("postgresql://x", "dep", okg="okg")
    conn = RequeueConn([("ticket-triage", "redmine-4820-v1", 1), ("ticket-triage", "redmine-4821-v1", 1)],
                       ["superseded", "new"])
    monkeypatch.setattr(io, "connect", lambda: conn)
    monkeypatch.setattr(io, "_client", lambda: Workflows(None))
    assert io.requeue_cancelled_signals() == [("ticket-triage", "redmine-4821-v1", 2)]
    updates = [sql for sql, _ in conn.sql if sql.startswith("UPDATE")]
    assert len(updates) == 2 and all("s.source = 'ticket'" in u and "n.created_at > s.created_at" in u
                                     and "THEN 'superseded' ELSE 'new'" in u for u in updates)
