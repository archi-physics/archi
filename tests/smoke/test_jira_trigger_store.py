"""Smoke test for Jira responder trigger state against real PostgreSQL.

Reads PGHOST, PGPORT, PGDATABASE, PGUSER, and PGPASSWORD from the smoke runner.
It can also be run against tests/smoke/docker-compose.integration.yaml defaults.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.jira_ticket_responder.store import JiraTriggerStore
from src.utils.connection_pool import ConnectionPool

PG_CONFIG = {
    "host": os.getenv("PGHOST", os.getenv("PG_HOST", "localhost")),
    "port": int(os.getenv("PGPORT", os.getenv("PG_PORT", "5439"))),
    "database": os.getenv("PGDATABASE", os.getenv("PG_DATABASE", "archi")),
    "user": os.getenv("PGUSER", os.getenv("PG_USER", "archi")),
    "password": os.getenv(
        "PGPASSWORD",
        os.getenv("PG_PASSWORD", os.getenv("POSTGRES_PASSWORD", "testpassword123")),
    ),
}


def wait_for_postgres(max_attempts=30, delay=1):
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**PG_CONFIG)
            conn.close()
            return
        except psycopg2.OperationalError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay)


def create_trigger_store():
    wait_for_postgres()
    pool = ConnectionPool(PG_CONFIG, min_conn=1, max_conn=2)
    store = JiraTriggerStore(SimpleNamespace(connection_pool=pool))
    store.ensure_schema()
    return store, pool


def _fetch_trigger(trigger_key):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, retry_used, issue_key, trigger_comment_id,
                       response_comment_id, conversation_id, last_error
                FROM jira_responder_triggers
                WHERE trigger_key = %s
                """,
                (trigger_key,),
            )
            return cursor.fetchone()
    finally:
        conn.close()


def _execute(sql, params):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _create_conversation():
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversation_metadata (client_id, title)
                VALUES (%s, %s)
                RETURNING conversation_id
                """,
                ("jira-smoke", "Jira trigger store smoke test"),
            )
            conversation_id = cursor.fetchone()[0]
        conn.commit()
        return conversation_id
    finally:
        conn.close()


def test_jira_trigger_store_lifecycle_with_real_postgres():
    trigger_store, pool = create_trigger_store()
    suffix = uuid.uuid4().hex
    issue_key = f"SMOKE-{suffix[:8]}"
    issue_trigger = f"issue:{issue_key}"
    comment_trigger = f"comment:{suffix}"
    posted_missing_trigger = f"comment:posted-{suffix}"

    try:
        assert (
            trigger_store.claim_trigger(
                trigger_key=issue_trigger,
                trigger_type="issue",
                issue_key=issue_key,
                trigger_comment_id=None,
            )
            is True
        )
        assert (
            trigger_store.claim_trigger(
                trigger_key=issue_trigger,
                trigger_type="issue",
                issue_key=issue_key,
                trigger_comment_id=None,
            )
            is False
        )

        _execute(
            """
            UPDATE jira_responder_triggers
            SET updated_at = NOW() - INTERVAL '61 seconds',
                retry_used = FALSE,
                status = 'answering'
            WHERE trigger_key = %s
            """,
            (issue_trigger,),
        )
        assert (
            trigger_store.claim_trigger(
                trigger_key=issue_trigger,
                trigger_type="issue",
                issue_key=issue_key,
                trigger_comment_id=None,
            )
            is True
        )
        status, retry_used, *_ = _fetch_trigger(issue_trigger)
        assert status == "answering"
        assert retry_used is True

        _execute(
            """
            UPDATE jira_responder_triggers
            SET updated_at = NOW() - INTERVAL '601 seconds',
                retry_used = TRUE,
                status = 'answering'
            WHERE trigger_key = %s
            """,
            (issue_trigger,),
        )
        assert (
            trigger_store.claim_trigger(
                trigger_key=issue_trigger,
                trigger_type="issue",
                issue_key=issue_key,
                trigger_comment_id=None,
            )
            is False
        )
        status, retry_used, *_ = _fetch_trigger(issue_trigger)
        assert status == "failed"
        assert retry_used is True

        assert (
            trigger_store.claim_trigger(
                trigger_key=comment_trigger,
                trigger_type="mention_comment",
                issue_key=issue_key,
                trigger_comment_id=suffix,
            )
            is True
        )
        trigger_store.mark_answered(comment_trigger, "response-1")
        conversation_id = _create_conversation()
        trigger_store.link_conversation(comment_trigger, conversation_id)

        row = _fetch_trigger(comment_trigger)
        assert row[:6] == (
            "answered",
            False,
            issue_key,
            suffix,
            "response-1",
            conversation_id,
        )

        trigger_store.mark_posted_but_unconfirmed(
            trigger_key=posted_missing_trigger,
            trigger_type="mention_comment",
            issue_key=issue_key,
            trigger_comment_id=f"posted-{suffix}",
            response_comment_id="response-2",
            last_error="posted before local state could be confirmed",
        )
        row = _fetch_trigger(posted_missing_trigger)
        assert row[0] == "failed"
        assert row[1] is True
        assert row[4] == "response-2"
        assert row[6] == "posted before local state could be confirmed"

        try:
            trigger_store.mark_failed(f"comment:missing-{suffix}", "missing")
        except RuntimeError as exc:
            assert "Expected exactly one" in str(exc)
        else:
            raise AssertionError("mark_failed should fail for a missing trigger row.")
    finally:
        _execute(
            """
            DELETE FROM jira_responder_triggers
            WHERE trigger_key IN (%s, %s, %s)
            """,
            (issue_trigger, comment_trigger, posted_missing_trigger),
        )
        pool.close()


if __name__ == "__main__":
    test_jira_trigger_store_lifecycle_with_real_postgres()
    print("Jira trigger store smoke test passed.")
