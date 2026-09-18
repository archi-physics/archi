from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.utils.postgres_service_factory import PostgresServiceFactory

JIRA_TRIGGER_RETRY_TIMEOUT_SECONDS = 60
JIRA_TRIGGER_FINAL_TIMEOUT_SECONDS = 600
# Canonical runtime DDL for the trigger ledger. Fresh-deployment SQL templates
# carry the same statements and are checked for parity in the Jira unit tests.
JIRA_RESPONDER_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS jira_responder_triggers (
        trigger_key TEXT PRIMARY KEY,
        trigger_type TEXT NOT NULL CHECK (trigger_type IN ('issue','mention_comment')),
        issue_key TEXT NOT NULL,
        trigger_comment_id TEXT,
        status TEXT NOT NULL CHECK (status IN ('answering','answered','failed')),
        retry_used BOOLEAN NOT NULL DEFAULT FALSE,
        last_error TEXT,
        conversation_id INTEGER REFERENCES conversation_metadata(conversation_id) ON DELETE SET NULL,
        response_comment_id TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_jira_responder_triggers_issue
        ON jira_responder_triggers(issue_key)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_jira_responder_triggers_status
        ON jira_responder_triggers(status, updated_at)
    """,
)


class JiraTriggerStore:
    def __init__(self, postgres_factory: PostgresServiceFactory) -> None:
        self.postgres_factory = postgres_factory

    def ensure_schema(self) -> None:
        with self.postgres_factory.connection_pool.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    for statement in JIRA_RESPONDER_SCHEMA_STATEMENTS:
                        cursor.execute(statement)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def claim_trigger(
        self,
        *,
        trigger_key: str,
        trigger_type: str,
        issue_key: str,
        trigger_comment_id: Optional[str],
    ) -> bool:
        now = datetime.now(timezone.utc)
        with self.postgres_factory.connection_pool.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT status, retry_used, updated_at
                        FROM jira_responder_triggers
                        WHERE trigger_key = %s
                        FOR UPDATE
                        """,
                        (trigger_key,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        cursor.execute(
                            """
                            INSERT INTO jira_responder_triggers (
                                trigger_key,
                                trigger_type,
                                issue_key,
                                trigger_comment_id,
                                status,
                                retry_used,
                                last_error,
                                conversation_id,
                                response_comment_id,
                                created_at,
                                updated_at
                            )
                            VALUES (
                                %s, %s, %s, %s, 'answering', FALSE, NULL,
                                NULL, NULL, NOW(), NOW()
                            )
                            """,
                            (
                                trigger_key,
                                trigger_type,
                                issue_key,
                                trigger_comment_id,
                            ),
                        )
                        conn.commit()
                        return True

                    status, retry_used, updated_at = row
                    should_answer = self._apply_existing_trigger_state(
                        cursor,
                        trigger_key,
                        str(status),
                        bool(retry_used),
                        updated_at,
                        now,
                    )
                conn.commit()
                return should_answer
            except Exception:
                conn.rollback()
                raise

    def _apply_existing_trigger_state(
        self,
        cursor: Any,
        trigger_key: str,
        status: str,
        retry_used: bool,
        updated_at: datetime,
        now: datetime,
    ) -> bool:
        if status in ("answered", "failed"):
            return False
        if status != "answering":
            raise ValueError(f"Unknown Jira responder trigger status: {status}")

        age_seconds = (now - ensure_aware_utc(updated_at)).total_seconds()
        if not retry_used:
            if age_seconds < JIRA_TRIGGER_RETRY_TIMEOUT_SECONDS:
                return False
            cursor.execute(
                """
                UPDATE jira_responder_triggers
                SET retry_used = TRUE,
                    last_error = NULL,
                    updated_at = NOW()
                WHERE trigger_key = %s
                """,
                (trigger_key,),
            )
            self._require_single_trigger_row(cursor)
            return True

        if age_seconds < JIRA_TRIGGER_FINAL_TIMEOUT_SECONDS:
            return False

        cursor.execute(
            """
            UPDATE jira_responder_triggers
            SET status = 'failed',
                last_error = %s,
                updated_at = NOW()
            WHERE trigger_key = %s
            """,
            ("Trigger remained answering after the final stale timeout.", trigger_key),
        )
        self._require_single_trigger_row(cursor)
        return False

    def mark_answered(
        self, trigger_key: str, response_comment_id: Optional[str]
    ) -> None:
        self._execute_update(
            """
            UPDATE jira_responder_triggers
            SET status = 'answered',
                last_error = NULL,
                response_comment_id = %s,
                updated_at = NOW()
            WHERE trigger_key = %s
            """,
            (response_comment_id, trigger_key),
        )

    def mark_posted_but_unconfirmed(
        self,
        *,
        trigger_key: str,
        trigger_type: str,
        issue_key: str,
        trigger_comment_id: Optional[str],
        response_comment_id: Optional[str],
        last_error: str,
    ) -> None:
        self._execute_update(
            """
            INSERT INTO jira_responder_triggers (
                trigger_key,
                trigger_type,
                issue_key,
                trigger_comment_id,
                status,
                retry_used,
                last_error,
                conversation_id,
                response_comment_id,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, 'failed', TRUE, %s, NULL, %s, NOW(), NOW()
            )
            ON CONFLICT (trigger_key) DO UPDATE
            SET status = 'failed',
                retry_used = TRUE,
                last_error = EXCLUDED.last_error,
                response_comment_id = EXCLUDED.response_comment_id,
                updated_at = NOW()
            """,
            (
                trigger_key,
                trigger_type,
                issue_key,
                trigger_comment_id,
                last_error,
                response_comment_id,
            ),
        )

    def mark_failed(self, trigger_key: str, last_error: str) -> None:
        self._execute_update(
            """
            UPDATE jira_responder_triggers
            SET status = 'failed',
                last_error = %s,
                updated_at = NOW()
            WHERE trigger_key = %s
            """,
            (last_error, trigger_key),
        )

    def record_last_error(self, trigger_key: str, last_error: str) -> None:
        self._execute_update(
            """
            UPDATE jira_responder_triggers
            SET last_error = %s,
                updated_at = NOW()
            WHERE trigger_key = %s
            """,
            (last_error, trigger_key),
        )

    def link_conversation(self, trigger_key: str, conversation_id: int) -> None:
        self._execute_update(
            """
            UPDATE jira_responder_triggers
            SET conversation_id = %s,
                updated_at = NOW()
            WHERE trigger_key = %s
            """,
            (conversation_id, trigger_key),
        )

    def _execute_update(self, query: str, params: tuple[Any, ...]) -> None:
        with self.postgres_factory.connection_pool.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    self._require_single_trigger_row(cursor)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _require_single_trigger_row(cursor: Any) -> None:
        if cursor.rowcount != 1:
            raise RuntimeError(
                "Expected exactly one Jira responder trigger row to be updated; "
                f"updated {cursor.rowcount} rows."
            )


def ensure_aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("Jira responder trigger updated_at must be a datetime.")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
