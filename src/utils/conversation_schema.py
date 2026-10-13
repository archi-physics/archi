"""
conversation_schema.py – Idempotent migration for conversation_metadata columns.

Called once at service startup (service_chat.py) so that existing Postgres
volumes gain the is_pinned / is_archived columns without needing a full
re-initialisation.  Pattern mirrors PlaybookService.ensure_schema().
"""
import psycopg2

from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_conversation_schema(pg_config: dict) -> None:
    """Add is_pinned and is_archived to conversation_metadata if absent.

    Safe to call on every startup — uses ADD COLUMN IF NOT EXISTS so repeated
    runs are no-ops.  Raises on unexpected DB errors so the caller can decide
    whether to block startup.
    """
    ddl_statements = [
        (
            "ALTER TABLE conversation_metadata "
            "ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN NOT NULL DEFAULT FALSE"
        ),
        (
            "ALTER TABLE conversation_metadata "
            "ADD COLUMN IF NOT EXISTS is_archived BOOLEAN NOT NULL DEFAULT FALSE"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_conv_meta_pinned "
            "ON conversation_metadata(user_id) WHERE is_pinned = TRUE"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_conv_meta_archived "
            "ON conversation_metadata(client_id) WHERE is_archived = TRUE"
        ),
    ]

    conn = psycopg2.connect(**pg_config)
    try:
        with conn.cursor() as cursor:
            for stmt in ddl_statements:
                cursor.execute(stmt)
        conn.commit()
        logger.info("conversation_schema: is_pinned / is_archived columns ensured")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
