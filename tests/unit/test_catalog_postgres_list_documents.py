from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService


def _build_service_with_cursor(cursor):
    service = PostgresCatalogService.__new__(PostgresCatalogService)

    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.cursor.return_value.__exit__.return_value = False

    @contextmanager
    def _connect():
        yield conn

    service._connect = _connect
    return service, cursor


def test_list_documents_returns_pagination_metadata():
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        {"count": 10},  # total
        {"count": 8},   # enabled_count
    ]
    cursor.fetchall.return_value = [
        {
            "resource_hash": "doc-1",
            "display_name": "Doc 1",
            "source_type": "local_files",
            "url": "file:///tmp/doc-1.md",
            "size_bytes": 10,
            "suffix": "md",
            "ingested_at": datetime(2026, 2, 20, tzinfo=timezone.utc),
            "ingestion_status": "embedded",
            "ingestion_error": None,
            "enabled": True,
        },
        {
            "resource_hash": "doc-2",
            "display_name": "Doc 2",
            "source_type": "web",
            "url": "https://example.com/doc-2",
            "size_bytes": 20,
            "suffix": "html",
            "ingested_at": None,
            "ingestion_status": "pending",
            "ingestion_error": None,
            "enabled": False,
        },
    ]

    service, _ = _build_service_with_cursor(cursor)
    result = service.list_documents(limit=2, offset=0)

    assert result["total"] == 10
    assert result["enabled_count"] == 8
    assert result["has_more"] is True
    assert result["next_offset"] == 2
    assert len(result["documents"]) == 2
    assert result["documents"][0]["enabled"] is True
    assert result["documents"][1]["enabled"] is False


def test_list_documents_limit_all_disables_pagination():
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        {"count": 2},  # total
        {"count": 2},  # enabled_count
    ]
    cursor.fetchall.return_value = [
        {
            "resource_hash": "doc-1",
            "display_name": "Doc 1",
            "source_type": "local_files",
            "url": "file:///tmp/doc-1.md",
            "size_bytes": 10,
            "suffix": "md",
            "ingested_at": datetime(2026, 2, 20, tzinfo=timezone.utc),
            "ingestion_status": "embedded",
            "ingestion_error": None,
            "enabled": True,
        }
    ]

    service, _ = _build_service_with_cursor(cursor)
    result = service.list_documents(limit=None, offset=0)

    assert result["has_more"] is False
    assert result["next_offset"] is None
    assert result["limit"] == 2

    data_query = cursor.execute.call_args_list[2][0][0]
    assert "LIMIT" not in data_query
