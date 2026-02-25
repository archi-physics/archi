"""
PostgreSQL-backed CatalogService.

Replaces SQLite-based catalog with PostgreSQL 'documents' table.
Provides the same interface as the original CatalogService.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor
from langchain_core.documents import Document

from src.data_manager.vectorstore.loader_utils import load_doc_from_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".pdf", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".html", ".htm", ".log", ".py", ".c", ".cpp", ".C", ".h",
}

# Map metadata keys to PostgreSQL column names
_METADATA_COLUMN_MAP = {
    "path": "file_path",
    "file_path": "file_path",
    "display_name": "display_name",
    "source_type": "source_type",
    "url": "url",
    "ticket_id": "ticket_id",
    "suffix": "suffix",
    "size_bytes": "size_bytes",
    "original_path": "original_path",
    "base_path": "base_path",
    "relative_path": "relative_path",
    "created_at": "created_at",
    "modified_at": "file_modified_at",
    "file_modified_at": "file_modified_at",
    "ingested_at": "ingested_at",
}


@dataclass
class PostgresCatalogService:
    """
    PostgreSQL-backed document catalog service.
    
    Stores document metadata in the PostgreSQL 'documents' table,
    replacing the legacy SQLite catalog.
    """

    data_path: Path | str
    pg_config: Dict[str, Any]
    include_extensions: Sequence[str] = field(default_factory=lambda: sorted(DEFAULT_TEXT_EXTENSIONS))
    _file_index: Dict[str, str] = field(init=False, default_factory=dict)
    _metadata_index: Dict[str, str] = field(init=False, default_factory=dict)
    _id_cache: Dict[str, int] = field(init=False, default_factory=dict)  # resource_hash -> document id

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        if self.include_extensions:
            self.include_extensions = tuple(ext.lower() for ext in self.include_extensions)
        self.refresh()

    def _connect_with_retry(self) -> psycopg2.extensions.connection:
        """Open a raw connection with retry logic for transient failures."""
        last_exc: Exception | None = None
        for attempt in range(1, 4):  # up to 3 attempts
            try:
                return psycopg2.connect(**self.pg_config)
            except psycopg2.OperationalError as exc:
                last_exc = exc
                if attempt < 3:
                    wait = attempt * 2  # 2s, 4s
                    logger.warning(
                        "Postgres connection attempt %d/3 failed (%s); retrying in %ds",
                        attempt, exc, wait,
                    )
                    time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    @contextmanager
    def _connect(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Context manager for database connections with retry."""
        conn = self._connect_with_retry()
        try:
            yield conn
        finally:
            conn.close()

    def refresh(self) -> None:
        """Reload file and metadata indices from PostgreSQL."""
        logger.debug("Refreshing catalog indices from PostgreSQL documents table")
        self._file_index = {}
        self._metadata_index = {}
        self._id_cache = {}
        
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, resource_hash, file_path 
                    FROM documents 
                    WHERE NOT is_deleted
                """)
                rows = cur.fetchall()
        
        for row in rows:
            resource_hash = row["resource_hash"]
            stored_path = row["file_path"]
            self._file_index[resource_hash] = stored_path
            self._metadata_index[resource_hash] = stored_path
            self._id_cache[resource_hash] = row["id"]

    @property
    def file_index(self) -> Dict[str, str]:
        return self._file_index

    @property
    def metadata_index(self) -> Dict[str, str]:
        return self._metadata_index

    def get_document_id(self, resource_hash: str) -> Optional[int]:
        """Get the PostgreSQL document ID for a resource hash."""
        if resource_hash in self._id_cache:
            return self._id_cache[resource_hash]
        
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM documents WHERE resource_hash = %s AND NOT is_deleted",
                    (resource_hash,)
                )
                row = cur.fetchone()
                if row:
                    self._id_cache[resource_hash] = row[0]
                    return row[0]
        return None

    def upsert_resource(
        self,
        resource_hash: str,
        path: str,
        metadata: Optional[Dict[str, str]],
    ) -> int:
        """
        Insert or update a resource in the documents table.
        
        Returns:
            The document ID (for linking to document_chunks)
        """
        payload = metadata or {}
        display_name = payload.get("display_name") or resource_hash
        source_type = payload.get("source_type") or "unknown"

        # Build extra_json from non-column fields
        extra = dict(payload)
        for key in _METADATA_COLUMN_MAP:
            extra.pop(key, None)
        extra_json = json.dumps(extra, sort_keys=True) if extra else None
        extra_text = _build_extra_text(payload)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (
                        resource_hash,
                        file_path,
                        display_name,
                        source_type,
                        url,
                        ticket_id,
                        suffix,
                        size_bytes,
                        original_path,
                        base_path,
                        relative_path,
                        file_modified_at,
                        ingested_at,
                        ingestion_status,
                        extra_json,
                        extra_text,
                        is_deleted
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s, FALSE)
                    ON CONFLICT (resource_hash) DO UPDATE SET
                        file_path = EXCLUDED.file_path,
                        display_name = EXCLUDED.display_name,
                        source_type = EXCLUDED.source_type,
                        url = EXCLUDED.url,
                        ticket_id = EXCLUDED.ticket_id,
                        suffix = EXCLUDED.suffix,
                        size_bytes = EXCLUDED.size_bytes,
                        original_path = EXCLUDED.original_path,
                        base_path = EXCLUDED.base_path,
                        relative_path = EXCLUDED.relative_path,
                        file_modified_at = EXCLUDED.file_modified_at,
                        ingested_at = EXCLUDED.ingested_at,
                        extra_json = EXCLUDED.extra_json,
                        extra_text = EXCLUDED.extra_text,
                        is_deleted = FALSE,
                        deleted_at = NULL
                    RETURNING id
                """, (
                    resource_hash,
                    path,
                    display_name,
                    source_type,
                    payload.get("url"),
                    payload.get("ticket_id"),
                    payload.get("suffix"),
                    _coerce_int(payload.get("size_bytes")),
                    payload.get("original_path"),
                    payload.get("base_path"),
                    payload.get("relative_path"),
                    _parse_timestamp(payload.get("modified_at") or payload.get("file_modified_at")),
                    _parse_timestamp(payload.get("ingested_at")),
                    extra_json,
                    extra_text,
                ))
                document_id = cur.fetchone()[0]
            conn.commit()

        self._file_index[resource_hash] = path
        self._metadata_index[resource_hash] = path
        self._id_cache[resource_hash] = document_id
        
        return document_id

    def delete_resource(self, resource_hash: str) -> None:
        """Soft-delete a resource."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents 
                    SET is_deleted = TRUE, deleted_at = NOW()
                    WHERE resource_hash = %s
                """, (resource_hash,))
            conn.commit()
        
        self._file_index.pop(resource_hash, None)
        self._metadata_index.pop(resource_hash, None)
        self._id_cache.pop(resource_hash, None)

    def get_resource_hashes_by_metadata_filter(self, metadata_field: str, value: str) -> List[str]:
        """Return resource hashes matching the metadata filter."""
        matches = self.get_metadata_by_filter(metadata_field, value=value)
        return [resource_hash for resource_hash, _ in matches]

    def get_metadata_by_filter(
        self,
        metadata_field: str,
        value: Optional[str] = None,
        metadata_keys: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Return (resource_hash, metadata) pairs matching the filter."""
        if value is None and metadata_field in kwargs:
            value = kwargs[metadata_field]

        column = _METADATA_COLUMN_MAP.get(metadata_field)
        
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if column:
                    if value is None:
                        cur.execute(f"""
                            SELECT * FROM documents 
                            WHERE NOT is_deleted AND {column} IS NOT NULL AND {column} != ''
                        """)
                    else:
                        cur.execute(f"""
                            SELECT * FROM documents 
                            WHERE NOT is_deleted AND {column} = %s
                        """, (str(value),))
                else:
                    cur.execute("SELECT * FROM documents WHERE NOT is_deleted")
                rows = cur.fetchall()

        matches: List[Tuple[str, Dict[str, Any]]] = []
        expected = str(value) if value is not None else None
        
        for row in rows:
            metadata = self._row_to_metadata(row)
            if metadata_field not in metadata:
                continue
            if expected is not None and metadata.get(metadata_field) != expected:
                continue
            if metadata_keys:
                metadata = {k: metadata[k] for k in metadata_keys if k in metadata}
            matches.append((row["resource_hash"], metadata))
        
        return matches

    def search_metadata(
        self,
        query: str,
        *,
        limit: Optional[int] = 5,
        filters: Optional[Dict[str, str] | List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documents by query and/or filters."""
        if not query and not filters:
            return []

        where_clauses: List[str] = ["NOT is_deleted"]
        params: List[object] = []

        if filters:
            filter_groups = [filters] if isinstance(filters, dict) else filters
            group_clauses: List[str] = []
            for group in filter_groups:
                if not isinstance(group, dict):
                    continue
                sub_clauses: List[str] = []
                for key, val in group.items():
                    column = _METADATA_COLUMN_MAP.get(key)
                    if column:
                        sub_clauses.append(f"{column} = %s")
                        params.append(str(val))
                    else:
                        sub_clauses.append("extra_text ILIKE %s")
                        params.append(f"%{key}:{val}%")
                if sub_clauses:
                    group_clauses.append("(" + " AND ".join(sub_clauses) + ")")
            if group_clauses:
                where_clauses.append("(" + " OR ".join(group_clauses) + ")")

        if query:
            like = f"%{query}%"
            where_clauses.append("""
                (display_name ILIKE %s OR source_type ILIKE %s OR url ILIKE %s 
                 OR ticket_id ILIKE %s OR file_path ILIKE %s OR original_path ILIKE %s 
                 OR relative_path ILIKE %s OR extra_text ILIKE %s)
            """)
            params.extend([like] * 8)

        sql = f"""
            SELECT * FROM documents
            WHERE {" AND ".join(where_clauses)}
            ORDER BY COALESCE(file_modified_at, created_at, ingested_at) DESC NULLS LAST
        """
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            path = self._resolve_path(row["file_path"])
            results.append({
                "hash": row["resource_hash"],
                "path": path,
                "metadata": self._row_to_metadata(row),
            })
        return results

    def iter_files(self) -> Iterable[Tuple[str, Path]]:
        """Iterate over all indexed files."""
        for resource_hash, stored_path in self._file_index.items():
            path = self._resolve_path(stored_path)
            if not path.exists():
                logger.debug("File for resource hash %s not found; skipping.", resource_hash)
                continue
            if self.include_extensions and path.suffix.lower() not in self.include_extensions:
                logger.debug("File %s has excluded extension; skipping.", path)
                continue
            yield resource_hash, path

    def get_metadata_for_hash(self, hash: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific resource hash."""
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM documents WHERE resource_hash = %s AND NOT is_deleted",
                    (hash,)
                )
                row = cur.fetchone()
        
        if not row:
            return None
        return self._row_to_metadata(row)

    def get_distinct_metadata(self, fields: Sequence[str]) -> Dict[str, List[str]]:
        """Return distinct values for the requested metadata columns."""
        result: Dict[str, List[str]] = {}
        allowed = {
            "source_type",
            "suffix",
            "ticket_id",
            "git_repo",
            "url",
        }
        wanted = [f for f in fields if f in allowed]
        if not wanted:
            return result

        with self._connect() as conn:
            with conn.cursor() as cur:
                for field in wanted:
                    cur.execute(
                        f"SELECT DISTINCT {field} FROM documents WHERE NOT is_deleted AND {field} IS NOT NULL"
                    )
                    vals = [row[0] for row in cur.fetchall() if row and row[0] is not None]
                    result[field] = vals
        return result

    def get_filepath_for_hash(self, hash: str) -> Optional[Path]:
        """Get the file path for a resource hash.
        
        First checks the in-memory cache, then falls back to database query
        to handle documents added after service startup.
        """
        stored = self._file_index.get(hash)
        if not stored:
            # Fall back to database query for documents added after startup
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT file_path FROM documents WHERE resource_hash = %s AND NOT is_deleted",
                        (hash,)
                    )
                    row = cur.fetchone()
            if row:
                stored = row[0]
                # Update cache for future lookups
                self._file_index[hash] = stored
            else:
                return None
        path = self._resolve_path(stored)
        return path if path.exists() else None

    def get_document_for_hash(self, hash: str) -> Optional[Document]:
        """Reconstruct a Document for the given resource hash."""
        path = self.get_filepath_for_hash(hash)
        if not path:
            return None
        doc = load_doc_from_path(path)
        metadata = self.get_metadata_for_hash(hash)
        if doc and metadata:
            doc.metadata.update(metadata)
        return doc

    # =========================================================================
    # Per-Chat Document Selection Methods (uses conversation_doc_overrides)
    # =========================================================================

    def is_document_enabled(self, conversation_id: str, document_hash: str) -> bool:
        """Check if a document is enabled for a conversation."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT enabled FROM conversation_doc_overrides 
                    WHERE conversation_id = %s AND document_hash = %s
                """, (int(conversation_id), document_hash))
                row = cur.fetchone()
        
        if row is None:
            return True  # Default: enabled
        return bool(row[0])

    def set_document_enabled(self, conversation_id: str, document_hash: str, enabled: bool) -> None:
        """Set whether a document is enabled for a conversation."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_doc_overrides (conversation_id, document_hash, enabled)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (conversation_id, document_hash) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        updated_at = NOW()
                """, (int(conversation_id), document_hash, enabled))
            conn.commit()

    def bulk_set_enabled(self, conversation_id: str, document_hashes: Sequence[str], enabled: bool) -> int:
        """Set enabled state for multiple documents."""
        if not document_hashes:
            return 0
        
        with self._connect() as conn:
            with conn.cursor() as cur:
                for doc_hash in document_hashes:
                    cur.execute("""
                        INSERT INTO conversation_doc_overrides (conversation_id, document_hash, enabled)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (conversation_id, document_hash) DO UPDATE SET
                            enabled = EXCLUDED.enabled,
                            updated_at = NOW()
                    """, (int(conversation_id), doc_hash, enabled))
            conn.commit()
        return len(document_hashes)

    def get_disabled_hashes(self, conversation_id: str) -> Set[str]:
        """Get document hashes disabled for a conversation."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document_hash FROM conversation_doc_overrides 
                    WHERE conversation_id = %s AND NOT enabled
                """, (int(conversation_id),))
                rows = cur.fetchall()
        return {row[0] for row in rows}

    def get_enabled_hashes(self, conversation_id: str) -> Set[str]:
        """Get document hashes explicitly enabled for a conversation."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document_hash FROM conversation_doc_overrides 
                    WHERE conversation_id = %s AND enabled
                """, (int(conversation_id),))
                rows = cur.fetchall()
        return {row[0] for row in rows}

    def get_selection_state(self, conversation_id: str) -> Dict[str, bool]:
        """Get selection state for all explicitly set documents."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document_hash, enabled FROM conversation_doc_overrides 
                    WHERE conversation_id = %s
                """, (int(conversation_id),))
                rows = cur.fetchall()
        return {row[0]: bool(row[1]) for row in rows}

    def get_stats(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get document statistics."""
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Total documents and size
                cur.execute("""
                    SELECT COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as total_size 
                    FROM documents WHERE NOT is_deleted
                """)
                row = cur.fetchone()
                total_documents = row["count"]
                total_size_bytes = row["total_size"]
                
                # Total chunks
                cur.execute("""
                    SELECT COUNT(*) as count 
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE NOT d.is_deleted
                """)
                chunk_row = cur.fetchone()
                total_chunks = chunk_row["count"]

                # Ingestion status counts
                cur.execute("""
                    SELECT ingestion_status, COUNT(*) as count
                    FROM documents
                    WHERE NOT is_deleted
                    GROUP BY ingestion_status
                """)
                status_rows = cur.fetchall()
                status_counts = {
                    "pending": 0,
                    "embedding": 0,
                    "embedded": 0,
                    "failed": 0,
                }
                for sr in status_rows:
                    key = sr["ingestion_status"]
                    if key in status_counts:
                        status_counts[key] = sr["count"]

                # By source type
                cur.execute("""
                    SELECT source_type, COUNT(*) as count 
                    FROM documents WHERE NOT is_deleted 
                    GROUP BY source_type
                """)
                type_rows = cur.fetchall()
                by_source_type = {r["source_type"]: {"total": r["count"], "enabled": r["count"]} for r in type_rows}

                # Last sync
                cur.execute("SELECT MAX(ingested_at) as last_sync FROM documents WHERE NOT is_deleted")
                last_row = cur.fetchone()
                last_sync = last_row["last_sync"].isoformat() if last_row and last_row["last_sync"] else None

                # Disabled count for conversation
                disabled_count = 0
                if conversation_id:
                    cur.execute("""
                        SELECT d.source_type, COUNT(*) as count
                        FROM conversation_doc_overrides o
                        JOIN documents d ON o.document_hash = d.resource_hash
                        WHERE o.conversation_id = %s AND NOT o.enabled AND NOT d.is_deleted
                        GROUP BY d.source_type
                    """, (int(conversation_id),))
                    for dr in cur.fetchall():
                        disabled_count += dr["count"]
                        if dr["source_type"] in by_source_type:
                            by_source_type[dr["source_type"]]["enabled"] -= dr["count"]

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "enabled_documents": total_documents - disabled_count,
            "disabled_documents": disabled_count,
            "total_size_bytes": total_size_bytes,
            "by_source_type": by_source_type,
            "status_counts": status_counts,
            "ingestion_in_progress": (status_counts["pending"] + status_counts["embedding"]) > 0,
            "last_sync": last_sync,
        }

    def list_documents(
        self,
        conversation_id: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        enabled_filter: Optional[str] = None,
        limit: Optional[int] = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List documents with optional filtering and pagination."""
        where_clauses = ["NOT d.is_deleted"]
        params: List[Any] = []

        join_clause = ""
        enabled_expr = "TRUE"
        if conversation_id:
            join_clause = (
                "LEFT JOIN conversation_doc_overrides o "
                "ON o.document_hash = d.resource_hash AND o.conversation_id = %s"
            )
            params.append(int(conversation_id))
            enabled_expr = "COALESCE(o.enabled, TRUE)"

        if source_type and source_type != "all":
            where_clauses.append("d.source_type = %s")
            params.append(source_type)

        if search:
            like = f"%{search}%"
            where_clauses.append("(d.display_name ILIKE %s OR d.url ILIKE %s)")
            params.extend([like, like])

        base_where_sql = " AND ".join(where_clauses)
        filtered_where_clauses = list(where_clauses)
        if enabled_filter == "enabled":
            filtered_where_clauses.append(enabled_expr)
        elif enabled_filter == "disabled":
            filtered_where_clauses.append(f"NOT {enabled_expr}")
        where_sql = " AND ".join(filtered_where_clauses)
        base_from = f"FROM documents d {join_clause}"

        query_params = list(params)
        count_sql = f"SELECT COUNT(*) as count {base_from} WHERE {where_sql}"

        enabled_count_sql = (
            f"SELECT COUNT(*) as count {base_from} "
            f"WHERE {base_where_sql} AND {enabled_expr}"
        )

        data_sql = (
            f"SELECT d.*, {enabled_expr} AS enabled "
            f"{base_from} "
            f"WHERE {where_sql} "
            "ORDER BY COALESCE(d.ingested_at, d.file_modified_at, d.created_at) DESC NULLS LAST, d.resource_hash ASC"
        )

        if limit is not None:
            data_sql += " LIMIT %s OFFSET %s"
            query_params.extend([limit, offset])

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(count_sql, params)
                total = cur.fetchone()["count"]

                cur.execute(enabled_count_sql, params)
                enabled_count = cur.fetchone()["count"]

                cur.execute(data_sql, query_params)
                rows = cur.fetchall()

        documents = []
        for row in rows:
            documents.append({
                "hash": row["resource_hash"],
                "display_name": row["display_name"],
                "source_type": row["source_type"],
                "url": row["url"],
                "size_bytes": row["size_bytes"],
                "suffix": row["suffix"],
                "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
                "ingestion_status": row.get("ingestion_status", "pending"),
                "ingestion_error": row.get("ingestion_error"),
                "enabled": bool(row.get("enabled", True)),
            })

        effective_limit = total if limit is None else limit
        has_more = False if limit is None else (offset + len(documents) < total)
        next_offset = None if not has_more else offset + len(documents)

        return {
            "documents": documents,
            "total": total,
            "enabled_count": enabled_count,
            "limit": effective_limit,
            "offset": offset,
            "has_more": has_more,
            "next_offset": next_offset,
        }

    def update_ingestion_status(
        self,
        resource_hash: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the ingestion status of a document.
        
        Args:
            resource_hash: The document's resource_hash
            status: One of 'pending', 'embedding', 'embedded', 'failed'
            error: Error message (only for 'failed' status)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                if status == "embedded":
                    cur.execute(
                        """UPDATE documents 
                           SET ingestion_status = %s, ingestion_error = NULL, indexed_at = NOW()
                           WHERE resource_hash = %s AND NOT is_deleted""",
                        (status, resource_hash),
                    )
                elif status == "failed":
                    cur.execute(
                        """UPDATE documents 
                           SET ingestion_status = %s, ingestion_error = %s
                           WHERE resource_hash = %s AND NOT is_deleted""",
                        (status, error, resource_hash),
                    )
                else:
                    cur.execute(
                        """UPDATE documents 
                           SET ingestion_status = %s, ingestion_error = NULL
                           WHERE resource_hash = %s AND NOT is_deleted""",
                        (status, resource_hash),
                    )
            conn.commit()

    def reset_failed_document(self, resource_hash: str) -> bool:
        """
        Reset a failed document back to 'pending' for retry.
        
        Args:
            resource_hash: The document's resource_hash
            
        Returns:
            True if a document was reset, False if not found or not in 'failed' state
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE documents 
                       SET ingestion_status = 'pending', ingestion_error = NULL
                       WHERE resource_hash = %s AND NOT is_deleted AND ingestion_status = 'failed'""",
                    (resource_hash,),
                )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def reset_all_failed_documents(self) -> int:
        """
        Reset all failed documents back to 'pending' for retry.
        
        Returns:
            Number of documents reset
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE documents 
                       SET ingestion_status = 'pending', ingestion_error = NULL
                       WHERE NOT is_deleted AND ingestion_status = 'failed'"""
                )
                count = cur.rowcount
            conn.commit()
        return count

    def list_documents_grouped(
        self,
        show_all: bool = False,
        expand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List documents grouped by source origin for the unified status section.
        
        Args:
            show_all: If False (default), only return groups with actionable (non-embedded) docs.
                      If True, return all groups.
            expand: Source group name to include full document list for.
            
        Returns:
            Dict with groups list and aggregate status counts
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get aggregate status counts
                cur.execute(
                    """SELECT ingestion_status, COUNT(*) as count 
                       FROM documents WHERE NOT is_deleted
                       GROUP BY ingestion_status"""
                )
                status_counts = {row["ingestion_status"]: row["count"] for row in cur.fetchall()}

                # Get groups with counts per status
                # Group by: domain for web, repo URL for git, 'Local files' for local_files, source_type otherwise
                cur.execute(
                    """SELECT 
                         CASE 
                           WHEN source_type = 'web' AND url IS NOT NULL THEN
                             regexp_replace(url, '^(https?://[^/]+).*', '\\1')
                           WHEN source_type = 'git' AND url IS NOT NULL THEN
                             regexp_replace(url, '^(https?://[^/]+/[^/]+/[^/]+).*', '\\1')
                           WHEN source_type = 'local_files' THEN 'Local files'
                           WHEN source_type = 'jira' THEN 'Jira'
                           ELSE COALESCE(source_type, 'Unknown')
                         END as source_name,
                         ingestion_status,
                         COUNT(*) as count
                       FROM documents WHERE NOT is_deleted
                       GROUP BY source_name, ingestion_status
                       ORDER BY source_name"""
                )
                group_rows = cur.fetchall()

                # Build group structures
                groups_map: Dict[str, Dict[str, Any]] = {}
                for row in group_rows:
                    name = row["source_name"]
                    if name not in groups_map:
                        groups_map[name] = {
                            "source_name": name,
                            "total": 0,
                            "pending": 0,
                            "embedding": 0,
                            "embedded": 0,
                            "failed": 0,
                            "documents": [],
                        }
                    groups_map[name][row["ingestion_status"]] = row["count"]
                    groups_map[name]["total"] += row["count"]

                # Filter to only actionable groups (with pending or failed) unless show_all
                groups = []
                for g in groups_map.values():
                    g["has_actionable"] = (g["pending"] + g["embedding"] + g["failed"]) > 0
                    if show_all or g["has_actionable"]:
                        groups.append(g)

                # Sort: groups with actionable items first, then alphabetical
                groups.sort(key=lambda g: (not g["has_actionable"], g["source_name"]))

                # If expand is specified, load documents for that group
                if expand:
                    for g in groups:
                        if g["source_name"] == expand:
                            # Build WHERE for this source group
                            source_where, source_params = self._source_group_where(expand)
                            # Only show actionable (non-embedded) docs in group expand
                            # Users can use "Show all documents" for full flat list
                            cur.execute(
                                f"""SELECT resource_hash, display_name, source_type, suffix, size_bytes,
                                           ingestion_status, ingestion_error, ingested_at, indexed_at, created_at
                                    FROM documents WHERE NOT is_deleted AND {source_where}
                                      AND ingestion_status != 'embedded'
                                    ORDER BY 
                                      CASE ingestion_status 
                                        WHEN 'failed' THEN 0 
                                        WHEN 'pending' THEN 1 
                                        WHEN 'embedding' THEN 2 
                                        ELSE 3 
                                      END,
                                      created_at DESC
                                    LIMIT 100""",
                                source_params,
                            )
                            g["documents"] = [self._row_to_doc(r) for r in cur.fetchall()]
                            break

        return {
            "groups": groups,
            "status_counts": {
                "pending": status_counts.get("pending", 0),
                "embedding": status_counts.get("embedding", 0),
                "embedded": status_counts.get("embedded", 0),
                "failed": status_counts.get("failed", 0),
            },
        }

    def _source_group_where(self, source_name: str) -> tuple:
        """Build WHERE clause fragment for a source group name."""
        if source_name == "Local files":
            return "source_type = %s", ["local_files"]
        elif source_name == "Jira":
            return "source_type = %s", ["jira"]
        elif source_name.startswith("http"):
            # Domain-based match for web, repo-based for git
            return (
                """(
                  (source_type = 'web' AND url LIKE %s) OR
                  (source_type = 'git' AND url LIKE %s)
                )""",
                [source_name + "%", source_name + "%"],
            )
        else:
            return "source_type = %s", [source_name]

    def _row_to_doc(self, row) -> Dict[str, Any]:
        """Convert a RealDictRow to a document dict."""
        return {
            "hash": row["resource_hash"],
            "display_name": row["display_name"],
            "source_type": row["source_type"],
            "suffix": row["suffix"],
            "size_bytes": row["size_bytes"],
            "ingestion_status": row["ingestion_status"],
            "ingestion_error": row["ingestion_error"],
            "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
            "indexed_at": row["indexed_at"].isoformat() if row["indexed_at"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    def list_documents_with_status(
        self,
        status_filter: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List documents with ingestion status info for the upload page.
        
        Args:
            status_filter: Filter by ingestion status ('pending', 'embedding', 'embedded', 'failed', or None for all)
            source_type: Filter by source type
            search: Search query for display_name
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            Dict with documents list, total count, and status counts
        """
        where_clauses = ["NOT is_deleted"]
        params: List[Any] = []

        if status_filter:
            where_clauses.append("ingestion_status = %s")
            params.append(status_filter)

        if source_type and source_type != "all":
            where_clauses.append("source_type = %s")
            params.append(source_type)

        if search:
            like = f"%{search}%"
            where_clauses.append("display_name ILIKE %s")
            params.append(like)

        where_sql = " AND ".join(where_clauses)

        with self._connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get total matching
                cur.execute(f"SELECT COUNT(*) as count FROM documents WHERE {where_sql}", params)
                total = cur.fetchone()["count"]

                # Get status counts (always unfiltered by status)
                status_where = ["NOT is_deleted"]
                status_params: List[Any] = []
                if source_type and source_type != "all":
                    status_where.append("source_type = %s")
                    status_params.append(source_type)
                if search:
                    status_where.append("display_name ILIKE %s")
                    status_params.append(f"%{search}%")
                
                cur.execute(
                    f"""SELECT ingestion_status, COUNT(*) as count 
                        FROM documents WHERE {" AND ".join(status_where)}
                        GROUP BY ingestion_status""",
                    status_params,
                )
                status_counts = {row["ingestion_status"]: row["count"] for row in cur.fetchall()}

                # Get paginated results
                cur.execute(
                    f"""SELECT resource_hash, display_name, source_type, suffix, size_bytes,
                               ingestion_status, ingestion_error, ingested_at, indexed_at, created_at
                        FROM documents WHERE {where_sql}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s""",
                    params + [limit, offset],
                )
                rows = cur.fetchall()

        documents = []
        for row in rows:
            documents.append({
                "hash": row["resource_hash"],
                "display_name": row["display_name"],
                "source_type": row["source_type"],
                "suffix": row["suffix"],
                "size_bytes": row["size_bytes"],
                "ingestion_status": row["ingestion_status"],
                "ingestion_error": row["ingestion_error"],
                "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
                "indexed_at": row["indexed_at"].isoformat() if row["indexed_at"] else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })

        return {
            "documents": documents,
            "total": total,
            "limit": limit,
            "offset": offset,
            "status_counts": {
                "pending": status_counts.get("pending", 0),
                "embedding": status_counts.get("embedding", 0),
                "embedded": status_counts.get("embedded", 0),
                "failed": status_counts.get("failed", 0),
            },
        }

    def get_document_chunks(self, document_hash: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document with their boundaries.
        
        Args:
            document_hash: The document's resource_hash
            
        Returns:
            List of chunk dicts with chunk_index, chunk_text, start_char, end_char
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT dc.chunk_index, dc.chunk_text, dc.start_char, dc.end_char
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.resource_hash = %s AND NOT d.is_deleted
                    ORDER BY dc.chunk_index
                """, (document_hash,))
                rows = cur.fetchall()
        
        chunks = []
        for row in rows:
            chunks.append({
                "index": row["chunk_index"],
                "text": row["chunk_text"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
            })
        
        return chunks
    def get_document_content(self, document_hash: str, max_size: int = 100000) -> Optional[Dict[str, Any]]:
        """Get document content for preview.
        
        Falls back to chunk-based or metadata-based preview when the
        original file is not available on disk (e.g. pending documents).
        """
        metadata = self.get_metadata_for_hash(document_hash)
        if metadata is None:
            return None

        display_name = metadata.get("display_name", document_hash)

        # Try reading from the file on disk first
        path = self.get_filepath_for_hash(document_hash)
        if path:
            suffix = metadata.get("suffix", path.suffix).lower()
            content_type_map = {
                ".md": "text/markdown", ".txt": "text/plain", ".py": "text/x-python",
                ".js": "text/javascript", ".html": "text/html", ".json": "application/json",
                ".yaml": "text/yaml", ".yml": "text/yaml", ".csv": "text/csv",
            }
            content_type = content_type_map.get(suffix, "text/plain")

            try:
                size_bytes = path.stat().st_size
                truncated = size_bytes > max_size
                read_size = min(size_bytes, max_size)

                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(read_size)

                return {
                    "hash": document_hash,
                    "display_name": display_name,
                    "content": content,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                    "truncated": truncated,
                }
            except Exception as e:
                logger.warning(f"Failed to read content for {document_hash}: {e}")
                # Fall through to chunk/metadata fallback

        # Fallback: reconstruct content from stored chunks
        chunks = self.get_document_chunks(document_hash)
        if chunks:
            content = "\n\n".join(c["text"] for c in chunks)
            if len(content) > max_size:
                content = content[:max_size]
                truncated = True
            else:
                truncated = False
            return {
                "hash": document_hash,
                "display_name": display_name,
                "content": content,
                "content_type": "text/plain",
                "size_bytes": len(content),
                "truncated": truncated,
                "source": "chunks",
            }

        # Fallback: metadata-only preview for pending documents
        status = metadata.get("ingestion_status", "pending")
        url = metadata.get("url", "")
        source_type = metadata.get("source_type", "unknown")
        lines = [f"# {display_name}", ""]
        if url:
            lines.append(f"**URL:** {url}")
        lines.append(f"**Source:** {source_type}")
        lines.append(f"**Status:** {status}")
        if status == "pending":
            lines.extend(["", "_Content will be available after documents are processed (embedded)._"])
        elif status == "failed":
            error = metadata.get("ingestion_error", "Unknown error")
            lines.extend(["", f"**Error:** {error}"])
        content = "\n".join(lines)
        return {
            "hash": document_hash,
            "display_name": display_name,
            "content": content,
            "content_type": "text/markdown",
            "size_bytes": 0,
            "truncated": False,
            "source": "metadata",
        }

    def _resolve_path(self, stored_path: str) -> Path:
        """Resolve a stored path to an absolute path."""
        path = Path(stored_path)
        if not path.is_absolute():
            path = (self.data_path / path).resolve()
        return path

    def _row_to_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a database row to metadata dict."""
        metadata: Dict[str, Any] = {}

        # Parse extra_json
        extra_json = row.get("extra_json")
        if extra_json:
            try:
                extra = json.loads(extra_json) if isinstance(extra_json, str) else extra_json
                if isinstance(extra, dict):
                    for key, value in extra.items():
                        if value is not None:
                            metadata[str(key)] = str(value)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to parse extra_json: %s", exc)

        # Map standard columns
        column_to_key = {v: k for k, v in _METADATA_COLUMN_MAP.items()}
        for col in ["display_name", "source_type", "url", "ticket_id", "suffix", 
                    "size_bytes", "original_path", "base_path", "relative_path",
                    "file_path", "file_modified_at", "ingested_at",
                    "ingestion_status", "ingestion_error", "resource_hash",
                    "git_repo"]:
            value = row.get(col)
            if value is None:
                continue
            # Use the metadata key name
            key = column_to_key.get(col, col)
            if key == "file_path":
                key = "path"
            elif key == "file_modified_at":
                key = "modified_at"
            
            if hasattr(value, 'isoformat'):
                metadata[key] = value.isoformat()
            else:
                metadata[key] = str(value)

        return metadata

    @classmethod
    def load_sources_catalog(
        cls,
        data_path: Path | str,
        pg_config: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Convenience helper that returns the resource index mapping with absolute paths.
        
        Args:
            data_path: Base data path for resolving relative paths
            pg_config: PostgreSQL connection configuration
            
        Returns:
            Dict mapping resource_hash to absolute file path
        """
        base_path = Path(data_path)
        
        conn = psycopg2.connect(**pg_config)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT resource_hash, file_path 
                    FROM documents 
                    WHERE NOT is_deleted
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        resolved: Dict[str, str] = {}
        for row in rows:
            stored_path = row["file_path"]
            path = Path(stored_path)
            if not path.is_absolute():
                path = (base_path / path).resolve()
            resolved[row["resource_hash"]] = str(path)
        return resolved


def _coerce_int(value: Optional[str]) -> Optional[int]:
    """Coerce a value to int or None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp string to datetime."""
    if not value:
        return None
    try:
        # Handle ISO format with or without timezone
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


def _build_extra_text(payload: Dict[str, str]) -> str:
    """Build searchable text from metadata payload."""
    parts: List[str] = []
    for key, value in payload.items():
        if value is None:
            continue
        value_str = str(value)
        parts.append(f"{key}:{value_str}")
        parts.append(value_str)
    return " ".join(parts)
