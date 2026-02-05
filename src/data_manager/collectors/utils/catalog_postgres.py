"""
PostgreSQL-backed CatalogService.

Replaces SQLite-based catalog with PostgreSQL 'documents' table.
Provides the same interface as the original CatalogService.
"""
from __future__ import annotations

import json
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

    @contextmanager
    def _connect(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Context manager for database connections."""
        conn = psycopg2.connect(**self.pg_config)
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
                        extra_json,
                        extra_text,
                        is_deleted
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
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
            "last_sync": last_sync,
        }

    def list_documents(
        self,
        conversation_id: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        enabled_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List documents with optional filtering."""
        selection_state = self.get_selection_state(conversation_id) if conversation_id else {}

        where_clauses = ["NOT is_deleted"]
        params: List[Any] = []

        if source_type and source_type != "all":
            where_clauses.append("source_type = %s")
            params.append(source_type)

        if search:
            like = f"%{search}%"
            where_clauses.append("(display_name ILIKE %s OR url ILIKE %s)")
            params.extend([like, like])

        sql = f"""
            SELECT * FROM documents
            WHERE {" AND ".join(where_clauses)}
            ORDER BY COALESCE(ingested_at, file_modified_at, created_at) DESC NULLS LAST
        """

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get total
                count_sql = f"SELECT COUNT(*) as count FROM documents WHERE {' AND '.join(where_clauses)}"
                cur.execute(count_sql, params)
                total = cur.fetchone()["count"]

                cur.execute(sql, params)
                rows = cur.fetchall()

        all_docs = []
        enabled_count = 0
        for row in rows:
            doc_hash = row["resource_hash"]
            is_enabled = selection_state.get(doc_hash, True)
            if is_enabled:
                enabled_count += 1

            if enabled_filter == "enabled" and not is_enabled:
                continue
            if enabled_filter == "disabled" and is_enabled:
                continue

            all_docs.append({
                "hash": doc_hash,
                "display_name": row["display_name"],
                "source_type": row["source_type"],
                "url": row["url"],
                "size_bytes": row["size_bytes"],
                "suffix": row["suffix"],
                "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
                "enabled": is_enabled,
            })

        paginated = all_docs[offset:offset + limit]

        return {
            "documents": paginated,
            "total": len(all_docs) if enabled_filter else total,
            "enabled_count": enabled_count,
            "limit": limit,
            "offset": offset,
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
        """Get document content for preview."""
        path = self.get_filepath_for_hash(document_hash)
        if not path:
            return None

        metadata = self.get_metadata_for_hash(document_hash)
        if not metadata:
            return None

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
        except Exception as e:
            logger.warning(f"Failed to read content for {document_hash}: {e}")
            return None

        return {
            "hash": document_hash,
            "display_name": metadata.get("display_name", document_hash),
            "content": content,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "truncated": truncated,
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
                    "file_path", "file_modified_at", "ingested_at"]:
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
