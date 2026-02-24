"""
DataViewerService - Orchestrates document viewing and per-chat selection.

This service provides the business logic layer for the Data Viewer UI,
coordinating between the CatalogService and the chat application.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataViewerService:
    """
    Service for managing document viewing and per-chat document selection.
    """

    def __init__(self, data_path: str | Path, pg_config: Dict[str, Any]):
        """
        Initialize the DataViewerService.

        Args:
            data_path: Path to the data directory containing the catalog.
            pg_config: PostgreSQL connection configuration.
        """
        self.data_path = Path(data_path)
        self.pg_config = pg_config
        self.catalog = PostgresCatalogService(data_path=self.data_path, pg_config=self.pg_config)
        logger.info(f"DataViewerService initialized with data_path: {self.data_path}")

    def list_documents(
        self,
        conversation_id: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        enabled_filter: Optional[str] = None,
        limit: Optional[int] = 100, # chat passes None (all docs); default 100 in case called in other contexts to prevent document overload...
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List documents with per-chat enabled state.

        Args:
            conversation_id: The conversation ID for per-chat state (optional for global listing)
            source_type: Filter by source type ("local", "web", "ticket", "all")
            search: Search query for display_name and url
            enabled_filter: Filter by enabled state ("all", "enabled", "disabled")
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            Dict with documents, total, enabled_count, limit, offset
        """
        return self.catalog.list_documents(
            conversation_id=conversation_id,
            source_type=source_type,
            search=search,
            enabled_filter=enabled_filter,
            limit=limit,
            offset=offset,
        )

    def get_document_content(
        self,
        document_hash: str,
        max_size: int = 100000,
    ) -> Optional[Dict[str, Any]]:
        """
        Get document content for preview.

        Args:
            document_hash: The document's SHA-256 hash
            max_size: Maximum content size to return (truncates if larger)

        Returns:
            Dict with hash, display_name, content, content_type, size_bytes, truncated
            or None if document not found
        """
        return self.catalog.get_document_content(document_hash, max_size)

    def enable_document(self, conversation_id: str, document_hash: str) -> Dict[str, Any]:
        """
        Enable a document for a conversation.

        Args:
            conversation_id: The conversation ID
            document_hash: The document's SHA-256 hash

        Returns:
            Dict with success, hash, enabled
        """
        self.catalog.set_document_enabled(conversation_id, document_hash, enabled=True)
        return {
            "success": True,
            "hash": document_hash,
            "enabled": True,
        }

    def disable_document(self, conversation_id: str, document_hash: str) -> Dict[str, Any]:
        """
        Disable a document for a conversation.

        Args:
            conversation_id: The conversation ID
            document_hash: The document's SHA-256 hash

        Returns:
            Dict with success, hash, enabled
        """
        self.catalog.set_document_enabled(conversation_id, document_hash, enabled=False)
        return {
            "success": True,
            "hash": document_hash,
            "enabled": False,
        }

    def bulk_enable(
        self,
        conversation_id: str,
        document_hashes: List[str],
    ) -> Dict[str, Any]:
        """
        Enable multiple documents for a conversation.

        Args:
            conversation_id: The conversation ID
            document_hashes: List of document hashes to enable

        Returns:
            Dict with success, enabled_count
        """
        count = self.catalog.bulk_set_enabled(conversation_id, document_hashes, enabled=True)
        return {
            "success": True,
            "enabled_count": count,
        }

    def bulk_disable(
        self,
        conversation_id: str,
        document_hashes: List[str],
    ) -> Dict[str, Any]:
        """
        Disable multiple documents for a conversation.

        Args:
            conversation_id: The conversation ID
            document_hashes: List of document hashes to disable

        Returns:
            Dict with success, disabled_count
        """
        count = self.catalog.bulk_set_enabled(conversation_id, document_hashes, enabled=False)
        return {
            "success": True,
            "disabled_count": count,
        }

    def get_stats(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for the data viewer.

        Args:
            conversation_id: Optional. The conversation ID for per-chat stats.
                            If None, shows all documents as enabled.

        Returns:
            Dict with total_documents, enabled_documents, disabled_documents,
            total_size_bytes, by_source_type, last_sync
        """
        return self.catalog.get_stats(conversation_id)

    def get_disabled_hashes(self, conversation_id: str) -> Set[str]:
        """
        Get the set of disabled document hashes for a conversation.
        Used for filtering retrieval results.

        Args:
            conversation_id: The conversation ID

        Returns:
            Set of disabled document hashes
        """
        return self.catalog.get_disabled_hashes(conversation_id)

    def is_document_enabled(self, conversation_id: str, document_hash: str) -> bool:
        """
        Check if a document is enabled for a conversation.

        Args:
            conversation_id: The conversation ID
            document_hash: The document's SHA-256 hash

        Returns:
            True if enabled (default), False if disabled
        """
        return self.catalog.is_document_enabled(conversation_id, document_hash)

    def get_document_chunks(self, document_hash: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document with their boundaries.
        
        Args:
            document_hash: The document's resource_hash
            
        Returns:
            List of chunk dicts with index, text, start_char, end_char
        """
        return self.catalog.get_document_chunks(document_hash)
