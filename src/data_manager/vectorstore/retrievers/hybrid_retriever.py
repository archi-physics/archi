"""
Hybrid retriever combining BM25 full-text search and semantic vector search.

Uses PostgreSQL-native hybrid search (ts_rank or pg_textsearch BM25) when available,
falling back to semantic-only search for other vectorstore backends.
"""

from typing import List, Tuple

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStore

from src.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever using Postgres-native BM25 + semantic vector search.
    
    Delegates to PostgresVectorStore.hybrid_search() which combines:
    - Semantic similarity via pgvector cosine distance
    - BM25/full-text search via ts_rank (GIN index) or pg_textsearch
    
    For non-Postgres vectorstores, falls back to semantic-only search.
    """
    
    vectorstore: VectorStore
    k: int = 5
    bm25_weight: float = 0.5
    semantic_weight: float = 0.5
    
    def __init__(
        self,
        vectorstore: VectorStore,
        k: int = 5,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        **kwargs,
    ):
        
        super().__init__(
            vectorstore=vectorstore,
            k=k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            **kwargs,
        )
        self.k = k
        
        # Check if vectorstore supports native hybrid search
        self._has_hybrid = hasattr(vectorstore, 'hybrid_search')
        if self._has_hybrid:
            logger.info("HybridRetriever using Postgres-native hybrid search")
        else:
            logger.warning(
                "Vectorstore does not support hybrid_search(); "
                "falling back to semantic-only search"
            )
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using hybrid search (BM25 + semantic).
        
        Returns:
            List of (Document, score) tuples ordered by combined score.
        """
        logger.debug("HybridRetriever query: %s", query[:100])
        
        if self._has_hybrid:
            # Use Postgres-native hybrid search
            logger.debug(
                "Using Postgres hybrid search: k=%d, semantic_weight=%.2f, bm25_weight=%.2f",
                self.k, self.semantic_weight, self.bm25_weight,
            )
            try:
                results = self.vectorstore.hybrid_search(
                    query=query,
                    k=self.k,
                    semantic_weight=self.semantic_weight,
                    bm25_weight=self.bm25_weight,
                )
                logger.debug("Hybrid search returned %d documents", len(results))
                return results
            except RuntimeError as exc:
                logger.warning("Hybrid search failed, falling back to semantic-only: %s", exc)
        
        # Fallback: semantic-only search
        logger.debug("Falling back to semantic-only search (k=%d)", self.k)
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        logger.debug("Semantic search returned %d documents", len(results))
        return results
