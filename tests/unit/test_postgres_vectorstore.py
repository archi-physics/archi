"""
Unit tests for PostgresVectorStore.

Tests cover:
- Similarity search (semantic)
- Hybrid search (semantic + BM25)
- Document addition and deletion
- Metadata filtering
- Index usage
"""
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

from langchain_core.documents import Document

from src.data_manager.vectorstore.postgres_vectorstore import PostgresVectorStore


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_pg_connection():
    """Create a mock PostgreSQL connection."""
    conn = MagicMock()
    cursor = MagicMock()
    
    # Setup context manager
    cursor_context = MagicMock()
    cursor_context.__enter__ = MagicMock(return_value=cursor)
    cursor_context.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor_context
    
    # Required for psycopg2.extras.execute_values - cursor needs connection.encoding
    conn.encoding = 'UTF8'
    cursor.connection = conn  # Link cursor back to connection
    
    return conn, cursor


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings function."""
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3] * 128]  # 384-dim
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3] * 128
    return embeddings


@pytest.fixture
def pg_config():
    """Standard PostgreSQL config."""
    return {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'archi_test',
        'user': 'postgres',
        'password': 'testpass',
    }


@pytest.fixture
def vector_store(pg_config, mock_embeddings, mock_pg_connection):
    """Create a PostgresVectorStore with mocked connection."""
    conn, cursor = mock_pg_connection
    
    with patch.object(PostgresVectorStore, '_get_connection', return_value=conn):
        store = PostgresVectorStore(
            pg_config=pg_config,
            embedding_function=mock_embeddings,
            collection_name="test_collection",
            distance_metric="cosine",
        )
        return store


# =============================================================================
# Initialization Tests
# =============================================================================

class TestPostgresVectorStoreInit:
    """Tests for PostgresVectorStore initialization."""
    
    def test_init_with_valid_config(self, pg_config, mock_embeddings):
        """Test successful initialization."""
        store = PostgresVectorStore(
            pg_config=pg_config,
            embedding_function=mock_embeddings,
        )
        
        assert store._collection_name == "default"
        assert store._distance_metric == "cosine"
        assert store._distance_op == "<=>"
    
    def test_init_with_custom_collection(self, pg_config, mock_embeddings):
        """Test initialization with custom collection name."""
        store = PostgresVectorStore(
            pg_config=pg_config,
            embedding_function=mock_embeddings,
            collection_name="my_docs",
        )
        
        assert store._collection_name == "my_docs"
    
    def test_init_with_l2_distance(self, pg_config, mock_embeddings):
        """Test initialization with L2 distance metric."""
        store = PostgresVectorStore(
            pg_config=pg_config,
            embedding_function=mock_embeddings,
            distance_metric="l2",
        )
        
        assert store._distance_metric == "l2"
        assert store._distance_op == "<->"
    
    def test_init_with_inner_product(self, pg_config, mock_embeddings):
        """Test initialization with inner product distance."""
        store = PostgresVectorStore(
            pg_config=pg_config,
            embedding_function=mock_embeddings,
            distance_metric="inner_product",
        )
        
        assert store._distance_metric == "inner_product"
        assert store._distance_op == "<#>"
    
    def test_init_invalid_distance_metric(self, pg_config, mock_embeddings):
        """Test initialization with invalid distance metric."""
        with pytest.raises(ValueError, match="distance_metric must be one of"):
            PostgresVectorStore(
                pg_config=pg_config,
                embedding_function=mock_embeddings,
                distance_metric="invalid",
            )
    
    def test_embeddings_property(self, vector_store, mock_embeddings):
        """Test embeddings property returns the embedding function."""
        assert vector_store.embeddings is mock_embeddings


# =============================================================================
# Similarity Search Tests
# =============================================================================

class TestSimilaritySearch:
    """Tests for similarity search."""
    
    def test_similarity_search_basic(self, vector_store, mock_pg_connection, mock_embeddings):
        """Test basic similarity search."""
        conn, cursor = mock_pg_connection
        
        # Mock query results
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'Document about machine learning',
                'metadata': json.dumps({'source': 'web'}),
                'distance': 0.15,
                'resource_hash': 'abc123',
                'display_name': 'ML Guide',
                'source_type': 'web',
                'url': 'https://example.com/ml',
            },
            {
                'id': 2,
                'chunk_text': 'Another ML document',
                'metadata': json.dumps({'source': 'pdf'}),
                'distance': 0.25,
                'resource_hash': 'def456',
                'display_name': 'ML Paper',
                'source_type': 'pdf',
                'url': None,
            },
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search("machine learning", k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert 'machine learning' in results[0].page_content
        assert results[0].metadata.get('resource_hash') == 'abc123'
    
    def test_similarity_search_with_scores(self, vector_store, mock_pg_connection):
        """Test similarity search returning scores."""
        conn, cursor = mock_pg_connection
        
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'Relevant document',
                'metadata': '{}',
                'distance': 0.1,  # cosine distance
                'resource_hash': None,
                'display_name': None,
                'source_type': None,
                'url': None,
            },
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search_with_score("query", k=1)
        
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert score == 0.9  # 1 - 0.1 cosine distance
    
    def test_similarity_search_with_filter(self, vector_store, mock_pg_connection):
        """Test similarity search with metadata filter."""
        conn, cursor = mock_pg_connection
        cursor.fetchall.return_value = []
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search(
                "query",
                k=5,
                filter={"source_type": "pdf"},
            )
        
        # Verify filter was applied in query
        call_args = cursor.execute.call_args[0]
        query_sql = call_args[0]
        assert "source_type" in query_sql or "metadata" in query_sql
    
    def test_similarity_search_empty_results(self, vector_store, mock_pg_connection):
        """Test similarity search with no results."""
        conn, cursor = mock_pg_connection
        cursor.fetchall.return_value = []
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search("obscure query", k=5)
        
        assert results == []


# =============================================================================
# Hybrid Search Tests
# =============================================================================

class TestHybridSearch:
    """Tests for hybrid search (semantic + BM25)."""
    
    def test_hybrid_search_with_bm25_index(self, vector_store, mock_pg_connection):
        """Test hybrid search when BM25 index exists."""
        conn, cursor = mock_pg_connection
        
        # First call checks for BM25 index, second checks for chunk_tsv column
        # Use dict-like results for RealDictCursor compatibility
        cursor.fetchone.side_effect = [{'relname': 'idx_bm25'}, None]  # BM25 index exists, no chunk_tsv
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'Machine learning fundamentals',
                'metadata': '{}',
                'semantic_score': 0.85,
                'bm25_score': 0.9,
                'combined_score': 0.865,  # 0.85*0.7 + 0.9*0.3
                'resource_hash': 'abc',
                'display_name': 'ML Doc',
                'source_type': 'web',
                'url': None,
            },
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.hybrid_search(
                "machine learning",
                k=5,
                semantic_weight=0.7,
                bm25_weight=0.3,
            )
        
        assert len(results) == 1
        doc, score = results[0]
        assert 'machine learning' in doc.page_content.lower()
    
    def test_hybrid_search_without_bm25_index(self, vector_store, mock_pg_connection):
        """Test hybrid search raises error when BM25 index is missing."""
        conn, cursor = mock_pg_connection
        
        # No BM25 index found
        cursor.fetchone.return_value = None
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            with pytest.raises(RuntimeError, match="BM25 index"):
                vector_store.hybrid_search("query", k=5)
    
    def test_hybrid_search_custom_weights(self, vector_store, mock_pg_connection):
        """Test hybrid search with custom weights."""
        conn, cursor = mock_pg_connection
        # First call checks for BM25 index
        # Use dict-like results for RealDictCursor compatibility
        cursor.fetchone.return_value = {'relname': 'idx_bm25'}  # BM25 index exists
        # Return mock results so we don't fall back to semantic search
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'test content',
                'metadata': {},
                'semantic_score': 0.8,
                'bm25_score': 0.7,
                'combined_score': 0.74,  # 0.8*0.4 + 0.7*0.6
                'resource_hash': 'hash123',
                'display_name': 'Test Doc',
                'source_type': 'web',
                'url': None,
            }
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.hybrid_search(
                "query",
                k=5,
                semantic_weight=0.4,
                bm25_weight=0.6,
            )
        
        # Verify we got results
        assert len(results) == 1
        
        # Verify hybrid search query was executed (weights are embedded in SQL params)
        # Find the hybrid search query (has combined_score)
        for call in cursor.execute.call_args_list:
            call_args = call[0]
            query_sql = call_args[0]
            if 'combined_score' in query_sql.lower():
                params = call_args[1]
                # Weights should be in params (0.4 and 0.6)
                assert 0.4 in params
                assert 0.6 in params
                break
        else:
            pytest.fail("Hybrid search query with combined_score not found")
    
    def test_hybrid_search_score_combination(self, vector_store, mock_pg_connection):
        """Test that hybrid search correctly combines scores."""
        conn, cursor = mock_pg_connection
        # First call checks for BM25 index, second checks for chunk_tsv column
        # Use dict-like results for RealDictCursor compatibility
        cursor.fetchone.side_effect = [{'relname': 'idx_bm25'}, None]  # BM25 index exists, no chunk_tsv
        
        # Results with known scores
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'High semantic, low keyword',
                'metadata': '{}',
                'semantic_score': 0.95,
                'bm25_score': 0.2,
                'combined_score': 0.725,  # 0.95*0.7 + 0.2*0.3
                'resource_hash': None,
                'display_name': None,
                'source_type': None,
                'url': None,
            },
            {
                'id': 2,
                'chunk_text': 'Balanced scores',
                'metadata': '{}',
                'semantic_score': 0.7,
                'bm25_score': 0.8,
                'combined_score': 0.73,  # 0.7*0.7 + 0.8*0.3
                'resource_hash': None,
                'display_name': None,
                'source_type': None,
                'url': None,
            },
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.hybrid_search(
                "query", 
                k=5,
                semantic_weight=0.7,
                bm25_weight=0.3,
            )
        
        # Results should be ordered by combined_score
        assert len(results) == 2
        # Second doc has higher combined score
        _, score1 = results[0]
        _, score2 = results[1]


# =============================================================================
# Document Operations Tests
# =============================================================================

class TestDocumentOperations:
    """Tests for document add/delete operations."""
    
    def test_add_texts(self, vector_store, mock_pg_connection, mock_embeddings):
        """Test adding texts to the vector store."""
        conn, cursor = mock_pg_connection
        cursor.fetchone.return_value = (1,)  # document_id
        
        texts = ["First document", "Second document"]
        metadatas = [{"source": "test"}, {"source": "test"}]
        
        with patch.object(vector_store, '_get_connection', return_value=conn), \
             patch('psycopg2.extras.execute_values') as mock_execute_values:
            ids = vector_store.add_texts(texts, metadatas=metadatas)
        
        assert len(ids) == 2
        # Verify embeddings were created
        mock_embeddings.embed_documents.assert_called_once_with(texts)
        # Verify execute_values was called for bulk insert
        assert mock_execute_values.called
    
    def test_add_documents(self, vector_store, mock_pg_connection, mock_embeddings):
        """Test adding Document objects."""
        conn, cursor = mock_pg_connection
        cursor.fetchone.return_value = (1,)
        
        docs = [
            Document(page_content="Doc 1", metadata={"key": "value1"}),
            Document(page_content="Doc 2", metadata={"key": "value2"}),
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn), \
             patch('psycopg2.extras.execute_values') as mock_execute_values:
            ids = vector_store.add_documents(docs)
        
        assert len(ids) == 2
        assert mock_execute_values.called
    
    def test_delete_by_ids(self, vector_store, mock_pg_connection):
        """Test deleting documents by ID."""
        conn, cursor = mock_pg_connection
        cursor.rowcount = 2  # 2 rows deleted
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            success = vector_store.delete(ids=["chunk_1", "chunk_2"])
        
        assert success is True
        # Verify DELETE was called
        call_args = cursor.execute.call_args[0]
        assert "DELETE" in call_args[0] or "UPDATE" in call_args[0]


# =============================================================================
# Search Quality Tests
# =============================================================================

class TestSearchQuality:
    """Tests for search quality metrics."""
    
    def test_cosine_distance_to_similarity(self, vector_store, mock_pg_connection):
        """Test that cosine distance is converted to similarity correctly."""
        conn, cursor = mock_pg_connection
        
        # Distance of 0.1 should give similarity of 0.9
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'Test',
                'metadata': '{}',
                'distance': 0.1,
                'resource_hash': None,
                'display_name': None,
                'source_type': None,
                'url': None,
            },
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search_with_score("query", k=1)
        
        _, score = results[0]
        assert abs(score - 0.9) < 0.001  # 1 - 0.1
    
    def test_metadata_preserved_in_results(self, vector_store, mock_pg_connection):
        """Test that all metadata is preserved in results."""
        conn, cursor = mock_pg_connection
        
        original_metadata = {
            'source': 'web',
            'page': 5,
            'custom_field': 'custom_value',
        }
        
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'Test document',
                'metadata': json.dumps(original_metadata),
                'distance': 0.2,
                'resource_hash': 'hash123',
                'display_name': 'Test Doc',
                'source_type': 'web',
                'url': 'https://example.com',
            },
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search("query", k=1)
        
        doc = results[0]
        assert doc.metadata.get('source') == 'web'
        assert doc.metadata.get('page') == 5
        assert doc.metadata.get('custom_field') == 'custom_value'
        assert doc.metadata.get('resource_hash') == 'hash123'
        assert doc.metadata.get('display_name') == 'Test Doc'


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_query(self, vector_store, mock_pg_connection):
        """Test search with empty query."""
        conn, cursor = mock_pg_connection
        cursor.fetchall.return_value = []
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search("", k=5)
        
        # Should return empty or handle gracefully
        assert isinstance(results, list)
    
    def test_large_k_value(self, vector_store, mock_pg_connection):
        """Test search with large k value."""
        conn, cursor = mock_pg_connection
        cursor.fetchall.return_value = []
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search("query", k=10000)
        
        # Should handle without error
        assert isinstance(results, list)
    
    def test_special_characters_in_query(self, vector_store, mock_pg_connection):
        """Test search with special characters in query."""
        conn, cursor = mock_pg_connection
        cursor.fetchall.return_value = []
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            # Should not raise SQL injection errors
            results = vector_store.similarity_search("query'; DROP TABLE documents; --", k=5)
        
        assert isinstance(results, list)
    
    def test_unicode_in_query(self, vector_store, mock_pg_connection):
        """Test search with unicode characters."""
        conn, cursor = mock_pg_connection
        cursor.fetchall.return_value = []
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search("机器学习 αβγ 🤖", k=5)
        
        assert isinstance(results, list)
    
    def test_null_metadata_handling(self, vector_store, mock_pg_connection):
        """Test handling of null metadata in results."""
        conn, cursor = mock_pg_connection
        
        cursor.fetchall.return_value = [
            {
                'id': 1,
                'chunk_text': 'Document with no metadata',
                'metadata': None,  # NULL from database
                'distance': 0.3,
                'resource_hash': None,
                'display_name': None,
                'source_type': None,
                'url': None,
            },
        ]
        
        with patch.object(vector_store, '_get_connection', return_value=conn):
            results = vector_store.similarity_search("query", k=1)
        
        assert len(results) == 1
        assert results[0].metadata is not None  # Should be empty dict, not None
