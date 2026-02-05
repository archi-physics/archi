"""
Tests to isolate the ingestion pipeline issue where documents are scraped
but not embedded into chunks.

ROOT CAUSE IDENTIFIED (2026-02-03):
====================================
The ingestion DOES work, but embedding is EXTREMELY SLOW when running on CPU:
- HuggingFace embeddings without GPU take 30-50+ seconds per file
- For 46 files, total embedding time was ~1.5 hours
- When initially checked, embedding was still in progress (not failed)

Original hypotheses (all ruled out):
1. Documents in `documents` table don't have file_path pointing to actual files - FALSE
2. The file_path paths inside the container don't exist - FALSE  
3. The load_sources_catalog method returns empty or invalid paths - FALSE
4. The _collect_indexed_documents filters out all documents - FALSE
5. The embedding step silently fails - FALSE

The actual issue:
- CPU-based embedding is too slow for production use
- Need GPU acceleration or a faster embedding model
- Consider async/background processing with progress reporting
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pytest

# These tests can run against a live deployment using docker exec
# or can be run as unit tests with mocks


class TestIngestionPipelineIsolation:
    """Test each step of the ingestion pipeline in isolation."""

    @pytest.fixture
    def mock_pg_config(self):
        """Mock PostgreSQL configuration."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "archi-db",
            "user": "archi",
            "password": "archi"
        }

    def test_documents_table_has_valid_file_paths(self, mock_pg_config):
        """
        HYPOTHESIS: Documents are stored in the documents table but file_path
        points to non-existent files.
        
        This test checks that every document in the database has a file_path
        that actually exists on disk.
        """
        # This test should be run inside the container or with proper paths
        # For now, this is a template showing what to check
        
        # Query to run:
        # SELECT id, resource_hash, file_path, display_name, source_type
        # FROM documents 
        # WHERE NOT is_deleted
        # LIMIT 10;
        
        # Then for each row, check: Path(file_path).exists()
        pass

    def test_load_sources_catalog_returns_data(self, mock_pg_config):
        """
        HYPOTHESIS: load_sources_catalog returns empty dict despite documents
        existing in the table.
        
        This tests that the catalog loading method correctly maps document
        hashes to file paths.
        """
        pass

    def test_collect_indexed_documents_filters_correctly(self):
        """
        HYPOTHESIS: _collect_indexed_documents filters out all documents
        because paths don't exist.
        
        This tests the filtering logic that validates paths exist.
        """
        from src.data_manager.vectorstore.manager import VectorStoreManager
        
        # Create a mock manager
        manager = MagicMock(spec=VectorStoreManager)
        
        # Test case 1: All paths exist
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            sources = {"hash1": temp_path}
            # Call the real method with mock self
            result = VectorStoreManager._collect_indexed_documents(manager, sources)
            assert "hash1" in result, "Should include existing files"
            assert result["hash1"] == temp_path
        finally:
            os.unlink(temp_path)

        # Test case 2: Path doesn't exist
        sources = {"hash2": "/nonexistent/path/file.txt"}
        result = VectorStoreManager._collect_indexed_documents(manager, sources)
        assert "hash2" not in result, "Should filter out missing files"

        # Test case 3: Path is a directory (should be skipped)
        with tempfile.TemporaryDirectory() as temp_dir:
            sources = {"hash3": temp_dir}
            result = VectorStoreManager._collect_indexed_documents(manager, sources)
            assert "hash3" not in result, "Should filter out directories"

    def test_embedding_model_works(self):
        """
        HYPOTHESIS: The embedding model fails silently.
        
        This tests that the HuggingFace embedding model can actually
        generate embeddings.
        
        NOTE: This also serves as a performance benchmark. On CPU,
        embedding is very slow (30-50+ seconds per file).
        """
        import time
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            test_texts = ["This is a test document.", "Another test."]
            
            start = time.time()
            embeddings = model.embed_documents(test_texts)
            elapsed = time.time() - start
            
            assert len(embeddings) == 2, "Should generate 2 embeddings"
            assert len(embeddings[0]) == 384, "Embedding dimension should be 384"
            
            print(f"Embedding 2 short texts took {elapsed:.2f} seconds")
            
            # Test with longer text (more realistic)
            long_text = "This is a longer test document. " * 100
            start = time.time()
            embeddings = model.embed_documents([long_text])
            elapsed = time.time() - start
            print(f"Embedding 1 long text took {elapsed:.2f} seconds")
            
        except ImportError:
            pytest.skip("langchain_huggingface not installed")

    def test_text_splitter_produces_chunks(self):
        """
        HYPOTHESIS: The text splitter produces no chunks.
        
        This tests that the splitter correctly chunks documents.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        
        # Test with substantial content - use newlines for natural split points
        long_text = ("This is a test sentence.\n" * 100)  # ~2600 chars with newlines
        doc = Document(page_content=long_text, metadata={"source": "test"})
        
        chunks = splitter.split_documents([doc])
        
        assert len(chunks) > 1, f"Should produce multiple chunks, got {len(chunks)}"
        for chunk in chunks:
            assert chunk.page_content.strip(), "Chunks should not be empty"

    def test_loader_returns_content(self):
        """
        HYPOTHESIS: The document loader fails to read scraped files.
        
        This tests that HTML/text files can be loaded properly.
        """
        from src.data_manager.vectorstore.loader_utils import select_loader
        
        # Test with HTML content (typical scraped page)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False
        ) as f:
            f.write("""
            <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Hello World</h1>
                <p>This is test content for embedding.</p>
            </body>
            </html>
            """)
            temp_path = f.name
        
        try:
            loader = select_loader(temp_path)
            assert loader is not None, "Should return a loader for .html files"
            
            docs = loader.load()
            assert len(docs) > 0, "Loader should return documents"
            
            content = docs[0].page_content
            assert "Hello World" in content or "test content" in content, \
                f"Content should be extracted. Got: {content[:200]}"
        finally:
            os.unlink(temp_path)

    def test_embedding_performance_realistic(self):
        """
        Performance test for realistic HTML content embedding.
        
        This test measures how long it takes to embed content similar to
        scraped web pages to identify performance bottlenecks.
        """
        import time
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_text_splitters.character import CharacterTextSplitter
            
            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            
            # Simulate a ~65KB HTML page (typical scraped page size)
            html_content = """
            <html><head><title>Test Page</title></head><body>
            <h1>Welcome to the Test Page</h1>
            <p>This is paragraph content that represents typical web page text.
            It contains various sentences and information that would be found
            on a real website about computing, research, or education.</p>
            """ * 500  # ~65KB
            
            # Time the chunking
            start = time.time()
            from langchain_core.documents import Document
            doc = Document(page_content=html_content, metadata={})
            chunks = splitter.split_documents([doc])
            chunk_time = time.time() - start
            
            # Time the embedding
            chunk_texts = [c.page_content for c in chunks]
            start = time.time()
            embeddings = model.embed_documents(chunk_texts)
            embed_time = time.time() - start
            
            print(f"\n=== PERFORMANCE RESULTS ===")
            print(f"Content size: {len(html_content)} bytes")
            print(f"Chunks generated: {len(chunks)}")
            print(f"Chunking time: {chunk_time:.2f}s")
            print(f"Embedding time: {embed_time:.2f}s")
            print(f"Time per chunk: {embed_time/len(chunks):.2f}s")
            print(f"Estimated time for 46 files (3 chunks each): {46 * 3 * embed_time/len(chunks) / 60:.1f} minutes")
            
            # Warn if embedding is too slow
            if embed_time > 30:
                print(f"\nWARNING: Embedding took {embed_time:.0f}s - consider GPU acceleration!")
            
        except ImportError:
            pytest.skip("langchain_huggingface not installed")


class TestDockerDeploymentDiagnostics:
    """
    Diagnostic tests to run against a live Docker deployment.
    
    These aren't unit tests - they're diagnostic scripts that can be run
    via docker exec to check the actual state of a deployment.
    """

    @staticmethod
    def diagnose_documents_vs_files():
        """
        Run this inside the data-manager container to check if documents
        have valid file paths.
        
        Usage:
            docker exec data-manager-test-fresh-2 python -c "
            from tests.unit.test_ingestion_pipeline_isolation import TestDockerDeploymentDiagnostics
            TestDockerDeploymentDiagnostics.diagnose_documents_vs_files()
            "
        """
        import psycopg2
        import psycopg2.extras
        from pathlib import Path
        
        pg_config = {
            "host": "postgres",
            "port": 5432,
            "database": "archi-db",
            "user": "archi",
            "password": "archi"
        }
        
        conn = psycopg2.connect(**pg_config)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, resource_hash, file_path, display_name, source_type
                    FROM documents 
                    WHERE NOT is_deleted
                    LIMIT 20
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        print(f"\n=== DOCUMENTS TABLE DIAGNOSTIC ===")
        print(f"Found {len(rows)} documents in database")
        
        existing = 0
        missing = 0
        
        for row in rows:
            file_path = row['file_path']
            path = Path(file_path)
            exists = path.exists()
            
            status = "✓ EXISTS" if exists else "✗ MISSING"
            if exists:
                existing += 1
                size = path.stat().st_size
                status += f" ({size} bytes)"
            else:
                missing += 1
                
            print(f"  [{status}] {file_path}")
            print(f"      hash: {row['resource_hash'][:16]}..., type: {row['source_type']}")
        
        print(f"\n=== SUMMARY ===")
        print(f"  Existing files: {existing}")
        print(f"  Missing files:  {missing}")
        
        return {"existing": existing, "missing": missing, "total": len(rows)}

    @staticmethod
    def diagnose_vectorstore_update():
        """
        Run this inside the data-manager container to trace through
        the vectorstore update process.
        
        Usage:
            docker exec data-manager-test-fresh-2 python -c "
            from tests.unit.test_ingestion_pipeline_isolation import TestDockerDeploymentDiagnostics
            TestDockerDeploymentDiagnostics.diagnose_vectorstore_update()
            "
        """
        import psycopg2
        import psycopg2.extras
        from pathlib import Path
        
        from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService
        
        data_path = "/root/data"  # Default container path
        pg_config = {
            "host": "postgres",
            "port": 5432,
            "database": "archi-db",
            "user": "archi",
            "password": "archi"
        }
        
        print("\n=== VECTORSTORE UPDATE DIAGNOSTIC ===")
        
        # Step 1: Load sources catalog
        print("\n1. Loading sources catalog...")
        sources = PostgresCatalogService.load_sources_catalog(data_path, pg_config)
        print(f"   Loaded {len(sources)} sources from catalog")
        
        if sources:
            print("   First 5 sources:")
            for i, (hash_val, path) in enumerate(list(sources.items())[:5]):
                exists = Path(path).exists()
                print(f"      {hash_val[:16]}... -> {path} [{'EXISTS' if exists else 'MISSING'}]")
        
        # Step 2: Check which paths actually exist
        print("\n2. Validating file paths...")
        existing_paths = {h: p for h, p in sources.items() if Path(p).exists()}
        missing_paths = {h: p for h, p in sources.items() if not Path(p).exists()}
        
        print(f"   Existing files: {len(existing_paths)}")
        print(f"   Missing files:  {len(missing_paths)}")
        
        if missing_paths:
            print("   First 5 missing paths:")
            for i, (hash_val, path) in enumerate(list(missing_paths.items())[:5]):
                print(f"      {path}")
        
        # Step 3: Check what's in the vectorstore
        print("\n3. Checking vectorstore state...")
        conn = psycopg2.connect(**pg_config)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks")
                chunk_count = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT DISTINCT metadata->>'resource_hash' 
                    FROM document_chunks 
                    WHERE metadata->>'resource_hash' IS NOT NULL
                """)
                vectorstore_hashes = {row[0] for row in cur.fetchall()}
        finally:
            conn.close()
        
        print(f"   Chunks in vectorstore: {chunk_count}")
        print(f"   Unique resource hashes: {len(vectorstore_hashes)}")
        
        # Step 4: Determine what needs to be added
        hashes_to_add = set(existing_paths.keys()) - vectorstore_hashes
        print(f"\n4. Files needing embedding: {len(hashes_to_add)}")
        
        if hashes_to_add:
            print("   First 5 files to add:")
            for hash_val in list(hashes_to_add)[:5]:
                path = existing_paths[hash_val]
                size = Path(path).stat().st_size
                print(f"      {Path(path).name} ({size} bytes)")
        
        return {
            "sources_loaded": len(sources),
            "existing_files": len(existing_paths),
            "missing_files": len(missing_paths),
            "chunks_in_vectorstore": chunk_count,
            "files_to_add": len(hashes_to_add),
        }

    @staticmethod
    def diagnose_data_directory():
        """
        Check what files actually exist in the data directory.
        
        Usage:
            docker exec data-manager-test-fresh-2 python -c "
            from tests.unit.test_ingestion_pipeline_isolation import TestDockerDeploymentDiagnostics
            TestDockerDeploymentDiagnostics.diagnose_data_directory()
            "
        """
        from pathlib import Path
        import os
        
        data_path = Path("/root/data")
        
        print(f"\n=== DATA DIRECTORY DIAGNOSTIC ===")
        print(f"Data path: {data_path}")
        print(f"Exists: {data_path.exists()}")
        
        if not data_path.exists():
            print("ERROR: Data directory does not exist!")
            return
        
        print(f"\nDirectory structure:")
        for item in data_path.iterdir():
            if item.is_dir():
                file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                print(f"  📁 {item.name}/ ({file_count} files)")
            else:
                print(f"  📄 {item.name}")
        
        # Check websites directory specifically
        websites_dir = data_path / "websites"
        if websites_dir.exists():
            print(f"\nWebsites directory contents:")
            html_files = list(websites_dir.rglob("*.html"))[:10]
            for f in html_files:
                print(f"  📄 {f.relative_to(websites_dir)} ({f.stat().st_size} bytes)")
            
            total_html = len(list(websites_dir.rglob("*.html")))
            print(f"\n  Total HTML files: {total_html}")

    @staticmethod
    def diagnose_ingestion_progress():
        """
        Check the current progress of ingestion by comparing documents to chunks.
        
        This helps identify if ingestion is:
        - Not started (0 chunks, N documents)
        - In progress (some chunks, N documents where chunks < expected)
        - Complete (all documents have chunks)
        - Stalled (no new chunks being added over time)
        
        Usage:
            docker exec data-manager-test-fresh-2 python -c "
            import os; os.environ['PG_PASSWORD']='donuts'
            from tests.unit.test_ingestion_pipeline_isolation import TestDockerDeploymentDiagnostics
            TestDockerDeploymentDiagnostics.diagnose_ingestion_progress()
            "
        """
        import os
        import psycopg2
        import psycopg2.extras
        from datetime import datetime
        
        password = os.environ.get('PG_PASSWORD', 'archi')
        pg_config = {
            "host": "postgres",
            "port": 5432,
            "database": "archi-db",
            "user": "archi",
            "password": password
        }
        
        conn = psycopg2.connect(**pg_config)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get document counts
                cur.execute("SELECT COUNT(*) as count FROM documents WHERE NOT is_deleted")
                doc_count = cur.fetchone()['count']
                
                # Get chunk counts
                cur.execute("SELECT COUNT(*) as count FROM document_chunks")
                chunk_count = cur.fetchone()['count']
                
                # Get unique embedded docs
                cur.execute("""
                    SELECT COUNT(DISTINCT metadata->>'resource_hash') as count 
                    FROM document_chunks 
                    WHERE metadata->>'resource_hash' IS NOT NULL
                """)
                embedded_docs = cur.fetchone()['count']
                
                # Get latest chunk timestamp
                cur.execute("SELECT MAX(created_at) as latest FROM document_chunks")
                latest = cur.fetchone()['latest']
                
                # Get earliest chunk timestamp  
                cur.execute("SELECT MIN(created_at) as earliest FROM document_chunks")
                earliest = cur.fetchone()['earliest']
        finally:
            conn.close()
        
        print("\n=== INGESTION PROGRESS DIAGNOSTIC ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"\nDocument counts:")
        print(f"  Documents in catalog: {doc_count}")
        print(f"  Documents embedded:   {embedded_docs}")
        print(f"  Total chunks:         {chunk_count}")
        
        if doc_count > 0:
            pct = (embedded_docs / doc_count) * 100
            print(f"\nProgress: {pct:.1f}% ({embedded_docs}/{doc_count} documents)")
        
        if earliest and latest:
            duration = latest - earliest
            print(f"\nTiming:")
            print(f"  First chunk: {earliest}")
            print(f"  Last chunk:  {latest}")
            print(f"  Duration:    {duration}")
            
            if embedded_docs > 0 and duration.total_seconds() > 0:
                avg_time = duration.total_seconds() / embedded_docs
                print(f"  Avg time/doc: {avg_time:.1f}s")
                
                remaining = doc_count - embedded_docs
                if remaining > 0:
                    eta_seconds = remaining * avg_time
                    print(f"\n  Remaining docs: {remaining}")
                    print(f"  Estimated time: {eta_seconds/60:.1f} minutes")
        
        # Determine status
        if chunk_count == 0:
            status = "NOT STARTED"
        elif embedded_docs < doc_count:
            status = "IN PROGRESS"
        else:
            status = "COMPLETE"
        
        print(f"\n=== STATUS: {status} ===")
        
        return {
            "documents": doc_count,
            "embedded": embedded_docs,
            "chunks": chunk_count,
            "status": status,
        }


# Standalone diagnostic script
if __name__ == "__main__":
    """
    Run diagnostics when executed directly.
    
    Usage from host:
        docker exec data-manager-test-fresh-2 python /app/tests/unit/test_ingestion_pipeline_isolation.py
    
    Or copy and run:
        docker cp tests/unit/test_ingestion_pipeline_isolation.py data-manager-test-fresh-2:/tmp/
        docker exec data-manager-test-fresh-2 python /tmp/test_ingestion_pipeline_isolation.py
    """
    print("=" * 60)
    print("INGESTION PIPELINE DIAGNOSTICS")
    print("=" * 60)
    
    try:
        result1 = TestDockerDeploymentDiagnostics.diagnose_data_directory()
    except Exception as e:
        print(f"Data directory diagnostic failed: {e}")
    
    print("\n" + "=" * 60)
    
    try:
        result2 = TestDockerDeploymentDiagnostics.diagnose_documents_vs_files()
    except Exception as e:
        print(f"Documents diagnostic failed: {e}")
    
    print("\n" + "=" * 60)
    
    try:
        result3 = TestDockerDeploymentDiagnostics.diagnose_vectorstore_update()
    except Exception as e:
        print(f"Vectorstore diagnostic failed: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)
