#!/usr/bin/env python3
"""Test script to run full document ingestion with chunking and embeddings."""
import os
import sys

# Set environment before imports
os.environ["ARCHI_CONFIGS_PATH"] = os.path.join(os.path.dirname(__file__), "local_dev_config/")
os.environ["PG_PASSWORD"] = "testpassword123"

from src.utils.config_access import get_full_config
from src.data_manager.vectorstore.manager import VectorStoreManager

def main():
    print("=" * 60)
    print("Archi Document Ingestion Test")
    print("=" * 60)
    
    # Use get_full_config with resolve_embeddings to resolve embedding class names to actual classes
    config = get_full_config(resolve_embeddings=True)
    print(f"Loaded config: {config.get('name', 'unknown')}")

    global_config = config.get("global", {})
    data_path = global_config.get("DATA_PATH", "/tmp/archi-data")
    print(f"Data path: {data_path}")

    pg_config = {
        "host": config["services"]["postgres"]["host"],
        "port": config["services"]["postgres"]["port"],
        "database": config["services"]["postgres"]["database"],
        "user": config["services"]["postgres"]["user"],
        "password": "testpassword123",
    }
    print(f"PostgreSQL: {pg_config['host']}:{pg_config['port']}")

    print("\n" + "-" * 40)
    print("Initializing VectorStoreManager...")
    print("-" * 40)
    
    manager = VectorStoreManager(
        config=config,
        global_config=global_config,
        data_path=data_path,
        pg_config=pg_config,
    )
    
    print(f"Collection: {manager.collection_name}")
    print(f"Distance metric: {manager.distance_metric}")
    print(f"Embedding model: {type(manager.embedding_model).__name__}")
    print(f"Chunk size: {manager._data_manager_config['chunk_size']}")
    print(f"Chunk overlap: {manager._data_manager_config['chunk_overlap']}")

    print("\n" + "-" * 40)
    print("Running vectorstore sync (this will chunk + embed)...")
    print("-" * 40)
    
    manager.update_vectorstore()
    
    print("\n" + "-" * 40)
    print("Checking results...")
    print("-" * 40)
    
    import psycopg2
    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM documents")
    doc_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM document_chunks")
    chunk_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL")
    embedded_count = cur.fetchone()[0]
    
    conn.close()
    
    print(f"Documents in catalog: {doc_count}")
    print(f"Total chunks: {chunk_count}")
    print(f"Chunks with embeddings: {embedded_count}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
