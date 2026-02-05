#!/usr/bin/env python3
"""Test semantic similarity search directly using pgvector."""
import os
import sys

# Set environment before imports
os.environ["ARCHI_CONFIGS_PATH"] = os.path.join(os.path.dirname(__file__), "local_dev_config/")
os.environ["PG_PASSWORD"] = "testpassword123"

import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("=" * 60)
    print("Semantic Similarity Search Test (pgvector)")
    print("=" * 60)
    
    print("\nLoading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pg_config = {
        "host": "localhost",
        "port": 5439,
        "database": "archi",
        "user": "archi",
        "password": "testpassword123",
    }

    queries = [
        "What is the prompt management refactor?",
        "How do I unify model providers?",
        "What tasks need cleanup?",
    ]

    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor()

    for query in queries:
        print("\n" + "-" * 60)
        print(f"Query: {query}")
        print("-" * 60)
        
        # Get embedding
        query_embedding = embedder.embed_query(query)
        
        # Search using pgvector cosine similarity
        cur.execute("""
            SELECT 
                dc.chunk_text,
                dc.metadata->>'filename' as filename,
                1 - (dc.embedding <=> %s::vector) as similarity
            FROM document_chunks dc
            WHERE dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> %s::vector
            LIMIT 5
        """, (query_embedding, query_embedding))
        
        results = cur.fetchall()
        print(f"Found {len(results)} results:\n")
        
        for i, (chunk, filename, similarity) in enumerate(results, 1):
            preview = chunk[:120].replace("\n", " ").strip()
            print(f"  {i}. {filename} (similarity: {similarity:.4f})")
            print(f"     {preview}...")
            print()

    cur.close()
    conn.close()
    print("Done!")

if __name__ == "__main__":
    main()
