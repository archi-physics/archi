from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import nltk
import psycopg2
import psycopg2.extras
from .loader_utils import select_loader
from .postgres_vectorstore import PostgresVectorStore
from langchain_text_splitters.character import CharacterTextSplitter

from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService
from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_DISTANCE_METRICS = ["l2", "cosine", "ip"]

class VectorStoreManager:
    """
    Encapsulates vectorstore configuration and synchronization.

    Uses PostgreSQL with pgvector for vector storage and similarity search.
    """

    def __init__(
        self,
        *,
        config: Dict,
        global_config: Dict,
        data_path: str,
        pg_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.global_config = global_config
        self.data_path = data_path

        self._data_manager_config = config["data_manager"]
        self._services_config = config.get("services", {})

        if pg_config is None:
            pg_config = {
                "password": read_secret("PG_PASSWORD"),
                **self._services_config["postgres"],
            }
        self._pg_config = pg_config
        self._catalog = PostgresCatalogService(self.data_path, pg_config=self._pg_config)

        embedding_name = self._data_manager_config["embedding_name"]
        self.collection_name = (
            f"{self._data_manager_config['collection_name']}_with_{embedding_name}"
        )

        self.distance_metric = self._data_manager_config["distance_metric"]
        if self.distance_metric not in SUPPORTED_DISTANCE_METRICS:
            raise ValueError(
                f"The selected distance metrics, '{self.distance_metric}', is not supported. "
                f"Must be one of {SUPPORTED_DISTANCE_METRICS}"
            )

        # Build embedding model
        embedding_class_map = self._data_manager_config["embedding_class_map"]
        from src.utils.config_service import ConfigService
        embedding_class_map = ConfigService._resolve_embedding_classes(embedding_class_map)

        embedding_entry = embedding_class_map[embedding_name]
        embedding_class = embedding_entry["class"]
        embedding_kwargs = embedding_entry.get("kwargs", {})
        self.embedding_model = embedding_class(**embedding_kwargs)

        self.text_splitter = CharacterTextSplitter(
            chunk_size=self._data_manager_config["chunk_size"],
            chunk_overlap=self._data_manager_config["chunk_overlap"],
        )

        self.stemmer = None
        stemming_cfg = self._data_manager_config.get("stemming", {})
        if stemming_cfg.get("enabled", False):
            nltk.download("punkt_tab")
            self.stemmer = nltk.stem.PorterStemmer()

        default_workers = min(64, (os.cpu_count() or 1) + 4)
        parallel_workers_config = self._data_manager_config.get("parallel_workers")
        if parallel_workers_config is None:
            self.parallel_workers = default_workers
        else:
            try:
                self.parallel_workers = int(parallel_workers_config)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid 'parallel_workers' value %r. Falling back to default.",
                    parallel_workers_config,
                )
                self.parallel_workers = default_workers
        self.parallel_workers = max(1, self.parallel_workers)

        logger.info(f"VectorStoreManager initialized: collection={self.collection_name}")

    def delete_existing_collection_if_reset(self) -> None:
        """Delete the collection if reset_collection is enabled.

        Truncates the ``document_chunks`` table and resets all documents'
        ingestion status to ``'pending'`` so they get re-embedded.
        """
        if not self._data_manager_config.get("reset_collection", False):
            return

        conn = psycopg2.connect(**self._pg_config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE document_chunks CASCADE")
                logger.info("Truncated document_chunks table")

                # Reset ingestion status so all documents get re-embedded.
                cursor.execute(
                    """
                    UPDATE documents
                    SET ingestion_status = 'pending',
                        ingestion_error = NULL,
                        indexed_at = NULL
                    WHERE NOT is_deleted
                    """
                )
                reset_docs = cursor.rowcount

                conn.commit()

                conn.autocommit = True
                cursor.execute("VACUUM FULL document_chunks")
                conn.autocommit = False

                logger.info(
                    "reset_collection is enabled; truncated document_chunks, "
                    "reset %d documents for collection %s",
                    reset_docs, self.collection_name,
                )
        except Exception as exc:
            logger.error("Failed during collection reset: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                conn.autocommit = False
            except Exception:
                pass
            conn.close()

    def fetch_collection(self):
        """
        Return the active PostgresVectorStore.
        """
        # Map distance metric names
        distance_metric_map = {
            "l2": "l2",
            "cosine": "cosine",
            "ip": "inner_product",
        }
        pg_distance = distance_metric_map.get(self.distance_metric, "cosine")

        store = PostgresVectorStore(
            pg_config=self._pg_config,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            distance_metric=pg_distance,
        )
        count = store.count()
        logger.info(f"N in PostgreSQL collection: {count}")
        return store

    def update_vectorstore(self) -> None:
        """Synchronise filesystem documents with the vectorstore."""
        store = self.fetch_collection()

        sources = PostgresCatalogService.load_sources_catalog(self.data_path, self._pg_config)
        logger.info(f"Loaded {len(sources)} sources from catalog")

        # Get hashes currently in vectorstore
        hashes_in_vstore = self._collect_postgres_hashes()
        files_in_data = self._collect_indexed_documents(sources)

        hashes_in_data = set(files_in_data.keys())

        logger.info(f"Files in catalog: {len(hashes_in_data)}, Files in vectorstore: {len(hashes_in_vstore)}")

        if hashes_in_data == hashes_in_vstore:
            logger.info("Vectorstore is up to date")
        else:
            logger.info("Vectorstore needs to be updated")

            hashes_to_remove = list(hashes_in_vstore - hashes_in_data)
            if hashes_to_remove:
                logger.info(f"Removing {len(hashes_to_remove)} stale documents")
                self._remove_from_postgres(hashes_to_remove)

            hashes_to_add = hashes_in_data - hashes_in_vstore
            files_to_add = {
                hash_value: files_in_data[hash_value] for hash_value in hashes_to_add
            }
            if files_to_add:
                logger.info(f"Adding {len(files_to_add)} new documents")
                try:
                    self._add_to_postgres(files_to_add)
                except Exception as e:
                    logger.error(f"Files could not be added",exc_info=e)
            logger.info("Vectorstore update has been completed")

        logger.info(f"N Collection: {store.count()}")

    def _collect_postgres_hashes(self) -> set:
        """Get all resource hashes currently in the PostgreSQL vectorstore."""
        conn = psycopg2.connect(**self._pg_config)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT metadata->>'resource_hash' as hash
                    FROM document_chunks
                    WHERE (metadata->>'collection' = %s OR metadata->>'collection' IS NULL)
                      AND metadata->>'resource_hash' IS NOT NULL
                    """,
                    (self.collection_name,)
                )
                return {row[0] for row in cursor.fetchall()}
        finally:
            conn.close()

    def _remove_from_postgres(self, hashes_to_remove: List[str]) -> None:
        """Remove chunks by resource hash from PostgreSQL."""
        conn = psycopg2.connect(**self._pg_config)
        try:
            with conn.cursor() as cursor:
                for resource_hash in hashes_to_remove:
                    cursor.execute(
                        """
                        DELETE FROM document_chunks
                        WHERE metadata->>'resource_hash' = %s
                          AND (metadata->>'collection' = %s OR metadata->>'collection' IS NULL)
                        """,
                        (resource_hash, self.collection_name)
                    )
                conn.commit()
                logger.debug(f"Removed {len(hashes_to_remove)} resource hashes from vectorstore")
        finally:
            conn.close()

    def _add_to_postgres(self, files_to_add: Dict[str, str]) -> None:
        """Add files to PostgreSQL vectorstore."""
        if not files_to_add:
            return
        commit_batch_size = 25

        # Mark all documents as 'embedding' before starting
        for filehash in files_to_add:
            self._catalog.update_ingestion_status(filehash, "embedding")

        files_to_add_items = list(files_to_add.items())
        apply_stemming = self._data_manager_config.get("stemming", {}).get("enabled", False)
        if apply_stemming:
            tokenize = nltk.tokenize.word_tokenize
            stem = self.stemmer.stem

        def process_file(filehash: str, file_path: str):
            filename = Path(file_path).name
            logger.debug(f"Processing file: {filename} (hash: {filehash})")

            try:
                loader = self.loader(file_path)
            except Exception as exc:
                logger.error(f"Failed to load file: {file_path}. Skipping. Exception: {exc}")
                self._catalog.update_ingestion_status(filehash, "failed", str(exc))
                return None

            if loader is None:
                self._catalog.update_ingestion_status(filehash, "failed", f"Unsupported file format: {file_path}")
                return None

            file_level_metadata = self._load_file_metadata(filehash)
            try:
                docs = loader.load()
            except Exception as exc:
                logger.error("Failed to read file %s. Skipping. Exception: %s", file_path, exc)
                self._catalog.update_ingestion_status(filehash, "failed", str(exc))
                return None

            split_docs = self.text_splitter.split_documents(docs)

            chunks: List[str] = []
            metadatas: List[Dict] = []

            for index, split_doc in enumerate(split_docs):
                chunk = split_doc.page_content or ""
                # Remove NUL bytes that PostgreSQL cannot handle
                chunk = chunk.replace('\x00', '')
                
                if apply_stemming:
                    words = tokenize(chunk)
                    chunk = " ".join(stem(word) for word in words)

                if not chunk.strip():
                    continue

                chunks.append(chunk)

                doc_metadata = getattr(split_doc, "metadata", {}) or {}
                if not isinstance(doc_metadata, dict):
                    doc_metadata = dict(doc_metadata)
                entry_metadata = {**file_level_metadata, **doc_metadata}
                entry_metadata["chunk_index"] = index
                entry_metadata["filename"] = filename
                entry_metadata["resource_hash"] = filehash
                entry_metadata["collection"] = self.collection_name
                metadatas.append(entry_metadata)

            if not chunks:
                logger.info(f"No chunks generated for {filename}; skipping.")
                self._catalog.update_ingestion_status(filehash, "failed", "No text chunks could be extracted")
                return None

            return filename, chunks, metadatas

        processed_results: Dict[str, tuple] = {}
        max_workers = max(1, self.parallel_workers)
        logger.info(f"Processing files with up to {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, filehash, file_path): filehash
                for filehash, file_path in files_to_add_items
            }
            for future in as_completed(futures):
                filehash = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error(
                        "Unexpected error while processing %s: %s",
                        files_to_add.get(filehash),
                        exc,
                    )
                    self._catalog.update_ingestion_status(filehash, "failed", str(exc))
                    continue
                if result:
                    processed_results[filehash] = result

        logger.info("Finished processing files; adding to vectorstore")

        # Batch insert to PostgreSQL
        conn = psycopg2.connect(**self._pg_config)
        try:
            with conn.cursor() as cursor:
                import json
                
                total_files = len(files_to_add_items)
                files_since_commit = 0
                for file_idx, (filehash, file_path) in enumerate(files_to_add_items):
                    processed = processed_results.get(filehash)
                    if not processed:
                        continue

                    filename, chunks, metadatas = processed
                    logger.info(f"Embedding file {file_idx+1}/{total_files}: {filename} ({len(chunks)} chunks)")

                    savepoint_name = f"sp_embed_{file_idx}"
                    cursor.execute(f"SAVEPOINT {savepoint_name}")
                    try:
                        embeddings = self.embedding_model.embed_documents(chunks)
                    except Exception as exc:
                        logger.error(f"Failed to embed {filename}: {exc}")
                        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        cursor.execute(
                            """UPDATE documents
                               SET ingestion_status = 'failed', ingestion_error = %s
                               WHERE resource_hash = %s AND NOT is_deleted""",
                            (str(exc), filehash),
                        )
                        cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        files_since_commit += 1
                        if files_since_commit >= commit_batch_size:
                            conn.commit()
                            logger.info("Committed embedding progress batch (%d files)", files_since_commit)
                            files_since_commit = 0
                        continue

                    logger.info(f"Finished embedding {filename}")
                    
                    # Get document_id from the catalog (documents table)
                    document_id = self._catalog.get_document_id(filehash)
                    if document_id is None:
                        logger.warning(f"No document record found for {filehash}, chunks will have NULL document_id")

                    insert_data = []
                    for idx, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadatas)):
                        # Ensure no NUL bytes in chunk or metadata JSON
                        clean_chunk = chunk.replace('\x00', '')
                        clean_metadata_json = json.dumps(metadata).replace('\x00', '')
                        
                        insert_data.append((
                            document_id,  # Link to documents table
                            idx,   # chunk_index
                            clean_chunk,
                            embedding,
                            clean_metadata_json,
                        ))

                    try:
                        logger.debug(f"Inserting data in {filename} document_id = {document_id}")
                        psycopg2.extras.execute_values(
                            cursor,
                            """
                            INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding, metadata)
                            VALUES %s
                            """,
                            insert_data,
                            template="(%s, %s, %s, %s::vector, %s::jsonb)",
                        )
                        logger.debug(f"Added {len(insert_data)} chunks for {filename} (document_id={document_id})")

                        # Update timestamps and mark as embedded
                        cursor.execute(
                            """UPDATE documents
                               SET ingested_at = NOW(), ingestion_status = 'embedded',
                                   ingestion_error = NULL, indexed_at = NOW()
                               WHERE resource_hash = %s AND NOT is_deleted""",
                            (filehash,),
                        )
                        cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    except Exception as exc:
                        logger.error(f"Failed to store vectors for {filename}: {exc}")
                        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        cursor.execute(
                            """UPDATE documents
                               SET ingestion_status = 'failed', ingestion_error = %s
                               WHERE resource_hash = %s AND NOT is_deleted""",
                            (str(exc), filehash),
                        )
                        cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")

                    files_since_commit += 1
                    if files_since_commit >= commit_batch_size:
                        conn.commit()
                        logger.info("Committed embedding progress batch (%d files)", files_since_commit)
                        files_since_commit = 0

                if files_since_commit > 0:
                    conn.commit()
                    logger.info("Committed final embedding progress batch (%d files)", files_since_commit)
        finally:
            conn.close()

        logger.info("All files have been added to the vectorstore")

    def loader(self, file_path: str):
        """Return the document loader for a given path."""
        loader = select_loader(file_path)
        if loader is None:
            logger.error(f"Format not supported -- {file_path}")
        return loader

    def _collect_indexed_documents(self, sources: Dict[str, str]) -> Dict[str, str]:
        """
        Build a mapping of resource hash -> absolute path from the persisted index.
        """
        files_in_data: Dict[str, str] = {}
        missing_files = []
        skipped_dirs = []
        for resource_hash, stored_path in sources.items():
            path = Path(stored_path)
            if not path.exists():
                missing_files.append((resource_hash, stored_path))
                logger.warning(
                    f"Indexed resource '{resource_hash}' points to missing file: {stored_path}"
                )
                continue
            if path.is_dir():
                skipped_dirs.append((resource_hash, stored_path))
                logger.debug(
                    f"Indexed resource '{resource_hash}' points to a directory; skipping."
                )
                continue

            if resource_hash in files_in_data and files_in_data[resource_hash] != str(path):
                logger.warning(
                    "Duplicate resource hash detected in index; keeping first occurrence. "
                    f"hash={resource_hash}, existing={files_in_data[resource_hash]}, ignored={path}"
                )
                continue

            files_in_data[resource_hash] = str(path)

        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing files in catalog (first 5): {missing_files[:5]}")
        if skipped_dirs:
            logger.debug(f"Skipped {len(skipped_dirs)} directories in catalog")
        logger.info(f"Collected {len(files_in_data)} valid indexed documents (after filtering missing/dirs)")

        return files_in_data

    def _load_file_metadata(self, resource_hash: str) -> Dict[str, str]:
        """
        Load persisted metadata stored in the catalog, if available.
        """
        metadata = self._catalog.get_metadata_for_hash(resource_hash) or {}
        sanitized: Dict[str, str] = {}
        for key, value in metadata.items():
            if key is None or value is None:
                continue
            sanitized[str(key)] = str(value)
        return sanitized
