import os
from typing import Callable, Optional

from src.data_manager.collectors.persistence import PersistenceService
from src.data_manager.collectors.scrapers.scraper_manager import ScraperManager
from src.data_manager.collectors.tickets.ticket_manager import TicketManager
from src.data_manager.collectors.localfile_manager import LocalFileManager
from src.data_manager.vectorstore.manager import VectorStoreManager
from src.utils.config_access import get_full_config
from src.utils.config_service import ConfigService
from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataManager:
    def __init__(self, *, run_ingestion: bool = True, factory=None):

        self.config = get_full_config()
        self.global_config = self.config["global"]
        self.data_path = self.global_config["DATA_PATH"]
        self.should_run_ingestion = run_ingestion

        os.makedirs(self.data_path, exist_ok=True)

        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.config["services"]["postgres"],
        }
        self.persistence = PersistenceService(self.data_path, pg_config=self.pg_config)
        self.config_service = (
            factory.config_service
            if factory
            else ConfigService(pg_config=self.pg_config)
        )
        static_config = self.config_service.get_static_config()
        if not static_config or static_config.sources_config is None:
            raise RuntimeError(
                "Static config missing sources_config; run deployment initialization first."
            )
        self.config["data_manager"]["sources"] = static_config.sources_config

        self.localfile_manager = LocalFileManager(dm_config=self.config["data_manager"])
        self.scraper_manager = ScraperManager(dm_config=self.config["data_manager"])
        self.ticket_manager = TicketManager(dm_config=self.config["data_manager"])

        self.vector_manager = VectorStoreManager(
            config=self.config,
            global_config=self.global_config,
            data_path=self.data_path,
            pg_config=self.pg_config,
        )

        self.collection_name = self.vector_manager.collection_name
        self.distance_metric = self.vector_manager.distance_metric
        self.embedding_model = self.vector_manager.embedding_model
        self.text_splitter = self.vector_manager.text_splitter
        self.stemmer = self.vector_manager.stemmer

        logger.info(f"Using collection: {self.collection_name}")

        if self.should_run_ingestion:
            self.run_ingestion()

    def run_ingestion(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Execute initial ingestion and vectorstore update."""
        reset_collection = self.config["data_manager"].get("reset_collection", False)

        source_aggregation = [
            (
                "Copying configured local files",
                lambda: self.localfile_manager.collect_all_from_config(
                    self.persistence
                ),
            ),
        ]

        if reset_collection:
            source_aggregation.extend(
                [
                    (
                        "Scraping documents onto filesystem",
                        lambda: self.scraper_manager.collect_all_from_config(
                            self.persistence
                        ),
                    ),
                    (
                        "Fetching ticket data onto filesystem",
                        lambda: self.ticket_manager.collect_all_from_config(
                            self.persistence
                        ),
                    ),
                ]
            )
        else:
            logger.info(
                "reset_collection is false; skipping ticket and scraper collection. "
                "Data will be updated via scheduled sync."
            )

        for message, step in source_aggregation:
            logger.info(message)
            if progress_callback:
                progress_callback(message)
            step()

        if progress_callback:
            progress_callback("Flushing indices")
        self.persistence.flush_index()

        # Verify catalog was updated
        catalog = self.persistence.catalog
        catalog.refresh()  # Ensure we have the latest data
        logger.info(f"Catalog contains {len(catalog.file_index)} resources after flush")

        if progress_callback:
            progress_callback("Resetting vectorstore (if configured)")
        self.vector_manager.delete_existing_collection_if_reset()
        if progress_callback:
            progress_callback("Updating vectorstore")
        self.vector_manager.update_vectorstore()

    def delete_existing_collection_if_reset(self, *, force: bool = False):
        """Proxy to the underlying vector manager."""
        if not (self.should_run_ingestion or force):
            logger.debug("Skipping collection reset check (ingestion disabled).")
            return None
        return self.vector_manager.delete_existing_collection_if_reset()

    def fetch_collection(self):
        """Proxy to the underlying vector manager."""
        return self.vector_manager.fetch_collection()

    def update_vectorstore(self, *, force: bool = False):
        """Proxy to the underlying vector manager."""
        if not (self.should_run_ingestion or force):
            logger.debug("Skipping vectorstore update (ingestion disabled).")
            return None
        self.vector_manager.update_vectorstore()

    def _update_after_collect(self) -> None:
        self.persistence.flush_index()
        self.vector_manager.update_vectorstore()
