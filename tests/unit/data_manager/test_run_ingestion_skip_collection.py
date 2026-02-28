"""
Tests for Issue #460: Skip re-fetching tickets/scraped data on restart.

Verifies that when reset_collection is false, ticket_manager and scraper_manager
collection methods are NOT called during run_ingestion(), while
localfile_manager is always called regardless of the setting.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.data_manager.data_manager import DataManager


@pytest.fixture()
def mock_deps():
    """Patch direct dependencies of DataManager so __init__ runs cheaply."""
    with (
        patch("src.data_manager.data_manager.get_full_config") as mock_config,
        patch("src.data_manager.data_manager.read_secret", return_value="mock_pw"),
        patch("src.data_manager.data_manager.PersistenceService"),
        patch("src.data_manager.data_manager.ConfigService") as mock_cs,
        patch("src.data_manager.data_manager.LocalFileManager") as mock_lf,
        patch("src.data_manager.data_manager.ScraperManager") as mock_sc,
        patch("src.data_manager.data_manager.TicketManager") as mock_tm,
        patch("src.data_manager.data_manager.VectorStoreManager") as mock_vm,
    ):
        mock_config.return_value = {
            "global": {"DATA_PATH": "/tmp/archi-test-data"},
            "services": {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "db",
                    "user": "u",
                },
            },
            "data_manager": {
                "reset_collection": False,
                "sources": {},
            },
        }

        static = MagicMock()
        static.sources_config = {}
        mock_cs.return_value.get_static_config.return_value = static

        vm = mock_vm.return_value
        vm.collection_name = "test-collection"
        vm.distance_metric = "cosine"
        vm.embedding_model = MagicMock()
        vm.text_splitter = MagicMock()
        vm.stemmer = MagicMock()

        yield {
            "config": mock_config,
            "localfile_manager": mock_lf,
            "scraper_manager": mock_sc,
            "ticket_manager": mock_tm,
            "vector_manager": mock_vm,
        }


class TestRunIngestionSkipCollection:
    """Verify reset_collection controls ticket/scraper collection."""

    def test_skip_collection_when_reset_false(self, mock_deps):
        """When reset_collection is false, tickets and scraped data are NOT fetched."""
        mock_deps["config"].return_value["data_manager"]["reset_collection"] = False

        dm = DataManager(run_ingestion=False)
        dm.run_ingestion()

        mock_deps[
            "localfile_manager"
        ].return_value.collect_all_from_config.assert_called_once()
        mock_deps[
            "scraper_manager"
        ].return_value.collect_all_from_config.assert_not_called()
        mock_deps[
            "ticket_manager"
        ].return_value.collect_all_from_config.assert_not_called()

    def test_collect_all_when_reset_true(self, mock_deps):
        """When reset_collection is true, all sources are fetched."""
        mock_deps["config"].return_value["data_manager"]["reset_collection"] = True

        dm = DataManager(run_ingestion=False)
        dm.run_ingestion()

        mock_deps[
            "localfile_manager"
        ].return_value.collect_all_from_config.assert_called_once()
        mock_deps[
            "scraper_manager"
        ].return_value.collect_all_from_config.assert_called_once()
        mock_deps[
            "ticket_manager"
        ].return_value.collect_all_from_config.assert_called_once()

    def test_localfiles_always_collected(self, mock_deps):
        """localfile_manager is called regardless of reset_collection."""
        for reset_value in (True, False):
            for key in ("localfile_manager", "scraper_manager", "ticket_manager"):
                mock_deps[key].return_value.collect_all_from_config.reset_mock()
            mock_deps["config"].return_value["data_manager"]["reset_collection"] = (
                reset_value
            )

            dm = DataManager(run_ingestion=False)
            dm.run_ingestion()

            (
                mock_deps[
                    "localfile_manager"
                ].return_value.collect_all_from_config.assert_called_once(),
                f"localfile_manager should always be called (reset_collection={reset_value})",
            )

    def test_default_reset_collection_is_false(self, mock_deps):
        """When the key is absent, it defaults to False (skip fetching)."""
        del mock_deps["config"].return_value["data_manager"]["reset_collection"]

        dm = DataManager(run_ingestion=False)
        dm.run_ingestion()

        mock_deps[
            "localfile_manager"
        ].return_value.collect_all_from_config.assert_called_once()
        mock_deps[
            "scraper_manager"
        ].return_value.collect_all_from_config.assert_not_called()
        mock_deps[
            "ticket_manager"
        ].return_value.collect_all_from_config.assert_not_called()
