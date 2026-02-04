import os
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.data_manager.collectors.persistence import PersistenceService
from src.data_manager.collectors.scrapers.scraped_resource import \
    ScrapedResource
from src.data_manager.collectors.scrapers.scraper import LinkScraper
from src.utils.config_access import get_global_config
from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.data_manager.collectors.scrapers.integrations.git_scraper import \
        GitScraper


class ScraperManager:
    """Coordinates scraper integrations and centralises persistence logic."""

    def __init__(self, dm_config: Optional[Dict[str, Any]] = None) -> None:
        global_config = get_global_config()

        sources_config = (dm_config or {}).get("sources", {}) or {}
        links_config = sources_config.get("links", {}) if isinstance(sources_config, dict) else {}
        selenium_config = links_config.get("selenium_scraper", {}) if isinstance(sources_config, dict) else {}

        git_config = sources_config.get("git", {}) if isinstance(sources_config, dict) else {}
        sso_config = sources_config.get("sso", {}) if isinstance(sources_config, dict) else {}
        self.base_depth = links_config.get('base_source_depth', 5)
        logger.debug(f"Using base depth of {self.base_depth} for weblist URLs")

        scraper_config = {}
        if isinstance(links_config, dict):
            scraper_config = links_config.get("html_scraper", {}) or {}
        self.config = scraper_config
        raw_max_pages = links_config.get("max_pages")
        self.max_pages = None
        if raw_max_pages not in (None, ""):
            try:
                self.max_pages = int(raw_max_pages)
            except (TypeError, ValueError):
                logger.warning(f"Invalid max_pages value {raw_max_pages}; ignoring.")

        self.links_enabled = True
        self.git_enabled = git_config.get("enabled", False) if isinstance(git_config, dict) else True
        self.git_config = git_config if isinstance(git_config, dict) else {}
        self.selenium_config = selenium_config or {}
        self.selenium_enabled = self.selenium_config.get("enabled", False)
        self.scrape_with_selenium = self.selenium_config.get("use_for_scraping", False)

        self.sso_enabled = bool(sso_config.get("enabled", False))

        self.data_path = Path(global_config["DATA_PATH"])
        self.input_lists = links_config.get("input_lists", [])
        self.git_dir = self.data_path / "git"

        self.data_path.mkdir(parents=True, exist_ok=True)

        self.web_scraper = LinkScraper(
            verify_urls=self.config.get("verify_urls", True),
            enable_warnings=self.config.get("enable_warnings", True),
        )
        self._git_scraper: Optional["GitScraper"] = None
          
    def collect_all_from_config(
        self, persistence: PersistenceService
    ) -> None:
        """Run the configured scrapers and persist their output."""
        link_urls, git_urls, sso_urls = self._collect_urls_from_lists_by_type(self.input_lists)

        if git_urls:
            self.git_enabled = True
        if sso_urls:
            self.sso_enabled = True
            self._ensure_sso_defaults()

        self.collect_links(persistence, link_urls=link_urls)
        self.collect_sso(persistence, sso_urls=sso_urls)
        self.collect_git(persistence, git_urls=git_urls)

        logger.info("Web scraping was completed successfully")

    def collect_links(
        self,
        persistence: PersistenceService,
        link_urls: List[str] = [],
    ) -> None:
        """Collect only standard link sources."""
        if not self.links_enabled:
            logger.info("Links disabled, skipping link scraping")
            return
        if not link_urls:
            return
        websites_dir = persistence.data_path / "websites"
        if not os.path.exists(websites_dir):
            os.makedirs(websites_dir, exist_ok=True)
        self._collect_links_from_urls(link_urls, persistence, websites_dir)

    def collect_git(
        self,
        persistence: PersistenceService,
        git_urls: Optional[List[str]] = None,
    ) -> None:
        """Collect only git sources."""
        if not self.git_enabled:
            logger.info("Git disabled, skipping git scraping")
            return
        if not git_urls:
            return
        git_dir = persistence.data_path / "git"
        if not os.path.exists(git_dir):
            os.makedirs(git_dir, exist_ok=True)
        self._collect_git_resources(git_urls, persistence, git_dir)

    def collect_sso(
        self,
        persistence: PersistenceService,
        sso_urls: Optional[List[str]] = None,
    ) -> None:
        """Collect only SSO sources."""
        if not self.sso_enabled:
            logger.info("SSO disabled, skipping SSO scraping")
            return
        self._ensure_sso_defaults()
        if not sso_urls:
            return
        sso_dir = persistence.data_path / "sso"
        if not os.path.exists(sso_dir):
            os.makedirs(sso_dir, exist_ok=True)
        self._collect_sso_from_urls(sso_urls, persistence, sso_dir)

    def schedule_collect_links(self, persistence: PersistenceService, last_run: Optional[str] = None) -> None:
        """
        Scheduled collection of link sources.
        For now, this behaves the same as a full collection, overriding last_run depending on the persistence layer.
        """
        metadata = persistence.catalog.get_metadata_by_filter("source_type", source_type="links", metadata_keys=["url"])
        catalog_urls = [m[1].get("url", "") for m in metadata]
        self.collect_links(persistence, link_urls=catalog_urls)

    def schedule_collect_git(self, persistence: PersistenceService, last_run: Optional[str] = None) -> None:
        metadata = persistence.catalog.get_metadata_by_filter("source_type", source_type="git", metadata_keys=["url"])
        catalog_urls = [m[1].get("url", "") for m in metadata]
        self.collect_git(persistence, git_urls=catalog_urls)

    def schedule_collect_sso(self, persistence: PersistenceService, last_run: Optional[str] = None) -> None:
        metadata = persistence.catalog.get_metadata_by_filter("source_type", source_type="sso", metadata_keys=["url"])
        catalog_urls = [m[1].get("url", "") for m in metadata]
        self.collect_sso(persistence, sso_urls=catalog_urls)

    def _collect_links_from_urls(
        self,
        urls: List[str],
        persistence: PersistenceService,
        output_dir: Path,
    ) -> None:
        # Initialize authenticator if selenium is enabled
        authenticator = None
        if self.selenium_enabled:
            authenticator_class, kwargs = self._resolve_scraper()
            if authenticator_class is not None:
                authenticator = authenticator_class(**kwargs)

        try:
            for url in urls:
                # For standard link collection, don't use selenium for scraping
                # (SSO urls are handled separately via collect_sso)
                self._handle_standard_url(
                    url, 
                    persistence, 
                    output_dir, 
                    max_depth=self.base_depth,
                    client=None,
                    use_client_for_scraping=False
                )
        finally:
            if authenticator is not None:
                authenticator.close()  # Close the authenticator properly and free the resources

    def _collect_sso_from_urls(
        self,
        urls: List[str],
        persistence: PersistenceService,
        output_dir: Path,
    ) -> None:
        """Collect SSO-protected URLs using selenium for authentication."""
        if not self.selenium_enabled:
            logger.error("SSO scraping requires data_manager.sources.links.selenium_scraper.enabled")
            return
        if not read_secret("SSO_USERNAME") or not read_secret("SSO_PASSWORD"):
            logger.error("SSO scraping requires SSO_USERNAME and SSO_PASSWORD secrets")
            return
        authenticator = None
        if self.selenium_enabled:
            authenticator_class, kwargs = self._resolve_scraper()
            if authenticator_class is not None:
                authenticator = authenticator_class(**kwargs)
        
        if authenticator is None:
            logger.error("SSO collection requires a valid selenium scraper configuration")
            return

        try:
            for url in urls:
                # For SSO URLs, use selenium client for authentication
                # scrape_with_selenium determines if we use selenium for scraping too
                self._handle_standard_url(
                    url,
                    persistence,
                    output_dir,
                    max_depth=self.base_depth,
                    client=authenticator,
                    use_client_for_scraping=self.scrape_with_selenium
                )
        finally:
            if authenticator is not None:
                authenticator.close()

    def _ensure_sso_defaults(self) -> None:
        if not self.selenium_config:
            self.selenium_config = {}

        if not self.selenium_enabled:
            self.selenium_config["enabled"] = True
            self.selenium_enabled = True

        if not self.selenium_config.get("selenium_class"):
            self.selenium_config["selenium_class"] = "CERNSSOScraper"

        class_map = self.selenium_config.setdefault("selenium_class_map", {})
        if "CERNSSOScraper" not in class_map:
            class_map["CERNSSOScraper"] = {
                "class": "CERNSSOScraper",
                "kwargs": {
                    "headless": True,
                    "max_depth": 2,
                },
            }

    def _collect_urls_from_lists(self, input_lists) -> List[str]:
        """Collect URLs from the configured weblists."""
        # Handle case where input_lists might be None
        urls: List[str] = []
        if not input_lists:
            return urls
        for list_name in input_lists:
            list_path = Path("weblists") / Path(list_name).name
            if not list_path.exists():
                logger.warning(f"Input list {list_path} not found.")
                continue

            urls.extend(self._extract_urls_from_file(list_path))

        return urls

    def _collect_urls_from_lists_by_type(self, input_lists: List[str]) -> tuple[List[str], List[str], List[str]]:
        """All types of URLs are in the same input lists, separate them via prefixes"""
        link_urls: List[str] = []
        git_urls: List[str] = []
        sso_urls: List[str] = []
        for raw_url in self._collect_urls_from_lists(input_lists):
            if raw_url.startswith("git-"):
                git_urls.append(raw_url.split("git-", 1)[1])
                continue
            if raw_url.startswith("sso-"):
                sso_urls.append(raw_url.split("sso-", 1)[1])
                continue
            link_urls.append(raw_url)
        return link_urls, git_urls, sso_urls
    def _resolve_scraper(self):
        class_name = self.selenium_config.get("selenium_class")
        class_map = self.selenium_config.get("selenium_class_map", {})

        entry = class_map.get(class_name)

        if not entry: 
            logger.error(f"Selenium class {class_name} is not defined in the configuration")
            return None, {}

        scraper_class = entry.get("class")
        if isinstance(scraper_class, str):
            module_name = entry.get(
                    "module", 
                    "src.data_manager.collectors.scrapers.integrations.sso_scraper",
                    )
            module = importlib.import_module(module_name)
            scraper_class = getattr(module, scraper_class)
        scraper_kwargs = entry.get("kwargs", {})
        return scraper_class, scraper_kwargs


    def _handle_standard_url(
            self, 
            url: str, 
            persistence: PersistenceService, 
            output_dir: Path, 
            max_depth: int, 
            client=None, 
            use_client_for_scraping: bool = False,
    ) -> None:
        logger.debug("Handling standard url")
        logger.debug(url)
        try:
            for resource in self.web_scraper.crawl_iter(
                url,
                browserclient=client,
                max_depth=max_depth,
                selenium_scrape=use_client_for_scraping,
                max_pages=self.max_pages,
            ):
                persistence.persist_resource(
                    resource, output_dir
                )
        except Exception as exc:
            logger.error(f"Failed to scrape {url}: {exc}",exc_info=exc)

    def _extract_urls_from_file(self, path: Path) -> List[str]:
        """Extract URLs from file, ignoring depth specifications for now."""
        urls: List[str] = []
        with path.open("r") as file:
            for line in file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # Extract just the URL part, ignoring depth specification if present
                url_depth = stripped.split(",")
                url = url_depth[0].strip()
                urls.append(url)
        return urls

    def _collect_git_resources(
        self,
        git_urls: List[str],
        persistence: PersistenceService,
        git_dir: Path,
    ) -> List[ScrapedResource]:
        git_scraper = self._get_git_scraper()
        resources = git_scraper.collect(git_urls)
        for resource in resources:
            persistence.persist_resource(resource, git_dir)
        return resources

    def _get_git_scraper(self) -> "GitScraper":
        if self._git_scraper is None:
            from src.data_manager.collectors.scrapers.integrations.git_scraper import \
                    GitScraper

            self._git_scraper = GitScraper(manager=self, git_config=self.git_config)
        return self._git_scraper
