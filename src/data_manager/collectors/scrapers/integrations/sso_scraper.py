import hashlib
import importlib
import json
import os
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.data_manager.collectors.scrapers.scraped_resource import \
    ScrapedResource, BrowserIntermediaryResult
from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SSOScraper(ABC):
    """Generic base class for SSO-authenticated web scrapers."""
    
    def __init__(self, username=None, password=None, headless=True, site_type="generic", max_depth=2):
        """Initialize the SSO scraper with credentials and browser settings.
        
        Args:
            username (str, optional): SSO username. If None, will try to get from env vars.
            password (str, optional): SSO password. If None, will try to get from env vars.
            headless (bool): Whether to run the browser in headless mode.
            site_type (str): Type of site to scrape ('generic' or 'mkdocs')
            max_depth (int): Maximum number of levels to crawl per page.
        """
        self.username = username or self.get_username_from_env()
        self.password = password or self.get_password_from_env()
        self.headless = headless
        self.max_depth = max_depth
        self.site_type = site_type
        self.driver = None
        self.visited_urls = set()
        
        if self.username:
            logger.info(f"Using username: {self.username}")
    
    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image file."""
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico', '.webp')
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path.lower()
        return any(path.endswith(ext) for ext in image_extensions)
    
    @abstractmethod
    def get_username_from_env(self):
        """Get username from environment variables. Override in subclasses."""
        pass
    
    @abstractmethod
    def get_password_from_env(self):
        """Get password from environment variables. Override in subclasses."""
        pass
    
    @abstractmethod
    def login(self):
        """Login to SSO with the provided credentials. Override in subclasses."""
        pass
    
    def setup_driver(self):
        """Configure and initialize the Firefox WebDriver."""
        firefox_options = FirefoxOptions()
        if self.headless:
            firefox_options.add_argument("--headless")
        
        # Additional options for better performance in containers
        firefox_options.add_argument("--no-sandbox")
        firefox_options.add_argument("--disable-dev-shm-usage")
        firefox_options.add_argument("--disable-gpu")
        firefox_options.add_argument("--window-size=1920,1080")
        
        # Create Firefox profile with preferences
        firefox_profile = webdriver.FirefoxProfile()
        firefox_profile.set_preference("dom.disable_open_during_load", False)
        firefox_profile.set_preference("browser.download.folderList", 2)
        firefox_profile.set_preference("browser.download.manager.showWhenStarting", False)
        firefox_profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
        
        # Initialize the driver with options
        self.driver = webdriver.Firefox(options=firefox_options)
        self.driver.set_page_load_timeout(30)
        logger.info(f"Starting Firefox browser in {'headless' if self.headless else 'visible'} mode...")
        return self.driver
    
    def navigate_to(self, url, wait_time=1):
        """Navigate to specified URL and wait for page to load."""
        if not self.driver:
            raise RuntimeError("WebDriver not initialized. Call setup_driver() first.")
            
        self.driver.get(url)
        time.sleep(wait_time)  # Enable wait time for page loading
        logger.info(f"Navigated to {url}")
        logger.info(f"Page title: {self.driver.title}")
        return self.driver.title
    
    def get_links_with_same_hostname(self, base_url):
        """Extract all links from the current page that have the same hostname as base_url."""
        base_hostname = urllib.parse.urlparse(base_url).netloc
        links = []

        # Find all anchor tags
        if self.site_type == "mkdocs":
            # For MkDocs, prioritize navigation links
            anchors = self.driver.find_elements(By.CSS_SELECTOR, ".md-nav__link, .md-content a")
        else:
            anchors = self.driver.find_elements(By.TAG_NAME, "a")
        
        for anchor in anchors:
            try:
                href = anchor.get_attribute("href")
                if href and href.strip():
                    parsed_url = urllib.parse.urlparse(href)
                    # Check if the link has the same hostname and is not a fragment
                    if parsed_url.netloc == base_hostname and parsed_url.scheme in ('http', 'https'):
                        # Normalize the URL to prevent duplicates
                        normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                        if parsed_url.query:
                            normalized_url += f"?{parsed_url.query}"

                        # this works for CMS twiki but should be generalized
                        normalized_url = normalized_url.split("?")[0]
                        if 'bin/rdiff' in normalized_url or 'bin/edit' in normalized_url or 'bin/oops' in normalized_url  or 'bin/attach' in normalized_url or 'bin/genpdf' in normalized_url or '/WebIndex' in normalized_url:
                            continue
                        
                        if not self._clear_url(normalized_url):
                            continue
                        
                        # Skip image files
                        if self._is_image_url(normalized_url):
                            logger.debug(f"Skipping image URL: {normalized_url}")
                            continue

                        links.append(normalized_url)
                        
            except Exception as e:
                logger.error(f"Error extracting link: {e}")
                
        return list(set(links))  # Remove duplicates

    def extract_page_data(self, current_url):
        """Return the raw HTML payload for the current page."""
        if not self.driver:
            raise RuntimeError("WebDriver not initialized. Call setup_driver() first.")

        title = self.driver.title or ""
        content = self.driver.page_source or ""

        return {
            "url": current_url,
            "title": title,
            "content": content,
            "suffix": "html",
        }
    
    def crawl(self, start_url):
        """Crawl pages starting from the given URL, storing title and content of each page.

        Args:
            start_url (str): The URL to start crawling from

        Returns:
            List[Dict]: A list of dictionaries describing each visited page.
        """
        max_depth = self.max_depth
        depth = 0
        
        if not self.driver:
            self.setup_driver()
            
        # Reset crawling state
        self.visited_urls = set()
        self.page_data = []
        to_visit = [start_url]
        level_links = []
        
        # First authenticate through the start URL
        self.authenticate_and_navigate(start_url)
        
        base_hostname = urllib.parse.urlparse(start_url).netloc
        logger.info(f"Base hostname for crawling: {base_hostname}")
        logger.info(f"Site type: {self.site_type}")

        # History record   
        pages_visited = 0
        self.visited_urls = set()
        
        while to_visit and depth < max_depth:
            current_url = to_visit.pop(0)
            
            # Skip if we've already visited this URL
            if current_url in self.visited_urls:
                continue
            
            # Skip image files
            if self._is_image_url(current_url):
                logger.debug(f"Skipping image URL: {current_url}")
                self.visited_urls.add(current_url)
                continue
                
            logger.info(f"Crawling page {depth + 1}/{max_depth}: {current_url}")

            try:
                # Navigate to the page
                self.navigate_to(current_url, wait_time=2)
                
                # Mark as visited
                self.visited_urls.add(current_url)
                pages_visited += 1

                # Extract and store page data
                page_data = self.extract_page_data(current_url)
                self.page_data.append(page_data)
                logger.info(f"Extracted data from {current_url} ({len(page_data['content'])} chars)")
                
                # Get links to follow
                new_links = self.get_links_with_same_hostname(current_url)
                logger.info(f"Found {len(new_links)} links on the page (nv: {pages_visited})")

                # Add new links to visit
                for link in new_links:
                    if link not in self.visited_urls and link not in to_visit and link not in level_links:
                        logger.info(f"Found new link: {link} (nv: {pages_visited})")
                        level_links.append(link)

                # Scan next level if to_visit is empty
                if not to_visit:
                    to_visit.extend(level_links)
                    level_links = []
                    depth += 1
                        
            except Exception as e:
                logger.info(f"Error crawling {current_url}: {e}")
                self.visited_urls.add(current_url)  # Mark as visited to avoid retrying           
            
        logger.info(f"Crawling complete. Visited {pages_visited} pages.")
        return list(self.page_data)

    def _clear_url(self, url: str) -> bool:
        """Basic filtering for duplicate or fragment-only URLs."""
        if not url:
            return False

        # Ignore pure fragments or JavaScript links
        if url.startswith("javascript:"):
            return False

        return True
    
    def close(self):
        """Close the browser and clean up resources."""
        if self.driver:
            logger.info("Closing browser...")
            self.driver.quit()
            self.driver = None
    
    def authenticate_and_navigate(self, url):
        """Complete authentication flow and navigate to target URL."""
        try:
            if not self.driver:
                self.setup_driver()
                
            # First navigate to trigger SSO
            self.driver.get(url)
            
            # Login
            if self.login():
                # Navigate back to target page
                title = self.navigate_to(url)
                return title
            else:
                return None
        except Exception as e:
            logger.warning(f"Error during authentication: {e}")
            return None

    def authenticate(self, url):
        """Complete authentication flow and navigate to target URL."""
        try:
            if not self.driver:
                self.setup_driver()
                
            # First navigate to trigger SSO
            self.driver.get(url)
            
            # Login
            if self.login():
                # Navigate back to target page
                return self.driver.get_cookies()
            else:
                return None
        except Exception as e:
            logger.warning(f"Error during authentication: {e}")
            return None
        
    def __enter__(self):
        """Context manager entry point."""
        self.setup_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()


class CERNSSOScraper(SSOScraper):
    """A scraper to handle CERN SSO authentication and page navigation."""
    
    def get_username_from_env(self):
        """Get CERN SSO username from environment variables."""
        return read_secret("SSO_USERNAME")

    def get_password_from_env(self):
        """Get CERN SSO password from environment variables."""
        return read_secret("SSO_PASSWORD")

    def login(self):
        """Login to CERN SSO with the provided credentials."""
        if not self.username or not self.password:
            raise ValueError("Missing credentials for CERN SSO")
            
        try:
            # Wait for login form to appear
            username_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            username_input.send_keys(self.username)
            # time.sleep(1)  # Optional sleep to ensure the input is registered
            
            password_input = self.driver.find_element(By.ID, "password")
            password_input.send_keys(self.password)
            # time.sleep(1)  # Optional sleep to ensure the input is registered
            
            sign_in = self.driver.find_element(By.ID, "kc-login")
            sign_in.click()
                
            logger.info("Login credentials submitted")
            return True
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return False


class SSOCollector:
    """Collects resources behind SSO-protected URLs using configured scrapers."""

    def __init__(self, selenium_config: Dict[str, Dict]) -> None:
        self._config = selenium_config or {}
        self._enabled = self._config.get("enabled", False)
        self._class_name = self._config.get("selenium_class", "")
        self._class_map = self._config.get("selenium_class_map", {})

    def collect(self, url: str) -> List[ScrapedResource]:
        if not self._enabled:
            logger.error("SSO is disabled or not configured")
            return []

        scraper_class, scraper_kwargs = self._resolve_scraper()
        if scraper_class is None:
            return []

        try:
            with scraper_class(**scraper_kwargs) as scraper:
                payload = scraper.crawl(url)
                resources = self._extract_resources(scraper, payload)
                if not resources:
                    logger.warning(f"No content extracted from SSO crawl for {url}")
                return resources
        except Exception as exc:  # pragma: no cover - defensive catch
            logger.error(f"SSO scraping failed for {url}: {exc}")
            return []

    def _resolve_scraper(self):
        entry = self._class_map.get(self._class_name)
        if not entry:
            logger.error(f"SSO class {self._class_name} not configured")
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

    def _extract_resources(self, scraper, payload) -> List[ScrapedResource]:
        resources: List[ScrapedResource] = []

        page_data = getattr(scraper, "page_data", None)
        if isinstance(page_data, list):
            for page in page_data:
                if not isinstance(page, dict):
                    continue
                page_url = page.get("url")
                content = page.get("content")
                if not page_url or content is None:
                    continue

                resources.append(
                    ScrapedResource(
                        url=page_url,
                        content=content,
                        suffix=page.get("suffix", "html"),
                        source_type="sso",
                        metadata={
                            "title": page.get("title"),
                        },
                    )
                )

        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                page_url = item.get("url")
                content = item.get("content")
                if not page_url or content is None:
                    continue
                resources.append(
                    ScrapedResource(
                        url=page_url,
                        content=content,
                        suffix=item.get("suffix", "html"),
                        source_type="sso",
                        metadata={
                            "visible": str(self._visible).lower(),
                        },
                    )
                )

        elif isinstance(payload, dict):
            for page_url in payload.values():
                logger.warning(
                    f"SSO scraper returned mapping without page content; skipping {page_url}"
                )

        elif payload is not None:
            logger.warning(
                f"Unsupported SSO payload type {type(payload).__name__}"
            )

        return resources
