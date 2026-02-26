"""
ELOG scraper for electronic logbooks (https://elog.sourceforge.net/).

Automatically discovers all entries by following pagination links on the
index page, then fetches each individual entry as a ScrapedResource.

Config (under data_manager.sources.elog):
    url:          Base URL of the logbook, e.g. https://www-enstore.fnal.gov/elog/dCache/
    max_entries:  Optional cap on total entries to fetch (default: unlimited)
    verify_ssl:   Whether to verify SSL certificates (default: False)
"""

import re
import requests
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.data_manager.collectors.scrapers.scraped_resource import ScrapedResource
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PAGE_HREF  = re.compile(r"^page\d+$")
_ENTRY_PATH = re.compile(r"/\d+$")


class ElogScraper:
    """Crawls an ELOG logbook index (with pagination) and yields each entry."""

    def __init__(self, config: dict) -> None:
        self.base_url   = config.get("url", "").rstrip("/") + "/"
        self.max_entries: Optional[int] = config.get("max_entries")
        self.verify_ssl = config.get("verify_ssl", False)
        self._session   = requests.Session()
        if not self.verify_ssl:
            import urllib3
            urllib3.disable_warnings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_entries(self) -> Iterator[ScrapedResource]:
        """Yield one ScrapedResource per logbook entry, newest first."""
        entry_urls = self._discover_entry_urls()
        fetched = 0
        for url in entry_urls:
            if self.max_entries is not None and fetched >= self.max_entries:
                logger.info(f"Reached max_entries={self.max_entries}; stopping.")
                break
            resource = self._fetch_entry(url)
            if resource is not None:
                yield resource
                fetched += 1
        logger.info(f"ElogScraper: fetched {fetched} entries from {self.base_url}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_entry_urls(self) -> list[str]:
        """Return deduplicated entry URLs collected from all index pages."""
        seen:   set[str]  = set()
        result: list[str] = []

        # Collect all index page URLs first (main page + all paginated pages)
        index_pages = [self.base_url]
        index_pages.extend(self._get_pagination_urls(self.base_url))

        logger.info(f"ElogScraper: found {len(index_pages)} index pages to scan")

        for page_url in index_pages:
            for entry_url in self._get_entry_urls_from_page(page_url):
                if entry_url not in seen:
                    seen.add(entry_url)
                    result.append(entry_url)

        # Sort descending by numeric entry ID (most recent first)
        result.sort(key=lambda u: int(u.rstrip("/").rsplit("/", 1)[-1]), reverse=True)
        logger.info(f"ElogScraper: discovered {len(result)} unique entries")
        return result

    def _get_pagination_urls(self, index_url: str) -> list[str]:
        """Return URLs for page2, page3, … found on the main index page."""
        html = self._fetch_html(index_url)
        if html is None:
            return []
        soup = BeautifulSoup(html, "html.parser")
        pages = []
        for a in soup.find_all("a", href=_PAGE_HREF):
            pages.append(urljoin(index_url, a["href"]))
        logger.debug(f"ElogScraper: found {len(pages)} pagination links")
        return pages

    def _get_entry_urls_from_page(self, page_url: str) -> list[str]:
        """Return all entry URLs found on a single index/listing page."""
        html = self._fetch_html(page_url)
        if html is None:
            return []
        soup  = BeautifulSoup(html, "html.parser")
        base_host = urlparse(self.base_url).netloc
        entries: set[str] = set()
        for a in soup.find_all("a", href=True):
            full = urljoin(page_url, a["href"])
            parsed = urlparse(full)
            if parsed.netloc == base_host and _ENTRY_PATH.search(parsed.path):
                # Strip query/fragment so we get the canonical entry URL
                entries.add(parsed._replace(query="", fragment="").geturl())
        return list(entries)

    def _fetch_entry(self, url: str) -> Optional[ScrapedResource]:
        """Fetch a single entry page, extract structured text, and return a ScrapedResource."""
        html = self._fetch_html(url)
        if html is None:
            return None
        text, metadata = self._parse_entry(html, url)
        return ScrapedResource(
            url=url,
            content=text,
            suffix="txt",
            source_type="web",
            metadata=metadata,
        )

    def _parse_entry(self, html: str, url: str) -> tuple[str, dict]:
        """Parse an ELOG entry page into clean text and structured metadata."""
        soup = BeautifulSoup(html, "html.parser")
        meta: dict = {"url": url, "elog_entry": True}

        # Extract entry ID from URL
        entry_id = url.rstrip("/").rsplit("/", 1)[-1]
        meta["entry_id"] = entry_id

        # Extract attribute rows (Incident Date, Tech, Node, Inst, Category, Fix Action)
        for row in soup.select("table.listframe tr td table tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True).rstrip(":")
                value = cells[1].get_text(strip=True)
                if key and value:
                    meta[key.lower().replace(" ", "_")] = value

        # Entry time from attribhead
        attribhead = soup.find("td", class_="attribhead")
        if attribhead:
            text = attribhead.get_text(" ", strip=True)
            for part in text.split():
                pass  # entry_time already in meta via hidden inputs if needed

        # Main message body
        body = ""
        pre = soup.find("pre", class_="messagepre")
        if pre:
            body = pre.get_text()

        # Build clean plain-text document
        lines = [f"ELOG Entry {entry_id} — {self.base_url}"]
        for k, v in meta.items():
            if k not in ("url", "elog_entry", "entry_id"):
                lines.append(f"{k.replace('_', ' ').title()}: {v}")
        lines.append("")
        lines.append(body.strip())

        return "\n".join(lines), meta

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            r = self._session.get(url, timeout=15, verify=self.verify_ssl)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            logger.warning(f"ElogScraper: could not fetch {url}: {exc}")
            return None
