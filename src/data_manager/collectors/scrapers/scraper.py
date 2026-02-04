import requests
import re

from typing import Dict, Iterator, List, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag

from src.data_manager.collectors.scrapers.scraped_resource import \
    ScrapedResource
from src.utils.logging import get_logger

logger = get_logger(__name__)

class LinkScraper:
    """
    Single scraper for all our link needs that handles Selenium and requests.
    This class explicitly handles requests, but if selenium scraping is enabled for a link
    everything is passed through to the driver including how the page data is collected and 
    how the next level of links are found. This class DOESNT own the selenium driver, that is 
    owned by the scraper manager class. 
    """

    def __init__(self, verify_urls: bool = True, enable_warnings: bool = True) -> None:
        self.verify_urls = verify_urls
        self.enable_warnings = enable_warnings
        # seen_urls tracks anything queued/visited; visited_urls tracks pages actually crawled.
        self.visited_urls = set()
        self.seen_urls = set()

    def reap(self, response, current_url: str, selenium_scrape: bool = False, authenticator = None):
        """
        probably the most complicated method here and most volatile in terms of maybe later needing a rewrite

        this method is here to deal with any result that it gets back. for a selenium resource it expects results as a 
        BrowserIntermediaryResult, otherwhise it will handle it as a normal http response. it handles getting the next set 
        of links and updating the page data gathered

        Args: 
            response (BrowserIntermediaryResult | requests.response): whatever has been collected for the current_url by the scraper
            selenium_scrape (bool): whether or not selenium was used to scrape this content
            authenticator (SSOAuthenticator | None): client being used to crawl websites or just for auth 

        Return (tuple[list[str], list[ScrapedResource]]): next links to crawl and resources collected
        """

        # mark as visited
        self._mark_visited(current_url)

        source_type = "links" if (authenticator is None) else "sso"
        
        resources = []

        if selenium_scrape: # deals with a selenium response (should work for both non authenitcated and authenticated sites in principle)
            assert(authenticator is not None) ## this shouldnt be tripped
            
            # For selenium scraping, we expect a simple dict from extract_page_data
            # containing url, title, content, suffix
            content = response.get("content", "")
            title = response.get("title", "")
            suffix = response.get("suffix", "html")
            
            resource = ScrapedResource(
                url=current_url,
                content=content, 
                suffix=suffix,
                source_type=source_type,
                metadata={
                    "title": title,
                    "content_type": "rendered_html",
                    "renderer": "selenium",
                },
            )
            res = authenticator.get_links_with_same_hostname(current_url)
            resources.append(resource)
                
        else: # deals with http response
            content_type = response.headers.get("Content-type")

            if current_url.lower().endswith(".pdf"):
                resource = ScrapedResource(
                    url=current_url,
                    content=response.content,
                    suffix="pdf",
                    source_type=source_type,
                    metadata={"content_type": content_type},
                )
            else:
                resource = ScrapedResource(
                    url=current_url,
                    content=response.text,
                    suffix="html",
                    source_type=source_type,
                    metadata={
                        "content_type": content_type,
                        "encoding": response.encoding,
                    },
                )
            res = self.get_links_with_same_hostname(current_url, resource)
            resources.append(resource)

        return res, resources # either collected via http or via authenticators method


    def crawl(
        self,
        start_url: str,
        browserclient = None,
        max_depth: int = 1,
        selenium_scrape: bool = False,
        max_pages: Optional[int] = None,
    ):
        """
        crawl pages from a given starting url up to a given depth either using basic http or a provided browser client

        Args : 
            start_url (str): Url to start crawling from
            authenticator (SSOAuthenticator): class used for handling authenticatoin for web resources
            max_depth (int): max depth of links to descend from the start url
            selenium_scrape (bool): tracks whether or not the page should be scraped through selenium or not
            max_pages (int | None): cap on total pages to visit before stopping

        Returns: List[ScrapedResource]

        """
        # Consume the iterator so page_data is populated for callers of crawl().
        for _ in self.crawl_iter(
            start_url,
            browserclient=browserclient,
            max_depth=max_depth,
            selenium_scrape=selenium_scrape,
            max_pages=max_pages,
            collect_page_data=True,
        ):
            pass
        return list(self.page_data)

    def crawl_iter(
        self,
        start_url: str,
        browserclient = None,
        max_depth: int = 1,
        selenium_scrape: bool = False,
        max_pages: Optional[int] = None,
        collect_page_data: bool = False,
    ) -> Iterator[ScrapedResource]:
        """
        crawl pages from a given starting url up to a given depth either using basic http or a provided browser client

        Args : 
            start_url (str): Url to start crawling from
            authenticator (SSOAuthenticator): class used for handling authenticatoin for web resources
            max_depth (int): max depth of links to descend from the start url
            selenium_scrape (bool): tracks whether or not the page should be scraped through selenium or not
            max_pages (int | None): cap on total pages to visit before stopping
            collect_page_data (bool): whether to store resources on the scraper instance

        Returns: Iterator[ScrapedResource]

        """

        if not self.enable_warnings:
            import urllib3
            urllib3.disable_warnings()
            
        depth = 0 
        self.visited_urls = set()
        self.seen_urls = set()
        self.page_data = []
        normalized_start_url = self._normalize_url(start_url)
        if not normalized_start_url:
            logger.error(f"Failed to crawl: {start_url}, could not normalize URL")
            return
        to_visit = [normalized_start_url]
        self.seen_urls.add(normalized_start_url)
        level_links = []
        pages_visited = 0

        base_hostname = urlparse(normalized_start_url).netloc
        logger.info(f"Base hostname for crawling: {base_hostname}")

        # session either stays none or becomes a requests.Session object if not selenium scraping
        session = None

        if selenium_scrape: # scrape page with pure selenium
            if browserclient is None: 
                logger.error(f"Failed to crawl: {start_url}, auth is needed but no browser clilent was passed through")
                return [] 
            browserclient.authenticate_and_navigate(normalized_start_url)

        elif not selenium_scrape and browserclient is not None: # use browser client for auth but scrape with http request
            session = requests.Session()
            cookies = browserclient.authenticate(normalized_start_url)
            if cookies is not None:
                for cookie_args in cookies:
                    cookie = requests.cookies.create_cookie(name=cookie_args['name'],
                                                            value=cookie_args['value'],
                                                            domain=cookie_args.get('domain'),
                                                            path=cookie_args.get('path', '/'),
                                                            expires=cookie_args.get('expires'),
                                                            secure=cookie_args.get('secure', False))
                    session.cookies.set_cookie(cookie)

        else: # pure html no browser client needed
            session = requests.Session()

        while to_visit and depth < max_depth:
            if max_pages is not None and pages_visited >= max_pages:
                logger.info(f"Reached max_pages={max_pages}; stopping crawl early.")
                break
            current_url = to_visit.pop(0)
            
            # Skip if we've already visited this URL
            if current_url in self.visited_urls:
                continue

            logger.info(f"Crawling depth {depth + 1}/{max_depth}: {current_url}")

            try:

                # grab the page content 
                if not selenium_scrape: 
                    assert (session is not None) # REMOVELATER
                    response = session.get(current_url, verify = self.verify_urls)
                    response.raise_for_status()
                else: 
                    assert (browserclient is not None) # REMOVELATER
                    browserclient.navigate_to(current_url, wait_time = 2)
                    response = browserclient.extract_page_data(current_url) # see the BrowserIntermediaryResult class to see what comes back here
                            
                
                # Mark as visited and store content
                pages_visited += 1
                new_links, resources = self.reap(response, current_url, selenium_scrape, browserclient)
                for resource in resources:
                    if collect_page_data:
                        self.page_data.append(resource)
                    yield resource
                        
                for link in new_links:
                    normalized_link = self._normalize_url(link)
                    if not normalized_link:
                        continue
                    if normalized_link in self.seen_urls:
                        continue
                    logger.info(f"Found new link: {normalized_link} (nv: {pages_visited})")
                    self.seen_urls.add(normalized_link)
                    level_links.append(normalized_link)

            except Exception as e:
                logger.info(f"Error crawling {current_url}: {e}")
                self._mark_visited(current_url)  # Mark as visited to avoid retrying           

            if not to_visit:
                to_visit.extend(level_links)
                level_links = []
                depth += 1
            
        logger.info(f"Crawling complete. Visited {pages_visited} pages.")
        return

    def _normalize_url(self, url: str) -> Optional[str]:
        if not url:
            return None

        normalized, _ = urldefrag(url)
        parsed = urlparse(normalized)
        if not parsed.scheme:
            return normalized
        return parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
        ).geturl()

    def _mark_visited(self, url: str) -> None:
        normalized = self._normalize_url(url)
        if not normalized:
            return
        self.visited_urls.add(normalized)
        self.seen_urls.add(normalized)

    def get_links_with_same_hostname(self, url: str, page_data: ScrapedResource):
        """Return all links on the page that share the same hostname as `url`. For now does not support PDFs"""

        base_url = self._normalize_url(url) or url
        base_hostname = urlparse(base_url).netloc
        links = set()
        a_tags = []
        
        if (page_data.suffix == "html"):
            soup = BeautifulSoup(page_data.content, "html.parser")
            a_tags = soup.find_all("a", href=True) 

        # how many  links found on the first level
        for tag in a_tags:
            full = urljoin(base_url, tag["href"])
            normalized = self._normalize_url(full)
            if not normalized:
                continue
            if urlparse(normalized).netloc == base_hostname:
                links.add(normalized)
        return list(links)
