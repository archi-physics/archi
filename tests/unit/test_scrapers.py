import time
import pytest
from src.data_manager.collectors.scrapers.scraper import LinkScraper
from tests.http.offline_router import OfflineRouter

# @pytest.mark.slow
@pytest.mark.routesets("twiki")
def test_link_scraper_basic_crawl(http_router: OfflineRouter):
    scraper = LinkScraper(delay=0)
    scraped = scraper.crawl_iter(
        "https://twiki.test/CMSPublic/SWGuide",
        browserclient=None,
        max_depth=2,
        selenium_scrape=False
    )
    assert len(list(scraped)) == 6

@pytest.mark.routesets("twiki")
def test_link_scraper_deep_crawl(http_router: OfflineRouter):
    scraper = LinkScraper(delay=0)
    scraped = scraper.crawl_iter(
        "https://twiki.test/CMSPublic/SWGuide",
        max_pages=100,
        browserclient=None,
        max_depth=10,
        selenium_scrape=False
    )
    assert len(list(scraped)) == 23

@pytest.mark.slow
@pytest.mark.routesets("twiki")
def test_link_scraper_delay(http_router: OfflineRouter):
    scraper = LinkScraper()

    start = time.perf_counter()
    scraped = list(scraper.crawl_iter(
        "https://twiki.test/CMSPublic/SWGuide",
        max_pages=100,
        browserclient=None,
        max_depth=10,
        selenium_scrape=False
    ))
    elapsed_time = time.perf_counter() - start
    assert elapsed_time >= 2.0

@pytest.mark.routesets("twiki", "deep_wiki")
def test_link_scraper_allowed_prefixes_and_sanitization(http_router: OfflineRouter):
    # If w/o allowed path regexes, denied path regexes we should get all valid links (23 links)
    scraper = LinkScraper(
        allowed_path_regexes=[".*Crab.*", ".*CRAB3.*", ".*WorkBook.*"],
        denied_path_regexes=["LeftBar", "diff"],
        delay=0
    )

    scraped_resources = list(scraper.crawl_iter(
        "https://twiki.test/CMSPublic/SWGuide",
        max_pages=100,
        browserclient=None,
        max_depth=10,
        selenium_scrape=False
    ))
    scraped_links = [link.url for link in scraped_resources]
    EXPECTED_LINKS = [
        "https://twiki.test/CMSPublic/SWGuideCrab",
        "https://twiki.test/CMSPublic/CRAB3AdvancedTutorial",
        "https://twiki.test/CMSPublic/CRAB3ConfigurationFile",
        "https://twiki.test/CMSPublic/CRAB3Commands",
        "https://twiki.test/CMSPublic/CRAB3FAQ",
        "https://twiki.test/CMSPublic/WorkBook",
        "https://twiki.test/CMSPublic/WorkBookCRAB3Tutorial",
        "https://twiki.test/CMSPublic/WorkBookGetAccount",
    ]
    assert len(scraped_links) == len(EXPECTED_LINKS)
    assert set(scraped_links) == set(EXPECTED_LINKS)