from typing import List
from functools import partial
from tests.http.offline_router import RouteTable, RouteValue

def make_links_html(base_url: str, paths: List[str]) -> RouteValue:
    anchors = "".join(f'<a href="{base_url}/{p}">{p}</a>' for p in paths)
    return (200, f"<!doctype html><html><body>{anchors}</body></html>")

def twiki_routes() -> RouteTable:
    """Route table: URL -> (status_code, body)."""
    base = "https://twiki.test/CMSPublic"
    twiki = partial(make_links_html, base)
    return {
        f"{base}/SWGuide": twiki(
            ["SWGuide", "SWGuideDQM", "SWGuideReco", "SWGuideHiggs", "SWGuideMuons", "SWGuideCrab"]
        ),
        f"{base}/SWGuideDQM": twiki(
            ["SWGuideDQMConverters", "SWGuideValidationTableDQM", "SWGuideTauDQM", "SWGuideHLTDQM"]
        ),
        f"{base}/SWGuideReco": twiki(
            ["SWGuideEcalReco", "SWGuideHcalReco", "SWGuideVertexReco"]
        ),
        f"{base}/SWGuideCrab": twiki(
            # Example: un-sanitized links, that should be cleaned, by the scraper
            ["CRAB3AdvancedTutorial?rev1=196;rev2=195", "CRAB3ConfigurationFile", "CRAB3Commands", "CRAB3FAQ", "WorkBook"]
        ),
        f"{base}/WorkBook": twiki([
            # Example: un-sanitized links, that should be cleaned, by the scraper
            "WorkBookCRAB3Tutorial?t=1771422672;nowysiwyg=1", "WorkBookGetAccount",
            # Example: that should be discarded, by the scraper with Negative pattern ^LeftBar$
            "WorkBookCRAB3TutorialLeftBar", "WorkBookCRAB3TutorialLeftBarLeftBar", "WorkBookCRAB3TutorialLeftBarLeftBarLeftBar"   
        ]),
        f"{base}/SWGuideHiggs": (200, "SWGuideHiggs"),
        f"{base}/SWGuideMuons": (200, "SWGuideMuons"),
        f"{base}/SWGuideDQMConverters": (200, "SWGuideDQMConverters"),
        f"{base}/SWGuideValidationTableDQM": (200, "SWGuideValidationTableDQM"),
        f"{base}/SWGuideTauDQM": (200, "SWGuideTauDQM"),
        f"{base}/SWGuideHLTDQM": (200, "SWGuideHLTDQM"),
        f"{base}/SWGuideEcalReco": (200, "SWGuideEcalReco"),
        f"{base}/SWGuideHcalReco": (200, "SWGuideHcalReco"),
        f"{base}/SWGuideVertexReco": (200, "SWGuideVertexReco"),
        f"{base}/WorkBookGetAccount": (200, "WorkBookGetAccount"),
        f"{base}/WorkBookCRAB3Tutorial": (200, "WorkBookCRAB3Tutorial"),
        # Example: that should be discarded, by the scraper with Negative pattern ^LeftBar$
        f"{base}/WorkBookCRAB3TutorialLeftBar": (200, "WorkBookCRAB3TutorialLeftBar"),
        f"{base}/WorkBookCRAB3TutorialLeftBarLeftBar": (200, "WorkBookCRAB3TutorialLeftBarLeftBar"),
        f"{base}/WorkBookCRAB3TutorialLeftBarLeftBarLeftBar": (200, "WorkBookCRAB3TutorialLeftBarLeftBarLeftBar"),
        # Example: Nested Deep Wiki link that should be discarded, by the scraper due to have different hostname, also not matching the allowed path regexes
        f"{base}/CRAB3AdvancedTutorial": (200, "<!doctype html><html><body><title>CRAB3AdvancedTutorial</title><a href='https://deepwiki.test/dmwm/CRABServer/1.1-system-architecture'>Deep Wiki Link</a></body></html>"),
        f"{base}/CRAB3ConfigurationFile": (200, "CRAB3ConfigurationFile"),
        f"{base}/CRAB3Commands": (200, "CRAB3Commands"),
        f"{base}/CRAB3FAQ": (200, "CRAB3FAQ"),
        # Example: that should be discarded, by the scraper since has different hostname
        "https://example.test/missing": (404, "404 Not Found"),
    }

def deep_wiki_routes() -> RouteTable:
    base = "https://deepwiki.test/dmwm/CRABServer"
    deepwiki = partial(make_links_html, base)
    return {
        f"{base}/1-overview": deepwiki(["1.1-system-architecture", "1.2-key-concepts-and-terminology"]),
        f"{base}/1.1-system-architecture": (200, "1.1-system-architecture"),
        f"{base}/1.2-key-concepts-and-terminology": (200, "1.2-key-concepts-and-terminology"),
    }
