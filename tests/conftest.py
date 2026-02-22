import pytest
import requests_mock

from tests.http.offline_router import OfflineRouter
from tests.http.routes_twikis import twiki_routes, deep_wiki_routes

ROUTESETS = {
    "twiki": twiki_routes,
    "deep_wiki": deep_wiki_routes,
}

@pytest.fixture
def http_router(request):
    with requests_mock.Mocker(real_http=False) as m:
        router = OfflineRouter(m)

        marker = request.node.get_closest_marker("routesets")
        names = marker.args if marker else ("twiki",)  # default

        for name in names:
            router.add_routes(ROUTESETS[name]())

        yield router
