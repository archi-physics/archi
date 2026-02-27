import requests_mock
from dataclasses import dataclass
from typing import Mapping, Tuple, Callable

RouteValue = Tuple[int, str]
RouteTable = Mapping[str, RouteValue]
RouteFactory = Callable[[], RouteTable]

@dataclass
class OfflineRouter:
    mocker: requests_mock.Mocker

    def add_routes(self, routes: RouteTable) -> None:
        for url, (status, body) in routes.items():
            self.mocker.get(
                url,
                text=body,
                status_code=status,
                headers={"Content-Type": "text/html"},
            )