#!/usr/bin/env python3
"""
Utility helpers for deleting Docker Hub tags used by CI workflows.

Supports deleting a single tag across multiple repositories and pruning tags
matching a prefix while keeping explicit exceptions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


DEFAULT_REPOSITORIES: Sequence[str] = (
    "a2rchi/a2rchi-python-base",
    "a2rchi/a2rchi-pytorch-base",
)


class DockerHubError(RuntimeError):
    """Raised when the Docker Hub API returns an unexpected error."""


@dataclass
class DockerHubClient:
    """Minimal Docker Hub API client for tag management."""

    username: str
    password: str
    token: Optional[str] = None

    API_ROOT = "https://hub.docker.com/v2/"

    def __post_init__(self) -> None:
        if not self.username or not self.password:
            raise DockerHubError("Docker Hub credentials are required.")
        self.token = self._login()

    def _login(self) -> str:
        payload = json.dumps({"username": self.username, "password": self.password}).encode("utf-8")
        request = Request(
            urljoin(self.API_ROOT, "users/login/"),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise DockerHubError(f"Failed to authenticate with Docker Hub: {exc}") from exc
        except URLError as exc:
            raise DockerHubError(f"Unable to reach Docker Hub: {exc}") from exc

        token = data.get("token")
        if not token:
            raise DockerHubError("Docker Hub login response did not include a token.")
        return token

    def _request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        url = urljoin(self.API_ROOT, path)
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(url, method=method, headers={"Authorization": f"JWT {self.token}"})
        try:
            with urlopen(request) as response:
                body = response.read()
                if not body:
                    return {}
                return json.loads(body.decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404 and method == "DELETE":
                # 404s during deletion imply the tag is already gone. Treat as success.
                return {}
            raise DockerHubError(f"Docker Hub API error ({exc.code}): {exc.reason}") from exc
        except URLError as exc:
            raise DockerHubError(f"Network error contacting Docker Hub: {exc}") from exc

    def delete_tag(self, repository: str, tag: str) -> None:
        path = f"repositories/{repository}/tags/{tag}/"
        self._request("DELETE", path)

    def list_tags(self, repository: str) -> List[str]:
        tags: List[str] = []
        next_path: Optional[str] = f"repositories/{repository}/tags/"
        while next_path:
            response = self._request("GET", next_path)
            tags.extend(result["name"] for result in response.get("results", []))
            next_url = response.get("next")
            if next_url:
                # The API returns an absolute URL; convert to a relative path for consistency.
                next_path = next_url.replace(self.API_ROOT, "", 1)
            else:
                next_path = None
        return tags


def resolve_credentials(username: Optional[str], password: Optional[str]) -> tuple[str, str]:
    resolved_username = username or os.environ.get("DOCKERHUB_USERNAME")
    resolved_password = password or os.environ.get("DOCKERHUB_TOKEN") or os.environ.get("DOCKERHUB_PASSWORD")
    if not resolved_username or not resolved_password:
        raise DockerHubError(
            "Docker Hub credentials are required. Provide --username/--password or set "
            "DOCKERHUB_USERNAME and DOCKERHUB_TOKEN."
        )
    return resolved_username, resolved_password


def delete_single_tag(client: DockerHubClient, repositories: Sequence[str], tag: str) -> None:
    any_deleted = False
    for repo in repositories:
        print(f"Deleting tag '{tag}' from {repo}...")
        client.delete_tag(repo, tag)
        any_deleted = True
    if any_deleted:
        print("Tag deletion complete.")


def prune_by_prefix(client: DockerHubClient, repositories: Sequence[str], prefix: str, keep: Set[str]) -> None:
    for repo in repositories:
        keep_display = ", ".join(sorted(keep)) or "none"
        print(f"Pruning tags for {repo} with prefix '{prefix}' (keeping: {keep_display}).")
        tags = client.list_tags(repo)
        to_delete = sorted(tag for tag in tags if tag.startswith(prefix) and tag not in keep)
        if not to_delete:
            print(f"No tags to delete for {repo}.")
            continue
        for tag in to_delete:
            print(f"Deleting {repo}:{tag}")
            client.delete_tag(repo, tag)
        print(f"Deleted {len(to_delete)} tag(s) for {repo}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Docker Hub tags for archi images.")
    parser.add_argument("--username", help="Docker Hub username (defaults to DOCKERHUB_USERNAME env var).")
    parser.add_argument(
        "--password",
        help="Docker Hub password or token (defaults to DOCKERHUB_TOKEN or DOCKERHUB_PASSWORD env vars).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    delete_parser = subparsers.add_parser("delete-tag", help="Delete a single tag across repositories.")
    delete_parser.add_argument("--tag", required=True, help="Tag name to delete.")
    delete_parser.add_argument(
        "--repositories",
        nargs="+",
        default=list(DEFAULT_REPOSITORIES),
        help="Repositories to target (default: %(default)s).",
    )

    prune_parser = subparsers.add_parser(
        "prune-prefix", help="Delete tags starting with a prefix while keeping specified tags."
    )
    prune_parser.add_argument("--prefix", required=True, help="Prefix used to select tags for deletion.")
    prune_parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Tag to keep. Repeat for multiple tags. (default: keep none)",
    )
    prune_parser.add_argument(
        "--repositories",
        nargs="+",
        default=list(DEFAULT_REPOSITORIES),
        help="Repositories to target (default: %(default)s).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        username, password = resolve_credentials(args.username, args.password)
        client = DockerHubClient(username=username, password=password)

        if args.command == "delete-tag":
            delete_single_tag(client, args.repositories, args.tag)
        elif args.command == "prune-prefix":
            keep_tags = set(args.keep or [])
            prune_by_prefix(client, args.repositories, args.prefix, keep_tags)
        else:
            parser.error(f"Unknown command: {args.command}")
    except DockerHubError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
