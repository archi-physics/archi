import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from git import Repo
from mkdocs.utils.yaml import yaml_load

from src.utils.config_access import get_global_config
from src.data_manager.collectors.scrapers.scraped_resource import ScrapedResource
from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.data_manager.collectors.scrapers.scraper_manager import \
        ScraperManager

global_config = get_global_config()

class GitScraper:
    """Scraper integration that clones Git repositories and indexes MkDocs sites and code files."""

    def __init__(self, manager: "ScraperManager", git_config: Optional[Dict[str, Any]] = None) -> None:
        self.manager = manager
        self.config = git_config or {}

        # where we clone our repos to
        self.data_path = global_config["DATA_PATH"]
        self.git_dir = Path(self.data_path) / "raw_git_repos"
        self.git_dir.mkdir(parents=True, exist_ok=True)

        self.code_suffixes = {
            suffix.lower()
            for suffix in (
                self.config.get(
                    "code_suffixes",
                    [
                        ".py",
                        ".js",
                        ".ts",
                        ".tsx",
                        ".jsx",
                        ".java",
                        ".go",
                        ".rs",
                        ".c",
                        ".cpp",
                        ".h",
                        ".hpp",
                        ".sh",
                        ".sql",
                        ".json",
                        ".yaml",
                        ".yml",
                        ".toml",
                        ".md",
                        ".txt",
                    ],
                )
                or []
            )
        }
        self.exclude_dirs = {
            dir_name
            for dir_name in (
                self.config.get(
                    "exclude_dirs",
                    [
                        ".git",
                        "node_modules",
                        ".venv",
                        "venv",
                        "__pycache__",
                        ".idea",
                        ".vscode",
                        "dist",
                        "build",
                    ],
                )
                or []
            )
        }
        self.max_file_size_bytes = int(self.config.get("max_file_size_bytes", 1_000_000))

        self.git_username = read_secret("GIT_USERNAME")
        self.git_token = read_secret("GIT_TOKEN")
        self._credentials_available = bool(self.git_username and self.git_token)
        if not self._credentials_available:
            logger.info("No git credentials supplied; git scraping will not work for private repositories.")

    def collect(self, git_urls: List[str]) -> List[ScrapedResource]:
        if not git_urls:
            logger.warning("No git URLs provided for scraping; skipping git scraper.")
            return []

        harvested: List[ScrapedResource] = []

        for url in git_urls:
            try:
                repo_info = self._prepare_repository(url)
            except ValueError as exc:
                logger.info(f"{exc}")
                continue
            except Exception as exc:
                logger.error(f"Failed to clone {url}: {exc}")
                continue

            try:
                harvested.extend(self._harvest_repository(repo_info))
            finally:
                shutil.rmtree(repo_info["repo_path"], ignore_errors=True)

        if harvested:
            logger.info("Git scraping was completed successfully")

        return harvested

    def _prepare_repository(self, url: str) -> Dict[str, Any]:
        url_dict = self._parse_url(url)
        repo_path = self._clone_repo(url_dict)
        mkdocs_site_url = self._read_mkdocs_site_url(repo_path)
        ref = self._determine_ref(repo_path, url_dict["branch"])
        web_base_url = self._compute_web_base_url(url_dict["original_url"])

        return {
            "repo_path": repo_path,
            "repo_name": url_dict["repo_name"],
            "mkdocs_site_url": mkdocs_site_url,
            "ref": ref,
            "web_base_url": web_base_url,
        }

    def _harvest_repository(self, repo_info: Dict[str, Any]) -> List[ScrapedResource]:
        resources: List[ScrapedResource] = []
        resources.extend(self._harvest_mkdocs(repo_info))
        resources.extend(self._harvest_code(repo_info))
        return resources

    def _harvest_mkdocs(self, repo_info: Dict[str, Any]) -> List[ScrapedResource]:
        repo_path = repo_info["repo_path"]
        mkdocs_site_url = repo_info["mkdocs_site_url"]
        base_url = repo_info["web_base_url"]
        ref = repo_info["ref"]
        docs_dir = repo_path / "docs"
        if not docs_dir.exists():
            logger.info(f"Skipping MkDocs harvesting for {repo_path}; missing docs directory")
            return []

        resources: List[ScrapedResource] = []
        parent_repo = repo_info["repo_name"]
        used_blob_links = False
        for markdown_path in docs_dir.rglob("*.md"):
            if mkdocs_site_url:
                current_url = mkdocs_site_url + markdown_path.relative_to(docs_dir).with_suffix("").as_posix()
            else:
                current_url = self._build_blob_url(base_url, ref, markdown_path.relative_to(repo_path))
                used_blob_links = True
            logger.info(f"Indexing Git doc: {current_url}")
            text_content = markdown_path.read_text(encoding="utf-8")
            relative_path = Path(parent_repo) / markdown_path.relative_to(repo_path)
            resource = ScrapedResource(
                url=current_url,
                content=text_content,
                suffix=markdown_path.suffix.lstrip(".") or "txt",
                source_type="git",
                metadata={
                    "repo_path": str(markdown_path.relative_to(repo_path)),
                    "title": markdown_path.stem.replace("_", " ").replace("-", " ").title(),
                    "parent": parent_repo,
                },
                file_name=markdown_path.name,
                relative_path=str(relative_path),
            )
            if resource.content:
                resources.append(resource)
            else:
                logger.info(f"Resource {current_url} is empty. Skipping...")

        if used_blob_links and not mkdocs_site_url:
            logger.info(f"Used repository blob URLs for MkDocs content in {repo_path} (site_url missing)")

        return resources

    def _harvest_code(self, repo_info: Dict[str, Any]) -> List[ScrapedResource]:
        repo_path = repo_info["repo_path"]
        ref = repo_info["ref"]
        base_url = repo_info["web_base_url"]
        repo_name = repo_info["repo_name"]

        resources: List[ScrapedResource] = []
        for file_path in self._iter_code_files(repo_path):
            logger.debug(file_path)
            rel_path = file_path.relative_to(repo_path)

            # avoid overlap wtih _harvest_mkdocs
            if rel_path.parts and rel_path.parts[0] == "docs" and file_path.suffix.lower() == ".md":
                continue

            try:
                if file_path.stat().st_size > self.max_file_size_bytes:
                    logger.warning(f"Skipping {file_path} due to file size")
                    continue
            except OSError:
                continue

            if not self._is_allowed_suffix(file_path):
                logger.warning(f"Skipping {file_path} due to disallowed suffix")
                continue

            if self._looks_binary(file_path):
                logger.warning(f"Skipping {file_path} due to likely binary content")
                continue

            try:
                text_content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not text_content.strip():
                continue

            resource_url = self._build_blob_url(base_url, ref, rel_path)
            relative_path = Path(repo_name) / rel_path
            resource = ScrapedResource(
                url=resource_url,
                content=text_content,
                suffix=file_path.suffix.lstrip("."),
                source_type="git",
                metadata={
                    "repo_path": str(rel_path),
                    "parent": repo_name,
                    "ref": ref,
                },
                file_name=file_path.name,
                relative_path=str(relative_path),
            )
            resources.append(resource)

        return resources

    def _parse_url(self, url: str) -> dict:
        branch_name = None

        regex_repo_name = r"(?:github|gitlab)\.[\w.]+\/[^\/]+\/([\w.-]+)(?:\.git|\/|$)"
        match = re.search(regex_repo_name, url, re.IGNORECASE)
        if not match:
            raise ValueError(f"The git url {url} does not match the expected format.")

        repo_name = match.group(1)

        if "gitlab" in url:
            clone_from_url = url.replace("gitlab", f"{self.git_username}:{self.git_token}@gitlab")
        elif "github" in url:
            clone_from_url = url.replace("github", f"{self.git_username}:{self.git_token}@github")
        else:
            raise ValueError(f"Unsupported git host in url {url}")

        branch_split = re.split(r"/(?:-/)?tree/", clone_from_url, maxsplit=1)
        if len(branch_split) > 1:
            branch_name = branch_split[1].strip("/") or None
            clone_from_url = branch_split[0].rstrip("/")

        return {
            "original_url": url,
            "clone_url": clone_from_url,
            "repo_name": repo_name,
            "branch": branch_name,
        }

    def _clone_repo(self, url_dict: dict) -> Path:
        clone_url = url_dict["clone_url"]
        branch = url_dict["branch"]
        repo_name = url_dict["repo_name"]

        logger.info(f"Cloning repository {repo_name}...")

        repo_path = self.git_dir / repo_name
        if branch is None:
            Repo.clone_from(clone_url, repo_path)
        else:
            Repo.clone_from(clone_url, repo_path, branch=branch)

        return repo_path

    def _read_mkdocs_site_url(self, repo_path: Path) -> Optional[str]:
        mkdocs_file = repo_path / "mkdocs.yml"
        if not mkdocs_file.exists():
            return None
        try:
            with mkdocs_file.open("r") as file:
                data = yaml_load(file)
            site_url = data.get("site_url")
            if not site_url:
                return None
            return site_url if site_url.endswith("/") else site_url + "/"
        except Exception:
            logger.info(f"Could not read mkdocs.yml in {repo_path}")
            return None

    def _compute_web_base_url(self, original_url: str) -> str:
        sanitized = re.sub(r"//[^@/]+@", "//", original_url)
        sanitized = re.split(r"/(?:-/)?tree/", sanitized, maxsplit=1)[0]
        if sanitized.endswith(".git"):
            sanitized = sanitized[:-4]
        return sanitized.rstrip("/")

    def _determine_ref(self, repo_path: Path, requested_branch: Optional[str]) -> str:
        if requested_branch:
            return requested_branch
        repo: Optional[Repo] = None
        try:
            repo = Repo(repo_path)
            return repo.active_branch.name
        except Exception:
            try:
                repo = repo or Repo(repo_path)
                return repo.head.commit.hexsha[:7]
            except Exception:
                return "main"

    def _iter_code_files(self, repo_path: Path):
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for filename in files:
                file_path = Path(root) / filename
                yield file_path

    def _is_allowed_suffix(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.code_suffixes

    def _looks_binary(self, file_path: Path) -> bool:
        try:
            with file_path.open("rb") as file:
                sample = file.read(8000)
            return b"\0" in sample
        except Exception:
            return True

    def _build_blob_url(self, base_url: str, ref: str, rel_path: Path) -> str:
        base = base_url.rstrip("/")
        rel = rel_path.as_posix()
        if "gitlab" in base:
            return f"{base}/-/blob/{ref}/{rel}"
        return f"{base}/blob/{ref}/{rel}"
