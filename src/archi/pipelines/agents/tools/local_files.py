from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.utils.logging import get_logger
from src.utils.env import read_secret
from src.archi.pipelines.agents.tools.base import require_tool_permission

logger = get_logger(__name__)


class RemoteCatalogClient:
    """HTTP client for the data-manager catalog API."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        host_mode: Optional[bool] = None,
        hostname: Optional[str] = None,
        port: int = 7871,
        external_port: Optional[int] = None,
        timeout: float = 10.0,
    ):
        host_mode_flag = self._resolve_host_mode(host_mode)

        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            host = hostname or ("localhost" if host_mode_flag else "data-manager")
            final_port = external_port if host_mode_flag and external_port else port
            self.base_url = f"http://{host}:{final_port}"
        self.timeout = timeout

    @classmethod
    def from_deployment_config(cls, config: Optional[Dict[str, object]]) -> "RemoteCatalogClient":
        """Create a client using the standard archi deployment config structure."""
        cfg = config or {}
        services_cfg = cfg.get("services", {}) if isinstance(cfg, dict) else {}
        data_manager_cfg = services_cfg.get("data_manager", {}) if isinstance(services_cfg, dict) else {}
        auth_cfg = data_manager_cfg.get("auth", {}) if isinstance(data_manager_cfg, dict) else {}

        return cls(
            base_url=data_manager_cfg.get("base_url"),
            host_mode=cfg.get("host_mode"),
            hostname=data_manager_cfg.get("hostname") or data_manager_cfg.get("host"),
            port=data_manager_cfg.get("port", 7871),
            external_port=data_manager_cfg.get("external_port"),
        )

    @staticmethod
    def _resolve_host_mode(host_mode: Optional[bool]) -> bool:
        if host_mode is None:
            env_host_mode = (
                os.environ.get("HOST_MODE")
                or os.environ.get("HOSTMODE")
                or os.environ.get("ARCHI_HOST_MODE")
            )
            return str(env_host_mode).lower() in {"1", "true", "yes", "on"}
        return bool(host_mode)


    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        search_content: bool = True,
        regex: bool = False,
        case_sensitive: bool = False,
        max_matches_per_file: Optional[int] = None,
        before: int = 0,
        after: int = 0,
        mode: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        params: Dict[str, object] = {
            "q": query,
            "limit": limit,
            "search_content": str(search_content).lower(),
        }
        if mode:
            params["mode"] = mode
        if search_content:
            params["regex"] = str(regex).lower()
            params["case_sensitive"] = str(case_sensitive).lower()
            params["before"] = before
            params["after"] = after
            if max_matches_per_file is not None:
                params["max_matches_per_file"] = max_matches_per_file
        resp = requests.get(
            f"{self.base_url}/api/catalog/search",
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", []) or []

    def get_document(self, resource_hash: str, *, max_chars: int = 4000) -> Optional[Dict[str, object]]:
        resp = requests.get(
            f"{self.base_url}/api/catalog/document/{resource_hash}",
            params={"max_chars": max_chars},
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def schema(self) -> Dict[str, object]:
        resp = requests.get(
            f"{self.base_url}/api/catalog/schema",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


def _render_metadata_preview(metadata: Optional[Dict[str, object]], *, max_chars: int = 800) -> str:
    if not metadata:
        return "(no metadata)"
    # render key: value lines
    lines: List[str] = []
    for key, value in sorted(metadata.items()):
        lines.append(f"{key}: {value}")
    meta_str = "\n".join(lines)
    if len(meta_str) > max_chars:
        return meta_str[: max_chars - 12].rstrip() + "\n... (truncated)"
    return meta_str


def _format_files_for_llm(hits: List[Tuple[str, Path, Optional[Dict[str, object]], str]], *, max_meta_chars: int = 800, max_content_chars: int = 800) -> str:
    if not hits:
        return "No local files matched that search query."
    lines: List[str] = []
    for idx, (resource_hash, path, metadata, snippet) in enumerate(hits, start=1):
        meta_preview = _render_metadata_preview(metadata, max_chars=max_meta_chars)
        content = snippet.strip() if snippet else ""
        if len(content) > max_content_chars:
            content = content[: max_content_chars - 3].rstrip() + "..."
        lines.append(
            f"[{idx}] {path} (hash={resource_hash})\nMetadata:\n{meta_preview}\n\nSnippet:\n{content}"
        )
    return "\n\n".join(lines)


def _format_grep_hits(hits: List[Dict[str, object]]) -> str:
    if not hits:
        return "No local files matched that search query."
    lines: List[str] = []
    for idx, item in enumerate(hits, start=1):
        resource_hash = item.get("hash")
        path = item.get("path", "")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        display_name = metadata.get("display_name") or metadata.get("file_name") or ""
        source_type = metadata.get("source_type") or ""
        meta_line = " ".join(part for part in [source_type, display_name] if part)
        lines.append(f"[{idx}] {path} (hash={resource_hash})")
        if meta_line:
            lines.append(f"Meta: {meta_line}")
        matches = item.get("matches") if isinstance(item.get("matches"), list) else []
        if matches:
            for match in matches:
                line_no = match.get("line", "?")
                text = (match.get("text") or "").strip()
                lines.append(f"L{line_no}: {text}")
                before_lines = match.get("before") if isinstance(match.get("before"), list) else []
                after_lines = match.get("after") if isinstance(match.get("after"), list) else []
                for ctx in before_lines:
                    lines.append(f"B: {ctx}")
                for ctx in after_lines:
                    lines.append(f"A: {ctx}")
        else:
            snippet = item.get("snippet") or ""
            if snippet:
                lines.append(f"Snippet: {snippet.strip()}")
    return "\n".join(lines)


def _collect_snippet(text: str, match: re.Match, *, window: int = 240) -> str:
    start = max(match.start() - window, 0)
    end = min(match.end() + window, len(text))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    excerpt = text[start:end].replace("\n", " ")
    return f"{prefix}{excerpt}{suffix}"


def create_file_search_tool(
    catalog: RemoteCatalogClient,
    *,
    name: str = "search_local_files",
    description: Optional[str] = None,
    max_results: int = 3,
    window: int = 240,
    store_docs: Optional[Callable[[str, Sequence[Path]], None]] = None,
    required_permission: Optional[str] = None,
) -> Callable[[str], str]:
    """Create a LangChain tool that performs keyword search in catalogued files.
    
    Args:
        catalog: The RemoteCatalogClient instance.
        name: The name of the tool.
        description: Human-readable description of the tool.
        max_results: Maximum number of results to return.
        window: Context window size for snippets.
        store_docs: Optional callback to store retrieved documents.
        required_permission: Optional RBAC permission required to use this tool.
            If None, no permission check is performed (allow all).
    """

    _default_description = (
        "Grep-like search over local document contents only (not filenames/paths).\n"
        "Input: query (string), regex=false, case_sensitive=false, max_results_override=None, "
        "max_matches_per_file=3, before=0, after=0.\n"
        "Output: lines grouped by file with hash/path and matching line numbers, plus context lines.\n"
        "Example input: \"timeout error\" (regex=false)."
    )
    tool_description = (
        description
        or _default_description
    )

    @tool(name, description=tool_description)
    @require_tool_permission(required_permission)
    def _search_local_files(
        query: str,
        regex: bool = False,
        case_sensitive: bool = False,
        max_results_override: Optional[int] = None,
        max_matches_per_file: int = 3,
        before: int = 0,
        after: int = 0,
    ) -> str:
        if not query.strip():
            return "Please provide a non-empty search query."

        hits: List[Dict[str, object]] = []
        docs: List[Document] = []
        limit = max_results_override or max_results

        try:
            results = catalog.search(
                query.strip(),
                limit=limit,
                search_content=True,
                regex=regex,
                case_sensitive=case_sensitive,
                max_matches_per_file=max_matches_per_file,
                before=before,
                after=after,
                mode="grep",
            )
        except Exception as exc:
            logger.warning("Catalog search failed: %s", exc)
            return "Catalog search failed."

        for item in results:
            hits.append(item)

        if store_docs and hits:
            for item in hits:
                try:
                    resource_hash = item.get("hash")
                    doc_payload = catalog.get_document(resource_hash, max_chars=4000) or {}
                    text = doc_payload.get("text") or ""
                    doc_meta = doc_payload.get("metadata") or item.get("metadata") or {}
                    docs.append(Document(page_content=text, metadata=doc_meta))
                except Exception:
                    continue

        if store_docs:
            store_docs(f"{name}: {query}", docs)

        return _format_grep_hits(hits)

    return _search_local_files


def _flatten_metadata(data: Dict[str, object], prefix: str = "") -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_metadata(value, prefix=full_key))
        else:
            flattened[full_key] = "" if value is None else str(value)
    return flattened


def create_metadata_search_tool(
    catalog: RemoteCatalogClient,
    *,
    name: str = "search_metadata_index",
    description: Optional[str] = None,
    max_results: int = 5,
    store_docs: Optional[Callable[[str, Sequence[Path]], None]] = None,
    required_permission: Optional[str] = None,
) -> Callable[[str], str]:
    """Create a LangChain tool to search resource metadata catalogues.
    
    Args:
        catalog: The RemoteCatalogClient instance.
        name: The name of the tool.
        description: Human-readable description of the tool.
        max_results: Maximum number of results to return.
        store_docs: Optional callback to store retrieved documents.
        required_permission: Optional RBAC permission required to use this tool.
            If None, no permission check is performed (allow all).
    """

    tool_description = (
        description
        or (
            "Search document metadata stored in PostgreSQL (tickets, git, local files).\n"
            "Input: query string with key:value filters; filters are exact matches and ANDed within a group, OR across groups.\n"
            "Canonical filter keys with examples: "
            "source_type:ticket | ticket_id:CMSPROD-1234 | display_name:\"Release Notes\" | "
            "relative_path:docs/readme.md | file_path:/data/foo.txt | url:github.com/org/repo | "
            "git_repo:org/repo | suffix:.py | created_at:2024-11-01 | ingested_at:2024-11-02.\n"
            "Legacy keys resource_type/resource_id are aliased automatically. Free text matches display_name/url/paths when used without filters.\n"
            "Output: list of matches with hash, path, metadata, and a short snippet."
        )
    )

    @tool(name, description=tool_description)
    @require_tool_permission(required_permission)
    def _search_metadata(query: str) -> str:
        if not query.strip():
            return "Please provide a non-empty search query."

        hits: List[Tuple[str, Path, Optional[Dict[str, object]], str]] = []
        docs: List[Document] = []

        try:
            results = catalog.search(query.strip(), limit=max_results, search_content=False)
        except Exception as exc:
            logger.warning("Metadata search failed: %s", exc)
            return "Metadata search failed."

        for item in results:
            resource_hash = item.get("hash")
            path = Path(item.get("path", "")) if item.get("path") else Path("")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            snippet = item.get("snippet") or ""
            hits.append((resource_hash, path, metadata, snippet))
            if len(hits) >= max_results:
                break

        if store_docs and hits:
            for resource_hash, path, metadata, _ in hits:
                try:
                    doc_payload = catalog.get_document(resource_hash, max_chars=4000) or {}
                    text = doc_payload.get("text") or ""
                    doc_meta = doc_payload.get("metadata") or metadata or {}
                    docs.append(Document(page_content=text, metadata=doc_meta))
                except Exception:
                    continue

        if store_docs:
            store_docs(f"{name}: {query}", docs)

        return _format_files_for_llm(hits)

    return _search_metadata


def create_metadata_schema_tool(
    catalog: RemoteCatalogClient,
    *,
    name: str = "list_metadata_schema",
    description: Optional[str] = None,
    required_permission: Optional[str] = None,
) -> Callable[[], str]:
    """Create a tool that returns supported metadata keys and distinct values.
    
    Args:
        catalog: The RemoteCatalogClient instance.
        name: The name of the tool.
        description: Human-readable description of the tool.
        required_permission: Optional RBAC permission required to use this tool.
            If None, no permission check is performed (allow all).
    """

    tool_description = (
        description
        or (
            "Return metadata schema hints: supported keys, distinct source_type values, and suffixes. "
            "Use this to learn which key:value filters are available."
        )
    )

    @tool(name, description=tool_description)
    @require_tool_permission(required_permission)
    def _schema_tool() -> str:
        try:
            payload = catalog.schema()
        except Exception as exc:
            logger.warning("Metadata schema fetch failed: %s", exc)
            return "Metadata schema fetch failed."
        keys = payload.get("keys") or []
        source_types = payload.get("source_types") or []
        suffixes = payload.get("suffixes") or []
        return (
            "Supported keys: " + ", ".join(keys) + "\n"
            "source_type values: " + (", ".join(source_types) or "none") + "\n"
            "suffix values: " + (", ".join(suffixes) or "none")
        )

    return _schema_tool


def create_document_fetch_tool(
    catalog: RemoteCatalogClient,
    *,
    name: str = "fetch_catalog_document",
    description: Optional[str] = None,
    default_max_chars: int = 4000,
    required_permission: Optional[str] = None,
) -> Callable[..., str]:
    """Create a LangChain tool to fetch a full document by resource hash.
    
    Args:
        catalog: The RemoteCatalogClient instance.
        name: The name of the tool.
        description: Human-readable description of the tool.
        default_max_chars: Default maximum characters to return.
        required_permission: Optional RBAC permission required to use this tool.
            If None, no permission check is performed (allow all).
    """

    tool_description = (
        description
        or (
            "Fetch a catalog document by resource hash after a search hit.\n"
            "Input: resource_hash (string), max_chars=4000.\n"
            "Output: path, metadata, and document text (truncated).\n"
            "Example input: \"abcd1234\"."
        )
    )

    @tool(name, description=tool_description)
    @require_tool_permission(required_permission)
    def _fetch_document(resource_hash: str, max_chars: int = default_max_chars) -> str:
        if not resource_hash.strip():
            return "Please provide a non-empty resource hash."

        try:
            doc_payload = catalog.get_document(resource_hash.strip(), max_chars=max_chars) or {}
        except Exception as exc:
            logger.warning("Document fetch failed: %s", exc)
            return "Document fetch failed."

        if not doc_payload:
            return "Document not found."

        path = doc_payload.get("path") or ""
        metadata = doc_payload.get("metadata") if isinstance(doc_payload.get("metadata"), dict) else {}
        text = doc_payload.get("text") or ""
        meta_preview = _render_metadata_preview(metadata)

        return (
            f"Path: {path}\n"
            f"Metadata:\n{meta_preview}\n\n"
            f"Content:\n{text}"
        )

    return _fetch_document


__all__ = [
    "RemoteCatalogClient",
    "create_retriever_tool",
    "create_file_search_tool",
    "create_metadata_search_tool",
    "create_document_fetch_tool",
]
