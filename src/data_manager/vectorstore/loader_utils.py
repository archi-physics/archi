from __future__ import annotations

from pathlib import Path
from typing import Optional, List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    PythonLoader,
)
from langchain_community.document_loaders.text import TextLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)


def select_loader(file_path: str | Path):
    """Return a document loader instance appropriate for the given path, or None.

    Mirrors the behavior used by VectorStoreManager.loader but is available to
    other modules so they can reuse the same loaders.
    """
    path = Path(file_path)
    _, file_extension = path.suffix, path.suffix
    file_extension = file_extension.lower()
    if file_extension in {".txt", ".c", ".C", ".sh", ".h", ".php", ".yaml", ".yml", ".json", ".csv", ".tsv", ".log", ".rst", ".md"}:
        return TextLoader(str(path))
    if file_extension == ".py":
        return PythonLoader(str(path))
    if file_extension in {".html", ".htm"}:
        return BSHTMLLoader(str(path), bs_kwargs={"features": "html.parser"})
    if file_extension == ".pdf":
        return PyPDFLoader(str(path))

    logger.error("No loader available for %s", path)
    return None


def load_doc_from_path(file_path: str | Path) -> Optional[Document]:

    path = Path(file_path)
    try:
        loader = select_loader(path)
        if loader is None:
            return None
        docs = loader.load()
        if docs:
            return docs[0]
        return None
    except Exception as exc:
        logger.warning("Failed to load document from %s: %s", file_path, exc)
        return None

def load_text_from_path(file_path: str | Path) -> Optional[str]:
    """Attempt to extract text from a file using an appropriate loader.

    For simple text-like files this will return the file contents. For other
    formats it will use the loader's .load() and concatenate page_content.
    Returns None if no loader is available or extraction fails.
    """
    path = Path(file_path)
    try:
        # For simple text files prefer direct read for speed and encoding handling
        if path.suffix.lower() in {".txt", ".md", ".rst", ".log", ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".html", ".htm"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        loader = select_loader(path)
        if loader is None:
            return None
        docs = loader.load()
        parts: List[str] = []
        for d in docs:
            content = getattr(d, "page_content", None)
            if content:
                parts.append(str(content))
        return "\n".join(parts) if parts else None
    except Exception as exc:
        logger.warning("Failed to extract text from %s: %s", file_path, exc)
        return None
