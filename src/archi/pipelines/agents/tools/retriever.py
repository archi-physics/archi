from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.utils.logging import get_logger
from src.archi.pipelines.agents.tools.base import require_tool_permission

logger = get_logger(__name__)


def _normalize_results(
    results: Iterable[object],
) -> Sequence[Tuple[Document, Optional[float]]]:
    """Coerce retriever outputs into (Document, score) tuples."""
    normalized: list[Tuple[Document, Optional[float]]] = []
    for item in results:
        if isinstance(item, Document):
            normalized.append((item, None))
        elif (
            isinstance(item, tuple)
            and len(item) >= 2
            and isinstance(item[0], Document)
        ):
            normalized.append((item[0], item[1]))
    return normalized


def _format_documents_for_llm(
    docs: Sequence[Tuple[Document, Optional[float]]],
    *,
    max_documents: int,
    max_chars: int,
) -> str:
    """Render retrieved documents into a compact string."""
    if not docs:
        return "No documents found in the knowledge base for this query."

    snippets = []
    for idx, (doc, score) in enumerate(docs[:max_documents], start=1):
        source = (
            doc.metadata.get("filename")
            or "unknown source"
        )
        hash = (
            doc.metadata.get("resource_hash")
            or "n/a"
        )
        text = doc.page_content.strip()
        if len(text) > max_chars:
            text = f"{text[:max_chars].rstrip()}..."
        header = f"[{idx}] {source} (hash={hash})"
        footer = f"Score: {score:.4f}" if isinstance(score, (float, int)) else "Score: n/a"
        snippets.append(f"{header}\n{footer}\n{text}")

    return "\n\n".join(snippets)


def create_retriever_tool(
    retriever: BaseRetriever,
    *,
    name: str = "search_knowledge_base",
    description: Optional[str] = None,
    max_documents: int = 4,
    max_chars: int = 800,
    store_docs: Optional[Callable[[str, Sequence[Document]], None]] = None,
    required_permission: Optional[str] = None,
) -> Callable[[str], str]:
    """
    Wrap a `BaseRetriever` instance in a LangChain tool.

    The resulting tool returns a formatted string combining the retrieved documents
    so the calling agent can ground its responses in the vector store content.
    If ``store_docs`` is provided, it will be invoked with the tool name and
    the list of retrieved ``Document`` objects before formatting the response.
    
    Args:
        retriever: The LangChain retriever instance to wrap.
        name: The name of the tool.
        description: Human-readable description of the tool.
        max_documents: Maximum number of documents to return.
        max_chars: Maximum characters per document snippet.
        store_docs: Optional callback to store retrieved documents.
        required_permission: Optional RBAC permission required to use this tool.
            If None, no permission check is performed (allow all).
    """

    tool_description = (
        description
        or (
            "Search the indexed knowledge base for relevant passages.\n"
            "Input: query string.\n"
            "Output: ranked snippets with source filename, resource hash, and score.\n"
            "Example input: \"transfer errors in CMS\"."
        )
    )

    @tool(name, description=tool_description)
    @require_tool_permission(required_permission)
    def _retriever_tool(query: str) -> str:
        results = retriever.invoke(query)
        docs = _normalize_results(results or [])
        if store_docs:
            store_docs(f"{name}: {query}", [doc for doc, _ in docs])
        return _format_documents_for_llm(docs, max_documents=max_documents, max_chars=max_chars)

    return _retriever_tool
