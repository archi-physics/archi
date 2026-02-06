from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence
import json
import os

from langchain_core.documents import Document
import asyncio
import nest_asyncio
from langchain.agents.middleware import TodoListMiddleware, LLMToolSelectorMiddleware

from src.utils.logging import get_logger
from src.archi.pipelines.agents.base_react import BaseReActAgent
from src.data_manager.vectorstore.retrievers import HybridRetriever
from src.archi.pipelines.agents.tools import (
    create_document_fetch_tool,
    create_file_search_tool,
    create_metadata_search_tool,
    create_metadata_schema_tool,
    create_retriever_tool,
    initialize_mcp_client,
    RemoteCatalogClient,
)
from src.archi.pipelines.agents.utils.history_utils import infer_speaker

logger = get_logger(__name__)


class CMSCompOpsAgent(BaseReActAgent):
    """Agent designed for CMS CompOps operations."""

    def __init__(
        self,
        config: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)

        self.mcp_client = None
        self.catalog_service = RemoteCatalogClient.from_deployment_config(self.config)
        self._vector_retrievers = None
        self._vector_tool = None
        self.enable_vector_tools = "search_vectorstore_hybrid" in self.selected_tool_names

        self.rebuild_static_tools()
        self.rebuild_static_middleware()
        self.refresh_agent()

    def get_tool_registry(self) -> Dict[str, Callable[[], Any]]:
        return {
            "search_local_files": self._build_file_search_tool,
            "search_metadata_index": self._build_metadata_search_tool,
            "list_metadata_schema": self._build_metadata_schema_tool,
            "fetch_catalog_document": self._build_fetch_tool,
            "search_vectorstore_hybrid": self._build_vector_tool_placeholder,
            "mcp": self._build_mcp_tools,
        }

    def _build_file_search_tool(self) -> Callable:
        return create_file_search_tool(
            self.catalog_service,
            description=(
                "Grep-like search over file contents. Provide a distinctive phrase or regex; optionally use regex=true, "
                "case_sensitive=true, and context (before/after). Returns matching lines with hashes; "
                "use fetch_catalog_document for full text."
            ),
            store_docs=self._store_documents,
        )

    def _build_metadata_search_tool(self) -> Callable:
        return create_metadata_search_tool(
            self.catalog_service,
            description=(
                "Query the files' metadata catalog (ticket IDs, source URLs, resource types, etc.). "
                "Supports key:value filters and OR (e.g., source_type:git OR url:https://... ticket_id:CMS-123). "
                "Returns matching files with metadata; use fetch_catalog_document to pull full text."
            ),
            store_docs=self._store_documents,
        )

    def _build_metadata_schema_tool(self) -> Callable:
        return create_metadata_schema_tool(
            self.catalog_service,
            description=(
                "List metadata schema hints: supported keys, distinct source_type values, and suffixes. "
                "Use this to learn which key:value filters are available before searching."
            ),
        )

    def _build_fetch_tool(self) -> Callable:
        return create_document_fetch_tool(
            self.catalog_service,
            description=(
                "Fetch full document text by resource hash after a search hit. "
                "Use this sparingly to pull only the most relevant files."
            ),
        )

    def _build_vector_tool_placeholder(self) -> List[Callable]:
        return []

    def _build_mcp_tools(self) -> List[Callable]:
        try:
            nest_asyncio.apply()
            mcp_servers = self._load_mcp_servers()
            client, mcp_tools = asyncio.run(initialize_mcp_client(mcp_servers))
            self.mcp_client = client

            def make_synchronous(async_tool):
                def sync_wrapper(*args, **kwargs):
                    return asyncio.run(async_tool.coroutine(*args, **kwargs))

                async_tool.func = sync_wrapper
                return async_tool

            if mcp_tools:
                synchronous_mcp_tools = [make_synchronous(t) for t in mcp_tools]
                logger.info("Loaded and patched %d MCP tools for sync execution.", len(synchronous_mcp_tools))
                return synchronous_mcp_tools
        except Exception as exc:
            logger.error("Failed to load MCP tools: %s", exc, exc_info=True)
        return []

    @staticmethod
    def _load_mcp_servers() -> Dict[str, Any]:
        raw = os.environ.get("ARCHI_MCP_SERVERS") or os.environ.get("MCP_SERVERS")
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid MCP servers JSON in ARCHI_MCP_SERVERS/MCP_SERVERS.")
            return {}
        if not isinstance(payload, dict):
            logger.warning("MCP servers payload must be a JSON object.")
            return {}
        return payload

    # def _build_static_middleware(self) -> List[Callable]:
    #     """
    #     Initialize middleware: currently, testing what works best.
    #     This is static.
    #     """
    #     todolist_middleware = TodoListMiddleware()
    #     llmtoolselector_middleware = LLMToolSelectorMiddleware(
    #         model=self.agent_llm,
    #         max_tools=3,
    #     )
    #     return [todolist_middleware, llmtoolselector_middleware]

    def _store_documents(self, stage: str, docs: Sequence[Document]) -> None:
        """Centralised helper used by tools to record documents into the active memory."""
        memory = self.active_memory
        if not memory:
            return
        # Prefer memory convenience method if available
        try:
            logger.debug("Recording %d documents from stage '%s' via record_documents", len(docs), stage)
            memory.record_documents(stage, docs)
        except Exception:
            # fallback to explicit record + note
            memory.record(stage, docs)
            memory.note(f"{stage} returned {len(list(docs))} document(s).")

    def _prepare_inputs(self, history: Any, **kwargs) -> Dict[str, Any]:
        """Create list of messages using LangChain's formatting."""
        history = history or []
        history_messages = [infer_speaker(msg[0])(msg[1]) for msg in history]
        return {"history": history_messages}

    def _prepare_agent_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare agent state and formatted inputs shared by invoke/stream."""

        # event-level memory (which documents were retrieved)
        memory = self.start_run_memory()

        # refresh vs connection
        vectorstore = kwargs.get("vectorstore")
        if vectorstore:
            self._update_vector_retrievers(vectorstore)
        else:
            self._vector_retrievers = None
            self._vector_tools = None
        extra_tools = self._vector_tools if self._vector_tools else None

        self.refresh_agent(extra_tools=extra_tools)

        inputs = self._prepare_inputs(history=kwargs.get("history"))
        history_messages = inputs["history"]
        if history_messages:
            memory.note(f"History contains {len(history_messages)} message(s).")
            last_message = history_messages[-1]
            content = self._message_content(last_message)
            if content:
                snippet = content if len(content) <= 200 else f"{content[:197]}..."
                memory.note(f"Latest user message: {snippet}")
        return {"messages": history_messages}

    def _update_vector_retrievers(self, vectorstore: Any) -> None:
        """Instantiate or refresh the vectorstore retriever tool using hybrid retrieval."""
        if not self.enable_vector_tools:
            self._vector_retrievers = None
            self._vector_tools = None
            return
        retrievers_cfg = self.dm_config.get("retrievers", {})
        hybrid_cfg = retrievers_cfg.get("hybrid_retriever", {})

        k = hybrid_cfg["num_documents_to_retrieve"]
        bm25_weight = hybrid_cfg["bm25_weight"]
        semantic_weight = hybrid_cfg["semantic_weight"]

        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            k=k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        hybrid_description = (
            "Hybrid search over the knowledge base that combines both lexical (BM25) and semantic (vector) search. "
            "This automatically finds documents matching exact keywords, error messages, ticket IDs, filenames, "
            "and function names (via BM25) as well as conceptually related content and paraphrased information "
            "(via semantic search). Use this for comprehensive retrieval - it handles both precise keyword matches "
            "and conceptual similarity automatically."
        )

        self._vector_retrievers = [hybrid_retriever]
        self._vector_tools = []
        self._vector_tools.append(
            create_retriever_tool(
                hybrid_retriever,
                name="search_vectorstore_hybrid",
                description=hybrid_description,
                store_docs=self._store_documents,
            )
        )
