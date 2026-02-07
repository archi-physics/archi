from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

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
    create_http_get_tool,
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

        self.rebuild_static_tools()
        self.rebuild_static_middleware()
        self.refresh_agent()

    def _build_static_tools(self) -> List[Callable]:
        """Initialise static tools that are always available to the agent."""
        file_search_tool = create_file_search_tool(
            self.catalog_service,
            description= (
                "Grep-like search over file contents. Provide a distinctive phrase or regex; optionally use regex=true, "
                "case_sensitive=true, and context (before/after). Returns matching lines with hashes; "
                "use fetch_catalog_document for full text."
            ),
            store_docs=self._store_documents,
        )
        metadata_search_tool = create_metadata_search_tool(
            self.catalog_service,
            description=(
                "Query the files' metadata catalog (ticket IDs, source URLs, resource types, etc.). "
                "Supports key:value filters and OR (e.g., source_type:git OR url:https://... ticket_id:CMS-123). "
                "Returns matching files with metadata; use fetch_catalog_document to pull full text."
            ),
            store_docs=self._store_documents,
        )
        metadata_schema_tool = create_metadata_schema_tool(
            self.catalog_service,
            description=(
                "List metadata schema hints: supported keys, distinct source_type values, and suffixes. "
                "Use this to learn which key:value filters are available before searching."
            ),
        )

        fetch_tool = create_document_fetch_tool(
            self.catalog_service,
            description=(
                "Fetch full document text by resource hash after a search hit. "
                "Use this sparingly to pull only the most relevant files."
            ),
        )

        http_get_tool = create_http_get_tool(
            name="fetch_url",
            description=(
                "Fetch live content from a URL via HTTP GET request. "
                "Input: A valid HTTP or HTTPS URL. "
                "Output: The response body text or an error message. "
                "Use this to retrieve real-time data from web endpoints, APIs, documentation, or status pages. "
                "Examples: checking endpoint status, fetching API data, retrieving documentation."
            ),
            timeout=15.0,
            max_response_chars=600000,
        )

        all_tools = [file_search_tool, metadata_search_tool, metadata_schema_tool, fetch_tool, http_get_tool]

        try:
            nest_asyncio.apply()

            # 1. Fetch the tools (async)
            client, mcp_tools = asyncio.run(initialize_mcp_client())
            self.mcp_client = client  # Keep client alive

            # 2. Patch tools to support synchronous execution
            # This wrapper allows the sync agent to call the async tools
            def make_synchronous(async_tool):
                def sync_wrapper(*args, **kwargs):
                    # We reuse the existing client session via the closure of the original tool
                    return asyncio.run(async_tool.coroutine(*args, **kwargs))

                # Assign the wrapper to the tool's 'func' attribute (standard LangChain sync entry point)
                async_tool.func = sync_wrapper
                return async_tool

            # Apply the patch to all fetched tools
            if mcp_tools:
                synchronous_mcp_tools = [make_synchronous(t) for t in mcp_tools]
                all_tools.extend(synchronous_mcp_tools)
                logger.info(f"Loaded and patched {len(synchronous_mcp_tools)} MCP tools for sync execution.")

        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}", exc_info=True)

        return all_tools

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
