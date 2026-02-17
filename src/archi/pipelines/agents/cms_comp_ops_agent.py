from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from src.utils.logging import get_logger
from src.utils.env import read_secret
from src.archi.pipelines.agents.base_react import BaseReActAgent
from src.data_manager.vectorstore.retrievers import HybridRetriever
from src.archi.pipelines.agents.tools import (
    create_document_fetch_tool,
    create_file_search_tool,
    create_metadata_search_tool,
    create_metadata_schema_tool,
    create_retriever_tool,
    RemoteCatalogClient,
    MONITOpenSearchClient,
    create_monit_opensearch_search_tool,
    create_monit_opensearch_aggregation_tool,
)
from src.archi.pipelines.agents.utils.skill_utils import load_skill

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

        self.catalog_service = RemoteCatalogClient.from_deployment_config(self.config)
        self._vector_retrievers = None
        self._vector_tool = None
        self.enable_vector_tools = "search_vectorstore_hybrid" in self.selected_tool_names

        # Initialize MONIT client (shared across search and aggregation tools)
        self._monit_client = None
        self._rucio_events_skill = None
        self._init_monit()

        self.rebuild_static_tools()
        self.rebuild_static_middleware()
        self.refresh_agent()

    @property
    def _chat_app_config(self) -> Dict[str, Any]:
        """Return the services.chat_app config section."""
        return self.config.get("services", {}).get("chat_app", {})

    def _init_monit(self) -> None:
        """Initialize the MONIT OpenSearch client if credentials and config are available."""
        monit_token = read_secret("MONIT_GRAFANA_TOKEN")
        monit_url = (
            self._chat_app_config.get("tools", {}).get("monit", {}).get("url")
        )

        if monit_token and monit_url:
            try:
                self._monit_client = MONITOpenSearchClient(url=monit_url, token=monit_token)
                self._rucio_events_skill = load_skill("rucio_events", self.config)
                logger.info("MONIT OpenSearch client initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize MONIT OpenSearch client: %s", e)
        elif not monit_url:
            logger.info(
                "No MONIT URL configured in services.chat_app.tools.monit.url; "
                "MONIT OpenSearch tools not available"
            )
        else:
            logger.info("MONIT_GRAFANA_TOKEN not found; MONIT OpenSearch tools not available")

    def get_tool_registry(self) -> Dict[str, Callable[[], Any]]:
        return {name: entry["builder"] for name, entry in self._tool_definitions().items()}

    def get_tool_descriptions(self) -> Dict[str, str]:
        return {name: entry["description"] for name, entry in self._tool_definitions().items()}

    def _tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        defs = {
            "search_local_files": {
                "builder": self._build_file_search_tool,
                "description": (
                    "Grep-like search over file contents. Provide a distinctive phrase or regex; optionally use "
                    "regex=true, case_sensitive=true, and context (before/after). Returns matching lines with hashes; "
                    "use fetch_catalog_document for full text."
                ),
            },
            "search_metadata_index": {
                "builder": self._build_metadata_search_tool,
                "description": (
                    "Query the files' metadata catalog (ticket IDs, source URLs, resource types, etc.). "
                    "Supports key:value filters and OR (e.g., source_type:git OR url:https://... ticket_id:CMS-123). "
                    "Returns matching files with metadata; use fetch_catalog_document to pull full text."
                ),
            },
            "list_metadata_schema": {
                "builder": self._build_metadata_schema_tool,
                "description": (
                    "List metadata schema hints: supported keys, distinct source_type values, and suffixes. "
                    "Use this to learn which key:value filters are available before searching."
                ),
            },
            "fetch_catalog_document": {
                "builder": self._build_fetch_tool,
                "description": (
                    "Fetch full document text by resource hash after a search hit. "
                    "Use this sparingly to pull only the most relevant files."
                ),
            },
            "search_vectorstore_hybrid": {
                "builder": self._build_vector_tool_placeholder,
                "description": (
                    "Hybrid search over the knowledge base that combines both lexical (BM25) and semantic (vector) search."
                ),
            },
            "mcp": {
                "builder": self._build_mcp_tools,
                "description": "Access tools served via configured MCP servers.",
            },
        }

        # Keep this safe for lightweight introspection paths that call
        # get_tool_registry()/get_tool_descriptions() on an uninitialized
        # instance (constructed via __new__).
        if getattr(self, "_monit_client", None) is not None:
            defs["monit_opensearch_search"] = {
                "builder": self._build_monit_opensearch_search_tool,
                "description": "Search MONIT OpenSearch for CMS Rucio events.",
            }
            defs["monit_opensearch_aggregation"] = {
                "builder": self._build_monit_opensearch_aggregation_tool,
                "description": "Run aggregation queries on MONIT OpenSearch for CMS Rucio events.",
            }

        return defs

    def _build_file_search_tool(self) -> Callable:
        description = self._tool_definitions()["search_local_files"]["description"]
        return create_file_search_tool(
            self.catalog_service,
            description=description,
            store_docs=self._store_documents,
        )

    def _build_metadata_search_tool(self) -> Callable:
        description = self._tool_definitions()["search_metadata_index"]["description"]
        return create_metadata_search_tool(
            self.catalog_service,
            description=description,
            store_docs=self._store_documents,
        )

    def _build_metadata_schema_tool(self) -> Callable:
        description = self._tool_definitions()["list_metadata_schema"]["description"]
        return create_metadata_schema_tool(
            self.catalog_service,
            description=description,
        )

    def _build_fetch_tool(self) -> Callable:
        description = self._tool_definitions()["fetch_catalog_document"]["description"]
        return create_document_fetch_tool(
            self.catalog_service,
            description=description,
        )

    def _build_vector_tool_placeholder(self) -> List[Callable]:
        return []

    def _build_monit_opensearch_search_tool(self) -> Callable:
        """Build the MONIT OpenSearch search tool for Rucio events."""
        return create_monit_opensearch_search_tool(
            self._monit_client,
            tool_name="rucio_events_search",
            index="monit_prod_cms_rucio_raw_events*",
            skill=self._rucio_events_skill,
        )

    def _build_monit_opensearch_aggregation_tool(self) -> Callable:
        """Build the MONIT OpenSearch aggregation tool for Rucio events."""
        return create_monit_opensearch_aggregation_tool(
            self._monit_client,
            tool_name="rucio_events_aggregation",
            index="monit_prod_cms_rucio_raw_events*",
            skill=self._rucio_events_skill,
        )

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

    def _update_vector_retrievers(self, vectorstore: Any) -> None:
        """Instantiate or refresh the vectorstore retriever tool using hybrid retrieval."""
        if not self.enable_vector_tools:
            self._vector_retrievers = None
            self._vector_tools = None
            return
        retrievers_cfg = self.dm_config.get("retrievers", {})
        hybrid_cfg = retrievers_cfg.get("hybrid_retriever", {})

        k = hybrid_cfg.get("num_documents_to_retrieve", 5)
        bm25_weight = hybrid_cfg.get("bm25_weight", 0.6)
        semantic_weight = hybrid_cfg.get("semantic_weight", 0.4)

        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            k=k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        hybrid_description = self._tool_definitions()["search_vectorstore_hybrid"]["description"]

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
