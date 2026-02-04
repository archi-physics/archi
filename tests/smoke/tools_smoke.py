#!/usr/bin/env python3
"""Direct tool smoke checks for catalog and vectorstore tools."""
import os
import sys
from typing import Dict

import yaml

from src.archi.pipelines.agents.tools import (
    RemoteCatalogClient,
    create_document_fetch_tool,
    create_file_search_tool,
    create_metadata_search_tool,
    create_retriever_tool,
)
from src.archi.utils.vectorstore_connector import VectorstoreConnector
from src.data_manager.vectorstore.retrievers import HybridRetriever


def _fail(message: str) -> None:
    print(f"[tools-smoke] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _info(message: str) -> None:
    print(f"[tools-smoke] {message}")


def _invoke_tool(tool, payload: Dict[str, object]) -> str:
    if hasattr(tool, "invoke"):
        return tool.invoke(payload)
    if hasattr(tool, "run"):
        return tool.run(payload)
    raise TypeError(f"Unsupported tool type: {type(tool)}")


def _load_config() -> Dict:
    config_path = os.getenv("ARCHI_CONFIG_PATH")
    if not config_path:
        _fail("ARCHI_CONFIG_PATH is required for tool smoke checks")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:
        _fail(f"Failed to load config at {config_path}: {exc}")
    return {}


def _map_embedding_classes(config: Dict) -> None:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import OpenAIEmbeddings
    except Exception as exc:
        _fail(f"Missing embedding dependencies: {exc}")

    embedding_map = config.get("data_manager", {}).get("embedding_class_map", {})
    for name, entry in embedding_map.items():
        class_name = entry.get("class") or entry.get("name") or name
        if class_name == "HuggingFaceEmbeddings":
            entry["class"] = HuggingFaceEmbeddings
        elif class_name == "OpenAIEmbeddings":
            entry["class"] = OpenAIEmbeddings
        else:
            _fail(f"Unsupported embedding class '{class_name}' in config")


def _build_catalog_client(config: Dict) -> RemoteCatalogClient:
    dm_base_url = os.getenv("DM_BASE_URL")
    if dm_base_url:
        return RemoteCatalogClient(base_url=dm_base_url)
    return RemoteCatalogClient.from_deployment_config(config)


def _run_catalog_tools(catalog: RemoteCatalogClient) -> None:
    file_query = os.getenv("FILE_SEARCH_QUERY", "Smoke test seed document")
    metadata_query = os.getenv("METADATA_SEARCH_QUERY", "file_name:seed.txt")

    file_search_tool = create_file_search_tool(catalog)
    metadata_search_tool = create_metadata_search_tool(catalog)
    fetch_tool = create_document_fetch_tool(catalog)

    _info("Running file search tool ...")
    file_result = _invoke_tool(file_search_tool, {"query": file_query})
    if "failed" in file_result.lower() or "no local files" in file_result.lower():
        _fail("File search tool returned no results or failed")

    _info("Running metadata search tool ...")
    meta_result = _invoke_tool(metadata_search_tool, {"query": metadata_query})
    if "failed" in meta_result.lower() or "no local files" in meta_result.lower():
        _fail("Metadata search tool returned no results or failed")

    _info("Running document fetch tool ...")
    hits = catalog.search(metadata_query, limit=1, search_content=False)
    if not hits:
        _fail("Metadata search returned no hits; cannot fetch document")
    resource_hash = hits[0].get("hash")
    if not resource_hash:
        _fail("Catalog hit missing resource hash")
    fetch_result = _invoke_tool(fetch_tool, {"resource_hash": resource_hash})
    if "content:" not in fetch_result.lower():
        _fail("Document fetch tool returned unexpected output")


def _run_vectorstore_tool(config: Dict) -> None:
    _map_embedding_classes(config)
    vectorstore = VectorstoreConnector(config).get_vectorstore()

    retriever_cfg = config.get("data_manager", {}).get("retrievers", {}).get("hybrid_retriever")
    if not retriever_cfg:
        _fail("Missing data_manager.retrievers.hybrid_retriever config for vectorstore tool")

    hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        k=retriever_cfg["num_documents_to_retrieve"],
        bm25_weight=retriever_cfg["bm25_weight"],
        semantic_weight=retriever_cfg["semantic_weight"],
    )

    retriever_tool = create_retriever_tool(hybrid_retriever)
    query = os.getenv("VECTORSTORE_QUERY", "Smoke test seed document")

    _info("Running vectorstore retriever tool ...")
    result = _invoke_tool(retriever_tool, {"query": query})
    if "no documents found" in result.lower():
        _fail("Vectorstore retriever tool returned no documents")


def main() -> None:
    config = _load_config()
    catalog = _build_catalog_client(config)
    _run_catalog_tools(catalog)
    _run_vectorstore_tool(config)
    _info("Tool smoke checks passed")


if __name__ == "__main__":
    main()
