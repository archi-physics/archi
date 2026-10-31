from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import flask  # noqa: F401
except ModuleNotFoundError:
    fake_flask = types.ModuleType("flask")

    class _Blueprint:
        def __init__(self, *args, **kwargs):
            pass

        def route(self, *args, **kwargs):
            def _decorator(func):
                return func

            return _decorator

    class _Response:
        def __init__(self, *args, **kwargs):
            pass

    fake_flask.Blueprint = _Blueprint
    fake_flask.Response = _Response
    fake_flask.jsonify = lambda payload=None, **kwargs: payload if payload is not None else kwargs
    fake_flask.request = types.SimpleNamespace(
        headers={},
        args={},
        host="localhost",
        scheme="http",
        get_json=lambda silent=True: None,
    )
    fake_flask.stream_with_context = lambda generator: generator
    sys.modules["flask"] = fake_flask

from src.interfaces.chat_app import mcp_sse


def _tool_text(result: dict) -> str:
    return result["content"][0]["text"]


@pytest.fixture
def wrapper():
    wrapper = MagicMock()
    wrapper.chat = MagicMock()
    wrapper.chat.data_viewer = MagicMock()
    return wrapper


def test_tool_list_includes_repo_backed_discovery_tools():
    names = {tool["name"] for tool in mcp_sse._TOOLS}

    assert "archi_search_document_metadata" in names
    assert "archi_list_metadata_schema" in names
    assert "archi_search_document_content" in names
    assert "archi_get_document_chunks" in names
    assert "archi_get_data_stats" in names
    assert "archi_get_agent_spec" in names


def test_list_documents_passes_conversation_and_enabled_filter(wrapper):
    wrapper.chat.data_viewer.list_documents.return_value = {
        "documents": [
            {
                "hash": "doc-1",
                "display_name": "Doc 1",
                "source_type": "git",
                "ingestion_status": "embedded",
                "enabled": False,
            }
        ],
        "total": 1,
    }

    result = mcp_sse._tool_list_documents(
        {
            "conversation_id": 42,
            "enabled": "disabled",
            "limit": 10,
            "offset": 5,
        },
        wrapper,
    )

    wrapper.chat.data_viewer.list_documents.assert_called_once_with(
        conversation_id=42,
        source_type=None,
        search=None,
        enabled_filter="disabled",
        limit=10,
        offset=5,
    )
    assert "enabled=no" in _tool_text(result)


def test_search_document_metadata_parses_or_filters(wrapper):
    catalog = wrapper.chat.data_viewer.catalog
    catalog.search_metadata.return_value = [
        {
            "hash": "hash-1",
            "path": Path("/tmp/docs/readme.md"),
            "metadata": {
                "display_name": "README",
                "source_type": "git",
                "relative_path": "docs/readme.md",
            },
        }
    ]

    result = mcp_sse._tool_search_document_metadata(
        {"query": "source_type:git OR source_type:web outage", "limit": 7},
        wrapper,
    )

    catalog.search_metadata.assert_called_once_with(
        "outage",
        limit=7,
        filters=[{"source_type": "git"}, {"source_type": "web"}],
    )
    assert "README" in _tool_text(result)
    assert "hash-1" in _tool_text(result)


def test_list_metadata_schema_formats_distinct_values(wrapper):
    catalog = wrapper.chat.data_viewer.catalog
    catalog.get_distinct_metadata.return_value = {
        "source_type": ["git", "web"],
        "suffix": [".md", ".py"],
    }

    result = mcp_sse._tool_list_metadata_schema(wrapper)
    text = _tool_text(result)

    assert "source_type values: git, web" in text
    assert "suffix values: .md, .py" in text
    assert "relative_path" in text


def test_search_document_content_greps_indexed_files(wrapper, tmp_path):
    catalog = wrapper.chat.data_viewer.catalog
    doc_path = tmp_path / "example.log"
    doc_path.write_text("alpha\nneedle here\nomega\n", encoding="utf-8")

    catalog.iter_files.return_value = [("hash-1", doc_path)]
    catalog.get_metadata_for_hash.return_value = {
        "display_name": "Example Log",
        "source_type": "local_files",
    }

    fake_loader_utils = types.ModuleType("src.data_manager.vectorstore.loader_utils")
    fake_loader_utils.load_text_from_path = lambda path: Path(path).read_text(encoding="utf-8")

    with patch.dict(sys.modules, {"src.data_manager.vectorstore.loader_utils": fake_loader_utils}):
        result = mcp_sse._tool_search_document_content(
            {"query": "needle", "before": 1, "after": 1},
            wrapper,
        )

    text = _tool_text(result)
    assert "Example Log" in text
    assert "L2: needle here" in text
    assert "B: alpha" in text
    assert "A: omega" in text


def test_get_document_chunks_paginates_and_truncates(wrapper):
    wrapper.chat.data_viewer.get_document_chunks.return_value = [
        {"index": 0, "text": "a" * 120, "start_char": 0, "end_char": 119},
        {"index": 1, "text": "b" * 120, "start_char": 120, "end_char": 239},
        {"index": 2, "text": "c" * 120, "start_char": 240, "end_char": 359},
    ]

    result = mcp_sse._tool_get_document_chunks(
        {
            "document_hash": "hash-1",
            "offset": 1,
            "limit": 1,
            "max_chars_per_chunk": 80,
        },
        wrapper,
    )

    text = _tool_text(result)
    assert "showing 1 from offset 1" in text
    assert "chunk 1" in text
    assert "chars=120-239" in text
    assert "..." in text


def test_get_data_stats_formats_source_breakdown(wrapper):
    wrapper.chat.data_viewer.get_stats.return_value = {
        "total_documents": 12,
        "total_chunks": 34,
        "enabled_documents": 10,
        "disabled_documents": 2,
        "total_size_bytes": 2048,
        "last_sync": "2026-03-17T12:00:00+00:00",
        "status_counts": {"pending": 1, "embedding": 2, "embedded": 8, "failed": 1},
        "by_source_type": {
            "git": {"total": 5, "enabled": 4},
            "web": {"total": 7, "enabled": 6},
        },
    }

    result = mcp_sse._tool_get_data_stats({"conversation_id": 99}, wrapper)
    wrapper.chat.data_viewer.get_stats.assert_called_once_with(99)

    text = _tool_text(result)
    assert "Total documents:      12" in text
    assert "git: total=5, enabled=4" in text
    assert "web: total=7, enabled=6" in text


def test_get_agent_spec_returns_full_markdown(wrapper, tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    agent_path = agents_dir / "ops.md"
    content = (
        "---\n"
        "name: Ops Agent\n"
        "tools:\n"
        "  - search_vectorstore_hybrid\n"
        "---\n\n"
        "You help with ops questions.\n"
    )
    agent_path.write_text(content, encoding="utf-8")
    wrapper._get_agents_dir.return_value = agents_dir

    result = mcp_sse._tool_get_agent_spec({"agent_name": "Ops Agent"}, wrapper)

    assert _tool_text(result) == content


def test_deployment_info_includes_active_agent_and_mcp_servers(wrapper):
    wrapper.chat.agent_spec = types.SimpleNamespace(name="Fallback Agent")

    fake_static = types.SimpleNamespace(
        available_providers=["openai", "anthropic"],
        available_pipelines=["QAPipeline", "CMSCompOpsAgent"],
    )
    fake_dynamic = types.SimpleNamespace(
        active_agent_name="Configured Agent",
        active_model="openai/gpt-4o",
        temperature=0.2,
        max_tokens=2048,
        num_documents_to_retrieve=6,
        use_hybrid_search=True,
        bm25_weight=0.4,
        semantic_weight=0.6,
    )

    with patch("src.utils.config_access.get_full_config") as mock_full, \
         patch("src.utils.config_access.get_static_config") as mock_static, \
         patch("src.utils.config_access.get_dynamic_config") as mock_dynamic:
        mock_full.return_value = {
            "name": "demo",
            "services": {
                "chat_app": {"pipeline": "QAPipeline", "agent_class": "CMSCompOpsAgent"},
                "data_manager": {"embedding": {"model": "text-embedding-3-small", "chunk_size": 800, "chunk_overlap": 100}},
                "mcp_server": {"enabled": True},
            },
            "mcp_servers": {"deepwiki": {}, "search": {}},
        }
        mock_static.return_value = fake_static
        mock_dynamic.return_value = fake_dynamic

        result = mcp_sse._tool_deployment_info(wrapper)

    text = _tool_text(result)
    assert "Active agent:          Configured Agent" in text
    assert "MCP servers:           deepwiki, search" in text
    assert "Available providers:   openai, anthropic" in text
