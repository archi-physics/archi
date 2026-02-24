import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Minimal stubs so tests can run without langchain-core installed.
if "langchain_core" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    sys.modules["langchain_core"] = langchain_core

if "langchain_core.documents" not in sys.modules:
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = object
    sys.modules["langchain_core.documents"] = documents_module

if "langchain_core.embeddings" not in sys.modules:
    embeddings_module = types.ModuleType("langchain_core.embeddings")
    embeddings_module.Embeddings = object
    sys.modules["langchain_core.embeddings"] = embeddings_module

if "langchain_core.vectorstores" not in sys.modules:
    vectorstores_module = types.ModuleType("langchain_core.vectorstores")
    vectorstores_module.VectorStore = object
    sys.modules["langchain_core.vectorstores"] = vectorstores_module

if "nltk" not in sys.modules:
    nltk_module = types.ModuleType("nltk")
    nltk_module.tokenize = types.SimpleNamespace(word_tokenize=lambda text: text.split())
    nltk_module.stem = types.SimpleNamespace(PorterStemmer=lambda: types.SimpleNamespace(stem=lambda w: w))
    nltk_module.download = lambda *_args, **_kwargs: None
    sys.modules["nltk"] = nltk_module

if "langchain_text_splitters" not in sys.modules:
    sys.modules["langchain_text_splitters"] = types.ModuleType("langchain_text_splitters")

if "langchain_text_splitters.character" not in sys.modules:
    character_module = types.ModuleType("langchain_text_splitters.character")

    class _DummyCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, docs):
            return docs

    character_module.CharacterTextSplitter = _DummyCharacterTextSplitter
    sys.modules["langchain_text_splitters.character"] = character_module

if "langchain_community" not in sys.modules:
    sys.modules["langchain_community"] = types.ModuleType("langchain_community")

if "langchain_community.document_loaders" not in sys.modules:
    loaders_module = types.ModuleType("langchain_community.document_loaders")

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self):
            return []

    loaders_module.BSHTMLLoader = _DummyLoader
    loaders_module.PyPDFLoader = _DummyLoader
    loaders_module.PythonLoader = _DummyLoader
    loaders_module.TextLoader = _DummyLoader
    sys.modules["langchain_community.document_loaders"] = loaders_module

if "langchain_community.document_loaders.text" not in sys.modules:
    text_module = types.ModuleType("langchain_community.document_loaders.text")
    text_module.TextLoader = sys.modules["langchain_community.document_loaders"].TextLoader
    sys.modules["langchain_community.document_loaders.text"] = text_module

from src.data_manager.vectorstore import manager as manager_module
from src.data_manager.vectorstore.manager import VectorStoreManager


class _InlineFuture:
    def __init__(self, fn, *args, **kwargs):
        self._exc = None
        self._result = None
        try:
            self._result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            self._exc = exc

    def result(self):
        if self._exc:
            raise self._exc
        return self._result


class _InlineExecutor:
    def __init__(self, max_workers=1):
        self.max_workers = max_workers
        self.futures = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = _InlineFuture(fn, *args, **kwargs)
        self.futures.append(fut)
        return fut


def test_add_to_postgres_commits_every_25_files(monkeypatch):
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.parallel_workers = 1
    manager.collection_name = "test_collection"
    manager._data_manager_config = {"stemming": {"enabled": False}}
    manager._pg_config = {"host": "localhost"}

    catalog = MagicMock()
    catalog.get_document_id.return_value = 1
    catalog.get_metadata_for_hash.return_value = {}
    manager._catalog = catalog

    split_doc = SimpleNamespace(page_content="hello world", metadata={})
    manager.text_splitter = SimpleNamespace(split_documents=lambda docs: [split_doc])
    manager.embedding_model = SimpleNamespace(embed_documents=lambda chunks: [[0.1, 0.2, 0.3] for _ in chunks])
    manager.loader = lambda _path: SimpleNamespace(load=lambda: [split_doc])

    fake_cursor = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    fake_conn.cursor.return_value.__exit__.return_value = False

    monkeypatch.setattr(manager_module.psycopg2, "connect", lambda **_kwargs: fake_conn)
    monkeypatch.setattr(manager_module.psycopg2.extras, "execute_values", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager_module, "ThreadPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(manager_module, "as_completed", lambda futures: list(futures))

    files_to_add = {f"hash-{i}": f"/tmp/file-{i}.txt" for i in range(26)}
    manager._add_to_postgres(files_to_add)

    # First commit at 25 files, second for final remainder.
    assert fake_conn.commit.call_count == 2
    # All documents are marked embedding at start of run.
    assert catalog.update_ingestion_status.call_count >= 26
