import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

# Minimal stub so tests can run without langchain-core installed.
if "langchain_core" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    sys.modules["langchain_core"] = langchain_core

if "langchain_core.documents" not in sys.modules:
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = object
    sys.modules["langchain_core.documents"] = documents_module

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
    sys.modules["langchain_community.document_loaders"] = loaders_module

if "langchain_community.document_loaders.text" not in sys.modules:
    text_module = types.ModuleType("langchain_community.document_loaders.text")
    text_module.TextLoader = sys.modules["langchain_community.document_loaders"].TextLoader if hasattr(sys.modules["langchain_community.document_loaders"], "TextLoader") else type("TextLoader", (), {"__init__": lambda self, *_a, **_k: None, "load": lambda self: []})
    # Ensure TextLoader exists on both modules for imports.
    if not hasattr(sys.modules["langchain_community.document_loaders"], "TextLoader"):
        setattr(sys.modules["langchain_community.document_loaders"], "TextLoader", text_module.TextLoader)
    sys.modules["langchain_community.document_loaders.text"] = text_module

from src.data_manager.collectors.persistence import PersistenceService


class _FakeResource:
    def __init__(self, resource_hash: str, filename: str, content: str):
        self._hash = resource_hash
        self._filename = filename
        self._content = content

    def get_hash(self) -> str:
        return self._hash

    def get_file_path(self, target_dir: Path) -> Path:
        return target_dir / self._filename

    def get_content(self):
        return self._content

    def get_metadata(self):
        # No size_bytes provided by resource metadata on purpose.
        return SimpleNamespace(as_dict=lambda: {"source_type": "ticket", "display_name": "Test Doc"})


def test_persist_resource_sets_size_bytes_from_written_file():
    with TemporaryDirectory() as tmp_dir:
        service = PersistenceService.__new__(PersistenceService)
        service.data_path = Path(tmp_dir)
        service.catalog = MagicMock()

        resource = _FakeResource("hash-1", "doc.txt", "hello persistence")
        target_dir = service.data_path / "tickets"
        persisted_path = service.persist_resource(resource, target_dir)

        assert persisted_path.exists()
        service.catalog.upsert_resource.assert_called_once()
        _, _, metadata = service.catalog.upsert_resource.call_args[0]
        assert metadata["size_bytes"] == str(len("hello persistence"))
