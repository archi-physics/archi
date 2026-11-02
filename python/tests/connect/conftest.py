"""Shared pieces for the `archi connect` tests: a throwaway instance and golden-file checks."""
import os
from pathlib import Path

import pytest

from archi.stack.instance import Instance, McpServer, Ports, write_env

TOKEN = "TESTTOKEN-0123456789abcdef"
GOLDEN = Path(__file__).parent / "golden"


def make_instance(tmp_path: Path, *, mcp_port: int = 18080, external=None, name: str = "demo") -> Instance:
    directory = tmp_path / "instance"
    directory.mkdir(parents=True, exist_ok=True)
    instance = Instance(
        dir=directory, name=name, bundles=["cern-team"], data_root=tmp_path / "data", runtime="docker",
        ports=Ports(graph_db=15432, mcp=mcp_port, webui=13000),
        mcp_servers=list(external if external is not None else []),
    )
    instance.save()
    write_env(directory / ".env", {"OKG_CHAT_MCP_TOKEN": TOKEN, "ARCHI_ADMIN_PASSWORD": "pw"})
    return instance


EXTERNAL = McpServer(name="notes", url="http://127.0.0.1:18090/mcp", headers={"X-Api-Key": "NOTES_API_KEY"})


@pytest.fixture
def instance(tmp_path):
    return make_instance(tmp_path, external=[EXTERNAL])


@pytest.fixture
def check_golden():
    def check(name: str, text: str) -> None:
        path = GOLDEN / name
        if os.environ.get("UPDATE_GOLDEN"):
            path.write_text(text, encoding="utf-8")
        assert text == path.read_text(encoding="utf-8"), f"{name} differs from its golden file"

    return check
