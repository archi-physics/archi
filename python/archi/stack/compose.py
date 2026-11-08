"""The instance's compose project: graph Postgres, chat Postgres, Open WebUI, MCP.

One project on one network replaces four hand-run pieces of the old
quickstart: the `podman run` for the graph database (with its preload flags
and shm size), the chat database published on the container gateway, `okg
chat-instance up`, and a long-running `okg mcp-serve` in its own terminal.
Inside the network every service reaches the others by name, so there is no
gateway address to look up and nothing listens on 0.0.0.0. Each service the
host needs is published on 127.0.0.1 only.

`compose.yaml` holds no secret: compose interpolates `${...}` from the
instance's `.env`, which git ignores.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml

from archi.stack.instance import COMPOSE_FILE, ENV_FILE, MCP_DOCKERFILE, Instance

#: AKMON's Postgres image, with the extensions okg needs. Three of them must
#: be preloaded at server start, so a stock Postgres cannot be adapted later.
GRAPH_DB_IMAGE = "ghcr.io/mitdbg/akmon-postgres:v{okg_version}"
PRELOAD_LIBRARIES = ("timescaledb", "pg_cron", "pg_textsearch")
#: The 64 MB container default is too small for okg's index builds.
GRAPH_DB_SHM = "2g"
#: PGDATA inside ghcr.io/mitdbg/akmon-postgres (`podman image inspect` of v0.2.0
#: shows PGDATA=/home/postgres/pgdata/data; not re-inspected on v0.4.0). The graph volume must cover its
#: parent; mounted anywhere else it holds nothing and the graph dies with the
#: container. Re-check when GRAPH_DB_IMAGE changes.
GRAPH_DB_PGDATA = "/home/postgres/pgdata/data"  # container-internal path
CHAT_DB_IMAGE = "docker.io/library/postgres:16"
#: Upstream Open WebUI at the version okg's chat code is measured against
#: (okg.substrate.chat.instance.PINNED_VERSION). okg v0.2.0 pins a patched
#: image under a placeholder registry (`your-registry.invalid/...`) that is
#: not published; every okg tool result is plain text now, which
#: the upstream image reads too. See docs/docs/operating/footguns.md.
WEBUI_IMAGE = "ghcr.io/open-webui/open-webui:v0.11.4"
#: Where the MCP server listens inside the network.
MCP_CONTAINER_PORT = 8765
#: Open WebUI loads a sentence-embedding model on first start; that alone
#: overran okg's 120 s default ready timeout in the quickstart.
WEBUI_READY_SECONDS = 600
#: AKMON's release wheel for one version. Its file and distribution are named
#: `akmon` from 0.6.0 and `okg` before; the import and the command stay `okg`.
OKG_WHEEL = (
    "https://github.com/mitdbg/akmon/releases/download/v{v}/{dist}-{v}-py3-none-any.whl"
)

COMPOSE_HEADER = (
    "# Rendered by archi from archi.yaml and the deployment's chat: block; `archi up`\n"
    "# renders it again before starting. Values in ${...} come from .env, which git\n"
    "# ignores. Edit archi.yaml or deployment.yaml, not this file.\n"
)

DOCKERFILE = """\
# The instance's MCP server: okg, at the version the host runs, and git,
# which okg needs to resolve the deployment's provenance even to serve reads.
FROM docker.io/library/python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \\
 && rm -rf /var/lib/apt/lists/* \\
 && git config --system --add safe.directory '*'
ARG OKG_WHEEL={wheel}
RUN pip install --no-cache-dir "{dist} @ ${{OKG_WHEEL}}"
ENTRYPOINT ["okg"]
"""


class ComposeError(Exception):
    """The container runtime is missing or a compose step failed."""


def okg_distribution(version: str) -> str:
    """The engine's distribution name at `version`: `akmon` from 0.6.0, `okg` before."""
    from packaging.version import InvalidVersion, Version

    try:
        return "okg" if Version(version).release < (0, 6) else "akmon"
    except InvalidVersion:
        return "akmon"


def okg_wheel(version: str) -> str:
    return OKG_WHEEL.format(v=version, dist=okg_distribution(version))


def render_dockerfile(okg_version: str) -> str:
    return DOCKERFILE.format(wheel=okg_wheel(okg_version), dist=okg_distribution(okg_version))


def chat_settings(deployment_dir: Path) -> dict[str, Any]:
    """The deployment's chat: block, as okg resolved it at install time."""
    manifest = yaml.safe_load((Path(deployment_dir) / "deployment.yaml").read_text(encoding="utf-8"))
    return manifest.get("chat") or {}


def mcp_port(deployment_dir: Path) -> int:
    """`chat.mcp.port`: `okg chat sync` looks for the endpoint there and nowhere else."""
    return int(((chat_settings(deployment_dir).get("mcp") or {}).get("port")) or MCP_CONTAINER_PORT)


def render_compose(instance: Instance) -> dict[str, Any]:
    """The compose document for `instance` (no secret values, only ${...})."""
    db = instance.database
    port = mcp_port(instance.deployment_dir) if instance.deployment_dir.is_dir() else MCP_CONTAINER_PORT
    site_name = (chat_settings(instance.deployment_dir).get("ui") or {}).get("site_name") \
        if instance.deployment_dir.is_dir() else None
    inst = str(instance.dir)
    webui_env: dict[str, str] = {
        "DATABASE_URL": "postgresql://postgres:${ARCHI_CHAT_DB_PASSWORD}@chat-db:5432/okg_chat_app",
        "WEBUI_SECRET_KEY": "${OKG_CHAT_WEBUI_SECRET_KEY}",
        # The settings okg's own launcher fixes (CONTAINER_FIXED_ENV).
        "ENABLE_VERSION_UPDATE_CHECK": "false",
        "ENABLE_EASTER_EGGS": "false",
    }
    if site_name:
        webui_env["WEBUI_NAME"] = str(site_name)
    return {
        "name": instance.project,
        "services": {
            "graph-db": {
                "image": GRAPH_DB_IMAGE.format(okg_version=instance.okg_version),
                "init": True,
                "shm_size": GRAPH_DB_SHM,
                "environment": {
                    "POSTGRES_PASSWORD": "${ARCHI_GRAPH_DB_PASSWORD}",
                    "POSTGRES_DB": db,
                    "POSTGRES_INITDB_ARGS": "--locale=C --encoding=UTF8",
                },
                "command": [
                    "postgres",
                    "-c", f"shared_preload_libraries={','.join(PRELOAD_LIBRARIES)}",
                    # pg_cron serves one database: one server per deployment.
                    "-c", f"cron.database_name={db}",
                    "-c", "max_connections=100",
                    "-c", "max_worker_processes=32",
                ],
                "ports": [f"127.0.0.1:{instance.ports.graph_db}:5432"],
                # The AKMON image keeps its cluster in PGDATA=/home/postgres/pgdata/data,
                # not in the stock /var/lib/postgresql/data; a volume mounted there
                # holds nothing, and the graph is lost when the container is recreated.
                "volumes": [f"graph-data:{GRAPH_DB_PGDATA.rsplit('/', 1)[0]}"],
                "healthcheck": {
                    "test": ["CMD-SHELL", f"pg_isready -U postgres -d {db}"],
                    "interval": "2s",
                    "retries": 90,
                },
            },
            "chat-db": {
                "image": CHAT_DB_IMAGE,
                "environment": {
                    "POSTGRES_PASSWORD": "${ARCHI_CHAT_DB_PASSWORD}",
                    "POSTGRES_DB": "okg_chat_app",
                },
                # No published port: only Open WebUI, on this network, reads it.
                "volumes": ["chat-data:/var/lib/postgresql/data"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U postgres -d okg_chat_app"],
                    "interval": "2s",
                    "retries": 90,
                },
            },
            "mcp": {
                "build": {"context": ".", "dockerfile": MCP_DOCKERFILE},
                "image": f"localhost/{instance.project}-mcp:okg-{instance.okg_version}",
                "command": [
                    "mcp-serve", "--deployment", instance.name,
                    "--transport", "streamable-http",
                    "--host", "0.0.0.0", "--port", str(port),
                    "--auth-token-env", "OKG_CHAT_MCP_TOKEN",
                    "--allowed-host", "mcp:*",
                    "--allowed-host", "127.0.0.1:*",
                    "--allowed-host", "localhost:*",
                ],
                "environment": {
                    "OKG_DEPLOYMENTS_DIR": f"{inst}/deployments",
                    "OKG_DSN": f"postgresql://postgres:${{ARCHI_GRAPH_DB_PASSWORD}}@graph-db:5432/{db}",
                    # The least-privilege read role; its password comes from
                    # `okg db role-passwords --apply`, copied into .env by init.
                    "OKG_MCP_RO_DSN": f"postgresql://okg_mcp:${{ARCHI_OKG_MCP_PASSWORD}}@graph-db:5432/{db}",
                    "OKG_CHAT_MCP_TOKEN": "${OKG_CHAT_MCP_TOKEN}",
                },
                # Same absolute path inside: the deployment's manifest names
                # absolute paths, and okg reads the git state around it.
                "volumes": [f"{inst}:{inst}:ro"],
                "ports": [f"127.0.0.1:{instance.ports.mcp}:{port}"],
                "depends_on": {"graph-db": {"condition": "service_healthy"}},
            },
            "webui": {
                "image": WEBUI_IMAGE,
                "environment": webui_env,
                "ports": [f"127.0.0.1:{instance.ports.webui}:8080"],
                # A model server on this host (a local ollama) is reached by name.
                "extra_hosts": ["host.docker.internal:host-gateway"],
                "volumes": ["webui-data:/app/backend/data"],
                "depends_on": {
                    "chat-db": {"condition": "service_healthy"},
                    "mcp": {"condition": "service_started"},
                },
                "healthcheck": {
                    "test": ["CMD-SHELL", "curl -fsS http://localhost:8080/health || exit 1"],
                    "interval": "5s",
                    "start_period": f"{WEBUI_READY_SECONDS}s",
                    "retries": 12,
                },
            },
        },
        "volumes": {"graph-data": {}, "chat-data": {}, "webui-data": {}},
    }


def write_stack_files(instance: Instance) -> list[Path]:
    compose = instance.dir / COMPOSE_FILE
    compose.write_text(
        COMPOSE_HEADER + yaml.safe_dump(render_compose(instance), sort_keys=False, width=100),
        encoding="utf-8",
    )
    dockerfile = instance.dir / MCP_DOCKERFILE
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text(render_dockerfile(instance.okg_version), encoding="utf-8")
    return [compose, dockerfile]


# ---------------------------------------------------------------------------
# The container runtime
# ---------------------------------------------------------------------------


Runner = Callable[..., subprocess.CompletedProcess]


def _works(cmd: Sequence[str], runner: Runner) -> bool:
    try:
        return runner(list(cmd), capture_output=True, text=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def detect_runtime(preferred: Optional[str] = None, runner: Runner = subprocess.run) -> str:
    """docker or podman: the first whose daemon answers and whose compose works."""
    candidates = [preferred] if preferred else ["docker", "podman"]
    tried = []
    for name in candidates:
        if shutil.which(name) is None:
            tried.append(f"{name}: not installed")
            continue
        if not _works([name, "info"], runner):
            tried.append(f"{name}: installed, but `{name} info` fails (is the daemon or VM running?)")
            continue
        if not _works([name, "compose", "version"], runner):
            tried.append(f"{name}: `{name} compose` is not available")
            continue
        return name
    raise ComposeError("no usable container runtime: " + "; ".join(tried))


@dataclass
class Compose:
    instance: Instance
    runner: Runner = subprocess.run

    def base(self) -> list[str]:
        # No --project-directory: podman-compose 1.4 rejects it, and both
        # runtimes default the project directory to the one holding the first
        # -f file, which is the instance directory. run() also sets cwd to it.
        return [
            self.instance.runtime, "compose",
            "-f", str(self.instance.dir / COMPOSE_FILE),
            "--env-file", str(self.instance.dir / ENV_FILE),
            "-p", self.instance.project,
        ]

    def run(self, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        cmd = self.base() + list(args)
        result = self.runner(cmd, cwd=self.instance.dir, text=True,
                             capture_output=capture)
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "")[-2000:] if capture else ""
            raise ComposeError(f"`{' '.join(cmd[1:])}` failed ({result.returncode}) {detail}".strip())
        return result

    def up(self, *services: str, build: bool = False) -> None:
        args = ["up", "-d"]
        if build:
            args.append("--build")
        self.run(*args, *services)

    def down(self, *, volumes: bool = False) -> None:
        self.run("down", *(["--volumes"] if volumes else []))

    def services(self) -> dict[str, dict[str, Any]]:
        """Service name -> {state, health} for the project's containers.

        `compose ps --format json` is asked first. podman-compose 1.4 has no
        such option, so on failure the runtime itself lists the containers
        carrying this project's compose label.
        """
        result = self.run("ps", "--all", "--format", "json", check=False, capture=True)
        if result.returncode != 0:
            result = self.runner(
                [self.instance.runtime, "ps", "--all", "--format", "json",
                 "--filter", f"label=com.docker.compose.project={self.instance.project}"],
                cwd=self.instance.dir, text=True, capture_output=True,
            )
            if result.returncode != 0:
                return {}
        text = (result.stdout or "").strip()
        if not text:
            return {}
        rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines()]
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            labels = row.get("Labels") or {}
            if isinstance(labels, str):  # docker ps: "k=v,k=v"
                labels = dict(item.partition("=")[::2] for item in labels.split(",") if item)
            name = row.get("Service") or labels.get("com.docker.compose.service", "")
            status = str(row.get("Status") or "").lower()
            health = (row.get("Health") or "").lower() or next(
                (h for h in ("unhealthy", "healthy", "starting") if f"({h}" in status), ""
            )
            out[name] = {"state": (row.get("State") or "").lower(), "health": health}
        return out


# ---------------------------------------------------------------------------
# Readiness probes, from the host
# ---------------------------------------------------------------------------


def http_status(url: str, timeout: float = 5.0, headers: Optional[Mapping[str, str]] = None) -> int:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError):
        return 0


def wait_for(what: str, ready: Callable[[], bool], timeout: float, echo=print, every: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    announced = False
    while not ready():
        if time.monotonic() > deadline:
            raise ComposeError(f"{what} was not ready after {int(timeout)} s")
        if not announced:
            echo(f"waiting for {what} ...")
            announced = True
        time.sleep(every)


def graph_db_ready(dsn: str) -> Callable[[], bool]:
    def probe() -> bool:
        try:
            import psycopg

            with psycopg.connect(dsn, connect_timeout=3) as conn:
                conn.execute("select 1")
            return True
        except Exception:
            return False

    return probe
