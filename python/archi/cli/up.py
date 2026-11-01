"""`archi up` and `archi down`: the instance's compose project.

`up` starts graph Postgres, chat Postgres, the MCP server and Open WebUI on
one network, waits until each answers, signs in to the site as its admin,
points it at a model server, and runs `okg chat sync`, which registers the
MCP connection and the preset and proves both by readback. `down` stops the
project and keeps every volume; `down --purge --yes` also deletes them.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from archi.cli.init import commit_all, declare_runtime_database, ensure_ops_tables, has_automations
from archi.stack import compose as stack_compose
from archi.stack import webui
from archi.stack.instance import Instance, InstanceError, graph_dsn, read_env
from archi.stack.okg import Okg, OkgError

#: Where a local ollama listens, from the host and from inside a container.
OLLAMA_HOST_URL = "http://127.0.0.1:11434"
OLLAMA_CONTAINER_URL = "http://host.docker.internal:11434"


def site_url(instance: Instance) -> str:
    return f"http://127.0.0.1:{instance.ports.webui}"


def mcp_url(instance: Instance) -> str:
    return f"http://127.0.0.1:{instance.ports.mcp}/mcp"


def up(
    instance: Instance,
    *,
    ollama_url: Optional[str] = None,
    sync: bool = True,
    echo: Callable[[str], None] = print,
    compose_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    okg_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    secrets = read_env(instance.env_path)
    if "ARCHI_OKG_MCP_PASSWORD" not in secrets:
        raise InstanceError(
            f"{instance.env_path} has no ARCHI_OKG_MCP_PASSWORD: `archi init` did not finish "
            "(it sets the MCP role password after the first publish). Start over with "
            "`archi down --purge --yes` and a fresh `archi init`"
        )
    stack_compose.write_stack_files(instance)
    # An instance made before archi declared the runtime's database (okg lint
    # deployment.runtime.database_undeclared) is repaired here; idempotent otherwise.
    declare_runtime_database(instance.deployment_dir, instance.database)
    commit_all(instance.dir, "archi up: render the stack")
    compose = stack_compose.Compose(instance, runner=compose_runner)
    echo("starting graph-db, chat-db, mcp and webui (the first start pulls and builds images) ...")
    compose.up(build=True)

    stack_compose.wait_for(
        "the graph database", stack_compose.graph_db_ready(graph_dsn(instance, secrets)), 300, echo
    )
    okg = Okg(instance, secrets, runner=okg_runner, echo=echo)
    if has_automations(instance.dir):
        # Idempotent. `archi init` already did this; it matters for an instance whose database was
        # restored or purged, and it never leaves the DBOS schema for a hand step.
        okg.run("runtime", "bootstrap", "--deployment", instance.name)
        ensure_ops_tables(instance, secrets)
        echo("ops runtime bootstrapped. The stack has no runtime worker yet: run `okg runtime worker "
             f"--deployment {instance.name}` (with ARCHI_INSTANCE, OKG_DSN, ARCHI_OPS_DEPLOYMENT and any source credentials from .env set) "
             "for automations to run")
    # Unauthenticated, the MCP endpoint answers 401: up, and refusing strangers.
    stack_compose.wait_for(
        "the MCP server", lambda: stack_compose.http_status(mcp_url(instance)) in (401, 405, 406), 300, echo
    )
    stack_compose.wait_for(
        "Open WebUI (its first start loads an embedding model)",
        lambda: stack_compose.http_status(f"{site_url(instance)}/health") == 200,
        stack_compose.WEBUI_READY_SECONDS, echo, every=5,
    )
    if not sync:
        echo(f"up: {site_url(instance)} (chat sync skipped)")
        return

    token = webui.admin_token(site_url(instance), instance.admin_email, secrets["ARCHI_ADMIN_PASSWORD"])
    if ollama_url is None and stack_compose.http_status(f"{OLLAMA_HOST_URL}/api/tags") == 200:
        ollama_url = OLLAMA_CONTAINER_URL
    if ollama_url:
        webui.configure_ollama(site_url(instance), token, ollama_url)
        echo(f"model server: ollama at {ollama_url}")
    else:
        echo("no model server configured: no ollama on this host and no --ollama-url. Add one "
             "under Settings > Admin > Connections; the graph tools work without it over MCP")

    port = stack_compose.mcp_port(instance.deployment_dir)
    okg.run(
        "chat", "sync", "--deployment", instance.name,
        "--mcp-probe-url", mcp_url(instance),
        # The site reaches the MCP server by its service name on the network.
        "--mcp-instance-url", f"http://mcp:{port}/mcp",
        extra_env={"OKG_CHAT_INSTANCE_URL": site_url(instance), "OKG_CHAT_ADMIN_TOKEN": token},
    )
    echo(f"up: chat {site_url(instance)}  (admin {instance.admin_email}; password: "
         f"ARCHI_ADMIN_PASSWORD in {instance.env_path})")
    echo(f"    MCP {mcp_url(instance)}  (bearer token: OKG_CHAT_MCP_TOKEN in {instance.env_path})")
    echo("    pick the preset named after the site in the model picker: the graph tools are "
         "bound to it, not to the raw model")
    if os.environ.get("SSH_CONNECTION"):
        echo(f"    this host is remote; from your machine: ssh -L {instance.ports.webui}:127.0.0.1:"
             f"{instance.ports.webui} <this-host>")


def down(instance: Instance, *, purge: bool = False, echo=print,
         compose_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
    stack_compose.Compose(instance, runner=compose_runner).down(volumes=purge)
    if purge:
        echo(f"stopped {instance.project} and deleted its volumes (graph, chat, webui). "
             f"The data root {instance.data_root} and the instance directory are untouched")
    else:
        echo(f"stopped {instance.project}; volumes kept. `archi up` starts it again")


def cmd_up(args: argparse.Namespace) -> int:
    try:
        up(Instance.load(args.instance), ollama_url=args.ollama_url, sync=not args.no_sync)
    except (InstanceError, OkgError, stack_compose.ComposeError, webui.WebUIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    if args.purge and not args.yes:
        print("error: --purge deletes the graph, chat and webui volumes; add --yes to confirm",
              file=sys.stderr)
        return 1
    try:
        down(Instance.load(args.instance), purge=args.purge)
    except (InstanceError, stack_compose.ComposeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def add_parsers(sub) -> None:
    p = sub.add_parser("up", help="start the instance's compose project and wire the chat to the graph")
    p.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    p.add_argument("--ollama-url",
                   help="model server as the webui container reaches it "
                        f"(default: {OLLAMA_CONTAINER_URL} when an ollama answers on this host)")
    p.add_argument("--no-sync", action="store_true",
                   help="start the containers only; skip the admin sign-in and `okg chat sync`")
    p.set_defaults(func=cmd_up)

    d = sub.add_parser("down", help="stop the instance's compose project (volumes are kept)")
    d.add_argument("--instance", type=Path, default=Path("."), help="instance directory")
    d.add_argument("--purge", action="store_true", help="also delete the volumes (needs --yes)")
    d.add_argument("--yes", action="store_true")
    d.set_defaults(func=cmd_down)
