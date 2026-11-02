"""Stand-ins for the scratch run (PACT r3-ops-runtime). Not for production.

`write_marker` is a live tool that stands in for a real mediator: its effect
is one row in a scratch table, so a test can show whether a mediator ran.
`marker_outcome` is the matching outcome check. `write_marker` is idempotent: the marker's key is the
run's idempotency key, so a repeat with the same key writes nothing. A note `slow:N` makes it linger N
seconds after writing, so a test can kill the worker in the middle of the call.

A mediator is named `module:function` (as in a tool pack's `live_tools`) and
is called as `function(payload=..., context=...)`, returning
`{"status": "succeeded" | "failed", "result": "..."}`.
"""
from __future__ import annotations

import importlib
import json
import time
from typing import Any, Callable, Mapping

DDL = """
CREATE SCHEMA IF NOT EXISTS archi_scratch;
CREATE TABLE IF NOT EXISTS archi_scratch.marker (
    idempotency_key text PRIMARY KEY,
    proposal_id text NOT NULL,
    payload jsonb NOT NULL,
    written_at timestamptz NOT NULL DEFAULT now()
);
"""


def resolve_mediator(entry: str) -> Callable[..., Mapping[str, Any]]:
    """The callable a mediator names: `module:function`, or a registered workflow step that has an implementation."""
    from archi.ops.steps import local_step

    step = local_step(entry)
    if step is not None:
        entry = step.impl
    module, _, name = entry.partition(":")
    if not module or not name:
        raise ValueError(f"mediator {entry!r} is not module:function")
    return getattr(importlib.import_module(module), name)


def write_marker(*, payload: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, str]:
    """Insert the marker row. `context` carries `dsn`, `proposal_id` and `idempotency_key`."""
    import psycopg

    with psycopg.connect(context["dsn"], autocommit=True) as conn:
        conn.execute(DDL)
        conn.execute(
            "INSERT INTO archi_scratch.marker (idempotency_key, proposal_id, payload) VALUES (%s, %s, %s) "
            "ON CONFLICT (idempotency_key) DO NOTHING",
            (context["idempotency_key"], context["proposal_id"], json.dumps(dict(payload))),
        )
    note = str(payload.get("note", ""))
    if note.startswith("slow:"):  # scratch runs only: linger after the effect, so a test can kill the worker mid-call
        time.sleep(float(note.split(":", 1)[1]))
    return {"status": "succeeded", "result": f"marker written for {context['proposal_id']}"}


def marker_outcome(*, dsn: str, entity: str) -> tuple[str, str]:
    """`(result, resolution)`: resolved when a marker names the entity."""
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(DDL)
        row = conn.execute(
            "SELECT count(*) FROM archi_scratch.marker WHERE payload->>'entity' = %s", (entity,)).fetchone()
    if row[0]:
        return "resolved", f"marker present for {entity}"
    return "not_resolved", f"no marker for {entity}"
