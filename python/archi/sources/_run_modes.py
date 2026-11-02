"""Run modes in which a source run covers its whole scope.

okg passes a source its run mode and trusts the source's
``completed_scope`` claim, so a reader must know which modes complete a
scope. okg keeps the table in ``okg.substrate.source_run_modes``
(``COMPLETED_SCOPE_RUN_MODES``), but a connector may not import
``okg.substrate``: ``okg lint deployments`` reports DEPLOYMENT-001 for
it. So archi keeps this copy, and
``tests/sources/test_run_modes.py`` asserts it equals okg's set at the
pinned okg commit, which catches drift at every pin bump.

Once the okg pin includes ``okg.deployment.COMPLETED_SCOPE_RUN_MODES``
(okg branch ``claude/connector-complete-scope-modes``), readers should
import that instead and this module should go.

Per profile, at okg ``ac078aabd``: ``discovery_crawl`` completes in
``scope_complete``, ``mutable_api`` in ``reconcile``, and
``reference_catalog`` in ``release_new`` and ``release_unchanged``.

:data:`RELEASE_RUN_MODES` are the reference_catalog profile's own modes,
and :func:`input_content_checkpoint` is the cursor a reference catalog
returns when it deliberately does not claim a complete scope in them
(DBS and CondDB; see those readers).
"""

from __future__ import annotations

from collections.abc import Iterable

from okg.deployment import Checkpoint

from archi.auth.cache import content_hash, resolve_repo_path

COMPLETED_SCOPE_RUN_MODES: frozenset[str] = frozenset(
    {"scope_complete", "reconcile", "release_new", "release_unchanged"}
)

#: The run modes okg gives a reference_catalog source.
RELEASE_RUN_MODES: frozenset[str] = frozenset({"release_new", "release_unchanged"})

#: The one cursor key :func:`input_content_checkpoint` writes.
INPUT_CONTENT_CURSOR_KEY = "input_content_sha256"


def input_content_checkpoint(
    paths: Iterable[str], *, base: str | None = None
) -> Checkpoint:
    """A cursor naming the exact content of the input files a run read.

    okg accepts a later run that emits nothing as a no-op only when the
    source already has a stored cursor or a reconcile time; a run that
    claims no complete scope never sets the reconcile time, so it needs
    this cursor. The value is the SHA-256 of each file's bytes (with its
    configured path as a separator), not its mtime, so a changed export
    yields a different cursor. Files that do not exist are left out,
    matching the change probe over the same paths.

    The files are hashed after the reader parsed them, so a file rewritten
    in between yields a cursor for the newer bytes. That is harmless while
    nothing compares the cursor (no reader declares
    ``probe_short_circuit_safe``, so okg re-reads every run); a reader
    that wants okg to skip unchanged reads must first hash the bytes it
    parsed.
    """
    present = [p for p in paths if resolve_repo_path(p, base=base).is_file()]
    return Checkpoint(
        next_cursor={INPUT_CONTENT_CURSOR_KEY: content_hash(present, base=base)}
    )
