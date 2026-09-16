"""Bridge this distribution's readers to the substrate's adapter contract.

The readers return ``ConnectorRun``, the public SDK's run result, which
deliberately omits the substrate-only fields (raw cursor, applied token,
bootstrap identity). A source registry names a class, and the substrate runs
that class through its ``SourceAdapter`` contract, reading exactly those
fields — so a reader cannot be registered directly. Since the connector
migration (#641) that combination raised
``AttributeError: 'ConnectorRun' object has no attribute 'next_cursor'`` on
every run.

``okg.deployment.ConnectorAdapter`` is the framework's supported bridge: it
drives a connector and returns what the runner expects. Readers keep the
``run(run_id, mode=...)`` signature they have, so this module wraps them in
the connector shape the adapter drives. ``bundles/cern-team/source-defaults``
names the adapter classes, so an installed instance runs the same reader code
through the supported contract.
"""

from __future__ import annotations

from typing import Any

from okg.deployment import ConnectorAdapter


class _ReaderConnector:
    """Call a reader that takes ``(run_id, mode=...)`` from a run context."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader
        self.name = reader.name
        self.profile = reader.profile

    def run(self, ctx: Any) -> Any:
        if ctx.sync_scope:
            # These readers replay their whole selected input; a narrowed
            # scope would silently return the full walk and claim it was
            # the narrowed one.
            raise ValueError(
                f"{self.name} does not support narrowed sync scopes",
            )
        return self._reader.run(ctx.run_id, mode=ctx.mode)


class ReaderAdapter(ConnectorAdapter):
    """Construct one reader from its registry parameters and drive it.

    Subclasses set ``reader_class`` and mirror the reader's ``profile`` and
    ``change_probe_kind`` as class attributes: the substrate reads both from
    the class, without instantiating it, before it constructs anything.
    """

    reader_class: Any = None

    def __init__(self, **params: Any) -> None:
        reader = type(self).reader_class(**params)
        super().__init__(_ReaderConnector(reader))
        #: The wrapped reader. Preflight, change probes and cache paths stay
        #: the reader's own behavior; this class adds no source semantics.
        self.reader = reader
        self.change_probe = reader.change_probe

    def preflight(self, *args: Any, **kwargs: Any) -> Any:
        return self.reader.preflight(*args, **kwargs)

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return self.reader.cache_paths
