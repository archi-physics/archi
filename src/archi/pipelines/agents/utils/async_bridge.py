import asyncio
import queue
import threading
from typing import Any, AsyncIterator, Iterator, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

_DONE = object()


class StreamCancelled(Exception):
    """Raised in the consuming thread when the underlying task was cancelled."""


class AsyncStreamBridge:
    """Run an async generator as a cancellable task on a background loop and
    consume its items from a synchronous iterator.

    The chat app serves requests from worker threads, where a model call blocks
    the thread inside a socket read and cannot be interrupted. Running the
    pipeline's astream() as an asyncio task changes that: cancel() raises
    CancelledError at the task's current await point, which unwinds through
    httpx and closes the HTTP connection — the signal Ollama (and any other
    provider) accepts to abort an in-flight generation, including mid-prefill.

    Items are pumped into an unbounded thread-safe queue (SSE events are small
    and consumed promptly, and a bounded put would block the shared event
    loop). cancel() is thread-safe and idempotent.
    """

    def __init__(self, agen: AsyncIterator[Any], loop: asyncio.AbstractEventLoop):
        self._agen = agen
        self._loop = loop
        self._queue: queue.Queue = queue.Queue()
        self._task: Optional[asyncio.Task] = None
        self._cancel_requested = threading.Event()
        self._started = threading.Event()
        loop.call_soon_threadsafe(self._start)

    def _start(self) -> None:
        self._task = self._loop.create_task(self._pump())
        self._started.set()
        if self._cancel_requested.is_set():
            self._task.cancel()

    async def _pump(self) -> None:
        try:
            async for item in self._agen:
                self._queue.put(item)
        except asyncio.CancelledError:
            self._queue.put(StreamCancelled("stream task cancelled"))
            raise
        except BaseException as exc:  # surfaced to the consuming thread
            self._queue.put(exc)
        else:
            self._queue.put(_DONE)
        finally:
            await _aclose_quietly(self._agen)

    def __iter__(self) -> Iterator[Any]:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                # Normally a terminal marker arrives; this guards against the
                # task dying without one (e.g. the loop shut down under it).
                if self._started.is_set() and self._task is not None and self._task.done() and self._queue.empty():
                    logger.warning("Async stream task ended without terminal marker.")
                    return
                continue
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def cancel(self) -> None:
        """Cancel the underlying task from any thread; safe to call repeatedly."""
        if self._cancel_requested.is_set():
            return
        self._cancel_requested.set()
        if self._started.is_set() and self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)


async def _aclose_quietly(agen: Any) -> None:
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        pass
