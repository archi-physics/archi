from typing import Optional, Any
import asyncio
import threading
from src.utils.logging import get_logger

logger = get_logger(__name__)

class AsyncLoopThread:
    """
    A dedicated background thread running a single event loop.

    This ensures all async operations (MCP client init, tool calls) happen
    on the same event loop, preventing ClosedResourceError.

    Usage:
        runner = AsyncLoopThread.get_instance()
        result = runner.run(some_async_coroutine())
    """

    _instance: Optional["AsyncLoopThread"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._started = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="mcp-async-loop"
        )
        self.thread.start()

        # Wait for the loop to actually start before returning
        if not self._started.wait(timeout=10.0):
            raise RuntimeError("Failed to start async loop thread")
        logger.info("Background async loop started for MCP operations")

    def _run(self):
        """Run the event loop forever in the background thread."""
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

    def run(self, coro, timeout: Optional[float] = 120.0) -> Any:
        """
        Schedule a coroutine on the background loop and wait for result.

        Args:
            coro: An awaitable coroutine
            timeout: Maximum seconds to wait (default 120s for MCP operations)

        Returns:
            The result of the coroutine

        Raises:
            TimeoutError: If the coroutine doesn't complete in time
            Any exception raised by the coroutine
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)

    def in_loop_thread(self) -> bool:
        """Return True if called from the background event-loop thread."""
        return threading.current_thread() is self.thread
        # or: return threading.get_ident() == self.thread.ident

    @classmethod
    def get_instance(cls) -> "AsyncLoopThread":
        """Get or create the singleton async runner instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern for thread safety
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def shutdown(self):
        """Gracefully shutdown the background loop."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5.0)
        logger.info("Background async loop stopped")
