import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

from src.archi.pipelines.agents.utils.mcp_utils import AsyncLoopThread


def test_run_cancels_coroutine_when_timeout_expires():
    runner = AsyncLoopThread()
    started = threading.Event()
    cleaned_up = threading.Event()
    release = None

    async def wait_forever():
        nonlocal release
        release = asyncio.Event()
        started.set()
        try:
            await release.wait()
        finally:
            cleaned_up.set()

    try:
        with pytest.raises(FutureTimeoutError):
            runner.run(wait_forever(), timeout=0.01)

        assert started.is_set()
        assert cleaned_up.wait(timeout=0.5)
    finally:
        # Let the old implementation finish cleanly after demonstrating that it
        # failed to cancel the coroutine, avoiding a leaked task in the test itself.
        if not cleaned_up.is_set() and release is not None:
            runner.loop.call_soon_threadsafe(release.set)
            cleaned_up.wait(timeout=1.0)
        runner.shutdown()
