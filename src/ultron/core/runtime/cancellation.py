"""
ultron.core.runtime.cancellation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cooperative cancellation token for AgentRuntime.
"""

from __future__ import annotations

import asyncio


class CancellationToken:
    """
    Cooperative cancellation handle passed into execution workflows.
    """

    def __init__(self) -> None:
        self._is_cancelled = False
        self._reason: str | None = None
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "Execution cancelled") -> None:
        """Flags cancellation and sets the internal asyncio event."""
        if not self._is_cancelled:
            self._is_cancelled = True
            self._reason = reason
            self._event.set()

    def check(self) -> None:
        """Raises asyncio.CancelledError if cancellation was requested."""
        if self._is_cancelled:
            raise asyncio.CancelledError(self._reason or "Execution cancelled")

    async def wait(self) -> None:
        """Waits until cancellation is flagged."""
        await self._event.wait()
