"""Rate-limited proactive notification bridge.

The curator decides what to say; Notifier decides whether now is the
right moment. The 1-per-24h ceiling lives here, not in the curator —
that way the curator can be aggressive about wanting to notify and
the user still isn't spammed.

State (the timestamp of the last successful send) persists in
data/soma/notify_state.json, so restarts don't reset the rate window.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from .events import (
    EVENT_NOTIFY_FAILED,
    EVENT_NOTIFY_SENT,
    EVENT_NOTIFY_SKIPPED,
    EventLog,
)

logger = logging.getLogger(__name__)


DEFAULT_MIN_INTERVAL_SECONDS = 24 * 3600.0


SendCallback = Callable[[str], Awaitable[None]]


class Notifier:
    """Rate-limits proactive sends. Returns True iff the message was delivered."""

    def __init__(
        self,
        send: SendCallback,
        *,
        state_path: os.PathLike | str,
        events: Optional[EventLog] = None,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ):
        self.send = send
        self.state_path = Path(state_path)
        self.events = events
        self.min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()

    # -- Public API ----------------------------------------------------------

    async def maybe_notify(self, message: str, *, now: Optional[float] = None) -> bool:
        message = (message or "").strip()
        if not message:
            return False
        now = now if now is not None else time.time()
        last = self._load_last_sent()
        if last and (now - last) < self.min_interval_seconds:
            wait = int(self.min_interval_seconds - (now - last))
            if self.events:
                await self.events.record_async(
                    EVENT_NOTIFY_SKIPPED,
                    reason="rate_limit",
                    seconds_until_allowed=wait,
                    message=message[:200],
                )
            return False
        try:
            await self.send(message)
        except Exception as exc:
            logger.exception("soma: notifier send failed")
            if self.events:
                await self.events.record_async(
                    EVENT_NOTIFY_FAILED,
                    error=str(exc),
                    message=message[:200],
                )
            return False
        self._save_last_sent(now)
        if self.events:
            await self.events.record_async(EVENT_NOTIFY_SENT, message=message)
        return True

    # -- State persistence ---------------------------------------------------

    def _load_last_sent(self) -> Optional[float]:
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        ts = data.get("last_sent_ts")
        try:
            ts = float(ts) if ts is not None else None
        except (TypeError, ValueError):
            ts = None
        return ts or None

    def _save_last_sent(self, ts: float) -> None:
        with self._lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps({"last_sent_ts": ts}), encoding="utf-8")
            os.replace(tmp, self.state_path)
