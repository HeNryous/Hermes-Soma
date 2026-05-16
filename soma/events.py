"""Append-only JSONL event log for Soma.

Records every notable thing that happens — incoming prompts, outgoing
responses, memory writes, background ticks, errors. The log is the
ground truth a replay or post-hoc analysis runs against.

Thread-safe via a single lock. Async callers should invoke record_async
or wrap record() in asyncio.to_thread.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


# The canonical event-type strings the curator and crystallization layers
# look for. Add new ones here rather than scattering string literals.
EVENT_PROMPT_RECEIVED = "prompt_received"
EVENT_RESPONSE_SENT = "response_sent"
EVENT_MEMORY_WRITTEN = "memory_written"
EVENT_MEMORY_FUSED = "memory_fused"
EVENT_BACKGROUND_TICK = "background_tick"
EVENT_CURATOR_ACTION = "curator_action"
EVENT_NOTIFY_SENT = "notify_sent"
EVENT_NOTIFY_SKIPPED = "notify_skipped"
EVENT_NOTIFY_FAILED = "notify_failed"
EVENT_CRYSTALLIZE = "crystallize"
EVENT_ERROR = "error"


class EventLog:
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, type: str, **data: Any) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "ts": time.time(),
            "type": type,
            **data,
        }
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
        return event

    async def record_async(self, type: str, **data: Any) -> Dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(self.record, type, **data)

    def replay(
        self,
        *,
        type: Optional[str] = None,
        since: Optional[float] = None,
    ) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if type is not None and event.get("type") != type:
                    continue
                if since is not None and event.get("ts", 0) < since:
                    continue
                yield event
