"""Tests for soma.main.SomaApp.handle_user_message — the per-turn pipeline.

Verifies the wiring: prompt event recorded, context built and passed to
the engine, history updated from the engine's result, reply sanitized,
response event recorded, extraction spawned in background, written
memories surfaced as events.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.context_builder import ContextBuilder
from soma.events import (
    EVENT_MEMORY_WRITTEN,
    EVENT_PROMPT_RECEIVED,
    EVENT_RESPONSE_SENT,
    EventLog,
)
from soma.main import SomaApp, SomaConfig
from soma.memory_extractor import MemoryExtractor
from soma.memory_store import MemoryStore


class _DictEmbedder:
    def __init__(self):
        self._slots: dict[str, int] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self._slots:
            self._slots[text] = len(self._slots)
        vec = [0.0] * 32
        vec[self._slots[text] % 32] = 1.0
        return vec


class _FakeEngine:
    """Stand-in for SomaEngine. Records calls; returns a canned result."""

    def __init__(self, *, final_response: str = "ok", messages=None):
        self.calls = []
        self._final_response = final_response
        self._messages = messages or [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "ok"},
        ]

    async def run_turn(self, user_message, *, conversation_history=None, system_message=None):
        self.calls.append({
            "user_message": user_message,
            "conversation_history": conversation_history,
            "system_message": system_message,
        })
        return {
            "final_response": self._final_response,
            "messages": self._messages,
            "api_calls": 1,
            "completed": True,
        }


class _FakeExtractor:
    """Stand-in for MemoryExtractor.extract_async. Returns a fixed list."""

    def __init__(self, written=None):
        self.calls = []
        self._written = written or []

    async def extract_async(self, user_text, assistant_text, **_):
        self.calls.append((user_text, assistant_text))
        return list(self._written)


class _FakeTransport:
    """Stand-in for TelegramTransport so SomaApp can be constructed without a token."""

    def __init__(self):
        self.sent: list[str] = []

    async def run_forever(self):
        await asyncio.Event().wait()  # never returns; tests don't call this

    async def send(self, text):
        self.sent.append(text)


def _make_app(*, engine=None, extractor=None) -> tuple[SomaApp, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="soma_test_"))
    config = SomaConfig(
        telegram_token="test-token",
        telegram_user_id=1,
        data_dir=tmp_dir,
    )
    embedder = _DictEmbedder()
    store = MemoryStore(tmp_dir / "memories.jsonl", embedder)
    events = EventLog(tmp_dir / "events.jsonl")
    extractor = extractor or _FakeExtractor()
    context_builder = ContextBuilder(store)
    transport = _FakeTransport()
    app = SomaApp(
        config,
        engine=engine or _FakeEngine(),
        embedder=embedder,
        store=store,
        extractor=extractor,
        context_builder=context_builder,
        events=events,
        transport=transport,
    )
    return app, tmp_dir


def _cleanup(tmp_dir: Path):
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class SomaAppPipelineTest(unittest.TestCase):
    def test_happy_path_records_events_and_returns_reply(self):
        engine = _FakeEngine(final_response="hello user")
        extractor = _FakeExtractor()
        app, tmp_dir = _make_app(engine=engine, extractor=extractor)
        self.addCleanup(_cleanup, tmp_dir)

        async def scenario():
            reply = await app.handle_user_message("hi there")
            # Let the background extraction task run.
            await asyncio.sleep(0)
            await asyncio.wait_for(asyncio.gather(*app._bg_tasks), timeout=2.0)
            return reply

        reply = asyncio.run(scenario())

        self.assertEqual(reply, "hello user")
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0]["user_message"], "hi there")

        events = list(app.events.replay())
        types = [e["type"] for e in events]
        self.assertIn(EVENT_PROMPT_RECEIVED, types)
        self.assertIn(EVENT_RESPONSE_SENT, types)

        self.assertEqual(extractor.calls, [("hi there", "hello user")])

    def test_extraction_writes_memory_events(self):
        from soma.memory_store import MemoryRecord

        written = [
            MemoryRecord(
                id="mem_xxx",
                type="semantic",
                content="User works at Supermicro",
                tags=["domain"],
                use_count=1,
                source="extractor",
            )
        ]
        extractor = _FakeExtractor(written=written)
        app, tmp_dir = _make_app(extractor=extractor)
        self.addCleanup(_cleanup, tmp_dir)

        async def scenario():
            await app.handle_user_message("Ich arbeite bei Supermicro")
            await asyncio.sleep(0)
            await asyncio.wait_for(asyncio.gather(*app._bg_tasks), timeout=2.0)

        asyncio.run(scenario())

        mem_events = list(app.events.replay(type=EVENT_MEMORY_WRITTEN))
        self.assertEqual(len(mem_events), 1)
        self.assertEqual(mem_events[0]["id"], "mem_xxx")
        # The event's own 'type' field must remain EVENT_MEMORY_WRITTEN —
        # the memory record's own type is carried under 'memory_type' to
        # avoid clobbering the event discriminator.
        self.assertEqual(mem_events[0]["type"], EVENT_MEMORY_WRITTEN)
        self.assertEqual(mem_events[0]["memory_type"], "semantic")
        self.assertEqual(mem_events[0]["content"], "User works at Supermicro")

    def test_engine_failure_returns_error_reply(self):
        class _BoomEngine:
            async def run_turn(self, *_a, **_k):
                raise RuntimeError("provider down")

        app, tmp_dir = _make_app(engine=_BoomEngine())
        self.addCleanup(_cleanup, tmp_dir)

        async def scenario():
            return await app.handle_user_message("hi")

        reply = asyncio.run(scenario())
        self.assertIsNotNone(reply)
        self.assertIn("internal error", reply.lower())

        types = [e["type"] for e in app.events.replay()]
        self.assertIn("error", types)

    def test_history_persists_across_turns(self):
        engine = _FakeEngine(
            final_response="reply",
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
            ],
        )
        app, tmp_dir = _make_app(engine=engine)
        self.addCleanup(_cleanup, tmp_dir)

        async def scenario():
            await app.handle_user_message("first")
            await app.handle_user_message("second")

        asyncio.run(scenario())
        # Second turn must have been called with the messages list from
        # the first turn as conversation_history.
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.calls[1]["conversation_history"][0]["content"], "first")

    def test_reply_is_sanitized(self):
        engine = _FakeEngine(
            final_response="<thinking>plan</thinking>the answer is 42"
        )
        app, tmp_dir = _make_app(engine=engine)
        self.addCleanup(_cleanup, tmp_dir)

        async def scenario():
            return await app.handle_user_message("q")

        reply = asyncio.run(scenario())
        self.assertEqual(reply, "the answer is 42")


if __name__ == "__main__":
    unittest.main()
