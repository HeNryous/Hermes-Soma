"""Tests for soma.transport.MessagePipeline — debounce + serialization.

TelegramTransport itself is exercised indirectly (lazy-imports the
SDK; runtime behavior covered by manual Phase-7 smoke). The pipeline
is where the real logic lives and is fully testable in isolation.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.transport import MessagePipeline


class MessagePipelineTest(unittest.TestCase):
    def test_single_message_fires_after_debounce(self):
        batches: list[str] = []

        async def handler(text):
            batches.append(text)

        async def scenario():
            pipe = MessagePipeline(handler, debounce_seconds=0.05)
            await pipe.submit("hello")
            await asyncio.sleep(0.15)

        asyncio.run(scenario())
        self.assertEqual(batches, ["hello"])

    def test_burst_messages_collapse_into_one_batch(self):
        batches: list[str] = []

        async def handler(text):
            batches.append(text)

        async def scenario():
            pipe = MessagePipeline(handler, debounce_seconds=0.10)
            await pipe.submit("a")
            await asyncio.sleep(0.02)
            await pipe.submit("b")
            await asyncio.sleep(0.02)
            await pipe.submit("c")
            await asyncio.sleep(0.20)

        asyncio.run(scenario())
        self.assertEqual(batches, ["a\nb\nc"])

    def test_handler_exception_does_not_crash_pipeline(self):
        batches: list[str] = []

        async def handler(text):
            if text == "boom":
                raise RuntimeError("boom")
            batches.append(text)

        async def scenario():
            pipe = MessagePipeline(handler, debounce_seconds=0.05)
            await pipe.submit("boom")
            await asyncio.sleep(0.15)
            await pipe.submit("ok")
            await asyncio.sleep(0.15)

        asyncio.run(scenario())
        self.assertEqual(batches, ["ok"])

    def test_empty_submit_ignored(self):
        batches: list[str] = []

        async def handler(text):
            batches.append(text)

        async def scenario():
            pipe = MessagePipeline(handler, debounce_seconds=0.05)
            await pipe.submit("   ")
            await pipe.submit("")
            await asyncio.sleep(0.15)

        asyncio.run(scenario())
        self.assertEqual(batches, [])

    def test_flush_now_drains_immediately(self):
        batches: list[str] = []

        async def handler(text):
            batches.append(text)

        async def scenario():
            pipe = MessagePipeline(handler, debounce_seconds=10.0)
            await pipe.submit("urgent")
            await pipe.flush_now()

        asyncio.run(scenario())
        self.assertEqual(batches, ["urgent"])

    def test_serialization_blocks_concurrent_handlers(self):
        order: list[str] = []
        in_flight = asyncio.Event()

        async def slow_handler(text):
            order.append(f"start:{text}")
            in_flight.set()
            await asyncio.sleep(0.10)
            order.append(f"end:{text}")

        async def scenario():
            pipe = MessagePipeline(slow_handler, debounce_seconds=0.05)
            await pipe.submit("first")
            # Wait until the first handler is actively running.
            await asyncio.sleep(0.08)
            await pipe.submit("second")
            await asyncio.sleep(0.40)

        asyncio.run(scenario())
        # The second handler must not interleave with the first.
        self.assertEqual(
            order,
            ["start:first", "end:first", "start:second", "end:second"],
        )


if __name__ == "__main__":
    unittest.main()
