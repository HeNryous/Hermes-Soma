"""Tests for soma.events — append-only JSONL event log."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.events import (
    EVENT_MEMORY_WRITTEN,
    EVENT_PROMPT_RECEIVED,
    EVENT_RESPONSE_SENT,
    EventLog,
)


class EventLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.tmp.close()
        self.addCleanup(lambda: os.path.exists(self.tmp.name) and os.unlink(self.tmp.name))

    def test_record_appends(self):
        log = EventLog(self.tmp.name)
        log.record(EVENT_PROMPT_RECEIVED, text="hi")
        log.record(EVENT_RESPONSE_SENT, text="hello")
        events = list(log.replay())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], EVENT_PROMPT_RECEIVED)
        self.assertEqual(events[0]["text"], "hi")
        self.assertEqual(events[1]["type"], EVENT_RESPONSE_SENT)
        self.assertIn("ts", events[0])

    def test_replay_filter_by_type(self):
        log = EventLog(self.tmp.name)
        log.record(EVENT_PROMPT_RECEIVED, text="a")
        log.record(EVENT_MEMORY_WRITTEN, id="m1")
        log.record(EVENT_PROMPT_RECEIVED, text="b")
        prompts = list(log.replay(type=EVENT_PROMPT_RECEIVED))
        self.assertEqual([e["text"] for e in prompts], ["a", "b"])

    def test_replay_filter_by_since(self):
        log = EventLog(self.tmp.name)
        log.record(EVENT_PROMPT_RECEIVED, text="old")
        # Forge a future event to verify the since cutoff.
        import json
        with open(self.tmp.name, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 9_999_999_999.0, "type": "x"}) + "\n")
        recent = list(log.replay(since=9_999_999_000.0))
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["type"], "x")

    def test_replay_skips_bad_lines(self):
        with open(self.tmp.name, "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write('{"ts": 1.0, "type": "x"}\n')
            f.write("\n")
        log = EventLog(self.tmp.name)
        events = list(log.replay())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "x")

    def test_replay_missing_file_yields_nothing(self):
        path = os.path.join(tempfile.gettempdir(), "nonexistent_soma_events.jsonl")
        if os.path.exists(path):
            os.unlink(path)
        log = EventLog(path)
        self.assertEqual(list(log.replay()), [])

    def test_async_record(self):
        log = EventLog(self.tmp.name)
        event = asyncio.run(log.record_async(EVENT_PROMPT_RECEIVED, text="async"))
        self.assertEqual(event["text"], "async")
        events = list(log.replay())
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
