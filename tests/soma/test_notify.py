"""Tests for soma.notify — rate limit, state persistence, event logging."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.events import (
    EVENT_NOTIFY_FAILED,
    EVENT_NOTIFY_SENT,
    EVENT_NOTIFY_SKIPPED,
    EventLog,
)
from soma.notify import Notifier


def _setup() -> tuple[Path, EventLog, list[str]]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="soma_notify_test_"))
    events = EventLog(tmp_dir / "events.jsonl")
    sent: list[str] = []

    async def send(text):
        sent.append(text)

    return tmp_dir, events, sent, send  # type: ignore


def _cleanup(tmp_dir: Path):
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class NotifierBasicsTest(unittest.TestCase):
    def test_first_send_succeeds(self):
        tmp_dir, events, sent, send = _setup()
        self.addCleanup(_cleanup, tmp_dir)
        notifier = Notifier(
            send,
            state_path=tmp_dir / "notify_state.json",
            events=events,
            min_interval_seconds=3600,
        )
        delivered = asyncio.run(notifier.maybe_notify("hello"))
        self.assertTrue(delivered)
        self.assertEqual(sent, ["hello"])
        notif_events = list(events.replay(type=EVENT_NOTIFY_SENT))
        self.assertEqual(len(notif_events), 1)

    def test_empty_message_does_not_send(self):
        tmp_dir, events, sent, send = _setup()
        self.addCleanup(_cleanup, tmp_dir)
        notifier = Notifier(send, state_path=tmp_dir / "notify_state.json")
        delivered = asyncio.run(notifier.maybe_notify(""))
        self.assertFalse(delivered)
        self.assertEqual(sent, [])
        delivered = asyncio.run(notifier.maybe_notify("   "))
        self.assertFalse(delivered)

    def test_state_persists(self):
        tmp_dir, events, sent, send = _setup()
        self.addCleanup(_cleanup, tmp_dir)
        state_path = tmp_dir / "notify_state.json"
        notifier = Notifier(send, state_path=state_path, events=events)
        asyncio.run(notifier.maybe_notify("a", now=100.0))
        self.assertTrue(state_path.exists())
        data = json.loads(state_path.read_text())
        self.assertEqual(data["last_sent_ts"], 100.0)


class RateLimitTest(unittest.TestCase):
    def test_second_send_within_window_is_skipped(self):
        tmp_dir, events, sent, send = _setup()
        self.addCleanup(_cleanup, tmp_dir)
        notifier = Notifier(
            send,
            state_path=tmp_dir / "notify_state.json",
            events=events,
            min_interval_seconds=3600,
        )
        a = asyncio.run(notifier.maybe_notify("first", now=1000.0))
        b = asyncio.run(notifier.maybe_notify("second", now=1500.0))
        self.assertTrue(a)
        self.assertFalse(b)
        self.assertEqual(sent, ["first"])
        skipped = list(events.replay(type=EVENT_NOTIFY_SKIPPED))
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].get("reason"), "rate_limit")
        # The wait should be > 0 and <= min_interval.
        wait = skipped[0].get("seconds_until_allowed")
        self.assertIsNotNone(wait)
        self.assertGreater(wait, 0)
        self.assertLessEqual(wait, 3600)

    def test_send_after_window_succeeds(self):
        tmp_dir, events, sent, send = _setup()
        self.addCleanup(_cleanup, tmp_dir)
        notifier = Notifier(
            send,
            state_path=tmp_dir / "notify_state.json",
            events=events,
            min_interval_seconds=3600,
        )
        asyncio.run(notifier.maybe_notify("first", now=1000.0))
        asyncio.run(notifier.maybe_notify("second", now=1000.0 + 3601))
        self.assertEqual(sent, ["first", "second"])

    def test_rate_limit_survives_restart(self):
        tmp_dir, events, sent, send = _setup()
        self.addCleanup(_cleanup, tmp_dir)
        state_path = tmp_dir / "notify_state.json"

        # First instance sends once.
        n1 = Notifier(send, state_path=state_path, events=events, min_interval_seconds=3600)
        asyncio.run(n1.maybe_notify("first", now=1000.0))

        # Second instance (simulates a process restart) sees the same state.
        sent2: list[str] = []

        async def send2(text):
            sent2.append(text)

        n2 = Notifier(send2, state_path=state_path, events=events, min_interval_seconds=3600)
        delivered = asyncio.run(n2.maybe_notify("second", now=1500.0))
        self.assertFalse(delivered)
        self.assertEqual(sent2, [])


class SendFailureTest(unittest.TestCase):
    def test_send_exception_returns_false_and_does_not_advance_state(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="soma_notify_fail_"))
        self.addCleanup(_cleanup, tmp_dir)
        events = EventLog(tmp_dir / "events.jsonl")
        state_path = tmp_dir / "notify_state.json"

        async def boom(_):
            raise RuntimeError("network down")

        notifier = Notifier(boom, state_path=state_path, events=events, min_interval_seconds=3600)
        delivered = asyncio.run(notifier.maybe_notify("hi", now=1000.0))
        self.assertFalse(delivered)

        failed = list(events.replay(type=EVENT_NOTIFY_FAILED))
        self.assertEqual(len(failed), 1)
        # State must NOT have advanced — rate window should not start
        # from a failed send.
        self.assertFalse(state_path.exists())


class CorruptStateTest(unittest.TestCase):
    def test_corrupt_state_treated_as_no_prior_send(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="soma_notify_corrupt_"))
        self.addCleanup(_cleanup, tmp_dir)
        state_path = tmp_dir / "notify_state.json"
        state_path.write_text("not json")
        events = EventLog(tmp_dir / "events.jsonl")
        sent: list[str] = []

        async def send(t):
            sent.append(t)

        notifier = Notifier(send, state_path=state_path, events=events, min_interval_seconds=3600)
        delivered = asyncio.run(notifier.maybe_notify("hi", now=1000.0))
        self.assertTrue(delivered)
        self.assertEqual(sent, ["hi"])


if __name__ == "__main__":
    unittest.main()
