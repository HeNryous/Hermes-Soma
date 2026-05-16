"""Tests for soma.curator — actions, anti-stagnation, adaptive sleep."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.curator import Curator, CycleResult, _looks_like_inactivity
from soma.events import (
    EVENT_BACKGROUND_TICK,
    EVENT_CURATOR_ACTION,
    EventLog,
)
from soma.memory_store import MemoryStore


class _DictEmbedder:
    def __init__(self):
        self._slots: dict[str, int] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self._slots:
            self._slots[text] = len(self._slots)
        vec = [0.0] * 64
        vec[self._slots[text] % 64] = 1.0
        return vec


def _fake_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _setup():
    tmp_dir = Path(tempfile.mkdtemp(prefix="soma_curator_test_"))
    store = MemoryStore(tmp_dir / "memories.jsonl", _DictEmbedder())
    events = EventLog(tmp_dir / "events.jsonl")
    return store, events, tmp_dir


def _cleanup(tmp_dir: Path):
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class InactivityDetectorTest(unittest.TestCase):
    def test_catches_obvious_idleness(self):
        self.assertTrue(_looks_like_inactivity("Curator is idle"))
        self.assertTrue(_looks_like_inactivity("Nothing happened in the last cycle"))
        self.assertTrue(_looks_like_inactivity("Soma was waiting for input"))
        self.assertTrue(_looks_like_inactivity("No new user message"))
        self.assertTrue(_looks_like_inactivity("Still waiting for the user"))
        self.assertTrue(_looks_like_inactivity("No actions taken this turn"))

    def test_does_not_flag_normal_facts(self):
        self.assertFalse(_looks_like_inactivity("User works at Supermicro"))
        self.assertFalse(_looks_like_inactivity("Project deadline is Friday"))
        self.assertFalse(_looks_like_inactivity("User speaks German and English"))


class WriteMemoryActionTest(unittest.TestCase):
    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def _curator(self, llm_response: str, **kw):
        call_llm = MagicMock(return_value=_fake_response(llm_response))
        return Curator(self.store, self.events, call_llm=call_llm, **kw)

    def test_writes_valid_memory(self):
        raw = '{"action": "write_memory", "type": "semantic", "content": "User likes hiking", "tags": ["hobby"]}'
        curator = self._curator(raw)
        result = asyncio.run(curator.cycle_once())
        self.assertEqual(result.action, "write_memory")
        self.assertTrue(result.productive)
        self.assertEqual(len(self.store.all()), 1)
        self.assertEqual(self.store.all()[0].content, "User likes hiking")
        self.assertEqual(self.store.all()[0].source, "curator")

    def test_rejects_inactivity_memory(self):
        raw = '{"action": "write_memory", "type": "semantic", "content": "Curator is idle"}'
        curator = self._curator(raw)
        result = asyncio.run(curator.cycle_once())
        self.assertEqual(result.action, "write_memory")
        self.assertFalse(result.productive)
        self.assertEqual(result.detail.get("reason"), "inactivity_pattern")
        self.assertEqual(len(self.store.all()), 0)
        # Rejection is logged as a curator action.
        actions = list(self.events.replay(type=EVENT_CURATOR_ACTION))
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "write_memory_rejected")

    def test_rejects_invalid_type(self):
        raw = '{"action": "write_memory", "type": "magic", "content": "x"}'
        curator = self._curator(raw)
        result = asyncio.run(curator.cycle_once())
        self.assertFalse(result.productive)
        self.assertEqual(len(self.store.all()), 0)


class ConsolidateActionTest(unittest.TestCase):
    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def _curator(self, llm_response: str):
        return Curator(self.store, self.events, call_llm=MagicMock(return_value=_fake_response(llm_response)))

    def test_merges_two_records(self):
        a = self.store.write("Fact A about project", type="semantic", tags=["project"])
        b = self.store.write("Fact B about project", type="semantic", tags=["project"])
        raw = (
            '{"action": "consolidate", "ids": ["%s", "%s"], '
            '"new_content": "Combined fact A+B", "type": "semantic", "tags": ["project"]}'
            % (a.id, b.id)
        )
        result = asyncio.run(self._curator(raw).cycle_once())
        self.assertTrue(result.productive)
        contents = [r.content for r in self.store.all()]
        self.assertNotIn("Fact A about project", contents)
        self.assertNotIn("Fact B about project", contents)
        self.assertIn("Combined fact A+B", contents)

    def test_refuses_to_consolidate_immortals(self):
        a = self.store.write("User identity fact", type="semantic", tags=["identity"])
        b = self.store.write("Other fact", type="semantic", tags=[])
        raw = (
            '{"action": "consolidate", "ids": ["%s", "%s"], '
            '"new_content": "Merged", "type": "semantic"}'
            % (a.id, b.id)
        )
        result = asyncio.run(self._curator(raw).cycle_once())
        self.assertFalse(result.productive)
        self.assertEqual(result.detail.get("reason"), "immortal_in_set")
        # Both originals still present.
        contents = [r.content for r in self.store.all()]
        self.assertIn("User identity fact", contents)
        self.assertIn("Other fact", contents)

    def test_missing_ids_refused(self):
        raw = '{"action": "consolidate", "ids": ["nope1", "nope2"], "new_content": "x", "type": "semantic"}'
        result = asyncio.run(self._curator(raw).cycle_once())
        self.assertFalse(result.productive)
        self.assertEqual(result.detail.get("reason"), "ids_not_found")


class ResearchActionTest(unittest.TestCase):
    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def _curator(self, llm_response: str, **kw):
        return Curator(
            self.store,
            self.events,
            call_llm=MagicMock(return_value=_fake_response(llm_response)),
            **kw,
        )

    def test_research_callback_invoked(self):
        seen: list[str] = []

        async def research(query):
            seen.append(query)
            return "found something useful"

        raw = '{"action": "research", "query": "current GPU prices"}'
        result = asyncio.run(self._curator(raw, research=research).cycle_once())
        self.assertTrue(result.productive)
        self.assertEqual(seen, ["current GPU prices"])

    def test_failed_topic_stops_after_threshold(self):
        async def research(query):
            return ""  # empty result counts as failure

        raw = '{"action": "research", "query": "obscure topic"}'
        curator = self._curator(raw, research=research, max_failed_research_per_topic=3)
        for _ in range(3):
            asyncio.run(curator.cycle_once())
        # 4th cycle should be skipped.
        result = asyncio.run(curator.cycle_once())
        self.assertFalse(result.productive)
        self.assertEqual(result.detail.get("reason"), "topic_failed_too_many_times")

    def test_no_callback_is_safe(self):
        raw = '{"action": "research", "query": "x"}'
        result = asyncio.run(self._curator(raw).cycle_once())
        self.assertFalse(result.productive)
        self.assertEqual(result.detail.get("reason"), "no_callback")


class NotifyActionTest(unittest.TestCase):
    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def test_notify_callback_invoked_when_callback_returns_true(self):
        sent: list[str] = []

        async def notify(msg):
            sent.append(msg)
            return True  # The notifier (Phase 6) is responsible for the rate-limit.

        raw = '{"action": "notify_user", "message": "Your build is done"}'
        curator = Curator(
            self.store,
            self.events,
            call_llm=MagicMock(return_value=_fake_response(raw)),
            notify=notify,
        )
        result = asyncio.run(curator.cycle_once())
        self.assertTrue(result.productive)
        self.assertEqual(sent, ["Your build is done"])
        actions = [
            e for e in self.events.replay(type=EVENT_CURATOR_ACTION)
            if e.get("kind") == "notify_user"
        ]
        self.assertEqual(len(actions), 1)
        self.assertTrue(actions[0].get("sent"))

    def test_notify_unproductive_when_callback_returns_false(self):
        async def notify(msg):
            return False  # rate-limited / suppressed

        raw = '{"action": "notify_user", "message": "Your build is done"}'
        curator = Curator(
            self.store,
            self.events,
            call_llm=MagicMock(return_value=_fake_response(raw)),
            notify=notify,
        )
        result = asyncio.run(curator.cycle_once())
        self.assertFalse(result.productive)
        self.assertFalse(result.detail.get("sent"))


class NoneAndUnknownActionTest(unittest.TestCase):
    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def _run(self, raw: str) -> CycleResult:
        curator = Curator(
            self.store, self.events,
            call_llm=MagicMock(return_value=_fake_response(raw)),
        )
        return asyncio.run(curator.cycle_once())

    def test_none_action(self):
        result = self._run('{"action": "none"}')
        self.assertEqual(result.action, "none")
        self.assertFalse(result.productive)

    def test_unknown_action(self):
        result = self._run('{"action": "magic", "thing": "x"}')
        self.assertEqual(result.action, "unknown")
        self.assertFalse(result.productive)

    def test_malformed_json_becomes_none(self):
        result = self._run("not json at all")
        self.assertEqual(result.action, "none")

    def test_empty_response_becomes_none(self):
        result = self._run("")
        self.assertEqual(result.action, "none")

    def test_fenced_object(self):
        raw = "```json\n{\"action\": \"none\"}\n```"
        result = self._run(raw)
        self.assertEqual(result.action, "none")


class AdaptiveSleepTest(unittest.TestCase):
    """The pause-after-empty-cycles behavior is the core anti-stagnation
    contract. Use a fake sleep to inspect the durations without waiting."""

    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def _curator(self):
        # Always returns {"action": "none"} → every cycle is non-productive.
        call_llm = MagicMock(return_value=_fake_response('{"action": "none"}'))
        curator = Curator(
            self.store, self.events,
            call_llm=call_llm,
            max_idle_cycles_before_pause=3,
            idle_pause_seconds=30.0,
            min_cycle_seconds=1.0,
        )
        durations: list[float] = []

        async def fake_sleep(seconds):
            durations.append(seconds)

        curator._sleep = fake_sleep
        return curator, durations

    def test_pause_after_n_idle_cycles(self):
        curator, durations = self._curator()

        async def run_n_cycles(n):
            for _ in range(n):
                result = await curator.cycle_once()
                await curator._adaptive_sleep(result)

        asyncio.run(run_n_cycles(3))
        # Two short sleeps, then a long pause.
        self.assertEqual(durations, [1.0, 1.0, 30.0])

    def test_productive_cycle_resets_counter(self):
        curator, durations = self._curator()

        async def run():
            # 2 idle, then 1 productive (we inject), then 2 more idle
            for _ in range(2):
                result = await curator.cycle_once()
                await curator._adaptive_sleep(result)
            productive = CycleResult(action="write_memory", productive=True)
            await curator._adaptive_sleep(productive)
            for _ in range(2):
                result = await curator.cycle_once()
                await curator._adaptive_sleep(result)

        asyncio.run(run())
        # All five should be min_cycle (no pause hit because productive reset).
        self.assertEqual(durations, [1.0, 1.0, 1.0, 1.0, 1.0])


class TickEventTest(unittest.TestCase):
    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def test_every_cycle_emits_background_tick(self):
        raw = '{"action": "none"}'
        curator = Curator(
            self.store, self.events,
            call_llm=MagicMock(return_value=_fake_response(raw)),
        )

        async def run_three():
            for _ in range(3):
                result = await curator.cycle_once()
                await curator._record_tick(result)

        asyncio.run(run_three())
        ticks = list(self.events.replay(type=EVENT_BACKGROUND_TICK))
        self.assertEqual(len(ticks), 3)
        self.assertTrue(all(t["action"] == "none" for t in ticks))
        self.assertTrue(all(t["productive"] is False for t in ticks))


class RunLoopTest(unittest.TestCase):
    """Smoke test for the full run() loop — uses fake sleep + cancels quickly."""

    def setUp(self):
        self.store, self.events, self.tmp = _setup()
        self.addCleanup(_cleanup, self.tmp)

    def test_run_cancels_cleanly(self):
        raw = '{"action": "none"}'
        curator = Curator(
            self.store, self.events,
            call_llm=MagicMock(return_value=_fake_response(raw)),
            min_cycle_seconds=0.01,
            idle_pause_seconds=0.01,
            max_idle_cycles_before_pause=2,
        )

        async def scenario():
            task = asyncio.create_task(curator.run())
            await asyncio.sleep(0.10)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())
        ticks = list(self.events.replay(type=EVENT_BACKGROUND_TICK))
        self.assertGreater(len(ticks), 0, "at least one tick should have fired")


if __name__ == "__main__":
    unittest.main()
