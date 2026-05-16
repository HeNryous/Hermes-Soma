"""Tests for soma.crystallize — clustering, threshold, dedup via fuse."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.crystallize import Crystallizer, _representative
from soma.events import EVENT_CRYSTALLIZE, EVENT_PROMPT_RECEIVED, EventLog
from soma.memory_store import MemoryStore


class _KeyedEmbedder:
    """Deterministic test embedder.

    Returns a fixed-length unit vector per text. Pre-mapped texts share a
    slot (so cosine = 1.0); unknown texts get cached on first sight so
    repeated calls for the same text always return the same vector.
    """

    def __init__(self, mapping: dict[str, int], dim: int = 32):
        self._mapping = dict(mapping)
        self._dim = dim
        self._next_slot = max(mapping.values(), default=-1) + 1

    def embed(self, text: str) -> list[float]:
        if text not in self._mapping:
            self._mapping[text] = self._next_slot % self._dim
            self._next_slot += 1
        vec = [0.0] * self._dim
        vec[self._mapping[text]] = 1.0
        return vec


def _setup(mapping: dict[str, int]):
    tmp_dir = Path(tempfile.mkdtemp(prefix="soma_crystallize_test_"))
    store = MemoryStore(tmp_dir / "memories.jsonl", _KeyedEmbedder(mapping))
    events = EventLog(tmp_dir / "events.jsonl")
    return tmp_dir, store, events


def _cleanup(tmp_dir: Path):
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class RepresentativeTest(unittest.TestCase):
    def test_picks_shortest(self):
        self.assertEqual(
            _representative(["please show me the git status", "git status", "show git status"]),
            "git status",
        )

    def test_falls_back_for_all_trivial(self):
        # All under 3 chars → use the cluster as-is.
        self.assertEqual(_representative(["a", "b"]), "a")


class CrystallizerThresholdTest(unittest.TestCase):
    def test_below_min_cluster_size_writes_nothing(self):
        # 4 prompts that all cluster together — below threshold of 5.
        mapping = {f"p{i}": 0 for i in range(4)}
        tmp_dir, store, events = _setup(mapping)
        self.addCleanup(_cleanup, tmp_dir)
        for i in range(4):
            events.record(EVENT_PROMPT_RECEIVED, text=f"p{i}")
        cryst = Crystallizer(store, events, embedder=store.embedder, min_cluster_size=5)
        written = asyncio.run(cryst.run_once())
        self.assertEqual(written, [])
        self.assertEqual(len(store.all()), 0)

    def test_meets_min_cluster_writes_one(self):
        mapping = {f"p{i}": 0 for i in range(5)}
        tmp_dir, store, events = _setup(mapping)
        self.addCleanup(_cleanup, tmp_dir)
        for i in range(5):
            events.record(EVENT_PROMPT_RECEIVED, text=f"p{i}")
        cryst = Crystallizer(store, events, embedder=store.embedder, min_cluster_size=5)
        written = asyncio.run(cryst.run_once())
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].type, "procedural")
        self.assertIn("pattern", written[0].tags)

    def test_two_distinct_clusters_each_above_threshold(self):
        mapping = {
            **{f"a{i}": 0 for i in range(5)},
            **{f"b{i}": 1 for i in range(5)},
        }
        tmp_dir, store, events = _setup(mapping)
        self.addCleanup(_cleanup, tmp_dir)
        for k in mapping:
            events.record(EVENT_PROMPT_RECEIVED, text=k)
        cryst = Crystallizer(store, events, embedder=store.embedder, min_cluster_size=5)
        written = asyncio.run(cryst.run_once())
        self.assertEqual(len(written), 2)

    def test_singleton_does_not_dilute_real_cluster(self):
        mapping = {
            **{f"p{i}": 0 for i in range(5)},
            "loner": 99,  # singleton, distinct slot
        }
        tmp_dir, store, events = _setup(mapping)
        self.addCleanup(_cleanup, tmp_dir)
        for k in mapping:
            events.record(EVENT_PROMPT_RECEIVED, text=k)
        cryst = Crystallizer(store, events, embedder=store.embedder, min_cluster_size=5)
        written = asyncio.run(cryst.run_once())
        # Loner forms its own cluster of 1; only the 5-cluster qualifies.
        self.assertEqual(len(written), 1)


class CrystallizeDedupTest(unittest.TestCase):
    def test_rerunning_does_not_create_duplicates(self):
        mapping = {f"p{i}": 0 for i in range(5)}
        tmp_dir, store, events = _setup(mapping)
        self.addCleanup(_cleanup, tmp_dir)
        for i in range(5):
            events.record(EVENT_PROMPT_RECEIVED, text=f"p{i}")
        cryst = Crystallizer(store, events, embedder=store.embedder, min_cluster_size=5)
        first = asyncio.run(cryst.run_once())
        second = asyncio.run(cryst.run_once())
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        # Same record id (fused), not two records.
        self.assertEqual(first[0].id, second[0].id)
        self.assertEqual(len(store.all()), 1)
        # use_count incremented on re-run.
        self.assertEqual(store.all()[0].use_count, 2)


class CrystallizeEventLogTest(unittest.TestCase):
    def test_writes_crystallize_event_when_productive(self):
        mapping = {f"p{i}": 0 for i in range(5)}
        tmp_dir, store, events = _setup(mapping)
        self.addCleanup(_cleanup, tmp_dir)
        for i in range(5):
            events.record(EVENT_PROMPT_RECEIVED, text=f"p{i}")
        cryst = Crystallizer(store, events, embedder=store.embedder, min_cluster_size=5)
        asyncio.run(cryst.run_once())
        cryst_events = list(events.replay(type=EVENT_CRYSTALLIZE))
        self.assertEqual(len(cryst_events), 1)
        self.assertEqual(cryst_events[0]["count"], 1)
        self.assertEqual(cryst_events[0]["clusters"], [5])


class RunLoopCancellationTest(unittest.TestCase):
    def test_cancellation_is_clean(self):
        tmp_dir, store, events = _setup({})
        self.addCleanup(_cleanup, tmp_dir)
        cryst = Crystallizer(
            store, events, embedder=store.embedder,
            interval_seconds=0.05, min_cluster_size=5,
        )

        async def scenario():
            task = asyncio.create_task(cryst.run())
            await asyncio.sleep(0.12)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
