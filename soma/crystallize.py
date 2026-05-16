"""Skill crystallization — recurring user-prompt patterns become procedural memories.

Reads the prompt_received events from the event log, embeds each
prompt, and greedily clusters by cosine similarity. Any cluster with
N+ members becomes a procedural memory ("User frequently asks
variations of: ..."). The store's fuse mechanism dedups across passes,
so running this repeatedly is safe.

This complements Hermes' own skill system rather than replacing it:
the memories we write live in Soma's pool and surface through the
context builder's PROCEDURES block; Hermes' skill directory stays
untouched.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from .embed import cosine_similarity
from .events import (
    EVENT_CRYSTALLIZE,
    EVENT_ERROR,
    EVENT_PROMPT_RECEIVED,
    EventLog,
)
from .memory_store import MemoryRecord, MemoryStore

logger = logging.getLogger(__name__)


class Crystallizer:
    def __init__(
        self,
        store: MemoryStore,
        events: EventLog,
        *,
        embedder,
        min_cluster_size: int = 5,
        cluster_threshold: float = 0.85,
        prompt_window: int = 200,
        interval_seconds: float = 30 * 60.0,
    ):
        self.store = store
        self.events = events
        self.embedder = embedder
        self.min_cluster_size = min_cluster_size
        self.cluster_threshold = cluster_threshold
        self.prompt_window = prompt_window
        self.interval_seconds = interval_seconds
        # Sleep indirection so tests can substitute a no-op.
        self._sleep = asyncio.sleep

    # -- Public API ----------------------------------------------------------

    async def run(self) -> None:
        """Periodic loop. Runs until cancelled."""
        logger.info("soma: crystallizer started (interval=%ss)", self.interval_seconds)
        try:
            while True:
                await self._sleep(self.interval_seconds)
                try:
                    await self.run_once()
                except Exception as exc:
                    logger.exception("soma: crystallizer cycle failed")
                    await self.events.record_async(
                        EVENT_ERROR, where="crystallize", message=str(exc)
                    )
        except asyncio.CancelledError:
            logger.info("soma: crystallizer stopped")
            raise

    async def run_once(self) -> List[MemoryRecord]:
        prompts = self._recent_prompts()
        if len(prompts) < self.min_cluster_size:
            return []

        clusters = await asyncio.to_thread(self._cluster, prompts)
        candidates = [c for c in clusters if len(c) >= self.min_cluster_size]
        if not candidates:
            return []

        written: List[MemoryRecord] = []
        for cluster in candidates:
            representative = _representative(cluster)
            content = f'User frequently asks variations of: "{representative}"'
            try:
                record = await asyncio.to_thread(
                    self.store.write,
                    content,
                    type="procedural",
                    tags=["pattern"],
                    source="crystallize",
                )
                written.append(record)
            except Exception:
                logger.exception("soma: crystallize write failed for %r", representative[:60])

        if written:
            await self.events.record_async(
                EVENT_CRYSTALLIZE,
                count=len(written),
                ids=[r.id for r in written],
                clusters=[len(c) for c in candidates],
            )
        return written

    # -- Internals -----------------------------------------------------------

    def _recent_prompts(self) -> List[str]:
        out: List[str] = []
        for event in self.events.replay(type=EVENT_PROMPT_RECEIVED):
            text = (event.get("text") or "").strip()
            if text:
                out.append(text)
        return out[-self.prompt_window :]

    def _cluster(self, prompts: List[str]) -> List[List[str]]:
        """Greedy single-linkage clustering by cosine.

        For each prompt, embed it; if it matches an existing cluster's
        centroid above the threshold, add it; otherwise spawn a new
        cluster. Centroid is the first prompt's embedding — simple and
        cheap, and good enough at this scale (~200 prompts).
        """
        clusters: List[List[str]] = []
        centroids: List[List[float]] = []
        for prompt in prompts:
            try:
                vec = self.embedder.embed(prompt)
            except Exception:
                logger.exception("soma: crystallize embed failed for %r", prompt[:60])
                continue
            assigned = False
            for i, centroid in enumerate(centroids):
                if cosine_similarity(vec, centroid) >= self.cluster_threshold:
                    clusters[i].append(prompt)
                    assigned = True
                    break
            if not assigned:
                clusters.append([prompt])
                centroids.append(vec)
        return clusters


def _representative(cluster: List[str]) -> str:
    """Pick the shortest non-trivial prompt as the cluster's representative —
    shorter prompts tend to be more general."""
    candidates = [p for p in cluster if len(p) >= 3] or cluster
    return min(candidates, key=len)
