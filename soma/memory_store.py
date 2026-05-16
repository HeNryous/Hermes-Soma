"""JSONL-backed memory pool with embedding-fuse, scoring, and prune.

Design contract (from the build plan):

- Three types: semantic, procedural, episodic.
- Immortal tags (domain, role, identity, preference, behavior) survive prune.
- Score = recency × frequency × type_weight. Higher = keep.
- Cosine > dup_threshold against any existing memory → fuse instead of insert:
  bump use_count and last_seen_at on the existing record.
- Auto-prune when len > max_memories → keep top prune_target by score
  (immortals always kept regardless of count).
- Conflict resolution is shallow: a newer write with the same content_hash
  wins. Semantic conflicts (e.g. "user prefers short" vs "user prefers long"
  with similar embedding) fuse via cosine — the newer content overwrites
  the older record while preserving use_count.

The store is thread-safe via a single lock. Async callers should invoke
write/query through asyncio.to_thread.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .embed import cosine_similarity


MEMORY_TYPES = ("semantic", "procedural", "episodic")

# Tags that keep a memory immortal — never pruned regardless of score.
IMMORTAL_TAGS = frozenset({"domain", "role", "identity", "preference", "behavior"})

# Multiplicative weight applied to each memory's score by type.
TYPE_WEIGHTS = {
    "semantic": 1.0,
    "procedural": 1.2,
    "episodic": 0.7,
}

# Recency half-life — recall score halves every RECENCY_HALF_LIFE_SECS.
RECENCY_HALF_LIFE_SECS = 7 * 24 * 3600.0


@dataclass
class MemoryRecord:
    id: str
    type: str
    content: str
    tags: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    created_at: float = 0.0
    last_seen_at: float = 0.0
    use_count: int = 1
    source: str = "extractor"
    session_id: Optional[str] = None

    def is_immortal(self) -> bool:
        return any(t in IMMORTAL_TAGS for t in self.tags)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(
            id=d["id"],
            type=d["type"],
            content=d["content"],
            tags=list(d.get("tags") or []),
            embedding=list(d.get("embedding") or []),
            created_at=float(d.get("created_at") or 0.0),
            last_seen_at=float(d.get("last_seen_at") or 0.0),
            use_count=int(d.get("use_count") or 1),
            source=d.get("source") or "extractor",
            session_id=d.get("session_id"),
        )


def score_record(record: MemoryRecord, *, now: float) -> float:
    """Score = recency × frequency × type_weight. Immortals return +inf."""
    if record.is_immortal():
        return math.inf
    age = max(0.0, now - record.last_seen_at)
    recency = math.exp(-age * math.log(2) / RECENCY_HALF_LIFE_SECS)
    frequency = math.log1p(max(0, record.use_count))
    type_weight = TYPE_WEIGHTS.get(record.type, 1.0)
    return recency * frequency * type_weight


class MemoryStore:
    def __init__(
        self,
        path: os.PathLike | str,
        embedder,
        *,
        max_memories: int = 150,
        prune_target: int = 100,
        dup_threshold: float = 0.80,
    ):
        if prune_target > max_memories:
            raise ValueError("prune_target must be <= max_memories")
        self.path = Path(path)
        self.embedder = embedder
        self.max_memories = max_memories
        self.prune_target = prune_target
        self.dup_threshold = dup_threshold
        self._lock = threading.Lock()
        self._records: List[MemoryRecord] = []
        self._loaded = False

    # -- I/O -----------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._records = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._records.append(MemoryRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._loaded = True

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for record in self._records:
                f.write(record.to_jsonl() + "\n")
        os.replace(tmp, self.path)

    # -- Public API ----------------------------------------------------------

    def all(self) -> List[MemoryRecord]:
        with self._lock:
            self._load()
            return list(self._records)

    def by_tag(self, tag: str) -> List[MemoryRecord]:
        with self._lock:
            self._load()
            return [r for r in self._records if tag in r.tags]

    def by_type(self, type_: str) -> List[MemoryRecord]:
        with self._lock:
            self._load()
            return [r for r in self._records if r.type == type_]

    def write(
        self,
        content: str,
        *,
        type: str = "semantic",
        tags: Optional[Sequence[str]] = None,
        source: str = "extractor",
        session_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> MemoryRecord:
        if type not in MEMORY_TYPES:
            raise ValueError(f"invalid memory type: {type!r}")
        content = (content or "").strip()
        if not content:
            raise ValueError("cannot write empty memory content")
        now = now if now is not None else time.time()
        embedding = self.embedder.embed(content)

        with self._lock:
            self._load()

            duplicate = self._find_duplicate(embedding)
            if duplicate is not None:
                duplicate.use_count += 1
                duplicate.last_seen_at = now
                # New content wins on near-duplicate (conflict resolution).
                duplicate.content = content
                if tags:
                    merged = list(dict.fromkeys([*duplicate.tags, *tags]))
                    duplicate.tags = merged
                self._flush()
                return duplicate

            record = MemoryRecord(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                type=type,
                content=content,
                tags=list(tags or []),
                embedding=list(embedding),
                created_at=now,
                last_seen_at=now,
                use_count=1,
                source=source,
                session_id=session_id,
            )
            self._records.append(record)
            self._maybe_prune(now=now)
            self._flush()
            return record

    def query(
        self,
        text: str,
        *,
        top_k: int = 8,
        min_similarity: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Return (record, similarity) sorted by similarity descending."""
        if not text.strip():
            return []
        query_embedding = self.embedder.embed(text)
        with self._lock:
            self._load()
            scored = [
                (r, cosine_similarity(query_embedding, r.embedding))
                for r in self._records
            ]
        scored = [pair for pair in scored if pair[1] >= min_similarity]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def prune(self, *, now: Optional[float] = None) -> int:
        """Force a prune pass; returns number removed."""
        now = now if now is not None else time.time()
        with self._lock:
            self._load()
            before = len(self._records)
            self._maybe_prune(now=now, force=True)
            removed = before - len(self._records)
            if removed:
                self._flush()
            return removed

    def get(self, id: str) -> Optional[MemoryRecord]:
        with self._lock:
            self._load()
            for record in self._records:
                if record.id == id:
                    return record
            return None

    def delete(self, id: str) -> bool:
        with self._lock:
            self._load()
            before = len(self._records)
            self._records = [r for r in self._records if r.id != id]
            removed = len(self._records) < before
            if removed:
                self._flush()
            return removed

    def delete_many(self, ids: Iterable[str]) -> int:
        id_set = set(ids)
        if not id_set:
            return 0
        with self._lock:
            self._load()
            before = len(self._records)
            self._records = [r for r in self._records if r.id not in id_set]
            removed = before - len(self._records)
            if removed:
                self._flush()
            return removed

    # -- Internals -----------------------------------------------------------

    def _find_duplicate(self, embedding: Sequence[float]) -> Optional[MemoryRecord]:
        best: Optional[MemoryRecord] = None
        best_sim = self.dup_threshold
        for record in self._records:
            sim = cosine_similarity(embedding, record.embedding)
            if sim >= best_sim:
                best_sim = sim
                best = record
        return best

    def _maybe_prune(self, *, now: float, force: bool = False) -> None:
        if not force and len(self._records) <= self.max_memories:
            return
        immortals = [r for r in self._records if r.is_immortal()]
        mortals = [r for r in self._records if not r.is_immortal()]
        mortal_budget = max(0, self.prune_target - len(immortals))
        mortals.sort(key=lambda r: score_record(r, now=now), reverse=True)
        kept_mortals = mortals[:mortal_budget]
        # Preserve original insertion order for stable on-disk layout.
        keep_ids = {r.id for r in immortals} | {r.id for r in kept_mortals}
        self._records = [r for r in self._records if r.id in keep_ids]
