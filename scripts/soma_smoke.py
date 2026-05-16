#!/usr/bin/env python3
"""Soma in-process smoke test.

Runs without Telegram, without Ollama, without a real LLM. Exercises
the full SomaApp graph end-to-end: handle a user message, watch the
extractor produce memories, run a curator cycle, run crystallize,
trigger notify (twice — the second one must be rate-limited).

Pass if all steps complete and the event log matches the expected
shape. Fails loud and prints what was missing.

Use this BEFORE the real-Telegram smoke test on the Spark — it
catches wiring regressions that pass per-component tests but break
under live composition.

Run:
    python scripts/soma_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soma.context_builder import ContextBuilder
from soma.crystallize import Crystallizer
from soma.curator import Curator
from soma.events import (
    EVENT_BACKGROUND_TICK,
    EVENT_CRYSTALLIZE,
    EVENT_CURATOR_ACTION,
    EVENT_MEMORY_WRITTEN,
    EVENT_NOTIFY_SENT,
    EVENT_NOTIFY_SKIPPED,
    EVENT_PROMPT_RECEIVED,
    EVENT_RESPONSE_SENT,
    EventLog,
)
from soma.main import SomaApp, SomaConfig
from soma.memory_extractor import MemoryExtractor
from soma.memory_store import MemoryStore
from soma.notify import Notifier


class _DictEmbedder:
    """Keyword-bag embedder.

    Real Ollama embeddings cluster paraphrases together by semantics.
    For an in-process smoke test we approximate that with a coarse
    bag-of-keywords: each text becomes a vector whose components are
    set on the slots assigned to its lowercase non-stopword tokens.
    Two prompts that share a salient keyword (e.g. "git status") end
    up with cosine near 1.0; unrelated prompts stay orthogonal.
    """

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "me", "my", "you",
        "your", "to", "of", "in", "on", "at", "for", "with", "and",
        "or", "but", "if", "show", "please", "now", "current", "this",
        "that", "what", "whats",
    }

    def __init__(self, dim: int = 128):
        self._slots: dict[str, int] = {}
        self._dim = dim

    def _tokenize(self, text: str) -> list[str]:
        import re
        tokens = re.findall(r"[a-z]+", text.lower())
        return [t for t in tokens if t not in self._STOPWORDS]

    def _slot(self, token: str) -> int:
        if token not in self._slots:
            self._slots[token] = len(self._slots) % self._dim
        return self._slots[token]

    def embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            # Fall back to a per-text slot so the vector is non-zero
            # and unique to this text.
            return self._fallback(text)
        vec = [0.0] * self._dim
        for tok in tokens:
            vec[self._slot(tok)] += 1.0
        return vec

    def _fallback(self, text: str) -> list[float]:
        if text not in self._slots:
            self._slots[text] = len(self._slots) % self._dim
        vec = [0.0] * self._dim
        vec[self._slots[text]] = 1.0
        return vec


class _FakeEngine:
    def __init__(self):
        self._turn = 0

    async def run_turn(self, user_message, *, conversation_history=None, system_message=None):
        self._turn += 1
        reply = f"[turn {self._turn}] received: {user_message}"
        history = (conversation_history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        return {
            "final_response": reply,
            "messages": history,
            "api_calls": 1,
            "completed": True,
        }


class _FakeTransport:
    def __init__(self):
        self.sent: List[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)


def _fake_llm_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if sys.stdout.isatty() else s


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if sys.stdout.isatty() else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


class SmokeRunner:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="soma_smoke_"))
        self.embedder = _DictEmbedder()
        self.store = MemoryStore(self.tmp / "memories.jsonl", self.embedder)
        self.events = EventLog(self.tmp / "events.jsonl")
        self.transport = _FakeTransport()

        # Extractor with a canned response: pull a fact out of every turn.
        extractor_llm = MagicMock(
            return_value=_fake_llm_response(
                '[{"type": "semantic", "content": "User mentioned: hiking", "tags": ["hobby"]}]'
            )
        )
        self.extractor = MemoryExtractor(self.store, call_llm=extractor_llm)

        self.context_builder = ContextBuilder(self.store)

        # Notifier with a deliberately tight window — the second call
        # within the window MUST be rate-limited.
        self.notifier = Notifier(
            self.transport.send,
            state_path=self.tmp / "notify_state.json",
            events=self.events,
            min_interval_seconds=60.0,
        )

        # Curator with a canned action sequence.
        self._curator_responses = iter([
            '{"action": "none"}',
            '{"action": "write_memory", "type": "semantic", "content": "User prefers short answers", "tags": ["preference"]}',
            '{"action": "notify_user", "message": "Smoke notification"}',
        ])
        curator_llm = MagicMock(
            side_effect=lambda **kw: _fake_llm_response(next(self._curator_responses))
        )
        self.curator = Curator(
            self.store, self.events,
            call_llm=curator_llm,
            notify=self.notifier.maybe_notify,
            min_cycle_seconds=0.0,
            idle_pause_seconds=0.0,
        )

        self.crystallizer = Crystallizer(
            self.store, self.events,
            embedder=self.embedder,
            min_cluster_size=3,
            cluster_threshold=0.85,
            interval_seconds=999.0,
        )

        self.config = SomaConfig(
            telegram_token="smoke",
            telegram_user_id=1,
            data_dir=self.tmp,
            enable_curator=False,
            enable_crystallize=False,
        )
        self.app = SomaApp(
            self.config,
            engine=_FakeEngine(),
            embedder=self.embedder,
            store=self.store,
            extractor=self.extractor,
            context_builder=self.context_builder,
            events=self.events,
            transport=self.transport,
            curator=self.curator,
            notifier=self.notifier,
            crystallizer=self.crystallizer,
        )

        self.failures: list[str] = []
        self.steps: list[str] = []

    async def run(self) -> bool:
        await self._step_user_turns()
        await self._step_curator_cycles()
        await self._step_crystallize()
        await self._step_notify_rate_limit()
        self._report()
        return not self.failures

    # -- Steps ---------------------------------------------------------------

    async def _step_user_turns(self) -> None:
        prompts = [
            "show me git status",
            "show me git status please",
            "git status now",
            "what's the current git status",
            "git status?",
        ]
        for p in prompts:
            reply = await self.app.handle_user_message(p)
            if not reply:
                self.failures.append(f"no reply for prompt: {p!r}")
        # Let extraction background tasks settle.
        if self.app._bg_tasks:
            await asyncio.wait_for(asyncio.gather(*self.app._bg_tasks), timeout=5.0)

        prompts_logged = list(self.events.replay(type=EVENT_PROMPT_RECEIVED))
        responses_logged = list(self.events.replay(type=EVENT_RESPONSE_SENT))
        memories_logged = list(self.events.replay(type=EVENT_MEMORY_WRITTEN))
        if len(prompts_logged) != len(prompts):
            self.failures.append(
                f"prompt_received events: got {len(prompts_logged)}, want {len(prompts)}"
            )
        if len(responses_logged) != len(prompts):
            self.failures.append(
                f"response_sent events: got {len(responses_logged)}, want {len(prompts)}"
            )
        # Memories: extractor returns the same fact each turn → fuse → 1 record,
        # but each extraction logs a memory_written event.
        if not memories_logged:
            self.failures.append("no memory_written events — extractor never wrote")
        if len(self.store.all()) == 0:
            self.failures.append("memory store is empty after 5 turns")
        self.steps.append(
            f"user turns: {len(prompts)} prompts → {len(responses_logged)} replies, "
            f"{len(memories_logged)} memory events, store has {len(self.store.all())} records"
        )

    async def _step_curator_cycles(self) -> None:
        # 3 cycles → none, write_memory, notify
        for _ in range(3):
            await self.curator.cycle_once()
        actions = [
            e for e in self.events.replay(type=EVENT_CURATOR_ACTION)
        ]
        kinds = {a.get("kind") for a in actions}
        if "write_memory" not in kinds:
            self.failures.append("curator did not log a write_memory action")
        if "notify_user" not in kinds:
            self.failures.append("curator did not log a notify_user action")
        self.steps.append(
            f"curator: 3 cycles → action kinds = {sorted(kinds)}"
        )

    async def _step_crystallize(self) -> None:
        written = await self.crystallizer.run_once()
        cryst_events = list(self.events.replay(type=EVENT_CRYSTALLIZE))
        if not written:
            self.failures.append("crystallize produced no procedural memories from 5 similar prompts")
        if not cryst_events:
            self.failures.append("crystallize did not emit a crystallize event")
        self.steps.append(
            f"crystallize: wrote {len(written)} procedural memory record(s)"
        )

    async def _step_notify_rate_limit(self) -> None:
        # We already sent one notification via the curator's notify action.
        # A second notify within 60s MUST be rate-limited.
        before_sent = len(list(self.events.replay(type=EVENT_NOTIFY_SENT)))
        before_skipped = len(list(self.events.replay(type=EVENT_NOTIFY_SKIPPED)))
        delivered = await self.notifier.maybe_notify("second notification")
        after_sent = len(list(self.events.replay(type=EVENT_NOTIFY_SENT)))
        after_skipped = len(list(self.events.replay(type=EVENT_NOTIFY_SKIPPED)))

        if delivered:
            self.failures.append("rate limit failed: second notification was delivered")
        if after_sent != before_sent:
            self.failures.append("rate limit failed: notify_sent count increased")
        if after_skipped != before_skipped + 1:
            self.failures.append("rate limit failed: notify_skipped did not log the rejection")
        self.steps.append(
            f"notify rate-limit: first delivered={before_sent} sent, "
            f"second rejected ({after_skipped - before_skipped} new skipped event)"
        )

    # -- Reporting -----------------------------------------------------------

    def _report(self) -> None:
        print()
        print(_bold("Soma smoke test"))
        print("-" * 60)
        for step in self.steps:
            print(f"  • {step}")
        print()
        # Event-log summary so a regression makes itself obvious.
        counts: dict[str, int] = {}
        for e in self.events.replay():
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        print(_bold("Event log:"))
        for k in sorted(counts):
            print(f"  {k:25s} {counts[k]}")
        print()
        if self.failures:
            print(_red(_bold(f"FAIL — {len(self.failures)} problem(s):")))
            for f in self.failures:
                print(_red(f"  - {f}"))
        else:
            print(_green(_bold("OK — all smoke checks passed")))
        print()
        print(f"Artifacts left at: {self.tmp}")


def main() -> int:
    runner = SmokeRunner()
    ok = asyncio.run(runner.run())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
