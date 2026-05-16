"""Background curator — Soma's always-on cognition layer.

Runs as an asyncio task next to the transport. Each cycle:

  1. Snapshot the recent memory pool and event log.
  2. Ask the LLM: "what one thing would make the next user turn better?"
  3. Parse the JSON action and dispatch (write / consolidate / research / notify / none).
  4. Record a background_tick event.
  5. Adaptive sleep — if many cycles in a row produce nothing, take a longer break
     so we don't burn API calls on an idle user.

Hard rules baked in (anti-stagnation):

  - The curator NEVER writes a memory about its own activity, idleness,
    "still waiting", or anything else describing the curator itself.
    Inactivity memories are the failure mode we explicitly prevent.
  - Consolidation refuses to touch memories with immortal tags. Those
    are the user's identity / domain / preferences — they survive forever.
  - Failed research on a topic is tracked. After max_failed_research_per_topic
    failures, the curator refuses to retry that topic.

Notifications go through an injectable callback (Phase 6 wires the real
Telegram send with the 1/24h rate limit). The curator just enqueues —
delivery and rate-limiting are not its concern.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .events import (
    EVENT_BACKGROUND_TICK,
    EVENT_CURATOR_ACTION,
    EVENT_ERROR,
    EVENT_NOTIFY_SENT,
    EventLog,
)
from .memory_extractor import _coerce_json_array  # reuse the forgiving JSON parser
from .memory_store import IMMORTAL_TAGS, MEMORY_TYPES, MemoryStore

logger = logging.getLogger(__name__)


# Phrases that betray an "I was idle" memory. The plan's hardest rule:
# never write a memory about doing nothing. Match defensively — substring,
# case-insensitive.
_INACTIVITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bidle\b",
        r"\bnothing (happened|new|to do)\b",
        # "no message", "no new message", "no user message",
        # "no new user message" — any combination of new/user qualifiers.
        r"\bno (?:new |user )*(?:message|input|activity)\b",
        r"\bstill waiting\b",
        r"\bcurator (is|was) ",
        r"\bsoma (is|was) (waiting|idle|thinking)\b",
        r"\bno actions? (taken|needed)\b",
    )
]


@dataclass
class CycleResult:
    action: str
    productive: bool
    detail: Dict[str, Any] = field(default_factory=dict)


# Optional callbacks the curator delegates to.
ResearchCallback = Callable[[str], Awaitable[Optional[str]]]
NotifyCallback = Callable[[str], Awaitable[None]]


CURATOR_SYSTEM = """You are Soma's background curator. Soma runs you on a tight
loop while the user is quiet — your job is to make the NEXT user turn better.

You receive:
- MEMORY_POOL: a sample of what Soma already knows about this user.
- RECENT_EVENTS: the last few prompts, responses, and memory writes.

You return ONE action as JSON. Schemas:

  {"action":"write_memory","type":"semantic|procedural|episodic","content":"...","tags":["..."]}
    Use when there's a durable fact about the user that's missing from the pool.

  {"action":"consolidate","ids":["mem_a","mem_b"],"new_content":"...","type":"semantic|procedural|episodic","tags":["..."]}
    Use to merge 2+ overlapping memories into one cleaner record.
    NEVER consolidate memories tagged domain/role/identity/preference/behavior.

  {"action":"research","query":"..."}
    Use when a recent user message left a question you can't answer from the pool.

  {"action":"notify_user","message":"..."}
    Use ONLY when you have time-sensitive information the user explicitly cares about.

  {"action":"none"}
    Use when nothing actionable is needed. This is the right answer most of the time.

HARD RULES:
- NEVER create a memory describing your own activity, idleness, or thinking.
- NEVER notify just to say hello, check in, or fill silence.
- Prefer "none" over noise. Empty cycles are healthy.

Return the JSON object and nothing else."""


class Curator:
    def __init__(
        self,
        store: MemoryStore,
        events: EventLog,
        *,
        call_llm: Optional[Callable] = None,
        model: Optional[str] = None,
        research: Optional[ResearchCallback] = None,
        notify: Optional[NotifyCallback] = None,
        max_idle_cycles_before_pause: int = 5,
        idle_pause_seconds: float = 30.0,
        min_cycle_seconds: float = 5.0,
        max_failed_research_per_topic: int = 3,
        recent_events_window: int = 50,
        memory_sample_size: int = 30,
    ):
        self.store = store
        self.events = events
        self.model = model
        self.research_cb = research
        self.notify_cb = notify
        self.max_idle_cycles_before_pause = max_idle_cycles_before_pause
        self.idle_pause_seconds = idle_pause_seconds
        self.min_cycle_seconds = min_cycle_seconds
        self.max_failed_research_per_topic = max_failed_research_per_topic
        self.recent_events_window = recent_events_window
        self.memory_sample_size = memory_sample_size

        if call_llm is None:
            from agent.auxiliary_client import call_llm as default_call_llm
            call_llm = default_call_llm
        self._call_llm = call_llm

        self._failed_research: Dict[str, int] = {}
        self._empty_in_a_row = 0
        # Sleep helper indirection so tests can substitute a no-op.
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    # -- Public API ----------------------------------------------------------

    async def run(self) -> None:
        """Main loop. Runs until cancelled."""
        logger.info("soma: curator started")
        try:
            while True:
                result = await self._safe_cycle()
                await self._record_tick(result)
                await self._adaptive_sleep(result)
        except asyncio.CancelledError:
            logger.info("soma: curator stopped")
            raise

    async def cycle_once(self) -> CycleResult:
        """Run a single cycle. Exposed for tests and manual ticks."""
        return await self._safe_cycle()

    # -- Cycle internals -----------------------------------------------------

    async def _safe_cycle(self) -> CycleResult:
        try:
            return await self._cycle()
        except Exception as exc:
            logger.exception("soma: curator cycle failed")
            await self.events.record_async(EVENT_ERROR, where="curator", message=str(exc))
            return CycleResult(action="error", productive=False, detail={"message": str(exc)})

    async def _cycle(self) -> CycleResult:
        prompt = self._build_prompt()
        raw = await asyncio.to_thread(self._invoke_llm, prompt)
        action = self._parse_action(raw)
        return await self._dispatch(action)

    def _build_prompt(self) -> str:
        memories = self.store.all()
        # Take the most recent N memories — last_seen_at as a proxy for relevance.
        memories.sort(key=lambda r: r.last_seen_at, reverse=True)
        mem_sample = memories[: self.memory_sample_size]
        mem_lines = [
            f"- [{r.id}] ({r.type}, tags={r.tags or '[]'}): {r.content}"
            for r in mem_sample
        ]

        events = list(self.events.replay())
        recent = events[-self.recent_events_window :]
        evt_lines = [
            f"- {e.get('type')}: {json.dumps({k: v for k, v in e.items() if k not in ('ts', 'type')}, ensure_ascii=False)[:200]}"
            for e in recent
        ]

        return (
            "MEMORY_POOL:\n"
            + ("\n".join(mem_lines) if mem_lines else "(empty)")
            + "\n\nRECENT_EVENTS:\n"
            + ("\n".join(evt_lines) if evt_lines else "(empty)")
            + "\n\nReturn the JSON action now."
        )

    def _invoke_llm(self, prompt: str) -> str:
        kwargs: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": CURATOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }
        if self.model:
            kwargs["model"] = self.model
        response = self._call_llm(**kwargs)
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    def _parse_action(self, raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        if not raw:
            return {"action": "none"}
        # Try as bare object first; fall back to array-coercion if the model
        # wrapped it.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Some models wrap a single action in [ ... ].
            coerced = _coerce_json_array(raw)
            if isinstance(coerced, list) and coerced and isinstance(coerced[0], dict):
                data = coerced[0]
            else:
                # Last resort: find the first {...} in the text.
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not match:
                    return {"action": "none"}
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {"action": "none"}
        if not isinstance(data, dict):
            return {"action": "none"}
        return data

    # -- Action dispatch -----------------------------------------------------

    async def _dispatch(self, action: Dict[str, Any]) -> CycleResult:
        kind = str(action.get("action") or "none").strip().lower()
        if kind == "none":
            return CycleResult(action="none", productive=False)
        if kind == "write_memory":
            return await self._do_write_memory(action)
        if kind == "consolidate":
            return await self._do_consolidate(action)
        if kind == "research":
            return await self._do_research(action)
        if kind == "notify_user":
            return await self._do_notify(action)
        return CycleResult(action="unknown", productive=False, detail={"raw": action})

    async def _do_write_memory(self, action: Dict[str, Any]) -> CycleResult:
        type_ = str(action.get("type") or "").strip().lower()
        content = str(action.get("content") or "").strip()
        raw_tags = action.get("tags") or []
        tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]

        if type_ not in MEMORY_TYPES or not content:
            return CycleResult(
                action="write_memory",
                productive=False,
                detail={"reason": "invalid", "type": type_, "content": content[:60]},
            )
        if _looks_like_inactivity(content):
            await self.events.record_async(
                EVENT_CURATOR_ACTION,
                kind="write_memory_rejected",
                reason="inactivity_pattern",
                content=content[:200],
            )
            return CycleResult(
                action="write_memory",
                productive=False,
                detail={"reason": "inactivity_pattern", "content": content[:200]},
            )
        try:
            record = await asyncio.to_thread(
                self.store.write,
                content,
                type=type_,
                tags=tags,
                source="curator",
            )
        except Exception as exc:
            logger.warning("soma: curator memory write failed: %s", exc)
            return CycleResult(
                action="write_memory",
                productive=False,
                detail={"reason": "write_failed", "message": str(exc)},
            )
        await self.events.record_async(
            EVENT_CURATOR_ACTION,
            kind="write_memory",
            id=record.id,
            memory_type=record.type,
            tags=record.tags,
        )
        return CycleResult(
            action="write_memory",
            productive=True,
            detail={"id": record.id},
        )

    async def _do_consolidate(self, action: Dict[str, Any]) -> CycleResult:
        ids = [str(i) for i in (action.get("ids") or []) if str(i).strip()]
        new_content = str(action.get("new_content") or "").strip()
        type_ = str(action.get("type") or "semantic").strip().lower()
        raw_tags = action.get("tags") or []
        tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]

        if len(ids) < 2 or not new_content or type_ not in MEMORY_TYPES:
            return CycleResult(
                action="consolidate",
                productive=False,
                detail={"reason": "invalid"},
            )
        records = [self.store.get(i) for i in ids]
        records = [r for r in records if r is not None]
        if len(records) < 2:
            return CycleResult(
                action="consolidate",
                productive=False,
                detail={"reason": "ids_not_found", "ids": ids},
            )
        if any(r.is_immortal() for r in records):
            await self.events.record_async(
                EVENT_CURATOR_ACTION,
                kind="consolidate_rejected",
                reason="immortal_in_set",
                ids=ids,
            )
            return CycleResult(
                action="consolidate",
                productive=False,
                detail={"reason": "immortal_in_set", "ids": ids},
            )
        # Preserve the immortal-tag union of the inputs (the new memory inherits any
        # behavior/preference tags, even if the LLM forgot to include them).
        union_tags = set(tags)
        for r in records:
            union_tags.update(t for t in r.tags if t in IMMORTAL_TAGS)
        merged_tags = sorted(union_tags) if union_tags else list(tags)

        try:
            await asyncio.to_thread(self.store.delete_many, ids)
            record = await asyncio.to_thread(
                self.store.write,
                new_content,
                type=type_,
                tags=merged_tags,
                source="curator",
            )
        except Exception as exc:
            logger.warning("soma: curator consolidate failed: %s", exc)
            return CycleResult(
                action="consolidate",
                productive=False,
                detail={"reason": "write_failed", "message": str(exc)},
            )
        await self.events.record_async(
            EVENT_CURATOR_ACTION,
            kind="consolidate",
            merged_into=record.id,
            removed_ids=ids,
        )
        return CycleResult(
            action="consolidate",
            productive=True,
            detail={"id": record.id, "removed": ids},
        )

    async def _do_research(self, action: Dict[str, Any]) -> CycleResult:
        query = str(action.get("query") or "").strip()
        if not query:
            return CycleResult(action="research", productive=False, detail={"reason": "empty"})
        if self._failed_research.get(query, 0) >= self.max_failed_research_per_topic:
            await self.events.record_async(
                EVENT_CURATOR_ACTION,
                kind="research_skipped",
                reason="topic_failed_too_many_times",
                query=query,
            )
            return CycleResult(
                action="research",
                productive=False,
                detail={"reason": "topic_failed_too_many_times", "query": query},
            )
        if self.research_cb is None:
            await self.events.record_async(
                EVENT_CURATOR_ACTION,
                kind="research_skipped",
                reason="no_research_callback",
                query=query,
            )
            return CycleResult(
                action="research",
                productive=False,
                detail={"reason": "no_callback", "query": query},
            )
        try:
            result = await self.research_cb(query)
        except Exception as exc:
            self._failed_research[query] = self._failed_research.get(query, 0) + 1
            logger.warning("soma: research callback raised for %r: %s", query, exc)
            return CycleResult(
                action="research",
                productive=False,
                detail={"reason": "callback_error", "query": query, "error": str(exc)},
            )
        if not result:
            self._failed_research[query] = self._failed_research.get(query, 0) + 1
            return CycleResult(
                action="research",
                productive=False,
                detail={"reason": "no_result", "query": query},
            )
        # Successful research is productive even before the result is digested.
        await self.events.record_async(
            EVENT_CURATOR_ACTION,
            kind="research",
            query=query,
            result_preview=result[:200],
        )
        return CycleResult(
            action="research",
            productive=True,
            detail={"query": query, "result": result},
        )

    async def _do_notify(self, action: Dict[str, Any]) -> CycleResult:
        message = str(action.get("message") or "").strip()
        if not message:
            return CycleResult(action="notify_user", productive=False, detail={"reason": "empty"})
        if self.notify_cb is None:
            await self.events.record_async(
                EVENT_CURATOR_ACTION,
                kind="notify_skipped",
                reason="no_notify_callback",
                message=message[:200],
            )
            return CycleResult(
                action="notify_user",
                productive=False,
                detail={"reason": "no_callback", "message": message},
            )
        try:
            await self.notify_cb(message)
        except Exception as exc:
            logger.warning("soma: notify callback raised: %s", exc)
            return CycleResult(
                action="notify_user",
                productive=False,
                detail={"reason": "callback_error", "error": str(exc)},
            )
        await self.events.record_async(EVENT_NOTIFY_SENT, message=message)
        return CycleResult(
            action="notify_user",
            productive=True,
            detail={"message": message},
        )

    # -- Loop pacing ---------------------------------------------------------

    async def _record_tick(self, result: CycleResult) -> None:
        await self.events.record_async(
            EVENT_BACKGROUND_TICK,
            action=result.action,
            productive=result.productive,
            **{k: v for k, v in result.detail.items() if k in ("reason", "id", "query")},
        )

    async def _adaptive_sleep(self, result: CycleResult) -> None:
        if result.productive:
            self._empty_in_a_row = 0
            await self._sleep(self.min_cycle_seconds)
            return
        self._empty_in_a_row += 1
        if self._empty_in_a_row >= self.max_idle_cycles_before_pause:
            await self._sleep(self.idle_pause_seconds)
            self._empty_in_a_row = 0
        else:
            await self._sleep(self.min_cycle_seconds)


def _looks_like_inactivity(content: str) -> bool:
    return any(p.search(content) for p in _INACTIVITY_PATTERNS)
