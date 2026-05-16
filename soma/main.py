"""Soma application — wires Phases 1–3 together behind the Telegram transport.

Run with:
    python -m soma.main

Environment:
    TELEGRAM_TOKEN          Bot token from BotFather (required).
    TELEGRAM_USER_ID        Numeric Telegram user id allowed to talk to the bot (required).
    SOMA_DATA_DIR           Directory for memories.jsonl and events.jsonl
                            (default: ./data/soma).
    SOMA_MODEL              Optional Hermes model override (default: Hermes' configured model).
    SOMA_OLLAMA_URL         Ollama base URL (default: http://localhost:11434).
    SOMA_EMBED_MODEL        Ollama embedding model (default: nomic-embed-text).
    SOMA_DEBOUNCE_SECS      Telegram debounce window (default: 3.0).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .context_builder import ContextBuilder
from .crystallize import Crystallizer
from .curator import Curator
from .embed import OllamaEmbedder
from .engine import EngineConfig, SomaEngine
from .events import (
    EVENT_ERROR,
    EVENT_MEMORY_WRITTEN,
    EVENT_PROMPT_RECEIVED,
    EVENT_RESPONSE_SENT,
    EventLog,
)
from .memory_extractor import MemoryExtractor
from .memory_store import MemoryStore
from .notify import DEFAULT_MIN_INTERVAL_SECONDS, Notifier
from .sanitizer import sanitize
from .transport import TelegramTransport

logger = logging.getLogger(__name__)


@dataclass
class SomaConfig:
    telegram_token: str
    telegram_user_id: int
    data_dir: Path = Path("data/soma")
    model: Optional[str] = None
    debounce_seconds: float = 3.0
    strip_markdown: bool = False
    max_reply_chars: int = 4000
    enable_curator: bool = True
    enable_crystallize: bool = True
    notify_min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS
    crystallize_interval_seconds: float = 30 * 60.0

    @classmethod
    def from_env(cls) -> "SomaConfig":
        token = os.environ.get("TELEGRAM_TOKEN", "").strip()
        user_id = os.environ.get("TELEGRAM_USER_ID", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_TOKEN is required")
        if not user_id.isdigit():
            raise RuntimeError("TELEGRAM_USER_ID must be a numeric Telegram user id")

        def _flag(name: str, default: str) -> bool:
            return os.environ.get(name, default).strip() not in ("0", "false", "False", "")

        return cls(
            telegram_token=token,
            telegram_user_id=int(user_id),
            data_dir=Path(os.environ.get("SOMA_DATA_DIR", "data/soma")),
            model=os.environ.get("SOMA_MODEL") or None,
            debounce_seconds=float(os.environ.get("SOMA_DEBOUNCE_SECS", "3.0")),
            enable_curator=_flag("SOMA_ENABLE_CURATOR", "1"),
            enable_crystallize=_flag("SOMA_ENABLE_CRYSTALLIZE", "1"),
            notify_min_interval_seconds=float(
                os.environ.get("SOMA_NOTIFY_MIN_INTERVAL_SECS", str(DEFAULT_MIN_INTERVAL_SECONDS))
            ),
            crystallize_interval_seconds=float(
                os.environ.get("SOMA_CRYSTALLIZE_INTERVAL_SECS", "1800")
            ),
        )


class SomaApp:
    """Composition root. Builds every component and runs the transport loop.

    Conversation history is in-memory for Phase 4 (single-user, single-session).
    Restarting the process starts a fresh conversation; memories persist.
    """

    def __init__(
        self,
        config: SomaConfig,
        *,
        engine: Optional[SomaEngine] = None,
        embedder=None,
        store: Optional[MemoryStore] = None,
        extractor: Optional[MemoryExtractor] = None,
        context_builder: Optional[ContextBuilder] = None,
        events: Optional[EventLog] = None,
        transport: Optional[TelegramTransport] = None,
        curator: Optional[Curator] = None,
        notifier: Optional[Notifier] = None,
        crystallizer: Optional[Crystallizer] = None,
    ):
        self.config = config
        config.data_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or OllamaEmbedder()
        self.store = store or MemoryStore(config.data_dir / "memories.jsonl", self.embedder)
        self.extractor = extractor or MemoryExtractor(self.store, model=config.model)
        self.context_builder = context_builder or ContextBuilder(self.store)
        self.events = events or EventLog(config.data_dir / "events.jsonl")
        self.engine = engine or SomaEngine(EngineConfig(model=config.model or ""))
        self.transport = transport or TelegramTransport(
            token=config.telegram_token,
            allowed_user_id=config.telegram_user_id,
            on_user_message=self.handle_user_message,
            debounce_seconds=config.debounce_seconds,
        )
        self.notifier = notifier or Notifier(
            self.transport.send,
            state_path=config.data_dir / "notify_state.json",
            events=self.events,
            min_interval_seconds=config.notify_min_interval_seconds,
        )
        if curator is not None:
            self.curator = curator
        elif config.enable_curator:
            self.curator = Curator(
                self.store,
                self.events,
                model=config.model,
                notify=self.notifier.maybe_notify,
            )
        else:
            self.curator = None
        if crystallizer is not None:
            self.crystallizer = crystallizer
        elif config.enable_crystallize:
            self.crystallizer = Crystallizer(
                self.store,
                self.events,
                embedder=self.embedder,
                interval_seconds=config.crystallize_interval_seconds,
            )
        else:
            self.crystallizer = None
        self._history: List[dict] = []
        self._bg_tasks: set[asyncio.Task] = set()

    async def handle_user_message(self, text: str) -> Optional[str]:
        await self.events.record_async(EVENT_PROMPT_RECEIVED, text=text)

        try:
            built = await asyncio.to_thread(self.context_builder.build, text)
            result = await self.engine.run_turn(
                text,
                conversation_history=self._history,
                system_message=built.text or None,
            )
        except Exception as exc:
            logger.exception("soma: turn failed")
            await self.events.record_async(EVENT_ERROR, where="run_turn", message=str(exc))
            return "Sorry — internal error. The logs will tell me why."

        self._history = list(result.get("messages") or [])
        raw_reply = result.get("final_response") or ""
        reply = sanitize(
            raw_reply,
            strip_markdown=self.config.strip_markdown,
            max_chars=self.config.max_reply_chars,
        )

        await self.events.record_async(
            EVENT_RESPONSE_SENT,
            text=reply,
            api_calls=result.get("api_calls"),
            completed=result.get("completed"),
        )

        # Extraction runs in the background — the user has their reply already.
        self._spawn_extraction(text, reply)
        return reply

    def _spawn_extraction(self, user_text: str, assistant_text: str) -> None:
        async def _run():
            try:
                written = await self.extractor.extract_async(user_text, assistant_text)
            except Exception as exc:
                logger.exception("soma: extraction failed")
                await self.events.record_async(EVENT_ERROR, where="extract", message=str(exc))
                return
            for record in written:
                await self.events.record_async(
                    EVENT_MEMORY_WRITTEN,
                    id=record.id,
                    memory_type=record.type,
                    tags=record.tags,
                    use_count=record.use_count,
                    content=record.content,
                )

        task = asyncio.create_task(_run())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _stop(*_a):
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main thread — fall back to default handler.
                signal.signal(sig, _stop)

        transport_task = asyncio.create_task(self.transport.run_forever())
        curator_task: Optional[asyncio.Task] = None
        crystallize_task: Optional[asyncio.Task] = None
        if self.curator is not None:
            curator_task = asyncio.create_task(self.curator.run())
        if self.crystallizer is not None:
            crystallize_task = asyncio.create_task(self.crystallizer.run())
        try:
            await stop.wait()
        finally:
            for t in (transport_task, curator_task, crystallize_task):
                if t is None:
                    continue
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            # Wait briefly for background extractions to settle.
            if self._bg_tasks:
                await asyncio.wait(self._bg_tasks, timeout=10.0)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("SOMA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = SomaConfig.from_env()
    app = SomaApp(config)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
