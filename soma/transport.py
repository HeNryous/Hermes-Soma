"""Telegram transport — receives messages, debounces follow-ups,
serializes Hermes calls, delivers sanitized replies.

Single-user by design (TELEGRAM_USER_ID gates inbound messages).
A separate MessagePipeline handles debounce + lock so the integration
logic is testable without booting a Telegram client.

Built on python-telegram-bot (v22) — same dependency the existing
gateway/platforms/telegram.py uses, so we don't pull in a parallel
Telegram SDK.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# A batch handler turns one debounced batch of user text into a reply.
# Returning None or "" suppresses the outbound message.
BatchHandler = Callable[[str], Awaitable[Optional[str]]]


class MessagePipeline:
    """Debounce + serialize per-user message processing.

    Multiple messages arriving within `debounce_seconds` are concatenated
    and handled as one batch. While the handler runs, additional messages
    queue up; the next batch fires only after the handler returns and the
    debounce window expires again.

    No Telegram dependency — feed it text and a handler, you get one
    handler call per logical user input.
    """

    def __init__(
        self,
        on_batch: BatchHandler,
        *,
        debounce_seconds: float = 3.0,
    ):
        self.on_batch = on_batch
        self.debounce_seconds = debounce_seconds
        self._buffer: list[str] = []
        self._timer: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._buffer_lock = asyncio.Lock()
        # True while a timer task is past its debounce sleep and inside
        # the handler. submit() must NOT cancel a draining timer — that
        # would propagate CancelledError into the running handler.
        self._draining = False

    async def submit(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        async with self._buffer_lock:
            self._buffer.append(text)
            if self._draining:
                # Handler is running; let it finish. _fire_after_debounce
                # checks the buffer after drain and reschedules itself.
                return
            if self._timer and not self._timer.done():
                self._timer.cancel()
            self._timer = asyncio.create_task(self._fire_after_debounce())

    async def flush_now(self) -> None:
        """Force an immediate flush (used by tests; can also be wired to /flush)."""
        async with self._buffer_lock:
            if self._timer and not self._timer.done() and not self._draining:
                self._timer.cancel()
        await self._drain()

    async def _fire_after_debounce(self) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            return
        self._draining = True
        try:
            await self._drain()
        finally:
            self._draining = False
        # If submits arrived while we were draining, schedule the next round.
        async with self._buffer_lock:
            if self._buffer:
                self._timer = asyncio.create_task(self._fire_after_debounce())
            else:
                self._timer = None

    async def _drain(self) -> None:
        async with self._buffer_lock:
            if not self._buffer:
                return
            batch = "\n".join(self._buffer)
            self._buffer.clear()

        # Lock ensures only one Hermes turn runs at a time even if a new
        # debounce fires while this one is still processing.
        async with self._lock:
            try:
                reply = await self.on_batch(batch)
            except Exception:
                logger.exception("soma: batch handler raised")
                return
        # Reply delivery is the handler's responsibility — keep the
        # pipeline focused on serialization. If the handler chooses to
        # return the reply, we don't dispatch it; transport callers
        # build their own send chain (see TelegramTransport.on_batch).
        _ = reply


class TelegramTransport:
    """Thin wrapper around python-telegram-bot's Application.

    Lazy-imports `telegram` so tests for the pipeline can run without
    the SDK installed.
    """

    def __init__(
        self,
        *,
        token: str,
        allowed_user_id: int,
        on_user_message: Callable[[str], Awaitable[Optional[str]]],
        debounce_seconds: float = 3.0,
    ):
        if not token:
            raise ValueError("TelegramTransport requires a bot token")
        if not allowed_user_id:
            raise ValueError("TelegramTransport requires an allowed_user_id (single-user)")
        self.token = token
        self.allowed_user_id = int(allowed_user_id)
        self.on_user_message = on_user_message
        self._chat_id: Optional[int] = None
        self._app = None  # built on start()
        self._pipeline = MessagePipeline(
            self._run_batch,
            debounce_seconds=debounce_seconds,
        )

    # -- Lifecycle -----------------------------------------------------------

    def build_application(self):
        from telegram.ext import Application, MessageHandler, filters

        app = Application.builder().token(self.token).build()
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self._app = app
        return app

    async def run_forever(self) -> None:
        app = self.build_application()
        await app.initialize()
        await app.start()
        try:
            await app.updater.start_polling(drop_pending_updates=True)
            logger.info("soma: telegram transport listening for user %d", self.allowed_user_id)
            stop = asyncio.Event()
            await stop.wait()  # block forever; cancellation tears it down
        finally:
            try:
                await app.updater.stop()
            except Exception:
                pass
            await app.stop()
            await app.shutdown()

    async def send(self, text: str) -> None:
        """Send a message to the known user chat. No-op if we don't have one yet."""
        if not text:
            return
        if self._app is None or self._chat_id is None:
            logger.warning("soma: send() before first inbound message; dropping")
            return
        try:
            await self._app.bot.send_message(chat_id=self._chat_id, text=text)
        except Exception:
            logger.exception("soma: failed to send telegram message")

    # -- Internals -----------------------------------------------------------

    async def _on_text(self, update, _context) -> None:
        message = getattr(update, "message", None)
        if message is None or message.text is None:
            return
        sender = getattr(message, "from_user", None)
        if sender is None or sender.id != self.allowed_user_id:
            logger.warning("soma: dropping message from unauthorized user %s",
                           sender.id if sender else "?")
            return
        self._chat_id = message.chat_id
        await self._pipeline.submit(message.text)

    async def _run_batch(self, text: str) -> Optional[str]:
        reply = await self.on_user_message(text)
        if reply:
            await self.send(reply)
        return reply
