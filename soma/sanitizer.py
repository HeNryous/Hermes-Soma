"""Response sanitization for outbound messages.

Hermes' final_response can contain:
  - Reasoning tags like <thinking>...</thinking>, <reasoning>, <reflection>
  - Markdown that renders poorly when forwarded across platforms
  - Run-on text longer than the platform message limit (Telegram = 4096)

sanitize() strips reasoning, optionally strips markdown, and truncates
on a word boundary. Default keeps markdown — most Telegram clients
render it fine and code blocks are useful.
"""

from __future__ import annotations

import re

# Reasoning / thinking / reflection wrappers that some models emit even
# when their final answer follows. Strip them whole.
_REASONING_TAG_RE = re.compile(
    r"<\s*(thinking|reasoning|reflection|scratch(?:pad)?|analysis)\b[^>]*>.*?"
    r"<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Markdown removal: headers, bold/italic markers, inline code ticks, links.
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def sanitize(
    text: str,
    *,
    strip_markdown: bool = False,
    max_chars: int = 4000,
) -> str:
    """Clean a model response for outbound delivery.

    - Always strips reasoning/thinking tags.
    - Optionally strips basic Markdown.
    - Always truncates to max_chars on a word boundary.
    - Always trims surrounding whitespace.
    """
    if not text:
        return ""
    text = _REASONING_TAG_RE.sub("", text)
    if strip_markdown:
        text = _MD_HEADER_RE.sub("", text)
        text = _MD_BOLD_RE.sub(r"\1", text)
        text = _MD_ITALIC_RE.sub(r"\1", text)
        text = _MD_INLINE_CODE_RE.sub(r"\1", text)
        text = _MD_LINK_RE.sub(r"\1", text)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Truncate on the last space before max_chars to avoid mid-word cuts.
    cut = text.rfind(" ", 0, max_chars)
    if cut < int(max_chars * 0.6):
        cut = max_chars
    return text[:cut].rstrip() + " […]"
