"""Tests for soma.sanitizer — reasoning strip, markdown strip, truncation."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.sanitizer import sanitize


class SanitizerTest(unittest.TestCase):
    def test_strips_thinking_tag(self):
        text = "<thinking>internal monologue</thinking>final answer"
        self.assertEqual(sanitize(text), "final answer")

    def test_strips_reasoning_tag(self):
        text = "<reasoning>steps</reasoning>\nthe result"
        self.assertEqual(sanitize(text), "the result")

    def test_strips_multiple_tags(self):
        text = "<thinking>a</thinking>between<reflection>b</reflection>after"
        self.assertEqual(sanitize(text), "betweenafter")

    def test_keeps_markdown_by_default(self):
        text = "**bold** and *italic* and `code`"
        self.assertEqual(sanitize(text), "**bold** and *italic* and `code`")

    def test_strips_markdown_when_requested(self):
        text = "## Header\n**bold** *italic* `code` [link](http://x)"
        out = sanitize(text, strip_markdown=True)
        self.assertNotIn("**", out)
        self.assertNotIn("##", out)
        self.assertNotIn("`", out)
        self.assertIn("bold", out)
        self.assertIn("italic", out)
        self.assertIn("link", out)
        self.assertNotIn("http://x", out)

    def test_truncates_long_text(self):
        text = "word " * 2000  # ~10000 chars
        out = sanitize(text, max_chars=100)
        self.assertLessEqual(len(out), 110)  # max_chars + " […]" suffix
        self.assertTrue(out.endswith("[…]"))

    def test_truncation_respects_word_boundary(self):
        text = ("hello " * 50).strip()
        out = sanitize(text, max_chars=20)
        # Should end with a complete "hello", not "hel"
        self.assertNotIn("hel […]", out)

    def test_short_text_not_truncated(self):
        text = "short"
        out = sanitize(text, max_chars=100)
        self.assertEqual(out, "short")

    def test_empty_returns_empty(self):
        self.assertEqual(sanitize(""), "")
        self.assertEqual(sanitize(None), "")

    def test_trims_whitespace(self):
        self.assertEqual(sanitize("   hi  \n\n  "), "hi")

    def test_case_insensitive_tag_match(self):
        text = "<THINKING>x</THINKING>y"
        self.assertEqual(sanitize(text), "y")


if __name__ == "__main__":
    unittest.main()
