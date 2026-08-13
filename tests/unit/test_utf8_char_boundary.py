"""Regression tests for the shared UTF-8 char-boundary helper and the
byte-cap truncation path used by llm_execution._format_snippet.

Background: a byte-granularity cap can split a multibyte UTF-8 character
(Korean 한 = 3 bytes). Decoding each fragment with errors="replace" turns the
split char into U+FFFD(s). For a one-shot snippet it is a cosmetic trailing
U+FFFD; the shared helper (utils.string_helper.utf8_trailing_incomplete_len)
trims the partial bytes so the cut lands on a character boundary.
"""
from __future__ import annotations


def test_canonical_helper_lives_in_string_helper():
    """The helper lives at its canonical home (utils.string_helper). The
    cursor-advancing design_chat tailer that shared it was removed in P12-4;
    the only remaining consumer is llm_execution._format_snippet.
    """
    from utils import string_helper

    assert callable(string_helper.utf8_trailing_incomplete_len)
    # a few spec spot-checks at the canonical home
    h = string_helper.utf8_trailing_incomplete_len
    assert h(b"") == 0
    assert h(b"ascii") == 0
    assert h("한".encode()) == 0          # complete 3-byte
    assert h("한".encode()[:1]) == 1      # leading only → defer 1
    assert h("한".encode()[:2]) == 2      # leading + 1 cont → defer 2
    assert h(b"\x80\x80") == 0            # orphan continuations → 0 (no stall)


def test_snippet_byte_cap_does_not_corrupt_multibyte():
    """Regression for llm_execution._format_snippet (two duplicated copies).

    The snippet encodes the body to bytes, caps at max_bytes (a BYTE boundary),
    backs off to the last newline (newlines are ASCII 0x0A, never a UTF-8
    continuation byte, so the backoff yields a char-safe cut) and then decodes.
    The newline backoff does NOT fire when a single long line leaves no newline
    in the capped window — a byte-cap cut there splits a multibyte char and the
    orphan bytes decode to a trailing U+FFFD. The char-boundary trim (the
    shared helper) closes that gap.

    This mirrors the EXACT truncation logic of _format_snippet; reverting the
    trim makes the assertion fail (output ends with U+FFFD).
    """
    from utils.string_helper import utf8_trailing_incomplete_len

    # Single long line of Korean (3 bytes/char), no newline → rfind(b"\n") = -1
    # so the newline backoff is a no-op. max_bytes=5000 is not a multiple of 3
    # → the cap lands mid-character (1666 full chars = 4998 bytes + ED 95).
    text = "한" * 4000
    max_bytes = 5000
    b = text.encode("utf-8")
    assert len(b) > max_bytes

    b = b[:max_bytes]
    last_nl = b.rfind(b"\n")
    if last_nl > 0:
        b = b[: last_nl + 1]

    trim = utf8_trailing_incomplete_len(b)
    assert trim > 0, "expected a partial multibyte char straddling the cap"
    b = b[: len(b) - trim]

    out = b.decode("utf-8", errors="replace")
    assert "\ufffd" not in out, "byte-cap cut leaked a replacement char"
    assert out.endswith("한")           # ends on a COMPLETE character
    assert len(out) == 1666             # 4998 bytes / 3


def test_snippet_byte_cap_newline_backoff_already_safe_for_multiline():
    """For normal multi-line snippets the newline backoff already lands on a
    char boundary, so the trim is a harmless no-op (trim == 0). Confirms the
    trim does not over-trim valid multi-line output.
    """
    from utils.string_helper import utf8_trailing_incomplete_len

    # Many short Korean lines; cap mid-stream. rfind lands on a '\n' (ASCII).
    text = "\n".join("한글" * 30 for _ in range(200))  # 200 lines, each Korean
    b = text.encode("utf-8")
    max_bytes = 4000
    assert len(b) > max_bytes
    b = b[:max_bytes]
    last_nl = b.rfind(b"\n")
    assert last_nl > 0
    b = b[: last_nl + 1]
    trim = utf8_trailing_incomplete_len(b)
    assert trim == 0, "newline backoff should already end on a char boundary"
    out = b.decode("utf-8", errors="replace")
    assert "\ufffd" not in out
    assert out.endswith("\n")
