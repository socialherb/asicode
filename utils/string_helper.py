"""
String utility functions for asicode.

Provides common string manipulation functions that can be used across the project.
"""
from __future__ import annotations

from typing import Optional


def utf8_trailing_incomplete_len(raw: bytes) -> int:
    """Number of trailing bytes in ``raw`` that form an INCOMPLETE multibyte
    UTF-8 sequence (a leading byte whose continuation bytes were cut off).

    Returns 0 when ``raw`` ends on a complete-character boundary — safe to
    decode the whole buffer (and, for a streaming cursor, advance past
    ``len(raw)``). Otherwise returns 1..3: those trailing bytes must be
    DEFERRED to the next read so a multibyte character is never split across
    two decode calls. Splitting a 3-byte CJK character (e.g. Korean 한) at a
    byte-cap boundary makes EACH fragment decode to U+FFFD with
    ``errors="replace"`` — one for the lone leading byte on this chunk, one
    for each orphan continuation byte on the next — corrupting the output,
    and (for a cursor-advancing stream) irrecoverably once the cursor jumps
    past the split point.

    Edge case: if the entire tail is orphan continuation bytes (the cursor
    landed mid-char, e.g. a corrupt offset into a multibyte sequence), there
    is no leading byte to wait for — returns 0 so the caller lets
    ``errors="replace"`` turn each orphan into U+FFFD rather than deferring
    forever and stalling the stream.

    Used by the byte-capped dev-tool tailer (cursor-advancing, must defer)
    and by snippet truncation (encode→byte-slice→decode, must trim).
    """
    if not raw:
        return 0
    n = len(raw)
    # Walk back over continuation bytes (0b10xxxxxx) to the leading byte of the
    # final sequence. A UTF-8 sequence is at most 4 bytes, so cap the scan at 3.
    i = n - 1
    back = 0
    while back < 3 and i >= 0 and (raw[i] & 0xC0) == 0x80:
        i -= 1
        back += 1
    if i < 0:
        # The entire tail is orphan continuation bytes (corrupt offset). No
        # incomplete char to wait for — let errors="replace" handle each byte.
        return 0
    lead = raw[i]
    if lead < 0x80:
        expected = 1          # ASCII
    elif lead < 0xE0:
        expected = 2          # 2-byte
    elif lead < 0xF0:
        expected = 3          # 3-byte (most CJK, incl. Korean)
    else:
        expected = 4          # 4-byte (emoji / astral plane)
    have = n - i
    return have if have < expected else 0


def reverse_string(s: str) -> str:
    """
    Return the reversed string.

    Args:
        s: Input string to reverse

    Returns:
        Reversed string

    Raises:
        TypeError: If input is not a string

    Examples:
        >>> reverse_string("hello")
        'olleh'
        >>> reverse_string("")
        ''
        >>> reverse_string("a")
        'a'
    """
    return s[::-1]
def count_vowels(s: str) -> int:
    """
    Count the number of vowels in a string (case-insensitive).

    Vowels are defined as: a, e, i, o, u

    Args:
        s: Input string to count vowels in

    Returns:
        Number of vowels in the string

    Examples:
        >>> count_vowels("hello")
        2
        >>> count_vowels("HELLO WORLD")
        3
        >>> count_vowels("xyz")
        0
        >>> count_vowels("")
        0
    """
    if not s:
        return 0

    # Count vowels (case-insensitive)
    vowels = set('aeiou')
    return sum(1 for char in s.lower() if char in vowels)
def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric characters).

    Returns True for empty string. Only alphanumeric characters are considered;
    all other characters (spaces, punctuation, etc.) are ignored.

    Args:
        s: Input string to check

    Returns:
        True if the string is a palindrome (ignoring case and non-alphanumeric characters), False otherwise

    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("A man, a plan, a canal: Panama")
        True
        >>> is_palindrome("hello")
        False
        >>> is_palindrome("")
        True
    """
    if not s:
        return True
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def has_exact_word(text: str, word: str) -> bool:
    """Check if `word` appears as a standalone word in `text` (no regex).

    Word boundary characters: space, punctuation, brackets, operators.
    Replaces patterns like ``re.search(r'\\b' + re.escape(k) + r'\\b', text)``.
    """
    if not text or not word:
        return False
    idx = text.find(word)
    wlen = len(word)
    _boundary_chars = frozenset(
        " \t\n\r()[]{}.,;:'\"!@#$%^&*+-=<>?/`~"
    )
    while idx != -1:
        before_ok = idx == 0 or text[idx - 1] in _boundary_chars
        after_ok = (idx + wlen >= len(text)
                    or text[idx + wlen] in _boundary_chars)
        if before_ok and after_ok:
            return True
        idx = text.find(word, idx + 1)
    return False


def extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from text, handling code fences and braces.

    Tries: 1) code-fenced JSON, 2) bare JSON object, 3) first dict-like substring.
    Replaces patterns like ``re.search(r'```(?:json)?\\s*(\\{.*?\\})\\s*```', text, re.DOTALL)``.
    """
    import json as _json
    # Strategy 1: code fence with json
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return _json.loads(part)
                except _json.JSONDecodeError:
                    continue
    # Strategy 2: find first {…} block, ignoring braces inside strings
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str and ch == '{':
            start = i
            break
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return _json.loads(text[start:i + 1])
                        except _json.JSONDecodeError:
                            return None


def parse_json(text: str) -> Optional[dict]:
    """Alias for extract_json. Thin wrapper for renamed import compatibility."""
    return extract_json(text)

if __name__ == "__main__":
    # Simple test cases
    test_cases = [
        ("hello", "olleh"),
        ("", ""),
        ("a", "a"),
        ("racecar", "racecar"),
    ]

    print("Testing reverse_string:")
    for input_str, expected in test_cases:
        result = reverse_string(input_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} reverse_string('{input_str}') = '{result}' (expected: '{expected}')")

    print("\nTesting count_vowels:")
    vowel_tests = [
        ("hello", 2),
        ("HELLO WORLD", 3),
        ("xyz", 0),
        ("", 0),
        ("aeiou", 5),
        ("AEIOU", 5),
    ]
    for input_str, expected in vowel_tests:
        result = count_vowels(input_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} count_vowels('{input_str}') = {result} (expected: {expected})")

    print("\nTesting is_palindrome:")
    palindrome_tests = [
        ("racecar", True),
        ("A man, a plan, a canal: Panama", True),
        ("hello", False),
        ("", True),
        ("a", True),
        ("race car", True),
        ("12321", True),
        ("12345", False),
    ]
    for input_str, expected in palindrome_tests:
        result = is_palindrome(input_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} is_palindrome('{input_str}') = {result} (expected: {expected})")
