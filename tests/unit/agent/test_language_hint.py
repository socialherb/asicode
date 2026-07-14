"""Tests for language_hint.py — CJK/Hangul script detection utilities."""

from __future__ import annotations

from external_llm.agent.language_hint import is_cjk, is_cjk_ideograph, is_hangul

# Unicode range boundaries
_HANGUL_START = '\uac00'
_HANGUL_END = '\ud7a3'
_CJK_START = '\u4e00'
_CJK_END = '\u9fff'
_CJK_SYM_START = '\u3000'
_CJK_SYM_END = '\u303f'
_KATAKANA_START = '\u30a0'
_KATAKANA_END = '\u30ff'
_HIRAGANA_START = '\u3040'
_HIRAGANA_END = '\u309f'


class TestIsHangul:
    """Coverage for is_hangul() — True, False, and edge cases."""

    def test_hangul_single_char(self):
        assert is_hangul('한') is True       # U+D55C inside range
        assert is_hangul('글') is True       # U+AE00 inside range

    def test_hangul_boundaries(self):
        assert is_hangul(_HANGUL_START) is True   # U+AC00
        assert is_hangul(_HANGUL_END) is True     # U+D7A3

    def test_non_hangul(self):
        assert is_hangul('a') is False       # ASCII
        assert is_hangul('中') is False      # CJK ideograph
        assert is_hangul('') is False        # empty string
        assert is_hangul('가나') is False     # multi-character string


class TestIsCjkIdeograph:
    """Coverage for is_cjk_ideograph() — True, False, and edge cases."""

    def test_cjk_ideograph_single_char(self):
        assert is_cjk_ideograph('中') is True
        assert is_cjk_ideograph('国') is True

    def test_cjk_ideograph_boundaries(self):
        assert is_cjk_ideograph(_CJK_START) is True   # U+4E00
        assert is_cjk_ideograph(_CJK_END) is True     # U+9FFF

    def test_non_cjk_ideograph(self):
        assert is_cjk_ideograph('a') is False
        assert is_cjk_ideograph('あ') is False   # Hiragana
        assert is_cjk_ideograph('') is False     # empty
        assert is_cjk_ideograph('中国') is False  # multi-char


class TestIsCjk:
    """Coverage for is_cjk() — all CJK sub-ranges, plus edge cases."""

    def test_hangul(self):
        assert is_cjk('한') is True

    def test_cjk_ideograph(self):
        assert is_cjk('中') is True

    def test_cjk_symbol(self):
        assert is_cjk(_CJK_SYM_START) is True   # U+3000
        assert is_cjk(_CJK_SYM_END) is True     # U+303F
        assert is_cjk('\u3001') is True          # 、
        assert is_cjk('\u3002') is True          # 。

    def test_katakana(self):
        assert is_cjk(_KATAKANA_START) is True  # U+30A0
        assert is_cjk(_KATAKANA_END) is True    # U+30FF
        assert is_cjk('ア') is True              # U+30A2

    def test_hiragana(self):
        assert is_cjk(_HIRAGANA_START) is True  # U+3040
        assert is_cjk(_HIRAGANA_END) is True    # U+309F
        assert is_cjk('あ') is True              # U+3042

    def test_non_cjk(self):
        assert is_cjk('a') is False
        assert is_cjk('1') is False
        assert is_cjk('') is False       # empty
        assert is_cjk('한글') is False    # multi-char (explicit else branch)
        assert is_cjk('中国') is False    # multi-char (explicit else branch)

    def test_boundary_outside_lower(self):
        """Just below each range should return False."""
        assert is_cjk('\u2fff') is False   # before CJK ideographs
        assert is_cjk('\u309f') is True    # Hiragana end (inside)
        assert is_cjk('\u30a0') is True    # Katakana start (inside)

    def test_multi_char_early_return(self):
        """len(c) != 1 short-circuits — tests the explicit else branch."""
        assert is_cjk('') is False
        assert is_cjk('ab') is False
        assert is_cjk('한글') is False
        assert is_cjk('   ') is False
