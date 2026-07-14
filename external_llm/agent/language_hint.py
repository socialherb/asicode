"""Language-neutral CJK script detection utilities.

Provides single-source-of-truth unicode range definitions and detection
functions used for word-count normalisation (task router) and other
language-agnostic CJK identification tasks.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# Unicode script ranges (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════

# Hangul Syllables (U+AC00–U+D7A3): 11,172 complete syllables
_HANGUL_START = '\uac00'
_HANGUL_END = '\ud7a3' # 힣

# CJK Unified Ideographs (U+4E00–U+9FFF)
# CJK Symbols and Punctuation (U+3000–U+303F)
_CJK_IDEOGRAPH_START = '\u4e00'
_CJK_IDEOGRAPH_END = '\u9fff'
_CJK_SYM_START = '\u3000'
_CJK_SYM_END = '\u303f'

# Katakana (U+30A0–U+30FF) and Hiragana (U+3040–U+309F)
_KATAKANA_START = '\u30a0'
_KATAKANA_END = '\u30ff'
_HIRAGANA_START = '\u3040'
_HIRAGANA_END = '\u309f'


def is_hangul(c: str) -> bool:
    """Check if character is a Hangul syllable (U+AC00–U+D7A3)."""
    return _HANGUL_START <= c <= _HANGUL_END if len(c) == 1 else False


def is_cjk_ideograph(c: str) -> bool:
    """Check if character is a CJK Unified Ideograph (U+4E00–U+9FFF)."""
    return _CJK_IDEOGRAPH_START <= c <= _CJK_IDEOGRAPH_END if len(c) == 1 else False


def is_cjk(c: str) -> bool:
    """Check if character is any CJK character (ideograph, symbol, kana, hangul)."""
    if len(c) != 1:
        return False
    return (
        is_hangul(c)
        or is_cjk_ideograph(c)
        or _CJK_SYM_START <= c <= _CJK_SYM_END
        or _KATAKANA_START <= c <= _KATAKANA_END
        or _HIRAGANA_START <= c <= _HIRAGANA_END
    )
