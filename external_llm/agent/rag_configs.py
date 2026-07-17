from __future__ import annotations

from typing import Optional

from external_llm.agent.language_hint import _HANGUL_END, _HANGUL_START


def _extract_identifiers(text: str) -> list[str]:
    """Extract identifier-like tokens via char scan (replaces ``re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*")``)."""
    if not text:
        return []
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        # Hangul run
        if _HANGUL_START <= ch <= _HANGUL_END:
            start = i
            i += 1
            while i < len(text) and _HANGUL_START <= text[i] <= _HANGUL_END:
                i += 1
            tokens.append(text[start:i])
            continue
        # Latin identifier start
        if ch.isalpha() or ch == '_':
            start = i
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] == '_'):
                i += 1
            tokens.append(text[start:i])
            continue
        i += 1
    return tokens


def _split_camel_case(token: str) -> list[str]:
    """Split CamelCase token into parts: ``XMLParser`` → ``['XML', 'Parser']``.
    Pure char-by-char scanner — no regex dependencies."""
    if not token:
        return []
    parts: list[str] = []
    start = 0
    for i in range(1, len(token)):
        ch = token[i]
        prev = token[i - 1]
        if ch.isupper():
            if prev.islower():
                # lower→Upper : boundary before Upper
                parts.append(token[start:i])
                start = i
            elif i + 1 < len(token) and token[i + 1].islower():
                # Upper→lower (acronym suffix): boundary before last Upper
                parts.append(token[start:i])
                start = i
        elif ch.isdigit() and not prev.isdigit():
            parts.append(token[start:i])
            start = i
    parts.append(token[start:])
    return [p for p in parts if p]  # drop empties


# ═══════════════════════════════════════════════════════════════════════════
# Default stop-word sets
# ═══════════════════════════════════════════════════════════════════════════

# Common English stopwords for code tokenization
_CODE_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "can", "may", "might", "must", "shall", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    # Code-specific stop words (keywords that appear in requests but carry
    # no semantic weight for matching).
    "def", "class", "return", "import", "from", "not",
    "this", "that", "these", "those", "then", "than",
})


class CodeTokenizer:
    """Tokenizer for code text with configurable CamelCase/non-Latin/underscore handling.

    Design: tokenization logic is centralized here instead of duplicated across
    ``_tokenize_request`` (relevance_scorer), ``_tokenize`` (rag_searcher), and
    ``_extract_symbol_candidates`` (context_packs).

    Usage::

        tok = CodeTokenizer()
        tok.tokenize("isAsync")         # -> ["is", "async"]
        tok.tokenize("XMLParser")       # -> ["xml", "parser"]
        tok.tokenize("is_async")        # -> ["is", "async"]
        tok.tokenize("로그인")           # -> ["로그인"] (Korean)
    """

    def __init__(
        self,
        stop_words: Optional[set[str]] = None,
        min_token_len: int = 2,
        split_underscore: bool = True,
        split_camel: bool = True
    ):
        """
        Args:
            stop_words: Custom stop-word set. Falls back to ``_CODE_STOP_WORDS``.
            min_token_len: Minimum token length to keep (default 2).
            split_underscore: Whether to split ``snake_case`` (default True).
            split_camel: Whether to split ``CamelCase`` (default True).
        """
        self.stop_words = stop_words or _CODE_STOP_WORDS
        self.min_token_len = min_token_len
        self._split_underscore = split_underscore
        self._split_camel = split_camel

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into filtered, lowercased tokens.

        Splits on whitespace, punctuation, CamelCase, snake_case boundaries.
        Filters out very short tokens and common stop words.
        """
        if not text:
            return []

        # Step 1: Extract raw identifier tokens (char scanner, no regex)
        raw_tokens = _extract_identifiers(text)

        # Step 2: Sub-split each raw token
        result: list[str] = []
        for token in raw_tokens:
            t_lower = token.lower()
            sub_tokens: set[str] = set()

            # Original token (if meaningful)
            if len(t_lower) >= self.min_token_len and t_lower not in self.stop_words:
                sub_tokens.add(t_lower)

            # CamelCase split: "isAsync" → {is, async}, "XMLParser" → {xml, parser}
            if self._split_camel and token.isascii():
                parts = _split_camel_case(token)
                for p in parts:
                    p_lower = p.lower()
                    if len(p_lower) >= self.min_token_len and p_lower not in self.stop_words:
                        sub_tokens.add(p_lower)

            # Underscore split: "is_async" → {is, async}
            if self._split_underscore and "_" in t_lower:
                for part in t_lower.split("_"):
                    if len(part) >= self.min_token_len and part not in self.stop_words:
                        sub_tokens.add(part)

            result.extend(sorted(sub_tokens))

        return result



__all__ = ['CodeTokenizer']
