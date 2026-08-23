from __future__ import annotations

import re
from functools import lru_cache

from external_llm.agent.language_hint import _HANGUL_END, _HANGUL_START

# Identifier scanner (see _extract_identifiers).  Two alternatives, Hangul
# FIRST so a Hangul run terminates at the first non-Hangul char instead of
# being swallowed by the ``\w*`` tail ("로그인test" → 로그인 + test).
#
#   [가-힣]+     one Hangul run (U+AC00-U+D7A3, matching _HANGUL_START/_END)
#   [^\W\d]\w*   identifier: start = word-char that is not a decimal digit,
#                tail = word-chars.  In Python's re, ``\w`` is exactly
#                "str.isalnum() or '_'", so the tail matches the char scan
#                verbatim; ``\d`` is category Nd only, so the START class is
#                slightly wider than ``str.isalpha() or '_'`` — reconciled by
#                the re-scan guard in _extract_identifiers.
_IDENT_RE = re.compile(r"[가-힣]+|[^\W\d]\w*")


def _scan_identifiers(text: str) -> list[str]:
    """Reference char-scan implementation of :func:`_extract_identifiers`.

    Kept as the exactness oracle for the regex fast path: it is the only place
    that expresses "identifier start = ``str.isalpha()`` or ``_``" precisely,
    which no ``re`` character class can (``re`` has no Unicode-category
    escapes).  Called from the fast path only for the rare match that begins
    with a non-alphabetic word char, and from the equivalence test.
    """
    if not text:
        return []
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # Hangul run
        if _HANGUL_START <= ch <= _HANGUL_END:
            start = i
            i += 1
            while i < n and _HANGUL_START <= text[i] <= _HANGUL_END:
                i += 1
            tokens.append(text[start:i])
            continue
        # Latin identifier start
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(text[start:i])
            continue
        i += 1
    return tokens


def _extract_identifiers(text: str) -> list[str]:
    """Extract identifier-like tokens (Latin/Unicode identifiers + Hangul runs).

    Regex-driven: the equivalent char-by-char scan (:func:`_scan_identifiers`)
    burned ~8.6s of self time and ~34M ``len()`` calls building this repo's own
    BM25 index, because every character cost several Python-level method calls.
    ``re`` runs the same scan in C.

    Exactness: ``_IDENT_RE``'s start class ``[^\\W\\d]`` excludes decimal digits
    (Nd) but NOT the two other numeric categories — Nl (``Ⅷ``) and the No
    digits (``①``, ``²``) — which ``str.isalpha()`` rejects.  Such a character
    would start a token here but be skipped by the scanner, so any match not
    beginning with a letter/underscore is re-scanned with the oracle.  Those
    characters occur only in prose/decorative comments, so the guard costs one
    ``str.isalpha()`` per token and fires essentially never.
    """
    if not text:
        return []
    tokens: list[str] = []
    for tok in _IDENT_RE.findall(text):
        first = tok[0]
        if first.isalpha() or first == "_":
            tokens.append(tok)
        else:
            # Nl/No lead char: the scanner skips it and restarts after, which
            # can also re-expose a Hangul boundary ("①로그인test").
            tokens.extend(_scan_identifiers(tok))
    return tokens


def _scan_camel_case(token: str) -> tuple[str, ...]:
    """Split a CamelCase token: ``XMLParser`` → ``('XML', 'Parser')``.

    Returns a tuple because :func:`_split_camel_case_cached` shares one result
    object across every caller — a list would let a single in-place mutation
    poison the cache for the process lifetime.
    """
    if not token:
        return ()
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
    return tuple(p for p in parts if p)  # drop empties


# Identifiers repeat heavily in source code — measured 19.1x (401k occurrences,
# 21k distinct) across this repo — so the split is memoised rather than redone
# per occurrence.  The cap bounds a long-lived process indexing many repos;
# distinct-identifier counts sit well under it for a single tree, so the LRU
# behaves as a plain memo in the common case.
_split_camel_case_cached = lru_cache(maxsize=100_000)(_scan_camel_case)


def _split_camel_case(token: str) -> list[str]:
    """List-returning wrapper over the memoised splitter.

    Copies out of the shared tuple so callers keep a mutable, private result.
    The hot path (:meth:`CodeTokenizer.tokenize`) only iterates, so it uses
    ``_split_camel_case_cached`` directly and skips this copy.
    """
    return list(_split_camel_case_cached(token))


# ═══════════════════════════════════════════════════════════════════════════
# Default stop-word sets
# ═══════════════════════════════════════════════════════════════════════════

# Common English stopwords for code tokenization
_CODE_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "shall",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        # Code-specific stop words (keywords that appear in requests but carry
        # no semantic weight for matching).
        "def",
        "class",
        "return",
        "import",
        "from",
        "not",
        "this",
        "that",
        "these",
        "those",
        "then",
        "than",
    }
)


class CodeTokenizer:
    """Tokenizer for code text with configurable CamelCase/non-Latin/underscore handling.

    Design: tokenization logic is centralized here instead of duplicated across
    ``_tokenize_request`` (relevance_scorer), ``_tokenize`` (rag_searcher), and
    ``_extract_symbol_candidates`` (context_packs).

    Every emitted form is kept — the joined identifier AND its parts — so a
    query for either matches.  Parts that are stop words are dropped, which is
    why ``is`` does not appear below.

    Usage::

        tok = CodeTokenizer()
        tok.tokenize("isAsync")         # -> ["async", "isasync"]
        tok.tokenize("XMLParser")       # -> ["parser", "xml", "xmlparser"]
        tok.tokenize("is_async")        # -> ["async", "is_async"]
        tok.tokenize("로그인")           # -> ["로그인"] (Korean)
    """

    def __init__(
        self,
        stop_words: set[str] | None = None,
        min_token_len: int = 2,
        split_underscore: bool = True,
        split_camel: bool = True,
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
        # Per-instance memo of a raw token's sorted sub-forms (see ``tokenize``).
        # Config is fixed at construction, so a token's expansion is a pure
        # function of the token within an instance.
        self._token_cache: dict[str, list[str]] = {}

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into filtered, lowercased tokens.

        Splits on whitespace, punctuation, CamelCase, snake_case boundaries.
        Filters out very short tokens and common stop words.
        """
        if not text:
            return []

        # Step 1: Extract raw identifier tokens
        raw_tokens = _extract_identifiers(text)

        # Step 2: Sub-split each raw token. A token's sub-forms are a pure
        # function of the token given this instance's (immutable) config, so
        # memoise per-instance: identifiers repeat heavily across source
        # (measured ~96% hit rate over a 1139-file index) and this collapses
        # the per-token set()/sorted() into a single dict lookup. The cache is
        # bounded by the vocabulary, which is far smaller than occurrence
        # counts, so it stays flat over a build. Mirrors the existing
        # _split_camel_case_cached memo one level down.
        result: list[str] = []
        cache = self._token_cache
        for token in raw_tokens:
            sub = cache.get(token)
            if sub is None:
                sub = self._expand_token(token)
                cache[token] = sub
            result.extend(sub)

        return result

    def _expand_token(self, token: str) -> list[str]:
        """Lowercased, filtered sub-token forms of one raw token, sorted.

        Pure function of ``token`` for a given (construction-time) config, so
        ``tokenize`` memoises its result per-instance. ``tokenize`` only reads
        the returned list (``extend``), so sharing one cached object across
        occurrences is safe.
        """
        t_lower = token.lower()
        sub_tokens: set[str] = set()

        # Original token (if meaningful)
        if len(t_lower) >= self.min_token_len and t_lower not in self.stop_words:
            sub_tokens.add(t_lower)

        # CamelCase split: "isAsync" → {is, async}, "XMLParser" → {xml, parser}
        if self._split_camel and token.isascii():
            # Cached (tuple) form: this loop only reads, so it must not pay
            # for the defensive copy _split_camel_case makes.
            parts = _split_camel_case_cached(token)
            for p in parts:
                p_lower = p.lower()
                if len(p_lower) >= self.min_token_len and p_lower not in self.stop_words:
                    sub_tokens.add(p_lower)

        # Underscore split: "is_async" → {is, async}
        if self._split_underscore and "_" in t_lower:
            for part in t_lower.split("_"):
                if len(part) >= self.min_token_len and part not in self.stop_words:
                    sub_tokens.add(part)

        return sorted(sub_tokens)


__all__ = ["CodeTokenizer"]
