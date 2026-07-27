"""Unit tests for the BM25 tokenizer in ``rag_configs``.

The tokenizer runs once per indexed document, so it is the hot path of every
cold ``find_relevant_files`` call.  ``_extract_identifiers`` is therefore a
compiled regex rather than a char-by-char scan, and ``_split_camel_case`` is
memoised.  Both are optimisations over a reference implementation that these
tests pin:

* ``_scan_identifiers`` is the exactness oracle — it is the only expression of
  "identifier start = ``str.isalpha()`` or ``_``" that ``re`` cannot encode
  (no Unicode-category escapes).  The regex fast path must agree with it on
  every input, so the oracle is differentially tested against the fast path.
* the memo shares one result object across callers, so the public
  list-returning wrapper must hand out a fresh copy each call.
"""
from __future__ import annotations

import random

import pytest

from external_llm.agent.rag_configs import (
    CodeTokenizer,
    _extract_identifiers,
    _scan_identifiers,
    _split_camel_case,
    _split_camel_case_cached,
)

# Inputs that separate the regex fast path from the oracle, or that pin a
# boundary rule the tokenizer depends on.
_EDGE_INPUTS = [
    "",
    " ",
    "_",
    "__future__",
    "123",
    "2abc",
    "abc2def",
    "123abc456",
    # Nl / No lead chars: `str.isalpha()` is False for these, but they are
    # word chars that are not category-Nd digits, so `[^\W\d]` admits them and
    # the regex alone would start a token there.  The oracle must win.
    "①",  # ① CIRCLED DIGIT ONE
    "①abc",  # ①abc
    "②abc",  # ②abc
    "Ⅷ",  # Ⅷ ROMAN NUMERAL EIGHT (Nl)
    "Ⅷabc",  # Ⅷabc
    "x²y",  # x²y (No superscript)
    # Hangul runs must terminate at the first non-Hangul char rather than being
    # swallowed by the identifier tail.
    "로그인",
    "로그인test",
    "test로그인",
    "_로그인",
    "로그인_test",
    "2로그인test",
    "①로그인test",  # ①로그인test — Nl/No lead AND a Hangul boundary
    "ㄱㄴㄷ",  # compatibility jamo: outside U+AC00-U+D7A3
    # CamelCase / acronym boundaries.
    "XMLParser",
    "isAsync",
    "HTTPSConnection",
    "parseHTML5Doc",
    "snake_case_ID2Name",
    "getID",
    "aB",
    "AB",
    "ABc",
    # Other scripts must not crash or split oddly.
    "café",
    "ЖизньNow",
    "中文abc",
    "こんにちは",
]


@pytest.mark.parametrize("text", _EDGE_INPUTS)
def test_regex_fast_path_matches_scan_oracle(text: str):
    """The regex path must agree with the char-scan oracle on every input."""
    assert _extract_identifiers(text) == _scan_identifiers(text), text


def test_fuzz_regex_fast_path_matches_scan_oracle():
    """Differential fuzz over Latin, Hangul, CJK, numeric and punctuation chars.

    Guards the `[^\\W\\d]` start-class gap generatively: any future widening of
    the regex that admits a non-alphabetic lead char without re-scanning shows
    up here even if it is not in the hand-written edge list.
    """
    rng = random.Random(20260727)
    pool = (
        [chr(c) for c in range(0x20, 0x2FFF)]
        + [chr(c) for c in range(0xAC00, 0xD7A4, 37)]  # Hangul syllables
        + [chr(c) for c in range(0x3040, 0x33FF, 7)]  # kana / CJK symbols
        + list("_ \t\n.,()[]{}0123456789")
    )
    for _ in range(1500):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 48)))
        assert _extract_identifiers(text) == _scan_identifiers(text), repr(text)


def test_nl_no_lead_char_is_not_part_of_the_token():
    """``①abc`` yields ``abc`` — the circled digit is a separator, not a start.

    Pinned explicitly because it is the single behaviour the regex cannot
    express on its own; if the re-scan guard is dropped as "dead", the token
    silently becomes ``①abc`` and stops matching a search for ``abc``.
    """
    assert _extract_identifiers("①abc") == ["abc"]
    assert _extract_identifiers("①로그인test") == ["로그인", "test"]


def test_hangul_run_terminates_at_non_hangul():
    assert _extract_identifiers("로그인test") == ["로그인", "test"]
    # …but a Hangul char *inside* a Latin identifier is a tail char, so the
    # run is not re-split from the left.
    assert _extract_identifiers("test로그인") == ["test로그인"]


def test_split_camel_case_returns_a_fresh_mutable_list():
    """The memo shares one tuple; the public wrapper must copy out of it.

    Without the copy, a caller doing ``parts.append(...)`` would corrupt the
    cached entry for every later caller in the process.
    """
    first = _split_camel_case("XMLParser")
    assert first == ["XML", "Parser"]
    first.append("MUTATED")
    assert _split_camel_case("XMLParser") == ["XML", "Parser"]


def test_split_camel_case_cached_is_memoised():
    """The tuple form is what the hot path calls; identical input must reuse it."""
    _split_camel_case_cached.cache_clear()
    a = _split_camel_case_cached("HTTPSConnection")
    b = _split_camel_case_cached("HTTPSConnection")
    assert a is b, "repeated identifiers must not be re-split"
    assert _split_camel_case_cached.cache_info().hits >= 1


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("XMLParser", ["XML", "Parser"]),
        ("isAsync", ["is", "Async"]),
        ("getID", ["get", "ID"]),
        ("parseHTML5Doc", ["parse", "HTML", "5", "Doc"]),
        ("lower", ["lower"]),
        ("", []),
    ],
)
def test_split_camel_case_boundaries(token: str, expected: list[str]):
    assert _split_camel_case(token) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # "is" is a stop word, so the camel/underscore part is dropped while the
        # joined form survives — the docstring examples are written accordingly.
        ("isAsync", ["async", "isasync"]),
        ("XMLParser", ["parser", "xml", "xmlparser"]),
        ("is_async", ["async", "is_async"]),
        ("로그인", ["로그인"]),
    ],
)
def test_code_tokenizer_documented_examples(text: str, expected: list[str]):
    """The examples in ``CodeTokenizer``'s docstring, pinned as behaviour."""
    assert CodeTokenizer().tokenize(text) == expected


def test_code_tokenizer_drops_stop_words_and_short_tokens():
    out = CodeTokenizer().tokenize("the def a xy")
    assert "the" not in out and "def" not in out
    assert "a" not in out, "single chars are below min_token_len"
    assert "xy" in out


def test_code_tokenizer_memoises_per_token():
    """``tokenize`` memoises each raw token's sorted sub-forms per-instance.

    Identifiers repeat heavily across source, so the per-token set()/sorted()
    is the hot loop of a cold ``find_relevant_files`` (measured ~96% cache hit
    over a 1139-file index). The memo turns repeated occurrences into a single
    dict lookup. Pinning: the cache is populated, reuses one list object per
    token (``extend`` only reads it, so sharing is safe), output stays
    identical, and a custom-config instance keeps its own cache separate from
    a default instance (config is part of the pure-function input).
    """
    tok = CodeTokenizer()
    out = tok.tokenize("isAsync isAsync XMLParser getHTTP getHTTP")
    assert "isasync" in out and "async" in out
    assert "xmlparser" in out and "xml" in out and "parser" in out
    assert "gethttp" in out and "get" in out and "http" in out

    # Raw identifiers in the input are memoised.
    assert "isAsync" in tok._token_cache
    assert "XMLParser" in tok._token_cache
    assert "getHTTP" in tok._token_cache
    # The cached form equals a fresh computation.
    assert tok._token_cache["XMLParser"] == tok._expand_token("XMLParser")

    # tokenize never mutates the cached list — it only extends ``result``.
    cached_before = tok._token_cache["isAsync"]
    tok.tokenize("isAsync")
    assert tok._token_cache["isAsync"] is cached_before

    # A custom-config instance has an independent cache: config is part of the
    # pure-function input, so the memo must not be shared across configs.
    custom = CodeTokenizer(min_token_len=5, split_camel=False)
    # split_camel=False drops the xml/parser parts that the default emits.
    assert custom.tokenize("XMLParser") != tok.tokenize("XMLParser")
    assert custom._token_cache is not tok._token_cache
