"""Tests for helper functions in external_llm/providers.py."""
from __future__ import annotations

import ast
import inspect

import pytest
import requests

import external_llm.providers as providers_module
from external_llm.providers import (
    _count_delimiters,
    _is_gemini_3,
    _is_gpt_oss,
    _normalize_gemini_finish_reason,
    _ollama_think_value,
)

# ── _normalize_gemini_finish_reason ────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("FINISH_REASON_UNSPECIFIED", "stop"),
        ("OTHER", "stop"),
        (None, None),
        ("stop", "stop"),  # already lowercase
        ("length", "length"),
        ("  STOP  ", "stop"),  # whitespace handling
        ("UNKNOWN_CODE", "unknown_code"),  # fallback: lowercased
        ("", ""),
    ],
)
def test_normalize_gemini_finish_reason(raw: str | None, expected: str | None) -> None:
    assert _normalize_gemini_finish_reason(raw) == expected


# ── _is_gpt_oss ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-oss", True),
        ("gpt-oss-v2", True),
        ("GPT-OSS", True),
        ("GPT-OSS-7B", True),
        ("gpt-4", False),
        ("deepseek-v4", False),
        ("", False),
        (None, False),  # type: ignore[arg-type]
    ],
)
def test_is_gpt_oss(model: str, expected: bool) -> None:
    assert _is_gpt_oss(model) == expected


# ── _is_gemini_3 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3-flash", True),
        ("gemini-3-pro", True),
        ("gemini-3", True),
        ("GEMINI-3-FLASH", True),
        ("  gemini-3-flash  ", True),
        ("gemini-2.5-flash", False),
        ("gemini-2.0-flash", False),
        ("gpt-4", False),
        ("", False),
        (None, False),  # type: ignore[arg-type]
    ],
)
def test_is_gemini_3(model: str, expected: bool) -> None:
    assert _is_gemini_3(model) == expected


# ── _ollama_think_value ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "thinking_mode", "reasoning_effort", "expected"),
    [
        # Non-GPT-OSS: boolean behavior
        ("deepseek-v4", True, None, True),
        ("deepseek-v4", False, None, False),
        ("deepseek-v4", None, None, None),
        # GPT-OSS: string levels
        ("gpt-oss", True, None, "medium"),
        ("gpt-oss", True, "high", "high"),
        ("gpt-oss", True, "max", "high"),
        ("gpt-oss", True, "low", "low"),
        ("gpt-oss", True, "medium", "medium"),
        ("gpt-oss", False, None, "low"),
        ("gpt-oss", False, "high", "low"),
        ("GPT-OSS", True, "high", "high"),
    ],
)
def test_ollama_think_value(
    model: str,
    thinking_mode: bool | None,
    reasoning_effort: str | None,
    expected: bool | str | None,
) -> None:
    assert _ollama_think_value(model, thinking_mode, reasoning_effort) == expected


# ── except-clause `requests.*` attribute references resolve ────────────────


def _attribute_chain(node: ast.expr) -> list[str] | None:
    """Return e.g. ['requests', 'exceptions', 'ChunkedEncodingError'] or None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def test_except_clauses_reference_real_requests_attributes() -> None:
    """Guard against typos like `requests.ChunkedEncodingError` (must be
    `requests.exceptions.ChunkedEncodingError`), which raise AttributeError
    at the moment an exception is actually raised, silently swallowing the
    real error and skipping the intended except branch."""
    source = inspect.getsource(providers_module)
    tree = ast.parse(source)

    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        type_nodes = (
            node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        )
        for type_node in type_nodes:
            chain = _attribute_chain(type_node)
            if chain is None or chain[0] != "requests":
                continue
            checked += 1
            obj = requests
            for attr in chain[1:]:
                assert hasattr(obj, attr), (
                    f"except clause references requests.{'.'.join(chain[1:])!s} "
                    f"but '{attr}' does not exist on {obj!r} "
                    f"(line {type_node.lineno})"
                )
                obj = getattr(obj, attr)
            assert isinstance(obj, type) and issubclass(obj, BaseException)

    assert checked > 0, "expected at least one requests.* except clause to check"


# ── _count_delimiters ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("{}", {"open_curly": 1, "close_curly": 1, "open_square": 0, "close_square": 0}),
        ("[]", {"open_curly": 0, "close_curly": 0, "open_square": 1, "close_square": 1}),
        # truncated / unclosed
        ("{", {"open_curly": 1, "close_curly": 0, "open_square": 0, "close_square": 0}),
        ("[1, 2,", {"open_curly": 0, "close_curly": 0, "open_square": 1, "close_square": 0}),
        # nested balanced
        ('{"a": [1, {"b": 2}]}', {"open_curly": 2, "close_curly": 2, "open_square": 1, "close_square": 1}),
        # empty input
        ("", {"open_curly": 0, "close_curly": 0, "open_square": 0, "close_square": 0}),
    ],
)
def test_count_delimiters_basic(text, expected):
    assert _count_delimiters(text) == expected


def test_count_delimiters_ignores_delimiters_inside_double_quoted_strings():
    # delimiters inside a string literal must NOT count
    text = '{"msg": "has } and { and [ ] inside"}'
    assert _count_delimiters(text) == {
        "open_curly": 1, "close_curly": 1, "open_square": 0, "close_square": 0,
    }


def test_count_delimiters_ignores_delimiters_inside_single_quoted_strings():
    text = "{'key': 'value with } and [ braces'}"
    d = _count_delimiters(text)
    assert d["open_curly"] == 1 and d["close_curly"] == 1
    assert d["open_square"] == 0 and d["close_square"] == 0


def test_count_delimiters_escaped_quote_does_not_terminate_string():
    # An escaped quote stays part of the string; a brace after it is ignored.
    text = r'{"msg": "a\"}{b"}'
    d = _count_delimiters(text)
    assert d == {"open_curly": 1, "close_curly": 1, "open_square": 0, "close_square": 0}


def test_count_delimiters_truncation_proxy_balanced_vs_truncated():
    # The helper's consumers detect truncation via open_count > close_count.
    assert _count_delimiters('{"partial":')['open_curly'] > 0
    assert _count_delimiters("[1,2,3")["open_square"] > 0
    bal = _count_delimiters('{"ok": true}')
    assert bal["open_curly"] == bal["close_curly"]


def test_count_delimiters_ssot_no_inline_copy_remains():
    """Pin: the string-aware delimiter counter is the single SSOT. No inline
    state-machine twin (the old _open_cb/_in_str/_str_char/_esc pattern) may
    remain in providers.py — guards against a copy-paste duplicate regrowing.
    """
    import inspect

    from external_llm import providers

    src = inspect.getsource(providers)
    for stale in ("_open_cb", "_close_cb", "_in_str", "_str_char", "_open_sb", "_close_sb", "_esc"):
        assert stale not in src, f"inline delimiter-counter variable {stale!r} regressed into providers.py"
    # The helper is wired (def + at least the two DeepSeek call sites), not orphaned.
    assert src.count("_count_delimiters") >= 3


def test_count_delimiters_contract_shape():
    """Pin: returns exactly the 4 documented keys (callers index by name)."""
    assert set(_count_delimiters("{}[]{}").keys()) == {
        "open_curly", "close_curly", "open_square", "close_square",
    }


def test_count_delimiters_parity_openai_streaming_uses_helper():
    """Pin: OpenAI streaming clients use the shared helper (truncation-detection
    parity with DeepSeek). Guards against the detection being removed/short-circuited
    in openai_client.py while DeepSeek keeps it — the two SSE paths must detect
    silent truncation symmetrically. Both ``_chat_streaming`` (content) and
    ``_chat_with_tools_streaming`` (content + tool-call args) reference it.
    """
    import inspect

    from external_llm import openai_client

    src = inspect.getsource(openai_client)
    # At least two call sites (content in _chat_streaming, content+args in the
    # tools path) — the lazy import + .count of 'truncated' rewrite confirm wiring.
    assert src.count("_count_delimiters") >= 2, "OpenAI streaming must use the shared helper"
    assert src.count('finish_reason = "truncated"') >= 3, (
        "OpenAI streaming must emit finish_reason='truncated' for content-curly, "
        "content-square, and tool-call-args paths (parity with DeepSeek)"
    )
