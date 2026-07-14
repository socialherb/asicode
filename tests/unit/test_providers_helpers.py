"""Tests for helper functions in external_llm/providers.py."""
from __future__ import annotations

import ast
import inspect

import pytest
import requests

import external_llm.providers as providers_module
from external_llm.providers import (
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
