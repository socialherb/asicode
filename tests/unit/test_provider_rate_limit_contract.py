"""Every ``raise LLMRateLimitError(...)`` site must pass the parsed
``retry_after=`` hint to the exception.

Regression guard for P0-1: the two Ollama 429 sites in providers.py computed
``retry_after = parse_retry_after(response.headers)`` and then only used it
inside the message f-string — the retry layers above (agent_loop's
``_retry_on_rate_limit``, design_chat's ``_call_llm_with_retry``) read
``e.retry_after`` via getattr to honor the server hint instead of a fixed
backoff, so the hint was silently dropped on exactly the provider that needs
it most (local/slow Ollama endpoints). The AST contract below pins the
kwarg on every raise site, including ones added in the future.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _rate_limit_raise_sites():
    """Yield (rel_path, lineno, call) for every ``raise LLMRateLimitError(...)``."""
    for file in sorted((_REPO_ROOT / "external_llm").rglob("*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            func = node.exc.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if name != "LLMRateLimitError":
                continue
            yield file.relative_to(_REPO_ROOT).as_posix(), node.lineno, node.exc


def test_every_rate_limit_raise_passes_retry_after_hint():
    """A computed-and-dropped server hint is a silent bug class; the kwarg
    must be present (``None`` explicitly) on every raise site."""
    missing = [
        f"{rel}:{lineno} — raise LLMRateLimitError(...) without retry_after="
        for rel, lineno, call in _rate_limit_raise_sites()
        if not any(kw.arg == "retry_after" for kw in call.keywords)
    ]
    assert not missing, (
        "LLMRateLimitError raise sites must pass retry_after= (parsed server "
        "hint, or None) so retry layers can honor it instead of fixed backoff:\n" + "\n".join(missing)
    )


def test_retry_after_hint_is_derived_from_the_response_header():
    """The kwarg must carry the parsed header value — not a hardcoded literal
    (which would defeat the purpose of honoring the server's hint)."""
    hardcoded = []
    for rel, lineno, call in _rate_limit_raise_sites():
        for kw in call.keywords:
            if kw.arg != "retry_after":
                continue
            value = kw.value
            if isinstance(value, ast.Constant) and value.value is None:
                continue  # explicit "no hint available" is allowed
            if isinstance(value, ast.Name):
                continue  # local alias (e.g. `retry_after = parse_retry_after(...)`)
            if not (isinstance(value, ast.Call) and getattr(value.func, "id", None) == "parse_retry_after"):
                hardcoded.append(f"{rel}:{lineno}")
    assert not hardcoded, (
        f"retry_after= must be parse_retry_after(response.headers) or None, got non-derived values at: {hardcoded}"
    )
