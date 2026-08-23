"""P21-2: internal call sites must not pass kwargs the definitions dropped.

The 6-week max_chars TypeError in external_llm/service.py lived because the
only generate_patch test stubbed the method with a permissive lambda(*a, **kw)
and no check compared call-site kwargs against the definition signature.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import external_llm.service as service_mod
from external_llm.service import ExternalLLMService

_SERVICE_SRC = pathlib.Path(service_mod.__file__).read_text(encoding="utf-8")
_SERVICE_TREE = ast.parse(_SERVICE_SRC)

_METHODS = [
    "_read_target_file_focused_snippet_best_effort",
    "_read_target_file_snippet_best_effort",
    "_noop_precheck_for_literal_add",
    "_build_llm_context_v7_best_effort",
    "_build_llm_context_super_best_effort",
]


def _internal_kwargs(method: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(_SERVICE_TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            out |= {kw.arg for kw in node.keywords if kw.arg is not None}
    return out


@pytest.mark.parametrize("method", _METHODS)
def test_internal_call_kwargs_exist_in_signature(method):
    sig = inspect.signature(getattr(ExternalLLMService, method))
    missing = _internal_kwargs(method) - set(sig.parameters)
    assert not missing, f"{method} call site passes kwargs missing from the definition: {sorted(missing)}"


def test_focused_snippet_max_chars_contract():
    """The P21 fix dropped max_chars=6_000 from the call site — keep it gone."""
    sig = inspect.signature(ExternalLLMService._read_target_file_focused_snippet_best_effort)
    assert "max_chars" not in sig.parameters
    assert "max_chars" not in _internal_kwargs("_read_target_file_focused_snippet_best_effort")


def test_validate_diff_compat_shim_is_gone():
    """P27-2 removed _validate_diff_best_effort — the TypeError-swallowing
    compat shim whose `except TypeError` retried WITHOUT target_file (silently
    dropping the auto-mode filter) and whose tail branches were unreachable
    (validate_diff always returns tuple[bool, str]). The call site now invokes
    validate_diff directly with the single real signature."""
    assert not hasattr(ExternalLLMService, "_validate_diff_best_effort")
    sig = inspect.signature(service_mod.validate_diff)
    assert "target_file" in sig.parameters


def test_normalize_candidate_patch_reports_precise_reason(tmp_path):
    """P27-2b: the normalization reason must flow out instead of being
    discarded — a non-diff patch reports WHY, so the caller can build an
    invalid_diff:reason instead of a generic validate_failed."""
    normalized, error = ExternalLLMService._normalize_candidate_patch("hello world", None, repo_root=str(tmp_path))
    assert normalized
    assert "unified diff" in (error or "")


def test_normalize_candidate_patch_ok_is_none_error(tmp_path):
    (tmp_path / "x.txt").write_text("old\n")
    diff = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-old\n+new\n"
    normalized, error = ExternalLLMService._normalize_candidate_patch(diff, "x.txt", repo_root=str(tmp_path))
    assert error is None
    assert "x.txt" in normalized
