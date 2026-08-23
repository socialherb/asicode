"""Unit tests for LocalAssistant and its pure collaborators
(external_llm/agent/local_assistant.py).

Strategy:
  * _extract_fenced_blocks / OutputCleaner / OutputValidator — pure, no mocks.
  * delegate_single_task / _execute_delegation — LocalAssistant constructed
    with a stub OllamaClient (no network) and Mock local client.

The execute() pipeline and _fallback_to_main_agent() were removed as dead
code (R7) — delegate_single_task is the only live entry point.
"""

from __future__ import annotations

from unittest.mock import Mock

import external_llm.providers as providers_mod
from external_llm.agent.local_assistant import (
    DelegationResult,
    DelegationSpec,
    LocalAssistant,
    OutputCleaner,
    OutputValidator,
    _extract_fenced_blocks,
)


class _Resp:
    """Minimal stand-in for an LLMResponse — effective_content reads .content."""

    def __init__(self, content: str):
        self.content = content


def _make_assistant(monkeypatch, *, local_client=None, max_local_calls=5):
    """Construct LocalAssistant without touching the network.

    Stub OllamaClient so __init__ never builds a real requests session, then
    always inject local_client (None for the no-client path, Mock otherwise).
    """
    monkeypatch.setattr(providers_mod, "OllamaClient", lambda **kw: object())
    la = LocalAssistant(
        local_model="l-model",
        repo_root="/tmp",
        max_local_calls=max_local_calls,
    )
    la._local_client = local_client
    return la


# ── _extract_fenced_blocks ──────────────────────────────────────────────────


class TestExtractFencedBlocks:
    def test_single_block(self):
        assert _extract_fenced_blocks("before\n```py\nx = 1\n```\nafter") == ["x = 1\n"]

    def test_multiple_blocks(self):
        out = _extract_fenced_blocks("```py\na\n```\ntext\n```js\nb\n```")
        assert out == ["a\n", "b\n"]

    def test_no_language_tag(self):
        assert _extract_fenced_blocks("```\ncode\n```") == ["code\n"]

    def test_no_fence_returns_empty(self):
        assert _extract_fenced_blocks("just plain text") == []


# ── OutputCleaner ───────────────────────────────────────────────────────────


class TestOutputCleaner:
    def setup_method(self):
        self.c = OutputCleaner()

    def test_fence_extraction(self):
        assert self.c.clean("Here:\n```py\nx = 1\n```") == "x = 1"

    def test_preamble_stripped(self):
        out = self.c.clean("Here is the code:\nx = 1\ny = 2")
        assert out.startswith("x = 1")

    def test_postamble_stripped(self):
        out = self.c.clean("x = 1\n\nNote: this code does X")
        assert out.strip() == "x = 1"

    def test_plain_code_unchanged(self):
        assert self.c.clean("x = 1\ny = 2") == "x = 1\ny = 2"

    def test_empty(self):
        assert self.c.clean("   ") == ""


# ── OutputValidator ─────────────────────────────────────────────────────────


class TestOutputValidator:
    def setup_method(self):
        self.v = OutputValidator()

    def _spec(self, **kw):
        base = {"role": "code_snippet", "instruction": "gen", "language": "python"}
        base.update(kw)
        return DelegationSpec(**base)

    def test_empty_output_fails(self):
        r = self.v.validate("   ", self._spec())
        assert r["overall_ok"] is False
        assert "Empty output" in r["issues"]

    def test_python_valid(self):
        r = self.v.validate("x = 1\n", self._spec())
        assert r["syntax_ok"] and r["overall_ok"]

    def test_python_syntax_error(self):
        r = self.v.validate("def (\n", self._spec(role="boilerplate"))
        assert r["syntax_ok"] is False
        assert any("syntax error" in i for i in r["issues"])

    def test_python_syntax_error_parses_once(self, monkeypatch):
        # Regression pin: the non-code_snippet/fim branch used to re-parse the
        # same output text a second time just to re-derive the SyntaxError
        # object. ast.parse must run exactly once on the failing output.
        import ast

        calls = []
        orig_parse = ast.parse

        def counting(*a, **k):
            calls.append(a)
            return orig_parse(*a, **k)

        monkeypatch.setattr(ast, "parse", counting)
        r = self.v.validate("def (\n", self._spec(role="boilerplate"))
        assert r["syntax_ok"] is False
        assert any("syntax error" in i for i in r["issues"])
        assert len(calls) == 1

    def test_python_code_snippet_body_wrap(self):
        # Bare function-body fragment (invalid as module) is re-validated wrapped
        r = self.v.validate("return x + 1\n", self._spec(role="code_snippet"))
        # Wrapped in def _tmp(): → valid → syntax_ok True
        assert r["syntax_ok"] is True

    def test_python_fim_body_wrap(self):
        r = self.v.validate("return 42\n", self._spec(role="fim"))
        assert r["syntax_ok"] is True

    def test_js_balanced(self):
        r = self.v.validate("function f() { return [1, 2]; }", self._spec(language="javascript"))
        assert r["syntax_ok"] and r["overall_ok"]

    def test_js_unbalanced(self):
        r = self.v.validate("function f() { return [1, 2; }", self._spec(language="javascript"))
        assert r["syntax_ok"] is False

    def test_js_unclosed(self):
        r = self.v.validate("const x = (1", self._spec(language="typescript"))
        assert r["syntax_ok"] is False

    def test_js_string_aware_brackets(self):
        # Brackets inside strings must not affect balance
        r = self.v.validate('const s = "[({";\n', self._spec(language="javascript"))
        assert r["syntax_ok"] is True

    def test_role_test_skeleton_ok(self):
        r = self.v.validate("def test_foo():\n    assert True\n", self._spec(role="test_skeleton"))
        assert r["pattern_match"] and r["overall_ok"]

    def test_role_test_skeleton_missing_test_fn(self):
        r = self.v.validate("x = 1\n", self._spec(role="test_skeleton"))
        assert r["pattern_match"] is False

    def test_role_docstring_ok(self):
        r = self.v.validate('"""A docstring."""', self._spec(role="docstring"))
        assert r["pattern_match"] and r["overall_ok"]

    def test_role_docstring_missing_quotes(self):
        r = self.v.validate("just text", self._spec(role="docstring"))
        assert r["pattern_match"] is False

    def test_hallucination_prefix_detected(self):
        r = self.v.validate("Here is the code:\nx = 1", self._spec())
        assert any("preamble" in i.lower() for i in r["issues"])

    def test_overall_ok_aggregates(self):
        # Syntax OK but role fails → overall False
        r = self.v.validate("x = 1\n", self._spec(role="test_skeleton"))
        assert r["overall_ok"] is False


# ── _execute_delegation ─────────────────────────────────────────────────────


class TestExecuteDelegation:
    def test_no_local_client_fails_gracefully(self, monkeypatch):
        la = _make_assistant(monkeypatch, local_client=None)
        spec = DelegationSpec(role="code_snippet", instruction="gen")
        dr = la._execute_delegation(spec)
        assert dr.accepted is False
        assert dr.validation["overall_ok"] is False
        assert "OllamaClient not available" in dr.validation["issues"]

    def test_happy_path_through_cleaner_and_validator(self, monkeypatch):
        local = Mock()
        local.chat.return_value = _Resp("```py\ndef foo():\n    return 1\n```")
        la = _make_assistant(monkeypatch, local_client=local)
        spec = DelegationSpec(role="code_snippet", instruction="gen foo")
        dr = la._execute_delegation(spec)
        assert dr.accepted is True
        assert "def foo():" in dr.cleaned_output
        assert dr.validation["overall_ok"] is True

    def test_chat_exception_caught(self, monkeypatch):
        local = Mock()
        local.chat.side_effect = RuntimeError("boom")
        la = _make_assistant(monkeypatch, local_client=local)
        spec = DelegationSpec(role="code_snippet", instruction="gen")
        dr = la._execute_delegation(spec)
        assert dr.accepted is False
        assert "boom" in dr.validation["issues"][0]


# ── delegate_single_task ────────────────────────────────────────────────────


class TestDelegateSingleTask:
    def test_success_response_shape(self, monkeypatch):
        local = Mock()
        local.chat.return_value = _Resp("x = 1\n")
        la = _make_assistant(monkeypatch, local_client=local)
        out = la.delegate_single_task(role="code_snippet", instruction="gen", language="python")
        assert out["success"] is True
        assert out["code"].strip() == "x = 1"
        assert out["role"] == "code_snippet"
        assert out["execution_time"] >= 0.0
        assert isinstance(out["validation"], dict)

    def test_failure_when_no_client(self, monkeypatch):
        la = _make_assistant(monkeypatch, local_client=None)
        out = la.delegate_single_task(role="code_snippet", instruction="gen")
        assert out["success"] is False
        assert out["code"] == ""
        assert out["issues"]  # non-empty

    def test_max_local_calls_limit_enforced(self, monkeypatch):
        """helper_max_calls → max_local_calls must cap delegations per session.

        Regression: max_local_calls was accepted but never stored or checked,
        so the webapp/tool-registry config had no effect at all.
        """
        local = Mock()
        local.chat.return_value = _Resp("x = 1\n")
        la = _make_assistant(monkeypatch, local_client=local, max_local_calls=2)
        first = la.delegate_single_task(role="code_snippet", instruction="gen", language="python")
        second = la.delegate_single_task(role="code_snippet", instruction="gen", language="python")
        third = la.delegate_single_task(role="code_snippet", instruction="gen", language="python")
        assert first["success"] is True
        assert second["success"] is True
        assert third["success"] is False
        assert any("limit" in issue.lower() for issue in third["issues"])
        assert "helper_max_calls" in third["error"]
        # The refused call must never reach the model.
        assert local.chat.call_count == 2

    def test_delegation_limit_is_per_instance(self, monkeypatch):
        """Each LocalAssistant session gets its own fresh budget."""
        local = Mock()
        local.chat.return_value = _Resp("x = 1\n")
        la1 = _make_assistant(monkeypatch, local_client=local, max_local_calls=1)
        la2 = _make_assistant(monkeypatch, local_client=local, max_local_calls=1)
        assert la1.delegate_single_task(role="code_snippet", instruction="gen")["success"] is True
        assert la1.delegate_single_task(role="code_snippet", instruction="gen")["success"] is False
        assert la2.delegate_single_task(role="code_snippet", instruction="gen")["success"] is True


# ── dataclasses ─────────────────────────────────────────────────────────────


class TestDataclasses:
    def test_delegation_spec_defaults(self):
        s = DelegationSpec(role="code_snippet", instruction="x")
        assert s.language == "python"
        assert s.function_signature == ""
        assert s.max_tokens > 0

    def test_delegation_result_defaults(self):
        r = DelegationResult(spec=DelegationSpec(role="x", instruction="y"))
        assert r.raw_output == "" and r.cleaned_output == ""
        assert r.accepted is False and r.execution_time == 0.0
        assert r.validation == {}
