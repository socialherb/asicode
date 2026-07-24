"""Regression tests for final-response LLM call metrics and error handling.

Verifies fixes for three related bugs:

1. **Double recording** — success-path ``record_llm_call(..., failed=False)`` and
   the outer ``except Exception`` failure-path both fired for a single call when
   an exception occurred after the success recording.
2. **Retry undercount** — empty-response retry path omitted
   ``result.total_llm_calls += 1``, so the counter underreported real LLM calls.
3. **Cancellation swallowed** — the two ``except Exception`` handlers in the
   final-response block lacked the ``isinstance(e, (LLMClientError, AgentCancelled)): raise``
   guard that the tool-loop block (L1626) has, allowing service-side errors and
   user cancellation to be caught and treated as "final response generation failed".

Testing strategy: every stub's ``chat_with_tools`` returns a tool-call request so
the tool loop runs until max_iterations, then ``respond()`` enters the final-response
block where ``chat()`` is tested.
"""
from __future__ import annotations

import pytest

from external_llm.agent.design_chat_loop import DesignChatLoop
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.client import (
    LLMClientError,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolCallResponse,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

_MAX_TOOL_ITERS = 1  # minimum — tool loop runs once, then final response


def _tool_calling_response(model: str) -> ToolCallResponse:
    """Return a ToolCallResponse that requests a single tool call."""
    return ToolCallResponse(
        content="", model=model, provider="test_stub", tokens_used=5,
        finish_reason="tool_calls", raw_response=None,
        tool_calls=[ToolCallRequest(
            call_id="c1", name="read_file",
            args={"file_path": "README.md"},
        )],
    )


@pytest.fixture
def _repo(tmp_path):
    """Set up a minimal git repo for ToolRegistry."""
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("hello\n")
    return str(tmp_path)


def _make_loop(client, repo_root: str) -> DesignChatLoop:
    """Build a DesignChatLoop with a stub client and minimal config."""
    config = AgentConfig(model_name="test-model")
    registry = ToolRegistry(repo_root, config, [])
    loop = DesignChatLoop(client, registry, config.model_name)
    return loop


def _respond(loop: DesignChatLoop):
    """Shorthand: call respond() with minimal args."""
    return loop.respond(
        messages=[LLMMessage(role="user", content="hello")],
        max_tool_iterations=_MAX_TOOL_ITERS,
        stream_callback=None, reasoning_callback=None, token_callback=None,
    )


# ── Stub implementations ──────────────────────────────────────────────────────

class _AlwaysToolCallStub:
    """Base: chat_with_tools always requests a tool call (drives max-iter path)."""

    @staticmethod
    def get_provider_name() -> str:
        return "test_stub"

    def chat(self, messages, model, **kw):
        raise NotImplementedError("subclass must implement chat()")

    def chat_with_tools(self, messages, tools, model, **kw):
        return _tool_calling_response(model)


class _EmptyFirstStub(_AlwaysToolCallStub):
    """chat() returns empty on first call, content on retry."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, model, **kw):
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                content="", model=model, provider="test_stub",
                tokens_used=10, prompt_tokens=5, completion_tokens=5,
            )
        return LLMResponse(
            content="Here is the final answer.", model=model,
            provider="test_stub", tokens_used=20, prompt_tokens=10,
            completion_tokens=10,
        )


class _AlwaysEmptyStub(_AlwaysToolCallStub):
    """chat() always returns empty (triggers retry, retry also empty)."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, model, **kw):
        self.call_count += 1
        return LLMResponse(
            content="", model=model, provider="test_stub",
            tokens_used=10, prompt_tokens=5, completion_tokens=5,
        )


class _OkStub(_AlwaysToolCallStub):
    """chat() returns a valid final response on first call (no retry)."""

    def chat(self, messages, model, **kw):
        return LLMResponse(
            content="Final answer.", model=model, provider="test_stub",
            tokens_used=5, prompt_tokens=2, completion_tokens=3,
        )


class _FailingStub(_AlwaysToolCallStub):
    """chat() always raises LLMClientError."""

    def chat(self, messages, model, **kw):
        raise LLMClientError("Service unavailable")


class _CancellingStub(_AlwaysToolCallStub):
    """chat() always raises AgentCancelled.
    Import delayed to avoid module-level raise.
    """

    def chat(self, messages, model, **kw):
        from external_llm.agent.agent_loop_types import AgentCancelled
        raise AgentCancelled("cancelled by user")


class _FirstEmptyThenFailStub(_AlwaysToolCallStub):
    """chat(): first call empty, retry raises LLMClientError."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, model, **kw):
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                content="", model=model, provider="test_stub", tokens_used=5,
            )
        raise LLMClientError("Retry also failed")


class _FirstEmptyThenCancelStub(_AlwaysToolCallStub):
    """chat(): first call empty, retry raises AgentCancelled."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, model, **kw):
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                content="", model=model, provider="test_stub", tokens_used=5,
            )
        from external_llm.agent.agent_loop_types import AgentCancelled
        raise AgentCancelled("cancelled during retry")


# ── Test classes ──────────────────────────────────────────────────────────────

class TestTotalLlmCallsRetry:
    """Bug 2: retry path must increment total_llm_calls."""

    def test_retry_increments_total_llm_calls(self, _repo):
        """Empty first response triggers retry; both calls must be counted."""
        loop = _make_loop(_EmptyFirstStub(), _repo)
        result = _respond(loop)
        # 1 tool-loop call + 1 initial final-attempt + 1 retry = 3
        assert result.total_llm_calls == 3, (
            f"Expected 3 LLM calls (tool-loop + initial + retry), "
            f"got {result.total_llm_calls}"
        )
        assert result.content.strip() == "Here is the final answer."

    def test_retry_failure_still_counts(self, _repo):
        """Retry that fails still counts toward total_llm_calls."""
        loop = _make_loop(_AlwaysEmptyStub(), _repo)
        result = _respond(loop)
        # 1 tool-loop + 1 initial + 1 retry = 3
        assert result.total_llm_calls == 3, (
            f"Expected 3 LLM calls (tool-loop + initial + retry), "
            f"got {result.total_llm_calls}"
        )


class TestFailureRecording:
    """After Bug 3 fix regression: provider failures must be recorded before raise."""

    def test_llm_client_error_records_and_propagates(self, _repo):
        """LLMClientError: failed=True recorded in metrics, result.is_error=True."""
        from external_llm.agent.performance_metrics import get_global_collector
        calls = []
        original_record = get_global_collector().record_llm_call
        try:
            def _spy(*args, **kw):
                calls.append(kw)
                return original_record(*args, **kw)
            get_global_collector().record_llm_call = _spy
            loop = _make_loop(_FailingStub(), _repo)
            result = _respond(loop)
            assert result.is_error is True, "LLMClientError must produce is_error result"
            failures = [c for c in calls if c.get("failed", False)]
            assert len(failures) >= 1, (
                f"Expected ≥1 failure recording for LLMClientError, got {len(failures)}"
            )
        finally:
            get_global_collector().record_llm_call = original_record

    def test_ok_path_no_spurious_failure(self, _repo):
        """Normal (no retry) path: no failure recordings for final response."""
        from external_llm.agent.performance_metrics import get_global_collector
        calls = []
        original_record = get_global_collector().record_llm_call
        try:
            def _spy(*args, **kw):
                calls.append(kw)
                return original_record(*args, **kw)
            get_global_collector().record_llm_call = _spy
            loop = _make_loop(_OkStub(), _repo)
            _respond(loop)
            failures = [c for c in calls if c.get("failed", False)]
            assert len(failures) == 0, (
                f"Expected 0 failure recordings, got {len(failures)}: {failures}"
            )
        finally:
            get_global_collector().record_llm_call = original_record


class TestCancellationGuard:
    """Bug 3: LLMClientError and AgentCancelled must propagate through except blocks."""

    def test_cancelled_propagates_from_outer_except(self, _repo):
        """AgentCancelled in final-response chat() must propagate to respond()."""
        loop = _make_loop(_CancellingStub(), _repo)
        from external_llm.agent.agent_loop_types import AgentCancelled
        with pytest.raises(AgentCancelled):
            _respond(loop)

    def test_llm_client_error_propagates_from_retry_except(self, _repo):
        """LLMClientError in retry chat() must propagate via inner except guard."""
        loop = _make_loop(_FirstEmptyThenFailStub(), _repo)
        result = _respond(loop)
        assert result.is_error is True

    def test_cancelled_propagates_from_retry_except(self, _repo):
        """AgentCancelled in retry chat() must propagate via inner except guard."""
        loop = _make_loop(_FirstEmptyThenCancelStub(), _repo)
        from external_llm.agent.agent_loop_types import AgentCancelled
        with pytest.raises(AgentCancelled):
            _respond(loop)


class TestFinalRecordedGuard:
    """Bug 1: _final_recorded must prevent double-recording on outer except.

    When the success-path ``record_llm_call(..., failed=False)`` fires and a
    *subsequent* statement in the try block raises, the outer ``except`` must
    NOT record a second (redundant) ``failed=True`` call. This test exercises
    the exact path: first ``chat()`` returns empty → success recording fires →
    ``_final_recorded = True`` → retry raises LLMClientError → inner except
    records ``failed=True`` → re-raises → outer except catches with
    ``_final_recorded=True``.

    Without the guard (``if not _final_recorded:``), the outer except would
    append a second ``failed=True``, inflating both ``calls`` and
    ``recent_failure_rate``.
    """

    def test_guard_prevents_double_failure_recording(self, _repo):
        """Only 1 failed=True recording (from inner except), not 2 (outer)."""
        from external_llm.agent.performance_metrics import get_global_collector
        calls: list[dict] = []
        original_record = get_global_collector().record_llm_call
        try:
            def _spy(*args, **kw):
                calls.append(kw)
                return original_record(*args, **kw)
            get_global_collector().record_llm_call = _spy

            loop = _make_loop(_FirstEmptyThenFailStub(), _repo)
            result = _respond(loop)
            assert result.is_error is True

            failures = [c for c in calls if c.get("failed", False)]
            successes = [c for c in calls if not c.get("failed", True)]

            assert len(failures) == 1, (
                f"Expected exactly 1 failed=True recording (from inner except), "
                f"got {len(failures)}: {failures}"
            )
            assert len(successes) >= 1, (
                f"Expected at least 1 failed=False recording (tool-loop + final), "
                f"got {len(successes)}: {successes}"
            )
        finally:
            get_global_collector().record_llm_call = original_record
