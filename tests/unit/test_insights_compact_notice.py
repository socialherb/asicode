from external_llm.agent.context_manager import _compress_failure_notice, _compress_fail_latch, _compress_fail_latch_lock
from external_llm.client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
)


class TestCompressFailureNoticeUnknownError:
    """Verify that _compress_failure_notice produces a message for unknown
    error types when use_latch=False (interactive /insights compact path).

    Background path (use_latch=True) stays silent for unknown errors —
    that's intentional (logged at debug by caller). The interactive path
    needs to surface the error so the user can diagnose the problem.
    """

    def _reset_latch(self):
        """Clear the latch between tests."""
        with _compress_fail_latch_lock:
            _compress_fail_latch.clear()

    def setup_method(self):
        self._reset_latch()

    def test_unknown_error_returns_message_with_use_latch_false(self):
        """LLMConnectionError (not auth/quota/rate) must produce a message
        with exception details when use_latch=False."""
        exc = LLMConnectionError("Connection refused")
        msg = _compress_failure_notice("test", "deepseek-chat", exc, use_latch=False)
        assert msg is not None, "interactive path must produce a message for unknown errors"
        assert "LLMConnectionError" in msg, "message must include the exception class"
        assert "Connection refused" in msg, "message must include the exception text"
        assert "deepseek-chat" in msg, "message must include the model name"

    def test_unknown_error_returns_none_with_use_latch_true(self):
        """Background path (default) must stay silent for unknown errors."""
        exc = LLMConnectionError("Connection refused")
        msg = _compress_failure_notice("test", "deepseek-chat", exc)
        assert msg is None, "background path must stay silent for unknown errors"

    def test_latch_bypass_use_latch_false(self):
        """When use_latch=False, the same error class should report every time
        (no once-latch suppression)."""
        exc = LLMAuthenticationError("bad key")
        msg1 = _compress_failure_notice("s1", "m", exc, use_latch=False)
        msg2 = _compress_failure_notice("s1", "m", exc, use_latch=False)
        assert msg1 is not None, "first call must produce a message"
        assert msg2 is not None, "second call must also produce (latch bypassed)"

    def test_latch_respected_use_latch_true(self):
        """When use_latch=True (background), the once-latch must suppress
        repeated notifications for the same failure class."""
        exc = LLMAuthenticationError("bad key")
        msg1 = _compress_failure_notice("s1", "m", exc, use_latch=True)
        msg2 = _compress_failure_notice("s1", "m", exc, use_latch=True)
        assert msg1 is not None, "first call must produce a message"
        assert msg2 is None, "second call must be suppressed by latch"

    def test_different_classes_not_suppressed_even_with_latch(self):
        """A switch from auth → rate must get one notice each, even with latch."""
        exc1 = LLMAuthenticationError("bad key")
        exc2 = LLMRateLimitError("too fast")
        msg1 = _compress_failure_notice("s1", "m", exc1, use_latch=True)
        msg2 = _compress_failure_notice("s1", "m", exc2, use_latch=True)
        assert msg1 is not None, "auth must be reported"
        assert msg2 is not None, "rate must also be reported (different class)"

    def test_generic_exception_with_use_latch_false(self):
        """A non-LLM exception (e.g. JSON decode) must also produce a message."""
        exc = ValueError("Invalid JSON response")
        msg = _compress_failure_notice("test", "model", exc, use_latch=False)
        assert msg is not None, "interactive path must handle generic exceptions"
        assert "ValueError" in msg
        assert "Invalid JSON response" in msg
