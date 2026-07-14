"""Tests for background-compress failure notice + once-latch + full-suppress filter.

Regression guard for the "ERROR DeepSeek authentication failed (401)" per-turn
spam bug: background compress is best-effort, so a persistent helper-model
auth/quota problem must (a) NOT surface the provider's raw logger.error every
turn, and (b) notify the user ONCE per failure class per session.
"""
import logging
from unittest import mock

import pytest

from external_llm.agent import context_manager as cm
from external_llm.client import (
    LLMAuthenticationError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMAPIError,
)


@pytest.fixture(autouse=True)
def _reset_latch():
    cm._compress_fail_latch.clear()
    yield
    cm._compress_fail_latch.clear()


class TestCompressFailureNotice:
    def test_auth_failure_returns_user_friendly_notice(self):
        msg = cm._compress_failure_notice("s1", "deepseek-v4-flash", LLMAuthenticationError("bad key"))
        assert msg is not None
        assert "deepseek-v4-flash" in msg
        assert "authentication" in msg.lower()
        assert "/helper" in msg

    def test_quota_failure_returns_user_friendly_notice(self):
        msg = cm._compress_failure_notice("s1", "glm-5.2", LLMQuotaExceededError("no credits"))
        assert msg is not None
        assert "glm-5.2" in msg
        assert "quota" in msg.lower()

    def test_rate_limit_returns_retry_notice(self):
        msg = cm._compress_failure_notice("s1", "glm-5.2", LLMRateLimitError("slow down"))
        assert msg is not None
        assert "rate" in msg.lower() or "retry" in msg.lower()

    def test_generic_error_returns_none(self):
        # Transient/generic errors stay silent (already at debug level)
        assert cm._compress_failure_notice("s1", "glm-5.2", LLMAPIError("oops")) is None
        assert cm._compress_failure_notice("s1", "glm-5.2", ValueError("oops")) is None

    def test_once_latch_same_class_same_session_returns_none(self):
        """Repeated auth errors for the same session must NOT re-notify."""
        first = cm._compress_failure_notice("s1", "deepseek-v4-flash", LLMAuthenticationError("bad key"))
        assert first is not None
        second = cm._compress_failure_notice("s1", "deepseek-v4-flash", LLMAuthenticationError("bad key"))
        assert second is None, "once-latch must suppress repeated same-class failures"

    def test_once_latch_different_class_still_notifies(self):
        """A switch from auth → quota should still get one notice."""
        first = cm._compress_failure_notice("s1", "m", LLMAuthenticationError("bad key"))
        assert first is not None
        second = cm._compress_failure_notice("s1", "m", LLMQuotaExceededError("no credits"))
        assert second is not None, "different failure class must still notify"

    def test_once_latch_is_per_session(self):
        """Different sessions each get their own first notice."""
        a = cm._compress_failure_notice("sA", "m", LLMAuthenticationError("bad key"))
        b = cm._compress_failure_notice("sB", "m", LLMAuthenticationError("bad key"))
        assert a is not None and b is not None

    def test_once_latch_rate_limit_also_latches(self):
        """Rate-limit is sticky within a session too (avoids per-turn spam)."""
        first = cm._compress_failure_notice("s1", "m", LLMRateLimitError("slow"))
        assert first is not None
        second = cm._compress_failure_notice("s1", "m", LLMRateLimitError("slow"))
        assert second is None


class TestSuppressInfoFilterFullSuppress:
    """The filter must suppress ALL levels (incl. ERROR) during background compress.

    Regression for the per-turn "ERROR DeepSeek authentication failed (401)" spam:
    the provider's logger.error was passing through because the old filter only
    blocked < WARNING. Background compress is best-effort, so the user-facing
    notice (via _compress_failure_notice) is the single source of truth — the
    raw provider error must NOT also leak to the terminal.
    """

    def test_suppresses_info(self):
        f = cm._SuppressInfoFilter()
        rec = logging.LogRecord("x", logging.INFO, "f", 1, "hi", (), None)
        assert f.filter(rec) is False

    def test_suppresses_warning(self):
        f = cm._SuppressInfoFilter()
        rec = logging.LogRecord("x", logging.WARNING, "f", 1, "hi", (), None)
        assert f.filter(rec) is False

    def test_suppresses_error(self):
        f = cm._SuppressInfoFilter()
        rec = logging.LogRecord("x", logging.ERROR, "f", 1, "hi", (), None)
        assert f.filter(rec) is False


class TestCompressOldTurnsRoutesNotice:
    """End-to-end: compress_old_turns must call notify() with the user-facing
    notice on auth failure (instead of relying on the provider's logger.error)."""

    def test_auth_failure_calls_notify_once(self, tmp_path):
        # Build a minimal session-like object with the attributes compress_old_turns reads.
        session = mock.MagicMock()
        session.session_id = "s1"
        session.compressed_summary = None
        session.compressed_up_to = 0
        session.archived_count = 0
        session.turns = [
            {"role": "user", "content": "hello", "preserve": False, "tool_results": None},
            {"role": "assistant", "content": "hi", "preserve": False, "tool_results": None},
            {"role": "user", "content": "hello2", "preserve": False, "tool_results": None},
            {"role": "assistant", "content": "hi2", "preserve": False, "tool_results": None},
            {"role": "user", "content": "hello3", "preserve": False, "tool_results": None},
            {"role": "assistant", "content": "hi3", "preserve": False, "tool_results": None},
            {"role": "user", "content": "hello4", "preserve": False, "tool_results": None},
            {"role": "assistant", "content": "hi4", "preserve": False, "tool_results": None},
        ]
        # Force the compress path: needs_compression True, compressible turns exist.
        mgr = mock.MagicMock()
        mgr.needs_compression.return_value = True
        # Use the real method but with a client that raises auth error.
        llm_client = mock.MagicMock()
        llm_client.chat.side_effect = LLMAuthenticationError("bad key")

        # Call the real compress_old_turns via the concrete class that has it.
        # SessionCompressionContext is the concrete subclass; instantiate minimally.
        from external_llm.agent.context_manager import SessionCompressionContext
        ctx = SessionCompressionContext.__new__(SessionCompressionContext)
        # compress_old_turns uses self._cfg.compression.MIN_RECENT_TURNS_KEEP
        ctx._cfg = mock.MagicMock()
        ctx._cfg.compression.MIN_RECENT_TURNS_KEEP = 4

        notifies = []
        ctx.compress_old_turns(
            session, llm_client, "deepseek-v4-flash",
            recent_keep=4, notify=notifies.append,
        )
        # First call: one user-facing notice routed via notify
        assert len(notifies) == 1
        assert "deepseek-v4-flash" in notifies[0]
        assert "authentication" in notifies[0].lower()

        # Second call (same session, same failure class): once-latch suppresses
        notifies2 = []
        ctx.compress_old_turns(
            session, llm_client, "deepseek-v4-flash",
            recent_keep=4, notify=notifies2.append,
        )
        assert notifies2 == [], "once-latch must suppress repeated same-class failures"


class TestCompressFailLatchBounded:
    """Regression guard: the per-session once-latch must be bounded.

    The latch is keyed by session_id and never cleared on session end, so a
    long-lived server accumulated one entry per distinct failed session
    forever. The cap evicts the oldest entry (a long-gone session's latch is
    purposeless — session_ids are unique and never reused).
    """

    def test_latch_evicts_oldest_over_cap(self, monkeypatch):
        # Lower the cap so the test is fast and the eviction is unambiguous.
        monkeypatch.setattr(cm, "_COMPRESS_FAIL_LATCH_MAX", 4)
        cm._compress_fail_latch.clear()

        # First failure for session "old" — latched, notified.
        assert cm._compress_failure_notice("old", "m", LLMAuthenticationError("x")) is not None
        # Fill past the cap with distinct sessions (each adds a new latch entry).
        for i in range(4):
            cm._compress_failure_notice(f"s{i}", "m", LLMAuthenticationError("x"))
        # "old" was evicted (oldest-first): the dict is at the cap.
        assert len(cm._compress_fail_latch) == cm._COMPRESS_FAIL_LATCH_MAX
        assert "old" not in cm._compress_fail_latch

    def test_evicted_session_re_notifies(self, monkeypatch):
        """An evicted session's next failure re-notifies (latch no longer held).

        This is correct, not a bug: the session was evicted only because it is
        long-gone. If it somehow recurs, a single re-notice is acceptable.
        """
        monkeypatch.setattr(cm, "_COMPRESS_FAIL_LATCH_MAX", 2)
        cm._compress_fail_latch.clear()

        cm._compress_failure_notice("a", "m", LLMAuthenticationError("x"))  # latched
        # Two more distinct sessions push "a" out.
        cm._compress_failure_notice("b", "m", LLMAuthenticationError("x"))
        cm._compress_failure_notice("c", "m", LLMAuthenticationError("x"))
        assert "a" not in cm._compress_fail_latch
        # "a" reappears: not latched anymore → re-notified.
        again = cm._compress_failure_notice("a", "m", LLMAuthenticationError("x"))
        assert again is not None

    def test_class_change_does_not_grow_beyond_cap(self, monkeypatch):
        """Re-setting an existing session_id (class change) must not overflow."""
        monkeypatch.setattr(cm, "_COMPRESS_FAIL_LATCH_MAX", 3)
        cm._compress_fail_latch.clear()

        cm._compress_failure_notice("a", "m", LLMAuthenticationError("x"))
        cm._compress_failure_notice("b", "m", LLMAuthenticationError("x"))
        cm._compress_failure_notice("c", "m", LLMAuthenticationError("x"))
        assert len(cm._compress_fail_latch) == 3
        # Same session "a", different class — updates in place, no growth.
        cm._compress_failure_notice("a", "m", LLMQuotaExceededError("x"))
        assert len(cm._compress_fail_latch) == 3
        assert cm._compress_fail_latch["a"] == "quota"
