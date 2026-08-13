"""Tests for _handle_loop_error error_type classification.

Replaces substring matching (`"connection" in str(error).lower()` /
`"rate" in str(error).lower() and "limit" in str(error).lower()`) with
typed isinstance checks against the LLM client error hierarchy.

Also pins the B2 outcome-channel contract: every path through
_handle_loop_error records the failed turn on the agent_result channel
(instance + global collectors), so loops that die with ``status="error"``
do not vanish from ``failure_rate`` / the recent-outcome window.
"""
from unittest.mock import MagicMock, patch

from external_llm.client import (
    LLMClientError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMServerUnavailableError,
)


def _drive_loop_error(error: Exception, cb=None):
    """Drive _handle_loop_error on a bare AgentLoop.

    Returns ``(loop, global_collector_mock)``. The module-level
    ``get_global_collector`` is patched so test recordings never touch the
    real global singleton (which would leak turns across tests).
    """
    from external_llm.agent import agent_turn_pipeline
    from external_llm.agent.agent_loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    loop._cb = cb if cb is not None else (lambda *a, **k: None)
    loop.registry = MagicMock()
    loop.registry.applied_patches = []
    loop.performance_collector = MagicMock()
    loop.performance_collector.get_summary.return_value = {}
    with patch.object(agent_turn_pipeline, "get_global_collector") as _gc:
        loop._handle_loop_error(
            error=error,
            turns=[],
            git_state=None,
            rollback_performed=False,
            rollback_result=None,
        )
    return loop, _gc


def _captured_error_type(error: Exception) -> str:
    """Drive _handle_loop_error and return the error_type sent over SSE."""
    captured = {}

    def _cb(event_name, payload):
        if event_name == "error":
            captured["error_type"] = payload.get("error_type")

    _drive_loop_error(error, cb=_cb)
    return captured.get("error_type")


class TestTypedClassification:
    def test_connection_error(self):
        assert _captured_error_type(LLMConnectionError("anything")) == "connection"

    def test_rate_limit_error(self):
        assert _captured_error_type(LLMRateLimitError("anything")) == "rate_limit"

    def test_server_unavailable_error(self):
        assert _captured_error_type(LLMServerUnavailableError("anything")) == "server_unavailable"

    def test_generic_llm_client_error_falls_to_api(self):
        # Any LLMClientError subclass without a more-specific category → "api"
        assert _captured_error_type(LLMClientError("anything")) == "api"

    def test_arbitrary_exception_falls_to_api(self):
        assert _captured_error_type(ValueError("anything")) == "api"


class TestNoSubstringConfusion:
    """Regression: error messages that *look* like a connection/rate-limit
    issue but are actually a different exception class must not be
    misclassified as connection/rate_limit."""

    def test_value_error_with_connection_in_message_is_api(self):
        # Pre-migration: "connection refused" substring → "connection"
        # Post-migration: ValueError isinstance → "api"
        err = ValueError("Database connection refused at startup")
        assert _captured_error_type(err) == "api"

    def test_runtime_error_with_rate_limit_in_message_is_api(self):
        err = RuntimeError("rate limit reached on internal scheduler")
        assert _captured_error_type(err) == "api"

    def test_connection_error_message_can_be_anything(self):
        # Pre-migration relied on 'connection' word in message — now it's
        # the type that matters, so any message works.
        err = LLMConnectionError("network unreachable")
        assert _captured_error_type(err) == "connection"

    def test_rate_limit_error_message_can_be_anything(self):
        # Many providers phrase rate limits as "Quota exceeded" — substring
        # matching would miss this, typed isinstance does not.
        err = LLMRateLimitError("Quota exceeded for model")
        assert _captured_error_type(err) == "rate_limit"


class TestOutcomeChannelRecording:
    """B2 regression: _handle_loop_error records the failed turn on both the
    instance and the global collector — otherwise a loop that dies with
    status="error" vanishes from failure_rate / the recent-outcome window."""

    def test_unexpected_exception_records_failed_on_both_collectors(self):
        loop, _gc = _drive_loop_error(error=RuntimeError("boom"))

        loop.performance_collector.record_agent_result.assert_called_once_with(failed=True)
        _gc.return_value.record_agent_result.assert_called_once_with(failed=True)

    def test_llm_client_error_escaping_inner_except_records_failed(self):
        # Generic LLMClientError is NOT among the 5 typed errors the turn body
        # catches — it escapes to _handle_loop_error, which must still record.
        loop, _gc = _drive_loop_error(error=LLMClientError("generic client failure"))

        loop.performance_collector.record_agent_result.assert_called_once_with(failed=True)
        _gc.return_value.record_agent_result.assert_called_once_with(failed=True)
