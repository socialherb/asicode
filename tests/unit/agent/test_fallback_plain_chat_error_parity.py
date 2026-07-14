"""Defense-depth parity (insight A38): _fallback_plain_chat must re-raise ALL
service-side LLM errors (LLMClientError subclasses), not just a stale 2-name
tuple of (LLMRateLimitError, LLMAuthenticationError).

When LLMQuotaExceededError was introduced (zai balance-exhaustion), this
fallback path was not updated, so a quota error raised during the fallback
chat got swallowed into a generic "An error occurred during LLM call: ..."
message — indistinguishable from a bug, and never reaching the canonical
error→message mapping (_user_facing_llm_error) or the error_type="quota"
classification in _respond_impl.

This test pins the invariant: every LLMClientError subclass propagates
unchanged; only non-LLMClientError (unexpected) exceptions are swallowed.
"""
from __future__ import annotations

from unittest import mock

import pytest

from external_llm.agent.design_chat_loop import _fallback_plain_chat
from external_llm.client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerUnavailableError,
)

# Every concrete service-side error that must propagate through the fallback.
_SERVICE_ERRORS = [
    LLMQuotaExceededError("balance exhausted"),
    LLMServerUnavailableError("503"),
    LLMConnectionError("timeout"),
    LLMAPIError("bad request"),
    LLMAuthenticationError("invalid key"),
    LLMRateLimitError("slow down"),
]


@pytest.mark.parametrize("exc", _SERVICE_ERRORS, ids=lambda e: type(e).__name__)
def test_fallback_plain_chat_propagates_service_errors(exc: LLMClientError):
    """Every LLMClientError subclass must re-raise, not get swallowed."""
    client = mock.MagicMock()
    client.chat.side_effect = exc

    with pytest.raises(type(exc)):
        _fallback_plain_chat(
            [LLMMessage(role="user", content="hi")], client, "test-model"
        )


def test_fallback_plain_chat_still_swallows_unexpected_exceptions():
    """Non-LLMClientError (unexpected) exceptions must still produce a fallback
    error dict — that is the whole point of the fallback (recover from bugs in
    tool-call parsing/serialization by trying a plain chat)."""
    client = mock.MagicMock()
    client.chat.side_effect = RuntimeError("unexpected parse bug")

    result = _fallback_plain_chat(
        [LLMMessage(role="user", content="hi")], client, "test-model"
    )

    assert result["error"] is True
    assert "unexpected parse bug" in result["content"]


def test_fallback_plain_chat_quota_error_regression():
    """Direct regression for the original bug: a quota error must NOT become a
    generic 'An error occurred during LLM call' dict (swallowed). Before the
    parity fix, LLMQuotaExceededError fell through to the generic branch."""
    client = mock.MagicMock()
    client.chat.side_effect = LLMQuotaExceededError("insufficient balance")

    with pytest.raises(LLMQuotaExceededError):
        _fallback_plain_chat(
            [LLMMessage(role="user", content="hi")], client, "test-model"
        )
