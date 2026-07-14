"""Tests for ``_prompt_auth_retry_key`` — the auth-failure recovery hook.

Covers the regression where an opencode server returns HTTP 401 for an
*unsupported model name* (not a bad key). Re-entering the key never fixes
that — the function must detect the "not supported" signal in the error
body and steer the user to ``/model`` instead of prompting for a key,
breaking the infinite 401 loop.
"""
from __future__ import annotations

import asi


class _FakeSvc:
    """Minimal stand-in for ExternalLLMService — only ``.model`` is read."""

    def __init__(self, model: str = "qwen3.7-max suggest bug/feature/performance improvements"):
        self.model = model
        self.llm_service = type("S", (), {"client": None, "provider": "opencode"})()


class TestAuthRetryDetectsUnsupportedModel:
    """When the 401 body says "not supported", the cause is the model name,
    not the key — refuse to prompt for a key and return False."""

    def test_not_supported_short_circuits_before_key_prompt(self, monkeypatch):
        # If the guard works, input() must never be called. A bomb makes the
        # test fail loudly if the guard is bypassed.
        monkeypatch.setattr("builtins.input", lambda *_: (_ for _ in ()).throw(AssertionError("input() must not be called for unsupported-model 401")))
        svc = _FakeSvc()

        result = asi._prompt_auth_retry_key(
            "opencode", svc,
            error_message="⚠️ LLM API authentication failed.\n(server message: Model qwen3.7-max is not supported)",
        )

        # The guard must short-circuit (return False) WITHOUT calling input().
        assert result is False
        # Source-contract: the "not supported" branch must exist and steer to
        # /model. (_print routes through a Rich console bound at import time,
        # so capsys can't capture it — we assert the source instead.)
        import inspect
        src = inspect.getsource(asi._prompt_auth_retry_key)
        assert "not supported" in src
        assert "/model" in src

    def test_genuine_auth_failure_still_prompts_for_key(self, monkeypatch):
        # A real 401 (no "not supported" signal) must still offer the key
        # prompt — the guard must not over-trigger and block legitimate retries.
        monkeypatch.setattr("builtins.input", lambda *_: "sk-newkey")
        # Stub create_llm_client so no network call happens.
        import external_llm.client as _client
        monkeypatch.setattr(_client, "create_llm_client", lambda **kw: object())
        svc = _FakeSvc(model="deepseek-chat")

        result = asi._prompt_auth_retry_key(
            "deepseek", svc,
            error_message="⚠️ LLM API authentication failed.\n(server message: Invalid API key)",
        )

        assert result is True
        assert svc.llm_service.client is not None

    def test_empty_error_message_falls_through_to_key_prompt(self, monkeypatch):
        # No error_message supplied (legacy call sites) → behave like a real
        # auth failure and prompt for the key. Backward-compat guard.
        monkeypatch.setattr("builtins.input", lambda *_: "")  # skip
        svc = _FakeSvc(model="some-model")

        result = asi._prompt_auth_retry_key("deepseek", svc)

        assert result is False  # user skipped
