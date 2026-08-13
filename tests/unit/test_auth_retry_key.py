"""Tests for ``_prompt_auth_retry_key`` — the auth-failure recovery hook.

Covers the regression where an opencode server returns HTTP 401 for an
*unsupported model name* (not a bad key). Re-entering the key never fixes
that — the function must detect the "not supported" signal in the error
body and steer the user to ``/model`` instead of prompting for a key,
breaking the infinite 401 loop.
"""
from __future__ import annotations

import asi


def _bomb_dotenv(*_a, **_k):
    """Fail loudly if a test reaches the real .env writer."""
    raise AssertionError(
        "_save_key_to_dotenv reached from a test — this writes the developer's "
        "real .env (it clobbered DEEPSEEK_API_KEY for months). Stub it."
    )


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
        monkeypatch.setattr(asi, "_save_key_to_dotenv", _bomb_dotenv)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder")
        svc = _FakeSvc(model="deepseek-chat")

        result = asi._prompt_auth_retry_key(
            "deepseek", svc,
            error_message="⚠️ LLM API authentication failed.\n(server message: Invalid API key)",
        )

        assert result is True
        assert svc.llm_service.client is not None


class TestUnverifiedKeyIsNeverPersisted:
    """The prompt must not write to .env until a live call accepts the key.

    This is a real-damage regression: ``create_llm_client`` only constructs an
    object, so ANY non-empty string "succeeded" and was written straight into
    the developer's ``.env``. This very test file used to drive that path with
    ``sk-newkey`` and silently clobbered the real DEEPSEEK_API_KEY on every
    ``pytest`` run.
    """

    def test_prompt_alone_does_not_touch_dotenv(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *_: "sk-unverified")
        import external_llm.client as _client
        monkeypatch.setattr(_client, "create_llm_client", lambda **kw: object())
        monkeypatch.setattr(asi, "_save_key_to_dotenv", _bomb_dotenv)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder")

        assert asi._prompt_auth_retry_key("deepseek", _FakeSvc(model="deepseek-chat")) is True
        # _bomb_dotenv would have exploded — reaching here IS the assertion.

    def test_commit_persists_only_after_verification(self, monkeypatch):
        saved: list[tuple] = []
        monkeypatch.setattr("builtins.input", lambda *_: "sk-verified")
        import external_llm.client as _client
        monkeypatch.setattr(_client, "create_llm_client", lambda **kw: object())
        monkeypatch.setattr(asi, "_save_key_to_dotenv",
                            lambda root, k, v: saved.append((k, v)))
        monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder")

        asi._prompt_auth_retry_key("deepseek", _FakeSvc(model="deepseek-chat"))
        assert saved == [], "must not persist before the retry proves the key"

        asi._commit_verified_api_key()
        assert saved == [("DEEPSEEK_API_KEY", "sk-verified")]

    def test_commit_is_a_noop_without_a_pending_key(self, monkeypatch):
        asi._PENDING_API_KEY.clear()
        monkeypatch.setattr(asi, "_save_key_to_dotenv", _bomb_dotenv)
        asi._commit_verified_api_key()   # skipped prompt / failed retry

    def test_skipped_prompt_leaves_nothing_pending(self, monkeypatch):
        asi._PENDING_API_KEY.clear()
        monkeypatch.setattr("builtins.input", lambda *_: "")
        assert asi._prompt_auth_retry_key("deepseek", _FakeSvc()) is False
        assert asi._PENDING_API_KEY == {}

    def test_shell_export_shadowing_is_reported(self, monkeypatch):
        """A .env write is inert while the shell exports the same key — say so."""
        warned: list[str] = []
        monkeypatch.setattr("builtins.input", lambda *_: "sk-verified")
        import external_llm.client as _client
        monkeypatch.setattr(_client, "create_llm_client", lambda **kw: object())
        monkeypatch.setattr(asi, "_save_key_to_dotenv", lambda *a: None)
        monkeypatch.setattr(asi, "_print", lambda msg, *a, **k: warned.append(msg))
        monkeypatch.setattr(asi, "_SHELL_PROVIDED_ENV_KEYS", {"DEEPSEEK_API_KEY"})

        asi._prompt_auth_retry_key("deepseek", _FakeSvc(model="deepseek-chat"))
        asi._commit_verified_api_key()
        assert any("overrides .env" in m for m in warned), warned

    def test_empty_error_message_falls_through_to_key_prompt(self, monkeypatch):
        # No error_message supplied (legacy call sites) → behave like a real
        # auth failure and prompt for the key. Backward-compat guard.
        monkeypatch.setattr("builtins.input", lambda *_: "")  # skip
        svc = _FakeSvc(model="some-model")

        result = asi._prompt_auth_retry_key("deepseek", svc)

        assert result is False  # user skipped
