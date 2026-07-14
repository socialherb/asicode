"""Unit tests for reasoning_utils.py — 100% branch coverage."""

from external_llm.agent.reasoning_utils import reasoning_ab_kwargs

ENV_VAR = "ASICODE_PLANNER_REASONING"


class TestReasoningAbKwargs:
    """Tests for reasoning_ab_kwargs with env var mocking."""

    def test_default_on(self, monkeypatch):
        """Default behavior: no env var → reasoning stays ON → empty dict."""
        monkeypatch.delenv(ENV_VAR, raising=False)
        monkeypatch.delenv("ASICODE_DEEPSEEK_NOTHINK_JSON", raising=False)
        assert reasoning_ab_kwargs(ENV_VAR) == {}

    def test_explicit_on(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "on")
        assert reasoning_ab_kwargs(ENV_VAR) == {}

    def test_off(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "off")
        result = reasoning_ab_kwargs(ENV_VAR)
        assert result == {"thinking": {"type": "disabled"}}

    def test_off_false_string(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "false")
        result = reasoning_ab_kwargs(ENV_VAR)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_off_no(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "no")
        result = reasoning_ab_kwargs(ENV_VAR)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_off_zero(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "0")
        result = reasoning_ab_kwargs(ENV_VAR)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_case_insensitive_off(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "OFF")
        result = reasoning_ab_kwargs(ENV_VAR)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_off_with_custom_json(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "off")
        monkeypatch.setenv("ASICODE_DEEPSEEK_NOTHINK_JSON", '{"reasoning": {"enabled": false}}')
        result = reasoning_ab_kwargs(ENV_VAR)
        assert result == {"reasoning": {"enabled": False}}

    def test_invalid_json_fallback_to_empty(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "off")
        monkeypatch.setenv("ASICODE_DEEPSEEK_NOTHINK_JSON", "not valid json{{{")
        result = reasoning_ab_kwargs(ENV_VAR)
        assert result == {}

    def test_json_not_a_dict_fallback_to_empty(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "off")
        monkeypatch.setenv("ASICODE_DEEPSEEK_NOTHINK_JSON", '"just a string"')
        result = reasoning_ab_kwargs(ENV_VAR)
        assert result == {}

    def test_unknown_mode_defaults_on(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "maybe")
        assert reasoning_ab_kwargs(ENV_VAR) == {}

    def test_empty_env_var_defaults_on(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "")
        assert reasoning_ab_kwargs(ENV_VAR) == {}

    def test_whitespace_env_var_defaults_on(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "  ")
        assert reasoning_ab_kwargs(ENV_VAR) == {}
