import os

import pytest

from external_llm.agent.config.thresholds import _env_flag, _env_int


class TestEnvFlag:
    """Cover _env_flag: truthy values (line 30), falsy values (line 32)."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", "TRUE", "Yes"])
    def test_truthy_values(self, value, monkeypatch):
        monkeypatch.setenv("TEST_FLAG", value)
        assert _env_flag("TEST_FLAG", False) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "FALSE", "No"])
    def test_falsy_values(self, value, monkeypatch):
        monkeypatch.setenv("TEST_FLAG", value)
        assert _env_flag("TEST_FLAG", True) is False

    def test_default_unset(self):
        os.environ.pop("TEST_FLAG_UNSET", None)
        assert _env_flag("TEST_FLAG_UNSET", True) is True

    def test_default_empty(self, monkeypatch):
        monkeypatch.setenv("TEST_FLAG_EMPTY", "")
        assert _env_flag("TEST_FLAG_EMPTY", False) is False

    def test_default_unknown(self, monkeypatch):
        monkeypatch.setenv("TEST_FLAG_OTHER", "xyz")
        assert _env_flag("TEST_FLAG_OTHER", True) is True


class TestEnvInt:
    """Cover _env_int: normal parse (lines 39-40) and exception handler (lines 41-42)."""

    def test_positive_int(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "42")
        assert _env_int("TEST_INT", 10) == 42

    def test_non_positive_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "0")
        assert _env_int("TEST_INT", 10) == 10

    def test_negative_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "-5")
        assert _env_int("TEST_INT", 10) == 10

    def test_invalid_string_raises_exception(self, monkeypatch):
        """Cover lines 41-42: env var is not parseable as int → except Exception → return default."""
        monkeypatch.setenv("TEST_INT", "not_a_number")
        assert _env_int("TEST_INT", 10) == 10

    def test_unset_returns_default(self):
        os.environ.pop("TEST_INT_UNSET", None)
        assert _env_int("TEST_INT_UNSET", 10) == 10

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_EMPTY", "")
        assert _env_int("TEST_INT_EMPTY", 10) == 10

    def test_trailing_whitespace(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "  42  ")
        assert _env_int("TEST_INT", 10) == 42
