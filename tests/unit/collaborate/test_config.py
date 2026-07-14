"""
Tests for Claude SDK config entries.
"""
from __future__ import annotations

import os

import config


class TestClaudeSdkConfig:
    """Verify Claude SDK environment variables."""

    def test_defaults(self):
        # Clear env vars to test defaults
        for key in ["CLAUDE_SDK_MAX_TURNS"]:
            os.environ.pop(key, None)

        # Reload config to test defaults
        import importlib
        importlib.reload(config)

        assert config.CLAUDE_SDK_MAX_TURNS == 100

    def test_env_override(self):
        os.environ["CLAUDE_SDK_MAX_TURNS"] = "20"

        import importlib
        importlib.reload(config)

        assert config.CLAUDE_SDK_MAX_TURNS == 20

        # Cleanup
        os.environ.pop("CLAUDE_SDK_MAX_TURNS", None)
        importlib.reload(config)
