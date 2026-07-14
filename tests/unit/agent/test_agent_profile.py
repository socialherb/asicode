"""Tests for AgentProfile, get_builtin_profile, load_profile."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from external_llm.agent.agent_profile import (
    BUILTIN_PROFILES,
    AgentProfile,
    get_builtin_profile,
    load_profile,
)


class TestAgentProfileFromDict:
    """Tests for AgentProfile.from_dict()."""

    def test_minimal_dict(self):
        profile = AgentProfile.from_dict({"name": "test"})
        assert profile.name == "test"
        assert profile.description == ""
        assert profile.allowed_tools == []
        assert profile.blocked_tools == []
        assert profile.model is None

    def test_full_dict(self):
        data = {
            "name": "custom",
            "description": "My custom agent",
            "allowed_tools": ["read_file", "grep"],
            "blocked_tools": ["bash"],
            "model": "claude-sonnet-4",
            "provider": "anthropic",
            "system_prompt_prefix": "Be careful.",
            "max_turns": 15,
            "planning_enabled": False,
        }
        profile = AgentProfile.from_dict(data)
        assert profile.name == "custom"
        assert profile.description == "My custom agent"
        assert profile.allowed_tools == ["read_file", "grep"]
        assert profile.blocked_tools == ["bash"]
        assert profile.model == "claude-sonnet-4"
        assert profile.provider == "anthropic"
        assert profile.system_prompt_prefix == "Be careful."
        assert profile.max_turns == 15
        assert profile.planning_enabled is False

    def test_none_allowed_tools_becomes_empty(self):
        """None or missing allowed_tools becomes empty list (no restriction)."""
        profile = AgentProfile.from_dict({"name": "test", "allowed_tools": None})
        assert profile.allowed_tools == []

    def test_empty_allowed_tools_no_restriction(self):
        profile = AgentProfile.from_dict({"name": "test", "allowed_tools": []})
        assert profile.allowed_tools == []

    def test_unknown_fields_ignored(self):
        profile = AgentProfile.from_dict({"name": "test", "extra_field": "value"})
        assert profile.name == "test"
        assert not hasattr(profile, "extra_field")


class TestAgentProfileToDict:
    """Tests for AgentProfile.to_dict()."""

    def test_roundtrip(self):
        data = {
            "name": "reviewer",
            "description": "Read-only",
            "allowed_tools": ["find_symbol"],
            "blocked_tools": [],
            "model": None,
            "provider": None,
            "system_prompt_prefix": None,
            "max_turns": None,
            "planning_enabled": None,
        }
        profile = AgentProfile.from_dict(data)
        assert profile.to_dict() == data

    def test_non_none_fields_preserved(self):
        profile = AgentProfile(
            name="myagent",
            description="desc",
            allowed_tools=["grep"],
            blocked_tools=["bash"],
            model="gpt-4",
        )
        d = profile.to_dict()
        assert d["name"] == "myagent"
        assert d["model"] == "gpt-4"
        assert d["provider"] is None


class TestAgentProfileLoad:
    """Tests for AgentProfile.load() — file I/O."""

    def test_load_valid_profile(self, tmp_path):
        agents_dir = tmp_path / ".asicode" / "agents"
        agents_dir.mkdir(parents=True)
        profile_file = agents_dir / "myagent.json"
        profile_file.write_text(json.dumps({
            "name": "myagent",
            "allowed_tools": ["grep", "read_file"],
            "blocked_tools": [],
        }), encoding="utf-8")

        profile = AgentProfile.load("myagent", str(tmp_path))
        assert profile.name == "myagent"
        assert profile.allowed_tools == ["grep", "read_file"]

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AgentProfile.load("nonexistent", str(tmp_path))

    def test_load_malformed_json(self, tmp_path):
        agents_dir = tmp_path / ".asicode" / "agents"
        agents_dir.mkdir(parents=True)
        profile_file = agents_dir / "bad.json"
        profile_file.write_text("not valid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed agent profile JSON"):
            AgentProfile.load("bad", str(tmp_path))


class TestAgentProfileApply:
    """Tests for AgentProfile.apply(agent_config)."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.model = "default-model"
        config.provider = "default-provider"
        config.max_turns = 100
        config.planning_enabled = True
        return config

    def test_none_fields_do_not_override(self, mock_config):
        profile = AgentProfile(name="test")  # all optional fields are None
        profile.apply(mock_config)
        assert mock_config.model == "default-model"
        assert mock_config.provider == "default-provider"
        assert mock_config.max_turns == 100
        assert mock_config.planning_enabled is True

    def test_set_model(self, mock_config):
        profile = AgentProfile(name="test", model="claude-opus-4")
        profile.apply(mock_config)
        assert mock_config.model == "claude-opus-4"
        assert mock_config.provider == "default-provider"  # unchanged

    def test_set_provider(self, mock_config):
        profile = AgentProfile(name="test", provider="deepseek")
        profile.apply(mock_config)
        assert mock_config.provider == "deepseek"

    def test_set_max_turns(self, mock_config):
        profile = AgentProfile(name="test", max_turns=5)
        profile.apply(mock_config)
        assert mock_config.max_turns == 5

    def test_set_planning_enabled(self, mock_config):
        profile = AgentProfile(name="test", planning_enabled=False)
        profile.apply(mock_config)
        assert mock_config.planning_enabled is False

    def test_set_all_fields(self, mock_config):
        profile = AgentProfile(
            name="full",
            model="m1",
            provider="p1",
            max_turns=10,
            planning_enabled=False,
        )
        profile.apply(mock_config)
        assert mock_config.model == "m1"
        assert mock_config.provider == "p1"
        assert mock_config.max_turns == 10
        assert mock_config.planning_enabled is False


class TestGetBuiltinProfile:
    """Tests for get_builtin_profile()."""

    def test_reviewer_exists(self):
        profile = get_builtin_profile("reviewer")
        assert profile is not None
        assert profile.name == "reviewer"
        assert "find_symbol" in profile.allowed_tools

    def test_patcher_exists(self):
        profile = get_builtin_profile("patcher")
        assert profile is not None
        assert profile.name == "patcher"
        assert "bash" in profile.blocked_tools

    def test_tester_exists(self):
        profile = get_builtin_profile("tester")
        assert profile is not None
        assert profile.name == "tester"
        assert profile.planning_enabled is False

    def test_unknown_returns_none(self):
        assert get_builtin_profile("nonexistent") is None

    def test_all_builtins_are_valid(self):
        for name in BUILTIN_PROFILES:
            profile = get_builtin_profile(name)
            assert profile is not None
            assert profile.name == name


class TestLoadProfile:
    """Tests for load_profile(name, repo_root)."""

    def test_file_based_takes_priority(self, tmp_path):
        """File-based profile should be loaded before built-in fallback."""
        agents_dir = tmp_path / ".asicode" / "agents"
        agents_dir.mkdir(parents=True)
        profile_file = agents_dir / "reviewer.json"  # same name as built-in
        profile_file.write_text(json.dumps({
            "name": "custom-reviewer",
            "allowed_tools": ["bash"],
        }), encoding="utf-8")

        profile = load_profile("reviewer", str(tmp_path))
        assert profile.name == "custom-reviewer"  # file version, not built-in
        assert profile.allowed_tools == ["bash"]

    def test_fallback_to_builtin(self, tmp_path):
        """When no file exists, fall back to built-in."""
        profile = load_profile("reviewer", str(tmp_path))
        assert profile is not None
        assert profile.name == "reviewer"

    def test_raises_when_not_found_anywhere(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_profile("no_such_profile", str(tmp_path))
