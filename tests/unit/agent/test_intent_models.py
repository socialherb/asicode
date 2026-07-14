"""Unit tests for intent_models.py — 100% branch coverage.

Tests:
  - IntentResult.__post_init__() — dedup/validation all branches
  - IntentResult.to_dict() / from_dict() round-trip
  - IntentResult.is_read_only()
  - IntentResult.has_edit_intent()
  - IntentResult.get_spec_hints()
  - IntentResolutionConfig.__post_init__() — validation
  - IntentResolutionConfig.get_cache_key()
"""

from __future__ import annotations

import hashlib

import pytest

from external_llm.agent.enums import Complexity, Scope
from external_llm.agent.intent_models import IntentResolutionConfig, IntentResult

# ═══════════════════════════════════════════════════════════════════════════
# IntentResult.__post_init__ — dedup / validation
# ═══════════════════════════════════════════════════════════════════════════


class TestPostInitDedup:
    """Coverage for IntentResult.__post_init__ dedup/validation logic."""

    def test_search_terms_dedup(self):
        r = IntentResult(
            original_request="test", normalized_query="test",
            search_terms=["foo", "bar", "foo", "baz", "bar"],
        )
        assert r.search_terms == ["foo", "bar", "baz"]

    def test_search_terms_empty(self):
        r = IntentResult(original_request="test", normalized_query="test")
        assert r.search_terms == []

    def test_target_files_dedup(self):
        r = IntentResult(
            original_request="test", normalized_query="test",
            target_files=["a.py", "b.py", "a.py", "c.py"],
        )
        assert r.target_files == ["a.py", "b.py", "c.py"]

    def test_target_files_empty(self):
        r = IntentResult(original_request="test", normalized_query="test")
        assert r.target_files == []

    def test_target_symbols_dedup(self):
        r = IntentResult(
            original_request="test", normalized_query="test",
            target_symbols=["foo", "bar", "foo"],
        )
        assert r.target_symbols == ["foo", "bar"]

    def test_target_symbols_empty(self):
        r = IntentResult(original_request="test", normalized_query="test")
        assert r.target_symbols == []

    def test_new_symbols_filter_invalid(self):
        r = IntentResult(
            original_request="test", normalized_query="test",
            new_symbols=[
                {"name": "valid_func", "kind": "function"},
                {"name": "valid_class", "kind": "class", "parent": "Base"},
                {},                 # empty dict → skip
                {"name": ""},       # empty name → skip
                "not_a_dict",       # string → skip
                42,                  # int → skip
                {"name": "valid_func"},  # duplicate name → skip
                {"name": "another", "kind": "method"},
            ],
        )
        assert len(r.new_symbols) == 3
        names = [s["name"] for s in r.new_symbols]
        assert "valid_func" in names
        assert "valid_class" in names
        assert "another" in names
        # Verify fields are normalized
        func = next(s for s in r.new_symbols if s["name"] == "valid_func")
        assert func["kind"] == "function"
        cls = next(s for s in r.new_symbols if s["name"] == "valid_class")
        assert cls["kind"] == "class"
        assert cls["parent"] == "Base"

    def test_new_symbols_empty(self):
        r = IntentResult(original_request="test", normalized_query="test")
        assert r.new_symbols == []

    def test_new_symbols_default_kind(self):
        """Missing kind defaults to 'function'."""
        r = IntentResult(
            original_request="test", normalized_query="test",
            new_symbols=[{"name": "auto_func"}],
        )
        assert r.new_symbols[0]["kind"] == "function"

    def test_new_symbols_parent_none(self):
        """Missing parent becomes None."""
        r = IntentResult(
            original_request="test", normalized_query="test",
            new_symbols=[{"name": "orphan"}],
        )
        assert r.new_symbols[0]["parent"] is None


class TestPostInitLaneHint:
    """Coverage for lane_hint normalization."""

    def test_lane_hint_normalized(self):
        r = IntentResult(
            original_request="test", normalized_query="test",
            lane_hint="  READ_ONLY  ",
        )
        assert r.lane_hint == "read_only"

    def test_lane_hint_empty(self):
        r = IntentResult(original_request="test", normalized_query="test")
        assert r.lane_hint == ""


# ═══════════════════════════════════════════════════════════════════════════
# IntentResult — classification methods
# ═══════════════════════════════════════════════════════════════════════════


class TestIsReadOnly:
    """Coverage for is_read_only()."""

    def test_exploration_is_read_only(self):
        r = IntentResult(
            original_request="explore", normalized_query="explore",
            intent_type="exploration",
        )
        assert r.is_read_only() is True

    def test_question_is_read_only(self):
        r = IntentResult(
            original_request="question", normalized_query="question",
            intent_type="question",
        )
        assert r.is_read_only() is True

    def test_read_only_lane_hint(self):
        r = IntentResult(
            original_request="check this", normalized_query="check this",
            intent_type="bugfix",
            lane_hint="read_only",
        )
        assert r.is_read_only() is True

    def test_bugfix_not_read_only(self):
        r = IntentResult(
            original_request="fix bug", normalized_query="fix bug",
            intent_type="bugfix",
        )
        assert r.is_read_only() is False


class TestHasEditIntent:
    """Coverage for has_edit_intent()."""

    def test_bugfix_has_edit_intent(self):
        r = IntentResult(
            original_request="fix", normalized_query="fix",
            intent_type="bugfix",
        )
        assert r.has_edit_intent() is True

    def test_feature_has_edit_intent(self):
        r = IntentResult(
            original_request="add", normalized_query="add",
            intent_type="feature",
        )
        assert r.has_edit_intent() is True

    def test_refactor_has_edit_intent(self):
        r = IntentResult(
            original_request="refactor", normalized_query="refactor",
            intent_type="refactor",
        )
        assert r.has_edit_intent() is True

    def test_modify_has_edit_intent(self):
        r = IntentResult(
            original_request="modify", normalized_query="modify",
            intent_type="modify",
        )
        assert r.has_edit_intent() is True

    def test_extend_has_edit_intent(self):
        r = IntentResult(
            original_request="extend", normalized_query="extend",
            intent_type="extend",
        )
        assert r.has_edit_intent() is True

    def test_create_has_edit_intent(self):
        r = IntentResult(
            original_request="create", normalized_query="create",
            intent_type="create",
        )
        assert r.has_edit_intent() is True

    def test_exploration_has_no_edit_intent(self):
        r = IntentResult(
            original_request="explore", normalized_query="explore",
            intent_type="exploration",
        )
        assert r.has_edit_intent() is False

    def test_question_has_no_edit_intent(self):
        r = IntentResult(
            original_request="ask", normalized_query="ask",
            intent_type="question",
        )
        assert r.has_edit_intent() is False


class TestGetSpecHints:
    """Coverage for get_spec_hints()."""

    def test_with_target_files(self):
        r = IntentResult(
            original_request="fix a.py", normalized_query="fix a.py",
            target_files=["a.py"],
        )
        hints = r.get_spec_hints()
        assert hints["modify_files"] == ["a.py"]

    def test_with_new_files_in_spec_hints(self):
        r = IntentResult(
            original_request="create", normalized_query="create",
            spec_hints={"new_files": ["new.py"]},
        )
        hints = r.get_spec_hints()
        assert hints.get("new_files") == ["new.py"]
        assert "modify_files" not in hints

    def test_with_target_and_new_files(self):
        r = IntentResult(
            original_request="modify and create",
            normalized_query="modify and create",
            target_files=["a.py"],
            spec_hints={"new_files": ["b.py"]},
        )
        hints = r.get_spec_hints()
        assert hints["modify_files"] == ["a.py"]
        assert hints["new_files"] == ["b.py"]

    def test_no_files(self):
        r = IntentResult(
            original_request="just ask", normalized_query="just ask",
        )
        assert r.get_spec_hints() == {}


# ═══════════════════════════════════════════════════════════════════════════
# IntentResult — serialization
# ═══════════════════════════════════════════════════════════════════════════


class TestToDict:
    """Coverage for to_dict()."""

    def test_basic_fields(self):
        r = IntentResult(
            original_request="fix bug in module",
            normalized_query="fix bug in module",
            search_terms=["bug"],
            intent_type="bugfix",
            target_files=["src/mod.py"],
            target_symbols=["buggy_func"],
            modify_symbols=["buggy_func"],
            reference_symbols=["helper"],
            lane_hint="main_agent",
            confidence=0.9,
        )
        d = r.to_dict()
        assert d["original_request"] == "fix bug in module"
        assert d["intent_type"] == "bugfix"
        assert d["target_files"] == ["src/mod.py"]
        assert d["scope_hint"] == "single_file"
        assert d["complexity_hint"] == "LOW"


class TestFromDict:
    """Coverage for from_dict()."""

    def test_round_trip(self):
        r = IntentResult(
            original_request="add feature X",
            normalized_query="add feature X",
            search_terms=["feature", "X"],
            intent_type="feature",
            target_files=["src/x.py"],
            target_symbols=["XClass"],
            new_symbols=[{"name": "helper", "kind": "function"}],
            modify_symbols=["XClass"],
            reference_symbols=["Base"],
            lane_hint="planner",
            scope_hint=Scope.MULTI_FILE,
            complexity_hint=Complexity.MEDIUM,
            is_test_write=True,
            is_style_fix=False,
            is_filesystem_op=False,
            is_ui_change=False,
            is_interface_preserving=True,
            confidence=0.85,
            metadata={"lang": "python"},
            spec_hints={"new_files": ["tests/test_x.py"]},
        )
        d = r.to_dict()
        restored = IntentResult.from_dict(d)
        assert restored.original_request == r.original_request
        assert restored.intent_type == r.intent_type
        assert restored.target_files == r.target_files
        assert restored.scope_hint == Scope.MULTI_FILE
        assert restored.complexity_hint == Complexity.MEDIUM
        assert restored.is_test_write is True
        assert restored.is_interface_preserving is True
        assert restored.confidence == 0.85
        assert restored.metadata == {"lang": "python"}

    def test_from_dict_strips_unknown_fields(self):
        d = {
            "original_request": "test",
            "normalized_query": "test",
            "unknown_field": "should be stripped",
            "scope_hint": "single_file",
            "complexity_hint": "LOW",
        }
        restored = IntentResult.from_dict(d)
        assert restored.original_request == "test"
        assert restored.scope_hint == Scope.SINGLE_FILE
        assert restored.complexity_hint == Complexity.LOW
        assert not hasattr(restored, "unknown_field")

    def test_from_dict_minimal(self):
        restored = IntentResult.from_dict({
            "original_request": "test",
            "normalized_query": "test",
        })
        assert restored.original_request == "test"
        assert restored.normalized_query == "test"
        assert restored.scope_hint == Scope.SINGLE_FILE
        assert restored.complexity_hint == Complexity.LOW


# ═══════════════════════════════════════════════════════════════════════════
# IntentResolutionConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentResolutionConfig:
    """Coverage for IntentResolutionConfig."""

    def test_valid_config(self):
        cfg = IntentResolutionConfig(model="claude-3-5-sonnet-20241022")
        assert cfg.model == "claude-3-5-sonnet-20241022"
        assert cfg.enable_cache is True
        assert cfg.cache_ttl_seconds == 300

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="IntentResolver requires a model name"):
            IntentResolutionConfig(model="")

    def test_get_cache_key(self):
        cfg = IntentResolutionConfig(model="test-model")
        key = cfg.get_cache_key("add feature X")
        expected = hashlib.md5(b"add feature X").hexdigest()[:16]
        assert key == expected

    def test_cache_key_different_for_different_inputs(self):
        cfg = IntentResolutionConfig(model="test-model")
        k1 = cfg.get_cache_key("request A")
        k2 = cfg.get_cache_key("request B")
        assert k1 != k2

    def test_cache_key_consistent(self):
        cfg = IntentResolutionConfig(model="test-model")
        k1 = cfg.get_cache_key("same input")
        k2 = cfg.get_cache_key("same input")
        assert k1 == k2
