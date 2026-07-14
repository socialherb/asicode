"""Tests for routing_policy — _size_bucket, _nesting_bucket, _matches, RoutingPolicy."""

from __future__ import annotations

import pytest
from unittest.mock import mock_open

from external_llm.agent.routing_policy import (
    RoutingPolicy,
    _matches,
    _nesting_bucket,
    _size_bucket,
)


class TestSizeBucket:
    """Tests for _size_bucket(lines) -> str."""

    @pytest.mark.parametrize("lines,expected", [
        (0, "unknown"),
        (-1, "unknown"),
        (1, "small"),
        (10, "small"),
        (29, "small"),
        (30, "medium"),
        (50, "medium"),
        (79, "medium"),
        (80, "large"),
        (100, "large"),
        (199, "large"),
        (200, "xlarge"),
        (1000, "xlarge"),
    ])
    def test_buckets(self, lines, expected):
        assert _size_bucket(lines) == expected


class TestNestingBucket:
    """Tests for _nesting_bucket(depth) -> str."""

    @pytest.mark.parametrize("depth,expected", [
        (0, "flat"),
        (1, "flat"),
        (2, "moderate"),
        (3, "moderate"),
        (4, "deep"),
        (5, "deep"),
        (10, "deep"),
    ])
    def test_buckets(self, depth, expected):
        assert _nesting_bucket(depth) == expected


class TestMatches:
    """Tests for _matches(cond, action_hint, size_bucket, nesting) -> bool."""

    def test_all_conditions_match(self):
        cond = {"action_hint": "bugfix", "size_bucket": "large", "nesting": "deep"}
        assert _matches(cond, "bugfix", "large", "deep")

    def test_empty_conditions_match_anything(self):
        assert _matches({}, "bugfix", "small", "flat")

    def test_action_hint_mismatch(self):
        cond = {"action_hint": "bugfix"}
        assert not _matches(cond, "refactor", "small", "flat")

    def test_size_bucket_mismatch(self):
        cond = {"size_bucket": "large"}
        assert not _matches(cond, "bugfix", "small", "flat")

    def test_nesting_mismatch(self):
        cond = {"nesting": "deep"}
        assert not _matches(cond, "bugfix", "small", "flat")

    def test_partial_conditions(self):
        """Only specified conditions are checked."""
        cond = {"size_bucket": "medium"}
        assert _matches(cond, "any_hint", "medium", "any_depth")


class TestRoutingPolicy:
    """Tests for RoutingPolicy class."""

    def test_empty_policy_returns_none(self):
        policy = RoutingPolicy([])
        assert policy.predict("bugfix") is None
        assert len(policy) == 0

    def test_single_rule_match(self):
        rules = [{
            "id": "r1",
            "conditions": {"action_hint": "bugfix", "size_bucket": "large"},
            "recommended_mode": "surgical_edit",
            "success_rate": 0.8,
            "n": 10,
        }]
        policy = RoutingPolicy(rules)
        assert policy.predict("bugfix", sym_lines=100) == "surgical_edit"

    def test_single_rule_no_match(self):
        rules = [{
            "id": "r1",
            "conditions": {"action_hint": "bugfix"},
            "recommended_mode": "surgical_edit",
            "success_rate": 0.8,
            "n": 10,
        }]
        policy = RoutingPolicy(rules)
        assert policy.predict("refactor") is None  # action_hint mismatch

    def test_most_specific_rule_wins(self):
        """Rules sorted by most conditions first."""
        rules = [
            {
                "id": "general",
                "conditions": {"action_hint": "bugfix"},
                "recommended_mode": "replace_symbol_body",
                "success_rate": 0.7,
                "n": 100,
            },
            {
                "id": "specific",
                "conditions": {"action_hint": "bugfix", "size_bucket": "small", "nesting": "flat"},
                "recommended_mode": "surgical_edit",
                "success_rate": 0.9,
                "n": 20,
            },
        ]
        policy = RoutingPolicy(rules)
        # Both rules match, but "specific" has more conditions → wins
        assert policy.predict("bugfix", sym_lines=10, nested_depth=0) == "surgical_edit"

    def test_empty_action_hint_returns_none(self):
        policy = RoutingPolicy([{
            "conditions": {"action_hint": "bugfix"},
            "recommended_mode": "surgical_edit",
            "success_rate": 0.8,
            "n": 10,
        }])
        assert policy.predict("") is None
        assert policy.predict(None) is None

    def test_ast_op_normalized_to_surgical_edit(self):
        rules = [{
            "id": "r1",
            "conditions": {"action_hint": "bugfix"},
            "recommended_mode": "ast_op",
            "success_rate": 0.8,
            "n": 10,
        }]
        policy = RoutingPolicy(rules)
        assert policy.predict("bugfix") == "surgical_edit"

    def test_unknown_mode_not_returned(self):
        """Modes other than surgical_edit and replace_symbol_body are not returned."""
        rules = [{
            "id": "r1",
            "conditions": {"action_hint": "bugfix"},
            "recommended_mode": "some_unknown_mode",
            "success_rate": 0.8,
            "n": 10,
        }]
        policy = RoutingPolicy(rules)
        assert policy.predict("bugfix") is None

    def test_no_conditions_rule_matches_anything_last(self):
        """An empty-conditions rule matches but is least specific (sorted last)."""
        rules = [
            {
                "id": "catchall",
                "conditions": {},
                "recommended_mode": "surgical_edit",
                "success_rate": 0.5,
                "n": 5,
            },
            {
                "id": "specific",
                "conditions": {"action_hint": "bugfix"},
                "recommended_mode": "replace_symbol_body",
                "success_rate": 0.8,
                "n": 10,
            },
        ]
        policy = RoutingPolicy(rules)
        # 'specific' has 1 condition, 'catchall' has 0 → 'specific' checked first
        assert policy.predict("bugfix") == "replace_symbol_body"
        # For unmatched action_hint, 'catchall' matches
        assert policy.predict("refactor") == "surgical_edit"

    def test_missing_conditions_key(self):
        """Rule without conditions dict is handled gracefully."""
        rules = [{
            "id": "r1",
            "recommended_mode": "surgical_edit",
            "success_rate": 0.8,
            "n": 10,
        }]
        policy = RoutingPolicy(rules)
        # conditions defaults to {} via .get, which matches everything
        assert policy.predict("bugfix") == "surgical_edit"

    def test_len(self):
        assert len(RoutingPolicy([])) == 0
        assert len(RoutingPolicy([{"conditions": {}, "recommended_mode": "x", "success_rate": 0.5, "n": 1}])) == 1
        assert len(RoutingPolicy([{}, {}])) == 2

    def test_rule_sorting_by_condition_count(self):
        """Internally, rules with more conditions come first."""
        rules = [
            {"conditions": {"a": "1"}, "recommended_mode": "surgical_edit", "success_rate": 0.5, "n": 1},
            {"conditions": {"a": "1", "b": "2"}, "recommended_mode": "replace_symbol_body", "success_rate": 0.5, "n": 1},
            {"conditions": {}, "recommended_mode": "surgical_edit", "success_rate": 0.5, "n": 1},
        ]
        policy = RoutingPolicy(rules)
        assert len(policy._rules) == 3
        # 2 conditions → 1 condition → 0 conditions
        assert len(policy._rules[0]["conditions"]) == 2
        assert len(policy._rules[1]["conditions"]) == 1
        assert len(policy._rules[2]["conditions"]) == 0


class TestLoadPolicy:
    """Tests for load_policy() — file I/O + caching."""

    def test_no_file_returns_none(self, monkeypatch):
        """When routing_policy.json does not exist, load_policy returns None."""
        import external_llm.agent.routing_policy as rp
        monkeypatch.setattr(rp, "_last_check", 0.0)
        monkeypatch.setattr(rp, "_cached_policy", None)
        monkeypatch.setattr(rp, "_cached_mtime", 0.0)
        monkeypatch.setattr("os.path.exists", lambda _: False)
        assert rp.load_policy() is None
        assert rp._cached_policy is None  # cache remains None

    def test_valid_file_returns_policy(self, monkeypatch):
        """When a valid policy JSON file exists, load_policy returns a RoutingPolicy."""
        import external_llm.agent.routing_policy as rp
        monkeypatch.setattr(rp, "_last_check", 0.0)
        monkeypatch.setattr(rp, "_cached_policy", None)
        monkeypatch.setattr(rp, "_cached_mtime", 0.0)
        monkeypatch.setattr("os.path.exists", lambda _: True)
        monkeypatch.setattr("os.path.getmtime", lambda _: 100.0)

        valid_json = '{"rules": [{"conditions": {"action_hint": "bugfix"}, "recommended_mode": "surgical_edit", "success_rate": 0.8, "n": 10}]}'
        monkeypatch.setattr("json.load", lambda _: __import__("json").loads(valid_json))
        # Hermetic: do not depend on a real POLICY_PATH file existing on disk.
        monkeypatch.setattr("builtins.open", mock_open(read_data="{}"))

        policy = rp.load_policy()
        assert policy is not None
        assert len(policy) == 1
        assert policy.predict("bugfix") == "surgical_edit"

    def test_cache_hit_skips_file_read(self, monkeypatch):
        """When cache is fresh, load_policy returns cached policy without re-reading."""
        import external_llm.agent.routing_policy as rp
        cached_policy = RoutingPolicy([])
        monkeypatch.setattr(rp, "_last_check", 9999999999.0)  # far in the future
        monkeypatch.setattr(rp, "_cached_policy", cached_policy)
        monkeypatch.setattr(rp, "_cached_mtime", 100.0)

        # Should return cached_policy without checking file existence
        exists_called = False

        def track_exists(_):
            nonlocal exists_called
            exists_called = True
            return True

        monkeypatch.setattr("os.path.exists", track_exists)
        policy = rp.load_policy()
        assert policy is cached_policy
        assert not exists_called  # file system not hit

    def test_cache_ttl_expired_rechecks_file(self, monkeypatch):
        """After cache TTL, load_policy re-checks file."""
        import time

        import external_llm.agent.routing_policy as rp
        monkeypatch.setattr(rp, "_last_check", time.monotonic() - 3600)  # 1 hour ago (expired)
        monkeypatch.setattr(rp, "_cached_policy", None)
        monkeypatch.setattr(rp, "_cached_mtime", 0.0)
        monkeypatch.setattr("os.path.exists", lambda _: False)
        assert rp.load_policy() is None

    def test_invalid_json_returns_none(self, monkeypatch):
        """When the JSON file is malformed, load_policy returns None."""
        import external_llm.agent.routing_policy as rp
        monkeypatch.setattr(rp, "_last_check", 0.0)
        monkeypatch.setattr(rp, "_cached_policy", None)
        monkeypatch.setattr(rp, "_cached_mtime", 0.0)
        monkeypatch.setattr("os.path.exists", lambda _: True)
        monkeypatch.setattr("os.path.getmtime", lambda _: 100.0)

        def broken_json(_):
            raise ValueError("Bad JSON")

        monkeypatch.setattr("json.load", lambda _: (_ for _ in ()).throw(ValueError("Bad JSON")))

        assert rp.load_policy() is None
        assert rp._cached_policy is None  # cache reset to None

    def test_cached_stale_mtime_reloads(self, monkeypatch):
        """When mtime changed, cached policy is refreshed."""
        import external_llm.agent.routing_policy as rp
        monkeypatch.setattr(rp, "_last_check", 0.0)
        monkeypatch.setattr(rp, "_cached_policy", RoutingPolicy([{
            "conditions": {"action_hint": "old"},
            "recommended_mode": "replace_symbol_body",
            "success_rate": 0.5,
            "n": 1,
        }]))
        monkeypatch.setattr(rp, "_cached_mtime", 50.0)  # stale
        monkeypatch.setattr("os.path.exists", lambda _: True)
        monkeypatch.setattr("os.path.getmtime", lambda _: 200.0)  # changed

        valid_json = '{"rules": [{"conditions": {"action_hint": "new"}, "recommended_mode": "surgical_edit", "success_rate": 0.9, "n": 20}]}'
        monkeypatch.setattr("json.load", lambda _: __import__("json").loads(valid_json))
        # Hermetic: do not depend on a real POLICY_PATH file existing on disk.
        monkeypatch.setattr("builtins.open", mock_open(read_data="{}"))

        policy = rp.load_policy()
        assert policy is not None
        assert policy._rules[0]["conditions"]["action_hint"] == "new"

    def test_same_mtime_returns_cached(self, monkeypatch):
        """When mtime matches cached and policy is not None, return cached without re-parsing."""
        import time

        import external_llm.agent.routing_policy as rp
        cached = RoutingPolicy([{
            "conditions": {"action_hint": "bugfix"},
            "recommended_mode": "surgical_edit",
            "success_rate": 0.8,
            "n": 10,
        }])
        # TTL expired → will check file
        monkeypatch.setattr(rp, "_last_check", time.monotonic() - 3600)
        monkeypatch.setattr(rp, "_cached_policy", cached)
        monkeypatch.setattr(rp, "_cached_mtime", 100.0)
        monkeypatch.setattr("os.path.exists", lambda _: True)
        monkeypatch.setattr("os.path.getmtime", lambda _: 100.0)  # same mtime

        json_called = False
        def fail_if_called(*_):
            nonlocal json_called
            json_called = True
            return {}
        monkeypatch.setattr("json.load", fail_if_called)

        policy = rp.load_policy()
        assert policy is cached  # same object, not re-parsed
        assert not json_called  # JSON not re-read
