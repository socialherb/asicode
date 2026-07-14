"""
Tests for JSON bracket repair.

qwen2.5-coder:7b and similar small models produce malformed JSON with
mismatched brackets. The _repair_json_brackets() helper fixes common
patterns before falling back to an error.

Run: pytest tests/unit/test_json_repair.py -v
"""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.json_repair import (
    repair_json_brackets,
    repair_truncated_json,
    try_parse_json,
)
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


@pytest.fixture
def loop(tmp_path):
    cfg = AgentConfig(max_turns=1)
    reg = ToolRegistry(str(tmp_path), cfg)
    client = Mock()
    client.get_provider_name.return_value = "openai"
    client.provider = "openai"
    return AgentLoop(llm_client=client, registry=reg, config=cfg, model="test")


# ── _repair_json_brackets unit tests ────────────────────────────────────────

class TestRepairJsonBrackets:

    def test_valid_json_unchanged(self, loop):
        text = '{"a": 1, "b": [1, 2, 3]}'
        assert loop._repair_json_brackets(text) == text

    def test_extra_closing_bracket_after_object(self, loop):
        """The screenshot case: ops array closes, then extra ] before }}."""
        malformed = '{"tool": "write_plan", "args": {"plan": {"ops": []}]}}'
        repaired = loop._repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["tool"] == "write_plan"
        assert isinstance(parsed["args"]["plan"]["ops"], list)

    def test_extra_closing_bracket_complex(self, loop):
        """Full reproduction of the screenshot JSON."""
        malformed = (
            '{"tool": "write_plan", "args": {"plan": {"kind": "ASICODE_PLAN_V1",'
            ' "ops": [{"op": "insert_before", "path": "main.py",'
            ' "anchor": "from __future__ import annotations",'
            ' "lines": ["# TESTEST"]}]}]}}'
        )
        repaired = loop._repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["tool"] == "write_plan"
        op = parsed["args"]["plan"]["ops"][0]
        assert op["op"] == "insert_before"
        assert op["lines"] == ["# TESTEST"]

    def test_brackets_inside_string_not_touched(self, loop):
        """Brackets inside string literals must not be counted."""
        text = '{"key": "value with ] and } inside", "ok": true}'
        assert loop._repair_json_brackets(text) == text

    def test_unclosed_array_closed_automatically(self, loop):
        """Unclosed array at end → closing bracket appended."""
        malformed = '{"a": [1, 2, 3}'
        repaired = loop._repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["a"] == [1, 2, 3]

    def test_unclosed_object_closed_automatically(self, loop):
        malformed = '{"a": {"b": 1}'
        repaired = loop._repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["a"]["b"] == 1

    def test_multiple_extra_brackets(self, loop):
        """Multiple extra ] characters are all stripped."""
        malformed = '{"a": [1]}]}'  # one extra ]
        repaired = loop._repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["a"] == [1]

    def test_escaped_quotes_in_string(self, loop):
        """Escaped quotes inside strings don't break string tracking."""
        text = '{"key": "value with \\"quote\\" and ]", "num": 42}'
        repaired = loop._repair_json_brackets(text)
        parsed = json.loads(repaired)
        assert parsed["num"] == 42


    def test_unterminated_string_closed(self, loop):
        """String that never closes → closing quote appended."""
        malformed = '{"key": "unterminated'
        repaired = loop._repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["key"] == "unterminated"

    def test_truncated_code_snippet_string(self, loop):
        """Truncated in code_snippet string value (common LLM truncation)."""
        malformed = '{"tool": "write_plan", "code_snippet": "def foo():'
        repaired = loop._repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["tool"] == "write_plan"


# ── Shared module-level function tests ──────────────────────────────────────

class TestSharedRepairJsonBrackets:
    """Direct tests for module-level repair_json_brackets."""

    def test_valid_json_unchanged(self):
        text = '{"a": 1, "b": [1, 2, 3]}'
        assert repair_json_brackets(text) == text

    def test_extra_closing_bracket(self):
        malformed = '{"tool": "write_plan", "args": {"plan": {"ops": []}]}}'
        repaired = repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["tool"] == "write_plan"

    def test_unterminated_string(self):
        malformed = '{"key": "unterminated'
        repaired = repair_json_brackets(malformed)
        parsed = json.loads(repaired)
        assert parsed["key"] == "unterminated"


class TestSharedTryParseJson:
    """Direct tests for module-level try_parse_json."""

    def test_valid_json_parses_directly(self):
        result = try_parse_json('{"a": 1}')
        assert result == {"a": 1}

    def test_malformed_with_extra_bracket_repaired(self):
        malformed = '{"tool": "write_plan", "args": {"ops": []}]}'
        result = try_parse_json(malformed)
        assert result is not None
        assert result["tool"] == "write_plan"

    def test_truncated_operations_array(self):
        malformed = '{"operations": [{"kind": "insert_import", "path": "test.py"}]'
        result = try_parse_json(malformed)
        assert result is not None
        assert len(result["operations"]) == 1

    def test_completely_invalid_returns_none(self):
        assert try_parse_json("not json at all") is None

    def test_empty_string_returns_none(self):
        assert try_parse_json("") is None
# ── _repair_truncated_json unit tests ──────────────────────────────────────

class TestRepairTruncatedJson:

    def test_truncated_operations_array(self):
        """Operations array with incomplete last object → last complete op recovered."""
        malformed = '''{
  "analysis": "fix imports",
  "operations": [
    {
      "kind": "insert_import",
      "path": "test.py",
      "intent": "from os import path"
    },
    {
      "kind": "insert_after_symbol",
      "path": "test.py",
      "intent": "add test"
  ]}'''
        result = repair_truncated_json(malformed)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert len(parsed["operations"]) == 1
        assert parsed["operations"][0]["kind"] == "insert_import"

    def test_no_truncation_returns_none(self):
        """Complete JSON should not trigger truncation recovery."""
        complete = '{"analysis": "ok", "operations": [{"kind": "test"}]}'
        assert repair_truncated_json(complete) is None

    def test_not_operation_json_returns_none(self):
        """Non-operations JSON should not trigger truncation recovery."""
        assert repair_truncated_json('{"a": 1}') is None

    def test_no_operations_key_returns_none(self):
        """JSON without operations array should not trigger recovery."""
        assert repair_truncated_json('{"tool": "write_plan", "args": {}}') is None

    def test_no_complete_operation_returns_empty(self):
        """When no complete operation object exists, should return empty operations array."""
        malformed = '{"operations": [{"kind": '
        result = repair_truncated_json(malformed)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["operations"] == []

    def test_via_try_parse_json(self, loop):
        """Truncated JSON should be recoverable via _try_parse_json."""
        malformed = '{"analysis": "fix imports", '
        malformed += '"operations": [{"kind": "insert_import", "path": "test.py"}]'  # missing closing }
        result = loop._try_parse_json(malformed)
        assert result is not None
        assert len(result["operations"]) == 1

    def test_truncated_mid_code_snippet_string(self, loop):
        """Truncated mid-string in code_snippet field (log pattern).
        The last operation's string is NOT closed (no trailing "), so
        bracket repair alone cannot recover it — truncation recovery is needed.
        """
        malformed = '{"analysis": "add tests", '
        malformed += '"operations": [{"kind": "insert_import", "path": "test.py"},'
        malformed += '{"kind": "insert_after_symbol", "symbol": "Foo", '
        malformed += '"code_snippet": "def bar():\n    pass\n'  # unterminated string!
        result = loop._try_parse_json(malformed)
        assert result is not None, "_try_parse_json should recover truncated JSON"
        assert len(result["operations"]) == 1, (
            "Only 1 complete op should be recovered; "
            "the second has an unterminated string"
        )
        assert result["operations"][0]["kind"] == "insert_import"
# ── _try_parse_json unit tests ───────────────────────────────────────────────

class TestTryParseJson:

    def test_valid_json_parses_directly(self, loop):
        result = loop._try_parse_json('{"a": 1}')
        assert result == {"a": 1}

    def test_malformed_with_extra_bracket_repaired(self, loop):
        malformed = '{"tool": "write_plan", "args": {"ops": []}]}'
        result = loop._try_parse_json(malformed)
        assert result is not None
        assert result["tool"] == "write_plan"

    def test_completely_invalid_returns_none(self, loop):
        result = loop._try_parse_json("not json at all")
        assert result is None

    def test_empty_string_returns_none(self, loop):
        result = loop._try_parse_json("")
        assert result is None
