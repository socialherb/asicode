from external_llm.agent.json_repair import repair_json_brackets, repair_truncated_json, try_parse_json


class TestSharedRepairJsonBrackets:
    """Edge-case tests for repair_json_brackets not covered in existing shared tests."""

    def test_empty_string(self):
        """Empty string should remain empty."""
        assert repair_json_brackets("") == ""

    def test_only_open_brackets(self):
        """String with only opening brackets should close them."""
        assert repair_json_brackets("[") == "[]"
        assert repair_json_brackets("{") == "{}"
        assert repair_json_brackets("[{") == '[{}]'

    def test_nested_unclosed(self):
        """Nested unclosed brackets should be properly closed."""
        assert repair_json_brackets('{"a": [1, 2') == '{"a": [1, 2]}'

    def test_unterminated_string_in_object(self):
        """String literal at end of text should be closed."""
        text = '{"key": "value'
        result = repair_json_brackets(text)
        # Should close the string, then close the brace
        assert result.endswith('"}')

    def test_truncated_operations_array_recovery(self):
        """Truncated operations array should recover incomplete last object."""
        text = '{"operations": [{"op": "add"}, {"op": "remove"}, {"op":'  # no closing
        result = repair_json_brackets(text)
        # The incomplete object {"op": is closed as {"op":} (empty op)
        assert '"remove"}' in result
        assert result.count('{"op":') == 3

    def test_extra_close_bracket(self):
        """Extra closing bracket should be silently dropped."""
        text = '{"a": [1, 2]]}'
        result = repair_json_brackets(text)
        import json
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}

    def test_extra_close_brace(self):
        """Extra closing brace should be silently dropped."""
        text = '{"a": 1}}'
        result = repair_json_brackets(text)
        import json
        parsed = json.loads(result)
        assert parsed == {"a": 1}

    def test_mismatched_close_brace_in_array(self):
        """Closing brace when stack expects bracket should be dropped."""
        text = '[1, 2}'
        result = repair_json_brackets(text)
        assert result == '[1, 2]'

    def test_mismatched_close_bracket_in_object(self):
        """Closing bracket when stack expects brace should be dropped."""
        text = '{"a": 1]'
        result = repair_json_brackets(text)
        assert result == '{"a": 1}'

    def test_already_valid_json_no_change(self):
        """Valid JSON should remain unchanged."""
        text = '{"key": "value", "arr": [1, 2, 3]}'
        assert repair_json_brackets(text) == text

    def test_only_closing_brackets(self):
        """String with only closing brackets should be stripped to empty."""
        assert repair_json_brackets("]]}}") == ""


class TestSharedRepairTruncatedJson:
    """Edge-case tests for repair_truncated_json focusing on truncated operations array recovery."""

    def test_single_operation_incomplete(self):
        """Single operation truncated mid-object."""
        text = '{"operations": [{"op": "add", "path": "/a"'
        result = repair_truncated_json(text)
        assert result is not None
        # After repair, should close the object, array, and outer JSON
        import json
        parsed = json.loads(result)
        assert len(parsed["operations"]) == 0

    def test_multiple_operations_last_incomplete(self):
        """Multiple operations, only last truncated."""
        text = '{"operations": [{"op": "add"}, {"op": "remove", "path": "/b"}'
        result = repair_truncated_json(text)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert len(parsed["operations"]) == 2

    def test_operations_key_missing(self):
        """No 'operations' key -> should return None (not truncatable)."""
        text = '{"other": "data"'
        assert repair_truncated_json(text) is None

    def test_not_operation_json(self):
        """Not an operations JSON at all -> return None."""
        text = 'plain text, not json'
        assert repair_truncated_json(text) is None

    def test_no_complete_operation_before_truncation(self):
        """Truncated inside first (and only) incomplete operation -> drop it, return empty operations."""
        text = '{"operations": [{"op":'
        result = repair_truncated_json(text)
        assert result is not None
        assert '"operations": []' in result

    def test_already_closed_json(self):
        """Already valid JSON with operations -> should return None (nothing to repair)."""
        text = '{"operations": [{"op": "add"}]}'
        assert repair_truncated_json(text) is None

    def test_operations_array_with_trailing_comma(self):
        """Operations array ends with a comma after last complete object."""
        text = '{"operations": [{"op": "add"},'
        result = repair_truncated_json(text)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["operations"] == [{"op": "add"}]


class TestSharedTryParseJson:
    """Direct tests for try_parse_json: 3-tier parsing with various inputs."""

    def test_valid_json_direct(self):
        """Valid JSON should parse directly."""
        text = '{"key": "value", "num": 42}'
        result = try_parse_json(text)
        assert result == {"key": "value", "num": 42}

    def test_valid_json_with_whitespace(self):
        """Valid JSON with extra whitespace should parse directly."""
        text = '  { "key" : 123 }  '
        result = try_parse_json(text)
        assert result == {"key": 123}

    def test_extra_bracket_repaired(self):
        """Extra closing bracket should be repaired by repair_json_brackets tier."""
        text = '{"a": [1, 2]]}'
        result = try_parse_json(text)
        assert result == {"a": [1, 2]}

    def test_extra_brace_repaired(self):
        """Extra closing brace should be repaired."""
        text = '{"a": 1}}'
        result = try_parse_json(text)
        assert result == {"a": 1}

    def test_truncated_operations_array(self):
        """Truncated operations array should be recovered by repair_truncated_json."""
        text = '{"operations": [{"op": "add"}, {"op": "remove"}'
        result = try_parse_json(text)
        assert result is not None
        assert "operations" in result
        assert len(result["operations"]) == 2

    def test_truncated_single_operation(self):
        """Truncated single operation -> empty operations array."""
        text = '{"operations": [{"op":'
        result = try_parse_json(text)
        assert result is not None
        assert result["operations"] == []

    def test_completely_invalid_returns_none(self):
        """Gibberish should return None."""
        assert try_parse_json("not even close") is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert try_parse_json("") is None

    def test_no_repair_needed_different_structure(self):
        """Valid JSON without operations array should parse directly."""
        text = '[1, 2, {"x": 3}]'
        result = try_parse_json(text)
        assert result == [1, 2, {"x": 3}]

    def test_truncated_operations_with_trailing_comma(self):
        """Operations array ends with comma after last complete object -> repaired."""
        text = '{"operations": [{"op": "add"},'
        result = try_parse_json(text)
        assert result is not None
        assert result["operations"] == [{"op": "add"}]
