"""Behavioral tests for try_parse_json fence/prose normalization.

Background: ``try_parse_json`` (the canonical LLM-JSON parser consumed by
``agent_loop``, ``instruction_handler``, and the planner candidate pipeline via
``_parse_json``) used to call ``json.loads`` directly with NO fence stripping
and NO prose isolation. Every fenced (```` ```json {...} ``` ````) or
prose-wrapped ("Here is the result:\\n{...}") response returned ``None``,
collapsing consumers to fallback. These tests pin the canonical normalization
(fence strip + outermost-JSON isolation) as a regression guard.
"""

from __future__ import annotations

from external_llm.agent.json_repair import (
    _isolate_outermost_json,
    _normalize_llm_json,
    try_parse_json,
)

# ── _isolate_outermost_json: pure bracket-balance scan ───────────────────


class TestIsolateOutermostJson:
    def test_plain_object_returned_as_is(self):
        assert _isolate_outermost_json('{"a": 1}') == '{"a": 1}'

    def test_plain_array_returned_as_is(self):
        assert _isolate_outermost_json("[1, 2, 3]") == "[1, 2, 3]"

    def test_strips_leading_prose(self):
        assert _isolate_outermost_json('Here is the plan:\n{"x": 1}') == '{"x": 1}'

    def test_strips_trailing_prose(self):
        assert _isolate_outermost_json('{"x": 1}\nDone.') == '{"x": 1}'

    def test_strips_both_sides(self):
        assert _isolate_outermost_json('Result:\n{"x": 1}\nend') == '{"x": 1}'

    def test_no_opener_returns_original(self):
        # No JSON → return unchanged (so "no JSON" stays None at parse time).
        assert _isolate_outermost_json("just prose, no json") == "just prose, no json"

    def test_unbalanced_returns_original(self):
        # Truncated opener → return original so downstream repair can try.
        assert _isolate_outermost_json('{"ops": [{"a": 1') == '{"ops": [{"a": 1'

    def test_braces_inside_strings_ignored(self):
        # { inside a string value must not affect balance tracking.
        assert _isolate_outermost_json('{"text": "use { carefully", "n": 1}') == '{"text": "use { carefully", "n": 1}'

    def test_nested_arrays_and_objects(self):
        text = 'prefix {"ops": [{"a": 1}, {"b": [2, 3]}]} suffix'
        assert _isolate_outermost_json(text) == '{"ops": [{"a": 1}, {"b": [2, 3]}]}'

    def test_escaped_quotes_in_strings(self):
        # \" inside a string must not toggle string state.
        text = r'{"code": "return \"x\""}'
        assert _isolate_outermost_json(text) == text


# ── _normalize_llm_json: fence strip + isolation ─────────────────────────


class TestNormalizeLlmJson:
    def test_strips_json_fence(self):
        assert _normalize_llm_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_plain_fence(self):
        assert _normalize_llm_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_fence_and_prose(self):
        assert _normalize_llm_json('Result:\n```json\n{"a": 1}\n```\nDone.') == '{"a": 1}'

    def test_idempotent_on_clean_json(self):
        assert _normalize_llm_json('{"a": 1}') == '{"a": 1}'


# ── try_parse_json: end-to-end parse recovery ────────────────────────────


class TestTryParseJsonFenceProse:
    def test_plain_json(self):
        assert try_parse_json('{"intent": "modify", "confidence": 0.9}') == {"intent": "modify", "confidence": 0.9}

    def test_json_fenced(self):
        # BUG CASE: returned None before the fix.
        assert try_parse_json('```json\n{"intent": "modify"}\n```') == {"intent": "modify"}

    def test_plain_fenced(self):
        # BUG CASE: returned None before the fix.
        assert try_parse_json('```\n{"intent": "modify"}\n```') == {"intent": "modify"}

    def test_prose_plus_fenced(self):
        # BUG CASE: returned None before the fix.
        assert try_parse_json('Here is the result:\n```json\n{"intent": "modify"}\n```\nDone.') == {"intent": "modify"}

    def test_prose_plus_bare_json(self):
        # BUG CASE: returned None before the fix.
        assert try_parse_json('Here is the result:\n{"intent": "modify"}\nDone.') == {"intent": "modify"}

    def test_prose_plus_array(self):
        assert try_parse_json("Result:\n[1, 2, 3]\nend") == [1, 2, 3]

    def test_nested_with_prose(self):
        assert try_parse_json('Plan:\n{"ops": [{"a": 1}, {"b": 2}]}\nok') == {"ops": [{"a": 1}, {"b": 2}]}

    def test_no_json_returns_none(self):
        assert try_parse_json("This has no json at all.") is None

    def test_braces_inside_strings_preserved(self):
        assert try_parse_json('{"text": "use { carefully", "n": 1}') == {"text": "use { carefully", "n": 1}

    def test_bracket_repair_still_works(self):
        # Regression: extra closing brace still repaired.
        assert try_parse_json('{"a": 1}}') == {"a": 1}

    def test_truncated_operations_recovery_still_works(self):
        # Regression: truncated operations array still recovers last complete op.
        text = '{"operations": [{"kind": "edit", "path": "a.py"}, {"kind": "edit", "path":'
        result = try_parse_json(text)
        assert result is not None
        assert "operations" in result
        assert len(result["operations"]) == 1


# ── _parse_json delegation parity (planner candidate entrypoint) ─────────


# ── repair_truncated_json: 남은 브랜치 ────────────────────────────────────


class TestRepairTruncatedJsonRemaining:
    def test_operations_marker_without_array_opener_returns_none(self):
        from external_llm.agent.json_repair import repair_truncated_json

        # '"operations"' 뒤에 '['가 없음 → 회복 불가
        assert repair_truncated_json('{"operations": "not-an-array"}') is None

    def test_closed_empty_array_returns_none(self):
        from external_llm.agent.json_repair import repair_truncated_json

        # 완전한 빈 배열 — 잘림 아님 (마지막 완결 객체도, 열린 배열도 없음)
        assert repair_truncated_json('{"operations": []}') is None

    def test_escaped_backslash_inside_string_scan(self):
        from external_llm.agent.json_repair import repair_truncated_json

        # 문자열 내 이스케이프된 백슬래시(\\, 4바이트)가 스캐너를 깨뜨리지 않아야 함
        text = '{"operations": [{"path": "a\\\\b"'
        out = repair_truncated_json(text)
        assert out is not None
        assert out.endswith("]}")

    def test_repaired_output_is_balanced_json(self):
        import json

        from external_llm.agent.json_repair import repair_truncated_json

        text = '{"operations": [{"tool": "read_file", "args": {"path": "x"}}'
        out = repair_truncated_json(text)
        assert out is not None
        parsed = json.loads(out)
        assert parsed["operations"][0]["tool"] == "read_file"


# ── repair_json_brackets: escape/배열/미종결 문자열 ───────────────────────


class TestRepairJsonBracketsRemaining:
    def test_escaped_quote_inside_string_preserved(self):
        from external_llm.agent.json_repair import repair_json_brackets

        text = '{"s": "he said \\"hi\\""}'
        assert repair_json_brackets(text) == text

    def test_escaped_backslash_inside_string_preserved(self):
        from external_llm.agent.json_repair import repair_json_brackets

        text = r'{"path": "a\\b\\c"}'
        assert repair_json_brackets(text) == text

    def test_balanced_array_popped(self):
        from external_llm.agent.json_repair import repair_json_brackets

        assert repair_json_brackets('{"a": [1, 2]}') == '{"a": [1, 2]}'

    def test_unterminated_string_closed(self):
        from external_llm.agent.json_repair import repair_json_brackets

        assert repair_json_brackets('{"a": "trunc') == '{"a": "trunc"}'

    def test_unmatched_close_bracket_skipped(self):
        from external_llm.agent.json_repair import repair_json_brackets

        assert repair_json_brackets('{"a": 1]}') == '{"a": 1}'


class TestRepairTruncatedJsonNoOpsMarker:
    def test_no_operations_marker_returns_none(self):
        from external_llm.agent.json_repair import repair_truncated_json

        assert repair_truncated_json('{"other": 1}') is None

    def test_already_properly_closed_returns_none(self):
        from external_llm.agent.json_repair import repair_truncated_json

        assert repair_truncated_json('{"operations": [{"a": 1}]}') is None
