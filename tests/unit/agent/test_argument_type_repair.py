"""Wrong argument TYPES must not reach the handlers as Python exceptions.

`ArgumentRepairer` absorbed one class of LLM mistake — the wrong NAME for the
right value — and let the other through. A wrong TYPE reached the handler,
where `args.get("x", "").strip()` and `int(args.get("y", 30))` raised; the
outer catch in `_dispatch_impl` turned that into a ToolResult, so nothing
crashed, but the model's only feedback was a Python exception naming neither
the argument nor the expected type. Measured before this layer: 67 of 138
malformed-argument dispatches (48.6%) across 15 of 24 tools came back that way,
write tools included.

The layer is deliberately conservative — see the split between "coerce",
"drop to the handler's default" and "refuse" pinned below.
"""
from __future__ import annotations

import pytest

from external_llm.agent.argument_repairer import ArgumentRepairer


@pytest.fixture()
def repairer() -> ArgumentRepairer:
    return ArgumentRepairer()


def _coerce(repairer, tool, args):
    return repairer.coerce_types(tool, args)


class TestCoercion:
    """One plausible reading -> rewrite it."""

    def test_number_for_a_string_param(self, repairer):
        out, repairs, errors = _coerce(repairer, "read_file", {"path": 12345})
        assert out["path"] == "12345"
        assert errors == []
        assert repairs

    def test_numeric_string_for_an_integer_param(self, repairer):
        out, _, errors = _coerce(repairer, "grep", {"pattern": "x", "max_results": "30"})
        assert out["max_results"] == 30
        assert errors == []

    def test_whitespace_padded_numeric_string(self, repairer):
        out, _, errors = _coerce(repairer, "grep", {"pattern": "x", "max_results": " 30 "})
        assert out["max_results"] == 30
        assert errors == []

    def test_integral_float_for_an_integer_param(self, repairer):
        out, _, errors = _coerce(repairer, "grep", {"pattern": "x", "max_results": 30.0})
        assert out["max_results"] == 30
        assert errors == []

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("TRUE", True), ("yes", True), ("1", True),
        ("false", False), ("no", False), ("0", False),
    ])
    def test_boolean_strings(self, repairer, raw, expected):
        out, _, errors = _coerce(repairer, "grep", {"pattern": "x", "ignore_case": raw})
        assert out["ignore_case"] is expected
        assert errors == []

    def test_correct_types_are_left_alone(self, repairer):
        args = {"pattern": "x", "max_results": 30, "ignore_case": True}
        out, repairs, errors = _coerce(repairer, "grep", dict(args))
        assert out == args
        assert repairs == [] and errors == []


class TestNullIsAbsence:
    """A JSON null for an optional argument means "no opinion", not a value."""

    def test_null_is_dropped_so_the_handler_default_applies(self, repairer):
        out, repairs, errors = _coerce(repairer, "grep", {"pattern": "x", "max_results": None})
        assert "max_results" not in out
        assert errors == []
        assert repairs

    def test_null_on_a_required_param_yields_the_handler_message(self, repairer):
        """Dropping lets the handler say "'pattern' is required" instead of
        AttributeError: 'NoneType' object has no attribute 'strip'."""
        out, _, errors = _coerce(repairer, "grep", {"pattern": None})
        assert "pattern" not in out
        assert errors == []


class TestUnreadableNumbersDropRatherThanRefuse:
    """Several handlers document a tolerant contract for a garbage number —
    bash clamps timeout="abc" to the default rather than failing the command.
    Refusing here would override a deliberate decision this layer cannot see."""

    def test_non_numeric_string_is_dropped(self, repairer):
        out, repairs, errors = _coerce(repairer, "bash", {"command": "echo hi", "timeout": "abc"})
        assert "timeout" not in out
        assert errors == []
        assert repairs

    def test_the_rest_of_the_call_survives(self, repairer):
        out, _, errors = _coerce(repairer, "bash", {"command": "echo hi", "timeout": "abc"})
        assert out["command"] == "echo hi"
        assert errors == []


class TestRefusals:
    """No scalar reading exists -> refuse, and say what was wanted."""

    @pytest.mark.parametrize("bad", [["a", "b"], {"k": "v"}, ("a",), {"a"}])
    def test_containers_are_never_stringified(self, repairer, bad):
        """str(["a"]) is "['a']" — a path that cannot match anything, failing
        later and further from the cause."""
        _, _, errors = _coerce(repairer, "read_file", {"path": bad})
        assert errors and "'path'" in errors[0] and "string" in errors[0]

    def test_bool_is_not_a_string(self, repairer):
        _, _, errors = _coerce(repairer, "read_file", {"path": True})
        assert errors

    def test_bool_is_not_an_integer(self, repairer):
        """isinstance(True, int) is True in Python, so an unguarded numeric
        branch would silently accept ignore_case=True as max_results=1."""
        _, _, errors = _coerce(repairer, "grep", {"pattern": "x", "max_results": True})
        assert errors and "max_results" in errors[0]

    def test_container_for_an_integer_param(self, repairer):
        _, _, errors = _coerce(repairer, "grep", {"pattern": "x", "max_results": ["3"]})
        assert errors

    def test_message_names_param_actual_and_expected(self, repairer):
        _, _, errors = _coerce(repairer, "read_file", {"path": ["a"]})
        msg = errors[0]
        assert "'path'" in msg and "a string" in msg and "list" in msg

    def test_long_values_are_truncated_in_the_message(self, repairer):
        _, _, errors = _coerce(repairer, "read_file", {"path": ["x" * 500]})
        assert len(errors[0]) < 200


class TestScopeLimits:
    def test_unknown_tool_is_untouched(self, repairer):
        args = {"anything": ["a"]}
        out, repairs, errors = _coerce(repairer, "not_a_tool", dict(args))
        assert out == args and repairs == [] and errors == []

    def test_object_params_are_left_to_their_handler(self, repairer):
        """write_plan's `plan` carries structure this layer has no business
        second-guessing; its handler validates shape and reports it well."""
        plan = {"version": "ASICODE_PLAN_V1", "operations": []}
        out, _, errors = _coerce(repairer, "write_plan", {"plan": plan})
        assert out["plan"] == plan and errors == []

    def test_undeclared_params_pass_through(self, repairer):
        out, _, errors = _coerce(repairer, "grep", {"pattern": "x", "made_up": ["a"]})
        assert out["made_up"] == ["a"] and errors == []


class TestNameRepairStillRuns:
    def test_alias_then_type(self, repairer):
        """Name repair must run FIRST so the type pass can find the declared
        type under the canonical name."""
        r = repairer.repair("read_file", {"file_path": 12345})
        assert r.repaired_args["path"] == "12345"
        assert r.errors == []


class TestDispatchIntegration:
    """The refusal has to reach the model as a tool result, not as an exception
    and not by invoking the handler anyway."""

    @pytest.fixture()
    def registry(self, tmp_path):
        import subprocess
        (tmp_path / "sample.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
        return ToolRegistry(str(tmp_path), AgentConfig(planning_enabled=False, rag_enabled=False))

    def test_refusal_is_a_clean_tool_result(self, registry):
        r = registry.dispatch("read_file", {"path": ["a", "b"]})
        assert r.ok is False
        assert "must be a string" in (r.error or "")
        assert r.metadata.get("blocked") == "argument_type"

    def test_refusal_carries_no_python_exception_text(self, registry):
        r = registry.dispatch("apply_patch", {"patch": {"k": "v"}})
        msg = (r.error or "") + (r.content or "")
        for leak in ("AttributeError", "TypeError", "has no attribute", "raised exception"):
            assert leak not in msg, f"Python internals leaked to the model: {msg[:160]}"

    def test_refusal_is_retryable(self, registry):
        """The model can fix a type and re-send; marking it terminal would
        strand a recoverable call."""
        assert registry.dispatch("read_file", {"path": ["a"]}).retryable is True

    def test_coerced_call_still_reaches_the_handler(self, registry):
        r = registry.dispatch("read_file", {"path": "sample.py", "start_line": "1", "end_line": "2"})
        assert r.ok, r.error
        assert "hello" in r.content

    def test_missing_required_arg_is_not_reported_as_success(self, registry):
        """read_symbol answered "Symbol name is required." with ok=True, which
        reads to the model as asked-and-answered, so the retry never happens."""
        r = registry.dispatch("read_symbol", {"name": None})
        assert r.ok is False
        assert "required" in (r.error or "").lower()
