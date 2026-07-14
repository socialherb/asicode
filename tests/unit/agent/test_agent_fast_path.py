"""Unit tests for agent_fast_path.py — 100% branch coverage."""
from external_llm.agent.agent_fast_path import FastPathMixin, _is_trivial_request

# ── _is_trivial_request ──────────────────────────────────────────────────────

class TestIsTrivialRequest:
    """Branch coverage for the standalone triviality check."""

    def test_none_request(self):
        assert _is_trivial_request(None) is False

    def test_empty_request(self):
        assert _is_trivial_request("") is False

    def test_whitespace_request(self):
        assert _is_trivial_request("   ") is False

    def test_underscore_var_name(self):
        """Underscore name < 40 chars with ASCII alpha → True."""
        assert _is_trivial_request("_max_tokens") is True

    def test_underscore_long_var_name(self):
        """≥ 40 chars → False (len check fails)."""
        long = "_" + "a" * 39
        assert _is_trivial_request(long) is False

    def test_underscore_numbers_only(self):
        """No ASCII alpha → False."""
        assert _is_trivial_request("400_000") is False

    def test_trivial_trigger_typo(self):
        assert _is_trivial_request("fix typo") is True

    def test_trivial_trigger_spelling(self):
        assert _is_trivial_request("spelling mistake") is True

    def test_trivial_trigger_rename(self):
        assert _is_trivial_request("rename variable") is True

    def test_trivial_trigger_header(self):
        assert _is_trivial_request("add header") is True

    def test_only_change_phrase(self):
        assert _is_trivial_request("only change the color") is True

    def test_only_modify_phrase(self):
        assert _is_trivial_request("only modify the text") is True

    def test_constant_short(self):
        """'constant' in short request (< 10 words) → True."""
        assert _is_trivial_request("constant value") is True

    def test_constant_long(self):
        """'constant' in long request (≥ 10 words) → False."""
        long_req = "constant " + " ".join(["word"] * 10)
        assert _is_trivial_request(long_req) is False

    def test_no_match(self):
        assert _is_trivial_request("refactor the entire module") is False


# ── FastPathMixin ────────────────────────────────────────────────────────────

class FakeConfig:
    """Minimal config stub for FastPathMixin tests."""
    def __init__(self, route_decision=None):
        self.route_decision = route_decision


class FakeRoute:
    def __init__(self, task_kind):
        self.task_kind = task_kind


class _Host(FastPathMixin):
    """Concrete host class with minimal config."""
    def __init__(self, config):
        self.config = config


class TestFastPathMixin:
    """Branch coverage for FastPathMixin._is_trivial_edit_request."""

    def test_route_micro_edit(self):
        """Route with MICRO_EDIT → True."""
        from external_llm.agent.task_router import TaskKind
        host = _Host(FakeConfig(route_decision=FakeRoute(TaskKind.MICRO_EDIT)))
        assert host._is_trivial_edit_request("any request") is True

    def test_route_other_kind(self):
        """Route with non-MICRO_EDIT kind → False."""
        from external_llm.agent.task_router import TaskKind
        host = _Host(FakeConfig(route_decision=FakeRoute(TaskKind.SINGLE_FILE_EDIT)))
        assert host._is_trivial_edit_request("any request") is False

    def test_route_none_fallback_trivial(self):
        """No route_decision + trivial request → True."""
        host = _Host(FakeConfig(route_decision=None))
        assert host._is_trivial_edit_request("fix typo") is True

    def test_route_none_fallback_non_trivial(self):
        """No route_decision + non-trivial request → False."""
        host = _Host(FakeConfig(route_decision=None))
        assert host._is_trivial_edit_request("refactor everything") is False
