"""Regression tests for DecompositionGuard candidate name filters.

Verifies that Python keywords (None, True, False, ...) and single-letter
placeholder names extracted from intent text are NOT treated as missing symbols.
"""
import keyword

# ── helper: invoke the filtering logic in isolation ──────────────────────────

def _apply_keyword_filter(candidate_names: set) -> set:
    """Replicate the filtering step from _find_missing_callees_for_guard."""
    return {n for n in candidate_names if not keyword.iskeyword(n)}


def _simulate_placeholder_filter(candidate_names: set, regex_names: set, contract_names: set) -> set:
    """Replicate the placeholder filter from fixed _find_missing_callees_for_guard."""
    _regex_single_letter = {n for n in regex_names if len(n) == 1}
    return (candidate_names - _regex_single_letter) | contract_names


# ── Part A: keyword/literal tokens are removed ───────────────────────────────

class TestKeywordFilter:
    def test_none_is_filtered(self):
        result = _apply_keyword_filter({"None"})
        assert "None" not in result

    def test_true_false_filtered_ellipsis_not(self):
        # True/False are keywords → filtered.
        # Ellipsis is a builtin constant, not a keyword → preserved (builtins not filtered).
        result = _apply_keyword_filter({"True", "False", "Ellipsis"})
        assert "True" not in result
        assert "False" not in result
        assert "Ellipsis" in result  # builtin, not a keyword

    def test_python_keywords_filtered(self):
        kws = {"return", "if", "for", "while", "class", "def", "import",
               "from", "as", "pass", "break", "continue", "raise", "yield"}
        result = _apply_keyword_filter(kws)
        assert result == set(), f"Keyword names should all be filtered: {result}"

    def test_builtins_are_NOT_filtered(self):
        # Builtins like ValueError, list, re are legitimate missing-symbol targets
        # and must NOT be filtered — only keywords are filtered.
        builtins = {"ValueError", "list", "dict", "Exception", "re", "int"}
        result = _apply_keyword_filter(builtins)
        assert result == builtins, f"Builtins must survive filter: {result}"

    def test_user_symbols_preserved(self):
        user_names = {"_my_helper", "compute_score", "MyClass", "do_work"}
        result = _apply_keyword_filter(user_names)
        assert result == user_names, f"User-defined symbols must survive: {result}"

    def test_mixed_set_keywords_removed_builtins_preserved(self):
        # Keywords (None, True, return) removed; builtins (isinstance) and user symbols kept
        mixed = {"None", "True", "_real_func", "isinstance", "_helper", "return"}
        result = _apply_keyword_filter(mixed)
        assert result == {"_real_func", "isinstance", "_helper"}

    def test_underscore_private_names_preserved(self):
        private = {"_check_f821", "_auto_repair", "_run_internal"}
        result = _apply_keyword_filter(private)
        assert result == private


# ── Part C: single-letter placeholder filter ─────────────────────────────────

class TestPlaceholderFilter:
    """Verifies single-letter names from regex are filtered out.

    Prevents spurious INSERT ops for placeholder names like 'X', 'Y', 'i'
    that appear in code explanations but are never meant to be created.
    Contract names (Tier 1) are exempt — they come from the structured plan.
    """

    def test_single_letter_placeholder_filtered(self):
        candidates = {"X"}
        regex_names = {"X"}
        contract_names = set()
        result = _simulate_placeholder_filter(candidates, regex_names, contract_names)
        assert "X" not in result

    def test_single_lowercase_letter_filtered(self):
        candidates = {"i", "n", "k", "v"}
        regex_names = {"i", "n", "k", "v"}
        contract_names = set()
        result = _simulate_placeholder_filter(candidates, regex_names, contract_names)
        assert result == set(), f"All single-letter placeholders should be filtered: {result}"

    def test_contract_name_preserved_even_when_single_letter(self):
        """Contract names (Tier 1) are exempt — they come from structured plan."""
        candidates = {"X"}
        regex_names = {"X"}
        contract_names = {"X"}  # would be unusual, but contract is authoritative
        result = _simulate_placeholder_filter(candidates, regex_names, contract_names)
        assert "X" in result, "Contract names must be preserved"

    def test_multi_letter_name_preserved(self):
        candidates = {"validate_input", "process_data"}
        regex_names = {"validate_input", "process_data"}
        contract_names = set()
        result = _simulate_placeholder_filter(candidates, regex_names, contract_names)
        assert result == {"validate_input", "process_data"}

    def test_mixed_multi_letter_preserved_single_filtered(self):
        candidates = {"validate_input", "X", "process_data", "Y"}
        regex_names = {"validate_input", "X", "process_data", "Y"}
        contract_names = set()
        result = _simulate_placeholder_filter(candidates, regex_names, contract_names)
        assert result == {"validate_input", "process_data"}


# ── Part B: SSE status consistency logic ─────────────────────────────────────
#
# The agent_loop status expression:
#   "partial_success" if (failed > 0 or _is_blocking_verdict) else "success"
# where _is_blocking_verdict = exec_final_status in {"failed", ...}

def _compute_sse_status(failed: int, exec_final_status: str) -> str:
    """Replicate the status computation from agent_loop._run_planner_lane."""
    _BLOCKING = {"failed", "verification_failed", "rollback", "execution_error"}
    _is_blocking = exec_final_status in _BLOCKING
    return "partial_success" if (failed > 0 or _is_blocking) else "success"


class TestSSEStatusConsistency:
    def test_all_ops_ok_and_verdict_success(self):
        assert _compute_sse_status(failed=0, exec_final_status="success") == "success"

    def test_failed_ops_gives_partial(self):
        assert _compute_sse_status(failed=1, exec_final_status="success") == "partial_success"

    def test_blocking_verdict_gives_partial_even_with_zero_failures(self):
        """Core regression: F821 blocking verdict + 0 failed ops → partial, NOT success."""
        assert _compute_sse_status(failed=0, exec_final_status="failed") == "partial_success"

    def test_verification_failed_verdict_gives_partial(self):
        assert _compute_sse_status(failed=0, exec_final_status="verification_failed") == "partial_success"

    def test_rollback_verdict_gives_partial(self):
        assert _compute_sse_status(failed=0, exec_final_status="rollback") == "partial_success"

    def test_execution_error_gives_partial(self):
        assert _compute_sse_status(failed=0, exec_final_status="execution_error") == "partial_success"

    def test_completed_verdict_gives_success(self):
        assert _compute_sse_status(failed=0, exec_final_status="completed") == "success"

    def test_empty_verdict_gives_success(self):
        # Unknown/missing status defaults to success (no blocking)
        assert _compute_sse_status(failed=0, exec_final_status="") == "success"

    def test_failed_ops_and_blocking_verdict_both_give_partial(self):
        assert _compute_sse_status(failed=2, exec_final_status="failed") == "partial_success"
