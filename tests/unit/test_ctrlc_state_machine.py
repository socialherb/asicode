"""Tests for the Ctrl+C single-press exit handler.

The ``_eval_ctrlc_armed`` function is a pure function extracted from the
prompt_toolkit keybinding handler in ``_collect_input``.  These tests
verify all 4 state transitions deterministically without any UI fixture.

Decision table (from docstring):

is_main_prompt  buffer_has_text → should_clear_buffer  should_raise
=============  ===============   ====================  ===========
False           False             False                 True
False           True              False                 True
True            False             False                 True
True            True              True                  False
"""

from __future__ import annotations

from asi import _eval_ctrlc_armed


class TestCtrlCArmedStateMachine:
    """Deterministic tests for the pure Ctrl+C handler decision."""

    # ── Non-main-prompt (e.g. y/N, model selector) ────────────────

    def test_auxiliary_empty_buffer_raises(self) -> None:
        """y/N prompt, empty buffer → immediate raise."""
        should_clear, should_raise = _eval_ctrlc_armed(
            is_main_prompt=False,
            buffer_has_text=False,
        )
        assert (should_clear, should_raise) == (False, True)

    def test_auxiliary_with_text_raises(self) -> None:
        """y/N prompt with partial input → immediate raise (never clear only)."""
        should_clear, should_raise = _eval_ctrlc_armed(
            is_main_prompt=False,
            buffer_has_text=True,
        )
        assert (should_clear, should_raise) == (False, True)

    # ── Main prompt, non-empty buffer ─────────────────────────────

    def test_main_prompt_with_text_clears(self) -> None:
        """User has typed something, Ctrl+C → clear buffer, no exit."""
        should_clear, should_raise = _eval_ctrlc_armed(
            is_main_prompt=True,
            buffer_has_text=True,
        )
        assert (should_clear, should_raise) == (True, False)

    # ── Main prompt, empty buffer (single-press exit) ─────────────

    def test_main_prompt_empty_raises(self) -> None:
        """Empty main prompt, single Ctrl+C → exit immediately (no arm)."""
        should_clear, should_raise = _eval_ctrlc_armed(
            is_main_prompt=True,
            buffer_has_text=False,
        )
        assert (should_clear, should_raise) == (False, True)

    # ── Exhaustive 2x2 = 4-state coverage ─────────────────────────

    def test_exhaustive_all_transitions(self) -> None:
        """All 4 (2²) input combinations produce exactly 2 distinct outputs."""
        results: set[tuple[bool, bool]] = set()
        for main in (False, True):
            for has_text in (False, True):
                r = _eval_ctrlc_armed(
                    is_main_prompt=main,
                    buffer_has_text=has_text,
                )
                results.add(r)
        assert results == {(True, False), (False, True)}
