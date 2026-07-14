"""Tests for the Ctrl+C two-button exit state machine.

The ``_eval_ctrlc_armed`` function is a pure function extracted from the
prompt_toolkit keybinding handler in ``_collect_input``.  These tests
verify all 4 state transitions deterministically without any UI fixture.

Table of transitions (from docstring):

current_armed  is_main_prompt  buffer_has_text → new_armed  should_raise
=============  ==============  ===============   =========  ===========
any            False           any               False      True
any            True            True              False      False
False          True            False             True       False
True           True            False             False      True
"""

from __future__ import annotations

from asi import _eval_ctrlc_armed


class TestCtrlCArmedStateMachine:
    """Deterministic tests for the pure Ctrl+C state machine."""

    # ── Non-main-prompt (e.g. y/N, model selector) ────────────────

    def test_non_main_prompt_disarmed_empty_buffer(self) -> None:
        """y/N prompt, first Ctrl+C → immediate raise."""
        new_armed, should_raise = _eval_ctrlc_armed(
            current_armed=False, is_main_prompt=False, buffer_has_text=False,
        )
        assert (new_armed, should_raise) == (False, True)

    def test_non_main_prompt_armed_empty_buffer(self) -> None:
        """y/N prompt, already somehow armed (corner case) → raise."""
        new_armed, should_raise = _eval_ctrlc_armed(
            current_armed=True, is_main_prompt=False, buffer_has_text=False,
        )
        assert (new_armed, should_raise) == (False, True)

    def test_non_main_prompt_with_text(self) -> None:
        """y/N prompt with partial input → immediate raise, never arm."""
        new_armed, should_raise = _eval_ctrlc_armed(
            current_armed=False, is_main_prompt=False, buffer_has_text=True,
        )
        assert (new_armed, should_raise) == (False, True)

    # ── Main prompt, non-empty buffer ─────────────────────────────

    def test_main_prompt_with_text_disarmed(self) -> None:
        """User has typed something, Ctrl+C → clear buffer + disarm."""
        new_armed, should_raise = _eval_ctrlc_armed(
            current_armed=False, is_main_prompt=True, buffer_has_text=True,
        )
        assert (new_armed, should_raise) == (False, False)

    def test_main_prompt_with_text_armed(self) -> None:
        """Previously armed, now with text → clear + disarm (no raise)."""
        new_armed, should_raise = _eval_ctrlc_armed(
            current_armed=True, is_main_prompt=True, buffer_has_text=True,
        )
        assert (new_armed, should_raise) == (False, False)

    # ── Main prompt, empty buffer, first Ctrl+C (arm) ─────────────

    def test_main_prompt_empty_first_ctrlc(self) -> None:
        """First Ctrl+C on empty main prompt → arm, do not raise."""
        new_armed, should_raise = _eval_ctrlc_armed(
            current_armed=False, is_main_prompt=True, buffer_has_text=False,
        )
        assert (new_armed, should_raise) == (True, False)

    # ── Main prompt, empty buffer, second Ctrl+C (raise) ──────────

    def test_main_prompt_empty_second_ctrlc(self) -> None:
        """Second Ctrl+C on empty main prompt → raise immediately."""
        new_armed, should_raise = _eval_ctrlc_armed(
            current_armed=True, is_main_prompt=True, buffer_has_text=False,
        )
        assert (new_armed, should_raise) == (False, True)

    # ── Exhaustive 2x2x2 = 8-state coverage ───────────────────────

    def test_exhaustive_all_transitions(self) -> None:
        """All 8 (2³) input combinations produce exactly 4 distinct outputs."""
        results: set[tuple[bool, bool]] = set()
        for armed in (False, True):
            for main in (False, True):
                for has_text in (False, True):
                    r = _eval_ctrlc_armed(
                        current_armed=armed, is_main_prompt=main,
                        buffer_has_text=has_text,
                    )
                    results.add(r)
        assert results == {(False, True), (False, False), (True, False)}
