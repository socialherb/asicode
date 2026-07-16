"""
Unit tests for the /auto auto-continue loop's pure decision logic
(_parse_auto_arg / _auto_continue_should_arm / _validate_next_suggestion
max_len budget in asi.py).

The auto-continue feature countdown-submits the ghost suggestion as the next
turn's instruction, so its safety rests on three pure decisions locked here:

  1. /auto argument parsing — explicit opt-in/out only, garbage rejected.
  2. Arming — never fire when off / no suggestion / depth cap reached, and the
     blocking reason is reported (cap stop must notify, not go silent).
  3. Suggestion budget — auto instructions are self-contained so they get a
     longer max_len; the display-hint contract keeps the original 140 so a
     rule-ignoring helper model still cannot leak long noise onto the prompt.
"""
import asi


# ─── _parse_auto_arg ─────────────────────────────────────────────────────────

def test_parse_auto_empty_toggles():
    assert asi._parse_auto_arg("", cur_on=False) == (True, None, None)
    assert asi._parse_auto_arg("", cur_on=True) == (False, None, None)


def test_parse_auto_on_off_keywords():
    assert asi._parse_auto_arg("on", cur_on=False) == (True, None, None)
    assert asi._parse_auto_arg("off", cur_on=True) == (False, None, None)
    # off is idempotent — not a toggle
    assert asi._parse_auto_arg("off", cur_on=False) == (False, None, None)
    assert asi._parse_auto_arg("STOP", cur_on=True) == (False, None, None)


def test_parse_auto_numeric_sets_cap_and_enables():
    assert asi._parse_auto_arg("8", cur_on=False) == (True, 8, None)
    assert asi._parse_auto_arg("3", cur_on=True) == (True, 3, None)


def test_parse_auto_rejects_garbage():
    for bad in ("0", "-2", "3.5", "eight", "on off"):
        new_on, new_cap, err = asi._parse_auto_arg(bad, cur_on=False)
        assert new_on is None and new_cap is None
        assert err  # usage message present


# ─── _auto_continue_should_arm ───────────────────────────────────────────────

def test_should_arm_happy_path():
    assert asi._auto_continue_should_arm(True, 0, 5, "run the tests") == (True, "")
    # last allowed step: depth 4 of cap 5
    assert asi._auto_continue_should_arm(True, 4, 5, "next") == (True, "")


def test_should_arm_blocks_when_off():
    arm, reason = asi._auto_continue_should_arm(False, 0, 5, "run the tests")
    assert not arm and reason == "off"


def test_should_arm_blocks_without_suggestion():
    arm, reason = asi._auto_continue_should_arm(True, 0, 5, "")
    assert not arm and reason == "no_suggestion"


def test_should_arm_blocks_at_depth_cap():
    arm, reason = asi._auto_continue_should_arm(True, 5, 5, "run the tests")
    assert not arm and reason == "cap_reached"
    arm, reason = asi._auto_continue_should_arm(True, 7, 5, "run the tests")
    assert not arm and reason == "cap_reached"


# ─── _validate_next_suggestion max_len budget ────────────────────────────────

def test_validator_default_budget_rejects_long_auto_style_instruction():
    # 141+ chars: fine as a self-contained auto instruction, too long for the
    # display-hint contract — default budget must still reject it.
    long_text = "run pytest tests/unit and then " + "x" * 120
    assert asi._validate_next_suggestion(long_text, "fix the bug") is None


def test_validator_auto_budget_accepts_self_contained_instruction():
    long_text = "run pytest tests/unit and then " + "x" * 120
    assert asi._validate_next_suggestion(
        long_text, "fix the bug", max_len=asi._AUTO_SUGGESTION_MAX_LEN) == long_text


def test_validator_auto_budget_still_rejects_none_and_language_mismatch():
    # NONE stays the stop signal regardless of budget.
    assert asi._validate_next_suggestion(
        "NONE", "fix the bug", max_len=asi._AUTO_SUGGESTION_MAX_LEN) is None
    # Script guard survives the budget change: Korean request → ASCII reply suppressed.
    assert asi._validate_next_suggestion(
        "verify the previous fix then run tests", "버그를 고치자",
        max_len=asi._AUTO_SUGGESTION_MAX_LEN) is None


def test_validator_auto_budget_has_a_ceiling_too():
    assert asi._validate_next_suggestion(
        "y" * (asi._AUTO_SUGGESTION_MAX_LEN + 1), "fix the bug",
        max_len=asi._AUTO_SUGGESTION_MAX_LEN) is None
