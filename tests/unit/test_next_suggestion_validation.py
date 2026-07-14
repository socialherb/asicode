"""
Unit tests for the "next-step ghost suggestion" validator
(_validate_next_suggestion / _text_has_hangul in asi.py).

Regression focus: a helper model that ignores _NEXT_SUGGEST_SYSTEM's rules
(no preamble, same language as user, one short line) must NOT leak noise onto
the prompt. The observed bug was an English restatement of a Korean request
appearing as ghost text:

    First, the user said: '개선 필요성이 있다면 개선하자' which is Korean for ...

The validator enforces the system-prompt rules structurally (character-class
based — no keyword/regex heuristics). These tests lock that behavior and serve
as mutation guards: remove any guard and the corresponding test fails.
"""
import asi


# ─── _text_has_hangul ────────────────────────────────────────────────────────

def test_text_has_hangul_detects_syllables():
    assert asi._text_has_hangul("개선하자")
    assert asi._text_has_hangul("mix 한글 and ascii")


def test_text_has_hangul_detects_compat_jamo():
    # Compatibility Jamo block (3130–318F), e.g. ﾟ-free Hangul consonants.
    assert asi._text_has_hangul("\u3131")  # ㄱ


def test_text_has_hangul_rejects_ascii_and_cjk():
    assert not asi._text_has_hangul("run the tests now")
    # CJK / Kana are NOT Hangul — must not be misdetected.
    assert not asi._text_has_hangul("実行する")
    assert not asi._text_has_hangul("运行测试")


# ─── accepted (no false positives) ───────────────────────────────────────────

def test_validate_accepts_korean_imperative_for_korean_request():
    text = asi._validate_next_suggestion(
        "실패한 테스트 로그를 확인한다", "개선 필요성이 있다면 개선하자")
    assert text == "실패한 테스트 로그를 확인한다"


def test_validate_accepts_english_for_english_request():
    text = asi._validate_next_suggestion(
        "run the failing tests", "refactor the validator")
    assert text == "run the failing tests"


def test_validate_accepts_none_for_none_request():
    # No Hangul in request → script guard does not fire; empty/short ascii ok.
    text = asi._validate_next_suggestion("commit the changes", "")
    assert text == "commit the changes"


# ─── rejected — the observed bug class ───────────────────────────────────────

def test_validate_rejects_english_preamble_for_korean_request():
    """THE regression: the exact ghost sentence observed in the wild must be
    suppressed (returns None) — English reply to a Korean request."""
    bug = ("First, the user said: '개선 필요성이 있다면 개선하자' "
           "which is Korean for 'improve if needed.'")
    assert asi._validate_next_suggestion(bug, "개선 필요성이 있다면 개선하자") is None


def test_validate_rejects_all_ascii_for_hangul_request():
    assert asi._validate_next_suggestion(
        "The user wants me to improve the code", "코드를 개선해줘") is None


def test_validate_rejects_verbatim_echo_same_language():
    """Same-language guard: a suggestion that quotes the user's request back
    is a past-restatement, not a next step."""
    echo = "다음으로 '개선 필요성이 있다면 개선하자'를 처리합니다"
    assert asi._validate_next_suggestion(
        echo, "개선 필요성이 있다면 개선하자") is None


# ─── rejected — sentinel / shape ─────────────────────────────────────────────

def test_validate_rejects_none_sentinel_variants():
    for s in ("NONE", "none", "None.", "NONE!", "none?"):
        assert asi._validate_next_suggestion(s, "do something") is None


def test_validate_rejects_empty_and_whitespace():
    for s in ("", "   ", "\t\n"):
        assert asi._validate_next_suggestion(s, "anything") is None


def test_validate_rejects_too_long():
    long_text = "이것은 백사십 자를 넘기는 매우 긴 제안 문장입니다 " * 6
    assert len(long_text) > 140
    assert asi._validate_next_suggestion(long_text, "한글 요청") is None
