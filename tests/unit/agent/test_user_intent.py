"""Tests for _user_intent — IntentPattern, classify_user_approval."""

from __future__ import annotations

from external_llm.agent._user_intent import (
    _AFFIRMATIVE_PATTERN,
    _DENIAL_PATTERN,
    IntentPattern,
    UserApproval,
    classify_user_approval,
)


class TestIntentPattern:
    """Tests for IntentPattern dataclass and its match() method."""

    def test_empty_text_returns_false(self):
        pat = IntentPattern(name="test", keywords={"yes", "ok"})
        assert pat.match("") is False
        assert pat.match(None) is False

    def test_affirmative_match_positive(self):
        for kw in ["yes", "Yes", "YES", "y"]:
            assert _AFFIRMATIVE_PATTERN.match(kw), f"should match {kw!r}"

    def test_affirmative_match_strip(self):
        assert _AFFIRMATIVE_PATTERN.match("  yes  ")

    def test_affirmative_match_korean(self):
        assert _AFFIRMATIVE_PATTERN.match("네")
        assert _AFFIRMATIVE_PATTERN.match("응")
        assert _AFFIRMATIVE_PATTERN.match("ㅇㅇ")

    def test_word_boundary_not_prefix(self):
        """A keyword must NOT match as a prefix of a longer word.

        This is the safety-critical property: ``"y"`` must not match ``"you"`` /
        ``"yes"`` / ``"yesterday"``, and ``"no"`` must not match ``"now"`` /
        ``"nope"`` (nope is a separate keyword). Previously IntentPattern used a
        bare ``startswith``, which let natural-language denials be read as
        approval at the dangerous-command gate.
        """
        # affirmative single-char 'y' must not leak into longer words
        assert not _AFFIRMATIVE_PATTERN.match("you should not run it")
        assert not _AFFIRMATIVE_PATTERN.match("yesterday it failed")
        assert not _AFFIRMATIVE_PATTERN.match("yesman")
        # denial single-char 'n' must not leak
        assert not _DENIAL_PATTERN.match("now is fine")
        assert not _DENIAL_PATTERN.match("next time")
        # 'no' must not match 'nope' via prefix — 'nope' is its own keyword
        assert _DENIAL_PATTERN.match("nope")  # exact keyword still matches
        # trailing punctuation is a valid boundary
        assert _AFFIRMATIVE_PATTERN.match("y!")
        assert _DENIAL_PATTERN.match("n.")

    def test_denial_match_positive(self):
        for kw in ["no", "No", "NO", "nope"]:
            assert _DENIAL_PATTERN.match(kw), f"should match {kw!r}"

    def test_denial_match_korean(self):
        assert _DENIAL_PATTERN.match("아니")
        assert _DENIAL_PATTERN.match("안 돼")
        assert _DENIAL_PATTERN.match("ㄴㄴ")

    def test_denial_does_not_match_affirmative(self):
        assert not _DENIAL_PATTERN.match("yes")
        assert not _DENIAL_PATTERN.match("ok")

    def test_affirmative_does_not_match_denial(self):
        assert not _AFFIRMATIVE_PATTERN.match("no")

    def test_multi_word_affirmative(self):
        assert _AFFIRMATIVE_PATTERN.match("go ahead")
        assert _AFFIRMATIVE_PATTERN.match("do it now")
        assert _AFFIRMATIVE_PATTERN.match("that's right")

    def test_multi_word_denial(self):
        assert _DENIAL_PATTERN.match("never mind")
        assert _DENIAL_PATTERN.match("nevermind")

    def test_dont_variants(self):
        assert _DENIAL_PATTERN.match("don't")
        assert _DENIAL_PATTERN.match("dont")


class TestClassifyUserApproval:
    """Tests for classify_user_approval(response, judge_fn=None)."""

    def test_empty_or_none_is_ambiguous(self):
        assert classify_user_approval("") == UserApproval.AMBIGUOUS
        assert classify_user_approval("   ") == UserApproval.AMBIGUOUS
        assert classify_user_approval(None) == UserApproval.AMBIGUOUS

    def test_heuristic_affirmative(self):
        assert classify_user_approval("yes") == UserApproval.APPROVED
        assert classify_user_approval("네") == UserApproval.APPROVED
        assert classify_user_approval("OK") == UserApproval.APPROVED

    def test_heuristic_denial(self):
        assert classify_user_approval("no") == UserApproval.DENIED
        assert classify_user_approval("아니") == UserApproval.DENIED
        assert classify_user_approval("stop") == UserApproval.DENIED

    def test_heuristic_ambiguous_fallback(self):
        """Unrecognized text without judge_fn → AMBIGUOUS."""
        assert classify_user_approval("maybe") == UserApproval.AMBIGUOUS
        assert classify_user_approval("i don't know") == UserApproval.AMBIGUOUS
        assert classify_user_approval("난몰라") == UserApproval.AMBIGUOUS

    def test_unambiguous_affirmative_approved(self):
        """A clean 'yes' is approved regardless of check ordering."""
        assert classify_user_approval("yes") == UserApproval.APPROVED

    def test_no_false_positive_approval_on_word_prefix(self):
        """Regression: prefix-matching must not approve denials.

        These inputs were previously classified APPROVED (or DENIED via 'n')
        because IntentPattern used bare ``startswith``. With word-boundary
        matching they are AMBIGUOUS — and AMBIGUOUS != APPROVED, so the
        dangerous-command gate denies them. This is the safety contract.
        """
        # Was APPROVED via 'y' prefix:
        assert classify_user_approval("you should not run it") == UserApproval.AMBIGUOUS
        assert classify_user_approval("yesterday it failed") == UserApproval.AMBIGUOUS
        # Was APPROVED via 'correct' prefix — conservative residual case:
        assert classify_user_approval("nice, go ahead") == UserApproval.AMBIGUOUS

    def test_denial_checked_before_affirmative(self):
        """Denial is evaluated first — conservative for the danger gate.

        With word-boundary matching the two patterns rarely collide on a single
        response, so this mainly documents and locks the ordering intent. A
        leading denial token always yields DENIED.
        """
        assert classify_user_approval("dont") == UserApproval.DENIED
        assert classify_user_approval("no") == UserApproval.DENIED

    def test_judge_fn_approved(self):
        def judge(prompt: str) -> str:
            return "approved"
        assert classify_user_approval("yes!", judge) == UserApproval.APPROVED

    def test_judge_fn_denied(self):
        def judge(prompt: str) -> str:
            return "denied"
        assert classify_user_approval("no!", judge) == UserApproval.DENIED

    def test_judge_fn_ambiguous(self):
        def judge(prompt: str) -> str:
            return "something else"
        assert classify_user_approval("hmm", judge) == UserApproval.AMBIGUOUS

    def test_judge_fn_strips_whitespace(self):
        def judge(prompt: str) -> str:
            return "  approved  "
        assert classify_user_approval("ok", judge) == UserApproval.APPROVED

    def test_judge_fn_exception_falls_back_to_heuristic(self):
        """When judge_fn raises, fall back to heuristic."""
        def broken_judge(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")
        # falls back to heuristic → "yes" is approved
        response = "yes"
        assert classify_user_approval(response, broken_judge) == UserApproval.APPROVED

    def test_judge_fn_exception_with_ambiguous_heuristic(self):
        """When judge_fn raises and heuristic doesn't match → AMBIGUOUS."""
        def broken_judge(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")
        assert classify_user_approval("maybe later", broken_judge) == UserApproval.AMBIGUOUS

    def test_judge_fn_prompt_format(self):
        """Verify the prompt template reaches judge_fn correctly."""
        captured: list[str] = []

        def capturing_judge(prompt: str) -> str:
            captured.append(prompt)
            return "approved"

        classify_user_approval("do it", capturing_judge)
        assert len(captured) == 1
        assert "do it" in captured[0]
        assert "approved" in captured[0]


class TestMixedResponseDemotion:
    """An affirmative prefix that the remainder qualifies is NOT a confident approval.

    Safety-critical: at the dangerous-command gate a mixed reply like
    "yes, but don't run it yet" must NOT be approved. The demotion scans the
    whole response for a denial/strong-negation token (word-boundary) and, on a
    hit, returns AMBIGUOUS (== deny).
    """

    def test_affirmative_then_explicit_dont_demoted(self):
        assert classify_user_approval("yes, but don't run it yet") == UserApproval.AMBIGUOUS

    def test_correct_then_do_not_demoted(self):
        assert classify_user_approval("correct the bug first, do not apply yet") == UserApproval.AMBIGUOUS

    def test_yes_not_yet_demoted(self):
        assert classify_user_approval("yes, not yet") == UserApproval.AMBIGUOUS

    def test_ok_wont_demoted(self):
        assert classify_user_approval("ok, won't proceed") == UserApproval.AMBIGUOUS

    def test_clean_affirmative_not_demoted(self):
        """A bare affirmative (no denial token) stays APPROVED."""
        assert classify_user_approval("yes") == UserApproval.APPROVED
        assert classify_user_approval("yes!") == UserApproval.APPROVED
        assert classify_user_approval("sure, definitely") == UserApproval.APPROVED

    def test_sure_not_a_problem_approved(self):
        """Affirmative idiom "not a problem" has no STRONG-negation token → APPROVED.

        Only explicit negation constructions (don't/do not/not yet/won't/...) trigger
        the demotion; bare "not" inside an affirmative idiom does not, so common
        approvals are not falsely denied.
        """
        assert classify_user_approval("sure, not a problem") == UserApproval.APPROVED

    def test_no_worries_conservatively_demoted(self):
        """Conservative trade-off: "no" inside "no worries" demotes → AMBIGUOUS.

        The literal token "no" is a denial signal; at a DANGER gate we prefer a
        safe false-deny (re-confirm) over a false-approval of a "yes... no..."
        self-correction. Documented intentional behavior.
        """
        assert classify_user_approval("ok, no worries go ahead") == UserApproval.AMBIGUOUS

    def test_dont_prefix_still_denied(self):
        """A leading denial still wins (denial-first) — not demoted to AMBIGUOUS."""
        assert classify_user_approval("don't, actually yes") == UserApproval.DENIED


class TestKoreanInflectedCoverage:
    """Politeness/verb-ending Korean forms that word-boundary (isalnum) would miss."""

    def test_korean_affirmative_endings(self):
        for kw in ["그래요", "좋아요", "맞아요", "진행해", "해줘", "실행해", "ㄱㄱ"]:
            assert classify_user_approval(kw) == UserApproval.APPROVED, f"{kw!r} should be APPROVED"

    def test_korean_denial_endings(self):
        for kw in ["아니요", "아니오", "아뇨", "취소", "취소해"]:
            assert classify_user_approval(kw) == UserApproval.DENIED, f"{kw!r} should be DENIED"


class TestApostropheNormalization:
    """Curly/smart apostrophes (don't → don't) must match the ASCII keyword."""

    def test_curly_apostrophe_dont_denied(self):
        assert classify_user_approval("don\u2019t run it") == UserApproval.DENIED

    def test_curly_apostrophe_in_mixed_demoted(self):
        # "yes, but don't-run-it-yet" with a smart quote → still demoted
        assert classify_user_approval("yes, but don\u2019t run it yet") == UserApproval.AMBIGUOUS

    def test_matches_anywhere_word_boundary_single_char(self):
        """matches_anywhere must use word boundaries so 'n'/'no' don't fire mid-word."""
        # 'n' inside "don't" / "any" must NOT count as a standalone denial token
        assert _DENIAL_PATTERN.matches_anywhere("yes, any time") is False
        # 'no' inside "nothing" must not fire
        assert _DENIAL_PATTERN.matches_anywhere("ok, nothing else") is False
        # but a standalone 'no' token does
        assert _DENIAL_PATTERN.matches_anywhere("ok, no go") is True
