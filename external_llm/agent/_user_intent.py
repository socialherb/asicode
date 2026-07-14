"""
User intent classification utilities.

Interprets a user's free-text response to a yes/no question, returning
approved / denied / ambiguous.

Two paths:
  * ``judge_fn`` (optional) — delegate interpretation to an LLM so natural-
    language variations, multi-language responses, and edge cases are handled
    structurally rather than enumerated. Used when a caller supplies one.
  * Heuristic classifier (default) — a conservative, deterministic, typed-
    keyword matcher. This is *first-class*, not merely a "last resort": the
    dangerous-command gate (git_tools) deliberately relies on it WITHOUT a
    judge_fn because a safety gate must be fast, predictable, and never
    hallucinate an approval. The heuristic's contract is deny-on-ambiguity:
    when it cannot confidently match a clean approval, it returns AMBIGUOUS,
    which the gate treats as deny. A mixed/conditional reply (affirmative
    prefix that the remainder qualifies with a denial token) is demoted to
    AMBIGUOUS for the same reason.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Normalize curly/smart apostrophes (U+2019 etc.) to ASCII "'" so that
# ``"don't"`` (typed with a smart quote) still matches the denial keyword
# ``"don't"``. Apostrophe variants are the only common case where Unicode
# confusables break Latin keyword matching; full Unicode confusable folding
# is deliberately out of scope (it would weaken the conservative bias).
_APOSTROPHE_VARIANTS = "'\u02bc\u2018\u2019\u02b9\u0060\u00b4"
_APOSTROPHE_TRANS = str.maketrans({c: "'" for c in _APOSTROPHE_VARIANTS})


def _normalize(text: str) -> str:
    """Fold smart apostrophes to ASCII so keyword matching is quote-agnostic."""
    return text.translate(_APOSTROPHE_TRANS) if text else text


class UserApproval(Enum):
    APPROVED = "approved"
    DENIED = "denied"
    AMBIGUOUS = "ambiguous"


_CLASSIFICATION_PROMPT = """\
You are classifying a user's response to a yes/no question.
Respond with exactly one word: approved, denied, or ambiguous.

Rules:
- approved  → clearly affirmative (yes, sure, go ahead, ok, 네, 응, do it, proceed, etc.)
- denied    → clearly negative (no, don't, stop, 아니, 하지마, cancel, etc.)
- ambiguous → unclear, conditional, unrelated, or contains both affirmation and hesitation

User response: {response}

Classification:"""


@dataclass
class IntentPattern:
    """Typed keyword pattern for user intent classification.

    Replaces ad-hoc regex patterns with a structured dataclass.
    Each pattern holds a set of keywords matched at word boundaries
    (anchored to start of line).
    """
    name: str
    keywords: set[str]

    def match(self, text: str) -> bool:
        """Check if *text* starts with any keyword at a word boundary.

        Conservative matcher — a keyword can NEVER spuriously match a longer
        word. A keyword matches only when the response *starts* with it AND is
        immediately followed by a word boundary (end-of-text, whitespace, or a
        non-alphanumeric character). This prevents ``"y"`` from matching
        ``"you should not"`` or ``"yes"`` from matching ``"yesterday"``.

        ``str.isalnum`` is used for the boundary check so CJK/Hangul characters
        (``네``, ``아니``, ``ㅇ``) are treated as part of a word — ``"y"`` thus
        never matches ``"yes"`` and ``"no"`` never matches ``"now"``.

        This makes the heuristic safe as the SOLE gate for dangerous commands:
        when it cannot confidently match, it returns False (→ AMBIGUOUS → deny).
        """
        if not text:
            return False
        stripped = _normalize(text.strip()).lower()
        for kw in self.keywords:
            kw_l = kw.lower()
            if stripped == kw_l:
                return True
            if not stripped.startswith(kw_l):
                continue
            # Keyword is a prefix — require a word boundary right after it so
            # we never match "y" in "you" or "no" in "now". A non-alphanumeric
            # char after the keyword (space, punctuation, ...) is a boundary.
            next_char = stripped[len(kw_l):len(kw_l) + 1]
            if not next_char or not next_char.isalnum():
                return True
        return False

    def matches_anywhere(self, text: str) -> bool:
        """True if any keyword appears as a word-boundary token *anywhere* in text.

        Unlike :meth:`match` (which anchors to the start of the response), this
        scans the whole text. It is used ONLY to *demote* an affirmative prefix
        match to AMBIGUOUS when the response also contains a denial token — i.e.
        a mixed/conditional reply such as ``"yes, but don't run it yet"`` or
        ``"correct, just don't apply it yet"``. A keyword counts as present only
        when it is surrounded by word boundaries (start/end of text or a
        non-alphanumeric char), so the single-char keywords ``"y"``/``"n"`` do
        not fire inside longer words ("any", "noted", "don't"). This makes the
        demotion conservative: a match here makes a classification MORE
        cautious (→ AMBIGUOUS → deny at the gate), never less.
        """
        if not text:
            return False
        s = _normalize(text.strip()).lower()
        for kw in self.keywords:
            kl = kw.lower()
            start = 0
            while True:
                idx = s.find(kl, start)
                if idx == -1:
                    break
                before_ok = idx == 0 or not s[idx - 1].isalnum()
                after = idx + len(kl)
                after_ok = after >= len(s) or not s[after].isalnum()
                if before_ok and after_ok:
                    return True
                start = idx + 1
        return False


# Typed intent patterns — replaces ad-hoc _AFFIRMATIVE_PATTERNS / _DENIAL_PATTERNS regex
_AFFIRMATIVE_PATTERN = IntentPattern(
    name="affirmative",
    keywords={
        "yes", "y", "yeah", "yep", "sure", "ok", "okay",
        "go ahead", "do it", "proceed", "approve", "allow", "run it",
        "correct", "right", "that's right", "that is right", "of course",
        "absolutely", "indeed",
        "네", "응", "맞아", "그래", "좋아", "응응", "넵", "예",
        "당연", "그렇지", "오케이", "ㅇㅇ", "ㅇ",
        # Korean inflected/verb forms. Hangul syllables are alnum for str, so a
 # bare keyword like "그래" does NOT boundary-match "그래요" (요 is alnum);
        # the politeness/verb-ending forms must be enumerated explicitly.
        "좋아요", "그래요", "맞아요", "그럼", "네네", "ㄱㄱ",
        "진행", "진행해", "해줘", "실행", "실행해", "시작해", "가요",
    },
)

_DENIAL_PATTERN = IntentPattern(
    name="denial",
    keywords={
        "no", "n", "nope", "nah", "don't", "dont", "stop", "cancel",
        "deny", "false", "abort", "never mind", "nevermind",
        "아니", "안 돼", "안돼", "하지마", "그만", "싫어", "노", "ㄴㄴ",
        # Politeness/verb-ending forms — see the affirmative block note.
        "아니요", "아니오", "아뇨", "안돼요", "하지마요",
        "취소", "취소해", "그만둬", "중단", "중단해",
        # Explicit negation constructions. These are added (rather than bare
        # "not"/"but") because they are unambiguously negative/withholding, so a
        # legitimate affirmation like "sure, not a problem" / "ok, no worries"
        # is NOT falsely denied — "not a problem" contains no such construction.
        # Word-boundary matching keeps them safe ("do not" won't fire inside
        # "do nothing"; "never" won't fire inside "nevertheless").
        "do not", "not yet", "should not", "shouldn't",
        "won't", "will not", "can't", "cannot", "never",
    },
)


def classify_user_approval(
    response: str,
    judge_fn: Optional[Callable[[str], str]] = None,
) -> UserApproval:
    """Interpret a user's free-text response as approval, denial, or ambiguous.

    Args:
        response: The raw text the user entered.
        judge_fn: Optional callable that takes a prompt string and returns
            a short classification string.  When omitted, a simple heuristic
            fallback is used.

    Returns:
        UserApproval.APPROVED, .DENIED, or .AMBIGUOUS.
    """
    text = _normalize((response or "").strip())
    if not text:
        return UserApproval.AMBIGUOUS

    if judge_fn is not None:
        try:
            prompt = _CLASSIFICATION_PROMPT.format(response=text)
            raw = (judge_fn(prompt) or "").strip().lower()
            if raw == "approved":
                return UserApproval.APPROVED
            elif raw == "denied":
                return UserApproval.DENIED
            else:
                logger.debug("LLM judge returned ambiguous classification: %r", raw)
                return UserApproval.AMBIGUOUS
        except Exception as e:
            logger.warning("LLM judge failed, falling back to heuristic: %s", e)

    # Heuristic classifier (default path; see module docstring). Denial is
    # checked BEFORE affirmative: this is the conservative choice for a gate
    # that protects dangerous shell commands — when a reading is mixed or
    # ambiguous, the negative interpretation must win (deny-by-default).
    if _DENIAL_PATTERN.match(text):
        return UserApproval.DENIED
    if _AFFIRMATIVE_PATTERN.match(text):
        # Mixed/conditional reply guard. An affirmative *prefix* match is not a
        # confident approval when the rest of the response also contains a
        # denial token — e.g. "yes, but don't run it yet", "correct, just don't
        # apply it yet", "ok, but not the destructive part". Word-boundary
        # matching keeps this from over-firing (single-char "n" won't trigger
        # inside "don't"/"noted"). At the danger gate AMBIGUOUS == deny, so a
        # demotion here can only make the verdict SAFER, never less safe.
        if _DENIAL_PATTERN.matches_anywhere(text):
            return UserApproval.AMBIGUOUS
        return UserApproval.APPROVED

    return UserApproval.AMBIGUOUS
