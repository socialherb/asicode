"""Behavior tests for extracted insights-compact pure functions.

Tests cover the three module-level helpers in ``asi`` that were extracted
from the ``_compact_insights_interactive`` closure for testability:

- ``_insights_compact_is_noop`` — no-op detection (normalized text + entry count)
- ``_size_compact_budget`` — max_tokens budget from content
- ``_dropped_entries`` — stable-identity-based entry loss detection

These are pure deterministic functions — no mocking needed.
"""
import pytest
from asi import (
    _insights_compact_is_noop,
    _size_compact_budget,
    _dropped_entries,
)
from external_llm.agent.insights_manager import InsightEntry


# ═══════════════════════════════════════════════════════════════
# _insights_compact_is_noop
# ═══════════════════════════════════════════════════════════════

def _mk_ent(header: str) -> InsightEntry:
    return InsightEntry(
        lines=[header + "\n", "  body text\n"],
        header_line=header,
        category="pattern",
    )


class TestIsNoop:
    def test_exact_same_content(self):
        h = "### [pattern] 2026-01-01"
        e = [_mk_ent(h)]
        assert _insights_compact_is_noop(h + "\nbody\n", h + "\nbody\n", e, e) is True

    def test_whitespace_collapsed(self):
        """Whitespace differences alone do NOT count as a change."""
        h = "### [pattern] 2026-01-01"
        e = [_mk_ent(h)]
        assert _insights_compact_is_noop(
            "  hello   world\nfoo",
            "hello world\n  foo",
            e, e,
        ) is True

    def test_content_differs(self):
        """Different body text → not no-op."""
        h = "### [pattern] 2026-01-01"
        e1 = [_mk_ent(h)]
        e2 = [_mk_ent(h)]  # same header, but body differs
        assert _insights_compact_is_noop(
            "### [pattern] 2026-01-01\nold body",
            "### [pattern] 2026-01-01\nnew body",
            e2, e1,
        ) is False

    def test_same_text_fewer_entries(self):
        """Same normalized text but fewer entries → not no-op (loss detected)."""
        h1 = "### [pattern] 2026-01-01"
        h2 = "### [pattern] 2026-01-02"
        before = [_mk_ent(h1), _mk_ent(h2)]
        after = [_mk_ent(h1)]
        text = "same text"
        assert _insights_compact_is_noop(text, text, after, before) is False

    def test_header_preserving_compact(self):
        """Compactor shortened body while preserving headers → not no-op."""
        h = "### [pattern] 2026-01-01"
        e = [_mk_ent(h)]
        assert _insights_compact_is_noop(
            "### [pattern] 2026-01-01\nlong body text here",
            "### [pattern] 2026-01-01\nshort",
            e, e,
        ) is False

    def test_empty_content(self):
        """Both empty → no-op."""
        assert _insights_compact_is_noop("", "", [], []) is True

    def test_one_empty_one_not(self):
        """One empty, one not → not no-op."""
        h = "### [pattern] 2026-01-01"
        e = [_mk_ent(h)]
        assert _insights_compact_is_noop("", "### [pattern] 2026-01-01\nbody", [], e) is False


# ═══════════════════════════════════════════════════════════════
# _size_compact_budget
# ═══════════════════════════════════════════════════════════════

class TestSizeCompactBudget:
    def test_empty_content(self):
        """Empty content → floor of 8192."""
        assert _size_compact_budget("") == 8192

    def test_short_content(self):
        """Short content (< ~12k bytes) → floor of 8192."""
        assert _size_compact_budget("hello world") == 8192
        assert _size_compact_budget("a" * 100) == 8192  # 100 bytes → 50 + 2048 < 8192

    def test_large_ascii(self):
        """Large ASCII → bytes/2 + 2048."""
        # 14000 ASCII bytes → 7000 tokens + 2048 = 9048
        assert _size_compact_budget("a" * 14000) == 9048

    def test_cjk_content(self):
        """CJK (3 bytes/char) → properly scaled."""
        # 2000 CJK chars = 6000 bytes → 3000 tokens + 2048 = 5048 (< 8192, floor)
        assert _size_compact_budget("안" * 2000) == 8192  # floor
        # 6000 CJK chars = 18000 bytes → 9000 tokens + 2048 = 11048
        assert _size_compact_budget("안" * 6000) == 11048

    def test_mixed_content(self):
        """Mixed ASCII + CJK → aggregate byte-based."""
        s = "hello world" + "안녕하세요" * 1000  # ~11 + 15000 = ~15011 bytes
        expected = 15011 // 2 + 2048  # 7505 + 2048 = 9553
        assert _size_compact_budget(s) == 9553


# ═══════════════════════════════════════════════════════════════
# _dropped_entries
# ═══════════════════════════════════════════════════════════════

class TestDroppedEntries:
    def test_none_dropped(self):
        """All entries preserved → empty list."""
        h = "### [pattern] 2026-01-01"
        e = [_mk_ent(h)]
        assert _dropped_entries(e, e) == []

    def test_some_dropped(self):
        """Some entries missing → those returned."""
        h1 = "### [pattern] 2026-01-01"
        h2 = "### [pattern] 2026-01-02"
        h3 = "### [pattern] 2026-01-03"
        before = [_mk_ent(h1), _mk_ent(h2), _mk_ent(h3)]
        after = [_mk_ent(h1), _mk_ent(h3)]
        dropped = _dropped_entries(before, after)
        assert len(dropped) == 1
        assert dropped[0].header_line == h2

    def test_position_independent(self):
        """Uses header_line identity, not position — order or position doesn't matter."""
        h1 = "### [pattern] 2026-01-01"
        h2 = "### [pattern] 2026-01-02"
        before = [_mk_ent(h1), _mk_ent(h2)]
        # After has h2 but not h1 — dropped is h1 (not the first N entries)
        after = [_mk_ent(h2)]
        dropped = _dropped_entries(before, after)
        assert len(dropped) == 1
        assert dropped[0].header_line == h1

    def test_all_dropped(self):
        """All entries missing → all returned."""
        h1 = "### [pattern] 2026-01-01"
        h2 = "### [pattern] 2026-01-02"
        before = [_mk_ent(h1), _mk_ent(h2)]
        dropped = _dropped_entries(before, [])
        assert len(dropped) == 2
        assert {e.header_line for e in dropped} == {h1, h2}

    def test_empty_before(self):
        """No entries before → empty list."""
        after = [_mk_ent("### [pattern] 2026-01-01")]
        assert _dropped_entries([], after) == []

    def test_entry_without_header_line(self):
        """Entry with empty header_line can still be compared."""
        e = InsightEntry(lines=[], header_line="", category="")
        assert _dropped_entries([e], []) == [e]
        assert _dropped_entries([e], [e]) == []


if __name__ == "__main__":
    pytest.main([__file__])
