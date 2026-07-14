"""Tests for insights_manager — parse/serialize round-trip, drop, stats, nudge.

The load-bearing invariant is **round-trip losslessness**:
``serialize_insights(*parse_insights(x)) == x`` for every file shape, because
the /insights commands and the LLM compactor both trust that parsing never
silently alters or drops content. Both on-disk shapes are covered: the
``### [category] timestamp`` blocks emitted by ``_save_insight_to_file`` and the
hand-edited ``#`` title + ``>`` blockquote "원칙" preamble that coexists in the
wild.
"""
from __future__ import annotations

import os

import pytest

from external_llm.agent.insights_manager import (
    COMPACT_BUDGET_BYTES,
    NUDGE_AGE_DAYS_THRESHOLD,
    NUDGE_BYTES_THRESHOLD,
    NUDGE_COUNT_THRESHOLD,
    InsightsStats,
    _active_invalidate,
    atomic_write_text,
    build_compact_messages,
    compute_stats,
    drop_entry,
    enforce_budget_by_demotion,
    entry_age_days,
    insights_path,
    load_active_insights_cached,
    load_archive_file,
    load_insights_file,
    parse_insights,
    parse_timestamp,
    select_demotion_candidates,
    select_entries_older_than,
    select_promotable_entries,
    build_archive_index,
    serialize_insights,
    should_nudge,
)


@pytest.fixture
def tmp_repo(tmp_path):
    """Temporary repo root with a ``.asicode`` directory."""
    (tmp_path / ".asicode").mkdir()
    return str(tmp_path)


# ── Round-trip losslessness ───────────────────────────────────────────────────

# Shape A: hand-edited prose preamble (the real current file shape).
HAND_EDITED = (
    "# Design Chat Insights (.asicode/design_insights.md)\n"
    "\n"
    "> **원칙**: 구조적/일반적 접근 (AST/graph/enum/typed policy 기반). "
    "키워드/regex 매칭 지양.\n"
    "> 핵심 설계 패턴/제약만 유지. 구현 전투 기록은 보관하지 않음.\n"
)

# Shape B: machine-emitted blocks (what _save_insight_to_file writes).
MACHINE_EMITTED = (
    "# Design Chat Insights\n\n"
    "Discoveries and insights saved by the design chat LLM across sessions.\n\n"
    "### [architecture] 2025-01-15 14:30\n"
    "Context compression runs every 15 turns, MIN_RECENT_TURNS_KEEP=4.\n\n"
    "### [gotcha] 2025-01-16 09:12\n"
    "UnicodeEncodeError is a ValueError subclass — escapes `except OSError`.\n\n"
    "### [pattern] 2025-01-16 10:00\n"
    "Prefer start_new_session=True + killpg for subprocess timeout safety.\n\n"
)

# Shape C: mixed — prose preamble + machine blocks (most realistic).
MIXED = HAND_EDITED + "### [bug] 2025-01-17 11:00\nrollback loop must use os.replace.\n\n"

# Shape D: preamble only, no entries at all.
PREAMBLE_ONLY = HAND_EDITED


@pytest.mark.parametrize(
    "content",
    [HAND_EDITED, MACHINE_EMITTED, MIXED, PREAMBLE_ONLY, "", "### \n", "no headers at all\n"],
    ids=["hand-edited", "machine-emitted", "mixed", "preamble-only", "empty", "bare-header", "no-headers"],
)
def test_roundtrip_lossless(content):
    """serialize(parse(x)) must equal x for every shape — nothing dropped/rewritten."""
    preamble, entries = parse_insights(content)
    assert serialize_insights(preamble, entries) == content


# ── parse_insights structure ──────────────────────────────────────────────────

def test_parse_separates_preamble_from_entries():
    preamble, entries = parse_insights(MIXED)
    assert "".join(preamble) == HAND_EDITED
    assert len(entries) == 1
    assert entries[0].category == "bug"
    assert "rollback loop" in entries[0].body


def test_parse_machine_emitted_has_three_entries():
    _, entries = parse_insights(MACHINE_EMITTED)
    assert [e.category for e in entries] == ["architecture", "gotcha", "pattern"]


def test_parse_empty_file_has_no_entries():
    preamble, entries = parse_insights("")
    assert preamble == []
    assert entries == []


def test_parse_preamble_only_keeps_all_lines():
    preamble, entries = parse_insights(PREAMBLE_ONLY)
    assert "".join(preamble) == PREAMBLE_ONLY
    assert entries == []


def test_parse_bare_header_is_one_entry_no_category():
    preamble, entries = parse_insights("### \n")
    assert preamble == []
    assert len(entries) == 1
    assert entries[0].category == ""


def test_entry_text_includes_header_line():
    _, entries = parse_insights(MACHINE_EMITTED)
    assert entries[0].text.startswith("### [architecture]")


# ── drop_entry ────────────────────────────────────────────────────────────────

def test_drop_removes_entry_by_1based_index():
    _, entries = parse_insights(MACHINE_EMITTED)
    dropped = drop_entry(entries, 2)
    assert [e.category for e in dropped] == ["architecture", "pattern"]


def test_drop_first_and_last():
    _, entries = parse_insights(MACHINE_EMITTED)
    assert [e.category for e in drop_entry(entries, 1)] == ["gotcha", "pattern"]
    assert [e.category for e in drop_entry(entries, 3)] == ["architecture", "gotcha"]


def test_drop_out_of_range_is_noop():
    _, entries = parse_insights(MACHINE_EMITTED)
    assert drop_entry(entries, 0) == entries
    assert drop_entry(entries, 99) == entries
    # returns a copy, not the same object
    assert drop_entry(entries, 99) is not entries


def test_drop_then_serialize_preserves_rest():
    _, entries = parse_insights(MACHINE_EMITTED)
    preamble, _ = parse_insights(MACHINE_EMITTED)
    dropped = drop_entry(entries, 1)
    result = serialize_insights(preamble, dropped)
    assert "### [architecture]" not in result
    assert "### [gotcha]" in result
    assert "### [pattern]" in result


# ── compute_stats ─────────────────────────────────────────────────────────────

def test_stats_missing_file(tmp_repo):
    stats = compute_stats(tmp_repo)
    assert stats.exists is False
    assert stats.count == 0


def test_stats_counts_entries_and_size(tmp_repo):
    path = insights_path(tmp_repo)
    atomic_write_text(path, MACHINE_EMITTED)
    stats = compute_stats(tmp_repo)
    assert stats.exists is True
    assert stats.count == 3
    assert stats.bytes_size == len(MACHINE_EMITTED.encode("utf-8"))
    assert stats.tokens == max(1, int(len(MACHINE_EMITTED) / 3.5))
    assert stats.mtime is not None
    assert stats.age_days is not None
    assert stats.age_days >= 0


def test_stats_preamble_only_counts_zero_entries(tmp_repo):
    atomic_write_text(insights_path(tmp_repo), HAND_EDITED)
    stats = compute_stats(tmp_repo)
    assert stats.count == 0
    assert stats.exists is True


# ── should_nudge ──────────────────────────────────────────────────────────────

def test_nudge_silent_when_file_missing(tmp_repo):
    stats = compute_stats(tmp_repo)
    fire, msg = should_nudge(stats)
    assert fire is False
    assert msg == ""


def test_nudge_fires_on_count_threshold(tmp_repo):
    stats = InsightsStats(
        exists=True,
        count=NUDGE_COUNT_THRESHOLD,
        bytes_size=100,
        age_days=1,
    )
    fire, msg = should_nudge(stats)
    assert fire is True
    assert f"{NUDGE_COUNT_THRESHOLD} items" in msg
    assert "{list|verify|compact|drop <n>}" in msg


def test_nudge_fires_on_bytes_threshold():
    stats = InsightsStats(
        exists=True,
        count=1,
        bytes_size=NUDGE_BYTES_THRESHOLD,
        age_days=1,
    )
    fire, msg = should_nudge(stats)
    assert fire is True
    assert "bytes" in msg
    assert "tokens" in msg


def test_nudge_fires_on_age_threshold():
    stats = InsightsStats(
        exists=True,
        count=1,
        bytes_size=100,
        age_days=NUDGE_AGE_DAYS_THRESHOLD + 5,
    )
    fire, msg = should_nudge(stats)
    assert fire is True
    assert "days old" in msg


def test_nudge_silent_below_all_thresholds():
    stats = InsightsStats(
        exists=True,
        count=NUDGE_COUNT_THRESHOLD - 1,
        bytes_size=NUDGE_BYTES_THRESHOLD - 1,
        age_days=NUDGE_AGE_DAYS_THRESHOLD - 1,
    )
    fire, msg = should_nudge(stats)
    assert fire is False
    assert msg == ""


def test_nudge_message_lists_all_crossed_thresholds():
    stats = InsightsStats(
        exists=True,
        count=NUDGE_COUNT_THRESHOLD + 5,
        bytes_size=NUDGE_BYTES_THRESHOLD + 1000,
        age_days=NUDGE_AGE_DAYS_THRESHOLD + 10,
    )
    fire, msg = should_nudge(stats)
    assert fire is True
    assert "items" in msg
    assert "bytes" in msg
    assert "tokens" in msg
    assert "days old" in msg


# ── atomic_write_text ─────────────────────────────────────────────────────────

def test_atomic_write_creates_and_overwrites(tmp_repo):
    path = insights_path(tmp_repo)
    atomic_write_text(path, "first content")
    assert load_insights_file(tmp_repo) == "first content"
    atomic_write_text(path, "second content\nwith newline\n")
    assert load_insights_file(tmp_repo) == "second content\nwith newline\n"


def test_atomic_write_creates_parent_dir(tmp_path):
    # .asicode does not exist yet
    repo = str(tmp_path)
    path = insights_path(repo)
    atomic_write_text(path, "created")
    assert os.path.exists(path)
    assert load_insights_file(repo) == "created"


def test_atomic_write_preserves_unicode(tmp_repo):
    path = insights_path(tmp_repo)
    text = "### [설계결정] 한국어 내용과 emoji 🎯\n본문입니다.\n"
    atomic_write_text(path, text)
    assert load_insights_file(tmp_repo) == text


# ── build_compact_messages ────────────────────────────────────────────────────

def test_compact_messages_structure():
    msgs = build_compact_messages(MACHINE_EMITTED)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert MACHINE_EMITTED in msgs[1]["content"]


def test_compact_messages_preserve_preamble_instruction():
    msgs = build_compact_messages(HAND_EDITED)
    system = msgs[0]["content"]
    # The compactor must be told to keep the '원칙' preamble verbatim.
    assert "VERBATIM" in system or "verbatim" in system
    assert "원칙" in system


def test_compact_messages_no_invent_instruction():
    system = build_compact_messages("x")[0]["content"]
    assert "NOT invent" in system or "Do not invent" in system


def test_compact_messages_has_value_triage():
    """Compaction must distinguish long-term-KEEP from ephemeral-DROP, not
    just dedup. The contract: both a KEEP criterion and a DROP criterion are
    named so the model makes a value judgment per entry."""
    system = build_compact_messages("x")[0]["content"].upper()
    assert "KEEP" in system and "DROP" in system
    # KEEP must be anchored to long-term architectural value.
    assert any(tok in system for tok in ("LONG-TERM", "ARCHITECTURAL", "FUTURE SESSION"))


def test_compact_messages_addresses_superseded():
    """When a later entry resolves an earlier one, the earlier must be
    dropped — the contract forbids preserving both (which yields a
    contradiction the next session can't reconcile)."""
    system = build_compact_messages("x")[0]["content"]
    assert "SUPERSEDED" in system.upper() or "superseded" in system.lower()


def test_compact_messages_drops_implementation_narrative():
    """KEEP entries must be stripped of implementation/debug battle records
    (step-by-step troubleshooting, commit-hash chasing, ad-hoc verification).
    Aligns with the file's 원칙: '구현 전투 기록은 보관하지 않음'."""
    system = build_compact_messages("x")[0]["content"].lower()
    assert "narrative" in system or "troubleshooting" in system
    assert "battle records" in system


def test_compact_messages_not_blind_keep():
    """The old 'when unsure, KEEP' rule is GONE — it preserved stale
    workarounds. The contract must not instruct unconditional retention."""
    system = build_compact_messages("x")[0]["content"].lower()
    assert "when unsure, keep" not in system
    assert "when unsure keep" not in system


def test_compact_messages_no_budget_directive_when_under_or_unset():
    """Under-budget (or no budget passed) ⇒ the MANDATORY reduction directive
    must NOT appear — behavior stays the original value-triage. This guards the
    backward-compat default and the normal steady state."""
    # No budget passed (default) → no directive.
    assert "BUDGET ENFORCEMENT" not in build_compact_messages("x")[0]["content"]
    # Budget passed but content well under it → no directive.
    under = "### [architecture] 2026-06-01 00:00\nshort invariant\n"
    assert "BUDGET ENFORCEMENT" not in build_compact_messages(
        under, budget_bytes=10_000)[0]["content"]


def test_compact_messages_enforces_budget_when_over():
    """Over-budget content + budget ⇒ the system prompt MUST carry a MANDATORY
    reduction directive. This is the fix for the 'already compact' trap: an
    all-durable file that trips the size nudge must still be tightened by the
    compactor rather than echoed unchanged."""
    # Build content guaranteed over a small budget.
    over = "### [architecture] 2026-06-01 00:00\n" + ("durable invariant " * 200) + "\n"
    assert len(over.encode("utf-8")) > 500  # sanity
    system = build_compact_messages(over, budget_bytes=500)[0]["content"].upper()
    assert "BUDGET ENFORCEMENT (MANDATORY)" in system
    assert "OVER THE 500-BYTE BUDGET" in system
    # Reduction is by MERGING + tightening, NOT by dropping durable facts.
    assert "MERGING" in system
    # The directive must forbid dropping valid facts just to hit the budget —
    # otherwise compact devolves into silent data loss.
    assert "DO NOT DROP A" in system and "STILL-VALID DURABLE FACT" in system


def test_compact_budget_bytes_equals_nudge_bytes_threshold():
    """Single source of truth: the compact budget and the size-nudge threshold
    are the SAME value. If these drift, the nudge warns about a size that
    compact no longer targets — the closed loop reopens."""
    assert COMPACT_BUDGET_BYTES == NUDGE_BYTES_THRESHOLD


# ── Integration with the real _save_insight_to_file shape ─────────────────────

def test_real_saved_file_roundtrips(tmp_repo):
    """The file _save_insight_to_file actually produces must round-trip."""
    from external_llm.agent.design_chat_loop import _save_insight_to_file

    _save_insight_to_file(tmp_repo, "First real insight.", "architecture")
    _save_insight_to_file(tmp_repo, "Second real insight.", "bug")
    content = load_insights_file(tmp_repo)
    # Round-trip lossless
    preamble, entries = parse_insights(content)
    assert serialize_insights(preamble, entries) == content
    assert len(entries) == 2
    assert [e.category for e in entries] == ["architecture", "bug"]


# ── CLI edit reconstruction round-trip (regression for body= crash + line damage) ──

def test_edit_reconstruction_roundtrips():
    """Simulate what ``/insights edit <n> <new_body>`` reconstructs and verify
    round-trip losslessness — this catches the crash from passing ``body=``
    (a @property, not a field) as well as header-line corruption."""
    from external_llm.agent.insights_manager import InsightEntry

    # Simulate an existing entry as parsed from a real file
    original_block = "### [architecture] 2025-06-28 10:32\ninstanced → module-level state promotion\n"
    preamble, entries = parse_insights(original_block)
    assert len(entries) == 1
    old = entries[0]

    # Reconstruct the way the fixed /insights edit does
    new_body = "new body line\n"
    hdr = old.lines[0]
    if not hdr.endswith("\n"):
        hdr += "\n"
    new_ent = InsightEntry(
        lines=[hdr, new_body.rstrip("\n") + "\n"],
        header_line=old.header_line,
        category=old.category,
    )
    # Round-trip: serialize → parse must recover the same structure
    result = serialize_insights(preamble, [new_ent])
    _, ents2 = parse_insights(result)
    assert len(ents2) == 1
    assert ents2[0].category == old.category
    assert ents2[0].header_line == old.header_line
    assert ents2[0].body == new_body
    # Full serialization must be byte-exact with what we expect
    expected = "### [architecture] 2025-06-28 10:32\n" + new_body
    assert result == expected


def test_edit_roundtrip_no_body_kwarg():
    """InsightEntry(body=...) must raise TypeError — body is a @property."""
    from external_llm.agent.insights_manager import InsightEntry
    import pytest as _pytest
    with _pytest.raises(TypeError):
        InsightEntry(
            lines=["### [x] 2025\n", "body\n"],
            header_line="### [x] 2025",
            category="x",
            body="should crash",  # was the cause of E1
        )


# ── Age policy: parse_timestamp / entry_age_days / select_entries_older_than ──

import calendar
import time as _time

from external_llm.agent.insights_manager import _age_reference_block


def _epoch(ts: str) -> float:
    """UTC epoch for a 'YYYY-MM-DD HH:MM' string (matches saved format)."""
    return calendar.timegm(_time.strptime(ts, "%Y-%m-%d %H:%M"))


def test_parse_timestamp_machine_header():
    assert parse_timestamp("### [architecture] 2026-06-26 05:30") == _epoch("2026-06-26 05:30")


def test_parse_timestamp_no_category():
    assert parse_timestamp("### 2026-06-26 05:30") == _epoch("2026-06-26 05:30")


def test_parse_timestamp_returns_none_for_prose_header():
    assert parse_timestamp("### Some hand-written principle") is None
    assert parse_timestamp("# title") is None
    assert parse_timestamp("### [cat] not-a-date") is None


def test_entry_age_days_from_header():
    _, ents = parse_insights("### [architecture] 2026-06-01 00:00\nbody\n")
    now = _epoch("2026-06-11 00:00")  # 10 days later
    assert entry_age_days(ents[0], now=now) == pytest.approx(10.0)


def test_entry_age_days_none_without_timestamp():
    _, ents = parse_insights("### hand written\nbody\n")
    assert entry_age_days(ents[0], now=_epoch("2026-06-11 00:00")) is None


def test_select_entries_older_than_picks_only_old():
    content = (
        "### [a] 2026-01-01 00:00\nold\n"
        "### [b] 2026-06-01 00:00\nrecent\n"
        "### hand-written\nno timestamp\n"
    )
    _, ents = parse_insights(content)
    now = _epoch("2026-06-10 00:00")
    # 90-day cutoff: only the Jan entry (#1) qualifies; #3 has no ts → never selected
    assert select_entries_older_than(ents, 90, now=now) == [1]


def test_select_entries_older_than_never_selects_untimestamped():
    _, ents = parse_insights("### no ts here\nbody\n")
    # even with a 0-day cutoff, an untimestamped entry is never age-pruned
    assert select_entries_older_than(ents, 0, now=_epoch("2030-01-01 00:00")) == []


def test_age_reference_block_lists_ages_and_marks_do_not_copy():
    content = "### [a] 2026-06-01 00:00\nbody\n### [b] 2026-06-09 00:00\nbody2\n"
    now = _epoch("2026-06-11 00:00")
    block = _age_reference_block(content, now=now)
    assert "DO NOT copy" in block
    assert "10d old" in block  # entry a
    assert "2d old" in block   # entry b


def test_age_reference_block_empty_when_no_entries():
    assert _age_reference_block("# preamble only\n> 원칙\n") == ""


def test_compact_messages_include_age_reference_when_timestamped():
    content = "### [architecture] 2020-01-01 00:00\ndurable invariant\n"
    msgs = build_compact_messages(content)
    user = msgs[1]["content"]
    assert "Entry ages" in user
    assert "RECENCY" in msgs[0]["content"]


def test_nudge_uses_oldest_entry_age_over_mtime():
    # oldest_age_days present → drives the age trigger (mtime ignored for age)
    stats = InsightsStats(
        exists=True, count=1, bytes_size=100,
        age_days=0,  # file just modified
        oldest_age_days=NUDGE_AGE_DAYS_THRESHOLD + 3,
    )
    fire, msg = should_nudge(stats)
    assert fire is True
    assert "oldest entry" in msg and "days old" in msg


# ── Two-tier archive: demotion / enforce-budget / index / promotion ──────────
# These cover the Layer 1 (hard budget via demotion), Layer 2 (always-on archive
# index), and Layer 3 (relevance promotion) behavior. The load-bearing invariants:
#   * timestamp-less / principle entries are NEVER demoted (mirror of
#     select_entries_older_than's carve-out);
#   * enforce_budget_by_demotion brings the active file to ≤ budget;
#   * nothing durable is deleted (demotion = move to archive);
#   * promotion is cheap, local, and silent when nothing matches.

import time as _time


def _ts_entry(days_ago: int, body: str, category: str = "architecture",
              now: float | None = None) -> object:
    """Build a timestamped InsightEntry `days_ago` in the past."""
    from external_llm.agent.insights_manager import InsightEntry
    now = now if now is not None else _time.time()
    ts = now - days_ago * 86400
    header = f"### [{category}] " + _time.strftime("%Y-%m-%d %H:%M", _time.gmtime(ts))
    return InsightEntry(
        lines=[header + "\n", body + "\n\n"],
        header_line=header, category=category,
    )


def _principle_entry(body: str = "Never use str.startswith for path containment.") -> object:
    """A timestamp-less, hand-written principle entry — must never be demoted."""
    from external_llm.agent.insights_manager import InsightEntry
    return InsightEntry(
        lines=["### Hand-written principle\n", body + "\n\n"],
        header_line="### Hand-written principle", category="",
    )


def test_select_demotion_empty_when_under_budget():
    entries = [_ts_entry(1, "x"), _ts_entry(2, "y")]
    assert select_demotion_candidates(entries, budget_bytes=10_000) == []


def test_select_demotion_protets_timestampless_principles():
    # Way over a tiny budget. The timestamp-less principle (idx 3) must NEVER be
    # selected, even though demoting it would be "cheapest".
    big = "durable invariant body " * 200
    entries = [_ts_entry(40, big), _ts_entry(10, big), _principle_entry(big)]
    idx = select_demotion_candidates(entries, budget_bytes=500)
    assert 3 not in idx           # principle protected
    assert idx == [1, 2]          # both timestamped demoted (deeply over budget)


def test_select_demotion_oldest_first():
    big = "durable invariant body " * 200  # ~4800 bytes each
    entries = [_ts_entry(40, big, "a"), _ts_entry(10, big, "b"), _ts_entry(2, big, "c")]
    total = sum(len("".join(e.lines).encode("utf-8")) for e in entries)
    one = len("".join(entries[0].lines).encode("utf-8"))
    # Budget needing exactly ONE demotion: demoting the oldest alone suffices.
    budget = total - one + 500
    idx = select_demotion_candidates(entries, budget_bytes=budget)
    assert idx == [1]  # oldest (40d) demoted, NOT the recent ones


def test_select_demotion_all_timestampless_returns_empty():
    # Everything is a principle → nothing can be demoted → [] (residual over-budget
    # accepted rather than ever demoting a principle).
    entries = [_principle_entry("one"), _principle_entry("two")]
    assert select_demotion_candidates(entries, budget_bytes=10) == []


def test_enforce_budget_demotes_to_archive_and_hits_budget(tmp_repo):
    big = "durable invariant about FileLockManager single identity " * 40
    entries = [_ts_entry(40, big, "a"), _ts_entry(10, big, "b"), _ts_entry(2, big, "c")]
    content = "# Design Chat Insights\n\n> Principle\n\n" + "".join(
        "".join(e.lines) for e in entries)
    atomic_write_text(insights_path(tmp_repo), content)
    total = len(content.encode("utf-8"))
    assert total > COMPACT_BUDGET_BYTES
    n, remaining = enforce_budget_by_demotion(tmp_repo, COMPACT_BUDGET_BYTES)
    assert n >= 1
    assert remaining <= COMPACT_BUDGET_BYTES
    # Archive received the demoted entries; active shrunk.
    _, arch = parse_insights(load_archive_file(tmp_repo))
    _, act = parse_insights(load_insights_file(tmp_repo))
    assert len(arch) == n
    assert len(act) == 3 - n


def test_enforce_budget_closes_preamble_band(tmp_repo):
    # Regression: when ENTRIES-ONLY <= budget < FULL-FILE, the non-entry
    # preamble must be subtracted before demotion accounting. Previously
    # select_demotion_candidates saw entries-only <= budget, returned [], and the
    # file stayed over budget — violating the hard-budget guarantee.
    preamble = "# Design Chat Insights\n\n> Principle block\n\n"
    e1 = _ts_entry(20, "a" * 800, "a")
    e2 = _ts_entry(5, "b" * 800, "b")
    content = preamble + "".join("".join(e.lines) for e in (e1, e2))
    atomic_write_text(insights_path(tmp_repo), content)

    preamble_bytes = len(preamble.encode("utf-8"))
    _, parsed = parse_insights(content)
    entries_bytes = sum(len("".join(e.lines).encode("utf-8")) for e in parsed)
    full = preamble_bytes + entries_bytes
    # Budget strictly between entries-only and full-file = the bug band.
    budget = entries_bytes + preamble_bytes // 2 + 1
    assert entries_bytes <= budget < full  # precondition: this IS the band

    n, remaining = enforce_budget_by_demotion(tmp_repo, budget)
    assert n >= 1, "must demote at least one entry in the preamble band"
    assert remaining <= budget, "file must fit budget after demotion"
def test_enforce_budget_idempotent(tmp_repo):
    content = "# Design Chat Insights\n\n" + "".join(
        "".join(e.lines) for e in [_ts_entry(5, "small entry")])
    atomic_write_text(insights_path(tmp_repo), content)
    n1, r1 = enforce_budget_by_demotion(tmp_repo, COMPACT_BUDGET_BYTES)
    assert n1 == 0  # under budget → no-op
    # Second call is also a no-op (no archive created).
    n2, r2 = enforce_budget_by_demotion(tmp_repo, COMPACT_BUDGET_BYTES)
    assert n2 == 0 and r2 == r1
    assert load_archive_file(tmp_repo) == ""


def test_enforce_budget_preserves_principle_in_active(tmp_repo):
    big = "durable invariant body " * 200
    entries = [_ts_entry(40, big), _principle_entry(big)]
    content = "# Design Chat Insights\n\n> P\n\n" + "".join(
        "".join(e.lines) for e in entries)
    atomic_write_text(insights_path(tmp_repo), content)
    enforce_budget_by_demotion(tmp_repo, COMPACT_BUDGET_BYTES)
    # Principle must still be in the ACTIVE file, never in the archive.
    active = load_insights_file(tmp_repo)
    archive = load_archive_file(tmp_repo)
    assert "Hand-written principle" in active
    assert "Hand-written principle" not in archive


def test_append_entries_to_archive_creates_with_preamble_and_roundtrips(tmp_repo):
    from external_llm.agent.insights_manager import append_entries_to_archive
    e = _ts_entry(5, "demoted durable body", "pattern")
    append_entries_to_archive(tmp_repo, [e])
    arc = load_archive_file(tmp_repo)
    assert "Archived Design Insights" in arc  # preamble present
    _, entries = parse_insights(arc)
    assert len(entries) == 1
    # Append a second → two entries, round-trip safe.
    append_entries_to_archive(tmp_repo, [_ts_entry(3, "second demoted", "issue")])
    _, entries2 = parse_insights(load_archive_file(tmp_repo))
    assert len(entries2) == 2


def test_append_entries_to_archive_empty_is_noop(tmp_repo):
    from external_llm.agent.insights_manager import append_entries_to_archive
    append_entries_to_archive(tmp_repo, [])
    assert load_archive_file(tmp_repo) == ""


def test_build_archive_index_empty_when_no_archive(tmp_repo):
    assert build_archive_index(tmp_repo) == ""


def test_build_archive_index_lists_and_caps(tmp_repo):
    from external_llm.agent.insights_manager import (
        append_entries_to_archive, ARCHIVE_INDEX_MAX_ENTRIES,
    )
    many = [_ts_entry(i, f"entry number {i} body", "pattern") for i in range(1, 25)]
    append_entries_to_archive(tmp_repo, many)
    idx = build_archive_index(tmp_repo)
    assert "ARCHIVED INSIGHTS" in idx
    # capped at ARCHIVE_INDEX_MAX_ENTRIES visible entry-lines + "and N more" note
    import re
    visible = len(re.findall(r"^  A\d+\.", idx, re.MULTILINE))
    assert visible <= ARCHIVE_INDEX_MAX_ENTRIES
    assert "more" in idx  # truncation note present (>15 archived)


def test_build_archive_index_labels_match_file_order(tmp_repo):
    """Archive index labels (A1, A2, ...) must match file-order positions so that
    ``/insights archive restore <n>`` and ``drop <n>`` operate on the correct entry.
    Display order is newest-first, but labels must reflect actual file positions.
    """
    from external_llm.agent.insights_manager import (
        append_entries_to_archive, load_archive_file, parse_insights,
    )
    import re
    # Add 3 distinct entries with unique bodies
    entries = [
        _ts_entry(100, "first entry body about alpha", "pattern"),
        _ts_entry(200, "second entry body about beta", "pattern"),
        _ts_entry(300, "third entry body about gamma", "pattern"),
    ]
    append_entries_to_archive(tmp_repo, entries)
    # Parse archive to get file-order entries (oldest first)
    _, archive_entries = parse_insights(load_archive_file(tmp_repo))
    assert len(archive_entries) == 3
    # Build index and verify labels match file positions
    idx = build_archive_index(tmp_repo)
    # Extract all A<n> labels and their associated body snippets
    label_pattern = re.compile(r"  A(\d+)\. \[.*?\] \(.*?\) (.+)$", re.MULTILINE)
    matches = label_pattern.findall(idx)
    assert len(matches) == 3
    # Verify each label maps to the correct file-order entry
    for label_str, body_snippet in matches:
        file_idx = int(label_str)
        assert 1 <= file_idx <= 3
        # file_idx is 1-based, archive_entries is 0-based
        expected_entry = archive_entries[file_idx - 1]
        expected_first_line = expected_entry.body.strip().split("\n", 1)[0][:70]
        assert body_snippet == expected_first_line


def test_select_promotable_empty_when_no_query_or_no_archive(tmp_repo):
    from external_llm.agent.insights_manager import append_entries_to_archive
    append_entries_to_archive(tmp_repo, [_ts_entry(5, "body about FileLockManager")])
    assert select_promotable_entries(tmp_repo, "") == []
    # Non-overlapping query → []
    assert select_promotable_entries(tmp_repo, "zzzzz totally unrelated qwx") == []


def test_select_promotable_matches_by_token_overlap(tmp_repo):
    from external_llm.agent.insights_manager import append_entries_to_archive
    e = _ts_entry(60, "FileLockManager uses WeakValueDictionary for lock identity")
    append_entries_to_archive(tmp_repo, [e])
    promoted = select_promotable_entries(
        tmp_repo, "FileLockManager lock identity WeakValueDictionary")
    assert len(promoted) == 1
    assert "FileLockManager" in promoted[0].body


def test_select_promotable_respects_min_overlap(tmp_repo):
    from external_llm.agent.insights_manager import append_entries_to_archive, PROMOTE_MIN_SCORE
    e = _ts_entry(60, "alpha beta gamma delta epsilon")
    append_entries_to_archive(tmp_repo, [e])
    # Single shared token leads to BM25 score ≈ 0.287 (< PROMOTE_MIN_SCORE=0.5)
    # → no promotion.
    assert select_promotable_entries(tmp_repo, "alpha") == []
    # Two shared tokens → BM25 score ≈ 0.574 (≥ 0.5) → promoted.
    assert len(select_promotable_entries(tmp_repo, "alpha beta")) >= 1
    assert PROMOTE_MIN_SCORE == 0.5


def test_select_promotable_score_scales_with_token_overlap(tmp_repo):
    """Perf #7 guard: tf_norm was hoisted out of the per-term BM25 loop. The score
    must still accumulate idf across ALL shared tokens (score = tf_norm * Σ idf),
    so an entry overlapping the query on MANY tokens scores far above the single-
    token baseline and is reliably promoted. A hoist that dropped tf_norm or
    double-counted would distort the score and break this monotonicity."""
    from external_llm.agent.insights_manager import append_entries_to_archive
    e = _ts_entry(
        60, "lock identity WeakValueDictionary thread safety guard reset pool",
    )
    append_entries_to_archive(tmp_repo, [e])
    # 5-token overlap → score ≈ 5 × idf ≫ PROMOTE_MIN_SCORE (0.5).
    promoted = select_promotable_entries(
        tmp_repo, "lock identity WeakValueDictionary thread safety",
    )
    assert len(promoted) == 1
    assert "WeakValueDictionary" in promoted[0].body


# ── B1: demotion durability (archive-first write order) ─────────────────────


def test_enforce_budget_demotion_is_archive_first(tmp_repo):
    """B1: the archive is appended BEFORE the active file is truncated, so a
    crash during the active write leaves the demoted entries recoverable in the
    archive instead of permanently lost.

    Non-vacuous: the OLD (active-first) order wrote active first, so faulting
    that write would leave the archive empty and this assertion would fail.
    """
    from unittest import mock
    from external_llm.agent import insights_manager as im

    many = [_ts_entry(i, f"durable entry body {i}", "pattern") for i in range(1, 10)]
    content = serialize_insights(["# Preamble\n\n"], many)
    atomic_write_text(insights_path(tmp_repo), content)
    budget = len(content.encode("utf-8")) // 2  # well under → forces demotion

    active_path = insights_path(tmp_repo)
    real_atomic = im.atomic_write_text

    def fault_active_only(path, data):
        # Simulate a crash specifically during the ACTIVE-file write.
        if os.path.abspath(path) == os.path.abspath(active_path):
            raise OSError("simulated crash during active write")
        return real_atomic(path, data)

    with mock.patch.object(im, "atomic_write_text", side_effect=fault_active_only):
        with pytest.raises(OSError):
            im.enforce_budget_by_demotion(tmp_repo, budget)

    arch = load_archive_file(tmp_repo)
    assert "durable entry body" in arch, (
        "demoted entries lost — archive was not written before the active truncate"
    )


def test_append_entries_to_archive_dedups_exact_duplicates(tmp_repo):
    """B1: re-appending an entry already in the archive (a crash-recovery
    re-demote) is absorbed — the archive never accumulates exact duplicates."""
    from external_llm.agent.insights_manager import append_entries_to_archive

    e = _ts_entry(5, "durable lock-identity insight", "pattern")
    append_entries_to_archive(tmp_repo, [e])
    # Simulate a recovery re-demote of the SAME entry.
    append_entries_to_archive(tmp_repo, [e])
    _, entries = parse_insights(load_archive_file(tmp_repo))
    assert len(entries) == 1, "crash-recovery re-append duplicated an archive entry"
    # A genuinely distinct entry is still kept.
    append_entries_to_archive(tmp_repo, [_ts_entry(4, "different insight body", "issue")])
    _, entries = parse_insights(load_archive_file(tmp_repo))
    assert len(entries) == 2


def test_enforce_budget_crash_recovery_leaves_no_archive_duplicates(tmp_repo):
    """B1: a crash during the active-file truncate leaves the demoted entries in
    BOTH files (active intact + archive appended). The recovery enforce run
    re-demotes them; append_entries_to_archive must dedup the re-append so the
    archive holds each entry exactly once.

    Non-vacuous: without dedup, the recovery re-append doubles every demoted
    header in the archive (each appears 2x).
    """
    from unittest import mock
    from external_llm.agent import insights_manager as im

    many = [_ts_entry(i, f"durable entry body {i}", "pattern") for i in range(1, 10)]
    content = serialize_insights(["# Preamble\n\n"], many)
    atomic_write_text(insights_path(tmp_repo), content)
    budget = len(content.encode("utf-8")) // 2  # forces demotion

    active_path = insights_path(tmp_repo)
    real_atomic = im.atomic_write_text

    def fault_active_only(path, data):
        if os.path.abspath(path) == os.path.abspath(active_path):
            raise OSError("simulated crash during active write")
        return real_atomic(path, data)

    # First enforce: archive append succeeds, active truncate crashes.
    with mock.patch.object(im, "atomic_write_text", side_effect=fault_active_only):
        with pytest.raises(OSError):
            im.enforce_budget_by_demotion(tmp_repo, budget)

    # Crash state: archive has the demoted entries; active still has them too.
    _, arch_before = parse_insights(load_archive_file(tmp_repo))
    assert len(arch_before) >= 1

    # Recovery run: re-demotes the still-present entries; dedup must absorb.
    im.enforce_budget_by_demotion(tmp_repo, budget)
    _, arch_after = parse_insights(load_archive_file(tmp_repo))

    headers = [e.header_line for e in arch_after]
    assert len(headers) == len(set(headers)), (
        f"crash recovery duplicated archive entries: {headers}"
    )
    assert len(arch_after) == len(arch_before), (
        "recovery run changed archive membership beyond absorbing the re-append"
    )
# ── B2: 0c archive index is byte-stable across the wall clock ────────────────


def test_build_archive_index_byte_stable_across_clock(tmp_repo):
    """B2: the always-on archive index (injected into the cached 0c prefix) must
    NOT depend on the wall clock. A relative age ('Nd') ticks at each UTC day
    boundary and would invalidate the compressed-summary + turns cache for
    nothing. The index uses the byte-stable creation date instead.

    Non-vacuous: re-introducing ``entry_age_days()``/``time.time()`` here makes
    the two clock snapshots differ.
    """
    import re
    from unittest import mock
    from external_llm.agent import insights_manager as im
    from external_llm.agent.insights_manager import append_entries_to_archive

    append_entries_to_archive(tmp_repo, [_ts_entry(3, "stable index body line", "pattern")])
    base = _time.time()
    with mock.patch.object(im.time, "time", return_value=base):
        out1 = im.build_archive_index(tmp_repo)
    with mock.patch.object(im.time, "time", return_value=base + 20 * 86400):
        out2 = im.build_archive_index(tmp_repo)

    assert out1 == out2
    # Absolute date present; relative age absent.
    assert re.search(r"\(\d{4}-\d{2}-\d{2}\)", out1)
    assert not re.search(r"\(\d+d\)", out1)


# ── P1: archive parse/token cache ────────────────────────────────────────────


def test_archive_parsed_cache_serves_consistent_entries(tmp_repo):
    """P1: repeated reads return consistent entries through the cache."""
    from external_llm.agent import insights_manager as im
    from external_llm.agent.insights_manager import append_entries_to_archive

    append_entries_to_archive(tmp_repo, [_ts_entry(5, "cache body alpha beta", "pattern")])
    e1 = im._parsed_archive_cached(tmp_repo)
    e2 = im._parsed_archive_cached(tmp_repo)
    assert [x.body for x in e1] == [x.body for x in e2]
    assert len(e1) == 1


def test_archive_parsed_cache_invalidates_on_write(tmp_repo):
    """P1: after append_entries_to_archive (which calls _archive_invalidate),
    the cache reflects the new content — even on a same-second write (mtime alone
    could collide, so the explicit invalidate + size key must carry it)."""
    from external_llm.agent import insights_manager as im
    from external_llm.agent.insights_manager import append_entries_to_archive

    append_entries_to_archive(tmp_repo, [_ts_entry(1, "first cached body", "pattern")])
    assert len(im._parsed_archive_cached(tmp_repo)) == 1
    append_entries_to_archive(tmp_repo, [_ts_entry(2, "second cached body", "pattern")])
    assert len(im._parsed_archive_cached(tmp_repo)) == 2


def test_promotion_consistent_through_cache(tmp_repo):
    """P1: select_promotable_entries returns correct results via the cached
    token-sets, and stays correct on repeated calls + after a change."""
    from external_llm.agent.insights_manager import append_entries_to_archive

    append_entries_to_archive(tmp_repo, [_ts_entry(10, "FileLockManager lock identity cache", "pattern")])
    q = "FileLockManager lock identity"
    assert len(select_promotable_entries(tmp_repo, q)) == 1
    assert len(select_promotable_entries(tmp_repo, q)) == 1  # cache hit, same answer
    # Unrelated query still yields nothing.
    assert select_promotable_entries(tmp_repo, "zzzz unrelated qwx") == []


# ── Active insights file content cache (block 0c) ─────────────────────────────
# load_design_insights reads the ACTIVE design_insights.md every turn (block 0c
# of the prompt prefix, which must stay byte-stable). The signature cache
# ``(mtime_ns, size, write_version)`` mirrors the archive family so the per-turn
# open()+read() is skipped when unchanged, and any writer-invalidated or
# mtime/size change is detected instantly. Mirrors the project.md mtime cache.


def test_active_cache_miss_reads_file(tmp_repo):
    """Cold cache: reads the file and populates the cache."""
    _active_invalidate(tmp_repo)  # ensure cold
    atomic_write_text(insights_path(tmp_repo), "### [pattern] 2026-01-01 00:00\nbody-A\n\n")
    assert "body-A" in load_active_insights_cached(tmp_repo)


def test_active_cache_hit_serves_cached(tmp_repo, monkeypatch):
    """Warm cache: a hit must NOT re-open the file."""
    from external_llm.agent import insights_manager as im

    atomic_write_text(insights_path(tmp_repo), "### [pattern] 2026-01-01 00:00\nbody-A\n\n")
    first = load_active_insights_cached(tmp_repo)
    # Sabotage the reader; a cache hit must bypass it entirely.
    called = {"n": 0}
    orig = im._load_file_safe

    def _spy(path):
        called["n"] += 1
        return orig(path)

    monkeypatch.setattr(im, "_load_file_safe", _spy)
    again = load_active_insights_cached(tmp_repo)
    assert again == first
    assert called["n"] == 0, "cache hit must not re-read the file"


def test_active_cache_invalidate_after_explicit_write(tmp_repo):
    """_active_invalidate drops the entry so the next read sees fresh content."""
    atomic_write_text(insights_path(tmp_repo), "### [pattern] 2026-01-01 00:00\nbody-A\n\n")
    assert "body-A" in load_active_insights_cached(tmp_repo)
    atomic_write_text(insights_path(tmp_repo), "### [pattern] 2026-01-02 00:00\nbody-B\n\n")
    _active_invalidate(tmp_repo)
    result = load_active_insights_cached(tmp_repo)
    assert "body-B" in result and "body-A" not in result


def test_active_cache_mtime_change_invalidates_without_explicit_call(tmp_repo):
    """A plain mtime/size change (no _active_invalidate) still invalidates — the
    belt-and-suspenders so a future writer that forgets the call stays correct."""
    atomic_write_text(insights_path(tmp_repo), "### [pattern] 2026-01-01 00:00\nbody-A\n\n")
    assert "body-A" in load_active_insights_cached(tmp_repo)
    import time
    time.sleep(0.02)  # ensure distinct st_mtime_ns
    atomic_write_text(insights_path(tmp_repo), "### [pattern] 2026-01-02 00:00\nbody-B\n\n")
    result = load_active_insights_cached(tmp_repo)
    assert "body-B" in result and "body-A" not in result


def test_active_cache_missing_file_returns_empty(tmp_repo):
    """No file → '' and stays cached as '' (no exception)."""
    _active_invalidate(tmp_repo)
    assert load_active_insights_cached(tmp_repo) == ""
    assert load_active_insights_cached(tmp_repo) == ""  # cached empty, no error


def test_active_cache_isolated_per_repo(tmp_path):
    """Two distinct repo roots have independent cache entries (path-keyed)."""
    (tmp_path / "a" / ".asicode").mkdir(parents=True)
    (tmp_path / "b" / ".asicode").mkdir(parents=True)
    ra, rb = str(tmp_path / "a"), str(tmp_path / "b")
    atomic_write_text(insights_path(ra), "### [pattern] 2026-01-01 00:00\nfrom-A\n\n")
    atomic_write_text(insights_path(rb), "### [pattern] 2026-01-01 00:00\nfrom-B\n\n")
    assert "from-A" in load_active_insights_cached(ra)
    assert "from-B" in load_active_insights_cached(rb)
    assert "from-B" not in load_active_insights_cached(ra)
    assert "from-A" not in load_active_insights_cached(rb)
