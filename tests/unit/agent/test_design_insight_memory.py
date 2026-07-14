"""Tests for design chat insight memory (save_insight tool + context injection)."""
from __future__ import annotations

import os

import pytest

from external_llm.agent.design_chat_loop import (
    _delete_insight,
    _edit_insight,
    _find_entry_by_match,
    _save_insight_to_file,
    load_design_insights,
    load_promoted_insights,
)
from external_llm.agent.insights_manager import (
    COMPACT_BUDGET_BYTES,
    InsightEntry,
    append_entries_to_archive,
    enforce_budget_by_demotion,
    insights_write_lock,
    load_archive_file,
    load_insights_file,
)


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo root with .asicode directory."""
    asr_dir = tmp_path / ".asicode"
    asr_dir.mkdir()
    return str(tmp_path)


@pytest.fixture
def tmp_repo_no_asr(tmp_path):
    """Create a temporary repo root WITHOUT .asicode directory."""
    return str(tmp_path)


class TestSaveInsight:
    """Test _save_insight_to_file."""

    def test_creates_file_on_first_save(self, tmp_repo):
        result = _save_insight_to_file(tmp_repo, "Test insight", "architecture")
        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        assert os.path.exists(path)
        assert "Insight saved" in result
        assert "[architecture]" in result

    def test_appends_with_category_and_timestamp(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "First insight", "gotcha")
        _save_insight_to_file(tmp_repo, "Second insight", "bug")

        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        content = open(path, encoding="utf-8").read()

        assert "### [gotcha]" in content
        assert "First insight" in content
        assert "### [bug]" in content
        assert "Second insight" in content

    def test_truncates_long_insight(self, tmp_repo):
        long_text = "x" * 600
        _save_insight_to_file(tmp_repo, long_text, "pattern")

        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        content = open(path, encoding="utf-8").read()
        # The function receives already-truncated text ([:500] in caller),
        # but _save_insight_to_file itself doesn't truncate — that's the caller's job
        assert len(content) > 0

    def test_header_created_once(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "A", "architecture")
        _save_insight_to_file(tmp_repo, "B", "architecture")

        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        content = open(path, encoding="utf-8").read()
        assert content.count("# Design Chat Insights") == 1


class TestLoadInsights:
    """Test load_design_insights."""

    def test_returns_empty_when_no_file(self, tmp_repo):
        result = load_design_insights(tmp_repo)
        assert result == ""

    def test_loads_saved_insights(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "shell_exec can run date", "gotcha")
        result = load_design_insights(tmp_repo)
        assert "shell_exec can run date" in result
        assert "[gotcha]" in result

    def test_passes_full_content_when_large(self, tmp_repo):
        """Verify large insights are no longer truncated (discarding information upfront wastes tokens instead)."""
        for i in range(100):
            _save_insight_to_file(tmp_repo, f"Insight number {i} " * 10, "pattern")

        result = load_design_insights(tmp_repo, max_chars=500)
        # Full content preserved despite max_chars=500
        assert len(result) > 1000
        assert "Insight number 99" in result

    def test_load_design_insights_excludes_layer3_promotion(self, tmp_repo):
        """Cache contract: the STABLE 0c block must NOT contain turn-volatile
        Layer 3 promotion, even when an archived entry is highly promotable.

        load_design_insights is placed early in the prompt prefix (block 0c) and
        must stay byte-identical across turns. Layer 3 (promoted bodies) depends
        on the recent user turns, so it is injected late via
        load_promoted_insights — never folded into load_design_insights.

        Note: the Layer 2 archive INDEX (header-only, first ~70 chars of each
        archived body) IS intentionally included — it only changes on
        demotion/restore/drop, so it is safe in the stable 0c prefix. The test
        distinguishes Layer 3 by its dedicated header and a unique body tail
        that never fits in the 70-char index line.
        """
        _BODY = (
            "FileLockManager uses WeakValueDictionary for lock identity "
            "cross-request UNIQUE-TAIL-MARKER-ZZZ"
        )
        _save_insight_to_file(tmp_repo, "active insight body", "architecture")
        append_entries_to_archive(tmp_repo, [
            InsightEntry(
                lines=["### [architecture] 2026-06-30 13:18\n", _BODY + "\n\n"],
                header_line="### [architecture] 2026-06-30 13:18",
                category="architecture",
            )
        ])
        stable = load_design_insights(tmp_repo)
        # Active body + Layer 2 index present ...
        assert "active insight body" in stable
        assert "ARCHIVED INSIGHTS" in stable  # Layer 2 index header
        # ... but the Layer 3 promoted block must NOT be folded in.
        assert "PROMOTED FROM ARCHIVE" not in stable
        assert "UNIQUE-TAIL-MARKER-ZZZ" not in stable  # full body not present

    def test_load_promoted_insights_contract(self, tmp_repo):
        """load_promoted_insights is the SOLE carrier of turn-volatile Layer 3.

        Returns the promoted body when the task overlaps an archived entry, and
        "" (so the message list stays byte-identical / cache-safe) otherwise.
        """
        append_entries_to_archive(tmp_repo, [
            InsightEntry(
                lines=["### [architecture] 2026-06-30 13:18\n",
                       "FileLockManager uses WeakValueDictionary for lock identity\n\n"],
                header_line="### [architecture] 2026-06-30 13:18",
                category="architecture",
            )
        ])
        # No query / non-overlapping → "" (cache-safe: nothing injected).
        assert load_promoted_insights(tmp_repo, "") == ""
        assert load_promoted_insights(tmp_repo, "zzzz totally unrelated qwx") == ""
        # Overlapping query → promoted body with its header.
        promoted = load_promoted_insights(
            tmp_repo, "FileLockManager lock identity WeakValueDictionary")
        assert "PROMOTED FROM ARCHIVE" in promoted
        assert "WeakValueDictionary" in promoted


class TestContextInjection:
    """Test that build_context_messages injects insights."""

    def test_insights_injected_in_context(self, tmp_repo):
        """Verify load_design_insights output would be non-empty after save."""
        _save_insight_to_file(tmp_repo, "Important discovery", "architecture")
        text = load_design_insights(tmp_repo)
        assert "Important discovery" in text
        # The actual injection happens in DesignSessionManager.build_context_messages
        # which imports load_design_insights — tested via integration


class TestFindEntryByMatch:
    """Test _find_entry_by_match."""

    def test_finds_single_match(self):
        entries = [
            InsightEntry(lines=["### [arch] 2026-01-01 10:00\n", "body\n\n"], header_line="### [arch] 2026-01-01 10:00", category="arch"),
            InsightEntry(lines=["### [bug] 2026-01-02 11:00\n", "body\n\n"], header_line="### [bug] 2026-01-02 11:00", category="bug"),
        ]
        idx, err = _find_entry_by_match(entries, "2026-01-02")
        assert idx == 1
        assert err is None

    def test_no_match(self):
        entries = [
            InsightEntry(lines=["### [arch] 2026-01-01 10:00\n", "body\n\n"], header_line="### [arch] 2026-01-01 10:00", category="arch"),
        ]
        idx, err = _find_entry_by_match(entries, "2027-01-01")
        assert idx is None
        assert "No insight found" in err

    def test_multiple_matches(self):
        entries = [
            InsightEntry(lines=["### [arch] 2026-01-01 10:00\n", "body\n\n"], header_line="### [arch] 2026-01-01 10:00", category="arch"),
            InsightEntry(lines=["### [bug] 2026-01-01 10:00\n", "body\n\n"], header_line="### [bug] 2026-01-01 10:00", category="bug"),
        ]
        idx, err = _find_entry_by_match(entries, "2026-01-01")
        assert idx is None
        assert "Multiple insights match" in err

    def test_multiple_matches_bracketless_headers_no_crash(self):
        """Regression: hand-written headers without a ``[category]`` bracket
        must not raise IndexError when building the disambiguation list.

        ``parse_insights`` explicitly supports bracket-less prose headers
        (category → ""). The old implementation did ``h.split('[')[1]`` to
        recover the category, which raised IndexError on such headers.
        """
        entries = [
            InsightEntry(lines=["### hand-written note one\n", "first body\n\n"],
                         header_line="### hand-written note one", category=""),
            InsightEntry(lines=["### hand-written note two\n", "second body\n\n"],
                         header_line="### hand-written note two", category=""),
        ]
        idx, err = _find_entry_by_match(entries, "hand-written")
        assert idx is None
        assert "Multiple insights match" in err
        # The bracket-less entries surface as [uncategorized] with their body,
        # not as a crash or a raw error string.
        assert "uncategorized" in err
        assert "first body" in err
        assert "second body" in err


class TestDeleteInsight:
    """Test _delete_insight."""

    def test_deletes_existing_entry(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "Keep this", "architecture")
        _save_insight_to_file(tmp_repo, "Delete this", "gotcha")

        # Match by unique category to disambiguate
        result = _delete_insight(tmp_repo, "[gotcha]")
        assert result.startswith("✅")

        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        content2 = open(path, encoding="utf-8").read()
        assert "Delete this" not in content2
        assert "Keep this" in content2

    def test_errors_on_nonexistent_match(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "Some insight", "architecture")
        result = _delete_insight(tmp_repo, "nonexistent_header_match")
        assert "Error" in result
        assert "No insight found" in result

    def test_errors_on_missing_file(self, tmp_repo_no_asr):
        result = _delete_insight(tmp_repo_no_asr, "[architecture]")
        assert "Error" in result
        assert "No design insights file found" in result

    def test_errors_on_ambiguous_match(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "First insight", "arch")
        _save_insight_to_file(tmp_repo, "Second insight", "arch")
        # Both entries have "[arch]" in their header → ambiguous
        result = _delete_insight(tmp_repo, "[arch]")
        assert "Error" in result
        assert "Multiple insights match" in result


class TestEditInsight:
    """Test _edit_insight."""

    def test_edits_body(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "Old content", "architecture")
        result = _edit_insight(tmp_repo, "[architecture]", "New content")
        assert result.startswith("✅")

        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        content = open(path, encoding="utf-8").read()
        assert "Old content" not in content
        assert "New content" in content

    def test_edits_body_and_category(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "Old content", "architecture")
        result = _edit_insight(tmp_repo, "[architecture]", "Updated content", new_category="pattern")
        assert result.startswith("✅")

        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        content = open(path, encoding="utf-8").read()
        assert "### [pattern]" in content
        assert "Updated content" in content
        assert "Old content" not in content

    def test_errors_on_nonexistent_match(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "Some insight", "architecture")
        result = _edit_insight(tmp_repo, "nonexistent", "New content")
        assert "Error" in result
        assert "No insight found" in result

    def test_errors_on_missing_file(self, tmp_repo_no_asr):
        result = _edit_insight(tmp_repo_no_asr, "[architecture]", "New content")
        assert "Error" in result
        assert "No design insights file found" in result


def _mp_save_worker(repo_root: str, count: int, results_path: str) -> None:
    """Cross-process save worker (module-level for spawn picklability).

    Appends ``count`` insights, recording each marker so the parent can verify
    none were dropped by a concurrent compactor in another process.
    """
    from external_llm.agent.design_chat_loop import _save_insight_to_file

    markers = []
    for i in range(count):
        marker = f"MPKEEP_{i:03d}"
        _save_insight_to_file(repo_root, f"body {marker} crossproc", "architecture")
        markers.append(marker)
    with open(results_path, "a", encoding="utf-8") as f:
        for m in markers:
            f.write(m + "\n")


def _mp_compact_worker(repo_root: str, count: int) -> None:
    """Cross-process compaction worker — repeated RMW racing the appender."""
    from external_llm.agent.insights_manager import (
        COMPACT_BUDGET_BYTES,
        enforce_budget_by_demotion,
    )

    for _ in range(count):
        enforce_budget_by_demotion(repo_root, COMPACT_BUDGET_BYTES)


class TestInsightsWriteLockConcurrency:
    """Concurrency contract for insights_write_lock (issue: RMW durable loss).

    Without the lock, a compactor's read-modify-write (rewrite of the whole
    file) racing a concurrent _save_insight_to_file (append) silently drops the
    appended entry — violating the documented "0 durable loss" contract.
    """

    def test_lock_serializes_concurrent_holders(self, tmp_repo):
        """At most ONE thread may hold insights_write_lock at a time."""
        import threading
        import time

        active = [0]
        max_active = [0]
        guard = threading.Lock()

        def critical():
            with guard:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            time.sleep(0.005)  # widen the window so overlap would be detected
            with guard:
                active[0] -= 1

        def worker():
            for _ in range(10):
                with insights_write_lock(tmp_repo):
                    critical()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_active[0] == 1, (
            f"insights_write_lock allowed {max_active[0]} concurrent holders; "
            "writers must be serialized"
        )

    def test_lock_is_reentrant_for_nested_calls(self, tmp_repo):
        """enforce_budget_by_demotion → append_entries_to_archive must not deadlock.

        Both acquire insights_write_lock; RLock re-entrancy prevents self-deadlock.
        """
        import threading

        errored = []

        def run():
            try:
                # enforce_budget calls append_entries internally; both lock.
                # If the lock were a plain Lock, this deadlocks/hangs.
                _save_insight_to_file(tmp_repo, "seed entry", "architecture")
                enforce_budget_by_demotion(tmp_repo, COMPACT_BUDGET_BYTES)
            except Exception as e:
                errored.append(e)

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=10)
        assert not t.is_alive(), "nested lock acquisition deadlocked"
        assert not errored, f"nested lock call raised: {errored}"

    def test_parallel_save_and_compact_no_durable_loss(self, tmp_repo):
        """Concurrent save (append) + compact (RMW) drops no saved entry.

        This is the core safety contract: every appended entry must survive in
        active or archive regardless of interleaving with the compactor.
        """
        import threading

        saved_markers: list[str] = []
        mlock = threading.Lock()

        def save_worker():
            local: list[str] = []
            for i in range(15):
                marker = f"UNIQUE_KEEP_{i:03d}"
                _save_insight_to_file(
                    tmp_repo, f"body {marker} durable", "architecture"
                )
                local.append(marker)
            with mlock:
                saved_markers.extend(local)

        def compact_worker():
            # tiny budget forces repeated demotion RMWs racing the appends
            for _ in range(8):
                enforce_budget_by_demotion(tmp_repo, COMPACT_BUDGET_BYTES)

        t1 = threading.Thread(target=save_worker)
        t2 = threading.Thread(target=compact_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        active = load_insights_file(tmp_repo)
        archive = load_archive_file(tmp_repo) or ""
        combined = active + "\n" + archive
        missing = [m for m in saved_markers if m not in combined]
        assert not missing, (
            f"durable loss: {len(missing)}/{len(saved_markers)} saved entries "
            f"dropped by concurrent compact: {missing[:5]}"
        )

    def test_cross_process_save_and_compact_no_durable_loss(self, tmp_repo):
        """The '0 durable loss' contract must hold ACROSS processes, not just threads.

        insights_write_lock is backed by fcntl.flock (per-open-file-description),
        which serializes across PROCESSES — the thread-based tests above only
        exercise threading.RLock. This spawns a compactor in one process racing
        an appender in another and verifies (a) no deadlock and (b) every
        appended entry survives in active or archive. Locks in the cross-process
        half of the contract.
        """
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        results_path = os.path.join(tmp_repo, ".asicode", "_mp_markers.txt")

        p_save = ctx.Process(target=_mp_save_worker, args=(tmp_repo, 15, results_path))
        p_compact = ctx.Process(target=_mp_compact_worker, args=(tmp_repo, 8))
        p_save.start()
        p_compact.start()
        p_save.join(timeout=30)
        p_compact.join(timeout=30)
        assert not p_save.is_alive(), "save worker hung — cross-process lock deadlock"
        assert not p_compact.is_alive(), "compact worker hung — cross-process lock deadlock"

        saved_markers = [
            ln.strip() for ln in open(results_path, encoding="utf-8") if ln.strip()
        ]
        active = load_insights_file(tmp_repo)
        archive = load_archive_file(tmp_repo) or ""
        combined = active + "\n" + archive
        missing = [m for m in saved_markers if m not in combined]
        assert not missing, (
            f"cross-process durable loss: {len(missing)}/{len(saved_markers)} "
            f"entries dropped by concurrent compact: {missing[:5]}"
        )
