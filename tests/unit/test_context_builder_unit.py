"""
Unit tests for external_llm/context_builder.py + context_collector.py.

Covers two defects fixed together:
  A. lstrip("./") is a character-SET {'.','/'} — it stripped a dotfile's
     leading dot (".config.py" -> "config.py"), so the target file appeared
     missing and/or leaked into its own Related Files. Fixed with
     removeprefix("./") (matches the go_provider.py precedent).
  B. _structure_hints_cache never evicted expired entries — keys that expired
     but were never re-accessed accumulated forever across distinct repo
     roots. Fixed with opportunistic GC on the miss path.
"""
from __future__ import annotations

import time

import pytest

from common import normalize_rel_path_fast
from context_collector import collect_related_files_shallow
from external_llm import context_builder as cb
from external_llm.context_builder import EnhancedContextBuilder

# ── Defect A: dotfile / dot-directory path normalization ────────────────────

def _norm(s: str) -> str:
    """Mirror of the inline normalization now used in both modules."""
    return normalize_rel_path_fast(s)


def test_rel_normalization_preserves_dotfiles():
    assert _norm(".config.py") == ".config.py"
    assert _norm(".hidden/mod.py") == ".hidden/mod.py"
    assert _norm("./.github/ci.yml") == ".github/ci.yml"
    assert _norm("./foo.py") == "foo.py"
    assert _norm("/foo.py") == "foo.py"
    assert _norm("foo.py") == "foo.py"


def test_collect_related_finds_dotfile_target(tmp_path):
    """Root fix: collect_related_files_shallow must FIND a dotfile target,
    not report it as 'target_missing' (the old char-set lstrip mangled it)."""
    (tmp_path / "helper.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / ".config.py").write_text("import helper\n", encoding="utf-8")

    selected, meta = collect_related_files_shallow(str(tmp_path), ".config.py")

    assert meta.get("reason") != "target_missing", f"dotfile not found: {meta}"
    assert ".config.py" in selected, f"target dropped: {selected}"


def test_find_related_excludes_dotfile_target(tmp_path):
    """Dedup fix: the dotfile target must be EXCLUDED from Related Files.
    Old bug: rel='.config.py'.lstrip('./')=='config.py' never matched, so the
    target leaked into its own Related Files list."""
    (tmp_path / "helper.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / ".config.py").write_text("import helper\n", encoding="utf-8")

    builder = EnhancedContextBuilder(str(tmp_path))
    related = builder._find_related_files(".config.py", max_files=3)

    assert ".config.py" not in related, f"target leaked in: {related}"
    assert "helper.py" in related, f"expected helper.py, got {related}"


def test_find_related_excludes_dotdir_target(tmp_path):
    """Same defect for a dot-DIRECTORY prefix (e.g. .hidden/mod.py)."""
    (tmp_path / "helper.py").write_text("X = 1\n", encoding="utf-8")
    dotpkg = tmp_path / ".hidden"
    dotpkg.mkdir()
    (dotpkg / "mod.py").write_text("import helper\n", encoding="utf-8")

    builder = EnhancedContextBuilder(str(tmp_path))
    related = builder._find_related_files(".hidden/mod.py", max_files=3)

    assert ".hidden/mod.py" not in related, f"dotdir target leaked: {related}"


# ── Defect B: bounded _structure_hints_cache ────────────────────────────────

@pytest.fixture
def isolated_hints_cache():
    """Snapshot/restore the process-wide cache + GC threshold around each test
    so module-level state never leaks across tests."""
    cache = cb._structure_hints_cache
    saved_cache = dict(cache)
    saved_thresh = cb._STRUCTURE_HINTS_GC_THRESHOLD
    cache.clear()
    try:
        yield
    finally:
        cache.clear()
        cache.update(saved_cache)
        cb._STRUCTURE_HINTS_GC_THRESHOLD = saved_thresh


def test_structure_hints_cache_serves_fresh(isolated_hints_cache, tmp_path):
    """A non-expired entry is returned without recomputation."""
    key = str(tmp_path)
    cb._structure_hints_cache[key] = ("CACHED_HINT", time.monotonic() + 9999.0)

    builder = EnhancedContextBuilder(str(tmp_path))
    result = builder._get_project_structure_hints()

    assert result == "CACHED_HINT"


def test_structure_hints_cache_evicts_expired(isolated_hints_cache, tmp_path):
    """Expired entries are purged on the miss path once the cache exceeds the
    GC threshold; fresh entries survive."""
    cb._STRUCTURE_HINTS_GC_THRESHOLD = 4
    now = time.monotonic()
    # 6 expired + 1 fresh => 7 > threshold(4) triggers GC on next miss.
    for i in range(6):
        cb._structure_hints_cache[f"/expired/{i}"] = ("stale", now - 1.0)
    cb._structure_hints_cache["/fresh"] = ("fresh", now + 9999.0)
    assert len(cb._structure_hints_cache) == 7

    builder = EnhancedContextBuilder(str(tmp_path))
    builder._get_project_structure_hints()  # miss on tmp_path -> GC runs

    cache = cb._structure_hints_cache
    expired_remaining = [k for k, (_, exp) in cache.items() if exp <= now]
    assert expired_remaining == [], f"expired not purged: {expired_remaining}"
    assert "/fresh" in cache, "fresh entry wrongly purged"


def test_structure_hints_cache_does_not_gc_below_threshold(isolated_hints_cache, tmp_path):
    """Below the GC threshold the miss path must NOT touch unrelated keys
    (keeps the common case O(1) and avoids surprising mutations)."""
    cb._STRUCTURE_HINTS_GC_THRESHOLD = 64
    now = time.monotonic()
    cb._structure_hints_cache["/stale-but-untouched"] = ("x", now - 1.0)

    builder = EnhancedContextBuilder(str(tmp_path))
    builder._get_project_structure_hints()

    # unrelated expired entry preserved (GC skipped: cache size < threshold)
    assert "/stale-but-untouched" in cb._structure_hints_cache


def test_structure_hints_no_vendor_dirs(tmp_path):
    """_get_project_structure_hints prunes node_modules from .py file counts
    via os.walk instead of rglob, which would descend into vendor trees."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("")

    builder = EnhancedContextBuilder(str(tmp_path))
    hints = builder._get_project_structure_hints()

    assert "src/" in hints
    assert "node_modules" not in hints
