"""
Unit tests for external_llm/context_builder.py + context_collector.py.

Covers defects fixed together:
  A. lstrip("./") is a character-SET {'.','/'} — it stripped a dotfile's
     leading dot (".config.py" -> "config.py"), so the target file appeared
     missing and/or leaked into its own Related Files. Fixed with
     removeprefix("./") (matches the go_provider.py precedent).
  B. _structure_hints_cache never evicted expired entries — keys that expired
     but were never re-accessed accumulated forever across distinct repo
     roots. Fixed with opportunistic GC on the miss path.
  C. build_context() re-spawned `git status` + `git log` on every call — the
     status half now delegates to the get_git_snapshot SSOT
     (agent_context_manager) and the recent-commits log is process-wide
     TTL-cached (P2).
"""
from __future__ import annotations

import subprocess
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


def test_structure_hints_list_dirs_and_files_in_sorted_order(isolated_hints_cache, tmp_path):
    """BUG-3: _get_project_structure_hints must list directories and files in
    case-insensitive sorted order, not iterdir()/readdir order — otherwise the
    LLM-visible project structure drifts between processes and machines."""
    # Scrambled creation order + mixed case exercises the name.lower() sort key.
    for name in ("zeta", "Mike", "alpha"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "mod.py").write_text("x = 1\n")
    for name in ("zzz.txt", "Beta.md", "aaa.json"):
        (tmp_path / name).write_text("x")

    builder = EnhancedContextBuilder(str(tmp_path))
    hints = builder._get_project_structure_hints()

    dir_pos = [hints.index(f"  - `{d}/`") for d in ("alpha", "Mike", "zeta")]
    assert dir_pos == sorted(dir_pos), f"dirs not sorted: {hints}"
    file_pos = [hints.index(f"  - `{f}`") for f in ("aaa.json", "Beta.md", "zzz.txt")]
    assert file_pos == sorted(file_pos), f"files not sorted: {hints}"


# ── Defect C: git context — status delegates to SSOT, log is TTL-cached ─────

@pytest.fixture
def isolated_git_log_cache():
    """Snapshot/restore the process-wide recent-commits TTL cache around each
    test so module-level state never leaks across tests."""
    cache = cb._git_log_cache
    saved_cache = dict(cache)
    saved_ttl = cb._GIT_LOG_TTL_S
    saved_thresh = cb._GIT_LOG_GC_THRESHOLD
    cache.clear()
    try:
        yield
    finally:
        cache.clear()
        cache.update(saved_cache)
        cb._GIT_LOG_TTL_S = saved_ttl
        cb._GIT_LOG_GC_THRESHOLD = saved_thresh


def _fake_git_run(monkeypatch, stdout="abc1234 feat: x\nabc1233 fix: y\n"):
    """Patch subprocess.run with a counting fake returning git-style stdout."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cb.subprocess, "run", fake_run)
    return calls


def test_git_status_delegates_to_snapshot_ssot(tmp_path, monkeypatch):
    """_get_git_status must route through get_git_snapshot, not spawn its own
    `git status` subprocess (single source of truth for git state)."""
    seen = []

    def fake_snapshot(root):
        seen.append(str(root))
        return {"status": " M x.py\n"}

    monkeypatch.setattr(cb, "get_git_snapshot", fake_snapshot)

    builder = EnhancedContextBuilder(str(tmp_path))
    assert builder._get_git_status() == " M x.py\n"
    assert builder._get_git_status() == " M x.py\n"
    # Delegation is per-call (the SSOT has its own 10s TTL cache) — no local
    # caching layer between the builder and the SSOT.
    assert seen == [str(tmp_path), str(tmp_path)]


def test_git_status_empty_when_snapshot_missing(tmp_path, monkeypatch):
    """A missing/empty snapshot (non-repo root) yields '' — same contract as
    the old subprocess path's failure sentinel."""
    monkeypatch.setattr(cb, "get_git_snapshot", lambda root: {})

    builder = EnhancedContextBuilder(str(tmp_path))
    assert builder._get_git_status() == ""


def test_git_log_cached_within_ttl(isolated_git_log_cache, tmp_path, monkeypatch):
    """Two rapid _get_recent_commits() calls spawn `git log` only once."""
    calls = _fake_git_run(monkeypatch)

    builder = EnhancedContextBuilder(str(tmp_path))
    # _fetch_recent_commits strips trailing whitespace, same as the old path.
    expected = "abc1234 feat: x\nabc1233 fix: y"
    assert builder._get_recent_commits() == expected
    assert builder._get_recent_commits() == expected

    assert len(calls) == 1
    assert calls[0][:2] == ["git", "log"]


def test_git_log_refetches_after_ttl_expiry(isolated_git_log_cache, tmp_path, monkeypatch):
    """Once the TTL expires the next call refetches and re-caches."""
    cb._GIT_LOG_TTL_S = 0.05
    calls = _fake_git_run(monkeypatch)

    builder = EnhancedContextBuilder(str(tmp_path))
    assert builder._get_recent_commits()
    time.sleep(0.06)
    assert builder._get_recent_commits()

    assert len(calls) == 2


def test_git_log_cache_keyed_by_root_and_count(isolated_git_log_cache, tmp_path, monkeypatch):
    """Distinct repo roots and log depths never share entries."""
    other = tmp_path / "other"
    other.mkdir()
    calls = _fake_git_run(monkeypatch)

    a = EnhancedContextBuilder(str(tmp_path))
    b = EnhancedContextBuilder(str(other))
    a._get_recent_commits(count=3)
    b._get_recent_commits(count=3)
    a._get_recent_commits(count=5)
    assert len(calls) == 3  # all three keys are distinct

    # All keys are now warm — no new subprocess spawns.
    a._get_recent_commits(count=3)
    b._get_recent_commits(count=3)
    a._get_recent_commits(count=5)
    assert len(calls) == 3


def test_git_log_cache_gc_preserves_fresh_and_unrelated(isolated_git_log_cache, tmp_path, monkeypatch):
    """Expired entries are purged once the cache exceeds the GC threshold;
    fresh entries survive (same contract as _structure_hints_cache)."""
    cb._GIT_LOG_GC_THRESHOLD = 4
    now = time.monotonic()
    for i in range(6):
        cb._git_log_cache[(f"/expired/{i}", 3)] = ("stale", now - 1.0)
    cb._git_log_cache[("/fresh", 3)] = ("fresh", now + 9999.0)

    calls = _fake_git_run(monkeypatch)
    builder = EnhancedContextBuilder(str(tmp_path))
    assert builder._get_recent_commits()  # miss on tmp_path -> GC runs

    cache = cb._git_log_cache
    expired_remaining = [k for k, (_, exp) in cache.items() if exp <= now]
    assert expired_remaining == [], f"expired not purged: {expired_remaining}"
    assert ("/fresh", 3) in cache, "fresh entry wrongly purged"
    assert len(calls) == 1


def test_super_builder_shares_git_ssot_and_log_cache(isolated_git_log_cache, tmp_path, monkeypatch):
    """SuperContextBuilder must route through the same SSOT + TTL cache — one
    `git log` spawn serves both builders."""
    from external_llm.super_context_builder import SuperContextBuilder

    snapshots = []

    def fake_snapshot(root):
        snapshots.append(str(root))
        return {"status": " M z.py\n"}

    monkeypatch.setattr(cb, "get_git_snapshot", fake_snapshot)
    monkeypatch.setattr("external_llm.super_context_builder.get_git_snapshot", fake_snapshot)

    builder = EnhancedContextBuilder(str(tmp_path))
    super_builder = SuperContextBuilder(str(tmp_path))
    assert builder._get_git_status() == " M z.py\n"
    assert super_builder._get_git_status() == " M z.py\n"
    assert snapshots == [str(tmp_path), str(tmp_path)]

    # Log cache is process-wide: one spawn for both builders.
    calls = _fake_git_run(monkeypatch)
    assert builder._get_recent_commits()
    assert super_builder._get_recent_commits()
    assert len(calls) == 1


# ── P21-3: file-context reads are head-bounded ───────────────────────────────

def test_file_context_small_full(tmp_path):
    builder = EnhancedContextBuilder(str(tmp_path))
    (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    out = builder._build_file_context("a.py")
    assert "def a()" in out
    assert "**Total lines**: 3" in out


def test_file_context_huge_head_bounded(tmp_path):
    from external_llm.context_builder import _FILE_CONTEXT_MAX_BYTES
    builder = EnhancedContextBuilder(str(tmp_path))
    with open(tmp_path / "big.py", "wb") as f:
        f.write(b"x = 1\n" * 200_000)  # 1.2 MiB — beyond the 1 MiB head
    out = builder._build_file_context("big.py")
    assert "head only" in out, "truncated block must say it is a head only"
    assert "**Total lines**: >=" in out
    assert len(out) < _FILE_CONTEXT_MAX_BYTES + 500


def test_file_context_utf8_boundary_clean(tmp_path):
    builder = EnhancedContextBuilder(str(tmp_path))
    with open(tmp_path / "ko.txt", "wb") as f:
        f.write(("가" * 400_000).encode())  # 1.2 MiB — 1 MiB % 3 == 1
    out = builder._build_file_context("ko.txt")
    head = out[: out.index("head only")] if "head only" in out else out
    assert "\ufffd" not in head, "cut must not split a multi-byte char"


def test_related_files_head_bounded(tmp_path, monkeypatch):
    from external_llm.context_builder import _FILE_CONTEXT_MAX_BYTES
    builder = EnhancedContextBuilder(str(tmp_path))
    with open(tmp_path / "mymod.py", "wb") as f:
        f.write(b"x = 1\n" * 200_000)
    monkeypatch.setattr(builder, "_find_related_files", lambda *a, **kw: ["mymod.py"])
    out = builder._build_related_files_context("main.py", max_files=3)
    assert "mymod.py" in out
    assert "TRUNCATED" in out
    assert len(out) < _FILE_CONTEXT_MAX_BYTES + 500


# ── P22-2: related-files fallback containment ────────────────────────────────

def test_find_related_files_rejects_escape_path(tmp_path):
    """P22-2: the fallback import-scan must not resolve outside the repo."""
    secret = tmp_path.parent / "SECRET.py"
    secret.write_text("SECRET = 1\n")
    builder = EnhancedContextBuilder(str(tmp_path))
    assert builder._find_related_files("../SECRET.py", max_files=5) == []
