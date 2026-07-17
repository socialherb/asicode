"""Tests for symbol_index incremental rebuild + FIFO cache cap.

The incremental rebuild must produce results identical to a full re-parse for
added / changed / removed files, and the cache must be bounded by an entry cap.
"""
import os
import textwrap
import threading

import pytest

from external_llm.agent import symbol_index as si


def _write(path, body):
    with open(path, "w") as f:
        f.write(textwrap.dedent(body))


@pytest.fixture()
def repo(tmp_path):
    """Minimal repo with three Python files."""
    _write(str(tmp_path / "a.py"), """\
        class Aa:
            pass


        def aa():
            pass
    """)
    _write(str(tmp_path / "b.py"), """\
        class Bb:
            pass


        CONST_B = 1
    """)
    (tmp_path / "sub").mkdir()
    _write(str(tmp_path / "sub" / "c.py"), """\
        async def cc():
            pass


        X: int = 5
    """)
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Each test gets a clean _INDEX_CACHE so they don't interfere."""
    saved = dict(si._INDEX_CACHE)
    si._INDEX_CACHE.clear()
    yield
    si._INDEX_CACHE.clear()
    si._INDEX_CACHE.update(saved)


def _force_ttl_expiry(root):
    """Rewrite the cache entry's timestamp to 0 so the next call re-walks."""
    file_idx, name_idx, mtimes, _ = si._INDEX_CACHE[str(root)]
    si._INDEX_CACHE[str(root)] = (file_idx, name_idx, mtimes, 0.0)


def _full_reparse_name_index(root):
    """Ground truth: full re-parse from scratch, derived into name index."""
    mtimes = si._collect_mtimes(str(root))
    full = si._rebuild_file_index(str(root), mtimes)
    return si._name_index_from_file_index(full)


def test_cold_cache_indexes_all_top_level_symbols(repo):
    idx = si.build_repo_symbol_index(str(repo))
    assert set(idx) == {"Aa", "aa", "Bb", "CONST_B", "cc", "X"}
    # Kinds are preserved.
    assert {loc.kind for loc in idx["Aa"]} == {"class"}
    assert {loc.kind for loc in idx["aa"]} == {"function"}
    assert {loc.kind for loc in idx["cc"]} == {"async_function"}
    assert {loc.kind for loc in idx["CONST_B"]} == {"constant"}


def test_no_change_reuses_index(repo):
    first = si.build_repo_symbol_index(str(repo))
    _force_ttl_expiry(repo)
    second = si.build_repo_symbol_index(str(repo))
    assert first == second


def test_changed_file_incremental_equals_full_reparse(repo):
    si.build_repo_symbol_index(str(repo))
    # Edit b.py: drop CONST_B, add bb_new.
    _write(str(repo / "b.py"), """\
        class Bb:
            pass


        def bb_new():
            pass
    """)
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "CONST_B" not in idx
    assert "bb_new" in idx
    assert idx == _full_reparse_name_index(repo)


def test_added_file_incremental_equals_full_reparse(repo):
    si.build_repo_symbol_index(str(repo))
    _write(str(repo / "d.py"), """\
        class Dd:
            pass
    """)
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "Dd" in idx
    assert idx == _full_reparse_name_index(repo)


def test_removed_file_incremental_equals_full_reparse(repo):
    si.build_repo_symbol_index(str(repo))
    os.remove(str(repo / "a.py"))
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "Aa" not in idx
    assert "aa" not in idx
    assert idx == _full_reparse_name_index(repo)


def test_incremental_reuses_untouched_entries(repo):
    """Only the changed file should be re-scanned; untouched files keep their
    parsed SymbolLocation objects verbatim (identity preserved)."""
    si.build_repo_symbol_index(str(repo))
    file_idx, name_idx, mtimes, _ = si._INDEX_CACHE[str(repo)]
    a_locs_before = file_idx["a.py"]
    # Touch b.py only.
    _write(str(repo / "b.py"), """\
        class Bb:
            pass
    """)
    _force_ttl_expiry(repo)
    si.build_repo_symbol_index(str(repo))
    file_idx2, _, _, _ = si._INDEX_CACHE[str(repo)]
    # a.py was untouched -> its list object is reused as-is (same identity).
    assert file_idx2["a.py"] is a_locs_before


def test_determinism_locations_sorted_by_file_then_kind(repo):
    """Same-named symbol in multiple files must be sorted (file_path, kind)."""
    _write(str(repo / "z.py"), """\
        def Aa():
            pass
    """)
    si.build_repo_symbol_index(str(repo))  # populate
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    aa_locs = idx["Aa"]
    assert [(l.file_path, l.kind) for l in aa_locs] == sorted(
        (l.file_path, l.kind) for l in aa_locs
    )


def test_cache_is_bounded_by_entry_cap(repo, tmp_path_factory):
    """Inserting more repos than the cap evicts the oldest (FIFO)."""
    orig_cap = si._INDEX_CACHE_MAX_ENTRIES
    si._INDEX_CACHE_MAX_ENTRIES = 2
    try:
        si._INDEX_CACHE.clear()
        roots = []
        for i in range(4):
            r = tmp_path_factory.mktemp(f"cap_{i}_")
            _write(str(r / "f.py"), f"class F{i}:\n    pass\n")
            si.build_repo_symbol_index(str(r))
            roots.append(str(r))
        assert len(si._INDEX_CACHE) == 2
        # oldest two evicted, newest two kept
        assert roots[0] not in si._INDEX_CACHE
        assert roots[1] not in si._INDEX_CACHE
        assert roots[2] in si._INDEX_CACHE
        assert roots[3] in si._INDEX_CACHE
    finally:
        si._INDEX_CACHE_MAX_ENTRIES = orig_cap


def test_apply_incremental_does_not_poison_cached_file_index(repo):
    """_apply_incremental must mutate a copy, leaving the cached alias intact."""
    si.build_repo_symbol_index(str(repo))
    file_idx, _name_idx, mtimes, _ = si._INDEX_CACHE[str(repo)]
    # Simulate a removed file by dropping it from cur_mtimes.
    cur = dict(mtimes)
    cur.pop("a.py")
    new_file_idx, _ = si._apply_incremental(str(repo), file_idx, mtimes, cur)
    # The cached (old) file_index still has a.py -- not mutated in place.
    assert "a.py" in file_idx
    assert "a.py" not in new_file_idx


# ── Cooperative-cancel contract ─────────────────────────────────────────
# Cancel is control flow, NOT a value. A cold-path cancel must RAISE
# _CancelledBuild (never collapse to {}), because {} is indistinguishable from
# "repo has no symbols" — consumers (executor_verification P6.9 / P6.10) would
# then treat an existing symbol as "missing repo-wide" and emit a hard
# "you MUST create it" directive, risking a DUPLICATE DEFINITION.


def test_cold_cancel_propagates_not_returns_empty(repo):
    """Cold-path cancel raises _CancelledBuild instead of returning {}.

    Regression for the duplicate-definition bug: the old `return {}` made the
    two consumers falsely mandate creating an already-existing symbol.
    """
    ev = threading.Event()
    ev.set()
    si._INDEX_CACHE.clear()  # force cold path
    with pytest.raises(si._CancelledBuild):
        si.build_repo_symbol_index(str(repo), cancel_event=ev)


def test_cold_cancel_does_not_poison_cache(repo):
    """After a cold cancel the cache stays empty; the next call rebuilds."""
    ev = threading.Event()
    ev.set()
    si._INDEX_CACHE.clear()
    with pytest.raises(si._CancelledBuild):
        si.build_repo_symbol_index(str(repo), cancel_event=ev)
    assert str(repo) not in si._INDEX_CACHE
    # A subsequent non-cancelled call performs the full build.
    ev.clear()
    idx = si.build_repo_symbol_index(str(repo))
    assert "Aa" in idx and "Bb" in idx


def test_apply_incremental_honors_cancel_event():
    """_apply_incremental raises _CancelledBuild mid-loop (branch-switch gap).

    A branch switch can flip thousands of mtimes at once, making this loop the
    most expensive part of the build and the exact moment cancel is needed.
    Previously this loop had no checkpoint; now it bails out without committing
    (new_file_index is a local copy → cache never poisoned).
    """
    ev = threading.Event()
    ev.set()
    # cur_mtimes has entries absent from old → the re-scan loop is entered →
    # the per-iteration checkpoint fires immediately.
    with pytest.raises(si._CancelledBuild):
        si._apply_incremental(
            "/unused", {}, {}, {"a.py": 1.0, "b.py": 2.0}, cancel_event=ev
        )
    # cancel_event=None stays inert (inline / non-interactive callers) — the
    # loop runs to completion without raising. (The file doesn't exist, so
    # _scan_file yields an empty list; only the no-raise contract matters here.)
    out, mtimes = si._apply_incremental(
        "/unused", {}, {}, {"a.py": 1.0}, cancel_event=None
    )
    assert mtimes == {"a.py": 1.0}
    assert "a.py" in out


def test_consumer_skip_pattern_prevents_spurious_create(repo):
    """Consumer contract: catch _CancelledBuild BEFORE `except Exception:`.

    Mirrors the executor_verification P6.9 / P6.10 fix. On cold cancel the caller
    skips the dependent decision (does NOT call decide_import_vs_create) rather
    than fall through to a hard "create" directive.
    """
    from external_llm.agent.symbol_index import decide_import_vs_create

    ev = threading.Event()
    ev.set()
    si._INDEX_CACHE.clear()

    # Correct (post-fix) consumer pattern: catch the cancel, skip the decision.
    cancelled = False
    index = None
    try:
        index = si.build_repo_symbol_index(str(repo), cancel_event=ev)
    except si._CancelledBuild:
        cancelled = True  # → caller skips decide_import_vs_create entirely
    assert cancelled is True
    assert index is None
    # Because decide_import_vs_create is skipped, no verdict is produced.

    # Contrast — the OLD `except Exception: index = {}` behaviour: feeding {} to
    # decide_import_vs_create would WRONGLY mandate creating an existing symbol.
    wrong = decide_import_vs_create("Aa", "x.py", {})
    assert wrong["action"] == "create"  # the bug the skip-pattern now avoids


# ── git ls-files SSOT integration (shared with write_tools._repo_file_index) ──
# These guard the duplicate-definition invariant: a symbol defined ONLY in a
# gitignored vendored/generated copy must NOT leak into the index, or
# decide_import_vs_create would flip a correct "create" into a wrong "import".

def _make_git_repo(tmp_path, name="grepo"):
    """Create a real (empty) git checkout under tmp_path and return its path."""
    import subprocess
    repo = tmp_path / name
    repo.mkdir()
    for args in (
        ["git", "-C", str(repo), "init", "-q"],
        ["git", "-C", str(repo), "config", "user.email", "t@t.test"],
        ["git", "-C", str(repo), "config", "user.name", "test"],
    ):
        subprocess.run(args, capture_output=True, check=True)
    return repo


def _git_commit(repo):
    import subprocess
    subprocess.run(["git", "-C", str(repo), "add", "-A"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], capture_output=True, check=True)


def test_gitignored_vendored_symbol_excluded_from_index(tmp_path):
    """Regression (duplicate-definition guard): a symbol defined ONLY in a
    gitignored vendored copy must NOT appear in the repo-wide symbol index.

    If it leaked in, decide_import_vs_create would see the symbol as "already
    exists elsewhere" and tell the LLM to import it — masking a genuine missing
    definition. The git-first _collect_mtimes path (shared SSOT with
    write_tools) respects .gitignore, so vendored copies are never scanned.
    """
    repo = _make_git_repo(tmp_path)
    (repo / "app.py").write_text("class EnemyBullet:\n    pass\n")
    (repo / "vendor").mkdir()
    (repo / "vendor" / "enemy.py").write_text("class EnemyBullet:\n    pass\n")
    (repo / ".gitignore").write_text("vendor/\n")
    _git_commit(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "EnemyBullet" in idx
    # ONLY from app.py, NOT the vendored copy under the gitignored vendor/.
    assert [loc.file_path for loc in idx["EnemyBullet"]] == ["app.py"]


def test_non_ascii_path_symbol_is_indexed(tmp_path):
    """Regression: a symbol in a Korean-named file is indexed via the git -z path.

    Confirms the SSOT change fixed the os.walk path's historical weakness on
    CJK paths (git porcelain C-quoting would otherwise break membership, so a
    legitimately-defined symbol could be missed → false "create" directive).
    """
    repo = _make_git_repo(tmp_path, name="krepo")
    (repo / "src").mkdir()
    (repo / "src" / "모듈.py").write_text("class 한글클래스:\n    pass\n")
    _git_commit(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "한글클래스" in idx
    assert idx["한글클래스"][0].file_path == "src/모듈.py"


def test_os_walk_fallback_when_not_git_checkout(tmp_path):
    """Non-git tree → git_list_repo_files returns None → os.walk fallback used.

    Ensures the fallback path still produces a valid index (the _SKIP_DIRS
    hardcoded set is narrower than .gitignore, acceptable for non-git trees).
    """
    repo = tmp_path / "plain"
    repo.mkdir()
    (repo / "plain.py").write_text("def plain_fn():\n    pass\n")
    idx = si.build_repo_symbol_index(str(repo))
    assert "plain_fn" in idx
