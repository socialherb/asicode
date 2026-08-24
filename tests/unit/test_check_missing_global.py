"""Zero-tolerance missing-``global`` gate (scripts/check_missing_global.py).

The gate covers the half of the missing-``global`` bug that ruff cannot see.
F823 (gated by scripts/check_f823_none.py) fires only when the name is also
READ before assignment; a write-only assignment raises nothing, warns nothing,
and silently discards the write.  Both shapes shipped in this repo — see the
script's docstring for the two incidents.

These tests pin the gate mechanics: which bindings count as a violation, which
deliberately do not, and that a planted violation actually fails ``main()``.
That last one matters more than it looks: an earlier hand-check of this gate
"passed" on a file full of planted bugs purely because the file sat outside the
repo root, so ``_resolve_scan_paths`` discarded it and the gate scanned the
(clean) repo instead.  A gate test that never reaches the failing branch is
indistinguishable from no test.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_missing_global.py"
_spec = importlib.util.spec_from_file_location("check_missing_global", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


def _names(source: str) -> list[tuple[str, str]]:
    """``(func, name)`` pairs the gate reports for *source*."""
    return [(v[1], v[2]) for v in g._violations_in(source, "m.py")]


def _shapes(source: str) -> list[str]:
    return [v[3] for v in g._violations_in(source, "m.py")]


# ── path normalization (shared contract with the other gates) ────────────────


def test_resolve_scan_paths_normalizes_abs_and_relative():
    rel = "scripts/check_missing_global.py"
    assert g._resolve_scan_paths([str(_SCRIPT)]) == [rel]
    assert g._resolve_scan_paths([rel]) == [rel]
    assert g._resolve_scan_paths([]) is None  # no args → full-repo scan
    assert g._resolve_scan_paths(["--flag"]) is None  # flags filtered by main


def test_resolve_scan_paths_rejects_out_of_repo_and_non_python():
    assert g._resolve_scan_paths(["/etc/passwd"]) is None  # '..' → rejected
    assert g._resolve_scan_paths(["README.md"]) is None  # not .py


def test_iter_py_files_prunes_skipped_dirs_entirely(tmp_path, monkeypatch):
    """Full scan must not descend into SKIP_DIRS (os.walk prune, not rglob+filter).

    Original code walked every skipped directory (14548 entries for ~954 kept)
    and discarded them after traversal. A pruned walk visits only the scanned
    trees, so a planted .venv with *.py files must never surface.
    """
    monkeypatch.setattr(g, "REPO", tmp_path)
    for skipped in (".venv", "__pycache__", "build", "node_modules", "dist", ".git"):
        d = tmp_path / skipped
        d.mkdir(parents=True, exist_ok=True)
        (d / "mod.py").write_text("_d = 0\ndef f():\n    _d = 1\n", encoding="utf-8")
    # full-repo scan (no args): skipped dirs are pruned — their .py never yields
    assert list(g._iter_py_files(None)) == []
    # explicit-path scan must still respect the skip (parity with rglob era)
    assert list(g._iter_py_files([str(tmp_path / ".venv" / "mod.py")])) == []
    # a real file outside skipped dirs is found
    (tmp_path / "real.py").write_text("_d = 0\ndef f():\n    _d = 1\n", encoding="utf-8")
    assert [rel for rel, p in g._iter_py_files(None)] == [Path("real.py")]


# ── what counts as module state ──────────────────────────────────────────────


def test_only_module_scope_assignments_count_as_globals():
    import ast

    tree = ast.parse(
        "import os\n"
        "from pathlib import Path\n"
        "PLAIN = 1\n"
        "ANNOTATED: int = 2\n"
        "AUGMENTED = 0\n"
        "AUGMENTED += 1\n"
        "TUPLE_A, TUPLE_B = 1, 2\n"
        "def fn(): pass\n"
        "class Cls: pass\n"
    )
    assert g._module_assigned_names(tree) == {
        "PLAIN",
        "ANNOTATED",
        "AUGMENTED",
        "TUPLE_A",
        "TUPLE_B",
    }
    # imports / def / class are NOT module state for this gate
    assert {"os", "Path", "fn", "Cls"}.isdisjoint(g._module_assigned_names(tree))


# ── violations ───────────────────────────────────────────────────────────────


def test_write_only_assignment_is_flagged():
    """The shape ruff cannot see: no read, so no F823, so no diagnostic."""
    src = "_dirty = False\ndef f():\n    _dirty = True\n"
    assert _names(src) == [("f", "_dirty")]
    assert "write-only" in _shapes(src)[0]


def test_read_then_write_is_flagged_and_labelled_as_f823():
    src = "_dirty = False\ndef f():\n    if not _dirty:\n        return\n    _dirty = True\n"
    assert _names(src) == [("f", "_dirty")]
    assert "read-then-write" in _shapes(src)[0]


def test_nested_function_scope_is_flagged():
    src = "_dirty = False\ndef outer():\n    def inner():\n        _dirty = True\n    inner()\n"
    assert _names(src) == [("inner", "_dirty")]


def test_walrus_rebind_is_flagged():
    src = "_dirty = False\ndef f():\n    if (_dirty := True):\n        pass\n"
    assert _names(src) == [("f", "_dirty")]


def test_walrus_inside_a_lambda_is_flagged():
    """Lambda scopes must be matched by symtable's name for them ("lambda").

    A lambda holds no statements, so it looked unreachable for this gate — but
    a walrus binds a discarded local there exactly like anywhere else, and
    naming the scope "<lambda>" silently skipped every lambda in the repo.
    """
    src = "_x = 0\nf = lambda: (_x := 1)\n"
    assert _names(src) == [("lambda", "_x")]


def test_augmented_assignment_is_flagged():
    src = "_count = 0\ndef f():\n    _count += 1\n"
    assert _names(src) == [("f", "_count")]


def test_method_body_is_flagged():
    src = "_dirty = False\nclass C:\n    def m(self):\n        _dirty = True\n"
    assert _names(src) == [("m", "_dirty")]


def test_every_violating_function_is_reported_separately():
    src = "_a = 0\n_b = 0\ndef f():\n    _a = 1\ndef h():\n    _b = 2\n"
    assert sorted(_names(src)) == [("f", "_a"), ("h", "_b")]


# ── deliberate non-violations ────────────────────────────────────────────────


def test_declared_global_is_clean():
    src = "_dirty = False\ndef f():\n    global _dirty\n    _dirty = True\n"
    assert _names(src) == []


def test_nonlocal_closure_is_clean():
    """`nonlocal` binds the enclosing local, not the module global."""
    src = (
        "_dirty = False\n"
        "def outer():\n"
        "    _dirty = False\n"
        "    def inner():\n"
        "        nonlocal _dirty\n"
        "        _dirty = True\n"
        "    inner()\n"
    )
    # outer()'s own `_dirty = False` IS a shadowing assignment and is reported;
    # inner() is not, because nonlocal makes its target outer's local.
    assert _names(src) == [("outer", "_dirty")]


def test_in_place_mutation_is_clean():
    """Mutating a global container never rebinds the name."""
    src = "_counts = {}\ndef f():\n    _counts['a'] = 1\n    _counts.clear()\n"
    assert _names(src) == []


def test_parameter_shadow_is_clean():
    src = "_dirty = False\ndef f(_dirty):\n    return _dirty\n"
    assert _names(src) == []


def test_for_target_and_with_and_except_shadows_are_clean():
    """Only ASSIGNMENT binds are in scope for this gate — see its docstring."""
    src = (
        "_x = 0\n"
        "def f(items):\n"
        "    for _x in items:\n"
        "        print(_x)\n"
        "def h(cm):\n"
        "    with cm as _x:\n"
        "        print(_x)\n"
        "def k():\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as _x:\n"
        "        print(_x)\n"
    )
    assert _names(src) == []


def test_local_not_matching_any_module_global_is_clean():
    src = "_dirty = False\ndef f():\n    scratch = 1\n    return scratch\n"
    assert _names(src) == []


def test_syntax_error_defers_to_the_syntax_gates():
    assert g._violations_in("def f(:\n", "m.py") == []


# ── end-to-end main() (the branch the manual probe originally missed) ────────


def test_main_fails_on_a_planted_violation(tmp_path, monkeypatch, capsys):
    (tmp_path / "mod.py").write_text("_dirty = False\ndef f():\n    _dirty = True\n", encoding="utf-8")
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py"])
    assert g.main() == 1
    out = capsys.readouterr().out
    assert "mod.py:3" in out
    assert "write-only" in out


def test_main_passes_when_global_is_declared(tmp_path, monkeypatch, capsys):
    (tmp_path / "mod.py").write_text(
        "_dirty = False\ndef f():\n    global _dirty\n    _dirty = True\n", encoding="utf-8"
    )
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py"])
    assert g.main() == 0
    assert "✅" in capsys.readouterr().out


def test_main_skips_vendored_and_cache_dirs(tmp_path, monkeypatch):
    for skipped in (".venv", "__pycache__", "build", "node_modules"):
        d = tmp_path / skipped
        d.mkdir()
        (d / "mod.py").write_text("_d = 0\ndef f():\n    _d = 1\n", encoding="utf-8")
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py"])
    assert g.main() == 0


@pytest.mark.slow
def test_repo_is_at_zero():
    """The gate has no baseline — the whole repo must be clean."""
    monkey_argv = sys.argv
    sys.argv = ["check_missing_global.py"]
    try:
        assert g.main() == 0
    finally:
        sys.argv = monkey_argv


# ── per-file analysis cache (P15, 2026-08-24) ─────────────────────────────────

# Cache contract is shared with the other analysis gates (A307, 786ffcdc):
#  * fingerprint (st_mtime_ns, st_size) — any content change drifts the key;
#  * fail-open — corruption / version mismatch / OSError → full recompute;
#  * per-repo path derived from REPO, so tmp_path REPO monkeypatch isolation.


def _plant(tmp_path, monkeypatch, *, with_cache: bool = True):
    """REPO→tmp_path, one clean module; returns (tmp_path, module_rel)."""
    monkeypatch.setattr(g, "REPO", tmp_path)
    rel = "mod.py"
    (tmp_path / rel).write_text("_d = 0\ndef f():\n    _d = 1\n", encoding="utf-8")
    if with_cache:
        g._save_cache({rel: (g._stat_fingerprint(tmp_path / rel), [])})
    return tmp_path, rel


def test_cache_hit_preserves_clean_verdict(tmp_path, monkeypatch, capsys):
    """A fingerprint hit returns the cached verdict without re-reading the file."""
    tmp_path, _rel = _plant(tmp_path, monkeypatch, with_cache=True)
    assert (tmp_path / ".cache" / "missing_global_v1.json").exists()
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py"])
    assert g.main() == 0
    assert "✅" in capsys.readouterr().out
    # the file was never re-read: a fictional planted violation in the cache
    # still surfaces (fail-open only on mismatch, not on hit)


def test_cache_miss_after_content_change_restores_recomputed_verdict(tmp_path, monkeypatch, capsys):
    """Changing the file (mtime+size drift) must invalidate and recompute."""
    tmp_path, rel = _plant(tmp_path, monkeypatch, with_cache=True)
    # drift: add a second planted violation (valid indentation) → size+mtime change
    (tmp_path / rel).write_text("_d = 0\ndef f():\n    _d = 1\ndef h():\n    _d = 2\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py"])
    assert g.main() == 1  # recomputed from the drifted file
    assert "mod.py:3" in capsys.readouterr().out


def test_cache_ignored_under_no_cache_flag(tmp_path, monkeypatch, capsys):
    """--no-cache forces a re-read even when the fingerprint matches."""
    tmp_path, rel = _plant(tmp_path, monkeypatch, with_cache=True)
    (tmp_path / rel).write_text("_d = 0\ndef f():\n    _d = 1\ndef h():\n    _d = 2\n", encoding="utf-8")
    # the cache entry now holds the CLEAN fingerprint for the drifted file —
    # only --no-cache can force the re-read that surfaces the violation
    g._save_cache({rel: (g._stat_fingerprint(tmp_path / rel), [])})
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py", "--no-cache"])
    assert g.main() == 1
    assert "mod.py:3" in capsys.readouterr().out


def test_corrupt_cache_fails_open(tmp_path, monkeypatch, capsys):
    """Garbage / version-mismatch / schema-broken cache must full-recompute."""
    tmp_path, _rel = _plant(tmp_path, monkeypatch, with_cache=True)
    cache = tmp_path / ".cache" / "missing_global_v1.json"
    cache.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py"])
    assert g.main() == 1  # recomputed the planted violation
    assert "mod.py:3" in capsys.readouterr().out

    # and version mismatch also fails open
    _plant(tmp_path, monkeypatch, with_cache=True)
    cache.write_text('{"version": 999, "files": {}}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_missing_global.py"])
    assert g.main() == 1
