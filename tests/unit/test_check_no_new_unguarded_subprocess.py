"""Tests for the net-new unguarded-subprocess baseline-diff gate.

Mirrors the IO/analysis split + precision-parametrize + vacuous-trap pattern of
``test_check_no_new_silent_except.py``: the live invariant (no new unguarded
calls beyond baseline) is vacuously-passable if the scanner is weakened, so the
*precision* parametrize tests are what guard detection capability.

The motivating defect (symbol_search rg P0): a ``subprocess.run`` to an optional
external binary inside a ``try`` whose ``except`` caught only
``subprocess.SubprocessError`` / ``(AttributeError, TypeError)`` — none of which
catch ``FileNotFoundError`` (an ``OSError``), so on a plain ``pip install`` the
binary's absence crashed the tool.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_new_unguarded_subprocess.py"
_spec = importlib.util.spec_from_file_location("check_no_new_unguarded_subprocess", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


def _src(body_except: str) -> str:
    """A function whose one subprocess.run is wrapped by a try with *body_except*."""
    return (
        "import subprocess\n"
        "def f():\n"
        "    try:\n"
        "        subprocess.run(['rg', 'x'])\n"
        f"    {body_except}\n"
        "        pass\n"
    )


# --- precision: the EXACT P0 defect shapes ARE flagged (≥1 key) ---------------
# These are the three except-clause shapes that shipped the rg crash. None catch
# OSError, so a missing binary's FileNotFoundError escapes the handler.
@pytest.mark.parametrize(
    "body_except",
    [
        "except subprocess.SubprocessError:",            # P0 site #2 verbatim
        "except (AttributeError, TypeError):",           # P0 site #1 verbatim
        "except (AttributeError, TypeError, subprocess.SubprocessError):",  # site #3
        "except subprocess.CalledProcessError:",         # exit-code-only
        "except subprocess.TimeoutExpired:",             # timeout-only
    ],
    ids=["SubprocessError", "AttrError-TypeError", "tuple-missing-OSError",
         "CalledProcessError", "TimeoutExpired"],
)
def test_scan_flags_p0_defect_shapes(body_except):
    keys = g._scan_source(_src(body_except), "m.py")
    assert keys == ["m.py::<module>::f::0"], keys


def test_scan_flags_no_try_at_all():
    src = "import subprocess\n" "def f():\n    subprocess.run(['rg'])\n"
    assert g._scan_source(src, "m.py") == ["m.py::<module>::f::0"]


def test_scan_flags_call_in_finally_region():
    # An OSError handler on the SAME try does NOT cover a call in `finally` —
    # the handler only applies to the try BODY. Validates the region check.
    src = (
        "import subprocess\n"
        "def f():\n"
        "    try:\n"
        "        x = 1\n"
        "    except OSError:\n"
        "        pass\n"
        "    finally:\n"
        "        subprocess.run(['rg'])\n"
    )
    assert g._scan_source(src, "m.py") == ["m.py::<module>::f::0"]


# --- precision: OSError-guarding shapes are NOT flagged (0 keys) --------------
@pytest.mark.parametrize(
    "body_except",
    [
        "except OSError:",
        "except FileNotFoundError:",
        "except (subprocess.SubprocessError, OSError):",   # the FIX shape
        "except (OSError, AttributeError):",
        "except Exception:",
        "except BaseException:",
        "except:",                                          # bare
        "except IOError:",                                  # OSError alias
    ],
    ids=["OSError", "FileNotFoundError", "tuple-with-OSError", "tuple-leading-OSError",
         "Exception", "BaseException", "bare", "IOError-alias"],
)
def test_scan_passes_oserror_guarded(body_except):
    assert g._scan_source(_src(body_except), "m.py") == []


def test_scan_passes_nested_outer_try_catches_oserror():
    # Nearest (inner) try catches only SubprocessError, but the OUTER try catches
    # OSError → the missing-binary error propagates through the inner try and is
    # caught by the outer. Must be reported GUARDED. This is the key correctness
    # check for the walk-up-to-ancestor logic.
    src = (
        "import subprocess\n"
        "def f():\n"
        "    try:\n"
        "        try:\n"
        "            subprocess.run(['rg'])\n"
        "        except subprocess.SubprocessError:\n"
        "            pass\n"
        "    except OSError:\n"
        "        pass\n"
    )
    assert g._scan_source(src, "m.py") == []


# --- precision: contextlib.suppress is an equivalent guard --------------------
# The silent-except program converts try/except/pass into suppress (SIM105 is
# selected repo-wide), so the subprocess guard must accept both shapes or the
# two gates fight each other: a suppress-wrapped call would look "unguarded".
@pytest.mark.parametrize(
    "suppress_args",
    [
        "OSError",
        "OSError, subprocess.SubprocessError",   # the FIX shape as suppress
        "Exception",                             # OSError supertype
        "BaseException",
    ],
    ids=["OSError", "tuple-with-OSError", "Exception", "BaseException"],
)
def test_scan_passes_suppress_guarded(suppress_args):
    src = (
        "import subprocess\n"
        "import contextlib\n"
        "def f():\n"
        f"    with contextlib.suppress({suppress_args}):\n"
        "        subprocess.run(['rg'])\n"
    )
    assert g._scan_source(src, "m.py") == []


@pytest.mark.parametrize(
    "suppress_args",
    [
        "ValueError",                             # misses OSError entirely
        "subprocess.SubprocessError",             # timeout-only, not OSError
    ],
    ids=["ValueError", "SubprocessError"],
)
def test_scan_flags_suppress_missing_oserror(suppress_args):
    # suppress without OSError (or a supertype) is the same defect as the
    # original rg crash: a missing binary's FileNotFoundError escapes.
    src = (
        "import subprocess\n"
        "import contextlib\n"
        "def f():\n"
        f"    with contextlib.suppress({suppress_args}):\n"
        "        subprocess.run(['rg'])\n"
    )
    assert g._scan_source(src, "m.py") == ["m.py::<module>::f::0"], src


def test_scan_flags_bare_with_is_not_a_guard():
    # A plain `with open(...)` is NOT an exception guard — only suppress counts.
    src = (
        "import subprocess\n"
        "def f():\n"
        "    with open('x', 'w') as fh:\n"
        "        subprocess.run(['rg'], stdout=fh)\n"
    )
    assert g._scan_source(src, "m.py") == ["m.py::<module>::f::0"], src


def test_scan_flags_nested_when_no_ancestor_catches_oserror():
    # Both inner and outer miss OSError → still unguarded. Counter-weight to the
    # above: the walk-up must not give up early at the inner try.
    src = (
        "import subprocess\n"
        "def f():\n"
        "    try:\n"
        "        try:\n"
        "            subprocess.run(['rg'])\n"
        "        except subprocess.SubprocessError:\n"
        "            pass\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    assert g._scan_source(src, "m.py") == ["m.py::<module>::f::0"]


# --- alias support: `import subprocess as _sp` then `_sp.run(...)` ------------
def test_scan_handles_import_alias():
    src = (
        "import subprocess as _sp\n"
        "def f():\n"
        "    try:\n"
        "        _sp.run(['rg'])\n"
        "    except subprocess.SubprocessError:\n"
        "        pass\n"
    )
    assert g._scan_source(src, "m.py") == ["m.py::<module>::f::0"]


# --- non-subprocess attribute calls of the same name are ignored --------------
def test_scan_ignores_same_named_non_subprocess_call():
    # A `.run(...)` on something not bound to subprocess is not in scope.
    src = (
        "import subprocess\n"
        "def f():\n"
        "    try:\n"
        "        other.run(['rg'])\n"
        "    except subprocess.SubprocessError:\n"
        "        pass\n"
    )
    assert g._scan_source(src, "m.py") == []


# --- key stability under line drift ------------------------------------------
def test_key_stable_under_line_drift():
    base = _src("except subprocess.SubprocessError:")
    drifted = "\n\n\n" + base  # blank lines prepended
    assert g._scan_source(base, "m.py") == g._scan_source(drifted, "m.py")


def test_ordinal_distinguishes_calls_in_same_scope():
    # Two unguarded calls in one function → ordinals 0 and 1.
    src = (
        "import subprocess\n"
        "def f():\n"
        "    try:\n"
        "        subprocess.run(['rg'])\n"
        "    except subprocess.SubprocessError:\n"
        "        pass\n"
        "    try:\n"
        "        subprocess.run(['git'])\n"
        "    except subprocess.SubprocessError:\n"
        "        pass\n"
    )
    assert g._scan_source(src, "m.py") == [
        "m.py::<module>::f::0",
        "m.py::<module>::f::1",
    ]


# --- net-new detection via main() exit code -----------------------------------
def test_main_returns_nonzero_on_net_new(tmp_path, monkeypatch):
    prod = tmp_path / "ext"
    prod.mkdir()
    (prod / "mod.py").write_text(
        "import subprocess\n"
        "def a():\n    subprocess.run(['rg'])\n\n"
        "def b():\n    subprocess.run(['rg'])\n"
    )
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    monkeypatch.setattr(g, "BASELINE", tmp_path / "b.txt")
    # Only call 'a' is baselined; 'b' is NET-NEW.
    (tmp_path / "b.txt").write_text("ext/mod.py::<module>::a::0\n")
    monkeypatch.setattr(sys, "argv", ["check"])  # main() parses file args from argv
    assert g.main() == 1


def test_main_returns_zero_when_in_sync(tmp_path, monkeypatch):
    prod = tmp_path / "ext"
    prod.mkdir()
    (prod / "mod.py").write_text(
        "import subprocess\n"
        "def a():\n    subprocess.run(['rg'])\n"
    )
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    monkeypatch.setattr(g, "BASELINE", tmp_path / "b.txt")
    (tmp_path / "b.txt").write_text("ext/mod.py::<module>::a::0\n")
    monkeypatch.setattr(sys, "argv", ["check"])  # main() parses file args from argv
    assert g.main() == 0


# --- scan scope -------------------------------------------------------------
def test_scope_covers_every_shipped_root_module():
    """All root-level *.py must be scanned, not just asi.py.

    They ship in the wheel exactly like asi.py, so scanning one by name left the
    other seven (common.py, config.py, path_security.py, ...) free to
    introduce an unguarded call unblocked.
    """
    scanned = {p.resolve() for p in g._iter_repo_py()}
    root_mods = {p.resolve() for p in g.REPO.glob("*.py") if p.is_file()}
    assert root_mods, "no root-level modules found — test would be vacuous"
    missing = root_mods - scanned
    assert not missing, f"root modules outside scan scope: {sorted(p.name for p in missing)}"


def test_scope_has_no_duplicate_paths():
    """asi.py must not be scanned twice once root globbing covers it.

    A duplicate would double-count ordinals for any file listed twice, silently
    corrupting the drift-stable key.
    """
    paths = g._iter_repo_py()
    dupes = [p for p in set(paths) if paths.count(p) > 1]
    assert not dupes, f"duplicated in scan set: {dupes}"


def test_root_module_unguarded_call_is_flagged(tmp_path, monkeypatch):
    """A net-new unguarded call in a ROOT module fails the gate."""
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ())
    monkeypatch.setattr(g, "BASELINE", tmp_path / "b.txt")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "root_like.py").write_text(
        "import subprocess\n"
        "def probe():\n"
        "    try:\n"
        "        subprocess.run(['git'])\n"
        "    except subprocess.SubprocessError:\n"
        "        pass\n"
    )
    monkeypatch.setattr(sys, "argv", ["check"])  # main() parses file args from argv
    assert g.main() == 1


# --- live repo invariant: current ⊆ baseline (drift guard) -------------------
# If a future commit adds an unguarded subprocess call without re-baselining,
# this fails — the intended regression signal. NOTE: vacuously-passable if the
# scanner is weakened; the precision tests above guard detection capability.
def test_no_new_unguarded_subprocess_beyond_baseline():
    current = g._get_current_keys()
    baseline = g._load_baseline()
    new = current - baseline
    assert not new, (
        f"{len(new)} new unguarded subprocess call(s) beyond baseline:\n"
        + "\n".join(sorted(new))
    )


# --- per-file mode: explicit paths scope the scan (parallel-write race fix) ---
def test_iter_repo_py_filters_paths_to_scan_scope(tmp_path, monkeypatch):
    prod = tmp_path / "ext"
    prod.mkdir()
    (prod / "mod.py").write_text("x = 1\n")
    (tmp_path / "asi.py").write_text("x = 1\n")
    (tmp_path / "rootmod.py").write_text("x = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("x = 1\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "s.py").write_text("x = 1\n")
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    paths = g._resolve_scan_paths([
        str(prod / "mod.py"),
        str(tmp_path / "asi.py"),
        str(tmp_path / "rootmod.py"),
        str(tmp_path / "tests" / "t.py"),
        str(tmp_path / "scripts" / "s.py"),
    ])
    scanned = g._iter_repo_py(paths)
    assert scanned == [
        prod / "mod.py",
        tmp_path / "asi.py",
        tmp_path / "rootmod.py",
    ], scanned


def test_main_with_explicit_paths_scopes_scan(tmp_path, monkeypatch):
    root = tmp_path / "rootmod.py"
    root.write_text("import subprocess\nsubprocess.run(['x'])\n")
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    monkeypatch.setattr(g, "BASELINE", tmp_path / "b.txt")
    (tmp_path / "b.txt").write_text("")
    # unguarded call in a root-level module → in scope → FAIL (net-new)
    monkeypatch.setattr(sys, "argv", ["check", str(root)])
    assert g.main() == 1
    # the same call under tests/ → out of scope → PASS
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("import subprocess\nsubprocess.run(['x'])\n")
    monkeypatch.setattr(sys, "argv", ["check", str(tmp_path / "tests" / "t.py")])
    assert g.main() == 0
