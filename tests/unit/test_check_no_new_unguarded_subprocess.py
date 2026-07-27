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
    assert g.main() == 0


# --- scan scope -------------------------------------------------------------
def test_scope_covers_every_shipped_root_module():
    """All root-level *.py must be scanned, not just asi.py.

    They ship in the wheel exactly like asi.py, so scanning one by name left the
    other eight (common.py, config.py, path_security.py, radio.py, ...) free to
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
    (tmp_path / "radio_like.py").write_text(
        "import subprocess\n"
        "def probe():\n"
        "    try:\n"
        "        subprocess.run(['git'])\n"
        "    except subprocess.SubprocessError:\n"
        "        pass\n"
    )
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
