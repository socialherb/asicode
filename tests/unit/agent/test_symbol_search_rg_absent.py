"""Regression: find_symbol / read_symbol must NOT crash when rg is absent.

ripgrep is an OPTIONAL dependency (pyproject ``[search]`` extra, NOT core deps).
A base ``pip install asicode`` has no rg, and pyproject explicitly promises a
graceful fallback. Three rg call sites in ``symbol_search.py`` called
``subprocess.run(["rg", ...])`` WITHOUT catching ``OSError`` (FileNotFoundError),
so the DEFAULT ``find_symbol(kind="any")`` — which reaches the non-Python index
branch whenever the Python/TS scan finds nothing — crashed on every lookup for
an absent / non-Python / mistyped name.

Why these tests run in a SUBPROCESS: the prior tests monkeypatched
``ss.shutil.which`` to simulate rg-absence, but the three crashing sites call
``subprocess.run`` directly — bypassing ``shutil.which`` — so the monkeypatch
never exercised them and they ran against the real rg. Only an actual PATH
without rg reproduces the failure. The subprocess isolates the PATH change so
the rest of the suite is unaffected.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
REAL_RG = __import__("shutil").which("rg")
pytestmark = pytest.mark.skipif(not REAL_RG, reason="ripgrep not installed")

# A Python snippet run with rg stripped from PATH. It exercises every crash site:
#   * find_symbol(kind="any") for an ABSENT name -> non-Python index branch
#     (_index_via_treesitter_batch + _nonpy_index_for regex loop)
#   * find_symbol for a REAL Python name -> must still be found
#   * read_symbol over a non-Python file via get_file_outline -> _outline_ripgrep
_PROBE = textwrap.dedent(
    """
    import os, sys
    import shutil as _sh
    assert _sh.which("rg") is None, (
        "PATH cleanup failed — rg still reachable, this regression test is invalid"
    )
    sys.path.insert(0, %r)
    from external_llm.agent.symbol_search import SymbolSearcher
    s = SymbolSearcher(".")
    # absent name, default kind="any": the branch that used to crash
    r1 = s.find_symbol("ZzzNoSuchSymbol_xyz_9988776655", kind="any")
    assert r1 == [], ("expected empty, got", r1)
    # absent name, kind="function": also reached the non-Python branch via
    # `not results` before the fix.
    r2 = s.find_symbol("ZzzNoSuchSymbol_xyz_9988776655", kind="function")
    assert r2 == [], r2
    # a REAL Python symbol must still be found (Python path is rg-independent)
    r3 = s.find_symbol("SymbolSearcher", kind="class")
    assert r3 and r3[0].name == "SymbolSearcher", r3
    # outline of a non-Python file -> _outline_ripgrep: must not crash, may be empty
    r4 = s.get_file_outline("webapp/ui/static/ui.js")
    assert isinstance(r4, list), type(r4)
    print("OK")
    """ % str(REPO)
)


def _run_without_rg(snippet: str) -> subprocess.CompletedProcess:
    """Run *snippet* in a subprocess whose PATH excludes rg.

    rg is frequently a symlink (Homebrew ``/opt/homebrew/bin/rg`` -> a
    ``Cellar/.../bin/rg`` target), so stripping only the resolved target dir
    leaves the raw ``/opt/homebrew/bin`` PATH entry intact and rg stays
    reachable — silently neutering the test (this is exactly why the prior
    bug slipped through). Strip BOTH the raw dir from ``which`` and its
    resolved target dir, and the probe below self-asserts rg is truly gone.
    """
    bad = {str(Path(REAL_RG).parent)}
    try:
        bad.add(str(Path(REAL_RG).resolve().parent))
    except OSError:
        pass
    clean_path = os.pathsep.join(
        p for p in os.environ["PATH"].split(os.pathsep) if p and p not in bad
    )
    env = dict(os.environ, PATH=clean_path)
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )


def test_find_symbol_no_crash_when_rg_absent():
    proc = _run_without_rg(_PROBE)
    assert proc.returncode == 0, (
        "find_symbol/read_symbol crashed with rg absent (P0 regression):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_three_unguarded_sites_all_degrade_gracefully():
    """Pin the contract directly at the function level with rg removed from PATH.

    Each of the three previously-crashing sites must return an empty/neutral
    result instead of raising FileNotFoundError.
    """
    snippet = textwrap.dedent(
        """
        import os, sys
        import shutil as _sh
        assert _sh.which("rg") is None, (
            "PATH cleanup failed — rg still reachable, this regression test is invalid"
        )
        sys.path.insert(0, %r)
        from pathlib import Path
        from external_llm.agent.symbol_search import SymbolSearcher
        s = SymbolSearcher(".")
        root = Path(".")
        # site 1: _outline_ripgrep (reachable via _outline_ts_js / _find_in_ts_js)
        out1 = s._outline_ripgrep(root / "webapp/ui/static/ui.js", "webapp/ui/static/ui.js")
        assert isinstance(out1, list), type(out1)
        # site 2 + 3: _nonpy_index_for (regex loop + _index_via_treesitter_batch)
        out2 = s._nonpy_index_for(root)
        assert isinstance(out2, dict), type(out2)
        print("OK")
        """ % str(REPO)
    )
    proc = _run_without_rg(snippet)
    assert proc.returncode == 0, (
        "an rg call site raised instead of degrading:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout
