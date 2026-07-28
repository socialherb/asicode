"""Regression: find_symbol / read_symbol must NOT crash when rg is absent.

ripgrep is an OPTIONAL dependency (pyproject ``[search]`` extra, NOT core deps).
A base ``pip install asicode`` ships no rg, and pyproject promises a graceful
fallback. Several call sites in ``symbol_search.py`` run
``subprocess.run(["rg", ...])``; when rg is absent that raises
``FileNotFoundError``, which EVERY site must catch — otherwise the DEFAULT
``find_symbol(kind="any")`` crashes on every lookup for an absent / non-Python /
mistyped name.

Why DETERMINISTIC MONKEYPATCH (not a subprocess with PATH scrubbed)
-------------------------------------------------------------------
The prior version ran a subprocess with rg stripped from PATH. That is
fundamentally incompatible with CI runners, where rg is installed at
``/usr/bin`` ALONGSIDE git and the interpreter binary: stripping ``/usr/bin``
breaks the subprocess, and rg is reachable from PATH directories a single
``shutil.which()`` does not enumerate (symlinked ``/bin`` -> ``/usr/bin``,
multiple installs). The self-check ``assert which("rg") is None`` fired on the
runner, the test could not run there at all, and it had to be ``--deselect``ed
in the release gate. A regression guard that cannot run in CI is half a guard.

This version simulates rg-absence DETERMINISTICALLY by patching the two symbols
the call sites resolve THROUGH — ``shutil.which`` (-> ``None`` for ``"rg"``) and
``subprocess.run`` (-> ``FileNotFoundError`` for any ``["rg", ...]`` command) —
at the ``symbol_search`` module level. That fires the exact
``FileNotFoundError`` every site must catch, on ANY host, whether or not rg is
installed. The guarded sites (``rg = shutil.which("rg"); if not rg:``) skip the
call; the hardcoded sites (``subprocess.run(["rg", ...])``) raise and must be
caught by their ``except (... OSError)`` clauses.

Verified to produce identical return types to the subprocess approach:
``_nonpy_index_for`` -> ``dict``, ``_outline_ripgrep`` -> ``list``,
``find_symbol(absent)`` -> ``[]``, a real Python symbol still resolves.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import external_llm.agent.symbol_search as ss
from external_llm.agent.symbol_search import SymbolSearcher

REPO = Path(__file__).resolve().parents[3]


def _simulate_rg_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``symbol_search`` behave EXACTLY as if rg were not installed.

    Patched at the module level (``ss.shutil`` / ``ss.subprocess``) so every
    call site — guarded (``rg = shutil.which("rg")``) or hardcoded
    (``subprocess.run(["rg", ...])``) — sees the absent-rg contract. ``which``
    delegates for non-rg names; ``run`` delegates for non-rg commands, so only
    the rg paths are affected.
    """
    real_which = shutil.which
    real_run = subprocess.run

    def fake_which(name, *args, **kwargs):
        return None if name == "rg" else real_which(name, *args, **kwargs)

    def fake_run(cmd, *args, **kwargs):
        head = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else cmd
        if isinstance(head, str) and (head == "rg" or head.endswith("/rg")):
            raise FileNotFoundError(2, "No such file or directory: 'rg'")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(ss.shutil, "which", fake_which)
    monkeypatch.setattr(ss.subprocess, "run", fake_run)


@pytest.fixture
def searcher_no_rg(monkeypatch: pytest.MonkeyPatch) -> SymbolSearcher:
    _simulate_rg_absent(monkeypatch)
    return SymbolSearcher(str(REPO))


def test_find_symbol_no_crash_when_rg_absent(searcher_no_rg: SymbolSearcher) -> None:
    s = searcher_no_rg
    # absent name, default kind="any": the branch that used to crash (non-Python
    # index lookup via _nonpy_index_for -> guarded rg sites skip gracefully).
    assert s.find_symbol("ZzzNoSuchSymbol_xyz_9988776655", kind="any") == [], "absent name crashed"
    # absent name, kind="function": also reached the non-Python branch via
    # `not results` before the fix.
    assert s.find_symbol("ZzzNoSuchSymbol_xyz_9988776655", kind="function") == [], "absent name crashed"
    # a REAL Python symbol must still be found (Python AST path is rg-independent).
    hit = s.find_symbol("SymbolSearcher", kind="class")
    assert hit and hit[0].name == "SymbolSearcher", hit
    # outline of a non-Python file -> _outline_ripgrep (hardcoded ["rg"]): must
    # not crash, returns a list. pyproject.toml exists in every checkout (incl.
    # the public export), so this exercises the real file path, not just the
    # missing-file early-return.
    out = s.get_file_outline("pyproject.toml")
    assert isinstance(out, list), type(out)


def test_rg_call_sites_degrade_gracefully(searcher_no_rg: SymbolSearcher) -> None:
    """Pin the contract directly at the function level with rg simulated absent.

    Every rg call site must return an empty/neutral result instead of letting
    ``FileNotFoundError`` propagate. ``_outline_ripgrep`` is the hardcoded-rg
    site (no ``shutil.which`` guard) — it MUST catch the exception. The guarded
    sites inside ``_nonpy_index_for`` skip the call and degrade to an empty dict.
    """
    s = searcher_no_rg
    # _outline_ripgrep: hardcoded ["rg", ...], no shutil.which guard -> must catch.
    out1 = s._outline_ripgrep(REPO / "pyproject.toml", "pyproject.toml")
    assert isinstance(out1, list), type(out1)
    # _nonpy_index_for: guarded (shutil.which); degrades to an empty dict.
    out2 = s._nonpy_index_for(REPO)
    assert isinstance(out2, dict), type(out2)
    # _find_subclasses: the SECOND hardcoded site (no shutil.which guard), and
    # the one neither this file nor its predecessor covered — verified by
    # mutation: dropping OSError from its `except` left both versions green.
    # It is live shipping code, not an internal helper:
    #   find_symbol(include_inheritance=True) -> get_symbol_info -> here
    # (read_tools._tool_find_symbol passes the flag straight through), so an
    # unguarded raise here crashes a user-reachable tool call.
    out3 = s._find_subclasses("SymbolSearcher")
    assert isinstance(out3, list), type(out3)
