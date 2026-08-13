"""P2-3: the non-tree-sitter fallback symbol index must spawn ONE rg per
provider (merged --glob / -e) instead of one per (glob, pattern) pair —
94 spawns (~1.1s) collapsed to ~13 (~0.13s) — with caps scaled to the merged
pattern count so no single pattern is starved of its per-pattern budget.
"""
from __future__ import annotations

import subprocess

import pytest

import external_llm.agent.symbol_search as ss


@pytest.fixture
def captured(tmp_path, monkeypatch):
    """Force the regex-fallback path (_HAS_TS=False) and capture rg cmds."""
    calls: list[list[str]] = []
    monkeypatch.setattr(ss, "_HAS_TS", False)

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        out = ""
        if "--glob" in cmd and "*.sh" in cmd:
            # Only the bash provider's merged call yields matches.
            out = (
                "run.sh:10:myfunc() {\n"
                "run.sh:20:function helper {\n"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(ss.subprocess, "run", _fake_run)
    searcher = ss.SymbolSearcher(str(tmp_path))
    index = searcher._nonpy_index_for(tmp_path)
    return calls, index


def _bash_call(calls):
    bash = [c for c in calls if "--glob" in c and "*.sh" in c]
    assert bash, "no bash-provider rg call captured"
    return bash


def test_one_spawn_per_provider_merged_args(captured):
    """Regression: the bash provider (4 globs x 2 patterns = 8 old spawns)
    must be indexed with exactly ONE rg call carrying all globs and all
    patterns."""
    calls, _ = captured
    bash = _bash_call(calls)
    assert len(bash) == 1

    cmd = bash[0]
    n_pats = cmd.count("-e")
    n_globs = cmd.count("--glob")
    assert n_pats == 2  # POSIX + function-keyword forms
    assert n_globs == 4 + 2  # provider globs + !node_modules* / !*.py
    # Per-file cap scales with the merged pattern count (rg -m is a per-file
    # total across the merged alternation).
    assert cmd[cmd.index("-m") + 1] == str(5 * n_pats)
    assert cmd[0] == "rg"
    assert cmd[-1].endswith("run.sh") is False  # last arg is search_root


def test_total_spawns_well_below_glob_pattern_combos(captured):
    """Aggregate sanity: the merged scheme must produce far fewer spawns than
    the old per-(glob, pattern) loop would (13 providers x several combos)."""
    calls, _ = captured
    assert 0 < len(calls) <= 20
    # Every call carries at least one pattern (no patternless spawns).
    assert all(c.count("-e") >= 1 for c in calls)


def test_merged_output_classified_by_first_matching_pattern(captured):
    """Lines from the merged output are classified by the FIRST pattern whose
    regex matches, using that pattern's name_capture and kind."""
    _, index = captured
    myfunc = index.get("myfunc", [])
    assert myfunc and myfunc[0].kind == "function"
    assert myfunc[0].file == "run.sh"
    assert myfunc[0].line == 10
    helper = index.get("helper", [])
    assert helper and helper[0].line == 20
    assert helper[0].signature == "function helper {"


def test_no_cap_starvation_single_pattern_still_gets_budget(captured):
    """The per-file -m must be 5x pattern-count so a file matching many
    patterns cannot exhaust the merged budget on the first pattern."""
    calls, _ = captured
    for cmd in calls:
        n_pats = cmd.count("-e")
        assert int(cmd[cmd.index("-m") + 1]) == 5 * n_pats
