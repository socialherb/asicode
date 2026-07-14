"""Tests for persisted dismissal state in dependency_checker.

Covers the regression where "Mark as done (pretend installed)" / "skip"
decisions were ephemeral and the prompt recurred on every launch.
"""
from __future__ import annotations

import json

import pytest

from external_llm.languages import dependency_checker as dc
from external_llm.languages.models import LanguageId

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect the global state file to a temp path; start empty."""
    state_file = tmp_path / "tool_state.json"
    monkeypatch.setattr(dc, "_STATE_PATH", str(state_file))
    return state_file


@pytest.fixture
def stub_resolution(monkeypatch):
    """Control tool availability without touching the real $PATH.

    Returns a dict the test mutates: {cmd: bool}.
    """
    avail: dict[str, bool] = {}

    def _fake_resolve(t):
        return avail.get(t.cmd, False)

    monkeypatch.setattr(dc, "_resolve_tool", _fake_resolve)
    monkeypatch.setattr(dc, "_detect_version", lambda cmd: "")
    return avail


@pytest.fixture
def non_interactive(monkeypatch):
    monkeypatch.setattr(dc, "_is_interactive", lambda: False)


# ── pure load/save round-trip ────────────────────────────────────────────────


def test_state_path_is_global_user_config():
    # Machine-global (per-user), not per-repo — tool availability is a host fact.
    assert dc._STATE_PATH.endswith(".asicode/tool_state.json")
    assert "~" not in dc._STATE_PATH  # expanded


def test_load_returns_empty_when_absent(isolated_state):
    assert dc._load_tool_state() == {}


def test_load_returns_empty_on_corrupt(isolated_state):
    isolated_state.write_text("{ not valid json", encoding="utf-8")
    assert dc._load_tool_state() == {}


def test_save_load_roundtrip(isolated_state):
    decisions = {"kotlinc": "pretend", "go": "skip"}
    dc._save_tool_state(decisions)
    assert dc._load_tool_state() == decisions


def test_save_is_valid_json_with_version(isolated_state):
    dc._save_tool_state({"kotlinc": "pretend"})
    data = json.loads(isolated_state.read_text(encoding="utf-8"))
    assert data["version"] == dc._STATE_VERSION
    assert data["pretend"] == {"kotlinc": True}
    assert data["skipped"] == {}


# ── _check_tools_with_state integration ──────────────────────────────────────


def test_persisted_pretend_suppresses_prompt(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    # kotlinc not on PATH, but user previously chose "pretend installed".
    stub_resolution["kotlinc"] = False
    dc._save_tool_state({"kotlinc": "pretend"})

    prompted = []
    monkeypatch.setattr(dc, "_prompt_and_install", lambda t: prompted.append(t.cmd))

    tools = dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)

    t = next(x for x in tools if x.cmd == "kotlinc")
    assert t.found is True                 # pretend → reported as found
    assert t.pretend_installed is True
    assert prompted == []                  # NOT re-prompted


def test_persisted_skip_suppresses_prompt(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    stub_resolution["go"] = False
    dc._save_tool_state({"go": "skip"})

    prompted = []
    monkeypatch.setattr(dc, "_prompt_and_install", lambda t: prompted.append(t.cmd))

    tools = dc._check_tools_with_state({LanguageId.GO}, no_prompt=False)
    t = next(x for x in tools if x.cmd == "go")
    assert t.skipped is True
    assert t.found is False
    assert prompted == []


def test_genuine_availability_beats_persisted_dismissal(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    # Previously dismissed, but now actually installed → found truthfully.
    stub_resolution["kotlinc"] = True
    dc._save_tool_state({"kotlinc": "pretend"})

    monkeypatch.setattr(dc, "_prompt_and_install", lambda t: None)
    tools = dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)
    t = next(x for x in tools if x.cmd == "kotlinc")
    assert t.found is True
    assert t.pretend_installed is False    # genuinely found, not pretend


def test_stale_dismissal_cleared_when_tool_becomes_available(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    stub_resolution["kotlinc"] = True
    dc._save_tool_state({"kotlinc": "pretend"})

    monkeypatch.setattr(dc, "_prompt_and_install", lambda t: None)
    dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)

    # State file rewritten with kotlinc removed (no longer dismissed).
    assert dc._load_tool_state() == {}


def test_new_pretend_decision_persisted_after_prompt(
    isolated_state, stub_resolution, monkeypatch
):
    stub_resolution["kotlinc"] = False

    def _simulate_pretend(t):
        t.found = True
        t.pretend_installed = True

    monkeypatch.setattr(dc, "_prompt_and_install", _simulate_pretend)
    monkeypatch.setattr(dc, "_is_interactive", lambda: True)

    dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)
    assert dc._load_tool_state() == {"kotlinc": "pretend"}


def test_new_skip_decision_persisted_after_prompt(
    isolated_state, stub_resolution, monkeypatch
):
    stub_resolution["go"] = False

    def _simulate_skip(t):
        t.skipped = True

    monkeypatch.setattr(dc, "_prompt_and_install", _simulate_skip)
    monkeypatch.setattr(dc, "_is_interactive", lambda: True)

    dc._check_tools_with_state({LanguageId.GO}, no_prompt=False)
    assert dc._load_tool_state() == {"go": "skip"}


def test_successful_install_not_recorded_as_dismissal(
    isolated_state, stub_resolution, monkeypatch
):
    # User picks "Install now" and it succeeds → NOT a dismissal.
    stub_resolution["pyright"] = False    # missing initially

    def _simulate_install(t):
        t.found = True                    # installed successfully (not pretend)

    monkeypatch.setattr(dc, "_prompt_and_install", _simulate_install)
    monkeypatch.setattr(dc, "_is_interactive", lambda: True)

    dc._check_tools_with_state({LanguageId.PYTHON}, no_prompt=False)
    assert dc._load_tool_state() == {}    # nothing dismissed


def test_no_prompt_does_not_touch_state(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    # In no_prompt/non-interactive mode the persisted state is applied but
    # never rewritten (no decisions can be made non-interactively).
    stub_resolution["kotlinc"] = False
    # start with empty state
    monkeypatch.setattr(dc, "_prompt_and_install", lambda t: None)
    dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=True)
    assert dc._load_tool_state() == {}
    assert not isolated_state.exists()


def test_pretend_installed_flag_reset_per_call(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    # Fresh instances must not leak pretend_installed from a previous run.
    stub_resolution["kotlinc"] = True     # genuinely available now
    dc._save_tool_state({"kotlinc": "pretend"})

    tools = dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)
    t1 = next(x for x in tools if x.cmd == "kotlinc")
    assert t1.pretend_installed is False

    # Second call with same availability — still genuinely found, not pretend.
    tools2 = dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)
    t2 = next(x for x in tools2 if x.cmd == "kotlinc")
    assert t2.pretend_installed is False
    assert t2.found is True


# ── cross-repo dismissal preservation (the machine-global-state regression) ──


def test_dismissal_preserved_when_language_absent_from_repo(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    # Regression: the state file is machine-global (shared across repos), yet
    # _sync_tool_state used to rewrite it from only the *current* repo's tools.
    # Launching asi from a repo whose language set does NOT include Kotlin
    # therefore erased a previously-saved ``kotlinc`` dismissal, so the "Mark
    # as done" prompt recurred next time a Kotlin project was opened.
    #
    # Setup: user already dismissed kotlinc in a Kotlin repo.
    dc._save_tool_state({"kotlinc": "pretend"})

    # Now open a *Python-only* repo — kotlinc is not in its tool set at all.
    stub_resolution["pyright"] = True      # genuinely available → not dismissed
    monkeypatch.setattr(dc, "_prompt_and_install", lambda t: None)

    dc._check_tools_with_state({LanguageId.PYTHON}, no_prompt=False)

    # The Kotlin dismissal must survive: the global state spans all repos, and
    # kotlinc was simply not examined this run.
    assert dc._load_tool_state() == {"kotlinc": "pretend"}


def test_dismissals_merged_across_disjoint_language_repos(
    isolated_state, stub_resolution, monkeypatch
):
    # Two repos with disjoint language sets; dismissals accumulate, not clobber.
    monkeypatch.setattr(dc, "_is_interactive", lambda: True)

    def _simulate_dismiss(t):
        # Simulate the user choosing skip (or pretend) at the prompt.
        t.skipped = True

    monkeypatch.setattr(dc, "_prompt_and_install", _simulate_dismiss)

    # Repo A (Kotlin): kotlinc not found → user dismisses it (pretend).
    stub_resolution["kotlinc"] = False

    def _simulate_pretend(t):
        t.found = True
        t.pretend_installed = True

    monkeypatch.setattr(dc, "_prompt_and_install", _simulate_pretend)
    dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)
    assert dc._load_tool_state() == {"kotlinc": "pretend"}

    # Repo B (Go): go not found → user skips it. kotlinc must be retained.
    stub_resolution["go"] = False
    monkeypatch.setattr(dc, "_prompt_and_install", _simulate_dismiss)
    dc._check_tools_with_state({LanguageId.GO}, no_prompt=False)

    state = dc._load_tool_state()
    assert state == {"kotlinc": "pretend", "go": "skip"}


def test_genuine_install_in_one_repo_clears_only_that_dismissal(
    isolated_state, stub_resolution, non_interactive, monkeypatch
):
    # Install-clears-stale must remain scoped to the examined tool: a genuinely
    # found kotlinc clears its dismissal but must not touch an unrelated go skip.
    monkeypatch.setattr(dc, "_prompt_and_install", lambda t: None)
    dc._save_tool_state({"kotlinc": "pretend", "go": "skip"})

    # Kotlin repo; kotlinc now genuinely on PATH.
    stub_resolution["kotlinc"] = True
    dc._check_tools_with_state({LanguageId.KOTLIN}, no_prompt=False)

    assert dc._load_tool_state() == {"go": "skip"}
