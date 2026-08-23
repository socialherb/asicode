"""RED→GREEN coverage completion for dependency_checker (46% → 100%).

Covers every branch that the state-persistence tests (test_dependency_checker_state)
and the REPL status-line contract test (test_dep_status_line_align) do not
touch: version detection, TTY detection, the yes/no prompt, the npm/pip
installers (incl. the PEP 668 retry), repo-language detection, tool
resolution, and the interactive install prompt.
"""

from __future__ import annotations

import builtins
import json
import logging
import subprocess
import sys

import pytest

from external_llm.languages import dependency_checker as dc
from external_llm.languages.models import LanguageId


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeStream:
    """Stand-in for sys.stdin/sys.stdout with controllable isatty()."""

    def __init__(self, tty=True, exc=None):
        self._tty = tty
        self._exc = exc

    def isatty(self):
        if self._exc is not None:
            raise self._exc
        return self._tty


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    f = tmp_path / "tool_state.json"
    monkeypatch.setattr(dc, "_STATE_PATH", str(f))
    return f


@pytest.fixture
def npm_on_path(monkeypatch):
    monkeypatch.setattr(dc.shutil, "which", lambda name: "/usr/bin/npm" if name == "npm" else None)


# ── _detect_version ──────────────────────────────────────────────────────────


def test_detect_version_stdout(monkeypatch):
    calls = []

    def _run(cmd, **kw):
        calls.append(cmd)
        return _FakeProc(0, stdout="pyright 1.2.3\n")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._detect_version("pyright") == "pyright 1.2.3"
    assert calls == [["pyright", "--version"]]


def test_detect_version_go_uses_version_subcommand(monkeypatch):
    calls = []

    def _run(cmd, **kw):
        calls.append(cmd)
        return _FakeProc(0, stdout="go version go1.22.5 darwin/arm64\n")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._detect_version("go") == "go version go1.22.5 darwin/arm64"
    assert calls == [["go", "version"]]


def test_detect_version_stderr_fallback(monkeypatch):
    # Empty (or whitespace-only) stdout falls back to stderr.
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **kw: _FakeProc(0, stdout="   ", stderr="v2.0"))
    assert dc._detect_version("tsc") == "v2.0"


def test_detect_version_nonzero_returncode_empty(monkeypatch):
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **kw: _FakeProc(1, stdout="x"))
    assert dc._detect_version("tsc") == ""


def test_detect_version_suppresses_oserror(monkeypatch):
    def _run(*a, **kw):
        raise FileNotFoundError("tsc")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._detect_version("tsc") == ""


def test_detect_version_suppresses_timeout(monkeypatch):
    def _run(*a, **kw):
        raise subprocess.TimeoutExpired(["tsc"], 5)

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._detect_version("tsc") == ""


# ── _is_interactive ──────────────────────────────────────────────────────────


def test_is_interactive_true(monkeypatch):
    monkeypatch.setattr(dc.sys, "stdin", _FakeStream(True))
    monkeypatch.setattr(dc.sys, "stdout", _FakeStream(True))
    assert dc._is_interactive() is True


def test_is_interactive_false_when_stdout_pipe(monkeypatch):
    monkeypatch.setattr(dc.sys, "stdin", _FakeStream(True))
    monkeypatch.setattr(dc.sys, "stdout", _FakeStream(False))
    assert dc._is_interactive() is False


def test_is_interactive_false_on_oserror(monkeypatch):
    monkeypatch.setattr(dc.sys, "stdin", _FakeStream(exc=OSError("not a tty")))
    monkeypatch.setattr(dc.sys, "stdout", _FakeStream(True))
    assert dc._is_interactive() is False


def test_is_interactive_false_on_attribute_error(monkeypatch):
    # Objects without isatty() (e.g. some embedded streams) → False.
    monkeypatch.setattr(dc.sys, "stdin", object())
    monkeypatch.setattr(dc.sys, "stdout", object())
    assert dc._is_interactive() is False


# ── _ask_yes_no ──────────────────────────────────────────────────────────────


def test_ask_yes_no_yes_forms(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt: "y")
    assert dc._ask_yes_no("Q") is True
    monkeypatch.setattr(builtins, "input", lambda prompt: "YES")
    assert dc._ask_yes_no("Q") is True


def test_ask_yes_no_no_forms(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt: "n")
    assert dc._ask_yes_no("Q") is False
    monkeypatch.setattr(builtins, "input", lambda prompt: "No")
    assert dc._ask_yes_no("Q") is False


def test_ask_yes_no_empty_uses_default_true(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt: "")
    assert dc._ask_yes_no("Q") is True


def test_ask_yes_no_empty_uses_default_false(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt: "")
    assert dc._ask_yes_no("Q", default=False) is False


def test_ask_yes_no_unknown_uses_default(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt: "maybe")
    assert dc._ask_yes_no("Q") is True
    assert dc._ask_yes_no("Q", default=False) is False


def test_ask_yes_no_eof_returns_default_and_newline(monkeypatch, capsys):
    def _raise(prompt):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _raise)
    assert dc._ask_yes_no("Q") is True
    assert capsys.readouterr().out == "\n"


def test_ask_yes_no_keyboard_interrupt_returns_default(monkeypatch, capsys):
    def _raise(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", _raise)
    assert dc._ask_yes_no("Q", default=False) is False
    assert capsys.readouterr().out == "\n"


def test_ask_yes_no_suffix_matches_default(monkeypatch):
    seen = []
    monkeypatch.setattr(builtins, "input", lambda prompt: seen.append(prompt) or "")
    dc._ask_yes_no("Q", default=True)
    dc._ask_yes_no("Q", default=False)
    assert seen == ["Q [Y/n] ", "Q [y/N] "]


# ── _npm_install ─────────────────────────────────────────────────────────────


def test_npm_install_success(monkeypatch, capsys):
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **kw: _FakeProc(0))
    assert dc._npm_install("pyright") is True
    assert "Installed" in capsys.readouterr().out


def test_npm_install_failure_prints_stderr_tail(monkeypatch, capsys):
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **kw: _FakeProc(1, stderr="e1\ne2\ne3\ne4"))
    assert dc._npm_install("pyright") is False
    out = capsys.readouterr().out
    assert "e2" in out and "e3" in out and "e4" in out
    assert "e1" not in out  # only the last 3 lines are shown
    assert "npm install failed (exit 1)" in out


def test_npm_install_failure_empty_stderr(monkeypatch, capsys):
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **kw: _FakeProc(2))
    assert dc._npm_install("pyright") is False
    assert "npm install failed (exit 2)" in capsys.readouterr().out


def test_npm_install_file_not_found(monkeypatch, capsys):
    def _run(*a, **kw):
        raise FileNotFoundError("npm")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._npm_install("pyright") is False
    assert "npm not found" in capsys.readouterr().out


def test_npm_install_timeout(monkeypatch, capsys):
    def _run(*a, **kw):
        raise subprocess.TimeoutExpired(["npm"], 120)

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._npm_install("pyright") is False
    assert "timed out" in capsys.readouterr().out


def test_npm_install_oserror(monkeypatch, capsys):
    def _run(*a, **kw):
        raise OSError("boom")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._npm_install("pyright") is False
    assert "install failed: boom" in capsys.readouterr().out


# ── _pip_install ─────────────────────────────────────────────────────────────


def test_pip_install_success(monkeypatch, capsys):
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **kw: _FakeProc(0))
    assert dc._pip_install("ruff") is True
    assert "Installed" in capsys.readouterr().out


def test_pip_install_failure_no_pep668(monkeypatch, capsys):
    monkeypatch.setattr(
        dc.subprocess,
        "run",
        lambda *a, **kw: _FakeProc(1, stderr="pip err", stdout="pip out"),
    )
    assert dc._pip_install("ruff") is False
    out = capsys.readouterr().out
    assert "pip err" in out and "pip out" in out
    assert "install failed (exit 1)" in out


def test_pip_install_pep668_retry_success(monkeypatch, capsys):
    calls = []

    def _run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            return _FakeProc(1, stderr="error: externally-managed-environment")
        return _FakeProc(0)

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._pip_install("ruff") is True
    assert calls[0] == [sys.executable, "-m", "pip", "install", "ruff"]
    assert calls[1][-2:] == ["--break-system-packages", "ruff"]
    assert "Installed" in capsys.readouterr().out


def test_pip_install_pep668_retry_failure(monkeypatch, capsys):
    def _run(cmd, **kw):
        return _FakeProc(1, stderr="error: externally-managed-environment")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._pip_install("ruff") is False
    out = capsys.readouterr().out
    assert "retrying with --break-system-packages" in out
    assert "install failed (exit 1)" in out


def test_pip_install_retry_oserror(monkeypatch, capsys):
    calls = []

    def _run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            return _FakeProc(1, stderr="error: externally-managed-environment")
        raise OSError("retry boom")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._pip_install("ruff") is False
    assert "retry failed: retry boom" in capsys.readouterr().out


def test_pip_install_timeout(monkeypatch, capsys):
    def _run(*a, **kw):
        raise subprocess.TimeoutExpired([sys.executable], 120)

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._pip_install("ruff") is False
    assert "timed out" in capsys.readouterr().out


def test_pip_install_oserror(monkeypatch, capsys):
    def _run(*a, **kw):
        raise OSError("pip boom")

    monkeypatch.setattr(dc.subprocess, "run", _run)
    assert dc._pip_install("ruff") is False
    assert "install failed: pip boom" in capsys.readouterr().out


# ── detect_repo_languages ────────────────────────────────────────────────────


def test_detect_repo_languages_mixed(monkeypatch):
    files = ["a.py", "b.ts", "c.js", "d.go", "e.java", "f.kt", "g.c", "h.cpp", "README.md"]
    monkeypatch.setattr(dc, "git_list_repo_files", lambda root: files)
    assert dc.detect_repo_languages("/repo") == {
        LanguageId.PYTHON,
        LanguageId.TYPESCRIPT,
        LanguageId.JAVASCRIPT,
        LanguageId.GO,
        LanguageId.JAVA,
        LanguageId.KOTLIN,
        LanguageId.C,
        LanguageId.CPP,
    }


def test_detect_repo_languages_non_ascii_names(monkeypatch):
    # Regression guard: git C-quotes non-ASCII paths without -z, which broke
    # suffix detection for every Korean/CJK-named file.
    monkeypatch.setattr(dc, "git_list_repo_files", lambda root: ["한글파일.py", "data.json"])
    assert dc.detect_repo_languages("/repo") == {LanguageId.PYTHON, LanguageId.JSON}


def test_detect_repo_languages_none_from_git_failure(monkeypatch):
    monkeypatch.setattr(dc, "git_list_repo_files", lambda root: None)
    assert dc.detect_repo_languages("/repo") == set()


def test_detect_repo_languages_empty_repo(monkeypatch):
    monkeypatch.setattr(dc, "git_list_repo_files", lambda root: [])
    assert dc.detect_repo_languages("/repo") == set()


def test_detect_repo_languages_unknown_only(monkeypatch):
    monkeypatch.setattr(dc, "git_list_repo_files", lambda root: ["README.md", "LICENSE"])
    assert dc.detect_repo_languages("/repo") == set()


# ── _clone_tools ─────────────────────────────────────────────────────────────


def test_clone_tools_deduplicates_by_cmd():
    # TS and JS maps both carry tsc + eslint → dedupe to one of each.
    fresh = dc._clone_tools({LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT})
    assert {t.cmd for t in fresh} == {"tsc", "eslint"}
    assert len(fresh) == 2


def test_clone_tools_returns_fresh_instances():
    fresh = dc._clone_tools({LanguageId.PYTHON})
    tmpl = dc._LANGUAGE_TOOL_MAP[LanguageId.PYTHON][0]
    assert fresh[0] is not tmpl
    assert fresh[0].cmd == tmpl.cmd
    fresh[0].found = True
    assert tmpl.found is False  # template untouched


def test_clone_tools_unknown_language_empty():
    assert dc._clone_tools({LanguageId.UNKNOWN}) == []


# ── _resolve_tool ────────────────────────────────────────────────────────────


def test_resolve_tool_bare_command(monkeypatch):
    monkeypatch.setattr(dc.shutil, "which", lambda name: "/usr/bin/pyright" if name == "pyright" else None)
    assert dc._resolve_tool(dc._Tool(cmd="pyright", label="L")) is True


def test_resolve_tool_npx_fallback(monkeypatch):
    # tsc/eslint are invoked via `npx <cmd>` — npx on $PATH suffices.
    monkeypatch.setattr(dc.shutil, "which", lambda name: "/usr/bin/npx" if name == "npx" else None)
    assert dc._resolve_tool(dc._Tool(cmd="tsc", label="L", use_npx=True)) is True


def test_resolve_tool_unavailable(monkeypatch):
    monkeypatch.setattr(dc.shutil, "which", lambda name: None)
    assert dc._resolve_tool(dc._Tool(cmd="go", label="L")) is False
    assert dc._resolve_tool(dc._Tool(cmd="tsc", label="L", use_npx=True)) is False


# ── _load_tool_state / _save_tool_state ──────────────────────────────────────


def test_load_state_non_dict(state_file):
    state_file.write_text("[1, 2]", encoding="utf-8")
    assert dc._load_tool_state() == {}


def test_load_state_non_dict_sections(state_file):
    state_file.write_text(json.dumps({"pretend": "x", "skipped": {"go": True}}), encoding="utf-8")
    assert dc._load_tool_state() == {"go": "skip"}


def test_load_state_falsy_values_filtered(state_file):
    state_file.write_text(json.dumps({"pretend": {"pyright": False}, "skipped": {"tsc": 0}}), encoding="utf-8")
    assert dc._load_tool_state() == {}


def test_save_state_oserror_swallowed(tmp_path, monkeypatch, caplog):
    # Parent dir is a regular file → makedirs fails → swallowed at debug level.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(dc, "_STATE_PATH", str(blocker / "sub" / "tool_state.json"))
    with caplog.at_level(logging.DEBUG, logger="external_llm.languages.dependency_checker"):
        dc._save_tool_state({"go": "skip"})  # must not raise
    assert "failed to persist" in caplog.text


def test_save_state_replace_into_directory_cleans_tmp(tmp_path, monkeypatch, caplog):
    # mkstemp succeeds but os.replace fails (target is a directory) → the temp
    # file must be unlinked by the finally block and the error swallowed.
    target = tmp_path / "tool_state.json"
    target.mkdir()
    monkeypatch.setattr(dc, "_STATE_PATH", str(target))
    with caplog.at_level(logging.DEBUG, logger="external_llm.languages.dependency_checker"):
        dc._save_tool_state({"go": "skip"})
    assert "failed to persist" in caplog.text
    assert list(tmp_path.glob(".tool_state_*.json")) == []  # tmp cleaned up


# ── _check_tools_with_state ──────────────────────────────────────────────────


def test_check_tools_with_state_no_languages(monkeypatch):
    monkeypatch.setattr(dc, "_is_interactive", lambda: True)
    assert dc._check_tools_with_state(set()) == []
    assert dc._check_tools_with_state({LanguageId.UNKNOWN}) == []


# ── check_and_install_all ────────────────────────────────────────────────────


def test_check_and_install_all_all_found(monkeypatch):
    monkeypatch.setattr(dc, "_resolve_tool", lambda t: True)
    monkeypatch.setattr(dc, "_detect_version", lambda cmd: "1.0")
    monkeypatch.setattr(dc, "_is_interactive", lambda: False)
    result = dc.check_and_install_all()
    assert result == {t.cmd: True for t in dc._TOOLS}


def test_check_and_install_all_missing_noninteractive(monkeypatch):
    monkeypatch.setattr(dc, "_resolve_tool", lambda t: False)
    monkeypatch.setattr(dc, "_is_interactive", lambda: False)
    result = dc.check_and_install_all()
    assert result == {t.cmd: False for t in dc._TOOLS}


def test_check_and_install_all_no_prompt_interactive(monkeypatch):
    monkeypatch.setattr(dc, "_resolve_tool", lambda t: False)
    monkeypatch.setattr(dc, "_is_interactive", lambda: True)
    result = dc.check_and_install_all(no_prompt=True)
    assert result == {t.cmd: False for t in dc._TOOLS}


def test_check_and_install_all_interactive_full_flow(monkeypatch, capsys):
    # Real _prompt_and_install: pyright/tsc via npm, go/javac/kotlinc via
    # "Mark as done (pretend installed)".
    monkeypatch.setattr(dc, "_resolve_tool", lambda t: False)
    monkeypatch.setattr(dc, "_is_interactive", lambda: True)
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: True)
    monkeypatch.setattr(dc, "_npm_install", lambda pkg: True)
    monkeypatch.setattr(dc, "_pip_install", lambda pkg: True)
    monkeypatch.setattr(dc.shutil, "which", lambda name: "/usr/bin/npm" if name == "npm" else None)
    result = dc.check_and_install_all()
    assert result == {t.cmd: True for t in dc._TOOLS}
    assert "Semantic validation tools" in capsys.readouterr().out


def test_check_and_install_all_exclude_go(monkeypatch):
    monkeypatch.setattr(dc, "_resolve_tool", lambda t: False)
    monkeypatch.setattr(dc, "_is_interactive", lambda: False)
    result = dc.check_and_install_all(include_go=False)
    assert "go" not in result
    assert set(result) == {"pyright", "tsc", "javac", "kotlinc"}
    assert all(v is False for v in result.values())


# ── _print_status_block ──────────────────────────────────────────────────────


def test_status_block_found_with_version(capsys):
    dc._print_status_block([dc._Tool(cmd="pyright", label="L", found=True, version="1.2.3")])
    out = capsys.readouterr().out
    assert "pyright" in out and "found (1.2.3)" in out


def test_status_block_found_without_version(capsys):
    dc._print_status_block([dc._Tool(cmd="tsc", label="L", found=True)])
    out = capsys.readouterr().out
    assert "found" in out and "(1.2.3)" not in out


def test_status_block_skipped(capsys):
    dc._print_status_block([dc._Tool(cmd="go", label="L", skipped=True)])
    assert "skipped by user" in capsys.readouterr().out


def test_status_block_not_found(capsys):
    dc._print_status_block([dc._Tool(cmd="go", label="L")])
    assert "not found" in capsys.readouterr().out


# ── _prompt_and_install ──────────────────────────────────────────────────────


def test_prompt_and_install_npm_success(monkeypatch, npm_on_path):
    t = dc._clone_tools({LanguageId.TYPESCRIPT})[0]  # tsc (npm only)
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: True)
    installed = []
    monkeypatch.setattr(dc, "_npm_install", lambda pkg: installed.append(pkg) or True)
    monkeypatch.setattr(dc, "_pip_install", lambda pkg: pytest.fail("pip must not run when npm succeeded"))
    dc._prompt_and_install(t)
    assert t.found is True
    assert installed == ["typescript"]


def test_prompt_and_install_npm_unavailable_pip_fallback(monkeypatch, capsys):
    monkeypatch.setattr(dc.shutil, "which", lambda name: None)
    t = dc._clone_tools({LanguageId.PYTHON})[0]  # pyright: npm + pip
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: True)
    monkeypatch.setattr(dc, "_npm_install", lambda pkg: pytest.fail("npm unavailable — must not run"))
    installed = []
    monkeypatch.setattr(dc, "_pip_install", lambda pkg: installed.append(pkg) or True)
    dc._prompt_and_install(t)
    assert t.found is True
    assert installed == ["pyright"]
    assert "(npm not available on $PATH)" in capsys.readouterr().out


def test_prompt_and_install_user_declines(monkeypatch, capsys, npm_on_path):
    t = dc._clone_tools({LanguageId.PYTHON})[0]
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: False)
    monkeypatch.setattr(dc, "_npm_install", lambda pkg: pytest.fail("must not install when declined"))
    dc._prompt_and_install(t)
    assert t.skipped is True
    assert "Skipped (semantic validation disabled for this language)" in capsys.readouterr().out


def test_prompt_and_install_all_methods_fail(monkeypatch, capsys, npm_on_path):
    t = dc._clone_tools({LanguageId.PYTHON})[0]
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: True)
    monkeypatch.setattr(dc, "_npm_install", lambda pkg: False)
    monkeypatch.setattr(dc, "_pip_install", lambda pkg: False)
    dc._prompt_and_install(t)
    assert t.skipped is True
    assert "All install methods failed" in capsys.readouterr().out


def test_prompt_and_install_manual_pretend(monkeypatch, capsys):
    t = dc._clone_tools({LanguageId.GO})[0]  # manual-hint-only tool
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: True)
    dc._prompt_and_install(t)
    assert t.found is True
    assert t.pretend_installed is True
    out = capsys.readouterr().out
    assert "Auto-install not available." in out
    assert "brew install go" in out  # manual hint rendered


def test_prompt_and_install_manual_skip(monkeypatch, capsys):
    t = dc._clone_tools({LanguageId.GO})[0]
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: False)
    dc._prompt_and_install(t)
    assert t.skipped is True
    assert t.found is False


def test_prompt_and_install_no_methods_no_hint():
    t = dc._Tool(cmd="x", label="X")  # no npm/pip/manual → cannot install
    dc._prompt_and_install(t)
    assert t.skipped is True


def test_prompt_and_install_npm_fails_pip_succeeds(monkeypatch, capsys, npm_on_path):
    t = dc._clone_tools({LanguageId.PYTHON})[0]
    monkeypatch.setattr(dc, "_ask_yes_no", lambda prompt, default=True: True)
    monkeypatch.setattr(dc, "_npm_install", lambda pkg: False)
    monkeypatch.setattr(dc, "_pip_install", lambda pkg: True)
    dc._prompt_and_install(t)
    assert t.found is True


# ── main ─────────────────────────────────────────────────────────────────────


def test_main_all_found_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr(dc, "_resolve_tool", lambda t: True)
    monkeypatch.setattr(dc, "_detect_version", lambda cmd: "")
    monkeypatch.setattr(dc, "_is_interactive", lambda: False)
    with pytest.raises(SystemExit) as ei:
        dc.main()
    assert ei.value.code == 0
    assert "Summary:" in capsys.readouterr().out


def test_main_missing_exit_one(monkeypatch, capsys):
    monkeypatch.setattr(dc, "_resolve_tool", lambda t: False)
    monkeypatch.setattr(dc, "_is_interactive", lambda: False)
    with pytest.raises(SystemExit) as ei:
        dc.main()
    assert ei.value.code == 1
