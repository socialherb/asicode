"""RED→GREEN coverage for asi.py — pure helpers, rendering, logging, git-mocked CLI utils.

Covers the miss regions from the 33% baseline (1735 stmts / 1165 miss):

  shimmer/color helpers, _MarginIO, bracketed paste, history rotation,
  diff renderers (_parse_diff_stats/_build_file_diff_renderable/_render_run_diff),
  git baseline/checkpoint/undo helpers (via _git mock), session summary,
  model resolution (_model_candidates/_resolve_model_arg/_resolve_model_interactive),
  auth retry + key commit, insights archive, clipboard, think suggestions,
  completer, help/status/banner renderers, dep-status printers, pip install,
  embedding cache helpers, dotenv, interrupt note, update notice.

Source-free: asi.py is not modified here.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

import asi

# ─── color / shimmer helpers ───────────────────────────────────────────────────


class TestLerpColor:
    def test_t0_returns_first(self):
        assert asi._lerp_color("#000000", "#ffffff", 0.0) == "#000000"

    def test_t1_returns_second(self):
        assert asi._lerp_color("#000000", "#ffffff", 1.0) == "#ffffff"

    def test_midpoint_rounds(self):
        assert asi._lerp_color("#000000", "#ffffff", 0.5) == "#808080"

    def test_handles_hashless(self):
        assert asi._lerp_color("102030", "405060", 1.0) == "#405060"


class TestShimmerBeam:
    def test_short_text_center_zero_width(self):
        assert asi._shimmer_beam(3, 0.0) == (1, 0)

    def test_long_text_returns_radius(self):
        center, beam_w = asi._shimmer_beam(30, 0.0)
        assert beam_w == max(3, 30 // 3)
        assert abs(center) <= beam_w  # triangle wave may start negative

    def test_phase_sweep_bounds(self):
        for t in (0.0, 0.5, 1.0, 2.0):
            center, beam_w = asi._shimmer_beam(30, t)
            assert -beam_w <= center <= 30 + beam_w


class TestShimmerStyle:
    def test_outside_beam_is_base(self):
        assert asi._shimmer_style_for(0, 50, 3) == asi._SHIMMER_BASE

    def test_center_is_brightest(self):
        s = asi._shimmer_style_for(5, 5, 4)
        assert s != asi._SHIMMER_BASE

    def test_edge_falls_back_to_base(self):
        assert asi._shimmer_style_for(5, 5, 1) != asi._SHIMMER_BASE


class TestRenderShimmerText:
    def test_blank_text(self):
        out = asi._render_shimmer_text("", 0.0)
        assert out.plain == ""

    def test_short_text_no_beam(self):
        out = asi._render_shimmer_text("ab", 0.0)
        assert out.plain == "ab"

    def test_spaces_preserved(self):
        out = asi._render_shimmer_text("a b c", 1.0)
        assert out.plain == "a b c"

    def test_long_text_has_styles(self):
        out = asi._render_shimmer_text("asicode", 0.3)
        assert out.plain == "asicode"
        assert len(out.spans) > 0


# ─── seq pad / MarginIO ────────────────────────────────────────────────────────


class TestSeqPad:
    def test_fills_to_fixed_width(self):
        assert len(asi._seq_pad("[1]")) == max(0, (asi._SEQ_W + 2) - len("[1]"))

    def test_wide_token_no_pad(self):
        assert asi._seq_pad("[1000]") == ""


class TestMarginIO:
    def test_writes_margin_at_bol(self):
        buf = io.StringIO()
        orig = sys.stdout
        try:
            sys.stdout = buf
            m = asi._MarginIO("stdout", margin=4)
            m.write("hello\n")
            assert buf.getvalue() == "    hello\n"
        finally:
            sys.stdout = orig

    def test_no_margin_on_continuation(self):
        buf = io.StringIO()
        orig = sys.stdout
        try:
            sys.stdout = buf
            m = asi._MarginIO("stdout", margin=4)
            m.write("line1\nline2\n")
            assert buf.getvalue() == "    line1\n    line2\n"
        finally:
            sys.stdout = orig

    def test_empty_write(self):
        buf = io.StringIO()
        orig = sys.stdout
        try:
            sys.stdout = buf
            m = asi._MarginIO("stdout", margin=4)
            assert m.write("") == 0
        finally:
            sys.stdout = orig

    def test_reset_bol(self):
        buf = io.StringIO()
        orig = sys.stdout
        try:
            sys.stdout = buf
            m = asi._MarginIO("stdout", margin=4)
            m.write("a")  # no newline → _bol False
            m.reset_bol()
            m.write("b")
            assert buf.getvalue() == "    a    b"
        finally:
            sys.stdout = orig

    def test_delegates(self):
        orig = sys.stdout
        try:
            sys.stdout = io.StringIO()
            m = asi._MarginIO("stdout")
            assert m.isatty() is False
            assert m.encoding in (None, "utf-8")
            assert m.errors in (None, "strict")
            m.flush()
        finally:
            sys.stdout = orig


# ─── bracketed paste / drain stdin ────────────────────────────────────────────


class TestBracketedPaste:
    def test_enable_writes_escape(self, monkeypatch, capsys):
        monkeypatch.setattr(asi.sys.stdout, "isatty", lambda: True)
        asi._enable_bracketed_paste()
        assert capsys.readouterr().out == "\x1b[?2004h"

    def test_disable_writes_escape(self, monkeypatch, capsys):
        monkeypatch.setattr(asi.sys.stdout, "isatty", lambda: True)
        asi._disable_bracketed_paste()
        assert capsys.readouterr().out == "\x1b[?2004l"

    def test_enable_noop_when_not_tty(self, monkeypatch, capsys):
        monkeypatch.setattr(asi.sys.stdout, "isatty", lambda: False)
        asi._enable_bracketed_paste()
        assert capsys.readouterr().out == ""


class TestDrainStdin:
    def test_non_tty_noop(self, monkeypatch):
        called = []
        monkeypatch.setattr(asi.sys.stdin, "fileno", lambda: 0)
        monkeypatch.setattr(os, "isatty", lambda fd: False)
        asi._drain_stdin()
        assert called == []


# ─── history rotation ──────────────────────────────────────────────────────────


class TestRotateCliHistory:
    def test_missing_file_noop(self, tmp_path):
        asi._rotate_cli_history_if_needed(str(tmp_path / "nope"))
        assert True

    def test_small_file_noop(self, tmp_path):
        p = tmp_path / "hist"
        p.write_text("# 1\n+a\n")
        asi._rotate_cli_history_if_needed(str(p))
        assert p.read_text() == "# 1\n+a\n"

    def test_rotates_keeping_tail(self, tmp_path, monkeypatch):
        p = tmp_path / "hist"
        lines = [f"# {i}\n+entry{i}\n" for i in range(30)]
        p.write_text("".join(lines))
        monkeypatch.setattr(asi, "_CLI_HISTORY_ROTATE_AT", 10)
        monkeypatch.setattr(asi, "_CLI_HISTORY_KEEP", 5)
        asi._rotate_cli_history_if_needed(str(p))
        out = p.read_text().splitlines()
        assert out[0].startswith("# ")
        assert len(out) <= 6  # 5 kept + boundary snap

    def test_rotation_failure_swallowed(self, tmp_path, monkeypatch):
        p = tmp_path / "hist"
        p.write_text("x" * 100)
        monkeypatch.setattr(asi, "_CLI_HISTORY_ROTATE_AT", 5)
        monkeypatch.setattr(asi, "_CLI_HISTORY_KEEP", 5)
        with contextlib.suppress(OSError):
            p.chmod(0o000)
        asi._rotate_cli_history_if_needed(str(p))  # must not raise


# ─── strip ansi / diff stats / elapsed / tokens ───────────────────────────────


class TestStripAnsi:
    def test_strips_sgr(self):
        assert asi._strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strips_cursor_moves(self):
        assert asi._strip_ansi("\x1b[2K\x1b[1A") == ""

    def test_keeps_plain(self):
        assert asi._strip_ansi("plain text") == "plain text"

    def test_malformed_escape_kept(self):
        assert asi._strip_ansi("a\x1b[zzb") == "a\x1b[zzb"


class TestParseDiffStats:
    def test_counts_adds_and_rems(self):
        body = "+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
        assert asi._parse_diff_stats(body) == (1, 1)

    def test_excludes_headers(self):
        body = "--- a/x\n+++ b/x\n"
        assert asi._parse_diff_stats(body) == (0, 0)


class TestFmtElapsed:
    def test_sub_minute(self):
        assert asi._fmt_elapsed(8.24) == "8.2s"

    def test_minutes(self):
        assert asi._fmt_elapsed(72) == "1m 12s"

    def test_hours(self):
        assert asi._fmt_elapsed(3725) == "1h 02m"


class TestAbbrevTokens:
    def test_small(self):
        assert asi._abbrev_tokens(690) == "690"

    def test_k(self):
        assert asi._abbrev_tokens(43606) == "43.6K"

    def test_m(self):
        assert asi._abbrev_tokens(11377708) == "11.38M"


# ─── session summary ───────────────────────────────────────────────────────────


class TestPrintSessionSummary:
    def test_silent_when_no_usage(self, monkeypatch):
        calls = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: calls.append(a))
        asi._print_session_summary({}, time.monotonic())
        assert calls == []

    def test_prints_with_usage(self, monkeypatch):
        calls = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: calls.append(a))
        asi._print_session_summary({"prompt": 1234, "completion": 56}, time.monotonic() - 10)
        assert calls and "session" in calls[0][0]
        assert "1.2K" in calls[0][0]


# ─── model resolution ──────────────────────────────────────────────────────────


class TestModelCandidates:
    def test_prefix_scan(self, monkeypatch):
        monkeypatch.setattr(asi, "_get_ollama_models", lambda timeout=3: ["qwen3:7b"])
        cands = asi._model_candidates("qwen", ollama_timeout=1)
        assert any(m.lower().startswith("qwen") for _, m in cands)

    def test_known_provider_scan(self):
        cands = asi._model_candidates("gpt")
        assert any(p == "openai" for p, _ in cands)


class TestResolveModelArg:
    def test_empty(self):
        assert asi._resolve_model_arg("") is None

    def test_provider_slash_model(self):
        assert asi._resolve_model_arg("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_slash_missing_model(self):
        assert asi._resolve_model_arg("openai/") is None

    def test_space_separated_known_provider(self):
        assert asi._resolve_model_arg("openai gpt-4o") == ("openai", "gpt-4o")

    def test_single_candidate(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p, ollama_timeout=3: [("anthropic", "claude-sonnet-4-6")])
        assert asi._resolve_model_arg("claude") == ("anthropic", "claude-sonnet-4-6")

    def test_exact_name_wins(self, monkeypatch):
        monkeypatch.setattr(
            asi,
            "_model_candidates",
            lambda p, ollama_timeout=3: [("anthropic", "claude-sonnet-4-6"), ("ollama", "claude-x")],
        )
        assert asi._resolve_model_arg("claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")

    def test_ambiguous_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            asi, "_model_candidates", lambda p, ollama_timeout=3: [("anthropic", "m1"), ("openai", "m2")]
        )
        assert asi._resolve_model_arg("m") is None


class TestResolveModelInteractive:
    def test_empty_arg(self, monkeypatch):
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "")
        assert asi._resolve_model_interactive("") is None

    def test_provider_slash(self):
        assert asi._resolve_model_interactive("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_slash_empty_model(self, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._resolve_model_interactive("openai/") is None
        assert any("model name required" in str(p) for p in printed)

    def test_slash_provider_with_spaces_rejected(self, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._resolve_model_interactive("openai x/gpt-4o") is None
        assert any("must not contain spaces" in str(p) for p in printed)

    def test_space_separated_known(self):
        assert asi._resolve_model_interactive("openai gpt-4o") == ("openai", "gpt-4o")

    def test_space_separated_multiword_model_rejected(self, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._resolve_model_interactive("openai gpt-4o bug") is None
        assert any("must not contain spaces" in str(p) for p in printed)

    def test_single_prefix_candidate(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [("openai", "gpt-4o")])
        assert asi._resolve_model_interactive("gpt") == ("openai", "gpt-4o")

    def test_multi_candidate_selection(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [("a", "m1"), ("b", "m2")])
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "2")
        assert asi._resolve_model_interactive("m") == ("b", "m2")

    def test_multi_candidate_cancel(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [("a", "m1"), ("b", "m2")])
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "")
        assert asi._resolve_model_interactive("m") is None

    def test_multi_candidate_eof(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [("a", "m1"), ("b", "m2")])

        def _eof(p=""):
            raise EOFError

        monkeypatch.setattr(asi, "_collect_input", _eof)
        assert asi._resolve_model_interactive("m") is None

    def test_unknown_model(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [])
        monkeypatch.setattr(asi, "_get_ollama_models", lambda timeout=3: [])
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._resolve_model_interactive("zzz") is None
        assert any("unknown model" in str(p) for p in printed)

    def test_unknown_model_silent(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [])
        monkeypatch.setattr(asi, "_get_ollama_models", lambda timeout=3: [])
        assert asi._resolve_model_interactive("zzz", warn_unknown=False) is None

    def test_alias_conversion(self, monkeypatch):
        monkeypatch.setattr(asi, "_MODEL_ALIASES", {"old-name": "new-name"})
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._resolve_model_interactive("openai/old-name") == ("openai", "new-name")
        assert any("auto-corrected" in str(p) for p in printed)


# ─── think suggestions ─────────────────────────────────────────────────────────


class TestGetThinkSuggestions:
    def test_openai_o_series(self):
        assert asi._get_think_suggestions("openai", "o3-mini") == ["low", "medium", "high"]

    def test_openai_generic(self):
        s = asi._get_think_suggestions("openai", "gpt-4o")
        assert "on" in s and "none" in s

    def test_anthropic_always_thinking(self):
        assert asi._get_think_suggestions("anthropic", "claude-opus-4-8") == ["low", "medium", "high"]

    def test_anthropic_generic(self):
        s = asi._get_think_suggestions("anthropic", "claude-sonnet-4-6")
        assert "off" in s

    def test_deepseek(self):
        assert asi._get_think_suggestions("deepseek", "deepseek-r1") == ["on", "off", "high", "max"]

    def test_google_25(self):
        assert asi._get_think_suggestions("google", "gemini-2.5-pro") == ["on", "off"]

    def test_google_3(self):
        s = asi._get_think_suggestions("google", "gemini-3-pro")
        assert "minimal" in s

    def test_zai(self):
        s = asi._get_think_suggestions("zai", "glm-5.2")
        assert "xhigh" in s

    def test_ollama(self):
        assert asi._get_think_suggestions("ollama", "qwen3") == ["on", "off"]

    def test_unknown_provider(self):
        s = asi._get_think_suggestions("", "")
        assert "max" in s and "minimal" in s


# ─── group / help / status / banner ────────────────────────────────────────────


class TestGroupedSlashCommands:
    def test_sections_and_other(self):
        groups = asi._grouped_slash_commands()
        titles = [t for t, _ in groups]
        assert "session" in titles
        all_cmds = [c for _, cmds in groups for c in cmds]
        assert len(all_cmds) == len(asi._SLASH_COMMANDS)  # nothing dropped


class TestRenderHelp:
    def test_plain_fallback(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        asi._render_help()
        out = capsys.readouterr().out
        assert "/help" in out

    def test_rich_branch(self, monkeypatch):
        printed = []
        fake = type("FakeConsole", (), {"print": staticmethod(lambda *a, **k: printed.append(a))})()
        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", fake)
        asi._render_help()
        assert printed


class TestRenderStatus:
    def test_plain_branch(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        asi._render_status(
            "/repo",
            "openai",
            "gpt-4o",
            "code",
            {"prompt": 100},
            thinking_state=True,
            reasoning_effort="high",
            helper="h",
        )
        out = capsys.readouterr().out
        assert "openai / gpt-4o" in out
        assert "thinking ON (high)" in out
        assert "helper" in out

    def test_rich_branch(self, monkeypatch):
        printed = []
        fake = type("FakeConsole", (), {"print": staticmethod(lambda *a, **k: printed.append(a))})()
        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", fake)
        asi._render_status("/repo", "", "model-x", "general", {"prompt": 0}, thinking_state=False)
        assert printed

    def test_auto_think(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        asi._render_status("/repo", "", "m", "code", {})
        out = capsys.readouterr().out
        assert "thinking (auto)" in out


class TestBarBoxPanel:
    def test_bar_box_cached(self, monkeypatch):
        monkeypatch.setattr(asi, "_BAR_BOX", None)
        box = asi._bar_box()
        assert box is not None
        assert asi._BAR_BOX is box

    def test_bar_panel(self, monkeypatch):
        panels = []

        class FakePanel:
            def __init__(self, *a, **k):
                panels.append((a, k))

        monkeypatch.setattr(asi, "_BAR_BOX", object())
        import rich.panel as rp

        monkeypatch.setattr(rp, "Panel", FakePanel)
        asi._bar_panel("content", title="t", color="red")
        assert panels and panels[0][1]["title"] == "t"


class TestPrint:
    def test_plain_branch(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        asi._print("hello", "red", end="\n")
        assert capsys.readouterr().out == "hello\n"

    def test_rich_branch(self, monkeypatch):
        printed = []

        class FakeOut:
            file = None

            def print(self, *a, **k):
                printed.append((a, k))

        fake = FakeOut()
        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", fake)
        asi._print("hi", "green")
        assert printed


class TestPrintBanner:
    def test_plain_branch(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        asi._print_banner("/repo")
        out = capsys.readouterr().out
        assert "asicode" in out and "/repo" in out

    def test_no_color_rich(self, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", True)
        printed = []

        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)

            def rule(self, *a, **k):
                pass

        monkeypatch.setattr(asi, "_out_console", FakeOut())
        monkeypatch.setenv("NO_COLOR", "1")
        asi._print_banner("/repo")
        assert printed


# ─── dep status ────────────────────────────────────────────────────────────────


class TestCheckDepStatus:
    def test_tool_states(self, monkeypatch):
        class T:
            def __init__(self, cmd, found, skipped):
                self.cmd, self.found, self.skipped = cmd, found, skipped

        tools = [T("eslint", True, False), T("black", False, False), T("mypy", False, True)]
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: False)
        status = asi._check_dep_status(tools)
        assert status["eslint"] == "ON"
        assert status["black"] == "OFF"
        assert status["mypy"] == "skip"
        assert "tree-sitter" in status
        assert status["vector"] in ("OFF", "no-model", "ON")


class TestGitLsFiles:
    def test_delegates(self, monkeypatch):
        from external_llm.common.repo_files import git_list_repo_files as real

        monkeypatch.setattr(asi, "_git_ls_files", lambda r: real(r) or [])
        files = asi._git_ls_files("/tmp")
        assert isinstance(files, list)


class TestDetectRepoTsLanguages:
    def test_detects(self):
        langs = asi._detect_repo_ts_languages(["a.py", "b.ts", "c.xyz"])
        assert "python" in langs
        assert "typescript" in langs


class TestPrintDepStatus:
    def _patch_dep_modules(self, monkeypatch, files=(), langs=()):
        import external_llm.languages.dependency_checker as dc
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(dc, "detect_repo_languages", lambda r: [])
        monkeypatch.setattr(dc, "_check_tools_with_state", lambda d, no_prompt=False: [])
        monkeypatch.setattr(asi, "_git_ls_files", lambda r: list(files))
        monkeypatch.setattr(asi, "_detect_repo_ts_languages", lambda f: set(langs))
        monkeypatch.setattr(tsu, "is_available", lambda: True)
        monkeypatch.setattr(tsu, "_get_language", lambda lang: None)
        monkeypatch.setattr(asi, "_maybe_prompt_vector_install", lambda: None)
        monkeypatch.setattr(asi, "_DEPS_RESTART_PENDING", False)

    def test_plain_branch_no_deps(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        self._patch_dep_modules(monkeypatch)
        asi._print_dep_status("/tmp", no_deps_check=True)
        out = capsys.readouterr().out
        assert "tree-sitter" in out

    def test_missing_grammar_warning_y(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        self._patch_dep_modules(monkeypatch, files=["a.py"], langs={"python"})
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "y")
        monkeypatch.setattr(asi, "_install_tree_sitter_grammars", lambda pkgs: None)
        asi._print_dep_status("/tmp", no_deps_check=True)
        out = capsys.readouterr().out
        assert "grammar" in out

    def test_missing_grammar_warning_n(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        self._patch_dep_modules(monkeypatch, files=["a.py"], langs={"python"})
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "n")
        asi._print_dep_status("/tmp", no_deps_check=True)
        out = capsys.readouterr().out
        assert "Skipped" in out


# ─── git baseline / changed files / undo (via _git mock) ──────────────────────


class TestGitBaseline:
    def test_not_a_repo(self, monkeypatch):
        monkeypatch.setattr(asi, "_git", lambda r, *a, **k: (128, ""))
        assert asi._git_baseline("/x") is None

    def test_stash_create_ref(self, monkeypatch):
        calls = []

        def fake_git(r, *args, **kw):
            calls.append(args)
            if args[0] == "rev-parse":
                return 0, "true"
            if args[0] == "stash":
                return 0, "abc123\n"
            if args[0] == "update-ref":
                return 0, ""
            if args[0] == "ls-files":
                return 0, "new1.py\x00new2.py\x00"
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        bl = asi._git_baseline("/x")
        assert bl == {"ref": "abc123", "untracked": frozenset({"new1.py", "new2.py"})}

    def test_head_fallback(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "rev-parse" and args[1] == "--is-inside-work-tree":
                return 0, "true"
            if args[0] == "stash":
                return 0, ""
            if args[0] == "rev-parse":
                return 0, "headref\n"
            if args[0] == "ls-files":
                return 0, ""
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        assert asi._git_baseline("/x")["ref"] == "headref"

    def test_no_ref(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "rev-parse" and args[1] == "--is-inside-work-tree":
                return 0, "true"
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        assert asi._git_baseline("/x") is None


class TestChangedFilesSince:
    def test_tracked_and_new(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "diff":
                return 0, "a.py\x00b.py\x00"
            if args[0] == "ls-files":
                return 0, "c.py\x00d.py\x00"
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        baseline = {"ref": "r", "untracked": frozenset({"d.py"})}
        assert asi._changed_files_since("/x", baseline) == ["a.py", "b.py", "c.py"]


class TestFileDiffText:
    def test_tracked_diff(self, monkeypatch):
        monkeypatch.setattr(
            asi, "_git", lambda r, *a, **k: (0, "+++ b/x\n+1\n") if a[0] == "diff" and a[1] == "--no-color" else (0, "")
        )
        body, is_new = asi._file_diff_text("/x", {"ref": "r"}, "x")
        assert is_new is False and body

    def test_new_file_diff(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if "--no-index" in args:
                return 0, "+new\n"
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        body, is_new = asi._file_diff_text("/x", {"ref": "r"}, "new.py")
        assert is_new is True and body


class TestBuildFileDiffRenderable:
    def _body(self):
        return "diff --git a/x b/x\nindex 111..222 100644\n--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-old\n+new\n context\n"

    def test_render(self):
        out = asi._build_file_diff_renderable("x", self._body(), False)
        assert "x" in out.plain and "+1" in out.plain and "−1" in out.plain  # noqa: RUF001

    def test_new_file_tag(self):
        out = asi._build_file_diff_renderable("y", "+a\n", True)
        assert "new" in out.plain

    def test_truncation(self):
        body = "".join(f"+line{i}\n" for i in range(80))
        out = asi._build_file_diff_renderable("z", body, False, max_lines=10)
        assert "more lines" in out.plain

    def test_hunk_gap(self):
        body = "@@ -1 +1 @@\n+a\n@@ -5 +5 @@\n+b\n"
        out = asi._build_file_diff_renderable("w", body, False)
        assert "⋯" in out.plain

    def test_no_newline_marker_skipped(self):
        body = "@@ -1 +1 @@\n+a\n\\ No newline at end of file\n"
        out = asi._build_file_diff_renderable("v", body, False)
        assert "No newline" not in out.plain


class TestRunChangedStats:
    def test_numstat_and_new_fallback(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "diff" and "--numstat" in args:
                return 0, "3\t1\ta.py\x00"
            if "--no-index" in args:
                return 0, "+new\n"
            if args[0] == "diff" and args[1] == "--no-color":
                return 0, ""  # tracked-file diff comes back empty → new-file path
            if args[0] == "diff":
                return 0, "a.py\x00b.py\x00"
            if args[0] == "ls-files":
                return 0, ""
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        baseline = {"ref": "r", "untracked": frozenset()}
        stats = asi._run_changed_stats("/x", baseline)
        assert stats[0] == ("a.py", 3, 1, False)
        assert stats[1][1] >= 1  # b.py fallback parsed (added > 0)

    def test_no_baseline(self):
        assert asi._run_changed_stats("/x", None) == []


class TestPrintRunChangeSummary:
    def test_nothing_changed(self, monkeypatch):
        monkeypatch.setattr(asi, "_run_changed_stats", lambda r, b: [])
        assert asi._print_run_change_summary("/x", {"ref": "r"}) is False

    def test_plain_print(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        monkeypatch.setattr(asi, "_run_changed_stats", lambda r, b: [("a.py", 2, 1, False)])
        assert asi._print_run_change_summary("/x", {"ref": "r"}) is True
        out = capsys.readouterr().out
        assert "M a.py" in out and "+2 -1" in out


class TestUndoRunChanges:
    def test_restore_and_delete(self, monkeypatch, tmp_path):
        (tmp_path / "new.py").write_text("x")

        def fake_git(r, *args, **kw):
            if args[0] == "cat-file":
                return (0, "") if args[2].endswith(":a.py") else (1, "")
            if args[0] == "restore":
                return 0, ""
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        baseline = {"ref": "r", "untracked": frozenset()}
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py", "new.py"])
        undone, failed = asi._undo_run_changes(str(tmp_path), baseline)
        assert undone == ["a.py", "new.py"]
        assert failed == []
        assert not (tmp_path / "new.py").exists()

    def test_restore_fallback_checkout(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "cat-file":
                return 0, ""
            if args[0] == "restore":
                return 1, ""
            if args[0] == "checkout":
                return 0, ""
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py"])
        undone, _failed = asi._undo_run_changes("/x", {"ref": "r"})
        assert undone == ["a.py"]

    def test_restore_failure(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "cat-file":
                return 0, ""
            return 1, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py"])
        _undone, failed = asi._undo_run_changes("/x", {"ref": "r"})
        assert failed == ["a.py"]

    def test_delete_oserror(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "cat-file":
                return 1, ""
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py"])
        monkeypatch.setattr(os, "remove", lambda p: (_ for _ in ()).throw(OSError("no")))
        _undone, failed = asi._undo_run_changes("/x", {"ref": "r"})
        assert failed == ["a.py"]


class TestRenderRunDiff:
    def test_no_baseline(self, monkeypatch):
        assert asi._render_run_diff("/x", None) is False

    def test_no_changes(self, monkeypatch):
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: [])
        assert asi._render_run_diff("/x", {"ref": "r"}) is False

    def test_plain_branch(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py"])
        monkeypatch.setattr(asi, "_file_diff_text", lambda r, b, p: ("+x\n-x\n", False))
        assert asi._render_run_diff("/x", {"ref": "r"}) is True
        out = capsys.readouterr().out
        assert "changes" in out and "a.py" in out

    def test_rich_branch(self, monkeypatch):
        printed = []

        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)

            def rule(self, *a, **k):
                pass

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py", "b.py"])
        monkeypatch.setattr(asi, "_file_diff_text", lambda r, b, p: ("+x\n", False))
        assert asi._render_run_diff("/x", {"ref": "r"}) is True
        assert printed

    def test_extra_files_note(self, monkeypatch):
        printed = []

        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)

            def rule(self, *a, **k):
                pass

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: [f"f{i}.py" for i in range(25)])
        monkeypatch.setattr(asi, "_file_diff_text", lambda r, b, p: ("+x\n", False))
        assert asi._render_run_diff("/x", {"ref": "r"}, max_files=20) is True
        assert any("more file(s)" in str(a) for a in printed)


# ─── checkpoint helpers (mock store) ───────────────────────────────────────────


class _FakeStore:
    def __init__(self, entries=None):
        self.entries = entries or []
        self.checkpoints = self.entries
        self.checkpoint_dir = Path("/tmp")

    def list(self):
        return self.entries

    def restore(self, cid):
        return cid == "good"


class TestCheckpointHelpers:
    def test_load_store_failure(self, monkeypatch):
        def _boom(*a, **k):
            raise ImportError("no checkpoint_store")

        monkeypatch.setattr(asi, "CheckpointStore", _boom) if hasattr(asi, "CheckpointStore") else None
        monkeypatch.setitem(sys.modules, "external_llm.agent.checkpoint_store", None)
        # force the lazy import path to fail
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "external_llm.agent.checkpoint_store":
                raise ImportError("nope")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert asi._load_checkpoint_store("/x") is None

    def test_newest_id(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: _FakeStore([{"id": "c1"}, {"id": "c2"}]))
        assert asi._newest_checkpoint_id("/x") == "c1"

    def test_newest_id_empty(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: _FakeStore([]))
        assert asi._newest_checkpoint_id("/x") is None

    def test_newest_id_list_error(self, monkeypatch):
        class Bad(_FakeStore):
            def list(self):
                raise OSError("nope")

        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: Bad())
        assert asi._newest_checkpoint_id("/x") is None

    def test_undo_via_checkpoint_ok(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: _FakeStore())
        assert asi._undo_via_checkpoint("/x", "good") is True

    def test_undo_via_checkpoint_fail(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: _FakeStore())
        assert asi._undo_via_checkpoint("/x", "bad") is False

    def test_undo_no_store(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: None)
        assert asi._undo_via_checkpoint("/x", "x") is False

    def test_changed_files(self, monkeypatch, tmp_path):
        (tmp_path / "cp.json").write_text(json.dumps({"file_hashes": {"a.py": "h"}, "absent": ["b.py"]}))
        store = _FakeStore([{"id": "c1", "path": "cp.json"}])
        store.checkpoint_dir = tmp_path
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: store)
        assert asi._checkpoint_changed_files(str(tmp_path), "c1") == ["a.py", "b.py"]

    def test_changed_files_missing_entry(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: _FakeStore([]))
        assert asi._checkpoint_changed_files("/x", "nope") == []

    def test_changed_files_bad_json(self, monkeypatch, tmp_path):
        (tmp_path / "cp.json").write_text("{not json")
        store = _FakeStore([{"id": "c1", "path": "cp.json"}])
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: store)
        assert asi._checkpoint_changed_files(str(tmp_path), "c1") == []


# ─── clipboard ─────────────────────────────────────────────────────────────────


class TestCopyToClipboard:
    def test_empty_text(self):
        assert asi._copy_to_clipboard("") == ""

    def test_darwin_pbcopy(self, monkeypatch):
        monkeypatch.setattr(asi.sys, "platform", "darwin")
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd == ["pbcopy"]:
                raise OSError("no")

        monkeypatch.setattr(asi.subprocess, "run", fake_run)
        assert asi._copy_to_clipboard("x") == ""

    def test_darwin_success(self, monkeypatch):
        monkeypatch.setattr(asi.sys, "platform", "darwin")
        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: None)
        assert asi._copy_to_clipboard("x") == "pbcopy"

    def test_linux_success(self, monkeypatch):
        monkeypatch.setattr(asi.sys, "platform", "linux")
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[0] == "wl-copy":
                raise OSError("no wl-copy")

        monkeypatch.setattr(asi.subprocess, "run", fake_run)
        assert asi._copy_to_clipboard("x") == "xclip"

    def test_osc52_fallback(self, monkeypatch, capsys):
        monkeypatch.setattr(asi.sys, "platform", "linux")

        def fake_run(cmd, **kw):
            raise OSError("none available")

        monkeypatch.setattr(asi.subprocess, "run", fake_run)
        monkeypatch.setattr(asi.sys.stdout, "isatty", lambda: True)
        assert asi._copy_to_clipboard("hi") == "OSC 52"
        assert "52;c;" in capsys.readouterr().out

    def test_all_fail_not_tty(self, monkeypatch, capsys):
        monkeypatch.setattr(asi.sys, "platform", "linux")
        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
        monkeypatch.setattr(asi.sys.stdout, "isatty", lambda: False)
        assert asi._copy_to_clipboard("hi") == ""


# ─── relativize / extract / preview ────────────────────────────────────────────


class TestRelativizeRepoPaths:
    def test_strips_cd_prefix(self, monkeypatch):
        monkeypatch.setattr(asi, "_REPO_ROOT", "/home/user/repo")
        text = "cd /home/user/repo && python run.py"
        assert asi._relativize_repo_paths(text) == "python run.py"

    def test_quoted_cd_prefix(self, monkeypatch):
        monkeypatch.setattr(asi, "_REPO_ROOT", "/home/user/repo")
        text = "cd '/home/user/repo' && ls"
        assert asi._relativize_repo_paths(text) == "ls"

    def test_replaces_occurrences(self, monkeypatch):
        monkeypatch.setattr(asi, "_REPO_ROOT", "/home/user/repo")
        assert asi._relativize_repo_paths("cat /home/user/repo/x.py") == "cat x.py"
        assert asi._relativize_repo_paths("cd /home/user/repo") == "cd ."

    def test_root_shortcut(self, monkeypatch):
        monkeypatch.setattr(asi, "_REPO_ROOT", "/")
        assert asi._relativize_repo_paths("anything") == "anything"


class TestExtractToolCmd:
    def test_command(self, monkeypatch):
        monkeypatch.setattr(asi, "_relativize_repo_paths", lambda t: t)
        assert asi._extract_tool_cmd({"command": "ls  -la"}) == "ls -la"

    def test_cmd_alias(self, monkeypatch):
        monkeypatch.setattr(asi, "_relativize_repo_paths", lambda t: t)
        assert asi._extract_tool_cmd({"cmd": "pwd"}) == "pwd"

    def test_pattern_path(self, monkeypatch):
        monkeypatch.setattr(asi, "_relativize_repo_paths", lambda t: t)
        assert asi._extract_tool_cmd({"pattern": "def x", "file_path": "a.py"}) == "'def x' in a.py"

    def test_name(self, monkeypatch):
        monkeypatch.setattr(asi, "_relativize_repo_paths", lambda t: t)
        assert asi._extract_tool_cmd({"name": "foo", "path": "b.py"}) == "'foo' in b.py"

    def test_query(self):
        assert asi._extract_tool_cmd({"query": "hello world"}) == "'hello world'"

    def test_empty(self):
        assert asi._extract_tool_cmd({}) == ""
        assert asi._extract_tool_cmd(None) == ""

    def test_long_truncated(self, monkeypatch):
        monkeypatch.setattr(asi, "_relativize_repo_paths", lambda t: t)
        assert len(asi._extract_tool_cmd({"query": "q" * 500})) <= 200


class TestSelectPreviewLines:
    def test_find_relevant_files_strips_header(self):
        lines = ["Top 5 relevant file(s) for: x", "a", "b", "c", "d"]
        assert asi._select_preview_lines("find_relevant_files", lines) == ["a", "b", "c"]

    def test_bash_total_line(self):
        assert asi._select_preview_lines("bash", ["total 8", "a", "b", "c", "d"]) == ["a", "b", "c"]

    def test_write_tools_post_edit(self):
        lines = ["head", "[POST-EDIT DIFF]", "path +2", "NO CHANGE", "tail"]
        out = asi._select_preview_lines("apply_patch", lines)
        assert out == ["head", "path +2", "NO CHANGE", "tail"]

    def test_write_tools_no_post_edit(self):
        assert asi._select_preview_lines("edit_text", ["a", "b", "c"]) == ["a", "b"]

    def test_update_plan(self):
        lines = ["plan updated", "[~] item", "Goal: x"]
        assert asi._select_preview_lines("update_plan", lines) == ["plan updated", "[~] item"]

    def test_update_plan_goal_fallback(self):
        lines = ["plan updated", "other", "Goal: x"]
        assert asi._select_preview_lines("update_plan", lines) == ["plan updated", "Goal: x"]

    def test_three_line_tools(self):
        lines = ["1", "2", "3", "4"]
        assert asi._select_preview_lines("grep", lines) == ["1", "2", "3"]

    def test_default_two(self):
        assert asi._select_preview_lines("unknown", ["1", "2", "3"]) == ["1", "2"]

    def test_binary_file_lines_removed(self):
        assert asi._select_preview_lines("x", ["Binary file a matches", "ok"]) == ["ok"]


# ─── interrupt note ────────────────────────────────────────────────────────────


class TestBuildInterruptNote:
    def test_tool_results_count(self):
        note = asi._build_interrupt_note(type("R", (), {"content": "partial", "tool_results": [1, 2, 3]})())
        assert "3 tool call(s)" in note
        assert "partial" in note

    def test_content_only(self):
        note = asi._build_interrupt_note(type("R", (), {"content": "hi", "tool_results": None})())
        assert "hi" in note

    def test_empty(self):
        note = asi._build_interrupt_note(type("R", (), {"content": "", "tool_results": []})())
        assert "Interrupted" not in note
        assert "resume" in note


# ─── dotenv ────────────────────────────────────────────────────────────────────


class TestLoadDotenv:
    def test_sets_keys(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("KEY1=val1\nKEY2 = val2\n")
        monkeypatch.delenv("ASI_TEST_KEY1", raising=False)
        monkeypatch.delenv("ASI_TEST_KEY2", raising=False)
        monkeypatch.setenv("ASI_TEST_KEY1", "x")
        (tmp_path / ".env").write_text("ASI_TEST_KEY1=overridden\nASI_TEST_KEY2=new\n")
        asi._load_dotenv(str(tmp_path))
        assert os.environ["ASI_TEST_KEY1"] == "x"  # existing wins
        assert os.environ["ASI_TEST_KEY2"] == "new"
        os.environ.pop("ASI_TEST_KEY2", None)

    def test_export_and_comments(self, tmp_path):
        (tmp_path / ".env").write_text(
            "# comment\nexport ASI_EXPORT_KEY=exported\nASI_COMMENT_KEY=value # trailing\n\n"
        )
        for k in ("ASI_EXPORT_KEY", "ASI_COMMENT_KEY"):
            os.environ.pop(k, None)
        asi._load_dotenv(str(tmp_path))
        assert os.environ["ASI_EXPORT_KEY"] == "exported"
        assert os.environ["ASI_COMMENT_KEY"] == "value"
        os.environ.pop("ASI_EXPORT_KEY", None)
        os.environ.pop("ASI_COMMENT_KEY", None)

    def test_quoted_values(self, tmp_path):
        (tmp_path / ".env").write_text("ASI_Q1=\"quoted\"\nASI_Q2='single'\n")
        for k in ("ASI_Q1", "ASI_Q2"):
            os.environ.pop(k, None)
        asi._load_dotenv(str(tmp_path))
        assert os.environ["ASI_Q1"] == "quoted"
        assert os.environ["ASI_Q2"] == "single"
        os.environ.pop("ASI_Q1", None)
        os.environ.pop("ASI_Q2", None)

    def test_missing_file_noop(self, tmp_path):
        asi._load_dotenv(str(tmp_path))  # must not raise

    def test_hash_without_space_kept(self, tmp_path):
        (tmp_path / ".env").write_text("ASI_URL=https://host/path#frag\n")
        os.environ.pop("ASI_URL", None)
        asi._load_dotenv(str(tmp_path))
        assert os.environ["ASI_URL"] == "https://host/path#frag"
        os.environ.pop("ASI_URL", None)


# ─── logging classes ───────────────────────────────────────────────────────────


class TestFsyncedFileHandler:
    def test_flush_close(self, tmp_path):
        h = asi._FsyncedFileHandler(str(tmp_path / "log.txt"))
        h.emit(logging.LogRecord("n", logging.INFO, __file__, 1, "msg", None, None))
        h.flush()
        h.close()
        assert (tmp_path / "log.txt").exists()


class TestToolRunningFilter:
    def test_filter_active(self):
        f = asi._ToolRunningFilter()
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
        f.active = True
        assert f.filter(rec) is False
        rec2 = logging.LogRecord("n", logging.WARNING, __file__, 1, "m", None, None)
        assert f.filter(rec2) is True

    def test_filter_inactive(self):
        f = asi._ToolRunningFilter()
        f.active = False
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
        assert f.filter(rec) is True

    def test_row_pending_cleared_on_inactive(self, monkeypatch):
        f = asi._ToolRunningFilter()
        monkeypatch.setattr(asi, "_set_term_row_pending", lambda v: None)
        f.active = True
        f.active = False


class TestTerminalInfoFilter:
    def test_warning_passes(self):
        f = asi._TerminalInfoFilter()
        assert f.filter(logging.LogRecord("n", logging.WARNING, __file__, 1, "m", None, None)) is True

    def test_asi_progress_suppressed(self):
        f = asi._TerminalInfoFilter()
        assert f.filter(logging.LogRecord("asi.progress", logging.INFO, __file__, 1, "m", None, None)) is False

    def test_external_llm_suppressed(self):
        f = asi._TerminalInfoFilter()
        assert f.filter(logging.LogRecord("external_llm.agent", logging.INFO, __file__, 1, "m", None, None)) is False

    def test_torch_suppressed(self):
        f = asi._TerminalInfoFilter()
        assert f.filter(logging.LogRecord("torch", logging.INFO, __file__, 1, "m", None, None)) is False

    def test_other_passes(self):
        f = asi._TerminalInfoFilter()
        assert f.filter(logging.LogRecord("other", logging.INFO, __file__, 1, "m", None, None)) is True


class TestSafeRichFormatter:
    def test_escapes_markup(self):
        fmt = asi._SafeRichFormatter("[dim]%(levelname)s %(message)s[/dim]")
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "[bold]raw[/bold]", None, None)
        out = fmt.format(rec)
        assert "\\[bold]" in out  # open tag escaped; rich passes [/ through

    def test_warning_indent(self):
        fmt = asi._SafeRichFormatter("%(message)s")
        rec = logging.LogRecord("n", logging.WARNING, __file__, 1, "w", None, None)
        assert fmt.format(rec).startswith("  ")

    def test_tuple_args(self):
        fmt = asi._SafeRichFormatter("%(message)s")
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "%s", ("a[bold]",), None)
        out = fmt.format(rec)
        assert "\\[bold]" in out  # one literal backslash

    def test_dict_args(self):
        fmt = asi._SafeRichFormatter("%(message)s")
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "%(k)s", ({"k": "v[bold]"},), None)
        assert isinstance(rec.args, dict)  # py3.14 unwraps a single Mapping arg
        out = fmt.format(rec)
        assert "\\[bold]" in out  # one literal backslash


class TestSetupLogging:
    """_setup_logging mutates the root logger — restore handlers+level afterwards."""

    @pytest.fixture(autouse=True)
    def _restore_root_logging(self):
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        yield
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        asi._LOG_FILE_HANDLER = None

    def test_no_log_file(self, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_log_console", None)
        asi._setup_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_with_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_log_console", None)
        logfile = str(tmp_path / "logs" / "run_{date}_{time}.log")
        asi._setup_logging("DEBUG", log_file=logfile)
        assert asi._LOG_FILE_HANDLER is not None
        assert any(tmp_path.glob("logs/run_*.log"))


class TestHandleTerminalResize:
    def test_updates_widths(self, monkeypatch):
        class C:
            def __init__(self):
                self.width = 100

        cons = C()
        monkeypatch.setattr(asi, "_console", cons)
        monkeypatch.setattr(asi, "_log_console", C())
        monkeypatch.setattr(asi, "_out_console", C())
        monkeypatch.setattr(asi, "_active_spinner_printer", None)
        monkeypatch.setattr(asi, "_console_width", 100)
        asi._handle_terminal_resize()
        assert cons.width >= 40

    def test_resize_error_suppressed(self, monkeypatch):
        import shutil

        def _boom(*a, **k):
            raise OSError("bad tty")

        monkeypatch.setattr(shutil, "get_terminal_size", _boom)
        asi._handle_terminal_resize()  # must not raise


# ─── embedding cache helpers ───────────────────────────────────────────────────


class TestEmbeddingHelpers:
    def test_model_folder(self):
        assert asi._embedding_model_folder("all-MiniLM") == "models--sentence-transformers--all-MiniLM"
        assert asi._embedding_model_folder("org/m") == "models--org--m"

    def test_cache_roots(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hc")
        monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", "/tmp/st")
        monkeypatch.setenv("HF_HOME", "/tmp/hf")
        roots = asi._embedding_cache_roots()
        assert "/tmp/hc" in roots and "/tmp/st" in roots
        assert "/tmp/hf/hub" in roots
        assert roots[-1].endswith(".cache/huggingface/hub")

    def test_is_cached_false(self, monkeypatch):
        monkeypatch.setattr(asi, "_embedding_cache_roots", lambda: ["/nonexistent-xyz"])
        assert asi._is_embedding_model_cached("all-MiniLM-L6-v2") is False

    def test_is_cached_true(self, tmp_path, monkeypatch):
        snap = tmp_path / "models--sentence-transformers--m" / "snapshots" / "s1"
        snap.mkdir(parents=True)
        (snap / "x.bin").write_bytes(b"1")
        monkeypatch.setattr(asi, "_embedding_cache_roots", lambda: [str(tmp_path)])
        assert asi._is_embedding_model_cached("m") is True

    def test_cache_bytes_zero(self, monkeypatch):
        monkeypatch.setattr(asi, "_embedding_cache_roots", lambda: ["/nonexistent-xyz"])
        assert asi._embedding_cache_bytes("m") == 0

    def test_cache_bytes_counts(self, tmp_path, monkeypatch):
        base = tmp_path / "models--sentence-transformers--m"
        base.mkdir(parents=True)
        (base / "a").write_bytes(b"12345")
        monkeypatch.setattr(asi, "_embedding_cache_roots", lambda: [str(tmp_path)])
        assert asi._embedding_cache_bytes("m") == 5


# ─── update notice ─────────────────────────────────────────────────────────────


class TestMaybeShowUpdateNotice:
    def _patch_vc(self, monkeypatch, notice):
        import utils.version_check as vc

        class H:
            def collect(self, wait_s=0.0):
                return notice

        monkeypatch.setattr(vc, "start_update_check", lambda: H())

    def test_no_notice(self, monkeypatch):
        self._patch_vc(monkeypatch, "")
        asi._maybe_show_update_notice()

    def test_notice_printed(self, monkeypatch, capsys):
        self._patch_vc(monkeypatch, "new version available\n")
        asi._maybe_show_update_notice()
        assert "new version" in capsys.readouterr().err

    def test_failure_swallowed(self, monkeypatch):
        import utils.version_check as vc

        def _boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(vc, "start_update_check", _boom)
        asi._maybe_show_update_notice()  # must not raise


# ─── _git itself ───────────────────────────────────────────────────────────────


class TestGit:
    def test_success(self):
        rc, _out = asi._git("/tmp", "rev-parse", "--is-inside-work-tree")
        assert rc in (0, 128)

    def test_failure_never_raises(self, monkeypatch):
        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
        assert asi._git("/tmp", "x") == (1, "")
