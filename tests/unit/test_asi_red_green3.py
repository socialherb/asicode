"""RED→GREEN coverage for asi.py — layer 3: spinner, emit mixins, rich branches,
remaining edge branches (resize, clipboard-win, dotenv debug, exact-name lookup,
insights budget warning, dep-status rich, main() env fallbacks). Source-free.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import asi

# ─── _ShimmerSpinner ───────────────────────────────────────────────────────────


class TestShimmerSpinner:
    def test_render_basic(self):
        sp = asi._ShimmerSpinner("working", "blue")
        out = sp.render(0.0)
        assert "working" in str(out)

    def test_render_empty_body(self):
        sp = asi._ShimmerSpinner("", "blue")
        out = sp.render(1.0)
        assert "working" not in str(out)

    def test_render_whitespace_body(self):
        sp = asi._ShimmerSpinner("   ", "blue")
        out = sp.render(1.0)
        assert out is not None

    def test_render_indented_body(self):
        sp = asi._ShimmerSpinner("  thinking", "blue")
        out = sp.render(0.2)
        assert "thinking" in str(out)

    def test_update_text_and_style(self):
        sp = asi._ShimmerSpinner("a", "blue")
        sp.update(text="b", style="red")
        assert sp._spinner.text is not None
        assert sp._spin_style == "red"

    def test_update_rich_text(self):
        from rich.text import Text

        sp = asi._ShimmerSpinner("a", "blue")
        sp.update(text=Text("rich"))
        assert sp._spinner.text is not None

    def test_start_time_property(self):
        sp = asi._ShimmerSpinner("a", "blue")
        sp.start_time = 1.5
        assert sp.start_time == 1.5

    def test_rich_console_protocol(self):
        sp = asi._ShimmerSpinner("x", "blue")
        console = type("C", (), {"get_time": staticmethod(lambda: 0.0)})()
        items = list(sp.__rich_console__(console, None))
        assert items

    def test_rich_measure(self):
        sp = asi._ShimmerSpinner("x", "blue")
        console = type("C", (), {"get_time": staticmethod(lambda: 0.0)})()
        options = type("O", (), {"max_width": 80})()
        m = sp.__rich_measure__(console, options)
        assert m is not None


# ─── console ensures ───────────────────────────────────────────────────────────


class TestConsoleEnsures:
    def test_ensure_console_imported_rich(self, monkeypatch):
        monkeypatch.setattr(asi, "_console", None)
        monkeypatch.setattr(asi, "_RICH", True)
        asi._ensure_console_imported()
        assert asi._console is not None

    def test_ensure_console_imported_no_rich(self, monkeypatch):
        monkeypatch.setattr(asi, "_console", None)
        monkeypatch.setattr(asi, "_RICH", False)
        asi._ensure_console_imported()
        assert asi._console is None

    def test_ensure_console_already_set(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(asi, "_console", sentinel)
        asi._ensure_console_imported()
        assert asi._console is sentinel

    def test_import_rich_console_no_rich(self, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", False)
        assert asi._import_rich_console() is None


# ─── _ToolRunningFilter / _RowSafeEmitMixin / _FsyncedFileHandler edges ────────


class TestToolRunningFilterEdges:
    def test_active_property(self):
        f = asi._ToolRunningFilter()
        assert f.active is False
        f.active = True
        assert f.active is True

    def test_row_pending_roundtrip(self, monkeypatch):
        f = asi._ToolRunningFilter()
        monkeypatch.setattr(asi, "_term_row_pending", lambda: True)
        assert f.row_pending is True
        monkeypatch.setattr(asi, "_set_term_row_pending", lambda v: None)
        f.row_pending = False


class TestFsyncedFileHandlerCloseError:
    def test_fsync_failure_prints(self, tmp_path, monkeypatch, capsys):
        h = asi._FsyncedFileHandler(str(tmp_path / "log.txt"))

        class BadStream:
            def fileno(self):
                return 999999

        h.stream = BadStream()
        h.close()
        assert "fsync failed" in capsys.readouterr().err


class TestRowSafeEmitMixin:
    def test_emit_plain(self, monkeypatch, capsys):
        class H(asi._RowSafeEmitMixin, logging.StreamHandler):
            def __init__(self):
                super().__init__(sys.stderr)

        h = H()
        monkeypatch.setattr(asi, "_active_spinner_printer", None)
        monkeypatch.setattr(asi, "_prompt_session", None)
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "hello", None, None)
        h.emit(rec)
        assert "hello" in capsys.readouterr().err

    def test_emit_row_pending_breaks_row(self, monkeypatch, capsys):
        class H(asi._RowSafeEmitMixin, logging.StreamHandler):
            def __init__(self):
                super().__init__(sys.stderr)

        h = H()
        monkeypatch.setattr(asi, "_active_spinner_printer", None)
        monkeypatch.setattr(asi, "_prompt_session", None)
        monkeypatch.setattr(asi, "_term_row_pending", lambda: True)
        monkeypatch.setattr(asi, "_set_term_row_pending", lambda v: None)
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
        h.emit(rec)
        # newline break written before the record
        out = capsys.readouterr().err
        assert "m" in out

    def test_emit_with_running_app_invalidate(self, monkeypatch):
        invalidated = []

        class App:
            is_running = True

            def invalidate(self):
                invalidated.append(True)

        class Sess:
            app = App()

        monkeypatch.setattr(asi, "_prompt_session", Sess())

        class H(asi._RowSafeEmitMixin, logging.StreamHandler):
            def __init__(self):
                super().__init__(sys.stderr)

        h = H()
        monkeypatch.setattr(asi, "_active_spinner_printer", None)
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "plain-message", None, None)
        h.emit(rec)
        assert invalidated == [True]


# ─── _SafeRichFormatter str-args / misc edges ──────────────────────────────────


class TestSafeRichFormatterEdges:
    def test_non_str_msg(self):
        fmt = asi._SafeRichFormatter("%(message)s")
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, 42, None, None)
        assert fmt.format(rec) == "42"


class TestRichMarkdownCls:
    def test_returns_class(self):
        cls = asi._rich_markdown_cls()
        assert cls is not None


# ─── _render_run_diff remaining edges ──────────────────────────────────────────


class TestRenderRunDiffEdges:
    def test_rendered_empty_returns_false(self, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py"])
        monkeypatch.setattr(asi, "_file_diff_text", lambda r, b, p: ("", False))
        assert asi._render_run_diff("/x", {"ref": "r"}) is False

    def test_plain_branch_diff_lines(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py"])
        monkeypatch.setattr(asi, "_file_diff_text", lambda r, b, p: ("@@ -1 +1 @@\n-old\n+new\n context\n", False))
        asi._render_run_diff("/x", {"ref": "r"})
        out = capsys.readouterr().out
        assert "-old" in out and "+new" in out

    def test_rich_branch_summary_print(self, monkeypatch):
        printed = []

        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)

            def rule(self, *a, **k):
                pass

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        monkeypatch.setattr(asi, "_changed_files_since", lambda r, b: ["a.py"])
        monkeypatch.setattr(asi, "_file_diff_text", lambda r, b, p: ("+x\n", False))
        assert asi._render_run_diff("/x", {"ref": "r"}) is True
        assert any("changes" in str(a) for a in printed)


class TestRunChangedStatsEdges:
    def test_numstat_skip_fields(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "diff" and "--numstat" in args:
                return 0, "\x00notab\x00"
            if "--no-index" in args:
                return 0, "+n\n"
            if args[0] == "diff" and args[1] == "--no-color":
                return 0, ""
            if args[0] == "diff":
                return 0, "a.py\x00"
            if args[0] == "ls-files":
                return 0, ""
            return 0, ""

        monkeypatch.setattr(asi, "_git", fake_git)
        stats = asi._run_changed_stats("/x", {"ref": "r", "untracked": frozenset()})
        assert stats[0][0] == "a.py"  # fallback path used


class TestPrintRunChangeSummaryRich:
    def test_rich_branch(self, monkeypatch):
        printed = []

        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        monkeypatch.setattr(asi, "_run_changed_stats", lambda r, b: [("a.py", 2, 1, False), ("b.py", 0, 0, True)])
        assert asi._print_run_change_summary("/x", {"ref": "r"}) is True
        assert any("a.py" in str(a) for a in printed)


# ─── _resolve_model_interactive exact-name / ollama-exact ─────────────────────


class TestResolveModelInteractiveExact:
    def test_exact_known_model(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [])
        monkeypatch.setattr(asi, "_get_ollama_models", lambda timeout=3: [])
        known = {}
        for prov, models in asi._KNOWN_MODELS.items():
            for m in models:
                known.setdefault(m, prov)
        sample = next(iter(known))
        assert asi._resolve_model_interactive(sample) == (known[sample], sample)

    def test_exact_ollama_model(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [])
        monkeypatch.setattr(asi, "_get_ollama_models", lambda timeout=3: ["qwen3:8b"])
        assert asi._resolve_model_interactive("qwen3:8b") == ("ollama", "qwen3:8b")

    def test_exact_unknown_without_warn(self, monkeypatch):
        monkeypatch.setattr(asi, "_model_candidates", lambda p: [])
        monkeypatch.setattr(asi, "_get_ollama_models", lambda timeout=3: [])
        assert asi._resolve_model_interactive("zzz-nope", warn_unknown=False) is None


# ─── insights archive budget warning / drop index ──────────────────────────────


class TestInsightsArchiveEdges:
    def _write(self, tmp_path, active_body, archive_body):
        d = tmp_path / ".asicode"
        d.mkdir()
        (d / "design_insights.md").write_text(active_body)
        (d / "design_insights_archive.md").write_text(archive_body)
        return tmp_path

    def test_restore_over_budget_warning(self, tmp_path, monkeypatch):
        repo = self._write(
            tmp_path,
            "# H\n\n### [pattern] 2026-01-01 10:00 +0900\na\n",
            "# A\n\n### [pattern] 2026-01-02 10:00 +0900\n" + "x" * 8000 + "\n",
        )
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive restore 1")
        assert any("over budget" in str(p) for p in printed)

    def test_drop_bad_index(self, tmp_path, monkeypatch):
        repo = self._write(tmp_path, "# H\n", "# A\n\n### [pattern] 2026-01-02 10:00 +0900\nb\n")
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive drop 5")
        assert any("no archive entry" in str(p) for p in printed)

    def test_drop_non_int(self, tmp_path, monkeypatch):
        repo = self._write(tmp_path, "# H\n", "# A\n\n### [pattern] 2026-01-02 10:00 +0900\nb\n")
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive drop zz")
        assert any("no archive entry" in str(p) for p in printed)


# ─── clipboard win branch / extract edges ──────────────────────────────────────


class TestClipboardWin:
    def test_win_clip(self, monkeypatch):
        monkeypatch.setattr(asi.sys, "platform", "win32")
        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: None)
        assert asi._copy_to_clipboard("x") == "clip"


class TestExtractToolCmdEdges:
    def test_fpath_only(self, monkeypatch):
        monkeypatch.setattr(asi, "_relativize_repo_paths", lambda t: t)
        assert asi._extract_tool_cmd({"file_path": "a/b.py"}) == "a/b.py"

    def test_query_truncated(self):
        q = "q" * 300
        assert asi._extract_tool_cmd({"query": q}) == f"'{q[:80]}'"


# ─── _load_dotenv debug-log branches ───────────────────────────────────────────


class TestLoadDotenvEdges:
    def test_quoted_inline_comment_logged(self, tmp_path, monkeypatch):
        # Never patch logging.getLogger — pytest's logging plugin owns the root
        # logger; a fake would break its teardown. The debug() call inside
        # _load_dotenv is harmless on the real "asi" logger.
        (tmp_path / ".env").write_text('ASI_DC="v # comment"\n')
        os.environ.pop("ASI_DC", None)
        asi._load_dotenv(str(tmp_path))
        assert os.environ.get("ASI_DC") == "v # comment"
        os.environ.pop("ASI_DC", None)

    def test_unquoted_inline_comment_logged(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("ASI_DC2=value # comment\n")
        os.environ.pop("ASI_DC2", None)
        asi._load_dotenv(str(tmp_path))
        assert os.environ.get("ASI_DC2") == "value"
        os.environ.pop("ASI_DC2", None)


# ─── _check_dep_status edges ───────────────────────────────────────────────────


class TestCheckDepStatusEdges:
    def test_skip_and_duplicate(self, monkeypatch):
        class T:
            def __init__(self, cmd, found, skipped):
                self.cmd, self.found, self.skipped = cmd, found, skipped

        tools = [
            T("tree-sitter", True, False),
            T("dup", True, False),
            T("dup", True, False),
            T("skipped-tool", False, True),
        ]
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: True)
        status = asi._check_dep_status(tools)
        assert status["dup"] == "ON"  # first wins
        assert status["skipped-tool"] == "skip"
        assert "vector" in status


# ─── _print_dep_status rich branch ─────────────────────────────────────────────


class TestPrintDepStatusRich:
    def _patch(self, monkeypatch, tools=(), files=(), langs=()):
        import external_llm.languages.dependency_checker as dc
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(dc, "detect_repo_languages", lambda r: [])
        monkeypatch.setattr(dc, "_check_tools_with_state", lambda d, no_prompt=False: list(tools))
        monkeypatch.setattr(asi, "_git_ls_files", lambda r: list(files))
        monkeypatch.setattr(asi, "_detect_repo_ts_languages", lambda f: set(langs))
        monkeypatch.setattr(tsu, "is_available", lambda: True)
        monkeypatch.setattr(tsu, "_get_language", lambda lang: None)
        monkeypatch.setattr(asi, "_maybe_prompt_vector_install", lambda: None)
        monkeypatch.setattr(asi, "_DEPS_RESTART_PENDING", False)
        monkeypatch.setattr(asi, "_check_dep_status", lambda t: {"vector": "ON"})

    def test_rich_branch_no_missing(self, monkeypatch):
        printed = []

        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        self._patch(monkeypatch)
        asi._print_dep_status("/tmp", no_deps_check=True)
        assert any("tree-sitter" in str(a) for a in printed)

    def test_rich_branch_missing_grammar(self, monkeypatch):
        printed = []

        class FakeOut:
            file = None

            def print(self, *a, **k):
                printed.append(a)

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        self._patch(monkeypatch, files=["a.py"], langs={"python"})
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "n")
        asi._print_dep_status("/tmp", no_deps_check=True)
        assert any("grammar" in str(a) for a in printed)

    def test_missing_grammar_eof(self, monkeypatch):
        def _eof(p=""):
            raise EOFError

        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        self._patch(monkeypatch, files=["a.py"], langs={"python"})
        monkeypatch.setattr(asi, "_collect_input", _eof)
        asi._print_dep_status("/tmp", no_deps_check=True)  # no raise

    def test_restart_pending(self, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        self._patch(monkeypatch)
        monkeypatch.setattr(asi, "_DEPS_RESTART_PENDING", True)
        restarted = []
        monkeypatch.setattr(asi, "_restart_cli", lambda: restarted.append(True))
        asi._print_dep_status("/tmp", no_deps_check=True)
        assert restarted == [True]


# ─── _render_status rich helper row ────────────────────────────────────────────


class TestRenderStatusRichHelper:
    def test_helper_row_rich(self, monkeypatch):
        printed = []

        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        asi._render_status("/r", "p", "m", "code", {"prompt": 10}, helper="h-helper")
        assert any("h-helper" in str(a) for a in printed)


# ─── main() env/config fallbacks ───────────────────────────────────────────────


class TestMainEnvFallbacks:
    def test_env_provider_model_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("EXTERNAL_LLM_MODEL", "deepseek-r1")
        cfg_dir = tmp_path / ".asicode"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text("{}")
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_terminal_config_path", lambda r: None)
        captured = {}
        monkeypatch.setattr(asi, "run_repl", lambda a: captured.update(args=a))
        monkeypatch.setattr(asi.sys, "argv", ["asi"])
        asi.main()
        assert captured["args"].provider == "deepseek"
        assert captured["args"].model == "deepseek-r1"

    def test_config_json_corrupt(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EXTERNAL_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("EXTERNAL_LLM_MODEL", raising=False)
        cfg_dir = tmp_path / ".asicode"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text("{corrupt")
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_terminal_config_path", lambda r: None)
        captured = {}
        monkeypatch.setattr(asi, "run_repl", lambda a: captured.update(args=a))
        monkeypatch.setattr(asi.sys, "argv", ["asi"])
        asi.main()
        assert captured["args"].provider == ""  # fell back to env, none set

    def test_config_json_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EXTERNAL_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("EXTERNAL_LLM_MODEL", raising=False)
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_terminal_config_path", lambda r: None)
        captured = {}
        monkeypatch.setattr(asi, "run_repl", lambda a: captured.update(args=a))
        monkeypatch.setattr(asi.sys, "argv", ["asi"])
        asi.main()
        assert captured["args"].model == ""

    def test_terminal_config_seed(self, monkeypatch, tmp_path):
        cfg_dir = tmp_path / ".asicode"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(json.dumps({"provider": "openai", "model": "gpt-4o"}))
        term_cfg = tmp_path / "term.json"
        seeded = []
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_terminal_config_path", lambda r: str(term_cfg))
        monkeypatch.setattr(asi, "_seed_terminal_config", lambda t, s: seeded.append((t, s)))
        captured = {}
        monkeypatch.setattr(asi, "run_repl", lambda a: captured.update(args=a))
        monkeypatch.setattr(asi.sys, "argv", ["asi"])
        (term_cfg).write_text(json.dumps({"provider": "anthropic", "model": "claude"}))
        asi.main()
        assert seeded  # terminal config seeded from shared
        assert captured["args"].provider == "anthropic"
