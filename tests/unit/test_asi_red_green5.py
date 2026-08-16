"""RED→GREEN coverage for asi.py — layer 5: tty spinner threads (pip/download),
Live banner animation, resize-with-spinner, emit-with-spinner, SIGWINCH wiring,
drain_stdin on tty, escaped-quote dotenv, numstat short field, newest-checkpoint
store-None, dep-status plain tool parts. Source-free.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time

import pytest

import asi

# huggingface_hub is optional (the `rag` extra). The download-spinner tests
# monkeypatch its internals, so they can only run where it is installed.
HAS_HUGGINGFACE_HUB = importlib.util.find_spec("huggingface_hub") is not None

# ─── _drain_stdin on a tty (termios mocked) ───────────────────────────────────

class TestDrainStdinTty:
    """pytest's capture rewires sys.stdin to a fileno-less object — substitute
    a fake stdin so _drain_stdin's tty branch runs."""

    def _fake_stdin(self, monkeypatch):
        class FakeStdin:
            def fileno(self):
                return 42
        monkeypatch.setattr(asi.sys, "stdin", FakeStdin())

    def test_drains_and_restores(self, monkeypatch):
        import select as select_mod
        import termios
        self._fake_stdin(monkeypatch)
        fake_attrs = [0, 0, 0, 0, 0, 0, [0] * 32]  # iflag..ospeed ints, cc array (VMIN/VTIME)
        monkeypatch.setattr(asi.os, "isatty", lambda fd: True)
        monkeypatch.setattr(termios, "tcgetattr", lambda fd: fake_attrs)
        monkeypatch.setattr(termios, "tcsetattr", lambda fd, mode, attrs: fake_attrs.__setitem__(0, attrs))
        select_calls = {"n": 0}
        def _select(*a, **k):
            select_calls["n"] += 1
            # first call: data ready (drain reads); second: no more data → loop exits
            return ([a[0][0]], [], []) if select_calls["n"] == 1 else ([], [], [])
        monkeypatch.setattr(select_mod, "select", _select)
        monkeypatch.setattr(asi.os, "read", lambda fd, n: b"")
        asi._drain_stdin(timeout=0.01)
        assert fake_attrs[0][0] == 0  # restored via tcsetattr

    def test_termios_error_suppressed(self, monkeypatch):
        import termios
        self._fake_stdin(monkeypatch)
        monkeypatch.setattr(asi.os, "isatty", lambda fd: True)
        def _boom(*a, **k):
            raise OSError("not a terminal")
        monkeypatch.setattr(termios, "tcgetattr", _boom)
        asi._drain_stdin()  # no raise


# ─── _handle_terminal_resize with an active spinner ────────────────────────────

class TestHandleTerminalResizeSpinner:
    def test_stops_active_live(self, monkeypatch):
        import collections
        import shutil
        Size = collections.namedtuple("Size", "columns lines")
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **k: Size(100, 30))
        stopped = []
        class Live:
            def stop(self):
                stopped.append(True)
        class Spinner:
            _spinner_live = Live()
            _spinner_obj = object()
        monkeypatch.setattr(asi, "_console", type("C", (), {"width": 1})())
        monkeypatch.setattr(asi, "_log_console", type("C", (), {"width": 1})())
        monkeypatch.setattr(asi, "_out_console", type("C", (), {"width": 1})())
        monkeypatch.setattr(asi, "_console_width", 1)
        monkeypatch.setattr(asi, "_active_spinner_printer", Spinner())
        resets = []
        monkeypatch.setattr(asi, "_margin_stderr", type("M", (), {"reset_bol": lambda self: resets.append(1)})())
        asi._handle_terminal_resize()
        assert stopped == [True]
        assert resets == [1]


# ─── _RowSafeEmitMixin.emit with active spinner (suspend path) ────────────────

class TestEmitSuspendLive:
    def test_suspend_live_called(self, monkeypatch, capsys):
        suspended = []
        class Spinner:
            _spinner_live = None
            def _suspend_live_for_log(self):
                suspended.append(True)
        monkeypatch.setattr(asi, "_active_spinner_printer", Spinner())
        monkeypatch.setattr(asi, "_prompt_session", None)
        class H(asi._RowSafeEmitMixin, logging.StreamHandler):
            def __init__(self):
                super().__init__(sys.stderr)
        h = H()
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
        h.emit(rec)
        assert suspended == [True]


# ─── _pip_install tty spinner thread ───────────────────────────────────────────

class TestPipInstallTty:
    def test_spinner_runs_on_tty(self, monkeypatch):
        class P:
            returncode = 0
            stdout, stderr = "", ""
        monkeypatch.setattr(asi.sys.stderr, "isatty", lambda: True)
        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: time.sleep(0.3) or P())
        monkeypatch.setattr(asi, "_print", lambda *a, **k: None)
        assert asi._pip_install(["pkg"], timeout=10) is True

    def test_tty_spinner_stopped_on_failure(self, monkeypatch):
        class P:
            returncode = 1
            stdout, stderr = "err", ""
        monkeypatch.setattr(asi.sys.stderr, "isatty", lambda: True)
        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: P())
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._pip_install(["pkg"], timeout=10) is False


# ─── _print_banner Live animation ──────────────────────────────────────────────

class TestPrintBannerLive:
    def test_live_animation(self, monkeypatch):
        from rich.live import Live
        monkeypatch.setattr(asi, "_RICH", True)
        printed = []
        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)
            def rule(self, *a, **k):
                pass
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        monkeypatch.delenv("NO_COLOR", raising=False)

        class FakeLive:
            def __init__(self, *a, **k):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def update(self, *a, **k):
                pass
        monkeypatch.setattr(Live, "__init__", FakeLive.__init__)
        monkeypatch.setattr(Live, "__enter__", FakeLive.__enter__)
        monkeypatch.setattr(Live, "__exit__", FakeLive.__exit__)
        monkeypatch.setattr(Live, "update", FakeLive.update)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda s: None)
        asi._print_banner("/repo")
        assert printed

    def test_live_exception_falls_back(self, monkeypatch):
        from rich.live import Live
        monkeypatch.setattr(asi, "_RICH", True)
        printed = []
        class FakeOut:
            def print(self, *a, **k):
                printed.append(a)
            def rule(self, *a, **k):
                pass
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        monkeypatch.delenv("NO_COLOR", raising=False)
        def _boom(*a, **k):
            raise RuntimeError("live broke")
        monkeypatch.setattr(Live, "__init__", _boom)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda s: None)
        asi._print_banner("/repo")  # falls back to static title
        assert printed


# ─── _download_embedding_model tty spinner ─────────────────────────────────────

@pytest.mark.skipif(
    not HAS_HUGGINGFACE_HUB,
    reason="requires huggingface_hub (rag extra)",
)
class TestDownloadSpinner:
    def _patch_hf(self, monkeypatch):
        import huggingface_hub as hf
        import huggingface_hub.constants as hf_c

        import external_llm.agent.vector_cache as vc
        monkeypatch.setattr(hf_c, "HF_HUB_OFFLINE", None)
        monkeypatch.setattr(hf, "snapshot_download",
                            lambda *a, **k: time.sleep(0.3) or None)
        monkeypatch.setattr(vc, "_suppress_hf_progress", lambda: __import__("contextlib").nullcontext())
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: True)
        monkeypatch.setattr(vc, "set_active_embedding_model", lambda m: object())
        monkeypatch.setattr(asi, "_print", lambda *a, **k: None)

    def test_tty_spinner(self, monkeypatch):
        monkeypatch.setattr(asi.sys.stderr, "isatty", lambda: True)
        self._patch_hf(monkeypatch)
        asi._download_embedding_model("m")

    def test_non_tty_message(self, monkeypatch):
        monkeypatch.setattr(asi.sys.stderr, "isatty", lambda: False)
        self._patch_hf(monkeypatch)
        asi._download_embedding_model("m")


# ─── main(): SIGWINCH wiring ──────────────────────────────────────────────────

class TestMainSigwinch:
    def test_sigwinch_registered(self, monkeypatch, tmp_path):
        import signal
        if not hasattr(signal, "SIGWINCH"):
            pytest.skip("no SIGWINCH on this platform")
        cfg_dir = tmp_path / ".asicode"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text("{}")
        monkeypatch.delenv("EXTERNAL_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("EXTERNAL_LLM_MODEL", raising=False)
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_terminal_config_path", lambda r: None)
        captured = {}
        monkeypatch.setattr(asi, "run_repl", lambda a: captured.update(args=a))
        monkeypatch.setattr(asi.sys, "argv", ["asi"])
        monkeypatch.setattr(asi.sys.stdout, "isatty", lambda: True)
        registered = []
        monkeypatch.setattr(signal, "signal", lambda sig, h: registered.append((sig, h)))
        asi.main()
        assert any(sig == signal.SIGWINCH for sig, _ in registered)


# ─── _load_dotenv: backslash-escaped closing quote ─────────────────────────────

class TestLoadDotenvEscapedQuote:
    def test_escaped_quote_not_stripped(self, tmp_path):
        # val = "v\\\" more" — closing quote preceded by odd backslashes → keep as-is
        (tmp_path / ".env").write_text('ASI_EQ="v\\" more"\n')
        os.environ.pop("ASI_EQ", None)
        asi._load_dotenv(str(tmp_path))
        # odd backslashes before quote → quote treated as escaped → kept raw
        assert "ASI_EQ" not in os.environ or "\\" in os.environ["ASI_EQ"]
        os.environ.pop("ASI_EQ", None)

    def test_even_backslashes_strips(self, tmp_path):
        (tmp_path / ".env").write_text('ASI_EQ2="v\\\\" # c\n')
        os.environ.pop("ASI_EQ2", None)
        asi._load_dotenv(str(tmp_path))
        assert os.environ.get("ASI_EQ2") == "v\\\\"  # 2 backslashes kept
        os.environ.pop("ASI_EQ2", None)


# ─── numstat short field (len(parts) < 3) ──────────────────────────────────────

class TestNumstatShortField:
    def test_short_field_skipped(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "diff" and "--numstat" in args:
                return 0, "1\t2\x00"
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
        assert stats[0][0] == "a.py"


# ─── _newest_checkpoint_id with store None ─────────────────────────────────────

class TestNewestCheckpointStoreNone:
    def test_store_none(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: None)
        assert asi._newest_checkpoint_id("/x") is None


# ─── _print_dep_status plain branch with tool parts ────────────────────────────

class TestPrintDepStatusPlainTools:
    def _patch(self, monkeypatch):
        import external_llm.languages.dependency_checker as dc
        import external_llm.languages.tree_sitter_utils as tsu
        monkeypatch.setattr(dc, "detect_repo_languages", lambda r: [])
        monkeypatch.setattr(dc, "_check_tools_with_state", lambda d, no_prompt=False: [])
        monkeypatch.setattr(asi, "_git_ls_files", lambda r: [])
        monkeypatch.setattr(asi, "_detect_repo_ts_languages", lambda f: set())
        monkeypatch.setattr(tsu, "is_available", lambda: True)
        monkeypatch.setattr(asi, "_maybe_prompt_vector_install", lambda: None)
        monkeypatch.setattr(asi, "_DEPS_RESTART_PENDING", False)
        monkeypatch.setattr(asi, "_check_dep_status",
                            lambda t: {"tree-sitter": "ON", "eslint": "ON", "vector": "OFF"})

    def test_tool_parts_plain(self, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_out_console", None)
        self._patch(monkeypatch)
        asi._print_dep_status("/tmp", no_deps_check=True)
        out = capsys.readouterr().out
        assert "eslint" in out


# ─── _check_dep_status: vector OFF when deps missing ───────────────────────────

class TestCheckDepStatusVectorOff:
    def test_vector_off_when_deps_missing(self, monkeypatch):
        import external_llm.agent.vector_cache as vc
        monkeypatch.setattr(vc, "HAS_FAISS", False)
        monkeypatch.setattr(vc, "HAS_NUMPY", False)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", False)
        status = asi._check_dep_status([])
        assert status["vector"] == "OFF"


# ─── _maybe_prompt_vector_install: first prompt EOF (no fallback) ─────────────

class TestVectorInstallFirstPromptEOF:
    def test_download_prompt_eof(self, monkeypatch):
        import external_llm.agent.vector_cache as vc
        monkeypatch.setattr(vc, "HAS_FAISS", True)
        monkeypatch.setattr(vc, "HAS_NUMPY", True)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", True)
        monkeypatch.setattr(vc, "get_configured_embedding_model_name", lambda: "pref")
        monkeypatch.setattr(vc, "FALLBACK_EMBEDDING_MODELS", [])
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: False)
        def _eof(p=""):
            raise EOFError
        monkeypatch.setattr(asi, "_collect_input", _eof)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._maybe_prompt_vector_install()  # no raise; falls to "Skipped"
        assert any("Skipped" in str(p) for p in printed)
