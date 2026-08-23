"""RED→GREEN coverage for asi.py — layer 4: log retention, checkpoint-store
exceptions, numstat garbage, empty active-file restore, vector no-model,
repo-files failure, dotenv debug logging, download-offline bypass, misc edges.
Source-free.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import pytest

import asi

# huggingface_hub is optional (the `rag` extra). The download-offline-bypass
# tests monkeypatch its internals, so they can only run where it is installed.
HAS_HUGGINGFACE_HUB = importlib.util.find_spec("huggingface_hub") is not None

# ─── _setup_logging: 30-day retention + 50-file cap ───────────────────────────


class TestLogRetention:
    @pytest.fixture(autouse=True)
    def _restore_root_logging(self):
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        yield
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        asi._LOG_FILE_HANDLER = None

    def test_deletes_old_logs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_log_console", None)
        old = tmp_path / "old.log"
        old.write_text("x")
        import time as _t

        _old_ts = _t.time() - 31 * 86400
        os.utime(old, (_old_ts, _old_ts))
        fresh = tmp_path / "fresh.log"
        fresh.write_text("y")
        logfile = str(tmp_path / "run_{date}_{time}.log")
        asi._setup_logging("INFO", log_file=logfile)
        assert not old.exists()
        assert fresh.exists()

    def test_keeps_at_most_50(self, tmp_path, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", False)
        monkeypatch.setattr(asi, "_log_console", None)
        for i in range(55):
            (tmp_path / f"f{i:02d}.log").write_text("x")
        asi._setup_logging("INFO", log_file=str(tmp_path / "run_{date}_{time}.log"))
        remaining = list(tmp_path.glob("*.log"))
        assert len(remaining) <= 51  # 50 kept + the new run log


# ─── checkpoint store exception paths ──────────────────────────────────────────


class _FakeStore:
    def __init__(self, entries=None, fail_list=False, fail_restore=False):
        self.entries = entries or []
        self.checkpoints = self.entries
        self.checkpoint_dir = Path("/tmp")
        self._fail_list = fail_list
        self._fail_restore = fail_restore

    def list(self):
        if self._fail_list:
            raise OSError("corrupt index")
        return self.entries

    def restore(self, cid):
        if self._fail_restore:
            raise OSError("restore failed")
        return True


class TestCheckpointExceptionPaths:
    def test_load_store_exception(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("checkpoint_store import ok but ctor fails")

        # CheckpointStore is imported lazily inside _load_checkpoint_store
        import external_llm.agent.checkpoint_store as cs_mod

        class Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("ctor boom")

        monkeypatch.setattr(cs_mod, "CheckpointStore", Boom)
        assert asi._load_checkpoint_store("/x") is None

    def test_undo_restore_exception(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: _FakeStore(fail_restore=True))
        assert asi._undo_via_checkpoint("/x", "c") is False

    def test_newest_list_exception(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: _FakeStore(fail_list=True))
        assert asi._newest_checkpoint_id("/x") is None

    def test_changed_files_no_store(self, monkeypatch):
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: None)
        assert asi._checkpoint_changed_files("/x", "c") == []

    def test_changed_files_oserror(self, monkeypatch, tmp_path):
        store = _FakeStore([{"id": "c", "path": "missing.json"}])
        store.checkpoint_dir = tmp_path
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: store)
        assert asi._checkpoint_changed_files(str(tmp_path), "c") == []

    def test_changed_files_bad_json(self, monkeypatch, tmp_path):
        (tmp_path / "cp.json").write_text("not json at all")
        store = _FakeStore([{"id": "c", "path": "cp.json"}])
        store.checkpoint_dir = tmp_path
        monkeypatch.setattr(asi, "_load_checkpoint_store", lambda r: store)
        assert asi._checkpoint_changed_files(str(tmp_path), "c") == []


# ─── _run_changed_stats: malformed numstat field ───────────────────────────────


class TestRunChangedStatsMalformed:
    def test_garbage_fields_skipped(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "diff" and "--numstat" in args:
                return 0, "\x00\x00"
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

    def test_short_field_skipped(self, monkeypatch):
        def fake_git(r, *args, **kw):
            if args[0] == "diff" and "--numstat" in args:
                return 0, "1\t2\tp.py\x00"
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


# ─── _commit_verified_api_key: missing env_var/key ─────────────────────────────


class TestCommitApiKeyEdges:
    def test_missing_env_var(self, monkeypatch):
        asi._PENDING_API_KEY.update({"env_var": "", "key": "k", "provider": "x"})
        calls = []
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: "/tmp")
        monkeypatch.setattr(asi, "_save_key_to_dotenv", lambda *a, **k: calls.append(a))
        asi._commit_verified_api_key()
        assert calls == []
        assert not asi._PENDING_API_KEY


# ─── insights archive: empty active file on restore ────────────────────────────


class TestInsightsArchiveEmptyActive:
    def test_restore_into_empty_active(self, tmp_path, monkeypatch):
        d = tmp_path / ".asicode"
        d.mkdir()
        (d / "design_insights.md").write_text("")
        (d / "design_insights_archive.md").write_text("# A\n\n### [pattern] 2026-01-02 10:00 +0900\nbody-b\n")
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(tmp_path), "archive restore 1")
        active = (d / "design_insights.md").read_text()
        assert "body-b" in active


# ─── _grouped_slash_commands: no "other" when everything is grouped ───────────


class TestGroupedNoOther:
    def test_leftover_absent(self):
        groups = asi._grouped_slash_commands()
        titles = [t for t, _ in groups]
        # every command in _SLASH_COMMANDS is listed in _SLASH_GROUPS
        grouped_names = {n for _, cmds in groups for c in cmds for n in (c[0],)}
        assert {c[0] for c in asi._SLASH_COMMANDS} == grouped_names
        assert "other" not in titles


# ─── _check_dep_status: vector no-model ────────────────────────────────────────


class TestCheckDepStatusNoModel:
    def test_vector_no_model(self, monkeypatch):
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: False)
        status = asi._check_dep_status([])
        assert status["vector"] in ("OFF", "no-model")


# ─── _git_ls_files failure → [] ────────────────────────────────────────────────


class TestGitLsFilesFailure:
    def test_delegation_failure_returns_empty(self, monkeypatch):
        import external_llm.common.repo_files as rf

        monkeypatch.setattr(rf, "git_list_repo_files", lambda r: None)
        assert asi._git_ls_files("/x") == []


# ─── _print_dep_status: rich branch with tool_parts ───────────────────────────


class TestPrintDepStatusRichToolParts:
    def _patch(self, monkeypatch, tools=()):
        import external_llm.languages.dependency_checker as dc
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(dc, "detect_repo_languages", lambda r: [])
        monkeypatch.setattr(dc, "_check_tools_with_state", lambda d, no_prompt=False: list(tools))
        monkeypatch.setattr(asi, "_git_ls_files", lambda r: [])
        monkeypatch.setattr(asi, "_detect_repo_ts_languages", lambda f: set())
        monkeypatch.setattr(tsu, "is_available", lambda: True)
        monkeypatch.setattr(asi, "_maybe_prompt_vector_install", lambda: None)
        monkeypatch.setattr(asi, "_DEPS_RESTART_PENDING", False)
        monkeypatch.setattr(asi, "_check_dep_status", lambda t: {"tree-sitter": "ON", "eslint": "ON", "vector": "OFF"})

    def test_tool_parts_rich(self, monkeypatch):
        printed = []

        class FakeOut:
            file = None

            def print(self, *a, **k):
                printed.append(a)

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        self._patch(monkeypatch)
        asi._print_dep_status("/tmp", no_deps_check=True)
        assert any("eslint" in str(a) for a in printed)

    def test_no_tool_parts_rich(self, monkeypatch):
        printed = []

        class FakeOut:
            file = None

            def print(self, *a, **k):
                printed.append(a)

        monkeypatch.setattr(asi, "_RICH", True)
        monkeypatch.setattr(asi, "_out_console", FakeOut())
        self._patch(monkeypatch)
        monkeypatch.setattr(asi, "_check_dep_status", lambda t: {"tree-sitter": "ON", "vector": "OFF"})
        asi._print_dep_status("/tmp", no_deps_check=True)
        assert printed


# ─── _maybe_prompt_vector_install EOF branches ─────────────────────────────────


class TestVectorInstallEOF:
    def _patch(self, monkeypatch, deps_ok=False):
        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(vc, "HAS_FAISS", deps_ok)
        monkeypatch.setattr(vc, "HAS_NUMPY", deps_ok)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", deps_ok)
        monkeypatch.setattr(vc, "get_configured_embedding_model_name", lambda: "pref")
        monkeypatch.setattr(vc, "FALLBACK_EMBEDDING_MODELS", ["fall"])
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: False)

    def test_deps_missing_eof(self, monkeypatch):
        self._patch(monkeypatch, deps_ok=False)

        def _eof(p=""):
            raise EOFError

        monkeypatch.setattr(asi, "_collect_input", _eof)
        asi._maybe_prompt_vector_install()  # no raise

    def test_fallback_prompt_eof(self, monkeypatch):
        self._patch(monkeypatch, deps_ok=True)
        answers = iter(["n"])

        def _input(p=""):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError from None

        monkeypatch.setattr(asi, "_collect_input", _input)
        asi._maybe_prompt_vector_install()  # no raise


# ─── _download_embedding_model: offline bypass ─────────────────────────────────


@pytest.mark.skipif(
    not HAS_HUGGINGFACE_HUB,
    reason="requires huggingface_hub (rag extra)",
)
class TestDownloadOfflineBypass:
    def test_hf_offline_bypass_restored(self, monkeypatch):
        import huggingface_hub as hf
        import huggingface_hub.constants as hf_c

        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(hf_c, "HF_HUB_OFFLINE", True)
        monkeypatch.setattr(hf, "snapshot_download", lambda *a, **k: None)
        monkeypatch.setattr(vc, "_suppress_hf_progress", lambda: __import__("contextlib").nullcontext())
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: True)
        monkeypatch.setattr(vc, "set_active_embedding_model", lambda m: object())
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._download_embedding_model("m")
        assert hf_c.HF_HUB_OFFLINE is True  # restored


# ─── _load_dotenv debug-log branches (caplog, real logger) ────────────────────


class TestLoadDotenvDebugLogs:
    def test_quoted_inline_comment_logged(self, tmp_path, caplog):
        # inline comment AFTER the closing quote triggers the quoted-branch debug log
        (tmp_path / ".env").write_text('ASI_DBG="v # comment" # trailing\n')
        os.environ.pop("ASI_DBG", None)
        with caplog.at_level(logging.DEBUG, logger="asi"):
            asi._load_dotenv(str(tmp_path))
        assert any("inline comment" in r.message for r in caplog.records)
        assert os.environ.get("ASI_DBG") == "v # comment"
        os.environ.pop("ASI_DBG", None)

    def test_unquoted_inline_comment_logged(self, tmp_path, caplog):
        (tmp_path / ".env").write_text("ASI_DBG2=value # comment\n")
        os.environ.pop("ASI_DBG2", None)
        with caplog.at_level(logging.DEBUG, logger="asi"):
            asi._load_dotenv(str(tmp_path))
        assert any("inline comment" in r.message for r in caplog.records)
        assert os.environ.get("ASI_DBG2") == "value"
        os.environ.pop("ASI_DBG2", None)


# ─── _extract_tool_cmd: truthy args but nothing known ──────────────────────────


class TestExtractToolCmdNothing:
    def test_unknown_args_returns_empty(self):
        assert asi._extract_tool_cmd({"unrelated": 1}) == ""


# ─── main(): NONE log level with file only ────────────────────────────────────


class TestMainLogNone:
    def test_log_level_none_file_saved(self, monkeypatch, tmp_path):
        cfg_dir = tmp_path / ".asicode"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text("{}")
        monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "x")
        monkeypatch.setenv("EXTERNAL_LLM_MODEL", "y")
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_terminal_config_path", lambda r: None)
        captured = {}
        monkeypatch.setattr(asi, "run_repl", lambda a: captured.update(args=a))
        monkeypatch.setattr(asi.sys, "argv", ["asi", "--log-level", "NONE", "--log-file", str(tmp_path / "none.log")])
        saved = []
        monkeypatch.setattr(asi, "_setup_logging", lambda level, log_file=None: saved.append((level, log_file)))
        asi.main()
        assert saved and saved[0][1] == str(tmp_path / "none.log")
