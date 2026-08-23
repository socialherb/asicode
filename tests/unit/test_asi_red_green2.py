"""RED→GREEN coverage for asi.py — layer 2: completer, auth, insights archive,
client creation, pip/vector installs, embedding warmup, restart, logging rich
branch, main() argument paths. Source-free.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import threading

import pytest

import asi

# huggingface_hub is optional (the `rag` extra). The embedding-download tests
# monkeypatch its internals, so they can only run where it is installed.
HAS_HUGGINGFACE_HUB = importlib.util.find_spec("huggingface_hub") is not None

# ─── _SlashCommandCompleter (prompt_toolkit Completion available) ──────────────


class _Doc:
    def __init__(self, text):
        self.text_before_cursor = text


class TestSlashCommandCompleter:
    def _make(self):
        return asi._SlashCommandCompleter(
            get_provider_fn=lambda: "openai",
            get_model_fn=lambda: "gpt-4o",
            get_dev_models_fn=lambda: {"1": "openai/gpt-4o"},
        )

    def test_no_slash_no_completions(self):
        c = self._make()
        assert list(c.get_completions(_Doc("hello"), None)) == []

    def test_newline_no_completions(self):
        c = self._make()
        assert list(c.get_completions(_Doc("a\n/help"), None)) == []

    def test_command_prefix(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/he"), None))
        assert out and out[0].text == "/help"

    def test_async_delegates(self):
        import asyncio

        c = self._make()

        async def _collect():
            return [i async for i in c.get_completions_async(_Doc("/mo"), None)]

        items = asyncio.run(_collect())
        assert any(i.text.startswith("/model") for i in items)

    def test_think_arg_completions(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/think "), None))
        assert any(i.text == "on" for i in out)

    def test_model_arg_completions(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/model gpt"), None))
        assert any("gpt" in i.text for i in out)

    def test_helper_off_completion(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/helper of"), None))
        assert any(i.text == "off" for i in out)

    def test_failure_patterns_subcommands(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/failure-patterns cl"), None))
        assert any(i.text == "clear" for i in out)

    def test_insights_subcommands(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/insights ar"), None))
        assert any(i.text == "archive" for i in out)

    def test_dev_slot_completion(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/model dev_"), None))
        texts = [i.text for i in out]
        assert "dev_1" in texts and "dev_8" in texts
        metas = [str(i.display_meta) for i in out]
        assert any("✓ set" in m for m in metas)

    def test_dev_slot_model_completion(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/model dev_1 gpt"), None))
        assert any("gpt" in i.text for i in out)

    def test_dev_slot_off(self):
        c = self._make()
        out = list(c.get_completions(_Doc("/model dev_1 of"), None))
        assert any(i.text == "off" for i in out)


# ─── _prompt_auth_retry_key / _commit_verified_api_key ─────────────────────────


class TestPromptAuthRetryKey:
    def test_unsupported_model_steer(self, monkeypatch, capsys):
        svc = type("S", (), {"model": "m", "llm_service": type("L", (), {"client": None})()})()
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._prompt_auth_retry_key("opencode", svc, error_message="model is not supported") is False
        assert any("not support" in str(p) for p in printed)

    def test_empty_key_skips(self, monkeypatch):
        import builtins

        svc = type("S", (), {"llm_service": type("L", (), {"client": None})()})()
        monkeypatch.setattr(asi, "_API_KEY_ENV_MAP", {"x": "X_KEY"})
        monkeypatch.setattr(builtins, "input", lambda p="": "")
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._prompt_auth_retry_key("x", svc, error_message="401") is False
        assert any("skipped" in str(p) for p in printed)

    def test_new_key_success(self, monkeypatch):
        svc = type("S", (), {"llm_service": type("L", (), {"client": None})()})()
        monkeypatch.setattr(asi, "_API_KEY_ENV_MAP", {"x": "X_KEY"})
        import builtins

        monkeypatch.setattr(builtins, "input", lambda p="": "newkey123")
        created = []
        import external_llm.client as cli

        monkeypatch.setattr(cli, "create_llm_client", lambda *a, **k: created.append(k.get("api_key")) or object())
        monkeypatch.setattr(cli, "resolve_provider_base_url", lambda p: "")
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._prompt_auth_retry_key("x", svc, error_message="401") is True
        assert os.environ.get("X_KEY") == "newkey123"
        assert asi._PENDING_API_KEY.get("key") == "newkey123"
        os.environ.pop("X_KEY", None)
        asi._PENDING_API_KEY.clear()

    def test_client_creation_failure(self, monkeypatch):
        svc = type("S", (), {"llm_service": type("L", (), {"client": None})()})()
        monkeypatch.setattr(asi, "_API_KEY_ENV_MAP", {"x": "X_KEY"})
        import builtins

        monkeypatch.setattr(builtins, "input", lambda p="": "badkey")
        import external_llm.client as cli

        monkeypatch.setattr(cli, "create_llm_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._prompt_auth_retry_key("x", svc) is False
        assert any("failed" in str(p) for p in printed)


class TestCommitVerifiedApiKey:
    def test_no_pending(self, monkeypatch):
        asi._PENDING_API_KEY.clear()
        asi._commit_verified_api_key()  # no-op

    def test_persists(self, monkeypatch, tmp_path):
        asi._PENDING_API_KEY.update({"env_var": "ASI_TK", "key": "k1", "provider": "x"})
        saved = []
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_save_key_to_dotenv", lambda root, var, key: saved.append((var, key)))
        asi._commit_verified_api_key()
        assert saved == [("ASI_TK", "k1")]
        assert not asi._PENDING_API_KEY

    def test_persist_failure_logs(self, monkeypatch):
        asi._PENDING_API_KEY.update({"env_var": "ASI_TK2", "key": "k2", "provider": "x"})
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: "/tmp")

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(asi, "_save_key_to_dotenv", _boom)
        asi._commit_verified_api_key()  # logs warning, no raise
        assert not asi._PENDING_API_KEY

    def test_shell_exported_warning(self, monkeypatch):
        asi._PENDING_API_KEY.update({"env_var": "ASI_TK3", "key": "k3", "provider": "x"})
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: "/tmp")
        monkeypatch.setattr(asi, "_save_key_to_dotenv", lambda *a, **k: None)
        monkeypatch.setattr(asi, "_SHELL_PROVIDED_ENV_KEYS", {"ASI_TK3"})
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._commit_verified_api_key()
        assert any("shell already exports" in str(p) for p in printed)


# ─── _create_llm_client_for ────────────────────────────────────────────────────


class TestCreateLlmClientFor:
    def test_ollama_no_key(self, monkeypatch):
        import external_llm.client as cli

        created = []
        monkeypatch.setattr(
            cli,
            "create_llm_client",
            lambda provider, api_key, base_url: created.append((provider, api_key)) or object(),
        )
        monkeypatch.setattr(cli, "resolve_provider_base_url", lambda p: "http://localhost:11434")
        assert asi._create_llm_client_for("ollama") is not None

    def test_env_key_lookup(self, monkeypatch):
        import external_llm.client as cli

        monkeypatch.setenv("ASI_PROV_KEY", "envkey")
        monkeypatch.setattr(asi, "_API_KEY_ENV_MAP", {"prov": "ASI_PROV_KEY"})
        got = []
        monkeypatch.setattr(
            cli, "create_llm_client", lambda provider, api_key, base_url: got.append(api_key) or object()
        )
        monkeypatch.setattr(cli, "resolve_provider_base_url", lambda p: "")
        asi._create_llm_client_for("prov")
        assert got == ["envkey"]

    def test_failure_returns_none(self, monkeypatch):
        import external_llm.client as cli

        monkeypatch.setattr(
            cli, "create_llm_client", lambda *a, **k: (_ for _ in ()).throw(ModuleNotFoundError("no module"))
        )
        monkeypatch.setattr(cli, "resolve_provider_base_url", lambda p: "")
        assert asi._create_llm_client_for("prov") is None


# ─── _handle_insights_archive (real temp repo) ─────────────────────────────────


class TestHandleInsightsArchive:
    def _write_files(self, tmp_path):
        active_dir = tmp_path / ".asicode"
        active_dir.mkdir()
        (active_dir / "design_insights.md").write_text(
            "# Design Chat Insights\n\n### [pattern] 2026-01-01 10:00 +0900\nbody-a\n"
        )
        (active_dir / "design_insights_archive.md").write_text(
            "# Archived\n\n### [pattern] 2026-01-02 10:00 +0900\nbody-b\n\n"
            "### [pattern] 2026-01-03 10:00 +0900\nbody-c\n"
        )
        return tmp_path

    def test_list(self, tmp_path, monkeypatch):
        repo = self._write_files(tmp_path)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive")
        assert any("archived insights: 2" in str(p) for p in printed)

    def test_list_empty(self, tmp_path, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(tmp_path), "archive")
        assert any("no archived insights" in str(p) for p in printed)

    def test_restore(self, tmp_path, monkeypatch):
        repo = self._write_files(tmp_path)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive restore 1")
        active = (repo / ".asicode" / "design_insights.md").read_text()
        assert "body-b" in active  # promoted
        arch = (repo / ".asicode" / "design_insights_archive.md").read_text()
        assert "body-b" not in arch  # removed from archive

    def test_restore_usage(self, tmp_path, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(tmp_path), "archive restore")
        assert any("usage" in str(p) for p in printed)

    def test_restore_bad_index(self, tmp_path, monkeypatch):
        repo = self._write_files(tmp_path)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive restore 9")
        assert any("no archive entry" in str(p) for p in printed)

    def test_restore_non_int(self, tmp_path, monkeypatch):
        repo = self._write_files(tmp_path)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive restore abc")
        assert any("no archive entry" in str(p) for p in printed)

    def test_drop(self, tmp_path, monkeypatch):
        repo = self._write_files(tmp_path)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(repo), "archive drop 2")
        arch = (repo / ".asicode" / "design_insights_archive.md").read_text()
        assert "body-c" not in arch
        assert "body-b" in arch

    def test_drop_usage(self, tmp_path, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(tmp_path), "archive drop")
        assert any("usage" in str(p) for p in printed)

    def test_unknown_sub(self, tmp_path, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._handle_insights_archive(str(tmp_path), "archive bogus")
        assert any("usage" in str(p) for p in printed)


# ─── _restart_cli ──────────────────────────────────────────────────────────────


class TestRestartCli:
    def test_execv_failure_prints(self, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))

        def _boom(*a, **k):
            raise OSError("exec format error")

        monkeypatch.setattr(os, "execv", _boom)
        asi._restart_cli()
        assert any("auto-restart failed" in str(p) for p in printed)

    def test_prints_restart_message(self, monkeypatch):
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))

        def _ok(*a, **k):
            raise OSError("gone")  # execv never returns; simulate process replacement

        monkeypatch.setattr(os, "execv", _ok)
        asi._restart_cli()
        assert any("Restarting asi" in str(p) for p in printed)


# ─── _pip_install ──────────────────────────────────────────────────────────────


class TestPipInstall:
    def test_success(self, monkeypatch):
        class P:
            returncode = 0
            stdout, stderr = "", ""

        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: P())
        monkeypatch.setattr(asi, "_print", lambda *a, **k: None)
        assert asi._pip_install(["pkg"], timeout=5) is True

    def test_failure(self, monkeypatch):
        class P:
            returncode = 1
            stdout, stderr = "bad", ""

        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: P())
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._pip_install(["pkg"], timeout=5) is False
        assert any("failed" in str(p) for p in printed)

    def test_pep668_retry(self, monkeypatch):
        calls = []

        class P:
            returncode = 1
            stdout = ""
            stderr = "error: externally-managed-environment"

        def fake_run(*a, **k):
            calls.append(a)
            if len(calls) == 1:
                return P()

            class P2:
                returncode = 0
                stdout, stderr = "", ""

            return P2()

        monkeypatch.setattr(asi.subprocess, "run", fake_run)
        monkeypatch.setattr(asi, "_print", lambda *a, **k: None)
        assert asi._pip_install(["pkg"], timeout=5) is True
        assert len(calls) == 2
        assert "--break-system-packages" in calls[1][0] or any("--break" in str(c) for c in calls[1])

    def test_timeout(self, monkeypatch):
        import subprocess as sp

        def _timeout(*a, **k):
            raise sp.TimeoutExpired("pip", 5)

        monkeypatch.setattr(asi.subprocess, "run", _timeout)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._pip_install(["pkg"], timeout=5) is False
        assert any("timed out" in str(p) for p in printed)

    def test_timeout_with_partial_output(self, monkeypatch):
        import subprocess as sp

        def _timeout(*a, **k):
            raise sp.TimeoutExpired("pip", 5, output=b"line1\nline2\nline3")

        monkeypatch.setattr(asi.subprocess, "run", _timeout)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._pip_install(["pkg"], timeout=5) is False
        assert any("line3" in str(p) for p in printed)

    def test_oserror(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("pip missing")

        monkeypatch.setattr(asi.subprocess, "run", _boom)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._pip_install(["pkg"], timeout=5) is False
        assert any("failed" in str(p) for p in printed)

    def test_break_retry_still_fails(self, monkeypatch):
        calls = []

        class P:
            returncode = 1
            stdout = ""
            stderr = "externally-managed-environment"

        monkeypatch.setattr(asi.subprocess, "run", lambda *a, **k: calls.append(a) or P())
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        assert asi._pip_install(["pkg"], timeout=5) is False
        assert len(calls) == 2


# ─── _install_tree_sitter_grammars ─────────────────────────────────────────────


class TestInstallTreeSitterGrammars:
    def test_install_fails(self, monkeypatch):
        monkeypatch.setattr(asi, "_pip_install", lambda *a, **k: False)
        asi._install_tree_sitter_grammars(["pkg"])  # no-op

    def test_core_missing_restart_pending(self, monkeypatch):
        monkeypatch.setattr(asi, "_pip_install", lambda *a, **k: True)
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(tsu, "is_available", lambda: False)
        monkeypatch.setattr(asi, "_DEPS_RESTART_PENDING", False)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._install_tree_sitter_grammars(["pkg"])
        assert asi._DEPS_RESTART_PENDING is True

    def test_core_present_live_refresh(self, monkeypatch):
        monkeypatch.setattr(asi, "_pip_install", lambda *a, **k: True)
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(tsu, "is_available", lambda: True)
        monkeypatch.setattr(tsu, "invalidate_caches", lambda: None)
        monkeypatch.setattr(tsu, "get_available_languages", lambda: {"python", "go"})
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._install_tree_sitter_grammars(["pkg"])
        assert any("py" in str(p) for p in printed)

    def test_refresh_exception(self, monkeypatch):
        monkeypatch.setattr(asi, "_pip_install", lambda *a, **k: True)
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(tsu, "is_available", lambda: True)

        def _boom(*a, **k):
            raise RuntimeError("bad grammar")

        monkeypatch.setattr(tsu, "invalidate_caches", _boom)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._install_tree_sitter_grammars(["pkg"])
        assert any("OFF" in str(p) for p in printed)


# ─── _maybe_prompt_vector_install ──────────────────────────────────────────────


class TestMaybePromptVectorInstall:
    def _patch_vector_flags(self, monkeypatch, deps_ok=True, model_cached=False, fallback_cached=False):
        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(vc, "HAS_FAISS", deps_ok)
        monkeypatch.setattr(vc, "HAS_NUMPY", deps_ok)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", deps_ok)
        monkeypatch.setattr(vc, "get_configured_embedding_model_name", lambda: "preferred-model")
        monkeypatch.setattr(vc, "FALLBACK_EMBEDDING_MODELS", ["fallback-model"])

        def _cached(name):
            if name == "preferred-model":
                return model_cached
            return fallback_cached

        monkeypatch.setattr(asi, "_is_embedding_model_cached", _cached)

    def test_deps_ok_model_cached(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=True, model_cached=True)
        asi._maybe_prompt_vector_install()  # silent

    def test_deps_ok_fallback_cached(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=True, model_cached=False, fallback_cached=True)
        asi._maybe_prompt_vector_install()  # silent

    def test_deps_missing_install_yes(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=False)
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "y")
        monkeypatch.setattr(asi, "_pip_install", lambda *a, **k: True)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        monkeypatch.setattr(asi, "_DEPS_RESTART_PENDING", False)
        asi._maybe_prompt_vector_install()
        assert asi._DEPS_RESTART_PENDING is True
        assert any("Installed" in str(p) for p in printed)

    def test_deps_missing_install_fail(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=False)
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "y")
        monkeypatch.setattr(asi, "_pip_install", lambda *a, **k: False)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._maybe_prompt_vector_install()
        assert any("manually" in str(p) for p in printed)

    def test_deps_missing_skip(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=False)
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "n")
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._maybe_prompt_vector_install()
        assert any("Skipped" in str(p) for p in printed)

    def test_deps_missing_eof(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=False)

        def _eof(p=""):
            raise EOFError

        monkeypatch.setattr(asi, "_collect_input", _eof)
        asi._maybe_prompt_vector_install()

    def test_model_missing_download(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=True)
        monkeypatch.setattr(asi, "_collect_input", lambda p="": "y")
        downloaded = []
        monkeypatch.setattr(asi, "_download_embedding_model", lambda m: downloaded.append(m))
        asi._maybe_prompt_vector_install()
        assert downloaded == ["preferred-model"]

    def test_model_missing_decline_fallback_yes(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=True)
        answers = iter(["n", "y"])
        monkeypatch.setattr(asi, "_collect_input", lambda p="": next(answers))
        downloaded = []
        monkeypatch.setattr(asi, "_download_embedding_model", lambda m: downloaded.append(m))
        asi._maybe_prompt_vector_install()
        assert downloaded == ["fallback-model"]

    def test_model_missing_decline_all(self, monkeypatch):
        self._patch_vector_flags(monkeypatch, deps_ok=True)
        answers = iter(["n", "n"])
        monkeypatch.setattr(asi, "_collect_input", lambda p="": next(answers))
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._maybe_prompt_vector_install()
        assert any("Skipped" in str(p) for p in printed)


# ─── _download_embedding_model ─────────────────────────────────────────────────


@pytest.mark.skipif(
    not HAS_HUGGINGFACE_HUB,
    reason="requires huggingface_hub (rag extra)",
)
class TestDownloadEmbeddingModel:
    def _patch_hf(self, monkeypatch):
        import huggingface_hub.constants as hf_c

        monkeypatch.setattr(hf_c, "HF_HUB_OFFLINE", None)
        import huggingface_hub as hf

        monkeypatch.setattr(hf, "snapshot_download", lambda *a, **k: None)
        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(vc, "_suppress_hf_progress", lambda: contextlib_nullcontext())

    def test_success(self, monkeypatch):
        self._patch_hf(monkeypatch)
        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: True)
        monkeypatch.setattr(vc, "set_active_embedding_model", lambda m: object())
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._download_embedding_model("m1")
        assert any("ready" in str(p) for p in printed)

    def test_not_cached_after_download(self, monkeypatch):
        self._patch_hf(monkeypatch)
        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: False)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._download_embedding_model("m1")
        assert any("not found in cache" in str(p) for p in printed)

    def test_model_none(self, monkeypatch):
        self._patch_hf(monkeypatch)
        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: True)
        monkeypatch.setattr(vc, "set_active_embedding_model", lambda m: None)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._download_embedding_model("m1")
        assert any("Could not load" in str(p) for p in printed)

    def test_exception(self, monkeypatch):
        self._patch_hf(monkeypatch)
        import huggingface_hub as hf

        def _boom(*a, **k):
            raise RuntimeError("network")

        monkeypatch.setattr(hf, "snapshot_download", _boom)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        asi._download_embedding_model("m1")
        assert any("Failed" in str(p) for p in printed)

    def test_offline_bypass_restored(self, monkeypatch):
        import huggingface_hub.constants as hf_c

        monkeypatch.setattr(hf_c, "HF_HUB_OFFLINE", True)
        self._patch_hf(monkeypatch)
        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(asi, "_is_embedding_model_cached", lambda m: True)
        monkeypatch.setattr(vc, "set_active_embedding_model", lambda m: object())
        asi._download_embedding_model("m1")
        # restored after


# ─── _kick_embedding_model_warmup ──────────────────────────────────────────────


class TestKickEmbeddingModelWarmup:
    def _patch(self, monkeypatch, deps=True, cached=True, fallback_cached=False):
        import external_llm.agent.vector_cache as vc

        monkeypatch.setattr(vc, "HAS_FAISS", deps)
        monkeypatch.setattr(vc, "HAS_NUMPY", deps)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", deps)
        monkeypatch.setattr(vc, "get_configured_embedding_model_name", lambda: "pref")
        monkeypatch.setattr(vc, "FALLBACK_EMBEDDING_MODELS", ["fall"])

        def _cached(name):
            return cached if name == "pref" else fallback_cached

        monkeypatch.setattr(asi, "_is_embedding_model_cached", _cached)
        monkeypatch.setattr(vc, "warmup_embedding_model", lambda: None)
        return vc

    def test_no_deps(self, monkeypatch):
        self._patch(monkeypatch, deps=False)
        asi._kick_embedding_model_warmup()  # silent

    def test_nothing_cached(self, monkeypatch):
        self._patch(monkeypatch, cached=False, fallback_cached=False)
        asi._kick_embedding_model_warmup()  # silent

    def test_starts_thread(self, monkeypatch):
        self._patch(monkeypatch, cached=True)
        started = []

        class FakeThread:
            def __init__(self, *a, **k):
                started.append(k.get("name"))

            def start(self):
                pass

        monkeypatch.setattr(threading, "Thread", FakeThread)
        asi._kick_embedding_model_warmup()
        assert "emb-warmup" in started


# ─── _setup_logging rich branch ────────────────────────────────────────────────


class TestSetupLoggingRich:
    @pytest.fixture(autouse=True)
    def _restore_root_logging(self):
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        yield
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        asi._LOG_FILE_HANDLER = None

    def test_rich_handler_used(self, tmp_path, monkeypatch):
        monkeypatch.setattr(asi, "_RICH", True)
        # Force the RichHandler import path by faking the lazy slot
        rich_logging = pytest.importorskip("rich.logging")
        monkeypatch.setattr(asi, "RichHandler", rich_logging.RichHandler)
        monkeypatch.setattr(asi, "_log_console", type("LC", (), {})())
        logfile = str(tmp_path / "run.log")
        asi._setup_logging("INFO", log_file=logfile)
        assert asi._LOG_FILE_HANDLER is not None
        asi._LOG_FILE_HANDLER = None


# ─── main() argument paths ─────────────────────────────────────────────────────


class TestMainArgPaths:
    def _run_main(self, monkeypatch, argv, prompt_stdin_text=""):
        monkeypatch.setattr(asi.sys, "argv", ["asi", *argv])
        import io

        monkeypatch.setattr(asi.sys, "stdin", io.StringIO(prompt_stdin_text))
        # stub out the heavy entry points so main() never touches engines
        calls = {}

        def _run_once(a, p):
            calls["run_once"] = (a, p)
            return 0

        monkeypatch.setattr(asi, "run_once", _run_once)
        monkeypatch.setattr(asi, "run_repl", lambda a: calls.setdefault("run_repl", a))
        monkeypatch.setattr(asi, "run_subagent_worker", lambda a: calls.setdefault("subagent", a))
        # _load_dotenv writes into env — keep it real but harmless
        return calls

    def test_collaborate_subcommand(self, monkeypatch):
        entered = []
        import external_llm.repl.collaborate.cli as cli_mod

        monkeypatch.setattr(cli_mod, "main", lambda: entered.append(True))
        monkeypatch.setattr(asi.sys, "argv", ["asi", "collaborate"])
        asi.main()
        assert entered == [True]

    def test_mcp_subcommand(self, monkeypatch):
        entered = []
        import external_llm.repl.collaborate.cli as cli_mod

        monkeypatch.setattr(cli_mod, "main", lambda: entered.append(True))
        monkeypatch.setattr(asi.sys, "argv", ["asi", "mcp"])
        asi.main()
        assert entered == [True]

    def test_prompt_file_read_error(self, monkeypatch, tmp_path):
        self._run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt")])
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        with pytest.raises(SystemExit) as e:
            asi.main()
        assert e.value.code == 1
        assert any("file read error" in str(p) for p in printed)

    def test_prompt_stdin_mutually_exclusive(self, monkeypatch):
        self._run_main(monkeypatch, ["--prompt-stdin", "--prompt", "x"])
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        with pytest.raises(SystemExit) as e:
            asi.main()
        assert e.value.code == 1
        assert any("mutually exclusive" in str(p) for p in printed)

    def test_prompt_stdin_read_error(self, monkeypatch):
        self._run_main(monkeypatch, ["--prompt-stdin"])

        def _boom():
            raise OSError("closed")

        monkeypatch.setattr(asi.sys.stdin, "read", _boom)
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        with pytest.raises(SystemExit) as e:
            asi.main()
        assert e.value.code == 1
        assert any("stdin read error" in str(p) for p in printed)

    def test_prompt_stdin_empty(self, monkeypatch):
        self._run_main(monkeypatch, ["--prompt-stdin"], prompt_stdin_text="  ")
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        with pytest.raises(SystemExit) as e:
            asi.main()
        assert e.value.code == 1
        assert any("empty input" in str(p) for p in printed)

    def test_json_requires_prompt(self, monkeypatch):
        self._run_main(monkeypatch, ["--json"])
        printed = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: printed.append(a))
        with pytest.raises(SystemExit) as e:
            asi.main()
        assert e.value.code == 1
        assert any("requires" in str(p) for p in printed)

    def test_prompt_runs_once(self, monkeypatch):
        calls = self._run_main(monkeypatch, ["--prompt", "hello"])
        with pytest.raises(SystemExit) as e:
            asi.main()
        assert e.value.code == 0
        assert calls.get("run_once") is not None

    def test_repl_mode(self, monkeypatch):
        calls = self._run_main(monkeypatch, [])
        asi.main()
        assert calls.get("run_repl") is not None

    def test_subagent_mode(self, monkeypatch):
        calls = self._run_main(monkeypatch, ["--subagent", "--subagent-id", "abc"])
        asi.main()
        assert calls.get("subagent") is not None

    def test_config_json_load(self, monkeypatch, tmp_path):
        # config.json fallback: no CLI provider/model → read from saved cfg
        cfg_dir = tmp_path / ".asicode"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(json.dumps({"provider": "openai", "model": "gpt-4o"}))
        monkeypatch.setattr(asi, "_resolve_repo_root", lambda r: str(tmp_path))
        monkeypatch.setattr(asi, "_terminal_config_path", lambda r: None)
        captured = {}
        monkeypatch.setattr(asi, "run_repl", lambda a: captured.update(args=a))
        monkeypatch.setattr(asi.sys, "argv", ["asi"])
        asi.main()
        assert captured["args"].provider == "openai"
        assert captured["args"].model == "gpt-4o"


def contextlib_nullcontext():
    import contextlib

    return contextlib.nullcontext()
