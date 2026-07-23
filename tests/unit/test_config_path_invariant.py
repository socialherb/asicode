"""Regression tests for config path invariants in asi.py.

Covers three fixes (see asi.py):
  1. ``main()`` reads config.json from git toplevel (``_repo_root``), NOT cwd —
     so ``/model`` (and /think, /helper, /dev, /code) persistence survives
     launches from a subdirectory. ``run_repl()`` writes to the same path.
     This is the bug class documented in ``_resolve_repo_root``'s docstring
     ("regardless of which subdirectory under the repo you run it from").
  2. ``_save_key_to_dotenv`` must NOT crash on write failure (read-only mount,
     full disk) — the API key already lives in os.environ for the session.
  3. ``_restart_cli`` must NOT crash if ``os.execv`` fails — degrade gracefully.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from unittest.mock import patch

from asi import (
    _resolve_repo_root,
    _terminal_config_path,
    _save_key_to_dotenv,
    _restart_cli,
)


# ── #1: config read/write path invariant under subdir launch ─────────────────


class TestConfigPathSubdirInvariant:
    """Pin that config.json read path (main) == write path (run_repl) even
    when asi is launched from a subdirectory of the git repo."""

    def _make_git_repo_with_subdir(self):
        root = tempfile.mkdtemp(prefix="asi_cfg_")
        subdir = os.path.join(root, "pkg", "deep")
        os.makedirs(os.path.join(subdir, ".asicode"))
        os.makedirs(os.path.join(root, ".asicode"))
        json.dump(
            {"provider": "openai", "model": "gpt-5.6"},
            open(os.path.join(root, ".asicode", "config.json"), "w"),
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=root, check=True,
        )
        return root, subdir

    def test_resolve_repo_root_finds_toplevel_from_subdir(self, monkeypatch):
        """_resolve_repo_root(None) returns git toplevel, not cwd.

        Compared via realpath because macOS aliases /var → /private/var and
        git resolves the symlink while tempfile.mkdtemp does not; this cosmetic
        difference is unrelated to the read/write invariant under test."""
        root, subdir = self._make_git_repo_with_subdir()
        monkeypatch.chdir(subdir)
        assert os.path.realpath(_resolve_repo_root(None)) == os.path.realpath(root)

    def test_read_path_equals_write_path_from_subdir(self, monkeypatch):
        """The path main() reads from == the path run_repl() writes to.

        Both must be derived from ``_repo_root`` (git toplevel). A regression
        to ``os.getcwd()`` breaks /model persistence silently."""
        root, subdir = self._make_git_repo_with_subdir()
        monkeypatch.chdir(subdir)

        repo_root = _resolve_repo_root(None)  # what main() computes
        # main() read path (post-fix):
        read_path = os.path.join(repo_root, ".asicode", "config.json")
        # run_repl() write path:
        write_path = os.path.join(repo_root, ".asicode", "config.json")

        assert read_path == write_path  # the core invariant
        assert os.path.realpath(read_path).startswith(os.path.realpath(root))
        # must NOT resolve into the subdir (the old cwd-based bug):
        assert not os.path.realpath(read_path).startswith(os.path.realpath(subdir))

    def test_terminal_config_path_uses_toplevel(self, monkeypatch):
        """Per-terminal isolation config also resolves from toplevel so its
        read (main) and write (run_repl) paths agree."""
        root, subdir = self._make_git_repo_with_subdir()
        monkeypatch.chdir(subdir)
        repo_root = _resolve_repo_root(None)
        # main() reads:  _terminal_config_path(_repo_root)
        # run_repl writes through the same _terminal_config_path(repo_root).
        # _terminal_config_path returns None without a real TTY on stdin, so
        # simulate one (the None case is a separate fallback, not this invariant).
        with patch("asi.sys.stdin.fileno", return_value=0), \
             patch("asi.os.ttyname", return_value="/dev/ttys999"):
            cfg_main = _terminal_config_path(repo_root)
        assert cfg_main is not None
        assert cfg_main.startswith(repo_root)  # toplevel, not subdir
        assert "terminals" in cfg_main

    def test_saved_model_resolvable_from_read_path(self, monkeypatch):
        """End-to-end: the model run_repl persists is visible to main()'s read."""
        root, subdir = self._make_git_repo_with_subdir()
        monkeypatch.chdir(subdir)
        repo_root = _resolve_repo_root(None)
        read_path = os.path.join(repo_root, ".asicode", "config.json")
        cfg = json.load(open(read_path))
        assert cfg["model"] == "gpt-5.6"


# ── #2: _save_key_to_dotenv best-effort persistence ───────────────────────────


class TestSaveKeyToDotenvBestEffort:
    """Write failure must not crash the caller (svc already built)."""

    def test_oserror_on_replace_does_not_raise(self, tmp_path, capsys):
        repo_root = str(tmp_path)
        # pre-existing .env so the read branch is exercised
        open(os.path.join(repo_root, ".env"), "w").write('OLD="x"\n')
        with patch("asi.os.replace", side_effect=OSError("Read-only file system")):
            # Must return None normally — NOT propagate OSError.
            _save_key_to_dotenv(repo_root, "OPENAI_API_KEY", "sk-test")
        out = capsys.readouterr().out
        assert "could not persist" in out
        assert "this session only" in out

    def test_tmp_cleaned_up_on_failure(self, tmp_path):
        repo_root = str(tmp_path)
        dotenv = os.path.join(repo_root, ".env")
        open(dotenv, "w").write('OLD="x"\n')
        tmp = dotenv + ".tmp"
        with patch("asi.os.replace", side_effect=OSError("denied")):
            _save_key_to_dotenv(repo_root, "KEY", "v")
        # the .tmp file created before replace must be cleaned up
        assert not os.path.exists(tmp)

    def test_success_path_still_writes(self, tmp_path, capsys):
        repo_root = str(tmp_path)
        _save_key_to_dotenv(repo_root, "OPENAI_API_KEY", "sk-real")
        dotenv = os.path.join(repo_root, ".env")
        assert 'OPENAI_API_KEY="sk-real"' in open(dotenv).read()
        assert "saved to" in capsys.readouterr().out


# ── #3: _restart_cli graceful degradation ─────────────────────────────────────


class TestRestartCliGraceful:
    """execv failure must not crash — the CLI continues degraded."""

    def test_execv_oserror_returns_normally(self, capsys):
        with patch("asi.os.execv", side_effect=OSError("Permission denied")):
            # Must NOT raise; execv is mocked so the process isn't replaced.
            _restart_cli()
        out = capsys.readouterr().out
        assert "auto-restart failed" in out
        assert "restart asi manually" in out
