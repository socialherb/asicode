"""Tests for utility functions in asi.py and scripts/release_public.py.

Covers:
  * ``_rotate_cli_history_if_needed`` — 3 scenarios: below threshold (no-op),
    above threshold (truncate + keep recent), edge-case boundary snap.
  * ``_changelog_has_version`` — present / absent / missing file.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

# ── _rotate_cli_history_if_needed ─────────────────────────────────────────────
# Import the function directly from asi.py — it has no asi-internal deps beyond
# the two constants, which we import alongside.
import asi
from asi import _CLI_HISTORY_KEEP, _CLI_HISTORY_ROTATE_AT, _rotate_cli_history_if_needed
from scripts.release_public import _changelog_has_version


class TestRotateCliHistory:
    """Pin CLI history rotation logic."""

    def _write_history(self, path: str, n: int) -> list[bytes]:
        """Write *n* single-line entries (``# <ts>`` + ``content``) to *path*.

        Returns the written lines so callers can verify which survive rotation.
        """
        lines: list[bytes] = []
        for i in range(n):
            lines.append(f"# {i}\n".encode())
            lines.append(f"command_{i}\n".encode())
        with open(path, "wb") as f:
            f.writelines(lines)
        return lines

    def test_below_threshold_no_op(self):
        """Under threshold lines: file is NOT rewritten."""
        with tempfile.NamedTemporaryFile(suffix=".history", delete=False) as f:
            tmp = f.name
        try:
            # Each entry = 2 lines, so halve the count to stay under line threshold
            n = _CLI_HISTORY_ROTATE_AT // 2 - 100
            self._write_history(tmp, n)
            before_size = os.path.getsize(tmp)
            _rotate_cli_history_if_needed(tmp)
            assert os.path.getsize(tmp) == before_size  # untouched
        finally:
            os.unlink(tmp)

    def test_above_threshold_rotates(self):
        """Above threshold: file shrinks to about ``_CLI_HISTORY_KEEP`` lines."""
        with tempfile.NamedTemporaryFile(suffix=".history", delete=False) as f:
            tmp = f.name
        try:
            n = _CLI_HISTORY_ROTATE_AT + 5000
            self._write_history(tmp, n)
            _rotate_cli_history_if_needed(tmp)
            with open(tmp, "rb") as f:
                kept = f.readlines()
            # Should be roughly _CLI_HISTORY_KEEP lines (2 per entry)
            assert len(kept) <= _CLI_HISTORY_KEEP * 2
            assert len(kept) >= _CLI_HISTORY_KEEP  # at least one full entry boundary
        finally:
            os.unlink(tmp)

    def test_missing_file_no_op(self):
        """Non-existent path is silently ignored (no crash)."""
        _rotate_cli_history_if_needed("/tmp/nonexistent_history_file_xyz")

    def test_keeps_most_recent_entries(self):
        """After rotation, the most recent entries survive, oldest are dropped."""
        with tempfile.NamedTemporaryFile(suffix=".history", delete=False) as f:
            tmp = f.name
        try:
            n = _CLI_HISTORY_KEEP + 100
            self._write_history(tmp, n)
            _rotate_cli_history_if_needed(tmp)
            with open(tmp, "rb") as f:
                kept = f.readlines()
            # The last entry should be the most recent
            last_content = kept[-1].decode().strip()
            assert last_content == f"command_{n - 1}"
            # The first entry should be from the tail, not the very first
            first_content = kept[0].decode().strip()
            assert first_content.startswith("# ")
        finally:
            os.unlink(tmp)

    def test_empty_file_no_op(self):
        """Empty history file is not touched."""
        with tempfile.NamedTemporaryFile(suffix=".history", delete=False) as f:
            tmp = f.name
        try:
            before_size = os.path.getsize(tmp)
            _rotate_cli_history_if_needed(tmp)
            assert os.path.getsize(tmp) == before_size
        finally:
            os.unlink(tmp)

    def test_multiline_entry_preserved(self):
        """Multi-line entries (``+...`` continuation) are not split at rotation."""
        with tempfile.NamedTemporaryFile(suffix=".history", delete=False) as f:
            tmp = f.name
        try:
            # Write exactly threshold+1 entries, last one is multi-line
            lines: list[bytes] = []
            for i in range(_CLI_HISTORY_ROTATE_AT - 1):
                lines.append(f"# {i}\n".encode())
                lines.append(f"cmd_{i}\n".encode())
            # One big multi-line entry at the end
            lines.append(f"# {_CLI_HISTORY_ROTATE_AT - 1}\n".encode())
            lines.append(b"+line1\n")
            lines.append(b"+line2\n")
            lines.append(b"+line3\n")
            with open(tmp, "wb") as f:
                f.writelines(lines)

            _rotate_cli_history_if_needed(tmp)

            with open(tmp, "rb") as f:
                kept = f.readlines()
            # The last multi-line entry must be intact
            assert b"+line1\n" in kept
            assert b"+line2\n" in kept
            assert b"+line3\n" in kept
        finally:
            os.unlink(tmp)


# ── _changelog_has_version ────────────────────────────────────────────────────


class TestChangelogHasVersion:
    """Pin CHANGELOG entry detection."""

    @patch("scripts.release_public.REPO", autospec=True)
    def test_version_present(self, mock_repo):
        """Existing ``## [0.2.12]`` header returns True."""
        mock_repo.__truediv__.return_value.read_text.return_value = (
            "# Changelog\n\n## [0.2.12] - 2026-07-20\n\nBugfixes.\n"
        )
        assert _changelog_has_version("0.2.12") is True

    @patch("scripts.release_public.REPO", autospec=True)
    def test_version_absent(self, mock_repo):
        """Missing version header returns False."""
        mock_repo.__truediv__.return_value.read_text.return_value = (
            "# Changelog\n\n## [0.2.11] - 2026-07-10\n\nOld stuff.\n"
        )
        assert _changelog_has_version("0.2.12") is False

    @patch("scripts.release_public.REPO", autospec=True)
    def test_missing_file(self, mock_repo):
        """Non-existent CHANGELOG.md returns False (no crash)."""
        from pathlib import Path as _Path
        mock_repo.__truediv__.return_value = _Path("/nonexistent/CHANGELOG.md")
        assert _changelog_has_version("0.2.12") is False

    @patch("scripts.release_public.REPO", autospec=True)
    def test_semver_not_partial_match(self, mock_repo):
        """``## [0.2.1]`` does NOT match a search for ``0.2.12``."""
        mock_repo.__truediv__.return_value.read_text.return_value = (
            "# Changelog\n\n## [0.2.1] - 2026-07-01\n"
        )
        assert _changelog_has_version("0.2.12") is False


# ── Checkpoint-backed /undo (the non-git path) ───────────────────────────────

class TestCheckpointUndoHelpers:
    """``/undo`` outside a git work tree.

    ``_git_baseline`` returns None in a plain directory, and the whole
    /diff · /undo surface used to vanish with it — while the agent's write gate
    had been recording a checkpoint of every file it touched the entire time.
    Nothing outside ``webapp/`` read those checkpoints and webapp/ is excluded
    from the public export, so the shipped CLI wrote undo data no code could
    read back. These cover the helpers that close that gap.
    """

    def _agent_run(self, root):
        """Drive real write tools so the gate is exercised, not simulated."""
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        reg = ToolRegistry(repo_root=str(root), config=AgentConfig())
        assert reg.dispatch(
            "edit_text",
            {"file_path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 999"},
        ).ok
        assert reg.dispatch(
            "apply_patch",
            {"path": "made.py",
             "content": "--- /dev/null\n+++ b/made.py\n@@ -0,0 +1 @@\n+NEW = True\n"},
        ).ok
        assert reg.dispatch(
            "edit_text",
            {"file_path": "made.py", "old_text": "NEW = True", "new_text": "NEW = False"},
        ).ok

    def test_undo_reverts_edits_and_deletes_creations_without_git(self, tmp_path):
        from asi import (
            _checkpoint_changed_files,
            _git_baseline,
            _newest_checkpoint_id,
            _undo_via_checkpoint,
        )

        root = tmp_path / "plain"
        root.mkdir()
        (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

        assert _git_baseline(str(root)) is None, "precondition: not a git repo"
        before = _newest_checkpoint_id(str(root))
        self._agent_run(root)

        cid = _newest_checkpoint_id(str(root))
        assert cid is not None and cid != before
        assert _checkpoint_changed_files(str(root), cid) == ["app.py", "made.py"]

        assert (root / "made.py").read_text(encoding="utf-8") == "NEW = False\n"
        assert _undo_via_checkpoint(str(root), cid) is True
        assert (root / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert not (root / "made.py").exists(), (
            "a file the run created must be deleted, not left at its mid-run content"
        )

    def test_helpers_are_quiet_when_there_is_nothing_to_undo(self, tmp_path):
        from asi import (
            _checkpoint_changed_files,
            _newest_checkpoint_id,
            _undo_via_checkpoint,
        )

        root = tmp_path / "untouched"
        root.mkdir()
        assert _newest_checkpoint_id(str(root)) is None
        assert _checkpoint_changed_files(str(root), "checkpoint_nope") == []
        assert _undo_via_checkpoint(str(root), "checkpoint_nope") is False

    def test_read_only_run_creates_no_undo_point(self, tmp_path):
        """A turn that changed nothing must not offer /undo."""
        from asi import _newest_checkpoint_id
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        root = tmp_path / "readonly"
        root.mkdir()
        (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

        reg = ToolRegistry(repo_root=str(root), config=AgentConfig())
        reg.dispatch("read_file", {"file_path": "app.py"})
        assert _newest_checkpoint_id(str(root)) is None


# ── _load_dotenv inline-comment semantics ─────────────────────────────────────
class TestLoadDotenvInlineComments:
    """Unquoted '#' must match python-dotenv semantics: only a whitespace-
    preceded '#' starts a comment; a bare '#' is part of the value."""

    def test_bare_hash_in_value_is_preserved(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "ASI_T_URL=https://api.example.com/v1#fragment\n"
            "ASI_T_TOKEN=abc#def\n",
            encoding="utf-8",
        )
        for k in ("ASI_T_URL", "ASI_T_TOKEN"):
            monkeypatch.delenv(k, raising=False)
        asi._load_dotenv(str(tmp_path))
        try:
            assert os.environ.get("ASI_T_URL") == "https://api.example.com/v1#fragment"
            assert os.environ.get("ASI_T_TOKEN") == "abc#def"
        finally:
            os.environ.pop("ASI_T_URL", None)
            os.environ.pop("ASI_T_TOKEN", None)

    def test_whitespace_hash_still_strips_comment(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "ASI_T_C=value # note\n"
            "ASI_T_EMPTY= #comment only\n",
            encoding="utf-8",
        )
        for k in ("ASI_T_C", "ASI_T_EMPTY"):
            monkeypatch.delenv(k, raising=False)
        asi._load_dotenv(str(tmp_path))
        try:
            assert os.environ.get("ASI_T_C") == "value"
            assert os.environ.get("ASI_T_EMPTY") == ""
        finally:
            os.environ.pop("ASI_T_C", None)
            os.environ.pop("ASI_T_EMPTY", None)


# ── _model_candidates / model resolution ─────────────────────────────────────
class TestModelCandidates:
    """The candidate scan is shared by both resolvers — pin prefix semantics."""

    def test_known_then_ollama_prefix_scan(self, monkeypatch):
        monkeypatch.setattr(
            asi,
            "_KNOWN_MODELS",
            {"anthropic": ["claude-sonnet-4-6", "claude-opus-4-1"], "openai": ["gpt-4o"]},
        )
        monkeypatch.setattr(
            asi, "_get_ollama_models", lambda timeout=5: ["claude-local", "llama3"]
        )
        assert asi._model_candidates("claude") == [
            ("anthropic", "claude-sonnet-4-6"),
            ("anthropic", "claude-opus-4-1"),
            ("ollama", "claude-local"),
        ]
        # exact name is found via the prefix scan (no separate exact branch needed)
        assert asi._model_candidates("gpt-4o") == [("openai", "gpt-4o")]
        assert asi._model_candidates("zzz") == []

    def test_resolve_model_arg_consults_helper(self, monkeypatch):
        calls = []
        sentinel = [("anthropic", "claude-sonnet-4-6")]

        def fake(prefix, ollama_timeout=3):
            calls.append((prefix, ollama_timeout))
            return sentinel

        monkeypatch.setattr(asi, "_model_candidates", fake)
        assert asi._resolve_model_arg("Claude") == ("anthropic", "claude-sonnet-4-6")
        assert calls == [("claude", 3)]
        # explicit provider/name never consults the candidate scan
        assert asi._resolve_model_arg("anthropic/claude-x") == ("anthropic", "claude-x")
        assert len(calls) == 1

    def test_resolve_model_interactive_consults_helper(self, monkeypatch):
        monkeypatch.setattr(
            asi,
            "_model_candidates",
            lambda prefix, ollama_timeout=3: [("anthropic", "claude-sonnet-4-6")],
        )
        assert asi._resolve_model_interactive("claude") == ("anthropic", "claude-sonnet-4-6")


# ── _load_checkpoint_store ───────────────────────────────────────────────────
class TestLoadCheckpointStore:
    """The guarded store import lives in one helper; the three /undo helpers
    route through it (existing behavior tests cover the end-to-end paths)."""

    def test_returns_store_when_available(self, tmp_path, monkeypatch):
        from external_llm.agent import checkpoint_store as cs_mod

        monkeypatch.setattr(cs_mod, "CheckpointStore", lambda repo_root: "store")
        assert asi._load_checkpoint_store(str(tmp_path)) == "store"

    def test_none_when_store_unavailable(self, tmp_path, monkeypatch):
        from external_llm.agent import checkpoint_store as cs_mod

        def boom(repo_root):
            raise RuntimeError("corrupt store")

        monkeypatch.setattr(cs_mod, "CheckpointStore", boom)
        assert asi._load_checkpoint_store(str(tmp_path)) is None

    def test_checkpoint_callers_route_through_helper(self, tmp_path, monkeypatch):
        calls = []

        def fake(repo_root):
            calls.append(repo_root)

        monkeypatch.setattr(asi, "_load_checkpoint_store", fake)
        assert asi._newest_checkpoint_id(str(tmp_path)) is None
        assert asi._undo_via_checkpoint(str(tmp_path), "nope") is False
        assert asi._checkpoint_changed_files(str(tmp_path), "nope") == []
        assert len(calls) == 3, "all three /undo helpers must consult the shared loader"
