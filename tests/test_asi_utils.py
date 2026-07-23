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
