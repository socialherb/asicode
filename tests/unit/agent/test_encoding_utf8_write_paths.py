"""Regression tests for UTF-8 encoding in WRITE/READ safety paths.

Background: ASCII locale (Docker C/POSIX, CI) makes bare ``open(path, "w")``
use ASCII as the default codec. Writing non-ASCII source (Korean comments,
emoji) then raises ``UnicodeEncodeError`` — which is a *ValueError* subclass,
NOT an OSError — so ``except OSError`` handlers let it escape, crashing the
rollback/repair loop mid-way and leaving files in an inconsistent state.

The fix adds ``encoding="utf-8"`` everywhere (UTF-8 encodes every str, so the
exception is impossible) and ``encoding="utf-8"`` on READ capture sites so
non-ASCII bytes round-trip losslessly instead of becoming replacement chars.

These tests guard against regressions by (a) exercising non-ASCII round-trips
and (b) asserting the ``encoding="utf-8"`` argument is still present in the
relevant call sites.
"""
from __future__ import annotations

import inspect

import pytest

from external_llm.agent.tool_safety import WriteSafetyManager
from external_llm.hybrid_parser import ParseResult
from external_llm.output_modes import OutputMode
from external_llm.patch_synthesizer import PatchSynthesizer

# ── restore_snapshots: non-ASCII round-trip ──────────────────────────────────

class TestRestoreSnapshotsEncoding:
    """restore_snapshots must write snapshot bytes losslessly regardless of locale."""

    @pytest.mark.parametrize("content", [
        "# Korean comment 한글\nx = 1\n",
        "# emoji 🎉 and 日本語\ny = 2\n",
        "# mixed: English + العربية + emoji ✓\nz = 3\n",
    ])
    def test_non_ascii_roundtrip(self, tmp_path, content):
        path = str(tmp_path / "module.py")
        WriteSafetyManager.restore_snapshots({path: content})
        # Read back with UTF-8 — content must survive losslessly.
        with open(path, encoding="utf-8") as f:
            assert f.read() == content

    def test_multiple_non_ascii_files(self, tmp_path):
        snapshots = {}
        expected = {}
        for i, label in enumerate(["한글", "日本語", "🎉"]):
            p = str(tmp_path / f"f{i}.py")
            snapshots[p] = f"# {label}\nv = {i}\n"
            expected[p] = snapshots[p]
        WriteSafetyManager.restore_snapshots(snapshots)
        for p, exp in expected.items():
            with open(p, encoding="utf-8") as f:
                assert f.read() == exp

    def test_overwrites_existing_non_ascii_file(self, tmp_path):
        path = str(tmp_path / "existing.py")
        # Pre-existing non-ASCII content that would be corrupted on a bad codec.
        with open(path, "w", encoding="utf-8") as f:
            f.write("# OLD 한글\nold = True\n")
        new_content = "# NEW 한글 교체\nnew = False\n"
        WriteSafetyManager.restore_snapshots({path: new_content})
        with open(path, encoding="utf-8") as f:
            assert f.read() == new_content

    def test_empty_snapshot_dict_is_noop(self, tmp_path):
        WriteSafetyManager.restore_snapshots({})  # must not raise


# ── patch_synthesizer._from_full_file: non-ASCII read ────────────────────────

class TestPatchSynthesizerEncoding:
    def test_full_file_diff_for_non_ascii_source(self, tmp_path):
        target = "src/module.py"
        (tmp_path / "src").mkdir()
        (tmp_path / target).write_text("# 원본 한글\nold = 1\n", encoding="utf-8")
        synth = PatchSynthesizer(str(tmp_path))
        result = ParseResult(success=True, mode=OutputMode.FULL_FILE,
                             content="# 수정됨 한글\nnew = 2\n")
        diff = synth._from_full_file(result, target)
        # Diff must reference both old and new non-ASCII content without raising.
        assert "old = 1" in diff
        assert "new = 2" in diff

    def test_full_file_missing_target_uses_empty_old(self, tmp_path):
        synth = PatchSynthesizer(str(tmp_path))
        result = ParseResult(success=True, mode=OutputMode.FULL_FILE,
                             content="# 새 파일 한글\nx = 0\n")
        diff = synth._from_full_file(result, "nonexistent.py")
        assert "x = 0" in diff


# ── Source-level guard: encoding="utf-8" must stay in WRITE/capture sites ─────

class TestEncodingSourceGuard:
    """If someone strips encoding="utf-8" the rollback safety net breaks under
    ASCII locales. These assertions catch that regression at the source level."""

    def test_restore_snapshots_uses_utf8(self):
        src = inspect.getsource(WriteSafetyManager.restore_snapshots)
        assert 'encoding="utf-8"' in src, (
            "restore_snapshots lost encoding=\"utf-8\" — non-ASCII rollback "
            "will raise UnicodeEncodeError under ASCII locale (CI/Docker)"
        )

    def test_from_full_file_uses_utf8_read(self):
        src = inspect.getsource(PatchSynthesizer._from_full_file)
        assert 'encoding="utf-8"' in src, (
            "_from_full_file lost encoding=\"utf-8\" — non-ASCII source will "
            "raise UnicodeDecodeError under ASCII locale (CI/Docker)"
        )
