"""P25-2: super-context prompt reads must be bounded.

``_build_enhanced_file_context`` read the WHOLE target file before deciding
how much to show — a multi-hundred-MB file was materialised to display at
most ``max_lines`` (500) lines, and the "small file" branch embedded the
entire content (with line numbers) into the prompt. ``_build_project_metadata``
read whole README/requirements files to extract one paragraph / first 10 deps.

Fixes: 1 MiB stat gate on the target embed (same budget as context_builder
heads) + 64 KiB bounded heads for README/requirements.
"""
from __future__ import annotations

from pathlib import Path

from external_llm.super_context_builder import SuperContextBuilder


def _boom(*args, **kwargs):
    raise AssertionError("read_text must not be called on oversized/whole files")


def test_enhanced_file_context_gates_oversized_target(tmp_path, monkeypatch):
    builder = SuperContextBuilder(str(tmp_path))
    big = tmp_path / "huge.py"
    with open(big, "wb") as fh:
        fh.truncate(2 * 1024 * 1024)  # 2 MiB sparse — gate must refuse before reading
    monkeypatch.setattr(Path, "read_text", _boom)

    out = builder._build_enhanced_file_context(big, max_lines=500)

    assert "too large" in out
    assert "read_file" in out  # actionable fallback hint


def test_project_metadata_reads_only_heads(tmp_path, monkeypatch):
    builder = SuperContextBuilder(str(tmp_path))
    (tmp_path / "README.md").write_text(
        "Hello world.\n\n" + ("rest of readme " * 200_000), encoding="utf-8"
    )
    (tmp_path / "requirements.txt").write_text(
        "requests==2.31.0\n" + ("junk_dep==0.1\n" * 100_000), encoding="utf-8"
    )
    monkeypatch.setattr(Path, "read_text", _boom)

    out = builder._build_project_metadata()

    assert "> Hello world." in out
    assert "requests" in out
