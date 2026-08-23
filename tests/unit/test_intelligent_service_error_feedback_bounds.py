"""P25-3: ``_build_error_feedback`` must read only the first 30 lines.

It read the ENTIRE target file to render a 30-line context block into the
git-apply error feedback (which is fed back into the LLM prompt on retry) —
a multi-hundred-MB file was fully materialised on every failed-patch retry.
"""

from __future__ import annotations

from pathlib import Path

from external_llm.intelligent_service import IntelligentLLMService


def _boom(*args, **kwargs):
    raise AssertionError("read_text must not be called — feedback needs only the 30-line head")


def test_error_feedback_reads_only_first_30_lines(tmp_path, monkeypatch):
    svc = object.__new__(IntelligentLLMService)
    target = tmp_path / "mod.py"
    with open(target, "w", encoding="utf-8") as fh:
        for i in range(1, 1001):
            fh.write(f"MARKER_{i:06d} = {i}\n")
    monkeypatch.setattr(Path, "read_text", _boom)

    out = svc._build_error_feedback("git_apply_check_failed", target, tmp_path)

    assert "MARKER_000001" in out
    assert "MARKER_000030" in out
    assert "MARKER_000031" not in out
