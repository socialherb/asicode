"""Regression: single-file fallback must NEVER overwrite an existing file.

Background: _handle_single_file's all-modes-failed fallback called
_create_default_file_patch() WITHOUT the "create-only" guard that the
multi-file path has (`operation.operation == "create"`). On an existing
file (modify), the placeholder content was written over the user's code
via target_path.write_text() AND the result was reported as success=True
(data loss + false success).

The fix: (1) single-file fallback only runs when the file does not exist,
(2) _create_default_file_patch() itself refuses to touch an existing file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from external_llm.intelligent_service import IntelligentLLMService
from external_llm.multi_planner import FileOperation


class _AlwaysFailLLM:
    """generate_patch always fails -> exercises the fallback path."""

    def generate_patch(self, **kwargs):
        return {"success": False, "error": "mock: all modes failed"}


def _make_service() -> IntelligentLLMService:
    svc = object.__new__(IntelligentLLMService)
    svc.llm_service = _AlwaysFailLLM()
    svc.provider = "mock"
    svc.model = "mock"
    return svc


def test_modify_failure_preserves_existing_file(tmp_path: Path):
    """Existing file + LLM failure -> file content untouched, failure reported."""
    target = tmp_path / "app.py"
    ORIGINAL = "# ORIGINAL USER CODE\ndef foo():\n    return 42\n"
    target.write_text(ORIGINAL, encoding="utf-8")

    result = _make_service().handle_request(
        repo_root=str(tmp_path),
        user_request="fix the bug in app.py",
        target_file="app.py",
        mode="single",
        temperature=0.0,
    )

    assert target.read_text(encoding="utf-8") == ORIGINAL, "existing file was overwritten!"
    assert result.get("success") is False
    assert result.get("fallback_used") is False


def test_create_failure_still_creates_placeholder(tmp_path: Path):
    """Missing file + LLM failure -> placeholder is still created (fallback kept)."""
    result = _make_service().handle_request(
        repo_root=str(tmp_path),
        user_request="create a new module",
        target_file="new_mod.py",
        mode="single",
        temperature=0.0,
    )

    created = tmp_path / "new_mod.py"
    assert result.get("success") is True
    assert result.get("fallback_used") is True
    assert created.exists()
    assert created.read_text(encoding="utf-8").startswith("# File: new_mod.py")


def test_create_default_file_patch_refuses_existing_target(tmp_path: Path):
    """Defense-in-depth: the low-level helper itself never overwrites."""
    target = tmp_path / "existing.py"
    ORIGINAL = "KEEP ME"
    target.write_text(ORIGINAL, encoding="utf-8")

    svc = _make_service()
    patch = svc._create_default_file_patch(tmp_path, FileOperation(
        file_path="existing.py",
        operation="create",
        description="d",
        instructions="i",
    ))

    assert patch == ""
    assert target.read_text(encoding="utf-8") == ORIGINAL
