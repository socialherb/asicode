"""NO_EFFECTIVE_PROGRESS hard gate wiring (apply_patch-only downgrade).

``_apply_no_effective_progress_gate`` is the apply_patch-only hard gate that
downgrades ``ok`` to ``False`` and tags ``failure_class="no_effective_change"``
when a patch applied "successfully" but left every touched file byte-identical
to its pre-edit snapshot (hunks matched already-present content). This pins the
end-to-end wiring contract between ``DesignChatLoop._process_tool_call`` and the
``WriteSafetyManager.all_files_unchanged`` predicate:

* apply_patch + byte-identical  → ok downgraded, metadata tagged;
* apply_patch + real change     → ok preserved, no tag;
* non-apply_patch tools         → NEVER downgraded (``anchor_edit`` /
  ``edit_text`` already_equal is a deliberate success — PARITY with the
  two-regime split documented on the gate);
* predicate error               → fail-open (ok preserved).
"""
from __future__ import annotations

from unittest import mock

from external_llm.agent.design_chat_loop import DesignChatLoop
from external_llm.agent.tool_safety import WriteSafetyManager


def _make_loop(repo_root: str) -> DesignChatLoop:
    """Lightweight DesignChatLoop with a real WriteSafetyManager (skip __init__)."""
    loop = DesignChatLoop.__new__(DesignChatLoop)
    loop.registry = mock.MagicMock()
    loop.registry._safety_manager = WriteSafetyManager(repo_root)
    return loop


def test_apply_patch_no_effective_change_downgrades_ok(tmp_path):
    """apply_patch that leaves the file byte-identical → ok=False + tagged."""
    loop = _make_loop(str(tmp_path))
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    pre = {str(target): "x = 1\n"}  # snapshot == current disk content → no change

    md: dict = {}
    ok = loop._apply_no_effective_progress_gate("apply_patch", True, pre, md)

    assert ok is False
    assert md["failure_class"] == "no_effective_change"


def test_apply_patch_real_change_keeps_ok(tmp_path):
    """apply_patch that actually changed bytes → ok stays True, no tag."""
    loop = _make_loop(str(tmp_path))
    target = tmp_path / "a.py"
    target.write_text("x = 2\n", encoding="utf-8")  # disk changed
    pre = {str(target): "x = 1\n"}  # snapshot differs from disk → real change

    md: dict = {}
    ok = loop._apply_no_effective_progress_gate("apply_patch", True, pre, md)

    assert ok is True
    assert "failure_class" not in md


def test_non_apply_patch_tools_never_downgraded(tmp_path):
    """anchor_edit / edit_text already_equal is a deliberate success (PARITY).

    These tools reaching byte-identity via already_equal is an intentional
    no-op success, not a failed edit — the apply_patch-only name guard keeps
    their ok untouched even when every file is byte-identical.
    """
    loop = _make_loop(str(tmp_path))
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    pre = {str(target): "x = 1\n"}  # byte-identical, but tool != apply_patch

    for name in ("anchor_edit", "edit_text", "modify_symbol", "edit_file", "edit_ast"):
        md: dict = {}
        ok = loop._apply_no_effective_progress_gate(name, True, pre, md)
        assert ok is True, name
        assert "failure_class" not in md, name


def test_gate_fail_open_on_predicate_error():
    """If all_files_unchanged raises, ok must stay True (fail-open, no tag)."""
    loop = DesignChatLoop.__new__(DesignChatLoop)
    loop.registry = mock.MagicMock()
    loop.registry._safety_manager = mock.MagicMock()
    loop.registry._safety_manager.all_files_unchanged.side_effect = RuntimeError("boom")

    md: dict = {}
    ok = loop._apply_no_effective_progress_gate("apply_patch", True, {"x": "y"}, md)

    assert ok is True
    assert "failure_class" not in md
