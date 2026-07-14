"""Tests for ``_write_touched_test_file`` — the conditional index-invalidation gate.

Verifies all argument layouts the gate handles:

* Direct ``"path"`` argument.
* ``apply_patch`` with ``"patch"`` text (``path`` is optional).
* ``write_plan`` with ``"plan"`` as ``dict``, JSON ``str``, or bare ``list``.
* Fallback to top-level ``"ops"`` / ``"operations"``.
"""
from __future__ import annotations

import json

from external_llm.agent.agent_turn_pipeline import _write_touched_test_file


# ── apply_patch: direct path ──────────────────────────────────────────────


def test_apply_patch_direct_test_path():
    """apply_patch with ``path`` pointing to a test file → True."""
    assert _write_touched_test_file("apply_patch", {"path": "tests/test_foo.py"}) is True


def test_apply_patch_direct_source_path():
    """apply_patch with ``path`` pointing to a source file → False."""
    assert _write_touched_test_file("apply_patch", {"path": "src/foo.py"}) is False


def test_apply_patch_empty_args():
    """apply_patch with empty tool_args → False."""
    assert _write_touched_test_file("apply_patch", {}) is False


# ── apply_patch: patch text extraction ────────────────────────────────────


def test_apply_patch_patch_text_with_test_file():
    """apply_patch with no ``path`` but patch touching a test file → True."""
    patch = (
        "--- a/tests/test_foo.py\n+++ b/tests/test_foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    assert _write_touched_test_file("apply_patch", {"patch": patch}) is True


def test_apply_patch_patch_text_with_source_only():
    """apply_patch with ``patch`` touching only source files → False."""
    patch = (
        "--- a/src/bar.py\n+++ b/src/bar.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    assert _write_touched_test_file("apply_patch", {"patch": patch}) is False


def test_apply_patch_patch_text_mixed_files():
    """apply_patch with patch touching both source and test files → True."""
    patch = (
        "--- a/src/bar.py\n+++ b/src/bar.py\n@@ -1 +1 @@\n-old\n+new\n"
        "--- a/tests/test_bar.py\n+++ b/tests/test_bar.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    assert _write_touched_test_file("apply_patch", {"patch": patch}) is True


def test_apply_patch_patch_text_git_diff_format():
    """apply_patch with ``diff --git`` format (alt. header syntax) → True."""
    patch = (
        "diff --git a/tests/test_foo.py b/tests/test_foo.py\n"
        "index abc..def 100644\n"
        "--- a/tests/test_foo.py\n+++ b/tests/test_foo.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    assert _write_touched_test_file("apply_patch", {"patch": patch}) is True


# ── write_plan: dict plan ─────────────────────────────────────────────────


def test_write_plan_dict_ops_with_test_file():
    """write_plan with ``plan`` as dict and ``ops`` containing a test file → True."""
    args = {"plan": {"ops": [{"path": "tests/test_foo.py"}]}}
    assert _write_touched_test_file("write_plan", args) is True


def test_write_plan_dict_ops_source_only():
    """write_plan with ``plan`` dict but only source files → False."""
    args = {"plan": {"ops": [{"path": "src/bar.py"}]}}
    assert _write_touched_test_file("write_plan", args) is False


def test_write_plan_dict_operations_key():
    """write_plan with ``operations`` (not ``ops``) → True."""
    args = {"plan": {"operations": [{"path": "tests/test_baz.py"}]}}
    assert _write_touched_test_file("write_plan", args) is True


def test_write_plan_dict_empty_ops():
    """write_plan with empty ops list → False."""
    args = {"plan": {"ops": []}}
    assert _write_touched_test_file("write_plan", args) is False


def test_write_plan_dict_no_ops_or_operations():
    """write_plan with no ops/operations in plan dict → False."""
    args = {"plan": {"something": "else"}}
    assert _write_touched_test_file("write_plan", args) is False


# ── write_plan: JSON string plan ──────────────────────────────────────────


def test_write_plan_json_string_plan_ops():
    """write_plan with ``plan`` as JSON string containing test file → True."""
    args = {"plan": json.dumps({"ops": [{"path": "tests/test_x.py"}]})}
    assert _write_touched_test_file("write_plan", args) is True


def test_write_plan_json_string_plan_source_only():
    """write_plan with ``plan`` as JSON string containing only source → False."""
    args = {"plan": json.dumps({"ops": [{"path": "src/x.py"}]})}
    assert _write_touched_test_file("write_plan", args) is False


def test_write_plan_json_string_invalid():
    """write_plan with invalid JSON string → False (falls through gracefully)."""
    args = {"plan": "not valid json {ops: ["}
    assert _write_touched_test_file("write_plan", args) is False


# ── write_plan: bare list plan ────────────────────────────────────────────


def test_write_plan_bare_list_test_file():
    """write_plan with ``plan`` as bare list containing test file path → True."""
    args = {"plan": [{"path": "tests/test_y.py"}]}
    assert _write_touched_test_file("write_plan", args) is True


def test_write_plan_bare_list_source_only():
    """write_plan with ``plan`` as bare list containing only source → False."""
    args = {"plan": [{"path": "src/y.py"}]}
    assert _write_touched_test_file("write_plan", args) is False


def test_write_plan_bare_list_empty():
    """write_plan with ``plan`` as empty bare list → False."""
    args = {"plan": []}
    assert _write_touched_test_file("write_plan", args) is False


# ── write_plan: top-level ops fallback ────────────────────────────────────


def test_write_plan_top_level_ops():
    """write_plan with no ``plan`` but top-level ``ops`` containing test → True."""
    args = {"ops": [{"path": "tests/test_z.py"}]}
    assert _write_touched_test_file("write_plan", args) is True


def test_write_plan_top_level_operations():
    """write_plan with no ``plan`` but top-level ``operations`` → True."""
    args = {"operations": [{"path": "tests/test_w.py"}]}
    assert _write_touched_test_file("write_plan", args) is True


def test_write_plan_top_level_ops_source_only():
    """write_plan with no ``plan`` and top-level ops with source only → False."""
    args = {"ops": [{"path": "src/w.py"}]}
    assert _write_touched_test_file("write_plan", args) is False


def test_write_plan_no_plan_and_no_ops():
    """write_plan with neither ``plan`` nor top-level ops → False."""
    assert _write_touched_test_file("write_plan", {}) is False


# ── misc ──────────────────────────────────────────────────────────────────


def test_unrecognised_tool_name():
    """An unrecognised tool name still checks ``path`` (tool-agnostic first pass).

    The gate always inspects ``tool_args["path"]`` before branching on the
    tool name, so a test file path triggers invalidation regardless of the
    tool name.  This is intentional future-proofing.
    """
    assert _write_touched_test_file("run_tests", {"path": "tests/test_foo.py"}) is True


def test_tool_name_not_checked_for_direct_path():
    """Direct path matching works for *any* tool name (future-proofing)."""
    assert _write_touched_test_file("unknown_tool", {"path": "tests/test_foo.py"}) is True


# ── file_path arg (edit_text/modify_symbol/edit_ast/anchor_edit) ───────────
# Regression: these tools carry their target under "file_path", not "path".
# Before the fix, the gate's first pass only checked tool_args["path"], so a
# test file edited via edit_text/modify_symbol/edit_ast/anchor_edit was missed
# and the test-impact index stayed stale (up to 600 s TTL).


def test_edit_text_file_path_test_file():
    """edit_text uses ``file_path`` → a test file must trigger invalidation."""
    assert _write_touched_test_file("edit_text", {"file_path": "tests/test_foo.py"}) is True


def test_edit_text_file_path_source_file():
    """edit_text with ``file_path`` pointing to a source file → False."""
    assert _write_touched_test_file("edit_text", {"file_path": "src/foo.py"}) is False


def test_modify_symbol_file_path_test_file():
    assert _write_touched_test_file("modify_symbol", {"file_path": "tests/unit/test_x.py"}) is True


def test_edit_ast_file_path_test_file():
    assert _write_touched_test_file("edit_ast", {"file_path": "tests/test_y.py"}) is True


def test_anchor_edit_file_path_test_file():
    assert _write_touched_test_file("anchor_edit", {"file_path": "tests/integration/test_z.py"}) is True


def test_edit_text_path_still_works():
    """The legacy ``path`` key (edit_file/apply_patch) still works."""
    assert _write_touched_test_file("edit_file", {"path": "tests/test_foo.py"}) is True


def test_edit_text_no_path_keys():
    """No ``path``/``file_path`` keys → not a direct-path test write."""
    assert _write_touched_test_file("edit_text", {"old_string": "x"}) is False
