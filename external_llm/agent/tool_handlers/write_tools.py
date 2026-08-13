"""Write tool handlers for ToolRegistry (P2-2 split barrel).

Implementation now lives in the sibling modules:

- ``write_tools_core`` — module-level helpers (repo file index, indentation /
  fragment-duplication / block-introducer guards, JSON repair).
- ``write_tools_patch_mixin`` — ``WriteToolsPatchMixin``: write_plan,
  apply_patch, apply_patch_text (git apply), anchor_edit.
- ``write_tools_edit_mixin`` — ``WriteToolsEditMixin``: edit_text, edit_file,
  create_file, modify_symbol, and the post-edit syntax/rollback gate helpers.
- ``write_tools_ast_mixin`` — ``WriteToolsAstMixin``: edit_ast.

This module re-exports the previous public surface (``WriteToolsMixin`` +
module-level helpers) so every existing ``from ...write_tools import X`` keeps
working. The write-safety 3-layer gate (syntax check → origin-skip → rollback)
lives once, in the mixin modules, and is shared via ``self`` calls — it is not
duplicated per tool.
"""
from __future__ import annotations

from .write_tools_ast_mixin import WriteToolsAstMixin
from .write_tools_core import (  # noqa: F401
    _FILE_INDEX_CACHE,
    _check_block_introducer_nesting,
    _detect_enclosing_scope,
    _detect_file_unit,
    _detect_fragment_duplication,
    _find_block_end_line,
    _git_list_tracked_files,
    _leading_indent_width,
    _reindent_to_match,
    _repair_plan_json,
    _repo_file_index,
    canonical_repo_key,
    invalidate_repo_file_index,
)
from .write_tools_edit_mixin import WriteToolsEditMixin
from .write_tools_patch_mixin import WriteToolsPatchMixin


class WriteToolsMixin(WriteToolsPatchMixin, WriteToolsEditMixin, WriteToolsAstMixin):
    """Mixin providing write tool implementations for ToolRegistry.

    Composition of the P2-2 split mixins — see the module docstring. Method
    resolution order: patch (write_plan/apply_patch/anchor_edit) → edit
    (edit_text/edit_file/modify_symbol) → ast (edit_ast). All handlers share
    the same post-edit syntax/rollback gate via ``self`` calls, so the
    write-safety contract is identical to the pre-split single class.
    """
