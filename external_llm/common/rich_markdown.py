"""
Shared ``rich.markdown`` accessor with CJK table-wrap fold patch.

A ``rich.markdown.TableElement`` that contains a table renders every cell using
``rich.table.Table``.  Calling ``Table.add_column(heading)`` with no
``overflow`` override leaves the default ``overflow="ellipsis"`` — a cell
whose content exceeds its proportionally-sized column is truncated mid-content
with ``"…"`` and the rest is permanently lost.

The patch wraps ``TableElement.__rich_console__`` so each yielded ``Table``
switches ellipsis columns to ``overflow="fold"``, making cells wrap to
multiple lines — consistent with how paragraphs/headings already wrap.
Idempotent via a marker attribute.
"""

from __future__ import annotations

import logging

__all__ = ["markdown_cls"]

_log = logging.getLogger(__name__)

_MD_CLS = None


def _patch_rich_md_tables_wrap() -> None:
    """Monkey-patch ``rich.markdown.TableElement`` to fold instead of ellipsize.

    See module docstring for rationale.  Idempotent (checks ``_asicode_fold_patched``
    marker attr before patching).
    """
    try:
        from rich.markdown import TableElement as _TE
    except Exception:
        _log.debug("rich.markdown not available — fold patch skipped")
        return
    if getattr(_TE, "_asicode_fold_patched", False):
        return
    _orig = _TE.__rich_console__

    def _fold(self, console, options):
        for _item in _orig(self, console, options):
            for _c in getattr(_item, "columns", None) or ():
                if _c.overflow == "ellipsis":
                    _c.overflow = "fold"
            yield _item

    _TE.__rich_console__ = _fold
    _TE._asicode_fold_patched = True


def markdown_cls():
    """Patched ``rich.markdown.Markdown`` class.

    Imports ``rich.markdown`` and applies the table-wrap fold patch on first
    call.  Subsequent calls return the cached class (no re-import, no re-patch).
    """
    global _MD_CLS
    if _MD_CLS is not None:
        return _MD_CLS
    from rich.markdown import Markdown

    _patch_rich_md_tables_wrap()
    _MD_CLS = Markdown
    return _MD_CLS
