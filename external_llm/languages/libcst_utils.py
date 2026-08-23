"""
LibCST integration utilities.

LibCST provides format-preserving CST transformations for Python.
Enables precise node manipulation without losing comments, whitespace,
or formatting.
"""

from __future__ import annotations

import logging
from typing import Any

# Late-bind contract: None when libcst is absent, real module/classes when
# present. All access is guarded by _LIBCST_AVAILABLE; Any keeps the lazy-import
# slots usable as type expressions (P45 vector_cache pattern).
_cst: Any = None
MetadataWrapper: Any = None
PositionProvider: Any = None

try:
    import libcst as _cst
    from libcst.metadata import MetadataWrapper, PositionProvider

    _LIBCST_AVAILABLE = True
except ImportError:
    _LIBCST_AVAILABLE = False

logger = logging.getLogger(__name__)

if not _LIBCST_AVAILABLE:
    logger.warning("libcst not installed — CST transforms disabled, falling back to stdlib ast")


# ── Core API ─────────────────────────────────────────────────────────────────


def parse_module(source: str):
    """Parse *source* into a ``libcst.Module``.

    Returns None if parsing fails or libcst is not installed.
    """
    if not _LIBCST_AVAILABLE:
        return None
    try:
        return _cst.parse_module(source)
    except Exception as e:
        logger.debug("libcst parse_module failed: %s", e)
        return None


def find_symbol_range(source: str, symbol_name: str) -> tuple[int, int] | None:
    """Find ``(start_line, end_line)`` of a top-level symbol using LibCST.

    Lines are 1-indexed.  Returns None if the symbol is not found or
    parsing fails.

    Uses a single ``MetadataWrapper`` traversal to find the symbol AND
    resolve its position — avoids the ``id()`` mismatch that occurs
    when positions are collected in a separate pass.
    """
    module = parse_module(source)
    if module is None:
        return None

    try:
        wrapper = MetadataWrapper(module)
        result: list[tuple[int, int]] = []

        # Split qualified name: "ClassName.method" → class="ClassName", method="method"
        parts = symbol_name.rsplit(".", 1)
        bare = parts[-1]
        parent_class = parts[0] if len(parts) == 2 else None

        class _Finder(_cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)  # noqa: V107 — libcst MetadataWrapper contract

            def visit_FunctionDef(self, node: Any) -> bool:  # noqa: N802 — stdlib/3rd-party dispatch protocol (name is fixed by caller)
                if result:
                    return False
                if node.name.value == bare:
                    try:
                        pos = self.get_metadata(PositionProvider, node)
                        if pos is not None:
                            # CodePosition.line is already 1-indexed
                            result.append((pos.start.line, pos.end.line))
                    except Exception as e:
                        logger.debug("find_symbol_range: metadata lookup failed for %r: %s", bare, e)
                return False  # don't descend

            def visit_ClassDef(self, node: Any) -> bool:  # noqa: N802 — stdlib/3rd-party dispatch protocol (name is fixed by caller)
                if result:
                    return False
                if node.name.value == bare:
                    try:
                        pos = self.get_metadata(PositionProvider, node)
                        if pos is not None:
                            # CodePosition.line is already 1-indexed
                            result.append((pos.start.line, pos.end.line))
                    except Exception as e:
                        logger.debug("find_symbol_range: metadata lookup failed for %r: %s", bare, e)
                    return False
                # If looking for Class.method, descend into matching class
                if parent_class is not None and node.name.value == parent_class:
                    # Look for method in class body
                    for stmt in node.body.body if hasattr(node.body, "body") else []:
                        if isinstance(stmt, _cst.FunctionDef) and stmt.name.value == bare:
                            try:
                                pos = self.get_metadata(PositionProvider, stmt)
                                if pos is not None:
                                    # CodePosition.line is already 1-indexed
                                    result.append((pos.start.line, pos.end.line))
                            except Exception as e:
                                logger.debug("find_symbol_range: metadata lookup failed for %r: %s", bare, e)
                            break
                    return False
                return False

        wrapper.visit(_Finder())
        return result[0] if result else None
    except Exception as e:
        logger.debug("find_symbol_range failed: %s", e)
        return None


# _get_node_line_range removed — LibCST nodes do not carry lineno attributes.
# Use _resolve_position_map() + CodeRange instead, or the higher-level
# find_symbol_range() which handles this internally.
