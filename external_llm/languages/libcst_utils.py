"""
LibCST integration utilities.

LibCST provides format-preserving CST transformations for Python.
Enables precise node manipulation without losing comments, whitespace,
or formatting.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

try:
    import libcst as _cst
    from libcst.metadata import MetadataWrapper, PositionProvider
    _LIBCST_AVAILABLE = True
except ImportError:
    _cst = None  # type: ignore[assignment]
    MetadataWrapper = None  # type: ignore[assignment,misc]
    PositionProvider = None  # type: ignore[assignment,misc]
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


def find_symbol_range(source: str, symbol_name: str) -> Optional[tuple[int, int]]:
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
            METADATA_DEPENDENCIES = (PositionProvider,)

            def visit_FunctionDef(self, node: _cst.FunctionDef) -> bool:
                if result:
                    return False
                if node.name.value == bare:
                    try:
                        pos = self.get_metadata(PositionProvider, node)
                        if pos is not None:
                            # CodePosition.line is already 1-indexed
                            result.append((pos.start.line, pos.end.line))
                    except Exception:
                        pass
                return False  # don't descend

            def visit_ClassDef(self, node: _cst.ClassDef) -> bool:
                if result:
                    return False
                if node.name.value == bare:
                    try:
                        pos = self.get_metadata(PositionProvider, node)
                        if pos is not None:
                            # CodePosition.line is already 1-indexed
                            result.append((pos.start.line, pos.end.line))
                    except Exception:
                        pass
                    return False
                # If looking for Class.method, descend into matching class
                if parent_class is not None and node.name.value == parent_class:
                    # Look for method in class body
                    for stmt in node.body.body if hasattr(node.body, 'body') else []:
                        if isinstance(stmt, _cst.FunctionDef) and stmt.name.value == bare:
                            try:
                                pos = self.get_metadata(PositionProvider, stmt)
                                if pos is not None:
                                    # CodePosition.line is already 1-indexed
                                    result.append((pos.start.line, pos.end.line))
                            except Exception:
                                pass
                            break
                    return False
                return False

        wrapper.visit(_Finder())
        return result[0] if result else None
    except Exception as e:
        logger.debug("find_symbol_range failed: %s", e)
        return None


def get_node_range(source: str, node) -> Optional[tuple[int, int, int, int]]:
    """Get ``(start_line, start_col, end_line, end_col)`` from a CST node.

    Lines and columns are 1-indexed.  Returns None if the position
    cannot be resolved.

    Because MetadataWrapper creates new node objects during traversal,
    this re-parses *source* and matches *node* by semantic identity
    (type + name) rather than Python ``id()``.
    """
    try:
        module = parse_module(source)
        if module is None:
            return None
        wrapper = MetadataWrapper(module)

        # Determine what to match on
        match_name = None
        match_type = type(node).__name__
        if isinstance(node, (_cst.FunctionDef, _cst.ClassDef)):
            match_name = node.name.value

        result: list[tuple[int, int, int, int]] = []

        class _Finder(_cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)

            def on_visit(self, found: _cst.CSTNode) -> bool:
                if result:
                    return False  # stop once found
                if type(found).__name__ != match_type:
                    return True
                if match_name is not None:
                    if isinstance(found, (_cst.FunctionDef, _cst.ClassDef)):
                        if found.name.value != match_name:
                            return True
                try:
                    pos = self.get_metadata(PositionProvider, found)
                    if pos is not None:
                        # CodePosition is already 1-indexed
                        result.append((pos.start.line, pos.start.column,
                                       pos.end.line, pos.end.column))
                except Exception:
                    pass
                return False if result else True

        wrapper.visit(_Finder())
        return result[0] if result else None
    except Exception as e:
        logger.debug("get_node_range failed: %s", e)
        return None


def get_node_text(source: str, node) -> Optional[str]:
    """Extract the exact source text covered by *node*.

    Uses LibCST metadata to locate the node's byte range in *source*,
    then slices the original text — preserving whitespace, comments,
    and formatting exactly as they appear.

    Returns None if the position cannot be resolved.
    """
    rng = get_node_range(source, node)
    if rng is None:
        return None
    start_line, start_col, end_line, end_col = rng
    lines = source.splitlines(keepends=True)
    # CodePosition: line is 1-indexed → subtract 1 for list index.
    # Column is 0-indexed → use directly.
    sl, sc = start_line - 1, start_col
    el, ec = end_line - 1, end_col

    if sl == el:
        return lines[sl][sc:ec]

    parts = [lines[sl][sc:]]
    for i in range(sl + 1, el):
        parts.append(lines[i])
    parts.append(lines[el][:ec])
    return "".join(parts)


# ── Structural helpers ───────────────────────────────────────────────────────


def structural_hash(module) -> str:
    """Compute a structure-only hash from a LibCST module.

    Ignores comments, whitespace, and formatting by using LibCST's
    ``deep_equals``-style structural comparison.  Returns a hex digest
    string.
    """
    import hashlib
    if module is None:
        return hashlib.sha256(b"").hexdigest()[:16]

    # Use LibCST's own deep_equals for semantic comparison; for hashing
    # we strip whitespace/comment nodes and serialize the remaining tree.
    class _StructureOnlyVisitor(_cst.CSTTransformer):
        def on_visit(self, node: _cst.CSTNode) -> bool:
            return True

        def leave_EmptyLine(
            self, node: _cst.EmptyLine, updated_node: _cst.EmptyLine
        ) -> _cst.EmptyLine:
            # Strip comments and whitespace from empty lines
            return _cst.EmptyLine(
                indent=False,
                whitespace=_cst.SimpleWhitespace(""),
                comment=None,
                newline=_cst.Newline(),
            )

        def leave_SimpleWhitespace(
            self, node: _cst.SimpleWhitespace, updated_node: _cst.SimpleWhitespace
        ) -> _cst.SimpleWhitespace:
            return _cst.SimpleWhitespace("")

        def leave_TrailingWhitespace(
            self, node: _cst.TrailingWhitespace, updated_node: _cst.TrailingWhitespace
        ) -> _cst.TrailingWhitespace:
            return _cst.TrailingWhitespace(
                whitespace=_cst.SimpleWhitespace(""),
                comment=None,
                newline=_cst.Newline(),
            )

        def leave_ParenthesizedWhitespace(
            self, node: _cst.ParenthesizedWhitespace,
            updated_node: _cst.ParenthesizedWhitespace
        ) -> _cst.ParenthesizedWhitespace:
            return _cst.ParenthesizedWhitespace(
                first_line=_cst.SimpleWhitespace(""),
                indent=_cst.SimpleWhitespace(""),
                last_line=_cst.SimpleWhitespace(""),
            )

    try:
        stripped = module.visit(_StructureOnlyVisitor())
        code = stripped.code
    except Exception:
        # Fallback: use str representation
        code = str(module)
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


# ── Internal helpers ─────────────────────────────────────────────────────────


# _get_node_line_range removed — LibCST nodes do not carry lineno attributes.
# Use _resolve_position_map() + CodeRange instead, or the higher-level
# find_symbol_range() which handles this internally.


def _classify_func_change(orig_node: Any, mod_node: Any) -> str:
    """Classify what changed between two function AST nodes.

    Returns a string describing the changed parts:
      ``"body_only"``       — only body statements changed
      ``"sig_only"``        — only signature (params/return) changed
      ``"dec_only"``        — only decorators changed
      ``"body+sig"``        — body and signature changed
      ``"body+dec"``        — body and decorators changed
      ``"sig+dec"``         — signature and decorators changed
      ``"body+sig+dec"``    — all three changed
      ``"unchanged"``       — nothing changed (caller should not call this)

    Uses ``ast.unparse`` for canonical comparison so quote style and
    whitespace differences are ignored.
    """
    import ast as _ast

    def _unparse_list(nodes: list[Any]) -> list[str]:
        out = []
        for n in nodes:
            try:
                out.append(_ast.unparse(n))
            except Exception:
                out.append(repr(n))
        return out

    # Decorator comparison
    dec_changed = _unparse_list(orig_node.decorator_list) != _unparse_list(mod_node.decorator_list)

    # Signature comparison: args + return annotation
    try:
        orig_sig = (
            _ast.unparse(orig_node.args),
            _ast.unparse(orig_node.returns) if orig_node.returns else "",
        )
        mod_sig = (
            _ast.unparse(mod_node.args),
            _ast.unparse(mod_node.returns) if mod_node.returns else "",
        )
        sig_changed = orig_sig != mod_sig
    except Exception:
        sig_changed = True

    # Body comparison (statement-level)
    body_changed = _unparse_list(orig_node.body) != _unparse_list(mod_node.body)

    parts = []
    if body_changed:
        parts.append("body")
    if sig_changed:
        parts.append("sig")
    if dec_changed:
        parts.append("dec")
    if not parts:
        return "unchanged"
    # Single-part changes get "_only" suffix to distinguish from multi-part
    return parts[0] + "_only" if len(parts) == 1 else "+".join(parts)


def _compute_symbol_diff(
    orig_body: list[Any],
    mod_body: list[Any],
    path_prefix: str = "",
) -> dict[str, tuple]:
    """Recursively detect changed functions/methods between two AST body lists.

    Handles arbitrary class nesting depth:
      ``"checkout"``             — top-level function
      ``"MyClass.method"``       — class method
      ``"Outer.Inner.helper"``   — doubly-nested class method

    Nested functions (closures inside functions) are not tracked — only class
    hierarchies are traversed.  New symbols (absent in original) get kind
    ``"new"``; changed symbols get the kind from :func:`_classify_func_change`.

    Returns:
        ``{qualified_name: (change_kind, orig_node_or_None, mod_node)}``
    """
    import ast as _ast

    _is_func = (_ast.FunctionDef, _ast.AsyncFunctionDef)

    orig_funcs   = {n.name: n for n in orig_body if isinstance(n, _is_func)}
    orig_classes = {n.name: n for n in orig_body if isinstance(n, _ast.ClassDef)}

    diff: dict[str, tuple] = {}

    for node in mod_body:
        if isinstance(node, _is_func):
            qname = f"{path_prefix}.{node.name}" if path_prefix else node.name
            if node.name not in orig_funcs:
                diff[qname] = ("new", None, node)
            else:
                kind = _classify_func_change(orig_funcs[node.name], node)
                if kind != "unchanged":
                    diff[qname] = (kind, orig_funcs[node.name], node)
        elif isinstance(node, _ast.ClassDef):
            cls_path = f"{path_prefix}.{node.name}" if path_prefix else node.name
            orig_cls      = orig_classes.get(node.name)
            orig_cls_body = orig_cls.body if orig_cls is not None else []
            diff.update(_compute_symbol_diff(orig_cls_body, node.body, cls_path))

    return diff


def splice_ast_changes(original_source: str, modified_ast: Any) -> Optional[str]:
    """Auto-detect changed functions/methods and splice via LibCST.

    Uses :func:`_compute_symbol_diff` to produce a structural change map
    that covers arbitrary class nesting depth.

    Change routing per function:
      - ``body_only`` → body splice (signature/decorators kept from original)
      - anything else → full node replacement via ast.unparse (correct but
        loses body formatting for that symbol)

    Returns None when nothing changed or the operation fails.
    """
    import ast as _ast

    try:
        orig_ast = _ast.parse(original_source)
    except SyntaxError:
        return None

    _is_func = (_ast.FunctionDef, _ast.AsyncFunctionDef)
    _is_cls  = _ast.ClassDef

    # ── Structural symbol diff (recursive, handles arbitrary nesting) ─────
    symbol_diff = _compute_symbol_diff(orig_ast.body, modified_ast.body)

    body_only_names  = {k for k, (kind, _, _) in symbol_diff.items() if kind == "body_only"}
    full_replace_names = {k for k, (kind, _, _) in symbol_diff.items() if kind != "body_only"}
    all_modified = body_only_names | full_replace_names

    # ── Module-level (non-func/non-class) changes ─────────────────────────
    def _non_func_canonical(tree: Any) -> list[str]:
        return [_ast.unparse(n) for n in tree.body
                if not isinstance(n, ((*_is_func, _is_cls)))]

    has_module_changes = _non_func_canonical(orig_ast) != _non_func_canonical(modified_ast)

    if not all_modified and not has_module_changes:
        return None

    return splice_modified_functions(
        original_source, modified_ast, all_modified,
        full_replace_names=full_replace_names,
        handle_module_changes=has_module_changes,
        orig_ast=orig_ast,
    )


def splice_modified_functions(
    original_source: str,
    modified_ast: Any,
    modified_func_names: set,
    full_replace_names: Optional[set] = None,
    handle_module_changes: bool = False,
    orig_ast: Any = None,
) -> Optional[str]:
    """Replace modified function/method bodies (or full nodes) via LibCST.

    Accepts two kinds of names in ``modified_func_names``:
      - Bare names (``"checkout"``) → top-level function
      - Qualified names (``"MyClass.method"``) → class method

    ``full_replace_names`` (subset of ``modified_func_names``):
      Names in this set replace the **entire** function node (decorators +
      signature + body) via ast.unparse.  Use for signature/decorator changes.
      Names NOT in this set replace only the body (best format preservation).
      Defaults to empty set (all names treated as body-only).

    ``handle_module_changes``:
      When True, also reconstructs the module body to apply module-level
      statement changes (import additions/removals, constant changes, etc.).
      Unchanged non-func statements are kept verbatim from the original CST.
      New or changed statements are generated via ast.unparse (format loss
      confined to those statements only).

    Returns None if the operation fails.
    """
    if not modified_func_names and not handle_module_changes:
        return None

    import ast as _ast
    import textwrap as _tw

    _full      = full_replace_names or set()
    body_names = modified_func_names - _full
    full_names = _full

    orig_cst = parse_module(original_source)
    if orig_cst is None:
        return None

    def _extract_body(func_node: Any, bare_name: str) -> Optional[list]:
        """ast.unparse a function → LibCST parse → body statement list."""
        try:
            unparsed = _ast.unparse(func_node)
        except Exception:
            return None
        for src in (unparsed, _tw.dedent(unparsed)):
            mod_cst = parse_module(src)
            if mod_cst is None:
                continue
            for stmt in mod_cst.body:
                if isinstance(stmt, _cst.FunctionDef) and stmt.name.value == bare_name:
                    return list(stmt.body.body)
        return None

    def _extract_full_node(func_node: Any, bare_name: str) -> Optional[Any]:
        """ast.unparse a function → LibCST parse → full FunctionDef node."""
        try:
            unparsed = _ast.unparse(func_node)
        except Exception:
            return None
        for src in (unparsed, _tw.dedent(unparsed)):
            mod_cst = parse_module(src)
            if mod_cst is None:
                continue
            for stmt in mod_cst.body:
                if isinstance(stmt, _cst.FunctionDef) and stmt.name.value == bare_name:
                    return stmt
        return None

    # ── Unified maps: qualified_path → CST data ───────────────────────────
    # Keyed by the same qualified path used in modified_func_names, e.g.
    # "checkout", "MyClass.run", "Outer.Inner.helper".
    # Handles arbitrary nesting via _build_cst_maps recursion.
    mod_body_map: dict[str, Any] = {}   # path → [body stmts]
    mod_full_map: dict[str, Any] = {}   # path → full FunctionDef CST node

    def _build_cst_maps(ast_body: list[Any], prefix: str = "") -> None:
        """Recursively build mod_body_map / mod_full_map from modified AST."""
        for node in ast_body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                qname = f"{prefix}.{node.name}" if prefix else node.name
                if qname in body_names:
                    body = _extract_body(node, node.name)
                    if body is not None:
                        mod_body_map[qname] = body
                elif qname in full_names:
                    fn = _extract_full_node(node, node.name)
                    if fn is not None:
                        mod_full_map[qname] = fn
            elif isinstance(node, _ast.ClassDef):
                cls_path = f"{prefix}.{node.name}" if prefix else node.name
                _build_cst_maps(node.body, cls_path)

    _build_cst_maps(modified_ast.body)

    if not handle_module_changes and not mod_body_map and not mod_full_map:
        return None

    import difflib as _difflib

    _mod_ast_body = list(modified_ast.body)   # authoritative ordering
    _is_func_ast = (_ast.FunctionDef, _ast.AsyncFunctionDef)
    def _is_nf_ast(n):
        return not isinstance(n, ((*_is_func_ast, _ast.ClassDef)))

    # ── Module-level diff (for leave_Module) ─────────────────────────────
    # Each entry: ('keep', orig_cst_stmt) | ('replace', orig_cst_stmt, new_canon)
    #            | ('insert', new_canon)
    # One entry per non-func/class node in _mod_ast_body, in order.
    # Deleted original stmts produce no entry (they simply vanish).
    #
    # Risk note: orig_nf_ast_nodes and orig_nf_cst_nodes are paired by
    # position (parallel iteration).  This holds for the vast majority of
    # Python module-level code.  The known edge case where it can break:
    # semicolon-separated statements (e.g. ``x=1; y=2``) — LibCST sees
    # one SimpleStatementLine; the AST sees two.  A length-guard below
    # detects this and falls back to regenerating all non-func stmts
    # (still better than full-file fallback).
    #
    # Known limitation: identical canonical strings that appear more than
    # once are matched by SequenceMatcher's LCS, which is the best available
    # heuristic but not a strict identity guarantee.
    _nf_diff: list[Any] = []   # entries described above

    if handle_module_changes:
        _parse_orig = orig_ast if orig_ast is not None else _ast.parse(original_source)

        orig_nf_ast = [n for n in _parse_orig.body if _is_nf_ast(n)]
        orig_nf_cst = [
            s for s in orig_cst.body
            if not isinstance(s, (_cst.FunctionDef, _cst.ClassDef))
        ]
        mod_nf_ast  = [n for n in _mod_ast_body if _is_nf_ast(n)]

        orig_nf_canonicals = []
        if len(orig_nf_ast) == len(orig_nf_cst):
            # Safe to pair AST nodes with CST nodes by position
            try:
                orig_nf_canonicals = [_ast.unparse(n) for n in orig_nf_ast]
            except Exception:
                pass
        # If lengths differ (semicolon edge case), orig_nf_canonicals stays []
        # → SequenceMatcher will treat everything as 'insert' → regenerate all

        mod_nf_canonicals = []
        try:
            mod_nf_canonicals = [_ast.unparse(n) for n in mod_nf_ast]
        except Exception:
            pass

        matcher = _difflib.SequenceMatcher(
            None, orig_nf_canonicals, mod_nf_canonicals, autojunk=False
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i, j in zip(range(i1, i2), range(j1, j2), strict=False):
                    _nf_diff.append(('keep', orig_nf_cst[i] if orig_nf_cst else None))
            elif tag == 'replace':
                pairs = min(i2 - i1, j2 - j1)
                for k in range(pairs):
                    orig_cst_stmt = orig_nf_cst[i1 + k] if orig_nf_cst else None
                    _nf_diff.append(('replace', orig_cst_stmt, mod_nf_canonicals[j1 + k]))
                # Extra inserts (mod longer than orig in this range)
                for k in range(pairs, j2 - j1):
                    _nf_diff.append(('insert', None, mod_nf_canonicals[j1 + k]))
                # Extra deletes (orig longer) → produce no entry (vanish)
            elif tag == 'insert':
                for j in range(j1, j2):
                    _nf_diff.append(('insert', None, mod_nf_canonicals[j]))
            # 'delete': produce no entry

    class _BodyReplacer(_cst.CSTTransformer):
        def __init__(self) -> None:
            # Tracks enclosing class names (not functions) to build qualified
            # paths, e.g. ["Outer", "Inner"] → "Outer.Inner.method".
            self._scope_stack: list[str] = []
            self._nf_iter = iter(_nf_diff)

        def _path(self, name: str) -> str:
            return ".".join([*self._scope_stack, name]) if self._scope_stack else name

        def visit_ClassDef(self, node: Any) -> bool:
            self._scope_stack.append(node.name.value)
            return True

        def leave_ClassDef(self, original: Any, updated: Any) -> Any:
            if self._scope_stack:
                self._scope_stack.pop()
            return updated

        def leave_FunctionDef(self, original: Any, updated: Any) -> Any:
            path = self._path(updated.name.value)
            if path in mod_full_map:
                # Preserve original leading_lines (e.g. class-level comment
                # above the method or blank lines above a top-level function).
                return mod_full_map[path].with_changes(
                    leading_lines=updated.leading_lines
                )
            if path in mod_body_map:
                return updated.with_changes(
                    body=updated.body.with_changes(body=mod_body_map[path])
                )
            return updated

        def leave_Module(self, original: Any, updated: Any) -> Any:
            if not handle_module_changes:
                return updated

            # Build name → already-transformed CST stmt for func/class nodes
            fc_by_name: dict[str, Any] = {}
            for stmt in updated.body:
                if isinstance(stmt, (_cst.FunctionDef, _cst.ClassDef)):
                    fc_by_name[stmt.name.value] = stmt

            new_body: list[Any] = []

            for mod_node in _mod_ast_body:
                if isinstance(mod_node, _is_func_ast):
                    cst_stmt = fc_by_name.get(mod_node.name)
                    if cst_stmt is not None:
                        new_body.append(cst_stmt)
                elif isinstance(mod_node, _ast.ClassDef):
                    cst_stmt = fc_by_name.get(mod_node.name)
                    if cst_stmt is not None:
                        new_body.append(cst_stmt)
                else:
                    # Consume one entry from the pre-computed diff
                    entry = next(self._nf_iter, None)
                    if entry is None:
                        continue
                    action = entry[0]
                    if action == 'keep':
                        orig_cst_stmt = entry[1]
                        if orig_cst_stmt is not None:
                            new_body.append(orig_cst_stmt)
                    else:  # 'replace' or 'insert'
                        orig_cst_stmt = entry[1]   # None for 'insert'
                        new_canon     = entry[2]
                        try:
                            parsed = parse_module(new_canon + "\n")
                            if parsed and parsed.body:
                                new_stmt = parsed.body[0]
                                # Inherit leading_lines from replaced original
                                # (preserves blank lines / comments ABOVE the stmt)
                                if orig_cst_stmt is not None:
                                    new_stmt = new_stmt.with_changes(
                                        leading_lines=orig_cst_stmt.leading_lines
                                    )
                                new_body.append(new_stmt)
                        except Exception:
                            pass

            if not new_body:
                return updated
            return updated.with_changes(body=new_body)

    try:
        result_cst = orig_cst.visit(_BodyReplacer())
        new_source = result_cst.code
    except Exception:
        return None

    try:
        _ast.parse(new_source)
    except SyntaxError:
        return None

    return new_source
